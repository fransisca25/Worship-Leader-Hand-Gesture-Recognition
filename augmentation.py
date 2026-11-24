"""Label-preserving augmentation for landmark windows.

Augmentation happens on the *raw* landmarks, before feature normalisation, so
the same code works for feature versions ``v1`` and ``v2``.  Source windows from
``MP_Data`` are never modified: every function returns new arrays.

A note on what is worth augmenting
---------------------------------
Feature version ``v2`` already removes translation, distance and in-plane
rotation, so those transforms are mostly only useful when training ``v1``.
What still varies in a real worship venue and survives normalisation is:

* out-of-plane hand tilt (toward / away from the camera),
* landmark jitter and briefly lost fingers (tracking noise),
* short detection gaps that freeze the pose (tracking hold),
* which part of a gesture the 50-frame window happens to cover.

Horizontal mirroring is available but **off by default**: handedness and the
palm/back twist that carries the ``medley`` meaning would be destroyed for any
gesture that is not symmetric.
"""

from dataclasses import dataclass

import numpy as np

from tool.kp_extractor import FEATURE_DIM, NUM_LANDMARKS


@dataclass
class AugmentConfig:
    """Which degradations to sample when generating extra training windows."""

    copies: int = 2
    in_plane_rotation_deg: float = 12.0
    tilt_x_deg: float = 10.0
    tilt_y_deg: float = 10.0
    scale_range: float = 0.12
    translate: float = 0.04
    jitter_sigma: float = 0.004
    frame_hold_prob: float = 0.05
    landmark_dropout_prob: float = 0.10
    max_dropped_landmarks: int = 2
    time_shift: int = 2
    mirror_prob: float = 0.0

    def validate(self):
        if self.copies < 0:
            raise ValueError("copies must be >= 0")
        if not 0.0 <= self.mirror_prob <= 1.0:
            raise ValueError("mirror_prob must be in [0, 1]")
        if self.max_dropped_landmarks < 0 or self.max_dropped_landmarks >= NUM_LANDMARKS:
            raise ValueError("max_dropped_landmarks must be in [0, 21)")
        return self


def _as_cube(window):
    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
        raise ValueError("window must have shape (T, %d), got %r" % (FEATURE_DIM, arr.shape))
    return arr.reshape(len(arr), NUM_LANDMARKS, 3).copy()


def _rotate_xyz(pts, rx=0.0, ry=0.0, rz=0.0):
    """Rotate (..., 21, 3) points about the wrist around x, y and z axes."""
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x
    x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y
    x, y = x * cos_z - y * sin_z, x * sin_z + y * cos_z
    pts[..., 0], pts[..., 1], pts[..., 2] = x, y, z
    return pts


def augment_window(window, rng, config=None):
    """Return one augmented copy of a raw (T, 63) landmark window."""
    cfg = config or AugmentConfig()
    cube = _as_cube(window)
    length = len(cube)

    scale = 1.0 + rng.uniform(-cfg.scale_range, cfg.scale_range)
    cube *= scale
    cube[..., 0] += rng.uniform(-cfg.translate, cfg.translate)
    cube[..., 1] += rng.uniform(-cfg.translate, cfg.translate)

    cube = _rotate_xyz(
        cube,
        rx=np.deg2rad(rng.uniform(-cfg.tilt_x_deg, cfg.tilt_x_deg)),
        ry=np.deg2rad(rng.uniform(-cfg.tilt_y_deg, cfg.tilt_y_deg)),
        rz=np.deg2rad(rng.uniform(-cfg.in_plane_rotation_deg, cfg.in_plane_rotation_deg)),
    )

    if cfg.mirror_prob and rng.random() < cfg.mirror_prob:
        cube[..., 0] *= -1.0

    if cfg.time_shift:
        shift = int(rng.integers(-cfg.time_shift, cfg.time_shift + 1))
        if shift > 0:
            cube = np.concatenate([np.repeat(cube[:1], shift, axis=0), cube[:-shift]], axis=0)
        elif shift < 0:
            cube = np.concatenate([cube[-shift:], np.repeat(cube[-1:], -shift, axis=0)], axis=0)

    if cfg.jitter_sigma > 0:
        cube += rng.normal(0.0, cfg.jitter_sigma, cube.shape)

    if cfg.landmark_dropout_prob > 0 and cfg.max_dropped_landmarks > 0:
        n_drop = int(rng.integers(0, cfg.max_dropped_landmarks + 1))
        if n_drop:
            drop = rng.choice(NUM_LANDMARKS, size=n_drop, replace=False)
            cube[:, drop, :] = 0.0

    if cfg.frame_hold_prob > 0:
        hold = rng.random(length) < cfg.frame_hold_prob
        last = None
        for i in range(length):
            if hold[i] and last is not None:
                cube[i] = last
            else:
                last = cube[i].copy()

    return cube.reshape(length, FEATURE_DIM)


def augment_batch(windows, labels, rng, config=None):
    """Return the original batch plus ``config.copies`` augmented copies."""
    cfg = (config or AugmentConfig()).validate()
    X = np.asarray(windows, dtype=np.float64)
    y = np.asarray(labels)
    if X.ndim != 3:
        raise ValueError("windows must be (N, T, 63), got %r" % (X.shape,))

    if cfg.copies == 0:
        return X.astype(np.float32), y

    out_x = [X.astype(np.float32)]
    out_y = [y]
    for _ in range(cfg.copies):
        augmented = np.stack([augment_window(w, rng, cfg) for w in X]).astype(np.float32)
        out_x.append(augmented)
        out_y.append(y)
    return np.concatenate(out_x, axis=0), np.concatenate(out_y, axis=0)



def transition_windows(windows_by_class, rng, copies_per_pair=2,
                       min_new_frames=8, max_new_frames=25, min_old_frames=8):
    """Build windows that straddle a gesture change.

    A window made of ``old`` frames followed by ``new`` frames is labelled with
    the *new* gesture, which teaches the model to answer as soon as enough of
    the new gesture is visible.  That is what makes a ~300 ms transition
    possible without shortening the 50-frame architecture.
    """
    classes = sorted(windows_by_class)
    if len(classes) < 2:
        raise ValueError("need at least two classes to build transitions")

    out_x, out_y = [], []
    for label_a in classes:
        for label_b in classes:
            if label_a == label_b:
                continue
            pool_a = windows_by_class[label_a]
            pool_b = windows_by_class[label_b]
            if len(pool_a) == 0 or len(pool_b) == 0:
                continue
            for _ in range(copies_per_pair):
                win_a = _as_cube(pool_a[int(rng.integers(0, len(pool_a)))])
                win_b = _as_cube(pool_b[int(rng.integers(0, len(pool_b)))])
                length = len(win_a)
                max_new = min(max_new_frames, length - min_old_frames)
                if max_new <= min_new_frames:
                    continue
                new_frames = int(rng.integers(min_new_frames, max_new + 1))
                window = np.concatenate(
                    [win_a[:length - new_frames], win_b[length - new_frames:]], axis=0
                )
                out_x.append(window.reshape(length, FEATURE_DIM))
                out_y.append(label_b)

    if not out_x:
        return np.zeros((0, 0, FEATURE_DIM), dtype=np.float64), np.zeros((0,), dtype=int)
    return np.stack(out_x), np.asarray(out_y, dtype=int)
