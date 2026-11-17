"""Keypoint / feature extraction for the worship-leader gesture pipeline.

The original project stored 21 MediaPipe hand landmarks as a flat vector of 63
numbers (x, y, z per landmark) taken directly from the normalised image
coordinates.  That representation is preserved as feature version ``v1`` so the
original checkpoint can still be served without any change.

Feature version ``v2`` keeps the *same 63 numbers* but normalises them, so the
classifier no longer has to relearn translation, distance and in-plane rotation:

    wrist-centred       removes where the hand sits inside the frame
    scale-normalised    removes how close the hand is to the camera
    rotation-normalised removes in-plane tilting of the hand
                        (the wrist -> middle-finger-MCP axis is aligned to -y)

None of these operations can alter the relative finger configuration, and none
of them cancel the out-of-plane hand twist that separates ``medley`` from the
three static hold gestures.  The feature dimension therefore stays 63 and the
custom transformer is untouched.

Coordinate contract (MediaPipe Hand Landmarker):
    x, y  normalised by frame width / height (0..1, may leave the frame)
    z     depth relative to the wrist, roughly in the same scale as x
"""

import numpy as np

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3
WRIST = 0
MIDDLE_MCP = 9
MIDDLE_TIP = 12

FEATURE_VERSIONS = ("v1", "v2")
DEFAULT_FEATURE_VERSION = "v2"

FEATURE_CONFIGS = {
    # original project representation, used by trained_model/wl_hand_gesture_transformer.pth
    "v1": {"version": "v1", "normalized": False},
    # wrist-centred, scale-normalised, in-plane rotation-normalised
    "v2": {"version": "v2", "normalized": True, "rotation": True, "scale_landmark": MIDDLE_MCP},
}


def feature_config(feature_version):
    """Return a copy of the preprocessing description for a feature version."""
    try:
        return dict(FEATURE_CONFIGS[feature_version])
    except KeyError:
        raise ValueError(
            "Unknown feature version %r (expected one of %s)" % (feature_version, FEATURE_VERSIONS)
        )


def is_valid_frame(keypoints):
    """True when a 63-value frame contains a usable hand.

    A frame is usable when it is finite, not all-zero (MediaPipe missing-hand
    sentinel) and has a non-degenerate wrist -> middle-MCP distance.
    """
    arr = np.asarray(keypoints, dtype=np.float64).reshape(-1)
    if arr.size != FEATURE_DIM:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    if not np.any(arr):
        return False
    pts = arr.reshape(NUM_LANDMARKS, 3)
    return bool(np.linalg.norm(pts[MIDDLE_MCP, :2] - pts[WRIST, :2]) > 1e-6)


def valid_mask(window):
    """Boolean mask of usable frames inside a (T, 63) window."""
    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
        raise ValueError("window must have shape (T, %d), got %r" % (FEATURE_DIM, arr.shape))
    finite = np.all(np.isfinite(arr), axis=1)
    non_zero = np.any(arr != 0.0, axis=1)
    pts = arr.reshape(-1, NUM_LANDMARKS, 3)
    scale = np.linalg.norm(pts[:, MIDDLE_MCP, :2] - pts[:, WRIST, :2], axis=1)
    return finite & non_zero & (scale > 1e-6)


def count_valid_frames(window):
    """Number of usable frames inside a window (used for min-valid-frame gating)."""
    return int(valid_mask(window).sum())


def repair_missing_frames(window, hold_limit=None):
    """Replace missing frames with the last usable pose (tracking-hold).

    This mirrors what a real tracker does when MediaPipe briefly loses the hand
    and removes the "all-zero frame == a gesture" label noise that made the
    original model answer "medley" for an empty frame.

    ``hold_limit`` caps how many consecutive frames may be held; longer gaps are
    left untouched so the caller can decide to discard the window.
    """
    arr = np.asarray(window, dtype=np.float64).copy()
    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
        raise ValueError("window must have shape (T, %d), got %r" % (FEATURE_DIM, arr.shape))

    mask = valid_mask(arr)
    if not mask.any():
        return arr

    last_index = None
    for i in range(len(arr)):
        if mask[i]:
            last_index = i
            continue
        if last_index is None:
            continue
        if hold_limit is not None and (i - last_index) > hold_limit:
            continue
        arr[i] = arr[last_index]

    first = int(np.flatnonzero(mask)[0])
    if first > 0 and (hold_limit is None or first <= hold_limit):
        arr[:first] = arr[first]
    return arr


def normalize_landmarks(points, rotation=True):
    """Normalise one (21, 3) hand pose into the v2 representation."""
    pts = np.asarray(points, dtype=np.float64).reshape(NUM_LANDMARKS, 3).copy()
    pts -= pts[WRIST]

    scale = float(np.linalg.norm(pts[MIDDLE_MCP, :2]))
    if scale < 1e-6:
        scale = float(np.linalg.norm(pts[MIDDLE_TIP, :2]))
    if scale < 1e-6:
        scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale < 1e-6:
        return np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)

    pts /= scale

    if rotation:
        vec = pts[MIDDLE_MCP, :2]
        angle = np.arctan2(vec[1], vec[0]) + np.pi / 2.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x = pts[:, 0] * cos_a + pts[:, 1] * sin_a
        y = -pts[:, 0] * sin_a + pts[:, 1] * cos_a
        pts[:, 0] = x
        pts[:, 1] = y
    return pts


def normalize_window(window, rotation=True):
    """Normalise a (T, 63) window into v2 features; unusable frames stay all-zero."""
    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
        raise ValueError("window must have shape (T, %d), got %r" % (FEATURE_DIM, arr.shape))
    out = np.zeros_like(arr)
    for i, frame in enumerate(arr):
        if is_valid_frame(frame):
            out[i] = normalize_landmarks(frame, rotation=rotation).reshape(FEATURE_DIM)
    return out


def window_features(window, feature_version=DEFAULT_FEATURE_VERSION, repair=True):
    """Convert a raw (T, 63) landmark window into model features of the requested version."""
    arr = np.asarray(window, dtype=np.float64)
    if repair:
        arr = repair_missing_frames(arr)
    if feature_version == "v1":
        return arr.astype(np.float32)
    cfg = feature_config(feature_version)
    return normalize_window(arr, rotation=cfg.get("rotation", True)).astype(np.float32)


def extract_keypoints(mp_detect):
    """Return the first detected hand as a flat (63,) array, or zeros when no hand.

    Kept with its original signature and behaviour so older scripts stay valid.
    Callers that must distinguish "no hand" from "hand at the origin" should use
    :func:`landmarks_of_result` or :func:`is_valid_frame` instead.
    """
    if mp_detect.hand_landmarks:
        hand = mp_detect.hand_landmarks[0]
        return np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten()
    return np.zeros(NUM_LANDMARKS * 3)


def landmarks_of_result(mp_detect, index=0):
    """Return one hand of a MediaPipe result as a (21, 3) array, or None."""
    if not mp_detect.hand_landmarks or index >= len(mp_detect.hand_landmarks):
        return None
    hand = mp_detect.hand_landmarks[index]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float64)
    if pts.shape != (NUM_LANDMARKS, 3) or not is_valid_frame(pts.reshape(-1)):
        return None
    return pts

    return int(valid_mask(window).sum())
