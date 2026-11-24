"""Streaming preprocessing for live inference.

Two responsibilities, both shared by ``main.py`` and the offline evaluation so
that the live pipeline and the measured pipeline are the same code:

``LandmarkSmoother``
    Optional exponential smoothing of a single hand pose, applied before the
    sliding window is built.  ``alpha = 1.0`` disables it (the default) because
    the temporal filter in :mod:`temporal` already stabilises predictions; the
    knob exists so the effect can be measured instead of assumed.

``LandmarkBuffer``
    The rolling 50-frame window, with everything the original ``deque`` was
    missing: detection validity, tracking-loss handling, buffer expiration and
    a minimum number of real (non-held) frames before a prediction is allowed.
"""

from collections import deque

import numpy as np

from tool.kp_extractor import (
    FEATURE_DIM,
    NUM_LANDMARKS,
    count_valid_frames,
    is_valid_frame,
)

DEFAULT_WINDOW_LENGTH = 50
DEFAULT_MAX_HOLD_FRAMES = 12
DEFAULT_MIN_VALID_FRAMES = 35


class LandmarkSmoother:
    """Exponential landmark smoothing for a single tracked pose."""

    def __init__(self, alpha=1.0):
        alpha = float(alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state = None

    @property
    def enabled(self):
        return self.alpha < 1.0

    def update(self, points):
        """Smooth one (21, 3) pose."""
        pts = np.asarray(points, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
        if self._state is None or not self.enabled:
            self._state = pts.copy()
            return pts.copy()
        self._state = self.alpha * pts + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self):
        self._state = None


class LandmarkBuffer:
    """Rolling landmark window with explicit handling of lost detections."""

    def __init__(
        self,
        length=DEFAULT_WINDOW_LENGTH,
        max_hold_frames=DEFAULT_MAX_HOLD_FRAMES,
        min_valid_frames=DEFAULT_MIN_VALID_FRAMES,
        smooth_alpha=1.0,
    ):
        if length < 2:
            raise ValueError("length must be >= 2")
        if min_valid_frames > length:
            raise ValueError("min_valid_frames cannot exceed the window length")
        self.length = int(length)
        self.max_hold_frames = int(max_hold_frames)
        self.min_valid_frames = int(min_valid_frames)
        self.smoother = LandmarkSmoother(smooth_alpha)

        self._frames = deque(maxlen=self.length)
        self._last_valid = None
        self.missing_streak = 0
        self.expired = False
        self.held_frames = 0

    # ------------------------------------------------------------------ state
    @property
    def frames(self):
        return len(self._frames)

    @property
    def is_full(self):
        return len(self._frames) == self.length

    @property
    def tracking_lost(self):
        return self.missing_streak > 0

    def reset(self):
        """Forget the current gesture (used on tracking loss and on demand)."""
        self._frames.clear()
        self._last_valid = None
        self.missing_streak = 0
        self.held_frames = 0
        self.expired = False

    # ------------------------------------------------------------------- push
    def push(self, landmarks):
        """Add one frame. ``None`` means MediaPipe did not return a usable hand."""
        if landmarks is None:
            self.missing_streak += 1
            if self._last_valid is not None and self.missing_streak <= self.max_hold_frames:
                # short dropout: hold the last known pose so timing stays intact
                self._frames.append(self._last_valid.copy())
                self.held_frames += 1
                return
            # long dropout: the gesture is gone, drop the stale window
            if self._frames:
                self.expired = True
            self._frames.clear()
            self.held_frames = 0
            return

        pts = self.smoother.update(landmarks)
        flat = pts.reshape(FEATURE_DIM)
        if not is_valid_frame(flat):
            self.push(None)
            return

        self._last_valid = pts
        self._frames.append(pts)
        self.missing_streak = 0
        self.held_frames = 0

    # ----------------------------------------------------------------- window
    def window(self):
        """Return the (50, 63) landmark window, or ``None`` when it is untrustworthy."""
        if not self.is_full:
            return None
        real_frames = self.length - min(self.held_frames, self.length)
        if real_frames < self.min_valid_frames:
            return None
        arr = np.stack(self._frames, axis=0).astype(np.float64).reshape(self.length, FEATURE_DIM)
        if count_valid_frames(arr) < self.min_valid_frames:
            return None
        return arr

    def valid_frames(self):
        """Number of frames that came from a real detection (held frames excluded)."""
        return max(self.length - min(self.held_frames, self.length), 0) if self._frames else 0
