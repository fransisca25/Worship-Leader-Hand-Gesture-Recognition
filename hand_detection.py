"""Hand detection: MediaPipe Hand Landmarker configuration and stable hand selection.

MediaPipe stays the underlying detector.  This module only centralises the
configuration that used to be hard-coded in several scripts, and adds a stable
selection policy so the pipeline keeps tracking the same hand when two hands
(or a hand plus another person's hand) are visible.

The selection logic is written against plain dictionaries so it can be unit
tested without a camera; ``mediapipe`` itself is imported lazily inside
:func:`create_landmarker`.
"""

import os
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL_PATH = "hand_landmarker.task"

# Lower detection/presence thresholds than the MediaPipe defaults (0.5) help in
# low-light worship venues.  False detections are cheap now because the
# inference gate rejects windows that do not look like a confident gesture,
# whereas a missed hand used to produce a fully confident wrong cue.
DEFAULT_MIN_HAND_DETECTION_CONFIDENCE = 0.4
DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE = 0.4
DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5

# A hand may move at most this much (normalised image units) between frames
# before the tracker considers the previous hand lost and re-acquires.
DEFAULT_MAX_JUMP = 0.35


@dataclass
class HandDetectionConfig:
    """Configuration for the MediaPipe Hand Landmarker."""

    model_path: str = DEFAULT_MODEL_PATH
    num_hands: int = 2
    min_hand_detection_confidence: float = DEFAULT_MIN_HAND_DETECTION_CONFIDENCE
    min_hand_presence_confidence: float = DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE
    min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE

    def resolved_model_path(self):
        """Resolve the model path relative to the project root when needed."""
        path = self.model_path
        if os.path.isabs(path):
            return path
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        return candidate if os.path.exists(candidate) else path


def create_landmarker(config=None, running_mode="video"):
    """Create a MediaPipe ``HandLandmarker`` from a :class:`HandDetectionConfig`."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    config = config or HandDetectionConfig()
    base_options = mp_python.BaseOptions(model_asset_path=config.resolved_model_path())
    mode = vision.RunningMode.VIDEO if str(running_mode).lower() == "video" else vision.RunningMode.IMAGE
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mode,
        num_hands=int(config.num_hands),
        min_hand_detection_confidence=float(config.min_hand_detection_confidence),
        min_hand_presence_confidence=float(config.min_hand_presence_confidence),
        min_tracking_confidence=float(config.min_tracking_confidence),
    )
    return vision.HandLandmarker.create_from_options(options)


def describe_hands(result):
    """Turn a MediaPipe result into a list of plain dictionaries.

    Each entry contains ``index``, ``landmarks`` (21, 3), ``score``,
    ``handedness`` ("Left"/"Right"), ``center`` (x, y) and ``size`` (bounding
    diagonal in normalised units).
    """
    hands = []
    landmarks = getattr(result, "hand_landmarks", None) or []
    handedness = getattr(result, "handedness", None) or []

    for index, hand in enumerate(landmarks):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float64)
        if pts.shape != (21, 3):
            continue

        score, side = 0.0, None
        if index < len(handedness) and handedness[index]:
            category = handedness[index][0]
            score = float(getattr(category, "score", 0.0) or 0.0)
            side = getattr(category, "category_name", None) or getattr(category, "display_name", None)

        span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
        hands.append(
            {
                "index": index,
                "landmarks": pts,
                "score": score,
                "handedness": side,
                "center": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "size": float(np.linalg.norm(span)),
            }
        )
    return hands



class HandSelector:
    """Keeps the pipeline locked on one hand while two hands are visible.

    The previous hand anchor (position + handedness) is used to pick the same
    hand again on the next frame.  Only when the anchored hand disappears, or
    jumps implausibly far, does the selector re-acquire by confidence.
    """

    def __init__(self, max_jump=DEFAULT_MAX_JUMP):
        self.max_jump = float(max_jump)
        self.previous = None
        self.switches = 0
        self.reacquisitions = 0

    def reset(self):
        self.previous = None

    def select(self, hands):
        """Return the chosen hand dictionary, or ``None`` when nothing usable was found."""
        if not hands:
            self.previous = None
            return None

        if self.previous is None:
            chosen = max(hands, key=lambda h: (h["score"], h["size"]))
            self.previous = self._anchor(chosen)
            return chosen

        pool = hands
        if self.previous["handedness"] is not None:
            same_side = [h for h in hands if h["handedness"] == self.previous["handedness"]]
            if same_side:
                pool = same_side

        nearest = min(pool, key=lambda h: self._distance(h, self.previous))
        if self._distance(nearest, self.previous) <= self.max_jump:
            if (self.previous["handedness"] is not None
                    and nearest["handedness"] != self.previous["handedness"]):
                self.switches += 1
            self.previous = self._anchor(nearest)
            return nearest

        chosen = max(pool, key=lambda h: (h["score"], h["size"]))
        self.reacquisitions += 1
        if (self.previous["handedness"] is not None
                and chosen["handedness"] != self.previous["handedness"]):
            self.switches += 1
        self.previous = self._anchor(chosen)
        return chosen

    @staticmethod
    def _anchor(hand):
        return {"center": hand["center"], "handedness": hand["handedness"], "index": hand["index"]}

    @staticmethod
    def _distance(hand, anchor):
        dx = hand["center"][0] - anchor["center"][0]
        dy = hand["center"][1] - anchor["center"][1]
        return float(np.hypot(dx, dy))
