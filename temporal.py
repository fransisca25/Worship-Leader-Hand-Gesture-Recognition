"""Temporal decision layer for live gesture output.

The transformer is a *window* classifier: it answers "which gesture is in these
50 frames?".  Turning that into a cue a multimedia operator can trust needs a
small state machine, because:

* a single frame inside a transition window can flip the answer;
* an ambiguous window should not force one of the four gestures;
* a stale answer must disappear when the hand leaves the frame.

``PredictionGate`` combines confidence, margin, hysteresis and a minimum hold
time in one place.  It is deliberately *not* a stack of smoothers: the audit
showed that stacking EMA or majority voting on top of a 50-frame window only
adds latency, so the gate keeps at most one pending candidate and one committed
gesture.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

NO_GESTURE = "no_gesture"


@dataclass
class GateConfig:
    """Thresholds for :class:`PredictionGate`.

    Defaults were chosen from the calibration sweep printed by ``evaluate.py``
    and the transition measurements recorded in the README.
    """

    confidence_threshold: float = 0.70
    margin_threshold: float = 0.15
    acquire_frames: int = 3
    switch_frames: int = 3
    min_hold_frames: int = 5
    expire_frames: int = 8
    history_size: int = 60

    def __post_init__(self):
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if self.margin_threshold < 0.0:
            raise ValueError("margin_threshold must be >= 0")
        for name in ("acquire_frames", "switch_frames", "min_hold_frames", "expire_frames"):
            if getattr(self, name) < 1:
                raise ValueError("%s must be >= 1" % name)


@dataclass
class GateResult:
    """What the gate wants the UI to show."""

    label: object
    confidence: float
    state: str
    pending: object = None
    streak: int = 0
    probabilities: np.ndarray = field(default=None, repr=False)

    @property
    def is_confident(self):
        return self.label is not None

    def display(self):
        return self.label if self.label is not None else NO_GESTURE


class PredictionGate:
    """Confidence + hysteresis + hold + expiry state machine."""

    def __init__(self, labels, config=None):
        self.labels = list(labels)
        if len(self.labels) < 2:
            raise ValueError("at least two labels are required")
        self.config = config or GateConfig()

        self.committed = None
        self.committed_confidence = 0.0
        self.hold_frames = 0
        self.candidate = None
        self.candidate_streak = 0
        self.silent_frames = 0
        self.history = deque(maxlen=self.config.history_size)
        self.transitions = 0

    # ----------------------------------------------------------------- helpers
    def _result(self, state, confidence=0.0, probabilities=None):
        self.history.append(
            {"label": self.committed, "state": state, "pending": self.candidate,
             "confidence": float(confidence)}
        )
        return GateResult(
            label=self.committed,
            confidence=float(confidence),
            state=state,
            pending=self.candidate,
            streak=self.candidate_streak,
            probabilities=probabilities,
        )

    def _drop(self):
        self.committed = None
        self.committed_confidence = 0.0
        self.hold_frames = 0
        self.candidate = None
        self.candidate_streak = 0

    def reset(self):
        self._drop()
        self.silent_frames = 0
        self.history.clear()

    # ------------------------------------------------------------------ update
    def update(self, probabilities):
        """Feed one model output (or ``None`` when no window is available)."""
        cfg = self.config

        if probabilities is None:
            self.silent_frames += 1
            if self.committed is not None and self.silent_frames >= cfg.expire_frames:
                self._drop()
            state = "no_hand" if self.committed is None else "holding"
            return self._result(state, self.committed_confidence)

        probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if probs.size != len(self.labels):
            raise ValueError("expected %d probabilities, got %d" % (len(self.labels), probs.size))
        if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
            probs = np.zeros_like(probs)
        else:
            probs = probs / probs.sum()

        order = np.argsort(probs)[::-1]
        best = float(probs[int(order[0])])
        second = float(probs[int(order[1])])
        strong = best >= cfg.confidence_threshold and (best - second) >= cfg.margin_threshold

        if not strong:
            self.silent_frames += 1
            self.candidate = None
            self.candidate_streak = 0
            if self.committed is not None:
                self.hold_frames += 1
                if self.silent_frames >= cfg.expire_frames:
                    self._drop()
                    return self._result("uncertain", 0.0, probs)
                return self._result("uncertain", self.committed_confidence, probs)
            return self._result("uncertain", best, probs)

        self.silent_frames = 0
        label = self.labels[int(order[0])]

        if self.committed is None:
            if label == self.candidate:
                self.candidate_streak += 1
            else:
                self.candidate, self.candidate_streak = label, 1
            if self.candidate_streak >= cfg.acquire_frames:
                self.committed = label
                self.committed_confidence = best
                self.hold_frames = 0
                self.candidate, self.candidate_streak = None, 0
                self.transitions += 1
                return self._result("stable", best, probs)
            return self._result("acquiring", best, probs)

        if label == self.committed:
            self.committed_confidence = best
            self.candidate, self.candidate_streak = None, 0
            self.hold_frames += 1
            return self._result("stable", best, probs)

        # a different gesture is winning: hysteresis + minimum hold time
        if label == self.candidate:
            self.candidate_streak += 1
        else:
            self.candidate, self.candidate_streak = label, 1
        self.hold_frames += 1

        if self.candidate_streak >= cfg.switch_frames and self.hold_frames >= cfg.min_hold_frames:
            self.committed = label
            self.committed_confidence = best
            self.hold_frames = 0
            self.candidate, self.candidate_streak = None, 0
            self.transitions += 1
            return self._result("switched", best, probs)

        return self._result("transition", self.committed_confidence, probs)

        return self.label if self.label is not None else NO_GESTURE
