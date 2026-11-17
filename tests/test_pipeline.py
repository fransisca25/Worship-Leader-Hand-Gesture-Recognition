"""Unit tests for the gesture pipeline.

Run with:
    python -m unittest discover -s tests -v
    python -m unittest tests.test_pipeline -v

Covered: feature normalisation, missing-frame handling, the prediction gate,
hand selection, augmentation, splits, checkpoints and metrics.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from augmentation import AugmentConfig, augment_batch, augment_window, transition_windows  # noqa: E402
from checkpoint import LEGACY_CHECKPOINT, load_model, save_checkpoint  # noqa: E402
from dataset_loader import (  # noqa: E402
    HandGestureDataset,
    grouped_folds,
    sequence_index,
    temporal_blocked_split,
)
from hand_detection import HandSelector  # noqa: E402
from metrics import classification_metrics, expected_calibration_error  # noqa: E402
from preprocessing import LandmarkBuffer, LandmarkSmoother  # noqa: E402
from temporal import GateConfig, PredictionGate  # noqa: E402
from tool.kp_extractor import (  # noqa: E402
    FEATURE_DIM,
    count_valid_frames,
    is_valid_frame,
    normalize_landmarks,
    normalize_window,
    repair_missing_frames,
    window_features,
)
from transformer import build_hand_gesture_transformer  # noqa: E402


def synthetic_pose(seed=0):
    """A plausible right-hand pose in normalised image coordinates."""
    rng = np.random.default_rng(seed)
    points = np.zeros((21, 3))
    points[:, 0] = 0.4
    points[:, 1] = 0.6
    for finger in range(5):
        base_angle = -0.9 + finger * 0.45
        for joint in range(4):
            radius = 0.03 + joint * 0.03
            points[1 + finger * 4 + joint, 0] = 0.4 + radius * np.cos(base_angle)
            points[1 + finger * 4 + joint, 1] = 0.6 + radius * np.sin(base_angle)
            points[1 + finger * 4 + joint, 2] = 0.01 * joint
    return points + rng.normal(0, 1e-4, points.shape)


def handedness_sign(points):
    wrist = points[0]
    index = points[5] - wrist
    pinky = points[17] - wrist
    return float(np.sign(index[0] * pinky[1] - index[1] * pinky[0]))


class FeatureTests(unittest.TestCase):
    def test_feature_dimension_is_unchanged(self):
        pose = synthetic_pose()
        self.assertEqual(pose.reshape(-1).size, FEATURE_DIM)
        self.assertEqual(FEATURE_DIM, 63)

    def test_v2_is_translation_invariant(self):
        pose = synthetic_pose()
        shifted = pose.copy()
        shifted[:, 0] += 0.25
        shifted[:, 1] -= 0.2
        np.testing.assert_allclose(normalize_landmarks(pose), normalize_landmarks(shifted), atol=1e-9)

    def test_v2_is_scale_invariant(self):
        """A hand twice as close produces twice as large x, y *and* z (all are
        normalised by frame width), so the normalised pose must be identical."""
        pose = synthetic_pose()
        scaled = pose.copy()
        scaled *= 2.7
        np.testing.assert_allclose(normalize_landmarks(pose), normalize_landmarks(scaled), atol=1e-9)

    def test_v2_is_in_plane_rotation_invariant(self):
        pose = synthetic_pose()
        angle = np.deg2rad(35.0)
        rotated = pose.copy()
        x, y = pose[:, 0].copy(), pose[:, 1].copy()
        rotated[:, 0] = x * np.cos(angle) - y * np.sin(angle)
        rotated[:, 1] = x * np.sin(angle) + y * np.cos(angle)
        np.testing.assert_allclose(normalize_landmarks(pose), normalize_landmarks(rotated), atol=1e-9)

    def test_v2_preserves_handedness_twist(self):
        """The palm/back twist that separates medley must survive normalisation."""
        for seed in range(5):
            pose = synthetic_pose(seed)
            mirrored = pose.copy()
            mirrored[:, 0] *= -1.0
            self.assertEqual(handedness_sign(pose), handedness_sign(normalize_landmarks(pose)))
            self.assertNotEqual(handedness_sign(pose), handedness_sign(normalize_landmarks(mirrored)))

    def test_v1_keeps_raw_coordinates(self):
        window = np.stack([synthetic_pose(i).reshape(-1) for i in range(4)])
        features = window_features(window, "v1", repair=False)
        np.testing.assert_allclose(features, window.astype(np.float32), rtol=0, atol=0)

    def test_invalid_frames_stay_zero_after_normalisation(self):
        window = np.stack([synthetic_pose(i).reshape(-1) for i in range(4)])
        window[2] = 0.0
        features = normalize_window(window)
        self.assertFalse(np.any(features[2]))
        self.assertTrue(is_valid_frame(features[0]))

    def test_unknown_feature_version_is_rejected(self):
        with self.assertRaises(ValueError):
            window_features(np.zeros((2, FEATURE_DIM)), "v9")


class MissingFrameTests(unittest.TestCase):
    def test_repair_fills_gaps_with_previous_pose(self):
        window = np.stack([synthetic_pose(i).reshape(-1) for i in range(6)])
        window[3] = 0.0
        window[4] = 0.0
        repaired = repair_missing_frames(window)
        np.testing.assert_allclose(repaired[3], repaired[2])
        np.testing.assert_allclose(repaired[4], repaired[2])
        self.assertEqual(count_valid_frames(repaired), 6)

    def test_repair_respects_hold_limit(self):
        window = np.stack([synthetic_pose(i).reshape(-1) for i in range(6)])
        window[4:] = 0.0
        # a gap of one frame may be held, a gap of two may not
        self.assertEqual(count_valid_frames(repair_missing_frames(window, hold_limit=0)), 4)
        self.assertEqual(count_valid_frames(repair_missing_frames(window, hold_limit=1)), 5)
        self.assertEqual(count_valid_frames(repair_missing_frames(window, hold_limit=5)), 6)

    def test_all_zero_window_is_left_alone(self):
        window = np.zeros((5, FEATURE_DIM))
        self.assertEqual(count_valid_frames(repair_missing_frames(window)), 0)

    def test_leading_gap_is_back_filled(self):
        window = np.stack([synthetic_pose(i).reshape(-1) for i in range(5)])
        window[:2] = 0.0
        repaired = repair_missing_frames(window)
        np.testing.assert_allclose(repaired[0], repaired[2])


class BufferTests(unittest.TestCase):
    def test_buffer_needs_full_window(self):
        buffer = LandmarkBuffer(length=5, min_valid_frames=3)
        for _ in range(4):
            buffer.push(synthetic_pose())
        self.assertIsNone(buffer.window())
        buffer.push(synthetic_pose())
        self.assertIsNotNone(buffer.window())

    def test_buffer_expires_after_loss(self):
        buffer = LandmarkBuffer(length=5, max_hold_frames=2, min_valid_frames=3)
        for _ in range(5):
            buffer.push(synthetic_pose())
        self.assertIsNotNone(buffer.window())
        buffer.push(None)
        buffer.push(None)
        self.assertFalse(buffer.expired)
        buffer.push(None)
        self.assertTrue(buffer.expired)
        self.assertIsNone(buffer.window())
        self.assertEqual(buffer.frames, 0)

    def test_buffer_rejects_windows_with_too_few_real_frames(self):
        buffer = LandmarkBuffer(length=5, max_hold_frames=5, min_valid_frames=5)
        for _ in range(2):
            buffer.push(synthetic_pose())
        for _ in range(3):
            buffer.push(None)
        self.assertIsNone(buffer.window())

    def test_smoother_disabled_is_identity(self):
        smoother = LandmarkSmoother(alpha=1.0)
        pose = synthetic_pose()
        np.testing.assert_allclose(smoother.update(pose), pose)
        self.assertFalse(smoother.enabled)

    def test_smoother_moves_towards_input(self):
        smoother = LandmarkSmoother(alpha=0.5)
        first = synthetic_pose(0)
        second = synthetic_pose(1)
        smoother.update(first)
        blended = smoother.update(second)
        self.assertLess(np.abs(blended - second).mean(), np.abs(first - second).mean())


class GateTests(unittest.TestCase):
    def setUp(self):
        self.codes = ["verse", "chorus", "bridge", "medley"]
        self.config = GateConfig(confidence_threshold=0.7, margin_threshold=0.15,
                                 acquire_frames=3, switch_frames=3,
                                 min_hold_frames=5, expire_frames=4)

    @staticmethod
    def _probs(index, confidence=0.9):
        row = np.full(4, (1.0 - confidence) / 3.0)
        row[index] = confidence
        return row

    def test_acquires_after_configured_frames(self):
        gate = PredictionGate(self.codes, self.config)
        gate.update(self._probs(0))
        gate.update(self._probs(0))
        self.assertIsNone(gate.committed)
        result = gate.update(self._probs(0))
        self.assertEqual(result.label, "verse")
        self.assertEqual(result.state, "stable")

    def test_ambiguous_input_is_not_forced(self):
        gate = PredictionGate(self.codes, self.config)
        flat = np.array([0.3, 0.3, 0.25, 0.15])
        for _ in range(10):
            result = gate.update(flat)
        self.assertIsNone(result.label)
        self.assertEqual(result.state, "uncertain")

    def test_switch_needs_hysteresis_and_hold(self):
        gate = PredictionGate(self.codes, self.config)
        for _ in range(6):
            gate.update(self._probs(0))
        self.assertEqual(gate.committed, "verse")
        gate.update(self._probs(1))
        self.assertEqual(gate.committed, "verse")
        gate.update(self._probs(1))
        self.assertEqual(gate.committed, "verse")
        result = gate.update(self._probs(1))
        self.assertEqual(result.label, "chorus")

    def test_short_hold_blocks_rapid_switching(self):
        config = GateConfig(confidence_threshold=0.7, margin_threshold=0.15,
                            acquire_frames=1, switch_frames=1, min_hold_frames=10,
                            expire_frames=4)
        gate = PredictionGate(self.codes, config)
        gate.update(self._probs(0))
        self.assertEqual(gate.committed, "verse")
        for _ in range(8):
            gate.update(self._probs(1))
        self.assertEqual(gate.committed, "verse")

    def test_missing_input_expires_the_cue(self):
        gate = PredictionGate(self.codes, self.config)
        for _ in range(4):
            gate.update(self._probs(0))
        self.assertEqual(gate.committed, "verse")
        for _ in range(self.config.expire_frames):
            result = gate.update(None)
        self.assertIsNone(result.label)

    def test_all_zero_probabilities_are_handled(self):
        gate = PredictionGate(self.codes, self.config)
        self.assertIsNone(gate.update(np.zeros(4)).label)

    def test_wrong_probability_length_is_rejected(self):
        gate = PredictionGate(self.codes, self.config)
        with self.assertRaises(ValueError):
            gate.update(np.zeros(3))


class HandSelectionTests(unittest.TestCase):
    @staticmethod
    def _hand(index, centre, score, side, size=0.2):
        points = np.zeros((21, 3))
        points[:, 0] = centre[0] + np.linspace(0, size, 21)
        points[:, 1] = centre[1]
        return {"index": index, "landmarks": points, "score": score,
                "handedness": side, "center": centre, "size": size}

    def test_prefers_highest_score_on_first_frame(self):
        selector = HandSelector()
        hands = [self._hand(0, (0.2, 0.5), 0.6, "Left"),
                 self._hand(1, (0.8, 0.5), 0.95, "Right")]
        self.assertEqual(selector.select(hands)["index"], 1)

    def test_keeps_the_same_hand_when_two_are_visible(self):
        selector = HandSelector()
        hands = [self._hand(0, (0.2, 0.5), 0.6, "Left"),
                 self._hand(1, (0.8, 0.5), 0.95, "Right")]
        selector.select(hands)
        hands[1]["score"] = 0.3
        self.assertEqual(selector.select(hands)["index"], 1)

    def test_handedness_is_preferred(self):
        selector = HandSelector()
        selector.select([self._hand(0, (0.3, 0.5), 0.9, "Left")])
        hands = [self._hand(0, (0.3, 0.5), 0.9, "Left"),
                 self._hand(1, (0.31, 0.51), 0.99, "Right")]
        self.assertEqual(selector.select(hands)["index"], 0)

    def test_reacquires_after_a_large_jump(self):
        selector = HandSelector(max_jump=0.1)
        selector.select([self._hand(0, (0.1, 0.1), 0.9, "Left")])
        hands = [self._hand(0, (0.9, 0.9), 0.5, "Left"),
                 self._hand(1, (0.9, 0.9), 0.8, "Left")]
        self.assertEqual(selector.select(hands)["index"], 1)
        self.assertEqual(selector.reacquisitions, 1)

    def test_empty_detection_clears_anchor(self):
        selector = HandSelector()
        selector.select([self._hand(0, (0.4, 0.4), 0.9, "Left")])
        self.assertIsNone(selector.select([]))
        self.assertIsNone(selector.previous)


class AugmentationTests(unittest.TestCase):
    def _windows(self):
        base = np.stack([synthetic_pose(i).reshape(-1) for i in range(6)])
        return np.stack([base, base + 0.001, base + 0.002]), np.array([1, 2, 3])

    def test_augmentation_does_not_touch_the_source(self):
        windows, labels = self._windows()
        original = windows.copy()
        augment_batch(windows, labels, np.random.default_rng(0), AugmentConfig(copies=2))
        np.testing.assert_array_equal(windows, original)

    def test_augmentation_shape_and_labels(self):
        windows, labels = self._windows()
        batch_x, batch_y = augment_batch(windows, labels, np.random.default_rng(1),
                                         AugmentConfig(copies=2))
        self.assertEqual(batch_x.shape, (len(windows) * 3, 6, FEATURE_DIM))
        self.assertEqual(len(batch_y), len(batch_x))
        np.testing.assert_array_equal(batch_y[:len(labels)], labels)

    def test_augmentation_is_reproducible_with_a_seed(self):
        windows, labels = self._windows()
        config = AugmentConfig(copies=1, mirror_prob=0.0)
        first, _ = augment_batch(windows, labels, np.random.default_rng(7), config)
        second, _ = augment_batch(windows, labels, np.random.default_rng(7), config)
        np.testing.assert_allclose(first, second)

    def test_mirroring_is_off_by_default(self):
        self.assertEqual(AugmentConfig().mirror_prob, 0.0)

    def test_augmented_window_keeps_shape(self):
        window = np.stack([synthetic_pose(i).reshape(-1) for i in range(50)])
        rng = np.random.default_rng(3)
        for _ in range(5):
            augmented = augment_window(window, rng, AugmentConfig(copies=1, jitter_sigma=0.0))
            self.assertEqual(augmented.shape, window.shape)

    def test_transition_windows_transition_into_their_label(self):
        pool = {0: [synthetic_pose(i).reshape(-1) for i in range(3)],
                1: [synthetic_pose(i + 10).reshape(-1) for i in range(3)]}
        by_class = {label: np.stack([np.tile(p, (50, 1)) for p in poses])
                    for label, poses in pool.items()}

        windows, labels = transition_windows(by_class, np.random.default_rng(0), copies_per_pair=2)
        self.assertGreater(len(windows), 0)
        self.assertEqual(set(labels.tolist()), {0, 1})

        for window, label in zip(windows, labels):
            head, tail = window[0], window[-1]
            self.assertTrue(any(np.allclose(tail, p) for p in pool[label]),
                            "window should end in its labelled gesture")
            self.assertTrue(any(np.allclose(head, p) for p in pool[1 - label]),
                            "window should start in the previous gesture")


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = os.path.join(ROOT, "MP_Data")
        cls.codes = ["verse", "chorus", "bridge", "medley"]
        if not os.path.isdir(os.path.join(cls.data, cls.codes[0])):
            raise unittest.SkipTest("MP_Data is not available")

    def test_grouped_folds_do_not_overlap_and_cover_everything(self):
        keys = sequence_index(self.data, self.codes, 50)
        folds = grouped_folds(keys, n_splits=5, seed=0)
        seen = set()
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            seen.update(test)
        self.assertEqual(len(seen), len(keys))

    def test_temporal_block_holds_out_the_tail_of_every_class(self):
        keys = sequence_index(self.data, self.codes, 50)
        train, held_out = temporal_blocked_split(keys, holdout_per_class=10)
        self.assertEqual(len(held_out), 40)
        for _, sequence in held_out:
            self.assertGreaterEqual(sequence, 40)
        self.assertFalse(set(train) & set(held_out))


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = os.path.join(ROOT, "MP_Data")
        cls.codes = ["verse", "chorus", "bridge", "medley"]
        if not os.path.isdir(os.path.join(cls.data, cls.codes[0])):
            raise unittest.SkipTest("MP_Data is not available")

    def test_dataset_builds_v2_features_and_caches(self):
        dataset = HandGestureDataset(self.data, self.codes, 50, 50, feature_version="v2",
                                     repair=True, min_valid_frames=0)
        self.assertEqual(len(dataset), 200)
        features, label = dataset[0]
        self.assertEqual(tuple(features.shape), (50, FEATURE_DIM))
        self.assertIn(int(label), range(4))
        sample = dataset.samples[0]
        self.assertIs(dataset.raw_window(sample["code"], sample["sequence"]),
                      dataset.raw_window(sample["code"], sample["sequence"]))

    def test_min_valid_frames_filter_skips_broken_recordings(self):
        # medley/0 lost 26 of its 50 frames during collection
        dataset = HandGestureDataset(self.data, self.codes, 50, 50,
                                     sequences=[(3, 0), (0, 0)], min_valid_frames=30)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.skipped, 1)

    def test_dataset_can_add_transition_windows(self):
        selection = [(0, 0), (1, 0)]
        base = HandGestureDataset(self.data, self.codes, 50, 50, sequences=selection)
        with_transitions = HandGestureDataset(self.data, self.codes, 50, 50,
                                              sequences=selection, add_transitions=1)
        self.assertGreater(len(with_transitions), len(base))
        self.assertEqual(sum(1 for s in with_transitions.samples if s["kind"] == "transition"), 2)


class CheckpointTests(unittest.TestCase):
    def test_round_trip_preserves_feature_version(self):
        model = build_hand_gesture_transformer({"input_dim": FEATURE_DIM, "max_seq_len": 50},
                                               num_classes=4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pth")
            save_checkpoint(path, model, None, ["verse", "chorus", "bridge", "medley"], 50, "v2",
                            model_config=model.config, metrics={"accuracy": 0.9})
            loaded = load_model(path, device="cpu")
            self.assertEqual(loaded.feature_version, "v2")
            self.assertEqual(len(loaded.codes), 4)
            self.assertFalse(loaded.is_legacy)
            self.assertEqual(loaded.metrics["accuracy"], 0.9)

    def test_refuses_to_overwrite_the_original_checkpoint(self):
        model = build_hand_gesture_transformer({}, num_classes=4)
        with self.assertRaises(ValueError):
            save_checkpoint(LEGACY_CHECKPOINT, model, None, ["a", "b", "c", "d"], 50, "v2",
                            model_config=model.config)

    def test_original_checkpoint_still_loads_as_v1(self):
        path = os.path.join(ROOT, LEGACY_CHECKPOINT)
        if not os.path.exists(path):
            self.skipTest("original checkpoint not present")
        loaded = load_model(path, device="cpu")
        self.assertEqual(loaded.feature_version, "v1")
        self.assertTrue(loaded.is_legacy)
        output = loaded.model(torch.zeros(1, loaded.sequence_len, FEATURE_DIM))
        self.assertEqual(tuple(output.shape), (1, len(loaded.codes)))

    def test_v2_model_forward_shape(self):
        model = build_hand_gesture_transformer({"input_dim": FEATURE_DIM, "max_seq_len": 50},
                                               num_classes=4)
        self.assertEqual(tuple(model(torch.zeros(2, 50, FEATURE_DIM)).shape), (2, 4))


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        metrics_ = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3], ["a", "b", "c", "d"])
        self.assertEqual(metrics_["accuracy"], 1.0)
        self.assertEqual(metrics_["macro_f1"], 1.0)
        self.assertEqual(metrics_["per_class"]["a"]["false_positive_rate"], 0.0)

    def test_confusion_matrix_layout(self):
        metrics_ = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1], ["a", "b", "c", "d"])
        self.assertEqual(metrics_["confusion_matrix"][0][0], 1)
        self.assertEqual(metrics_["confusion_matrix"][0][1], 1)
        self.assertEqual(metrics_["accuracy"], 0.5)

    def test_calibration_error_is_zero_when_perfect_and_certain(self):
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertLess(expected_calibration_error(probabilities, [0, 1]), 1e-6)


if __name__ == "__main__":
    unittest.main()

