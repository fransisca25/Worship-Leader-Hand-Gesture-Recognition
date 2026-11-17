"""Reproducible offline evaluation for the gesture pipeline.

Everything here runs on the recorded dataset and on synthetic degradations of
it, so the same command can be pointed at the original checkpoint and at the
enhanced one to produce a like-for-like comparison:

    python evaluate.py --checkpoint trained_model/wl_hand_gesture_transformer.pth
    python evaluate.py --checkpoint trained_model/wl_hand_gesture_transformer_v2.pth

Sections
--------
1. dataset audit          missing detections, per-landmark stability
2. clean metrics          accuracy / macro-F1 / per-class report / confusion
3. perturbations          translation, scale, rotation, jitter, aspect, occlusion
4. rejection              does the gate abstain on empty / ambiguous input?
5. threshold sweep        confidence x margin trade-off for the gate
6. transitions            end-to-end latency and flicker of a gesture change
7. latency                classifier inference time on this machine

What this cannot measure is stated in the report: real lighting, real users and
real cameras need new recordings.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from checkpoint import load_model, resolve_checkpoint
from dataset_loader import HandGestureDataset
from metrics import classification_metrics, expected_calibration_error, format_metrics
from temporal import GateConfig, PredictionGate
from tool.kp_extractor import (
    FEATURE_DIM,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    WRIST,
    count_valid_frames,
    window_features,
)

DEFAULT_DATA = "MP_Data"


# ------------------------------------------------------------------------ helpers
def load_windows(data_path, codes, no_sequences, sequence_len):
    """Load every recording as a raw (T, 63) window, without any preprocessing."""
    dataset = HandGestureDataset(
        data_path=data_path, codes=codes, no_sequences=no_sequences,
        sequence_len=sequence_len, feature_version="v1", augment=None, repair=False,
    )
    raw = [dataset.raw_window(sample["code"], sample["sequence"]) for sample in dataset.samples]
    labels = np.array([sample["label"] for sample in dataset.samples], dtype=int)
    keys = [(sample["code"], sample["sequence"]) for sample in dataset.samples]
    return np.stack(raw), labels, keys


def featurize(raw, feature_version, repair=True):
    """Turn raw windows into model input for the requested feature version."""
    return np.stack([window_features(window, feature_version, repair=repair) for window in raw]).astype(np.float32)


def predict(model, features, device, batch_size=32):
    """Softmax probabilities for a batch of windows."""
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(features[start:start + batch_size])).to(device)
            chunks.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 4))


def dataset_audit(raw, labels, codes):
    """Missing detections and landmark stability of the recordings themselves."""
    valid = np.stack([count_valid_frames(window) for window in raw])
    total_frames = int(raw.shape[0] * raw.shape[1])
    valid_frames = int(valid.sum())

    scale = np.linalg.norm(
        raw.reshape(-1, raw.shape[1], NUM_LANDMARKS, 3)[:, :, MIDDLE_MCP, :2]
        - raw.reshape(-1, raw.shape[1], NUM_LANDMARKS, 3)[:, :, WRIST, :2],
        axis=2,
    )
    delta = np.abs(np.diff(raw, axis=1))
    relative = delta / np.where(scale[:, 1:] < 1e-6, np.nan, scale[:, 1:])[:, :, None]

    return {
        "n_windows": int(raw.shape[0]),
        "window_length": int(raw.shape[1]),
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "missing_frames": total_frames - valid_frames,
        "missing_frame_rate": round(1.0 - valid_frames / max(total_frames, 1), 4),
        "windows_with_missing_frames": int((valid < raw.shape[1]).sum()),
        "fully_missing_windows": int((valid == 0).sum()),
        "min_valid_frames": int(valid.min()),
        "per_class_missing_rate": {
            code: round(1.0 - float(valid[labels == index].sum()) /
                        max(float(raw.shape[1] * max((labels == index).sum(), 1)), 1.0), 4)
            for index, code in enumerate(codes)
        },
        "landmark_jitter_relative_p50": round(float(np.nanpercentile(relative, 50)), 4),
        "landmark_jitter_relative_p95": round(float(np.nanpercentile(relative, 95)), 4),
        "landmark_jitter_relative_p99": round(float(np.nanpercentile(relative, 99)), 4),
        "jitter_is_relative_to_wrist_middle_mcp": True,
    }


def clean_evaluation(loaded, raw, labels, repair, codes):
    """Window-level accuracy of the checkpoint on untouched recordings."""
    features = featurize(raw, loaded.feature_version, repair=repair)
    probabilities = predict(loaded.model, features, loaded.device)
    metrics = classification_metrics(labels, probabilities.argmax(axis=1), codes)
    metrics["mean_confidence"] = round(float(probabilities.max(axis=1).mean()), 4)
    metrics["p10_confidence"] = round(float(np.percentile(probabilities.max(axis=1), 10)), 4)
    metrics["expected_calibration_error"] = expected_calibration_error(probabilities, labels)
    return metrics, probabilities, features



# ------------------------------------------------------------------ perturbations
def _cube(raw):
    arr = np.asarray(raw, dtype=np.float64)
    return arr.reshape(len(arr), arr.shape[1], NUM_LANDMARKS, 3).copy()


def _flat(cube):
    return cube.reshape(len(cube), cube.shape[1], FEATURE_DIM)


def perturbation_suite(loaded, raw, labels, repair, codes):
    """Accuracy under realistic landmark distortions (applied to the raw windows)."""
    results = {}

    def run(name, batch):
        features = featurize(batch, loaded.feature_version, repair=repair)
        probabilities = predict(loaded.model, features, loaded.device)
        metrics = classification_metrics(labels, probabilities.argmax(axis=1), codes)
        results[name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mean_confidence": round(float(probabilities.max(axis=1).mean()), 4),
        }

    run("identity", raw)

    cube = _cube(raw)
    cube[..., 0] += 0.10
    cube[..., 1] += 0.15
    run("translate_x0.10_y0.15", _flat(cube))

    for factor in (0.7, 1.4):
        cube = _cube(raw)
        centre = cube[..., :2].mean(axis=2, keepdims=True)
        cube[..., :2] = centre + (cube[..., :2] - centre) * factor
        run("scale_%.1fx" % factor, _flat(cube))

    for degrees in (20, 30):
        cube = _cube(raw)
        centre = cube[..., :2].mean(axis=2, keepdims=True)
        angle = np.deg2rad(degrees)
        shifted = cube[..., :2] - centre
        cube[..., 0] = centre[..., 0] + shifted[..., 0] * np.cos(angle) - shifted[..., 1] * np.sin(angle)
        cube[..., 1] = centre[..., 1] + shifted[..., 0] * np.sin(angle) + shifted[..., 1] * np.cos(angle)
        run("rotate_%ddeg" % degrees, _flat(cube))

    for axis, factor in (("x", 1.333), ("y", 1.333)):
        cube = _cube(raw)
        cube[..., 0 if axis == "x" else 1] *= factor
        run("aspect_%s_times_%.3f" % (axis, factor), _flat(cube))

    rng = np.random.default_rng(0)
    cube = _cube(raw) + rng.normal(0.0, 0.02, _cube(raw).shape)
    run("jitter_sigma0.020", _flat(cube))

    for name, indices in (("occlude_index_finger", [5, 6, 7, 8]), ("occlude_thumb", [1, 2, 3, 4])):
        cube = _cube(raw)
        cube[:, :, indices, :] = 0.0
        run(name, _flat(cube))

    cube = _cube(raw)
    cube[:, 20:35, :, :] = 0.0
    run("detection_gap_15_of_50", _flat(cube))

    zeros = np.zeros((64, raw.shape[1], FEATURE_DIM), dtype=np.float64)
    features = featurize(zeros, loaded.feature_version, repair=repair)
    probabilities = predict(loaded.model, features, loaded.device)
    results["all_zero_window"] = {
        "accuracy": None,
        "macro_f1": None,
        "mean_confidence": round(float(probabilities.max(axis=1).mean()), 4),
        "most_common_prediction": codes[int(np.bincount(probabilities.argmax(axis=1)).argmax())],
        "note": "meaningless input: the live gate must reject it (see rejection section)",
    }
    return results


# ---------------------------------------------------------------------- replaying
def replay_probabilities(loaded, frames, repair=True, smooth_alpha=1.0, min_valid_frames=35):
    """Feed frames through the live buffer and return per-frame probabilities."""
    from preprocessing import LandmarkBuffer
    from tool.kp_extractor import is_valid_frame

    buffer = LandmarkBuffer(
        length=loaded.sequence_len,
        min_valid_frames=min(int(min_valid_frames), loaded.sequence_len),
        smooth_alpha=smooth_alpha,
    )
    outputs = []
    for frame in frames:
        flat = np.asarray(frame, dtype=np.float64).reshape(-1)
        buffer.push(flat.reshape(NUM_LANDMARKS, 3) if is_valid_frame(flat) else None)
        window = buffer.window()
        if window is None:
            outputs.append(None)
            continue
        features = window_features(window, loaded.feature_version, repair=repair)[None]
        outputs.append(predict(loaded.model, features, loaded.device)[0])
    return outputs



# ------------------------------------------------------------- rejection / gate
def settle_gate(probabilities, code_names, config, repeats=10):
    """Feed one probability row through a fresh gate ``repeats`` times."""
    labels_out = []
    for row in probabilities:
        gate = PredictionGate(code_names, config)
        result = None
        for _ in range(repeats):
            result = gate.update(row)
        labels_out.append(result.label)
    return labels_out


def build_negative_sets(raw, labels, codes, rng, per_set=24):
    """Inputs that must *not* produce a cue."""
    sizes = len(codes)
    empty = np.zeros((per_set, raw.shape[1], FEATURE_DIM), dtype=np.float64)

    gap = raw[:per_set].copy()
    gap[:, raw.shape[1] // 3:, :] = 0.0     # hand disappears for the rest of the window

    mixtures = []
    per_pair = max(per_set // (sizes * (sizes - 1)), 1)
    for a in range(sizes):
        for b in range(sizes):
            if a == b:
                continue
            index_a = np.flatnonzero(labels == a)
            index_b = np.flatnonzero(labels == b)
            for _ in range(per_pair):
                left = raw[int(rng.choice(index_a))]
                right = raw[int(rng.choice(index_b))]
                half = raw.shape[1] // 2
                mixtures.append(np.concatenate([left[:half], right[half:]], axis=0))

    return {
        "empty_no_hand": empty,
        "long_detection_gap": gap,
        "ambiguous_transition": np.stack(mixtures),
    }


def rejection_evaluation(loaded, raw, labels, codes, config, repair, per_set=24,
                          live_samples=6, repeats=3):
    """Measure false cues on empty/ambiguous input, model-only and end-to-end.

    Live replays repeat the sample ``repeats`` times so the gate has time to
    acquire a gesture; without that a 50-frame stream would never produce a
    single stable decision.
    """
    rng = np.random.default_rng(0)
    negatives = build_negative_sets(raw, labels, codes, rng, per_set=per_set)
    report = {}

    for name, batch in negatives.items():
        probabilities = predict(loaded.model, featurize(batch, loaded.feature_version, repair=repair),
                                loaded.device)
        labels_out = settle_gate(probabilities, codes, config)
        model_false_cue = float(np.mean([label is not None for label in labels_out]))

        live_abstain = []
        if name != "ambiguous_transition":
            for sample in batch[:live_samples]:
                stream = replay_probabilities(loaded, np.tile(sample, (repeats, 1)), repair=repair)
                gate = PredictionGate(codes, config)
                result = None
                for row in stream:
                    result = gate.update(row)
                live_abstain.append(result.label is None)

        report[name] = {
            "n": int(len(batch)),
            "model_mean_confidence": round(float(probabilities.max(axis=1).mean()), 4),
            "model_false_cue_rate_gate_only": round(model_false_cue, 3),
            "live_abstain_rate": round(float(np.mean(live_abstain)), 3) if live_abstain else None,
            "live_samples": len(live_abstain),
        }

    positive_probs = predict(loaded.model, featurize(raw[:per_set], loaded.feature_version, repair=repair),
                             loaded.device)
    positive_labels = settle_gate(positive_probs, codes, config)
    accepted = np.array([label is not None for label in positive_labels])
    correct = np.array([label == codes[int(truth)] for label, truth in zip(positive_labels, labels)])

    live_accepted, live_correct = [], []
    for index, window in enumerate(raw[:live_samples]):
        stream = replay_probabilities(loaded, np.tile(window, (repeats, 1)), repair=repair)
        gate = PredictionGate(codes, config)
        result = None
        for row in stream:
            result = gate.update(row)
        live_accepted.append(result.label is not None)
        live_correct.append(result.label == codes[int(labels[index])])

    report["true_gesture"] = {
        "n": int(len(accepted)),
        "acceptance_rate": round(float(accepted.mean()), 3),
        "accuracy_when_accepted": round(float(correct[accepted].mean()) if accepted.any() else 0.0, 3),
        "false_suppression_rate": round(float(np.mean([not a for a in accepted])), 3),
        "live_acceptance_rate": round(float(np.mean(live_accepted)), 3),
        "live_accuracy_when_accepted": round(
            float(np.mean([c for a, c in zip(live_accepted, live_correct) if a]))
            if any(live_accepted) else 0.0, 3),
    }
    return report


def threshold_sweep(loaded, raw, labels, codes, repair):
    """Confidence x margin trade-off: acceptance on real gestures vs abstention on noise."""
    positive_probs = predict(loaded.model, featurize(raw, loaded.feature_version, repair=repair),
                             loaded.device)

    rng = np.random.default_rng(0)
    negatives = build_negative_sets(raw, labels, codes, rng, per_set=12)
    negative_probs = np.concatenate([
        predict(loaded.model, featurize(batch, loaded.feature_version, repair=repair), loaded.device)
        for batch in negatives.values()
    ], axis=0)

    rows = []
    for confidence in (0.5, 0.6, 0.7, 0.8, 0.9):
        for margin in (0.0, 0.1, 0.2):
            config = GateConfig(confidence_threshold=confidence, margin_threshold=margin)
            positive_labels = settle_gate(positive_probs, codes, config, repeats=6)
            negative_labels = settle_gate(negative_probs, codes, config, repeats=6)
            accepted = np.array([label is not None for label in positive_labels])
            correct = np.array([label == codes[int(truth)]
                                for label, truth in zip(positive_labels, labels)])
            rows.append({
                "confidence_threshold": confidence,
                "margin_threshold": margin,
                "positive_acceptance": round(float(accepted.mean()), 3),
                "accuracy_when_accepted": round(
                    float(correct[accepted].mean()) if accepted.any() else 0.0, 3),
                "negative_abstain": round(float(np.mean([label is None for label in negative_labels])), 3),
            })
    return rows


# ---------------------------------------------------------------- transitions
def transition_evaluation(loaded, raw, labels, codes, repair, config, trials=2, fps=30):
    """End-to-end latency of a gesture change, original behaviour vs gated pipeline."""
    by_label = {index: np.flatnonzero(labels == index) for index in range(len(codes))}
    rows = []

    for a in range(len(codes)):
        for b in range(len(codes)):
            if a == b:
                continue
            for trial in range(trials):
                left = raw[int(by_label[a][(trial * 7) % len(by_label[a])])]
                right = raw[int(by_label[b][((trial * 5) + 3) % len(by_label[b])])]
                # 50 frames of A followed by 100 frames of B: enough room for the
                # window (50 frames) plus the gate hysteresis to settle.
                stream = np.concatenate([left, right, right], axis=0)
                switch = len(left)
                probabilities = replay_probabilities(loaded, stream, repair=repair)

                raw_labels = [None if row is None else int(row.argmax()) for row in probabilities]
                gate = PredictionGate(codes, config)
                gated = [gate.update(row).label for row in probabilities]

                def first_hit(sequence, target):
                    for position in range(switch, len(sequence)):
                        if sequence[position] == target:
                            return position
                    return None

                def flicker(sequence, start):
                    if start is None or start >= len(sequence) - 1:
                        return 0
                    tail = sequence[start:]
                    return int(sum(1 for x, y in zip(tail[1:], tail[:-1]) if x != y))

                raw_hit = first_hit(raw_labels, b)
                gate_hit = first_hit(gated, codes[b])
                rows.append({
                    "pair": "%s->%s" % (codes[a], codes[b]),
                    "trial": trial,
                    "raw_latency_frames": None if raw_hit is None else raw_hit - switch,
                    "gated_latency_frames": None if gate_hit is None else gate_hit - switch,
                    "raw_flicker": flicker(raw_labels, raw_hit),
                    "gated_flicker": flicker(gated, gate_hit),
                    "raw_final_correct": bool(raw_labels[-1] == b),
                    "gated_final_correct": bool(gated[-1] == codes[b]),
                })

    summary = {}
    for mode in ("raw", "gated"):
        latencies = [row["%s_latency_frames" % mode] for row in rows
                     if row["%s_latency_frames" % mode] is not None]
        summary[mode] = {
            "n_transitions": len(rows),
            "median_latency_frames": round(float(np.median(latencies)), 1) if latencies else None,
            "p90_latency_frames": round(float(np.percentile(latencies, 90)), 1) if latencies else None,
            "median_latency_ms": round(float(np.median(latencies)) * 1000.0 / fps, 0) if latencies else None,
            "never_reached_target_rate": round(float(np.mean(
                [row["%s_latency_frames" % mode] is None for row in rows])), 3),
            "mean_flicker_after_hit": round(float(np.mean([row["%s_flicker" % mode] for row in rows])), 2),
            "final_label_correct_rate": round(float(np.mean(
                [row["%s_final_correct" % mode] for row in rows])), 3),
        }
    return {"summary": summary, "rows": rows}


# -------------------------------------------------------------------- latency
def latency_benchmark(loaded, features, repetitions=60):
    """Classifier-only inference time on this machine (batch size 1, like the app)."""
    model = loaded.model
    model.eval()
    sample = torch.from_numpy(np.ascontiguousarray(features[:1])).to(loaded.device)
    timings = []
    with torch.no_grad():
        for _ in range(repetitions):
            if loaded.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(sample)
            if loaded.device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    timings = np.array(timings)
    return {
        "device": loaded.device,
        "mean_ms": round(float(timings.mean()), 3),
        "median_ms": round(float(np.median(timings)), 3),
        "p95_ms": round(float(np.percentile(timings, 95)), 3),
        "implied_fps": round(1000.0 / float(np.median(timings)), 1),
        "note": "classifier only; MediaPipe detection dominates end-to-end latency",
    }




# ----------------------------------------------------------------------- main
def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a gesture checkpoint offline.")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint to evaluate (default: enhanced model, then original)")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--no-sequences", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-repair", action="store_true",
                        help="evaluate without tracking-hold repair (original behaviour)")
    parser.add_argument("--smooth-alpha", type=float, default=1.0,
                        help="landmark EMA used while replaying streams (1.0 disables)")
    parser.add_argument("--conf-threshold", type=float, default=0.70)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--fps", type=float, default=30.0, help="assumed capture rate")
    parser.add_argument("--trials", type=int, default=2, help="transition streams per class pair")
    parser.add_argument("--skip-rejection", action="store_true")
    parser.add_argument("--skip-transitions", action="store_true")
    parser.add_argument("--out", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.checkpoint:
        checkpoint_path, used_fallback = args.checkpoint, False
    else:
        checkpoint_path, used_fallback = resolve_checkpoint()

    loaded = load_model(checkpoint_path, device=args.device)
    codes = loaded.codes
    print("checkpoint : %s%s" % (checkpoint_path, "  (fallback to original model)" if used_fallback else ""))
    print("features   : %s %s" % (loaded.feature_version, loaded.feature_config))
    print("model      : input_dim=%s d_model=%s layers=%s heads=%s"
          % (loaded.model.config["input_dim"], loaded.model.config["d_model"],
             loaded.model.config["num_layers"], loaded.model.config["num_heads"]))
    print("classes    : %s\n" % ", ".join(codes))

    raw, labels, keys = load_windows(args.data, codes, args.no_sequences, loaded.sequence_len)
    repair = not args.no_repair
    gate_config = GateConfig(confidence_threshold=args.conf_threshold, margin_threshold=args.margin)

    report = {
        "checkpoint": checkpoint_path,
        "feature_version": loaded.feature_version,
        "feature_config": loaded.feature_config,
        "model_config": loaded.model.config,
        "codes": codes,
        "repair_missing_frames": repair,
        "gate_config": gate_config.__dict__,
        "dataset": dataset_audit(raw, labels, codes),
    }
    print("[1] dataset audit")
    print("    windows=%d  missing frames=%.2f%%  windows with gaps=%d  jitter p95=%.3f (hand scale)"
          % (report["dataset"]["n_windows"], 100 * report["dataset"]["missing_frame_rate"],
             report["dataset"]["windows_with_missing_frames"],
             report["dataset"]["landmark_jitter_relative_p95"]))

    clean, probabilities, features = clean_evaluation(loaded, raw, labels, repair, codes)
    report["clean"] = clean
    print("\n[2] clean window metrics (same-session data, may include training windows)")
    print(format_metrics(clean, codes))
    print("    mean confidence %.3f | ECE %.3f"
          % (clean["mean_confidence"], clean["expected_calibration_error"]))

    report["perturbations"] = perturbation_suite(loaded, raw, labels, repair, codes)
    print("\n[3] perturbation robustness (accuracy)")
    for name, values in report["perturbations"].items():
        if values["accuracy"] is None:
            print("    %-26s n/a (predicted '%s' at %.2f confidence)"
                  % (name, values["most_common_prediction"], values["mean_confidence"]))
        else:
            print("    %-26s %.4f" % (name, values["accuracy"]))

    if not args.skip_rejection:
        report["rejection"] = rejection_evaluation(loaded, raw, labels, codes, gate_config, repair)
        print("\n[4] rejection of empty / ambiguous input")
        for name, values in report["rejection"].items():
            if name == "true_gesture":
                print("    %-22s model accept %.2f (acc %.2f) | live accept %.2f (acc %.2f)"
                      % (name, values["acceptance_rate"], values["accuracy_when_accepted"],
                         values["live_acceptance_rate"], values["live_accuracy_when_accepted"]))
            else:
                abstain = values["live_abstain_rate"]
                print("    %-22s model false-cue %.2f | live abstain %s"
                      % (name, values["model_false_cue_rate_gate_only"],
                         "n/a" if abstain is None else "%.2f" % abstain))

    report["threshold_sweep"] = threshold_sweep(loaded, raw, labels, codes, repair)
    print("\n[5] gate threshold sweep (acceptance / abstention)")
    for row in report["threshold_sweep"]:
        print("    conf %.1f margin %.1f -> accept %.3f  acc %.3f  abstain %.3f"
              % (row["confidence_threshold"], row["margin_threshold"],
                 row["positive_acceptance"], row["accuracy_when_accepted"], row["negative_abstain"]))

    if not args.skip_transitions:
        report["transitions"] = transition_evaluation(
            loaded, raw, labels, codes, repair, gate_config, trials=args.trials, fps=args.fps
        )
        print("\n[6] gesture transition behaviour")
        for mode, values in report["transitions"]["summary"].items():
            print("    %-6s median %s frames (%s ms) | p90 %s | never %.2f | flicker %.2f | final correct %.2f"
                  % (mode, values["median_latency_frames"], values["median_latency_ms"],
                     values["p90_latency_frames"], values["never_reached_target_rate"],
                     values["mean_flicker_after_hit"], values["final_label_correct_rate"]))

    report["latency"] = latency_benchmark(loaded, features)
    print("\n[7] classifier latency: median %.2f ms (p95 %.2f ms, ~%.0f fps on %s)"
          % (report["latency"]["median_ms"], report["latency"]["p95_ms"],
             report["latency"]["implied_fps"], report["latency"]["device"]))

    out_path = args.out or os.path.join(
        "trained_model", "evaluation_%s.json" % os.path.splitext(os.path.basename(checkpoint_path))[0]
    )
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print("\nreport saved to %s" % out_path)
    return report


if __name__ == "__main__":
    main()
