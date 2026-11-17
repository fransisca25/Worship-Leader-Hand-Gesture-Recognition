"""Classification metrics shared by training and evaluation.

Keeping them in one place guarantees that the numbers printed during training
and the numbers in the evaluation report are computed the same way.
"""

import numpy as np


def classification_metrics(y_true, y_pred, codes):
    """Accuracy, macro-F1 and the full per-class report for integer labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(codes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for truth, guess in zip(y_true, y_pred):
        cm[int(truth), int(guess)] += 1

    per_class, f1_scores = {}, []
    for index, code in enumerate(codes):
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum() - tp)
        fn = int(cm[index, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        per_class[code] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(cm[index, :].sum()),
            "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        }

    return {
        "accuracy": round(float((y_true == y_pred).mean()) if len(y_true) else 0.0, 4),
        "macro_f1": round(float(np.mean(f1_scores)), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "n_samples": int(len(y_true)),
    }


def format_metrics(metrics, codes):
    """Human readable one-block summary used in logs and in the console."""
    lines = [
        "accuracy=%.4f  macro_f1=%.4f  n=%d"
        % (metrics["accuracy"], metrics["macro_f1"], metrics["n_samples"]),
        "            " + "  ".join("%-8s" % code for code in codes),
        "precision   " + "  ".join("%-8.3f" % metrics["per_class"][c]["precision"] for c in codes),
        "recall      " + "  ".join("%-8.3f" % metrics["per_class"][c]["recall"] for c in codes),
        "f1          " + "  ".join("%-8.3f" % metrics["per_class"][c]["f1"] for c in codes),
        "confusion   " + str(metrics["confusion_matrix"]),
    ]
    return "\n".join(lines)


def expected_calibration_error(probabilities, labels, n_bins=10):
    """Top-label expected calibration error (lower is better calibrated)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if len(probabilities) == 0:
        return 0.0

    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)

    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    total = len(confidence)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        bin_confidence = confidence[mask].mean()
        bin_accuracy = correct[mask].mean()
        error += (mask.sum() / total) * abs(bin_confidence - bin_accuracy)
    return round(float(error), 4)
