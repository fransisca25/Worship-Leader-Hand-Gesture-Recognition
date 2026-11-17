"""Training entry point for the worship-leader gesture transformer.

The model, the loss and the optimiser are the original ones; the training
protocol is what changed:

* reproducible seeds,
* a real validation set (temporal block by default) used for model selection
  and early stopping instead of reporting the epoch-50 weights,
* grouped cross-validation by recording sequence, which is the only honest
  number when the whole dataset comes from one session,
* label smoothing, optional class weights and a learning-rate schedule,
* per-class precision / recall / F1 and a confusion matrix written to disk,
* on-the-fly augmentation and transition windows (training only, the source
  dataset in ``MP_Data`` is never modified),
* the original checkpoint is never overwritten: the enhanced model is saved as
  ``trained_model/wl_hand_gesture_transformer_v2.pth``.

Usage
-----
    python train.py                     # enhanced model, temporal hold-out
    python train.py --mode cv           # 5-fold grouped CV metrics only
    python train.py --feature-version v1 --augment none --no-repair
                                        # reproduce the original baseline
"""

import argparse
import copy
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from augmentation import AugmentConfig  # noqa: E402
from checkpoint import DEFAULT_CHECKPOINT, save_checkpoint  # noqa: E402
from dataset_loader import (  # noqa: E402
    HandGestureDataset,
    create_dataloaders,
    grouped_folds,
    sequence_index,
    temporal_blocked_split,
)
from metrics import classification_metrics, format_metrics  # noqa: E402
from tool.dir_generator import codes as DEFAULT_CODES  # noqa: E402
from tool.dir_generator import no_sequences as DEFAULT_NO_SEQUENCES  # noqa: E402
from tool.dir_generator import sequence_len as DEFAULT_SEQUENCE_LEN  # noqa: E402
from tool.kp_extractor import DEFAULT_FEATURE_VERSION, FEATURE_DIM  # noqa: E402
from transformer import DEFAULT_MODEL_CONFIG, build_hand_gesture_transformer  # noqa: E402

SAVE_DIR = "trained_model"


# --------------------------------------------------------------------------- utils
def set_seed(seed):
    """Seed every RNG that can influence a training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference="auto"):
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_augmentation(preset, feature_version):
    """Turn a CLI augmentation preset into an :class:`AugmentConfig` (or ``None``)."""
    if preset == "none":
        return None

    if preset == "geo":
        config = AugmentConfig(
            copies=1, jitter_sigma=0.0, frame_hold_prob=0.0, landmark_dropout_prob=0.0,
            tilt_x_deg=8.0, tilt_y_deg=8.0, in_plane_rotation_deg=12.0,
            scale_range=0.10, translate=0.03, time_shift=2,
        )
    elif preset == "full":
        config = AugmentConfig(
            copies=2, jitter_sigma=0.004, frame_hold_prob=0.05, landmark_dropout_prob=0.10,
            tilt_x_deg=10.0, tilt_y_deg=10.0, in_plane_rotation_deg=12.0,
            scale_range=0.12, translate=0.04, time_shift=2,
        )
    else:
        raise ValueError("unknown augmentation preset %r" % preset)

    if feature_version == "v2":
        # translation, scale and in-plane rotation are removed by the v2
        # normalisation, so those copies would train on identical inputs.
        config.scale_range = 0.0
        config.translate = 0.0
        config.in_plane_rotation_deg = 0.0
    return config


def class_weights_from_loader(train_loader, n_classes):
    """Inverse-frequency class weights, or ``None`` when the data is balanced."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for sample in train_loader.dataset.samples:
        counts[int(sample["label"])] += 1
    if counts.min() == 0 or counts.max() / counts.min() < 1.2:
        return None
    weights = counts.sum() / (counts * len(counts))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model, loader, device, criterion=None):
    """Run one pass over ``loader`` and return metrics plus raw predictions."""
    model.eval()
    probabilities, labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            if criterion is not None:
                total_loss += float(criterion(logits, targets).item()) * len(targets)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(targets.cpu().numpy())

    probabilities = np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, 4))
    labels = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=int)
    predictions = probabilities.argmax(axis=1) if len(probabilities) else labels
    return {
        "loss": (total_loss / max(len(labels), 1)) if criterion is not None else None,
        "probabilities": probabilities,
        "labels": labels,
        "predictions": predictions,
    }


def train_model(model, train_loader, val_loader, device, code_names, args,
                log_file=None, description="train"):
    """Train with early stopping on validation macro-F1 and keep the best weights."""
    weights = class_weights_from_loader(train_loader, len(code_names))
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=float(args.label_smoothing),
    )
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 3),
    )

    best = {"macro_f1": -1.0, "epoch": 0, "state": None}
    history = {"epoch": [], "train_loss": [], "train_accuracy": [],
               "val_loss": [], "val_accuracy": [], "val_macro_f1": [], "lr": []}
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0

        for features, targets in tqdm(
            train_loader, desc="%s epoch %d/%d" % (description, epoch, args.epochs), leave=False
        ):
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += len(targets)

        train_loss = running_loss / max(seen, 1)
        train_accuracy = correct / max(seen, 1)

        evaluation = evaluate_model(model, val_loader, device, criterion)
        metrics = classification_metrics(evaluation["labels"], evaluation["predictions"], code_names)
        scheduler.step(metrics["macro_f1"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 5))
        history["train_accuracy"].append(round(train_accuracy, 5))
        history["val_loss"].append(round(float(evaluation["loss"] or 0.0), 5))
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_macro_f1"].append(metrics["macro_f1"])
        history["lr"].append(current_lr)

        improved = metrics["macro_f1"] > best["macro_f1"] + 1e-4
        if improved:
            best = {"macro_f1": metrics["macro_f1"], "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
            stale = 0
        else:
            stale += 1

        print("epoch %3d | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f macro_f1 %.4f | lr %.2e%s"
              % (epoch, train_loss, train_accuracy, float(evaluation["loss"] or 0.0),
                 metrics["accuracy"], metrics["macro_f1"], current_lr, " *" if improved else ""))
        if log_file is not None:
            log_file.write("%d, %.5f, %.5f, %.5f, %.4f, %.4f, %.6f\n" % (
                epoch, train_loss, train_accuracy, float(evaluation["loss"] or 0.0),
                metrics["accuracy"], metrics["macro_f1"], current_lr))
            log_file.flush()

        if epoch >= args.min_epochs and stale >= args.patience:
            print("Early stopping at epoch %d (best macro_f1 %.4f at epoch %d)"
                  % (epoch, best["macro_f1"], best["epoch"]))
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    final = evaluate_model(model, val_loader, device, criterion)
    final_metrics = classification_metrics(final["labels"], final["predictions"], code_names)
    return model, history, best, final_metrics


def model_config_for(sequence_len, num_classes, dropout):
    """The model configuration used by every enhanced run (architecture unchanged)."""
    config = dict(DEFAULT_MODEL_CONFIG)
    config.update({
        "input_dim": FEATURE_DIM,
        "max_seq_len": int(sequence_len),
        "num_classes": int(num_classes),
        "dropout": float(dropout),
    })
    return config


def run_final(args):
    """Train the deployable model on a temporal block, save it under a new name."""
    codes = [str(code) for code in DEFAULT_CODES]
    keys = sequence_index(args.data, codes, args.no_sequences)
    train_keys, val_keys = temporal_blocked_split(keys, args.holdout_per_class)

    augment = build_augmentation(args.augment, args.feature_version)
    train_loader, val_loader = create_dataloaders(
        data_path=args.data, codes=codes, no_sequences=args.no_sequences,
        sequence_len=args.sequence_len, batch_size=args.batch_size,
        feature_version=args.feature_version, seed=args.seed, augment=augment,
        add_transitions=args.transitions, train_sequences=train_keys,
        val_sequences=val_keys, repair=not args.no_repair,
        min_valid_frames=args.min_valid_frames,
    )

    print("train windows: %d | validation windows: %d (temporal hold-out of %d sequences/class)"
          % (len(train_loader.dataset), len(val_loader.dataset), args.holdout_per_class))
    if train_loader.dataset.skipped:
        print("skipped %d sequence(s) with too few detected frames" % train_loader.dataset.skipped)

    set_seed(args.seed)
    device = resolve_device(args.device)
    config = model_config_for(args.sequence_len, len(codes), args.dropout)
    model = build_hand_gesture_transformer(config, num_classes=len(codes)).to(device)

    os.makedirs(SAVE_DIR, exist_ok=True)
    log_path = os.path.join(SAVE_DIR, "training_log_v2.txt" if args.feature_version == "v2"
                            else "training_log_v1.txt")

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("epoch, train_loss, train_accuracy, val_loss, val_accuracy, val_macro_f1, lr\n")
        model, history, best, final_metrics = train_model(
            model, train_loader, val_loader, device, codes, args,
            log_file=log_file, description="fit",
        )

    checkpoint_path = args.out or (
        DEFAULT_CHECKPOINT if args.feature_version == "v2"
        else os.path.join(SAVE_DIR, "wl_hand_gesture_transformer_v1_retrained.pth")
    )
    save_checkpoint(
        checkpoint_path, model, None, codes, args.sequence_len, args.feature_version,
        model_config=config,
        train_config={
            "feature_version": args.feature_version, "augment": args.augment,
            "transitions": args.transitions, "repair": not args.no_repair,
            "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
            "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
            "batch_size": args.batch_size, "dropout": args.dropout,
            "split": "temporal_blocked", "holdout_per_class": args.holdout_per_class,
            "train_windows": len(train_loader.dataset),
            "val_windows": len(val_loader.dataset),
        },
        metrics=final_metrics,
        history={**history, "best_epoch": best["epoch"], "best_macro_f1": best["macro_f1"]},
    )

    report_path = os.path.join(SAVE_DIR, "training_metrics_v2.json" if args.feature_version == "v2"
                               else "training_metrics_v1.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({"config": vars(args), "history": history, "best_epoch": best["epoch"],
                   "validation_metrics": final_metrics}, handle, indent=2)

    if not args.no_plot:
        plot_path = os.path.join(SAVE_DIR, "training_plot_v2.png" if args.feature_version == "v2"
                                 else "training_plot_v1.png")
        plot_history(history, plot_path)
        print("training plot saved to %s" % plot_path)

    print("\nValidation (temporal hold-out) results:")
    print(format_metrics(final_metrics, codes))
    print("\nModel saved to %s" % checkpoint_path)
    print("Training log: %s\nMetrics: %s" % (log_path, report_path))
    return checkpoint_path, final_metrics


def plot_history(history, path):
    """Save the training curves (same idea as the original script)."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train Accuracy")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy")
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Accuracy / Macro-F1")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)



def run_cv(args):
    """Measure the model with whole recording sequences held out (grouped CV)."""
    codes = [str(code) for code in DEFAULT_CODES]
    keys = sequence_index(args.data, codes, args.no_sequences)
    folds = grouped_folds(keys, n_splits=args.folds, seed=args.seed)
    augment = build_augmentation(args.augment, args.feature_version)
    device = resolve_device(args.device)

    fold_results, pooled_true, pooled_pred = [], [], []
    fold_accuracies, fold_f1 = [], []

    for number, (train_keys, test_keys) in enumerate(folds, start=1):
        inner_train, inner_val = temporal_blocked_split(train_keys, args.cv_inner_holdout)
        train_loader, inner_val_loader = create_dataloaders(
            data_path=args.data, codes=codes, no_sequences=args.no_sequences,
            sequence_len=args.sequence_len, batch_size=args.batch_size,
            feature_version=args.feature_version, seed=args.seed + number, augment=augment,
            add_transitions=args.transitions, train_sequences=inner_train,
            val_sequences=inner_val, repair=not args.no_repair,
            min_valid_frames=args.min_valid_frames,
        )

        set_seed(args.seed + number)
        config = model_config_for(args.sequence_len, len(codes), args.dropout)
        model = build_hand_gesture_transformer(config, num_classes=len(codes)).to(device)
        model, _, best, _ = train_model(
            model, train_loader, inner_val_loader, device, codes, args,
            description="fold %d" % number,
        )

        test_dataset = HandGestureDataset(
            data_path=args.data, codes=codes, no_sequences=args.no_sequences,
            sequence_len=args.sequence_len, sequences=test_keys,
            feature_version=args.feature_version, augment=None, repair=not args.no_repair,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        evaluation = evaluate_model(model, test_loader, device)
        metrics = classification_metrics(evaluation["labels"], evaluation["predictions"], codes)

        fold_accuracies.append(metrics["accuracy"])
        fold_f1.append(metrics["macro_f1"])
        pooled_true.extend(evaluation["labels"].tolist())
        pooled_pred.extend(evaluation["predictions"].tolist())
        fold_results.append({"fold": number, "best_epoch": best["epoch"],
                             "n_test_windows": metrics["n_samples"], "metrics": metrics})
        print("fold %d held-out windows: %s" % (number, format_metrics(metrics, codes)))

    pooled = classification_metrics(pooled_true, pooled_pred, codes)
    summary = {
        "protocol": "grouped cross-validation, whole recording sequences held out",
        "n_folds": args.folds,
        "folds": fold_results,
        "pooled": pooled,
        "mean_accuracy": round(float(np.mean(fold_accuracies)), 4),
        "std_accuracy": round(float(np.std(fold_accuracies)), 4),
        "mean_macro_f1": round(float(np.mean(fold_f1)), 4),
        "std_macro_f1": round(float(np.std(fold_f1)), 4),
        "config": vars(args),
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = args.out or os.path.join(
        SAVE_DIR, "cv_metrics_v2.json" if args.feature_version == "v2" else "cv_metrics_v1.json"
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nGrouped CV summary (%d folds): accuracy %.4f +/- %.4f | macro-F1 %.4f +/- %.4f"
          % (args.folds, summary["mean_accuracy"], summary["std_accuracy"],
             summary["mean_macro_f1"], summary["std_macro_f1"]))
    print(format_metrics(pooled, codes))
    print("\nCV metrics saved to %s" % path)
    return summary



def build_parser():
    parser = argparse.ArgumentParser(description="Train the worship-leader gesture transformer.")
    parser.add_argument("--mode", choices=("final", "cv"), default="final",
                        help="final: train the deployable model; cv: grouped cross-validation")
    parser.add_argument("--feature-version", choices=("v1", "v2"), default=DEFAULT_FEATURE_VERSION,
                        help="v1 = original raw landmarks, v2 = normalised landmarks")
    parser.add_argument("--augment", choices=("none", "geo", "full"), default="full",
                        help="augmentation preset (source data is never modified)")
    parser.add_argument("--no-repair", action="store_true",
                        help="disable tracking-hold repair of missing frames (original behaviour)")
    parser.add_argument("--transitions", type=int, default=2,
                        help="transition windows per class pair (0 disables)")
    parser.add_argument("--data", default="MP_Data")
    parser.add_argument("--no-sequences", type=int, default=int(DEFAULT_NO_SEQUENCES))
    parser.add_argument("--sequence-len", type=int, default=int(DEFAULT_SEQUENCE_LEN))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, help="folds for --mode cv")
    parser.add_argument("--cv-inner-holdout", type=int, default=8,
                        help="sequences per class reserved inside each CV training fold")
    parser.add_argument("--holdout-per-class", type=int, default=10,
                        help="sequences per class held out for validation in --mode final")
    parser.add_argument("--min-valid-frames", type=int, default=25,
                        help="drop windows with fewer real (non-missing) frames")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", default=None,
                        help="output path for the checkpoint (final) or metrics JSON (cv)")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    if args.mode == "cv":
        run_cv(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
