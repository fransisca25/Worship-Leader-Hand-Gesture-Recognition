"""Checkpoint loading/saving with explicit feature-version handling.

The original checkpoint (``wl_hand_gesture_transformer.pth``) has no
``feature_version`` entry, so it is treated as ``v1`` (raw landmarks) and keeps
working exactly as before.  Enhanced checkpoints store the preprocessing they
were trained with, so inference can never silently feed the wrong features to
the wrong model.
"""

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from tool.kp_extractor import (
    DEFAULT_FEATURE_VERSION,
    FEATURE_VERSIONS,
    feature_config,
)
from transformer import build_hand_gesture_transformer

LEGACY_CHECKPOINT = os.path.join("trained_model", "wl_hand_gesture_transformer.pth")
DEFAULT_CHECKPOINT = os.path.join("trained_model", "wl_hand_gesture_transformer_v2.pth")


@dataclass
class LoadedModel:
    """A checkpoint plus everything the pipeline needs to feed it correctly."""

    model: torch.nn.Module
    codes: list
    sequence_len: int
    feature_version: str
    feature_config: dict
    path: str
    device: str
    metrics: dict = field(default_factory=dict)
    train_config: dict = field(default_factory=dict)
    is_legacy: bool = False


def resolve_checkpoint(preferred=DEFAULT_CHECKPOINT, fallback=LEGACY_CHECKPOINT):
    """Return ``(path, used_fallback)``, falling back to the original model."""
    if preferred and os.path.exists(preferred):
        return preferred, False
    if fallback and os.path.exists(fallback):
        return fallback, True
    raise FileNotFoundError(
        "No checkpoint found. Looked for %r and %r." % (preferred, fallback)
    )


def load_model(path=DEFAULT_CHECKPOINT, device="cpu"):
    """Load a checkpoint and rebuild the model with its stored configuration."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint or "codes" not in checkpoint:
        raise ValueError("%s does not look like a gesture checkpoint" % path)

    codes = [str(code) for code in np.asarray(checkpoint["codes"]).tolist()] \
        if isinstance(checkpoint["codes"], np.ndarray) else [str(c) for c in checkpoint["codes"]]

    feature_version = str(checkpoint.get("feature_version", "v1"))
    if feature_version not in FEATURE_VERSIONS:
        raise ValueError(
            "Checkpoint %s asks for unknown feature version %r" % (path, feature_version)
        )

    model = build_hand_gesture_transformer(checkpoint, num_classes=len(codes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        codes=codes,
        sequence_len=int(checkpoint.get("sequence_len", 50)),
        feature_version=feature_version,
        feature_config=dict(checkpoint.get("feature_config") or feature_config(feature_version)),
        path=path,
        device=device,
        metrics=dict(checkpoint.get("metrics") or {}),
        train_config=dict(checkpoint.get("train_config") or {}),
        is_legacy=("feature_version" not in checkpoint),
    )


def save_checkpoint(path, model, optimizer, codes, sequence_len, feature_version,
                    model_config, train_config=None, metrics=None, history=None):
    """Save an enhanced checkpoint next to (never on top of) the original one."""
    if os.path.abspath(path) == os.path.abspath(LEGACY_CHECKPOINT):
        raise ValueError("Refusing to overwrite the original checkpoint; choose a new filename")

    config = dict(model_config)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "codes": list(codes),
        "sequence_len": int(sequence_len),
        "input_dim": int(config.get("input_dim", 63)),
        "d_model": int(config["d_model"]),
        "d_ff": int(config["d_ff"]),
        "num_heads": int(config["num_heads"]),
        "dropout": float(config["dropout"]),
        "num_layers": int(config["num_layers"]),
        "num_classes": int(config.get("num_classes", len(codes))),
        "feature_version": feature_version,
        "feature_config": feature_config(feature_version),
        "train_config": train_config or {},
        "metrics": metrics or {},
        "history": history or {},
        "format_version": 2,
    }
    torch.save(payload, path)
    return path


def default_feature_version():
    return DEFAULT_FEATURE_VERSION
