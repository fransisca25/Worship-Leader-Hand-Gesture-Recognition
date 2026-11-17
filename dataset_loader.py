"""Dataset, split and batching utilities for the gesture transformer.

Changes compared with the original loader
-----------------------------------------
* all windows are cached in memory (the original opened 50 ``.npy`` files per
  sample on *every* epoch, i.e. 10 000 file reads per epoch),
* the feature version used to build the model input is explicit (``v1`` = the
  original raw landmarks, ``v2`` = normalised landmarks),
* windows whose landmarks are mostly missing can be rejected instead of being
  trained as if "no hand" were a gesture,
* optional on-the-fly augmentation and optional transition windows,
* splits can be grouped by recording sequence or blocked in time.

``create_dataloaders`` keeps its original signature and default behaviour, so
existing scripts keep working.
"""

import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from augmentation import augment_window
from tool.kp_extractor import (
    DEFAULT_FEATURE_VERSION,
    FEATURE_DIM,
    count_valid_frames,
    window_features,
)


def sequence_index(data_path, codes, no_sequences):
    """Return every ``(code_index, sequence)`` pair that exists on disk."""
    found = []
    for code_index, code in enumerate([str(c) for c in codes]):
        for sequence in range(no_sequences):
            sequence_path = os.path.join(data_path, str(code), str(sequence))
            if os.path.isdir(sequence_path):
                found.append((code_index, sequence))
    return found


class HandGestureDataset(Dataset):
    """One sample = one 50-frame landmark window."""

    def __init__(
        self,
        data_path,
        codes,
        no_sequences,
        sequence_len,
        sequences=None,
        feature_version=DEFAULT_FEATURE_VERSION,
        augment=None,
        repair=True,
        min_valid_frames=0,
        add_transitions=0,
        seed=0,
    ):
        self.data_path = data_path
        self.codes = [str(code) for code in codes]
        self.no_sequences = no_sequences
        self.sequence_len = sequence_len
        self.feature_version = feature_version
        self.augment = augment
        self.repair = repair
        self.min_valid_frames = int(min_valid_frames)
        self.label_map = {code: idx for idx, code in enumerate(self.codes)}

        self._cache = {}
        self._files = {}
        self._rng = np.random.default_rng(seed)
        self.samples = []
        self.skipped = 0

        wanted = None
        if sequences is not None:
            wanted = {(int(c), int(s)) for c, s in sequences}

        for code_index, sequence in sequence_index(data_path, self.codes, no_sequences):
            if wanted is not None and (code_index, sequence) not in wanted:
                continue
            files = self._frame_files(code_index, sequence)
            if len(files) < self.sequence_len:
                warnings.warn(
                    "Skipping incomplete sequence %s/%s (%d/%d frames)"
                    % (self.codes[code_index], sequence, len(files), self.sequence_len)
                )
                continue
            if self.min_valid_frames:
                window = self._load(code_index, sequence)
                if window is None:
                    continue
                if count_valid_frames(window) < self.min_valid_frames:
                    self.skipped += 1
                    continue
            self.samples.append(
                {"kind": "real", "code": code_index, "sequence": sequence, "label": code_index}
            )

        if add_transitions > 0:
            self._add_transitions(add_transitions)

    # ------------------------------------------------------------------ loading
    def _frame_files(self, code_index, sequence):
        key = (code_index, sequence)
        if key not in self._files:
            sequence_path = os.path.join(self.data_path, self.codes[code_index], str(sequence))
            self._files[key] = sorted(
                [os.path.join(sequence_path, f) for f in os.listdir(sequence_path)
                 if f.endswith(".npy")],
                key=lambda path: int(os.path.basename(path)[:-4]),
            )
        return self._files[key]

    def _load(self, code_index, sequence):
        key = (code_index, sequence)
        if key in self._cache:
            return self._cache[key]

        files = self._frame_files(code_index, sequence)
        if len(files) < self.sequence_len:
            self._cache[key] = None
            return None

        window = np.stack([np.load(path) for path in files[: self.sequence_len]]).astype(np.float64)
        if window.shape != (self.sequence_len, FEATURE_DIM):
            warnings.warn(
                "Skipping malformed sequence %s/%s: %r"
                % (self.codes[code_index], sequence, window.shape)
            )
            self._cache[key] = None
            return None

        self._cache[key] = window
        return window

    def raw_window(self, code_index, sequence):
        """Raw (T, 63) landmarks for one recording (cached, no preprocessing)."""
        return self._load(int(code_index), int(sequence))

    def _add_transitions(self, count_per_pair):
        """Add windows that straddle a gesture change (labelled as the new gesture)."""
        real = [s for s in self.samples if s["kind"] == "real"]
        by_label = {}
        for sample in real:
            by_label.setdefault(sample["label"], []).append(sample)

        labels = sorted(by_label)
        for label_a in labels:
            for label_b in labels:
                if label_a == label_b:
                    continue
                for _ in range(count_per_pair):
                    a = by_label[label_a][int(self._rng.integers(0, len(by_label[label_a])))]
                    b = by_label[label_b][int(self._rng.integers(0, len(by_label[label_b])))]
                    self.samples.append(
                        {
                            "kind": "transition",
                            "a": (a["code"], a["sequence"]),
                            "b": (b["code"], b["sequence"]),
                            "new_frames": int(self._rng.integers(8, 26)),
                            "label": label_b,
                        }
                    )

    # ------------------------------------------------------------------- access
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        if sample["kind"] == "real":
            window = self._load(sample["code"], sample["sequence"]).copy()
        else:
            code_a, seq_a = sample["a"]
            code_b, seq_b = sample["b"]
            new_frames = int(sample["new_frames"])
            window = np.concatenate(
                [
                    self._load(code_a, seq_a)[: self.sequence_len - new_frames],
                    self._load(code_b, seq_b)[self.sequence_len - new_frames:],
                ],
                axis=0,
            )

        if self.augment is not None:
            window = augment_window(window, self._rng, self.augment)

        features = window_features(window, self.feature_version, repair=self.repair)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(int(sample["label"]), dtype=torch.long),
        )



# ----------------------------------------------------------------------- splits
def grouped_folds(keys, n_splits=5, seed=0):
    """Split ``keys`` into folds where every fold holds out whole sequences."""
    keys = list(keys)
    if len(keys) < n_splits:
        raise ValueError("need at least as many sequences as folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    buckets = [[] for _ in range(n_splits)]
    for position, index in enumerate(order):
        buckets[position % n_splits].append(keys[int(index)])

    folds = []
    for bucket in buckets:
        held_out = set(bucket)
        train = [key for key in keys if key not in held_out]
        folds.append((train, list(bucket)))
    return folds


def temporal_blocked_split(keys, holdout_per_class=10):
    """Hold out the last ``holdout_per_class`` sequences of every class.

    The dataset was recorded class by class, so the tail of each class is the
    closest thing to "a later session" that can be built without new recordings.
    """
    by_label = {}
    for key in keys:
        by_label.setdefault(key[0], []).append(key)

    train, held_out = [], []
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda key: key[1])
        if len(ordered) <= holdout_per_class:
            raise ValueError("class %d has too few sequences to hold out" % label)
        held_out.extend(ordered[-holdout_per_class:])
        train.extend(ordered[:-holdout_per_class])
    return train, held_out


# ------------------------------------------------------------------- dataloader
def create_dataloaders(
    data_path,
    codes,
    no_sequences,
    sequence_len,
    batch_size=8,
    train_split=0.8,
    feature_version=DEFAULT_FEATURE_VERSION,
    seed=0,
    augment=None,
    add_transitions=0,
    train_sequences=None,
    val_sequences=None,
    repair=True,
    min_valid_frames=0,
    num_workers=0,
):
    """Build the train/validation loaders.

    Without explicit ``train_sequences`` / ``val_sequences`` this reproduces the
    original behaviour (a random split over sequences), but seeded so runs are
    reproducible.  Pass the output of :func:`grouped_folds` or
    :func:`temporal_blocked_split` for an honest evaluation.
    """
    keys = sequence_index(data_path, codes, no_sequences)

    if train_sequences is None or val_sequences is None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(keys))
        cut = int(len(keys) * train_split)
        train_sequences = [keys[int(i)] for i in order[:cut]]
        val_sequences = [keys[int(i)] for i in order[cut:]]

    train_dataset = HandGestureDataset(
        data_path=data_path,
        codes=codes,
        no_sequences=no_sequences,
        sequence_len=sequence_len,
        sequences=train_sequences,
        feature_version=feature_version,
        augment=augment,
        repair=repair,
        min_valid_frames=min_valid_frames,
        add_transitions=add_transitions,
        seed=seed,
    )

    val_dataset = HandGestureDataset(
        data_path=data_path,
        codes=codes,
        no_sequences=no_sequences,
        sequence_len=sequence_len,
        sequences=val_sequences,
        feature_version=feature_version,
        augment=None,
        repair=repair,
        min_valid_frames=0,
        add_transitions=0,
        seed=seed,
    )

    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader

