import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HandGestureDataset(Dataset):
    def __init__(self, data_path, codes, no_sequences, sequence_len):
        self.data_path = data_path
        self.codes = codes
        self.no_sequences = no_sequences
        self.sequence_len = sequence_len

        self.label_map = {code: idx for idx, code in enumerate(codes)}

        self.samples = []

        for code in codes:
            for sequence in range(no_sequences):
                sequence_path = os.path.join(data_path, code, str(sequence))

                if os.path.exists(sequence_path):
                    self.samples.append((code, sequence))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        code, sequence = self.samples[index]

        window = []

        for frame_num in range(self.sequence_len):
            npy_path = os.path.join(
                self.data_path,
                code,
                str(sequence),
                f"{frame_num}.npy"
            )

            keypoints = np.load(npy_path)
            window.append(keypoints)

        X = np.array(window, dtype=np.float32)
        y = self.label_map[code]

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return X, y


def create_dataloaders(
        data_path,
        codes,
        no_sequences,
        sequence_len,
        batch_size=8,
        train_split=0.8
):
    dataset = HandGestureDataset(
        data_path=data_path,
        codes=codes,
        no_sequences=no_sequences,
        sequence_len=sequence_len
    )

    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader



