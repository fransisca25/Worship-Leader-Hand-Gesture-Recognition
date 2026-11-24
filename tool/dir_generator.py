"""Dataset folder layout for the gesture recordings.

The folders are created next to this file's project (``<project>/MP_Data``)
instead of relying on the current working directory, so the collection script
works the same from the project root or from inside ``tool/``.
"""

import os

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "MP_Data")

codes = np.array(["verse", "chorus", "bridge", "medley"])
no_sequences = 50
sequence_len = 50


def create_folders():
    """Create one folder per class/sequence (idempotent)."""
    for code in codes:
        for sequence in range(no_sequences):
            os.makedirs(os.path.join(DATA_PATH, code, str(sequence)), exist_ok=True)


if __name__ == "__main__":
    create_folders()
    print("Dataset folders ready in %s" % DATA_PATH)

