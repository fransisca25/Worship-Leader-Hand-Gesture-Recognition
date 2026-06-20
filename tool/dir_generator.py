import os
import numpy as np


# setup folder for dataset collection
DATA_PATH = os.path.join("../MP_Data")
codes = np.array(['verse', 'chorus', 'bridge', 'medley'])
no_sequences = 50
sequence_len = 50

for code in codes:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, code, str(sequence)))
        except:
            pass

