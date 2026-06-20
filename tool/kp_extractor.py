import numpy as np


def extract_keypoints(mp_detect):
    if mp_detect.hand_landmarks:
        hand = mp_detect.hand_landmarks[0]
        return np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten()
    else:
        return np.zeros(21 * 3)
