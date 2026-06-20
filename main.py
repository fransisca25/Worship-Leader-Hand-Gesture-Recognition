import cv2
import numpy as np
import torch
import mediapipe as mp

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from transformer import HandGestureTransformer
from tool.kp_extractor import extract_keypoints


# ------------------ PATH --------------------
MODEL_PATH = "trained_model/wl_hand_gesture_transformer.pth"
HAND_MODEL_PATH = "hand_landmarker.task"

# ----------------- DEVICE -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ LOAD TRAINED MODEL ------------
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

codes = checkpoint["codes"]
sequence_len = checkpoint["sequence_len"]

model = HandGestureTransformer(
    input_dim=checkpoint["input_dim"],
    d_model=checkpoint["d_model"],
    d_ff=checkpoint["d_ff"],
    num_heads=checkpoint["num_heads"],
    dropout=checkpoint["dropout"],
    num_layers=checkpoint["num_layers"],
    num_classes=checkpoint["num_classes"],
    max_seq_len=checkpoint["sequence_len"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ---------- MEDIAPIPE HAND LANDMARKER ---------
base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)

# --------------- VIDEO INPUT -------------
cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps != 0 else 30

timestamp = 0

sequence = deque(maxlen=sequence_len)

predicted_label = ""
confidence = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    mp_detection = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += int(1000 / fps)

    keypoints = extract_keypoints(mp_detection)
    sequence.append(keypoints)

    h, w = frame.shape[:2]

    # draw landmarks
    if len(sequence) == sequence_len:
        input_data = np.expand_dims(sequence, axis=0)  # (1, sequence_len, 63)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            pred_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_idx].item()
            predicted_label = codes[pred_idx]

    # show prediction result
    cv2.putText(
        frame,
        f"{predicted_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("GBI Seoul WL Gesture Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()

