import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dir_generator import DATA_PATH, codes, no_sequences, sequence_len
from kp_extractor import extract_keypoints

handtrack_model_path = "../hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=handtrack_model_path)
hand_landmarker = vision.HandLandmarker
hand_landmarker_options = vision.HandLandmarkerOptions
vision_running_mode = vision.RunningMode

# setting configuration for hand landmark
options = hand_landmarker_options(
    base_options=base_options,
    running_mode=vision_running_mode.VIDEO,
    num_hands=1
)

landmarker = hand_landmarker.create_from_options(options)


# --------------------- VIDEO INPUT ---------------------
cap = cv2.VideoCapture(0)

timestamp = 0

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps != 0 else 30

for code in codes:
    for sequence in range(no_sequences):
        for frame_num in range(sequence_len):

            ret, img = cap.read()

            if not ret:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=img_rgb
            )

            mp_detection = landmarker.detect_for_video(mp_img, timestamp)
            keypoints = extract_keypoints(mp_detection)

            timestamp += int(1000 / fps)

            h, w = img.shape[:2]

            # draw on window
            if mp_detection.hand_landmarks:
                for hand_landmarks in mp_detection.hand_landmarks:
                    for lm in hand_landmarks:
                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        # draw landmark
                        cv2.circle(img, (x, y), 1, (0, 0, 255), 2)

            # waiting logic
            if frame_num == 0:
                cv2.putText(
                    img, "STARTING COLLECTION", (120, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    img, "Collecting frames for {} Video Number {}".format(code, sequence),
                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA
                )
                cv2.imshow("OpenCV Feed", img)
                cv2.waitKey(2000)
            else:
                cv2.putText(
                    img, "Collecting frames for {} Video Number {}".format(code, sequence),
                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA
                )
                cv2.imshow("OpenCV Feed", img)

            npy_path = os.path.join(DATA_PATH, code, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            # break smoothly
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
