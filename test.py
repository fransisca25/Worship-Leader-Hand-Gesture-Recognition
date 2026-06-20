import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# define path to the trained hand landmark model
model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
hand_landmarker = vision.HandLandmarker
hand_landmarker_options = vision.HandLandmarkerOptions
vision_running_mode = vision.RunningMode

# create configuration settings for the hand landmark
options = hand_landmarker_options(
    base_options=base_options,
    running_mode=vision_running_mode.VIDEO,
    num_hands=1
)

landmarker = hand_landmarker.create_from_options(options)

# -------------------- VIDEO INPUT ------------------------
cap = cv2.VideoCapture(0)
timestamp = 0
fps = cap.get(cv2.CAP_PROP_FPS)

# forcing fps to be 30
if fps == 0:
    fps = 30

while True:
    ret, img = cap.read()

    if not ret:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    result = landmarker.detect_for_video(mp_image, timestamp)

    timestamp += int(1000 / fps)

    h, w = img.shape[:2]

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)

                # draw landmark points
                cv2.circle(img, (x, y), 1, (0, 0, 255), 3)

    cv2.imshow("Testing", img)

    # break smoothly, window close by pushing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()

