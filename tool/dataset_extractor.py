"""Record gesture sequences from the webcam into ``MP_Data``.

Same workflow as the original collector (50 frames per sequence, folder per
class/sequence), with two robustness fixes:

* a frame is only saved when MediaPipe actually returned a usable hand, so the
  dataset can no longer contain "all-zero = gesture" samples;
* MediaPipe is configured from :mod:`hand_detection` (two hands, tuned
  confidences) and the tracked hand is drawn, so collection is verifiable.

Run from the project root:
    python tool/dataset_extractor.py
"""

import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from hand_detection import HandDetectionConfig, HandSelector, create_landmarker, describe_hands  # noqa: E402
from tool.dir_generator import DATA_PATH, codes, no_sequences, sequence_len, create_folders  # noqa: E402
from tool.kp_extractor import FEATURE_DIM  # noqa: E402

MAX_ATTEMPTS_PER_FRAME = 60   # give the user ~2 s to get the hand back into view


def main():
    create_folders()

    landmarker = create_landmarker(HandDetectionConfig(num_hands=1), running_mode="video")
    selector = HandSelector()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        landmarker.close()
        raise RuntimeError("Could not open the webcam")

    started = time.monotonic()
    quit_requested = False
    skipped_total = 0

    try:
        for code in codes:
            for sequence in range(no_sequences):
                if quit_requested:
                    break

                saved = 0
                attempts = 0

                while saved < sequence_len:
                    ok, image = capture.read()
                    if not ok:
                        quit_requested = True
                        break

                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp = int((time.monotonic() - started) * 1000)
                    detection = landmarker.detect_for_video(mp_image, timestamp)

                    hands = describe_hands(detection)
                    selected = selector.select(hands)

                    for hand in hands:
                        for point in hand["landmarks"]:
                            centre = (int(point[0] * image.shape[1]), int(point[1] * image.shape[0]))
                            cv2.circle(image, centre, 2, (0, 0, 255), 2)

                    if saved == 0:
                        cv2.putText(image, "STARTING COLLECTION", (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(
                        image,
                        "Collecting %s sequence %d  frame %d/%d  (attempt %d)"
                        % (code, sequence, saved, sequence_len, attempts),
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                    )
                    status = "hand: %s" % (selected["handedness"] if selected else "not detected")
                    cv2.putText(image, status, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 128, 0) if selected else (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow("OpenCV Feed", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        quit_requested = True
                        break

                    attempts += 1
                    if selected is None:
                        skipped_total += 1
                        if attempts > sequence_len * MAX_ATTEMPTS_PER_FRAME:
                            print("No hand for %s/%d - skipping sequence" % (code, sequence))
                            break
                        continue

                    keypoints = selected["landmarks"].reshape(FEATURE_DIM)
                    path = os.path.join(DATA_PATH, code, str(sequence), "%d.npy" % saved)
                    np.save(path, keypoints)
                    saved += 1

                print("%s/%d stored %d/%d frames (%d skipped reads, %d runs used)"
                      % (code, sequence, saved, sequence_len, skipped_total, attempts))

            if quit_requested:
                break
    finally:
        capture.release()
        landmarker.close()
        cv2.destroyAllWindows()

    print("Collection finished. Missing-hand reads skipped: %d" % skipped_total)


if __name__ == "__main__":
    main()
