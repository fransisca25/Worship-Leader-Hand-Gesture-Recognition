"""Real-time worship-leader gesture recognition.

Same application as before - MediaPipe finds the hand, the custom transformer
classifies the 50-frame window, the prediction is drawn on the webcam feed -
with the robustness work from the audit wired into the loop:

* a hand that is not detected is no longer fed to the model as a zero frame;
  the buffer holds the last pose for a few frames and then expires,
* the buffer must be full and mostly real before a prediction is allowed,
* the prediction passes through :class:`temporal.PredictionGate`
  (confidence + margin + hysteresis + hold + expiry), so an ambiguous window
  shows "no gesture" instead of a confident wrong cue,
* MediaPipe runs with configurable confidences, two hands, handedness-aware
  selection and monotonic timestamps,
* landmarks are drawn on the frame so the operator can see what the detector
  sees.

Run:
    python main.py
    python main.py --checkpoint trained_model/wl_hand_gesture_transformer.pth   # original model
"""

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import vision

from checkpoint import DEFAULT_CHECKPOINT, LEGACY_CHECKPOINT, load_model, resolve_checkpoint
from hand_detection import HandDetectionConfig, HandSelector, create_landmarker, describe_hands
from preprocessing import LandmarkBuffer
from temporal import GateConfig, PredictionGate
from tool.kp_extractor import window_features

WINDOW_TITLE = "Worship Leader Gesture Recognition"


def build_parser():
    parser = argparse.ArgumentParser(description="Real-time worship gesture recognition.")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint to load (default: enhanced model, then the original)")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--confidence", type=float, default=0.70,
                        help="minimum probability before a gesture is accepted")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="minimum gap between best and second best probability")
    parser.add_argument("--switch-frames", type=int, default=3,
                        help="consecutive confident frames needed to change gesture")
    parser.add_argument("--hold-frames", type=int, default=5,
                        help="minimum frames a gesture stays on screen")
    parser.add_argument("--expire-frames", type=int, default=8,
                        help="frames without confident input before the cue disappears")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.4)
    parser.add_argument("--min-presence-confidence", type=float, default=0.4)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--smooth-alpha", type=float, default=1.0,
                        help="landmark EMA (1.0 disables it)")
    parser.add_argument("--max-hold", type=int, default=12,
                        help="frames the last pose is reused while detection is lost")
    parser.add_argument("--min-valid-frames", type=int, default=35,
                        help="real frames required inside the 50-frame window")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-landmarks", action="store_true", help="hide the landmark overlay")
    return parser


def resolve_device(preference):
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hand_connections():
    """MediaPipe hand skeleton, with a manual fallback for older/newer builds."""
    try:
        pairs = [(connection.start, connection.end)
                 for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS]
        if pairs:
            return pairs
    except Exception:  # pragma: no cover - depends on the MediaPipe build
        pass
    return [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    ]


def draw_hands(frame, hands, selected_index, connections):
    """Draw every detected hand faintly and the tracked hand clearly."""
    height, width = frame.shape[:2]
    for hand in hands:
        active = hand["index"] == selected_index
        point_colour = (0, 200, 0) if active else (140, 140, 140)
        bone_colour = (0, 170, 0) if active else (110, 110, 110)
        points = hand["landmarks"]
        for start, end in connections:
            first = (int(points[start][0] * width), int(points[start][1] * height))
            second = (int(points[end][0] * width), int(points[end][1] * height))
            cv2.line(frame, first, second, bone_colour, 2 if active else 1, cv2.LINE_AA)
        for landmark in points:
            centre = (int(landmark[0] * width), int(landmark[1] * height))
            cv2.circle(frame, centre, 3 if active else 2, point_colour, -1, cv2.LINE_AA)
        if active:
            label = "tracked %s" % (hand["handedness"] or "hand")
            anchor = (int(hand["center"][0] * width), int(hand["center"][1] * height) - 20)
            cv2.putText(frame, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 0), 1, cv2.LINE_AA)


def _draw_overlay(frame, result, buffer, hands, fps_display):
    """Same visual language as the original UI, plus the pipeline state."""
    label = result.display().replace("_", " ")
    confidence = result.confidence if result.label is not None else 0.0
    colour = (0, 0, 255) if result.label is not None else (160, 160, 160)

    cv2.putText(frame, label.upper(), (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3, cv2.LINE_AA)
    cv2.putText(frame, "Confidence: %.2f" % confidence, (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "State: %s" % result.state, (10, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Buffer: %d/%d  hands: %d" % (buffer.frames, buffer.length, len(hands)),
                (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "FPS: %.1f" % fps_display, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.checkpoint:
        checkpoint_path, used_fallback = args.checkpoint, False
    else:
        checkpoint_path, used_fallback = resolve_checkpoint(DEFAULT_CHECKPOINT, LEGACY_CHECKPOINT)

    device = resolve_device(args.device)
    loaded = load_model(checkpoint_path, device=device)
    codes = loaded.codes

    print("model      : %s" % checkpoint_path)
    if used_fallback:
        print("             (enhanced model not found, running the original checkpoint)")
    print("features   : %s" % loaded.feature_version)
    print("classes    : %s" % ", ".join(codes))
    print("device     : %s" % device)
    print("quit       : press q in the video window")

    detection_config = HandDetectionConfig(
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    landmarker = create_landmarker(detection_config, running_mode="video")

    selector = HandSelector()
    buffer = LandmarkBuffer(
        length=loaded.sequence_len,
        max_hold_frames=args.max_hold,
        min_valid_frames=min(args.min_valid_frames, loaded.sequence_len),
        smooth_alpha=args.smooth_alpha,
    )
    gate = PredictionGate(
        codes,
        GateConfig(
            confidence_threshold=args.confidence,
            margin_threshold=args.margin,
            switch_frames=args.switch_frames,
            min_hold_frames=args.hold_frames,
            expire_frames=args.expire_frames,
        ),
    )

    connections = hand_connections()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        landmarker.close()
        raise RuntimeError("Could not open camera index %d" % args.camera)

    started = time.monotonic()
    last_frame_time = started
    last_timestamp_ms = -1
    fps_display = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("camera stream ended")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((time.monotonic() - started) * 1000)
            timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            detection = landmarker.detect_for_video(mp_image, timestamp_ms)

            hands = describe_hands(detection)
            selected = selector.select(hands)
            buffer.push(selected["landmarks"] if selected is not None else None)

            probabilities = None
            window = buffer.window()
            if window is not None:
                features = window_features(window, loaded.feature_version, repair=True)
                tensor = torch.from_numpy(features[None]).to(device)
                with torch.no_grad():
                    probabilities = torch.softmax(loaded.model(tensor), dim=1).cpu().numpy()[0]

            result = gate.update(probabilities)

            if not args.no_landmarks:
                draw_hands(frame, hands, selected["index"] if selected else None, connections)

            _draw_overlay(frame, result, buffer, hands, fps_display)

            now = time.monotonic()
            instant = 1.0 / max(now - last_frame_time, 1e-6)
            fps_display = instant if fps_display == 0.0 else 0.9 * fps_display + 0.1 * instant
            last_frame_time = now

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

