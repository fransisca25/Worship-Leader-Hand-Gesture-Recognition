# Worship Gesture Recognition for Multimedia Team

A real-time hand gesture recognition system designed to help church multimedia teams identify worship song sections such as **Verse**, **Chorus**, **Bridge**, and **Medley / Song Change** during live worship sessions.

This project uses **MediaPipe hand landmarks** and a **Transformer-based deep learning model** to classify hand gestures from webcam input.

---

## Overview

During live worship, the multimedia operator needs to display the correct lyrics at the right time. However, communication between the worship leader and the multimedia team can be difficult because of loud music, spontaneous song arrangements, and quick transitions between song sections.

This project provides a visual cue system using hand gestures. By recognizing predefined gestures, the system helps the multimedia team understand when to prepare or switch to the correct song section.

The main goal is to reduce miscommunication and support smoother worship flow during church services.

---

## Problem Statement

In many live worship settings, song arrangements are not always fixed. The worship leader may repeat a chorus, move to a bridge, return to a verse, or transition into a medley depending on the atmosphere of worship.

For the multimedia team, this can be challenging because they need to react quickly while displaying the correct lyrics.

Common challenges include:

- Loud music makes verbal communication difficult.
- The worship leader may change the song flow spontaneously.
- Medley or song transitions can happen quickly.
- The multimedia operator needs clear and fast cues.
- Wrong lyrics on screen can distract the congregation.

This project aims to solve this problem using computer vision and gesture recognition.

---

## Features

- Real-time hand gesture recognition using webcam
- Hand landmark detection using MediaPipe
- Sequence-based gesture classification
- Transformer-based model architecture
- Worship section classification
- Designed for church multimedia team workflow
- Can be expanded for more worship cues and slide commands

---

## Gesture Classes

The model is trained to classify hand gestures related to worship song flow.

| Class | Description |
|---|---|
| Verse | Indicates that the multimedia team should show the verse lyrics |
| Chorus | Indicates that the multimedia team should show the chorus lyrics |
| Bridge | Indicates that the multimedia team should show the bridge lyrics |
| Medley / Song Change | Indicates that the team should prepare for a medley or switch to another song |

These classes can be adjusted or expanded depending on the needs of the worship team.

---

## System Workflow

The system follows these steps:

1. The webcam captures the user's hand gesture.
2. MediaPipe detects hand landmarks from each video frame.
3. The detected hand landmarks are converted into numerical keypoints.
4. Each frame produces 63 features from 21 hand landmarks.
5. A sequence of keypoints is passed into the Transformer model.
6. The model predicts the gesture class.
7. The predicted result is displayed in real time.
8. The multimedia team can use the prediction as a cue for lyric changes.

---

## Model Architecture

This project uses a Transformer-based neural network for hand gesture classification.

The model consists of:

- Input projection layer
- Positional embedding
- Transformer encoder
- Transformer decoder
- Learnable query embedding
- Layer normalization
- Linear classification layer

The input format is:

```txt
(batch_size, sequence_length, 63)
```

Where:

```txt
63 = 21 hand landmarks × 3 coordinates
```

Each hand landmark contains:

- x coordinate
- y coordinate
- z coordinate

The model processes a sequence of frames, allowing it to learn both hand position and hand movement patterns.

---

## Why Transformer?

Hand gestures are not always static. Some gestures may depend on motion, direction, or timing across multiple frames.

A Transformer model is suitable for this task because it can learn relationships between frames in a sequence. This allows the model to understand gesture patterns more effectively than using a single frame only.

In this project, the Transformer helps classify worship gestures based on the sequence of hand landmarks.

---

## Training Result

The model was trained for 50 epochs.

### Final Result

| Metric | Value |
|---|---:|
| Train Loss | 0.2443 |
| Train Accuracy | 92.50% |
| Test Loss | 0.3944 |
| Test Accuracy | 92.50% |

### Best Test Accuracy

| Epoch | Test Accuracy |
|---:|---:|
| 39 | 95.00% |

The result shows that the model was able to learn the gesture patterns well and achieved strong performance on the test data.

---

## Tech Stack

- Python
- PyTorch
- OpenCV
- MediaPipe
- NumPy

---

## Project Structure

```txt
project/
│
├── data/
│   └── gesture_sequences/
│
├── trained_model/
│   └── wl_hand_gesture_transformer.pth
│
├── tool/
│   └── kp_extractor.py
│
├── transformer.py
├── train.py
├── main.py
├── hand_landmarker.task
└── README.md
```

---

## Installation

Install the required dependencies:

```bash
pip install torch opencv-python mediapipe numpy
```

---

## MediaPipe Model

This project uses the MediaPipe Hand Landmarker model.

Make sure the following file is available in the project directory:

```txt
hand_landmarker.task
```

If the file is placed in another directory, update the model path in the code.

---

## Training

To train the model, run:

```bash
python train.py
```

The training process will load the collected gesture keypoint sequences and train the Transformer model.

After training, the model can be saved into the `trained_model/` directory.

---

## Real-Time Prediction

To run real-time gesture prediction, use:

```bash
python main.py
```

The webcam will open and the system will display the predicted gesture class on the screen.

---

## Example Use Case

During worship, the worship leader or assigned team member can give a hand gesture to indicate the next song section.

Example:

| Gesture | Multimedia Action |
|---|---|
| Verse gesture | Show verse lyrics |
| Chorus gesture | Show chorus lyrics |
| Bridge gesture | Show bridge lyrics |
| Medley gesture | Prepare for medley or next song |

This helps the multimedia operator respond faster and display the correct lyrics with better timing.

---

## Limitations

This project is still an early prototype. Some limitations include:

- The model depends on clear hand visibility.
- Prediction may be affected by poor lighting.
- Fast hand movement may reduce detection accuracy.
- Camera angle can affect landmark detection.
- The current dataset is still limited.
- More testing is needed in real worship environments.

---

## Future Improvements

Possible future improvements include:

- Add more gesture classes
- Collect more training data from different people
- Improve prediction stability
- Add confidence threshold before accepting predictions
- Add prediction smoothing to reduce flickering results
- Support two-hand gestures
- Add automatic keyboard shortcuts for lyric slide control
- Integrate with presentation or lyric software
- Build a simple user interface for multimedia operators
- Test the system in real church service environments

---

## Project Goal

The goal of this project is to create a practical AI-based tool that supports church multimedia teams during live worship.

By combining computer vision and deep learning, this system helps improve communication, reduce confusion, and support a smoother worship experience for both the team and the congregation.

---

## Author

Created by Fransisca Celia K.
