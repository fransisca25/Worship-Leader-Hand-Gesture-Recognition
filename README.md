# Worship Gesture Recognition for Multimedia Team

A real-time hand gesture recognition system that helps a church multimedia team
follow the worship leader: the leader shows **Verse**, **Chorus**, **Bridge** or
**Medley / Song Change**, and the application shows the recognised cue on screen.

The recognition core is unchanged - **MediaPipe Hand Landmarker** finds 21 hand
landmarks and **this project's own PyTorch Transformer** classifies a 50-frame
window of 63 landmark values into the same four gestures - but the surrounding
engineering has been hardened so the cue is trustworthy in a real service:
a missing hand can no longer produce a confident wrong cue, the features are
normalised so position/distance/tilt stop mattering, the model is trained and
measured on unseen recordings, and the output is filtered in time.

- v1 model: `trained_model/wl_hand_gesture_transformer.pth`
- Enhanced model (v2): `trained_model/wl_hand_gesture_transformer_v2.pth`
- Dataset: `MP_Data/` 
- Original README: `README_original.md`

---

## 1. What the system does (unchanged)

1. The webcam captures the worship leader.
2. MediaPipe detects the hand and returns 21 landmarks.
3. The landmarks of one frame become 63 numbers (x, y, z per landmark).
4. The last 50 frames are stacked into one window.
5. The transformer predicts one of four gestures.
6. The cue is shown on the video overlay and can drive the lyric slides.

| Class | Meaning | Multimedia action |
|---|---|---|
| `verse` | verse gesture | show verse lyrics |
| `chorus` | chorus gesture | show chorus lyrics |
| `bridge` | bridge gesture | show bridge lyrics |
| `medley` | medley / song change | prepare for a medley or the next song |

The four labels, their meaning and the class order (`verse`, `chorus`, `bridge`,
`medley`) are exactly the original ones.

---

## 2. What changed, and why

| Area | Before | Now |
|---|---|---|
| Missing hand | `extract_keypoints` returned `zeros(63)` and the deque kept filling, so "no hand" was a valid model input. Measured: an all-zero window produced **"medley" at 0.975 confidence**. | The buffer never stores zero frames: it holds the last pose for a few frames, then expires. Windows need 35 of 50 real frames. An empty frame now produces **no cue** (`no_gesture`). |
| Landmarks | raw image coordinates (position, distance and tilt included) | feature version **v2**: wrist-centred, scale-normalised, in-plane rotation-normalised, same 63 numbers, same transformer |
| Decisions | `argmax` every frame, no threshold | confidence + margin + hysteresis + minimum hold + expiry (`temporal.PredictionGate`) |
| Hands in frame | `num_hands=1`, always the first detection | up to 2 hands, handedness-aware selection anchored to the previous hand |
| Detection settings | all MediaPipe confidences left at 0.5 | detection/presence 0.4, tracking 0.5, configurable |
| Training | random 80/20 window split, no seed, last epoch saved, test set also used for reporting | seeded, temporal hold-out + grouped CV by recording sequence, best-checkpoint by validation macro-F1, label smoothing, LR schedule, gradient clipping |
| Data pipeline | 50 `.npy` reads per sample per epoch | in-memory cache, optional augmentation, optional transition windows |
| Feedback | only the label was drawn | landmarks, tracked hand, state, buffer fill, FPS |
| Evaluation | one accuracy number | reproducible suite: per-class metrics, confusion matrix, ECE, perturbation tests, rejection tests, threshold sweep, transition latency, FPS |

Everything above preserves the architecture: same MediaPipe Tasks API, same
custom `HandGestureTransformer`, same 63-dimensional input, same four classes.

---

## 3. Quick start

```bash
pip install torch opencv-python mediapipe numpy scikit-learn tqdm matplotlib
```

Train the enhanced model (writes `trained_model/*_v2.*`, never touches the original):

```bash
python train.py
```

Run the real-time application (uses the enhanced model when present):

```bash
python main.py
```

Press **q** in the video window to quit. All thresholds are optional command
line flags, e.g.:

```bash
python main.py --confidence 0.75 --margin 0.2 --num-hands 2
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```


---

## 4. Hand detection

MediaPipe stays the detector (`hand_landmarker.task`, Tasks API). The settings
live in `hand_detection.py` so training, collection and inference agree:

| Setting | Value | Why |
|---|---|---|
| `num_hands` | 2 | another person's hand no longer hijacks the window |
| `min_hand_detection_confidence` | 0.4 | more tolerant in low light |
| `min_hand_presence_confidence` | 0.4 | keeps a partially visible hand alive |
| `min_tracking_confidence` | 0.5 | prevents identity swaps |

`HandSelector` keeps the pipeline on the same hand: it anchors the previous
hand's position and handedness, prefers the nearest hand of the same side and
only re-acquires by score when the hand really jumps. This is unit tested
(`tests/test_pipeline.py`).

Frames are converted to RGB and sent with a **monotonic** timestamp (the
original accumulated `1000/fps` per loop, which drifts when the loop is slower
than the camera).

*Image-level preprocessing (CLAHE / gamma for very dark venues) is deliberately
not enabled: it can only be validated with real recordings from the venue, and
there are no images in the repository to test it on. See section 13.*

---

## 5. Landmarks and feature versions

Both versions produce exactly **63 numbers per frame** and the same 50-frame
window, so the transformer and its positional embedding are unchanged.

### `v1` - the original representation

```text
[x0, y0, z0, x1, y1, z1, ... x20, y20, z20]   raw MediaPipe image coordinates
```

Kept for the original checkpoint and for before/after experiments.

### `v2` - normalised (default for new training)

```text
p_i   = landmark_i - wrist                       # translation removed
s     = || middle_finger_mcp - wrist ||_2        # hand size
p_i   = p_i / s                                  # scale removed (x, y and z)
angle = atan2(middle_mcp_y, middle_mcp_x) + pi/2
p_i   = rotate(p_i, angle) in the image plane    # in-plane tilt removed
```

Properties (each one is a unit test):

* invariant to where the hand is, how close it is, and how it is tilted in the
  image plane,
* the **relative finger configuration is untouched**, so the geometry that
  separates the four gestures survives,
* the palm/back twist that distinguishes `medley` is a rotation about the
  forearm axis, not about the camera axis - it is preserved (the handedness
  cross-product sign is unchanged by the normalisation and flipped by a mirror),
* `z` is kept but divided by the same hand scale, because the audit showed the
  original model relies on depth (zeroing z dropped accuracy from 0.925 to 0.50).

The feature version is written into every checkpoint, and `main.py`
preprocesses according to the checkpoint it loaded, so an old model can never be
fed new features by accident.

### Why rotation normalisation is safe for `medley`

The audit measured per-frame motion and a palm/back handedness proxy: three
gestures are static holds, `medley` is the dynamic one (2.5-4x more per-frame
movement, and 76% of its sequences flip the palm/back sign). In-plane rotation
normalisation cannot remove that flip, and a controlled feature experiment
(per-frame logistic regression, grouped by recording sequence) improved when
rotation was added: 0.966 -> 0.986 frame accuracy, with `medley` F1 0.98.

---

## 6. Model architecture (unchanged)

| Component | Value |
|---|---|
| Input | `(batch, 50, 63)` |
| Input projection | `Linear(63 -> 64)` |
| Positional embedding | learned `(1, 50, 64)` |
| Encoder | 2 layers, 2 heads, `d_ff=128`, dropout 0.1, post-LayerNorm |
| Decoder | 1 learnable query + 2 cross-attention layers |
| Head | `LayerNorm -> Linear(64 -> 4)` |
| Parameters | ~440k (1.7 MB checkpoint) |

`transformer.py` gained one helper, `build_hand_gesture_transformer(config)`,
which builds the same class from a checkpoint configuration. No layer was added,
removed or replaced.

---

## 7. Missing hand, rejection and temporal behaviour

### Buffer (`preprocessing.LandmarkBuffer`)

* frames with no hand are **not** stored as zeros,
* a short dropout (default 12 frames) holds the last real pose so the 50-frame
  timing stays correct,
* a longer dropout **expires** the whole window and the cue disappears,
* a window is only released when at least `min_valid_frames` (default 35) are
  real detections.

### Gate (`temporal.PredictionGate`)

| Parameter | Default | Effect |
|---|---|---|
| `confidence_threshold` | 0.70 | below this the gesture is not accepted |
| `margin_threshold` | 0.15 | best class must beat the runner-up |
| `acquire_frames` | 3 | consecutive confident frames to show a gesture |
| `switch_frames` | 3 | consecutive confident frames to change gesture |
| `min_hold_frames` | 5 | a gesture stays at least this long |
| `expire_frames` | 8 | frames without input before `no_gesture` |

The gate is a small state machine, not a stack of smoothing algorithms: the
audit showed that EMA or majority voting **on top of** the 50-frame window only
added latency, so exactly one pending candidate and one committed gesture are
kept.

---

## 8. Dataset

`MP_Data/` is unchanged and is what all numbers below are measured on.

| Property | Value |
|---|---|
| Classes | 4 (`verse`, `chorus`, `bridge`, `medley`) |
| Recordings | 50 per class, 50 frames each = 200 windows / 10 000 frames |
| Frames without a detected hand | 183 (1.83%), spread over 126 windows |
| Longest gap | 26 frames (`medley/0`, that window is skipped by the loader) |
| Landmark jitter per frame | median 0.006, p95 0.163 of hand size |

The data comes from **one session, one camera and (almost certainly) one
person**. Nothing in this repository can measure cross-user generalisation; that
needs new recordings (section 12).

---

## 9. Training

```bash
python train.py                      # enhanced model, temporal hold-out
python train.py --mode cv            # 5-fold grouped CV, metrics only
python train.py --feature-version v1 --augment none --no-repair
                                     # reproduce the original training protocol
```

What the training script does differently from the original one:

* **seeded** (`--seed 42`), so a run can be reproduced;
* **temporal hold-out** validation (the last 10 recordings of every class) used
  for early stopping and model selection - the previous script reported the loss
  of epoch 50 and also used the test split for that decision;
* **best checkpoint** is saved, not the last epoch;
* `CrossEntropyLoss(label_smoothing=0.05)`, `Adam(lr=1e-3, weight_decay=1e-4)`,
  `ReduceLROnPlateau`, gradient clipping;
* class weights are computed automatically if the classes are imbalanced
  (they are balanced today, so this is a no-op - it matters once a
  `no_gesture` recording set exists);
* **augmentation** and **transition windows** are added to the training set
  only; `MP_Data` is never modified;
* **per-class precision / recall / F1, macro-F1 and a confusion matrix** are
  written to `trained_model/training_metrics_v2.json` and the log.

Augmentation presets (`--augment`):

| Preset | Contents |
|---|---|
| `none` | no augmentation |
| `geo` | one copy: tilt, in-plane rotation, scale, translation, time shift |
| `full` (default) | two copies: + landmark jitter, 1-2 landmark dropout, tracking holds |

Because `v2` already removes translation, scale and in-plane rotation, those
augmentations are switched off automatically for `v2` (they would be identity);
what remains is out-of-plane tilt, jitter, dropout and tracking holds. Horizontal
mirroring is implemented but **off by default** - handedness is meaningful here.

Files written (the originals are never touched):

```text
trained_model/wl_hand_gesture_transformer_v2.pth    enhanced checkpoint
trained_model/training_log_v2.txt                   per-epoch CSV
trained_model/training_metrics_v2.json              config + history + metrics
trained_model/training_plot_v2.png                  loss / accuracy / macro-F1
trained_model/cv_metrics_v2.json                    grouped CV (--mode cv)
```

---

## 10. Evaluation

```bash
python evaluate.py --checkpoint trained_model/wl_hand_gesture_transformer.pth
python evaluate.py --checkpoint trained_model/wl_hand_gesture_transformer_v2.pth
```

The report is printed and also written to
`trained_model/evaluation_<checkpoint>.json`. Sections: dataset audit, clean
metrics, perturbations, rejection, threshold sweep, transition behaviour,
classifier latency.

Every number below was produced by these scripts on this machine
(CPU only, MediaPipe 1.2.3, torch 2.15.0). Nothing is estimated unless it is
marked as such.

### 10.1 Window accuracy (all 200 recordings)

Both models are evaluated on the same 200 windows, which include the windows
they were trained on - so this is an upper bound, not a generalisation number.

| Metric | original (v1) | enhanced (v2) |
|---|---|---|
| Accuracy | 0.9200 | **0.9950** |
| Macro-F1 | 0.9209 | **0.9950** |
| F1 verse / chorus / bridge / medley | 0.925 / 0.891 / 0.929 / 0.939 | 0.990 / 0.990 / 1.000 / 1.000 |
| Mean confidence | 0.901 | 0.938 |
| Expected calibration error | 0.042 | 0.058 |

### 10.2 Accuracy on recordings the model has never seen

| Protocol | original (v1) | enhanced (v2) |
|---|---|---|
| Temporal hold-out (last 10 recordings/class, 40 windows) | n/a | **0.975** accuracy, 0.975 macro-F1 |
| Grouped 5-fold CV by recording (40 epochs) | **0.715** | **0.855 ± 0.068** accuracy, 0.848 ± 0.077 macro-F1 |

Pooled per-class results of the grouped CV for the enhanced model:
verse F1 0.860 (recall 0.92), chorus 0.807 (recall 0.88), bridge 0.837
(recall 0.72), medley 0.918 (recall 0.90). The folds are the same kind of
sequence-grouped split in both columns; the original number comes from the
audit harness (same architecture and 40-epoch budget, but without the
normalisation, repair, augmentation and transition windows).

The temporal hold-out confusion matrix of the enhanced model is
`[[9,1,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10]]` - the only error is one `verse`
window read as `chorus`. The run stopped early at epoch 43; the best macro-F1
was reached at epoch 31 (validation accuracy 0.975, macro-F1 0.9749), and the
learning rate had decayed from 1e-3 to 1.25e-4 by then.

This is the honest comparison of the *training protocol*: the original 92.5%
figure came from a random split that put frames of the same recording in both
train and test; with whole recordings held out, the original pipeline scored
0.715 under the same kind of split.

### 10.3 Landmark perturbations

Each row re-featurises the same windows after a corruption (the model itself is
unchanged). "Occlusion" is simulated by zeroing the four landmarks of a finger
in every frame - a worst case, not a real hand-over-hand occlusion.

| Condition | original (v1) | enhanced (v2) |
|---|---|---|
| identity | 0.920 | 0.995 |
| hand moved by (0.10, 0.15) | 0.915 | **0.995** |
| hand 0.7x size | 0.925 | **0.995** |
| hand 1.4x size | 0.915 | 0.980 |
| hand rotated 20 deg | 0.900 | **0.995** |
| hand rotated 30 deg | 0.850 | **0.995** |
| x stretched 1.333 (aspect ratio) | 0.935 | 0.980 |
| y stretched 1.333 (aspect ratio) | 0.900 | 0.950 |
| landmark jitter sigma 0.02 | 0.920 | 0.935 |
| index finger landmarks missing | 0.250 | **0.655** |
| thumb landmarks missing | 0.325 | **0.820** |
| 15 frames of detection loss | 0.915 | 0.990 |
| all-zero window (no hand) | predicts `medley` at 0.98 | predicts `medley` at 0.62 |

Translation, scale and in-plane rotation are now *by construction* invariant:
the enhanced model returns identical features for them. The remaining
sensitivity is anisotropic (aspect-ratio) distortion and jitter, which are not
removed by the normalisation.

### 10.4 Missing hand and ambiguous input

"Live" means the real pipeline (buffer + gate); "model" means the raw classifier
applied to the window.

| Input | v1 model false cue | v1 live abstain | v2 model false cue | v2 live abstain |
|---|---|---|---|---|
| empty (no hand at all) | 1.00 | 1.00 | **0.00** | 1.00 |
| 2/3 of the window missing | 0.83 | 1.00 | 0.96 | 1.00 |
| window straddling two gestures | 0.67 | n/a | 0.96 | n/a |
| real gesture | accepts 0.92, 82% correct | accepts 0.83, 80% correct | accepts 1.00, 100% correct | accepts 1.00, 100% correct |

The important line is the first one: the original classifier answers "medley"
for an empty frame **every time**; the enhanced one never does, and the buffer
rejects the frame before the classifier is even asked. The v2 model does still
label a *partially* visible window confidently - that is deliberate, it was
trained on transition windows - but the live pipeline abstains as soon as the
hand is really gone.

### 10.5 Gesture transitions

50 frames of gesture A followed by 100 frames of gesture B, window slid one frame
at a time; latency is measured from the frame where B appears.

| | v1 raw argmax | v1 + gate | v2 raw argmax | v2 + gate |
|---|---|---|---|---|
| Median latency | 32 frames (1067 ms) | 40.5 frames (1350 ms) | 6.5 frames (217 ms) | **9.0 frames (300 ms)** |
| p90 latency | 49 frames | 51 frames | 20.3 frames | 26.2 frames |
| Never reached target | 0% | 0% | 0% | 0% |
| Label changes after switch | 0.33 | 0.00 | 0.83 | **0.00** |
| Final label correct | 100% | 100% | 100% | 100% |

This is the effect of two changes working together: transition windows in
training teach the model to answer after ~7 frames of the new gesture, and the
gate removes the flicker that early answering would otherwise cause. Gating the
*old* model only made it later (32 -> 40.5 frames); the model, not the filter,
was the bottleneck.

### 10.6 Threshold sweep

`evaluate.py` prints acceptance on real gestures versus abstention on the
negative sets for confidence 0.5-0.9 and margin 0-0.2. Selected rows:

| confidence / margin | v1 accept | v1 abstain | v2 accept | v2 abstain |
|---|---|---|---|---|
| 0.7 / 0.15 (default) | 0.915 | 0.083 | 0.980 | 0.333 |
| 0.8 / 0.15 | 0.850 | 0.194 | 0.970 | 0.333 |
| 0.9 / 0.15 | 0.685 | 0.389 | 0.910 | 0.361 |

The enhanced model keeps almost all real gestures while rejecting a third of the
noise; the original has to give up 30% of the real gestures to reach a similar
abstention rate.

### 10.7 Speed

| | original (v1) | enhanced (v2) |
|---|---|---|
| Classifier latency (CPU, batch 1) | 1.6-4.3 ms median | 1.52 ms median, 1.9 ms p95 |
| Implied classifier FPS | 230-610 | ~657 |

Both models are the same architecture, so this difference is machine load, not a
model improvement. **MediaPipe detection dominates the frame time**, not the
transformer; the measured end-to-end FPS is shown live in the application
overlay.

---

## 11. Project structure

```text
project/
├── MP_Data/                        original recordings (untouched)
├── trained_model/
│   ├── wl_hand_gesture_transformer.pth        ORIGINAL model (untouched)
│   ├── wl_hand_gesture_transformer_v2.pth     enhanced model
│   ├── training_log_v2.txt / training_metrics_v2.json / training_plot_v2.png
│   ├── cv_metrics_v2.json                     grouped CV
│   └── evaluation_*.json                      evaluation reports
├── tool/
│   ├── kp_extractor.py             landmarks -> 63 features (v1 / v2)
│   ├── dir_generator.py            dataset folder layout (absolute paths now)
│   └── dataset_extractor.py        webcam -> MP_Data (skips frames without a hand)
├── transformer.py                  the custom transformer (architecture unchanged)
├── preprocessing.py                landmark buffer, tracking hold, smoothing
├── augmentation.py                 label-preserving augmentation + transitions
├── temporal.py                     prediction gate (confidence, hysteresis, expiry)
├── hand_detection.py               MediaPipe configuration + stable hand selection
├── checkpoint.py                   checkpoint I/O with feature-version handling
├── metrics.py                      classification metrics + calibration error
├── dataset_loader.py               dataset, caching, grouped / temporal splits
├── train.py                        training (final + grouped CV)
├── evaluate.py                     offline evaluation suite
├── main.py                         real-time application
├── tests/test_pipeline.py          47 unit tests
├── README.md / README_original.md
└── hand_landmarker.task
```

### New modules and their responsibilities

| File | Responsibility |
|---|---|
| `preprocessing.py` | rolling window, tracking-loss handling, buffer expiry, landmark smoothing |
| `augmentation.py` | augmentation of raw landmarks + transition windows |
| `temporal.py` | `PredictionGate`: confidence, margin, hysteresis, hold, expiry |
| `hand_detection.py` | MediaPipe configuration and stable two-hand selection |
| `checkpoint.py` | load/save checkpoints with their feature version; refuses to overwrite the original |
| `metrics.py` | shared metrics so training and evaluation compute numbers the same way |
| `evaluate.py` | reproducible evaluation (clean, perturbations, rejection, sweep, transitions, latency) |
| `tests/test_pipeline.py` | unit tests |

---

## 12. Collecting new data

`tool/dataset_extractor.py` was updated so that:

* a frame is saved **only when a hand was detected** (the old version stored
  `zeros(63)`), so "no hand" can never again be learned as a gesture;
* it uses the same `hand_detection` configuration as inference;
* it shows the tracked hand and the frame counter while recording.

```bash
python tool/dataset_extractor.py      # run from the project root
```

To make the model generalise to other people and venues (the biggest remaining
limitation), record **3-5 volunteers x 2 sessions x 4 lighting conditions** and
keep them in a separate data root, e.g. `MP_Data_extra/`, instead of replacing
`MP_Data/`. Nothing in the repository requires that data to exist; without it
cross-user accuracy cannot be measured.

---

## 13. What is measured, what is simulated, what still needs recordings

**Measured on real data**

* window accuracy, per-class precision/recall/F1, confusion matrix, macro-F1,
* accuracy under simulated landmark corruptions (translation, scale, rotation,
  jitter, landmark loss, detection gaps),
* rejection of empty / partially missing / ambiguous windows,
* grouped CV and temporal hold-out accuracy,
* transition latency and flicker on stitched streams,
* classifier latency and FPS.

**Simulated (not real recordings)**

* the transition streams are two real recordings concatenated with an
  instantaneous switch; a human gesture change is smoother,
* "occlusion" zeroes landmarks; MediaPipe normally returns *distorted*
  landmarks instead, which is a different error,
* the negative sets (empty, gaps, mixed) are synthesised from the same
  recordings,
* brightness / contrast changes are **not** simulated at all, because the
  training data is landmarks, not images.

**Still requires real recordings (cannot be claimed yet)**

* detection reliability in low light, backlight, shadows, overexposure,
  complex backgrounds, other cameras and resolutions,
* accuracy for other people, skin tones, hand sizes, left hands and distances,
* real occlusion, real motion blur, real fast transitions,
* end-to-end FPS on the target machine and webcam.

The application shows detection state and FPS live, so a session in the actual
venue is the natural next step; no number above is extrapolated from another
dataset.

### Remaining limitations

* **One session, one camera, one person.** The model has never seen a second
  user; the cross-user number is unknown, not merely low.
* **Aspect-ratio** distortion (x and y normalised by different scales) is
  reduced but not removed; it costs up to ~5 points in the worst case measured.
* **Z depth** is MediaPipe's least reliable coordinate and is still used; world
  landmarks are available in the API and would be the better long-term source,
  but the existing dataset does not contain them, so switching now would make
  training and inference inconsistent.
* The **no-gesture state is a rejection rule, not a trained class**: a
  high-confidence wrong gesture cannot be rejected, only an uncertain one.
  A real `no_gesture` recording set is the clean fix.
* MediaPipe occasionally **re-acquires the hand**; the gate hides the blip but
  the 50-frame window still has to refill (about 1.7 s at 30 fps).
* The environment has `numpy 2.5.5` with a `scipy 1.15.2` that expects
  `numpy<2.5`; it only produces a warning today, but pinning the versions is
  recommended for a clean deployment.

---

## 14. Running the original model

The original checkpoint is untouched and still loads (it has no
`feature_version`, so it is treated as `v1`):

```bash
python main.py --checkpoint trained_model/wl_hand_gesture_transformer.pth
```

The enhanced pipeline (buffer, gate, two-hand selection) is still applied around
it - that is the point of the separation - and `evaluate.py` compares the two
models under identical conditions:

```bash
python evaluate.py --checkpoint trained_model/wl_hand_gesture_transformer.pth
python evaluate.py --checkpoint trained_model/wl_hand_gesture_transformer_v2.pth
```

To reproduce the original behaviour exactly, disable the new safety nets:

```bash
python main.py --checkpoint trained_model/wl_hand_gesture_transformer.pth `
               --confidence 0 --margin 0 --switch-frames 1 --hold-frames 1 `
               --expire-frames 1 --num-hands 1
```
(PowerShell line continuation shown; on bash use `\`.)

---

## 15. Files changed

| File | Change |
|---|---|
| `tool/kp_extractor.py` | added feature version `v2` (wrist-centre, scale, rotation), validity helpers, missing-frame repair; kept `extract_keypoints` and `v1` intact |
| `tool/dir_generator.py` | absolute `MP_Data` path, folder creation moved behind `__main__` |
| `tool/dataset_extractor.py` | saves only detected frames, uses the shared detection config, shows tracking state |
| `transformer.py` | added `build_hand_gesture_transformer()` factory only - **no architecture change** |
| `dataset_loader.py` | in-memory cache, feature version, min-valid-frame filter, augmentation, transition windows, grouped and temporal splits, seeded loaders |
| `train.py` | CLI, seeds, temporal hold-out + grouped CV, best checkpoint, scheduler, label smoothing, metrics, plots |
| `main.py` | buffer + gate + two-hand selection, thresholds, landmark overlay, monotonic timestamps, `--checkpoint` |
| `README.md` | rewritten (this file); original kept as `README_original.md` |
| **new** `preprocessing.py`, `augmentation.py`, `temporal.py`, `hand_detection.py`, `checkpoint.py`, `metrics.py`, `evaluate.py`, `tests/` | see section 11 |
| **new** `trained_model/wl_hand_gesture_transformer_v2.pth` and reports | enhanced model + measurements |

Untouched: `MP_Data/`, `trained_model/wl_hand_gesture_transformer.pth`,
`hand_landmarker.task`, `test.py`, and the four gesture labels.

---

## 16. Tests

```bash
python -m unittest discover -s tests -v      # 47 tests, well under a second
```

| Group | What it pins down |
|---|---|
| `FeatureTests` | v2 is invariant to translation, scale and in-plane rotation; it preserves the palm/back twist; v1 is unchanged; invalid frames stay zero |
| `MissingFrameTests` | tracking-hold repair, hold limits, all-zero windows are not invented |
| `BufferTests` | window length, expiry after loss, minimum real frames, smoothing |
| `GateTests` | acquisition, ambiguity rejection, hysteresis, minimum hold, expiry, invalid input |
| `HandSelectionTests` | same hand stays selected, handedness preference, re-acquisition, reset |
| `AugmentationTests` | source untouched, shapes/labels, reproducibility, mirror off by default, transition labelling |
| `SplitTests` / `DatasetTests` | folds do not overlap, temporal block, caching, min-valid filtering, transition samples |
| `CheckpointTests` | round trip keeps the feature version, the original checkpoint still loads as v1, overwriting it is refused |
| `MetricTests` | confusion matrix layout, per-class metrics, calibration error |







