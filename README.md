# LivePortrait — Real-Time Webcam Demo

Animate a portrait image in real time using your webcam as the driving source.
Three panels are shown side-by-side: **webcam feed · source portrait · animated result**.

![demo panels: webcam | source | animated]

---

## How it works

| Component                        | Role                                                                                                      |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Appearance extractor F** | Encodes the source portrait into a 3-D feature volume`f_s` (32×16×64×64) — computed once at startup |
| **Motion extractor M**     | Predicts 21 canonical 3-D keypoints + head pose from each webcam frame                                    |
| **Warping module W**       | Warps`f_s` using the relative keypoint motion                                                           |
| **SPADE generator G**      | Decodes the warped features into a 512×512 animated image                                                |
| **Stitching network S**    | Blends boundaries to keep the background intact                                                           |

---

## Requirements

- **OS:** Windows 10/11, Linux, or macOS
- **GPU:** NVIDIA GPU with CUDA 11.8+ recommended (RTX 30xx / 40xx or better)
- **Python:** 3.10 or 3.11
- **Pretrained weights:** see [Pretrained Weights](#pretrained-weights)

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-repo/LivePortrait.git
cd LivePortrait

# 2. Create a conda environment
conda create -n liveportrait python=3.10
conda activate liveportrait

# 3. Install PyTorch with CUDA (adjust cu121 to match your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt
```

### Pretrained Weights

Download and place under `pretrained_weights/`:

```
pretrained_weights/
  insightface/
    models/
      buffalo_l/         ← face detection (RetinaFace)
  liveportrait/
    base_models/
      appearance_feature_extractor.pth
      motion_extractor.pth
      warping_module.pth
      spade_generator.pth
    retargeting_models/
      stitching_retargeting_module.pth
    landmark.onnx        ← 203-point landmark refiner
```

---

## Quick Run

```bash
python webcam_inference.py --source assets/examples/source/s6.jpg
```

---

## All Arguments

| Argument                 | Default        | Description                                                                                                                                 |
| ------------------------ | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `--source` / `-s`    | *(required)* | Path to source portrait image (JPG/PNG)                                                                                                     |
| `--camera` / `-c`    | `0`          | OpenCV camera index (`0` = built-in, `1` = first external USB)                                                                          |
| `--width`              | `640`        | Camera capture width                                                                                                                        |
| `--height`             | `480`        | Camera capture height                                                                                                                       |
| `--driving_multiplier` | `1.0`        | Motion amplitude scale —`2.0` = exaggerated, `0.5` = subtle                                                                            |
| `--det_interval`       | `30`         | Run full RetinaFace detection every N frames; track landmarks cheaply between. Lower = more robust, slower                                  |
| `--no_crop_driving`    | off            | Skip face detection entirely — fastest mode, use when your face already fills the frame                                                    |
| `--compile`            | off            | `torch.compile` the W+G networks. On Linux+Triton: kernel fusion (~30% faster after warm-up). On Windows: CUDA graph replay (~10% faster) |
| `--decode_every`       | `1`          | Render the animated output only every N frames; show previous result on skipped frames.`3` → ~20 FPS on RTX 4060                         |
| `--profile`            | off            | Print per-step timing every 30 frames to identify bottlenecks                                                                               |

### Keyboard shortcuts while running

| Key             | Action                                                     |
| --------------- | ---------------------------------------------------------- |
| `q` / `ESC` | Quit                                                       |
| `r`           | Reset reference pose (re-calibrate to current webcam pose) |

---

## Performance Guide (RTX 4060, Windows)

| Command                                   | FPS                         |
| ----------------------------------------- | --------------------------- |
| Default (original)                        | ~6                          |
| `--det_interval 30` (landmark tracking) | ~18                         |
| `+ --compile`                           | ~20                         |
| `+ --decode_every 3`                    | **~20 FPS**           |
| `--no_crop_driving --compile`           | ~25 (no head-pose tracking) |

The SPADE generator outputs 512×512 using 6 SPADE ResnetBlocks — this is the hard GPU ceiling (~90 ms/frame on RTX 4060 without Triton). `--decode_every` bypasses it by reusing the previous rendered frame on skipped frames while keeping keypoint tracking responsive every frame.

### Linux / WSL2 with Triton

```bash
pip install triton
python webcam_inference.py --source s6.jpg --compile --decode_every 2
```

With Triton, `--compile` uses `max-autotune` kernel fusion and can cut GPU decode time roughly in half, allowing `--decode_every 2` to hit 25+ FPS with better output quality.

---

## Feature Vector Sizes

| Tensor                                 | Shape                 | Values    | Role                              |
| -------------------------------------- | --------------------- | --------- | --------------------------------- |
| `f_s` appearance volume              | `1×32×16×64×64` | 2,097,152 | Texture/appearance of source face |
| Keypoints`kp`                        | `1×21×3`          | 63        | Canonical 3-D keypoint positions  |
| Expression`exp`                      | `1×21×3`          | 63        | Per-keypoint deformation offsets  |
| Transformed keypoints`x_s` / `x_d` | `1×21×3`          | 63        | Final driving keypoints per frame |

`f_s` is computed once from the source image and reused every frame. Only the 63-float keypoint vectors are recomputed each frame from the webcam feed.

---

## Project Structure

```
LivePortrait/
├── webcam_inference.py          ← main entry point
├── assets/examples/source/      ← sample source portraits
├── pretrained_weights/          ← model checkpoints (download separately)
└── src/
    ├── live_portrait_wrapper.py ← model orchestration
    ├── config/
    │   ├── inference_config.py
    │   ├── crop_config.py
    │   └── models.yaml          ← architecture hyperparameters
    ├── modules/                 ← neural network definitions (F, M, W, G, S)
    │   ├── appearance_feature_extractor.py
    │   ├── motion_extractor.py
    │   ├── warping_network.py
    │   ├── dense_motion.py
    │   ├── spade_generator.py
    │   ├── stitching_retargeting_network.py
    │   ├── convnextv2.py
    │   └── util.py
    └── utils/
        ├── cropper.py           ← face detection + landmark pipeline
        ├── human_landmark_runner.py
        ├── face_analysis_diy.py
        ├── camera.py
        ├── crop.py
        ├── io.py
        ├── retargeting_utils.py
        ├── helper.py
        ├── timer.py
        └── rprint.py
```
