# Quickstart

Minimum steps to go from a fresh clone to a running webcam demo.

---

## Step 1 — Environment

```bash
conda create -n liveportrait python=3.10 -y
conda activate liveportrait
```

## Step 2 — PyTorch + CUDA

Find your CUDA version: `nvidia-smi` (top-right corner shows e.g. `CUDA Version: 12.1`)

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Step 3 — Dependencies

```bash
pip install -r requirements.txt
```

## Step 4 — Pretrained weights

Download from the [official HuggingFace repo](https://huggingface.co/KwaiVGI/LivePortrait) and place under `pretrained_weights/`:

```
pretrained_weights/
  insightface/models/buffalo_l/     ← RetinaFace face detector
  liveportrait/
    base_models/
      appearance_feature_extractor.pth
      motion_extractor.pth
      warping_module.pth
      spade_generator.pth
    retargeting_models/
      stitching_retargeting_module.pth
    landmark.onnx
```

## Step 5 — Run

```bash
# Basic
python webcam_inference.py --source assets/examples/source/s6.jpg

# Best performance on most GPUs (RTX 30xx / 40xx, Windows)
python webcam_inference.py --source assets/examples/source/s6.jpg --compile --decode_every 3

# If your face fills the frame and you want maximum FPS
python webcam_inference.py --source assets/examples/source/s6.jpg --no_crop_driving --compile
```

Press `q` or `ESC` to quit. Press `r` to reset the reference pose.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No face detected` in webcam panel | Move closer to camera, improve lighting, or use `--no_crop_driving` |
| Very low FPS (~6) | Add `--compile --decode_every 3` |
| `TritonMissing` warning on Windows | Normal — falls back to cudagraphs automatically |
| Camera not opening | Try `--camera 1` (or `2`) for external USB cameras |
| Out of GPU memory | Add `--width 320 --height 240` to reduce capture resolution |
| Face crop drifts when moving fast | Lower `--det_interval` (e.g. `--det_interval 10`) |
