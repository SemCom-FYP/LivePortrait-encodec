# coding: utf-8
"""
Real-time LivePortrait webcam demo.

Shows three panels side-by-side:
  [Webcam / driving]  |  [Source portrait]  |  [Animated result]

Usage:
    python webcam_inference.py --source assets/examples/source/s6.jpg
    python webcam_inference.py --source path/to/face.jpg --camera 1

Keyboard shortcuts while running:
    q / ESC  — quit
    r        — reset the reference driving frame (re-calibrate to current webcam pose)

Performance tips:
    --det_interval 30   Re-run face detection every N frames; track landmarks between (default: 30)
    --no_crop_driving   Skip face detection entirely (fastest; works when face fills frame)
    --compile           Enable torch.compile on W+G networks (~30 s warm-up, then faster)
    --width 320 --height 240  Smaller capture → faster face detection
"""

import sys
import os
import argparse
import time
import threading

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix
from src.utils.io import load_image_rgb, resize_to_limit
from src.utils.crop import crop_image


# ──────────────────────────────────────────────────────────────────────────────
# Threaded camera reader — keeps the latest frame in a slot so inference
# never blocks waiting for cap.read().
# ──────────────────────────────────────────────────────────────────────────────

class CameraReader:
    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._running = False
        self._cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_label(img, text, pos=(8, 22), color=(0, 255, 80)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


DISPLAY_SIZE = 512


def make_display(driving_bgr, source_bgr, result_bgr, fps=None):
    panels = [cv2.resize(p, (DISPLAY_SIZE, DISPLAY_SIZE)) for p in
              (driving_bgr, source_bgr, result_bgr)]
    labels = ["Webcam (driving)", "Source portrait", "Animated result"]
    for panel, label in zip(panels, labels):
        draw_label(panel, label)
    row = np.hstack(panels)
    if fps is not None:
        cv2.putText(row, f"FPS: {fps:.1f}", (row.shape[1] - 110, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Fast crop via landmark tracking (no RetinaFace).
# Uses the previous 203-pt landmarks to anchor the landmark ONNX model,
# then crops the face with the new landmarks — same as crop_source_video tracking.
# ──────────────────────────────────────────────────────────────────────────────

def track_crop(frame_rgb: np.ndarray, prev_lmk: np.ndarray,
               landmark_runner, crop_cfg: CropConfig):
    """Return (img_crop_256, new_lmk) using landmark tracking only (no face detector)."""
    lmk = landmark_runner.run(frame_rgb, prev_lmk)
    ret_dct = crop_image(
        frame_rgb, lmk,
        dsize=crop_cfg.dsize,
        scale=crop_cfg.scale,
        vx_ratio=crop_cfg.vx_ratio,
        vy_ratio=crop_cfg.vy_ratio,
        flag_do_rot=crop_cfg.flag_do_rot,
    )
    img_crop_256 = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
    return img_crop_256, lmk


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LivePortrait real-time webcam demo")
    parser.add_argument("--source", "-s", required=True,
                        help="Path to source portrait image (JPG/PNG)")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="OpenCV camera device index (default: 0)")
    parser.add_argument("--width",  type=int, default=640, help="Camera capture width")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height")
    parser.add_argument("--driving_multiplier", type=float, default=1.0,
                        help="Scale factor for motion amplitude (default: 1.0)")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-step timing every 30 frames to identify the bottleneck")
    parser.add_argument("--decode_every", type=int, default=1,
                        help="Run warp+decode only every N frames; show previous result on skipped frames. "
                             "2 → ~14 FPS, 3 → ~18 FPS. Tracking stays responsive on all frames.")
    parser.add_argument("--no_crop_driving", action="store_true",
                        help="Skip face detection entirely (fastest; use when face fills the frame)")
    parser.add_argument("--det_interval", type=int, default=30,
                        help="Run full RetinaFace detection every N frames; "
                             "track landmarks between (default: 30). Lower = more robust, slower.")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the W+G networks for extra GPU speed "
                             "(~30 s one-time warm-up; RTX 40xx recommended)")
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────────────
    inf_cfg = InferenceConfig()
    inf_cfg.driving_multiplier = args.driving_multiplier
    crop_cfg = CropConfig()

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading LivePortrait models …")
    wrapper = LivePortraitWrapper(inference_cfg=inf_cfg)
    cropper = Cropper(crop_cfg=crop_cfg)
    device  = wrapper.device

    if args.compile:
        torch._dynamo.config.suppress_errors = True
        try:
            import triton  # noqa: F401
            compile_kwargs = dict(mode="max-autotune")
            print("Compiling W+G with torch.compile max-autotune (first ~30 s slow) …")
        except ImportError:
            # Triton is Linux-only; cudagraphs backend works on Windows
            compile_kwargs = dict(backend="cudagraphs")
            print("Triton not available (Windows) — using cudagraphs backend instead …")
        try:
            wrapper.warping_module  = torch.compile(wrapper.warping_module,  **compile_kwargs)
            wrapper.spade_generator = torch.compile(wrapper.spade_generator, **compile_kwargs)
            wrapper.compile = True  # tells warp_decode to call cudagraph_mark_step_begin()
        except Exception as e:
            print(f"torch.compile failed: {e}\nContinuing without compilation.")

    # ── Load & pre-process source image (done once) ───────────────────────────
    print(f"Loading source image: {args.source}")
    if not os.path.exists(args.source):
        print(f"ERROR: source file not found: {args.source}")
        sys.exit(1)

    img_rgb = load_image_rgb(args.source)
    img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)

    crop_info = cropper.crop_source_image(img_rgb, crop_cfg)
    if crop_info is None:
        print("ERROR: No face detected in the source image.")
        sys.exit(1)

    img_crop_256  = crop_info["img_crop_256x256"]
    source_lmk    = crop_info["lmk_crop"]

    I_s      = wrapper.prepare_source(img_crop_256)
    x_s_info = wrapper.get_kp_info(I_s)
    x_c_s    = x_s_info["kp"]
    R_s      = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
    f_s      = wrapper.extract_feature_3d(I_s)
    x_s      = wrapper.transform_keypoint(x_s_info)

    lip_delta_before_animation = None
    if inf_cfg.flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
        c_d_lip_before = [0.0]
        combined_lip_ratio = wrapper.calc_combined_lip_ratio(c_d_lip_before, source_lmk)
        if combined_lip_ratio[0][0] >= inf_cfg.lip_normalize_threshold:
            lip_delta_before_animation = wrapper.retarget_lip(x_s, combined_lip_ratio)

    source_bgr_256 = cv2.cvtColor(img_crop_256, cv2.COLOR_RGB2BGR)

    # ── Open webcam ───────────────────────────────────────────────────────────
    print(f"Opening camera {args.camera} …")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)

    cam = CameraReader(cap)

    print("Running!  Press  q / ESC  to quit,  r  to reset reference frame.")
    cv2.namedWindow("LivePortrait Realtime", cv2.WINDOW_NORMAL)

    R_d_0      = None
    x_d_0_info = None

    blank = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 3), dtype=np.uint8)
    draw_label(blank, "waiting…", color=(100, 100, 100))

    fps_acc   = 0.0
    fps_alpha = 0.1
    t_prev    = time.perf_counter()

    # Landmark tracking state
    prev_lmk   = None   # 203-pt landmarks in original-frame coords from last detection
    det_counter = 0     # frames since last full RetinaFace detection
    frame_count = 0
    result_bgr  = np.zeros((512, 512, 3), dtype=np.uint8)  # placeholder until first decode

    while True:
        t_frame_start = time.perf_counter()
        ret, frame_bgr = cam.read()
        frame_count += 1
        if not ret or frame_bgr is None:
            time.sleep(0.001)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── Driving face crop ─────────────────────────────────────────────────
        if args.no_crop_driving:
            driving_256     = cv2.resize(frame_rgb, (256, 256))
            driving_bgr_256 = cv2.resize(frame_bgr, (256, 256))

        else:
            do_full_detect = (prev_lmk is None) or (det_counter >= args.det_interval)

            if do_full_detect:
                # Full detection: RetinaFace + landmark runner (expensive, runs every det_interval frames)
                try:
                    d_crop = cropper.crop_source_image(frame_rgb, crop_cfg)
                except Exception:
                    d_crop = None

                if d_crop is None:
                    prev_lmk = None
                    det_counter = 0
                    cam_panel = cv2.resize(frame_bgr, (256, 256))
                    draw_label(cam_panel, "No face detected", color=(0, 0, 220))
                    cv2.imshow("LivePortrait Realtime",
                               make_display(cam_panel, source_bgr_256, blank))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    continue

                prev_lmk    = d_crop["lmk_crop"]   # 203-pt in original-frame coords
                det_counter = 0
                driving_256 = d_crop["img_crop_256x256"]

            else:
                # Cheap tracking: landmark ONNX only (no face detector)
                try:
                    driving_256, prev_lmk = track_crop(
                        frame_rgb, prev_lmk,
                        cropper.human_landmark_runner, crop_cfg,
                    )
                except Exception:
                    prev_lmk = None   # lost face — force re-detect next frame
                    det_counter = 0
                    continue

            det_counter += 1
            driving_bgr_256 = cv2.cvtColor(driving_256, cv2.COLOR_RGB2BGR)

        # ── Extract driving keypoints ─────────────────────────────────────────
        _t0 = time.perf_counter()
        I_d        = wrapper.prepare_source(driving_256)
        x_d_i_info = wrapper.get_kp_info(I_d)
        R_d_i      = get_rotation_matrix(x_d_i_info["pitch"], x_d_i_info["yaw"], x_d_i_info["roll"])
        _t1 = time.perf_counter()

        if R_d_0 is None:
            R_d_0 = R_d_i.clone()
            x_d_0_info = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in x_d_i_info.items()
            }
            print("Reference frame set.")

        # ── Compose driving keypoints (relative motion) ───────────────────────
        R_new     = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
        delta_new = x_s_info["exp"] + (x_d_i_info["exp"] - x_d_0_info["exp"])
        scale_new = x_s_info["scale"] * (x_d_i_info["scale"] / x_d_0_info["scale"])
        t_new     = x_s_info["t"]   + (x_d_i_info["t"]   - x_d_0_info["t"])
        t_new[..., 2].fill_(0)

        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

        if inf_cfg.flag_stitching:
            x_d_i_new = wrapper.stitching(x_s, x_d_i_new)
        if lip_delta_before_animation is not None:
            x_d_i_new = x_d_i_new + lip_delta_before_animation

        x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier

        # ── Warp & decode (optionally skipped for speed) ──────────────────────
        _t2 = time.perf_counter()
        if frame_count % args.decode_every == 0:
            out        = wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i      = wrapper.parse_output(out["out"])[0]
            result_bgr = cv2.cvtColor(I_p_i, cv2.COLOR_RGB2BGR)
        # else: reuse result_bgr from previous decoded frame
        _t3 = _t4 = time.perf_counter()

        # ── FPS ───────────────────────────────────────────────────────────────
        t_now    = time.perf_counter()
        inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev   = t_now
        fps_acc  = fps_alpha * inst_fps + (1 - fps_alpha) * fps_acc if fps_acc > 0 else inst_fps

        if args.profile and frame_count % 30 == 1:
            crop_ms    = (_t0 - t_frame_start) * 1000
            motion_ms  = (_t1 - _t0) * 1000
            warp_ms    = (_t3 - _t2) * 1000
            output_ms  = (_t4 - _t3) * 1000
            total_ms   = (_t4 - t_frame_start) * 1000
            print(f"[profile] crop={crop_ms:5.1f}ms  motion={motion_ms:5.1f}ms  "
                  f"warp={warp_ms:5.1f}ms  output={output_ms:5.1f}ms  total={total_ms:5.1f}ms")

        # ── Display ───────────────────────────────────────────────────────────
        display = make_display(driving_bgr_256, source_bgr_256, result_bgr, fps=fps_acc)
        cv2.imshow("LivePortrait Realtime", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            R_d_0 = None
            x_d_0_info = None
            prev_lmk = None
            det_counter = 0
            print("Reference frame cleared — will re-calibrate on next face detection.")

    cam.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
