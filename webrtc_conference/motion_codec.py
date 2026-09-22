# coding: utf-8
"""
LivePortrait as a conferencing video codec.

Sender side (`MotionEncoder`) runs only the motion extractor M on the cropped
webcam face and emits the implicit-keypoint state: head pose, scale, translation
and the 21x3 expression deltas — 70 numbers, 140 bytes quantised.

Receiver side (`AvatarRenderer`) holds the peer's source portrait, so it has
already paid for the expensive parts once: the appearance volume f_s from F and
the transformed source keypoints x_s. Each arriving motion vector is composed
against that using the same relative-motion equation as the offline pipeline,
then warped and decoded by W+G into a 512x512 frame.

The reference pose (the peer's neutral frame) is established by the first packet
of a stream, or by any packet carrying FLAG_CALIBRATE. Sender and receiver
therefore never need to agree on anything beyond the packets themselves.
"""

import os
import sys
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils import cropper as _cropper_mod
from src.utils.camera import get_rotation_matrix
from src.utils.crop import crop_image
from src.utils.cropper import Cropper
from src.utils.io import load_image_rgb, resize_to_limit

RENDER_SIZE = 512


def _throttle_cropper_logging(period: float = 5.0):
    """The cropper logs once per failed detection, which at 20 fps buries the
    console whenever someone steps away from the camera. Collapse repeats."""
    original = _cropper_mod.log
    last: dict[str, float] = {}

    def throttled(message, *args, **kwargs):
        import time as _time
        now = _time.monotonic()
        key = str(message)
        if now - last.get(key, -1e9) < period:
            return
        last[key] = now
        return original(message, *args, **kwargs)

    _cropper_mod.log = throttled


_throttle_cropper_logging()


@dataclass
class MotionState:
    """The per-frame quantities that actually travel over the wire."""
    pitch: float
    yaw: float
    roll: float
    scale: float
    t: np.ndarray               # (3,)
    exp: np.ndarray             # (21, 3)


class LivePortraitEngine:
    """Owns the four networks once per process; encoders and renderers share it.

    The wrapper is not re-entrant, so every call is serialised behind one lock.
    In practice that is what we want anyway: a single GPU rendering N peers is
    better off round-robining than thrashing between contexts.
    """

    def __init__(self, force_cpu: bool = False, half: bool = True):
        cfg = InferenceConfig()
        cfg.flag_force_cpu = force_cpu
        cfg.flag_use_half_precision = half and not force_cpu
        self.inf_cfg = cfg
        self.crop_cfg = CropConfig()
        self.wrapper = LivePortraitWrapper(inference_cfg=cfg)
        self.cropper = Cropper(crop_cfg=self.crop_cfg)
        self.device = self.wrapper.device
        self.lock = threading.RLock()


class MotionEncoder:
    """Webcam frame -> MotionState. Runs face cropping plus the motion extractor."""

    def __init__(self, engine: LivePortraitEngine, det_interval: int = 30,
                 crop_driving: bool = True):
        self.engine = engine
        self.det_interval = det_interval
        self.crop_driving = crop_driving
        self._prev_lmk: Optional[np.ndarray] = None
        self._since_detect = 0
        self.last_crop_bgr: Optional[np.ndarray] = None

    def reset_tracking(self) -> None:
        self._prev_lmk = None
        self._since_detect = 0

    def extract(self, frame_bgr: np.ndarray) -> Optional[MotionState]:
        """Return the motion state for this frame, or None if no face is visible."""
        crop_rgb = self._crop(frame_bgr)
        if crop_rgb is None:
            return None
        self.last_crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

        with self.engine.lock:
            I_d = self.engine.wrapper.prepare_source(crop_rgb)
            info = self.engine.wrapper.get_kp_info(I_d)

        return MotionState(
            pitch=float(info["pitch"].item()),
            yaw=float(info["yaw"].item()),
            roll=float(info["roll"].item()),
            scale=float(info["scale"].item()),
            t=info["t"][0].detach().cpu().numpy().astype(np.float32),
            exp=info["exp"][0].detach().cpu().numpy().astype(np.float32),
        )

    def _crop(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if not self.crop_driving:
            return cv2.resize(frame_rgb, (256, 256))

        cropper, crop_cfg = self.engine.cropper, self.engine.crop_cfg
        need_detect = self._prev_lmk is None or self._since_detect >= self.det_interval

        if need_detect:
            try:
                with self.engine.lock:
                    info = cropper.crop_source_image(frame_rgb, crop_cfg)
            except Exception:
                info = None
            if info is None:
                self.reset_tracking()
                return None
            self._prev_lmk = info["lmk_crop"]
            self._since_detect = 1
            return info["img_crop_256x256"]

        # Cheap path: anchor the landmark model on the previous landmarks.
        try:
            with self.engine.lock:
                lmk = cropper.human_landmark_runner.run(frame_rgb, self._prev_lmk)
            ret = crop_image(
                frame_rgb, lmk,
                dsize=crop_cfg.dsize, scale=crop_cfg.scale,
                vx_ratio=crop_cfg.vx_ratio, vy_ratio=crop_cfg.vy_ratio,
                flag_do_rot=crop_cfg.flag_do_rot,
            )
        except Exception:
            self.reset_tracking()
            return None

        self._prev_lmk = lmk
        self._since_detect += 1
        return cv2.resize(ret["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)


class AvatarRenderer:
    """Holds one peer's source portrait and re-synthesises them from motion packets."""

    def __init__(self, engine: LivePortraitEngine, source_rgb: np.ndarray,
                 driving_multiplier: float = 1.0):
        self.engine = engine
        self.driving_multiplier = driving_multiplier
        cfg = engine.inf_cfg
        w = engine.wrapper

        img = resize_to_limit(source_rgb, cfg.source_max_dim, cfg.source_division)
        with engine.lock:
            crop_info = engine.cropper.crop_source_image(img, engine.crop_cfg)
        if crop_info is None:
            raise ValueError("no face detected in the source portrait")

        crop_256 = crop_info["img_crop_256x256"]
        source_lmk = crop_info["lmk_crop"]
        self.source_bgr_256 = cv2.cvtColor(crop_256, cv2.COLOR_RGB2BGR)

        with engine.lock:
            I_s = w.prepare_source(crop_256)
            self.x_s_info = w.get_kp_info(I_s)
            self.x_c_s = self.x_s_info["kp"]
            self.R_s = get_rotation_matrix(
                self.x_s_info["pitch"], self.x_s_info["yaw"], self.x_s_info["roll"])
            self.f_s = w.extract_feature_3d(I_s)
            self.x_s = w.transform_keypoint(self.x_s_info)

            self.lip_delta = None
            if cfg.flag_normalize_lip and cfg.flag_relative_motion and source_lmk is not None:
                ratio = w.calc_combined_lip_ratio([0.0], source_lmk)
                if ratio[0][0] >= cfg.lip_normalize_threshold:
                    self.lip_delta = w.retarget_lip(self.x_s, ratio)

        self._ref: Optional[dict] = None
        self.last_frame_bgr = np.zeros((RENDER_SIZE, RENDER_SIZE, 3), dtype=np.uint8)

    def reset_reference(self) -> None:
        self._ref = None

    def render(self, pkt) -> np.ndarray:
        """Compose an incoming MotionPacket against the source and decode a frame."""
        device = self.engine.device
        w = self.engine.wrapper

        pitch = torch.tensor([[pkt.pitch]], dtype=torch.float32, device=device)
        yaw = torch.tensor([[pkt.yaw]], dtype=torch.float32, device=device)
        roll = torch.tensor([[pkt.roll]], dtype=torch.float32, device=device)
        scale = torch.tensor([[pkt.scale]], dtype=torch.float32, device=device)
        t = torch.from_numpy(pkt.t).to(device).float().reshape(1, 3)
        exp = torch.from_numpy(pkt.exp).to(device).float().reshape(1, -1, 3)
        R_d = get_rotation_matrix(pitch, yaw, roll)

        if self._ref is None or pkt.calibrate:
            self._ref = {"R": R_d.clone(), "exp": exp.clone(),
                         "scale": scale.clone(), "t": t.clone()}

        ref = self._ref
        with self.engine.lock:
            R_new = (R_d @ ref["R"].permute(0, 2, 1)) @ self.R_s
            delta_new = self.x_s_info["exp"] + (exp - ref["exp"])
            scale_new = self.x_s_info["scale"] * (scale / ref["scale"])
            t_new = self.x_s_info["t"] + (t - ref["t"])
            t_new[..., 2].fill_(0)

            x_d = scale_new * (self.x_c_s @ R_new + delta_new) + t_new
            if self.engine.inf_cfg.flag_stitching:
                x_d = w.stitching(self.x_s, x_d)
            if self.lip_delta is not None:
                x_d = x_d + self.lip_delta
            x_d = self.x_s + (x_d - self.x_s) * self.driving_multiplier

            out = w.warp_decode(self.f_s, self.x_s, x_d)
            rgb = w.parse_output(out["out"])[0]

        self.last_frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.last_frame_bgr


# ── avatar image helpers ─────────────────────────────────────────────────────

def load_avatar_rgb(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load_image_rgb(path)


def encode_avatar_jpeg(rgb: np.ndarray, max_dim: int = 720, quality: int = 90) -> bytes:
    """JPEG-compress a portrait for the one-time transfer at call setup."""
    h, w = rgb.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to JPEG-encode avatar")
    return buf.tobytes()


def decode_avatar_jpeg(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("corrupt avatar JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
