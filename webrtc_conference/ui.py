# coding: utf-8
"""OpenCV front end for the conference: a pre-join lobby and the call grid.

Lobby   — pick a reference portrait (file, example or webcam capture), a display
          name, and either create a new room or join a live one.
CallView — the in-call grid with per-tile bandwidth readouts, clickable
          controls and transient toasts.

Everything is drawn with OpenCV primitives. Text goes through Pillow when a
TrueType font can be found, because Hershey fonts are both ugly and ASCII-only
(the client's status strings use "…"); without Pillow it falls back to cv2.
"""

import json
import math
import os
import random
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:                                   # pragma: no cover
    Image = None

# ── palette (BGR) ────────────────────────────────────────────────────────────

BG = (22, 18, 16)
SURFACE = (38, 32, 29)
SURFACE_HI = (52, 45, 40)
BORDER = (72, 63, 57)
TEXT = (242, 238, 235)
MUTED = (158, 148, 140)
ACCENT = (250, 145, 70)                             # blue
GREEN = (120, 215, 110)
AMBER = (60, 190, 255)
GREY = (150, 150, 150)
RED = (85, 85, 235)

FONT = cv2.FONT_HERSHEY_SIMPLEX
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".liveportrait_conference")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Tile:
    title: str
    image: Optional[np.ndarray]          # BGR, any size
    subtitle: str = ""
    status: str = ""                     # shown centred when `image` is None
    accent: tuple = GREEN
    inset: Optional[np.ndarray] = None   # picture-in-picture, top-right corner
    inset_label: str = ""
    badge: str = ""                      # chip in the top-left corner
    badge_color: tuple = ACCENT


# ── text ─────────────────────────────────────────────────────────────────────

_FONT_FILES = {
    False: ["C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    True: ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf",
           "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}


@lru_cache(maxsize=64)
def _font(size: int, bold: bool):
    if Image is None:
        return None
    for path in _FONT_FILES[bold]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return None


class Canvas:
    """A BGR image plus a queue of text draws, rendered in one Pillow pass.

    Shapes go straight onto `img`; text is deferred until `flush()`, so call it
    before drawing anything that must cover text already queued.
    """

    def __init__(self, img: np.ndarray):
        self.img = img
        self._texts = []

    def measure(self, text: str, size: int, bold: bool = False) -> tuple[int, int]:
        font = _font(size, bold)
        if font is None:
            (w, h), _ = cv2.getTextSize(_ascii(text), FONT, size / 30, _cv_thick(size, bold))
            return w, h
        l, _, r, _ = font.getbbox(text)
        return int(r - l), size

    def fit(self, text: str, size: int, max_w: int, bold: bool = False) -> str:
        """Truncate with an ellipsis so `text` fits in `max_w` pixels."""
        if self.measure(text, size, bold)[0] <= max_w:
            return text
        ell = "…" if _font(size, bold) is not None else "..."
        while text and self.measure(text + ell, size, bold)[0] > max_w:
            text = text[:-1]
        return text.rstrip() + ell

    def text(self, text: str, x: float, y: float, size: int, color=TEXT,
             bold: bool = False, anchor: str = "lm", outline: bool = False):
        """`anchor`: horizontal l/m/r then vertical a (top) / m (middle)."""
        if text:
            self._texts.append((text, int(x), int(y), size, color, bold, anchor, outline))

    def flush(self) -> np.ndarray:
        if not self._texts:
            return self.img
        if _font(14, False) is None:
            for item in self._texts:
                self._cv_text(*item)
        else:
            # The array is BGR and the colours are BGR, so Pillow never needs to
            # know: it just sees three channels in the same order on both sides.
            pil = Image.fromarray(self.img)
            draw = ImageDraw.Draw(pil)
            for text, x, y, size, color, bold, anchor, outline in self._texts:
                draw.text((x, y), text, font=_font(size, bold), fill=tuple(color),
                          anchor=anchor, stroke_width=2 if outline else 0,
                          stroke_fill=(0, 0, 0))
            self.img[:] = np.asarray(pil)
        self._texts.clear()
        return self.img

    def _cv_text(self, text, x, y, size, color, bold, anchor, outline):
        text = _ascii(text)
        scale, thick = size / 30, _cv_thick(size, bold)
        (w, h), _ = cv2.getTextSize(text, FONT, scale, thick)
        x -= {"l": 0, "m": w // 2, "r": w}[anchor[0]]
        y += h if anchor[1] == "a" else h // 2
        if outline:
            cv2.putText(self.img, text, (x, y), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(self.img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _ascii(text: str) -> str:
    return text.replace("…", "...").replace("·", "-").encode("ascii", "replace").decode()


def _cv_thick(size: int, bold: bool) -> int:
    return 2 if bold or size >= 22 else 1


# ── drawing primitives ───────────────────────────────────────────────────────

def _mix(a, b, t: float):
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


def rounded_rect(img, rect, radius: int, color, thickness: int = -1, alpha: float = 1.0):
    x0, y0, x1, y1 = (int(v) for v in rect)
    if alpha < 1.0:
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        if cx1 <= cx0 or cy1 <= cy0:
            return
        roi = img[cy0:cy1, cx0:cx1]
        layer = roi.copy()
        rounded_rect(layer, (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0), radius, color, thickness)
        cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)
        return
    x1, y1 = x1 - 1, y1 - 1
    r = int(max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2)))
    corners = ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270),
               (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90))
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy, _ in corners:
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)
        for cx, cy, start in corners:
            cv2.ellipse(img, (cx, cy), (r, r), start, 0, 90, color, thickness, cv2.LINE_AA)


@lru_cache(maxsize=32)
def _round_mask(w: int, h: int, r: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    rounded_rect(mask, (0, 0, w, h), r, 255)
    return (mask.astype(np.float32) / 255.0)[..., None]


def fit_cover(src: np.ndarray, w: int, h: int) -> np.ndarray:
    """Centre-crop to the target aspect, then resize — no letterboxing."""
    sh, sw = src.shape[:2]
    scale = max(w / sw, h / sh)
    cw, ch = int(round(w / scale)), int(round(h / scale))
    x, y = (sw - cw) // 2, (sh - ch) // 2
    crop = src[max(y, 0):y + ch, max(x, 0):x + cw]
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(crop, (w, h), interpolation=interp)


def paste_rounded(img, src: np.ndarray, rect, radius: int):
    x0, y0, x1, y1 = (int(v) for v in rect)
    w, h = x1 - x0, y1 - y0
    if src.shape[:2] != (h, w):
        src = fit_cover(src, w, h)
    a = _round_mask(w, h, radius)
    dst = img[y0:y1, x0:x1]
    dst[:] = (src.astype(np.float32) * a + dst.astype(np.float32) * (1 - a)).astype(np.uint8)


def _darken_bottom(img: np.ndarray, height: int, strength: float = 0.75):
    height = min(height, img.shape[0])
    if height <= 0:
        return
    ramp = np.linspace(0.0, strength, height, dtype=np.float32) ** 1.4
    band = img[-height:].astype(np.float32)
    img[-height:] = (band * (1 - ramp)[:, None, None]).astype(np.uint8)


def _spinner(img, center, radius: int, color, thickness: int = 3):
    start = (time.monotonic() * 300) % 360
    cv2.ellipse(img, center, (radius, radius), start, 0, 100, color, thickness, cv2.LINE_AA)


def _initials(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name).strip()
    parts = [p for p in re.split(r"[\s_\-.]+", name) if p]
    if not parts:
        return "?"
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else "")).upper()


def _contains(rect, pt) -> bool:
    return pt is not None and rect[0] <= pt[0] < rect[2] and rect[1] <= pt[1] < rect[3]


def _button(cv: Canvas, rect, label: str, hover: bool, kind: str = "default",
            enabled: bool = True, size: int = 14, key: str = ""):
    fill = {"default": SURFACE_HI, "primary": ACCENT, "danger": RED,
            "ghost": SURFACE, "toggle_on": _mix(SURFACE_HI, GREEN, 0.35)}[kind]
    if not enabled:
        fill = _mix(fill, SURFACE, 0.65)
    elif hover:
        fill = _mix(fill, (255, 255, 255), 0.12)
    rounded_rect(cv.img, rect, 10, fill)
    if kind in ("default", "ghost"):
        rounded_rect(cv.img, rect, 10, BORDER, 1)
    color = TEXT if enabled else MUTED
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    if key:
        # Keycap on the left, label centred in what remains.
        kw = cv.measure(key, 12, True)[0] + 12
        lw = cv.measure(label, size, True)[0]
        x = cx - (kw + 8 + lw) / 2
        rounded_rect(cv.img, (x, cy - 11, x + kw, cy + 11), 5, (0, 0, 0), alpha=0.3)
        cv.text(key, x + kw / 2, cy, 12, color, True, "mm")
        cv.text(label, x + kw + 8, cy, size, color, True, "lm")
    else:
        cv.text(label, cx, cy, size, color, True, "mm")


# ── call grid ────────────────────────────────────────────────────────────────

HEADER_H = 64
FOOTER_H = 60
GAP = 12
MIN_W = 760


def _wrap_stats(cv: Canvas, subtitle: str, max_w: int, size: int, max_lines: int = 3):
    chunks = [c for c in re.split(r"\s{2,}", subtitle.strip()) if c]
    lines, cur = [], ""
    for chunk in chunks:
        cand = f"{cur}  ·  {chunk}" if cur else chunk
        if cur and cv.measure(cand, size)[0] > max_w:
            lines.append(cur)
            cur = chunk
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return [cv.fit(l, size, max_w) for l in lines[:max_lines]]


def _draw_tile(cv: Canvas, tile: Tile, x: int, y: int, size: int):
    radius = max(8, size // 24)
    stats = _wrap_stats(cv, tile.subtitle, size - 28, 12) if tile.subtitle else []
    bottom_h = 48 + 17 * len(stats)

    if tile.image is not None:
        face = fit_cover(tile.image, size, size)
        _darken_bottom(face, bottom_h + 30)
        paste_rounded(cv.img, face, (x, y, x + size, y + size), radius)
    else:
        rounded_rect(cv.img, (x, y, x + size, y + size), radius, SURFACE)
        cx, cy, r = x + size // 2, y + int(size * 0.40), int(size * 0.16)
        cv2.circle(cv.img, (cx, cy), r, _mix(SURFACE, tile.accent, 0.22), -1, cv2.LINE_AA)
        cv.text(_initials(tile.title), cx, cy, max(14, int(r * 0.8)), tile.accent, True, "mm")
        status = tile.status or "waiting…"
        if status.endswith(("…", "...")):
            _spinner(cv.img, (cx, cy), r + 8, tile.accent)
        cv.text(cv.fit(status, 15, size - 32), cx, cy + r + 30, 15, MUTED, anchor="mm")
        _darken_bottom(cv.img[y:y + size, x:x + size], bottom_h, 0.35)

    if tile.inset is not None:
        iw = max(48, int(size * 0.30))
        ih = int(iw * 4 / 3) if tile.inset.shape[0] > tile.inset.shape[1] * 1.1 else iw
        ix0, iy0 = x + size - 10 - iw, y + 10
        rounded_rect(cv.img, (ix0 - 2, iy0 - 2, ix0 + iw + 2, iy0 + ih + 2), radius // 2 + 2,
                     (235, 235, 235))
        paste_rounded(cv.img, fit_cover(tile.inset, iw, ih), (ix0, iy0, ix0 + iw, iy0 + ih),
                      radius // 2)
        if tile.inset_label:
            label = cv.fit(tile.inset_label, 11, iw - 8, True)
            cv.text(label, ix0 + iw / 2, iy0 + ih - 10, 11, TEXT, True, "mm", outline=True)

    if tile.badge:
        label = cv.fit(tile.badge, 12, size - 40 - (int(size * 0.30) if tile.inset is not None else 0), True)
        bw = cv.measure(label, 12, True)[0] + 20
        rounded_rect(cv.img, (x + 10, y + 10, x + 10 + bw, y + 34), 12,
                     _mix((0, 0, 0), tile.badge_color, 0.75), alpha=0.9)
        cv.text(label, x + 20, y + 22, 12, TEXT, True, "lm")

    rounded_rect(cv.img, (x, y, x + size, y + size), radius, BORDER, 1)

    # Name pill, then the stats lines underneath it.
    name_y = y + size - 14 - 17 * len(stats) - 14
    name = cv.fit(tile.title, 15, size - 60, True)
    pill_w = cv.measure(name, 15, True)[0] + 36
    rounded_rect(cv.img, (x + 12, name_y - 14, x + 12 + pill_w, name_y + 14), 14,
                 (0, 0, 0), alpha=0.55)
    cv2.circle(cv.img, (x + 26, name_y), 5, tile.accent, -1, cv2.LINE_AA)
    cv.text(name, x + 38, name_y, 15, TEXT, True, "lm")
    for i, line in enumerate(stats):
        cv.text(line, x + 14, name_y + 26 + 17 * i, 12, AMBER, anchor="lm", outline=True)


def render_tile(tile: Tile, size: int) -> np.ndarray:
    cv = Canvas(np.full((size, size, 3), BG, np.uint8))
    _draw_tile(cv, tile, 0, 0, size)
    return cv.flush()


@dataclass
class Control:
    key: str
    label: str
    action: str
    kind: str = "default"


SELF_VIEW_CONTROL = Control("V", "Self view", "selfview")

CALL_CONTROLS = [
    Control("R", "Recalibrate", "recalibrate"),
    Control("S", "Snapshot", "snapshot"),
    Control("Q", "Leave", "leave", "danger"),
]


def _compose(tiles: list[Tile], tile_size: int, header: str, footer: str,
             room: str = "", controls=None, mouse=None, toast: str = "",
             toast_alpha: float = 1.0):
    n = max(1, len(tiles))
    cols = min(n, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    grid_w = cols * tile_size + (cols - 1) * GAP
    width = max(MIN_W, grid_w + 2 * 24)
    height = HEADER_H + rows * tile_size + (rows - 1) * GAP + GAP + FOOTER_H
    cv = Canvas(np.full((height, width, 3), BG, np.uint8))

    # Header: title + room chip on the left, head count on the right, stats below.
    cv.text("Neural Conference", 24, 22, 18, TEXT, True, "lm")
    x = 24 + cv.measure("Neural Conference", 18, True)[0] + 14
    if room:
        label = cv.fit(f"# {room}", 13, 220, True)
        w = cv.measure(label, 13, True)[0] + 20
        rounded_rect(cv.img, (x, 10, x + w, 34), 12, _mix(BG, ACCENT, 0.3))
        cv.text(label, x + 10, 22, 13, TEXT, True, "lm")
    count = f"{len(tiles)} in call"
    cw = cv.measure(count, 13, True)[0]
    cv2.circle(cv.img, (width - 24 - cw - 12, 22), 4, GREEN, -1, cv2.LINE_AA)
    cv.text(count, width - 24, 22, 13, MUTED, True, "rm")
    if header:
        cv.text(cv.fit(header, 12, width - 48), 24, 47, 12, MUTED, anchor="lm")

    x_off = (width - grid_w) // 2
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        _draw_tile(cv, tile, x_off + c * (tile_size + GAP),
                   HEADER_H + r * (tile_size + GAP), tile_size)

    # Footer: clickable controls (with their key), or the legacy hint line.
    hits = []
    fy = height - FOOTER_H // 2
    if controls:
        widths = [cv.measure(c.label, 14, True)[0] + cv.measure(c.key, 12, True)[0] + 52
                  for c in controls]
        x = (width - sum(widths) - 10 * (len(widths) - 1)) // 2
        for ctl, w in zip(controls, widths):
            rect = (x, fy - 19, x + w, fy + 19)
            _button(cv, rect, ctl.label, _contains(rect, mouse), ctl.kind, key=ctl.key)
            hits.append((rect, ctl.action))
            x += w + 10
    elif footer:
        cv.text(cv.fit(footer, 13, width - 48), width // 2, fy, 13, MUTED, anchor="mm")

    if toast and toast_alpha > 0:
        cv.flush()                                  # the toast must cover tile text
        tw = cv.measure(toast, 14, True)[0] + 36
        ty = HEADER_H + rows * tile_size + (rows - 1) * GAP - 36
        rect = ((width - tw) // 2, ty - 18, (width + tw) // 2, ty + 18)
        rounded_rect(cv.img, rect, 18, (12, 12, 12), alpha=0.85 * toast_alpha)
        cv.text(toast, width // 2, ty, 14, _mix(BG, TEXT, toast_alpha), True, "mm")
    return cv.flush(), hits


def compose(tiles: list[Tile], tile_size: int = 384,
            header: str = "", footer: str = "", room: str = "") -> np.ndarray:
    return _compose(tiles, tile_size, header, footer, room)[0]


class CallView:
    """The call window: draws the grid and turns keys and clicks into actions.

    `show()` returns one of "leave", "recalibrate", "snapshot" or None.
    """

    def __init__(self, window: str = "Neural Conference", tile_size: int = 384,
                 controls=None):
        self.window = window
        self.tile_size = tile_size
        self.controls = controls if controls is not None else CALL_CONTROLS
        self.frame: Optional[np.ndarray] = None
        self._mouse = None
        self._clicks = []
        self._hits = []
        self._toast = ("", 0.0, 0.0)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self._on_mouse)
        self._sized = False

    def _on_mouse(self, event, x, y, flags, param):
        self._mouse = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self._clicks.append((x, y))

    def toast(self, message: str, seconds: float = 2.5):
        now = time.monotonic()
        self._toast = (message, now, now + seconds)

    def show(self, tiles: list[Tile], room: str = "", header: str = "") -> Optional[str]:
        msg, t0, t1 = self._toast
        now = time.monotonic()
        alpha = 0.0 if now >= t1 else min(1.0, (t1 - now) / 0.4, (now - t0) / 0.15)
        self.frame, self._hits = _compose(tiles, self.tile_size, header, "", room,
                                          self.controls, self._mouse, msg, alpha)
        if not self._sized:
            cv2.resizeWindow(self.window, self.frame.shape[1], self.frame.shape[0])
            self._sized = True
        cv2.imshow(self.window, self.frame)
        key = cv2.waitKey(15) & 0xFF

        if _window_closed(self.window):
            return "leave"
        while self._clicks:
            pt = self._clicks.pop(0)
            for rect, action in self._hits:
                if _contains(rect, pt):
                    return action
        if key in (ord("q"), ord("Q"), 27):
            return "leave"
        for ctl in self.controls:
            if key in (ord(ctl.key.lower()), ord(ctl.key.upper())):
                return ctl.action
        return None

    def close(self):
        try:
            cv2.destroyWindow(self.window)
        except cv2.error:
            pass


def _window_closed(window: str) -> bool:
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def show_splash(window: str, title: str, detail: str = "", size=(760, 420)):
    """Paint a static 'loading' frame; stays up until the window is reused."""
    w, h = size
    cv = Canvas(np.full((h, w, 3), BG, np.uint8))
    cv2.circle(cv.img, (w // 2, h // 2 - 40), 26, SURFACE_HI, 4, cv2.LINE_AA)
    _spinner(cv.img, (w // 2, h // 2 - 40), 26, ACCENT, 4)
    cv.text(title, w // 2, h // 2 + 22, 20, TEXT, True, "mm")
    if detail:
        cv.text(cv.fit(detail, 13, w - 48), w // 2, h // 2 + 52, 13, MUTED, anchor="mm")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, w, h)
    cv2.imshow(window, cv.flush())
    cv2.waitKey(1)


def bitrate_label(kbps: float) -> str:
    return f"{kbps:.1f} kbps" if kbps < 1000 else f"{kbps / 1000:.2f} Mbps"


def h264_reference_kbps(width: int = 640, height: int = 480, fps: int = 25) -> float:
    """Rough bitrate a conventional codec would need for the same tile.

    Not a measurement — a yardstick for the HUD, at the low end of what
    conferencing encoders are typically configured for at this resolution.
    """
    return 0.06 * width * height * fps / 1000


# ── lobby ────────────────────────────────────────────────────────────────────

ROOM_CHARS = re.compile(r"[A-Za-z0-9_.\-]")
_ADJ = ["amber", "brisk", "calm", "clever", "cosmic", "dusky", "eager", "fuzzy",
        "gentle", "golden", "jolly", "lucky", "mellow", "misty", "nimble", "quiet",
        "rapid", "silver", "sunny", "witty"]
_NOUN = ["badger", "comet", "falcon", "fern", "harbor", "lagoon", "lynx", "maple",
         "meadow", "otter", "panda", "pebble", "quartz", "raven", "river", "sparrow",
         "tiger", "tulip", "walrus", "willow"]


def random_room_code() -> str:
    return f"{random.choice(_ADJ)}-{random.choice(_NOUN)}-{random.randint(10, 99)}"


def rooms_url(signaling_url: str) -> str:
    """ws://host:port/ws -> http://host:port/rooms (wss -> https)."""
    p = urllib.parse.urlparse(signaling_url)
    scheme = "https" if p.scheme in ("wss", "https") else "http"
    path = p.path[:-3] if p.path.endswith("/ws") else p.path.rstrip("/")
    return f"{scheme}://{p.netloc}{path}/rooms"


def load_prefs() -> dict:
    try:
        with open(os.path.join(CONFIG_DIR, "lobby.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_prefs(**prefs):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        merged = {**load_prefs(), **prefs}
        with open(os.path.join(CONFIG_DIR, "lobby.json"), "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
    except OSError:
        pass


def imread_any(path: str) -> Optional[np.ndarray]:
    """cv2.imread that also copes with non-ASCII paths on Windows."""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _ask_image_file(initialdir: Optional[str], owner: str = "") -> Optional[str]:
    """Native "open file" dialog; returns None if the user cancels.

    Tries the platform's own dialog before tkinter, because tkinter is the piece
    most often broken in conda and embedded Pythons (a missing Tcl/Tk DLL), and
    nothing here should depend on it just to pick a file.
    """
    initialdir = initialdir or os.getcwd()
    errors = []
    for ask in (_ask_file_win32, _ask_file_tk, _ask_file_external):
        try:
            return ask(initialdir, owner)
        except NotImplementedError:
            continue
        except Exception as exc:
            errors.append(f"{ask.__name__[10:]}: {exc}")
    raise RuntimeError("; ".join(errors) or "no file dialog available")


def _ask_file_win32(initialdir: str, owner: str) -> Optional[str]:
    if os.name != "nt":
        raise NotImplementedError
    import ctypes
    from ctypes import wintypes

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [("lStructSize", wintypes.DWORD), ("hwndOwner", wintypes.HWND),
                    ("hInstance", wintypes.HINSTANCE), ("lpstrFilter", ctypes.c_void_p),
                    ("lpstrCustomFilter", wintypes.LPWSTR), ("nMaxCustFilter", wintypes.DWORD),
                    ("nFilterIndex", wintypes.DWORD), ("lpstrFile", ctypes.c_void_p),
                    ("nMaxFile", wintypes.DWORD), ("lpstrFileTitle", wintypes.LPWSTR),
                    ("nMaxFileTitle", wintypes.DWORD), ("lpstrInitialDir", wintypes.LPCWSTR),
                    ("lpstrTitle", wintypes.LPCWSTR), ("Flags", wintypes.DWORD),
                    ("nFileOffset", wintypes.WORD), ("nFileExtension", wintypes.WORD),
                    ("lpstrDefExt", wintypes.LPCWSTR), ("lCustData", wintypes.LPARAM),
                    ("lpfnHook", ctypes.c_void_p), ("lpTemplateName", wintypes.LPCWSTR),
                    ("pvReserved", ctypes.c_void_p), ("dwReserved", wintypes.DWORD),
                    ("FlagsEx", wintypes.DWORD)]

    # The filter is a list of NUL-separated pairs ending in a double NUL, which
    # a plain c_wchar_p cannot hold, so it goes in as a raw character array.
    patterns = ";".join(f"*{e}" for e in IMAGE_EXTS)
    filt = f"Images ({patterns})\0{patterns}\0All files\0*.*\0\0"
    filt_buf = (ctypes.c_wchar * len(filt))(*filt)
    file_buf = ctypes.create_unicode_buffer(4096)

    user32, comdlg32 = ctypes.windll.user32, ctypes.windll.comdlg32
    user32.FindWindowW.restype = wintypes.HWND
    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.hwndOwner = user32.FindWindowW(None, owner) if owner else None
    ofn.lpstrFilter = ctypes.addressof(filt_buf)
    ofn.nFilterIndex = 1
    ofn.lpstrFile = ctypes.addressof(file_buf)
    ofn.nMaxFile = len(file_buf)
    ofn.lpstrInitialDir = initialdir
    ofn.lpstrTitle = "Choose a reference image"
    # EXPLORER | FILEMUSTEXIST | PATHMUSTEXIST | NOCHANGEDIR
    ofn.Flags = 0x00080000 | 0x00001000 | 0x00000800 | 0x00000008
    if comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
        return file_buf.value or None
    err = comdlg32.CommDlgExtendedError()
    if err:
        raise OSError(f"GetOpenFileNameW failed (CommDlgExtendedError {err:#x})")
    return None                                     # cancelled


def _ask_file_tk(initialdir: str, owner: str) -> Optional[str]:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        path = filedialog.askopenfilename(
            parent=root, title="Choose a reference image", initialdir=initialdir,
            filetypes=[("Images", " ".join(f"*{e}" for e in IMAGE_EXTS)),
                       ("All files", "*.*")])
    finally:
        root.destroy()
    return path or None


def _ask_file_external(initialdir: str, owner: str) -> Optional[str]:
    import shutil
    import subprocess
    import sys
    if sys.platform == "darwin":
        cmd = ["osascript", "-e", 'POSIX path of (choose file with prompt '
               '"Choose a reference image" of type {"public.image"})']
    elif shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", "--title=Choose a reference image",
               f"--filename={initialdir.rstrip('/')}/",
               "--file-filter=Images | " + " ".join(f"*{e}" for e in IMAGE_EXTS)]
    elif shutil.which("kdialog"):
        cmd = ["kdialog", "--getopenfilename", initialdir,
               " ".join(f"*{e}" for e in IMAGE_EXTS)]
    else:
        raise NotImplementedError
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout.strip() or None               # non-zero exit = cancelled


def _clipboard_text() -> str:
    if os.name == "nt":
        try:
            return _clipboard_win32()
        except Exception:
            pass
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        try:
            return root.clipboard_get()
        finally:
            root.destroy()
    except Exception:
        return ""


def _clipboard_win32() -> str:
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    CF_UNICODETEXT = 13
    if not user32.OpenClipboard(None):
        return ""
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        ptr = kernel32.GlobalLock(handle)
        try:
            return ctypes.wstring_at(ptr) if ptr else ""
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


@dataclass
class LobbyResult:
    name: str
    room: str
    avatar: str
    signaling: str
    created: bool                        # True: "create room", False: "join room"
    preview_self: bool = True            # render your own avatar during the call


@dataclass
class _Field:
    label: str
    value: str
    allowed: Callable[[str], bool]
    maxlen: int
    placeholder: str = ""
    rect: tuple = (0, 0, 0, 0)


class Lobby:
    """Pre-join screen. `run()` blocks and returns a LobbyResult, or None on Esc."""

    W, H = 1040, 740
    WINDOW = "Neural Conference - Lobby"

    def __init__(self, name: str = "", room: str = "", avatar: Optional[str] = None,
                 signaling: str = "ws://127.0.0.1:8765/ws", camera: int = 0,
                 examples_dir: Optional[str] = None, capture_dir: Optional[str] = None,
                 preview_self: bool = True):
        printable = lambda ch: 32 <= ord(ch) < 127
        self.fields = {
            "name": _Field("Your name", name, printable, 32, "e.g. Alice"),
            "create": _Field("Room code", random_room_code(),
                             lambda ch: bool(ROOM_CHARS.match(ch)), 40),
            "join": _Field("Room code", room, lambda ch: bool(ROOM_CHARS.match(ch)), 40,
                           "type a code or pick a live room"),
            "server": _Field("Signaling server", signaling,
                             lambda ch: printable(ch) and ch != " ", 120),
        }
        self.mode = "join" if room else "create"
        self.focus: Optional[str] = None if name else "name"
        self.camera_index = camera
        self.capture_dir = capture_dir or os.path.join(CONFIG_DIR, "captures")

        self.avatar_path: Optional[str] = None
        self.avatar_img: Optional[np.ndarray] = None
        self.message = ("", MUTED)                  # (text, colour) under the preview
        self.preview_self = preview_self
        if avatar and os.path.isfile(avatar):
            self._set_avatar(avatar)

        self.examples = []
        if examples_dir and os.path.isdir(examples_dir):
            for fn in sorted(os.listdir(examples_dir))[:6]:
                if fn.lower().endswith(IMAGE_EXTS):
                    img = imread_any(os.path.join(examples_dir, fn))
                    if img is not None:
                        self.examples.append((os.path.join(examples_dir, fn),
                                              fit_cover(img, 56, 56)))

        self.cam = None                             # media.CameraReader while active
        self.countdown_end: Optional[float] = None
        self.flash_end = 0.0

        self.rooms: dict = {}
        self.rooms_error: Optional[str] = None
        self.rooms_server = ""
        self._fetching = False
        self._next_fetch = 0.0

        self.error = ""
        self._mouse = None
        self._clicks = []
        self._hits = []
        self._result: Optional[LobbyResult] = None
        self._done = False

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def room_field(self) -> _Field:
        return self.fields[self.mode]

    def _set_avatar(self, path: str) -> bool:
        img = imread_any(path) if os.path.isfile(path) else None
        if img is None:
            self.message = (f"Could not read {os.path.basename(path)}", RED)
            return False
        self.avatar_path, self.avatar_img = os.path.abspath(path), img
        self.message = (os.path.basename(path), MUTED)
        self.error = ""
        return True

    def _start_camera(self):
        from .media import CameraReader
        try:
            self.cam = CameraReader(self.camera_index, 640, 480)
            self.message = ("Centre your face, look at the lens, press Capture.", MUTED)
        except Exception as exc:
            self.cam = None
            self.message = (f"Camera {self.camera_index} unavailable: {exc}", RED)
        self.focus = None
        self.countdown_end = None

    def _stop_camera(self):
        if self.cam is not None:
            self.cam.close()
            self.cam = None
        self.countdown_end = None

    def _snap(self):
        frame = self.cam.read() if self.cam is not None else None
        if frame is None:
            self.message = ("No frame from the camera yet — try again.", RED)
            return
        os.makedirs(self.capture_dir, exist_ok=True)
        path = os.path.join(self.capture_dir, f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            self.message = ("Failed to encode the capture.", RED)
            return
        buf.tofile(path)
        self._stop_camera()
        self._set_avatar(path)
        self.message = (f"Captured -> {path}", GREEN)
        self.flash_end = time.monotonic() + 0.25

    def _browse(self):
        self._stop_camera()
        start = os.path.dirname(self.avatar_path) if self.avatar_path else None
        try:
            path = _ask_image_file(start, self.WINDOW)
        except Exception as exc:
            print(f"File dialog failed: {exc}")
            self.message = ("No file dialog available here; pass --avatar <image> instead.", RED)
            return
        if path:
            self._set_avatar(path)

    def _maybe_fetch_rooms(self):
        server = self.fields["server"].value.strip()
        now = time.monotonic()
        if self._fetching or now < self._next_fetch or not server.startswith(("ws://", "wss://")):
            return
        self._fetching = True
        self._next_fetch = now + 4.0

        def work():
            try:
                with urllib.request.urlopen(rooms_url(server), timeout=2.0) as resp:
                    data = json.load(resp)
                self.rooms, self.rooms_error = (data if isinstance(data, dict) else {}), None
            except Exception as exc:
                self.rooms, self.rooms_error = {}, str(getattr(exc, "reason", exc))
            self.rooms_server = server
            self._fetching = False

        threading.Thread(target=work, daemon=True).start()

    def _validate(self) -> str:
        if self.avatar_path is None:
            return "Add a reference image: choose a file, an example, or capture one."
        if not self.fields["name"].value.strip():
            return "Enter your name."
        if not self.room_field.value.strip():
            return "Enter a room code." if self.mode == "join" else "Pick a room code."
        if not self.fields["server"].value.strip().startswith(("ws://", "wss://")):
            return "Signaling server must start with ws:// or wss://"
        return ""

    def _submit(self):
        self.error = self._validate()
        if self.error:
            return
        self._result = LobbyResult(
            name=self.fields["name"].value.strip(),
            room=self.room_field.value.strip(),
            avatar=self.avatar_path,
            signaling=self.fields["server"].value.strip(),
            created=self.mode == "create",
            preview_self=self.preview_self)
        self._done = True

    # ── input ────────────────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
        self._mouse = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self._clicks.append((x, y))

    def _act(self, action: str):
        kind, _, arg = action.partition(":")
        if kind == "focus":
            self.focus = arg
        elif kind == "mode":
            self.mode, self.focus, self.error = arg, None, ""
            self._next_fetch = 0.0
        elif kind == "browse":
            self._browse()
        elif kind == "camera":
            self._start_camera()
        elif kind == "capture":
            self.countdown_end = time.monotonic() + 3.0
        elif kind == "cancel_cam":
            self._stop_camera()
            self.message = (os.path.basename(self.avatar_path) if self.avatar_path else "", MUTED)
        elif kind == "example":
            self._stop_camera()
            self._set_avatar(self.examples[int(arg)][0])
        elif kind == "newcode":
            self.fields["create"].value = random_room_code()
        elif kind == "room":
            self.fields["join"].value = arg
        elif kind == "refresh":
            self._next_fetch = 0.0
        elif kind == "preview":
            self.preview_self = not self.preview_self
        elif kind == "submit":
            self._submit()

    def _on_key(self, key: int):
        if key == 27:                                       # Esc
            if self.cam is not None:
                self._stop_camera()
            else:
                self._done = True
            return
        if key == 13:                                       # Enter
            self._submit()
            return
        if key == 9:                                        # Tab
            order = ["name", self.mode, "server"]
            i = order.index(self.focus) + 1 if self.focus in order else 0
            self.focus = order[i % len(order)]
            return
        if self.focus is None:
            if key == 32 and self.cam is not None and self.countdown_end is None:
                self.countdown_end = time.monotonic() + 3.0
            return
        field = self.fields[self.focus]
        if key == 8:                                        # Backspace
            field.value = field.value[:-1]
        elif key == 22:                                     # Ctrl+V
            for ch in _clipboard_text().strip():
                if len(field.value) < field.maxlen and field.allowed(ch):
                    field.value += ch
        elif 0 < key < 256 and len(field.value) < field.maxlen and field.allowed(chr(key)):
            field.value += chr(key)
        else:
            return
        self.error = ""
        if self.focus == "server":
            self._next_fetch = time.monotonic() + 0.8

    # ── drawing ──────────────────────────────────────────────────────────────

    def _hit(self, rect, action: str) -> bool:
        self._hits.append((rect, action))
        return _contains(rect, self._mouse)

    def _draw_field(self, cv: Canvas, key: str, x0: int, y: int, x1: int):
        f = self.fields[key]
        cv.text(f.label, x0, y, 12, MUTED, True, "lm")
        rect = (x0, y + 14, x1, y + 54)
        f.rect = rect
        hover = self._hit(rect, f"focus:{key}")
        focused = self.focus == key
        rounded_rect(cv.img, rect, 10, SURFACE_HI if focused or hover else _mix(SURFACE, SURFACE_HI, 0.5))
        rounded_rect(cv.img, rect, 10, ACCENT if focused else BORDER, 2 if focused else 1)
        cy = (rect[1] + rect[3]) // 2
        inner = rect[2] - rect[0] - 28
        if f.value:
            shown = f.value
            while shown and cv.measure(shown, 16)[0] > inner:
                shown = shown[1:]                   # keep the caret end visible
            cv.text(shown, x0 + 14, cy, 16, TEXT, anchor="lm")
            caret_x = x0 + 14 + cv.measure(shown, 16)[0] + 2
        else:
            cv.text(f.placeholder, x0 + 14, cy, 15, _mix(MUTED, SURFACE, 0.3), anchor="lm")
            caret_x = x0 + 14
        if focused and int(time.monotonic() * 2) % 2 == 0:
            cv2.line(cv.img, (caret_x, cy - 10), (caret_x, cy + 10), TEXT, 2)

    def _draw_reference(self, cv: Canvas):
        x0, y0, x1, y1 = 24, 96, 500, 680
        rounded_rect(cv.img, (x0, y0, x1, y1), 16, SURFACE)
        cv.text("Reference image", x0 + 22, y0 + 28, 17, TEXT, True, "lm")
        cv.text("This face is what everyone else in the room will see.",
                x0 + 22, y0 + 52, 12, MUTED, anchor="lm")

        ps = 280
        px0, py0 = (x0 + x1 - ps) // 2, y0 + 84
        rect = (px0, py0, px0 + ps, py0 + ps)
        now = time.monotonic()
        frame = self.cam.read() if self.cam is not None else None
        if self.cam is not None and frame is not None:
            view = fit_cover(cv2.flip(frame, 1), ps, ps)    # mirrored, like a selfie
            paste_rounded(cv.img, view, rect, 14)
            cx, cy = px0 + ps // 2, py0 + ps // 2 - 10
            cv2.ellipse(cv.img, (cx, cy), (78, 102), 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
            rounded_rect(cv.img, (px0 + 10, py0 + 10, px0 + 66, py0 + 34), 12, RED)
            cv.text("LIVE", px0 + 38, py0 + 22, 12, TEXT, True, "mm")
            if self.countdown_end is not None:
                left = self.countdown_end - now
                if left <= 0:
                    self._snap()
                else:
                    cv2.circle(cv.img, (cx, cy), 46, (0, 0, 0), -1, cv2.LINE_AA)
                    cv.text(str(int(math.ceil(left))), cx, cy, 48, TEXT, True, "mm")
        elif self.cam is not None:
            rounded_rect(cv.img, rect, 14, BG)
            _spinner(cv.img, (px0 + ps // 2, py0 + ps // 2), 22, ACCENT)
        elif self.avatar_img is not None:
            paste_rounded(cv.img, fit_cover(self.avatar_img, ps, ps), rect, 14)
        else:
            rounded_rect(cv.img, rect, 14, BG)
            c = (px0 + ps // 2, py0 + ps // 2 - 20)
            cv2.circle(cv.img, (c[0], c[1] - 18), 34, BORDER, 2, cv2.LINE_AA)
            cv2.ellipse(cv.img, (c[0], c[1] + 62), (60, 38), 0, 180, 360, BORDER, 2, cv2.LINE_AA)
            cv.text("No reference image yet", c[0], py0 + ps - 40, 14, MUTED, anchor="mm")
        if now < self.flash_end:
            rounded_rect(cv.img, rect, 14, (255, 255, 255),
                         alpha=float((self.flash_end - now) / 0.25))
        rounded_rect(cv.img, rect, 14, BORDER, 1)

        by = py0 + ps + 16
        bw = 196
        bx = (x0 + x1) // 2 - bw - 6
        r1, r2 = (bx, by, bx + bw, by + 40), (bx + bw + 12, by, bx + 2 * bw + 12, by + 40)
        if self.cam is None:
            _button(cv, r1, "Choose file…", self._hit(r1, "browse"))
            _button(cv, r2, "Capture from camera", self._hit(r2, "camera"))
        else:
            counting = self.countdown_end is not None
            _button(cv, r1, "Counting down…" if counting else "Capture",
                    self._hit(r1, "capture") and not counting, "primary",
                    enabled=not counting, key="" if counting else "Space")
            _button(cv, r2, "Cancel", self._hit(r2, "cancel_cam"), key="Esc")

        text, color = self.message
        if text:
            cv.text(cv.fit(text, 12, x1 - x0 - 40), (x0 + x1) // 2, by + 60, 12, color, anchor="mm")

        if self.examples:
            ey = by + 82
            cv.text("Or pick an example", x0 + 22, ey, 12, MUTED, True, "lm")
            n, s, gap = len(self.examples), 56, 10
            ex = (x0 + x1 - (n * s + (n - 1) * gap)) // 2
            for i, (path, thumb) in enumerate(self.examples):
                r = (ex + i * (s + gap), ey + 14, ex + i * (s + gap) + s, ey + 14 + s)
                hover = self._hit(r, f"example:{i}")
                paste_rounded(cv.img, thumb, r, 10)
                selected = self.avatar_path == os.path.abspath(path)
                if selected or hover:
                    rounded_rect(cv.img, (r[0] - 3, r[1] - 3, r[2] + 3, r[3] + 3), 12,
                                 ACCENT if selected else MUTED, 2)

        self._draw_preview_toggle(cv, x0 + 22, y1 - 40, x1 - 22)

    def _draw_preview_toggle(self, cv: Canvas, x0: int, y: int, x1: int):
        label = "Show my generated avatar during the call"
        hint = "uses one extra GPU render per frame · toggle anytime with P"
        rect = (x0, y - 14, x1, y + 30)
        hover = self._hit(rect, "preview")
        box = (x0, y - 10, x0 + 20, y + 10)
        if self.preview_self:
            rounded_rect(cv.img, box, 5, ACCENT)
            cv2.polylines(cv.img, [np.array([(x0 + 5, y), (x0 + 9, y + 5), (x0 + 16, y - 5)])],
                          False, TEXT, 2, cv2.LINE_AA)
        else:
            rounded_rect(cv.img, box, 5, SURFACE_HI if hover else BG)
            rounded_rect(cv.img, box, 5, MUTED if hover else BORDER, 1)
        cv.text(label, x0 + 30, y, 13, TEXT if hover or self.preview_self else MUTED,
                True, "lm")
        cv.text(hint, x0 + 30, y + 19, 11, MUTED, anchor="lm")

    def _draw_room(self, cv: Canvas):
        x0, y0, x1, y1 = 524, 96, 1016, 680
        rounded_rect(cv.img, (x0, y0, x1, y1), 16, SURFACE)
        ix0, ix1 = x0 + 22, x1 - 22

        self._draw_field(cv, "name", ix0, y0 + 30, ix1)

        # Segmented control.
        sy = y0 + 104
        seg = (ix0, sy, ix1, sy + 42)
        rounded_rect(cv.img, seg, 12, BG)
        half = (ix1 - ix0) // 2
        for i, (mode, label) in enumerate((("create", "Create room"), ("join", "Join room"))):
            r = (ix0 + i * half + 4, sy + 4, ix0 + (i + 1) * half - 4, sy + 38)
            hover = self._hit(r, f"mode:{mode}")
            if self.mode == mode:
                rounded_rect(cv.img, r, 9, ACCENT)
            elif hover:
                rounded_rect(cv.img, r, 9, SURFACE_HI)
            cv.text(label, (r[0] + r[2]) // 2, sy + 21, 14,
                    TEXT if self.mode == mode or hover else MUTED, True, "mm")

        hint = ("Start a fresh room, then share its code with the people you invite."
                if self.mode == "create" else
                "Type the code you were given, or pick one of the live rooms below.")
        cv.text(hint, ix0, sy + 62, 12, MUTED, anchor="lm")

        fy = sy + 92
        if self.mode == "create":
            self._draw_field(cv, "create", ix0, fy, ix1 - 128)
            r = (ix1 - 116, fy + 14, ix1, fy + 54)
            _button(cv, r, "New code", self._hit(r, "newcode"))
        else:
            self._draw_field(cv, "join", ix0, fy, ix1)
        self._draw_rooms(cv, ix0, fy + 76, ix1)

        self._draw_field(cv, "server", ix0, y1 - 154, ix1)

        problem = self._validate()
        note, color = self.error, RED
        if not note:
            note, color = self._room_note(), AMBER
        if not note and problem:
            note, color = problem, MUTED
        cv.text(cv.fit(note, 12, ix1 - ix0), ix0, y1 - 78, 12, color, anchor="lm")

        r = (ix0, y1 - 62, ix1, y1 - 18)
        label = "Create room & join" if self.mode == "create" else "Join room"
        _button(cv, r, label, self._hit(r, "submit") and not problem, "primary",
                enabled=not problem, size=16, key="Enter")

    def _room_note(self) -> str:
        room = self.room_field.value.strip()
        if not room or self.rooms_error or self.rooms_server != self.fields["server"].value.strip():
            return ""
        people = self.rooms.get(room)
        if self.mode == "create" and people:
            return f"'{room}' already has {len(people)} people — you will join them."
        if self.mode == "join" and not people:
            return f"Nobody is in '{room}' yet — you will be the first one there."
        return ""

    def _draw_rooms(self, cv: Canvas, x0: int, y: int, x1: int):
        live = sorted(self.rooms.items(), key=lambda kv: -len(kv[1]))
        title = "Live rooms" if self.mode == "join" else "Rooms on this server"
        if live:
            title += f" ({len(live)})"
        cv.text(title, x0, y, 12, MUTED, True, "lm")
        r = (x1 - 84, y - 13, x1, y + 13)
        _button(cv, r, "Refresh", self._hit(r, "refresh"), "ghost", size=12)

        y += 22
        if self.rooms_error:
            cv.text(cv.fit(f"Server unreachable: {self.rooms_error}", 12, x1 - x0), x0, y + 16,
                    12, RED, anchor="lm")
            cv.text("Start it with:  python -m webrtc_conference.signaling", x0, y + 36,
                    12, MUTED, anchor="lm")
            return
        if not live:
            cv.text("Checking…" if self._fetching and not self.rooms_server else
                    "No one is online yet.", x0, y + 16, 12, MUTED, anchor="lm")
            return
        rows = live[:3]
        for room, people in rows:
            rect = (x0, y, x1, y + 34)
            hover = self._hit(rect, f"room:{room}") if self.mode == "join" else False
            selected = self.mode == "join" and self.fields["join"].value == room
            rounded_rect(cv.img, rect, 9, _mix(BG, ACCENT, 0.25) if selected else
                         (SURFACE_HI if hover else BG))
            cv2.circle(cv.img, (x0 + 14, y + 17), 4, GREEN, -1, cv2.LINE_AA)
            cv.text(cv.fit(room, 14, 180, True), x0 + 26, y + 17, 14, TEXT, True, "lm")
            names = ", ".join(p.get("name", "?") for p in people)
            cv.text(cv.fit(f"{len(people)} · {names}", 12, x1 - x0 - 230), x1 - 12, y + 17,
                    12, MUTED, anchor="rm")
            y += 38

    def render(self) -> np.ndarray:
        self._hits = []
        cv = Canvas(np.full((self.H, self.W, 3), BG, np.uint8))
        cv.text("Neural Conference", 24, 40, 26, TEXT, True, "lm")
        cv.text("Choose your reference face and a name, then create or join a room.",
                24, 72, 14, MUTED, anchor="lm")
        self._draw_reference(cv)
        self._draw_room(cv)
        cv.text("Tab  next field     Enter  start     Esc  quit", self.W // 2, self.H - 30,
                12, _mix(MUTED, BG, 0.3), anchor="mm")
        return cv.flush()

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self) -> Optional[LobbyResult]:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)
        try:
            while not self._done:
                self._maybe_fetch_rooms()
                cv2.imshow(self.WINDOW, self.render())
                key = cv2.waitKeyEx(30)
                if _window_closed(self.WINDOW):
                    break
                while self._clicks and not self._done:
                    pt = self._clicks.pop(0)
                    hit = next((a for r, a in reversed(self._hits) if _contains(r, pt)), None)
                    if hit is None:
                        self.focus = None
                    else:
                        self._act(hit)
                if key != -1:
                    self._on_key(key & 0xFFFF if key < 0x10000 else -1)
        finally:
            self._stop_camera()
            try:
                cv2.destroyWindow(self.WINDOW)
            except cv2.error:
                pass
        return self._result
