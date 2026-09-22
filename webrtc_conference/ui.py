# coding: utf-8
"""OpenCV grid view for the conference, with a per-tile bandwidth readout."""

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (120, 255, 140)
AMBER = (80, 200, 255)
GREY = (150, 150, 150)
RED = (80, 80, 240)


@dataclass
class Tile:
    title: str
    image: Optional[np.ndarray]          # BGR, any size
    subtitle: str = ""
    status: str = ""                     # shown centred when `image` is None
    accent: tuple = GREEN


def _shadowed(img, text, org, scale, color, thickness=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def render_tile(tile: Tile, size: int) -> np.ndarray:
    canvas = np.full((size, size, 3), 24, dtype=np.uint8)
    if tile.image is not None:
        canvas[:] = cv2.resize(tile.image, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        text = tile.status or "waiting…"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
        _shadowed(canvas, text, ((size - tw) // 2, (size + th) // 2), 0.6, GREY)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, size - 46), (size, size), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    _shadowed(canvas, tile.title, (10, size - 26), 0.58, tile.accent)
    if tile.subtitle:
        _shadowed(canvas, tile.subtitle, (10, size - 8), 0.45, AMBER)
    cv2.rectangle(canvas, (0, 0), (size - 1, size - 1), (60, 60, 60), 1)
    return canvas


def compose(tiles: list[Tile], tile_size: int = 384,
            header: str = "", footer: str = "") -> np.ndarray:
    n = max(1, len(tiles))
    cols = min(n, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))

    pad_top = 34 if header else 0
    pad_bottom = 28 if footer else 0
    grid = np.full((rows * tile_size + pad_top + pad_bottom,
                    cols * tile_size, 3), 18, dtype=np.uint8)

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad_top + r * tile_size, c * tile_size
        grid[y:y + tile_size, x:x + tile_size] = render_tile(tile, tile_size)

    if header:
        _shadowed(grid, header, (12, 23), 0.6, (230, 230, 230))
    if footer:
        _shadowed(grid, footer, (12, grid.shape[0] - 9), 0.45, GREY)
    return grid


def bitrate_label(kbps: float) -> str:
    return f"{kbps * 1000 / 1000:.1f} kbps" if kbps < 1000 else f"{kbps / 1000:.2f} Mbps"


def h264_reference_kbps(width: int = 640, height: int = 480, fps: int = 25) -> float:
    """Rough bitrate a conventional codec would need for the same tile.

    Not a measurement — a yardstick for the HUD, at the low end of what
    conferencing encoders are typically configured for at this resolution.
    """
    return 0.06 * width * height * fps / 1000
