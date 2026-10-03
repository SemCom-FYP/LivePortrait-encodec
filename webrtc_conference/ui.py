# coding: utf-8
"""PySide6 front end for the conference: a pre-join lobby and the call window.

LobbyDialog — pick a reference portrait (file, example or webcam capture), a
              display name, and either create a new room or join a live one.
CallWindow  — the in-call grid with per-tile bandwidth readouts, buttons with
              keyboard shortcuts, and transient toasts.

Qt owns the main thread. The client never touches widgets from its workers:
the call window polls a `refresh()` callback on a timer and gets plain `Tile`
records back, so everything Qt-side runs on the GUI thread.
"""

import json
import os
import random
import re
import signal
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

# opencv-python wheels on Linux bundle their own Qt and point Qt's plugin path
# at it on import; PySide6 would then load those plugins and abort.
if "cv2" in os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", ""):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

from PySide6.QtCore import (QEventLoop, QPointF, QRectF, QRegularExpression, QSize, Qt,
                            QTimer, QUrl, Signal)
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QIcon, QImage, QKeySequence,
                           QLinearGradient, QPainter, QPainterPath, QPalette, QPen, QPixmap,
                           QRegularExpressionValidator, QShortcut)
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QDialog, QFileDialog,
                               QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

# ── palette ──────────────────────────────────────────────────────────────────

BG = "#101216"
SURFACE = "#1d2026"
SURFACE_HI = "#282c34"
BORDER = "#39404a"
TEXT = "#ebeef2"
MUTED = "#8c949e"
ACCENT = "#4691fa"
GREEN = "#6ed778"
AMBER = "#ffbe3c"
GREY = "#969696"
RED = "#eb5555"

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".liveportrait_conference")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

STYLESHEET = f"""
QWidget {{ color: {TEXT}; font-size: 13px; }}
QDialog, QWidget#root {{ background: {BG}; }}
QFrame#card {{ background: {SURFACE}; border-radius: 16px; }}
QLabel[role="h1"] {{ font-size: 26px; font-weight: 600; }}
QLabel[role="h2"] {{ font-size: 17px; font-weight: 600; }}
QLabel[role="title"] {{ font-size: 18px; font-weight: 600; }}
QLabel[role="muted"] {{ color: {MUTED}; font-size: 12px; }}
QLabel[role="field"] {{ color: {MUTED}; font-size: 12px; font-weight: 600; }}
QLabel[role="key"] {{ background: rgba(0, 0, 0, 80); border-radius: 5px;
                      padding: 1px 6px; font-size: 11px; font-weight: 700; }}
QLabel#roomChip {{ background: #1d3456; border-radius: 11px; padding: 3px 10px;
                   font-size: 13px; font-weight: 600; }}
QLineEdit {{ background: {SURFACE_HI}; border: 1px solid {BORDER}; border-radius: 10px;
             padding: 8px 12px; font-size: 15px; selection-background-color: {ACCENT}; }}
QLineEdit:focus {{ border: 2px solid {ACCENT}; padding: 7px 11px; }}
QPushButton {{ background: {SURFACE_HI}; border: 1px solid {BORDER}; border-radius: 10px;
               padding: 9px 16px; font-weight: 600; }}
QPushButton:hover {{ background: #333842; }}
QPushButton:disabled {{ color: {MUTED}; background: #22262d; }}
QPushButton[kind="primary"] {{ background: {ACCENT}; border: none; font-size: 15px; }}
QPushButton[kind="primary"]:hover {{ background: #5aa0ff; }}
QPushButton[kind="primary"]:disabled {{ background: #2a3f5e; color: #9aa7b8; }}
QPushButton[kind="danger"] {{ background: {RED}; border: none; }}
QPushButton[kind="danger"]:hover {{ background: #f06a6a; }}
QPushButton[kind="toggle_on"] {{ background: #2f5a3a; border: 1px solid #3f7a4c; }}
QPushButton[kind="toggle_on"]:hover {{ background: #386b45; }}
QPushButton[kind="ghost"] {{ background: transparent; padding: 4px 12px; font-size: 12px; }}
QPushButton[kind="ghost"]:hover {{ background: {SURFACE_HI}; }}
QFrame#segment {{ background: {BG}; border-radius: 12px; }}
QPushButton[kind="segment"] {{ background: transparent; border: none; color: {MUTED};
                               border-radius: 9px; padding: 8px; }}
QPushButton[kind="segment"]:hover {{ background: {SURFACE_HI}; color: {TEXT}; }}
QPushButton[kind="segment"]:checked {{ background: {ACCENT}; color: {TEXT}; }}
QPushButton[kind="room"] {{ background: {BG}; border: none; border-radius: 9px;
                            padding: 0px; text-align: left; }}
QPushButton[kind="room"]:hover {{ background: {SURFACE_HI}; }}
QPushButton[kind="room"]:checked {{ background: #1f3550; }}
QPushButton[kind="room"]:disabled {{ background: {BG}; }}
QToolButton[kind="thumb"] {{ border: 2px solid transparent; border-radius: 12px;
                             padding: 1px; background: transparent; }}
QToolButton[kind="thumb"]:hover {{ border-color: {MUTED}; }}
QToolButton[kind="thumb"]:checked {{ border-color: {ACCENT}; }}
QCheckBox {{ spacing: 10px; font-weight: 600; }}
QMessageBox {{ background: {SURFACE}; }}
"""


@dataclass
class Tile:
    title: str
    image: Optional[np.ndarray]          # BGR, any size
    subtitle: str = ""
    status: str = ""                     # shown centred when `image` is None
    accent: str = GREEN
    inset: Optional[np.ndarray] = None   # picture-in-picture, top-right corner
    inset_label: str = ""
    badge: str = ""                      # chip in the top-left corner
    badge_color: str = ACCENT


@dataclass(frozen=True)
class Control:
    key: str
    label: str
    action: str
    kind: str = "default"                # default | danger | toggle_on


SELF_VIEW_CONTROL = Control("V", "Self view", "selfview")

CALL_CONTROLS = [
    Control("R", "Recalibrate", "recalibrate"),
    Control("S", "Snapshot", "snapshot"),
    Control("Q", "Leave", "leave", "danger"),
]


@dataclass
class CallState:
    """What the call window shows on one refresh."""
    tiles: list
    header: str = ""
    controls: list = field(default_factory=lambda: list(CALL_CONTROLS))


# ── application ──────────────────────────────────────────────────────────────

_interrupted = False


def ensure_app() -> QApplication:
    """Create (once) the QApplication with the dark theme and Ctrl+C handling."""
    app = QApplication.instance()
    if app is not None:
        return app
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Neural Conference")
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.Window, BG), (QPalette.WindowText, TEXT),
                        (QPalette.Base, SURFACE_HI), (QPalette.AlternateBase, SURFACE),
                        (QPalette.Text, TEXT), (QPalette.Button, SURFACE_HI),
                        (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT),
                        (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, SURFACE),
                        (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, MUTED)):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)
    # Every window here runs its own local event loop and ends it explicitly.
    app.setQuitOnLastWindowClosed(False)

    # Qt's event loop never returns to Python on its own, so Ctrl+C would sit
    # unhandled until the next callback. A no-op timer gives the interpreter a
    # chance to run the handler, which closes every window like a click would.
    def on_sigint(*_):
        global _interrupted
        _interrupted = True
        QApplication.closeAllWindows()
    signal.signal(signal.SIGINT, on_sigint)
    wake = QTimer(app)
    wake.timeout.connect(lambda: None)
    wake.start(200)
    app._wake_timer = wake
    return app


def interrupted() -> bool:
    return _interrupted


def exec_window(window: QWidget):
    """Show `window` and block, processing events, until it closes."""
    loop = QEventLoop()
    window.closed.connect(loop.quit)
    window.show()
    loop.exec()


def show_error(title: str, text: str):
    ensure_app()
    QMessageBox.critical(None, title, text)


# ── helpers ──────────────────────────────────────────────────────────────────

def _font(px: float, bold: bool = False) -> QFont:
    f = QFont(QApplication.font())
    f.setPixelSize(int(px))
    f.setWeight(QFont.DemiBold if bold else QFont.Normal)
    return f


def _mix(a, b, t: float) -> QColor:
    a, b = QColor(a), QColor(b)
    return QColor(int(a.red() * (1 - t) + b.red() * t),
                  int(a.green() * (1 - t) + b.green() * t),
                  int(a.blue() * (1 - t) + b.blue() * t))


def _alpha(color, a: int) -> QColor:
    c = QColor(color)
    c.setAlpha(a)
    return c


def to_qimage(bgr: np.ndarray) -> QImage:
    """BGR (or grey) uint8 array -> QImage that owns its own copy of the pixels."""
    arr = np.ascontiguousarray(bgr)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        return QImage(arr.data, w, h, arr.strides[0], QImage.Format_Grayscale8).copy()
    if arr.shape[2] == 4:
        arr = np.ascontiguousarray(arr[..., :3])
    return QImage(arr.data, w, h, arr.strides[0], QImage.Format_BGR888).copy()


def _draw_cover(p: QPainter, target: QRectF, img, radius: float = 0.0):
    """Fill `target` with `img` (QImage or BGR array), centre-cropped, no bars."""
    if isinstance(img, np.ndarray):
        img = to_qimage(img)
    if img.isNull():
        return
    iw, ih = img.width(), img.height()
    scale = max(target.width() / iw, target.height() / ih)
    sw, sh = target.width() / scale, target.height() / scale
    src = QRectF((iw - sw) / 2, (ih - sh) / 2, sw, sh)
    if radius:
        p.save()
        path = QPainterPath()
        path.addRoundedRect(target, radius, radius)
        p.setClipPath(path, Qt.IntersectClip)
    p.drawImage(target, img, src)
    if radius:
        p.restore()


def _text(p: QPainter, rect: QRectF, text: str, font: QFont, color, flags,
          shadow: bool = False):
    p.setFont(font)
    if shadow:
        p.setPen(QColor(0, 0, 0, 210))
        p.drawText(rect.translated(1, 1), flags, text)
    p.setPen(QColor(color))
    p.drawText(rect, flags, text)


def _spinner(p: QPainter, center: QPointF, radius: float, color, width: float = 3.0):
    start = (time.monotonic() * 300) % 360
    pen = QPen(QColor(color), width)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawArc(QRectF(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius),
              int(-start * 16), int(-100 * 16))


def _bottom_shade(p: QPainter, r: QRectF, height: float, alpha: int):
    grad = QLinearGradient(0, r.bottom() - height, 0, r.bottom())
    grad.setColorAt(0.0, QColor(0, 0, 0, 0))
    grad.setColorAt(1.0, QColor(0, 0, 0, alpha))
    p.fillRect(QRectF(r.left(), r.bottom() - height, r.width(), height), grad)


def _initials(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name).strip()
    parts = [x for x in re.split(r"[\s_\-.]+", name) if x]
    if not parts:
        return "?"
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else "")).upper()


def _wrap_stats(subtitle: str, fm: QFontMetrics, max_w: float, max_lines: int = 3):
    chunks = [c for c in re.split(r"\s{2,}", subtitle.strip()) if c]
    lines, cur = [], ""
    for chunk in chunks:
        cand = f"{cur}  ·  {chunk}" if cur else chunk
        if cur and fm.horizontalAdvance(cand) > max_w:
            lines.append(cur)
            cur = chunk
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return [fm.elidedText(line, Qt.ElideRight, int(max_w)) for line in lines[:max_lines]]


def _label(text: str, role: str = "") -> QLabel:
    lab = QLabel(text)
    if role:
        lab.setProperty("role", role)
    return lab


def _button(text: str, kind: str = "default") -> QPushButton:
    b = QPushButton(text)
    b.setProperty("kind", kind)
    b.setCursor(Qt.PointingHandCursor)
    return b


# ── tiles ────────────────────────────────────────────────────────────────────

def paint_tile(p: QPainter, t: Tile, r: QRectF):
    s = r.width()
    radius = max(8.0, s / 24)
    path = QPainterPath()
    path.addRoundedRect(r, radius, radius)
    stats_font, name_font = _font(12), _font(15, True)
    stats = _wrap_stats(t.subtitle, QFontMetrics(stats_font), s - 28) if t.subtitle else []
    bottom_h = 48 + 17 * len(stats)
    accent = QColor(t.accent)

    p.save()
    p.setClipPath(path)
    if t.image is not None:
        _draw_cover(p, r, t.image)
        _bottom_shade(p, r, bottom_h + 30, 200)
    else:
        p.fillRect(r, QColor(SURFACE))
        center = QPointF(r.center().x(), r.top() + s * 0.40)
        rad = s * 0.16
        p.setPen(Qt.NoPen)
        p.setBrush(_mix(SURFACE, accent, 0.22))
        p.drawEllipse(center, rad, rad)
        _text(p, QRectF(center.x() - rad, center.y() - rad, 2 * rad, 2 * rad),
              _initials(t.title), _font(max(14, rad * 0.8), True), accent, Qt.AlignCenter)
        status = t.status or "waiting…"
        if status.endswith(("…", "...")):
            _spinner(p, center, rad + 8, accent)
        f = _font(15)
        status = QFontMetrics(f).elidedText(status, Qt.ElideRight, int(s - 32))
        _text(p, QRectF(r.left() + 16, center.y() + rad + 16, s - 32, 28), status, f, MUTED,
              Qt.AlignCenter)
        _bottom_shade(p, r, bottom_h, 90)
    p.restore()

    p.setPen(QPen(QColor(BORDER), 1))
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)

    inset_w = 0.0
    if t.inset is not None:
        inset_w = max(48.0, s * 0.30)
        ih_ = inset_w * 4 / 3 if t.inset.shape[0] > t.inset.shape[1] * 1.1 else inset_w
        ir = QRectF(r.right() - 10 - inset_w, r.top() + 10, inset_w, ih_)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#ebebeb"))
        p.drawRoundedRect(ir.adjusted(-2, -2, 2, 2), radius / 2 + 2, radius / 2 + 2)
        _draw_cover(p, ir, t.inset, radius / 2)
        if t.inset_label:
            _text(p, QRectF(ir.left(), ir.bottom() - 22, ir.width(), 18), t.inset_label,
                  _font(11, True), TEXT, Qt.AlignCenter, shadow=True)

    if t.badge:
        f = _font(12, True)
        fm = QFontMetrics(f)
        label = fm.elidedText(t.badge, Qt.ElideRight, int(s - 60 - inset_w))
        br = QRectF(r.left() + 10, r.top() + 10, fm.horizontalAdvance(label) + 20, 24)
        p.setPen(Qt.NoPen)
        p.setBrush(_alpha(_mix("#000000", t.badge_color, 0.75), 230))
        p.drawRoundedRect(br, 12, 12)
        _text(p, br, label, f, TEXT, Qt.AlignCenter)

    # Name pill, then the stats lines underneath it.
    fm = QFontMetrics(name_font)
    name = fm.elidedText(t.title, Qt.ElideRight, int(s - 60))
    name_y = r.bottom() - 14 - 17 * len(stats) - 14
    pill = QRectF(r.left() + 12, name_y - 14, fm.horizontalAdvance(name) + 36, 28)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(0, 0, 0, 140))
    p.drawRoundedRect(pill, 14, 14)
    p.setBrush(accent)
    p.drawEllipse(QPointF(pill.left() + 14, name_y), 5, 5)
    _text(p, QRectF(pill.left() + 26, pill.top(), pill.width() - 26, 28), name, name_font,
          TEXT, Qt.AlignLeft | Qt.AlignVCenter)
    for i, line in enumerate(stats):
        _text(p, QRectF(r.left() + 14, name_y + 17 + 17 * i, s - 28, 17), line, stats_font,
              AMBER, Qt.AlignLeft | Qt.AlignVCenter, shadow=True)


class TileGrid(QWidget):
    """Lays the tiles out as large as the widget allows, plus the toast."""

    MARGIN, GAP = 16, 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tiles: list = []
        self._toast = ("", 0.0, 0.0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)

    def set_tiles(self, tiles: list):
        self.tiles = tiles
        self.update()

    def toast(self, message: str, seconds: float = 2.5):
        now = time.monotonic()
        self._toast = (message, now, now + seconds)
        self.update()

    @staticmethod
    def layout_for(n: int) -> tuple[int, int]:
        n = max(1, n)
        cols = int(np.ceil(np.sqrt(n)))
        return cols, int(np.ceil(n / cols))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform
                         | QPainter.TextAntialiasing)
        cols, rows = self.layout_for(len(self.tiles))
        m, g = self.MARGIN, self.GAP
        size = min((self.width() - 2 * m - (cols - 1) * g) / cols,
                   (self.height() - 2 * m - (rows - 1) * g) / rows)
        size = max(size, 64.0)
        gw, gh = cols * size + (cols - 1) * g, rows * size + (rows - 1) * g
        x0, y0 = (self.width() - gw) / 2, (self.height() - gh) / 2
        for i, tile in enumerate(self.tiles):
            r, c = divmod(i, cols)
            paint_tile(p, tile, QRectF(x0 + c * (size + g), y0 + r * (size + g), size, size))

        msg, t0, t1 = self._toast
        now = time.monotonic()
        if msg and now < t1:
            a = max(0.0, min(1.0, (t1 - now) / 0.4, (now - t0) / 0.15))
            f = _font(14, True)
            w = QFontMetrics(f).horizontalAdvance(msg) + 36
            tr = QRectF((self.width() - w) / 2, y0 + gh - 54, w, 36)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(12, 12, 12, int(215 * a)))
            p.drawRoundedRect(tr, 18, 18)
            _text(p, tr, msg, f, _alpha(TEXT, int(255 * a)), Qt.AlignCenter)
        p.end()


# ── call window ──────────────────────────────────────────────────────────────

class _ControlButton(QPushButton):
    """A button with a keycap on the left, e.g. [R] Recalibrate."""

    def __init__(self, ctl: Control):
        super().__init__()
        self.setProperty("kind", ctl.kind)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)             # keys belong to the window
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 16, 0)
        lay.setSpacing(8)
        for text, role in ((ctl.key, "key"), (ctl.label, "")):
            lab = _label(text, role)
            lab.setAttribute(Qt.WA_TransparentForMouseEvents)
            if not role:
                lab.setStyleSheet("font-weight: 600; background: transparent;")
            lay.addWidget(lab)
        self.setFixedHeight(40)
        self.setMinimumWidth(lay.sizeHint().width())


class CallWindow(QWidget):
    """The call window.

    `refresh()` is called on the GUI thread about 30 times a second and returns
    a CallState (or None to keep the current one). Button presses, shortcut keys
    and closing the window arrive at `on_action(name)`; closing sends "leave".
    """

    closed = Signal()

    def __init__(self, title: str, room: str, tile_size: int,
                 refresh: Callable[[], Optional[CallState]],
                 on_action: Callable[[str], None], fps: int = 30):
        super().__init__()
        ensure_app()
        self.setObjectName("root")
        self.setWindowTitle(f"{title} — {room}")
        self._refresh, self._on_action = refresh, on_action
        self._controls: tuple = ()
        self._count = -1
        self._tile_size = tile_size
        self._closed = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QWidget()
        hl = QVBoxLayout(header)
        hl.setContentsMargins(24, 14, 24, 0)
        hl.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(_label(title, "title"))
        chip = QLabel(f"# {room}")
        chip.setObjectName("roomChip")
        top.addWidget(chip)
        top.addStretch(1)
        self._count_label = _label("", "muted")
        self._count_label.setStyleSheet("font-weight: 600; font-size: 13px;")
        top.addWidget(self._count_label)
        hl.addLayout(top)
        self._stats_label = _label("", "muted")
        hl.addWidget(self._stats_label)
        root.addWidget(header)

        self.grid = TileGrid()
        root.addWidget(self.grid, 1)

        footer = QWidget()
        self._footer = QHBoxLayout(footer)
        self._footer.setContentsMargins(16, 4, 16, 16)
        self._footer.setSpacing(10)
        root.addWidget(footer)

        self.resize(*self._size_for(1))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / fps))

    def _size_for(self, n: int) -> tuple[int, int]:
        cols, rows = TileGrid.layout_for(n)
        g, m = TileGrid.GAP, TileGrid.MARGIN
        w = max(760, cols * self._tile_size + (cols - 1) * g + 2 * m)
        h = rows * self._tile_size + (rows - 1) * g + 2 * m + 64 + 60
        screen = self.screen().availableGeometry() if self.screen() else None
        if screen is not None:
            w, h = min(w, int(screen.width() * 0.95)), min(h, int(screen.height() * 0.9))
        return w, h

    def _tick(self):
        state = self._refresh()
        if state is None or self._closed:
            return
        self.grid.set_tiles(state.tiles)
        self._stats_label.setText(state.header)
        n = len(state.tiles)
        if n != self._count:
            self._count_label.setText(f"<span style='color:{GREEN}'>●</span> {n} in call")
            if self._count >= 0 and not (self.isMaximized() or self.isFullScreen()):
                w, h = self._size_for(n)
                self.resize(max(w, self.width()), max(h, self.height()))
            self._count = n
        controls = tuple(state.controls)
        if controls != self._controls:
            self._set_controls(controls)

    def _set_controls(self, controls: tuple):
        self._controls = controls
        while self._footer.count():
            item = self._footer.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._footer.addStretch(1)
        for ctl in controls:
            b = _ControlButton(ctl)
            b.clicked.connect(lambda _=False, a=ctl.action: self._on_action(a))
            self._footer.addWidget(b)
        self._footer.addStretch(1)

    def toast(self, message: str, seconds: float = 2.5):
        self.grid.toast(message, seconds)

    def snapshot(self, path: str) -> bool:
        return self.grab().save(path)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        key = event.text().upper()
        for ctl in self._controls:
            if key and key == ctl.key.upper():
                self._on_action(ctl.action)
                return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self._timer.stop()
            self._on_action("leave")
            self.closed.emit()
        event.accept()


# ── loading ──────────────────────────────────────────────────────────────────

class _Spinner(QWidget):
    def __init__(self, size: int = 64):
        super().__init__()
        self.setFixedSize(size, size)
        t = QTimer(self)
        t.timeout.connect(self.update)
        t.start(16)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c, r = QPointF(self.width() / 2, self.height() / 2), self.width() / 2 - 6
        p.setPen(QPen(QColor(SURFACE_HI), 4))
        p.drawEllipse(c, r, r)
        _spinner(p, c, r, ACCENT, 4)
        p.end()


class LoadingWindow(QWidget):
    closed = Signal()

    def __init__(self, window_title: str, title: str, detail: str = ""):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle(window_title)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(14)
        lay.addStretch(1)
        lay.addWidget(_Spinner(), 0, Qt.AlignHCenter)
        t = _label(title)
        t.setStyleSheet("font-size: 20px; font-weight: 600;")
        lay.addWidget(t, 0, Qt.AlignHCenter)
        if detail:
            lay.addWidget(_label(detail, "muted"), 0, Qt.AlignHCenter)
        lay.addStretch(1)
        self.resize(560, 320)

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()


def run_with_loading(window_title: str, title: str, detail: str, fn: Callable):
    """Run `fn()` on a worker thread behind a loading window; return its result.

    Keeps the GUI responsive through multi-second model loads. Re-raises the
    worker's exception, or KeyboardInterrupt if the user closed the window or
    pressed Ctrl+C first (the worker is a daemon thread and is abandoned).
    """
    ensure_app()
    win = LoadingWindow(window_title, title, detail)
    box: dict = {}
    abandoned = []

    def work():
        try:
            box["result"] = fn()
        except BaseException as exc:                # re-raised on the GUI thread
            box["error"] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: loop.quit() if not worker.is_alive() else None)
    win.closed.connect(lambda: (abandoned.append(True), loop.quit()))
    poll.start(50)
    win.show()
    loop.exec()
    poll.stop()
    if abandoned and worker.is_alive():
        win.deleteLater()
        raise KeyboardInterrupt
    win.hide()
    win.deleteLater()
    if "error" in box:
        raise box["error"]
    return box["result"]


# ── lobby helpers ────────────────────────────────────────────────────────────

ROOM_RE = r"[A-Za-z0-9_.\-]{0,40}"
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


def bitrate_label(kbps: float) -> str:
    return f"{kbps:.1f} kbps" if kbps < 1000 else f"{kbps / 1000:.2f} Mbps"


def h264_reference_kbps(width: int = 640, height: int = 480, fps: int = 25) -> float:
    """Rough bitrate a conventional codec would need for the same tile.

    Not a measurement — a yardstick for the HUD, at the low end of what
    conferencing encoders are typically configured for at this resolution.
    """
    return 0.06 * width * height * fps / 1000


@dataclass
class LobbyResult:
    name: str
    room: str
    avatar: str
    signaling: str
    created: bool                        # True: "create room", False: "join room"
    preview_self: bool = True            # render your own avatar during the call


# ── lobby ────────────────────────────────────────────────────────────────────

class _Preview(QWidget):
    """The big square: chosen portrait, live camera with countdown, or empty."""

    def __init__(self, size: int = 280):
        super().__init__()
        self.setFixedSize(size, size)
        self.image: Optional[QImage] = None
        self.live: Optional[np.ndarray] = None
        self.camera_on = False
        self.countdown: Optional[int] = None
        self.flash_until = 0.0

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        path = QPainterPath()
        path.addRoundedRect(r, 14, 14)
        p.fillPath(path, QColor(BG))
        if self.camera_on and self.live is not None:
            _draw_cover(p, r, self.live, 14)
            c = r.center() - QPointF(0, 10)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 78, 102)
            live = QRectF(10, 10, 56, 24)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(RED))
            p.drawRoundedRect(live, 12, 12)
            _text(p, live, "LIVE", _font(12, True), TEXT, Qt.AlignCenter)
            if self.countdown is not None:
                p.setBrush(QColor(0, 0, 0, 170))
                p.drawEllipse(c, 46, 46)
                _text(p, QRectF(c.x() - 46, c.y() - 46, 92, 92), str(self.countdown),
                      _font(48, True), TEXT, Qt.AlignCenter)
        elif self.camera_on:
            _spinner(p, r.center(), 22, ACCENT)
        elif self.image is not None:
            _draw_cover(p, r, self.image, 14)
        else:
            c = r.center() - QPointF(0, 20)
            p.setPen(QPen(QColor(BORDER), 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c - QPointF(0, 18), 34, 34)
            p.drawArc(QRectF(c.x() - 60, c.y() + 24, 120, 76), 0, 180 * 16)
            _text(p, QRectF(r.left(), r.bottom() - 60, r.width(), 30),
                  "No reference image yet", _font(14), MUTED, Qt.AlignCenter)
        now = time.monotonic()
        if now < self.flash_until:
            p.fillPath(path, QColor(255, 255, 255, int(255 * (self.flash_until - now) / 0.25)))
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        p.end()


class _RoomRow(QPushButton):
    """One live room: green dot, name, and who is in it, right-aligned."""

    def __init__(self, room: str, people: list, clickable: bool):
        super().__init__()
        self.room = room
        self.setProperty("kind", "room")
        self.setFixedHeight(36)
        self.setCheckable(clickable)
        self.setEnabled(clickable)
        self.setFocusPolicy(Qt.NoFocus)
        if clickable:
            self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(8)
        names = ", ".join(str(x.get("name", "?")) for x in people)
        parts = ((f"<span style='color:{GREEN}'>●</span>", "font-size: 10px;"),
                 (room, "font-size: 14px; font-weight: 600;"),
                 (None, ""),
                 (f"{len(people)} · {names}", f"color: {MUTED}; font-size: 12px;"))
        for text, style in parts:
            if text is None:
                lay.addStretch(1)
                continue
            lab = QLabel(text)
            lab.setStyleSheet(style + " background: transparent;")
            lab.setAttribute(Qt.WA_TransparentForMouseEvents)
            lay.addWidget(lab)


class LobbyDialog(QDialog):
    """Pre-join screen. `result_value` holds a LobbyResult once accepted."""

    def __init__(self, name: str = "", room: str = "", avatar: Optional[str] = None,
                 signaling: str = "ws://127.0.0.1:8765/ws", camera: int = 0,
                 examples_dir: Optional[str] = None, capture_dir: Optional[str] = None,
                 preview_self: bool = True):
        super().__init__()
        self.setWindowTitle("Neural Conference — Lobby")
        self.camera_index = camera
        self.capture_dir = capture_dir or os.path.join(CONFIG_DIR, "captures")
        self.avatar_path: Optional[str] = None
        self.result_value: Optional[LobbyResult] = None
        self.mode = "join" if room else "create"
        self._codes = {"create": random_room_code(), "join": room}
        self.cam = None                             # media.CameraReader while active
        self.countdown_end: Optional[float] = None
        self.rooms: dict = {}
        self.rooms_error: Optional[str] = None
        self.rooms_server = ""
        self._reply: Optional[QNetworkReply] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 14)
        outer.setSpacing(6)
        outer.addWidget(_label("Neural Conference", "h1"))
        sub = _label("Choose your reference face and a name, then create or join a room.")
        sub.setStyleSheet(f"color: {MUTED}; font-size: 14px;")
        outer.addWidget(sub)
        outer.addSpacing(12)
        cards = QHBoxLayout()
        cards.setSpacing(24)
        cards.addWidget(self._build_reference_card(examples_dir, preview_self), 1)
        cards.addWidget(self._build_room_card(name, signaling), 1)
        outer.addLayout(cards, 1)
        # Only once the button sits inside the dialog does setDefault register it
        # as the button Enter presses.
        self.submit_btn.setDefault(True)
        outer.addWidget(_label("Tab  next field     Enter  start     Esc  quit", "muted"),
                        0, Qt.AlignHCenter)

        self._net = QNetworkAccessManager(self)
        self._fetch_timer = QTimer(self)
        self._fetch_timer.timeout.connect(self._fetch_rooms)
        self._fetch_timer.start(4000)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._fetch_rooms)
        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self._camera_tick)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._space)

        self._set_mode(self.mode)
        if avatar and os.path.isfile(avatar):
            self._set_avatar(avatar)
        (self.room_edit if name else self.name_edit).setFocus()
        QTimer.singleShot(0, self._fetch_rooms)
        self.resize(1040, 720)

    # ── construction ────────────────────────────────────────────────────────

    def _card(self) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(22, 20, 22, 20)
        lay.setSpacing(6)
        return card, lay

    def _build_reference_card(self, examples_dir, preview_self) -> QFrame:
        card, lay = self._card()
        lay.addWidget(_label("Reference image", "h2"))
        lay.addWidget(_label("This face is what everyone else in the room will see.", "muted"))
        lay.addSpacing(10)
        self.preview = _Preview()
        lay.addWidget(self.preview, 0, Qt.AlignHCenter)
        lay.addSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(12)
        self.browse_btn = _button("Choose file…")
        self.camera_btn = _button("Capture from camera")
        self.browse_btn.clicked.connect(self._on_left_button)
        self.camera_btn.clicked.connect(self._on_right_button)
        row.addWidget(self.browse_btn)
        row.addWidget(self.camera_btn)
        lay.addLayout(row)

        self.message = _label("", "muted")
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setMinimumHeight(22)
        lay.addWidget(self.message)

        self.example_buttons = []
        if examples_dir and os.path.isdir(examples_dir):
            files = [f for f in sorted(os.listdir(examples_dir))
                     if f.lower().endswith(IMAGE_EXTS)][:6]
            if files:
                lay.addWidget(_label("Or pick an example", "field"))
                ex_row = QHBoxLayout()
                ex_row.setSpacing(8)
                ex_row.addStretch(1)
                for fn in files:
                    path = os.path.abspath(os.path.join(examples_dir, fn))
                    pix = QPixmap(path)
                    if pix.isNull():
                        continue
                    b = QToolButton()
                    b.setProperty("kind", "thumb")
                    b.setCheckable(True)
                    b.setCursor(Qt.PointingHandCursor)
                    b.setToolTip(fn)
                    b.setIcon(QIcon(_rounded_thumb(pix, 56)))
                    b.setIconSize(QSize(56, 56))
                    b.clicked.connect(lambda _=False, p=path: self._pick_example(p))
                    self.example_buttons.append((path, b))
                    ex_row.addWidget(b)
                ex_row.addStretch(1)
                lay.addLayout(ex_row)
        lay.addStretch(1)

        self.preview_check = QCheckBox("Show my generated avatar during the call")
        self.preview_check.setChecked(preview_self)
        self.preview_check.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.preview_check)
        note = _label("uses one extra GPU render per frame · toggle anytime with P", "muted")
        note.setContentsMargins(30, 0, 0, 0)
        lay.addWidget(note)
        return card

    def _build_room_card(self, name: str, signaling: str) -> QFrame:
        card, lay = self._card()
        lay.addWidget(_label("Your name", "field"))
        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("e.g. Alice")
        self.name_edit.setMaxLength(32)
        lay.addWidget(self.name_edit)
        lay.addSpacing(14)

        seg = QFrame()
        seg.setObjectName("segment")
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(4, 4, 4, 4)
        sl.setSpacing(4)
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for mode, label in (("create", "Create room"), ("join", "Join room")):
            b = _button(label, "segment")
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _=False, m=mode: self._set_mode(m))
            self.mode_group.addButton(b)
            self.mode_buttons[mode] = b
            sl.addWidget(b)
        lay.addWidget(seg)
        self.mode_hint = _label("", "muted")
        lay.addWidget(self.mode_hint)
        lay.addSpacing(10)

        lay.addWidget(_label("Room code", "field"))
        code_row = QHBoxLayout()
        code_row.setSpacing(12)
        self.room_edit = QLineEdit()
        self.room_edit.setMaxLength(40)
        self.room_edit.setValidator(
            QRegularExpressionValidator(QRegularExpression(ROOM_RE), self))
        self.new_code_btn = _button("New code")
        self.new_code_btn.clicked.connect(
            lambda: self.room_edit.setText(random_room_code()))
        code_row.addWidget(self.room_edit, 1)
        code_row.addWidget(self.new_code_btn)
        lay.addLayout(code_row)
        lay.addSpacing(10)

        head = QHBoxLayout()
        self.rooms_title = _label("", "field")
        head.addWidget(self.rooms_title)
        head.addStretch(1)
        refresh = _button("Refresh", "ghost")
        refresh.clicked.connect(self._fetch_rooms)
        head.addWidget(refresh)
        lay.addLayout(head)
        self.rooms_box = QVBoxLayout()
        self.rooms_box.setSpacing(4)
        self.room_rows: list = []
        lay.addLayout(self.rooms_box)
        self.rooms_status = _label("", "muted")
        self.rooms_status.setWordWrap(True)
        lay.addWidget(self.rooms_status)
        lay.addStretch(1)

        lay.addWidget(_label("Signaling server", "field"))
        self.server_edit = QLineEdit(signaling)
        self.server_edit.setMaxLength(200)
        lay.addWidget(self.server_edit)
        lay.addSpacing(6)
        self.note = _label("", "muted")
        self.note.setWordWrap(True)
        self.note.setMinimumHeight(20)
        lay.addWidget(self.note)
        self.submit_btn = _button("", "primary")
        self.submit_btn.setMinimumHeight(46)
        self.submit_btn.clicked.connect(self._submit)
        lay.addWidget(self.submit_btn)

        self.name_edit.textChanged.connect(self._refresh_state)
        self.room_edit.textChanged.connect(self._room_edited)
        self.server_edit.textChanged.connect(self._server_edited)
        return card

    # ── state ────────────────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        if self.room_edit.text() or mode != self.mode:
            self._codes[self.mode] = self.room_edit.text()
        self.mode = mode
        self.mode_buttons[mode].setChecked(True)
        self.room_edit.blockSignals(True)
        self.room_edit.setText(self._codes.get(mode, ""))
        self.room_edit.blockSignals(False)
        create = mode == "create"
        self.room_edit.setPlaceholderText("" if create else "type a code or pick a live room")
        self.new_code_btn.setVisible(create)
        self.mode_hint.setText(
            "Start a fresh room, then share its code with the people you invite."
            if create else "Type the code you were given, or pick one of the live rooms below.")
        self.submit_btn.setText("Create room && join" if create else "Join room")
        self._populate_rooms()
        self._refresh_state()

    def _room_edited(self, text: str):
        self._codes[self.mode] = text
        self._sync_room_selection()
        self._refresh_state()

    def _server_edited(self, _text: str):
        self._debounce.start(800)
        self._refresh_state()

    def _set_avatar(self, path: str) -> bool:
        img = QImage(path) if os.path.isfile(path) else QImage()
        if img.isNull():
            self._msg(f"Could not read {os.path.basename(path)}", RED)
            return False
        self.avatar_path = os.path.abspath(path)
        self.preview.image = img
        self.preview.update()
        self._msg(os.path.basename(path))
        for p, b in self.example_buttons:
            b.setChecked(p == self.avatar_path)
        self._refresh_state()
        return True

    def _msg(self, text: str, color: str = MUTED):
        self.message.setText(text)
        self.message.setStyleSheet(f"color: {color};")

    def validate(self) -> str:
        if self.avatar_path is None:
            return "Add a reference image: choose a file, an example, or capture one."
        if not self.name_edit.text().strip():
            return "Enter your name."
        if not self.room_edit.text().strip():
            return "Enter a room code." if self.mode == "join" else "Pick a room code."
        if not self.server_edit.text().strip().startswith(("ws://", "wss://")):
            return "Signaling server must start with ws:// or wss://"
        return ""

    def room_note(self) -> str:
        room = self.room_edit.text().strip()
        if not room or self.rooms_error or self.rooms_server != self.server_edit.text().strip():
            return ""
        people = self.rooms.get(room)
        if self.mode == "create" and people:
            return f"'{room}' already has {len(people)} people — you will join them."
        if self.mode == "join" and not people:
            return f"Nobody is in '{room}' yet — you will be the first one there."
        return ""

    def _refresh_state(self, *_):
        problem = self.validate()
        self.submit_btn.setEnabled(not problem)
        note, color = self.room_note(), AMBER
        if not note and problem:
            note, color = problem, MUTED
        self.note.setText(note)
        self.note.setStyleSheet(f"color: {color};")

    def _submit(self):
        if self.validate():
            return
        self.result_value = LobbyResult(
            name=self.name_edit.text().strip(),
            room=self.room_edit.text().strip(),
            avatar=self.avatar_path,
            signaling=self.server_edit.text().strip(),
            created=self.mode == "create",
            preview_self=self.preview_check.isChecked())
        self.accept()

    # ── reference image ─────────────────────────────────────────────────────

    def _on_left_button(self):
        if self.cam is None:
            self._browse()
        elif self.countdown_end is None:
            self.countdown_end = time.monotonic() + 3.0
            self._update_camera_buttons()

    def _on_right_button(self):
        if self.cam is None:
            self._start_camera()
        else:
            self._stop_camera()
            self._msg(os.path.basename(self.avatar_path) if self.avatar_path else "")

    def _browse(self):
        start = os.path.dirname(self.avatar_path) if self.avatar_path else os.getcwd()
        patterns = " ".join(f"*{e}" for e in IMAGE_EXTS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a reference image", start,
            f"Images ({patterns});;All files (*)")
        if path:
            self._set_avatar(path)

    def _pick_example(self, path: str):
        self._stop_camera()
        self._set_avatar(path)

    def _start_camera(self):
        from .media import CameraReader
        try:
            self.cam = CameraReader(self.camera_index, 640, 480)
        except Exception as exc:
            self.cam = None
            self._msg(f"Camera {self.camera_index} unavailable: {exc}", RED)
            return
        self.preview.camera_on = True
        self.preview.live = None
        self._msg("Centre your face, look at the lens, press Capture.")
        self._cam_timer.start(33)
        self._update_camera_buttons()

    def _stop_camera(self):
        self._cam_timer.stop()
        if self.cam is not None:
            self.cam.close()
            self.cam = None
        self.countdown_end = None
        self.preview.camera_on = False
        self.preview.countdown = None
        self.preview.live = None
        self.preview.update()
        self._update_camera_buttons()

    def _update_camera_buttons(self):
        if self.cam is None:
            self.browse_btn.setText("Choose file…")
            self.browse_btn.setProperty("kind", "default")
            self.browse_btn.setEnabled(True)
            self.camera_btn.setText("Capture from camera")
        else:
            counting = self.countdown_end is not None
            self.browse_btn.setText("Counting down…" if counting else "Capture  (Space)")
            self.browse_btn.setProperty("kind", "primary")
            self.browse_btn.setEnabled(not counting)
            self.camera_btn.setText("Cancel  (Esc)")
        self.browse_btn.style().unpolish(self.browse_btn)
        self.browse_btn.style().polish(self.browse_btn)

    def _space(self):
        if self.cam is not None and self.countdown_end is None:
            self._on_left_button()

    def _camera_tick(self):
        frame = self.cam.read() if self.cam is not None else None
        if frame is not None:
            self.preview.live = cv2.flip(frame, 1)      # mirrored, like a selfie
        if self.countdown_end is not None:
            left = self.countdown_end - time.monotonic()
            if left <= 0:
                self._snap()
                return
            self.preview.countdown = int(np.ceil(left))
        self.preview.update()

    def _snap(self):
        frame = self.cam.read() if self.cam is not None else None
        if frame is None:
            self.countdown_end = None
            self.preview.countdown = None
            self._update_camera_buttons()
            self._msg("No frame from the camera yet — try again.", RED)
            return
        os.makedirs(self.capture_dir, exist_ok=True)
        path = os.path.join(self.capture_dir,
                            f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            self._msg("Failed to encode the capture.", RED)
            return
        buf.tofile(path)
        self._stop_camera()
        self._set_avatar(path)
        self._msg(f"Captured → {path}", GREEN)
        self.preview.flash_until = time.monotonic() + 0.25
        for delay in (0, 60, 120, 180, 260):
            QTimer.singleShot(delay, self.preview.update)

    # ── rooms ────────────────────────────────────────────────────────────────

    def _fetch_rooms(self):
        server = self.server_edit.text().strip()
        if not server.startswith(("ws://", "wss://")):
            return
        if self._reply is not None and self._reply.isRunning():
            return
        req = QNetworkRequest(QUrl(rooms_url(server)))
        req.setTransferTimeout(2000)
        reply = self._net.get(req)
        reply.setProperty("server", server)
        reply.finished.connect(lambda r=reply: self._rooms_arrived(r))
        self._reply = reply

    def _rooms_arrived(self, reply: QNetworkReply):
        server = reply.property("server")
        if reply.error() == QNetworkReply.NoError:
            try:
                data = json.loads(bytes(reply.readAll()).decode("utf-8"))
                self.rooms = data if isinstance(data, dict) else {}
                self.rooms_error = None
            except ValueError as exc:
                self.rooms, self.rooms_error = {}, f"bad response ({exc})"
        else:
            self.rooms, self.rooms_error = {}, reply.errorString()
        self.rooms_server = server
        if self._reply is reply:
            self._reply = None
        reply.deleteLater()
        self._populate_rooms()
        self._refresh_state()

    MAX_ROOM_ROWS = 3

    def _populate_rooms(self):
        live = sorted(self.rooms.items(), key=lambda kv: -len(kv[1]))
        title = "Live rooms" if self.mode == "join" else "Rooms on this server"
        self.rooms_title.setText(f"{title} ({len(live)})" if live else title)
        for row in self.room_rows:
            self.rooms_box.removeWidget(row)
            row.deleteLater()
        self.room_rows = []
        for room, people in live[:self.MAX_ROOM_ROWS]:
            row = _RoomRow(room, people, clickable=self.mode == "join")
            row.clicked.connect(lambda _=False, r=room: self.room_edit.setText(r))
            self.rooms_box.addWidget(row)
            self.room_rows.append(row)
        if self.rooms_error:
            self.rooms_status.setText(
                f"<span style='color:{RED}'>Server unreachable: {self.rooms_error}</span><br>"
                "Start it with:&nbsp; <code>python -m webrtc_conference.signaling</code>")
        elif not live:
            self.rooms_status.setText("Checking…" if not self.rooms_server
                                      else "No one is online yet.")
        elif len(live) > self.MAX_ROOM_ROWS:
            self.rooms_status.setText(f"+{len(live) - self.MAX_ROOM_ROWS} more — type the "
                                      "code to join one of those")
        else:
            self.rooms_status.setText("")
        self.rooms_status.setVisible(bool(self.rooms_status.text()))
        self._sync_room_selection()

    def _sync_room_selection(self):
        code = self.room_edit.text()
        for row in self.room_rows:
            row.setChecked(self.mode == "join" and row.room == code)

    # ── closing ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.cam is not None:
            self._stop_camera()                     # Esc leaves the camera first
            self._msg(os.path.basename(self.avatar_path) if self.avatar_path else "")
            return
        super().keyPressEvent(event)

    def done(self, code: int):
        self._stop_camera()
        self._fetch_timer.stop()
        super().done(code)


def _rounded_thumb(pix: QPixmap, size: int) -> QPixmap:
    scaled = pix.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    x, y = (scaled.width() - size) // 2, (scaled.height() - size) // 2
    out = QPixmap(size, size)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, size, size), 10, 10)
    p.setClipPath(path)
    p.drawPixmap(0, 0, scaled, x, y, size, size)
    p.end()
    return out


def run_lobby(**kwargs) -> Optional[LobbyResult]:
    """Show the lobby and block; returns the choices, or None if it was closed."""
    ensure_app()
    dlg = LobbyDialog(**kwargs)
    dlg.exec()
    result = dlg.result_value
    dlg.deleteLater()
    return result
