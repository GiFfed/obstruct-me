import os
import sys
import threading
import time
from dataclasses import replace

import cv2
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt, QThread, Signal
from PySide6.QtGui import QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from censor.core import Censor, Settings
from censor.shapes import SHAPES

CAMERA_INDEX = 0
CAPTURE_SIZE = (640, 480)

REGION_OPTIONS = [("eyes", "Eyes"), ("mouth", "Mouth"), ("face", "Face"),
                  ("silhouette", "Silhouette")]
REGION_NAMES = {"eyes": "eyes", "mouth": "mouth", "face": "face", "silhouette": "silhouette"}
REGION_HOTKEYS = ["eyes", "mouth", "face", "silhouette"]
SHAPE_OPTIONS = list(SHAPES.items())
EFFECT_OPTIONS = [("fill", "Fill"), ("blur", "Blur")]

STYLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.qss")

def resource_path(relative):
    """Path to a bundled file: works both from source and inside the PyInstaller exe."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)) + "/..")
    return os.path.join(base, relative)

def load_stylesheet():
    with open(STYLE_PATH, encoding="utf-8") as f:
        return f.read()


def to_qimage(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class CameraWorker(QThread):
    """Reads the camera, applies the filter and sends frames to the UI."""

    frame_ready = Signal(QImage)
    error = Signal(str)
    fps = Signal(float)

    def __init__(self, settings):
        super().__init__()
        self._settings = replace(settings)
        self._lock = threading.Lock()
        self._can_send = threading.Event()
        self._can_send.set()

    def set_settings(self, settings):
        with self._lock:
            self._settings = replace(settings)

    def frame_consumed(self):
        """The UI has drawn the frame and is ready for the next one."""
        self._can_send.set()

    def stop(self):
        self.requestInterruption()
        self.wait(3000)

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error.emit("Camera not found or in use by another application.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])

        censor = Censor()
        last, fps = time.perf_counter(), 0.0
        failures = 0
        try:
            while not self.isInterruptionRequested():
                ok, frame = cap.read()
                if not ok:
                    # the first frames after opening are often empty, give the camera time
                    failures += 1
                    if failures > 50:
                        self.error.emit("Failed to get a frame from the camera.")
                        break
                    self.msleep(20)
                    continue
                failures = 0
                frame = cv2.flip(frame, 1)

                with self._lock:
                    settings = self._settings
                out = censor.process(frame, settings)

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
                last = now
                self.fps.emit(fps)

                if self._can_send.is_set():
                    self._can_send.clear()
                    self.frame_ready.emit(to_qimage(out))
        finally:
            censor.close()
            cap.release()


class Segmented(QWidget):
    """A row of buttons with a sliding indicator under the selected one."""

    changed = Signal(str)

    def __init__(self, options, parent=None):
        super().__init__(parent)
        self._buttons = {}
        self._current = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, label in options:
            btn = QPushButton(label)
            btn.setObjectName("segment")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self._select(k))
            group.addButton(btn)
            row.addWidget(btn)
            self._buttons[key] = btn

        self._indicator = QFrame(self)
        self._indicator.setObjectName("indicator")
        self._anim = QPropertyAnimation(self._indicator, b"geometry", self)
        self._anim.setDuration(240)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def set_current(self, key, animate=True):
        self._current = key
        self._buttons[key].setChecked(True)
        self._move_indicator(animate)

    def _select(self, key):
        if key != self._current:
            self.set_current(key)
            self.changed.emit(key)

    def _move_indicator(self, animate):
        btn = self._buttons[self._current]
        target = QRect(btn.x(), self.height() - 3, btn.width(), 3)
        self._anim.stop()
        if animate and self.isVisible():
            self._anim.setStartValue(self._indicator.geometry())
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._indicator.setGeometry(target)
        self._indicator.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current is not None:
            self._move_indicator(animate=False)


class ChoiceGrid(QWidget):
    """A grid of mutually exclusive buttons."""

    changed = Signal(str)

    def __init__(self, options, columns=2, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self._buttons = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for i, (key, label) in enumerate(options):
            btn = QPushButton(label)
            btn.setMinimumHeight(34)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            group.addButton(btn)
            grid.addWidget(btn, i // columns, i % columns)
            self._buttons[key] = btn

    def set_current(self, key):
        self._buttons[key].setChecked(True)


class LabeledSlider(QWidget):
    """A slider with a label and the current value. Uses an int scale for floats."""

    changed = Signal(float)

    def __init__(self, title, minimum, maximum, value, scale=1, decimals=0, parent=None):
        super().__init__(parent)
        self._scale = scale
        self._decimals = decimals

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._title = QLabel(title)
        self._title.setObjectName("sliderTitle")
        self._value = QLabel("")
        self._value.setObjectName("sliderValue")
        top.addWidget(self._title)
        top.addStretch()
        top.addWidget(self._value)
        layout.addLayout(top)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setObjectName("slider")
        self._slider.setRange(int(minimum * scale), int(maximum * scale))
        self._slider.setValue(int(value * scale))
        self._slider.valueChanged.connect(self._on_change)
        layout.addWidget(self._slider)

        self._update_label(value)

    def _on_change(self, raw):
        value = raw / self._scale
        self._update_label(value)
        self.changed.emit(value)

    def _update_label(self, value):
        self._value.setText(f"{value:.{self._decimals}f}")


def _section(text):
    label = QLabel(text)
    label.setObjectName("section")
    return label


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ObstructMe")
        self.setWindowIcon(QIcon(resource_path(os.path.join("docs", "logo.ico"))))
        self.setMinimumSize(900, 560)
        self.resize(1040, 680)

        self._settings = Settings()
        self._build_ui()
        self._build_shortcuts()

        self._worker = CameraWorker(self._settings)
        self._worker.frame_ready.connect(self._show_frame)
        self._worker.error.connect(self._on_error)
        self._worker.fps.connect(self._on_fps)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(450)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)
        self.setWindowOpacity(0.0)
        self._fade.start()

        self._flash_anim = QPropertyAnimation(self._status_fx, b"opacity", self)
        self._flash_anim.setDuration(260)
        self._flash_anim.setStartValue(0.2)
        self._flash_anim.setEndValue(1.0)
        self._flash_anim.setEasingCurve(QEasingCurve.OutCubic)

        self._worker.start()
        self.flash("Starting camera…")

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 12, 20, 12)
        title = QLabel("OBSTRUCTME")
        title.setObjectName("title")
        subtitle = QLabel("real-time masking")
        subtitle.setObjectName("subtitle")
        self.fps_label = QLabel("")
        self.fps_label.setObjectName("meta")
        hl.addWidget(title)
        hl.addSpacing(12)
        hl.addWidget(subtitle)
        hl.addStretch()
        hl.addWidget(self.fps_label)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.video = QLabel("Starting camera…")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 360)
        body.addWidget(self.video, 1)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(282)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(24, 24, 24, 24)
        sl.setSpacing(10)

        sl.addWidget(_section("REGION"))
        self.region = Segmented(REGION_OPTIONS)
        sl.addWidget(self.region)
        sl.addSpacing(18)

        sl.addWidget(_section("SHAPE"))
        self.shape = ChoiceGrid(SHAPE_OPTIONS, columns=2)
        sl.addWidget(self.shape)
        sl.addSpacing(18)

        sl.addWidget(_section("EFFECT"))
        self.effect = ChoiceGrid(EFFECT_OPTIONS, columns=2)
        sl.addWidget(self.effect)

        sl.addSpacing(18)
        sl.addWidget(_section("SETTINGS"))
        s = self._settings

        self.coverage_slider = LabeledSlider("Coverage", 0.6, 1.6, s.coverage,
                                             scale=100, decimals=2)
        self.coverage_slider.changed.connect(lambda v: self._update("coverage", v))
        sl.addWidget(self.coverage_slider)

        self.smooth_slider = LabeledSlider("Smoothing", 0.05, 1.0, s.smooth,
                                           scale=100, decimals=2)
        self.smooth_slider.changed.connect(lambda v: self._update("smooth", v))
        sl.addWidget(self.smooth_slider)

        self.feather_slider = LabeledSlider("Edge softness", 1, 31, s.feather)
        self.feather_slider.changed.connect(
            lambda v: self._update("feather", int(v)))
        sl.addWidget(self.feather_slider)

        self.blur_slider = LabeledSlider("Blur strength", 15, 151, s.blur_strength)
        self.blur_slider.changed.connect(
            lambda v: self._update("blur_strength", int(v)))
        sl.addWidget(self.blur_slider)

        sl.addStretch()
        hint = QLabel("1–4  region\nQ  quit")
        hint.setObjectName("meta")
        sl.addWidget(hint)

        scroll = QScrollArea()
        scroll.setObjectName("sideScroll")
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(side)
        body.addWidget(scroll)
        root.addLayout(body, 1)

        footer = QFrame()
        footer.setObjectName("footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(20, 8, 20, 8)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self._status_fx = QGraphicsOpacityEffect(self.status)
        self.status.setGraphicsEffect(self._status_fx)
        fl.addWidget(self.status)
        root.addWidget(footer)

        self.region.set_current("eyes", animate=False)
        self.shape.set_current("rect")
        self.effect.set_current("fill")

        self.region.changed.connect(self._on_region)
        self.shape.changed.connect(self._on_shape)
        self.effect.changed.connect(self._on_effect)

    def _build_shortcuts(self):
        for i, key in enumerate(REGION_HOTKEYS, start=1):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.activated.connect(lambda k=key: self._select_region(k))
        QShortcut(QKeySequence("Q"), self).activated.connect(self.close)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    def _select_region(self, key):
        self.region.set_current(key)
        self._on_region(key)

    def _on_region(self, key):
        if key == self._settings.region:
            return
        self._settings.region = key
        self.shape.setEnabled(key != "silhouette")
        self._apply_settings()
        self.flash(f"Region: {REGION_NAMES[key]}")

    def _on_shape(self, key):
        if key == self._settings.shape:
            return
        self._settings.shape = key
        self._apply_settings()
        self.flash(f"Shape: {SHAPES[key].lower()}")

    def _on_effect(self, key):
        blur = key == "blur"
        if blur == self._settings.blur:
            return
        self._settings.blur = blur
        self._apply_settings()
        self.flash("Effect: blur" if blur else "Effect: fill")

    def _update(self, field, value):
        setattr(self._settings, field, value)
        self._apply_settings()

    def _apply_settings(self):
        self._worker.set_settings(self._settings)

    def _show_frame(self, image):
        pixmap = QPixmap.fromImage(image).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video.setPixmap(pixmap)
        self._worker.frame_consumed()

    def _on_fps(self, value):
        self.fps_label.setText(f"{value:.0f} FPS")

    def _on_error(self, message):
        self.flash(message)
        QMessageBox.warning(self, "Error", message)

    def flash(self, text):
        """Update the status line with a smooth fade-in."""
        self.status.setText(text)
        self._flash_anim.stop()
        self._flash_anim.start()

    def closeEvent(self, event):
        self._worker.stop()
        super().closeEvent(event)