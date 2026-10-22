"""Frame processing: MediaPipe detection and shape overlay.

Independent of the UI, so it can be tested and reused.
"""
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from .shapes import shape_polygon

# MediaPipe Face Mesh landmark indices
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
              0, 37, 39, 40, 185]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
             378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]

REGION_LANDMARKS = {"eyes": LEFT_EYE + RIGHT_EYE, "mouth": LIPS_OUTER, "face": FACE_OVAL}
REGION_BASE_PAD = {"eyes": 0.25, "mouth": 0.15, "face": 0.05}

HOLD_FRAMES = 10  # how many frames to keep the mask when no face is detected


@dataclass
class Settings:
    region: str = "eyes"      # eyes | mouth | face | silhouette
    shape: str = "rect"       # key from censor.shapes.SHAPES
    blur: bool = False        # False: black fill, True: blur
    coverage: float = 1.0     # region size multiplier: 0.6 .. 1.6
    smooth: float = 0.4       # 0..1: lower is smoother but lags more
    feather: int = 5          # edge softness, 1 .. 31
    blur_strength: int = 61   # blur kernel size, odd number 15 .. 151


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _padded_bbox(landmarks, indices, w, h, pad):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
                   dtype=np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return np.array([x1 - px, y1 - py, x2 + px, y2 + py], dtype=np.float32)


def _poly_mask(poly, h, w, feather):
    """Shape mask with soft edges, values from 0 to 1."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    if feather > 1:
        k = _odd(feather)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def _apply(frame, alpha, blur, blur_strength):
    """Blend the frame with black or a blurred copy of the frame using the mask."""
    if blur:
        k = _odd(blur_strength)
        background = cv2.GaussianBlur(frame, (k, k), 0).astype(np.float32)
    else:
        background = np.zeros(frame.shape, dtype=np.float32)
    a = alpha[..., None]
    out = frame.astype(np.float32) * (1.0 - a) + background * a
    return out.astype(np.uint8)


class Censor:
    """Holds the MediaPipe models and the smoothed state between frames."""

    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1,
        )
        self._region = None
        self._box = None   # smoothed bounding box of the face region
        self._seg = None   # smoothed person mask
        self._missed = 0   # consecutive frames without a face

    def close(self):
        self._face_mesh.close()
        self._segmenter.close()

    def process(self, frame, settings):
        if settings.region != self._region:
            # old state must not carry over into the new region
            self._region = settings.region
            self._box = None
            self._seg = None
            self._missed = 0

        if settings.region == "silhouette":
            return self._silhouette(frame, settings)
        return self._face_region(frame, settings)

    def _face_region(self, frame, settings):
        h, w = frame.shape[:2]
        result = self._face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            pad = REGION_BASE_PAD[settings.region] * settings.coverage
            target = _padded_bbox(landmarks, REGION_LANDMARKS[settings.region], w, h, pad)
            if self._box is None:
                self._box = target
            else:
                self._box = self._box + (target - self._box) * settings.smooth
            self._missed = 0
        else:
            self._missed += 1
            if self._missed > HOLD_FRAMES:
                self._box = None

        if self._box is None:
            return frame

        poly = shape_polygon(settings.shape, *self._box)
        alpha = _poly_mask(poly, h, w, settings.feather)
        return _apply(frame, alpha, settings.blur, settings.blur_strength)

    def _silhouette(self, frame, settings):
        result = self._segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.segmentation_mask is None:
            return frame

        mask = result.segmentation_mask.astype(np.float32)
        if self._seg is None:
            self._seg = mask
        else:
            self._seg = self._seg + (mask - self._seg) * settings.smooth
        k = _odd(max(settings.feather, 1))
        alpha = np.clip(cv2.GaussianBlur(self._seg, (k, k), 0), 0.0, 1.0)
        return _apply(frame, alpha, settings.blur, settings.blur_strength)