"""Shape geometry for masks.

Each shape is built around a bounding rectangle (x1, y1, x2, y2) and fully
covers it, so the censored area never shows through at any shape.
"""
import math

import cv2
import numpy as np

SHAPES = {
    "rect": "Rectangle",
    "square": "Square",
    "oval": "Oval",
    "circle": "Circle",
    "triangle": "Triangle",
    "diamond": "Diamond",
}


def _ellipse(cx, cy, a, b):
    pts = cv2.ellipse2Poly(
        (int(cx), int(cy)), (max(int(a), 1), max(int(b), 1)), 0, 0, 360, 5
    )
    return pts.astype(np.float32)


def _rect(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _square(x1, y1, x2, y2):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    return _rect(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def _oval(x1, y1, x2, y2):
    # ellipse passing through the corners of the rectangle: semi-axes scaled by √2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a = (x2 - x1) / 2 * math.sqrt(2)
    b = (y2 - y1) / 2 * math.sqrt(2)
    return _ellipse(cx, cy, a, b)


def _circle(x1, y1, x2, y2):
    # radius is half the diagonal, so the circle covers all corners
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = math.hypot(x2 - x1, y2 - y1) / 2
    return _ellipse(cx, cy, r, r)


def _triangle(x1, y1, x2, y2):
    # apex above the region, base below; base width is chosen
    # so that all corners of the rectangle fall inside the triangle
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    apex = (cx, y1 - h)
    base_y = y2 + h / 2
    half = 1.3 * w
    return np.array([apex, (cx - half, base_y), (cx + half, base_y)], dtype=np.float32)


def _diamond(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = 0.8 * w
    return np.array(
        [(cx, y1 - h), (cx + half_w, cy), (cx, y2 + h), (cx - half_w, cy)],
        dtype=np.float32,
    )


_FUNCS = {
    "rect": _rect,
    "square": _square,
    "oval": _oval,
    "circle": _circle,
    "triangle": _triangle,
    "diamond": _diamond,
}


def shape_polygon(kind, x1, y1, x2, y2):
    """Vertices of the shape of the given kind (float32, shape (N, 2))."""
    return _FUNCS[kind](float(x1), float(y1), float(x2), float(y2))