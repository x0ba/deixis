"""Gaze features from one MediaPipe Face Landmarker result.

Landmark indices are from the 478-point mesh. Each eye is listed as
(image-left corner, image-right corner, upper lid, lower lid, iris center) so
both horizontal ratios grow in the same direction when the eyes move.
Subject's right eye: corners 33/133, lids 159/145, iris 468.
Subject's left eye: corners 362/263, lids 386/374, iris 473.
"""

import math

import numpy as np

RIGHT_EYE = (33, 133, 159, 145, 468)
LEFT_EYE = (362, 263, 386, 374, 473)

LOOK_BLENDSHAPES = (
    "eyeLookInLeft",
    "eyeLookOutLeft",
    "eyeLookUpLeft",
    "eyeLookDownLeft",
    "eyeLookInRight",
    "eyeLookOutRight",
    "eyeLookUpRight",
    "eyeLookDownRight",
)
BLINK_BLENDSHAPES = ("eyeBlinkLeft", "eyeBlinkRight")
BLINK_THRESHOLD = 0.45

FEATURE_NAMES = (
    "r_h",
    "r_v",
    "l_h",
    "l_v",
    "yaw",
    "pitch",
    "roll",
    "tx",
    "ty",
    "tz",
    *LOOK_BLENDSHAPES,
)
FEATURE_DIM = len(FEATURE_NAMES)


def eye_ratios(points: np.ndarray, eye: tuple[int, int, int, int, int]) -> tuple[float, float]:
    """Iris position along the corner axis (h) and lid axis (v), both roughly 0..1."""
    left, right, upper, lower, iris = (points[i] for i in eye)
    return _project(left, right, iris), _project(upper, lower, iris)


def _project(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    axis = b - a
    denom = float(axis @ axis)
    if denom < 1e-9:
        raise ValueError("degenerate eye")
    return float((p - a) @ axis / denom)


def head_pose(matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Yaw, pitch, roll (radians) and translation (cm) from the 4x4 face transform."""
    r = matrix[:3, :3]
    scale = np.linalg.norm(r[:, 0])
    if scale > 0:
        r = r / scale
    yaw = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    pitch = math.atan2(r[2, 1], r[2, 2])
    roll = math.atan2(r[1, 0], r[0, 0])
    tx, ty, tz = (float(v) for v in matrix[:3, 3])
    return yaw, pitch, roll, tx, ty, tz


def is_blinking(blendshapes: dict[str, float]) -> bool:
    return any(blendshapes.get(name, 0.0) > BLINK_THRESHOLD for name in BLINK_BLENDSHAPES)


def extract(
    landmarks: np.ndarray,
    matrix: np.ndarray,
    blendshapes: dict[str, float],
    width: int,
    height: int,
) -> np.ndarray:
    """Build the feature vector. `landmarks` is (478, 3) in normalized image coords."""
    if landmarks.shape[0] < 478:
        raise ValueError(f"expected 478 landmarks, got {landmarks.shape[0]}")
    points = landmarks[:, :2] * np.array([width, height], dtype=float)
    r_h, r_v = eye_ratios(points, RIGHT_EYE)
    l_h, l_v = eye_ratios(points, LEFT_EYE)
    pose = head_pose(matrix)
    looks = [blendshapes.get(name, 0.0) for name in LOOK_BLENDSHAPES]
    vector = np.array([r_h, r_v, l_h, l_v, *pose, *looks], dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("non-finite features")
    return vector
