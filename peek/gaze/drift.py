"""Corrects slow drift in webcam gaze from moments when we know where the user looked.

People look at what they click. When the user clicks with a mouse, or winks to
click the highlighted element, the gaze just before should have been on that
spot. If the predictions sat consistently to one side of it, the model's
offset is nudged part of the way back. A large miss is ignored, since the user
probably wasn't looking at what they clicked.
"""

import numpy as np

from peek.screen.ax import Rect

MIN_SAMPLES = 5
MAX_MISS_ERRORS = 2.5  # ignore misses farther than this many typical errors
MAX_MISS_PX = 220.0
POINT_GAIN = 0.5  # a mouse click is an exact point
RECT_GAIN = 0.3  # a winked element is only roughly where they looked


def shift(predicted: np.ndarray, target: Rect | tuple[float, float], error_px: float):
    """The offset change that moves `predicted` gaze (n, 2) part of the way onto
    `target`, or None if there's too little gaze or it was looking elsewhere.

    For a rectangle the target is its point nearest the gaze, so looking anywhere
    inside a big element counts as on target.
    """
    if len(predicted) < MIN_SAMPLES:
        return None
    gx, gy = np.median(predicted, axis=0)
    if isinstance(target, Rect):
        tx = min(max(gx, target.x), target.x + target.w)
        ty = min(max(gy, target.y), target.y + target.h)
        gain = RECT_GAIN
    else:
        tx, ty = target
        gain = POINT_GAIN
    miss = np.array([tx - gx, ty - gy])
    if np.linalg.norm(miss) > max(MAX_MISS_ERRORS * error_px, MAX_MISS_PX):
        return None
    return gain * miss
