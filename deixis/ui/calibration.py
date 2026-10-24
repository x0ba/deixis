"""Hands-free eye calibration.

Dots appear one at a time on the main screen. The user just looks at each
one; nothing needs clicking. Esc cancels. Main thread only.
"""

import time
from collections.abc import Callable

import AppKit
import numpy as np
import objc

from deixis.gaze import model as gaze_model

SETTLE_S = 0.9
COLLECT_S = 1.1
INTRO_S = 2.5
MIN_SAMPLES = 8
GRID = [
    (0.5, 0.5),
    (0.08, 0.08),
    (0.5, 0.08),
    (0.92, 0.08),
    (0.92, 0.5),
    (0.92, 0.92),
    (0.5, 0.92),
    (0.08, 0.92),
    (0.08, 0.5),
    (0.3, 0.3),
    (0.7, 0.3),
    (0.7, 0.7),
    (0.3, 0.7),
]


class CalibrationWindow(AppKit.NSWindow):
    def canBecomeKeyWindow(self):
        return True

    def keyDown_(self, event):
        if event.keyCode() == 53 and self.on_cancel:
            self.on_cancel()


class CalibrationView(AppKit.NSView):
    def initWithFrame_(self, frame):
        self = objc.super(CalibrationView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.target = None
        self.radius = 18.0
        self.message = ""
        return self

    def isFlipped(self):
        return True

    def drawRect_(self, rect):
        AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.55).setFill()
        AppKit.NSRectFill(self.bounds())
        if self.target is not None:
            x, y = self.target
            r = self.radius
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.75, 0.2, 1.0).setFill()
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                AppKit.NSMakeRect(x - r, y - r, 2 * r, 2 * r)
            ).fill()
            AppKit.NSColor.blackColor().setFill()
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                AppKit.NSMakeRect(x - 3, y - 3, 6, 6)
            ).fill()
        if self.message:
            attrs = {
                AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_weight_(
                    28, AppKit.NSFontWeightSemibold
                ),
                AppKit.NSForegroundColorAttributeName: AppKit.NSColor.whiteColor(),
            }
            text = AppKit.NSString.stringWithString_(self.message)
            size = text.sizeWithAttributes_(attrs)
            b = self.bounds().size
            text.drawAtPoint_withAttributes_(
                ((b.width - size.width) / 2, b.height * 0.5 + 60), attrs
            )


class Calibrator:
    """Runs one calibration. `done(model or None, message)` is called at the end."""

    def __init__(self, camera, done: Callable[[object, str], None]):
        self._camera = camera
        self._done = done
        screen = AppKit.NSScreen.screens()[0]
        frame = screen.frame()
        self._size = (frame.size.width, frame.size.height)
        self._window = CalibrationWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False
        )
        self._window.on_cancel = self.cancel
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(AppKit.NSColor.clearColor())
        self._window.setLevel_(AppKit.NSScreenSaverWindowLevel + 1)
        self._window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        self._window.setReleasedWhenClosed_(False)
        self._view = CalibrationView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, frame.size.width, frame.size.height)
        )
        self._window.setContentView_(self._view)
        self._samples: list[tuple[np.ndarray, tuple[float, float], int]] = []
        self._index = -1
        self._phase_start = 0.0
        self._collecting = False
        self._timer = None
        self._finished = False

    def start(self) -> None:
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        self._window.makeKeyAndOrderFront_(None)
        self._view.message = "Look at each dot until it moves. Press Esc to cancel."
        self._view.setNeedsDisplay_(True)
        self._camera.on_features = self._on_features
        self._phase_start = time.monotonic() + INTRO_S
        self._timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            1 / 30, True, lambda _t: self._tick()
        )

    def _target_px(self, index: int) -> tuple[float, float]:
        fx, fy = GRID[index]
        return fx * self._size[0], fy * self._size[1]

    def _on_features(self, t: float, vector: np.ndarray) -> None:
        if self._collecting and self._index >= 0:
            self._samples.append((vector, self._target_px(self._index), self._index))

    def _tick(self) -> None:
        now = time.monotonic()
        if now < self._phase_start:
            return
        if self._index < 0:
            self._index = 0
            self._phase_start = now
            self._view.message = ""
        elapsed = now - self._phase_start
        self._view.target = self._target_px(self._index)
        if not self._camera.face_visible:
            self._view.message = "I can't see your face. Check the camera and lighting."
        elif self._view.message:
            self._view.message = ""
        if elapsed < SETTLE_S:
            self._collecting = False
            self._view.radius = 26 - 12 * (elapsed / SETTLE_S)
        elif elapsed < SETTLE_S + COLLECT_S:
            self._collecting = True
            self._view.radius = 14 - 6 * ((elapsed - SETTLE_S) / COLLECT_S)
        else:
            self._collecting = False
            self._index += 1
            self._phase_start = now
            if self._index >= len(GRID):
                self._finish()
                return
        self._view.setNeedsDisplay_(True)

    def _close(self) -> None:
        self._finished = True
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self._camera.on_features = None
        self._window.orderOut_(None)

    def cancel(self) -> None:
        if not self._finished:
            self._close()
            self._done(None, "Calibration cancelled.")

    def _finish(self) -> None:
        self._close()
        counts = np.bincount([s[2] for s in self._samples], minlength=len(GRID))
        good = [i for i, c in enumerate(counts) if c >= MIN_SAMPLES]
        if len(good) < 8:
            self._done(None, "I couldn't see your eyes well enough. Let's try again later.")
            return
        rows = [s for s in self._samples if s[2] in good]
        features = np.array([r[0] for r in rows])
        targets = np.array([r[1] for r in rows], dtype=float)
        groups = np.array([r[2] for r in rows])
        try:
            fitted = gaze_model.fit(features, targets, groups)
        except (ValueError, np.linalg.LinAlgError) as err:
            self._done(None, f"Calibration failed: {err}")
            return
        self._done(fitted, f"Calibrated. Accuracy is about {round(fitted.cv_error_px)} pixels.")
