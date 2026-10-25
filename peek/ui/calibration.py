"""Hands-free eye calibration and re-centering.

Dots appear one at a time on the main screen. The user just looks at each
one; nothing needs clicking. Esc cancels. Main thread only.

Full calibration covers a 5x5 grid, then repeats 9 of the points after the
user shifts their head, so the model sees more than one posture. Re-centering
shows a single dot and only corrects the model's offset, for drift during a
session.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import AppKit
import numpy as np
import objc

from peek.gaze import model as gaze_model

SETTLE_S = 0.8
COLLECT_S = 1.0
INTRO_S = 2.5
MIN_SAMPLES = 8
MIN_COVERAGE = 0.6


@dataclass(frozen=True)
class Pause:
    text: str
    seconds: float
    spoken: str = ""


@dataclass(frozen=True)
class Dot:
    fx: float
    fy: float


_STEPS = (0.06, 0.28, 0.5, 0.72, 0.94)
GRID = [
    Dot(fx, fy)
    for row, fy in enumerate(_STEPS)
    for fx in (_STEPS if row % 2 == 0 else _STEPS[::-1])
]
HEAD_GRID = [Dot(fx, fy) for fy in (0.06, 0.5, 0.94) for fx in (0.06, 0.5, 0.94)]
FULL_PLAN = [
    Pause("Look at each dot until it moves. Press Esc to cancel.", INTRO_S),
    *GRID,
    Pause(
        "Now move your head a little, closer or to one side, and keep following the dots.",
        3.5,
        spoken="Now move your head a little, and keep following the dots.",
    ),
    *HEAD_GRID,
]
RECENTER_PLAN = [Pause("Look at the dot.", 1.0), Dot(0.5, 0.5)]


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


@dataclass(frozen=True)
class Samples:
    features: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    visits: np.ndarray


class Calibrator:
    """Plays one dot plan. `done(model or None, message)` is called at the end.

    `quick` re-centers the camera's current model instead of fitting a new one.
    """

    def __init__(
        self,
        camera,
        done: Callable[[object, str], None],
        say: Callable[[str], None] = lambda _text: None,
        quick: bool = False,
    ):
        self._camera = camera
        self._done = done
        self._say = say
        self._quick = quick
        self._plan = RECENTER_PLAN if quick else FULL_PLAN
        positions = sorted({(d.fx, d.fy) for d in self._plan if isinstance(d, Dot)})
        self._group = {p: i for i, p in enumerate(positions)}
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
        self._samples: list[tuple[np.ndarray, tuple[float, float], int, int]] = []
        self._index = 0
        self._phase_start = 0.0
        self._collecting = False
        self._timer = None
        self._finished = False

    def start(self) -> None:
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        self._window.makeKeyAndOrderFront_(None)
        self._camera.on_features = self._on_features
        self._begin(0, time.monotonic())
        self._timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            1 / 30, True, lambda _t: self._tick()
        )

    def _target_px(self, dot: Dot) -> tuple[float, float]:
        return dot.fx * self._size[0], dot.fy * self._size[1]

    def _on_features(self, t: float, vector: np.ndarray) -> None:
        step = self._plan[self._index] if self._index < len(self._plan) else None
        if self._collecting and isinstance(step, Dot):
            group = self._group[(step.fx, step.fy)]
            self._samples.append((vector, self._target_px(step), group, self._index))

    def _begin(self, index: int, now: float) -> None:
        self._index = index
        self._phase_start = now
        self._collecting = False
        step = self._plan[index]
        if isinstance(step, Pause):
            self._view.target = None
            self._view.message = step.text
            if step.spoken:
                self._say(step.spoken)
        else:
            self._view.target = self._target_px(step)
            self._view.message = ""
        self._view.setNeedsDisplay_(True)

    def _tick(self) -> None:
        now = time.monotonic()
        step = self._plan[self._index]
        elapsed = now - self._phase_start
        if isinstance(step, Pause):
            if elapsed >= step.seconds:
                self._advance(now)
            return
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
            self._advance(now)
            return
        self._view.setNeedsDisplay_(True)

    def _advance(self, now: float) -> None:
        if self._index + 1 >= len(self._plan):
            self._finish()
        else:
            self._begin(self._index + 1, now)

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
            self._done(None, "Re-centering cancelled." if self._quick else "Calibration cancelled.")

    def _usable(self) -> Samples | None:
        """Samples from dots that were seen long enough, or None if there are none."""
        visits = np.array([s[3] for s in self._samples], dtype=int)
        counts = np.bincount(visits, minlength=len(self._plan)) if len(visits) else visits
        rows = [s for s in self._samples if counts[s[3]] >= MIN_SAMPLES]
        if not rows:
            return None
        return Samples(
            features=np.array([r[0] for r in rows]),
            targets=np.array([r[1] for r in rows], dtype=float),
            groups=np.array([r[2] for r in rows], dtype=int),
            visits=np.array([r[3] for r in rows], dtype=int),
        )

    def _finish(self) -> None:
        self._close()
        usable = self._usable()
        if self._quick:
            self._finish_recenter(usable)
        else:
            self._finish_full(usable)

    def _finish_recenter(self, usable: Samples | None) -> None:
        model = self._camera.model
        if model is None or usable is None:
            self._done(None, "I couldn't see your eyes well enough to re-center.")
            return
        before = np.median(model.predict(usable.features), axis=0)
        moved = float(np.linalg.norm(usable.targets[0] - before))
        self._done(
            model.recentered(usable.features, tuple(usable.targets[0])),
            f"Re-centered. It was off by about {round(moved)} pixels.",
        )

    def _finish_full(self, usable: Samples | None) -> None:
        if usable is None or len(np.unique(usable.groups)) < MIN_COVERAGE * len(self._group):
            self._done(None, "I couldn't see your eyes well enough. Let's try again later.")
            return
        np.savez(
            self._camera.calibration_path.with_name("calibration_samples.npz"),
            features=usable.features,
            targets=usable.targets,
            groups=usable.groups,
            visits=usable.visits,
            screen=np.array(self._size),
        )
        try:
            fitted = gaze_model.fit(usable.features, usable.targets, usable.groups, usable.visits)
        except (ValueError, np.linalg.LinAlgError) as err:
            self._done(None, f"Calibration failed: {err}")
            return
        self._done(fitted, f"Calibrated. Accuracy is about {round(fitted.cv_error_px)} pixels.")
