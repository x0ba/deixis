"""Click-through overlays: a gaze dot and a status pill. Main thread only."""

import AppKit
import objc

from deixis.screen.desktop import screens

DOT = 34.0
PILL_W, PILL_H = 640.0, 64.0

COLORS = {
    "idle": (0.55, 0.55, 0.6),
    "listening": (0.2, 0.75, 1.0),
    "working": (1.0, 0.72, 0.2),
    "speaking": (0.45, 0.9, 0.5),
    "error": (1.0, 0.4, 0.35),
}


def _rgba(rgb: tuple[float, float, float], a: float):
    return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgb, a)


def _overlay_window(rect) -> AppKit.NSWindow:
    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        rect, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False
    )
    window.setOpaque_(False)
    window.setBackgroundColor_(AppKit.NSColor.clearColor())
    window.setIgnoresMouseEvents_(True)
    window.setHasShadow_(False)
    window.setLevel_(AppKit.NSScreenSaverWindowLevel)
    window.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorStationary
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorIgnoresCycle
    )
    window.setReleasedWhenClosed_(False)
    return window


class DotView(AppKit.NSView):
    def initWithFrame_(self, frame):
        self = objc.super(DotView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.rgb = COLORS["listening"]
        return self

    def drawRect_(self, rect):
        bounds = self.bounds()
        inset = AppKit.NSInsetRect(bounds, 5, 5)
        ring = AppKit.NSBezierPath.bezierPathWithOvalInRect_(inset)
        _rgba(self.rgb, 0.22).setFill()
        ring.fill()
        ring.setLineWidth_(3)
        _rgba(self.rgb, 0.85).setStroke()
        ring.stroke()
        center = AppKit.NSInsetRect(bounds, DOT / 2 - 3, DOT / 2 - 3)
        _rgba(self.rgb, 0.95).setFill()
        AppKit.NSBezierPath.bezierPathWithOvalInRect_(center).fill()


class PillView(AppKit.NSView):
    def initWithFrame_(self, frame):
        self = objc.super(PillView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.rgb = COLORS["idle"]
        self.message = ""
        return self

    def drawRect_(self, rect):
        bounds = self.bounds()
        path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, bounds.size.height / 2, bounds.size.height / 2
        )
        AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.88).setFill()
        path.fill()
        dot = AppKit.NSMakeRect(22, bounds.size.height / 2 - 9, 18, 18)
        _rgba(self.rgb, 1.0).setFill()
        AppKit.NSBezierPath.bezierPathWithOvalInRect_(dot).fill()
        style = AppKit.NSMutableParagraphStyle.alloc().init()
        style.setLineBreakMode_(AppKit.NSLineBreakByTruncatingHead)
        attrs = {
            AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_weight_(
                22, AppKit.NSFontWeightMedium
            ),
            AppKit.NSForegroundColorAttributeName: AppKit.NSColor.whiteColor(),
            AppKit.NSParagraphStyleAttributeName: style,
        }
        text_rect = AppKit.NSMakeRect(54, bounds.size.height / 2 - 15, bounds.size.width - 76, 30)
        AppKit.NSString.stringWithString_(self.message).drawInRect_withAttributes_(text_rect, attrs)


class Overlay:
    def __init__(self):
        self._dot_window = _overlay_window(AppKit.NSMakeRect(0, 0, DOT, DOT))
        self._dot = DotView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, DOT, DOT))
        self._dot_window.setContentView_(self._dot)
        self._pill_window = _overlay_window(AppKit.NSMakeRect(0, 0, PILL_W, PILL_H))
        self._pill = PillView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, PILL_W, PILL_H))
        self._pill_window.setContentView_(self._pill)
        self._main_h = AppKit.NSScreen.screens()[0].frame().size.height
        self._dot_enabled = True
        self._place_pill()

    def _place_pill(self) -> None:
        found = screens()
        if not found:
            return
        visible = found[0].visible
        x = visible.x + (visible.w - PILL_W) / 2
        top = visible.y + 14
        self._pill_window.setFrameOrigin_((x, self._main_h - top - PILL_H))

    def set_dot_enabled(self, enabled: bool) -> None:
        self._dot_enabled = enabled
        if not enabled:
            self._dot_window.orderOut_(None)

    def move_dot(self, x: float, y: float) -> None:
        if not self._dot_enabled:
            return
        self._dot_window.setFrameOrigin_((x - DOT / 2, self._main_h - y - DOT / 2))
        if not self._dot_window.isVisible():
            self._dot_window.orderFrontRegardless()

    def status(self, state: str, message: str) -> None:
        rgb = COLORS.get(state, COLORS["idle"])
        self._pill.rgb = rgb
        self._pill.message = message
        self._pill.setNeedsDisplay_(True)
        self._dot.rgb = rgb
        self._dot.setNeedsDisplay_(True)
        if message:
            self._place_pill()
            self._pill_window.orderFrontRegardless()
        else:
            self._pill_window.orderOut_(None)
