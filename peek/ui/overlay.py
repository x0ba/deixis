"""Click-through overlays: a highlight around the gazed element and a status pill.

Main thread only.
"""

import AppKit
import objc

from peek.screen.ax import Rect
from peek.screen.desktop import screens

HIGHLIGHT_RGB = (0.04, 0.52, 1.0)
HIGHLIGHT_PAD = 4.0
HIGHLIGHT_MOVE_S = 0.09
BORDER_W, PINNED_BORDER_W = 3.0, 5.0
FILL_ALPHA, PINNED_FILL_ALPHA = 0.12, 0.22
PILL_W, PILL_H = 640.0, 64.0
PILL_TEXT_X, PILL_PAD_Y = 54.0, 17.0
PILL_MAX_LINES = 4

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


def _highlight_view() -> AppKit.NSView:
    view = AppKit.NSView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 10, 10))
    view.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
    view.setWantsLayer_(True)
    layer = view.layer()
    layer.setCornerRadius_(7.0)
    layer.setBorderColor_(_rgba(HIGHLIGHT_RGB, 0.95).CGColor())
    _style_highlight(view, pinned=False)
    return view


def _style_highlight(view, pinned: bool) -> None:
    """A pinned highlight is bolder, so it's clear Peek is holding on to it."""
    layer = view.layer()
    layer.setBorderWidth_(PINNED_BORDER_W if pinned else BORDER_W)
    alpha = PINNED_FILL_ALPHA if pinned else FILL_ALPHA
    layer.setBackgroundColor_(_rgba(HIGHLIGHT_RGB, alpha).CGColor())


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
        h = bounds.size.height
        radius = min(h / 2, PILL_H / 2)
        path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, radius, radius
        )
        AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.88).setFill()
        path.fill()
        # The dot sits beside the first line of text.
        dot = AppKit.NSMakeRect(22, h - PILL_H / 2 - 9, 18, 18)
        _rgba(self.rgb, 1.0).setFill()
        AppKit.NSBezierPath.bezierPathWithOvalInRect_(dot).fill()
        text_rect = AppKit.NSMakeRect(
            PILL_TEXT_X, PILL_PAD_Y, bounds.size.width - PILL_TEXT_X - 22, h - 2 * PILL_PAD_Y
        )
        AppKit.NSString.stringWithString_(self.message).drawWithRect_options_attributes_(
            text_rect,
            AppKit.NSStringDrawingUsesLineFragmentOrigin
            | AppKit.NSStringDrawingTruncatesLastVisibleLine,
            _text_attrs(),
        )


def _text_attrs() -> dict:
    style = AppKit.NSMutableParagraphStyle.alloc().init()
    style.setLineBreakMode_(AppKit.NSLineBreakByWordWrapping)
    return {
        AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_weight_(
            22, AppKit.NSFontWeightMedium
        ),
        AppKit.NSForegroundColorAttributeName: AppKit.NSColor.whiteColor(),
        AppKit.NSParagraphStyleAttributeName: style,
    }


def _pill_height(message: str) -> float:
    """Tall enough for the wrapped message, up to PILL_MAX_LINES lines."""
    attrs = _text_attrs()
    width = PILL_W - PILL_TEXT_X - 22
    text = AppKit.NSString.stringWithString_(message or " ")
    options = AppKit.NSStringDrawingUsesLineFragmentOrigin
    bounds = text.boundingRectWithSize_options_attributes_((width, 10_000), options, attrs)
    needed = bounds.size.height
    line = attrs[AppKit.NSFontAttributeName]
    line_h = line.ascender() - line.descender() + line.leading()
    text_h = min(needed, line_h * PILL_MAX_LINES)
    return max(PILL_H, text_h + 2 * PILL_PAD_Y)


class Overlay:
    def __init__(self):
        self._highlight_window = _overlay_window(AppKit.NSMakeRect(0, 0, 10, 10))
        self._highlight_window.setContentView_(_highlight_view())
        self._pill_window = _overlay_window(AppKit.NSMakeRect(0, 0, PILL_W, PILL_H))
        self._pill = PillView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, PILL_W, PILL_H))
        self._pill_window.setContentView_(self._pill)
        self._main_h = AppKit.NSScreen.screens()[0].frame().size.height
        self._highlight_enabled = True
        self._place_pill()

    def _place_pill(self) -> None:
        found = screens()
        if not found:
            return
        visible = found[0].visible
        x = visible.x + (visible.w - PILL_W) / 2
        top = visible.y + 14
        h = self._pill_window.frame().size.height
        self._pill_window.setFrameOrigin_((x, self._main_h - top - h))

    def set_highlight_enabled(self, enabled: bool) -> None:
        self._highlight_enabled = enabled
        if not enabled:
            self._highlight_window.orderOut_(None)

    def highlight(self, rect: Rect | None, pinned: bool = False) -> None:
        """Outline `rect` (global top-left coordinates), or hide the outline."""
        window = self._highlight_window
        if rect is None or not self._highlight_enabled:
            window.orderOut_(None)
            return
        _style_highlight(window.contentView(), pinned)
        frame = AppKit.NSMakeRect(
            rect.x - HIGHLIGHT_PAD,
            self._main_h - rect.y - rect.h - HIGHLIGHT_PAD,
            rect.w + 2 * HIGHLIGHT_PAD,
            rect.h + 2 * HIGHLIGHT_PAD,
        )
        if window.isVisible():
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(HIGHLIGHT_MOVE_S)
            window.animator().setFrame_display_(frame, True)
            AppKit.NSAnimationContext.endGrouping()
        else:
            window.setFrame_display_(frame, True)
            window.orderFrontRegardless()

    def status(self, state: str, message: str) -> None:
        rgb = COLORS.get(state, COLORS["idle"])
        self._pill.rgb = rgb
        self._pill.message = message
        self._pill.setNeedsDisplay_(True)
        if message:
            h = _pill_height(message)
            self._pill_window.setContentSize_((PILL_W, h))
            self._pill.setFrame_(AppKit.NSMakeRect(0, 0, PILL_W, h))
            self._place_pill()
            self._pill_window.orderFrontRegardless()
        else:
            self._pill_window.orderOut_(None)
