"""Window actions through Accessibility.

Native Split View can't be scripted, so "split screen" tiles two windows
side by side on the same screen instead.
"""

import time

from peek.screen import ax, desktop
from peek.screen.ax import Rect


class ActionError(Exception):
    pass


def _visible_rect(window) -> Rect:
    rect = ax.frame(window)
    if rect is None:
        raise ActionError("can't read that window's position")
    screen = desktop.screen_for(rect)
    if screen is None:
        raise ActionError("no screen found")
    return screen.visible


def focus(window, pid: int) -> None:
    app = desktop.running_app(pid)
    if app is not None:
        app.activateWithOptions_(0)
    if ax.attr(window, "AXMinimized"):
        ax.set_attr(window, "AXMinimized", False)
    ax.perform(window, "AXRaise")
    ax.set_attr(window, "AXMain", True)


def _press_button(window, attribute: str, what: str) -> None:
    button = ax.attr(window, attribute)
    if button is None or not ax.perform(button, "AXPress"):
        raise ActionError(f"that window has no {what} button I can press")


def _halves(area: Rect, side: str) -> Rect:
    hw, hh = area.w / 2, area.h / 2
    return {
        "left_half": Rect(area.x, area.y, hw, area.h),
        "right_half": Rect(area.x + hw, area.y, hw, area.h),
        "top_half": Rect(area.x, area.y, area.w, hh),
        "bottom_half": Rect(area.x, area.y + hh, area.w, hh),
        "maximize": area,
        "center": Rect(area.x + area.w * 0.15, area.y + area.h * 0.1, area.w * 0.7, area.h * 0.8),
    }[side]


def _leave_fullscreen(window) -> None:
    if ax.attr(window, "AXFullScreen"):
        ax.set_attr(window, "AXFullScreen", False)
        time.sleep(0.9)


def act(window, pid: int, action: str) -> str:
    if window is None:
        raise ActionError("I can't control that window through Accessibility")
    if action == "fullscreen":
        if not ax.set_attr(window, "AXFullScreen", True):
            _press_button(window, "AXFullScreenButton", "full screen")
        return "entered full screen"
    if action == "exit_fullscreen":
        if not ax.set_attr(window, "AXFullScreen", False):
            raise ActionError("couldn't leave full screen")
        return "left full screen"
    if action == "close":
        _press_button(window, "AXCloseButton", "close")
        return "closed"
    if action == "minimize":
        if not ax.set_attr(window, "AXMinimized", True):
            _press_button(window, "AXMinimizeButton", "minimize")
        return "minimized"
    if action == "focus":
        focus(window, pid)
        return "brought to front"
    if action in ("left_half", "right_half", "top_half", "bottom_half", "maximize", "center"):
        _leave_fullscreen(window)
        target = _halves(_visible_rect(window), action)
        if not ax.set_frame(window, target):
            raise ActionError("that window can't be moved or resized")
        focus(window, pid)
        return f"moved to {action.replace('_', ' ')}"
    if action == "next_display":
        _leave_fullscreen(window)
        rect = ax.frame(window)
        all_screens = desktop.screens()
        if rect is None or len(all_screens) < 2:
            raise ActionError("there's only one display")
        current = desktop.screen_for(rect)
        index = all_screens.index(current) if current in all_screens else 0
        dest = all_screens[(index + 1) % len(all_screens)].visible
        dx, dy = rect.x - current.visible.x, rect.y - current.visible.y
        w, h = min(rect.w, dest.w), min(rect.h, dest.h)
        ax.set_frame(window, Rect(dest.x + min(dx, dest.w - w), dest.y + min(dy, dest.h - h), w, h))
        return "moved to the next display"
    raise ActionError(f"unknown window action {action}")


def split(left, left_pid: int, right, right_pid: int) -> str:
    if left is None or right is None:
        raise ActionError("I can't control one of those windows")
    _leave_fullscreen(left)
    _leave_fullscreen(right)
    area = _visible_rect(left)
    ax.set_frame(left, _halves(area, "left_half"))
    ax.set_frame(right, _halves(area, "right_half"))
    focus(right, right_pid)
    focus(left, left_pid)
    return "split side by side"
