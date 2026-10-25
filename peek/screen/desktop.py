"""Screens and on-screen windows, in global top-left coordinates."""

import os
from dataclasses import dataclass

import AppKit
import Quartz

from peek.screen.ax import Rect


@dataclass(frozen=True)
class Screen:
    frame: Rect
    visible: Rect


@dataclass(frozen=True)
class WindowInfo:
    number: int
    pid: int
    app: str
    bounds: Rect


def screens() -> list[Screen]:
    all_screens = AppKit.NSScreen.screens()
    if not all_screens:
        return []
    main_h = all_screens[0].frame().size.height

    def flip(r) -> Rect:
        return Rect(r.origin.x, main_h - r.origin.y - r.size.height, r.size.width, r.size.height)

    return [Screen(flip(s.frame()), flip(s.visibleFrame())) for s in all_screens]


def screen_for(rect: Rect) -> Screen | None:
    cx, cy = rect.center
    found = screens()
    for s in found:
        if s.frame.contains(cx, cy):
            return s
    return found[0] if found else None


def windows() -> list[WindowInfo]:
    """Normal app windows, front to back, excluding Peek's own overlays."""
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    raw = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
    me = os.getpid()
    out = []
    for w in raw:
        if w.get("kCGWindowLayer", 1) != 0 or w.get("kCGWindowOwnerPID") == me:
            continue
        if w.get("kCGWindowAlpha", 1) <= 0:
            continue
        b = w.get("kCGWindowBounds") or {}
        rect = Rect(b.get("X", 0), b.get("Y", 0), b.get("Width", 0), b.get("Height", 0))
        if rect.w < 40 or rect.h < 40:
            continue
        out.append(
            WindowInfo(
                number=int(w["kCGWindowNumber"]),
                pid=int(w["kCGWindowOwnerPID"]),
                app=str(w.get("kCGWindowOwnerName", "")),
                bounds=rect,
            )
        )
    return out


def window_at(x: float, y: float) -> WindowInfo | None:
    for w in windows():
        if w.bounds.contains(x, y):
            return w
    return None


def running_app(pid: int):
    return AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)


def frontmost_app():
    return AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
