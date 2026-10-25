"""Synthetic input: clicks, typing, key chords, scrolling, launching.

Typing posts real key events carrying Unicode strings rather than setting
AXValue, so web apps see ordinary input events.
"""

import subprocess
import time

import AppKit
import Quartz

from peek.actions.windows import ActionError
from peek.screen import ax, desktop

KEYCODES = {
    "a": 0,
    "s": 1,
    "d": 2,
    "f": 3,
    "h": 4,
    "g": 5,
    "z": 6,
    "x": 7,
    "c": 8,
    "v": 9,
    "b": 11,
    "q": 12,
    "w": 13,
    "e": 14,
    "r": 15,
    "y": 16,
    "t": 17,
    "1": 18,
    "2": 19,
    "3": 20,
    "4": 21,
    "6": 22,
    "5": 23,
    "=": 24,
    "9": 25,
    "7": 26,
    "-": 27,
    "8": 28,
    "0": 29,
    "]": 30,
    "o": 31,
    "u": 32,
    "[": 33,
    "i": 34,
    "p": 35,
    "return": 36,
    "enter": 36,
    "l": 37,
    "j": 38,
    "'": 39,
    "k": 40,
    ";": 41,
    "\\": 42,
    ",": 43,
    "/": 44,
    "n": 45,
    "m": 46,
    ".": 47,
    "tab": 48,
    "space": 49,
    "`": 50,
    "delete": 51,
    "backspace": 51,
    "escape": 53,
    "esc": 53,
    "forwarddelete": 117,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "pagedown": 121,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "f1": 122,
    "f2": 120,
    "f3": 99,
    "f4": 118,
    "f5": 96,
    "f6": 97,
    "f7": 98,
    "f8": 100,
    "f9": 101,
    "f10": 109,
    "f11": 103,
    "f12": 111,
}
MODIFIERS = {
    "cmd": Quartz.kCGEventFlagMaskCommand,
    "command": Quartz.kCGEventFlagMaskCommand,
    "shift": Quartz.kCGEventFlagMaskShift,
    "opt": Quartz.kCGEventFlagMaskAlternate,
    "option": Quartz.kCGEventFlagMaskAlternate,
    "alt": Quartz.kCGEventFlagMaskAlternate,
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "control": Quartz.kCGEventFlagMaskControl,
    "fn": Quartz.kCGEventFlagMaskSecondaryFn,
}


def _post(event) -> None:
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def click_point(x: float, y: float, double: bool = False) -> None:
    point = Quartz.CGPointMake(x, y)
    _post(Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, point, 0))
    time.sleep(0.03)
    for n in (1, 2) if double else (1,):
        down = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown, point, 0)
        up = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp, point, 0)
        Quartz.CGEventSetIntegerValueField(down, Quartz.kCGMouseEventClickState, n)
        Quartz.CGEventSetIntegerValueField(up, Quartz.kCGMouseEventClickState, n)
        _post(down)
        time.sleep(0.02)
        _post(up)
        time.sleep(0.04)


def activate_owner(element) -> None:
    pid = ax.pid_of(element)
    app = desktop.running_app(pid) if pid else None
    if app is not None and not app.isActive():
        app.activateWithOptions_(0)
        time.sleep(0.15)
    window = ax.window_of(element)
    if window is not None:
        ax.perform(window, "AXRaise")


def click(element, double: bool = False) -> str:
    activate_owner(element)
    if not double and "AXPress" in ax.action_names(element) and ax.perform(element, "AXPress"):
        return "pressed"
    rect = ax.frame(element)
    if rect is None:
        raise ActionError("can't find where that element is")
    click_point(*rect.center, double=double)
    return "clicked"


def focus_for_typing(element) -> None:
    activate_owner(element)
    ax.set_attr(element, "AXFocused", True)
    time.sleep(0.08)
    if ax.attr(element, "AXFocused"):
        return
    rect = ax.frame(element)
    if rect is None:
        raise ActionError("can't focus that field")
    click_point(*rect.center)
    time.sleep(0.1)


def type_string(text: str) -> None:
    source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for start in range(0, len(text), 16):
        chunk = text[start : start + 16]
        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(source, 0, down)
            Quartz.CGEventKeyboardSetUnicodeString(event, len(chunk), chunk)
            _post(event)
        time.sleep(0.012)


def chord(spec: str) -> None:
    """Press a chord like "cmd+shift+t" or "return"."""
    parts = [p.strip().lower() for p in spec.replace(" ", "").split("+") if p.strip()]
    if not parts:
        raise ActionError("empty key chord")
    *mods, key = parts
    flags = 0
    for m in mods:
        if m not in MODIFIERS:
            raise ActionError(f"unknown modifier {m}")
        flags |= MODIFIERS[m]
    if key not in KEYCODES:
        raise ActionError(f"unknown key {key}")
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, KEYCODES[key], down)
        Quartz.CGEventSetFlags(event, flags)
        _post(event)
        time.sleep(0.02)


def type_into(element, text: str, replace: bool, submit: bool) -> str:
    if element is not None:
        focus_for_typing(element)
    if replace:
        chord("cmd+a")
        time.sleep(0.05)
    type_string(text)
    if submit:
        time.sleep(0.05)
        chord("return")
    return "typed"


def scroll(x: float, y: float, direction: str, pages: float) -> str:
    step = {"up": (3, 0), "down": (-3, 0), "left": (0, 3), "right": (0, -3)}.get(direction)
    if step is None:
        raise ActionError(f"unknown scroll direction {direction}")
    point = Quartz.CGPointMake(x, y)
    _post(Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, point, 0))
    for _ in range(max(1, int(pages * 5))):
        event = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitLine, 2, step[0], step[1]
        )
        Quartz.CGEventSetLocation(event, point)
        _post(event)
        time.sleep(0.012)
    return f"scrolled {direction}"


def open_app(name: str) -> str:
    result = subprocess.run(["open", "-a", name], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ActionError(f"couldn't open {name}")
    return f"opened {name}"


def open_url(url: str) -> str:
    if "://" not in url:
        url = "https://" + url
    ok = AppKit.NSWorkspace.sharedWorkspace().openURL_(AppKit.NSURL.URLWithString_(url))
    if not ok:
        raise ActionError("couldn't open that link")
    return f"opened {url}"
