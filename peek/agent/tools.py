"""Tools the agent can call, and how each one runs against the live screen."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from peek.actions import input as inp
from peek.actions import windows as win
from peek.actions.windows import ActionError
from peek.screen import ax
from peek.screen.context import Registry, Snapshot, describe

WINDOW_ACTIONS = [
    "fullscreen",
    "exit_fullscreen",
    "close",
    "minimize",
    "maximize",
    "left_half",
    "right_half",
    "top_half",
    "bottom_half",
    "center",
    "focus",
    "next_display",
]

TOOLS = [
    {
        "name": "window_action",
        "description": "Act on one window: full screen, close, minimize, maximize (fill the "
        "screen without entering full screen), tile to a half, center, bring to front, or "
        "move to the next display.",
        "input_schema": {
            "type": "object",
            "properties": {
                "window_id": {"type": "string", "description": "A window id such as w1."},
                "action": {"type": "string", "enum": WINDOW_ACTIONS},
            },
            "required": ["window_id", "action"],
        },
    },
    {
        "name": "split_screen",
        "description": "Tile two windows side by side on the left window's screen.",
        "input_schema": {
            "type": "object",
            "properties": {
                "left_window_id": {"type": "string"},
                "right_window_id": {"type": "string"},
            },
            "required": ["left_window_id", "right_window_id"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into a field. Focuses the element first (clicking it if "
        "needed). Omit element_id to type wherever keyboard focus already is.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element_id": {"type": "string"},
                "text": {"type": "string"},
                "replace": {
                    "type": "boolean",
                    "description": "Select all existing text in the field first.",
                },
                "submit": {"type": "boolean", "description": "Press Return afterwards."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "click",
        "description": "Press or click an element by id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element_id": {"type": "string"},
                "double": {"type": "boolean"},
            },
            "required": ["element_id"],
        },
    },
    {
        "name": "click_point",
        "description": "Click at global screen coordinates (points, top-left origin). Use "
        "only when no element id fits, e.g. a spot found in the screenshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "double": {"type": "boolean"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "press_keys",
        "description": "Press a key or chord in the frontmost app, e.g. 'cmd+t', 'escape', "
        "'cmd+shift+]'. Use window_action to activate the right window first if needed.",
        "input_schema": {
            "type": "object",
            "properties": {"keys": {"type": "string"}},
            "required": ["keys"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the content under the user's gaze (or under an element).",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                "pages": {"type": "number", "description": "Roughly how far; 1 is a screenful."},
                "element_id": {"type": "string"},
            },
            "required": ["direction"],
        },
    },
    {
        "name": "read_element",
        "description": "Get the full text of an element (up to 6000 characters), for reading "
        "aloud, summarizing or answering questions about it.",
        "input_schema": {
            "type": "object",
            "properties": {"element_id": {"type": "string"}},
            "required": ["element_id"],
        },
    },
    {
        "name": "open_app",
        "description": "Open or switch to an app by name, e.g. 'Safari'.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "open_url",
        "description": "Open a web address in the default browser.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "ask",
        "description": "Say a short question out loud and wait for the user's spoken answer. "
        "Use it whenever you need something from them before you can go on: which one they "
        "mean, a missing detail, or a yes before something destructive. The result has "
        "their answer and, if they looked at something while answering, that element (its "
        "new id works with the other tools). Earlier ids keep working.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "refresh_context",
        "description": "Re-read the screen at the user's current gaze, e.g. to check the "
        "result of an action or after something opened. Ids from earlier snapshots stop "
        "working.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "recalibrate",
        "description": "Start eye-tracking calibration (the user follows dots with their eyes). "
        "Use quick=true to re-center with a single dot when the gaze is consistently off "
        "in one direction; use a full calibration when it is wrong in different ways in "
        "different places.",
        "input_schema": {
            "type": "object",
            "properties": {"quick": {"type": "boolean"}},
        },
    },
]


def _full_text(element) -> str:
    parts = []
    for name in ("AXValue", "AXTitle", "AXDescription"):
        value = ax.attr(element, name)
        if value:
            parts.append(str(value))
    if not parts:
        stack, seen = [element], 0
        while stack and seen < 400:
            current = stack.pop(0)
            seen += 1
            value = ax.attr(current, "AXValue")
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
            stack.extend(ax.attr(current, "AXChildren") or [])
    joined = "\n".join(parts)
    return joined[:6000] or "(no readable text)"


@dataclass
class Answer:
    """What the user said back to a question, and what they looked at meanwhile."""

    text: str
    element: object | None = None
    point: tuple[float, float] | None = None


class Executor:
    def __init__(
        self,
        snapshot: Snapshot,
        refresh: Callable[[], Snapshot],
        recalibrate: Callable[[bool], None],
        ask: Callable[[str], Answer | None] = lambda _: None,
    ):
        self.snapshot = snapshot
        self._refresh = refresh
        self._recalibrate = recalibrate
        self._ask = ask

    @property
    def registry(self) -> Registry:
        return self.snapshot.registry

    def _window(self, key: str):
        if key not in self.registry.window_info:
            raise ActionError(f"no window {key}")
        return self.registry.windows.get(key), self.registry.window_info[key].pid

    def _element(self, key: str):
        element = self.registry.elements.get(key)
        if element is None:
            raise ActionError(f"no element {key}")
        return element

    def _gaze_point(self) -> tuple[float, float]:
        gaze = self.snapshot.data.get("gaze")
        if not gaze:
            raise ActionError("I don't know where you're looking")
        return gaze["x"], gaze["y"]

    def run(self, name: str, args: dict) -> tuple[str, Snapshot | None]:
        """Returns a short result string, and a new snapshot if the screen was re-read."""
        try:
            if name == "window_action":
                window, pid = self._window(args["window_id"])
                return win.act(window, pid, args["action"]), None
            if name == "split_screen":
                left, lp = self._window(args["left_window_id"])
                right, rp = self._window(args["right_window_id"])
                return win.split(left, lp, right, rp), None
            if name == "type_text":
                element = self._element(args["element_id"]) if args.get("element_id") else None
                return inp.type_into(
                    element, args["text"], bool(args.get("replace")), bool(args.get("submit"))
                ), None
            if name == "click":
                return inp.click(self._element(args["element_id"]), bool(args.get("double"))), None
            if name == "click_point":
                inp.click_point(float(args["x"]), float(args["y"]), bool(args.get("double")))
                return "clicked", None
            if name == "press_keys":
                inp.chord(args["keys"])
                return f"pressed {args['keys']}", None
            if name == "scroll":
                if args.get("element_id"):
                    rect = ax.frame(self._element(args["element_id"]))
                    if rect is None:
                        raise ActionError("can't find that element")
                    x, y = rect.center
                else:
                    x, y = self._gaze_point()
                return inp.scroll(x, y, args["direction"], float(args.get("pages", 1))), None
            if name == "read_element":
                return _full_text(self._element(args["element_id"])), None
            if name == "open_app":
                return inp.open_app(args["name"]), None
            if name == "open_url":
                return inp.open_url(args["url"]), None
            if name == "ask":
                return self._run_ask(args["question"]), None
            if name == "refresh_context":
                time.sleep(0.3)
                self.snapshot = self._refresh()
                return json.dumps(self.snapshot.data), self.snapshot
            if name == "recalibrate":
                quick = bool(args.get("quick"))
                self._recalibrate(quick)
                return "re-centering started" if quick else "calibration started", None
            return f"unknown tool {name}", None
        except ActionError as err:
            return f"error: {err}", None
        except KeyError as err:
            return f"error: missing argument {err}", None

    def _run_ask(self, question: str) -> str:
        answer = self._ask(question)
        if answer is None:
            return "(no answer: the user didn't reply)"
        out: dict = {"answer": answer.text}
        if answer.element is not None:
            x, y = answer.point or (None, None)
            out["looking_at_while_answering"] = describe(answer.element, self.registry, x, y)
        if answer.point is not None:
            for key, info in self.registry.window_info.items():
                if info.bounds.contains(*answer.point):
                    out["window_while_answering"] = key
                    break
        return json.dumps(out)
