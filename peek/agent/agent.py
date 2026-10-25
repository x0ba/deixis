"""The planner: Claude decides what the user meant and drives the tools."""

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import anthropic

from peek.agent.tools import TOOLS, Answer, Executor
from peek.screen.context import Snapshot, image_b64

log = logging.getLogger(__name__)

MAX_STEPS = 10
MEMORY_S = 600.0
SILENT = "[silent]"

SYSTEM = f"""\
You are Peek, a hands-free assistant on the user's Mac. The user may not be \
able to use a keyboard or mouse. They control the computer by looking at things \
and talking to you, the way they'd ask a friend sitting next to them for help. \
A webcam tracks their eyes, so "this", "that", "here" and "it" usually mean what \
they were looking at while they spoke.

Each request comes with a JSON snapshot of the screen:
- `gaze` is where they looked longest while speaking, in global screen points \
(top-left origin). Webcam gaze is imprecise; `typical_error_px` says by how much.
- `gaze.highlighted`, when present, is the element outlined on screen while they \
spoke. The user could see the outline around it, so it is the best guess for \
"this" unless it clearly doesn't fit the request.
- `looking_at` is the window and element under the gaze, plus `nearby_elements` \
sorted by distance. When the element directly under the gaze doesn't fit the \
request (say they want to type but it's a label), choose the nearest one that \
fits, e.g. the closest editable field.
- `looking_at.item`, when present, is the post, message or row that what they're \
looking at belongs to, with its buttons, links and fields in reading order in \
`controls`. For requests about the whole thing ("like this", "reply to this post", \
"delete this message"), choose from `item.controls` by label, not a similar control \
in `nearby_elements`, which may belong to the item above or below. Labels often \
show state: a button labelled "Liked" or "Unlike" means it's already liked, and \
pressing it would undo that.
- `other_fixations_in_order` are other places they looked, in time order. Use \
them for requests like "move this there" or "put that next to this".
- `windows_front_to_back` lists visible windows; `w` ids work with window tools.
- `focused_element` is where keyboard focus is now.
- A screenshot of the area around the gaze may be attached, with a red ring at \
the gaze point. `screenshot.origin` and `points_per_pixel` map image pixels to \
screen points.

This is a spoken conversation:
- Everything you write is read aloud. Talk like a helpful person: short, warm, \
plain. No markdown, lists or emoji.
- Once they've said your name, they can keep talking without saying it again, so \
a request may follow on from earlier ones ("now the other one", "undo that", \
"a bit more"). Use the conversation so far to work out what they mean.
- When you need something from them to go on (which one, a missing detail, a yes \
before closing, sending or deleting something that's ambiguous), use the `ask` \
tool. It speaks your question and waits for their answer, so you can finish the \
job in the same turn. Don't end your turn on a question you need answered.
- Text you write alongside tool calls is spoken right away. For a request that \
needs several steps or a slow lookup, it's fine to say a few words first ("Let \
me find that."). Don't narrate simple one-step actions.
- When you're done, reply in one short sentence, e.g. "Done." or "Closed Safari." \
For questions, answer briefly and plainly. If you can't do something, say so \
plainly and, if there's a way forward, offer it.
- If a request came without your name and it clearly isn't meant for you (they're \
talking to someone else, or thinking out loud), reply with exactly {SILENT} and do \
nothing else.
- If they interrupted you with a new request before you finished the last one, \
decide whether it replaces the old one (a correction like "no, the other one") or \
adds to it, and finish whatever still applies.

How to act:
- Do what was asked with the fewest steps. Don't do extra things.
- When asked to write something, write exactly the text they dictated, with \
sensible punctuation and capitalization. If they ask you to compose something \
("reply saying I'll be late"), write it naturally in their voice.
"""


@dataclass
class Exchange:
    at: float
    said: str
    actions: list[str] = field(default_factory=list)
    reply: str = ""
    interrupted: str = ""

    def describe(self, now: float) -> str:
        line = f'- {round(now - self.at)} s ago the user said "{self.said}".'
        line += f" You did: {'; '.join(self.actions)}." if self.actions else ""
        if self.interrupted:
            line += f" {self.interrupted}"
        elif self.reply:
            line += f' You said: "{self.reply}"'
        return line


class Agent:
    def __init__(self, api_key: str, model: str):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._history: deque[Exchange] = deque(maxlen=8)

    def forget(self) -> None:
        self._history.clear()

    def _conversation(self, now: float) -> str | None:
        while self._history and now - self._history[0].at > MEMORY_S:
            self._history.popleft()
        if not self._history:
            return None
        lines = [e.describe(now) for e in self._history]
        return "The conversation so far, oldest first (ids in it are from old snapshots):\n" + (
            "\n".join(lines)
        )

    def handle(
        self,
        utterance: str,
        snapshot: Snapshot,
        refresh: Callable[[], Snapshot],
        recalibrate: Callable[[str], None],
        *,
        said_name: bool = True,
        ask: Callable[[str], Answer | None] = lambda _: None,
        say: Callable[[str], None] = lambda _: None,
        on_step: Callable[[str], None] = lambda _: None,
        cancelled: Callable[[], str | None] = lambda: None,
    ) -> str | None:
        """The reply to speak ("" if there's nothing more to say, SILENT if the
        utterance wasn't for Peek), or None if it was cancelled.

        `cancelled` is polled between steps; it returns why the request was
        cancelled, which is remembered for the next one."""
        executor = Executor(snapshot, refresh, recalibrate, ask)
        exchange = Exchange(time.monotonic(), utterance)
        content: list[dict] = []
        earlier = self._conversation(exchange.at)
        if earlier:
            content.append({"type": "text", "text": earlier})
        content.append({"type": "text", "text": f"Screen snapshot:\n{json.dumps(snapshot.data)}"})
        if snapshot.image_png:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64(snapshot.image_png),
                    },
                }
            )
        how = (
            "They said your name first."
            if said_name
            else "They didn't say your name; it came during your conversation."
        )
        content.append({"type": "text", "text": f'The user said: "{utterance}" ({how})'})
        messages = [{"role": "user", "content": content}]

        def stopped() -> bool:
            reason = cancelled()
            if reason:
                exchange.interrupted = reason
                self._history.append(exchange)
            return bool(reason)

        spoke = False
        reply = ""
        for _ in range(MAX_STEPS):
            if stopped():
                return None
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                messages=messages,
            )
            if stopped():
                return None
            messages.append({"role": "assistant", "content": response.content})
            text = " ".join(b.text.strip() for b in response.content if b.type == "text").strip()
            calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not calls:
                reply = text
                break
            if text and text != SILENT:
                say(text)
                spoke = True
            results = []
            for call in calls:
                if stopped():
                    return None
                on_step(call.name.replace("_", " "))
                result, _ = executor.run(call.name, dict(call.input))
                log.info("tool %s %s -> %s", call.name, call.input, result[:200])
                exchange.actions.append(f"{call.name} {json.dumps(call.input)} -> {result[:120]}")
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": result})
            messages.append({"role": "user", "content": results})
        else:
            reply = "That took too many steps, so I stopped."

        if stopped():
            return None
        if reply == SILENT and not exchange.actions:
            return SILENT
        reply = reply.replace(SILENT, "").strip() or ("" if spoke else "Done.")
        exchange.reply = reply
        self._history.append(exchange)
        return reply
