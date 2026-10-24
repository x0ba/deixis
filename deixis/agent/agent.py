"""The planner: Claude decides what the user meant and drives the tools."""

import json
import logging
from collections import deque
from collections.abc import Callable

import anthropic

from deixis.agent.tools import TOOLS, Executor
from deixis.screen.context import Snapshot, image_b64

log = logging.getLogger(__name__)

MAX_STEPS = 8

SYSTEM = """\
You are Deixis, a hands-free assistant on the user's Mac. The user may not be \
able to use a keyboard or mouse. They control the computer by looking at things \
and speaking. A webcam tracks their eyes, so "this", "that", "here" and "it" \
usually mean what they were looking at while they spoke.

Each request comes with a JSON snapshot of the screen:
- `gaze` is where they looked longest while speaking, in global screen points \
(top-left origin). Webcam gaze is imprecise; `typical_error_px` says by how much.
- `looking_at` is the window and element under the gaze, plus `nearby_elements` \
sorted by distance. When the element directly under the gaze doesn't fit the \
request (say they want to type but it's a label), choose the nearest one that \
fits, e.g. the closest editable field.
- `other_fixations_in_order` are other places they looked, in time order. Use \
them for requests like "move this there" or "put that next to this".
- `windows_front_to_back` lists visible windows; `w` ids work with window tools.
- `focused_element` is where keyboard focus is now.
- A screenshot of the area around the gaze may be attached, with a red ring at \
the gaze point. `screenshot.origin` and `points_per_pixel` map image pixels to \
screen points.

How to act:
- Do what was asked with the fewest steps. Don't do extra things.
- If the request is ambiguous and a wrong guess would be destructive (closing, \
sending, deleting), ask a short question instead of acting.
- When asked to write something, write exactly the text they dictated, with \
sensible punctuation and capitalization. If they ask you to compose something \
("reply saying I'll be late"), write it naturally in their voice.
- After you finish, reply with ONE short spoken sentence (it is read aloud), \
e.g. "Done." or "Closed Safari." For questions, answer briefly and plainly. Never \
use markdown, lists or emoji in the reply.
- If you can't do something, say so plainly in one sentence.
"""


class Agent:
    def __init__(self, api_key: str, model: str):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._memory: deque[dict] = deque(maxlen=8)

    def handle(
        self,
        utterance: str,
        snapshot: Snapshot,
        refresh: Callable[[], Snapshot],
        recalibrate: Callable[[], None],
        on_step: Callable[[str], None] = lambda _: None,
    ) -> str:
        executor = Executor(snapshot, refresh, recalibrate)
        content: list[dict] = [
            {"type": "text", "text": f"Screen snapshot:\n{json.dumps(snapshot.data)}"},
        ]
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
        content.append({"type": "text", "text": f'The user said: "{utterance}"'})
        messages = [*self._memory, {"role": "user", "content": content}]

        reply = ""
        for _ in range(MAX_STEPS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            texts = [b.text for b in response.content if b.type == "text"]
            if texts:
                reply = " ".join(t.strip() for t in texts if t.strip())
            calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not calls:
                break
            results = []
            for call in calls:
                on_step(call.name.replace("_", " "))
                result, _ = executor.run(call.name, dict(call.input))
                log.info("tool %s %s -> %s", call.name, call.input, result[:200])
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": result})
            messages.append({"role": "user", "content": results})
        else:
            reply = reply or "That took too many steps, so I stopped."

        self._memory.append({"role": "user", "content": f"(earlier) The user said: {utterance}"})
        self._memory.append({"role": "assistant", "content": reply or "Done."})
        return reply or "Done."
