"""Hands-free listening with Cartesia Ink-2 and its native turn detection.

The microphone streams continuously into the auto-finalize socket. Cartesia
tells us when a turn starts (the moment to sample gaze) and when it ends
(the moment to act). The mic stays live while Peek is speaking so the user
can interrupt; the app strips Peek's own voice from those transcripts. When
`muted` is set the mic sends silence instead, since the API expects a
continuous stream. A noise gate replaces background noise with silence the
same way.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from cartesia import AsyncCartesia

from peek.voice.gate import NoiseGate

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = SAMPLE_RATE // 10
SILENCE = bytes(CHUNK_SAMPLES * 2)


@dataclass
class Turn:
    transcript: str
    started: float
    ended: float


class Listener:
    def __init__(
        self,
        api_key: str,
        keyterms: list[str],
        on_turn_start: Callable[[float], None],
        on_partial: Callable[[str], None],
        on_turn: Callable[[Turn], None],
        on_status: Callable[[str], None],
        noise_gate_db: float = 12.0,
    ):
        self._client = AsyncCartesia(api_key=api_key)
        self._keyterms = keyterms
        self._on_turn_start = on_turn_start
        self._on_partial = on_partial
        self._on_turn = on_turn
        self._on_status = on_status
        self.muted = threading.Event()
        self.paused = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue[bytes] | None = None
        self._stop = threading.Event()
        self._turn_started: float | None = None
        self.level = 0.0
        self._gate = NoiseGate(margin_db=noise_gate_db) if noise_gate_db > 0 else None

    def start(self) -> None:
        threading.Thread(
            target=self._loop.run_until_complete, args=(self._main(),), name="listener", daemon=True
        ).start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def turn_started(self) -> float | None:
        """When the turn in progress began, or None if nobody is talking."""
        return self._turn_started

    def _mic_callback(self, indata, frames, time_info, status) -> None:
        data = bytes(indata)
        if self.muted.is_set() or self.paused.is_set():
            data = SILENCE[: len(data)]
            if self._gate is not None:
                self._gate.reset()
        else:
            self.level = float(np.abs(np.frombuffer(data, dtype=np.int16)).max(initial=0)) / 32768
            if self._gate is not None:
                data = self._gate.process(data)
        if self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, data)

    async def _main(self) -> None:
        self._queue = asyncio.Queue(maxsize=200)
        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=self._mic_callback,
        )
        stream.start()
        try:
            while not self._stop.is_set():
                try:
                    await self._session()
                except Exception as err:
                    log.warning("listener session ended: %s", err)
                    self._on_status(f"Speech connection dropped ({err}); reconnecting")
                    await asyncio.sleep(1.5)
        finally:
            stream.stop()
            stream.close()

    async def _session(self) -> None:
        async with self._client.stt.auto_finalize.websocket(
            encoding="pcm_s16le",
            model="ink-2",
            sample_rate=SAMPLE_RATE,
            keyterm=self._keyterms,
            turn_end_timeout_ms=2500,
        ) as connection:
            log.info("speech connection open")
            self._on_status("listening")
            sender = asyncio.create_task(self._send(connection))
            try:
                async for event in connection:
                    self._handle(event)
            finally:
                sender.cancel()

    async def _send(self, connection) -> None:
        assert self._queue is not None
        while not self._stop.is_set():
            chunk = await self._queue.get()
            await connection.send_raw(chunk)
        await connection.send({"type": "close"})

    def _handle(self, event) -> None:
        kind = event.type
        if kind == "turn.start":
            self._turn_started = time.monotonic()
            self._on_turn_start(self._turn_started)
        elif kind in ("turn.update", "turn.eager_end"):
            self._on_partial(event.transcript or "")
        elif kind == "turn.end":
            started = self._turn_started or time.monotonic()
            self._turn_started = None
            text = (event.transcript or "").strip()
            if text:
                self._on_turn(Turn(text, started, time.monotonic()))
        elif kind == "error":
            raise RuntimeError(getattr(event, "message", "speech error"))
