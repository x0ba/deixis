"""Spoken replies with Cartesia Sonic, streamed straight to the speakers.

Falls back to macOS `say` when there is no Cartesia key or the request fails,
so the user always hears an answer.
"""

import logging
import queue
import subprocess
import threading
from collections.abc import Callable

import sounddevice as sd
from cartesia import Cartesia

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
MODEL = "sonic-3.6"


class Speaker:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        on_speaking: Callable[[bool], None] = lambda _: None,
    ):
        self._client = Cartesia(api_key=api_key) if api_key else None
        self._voice_id = voice_id
        self._on_speaking = on_speaking
        self._queue: queue.Queue[str] = queue.Queue()
        self._ws = None
        self._interrupt = threading.Event()
        threading.Thread(target=self._run, name="speaker", daemon=True).start()

    def say(self, text: str) -> None:
        if text.strip():
            self._queue.put(text.strip())

    def interrupt(self) -> None:
        self._interrupt.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            self._interrupt.clear()
            self._on_speaking(True)
            try:
                if self._client is None:
                    raise RuntimeError("no Cartesia key")
                self._stream(text)
            except Exception as err:
                log.warning("Cartesia TTS failed (%s); using macOS say", err)
                self._ws = None
                subprocess.run(["say", text], check=False)
            finally:
                if self._queue.empty():
                    self._on_speaking(False)

    def _stream(self, text: str) -> None:
        if self._ws is None:
            self._ws = self._client.tts.websocket_connect().__enter__()
        ctx = self._ws.context(
            model_id=MODEL,
            voice=self._voice_id,
            output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
            language="en",
        )
        ctx.push(text)
        ctx.no_more_inputs()
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as out:
            for response in ctx.receive():
                if self._interrupt.is_set():
                    ctx.cancel()
                    break
                if response.type == "chunk" and response.audio:
                    out.write(response.audio)
                elif response.type == "error":
                    raise RuntimeError(response.message or response.title)
