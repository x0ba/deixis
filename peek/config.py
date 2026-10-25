import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "Peek"


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    cartesia_api_key: str
    anthropic_api_key: str
    anthropic_model: str
    voice_id: str
    gaze_source: str
    wake_words: list[str] = field(default_factory=list)
    require_wake_word: bool = True
    send_screenshot: bool = True
    show_highlight: bool = True
    camera_index: int = 0
    noise_gate_db: float = 12.0
    barge_in: bool = True
    conversation_s: float = 15.0
    sounds: bool = True
    wink_click: bool = True
    learn_from_clicks: bool = True
    face_model_path: Path = ROOT / "models" / "face_landmarker.task"
    calibration_path: Path = SUPPORT_DIR / "calibration.npz"


def load_config() -> Config:
    load_dotenv(ROOT / ".env")
    wake = os.environ.get("PEEK_WAKE_WORDS", "peek,hey computer")
    return Config(
        cartesia_api_key=os.environ.get("CARTESIA_API_KEY", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.environ.get("PEEK_MODEL", "claude-sonnet-5-5"),
        voice_id=os.environ.get("PEEK_VOICE_ID", "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"),
        gaze_source=os.environ.get("PEEK_SOURCE", "camera"),
        wake_words=[w.strip().lower() for w in wake.split(",") if w.strip()],
        require_wake_word=_flag("PEEK_REQUIRE_WAKE_WORD", True),
        send_screenshot=_flag("PEEK_SEND_SCREENSHOT", True),
        show_highlight=_flag("PEEK_HIGHLIGHT", True),
        camera_index=int(os.environ.get("PEEK_CAMERA", "0")),
        noise_gate_db=float(os.environ.get("PEEK_NOISE_GATE_DB") or 12),
        barge_in=_flag("PEEK_BARGE_IN", True),
        conversation_s=float(os.environ.get("PEEK_CONVERSATION_S") or 15),
        sounds=_flag("PEEK_SOUNDS", True),
        wink_click=_flag("PEEK_WINK_CLICK", True),
        learn_from_clicks=_flag("PEEK_LEARN_FROM_CLICKS", True),
    )
