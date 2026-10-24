import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "Deixis"


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
    show_gaze_dot: bool = True
    camera_index: int = 0
    face_model_path: Path = ROOT / "models" / "face_landmarker.task"
    calibration_path: Path = SUPPORT_DIR / "calibration.npz"


def load_config() -> Config:
    load_dotenv(ROOT / ".env")
    wake = os.environ.get("DEIXIS_WAKE_WORDS", "deixis,dixie,dexis,daxis,hey computer")
    return Config(
        cartesia_api_key=os.environ.get("CARTESIA_API_KEY", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.environ.get("DEIXIS_MODEL", "claude-sonnet-5-5"),
        voice_id=os.environ.get("DEIXIS_VOICE_ID", "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"),
        gaze_source=os.environ.get("DEIXIS_GAZE", "camera"),
        wake_words=[w.strip().lower() for w in wake.split(",") if w.strip()],
        require_wake_word=_flag("DEIXIS_REQUIRE_WAKE_WORD", True),
        send_screenshot=_flag("DEIXIS_SEND_SCREENSHOT", True),
        show_gaze_dot=_flag("DEIXIS_SHOW_GAZE", True),
        camera_index=int(os.environ.get("DEIXIS_CAMERA", "0")),
    )
