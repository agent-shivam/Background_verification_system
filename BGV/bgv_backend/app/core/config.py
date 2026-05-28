"""
app/core/config.py
──────────────────
Centralised application configuration loaded from environment variables / .env
All other modules import `settings` from here — never read env vars directly.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = Field(default="BGV System")
    app_version: str = Field(default="2.0.0")
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)

    # ── Server ─────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # ── Upload ─────────────────────────────────────────────────────────────────
    upload_dir: Path = Field(default=Path("app/uploads"))
    max_file_size_mb: int = Field(default=20)
    allowed_extensions: str = Field(default="pdf,jpg,jpeg,png,tiff")

    # ── Poppler ────────────────────────────────────────────────────────────────
    poppler_path: str = Field(default=r"C:/poppler/Library/bin")

    # ── OCR ────────────────────────────────────────────────────────────────────
    ocr_lang: str = Field(default="en")
    ocr_use_gpu: bool = Field(default=False)

    # ── Risk Scoring ───────────────────────────────────────────────────────────
    risk_low_threshold: int = Field(default=30)
    risk_medium_threshold: int = Field(default=60)

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = Field(default="DEBUG")
    log_file: str = Field(default="logs/bgv.log")

    # ── VLM (OpenRouter) ───────────────────────────────────────────────────────
    vlm_api_key: str = Field(default="")
    vlm_timeout: int = Field(default=60)
    openrouter_site_url: str = Field(default="http://localhost:8000")
    openrouter_app_name: str = Field(default="BGV System")

    @property
    def allowed_ext_list(self) -> list[str]:
        return [e.strip().lower() for e in self.allowed_extensions.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


# Singleton — import this everywhere
settings = Settings()

# Ensure required directories exist at startup
settings.upload_dir.mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# ── Inject VLM keys into os.environ ──────────────────────────────────────────
# claude_vlm.py reads os.environ directly; pydantic-settings does NOT populate
# os.environ automatically, so we bridge that here.
import os as _os

_os.environ.setdefault("VLM_API_KEY",            settings.vlm_api_key)
_os.environ.setdefault("VLM_TIMEOUT",            str(settings.vlm_timeout))

