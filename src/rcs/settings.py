"""Validated, fail-fast application settings."""

from pathlib import Path

from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables prefixed with ``RCS_``."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RCS_", extra="ignore")

    api_key: str | None = None
    model_path: Path | None = None
    max_upload_bytes: PositiveInt = Field(default=10 * 1024 * 1024)
    max_image_pixels: PositiveInt = Field(default=40_000_000)
    queue_capacity: PositiveInt = Field(default=128)
