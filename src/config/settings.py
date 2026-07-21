from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    data_dir: Path = Field(default=Path("./datasets"), alias="DATA_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def get_app_settings() -> AppSettings:
    return AppSettings()
