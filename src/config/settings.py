from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")


def get_app_settings() -> AppSettings:
    return AppSettings()
