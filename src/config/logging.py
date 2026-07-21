import logging

from dotenv import load_dotenv

from src.config.settings import get_app_settings


def configure_logging(level: int | str | None = None) -> None:
    """Set up timestamped console logging. Defaults to the LOG_LEVEL env var (INFO if unset).
    force=True so it wins even if a library (e.g. uvicorn) already attached its own handlers
    to the root logger first."""
    if level is None:
        load_dotenv()  # standalone scripts don't otherwise load .env like the fastapi CLI does
        level = get_app_settings().log_level
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
