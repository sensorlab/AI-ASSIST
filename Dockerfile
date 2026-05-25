FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN uv sync --frozen --no-dev

COPY src ./src

RUN uv sync --frozen --no-dev

EXPOSE 8000

#CMD ["uv", "run", "fastapi", "", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["uv", "run", "fastapi", "run", "src/api/main.py", "--port", "8000", "--workers", "4"]
