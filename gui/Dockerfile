FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1

ARG APT_PACKAGES="libgl1 libglib2.0-0"

RUN apt-get update \
  && apt-get install -y --no-install-recommends ${APT_PACKAGES} \
  && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY pyproject.toml .
RUN uv sync

COPY --chown=app:app . .

RUN mkdir -p /app/launches \
  && chown app:app /app/launches

USER app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "600", "app:app"]
