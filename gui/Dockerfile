FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1

# Extra OS packages needed by your tasks (OpenCV, Tesseract, poppler, ...).
# Override at build time, e.g.:
#   docker compose build --build-arg APT_PACKAGES="tesseract-ocr poppler-utils"
ARG APT_PACKAGES="libgl1 libglib2.0-0"
RUN apt-get update \
  && apt-get install -y --no-install-recommends ${APT_PACKAGES} \
  && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Run as a non-root user.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY pyproject.toml .
RUN uv sync

COPY --chown=app:app . .
RUN chmod +x /app/docker-entrypoint.sh \
  && mkdir -p /app/launches \
  && chown -R app:app /app

USER app

EXPOSE 5000

ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default role: the web UI. The worker overrides this in docker-compose.yml.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "600", "app:app"]
