FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libjpeg62-turbo libopenjp2-7 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/image-processing/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "impulse.handlers.image_handler"]
