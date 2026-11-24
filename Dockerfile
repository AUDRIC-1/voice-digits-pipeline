FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    NUMBA_CACHE_DIR=/tmp/numba \
    STORAGE_DIR=/app/storage \
    PORT=8000

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY ui ./ui
COPY models ./models
COPY data/train ./data/train
COPY data/test ./data/test
COPY data/new_voice ./data/new_voice
COPY data/noise ./data/noise
COPY results ./results

# Run as a normal user (Hugging Face Spaces uses uid 1000)
RUN useradd -m -u 1000 app && mkdir -p /app/storage && chown -R app /app/storage
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').environ['PORT'] + '/health')"
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
