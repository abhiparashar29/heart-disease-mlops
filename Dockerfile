# Heart Disease Risk API - serving image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# 1. Install dependencies first: this layer is cached and only rebuilt
#    when requirements-serve.txt changes -> fast rebuilds after code edits
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# 2. Copy application code and the trained model pipeline
COPY src/ src/
COPY app/ app/
COPY models/ models/

# 3. Run as a non-root user (security best practice)
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# 4. Docker checks the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
