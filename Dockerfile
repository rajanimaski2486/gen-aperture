FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application
COPY backend/app ./app

# Copy pre-built frontend (from CI/CD)
COPY backend/static ./app/static 2>/dev/null || echo "No static files yet"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENSEARCH_ENDPOINT="http://mmr-test-v1-prod.sstk-search-prod.ct.shuttercloud.org"
ENV ENVIRONMENT="production"

EXPOSE 8000

# Run with 4 workers for 10 RPS capacity
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
