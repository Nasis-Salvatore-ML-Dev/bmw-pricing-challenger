FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY src/ ./src/
COPY config/ ./config/
COPY data/models/checkpoints/rand_forest_v1.pkl ./data/models/checkpoints/

# Expose the port (Cloud Run will override with $PORT)
EXPOSE 8080

# Use environment variable for port (Cloud Run sets $PORT)
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]