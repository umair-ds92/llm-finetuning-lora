# Dockerfile for LLM LoRA Inference API

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Ensure src package is importable
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-deployment.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-deployment.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Create directories for models
RUN mkdir -p outputs models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API server
CMD ["python", "src/deployment/serve.py", "--model-path", "/app/models/deployed-model"]
