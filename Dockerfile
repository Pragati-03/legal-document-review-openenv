# Legal Document Review — OpenEnv
# Dockerfile for containerized deployment (HF Spaces + local Docker)
FROM python:3.11-slim

# Metadata
LABEL maintainer="openenv-legal-review"
LABEL org.opencontainers.image.title="Legal Document Review OpenEnv"
LABEL org.opencontainers.image.description="AI agent environment for legal contract review"
LABEL org.opencontainers.image.version="1.0.0"
LABEL space_sdk="docker"
LABEL tags="openenv"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY env/        ./env/
COPY tasks/      ./tasks/
COPY graders/    ./graders/
COPY server.py   .
COPY openenv.yaml .
COPY baseline_inference.py .
COPY . .
# Create __init__.py files for packages
RUN touch env/__init__.py tasks/__init__.py graders/__init__.py

# Expose the API port
# HF Spaces expects port 7860; override with PORT env var
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]