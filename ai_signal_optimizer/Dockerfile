# ── Build stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="Your Name <your@email.com>"
LABEL description="AI Signal Strength Optimization System"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create required directories
RUN mkdir -p data models outputs

# Expose Flask port
EXPOSE 5000

# Default command: run full pipeline then start server
CMD ["python", "main.py"]
