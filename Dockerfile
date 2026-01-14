# Slim Python image
from python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential needed for some pip packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Install asyncpg for Postgres support
    pip install --no-cache-dir asyncpg

# Copy application code
COPY . .

# Create logs and data directories
RUN mkdir -p logs && mkdir -p data

# Run as non-root user (Security Best Practice)
# RUN useradd -m botuser
# USER botuser

# Default command
CMD ["python", "main.py"]
