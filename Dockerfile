# Multi-stage build for smaller image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (caching layer)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ===== Final stage =====
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Add .local/bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create directories
RUN mkdir -p logs data/backtest_cache config monitoring

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sqlite3; conn = sqlite3.connect('data/trading_data.db'); conn.close()" || exit 1

# Run bot
CMD ["python", "main_integrated.py"]
