# AI Nutrition - Docker Container
# Single-stage build for ML application compatibility

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OCR, OpenCV, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer - only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY src/ ./src/
COPY static/ ./static/
COPY data/ ./data/
COPY models/ ./models/

# Create directories for runtime
RUN mkdir -p /app/logs

# Environment configuration
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PORT=8081
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8081

# Run application
CMD ["python", "src/main.py"]
