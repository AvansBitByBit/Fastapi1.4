# Enhanced Dockerfile for Waste Detection API v2.0.0
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirementsgoat.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirementsgoat.txt

# Copy application files
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port (using port 80 for Azure compatibility)
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:80/health')" || exit 1

# Run the enhanced application
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "80"]