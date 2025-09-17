# Dockerfile.prod - Production build for self-contained chromadb-vectorstore
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (self-contained)
COPY ./app ./

# Create necessary directories
RUN mkdir -p /app/chroma_data && chmod 755 /app/chroma_data

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    PORT=8088 \
    CHROMA_PERSIST_DIRECTORY=/app/chroma_data

EXPOSE 8088

# Run with uvicorn in production mode
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1