# MediAI Disease Predictor - Ultra Lightweight for Railway
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install Python dependencies in one layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy only essential application files
COPY app.py .
COPY run_app.py .
COPY ocr/ ./ocr/
COPY nlp/ ./nlp/
COPY disease_predictor/ ./disease_predictor/
COPY templates/ ./templates/
COPY static/ ./static/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Expose port
EXPOSE $PORT

# Simple startup
CMD python run_app.py