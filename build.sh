#!/bin/bash
# Render Build Script for MediAI Disease Predictor

echo "🚀 Starting MediAI Disease Predictor build..."

# Install system dependencies for OCR and image processing
echo "📦 Installing system dependencies..."
apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Upgrade pip and install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spacy language model
echo "🧠 Downloading NLP models..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p static/uploads temp_uploads logs uploads

# Set proper permissions for startup script
echo "🔧 Setting permissions..."
chmod +x start.sh

echo "✅ Build complete! Ready to start application."
echo "🌟 Application will be started with gunicorn on port $PORT"