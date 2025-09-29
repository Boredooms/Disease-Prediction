#!/bin/bash#!/bin/bash

# Render build script for MediAI Disease Predictor# Render Build Script for MediAI Disease Predictor



echo "🚀 Starting Render build for MediAI Disease Predictor..."echo "🚀 Starting MediAI Disease Predictor build..."



# Update pip# Install system dependencies for OCR and image processing

pip install --upgrade pipecho "📦 Installing system dependencies..."

apt-get update && apt-get install -y \

# Install Python dependencies    tesseract-ocr \

echo "📦 Installing Python packages..."    tesseract-ocr-eng \

pip install -r requirements.txt    libglib2.0-0 \

    libsm6 \

# Download NLTK data    libxext6 \

echo "📚 Downloading NLTK data..."    libxrender-dev \

python -c "    libgomp1 \

import nltk    libgtk-3-0 \

nltk.download('punkt')    libavcodec-dev \

nltk.download('stopwords')    libavformat-dev \

nltk.download('wordnet')    libswscale-dev

print('✅ NLTK data downloaded')

"# Upgrade pip and install Python dependencies

echo "🐍 Installing Python dependencies..."

echo "✅ Build completed successfully!"pip install --upgrade pip
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