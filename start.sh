#!/bin/bash

# Render startup script for MediAI Disease Predictor
echo "🚀 Starting MediAI Disease Predictor on Render..."

# Set environment variables
export PYTHONUNBUFFERED=1
export FLASK_ENV=production
export PORT=${PORT:-5000}

echo "📡 Port configuration: $PORT"

# Create required directories
echo "📁 Creating directories..."
mkdir -p uploads logs temp_uploads static/uploads

# Test app startup
echo "🧪 Testing app startup..."
python test_startup.py
if [ $? -ne 0 ]; then
    echo "❌ App startup test failed!"
    exit 1
fi

# Start the application with gunicorn
echo "🌟 Starting gunicorn server on port $PORT..."
echo "🔧 Using configuration: gunicorn.conf.py"

# Start gunicorn with explicit port binding
exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --timeout 300 \
    --graceful-timeout 300 \
    --worker-class sync \
    --max-requests 100 \
    --preload-app \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app