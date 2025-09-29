#!/usr/bin/env python3
"""
Simple startup script for Railway deployment
"""
import os
import sys
from app import app

if __name__ == "__main__":
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting MediAI Disease Predictor on Railway")
    print(f"🌐 Port: {port}")
    print(f"🏥 AI Models: OCR → Clinical BERT → Disease Prediction")
    
    # Railway deployment
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)