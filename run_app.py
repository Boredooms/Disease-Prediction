#!/usr/bin/env python3
"""
Simple startup script for Render deployment
"""
import os
import sys
from app import app

if __name__ == "__main__":
    # Render provides PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting MediAI Disease Predictor on Render")
    print(f"🌐 Port: {port}")
    print(f"🏥 AI Models: OCR → Clinical BERT → Disease Prediction")
    
    # CRITICAL: Render deployment needs explicit host and port
    print(f"🔗 Binding to 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)