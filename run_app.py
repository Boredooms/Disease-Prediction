#!/usr/bin/env python3
"""
Simple startup script for Render deployment
"""
import os
import sys
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting MediAI Disease Predictor on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)