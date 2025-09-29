# Railway Deployment Guide for MediAI Disease Predictor

## 🚅 Quick Railway Deployment

### Step 1: Connect to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up/Login with GitHub
3. Click "Deploy from GitHub repo"
4. Select your `Disease-Prediction` repository

### Step 2: Configure Environment Variables
In Railway dashboard, add these environment variables:
```
PYTHONUNBUFFERED=1
FLASK_ENV=production
```

### Step 3: Deploy
Railway will automatically:
- ✅ Detect Python project
- ✅ Install system dependencies (Tesseract OCR)
- ✅ Install Python packages
- ✅ Download NLP models
- ✅ Start the application

### Step 4: Access Your App
- Railway provides a custom domain: `your-app.railway.app`
- Health check: `https://your-app.railway.app/health`
- Main app: `https://your-app.railway.app/`

## 🔧 Railway Configuration Files

- `railway.json` - Railway deployment config
- `nixpacks.toml` - Build configuration 
- `run_app.py` - Startup script
- `Procfile` - Process definition (fallback)

## 🎯 Why Railway for MediAI?

✅ **Better AI/ML Support**: Optimized for ML workloads  
✅ **Faster Cold Starts**: Better than Render for ML apps  
✅ **More Memory**: Better handling of large AI models  
✅ **Auto-scaling**: Handles traffic spikes well  
✅ **Simple Config**: Minimal setup required  

## 📊 Expected Deploy Time
- Build: ~3-5 minutes (includes ML packages)
- Start: ~30-60 seconds (AI model loading)
- Total: ~5-6 minutes first deployment

Your **MediAI Disease Predictor** is Railway-ready! 🌟