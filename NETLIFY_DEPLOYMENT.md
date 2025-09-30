# MediAI Disease Predictor - Netlify Deployment

## Why Netlify for Full ML Stack?

✅ **Better for heavy dependencies** - Handles TensorFlow, PyTorch, transformers  
✅ **Longer build times** - Up to 45 minutes (good for ML models)  
✅ **Larger deployment size** - Can handle your full ML stack  
✅ **Serverless functions** - Good performance for AI/ML workloads  
✅ **Free tier** - 300 build minutes/month, 125k function calls  

## Deploy to Netlify

### Method 1: GitHub Integration (Recommended)

1. **Connect to Netlify**:
   - Go to [netlify.com](https://netlify.com)
   - Sign up/in with GitHub
   - Click "New site from Git" → GitHub → Select your `Disease-Prediction` repo

2. **Build Settings**:
   - Build command: `pip install -r requirements.txt`
   - Publish directory: `static`
   - Functions directory: `netlify/functions`

3. **Environment Variables**:
   - `PYTHON_VERSION` = `3.10`
   - `FAST_START` = `false` (to enable full ML stack)
   - `PYTHONUNBUFFERED` = `1`

### Method 2: Netlify CLI

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login to Netlify
netlify login

# Deploy from project root
netlify deploy --prod
```

## Full End-to-End Pipeline Available

With full ML libraries, your app will have:

✅ **Stage 1: OCR Processing**
- DirectPrescriptionReader extracts text from prescription images
- Smart medicine name enhancement and correction

✅ **Stage 2: Clinical BERT Analysis**  
- Medical entity extraction (medications, conditions, dosages)
- Clinical context understanding
- Disease pattern recognition

✅ **Stage 3: Disease Prediction**
- Multiple ML models (Random Forest, SVM, Neural Networks, Deep Learning)
- Medicine → Symptoms → Disease mapping
- Confidence scoring and recommendations

✅ **Stage 4: Comprehensive Results**
- Detailed medical analysis
- Disease predictions with confidence scores
- Treatment recommendations

## Full ML Stack Included

Your `requirements.txt` now includes:
- `tensorflow` - Deep learning models
- `torch` + `torchvision` - PyTorch ecosystem  
- `transformers` - Hugging Face Clinical BERT
- `scikit-learn` - Traditional ML algorithms
- `opencv-python-headless` - Image processing
- `pytesseract` - OCR engine
- `nltk` + `textblob` - Natural language processing

## Expected Build Time

- **First deploy**: 15-25 minutes (downloading heavy ML libraries)
- **Subsequent deploys**: 5-10 minutes (cached dependencies)

## Netlify vs Other Platforms

| Feature | Netlify Free | Vercel Free | Render Free |
|---------|--------------|-------------|-------------|
| Max deployment size | 1GB+ | 500MB | No limit |
| Build timeout | 45 min | 45 min | 15 min |
| Function timeout | 30s | 30s | No limit |
| ML library support | ✅ Good | ⚠️ Limited | ✅ Excellent |
| Cold starts | ~2-5s | ~1-2s | ~30s |

## Files Created

- `netlify.toml` - Netlify configuration
- `netlify/functions/app.py` - Serverless function handler
- Updated `requirements.txt` - Full ML stack

## Deploy Now!

Push these changes and connect your repo at [netlify.com/start](https://netlify.com/start)

Your full AI-powered disease prediction system will be live! 🚀🏥