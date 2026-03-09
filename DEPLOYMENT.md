# Indonesian Fake News Detector - Deployment Guide

## 🚀 Quick Start (Local Testing)

### 1. Start the Backend API

```bash
# Make sure you're in the project directory
cd anTech

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 2. Open the Frontend

```bash
# Open in browser (Windows)
start frontend/index.html

# Or navigate to:
# file:///C:/Users/User%20NB/Desktop/anTech/frontend/index.html
```

---

## 🌐 Production Deployment

### Option A: Vercel (Frontend) + Render/Railway (Backend)

#### **Step 1: Deploy Backend to Render**

1. **Create account at [render.com](https://render.com)**

2. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add fake news detector"
   git push origin main
   ```

3. **On Render Dashboard:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name:** `indobert-fake-news-api`
     - **Environment:** `Python 3`
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
     - **Instance Type:** Free (or Starter for faster performance)

4. **Wait for deployment** (5-10 minutes)

5. **Get your API URL:** `https://indobert-fake-news-api.onrender.com`

#### **Step 2: Update Frontend API URL**

Edit `frontend/script.js` line 2:
```javascript
const API_URL = 'https://indobert-fake-news-api.onrender.com';
```

#### **Step 3: Deploy Frontend to Vercel**

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd anTech
vercel --prod
```

Or via Vercel Dashboard:
1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repository
3. Vercel will auto-detect and deploy

Your app will be live at: `https://your-app.vercel.app`

---

### Option B: Hugging Face Spaces (All-in-One)

1. **Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)**
   - Choose "Gradio" or "Streamlit" template

2. **Create a Gradio interface** (simpler than FastAPI + frontend):

```python
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./model/indobert_fake_news"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    
    real_prob = probs[0][0].item()
    fake_prob = probs[0][1].item()
    
    return {
        "Real": real_prob,
        "Fake": fake_prob
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Masukkan berita dalam Bahasa Indonesia..."),
    outputs=gr.Label(num_top_classes=2),
    title="🔍 Indonesian Fake News Detector",
    description="Detect fake news in Indonesian language using IndoBERT",
    examples=[
        ["Pemerintah Indonesia mengumumkan kebijakan baru..."],
        ["HEBOH! Presiden tertangkap kamera sedang bertemu dengan alien!"]
    ]
)

demo.launch()
```

3. **Push to Hugging Face Spaces** - it will auto-deploy!

---

## 🔧 Environment Variables (Production)

For Render/Railway, add these environment variables:

```bash
PYTHON_VERSION=3.11
PORT=8000
```

---

## 📊 Testing Your Deployment

### Test Backend API:
```bash
curl -X POST "https://your-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Berita palsu tentang alien di Jakarta"}'
```

### Test Frontend:
Open `https://your-app.vercel.app` in browser

---

## 🐛 Troubleshooting

### CORS Errors:
Update `app/main.py` line 17 to include your frontend domain:
```python
allow_origins=["https://your-app.vercel.app"]
```

### Model Too Large for Deployment:
Consider using:
- Render Starter plan ($7/month) - supports larger apps
- Hugging Face Spaces (free, optimized for ML models)
- Railway ($5/month compute)

### API Timeout:
First request may be slow (~30s) as model loads. Subsequent requests are fast.

---

## 💡 Recommended Setup

For **demo/portfolio**: 
- **Hugging Face Spaces** (easiest, free, designed for ML)

For **production app**:
- **Backend**: Render Starter ($7/mo) or Railway
- **Frontend**: Vercel (free)

---

## 📝 Notes

- Model size: ~500MB (may be too large for free tiers)
- First prediction takes longer (model loading)
- Consider model quantization for smaller size
- Add rate limiting for production use
