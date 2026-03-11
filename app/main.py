from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from app.scraper import scrape_news_article, NewsScraperException, validate_indonesian_content

app = FastAPI(
    title="Indonesian Fake News Detector API",
    description="API for detecting fake news in Indonesian language using IndoBERT",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
MODEL_PATH = "./model/indobert_fake_news"
print("Loading IndoBERT model...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

# Request/Response models
class NewsText(BaseModel):
    text: str

class NewsURL(BaseModel):
    url: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    text_length: int
    scraped_text: str = None  # Optional: include scraped text
    scrape_method: str = None  # Optional: scraping method used

@app.get("/")
def root():
    return {
        "message": "Indonesian Fake News Detector API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict (POST with text)",
            "predict_url": "/predict-url (POST with URL)",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if model is not None else "N/A"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fake_news(news: NewsText):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not news.text or len(news.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short. Minimum 10 characters required.")
    
    try:
        # Tokenize
        inputs = tokenizer(
            news.text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()
        
        # Get probabilities
        prob_real = probs[0][0].item()
        prob_fake = probs[0][1].item()
        
        return PredictionResponse(
            prediction="FAKE" if predicted_class == 1 else "REAL",
            confidence=confidence,
            probabilities={
                "real": round(prob_real, 4),
                "fake": round(prob_fake, 4)
            },
            text_length=len(news.text)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-url", response_model=PredictionResponse)
async def predict_from_url(news_url: NewsURL):
    """
    Predict fake news from a URL by scraping the article content
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Scrape article text from URL
        scrape_result = scrape_news_article(news_url.url)
        article_text = scrape_result['text']
        scrape_method = scrape_result['method']
        
        # Validate minimum length
        if len(article_text) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Extracted text too short. The URL may not contain a valid news article."
            )
        
        # Check if content appears to be Indonesian
        if not validate_indonesian_content(article_text):
            raise HTTPException(
                status_code=400,
                detail="The extracted text does not appear to be in Indonesian language."
            )
        
        # Tokenize
        inputs = tokenizer(
            article_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()
        
        # Get probabilities
        prob_real = probs[0][0].item()
        prob_fake = probs[0][1].item()
        
        return PredictionResponse(
            prediction="FAKE" if predicted_class == 1 else "REAL",
            confidence=confidence,
            probabilities={
                "real": round(prob_real, 4),
                "fake": round(prob_fake, 4)
            },
            text_length=len(article_text),
            scraped_text=article_text[:500] + "..." if len(article_text) > 500 else article_text,  # Preview
            scrape_method=scrape_method
        )
    
    except NewsScraperException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")