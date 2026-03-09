import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Load the model and tokenizer
MODEL_PATH = "./model/indobert_fake_news"

print("Loading IndoBERT fake news classifier...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Set to evaluation mode
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_fake_news(text, show_confidence=True):
    """
    Predict if Indonesian news text is real or fake
    
    Args:
        text: Indonesian news text to classify
        show_confidence: Whether to show prediction confidence
    
    Returns:
        prediction: 'REAL' or 'FAKE'
        confidence: probability score
    """
    # Tokenize
    inputs = tokenizer(
        text,
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
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    
    # Map to label
    label = "FAKE" if predicted_class == 1 else "REAL"
    
    if show_confidence:
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Real: {probs[0][0]:.2%} | Fake: {probs[0][1]:.2%}")
    
    return label, confidence

# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("IndoBERT Indonesian Fake News Classifier")
    print("="*60)
    
    # Test examples
    test_texts = [
        "Presiden Indonesia mengumumkan kebijakan baru untuk meningkatkan ekonomi nasional melalui reformasi struktural yang komprehensif.",
        "BREAKING NEWS! Alien ditemukan di Jakarta dan sudah bertemu dengan presiden! Ini bukti fotonya yang mengejutkan!",
        "Menteri Kesehatan melaporkan penurunan kasus COVID-19 di berbagai provinsi setelah program vaksinasi massal."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {text[:100]}...")
        predict_fake_news(text)
    
    print("\n" + "="*60)
    
    # Interactive mode
    print("\nInteractive Mode - Enter Indonesian news text to classify")
    print("(Type 'quit' to exit)\n")
    
    while True:
        user_input = input("\nEnter news text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input:
            predict_fake_news(user_input)
        else:
            print("Please enter some text.")