import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
print("Loading datasets...")
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Train label distribution:\n{train_df['label'].value_counts()}")

# IndoBERT model name
MODEL_NAME = "indobenchmark/indobert-base-p1"  # Alternative: "indolem/indobert-base-uncased"

# Load tokenizer
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create custom dataset class
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
print("\nCreating datasets...")
train_dataset = FakeNewsDataset(
    texts=train_df['text'].values,
    labels=train_df['label'].values,
    tokenizer=tokenizer,
    max_length=512
)

val_dataset = FakeNewsDataset(
    texts=val_df['text'].values,
    labels=val_df['label'].values,
    tokenizer=tokenizer,
    max_length=512
)

# Load model
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    problem_type="single_label_classification"
)

# Define metrics computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results/indobert',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    report_to="none"  # Disable wandb/tensorboard for now
)

# Create Trainer
print("\nInitializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

trainer.train()

# Evaluate on validation set
print("\n" + "="*50)
print("Evaluating on validation set...")
print("="*50 + "\n")

eval_results = trainer.evaluate()
print("\nEvaluation Results:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# Save the model
print("\nSaving model...")
model.save_pretrained("./model/indobert_fake_news")
tokenizer.save_pretrained("./model/indobert_fake_news")

# Get predictions for confusion matrix
predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

# Print confusion matrix
cm = confusion_matrix(labels, preds)
print("\nConfusion Matrix:")
print("                Predicted")
print("              0 (Real)  1 (Fake)")
print(f"Actual 0 (Real)   {cm[0][0]:4d}     {cm[0][1]:4d}")
print(f"Actual 1 (Fake)   {cm[1][0]:4d}     {cm[1][1]:4d}")

print("\nTraining completed successfully!")
print(f"Model saved to: ./model/indobert_fake_news")
