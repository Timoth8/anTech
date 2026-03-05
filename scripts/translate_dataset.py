import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

# load dataset
df = pd.read_csv("data/raw/WELFake_Dataset.csv")

print("Dataset loaded:", df.shape)

# combine title + text
df["content"] = df["title"].fillna('') + " " + df["text"].fillna('')

translator = GoogleTranslator(source="en", target="id")

# Translate more samples for better model training
# Note: 10,000 samples will take ~15-30 minutes depending on internet speed
NUM_SAMPLES = 10000
print(f"\n🔄 Starting translation of {NUM_SAMPLES} samples...")
print("⏱️  This will take approximately 15-30 minutes. Please be patient.\n")

translated_texts = []

for text in tqdm(df["content"][:NUM_SAMPLES], desc="Translating"):
    try:
        translated = translator.translate(text)
    except Exception as e:
        # If translation fails, keep empty string
        translated = ""
    translated_texts.append(translated)

print(f"\n✅ Translation complete! Translated {len(translated_texts)} samples")

df_small = df[:NUM_SAMPLES].copy()
df_small["content_id"] = translated_texts

df_small = df_small[["content_id", "label"]]

print(f"💾 Saving to data/processed/indonesian_fake_news.csv...")
df_small.to_csv("data/processed/indonesian_fake_news.csv", index=False)
print("✅ Dataset saved successfully!")

print("Translation completed!")