import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/indonesian_fake_news_clean.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

print("Train size:", train_df.shape)
print("Validation size:", val_df.shape)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)