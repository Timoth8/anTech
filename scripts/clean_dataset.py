import pandas as pd

# load dataset
df = pd.read_csv("data/processed/indonesian_fake_news.csv")

print("Original dataset shape:", df.shape)

# remove rows with empty text
df = df.dropna(subset=["content_id"])

# remove duplicates
df = df.drop_duplicates(subset=["content_id"])

# rename column
df = df.rename(columns={"content_id": "text"})

# reset index
df = df.reset_index(drop=True)

print("Cleaned dataset shape:", df.shape)
print(df.head())

# save cleaned dataset
df.to_csv("data/processed/indonesian_fake_news_clean.csv", index=False)