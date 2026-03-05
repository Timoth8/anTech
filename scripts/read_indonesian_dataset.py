import pandas as pd

df = pd.read_csv("data/processed/indonesian_fake_news.csv")

print(df.head())
print(df.label.value_counts())