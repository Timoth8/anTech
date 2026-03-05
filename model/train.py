import pandas as pd
df = pd.read_csv('data/WELFake_Dataset.csv')

print("Dataset shape:", df.shape)
print(df.head())
print(df["label"].value_counts()) 