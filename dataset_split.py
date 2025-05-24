import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset to split
df = pd.read_csv("data/labeled_img/labeled_img.csv")

class_counts = df['class'].value_counts()

if (class_counts >= 3).all():
    stratify_col = df['class']
else:
    stratify_col = None

# Split (70% train, 15% val, 15% test)
df_train, df_temp = train_test_split(df, test_size=0.3, stratify=stratify_col, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=stratify_col[df_temp.index] if stratify_col is not None else None, random_state=42)

# Save the splits to CSV files
df_train.to_csv("data/train.csv", index=False)
df_val.to_csv("data/val.csv", index=False)
df_test.to_csv("data/test.csv", index=False)
