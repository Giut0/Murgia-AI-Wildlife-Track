import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def split_folds(df, base_dir, kf):    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for fold, (train_index, test_index) in enumerate(kf.split(df, df['class']), 1):
        fold_dir = os.path.join(base_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        
        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

# Setup
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Load dataset
animals_dataset = pd.read_csv('data/labeled_img/labeled_img.csv')

# Accorpa le classi con meno di n_splits campioni in 'other'
min_samples = n_splits
class_counts = animals_dataset['class'].value_counts()

def map_class(x):
    if class_counts[x] < min_samples:
        return 'other'
    else:
        return x

animals_dataset['class'] = animals_dataset['class'].map(map_class)

print("Classi rare accorpate in 'other'.")

split_folds(animals_dataset, 'data/fold_split', kf)