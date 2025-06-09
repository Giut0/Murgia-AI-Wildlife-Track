from training_utils import augment_minority_classes
from torchvision import transforms
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the augmentation transformations for minority classes
minority_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

df = pd.read_csv("data/labeled_img.csv")

augmented_df = augment_minority_classes(
    df=df,
    image_dir="data/cropped_images/",
    output_dir="data/labeled_img_aug/",
    transform_fn=minority_augmentation,
    min_samples=50,
    save_csv_path="data/labeled_img_augmented.csv",
    device=device
)