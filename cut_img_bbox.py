import os
import pandas as pd
from PIL import Image

csv_path = 'data/labeled_img.csv'
input_dir = 'data/labeled_img'
output_dir = 'data/cropped_images'

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

for index, row in df.iterrows():
    image_path = os.path.join(input_dir, row['filepath'])

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Coordinates for cropping
            left = int(row['x_min'] * width)
            top = int(row['y_min'] * height)
            right = int((row['x_min'] + row['width']) * width)
            bottom = int((row['y_min'] + row['height']) * height)
            
            # Crop the image
            cropped = img.crop((left, top, right, bottom))
            
            output_path = os.path.join(output_dir, row['filepath'])
            cropped.save(output_path)
            print(f"Saved: {output_path}")
    
    except Exception as e:
        print(f"Error: {row['filepath']}: {e}")
