import pandas as pd
import os
from PIL import Image
import torch


class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, transform=None, crop_bbox=False, label_map=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.crop_bbox = crop_bbox
        self.label_map = label_map or self._create_label_map()

    def _create_label_map(self):
        classes = sorted(self.df['class'].unique())
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filepath'])

        # Carica immagine
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Bounding box (relativo -> assoluto)
        if self.crop_bbox:
            x_min = int(row['x_min'] * w)
            y_min = int(row['y_min'] * h)
            x_max = int((row['x_min'] + row['width']) * w)
            y_max = int((row['y_min'] + row['height']) * h)
            image = image.crop((x_min, y_min, x_max, y_max))

        if self.transform:
            image = self.transform(image)

        label = self.label_map[row['class']]
        return image, label
    

# Training function
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Validation function
def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc