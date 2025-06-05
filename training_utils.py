import pandas as pd
import os
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, fbeta_score, precision_score, recall_score, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader 

class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_dir, transform=None, crop_bbox=False, label_map=None):
        self.df = dataset
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

from PIL import Image
import os
from collections import Counter
import torch
from torchvision.transforms import ToTensor, ToPILImage

def augment_minority_classes(df, image_dir, output_dir, transform_fn, min_samples=50, save_csv_path=None, device=None):
    os.makedirs(output_dir, exist_ok=True)
    class_counts = Counter(df['class'])

    augmented_rows = []

    to_tensor = ToTensor()
    to_pil = ToPILImage()

    for cls in class_counts:
        count = class_counts[cls]
        if count >= min_samples:
            continue

        needed = min_samples - count
        class_df = df[df['class'] == cls]

        for i in range(needed):
            row = class_df.iloc[i % len(class_df)]
            img_path = os.path.join(image_dir, row['filepath'])

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error {img_path}: {e}")
                continue

            img_tensor = to_tensor(image).unsqueeze(0).to(device)

            with torch.no_grad():
                aug_tensor = transform_fn(img_tensor) 

            aug_tensor_cpu = aug_tensor.squeeze(0).cpu()
            aug_image = to_pil(aug_tensor_cpu)

            # Save the augmented image
            filename, ext = os.path.splitext(os.path.basename(row['filepath']))
            new_filename = f"{filename}_aug_{i}{ext}"
            new_path = os.path.join(output_dir, new_filename)
            aug_image.save(new_path)
            print(f"Saved: {new_path}")

            new_row = row.copy()
            new_row['filepath'] = os.path.relpath(new_path, output_dir)
            augmented_rows.append(new_row)

    augmented_df = pd.DataFrame(augmented_rows)
    full_df = pd.concat([df, augmented_df], ignore_index=True)

    if save_csv_path:
        full_df.to_csv(save_csv_path, index=False)

    return full_df


def extract_features(model, dataloader, device):
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            out = model(images)
            features.append(out.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def cross_validation(base_dir, model, k, device, transform, label_map, model_feat=None):

    results = []

    for fold_num in range(1, k+1):
        fold_dir = os.path.join(base_dir, f"fold_{fold_num}")
        
        # Load current folder datasets
        train_path = os.path.join(fold_dir, "train.csv")
        test_path = os.path.join(fold_dir, "test.csv")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        # Dataset and DataLoader
        train_ds = AnimalDataset(train_data, "data/labeled_img_aug", transform=transform, label_map=label_map, crop_bbox=False)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_ds = AnimalDataset(test_data, "data/labeled_img_aug", transform=transform, label_map=label_map, crop_bbox=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)  

        X_train_validation, y_train_validation = extract_features(model_feat, train_loader, device)
        X_test_validation, y_test_validation = extract_features(model_feat, test_loader, device)
        
        # Learning 
        model.fit(X_train_validation, y_train_validation)
        
        # Get metrics
        val_predictions = model.predict(X_test_validation)
        val_accuracy = accuracy_score(y_test_validation, val_predictions)
        precision = precision_score(y_test_validation, val_predictions, zero_division=0, average=None)
        recall = recall_score(y_test_validation, val_predictions, zero_division=0, average=None)
        f1score = fbeta_score(y_test_validation, val_predictions, beta=1, zero_division=0, average=None)
        conf = confusion_matrix(y_test_validation, val_predictions)
                
        # Save results for each folder
        results.append({
            'fold': fold_num,
            'val_accuracy': val_accuracy,
            'precision' : precision,
            'recall' : recall,
            'f1score' : f1score,
            'confusion_matrix' : conf.flatten()
        })
    return results

def print_cross_validation_results(results):
    accuracy_values = [result['val_accuracy'] for result in results]
    precision_values = [result['precision'] for result in results]
    recall_values = [result['recall'] for result in results]
    f1score_values = [result['f1score'] for result in results]
    conf_matrices_flat = [np.array(r['confusion_matrix']) for r in results]
    n_classes = int(np.sqrt(len(conf_matrices_flat[0])))

    conf_matrices = np.stack([cm.reshape(n_classes, n_classes) for cm in conf_matrices_flat])

    # Mean
    accuracy_mean= np.average(accuracy_values)
    precision_mean= np.average(precision_values)
    recall_mean= np.average(recall_values)
    f1score_mean= np.average(f1score_values)
    conf_matrices_mean= np.mean(conf_matrices, axis=0)

    # Standard deviation
    accuracy_std = np.std(accuracy_values)
    precision_std = np.std(precision_values)
    recall_std = np.std(recall_values)
    f1score_std = np.std(f1score_values)
    conf_matrices_std = np.std(conf_matrices, axis=0)

    print(f"Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")
    print(f"f1score: {f1score_mean:.4f} ± {f1score_std:.4f}")
    print("Confusion matrix:")
    for i in range(n_classes):
        row = ""
        for j in range(n_classes):
            row += f"{conf_matrices_mean[i, j]:5.1f}±{conf_matrices_std[i, j]:.1f}  "
        print(row)