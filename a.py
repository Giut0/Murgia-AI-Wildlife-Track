import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import numpy as np
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from train_utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Trasformazioni immagine per ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
label_map = {
    'badger': 0,
    'bird': 1,
    'boar': 2,
    'butterfly': 3,
    'cat': 4,
    'dog': 5,
    'fox': 6,
    'lizard': 7,
    'podolic_cow': 8,
    'porcupine': 9,
    'weasel': 10,
    'wolf': 11
}

# Dataset e DataLoader
train_ds = AnimalDataset("data/train.csv", "data/labeled_img/", transform=transform, label_map=label_map, crop_bbox=True)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

val_ds = AnimalDataset("data/val.csv", "data/labeled_img/", transform=transform, label_map=label_map, crop_bbox=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

test_ds = AnimalDataset("data/test.csv", "data/labeled_img/", transform=transform, label_map=label_map, crop_bbox=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Carica il modello ViT
model_feat = timm.create_model('vit_base_patch16_224', pretrained=True)
model_feat.reset_classifier(0)  # rimuove la testa (classificazione)

model_feat.eval()  # modalità valutazione
model_feat = model_feat.to(device)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)


def extract_features(model, dataloader):
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            out = model(images)  # output shape: [batch_size, feature_dim]
            features.append(out.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

X_train, y_train = extract_features(model_feat, train_loader)
X_val, y_val = extract_features(model_feat, val_loader)

X_test, y_test = extract_features(model_feat, test_loader)



clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

bayes = ba

y_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average='macro')
precision = precision_score(y_test, y_test_pred, average='macro')
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)
print(f"Validation Recall: {recall:.7f}")
print(f"Validation Precision: {precision:.7f}")
print(f"Validation F1 Score: {f1:.7f}")
print(f"Validation Accuracy: {acc:.7f}")