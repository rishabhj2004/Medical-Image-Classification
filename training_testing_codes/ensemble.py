import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, confusion_matrix

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

IMAGE_DIR = "/kaggle/input/datasets/rishabhjain200407/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
MASKED_IMAGE_DIR = "/kaggle/input/datasets/rishabhjain200407/masked-training/ISIC_2019_masked-20260427T102454Z-3-001/ISIC_2019_masked"

META_PATH = "/kaggle/input/datasets/rishabhjain200407/masked-training/ISIC_2019_Training_Metadata.csv"
LABEL_PATH = "/kaggle/input/datasets/rishabhjain200407/masked-training/ISIC_2019_Training_GroundTruth.csv"

UNMASKED_PATH = "/kaggle/input/datasets/rishabhjain200407/final-models/unmasked_best_model.pth"
MASKED_CKPT   = "/kaggle/input/datasets/rishabhjain200407/final-models/masked_best_model.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

meta = pd.read_csv(META_PATH)

meta["sex"] = meta["sex"].map({"male":0,"female":1}).fillna(0.5)
meta["age_approx"] = meta["age_approx"].fillna(meta["age_approx"].median())

meta = pd.get_dummies(meta, columns=["anatom_site_general"])
meta = meta.drop(["lesion_id"], axis=1).fillna(0)

labels = pd.read_csv(LABEL_PATH)
labels["label"] = labels.iloc[:,1:].idxmax(axis=1).astype("category").cat.codes

df = meta.merge(labels[["image","label"]], on="image")

# SAME SPLIT AS TRAINING
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=1/3, stratify=temp_df["label"], random_state=42
)

# SCALING (IMPORTANT)
scaler = StandardScaler()
train_df[["age_approx"]] = scaler.fit_transform(train_df[["age_approx"]])
test_df[["age_approx"]]  = scaler.transform(test_df[["age_approx"]])

meta_dim = train_df.drop(["image","label"], axis=1).shape[1]
num_classes = train_df["label"].nunique()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

class DualDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row["image"]

        # unmasked
        path1 = os.path.join(IMAGE_DIR, img_name + ".jpg")
        if not os.path.exists(path1):
            path1 = os.path.join(IMAGE_DIR, img_name + ".png")

        # masked
        path2 = os.path.join(MASKED_IMAGE_DIR, img_name + ".jpg")
        if not os.path.exists(path2):
            path2 = os.path.join(MASKED_IMAGE_DIR, img_name + ".png")

        img1 = transform(Image.open(path1).convert("RGB"))
        img2 = transform(Image.open(path2).convert("RGB"))

        meta = torch.tensor(row.drop(["image","label"]).values.astype("float32"))
        label = torch.tensor(row["label"]).long()

        return img1, img2, meta, label

test_loader = DataLoader(DualDataset(test_df), batch_size=32, shuffle=False)

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model)
        )

    def forward(self, img, meta):
        img = img + self.attn1(self.norm1(img), meta, meta)[0]
        meta = meta + self.attn2(self.norm1(meta), img, img)[0]
        img = img + self.ffn(self.norm2(img))
        meta = meta + self.ffn(self.norm2(meta))
        return img, meta


class SkinModel(nn.Module):
    def __init__(self, meta_dim, num_classes):
        super().__init__()

        backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.features = backbone.features

        self.img_proj = nn.Linear(1536, 256)

        self.meta_tokenizer = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256 * 4)
        )

        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock() for _ in range(4)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, meta):
        x = self.features(img)

        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0,2,1)

        img_tokens = self.img_proj(x)
        meta_tokens = self.meta_tokenizer(meta).view(B, 4, 256)

        for layer in self.cross_layers:
            img_tokens, meta_tokens = layer(img_tokens, meta_tokens)

        img_feat = img_tokens.mean(dim=1)
        meta_feat = meta_tokens.mean(dim=1)

        fused = torch.cat([img_feat * 1.2, meta_feat * 0.8], dim=1)

        return self.classifier(fused)

unmasked_model = SkinModel(meta_dim, num_classes).to(device)
unmasked_model.load_state_dict(torch.load(UNMASKED_PATH, map_location=device))
unmasked_model.eval()

masked_model = SkinModel(meta_dim, num_classes).to(device)
ckpt = torch.load(MASKED_CKPT, map_location=device)
masked_model.load_state_dict(ckpt["model"])
masked_model.eval()

class_weights_masked = torch.tensor([
    0.4, 0.2, 0.3, 0.6, 0.3, 0.1, 0.5, 0.5
]).to(device)

y_true, y_pred = [], []

with torch.no_grad():
    for img1, img2, meta, label in test_loader:

        img1 = img1.to(device)
        img2 = img2.to(device)
        meta = meta.to(device)

        out_u = unmasked_model(img1, meta)
        out_m = masked_model(img2, meta)

        prob_u = torch.softmax(out_u, dim=1)
        prob_m = torch.softmax(out_m, dim=1)

        w = class_weights_masked.unsqueeze(0)
        probs = (1 - w) * prob_u + w * prob_m

        preds = torch.argmax(probs, dim=1)

        y_pred.extend(preds.cpu().numpy())
        y_true.extend(label.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n FINAL ENSEMBLE RESULTS")
print(f"Accuracy           : {(y_true==y_pred).mean():.4f}")
print(f"Balanced Accuracy  : {balanced_accuracy_score(y_true,y_pred):.4f}")
print(f"Macro F1           : {f1_score(y_true,y_pred,average='macro'):.4f}")
print(f"Weighted F1        : {f1_score(y_true,y_pred,average='weighted'):.4f}")

print("\n CLASSIFICATION REPORT")
print(classification_report(y_true, y_pred))

print("\n CONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

print(confusion)
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "AK", "BCC", "BKL", "DF",
        "MEL", "NV", "SCC", "VASC"
    ]
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", values_format="d")

plt.title("Confusion Matrix - Ensemble Model")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
