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
from sklearn.metrics import f1_score, classification_report

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = (device.type == "cuda")

print("Device:", device)

#paths
META_PATH= "/kaggle/input/datasets/rishabhjain200407/masked-training/ISIC_2019_Training_Metadata.csv"
LABEL_PATH = "/kaggle/input/datasets/rishabhjain200407/masked-training/ISIC_2019_Training_GroundTruth.csv"
IMAGE_DIR = "/kaggle/input/datasets/rishabhjain200407/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input"

CKPT_PATH = "/kaggle/working/models/ckpt.pth"
SAVE_DIR = "/kaggle/working/models"
os.makedirs(SAVE_DIR, exist_ok=True)
CheckPoint= f"{SAVE_DIR}/ckpt_model.pth"
BEST_PATH = f"{SAVE_DIR}/best_model.pth"

# Data proeprocessing

meta = pd.read_csv(META_PATH)

meta["sex"] = meta["sex"].map({"male":0,"female":1}).fillna(0.5)
meta["age_approx"] = meta["age_approx"].fillna(meta["age_approx"].median())

meta = pd.get_dummies(meta, columns=["anatom_site_general"])
meta = meta.drop(["lesion_id"], axis=1).fillna(0)

labels = pd.read_csv(LABEL_PATH)
labels["label"] = labels.iloc[:,1:].idxmax(axis=1).astype("category").cat.codes

df = meta.merge(labels[["image","label"]], on="image")

#Splits
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=1/3, stratify=temp_df["label"], random_state=42
)

scaler_meta = StandardScaler()
train_df[["age_approx"]] = scaler_meta.fit_transform(train_df[["age_approx"]])
val_df[["age_approx"]]   = scaler_meta.transform(val_df[["age_approx"]])
test_df[["age_approx"]]  = scaler_meta.transform(test_df[["age_approx"]])

meta_dim = train_df.drop(["image","label"], axis=1).shape[1]
num_classes = train_df["label"].nunique()

print("done")

scaler = torch.amp.GradScaler(enabled=use_amp)

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

class ISICDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["image"] + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, row["image"] + ".png")

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (224,224))

        img = self.transform(img)

        meta = torch.tensor(row.drop(["image","label"]).values.astype("float32"))
        label = torch.tensor(row["label"]).long()

        return img, meta, label

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

        for p in self.features.parameters():
            p.requires_grad = False

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

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True
        print("Backbone unfrozen")

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

train_loader = DataLoader(ISICDataset(train_df, IMAGE_DIR, train_transform),
                          batch_size=32, shuffle=True, num_workers=2, pin_memory=use_amp)

val_loader = DataLoader(ISICDataset(val_df, IMAGE_DIR, val_transform),
                        batch_size=32, shuffle=False, num_workers=2, pin_memory=use_amp)

test_loader = DataLoader(ISICDataset(test_df, IMAGE_DIR, val_transform),
                         batch_size=32, shuffle=False, num_workers=2, pin_memory=use_amp)

counts = train_df["label"].value_counts().sort_index()
weights = 1.0 / (counts ** 0.5)
weights = weights / weights.sum() * len(counts)

class_weights = torch.tensor(weights.values, dtype=torch.float32).to(device)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

focal = FocalLoss()
ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)


def evaluate_test(model, loader, epoch):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img, meta, label in loader:
            img, meta, label = img.to(device), meta.to(device), label.to(device)

            x = model(img, meta)
            probs = torch.softmax(x, dim=1)
            preds = torch.argmax(probs, dim=1)

            correct += (preds == label).sum().item()
            total += label.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, f1, all_preds, all_labels

model = SkinModel(meta_dim, num_classes).to(device)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

start_epoch = 0
best_f1 = 0

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt["epoch"] + 1
    best_f1 = ckpt["best_f1"]

    print("Resumed from epoch:", start_epoch)

num_epochs = 40
final_test_preds, final_test_labels = None, None

for epoch in range(start_epoch, num_epochs):

    if epoch == 4:
        model.unfreeze_backbone()

        optimizer = torch.optim.AdamW([
            {"params": model.features.parameters(), "lr": 5e-5},
            {"params": model.img_proj.parameters(), "lr": 1e-4},
            {"params": model.meta_tokenizer.parameters(), "lr": 1e-4},
            {"params": model.cross_layers.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ], weight_decay=1e-4)

    print(f"\nEpoch {epoch+1}")

    model.train()
    train_loss, correct, total = 0, 0, 0

    for i, (img, meta, label) in enumerate(train_loader):
        img, meta, label = img.to(device), meta.to(device), label.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(img, meta)
                loss = 0.4*focal(out,label) + 0.6*ce_loss(out,label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(img, meta)
            loss = 0.4*focal(out,label) + 0.6*ce_loss(out,label)
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            print(f"Batch {i}/{len(train_loader)}")

        train_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    train_loss /= len(train_loader)
    train_acc = correct / total
    model.eval()
    val_correct, val_total = 0, 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for img, meta, label in val_loader:
            img, meta, label = img.to(device), meta.to(device), label.to(device)

            out = model(img, meta)
            preds = torch.argmax(out, dim=1)

            val_correct += (preds == label).sum().item()
            val_total += label.size(0)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(label.cpu().numpy())

    val_acc = val_correct / val_total
    val_f1 = f1_score(val_labels, val_preds, average="macro")

    test_acc, test_f1, test_preds, test_labels = evaluate_test(model, test_loader, epoch)

    print(f"Train Acc {train_acc:.4f}")
    print(f"Val Acc {val_acc:.4f} F1 {val_f1:.4f}")
    print(f"Test Acc {test_acc:.4f} F1 {test_f1:.4f}")

    scheduler.step()

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "best_f1": best_f1
    }, CheckPoint)

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), BEST_PATH)
        print("Best model saved")

    final_test_preds = test_preds
    final_test_labels = test_labels

    if epoch == 29:
        print("\n TEST CLASSIFICATION REPORT")
        print(classification_report(final_test_labels, final_test_preds))
