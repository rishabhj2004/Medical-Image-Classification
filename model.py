import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights
from unet_model import UNet 

DEVICE = torch.device("cpu")
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

# Split transforms: 0-1 range for masking, then standard normalization for EfficientNet
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
norm_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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

class SkinModelInference(nn.Module):
    def __init__(self, meta_dim=10, num_classes=8):
        super().__init__()
        backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.features = backbone.features
        self.img_proj = nn.Linear(1536, 256)
        self.meta_tokenizer = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1024)
        )
        self.cross_layers = nn.ModuleList([CrossAttentionBlock() for _ in range(4)])
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, meta):
        x = self.features(img)
        B, C, _, _ = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        img_tokens = self.img_proj(x)
        meta_tokens = self.meta_tokenizer(meta).view(B, 4, 256)
        for layer in self.cross_layers:
            img_tokens, meta_tokens = layer(img_tokens, meta_tokens)
        img_feat, meta_feat = img_tokens.mean(dim=1), meta_tokens.mean(dim=1)
        fused = torch.cat([img_feat * 1.2, meta_feat * 0.8], dim=1)
        return self.classifier(fused)

def load_models():
    mu = SkinModelInference(10, 8).to(DEVICE)
    mm = SkinModelInference(10, 8).to(DEVICE)
    # Helper to handle different checkpoint formats
    def load_weights(m, p):
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=DEVICE)
            m.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        m.eval()
        return m
    return load_weights(mu, "models/unmasked_best_model.pth"), load_weights(mm, "models/masked_best_model.pth")

def load_unet():
    model = UNet().to(DEVICE)
    if os.path.exists("models/best_unet.pth"):
        ckpt = torch.load("models/best_unet.pth", map_location=DEVICE)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()
    return model

def predict(img_path, age, sex, mu, mm, unet, meta_override=None):
    img = Image.open(img_path).convert("RGB")
    raw_tensor = base_transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        mask = torch.sigmoid(unet(raw_tensor))
        masked_raw = raw_tensor * mask
    
    img_tensor = norm_transform(raw_tensor)
    masked_img_tensor = norm_transform(masked_raw)
    meta_tensor = torch.tensor(meta_override).unsqueeze(0).to(DEVICE).float()

    with torch.no_grad():
        log_u, log_m = mu(img_tensor, meta_tensor), mm(masked_img_tensor, meta_tensor)
        probs = torch.softmax((log_u + log_m) / 2, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    return CLASS_NAMES[pred_idx], probs[0].cpu(), mask, img_tensor, meta_tensor
