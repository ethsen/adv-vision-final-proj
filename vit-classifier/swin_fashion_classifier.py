import timm
import os, json, math, random, csv
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18, resnet34, resnet50
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- Dataset that loads full shards ----------------
class ShardDataset(Dataset):
    def __init__(self, shard_paths, normalize=True):
        self.shards = shard_paths
        self.samples = []  # (file_idx, local_idx)
        self.data = []
        self.normalize = normalize
        # Load all shards fully into memory
        for fi, f in enumerate(self.shards):
            d = torch.load(f, map_location="cpu")
            self.data.append(d)
            n = d["labels"].shape[0]
            self.samples.extend([(fi, i) for i in range(n)])
        self.num_classes = self.data[0]["one_hot"].shape[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fi, ii = self.samples[idx]
        shard = self.data[fi]
        x = shard["images"][ii].float() / 255.0
        y = shard["labels"][ii].long()
        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            x = (x - mean) / std
        return x, y


# ---------------- Training helpers ----------------
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = ce(out, y)
            total_loss += float(loss) * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total


class MultiStageDesignerHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(c) for c in channels])
        total_c = sum(channels)
        self.classifier = nn.Linear(total_c, num_classes)

    def forward(self, feats):
        # feats: list of [B, C_i, H_i, W_i]
        pooled = []
        for f, norm in zip(feats, self.norms):
            f_pool = f.mean(dim=(2, 3))          # B × C_i
            f_pool = norm(f_pool)                # layernorm per stage
            pooled.append(f_pool)
        f_cat = torch.cat(pooled, dim=-1)        # B × sum(C_i)
        logits = self.classifier(f_cat)          # B × num_classes
        return logits

class DesignerMultiStageViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True  # returns list of stage outputs
        )
        # Swin-T channels:
        channels = [96, 192, 384, 768]
        self.head = MultiStageDesignerHead(channels, num_classes)

    def forward(self, x):
        feats = self.backbone(x)  # [x1, x2, x3, x4]
        logits = self.head(feats)
        return logits


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="torch_data")
    ap.add_argument("--train_count", type=int, default=45000)
    ap.add_argument("--val_count", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--from_scratch", action="store_false")
    ap.add_argument("--out_dir", default="outputs/vit_outputs")
    args = ap.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = ds_train_full.num_classes

    # ---- Random subsampling ----
    rng = random.Random(19)
    train_idx = rng.sample(range(len(ds_train_full)), min(args.train_count, len(ds_train_full)))
    val_idx   = rng.sample(range(len(ds_val_full)),   min(args.val_count, len(ds_val_full)))
    ds_train = Subset(ds_train_full, train_idx)
    ds_val   = Subset(ds_val_full,   val_idx)

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )
    val_loader   = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    model = DesignerMultiStageViT(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)