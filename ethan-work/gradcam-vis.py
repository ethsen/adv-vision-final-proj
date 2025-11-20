import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18, resnet34, resnet50
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


# ---------------- Dataset (same as training) ----------------
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

        # ImageNet normalization used during training
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fi, ii = self.samples[idx]
        shard = self.data[fi]
        x = shard["images"][ii].float() / 255.0
        y = shard["labels"][ii].long()
        if self.normalize:
            x = (x - self.mean) / self.std
        return x, y


# ---------------- Grad-CAM helper ----------------
class GradCAM:
    """
    Simple Grad-CAM implementation for a single target layer.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            # grad_out is a tuple; we want the gradient w.r.t. the output
            self.gradients = grad_out[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.handles.append(self.target_layer.register_backward_hook(bwd_hook))

    def __call__(self, x, target_index=None):
        """
        x: [1, 3, H, W]
        target_index: class index to compute Grad-CAM for (int or tensor).
                      If None, use predicted class.
        Returns:
            cam: [1, 1, Hc, Wc] normalized to [0, 1]
            logits: [1, num_classes]
        """
        self.model.zero_grad()
        logits = self.model(x)  # [1, C]

        if target_index is None:
            target_index = logits.argmax(dim=1)

        if isinstance(target_index, int):
            target_index = torch.tensor([target_index], device=logits.device)

        # One-hot for backward
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_index.item()] = 1.0

        logits.backward(gradient=one_hot, retain_graph=True)

        # activations: [1, C, Hc, Wc], gradients: [1, C, Hc, Wc]
        grads = self.gradients
        acts = self.activations

        # Global-average-pool the gradients over spatial dims
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted sum of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, Hc, Wc]
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, logits

    def close(self):
        for h in self.handles:
            h.remove()


# ---------------- Utility: denormalize for plotting ----------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(img_tensor):
    """
    img_tensor: [3, H, W] normalized with ImageNet stats
    returns: [3, H, W] in [0,1]
    """
    x = img_tensor * IMAGENET_STD + IMAGENET_MEAN
    x = torch.clamp(x, 0.0, 1.0)
    return x


def save_gradcam_separate(
    img_tensor, cam_tensor, pred_idx, target_idx,
    out_dir, base_name, idx_to_label=None
):
    """
    Saves two separate images:
      1. Input image only  → <base_name>_input.png
      2. Grad-CAM overlay → <base_name>_cam.png
    """

    # Upsample CAM to image size
    _, _, Hc, Wc = cam_tensor.shape
    _, H, W = img_tensor.shape
    cam_up = F.interpolate(
        cam_tensor, size=(H, W), mode="bilinear", align_corners=False
    )
    cam_up = cam_up.squeeze().cpu().numpy()  # [H, W]

    # ---- Denormalize image ----
    img_denorm = denormalize(img_tensor.cpu())
    img_np = img_denorm.permute(1, 2, 0).numpy()  # [H, W, 3]

    # ---- Label strings ----
    if idx_to_label is not None:
        pred_label = idx_to_label.get(str(pred_idx), str(pred_idx))
        target_label = idx_to_label.get(str(target_idx), str(target_idx))
    else:
        pred_label = str(pred_idx)
        target_label = str(target_idx)

    title = f"Pred: {pred_label} ({pred_idx}) | Target: {target_label} ({target_idx})"

    # ============= SAVE INPUT IMAGE =============
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_np)
    ax.axis("off")
    fig.savefig(out_dir / f"{base_name}_input.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============= SAVE ONLY CAM IMAGE =============
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_np)
    ax.imshow(cam_up, cmap="jet", alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.savefig(out_dir / f"{base_name}_cam.png", dpi=150, bbox_inches="tight")
    plt.close(fig)



# ---------------- Main script ----------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="torch_data",
                        help="Root of shard dataset (same as training).")
    parser.add_argument("--split", type=str, default="val",
                        help="Which split to visualize from: train/val/test.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to checkpoint .pt file (saved in training).")
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="ResNet architecture to instantiate.")
    parser.add_argument("--out_dir", type=str, default="gradcam_outputs",
                        help="Directory to save Grad-CAM figures.")
    parser.add_argument("--num_images", type=int, default=16,
                        help="Number of images to visualize.")
    parser.add_argument("--label_map", type=str, default="torch_data/index_to_label.json",
                        help="Optional path to JSON mapping from class index to label.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load shards ----
    split_dir = Path(args.data_root) / args.split
    shard_paths = sorted(split_dir.glob("*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No .pt shards found in {split_dir}")

    print(f"Loading {len(shard_paths)} {args.split} shards...")
    ds_full = ShardDataset(shard_paths)
    num_classes = ds_full.num_classes
    print(f"Found {num_classes} classes in dataset.")

    # Subsample if dataset is huge
    rng = random.Random(42)
    indices = list(range(len(ds_full)))
    rng.shuffle(indices)
    indices = indices[:args.num_images * 2]  # extra in case of mis-loads
    ds_subset = Subset(ds_full, indices)

    loader = DataLoader(
        ds_subset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ---- Build model ----
    if args.arch == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif args.arch == "resnet34":
        model = resnet34(num_classes=num_classes)
    else:
        model = resnet50(num_classes=num_classes)

    # Load checkpoint (expects {"model": state_dict, "num_classes": ...})
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)  # handle raw state_dict too
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Target layer: last block of layer4 (works for 18/34/50)
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    # Optional index->label mapping
    idx_to_label = None
    label_map_path = Path(args.label_map)
    if label_map_path.is_file():
        with open(label_map_path, "r") as f:
            idx_to_label = json.load(f)
        print(f"Loaded label map from {label_map_path}")
    else:
        print(f"No label map found at {label_map_path}, using raw indices.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Iterate and visualize ----
    saved = 0
    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Grad-CAM")):
        if saved >= args.num_images:
            break

        x = x.to(device)  # [1,3,H,W]
        y = y.to(device)  # [1]

        with torch.enable_grad():
            cam, logits = gradcam(x, target_index=None)
        pred_idx = logits.argmax(dim=1).item()
        target_idx = y.item()

        out_path = out_dir / f"gradcam_{saved:03d}.png"
        base_name = f"resnet{args.arch.replace('resnet','')}_{saved:03d}"

        save_gradcam_separate(
            img_tensor=x[0].detach().cpu(),
            cam_tensor=cam.detach().cpu(),
            pred_idx=pred_idx,
            target_idx=target_idx,
            out_dir=out_dir,
            base_name=base_name,
            idx_to_label=idx_to_label
        )
        saved += 1

    gradcam.close()
    print(f"Saved {saved} Grad-CAM figures to {out_dir}")


if __name__ == "__main__":
    main()
