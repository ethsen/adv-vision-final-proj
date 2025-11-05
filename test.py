import torch
from pathlib import Path
from collections import Counter

root = Path("processed_data/train")  # change to val/test if needed
sizes = Counter()

for f in sorted(root.glob("*.pt")):
    data = torch.load(f, map_location="cpu")
    imgs = data["images"]  # [N,3,H,W]
    N, C, H, W = imgs.shape
    sizes[(H, W)] += N
    print(f"{f.name}: {H}x{W} ({N} images)")

print("\nSummary of all shard sizes:")
for (H, W), count in sizes.items():
    print(f"{H}x{W}: {count} images")

print(f"\nTotal images checked: {sum(sizes.values())}")
