import argparse, os, json, random, math, sys
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

def list_classes(root: Path) -> List[str]:
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

def find_images_for_class(root: Path, cls_name: str) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    cls_dir = root / cls_name
    return sorted([p for p in cls_dir.rglob("*") if p.suffix.lower() in exts])

def load_resize_to_uint8(path: Path, size: int) -> torch.Tensor:
    # returns [3, H, W] uint8
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((size, size), Image.LANCZOS)
        arr = torch.frombuffer(im.tobytes(), dtype=torch.uint8)
        # PIL gives bytes in HWC, contiguous
        arr = arr.view(size, size, 3).permute(2, 0, 1).contiguous()
        return arr

def shard_save(out_dir: Path, split: str, shard_idx: int,
               images: torch.Tensor, labels: torch.Tensor,
               one_hot: torch.Tensor):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_{shard_idx:04d}.pt"
    payload = {
        "images": images,
        "labels": labels,
        "one_hot": one_hot,
    }
    torch.save(payload, out_path)

def stratified_split(items_by_class: Dict[int, List[Path]],
                     train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for cls_idx, paths in items_by_class.items():
        paths = paths[:] 
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * train_frac))
        n_val   = int(round(n * val_frac))
        n_test  = n - n_train - n_val
        splits["train"] += [(p, cls_idx) for p in paths[:n_train]]
        splits["val"]   += [(p, cls_idx) for p in paths[n_train:n_train+n_val]]
        splits["test"]  += [(p, cls_idx) for p in paths[n_train+n_val:]]
    for k in splits:
        rng.shuffle(splits[k])
    return splits

def process_split(pairs: List[Tuple[Path,int]], split: str, out_dir: Path,
                  num_classes: int, image_size: int, shard_size: int):
    total = len(pairs)
    if total == 0:
        return 0, 0
    n_shards = math.ceil(total / shard_size)
    count_ok = 0
    skipped = 0
    for si in range(n_shards):
        start = si * shard_size
        end   = min((si+1)*shard_size, total)
        batch = pairs[start:end]

        imgs = []
        labels = []
        for p, lbl in tqdm(batch, desc=f"{split} shard {si+1}/{n_shards}", leave=False):
            try:
                t = load_resize_to_uint8(p, image_size) # [3,H,W] uint8
                imgs.append(t)
                labels.append(lbl)
                count_ok += 1
            except Exception as e:
                skipped += 1

        if len(imgs) == 0:
            continue

        images = torch.stack(imgs, dim=0)  #[N,3,H,W] uint8
        labels_t = torch.tensor(labels, dtype=torch.long) # [N]
        one_hot = F.one_hot(labels_t, num_classes=num_classes)       
        one_hot = one_hot.to(torch.uint8) # [N,C]

        shard_save(out_dir, split, si, images, labels_t, one_hot)
    return count_ok, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="raw_data", help="root with class folders")
    ap.add_argument("--out_dir", type=str, default="torch_data", help="where to save shards")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--shard_size", type=int, default=10000)  # 50k -> 5 shards per split at most
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_dir)
    out_root  = Path(args.out_dir)

    classes = list_classes(data_root)
    if len(classes) == 0:
        print(f"No class folders found under {data_root}", file=sys.stderr)
        sys.exit(1)

    label_to_index = {c: i for i, c in enumerate(classes)}
    index_to_label = {i: c for c, i in label_to_index.items()}
    num_classes = len(classes)

    # Gather files per class
    items_by_class: Dict[int, List[Path]] = {}
    total_imgs = 0
    for cls in classes:
        paths = find_images_for_class(data_root, cls)
        items_by_class[label_to_index[cls]] = paths
        total_imgs += len(paths)

    print(f"Found {num_classes} classes, {total_imgs} images.")

    # Save mapping
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "label_to_index.json", "w") as f:
        json.dump(label_to_index, f, indent=2)
    with open(out_root / "index_to_label.json", "w") as f:
        json.dump(index_to_label, f, indent=2)

    # Stratified splits
    splits = stratified_split(
        items_by_class,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    for k, v in splits.items():
        with open(out_root / f"{k}_files.json", "w") as f:
            # store just paths + labels for traceability
            json.dump([[str(p), lbl] for p, lbl in v], f)

    # Process each split into shards
    meta = {
        "image_size": args.image_size,
        "num_classes": num_classes,
        "classes": classes,
        "shard_size": args.shard_size,
        "counts": {},
    }

    for split in ["train", "val", "test"]:
        split_pairs = splits[split]
        split_dir = out_root / split
        ok, skipped = process_split(
            split_pairs, split, split_dir, num_classes, args.image_size, args.shard_size
        )
        meta["counts"][split] = {"ok": ok, "skipped": skipped}

    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
