import argparse, json, random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_meta(src_root: Path):
    with open(src_root / "meta.json", "r") as f:
        meta = json.load(f)
    return meta


def build_class_pointers(split_dir: Path, split: str, num_classes: int):
    """
    Return:
      class_to_ptrs: dict[int, list[(Path, int)]]
        maps class index -> list of (shard_path, idx_in_shard)
    """
    class_to_ptrs: Dict[int, List[Tuple[Path, int]]] = {c: [] for c in range(num_classes)}

    shard_paths = sorted(split_dir.glob(f"{split}_*.pt"))
    if not shard_paths:
        raise RuntimeError(f"No shards found in {split_dir} for split '{split}'")

    for sp in shard_paths:
        d = torch.load(sp, map_location="cpu")
        labels = d["labels"].tolist()
        for i, lbl in enumerate(labels):
            class_to_ptrs[int(lbl)].append((sp, i))

    return class_to_ptrs


def stratified_sample_pointers(
    class_to_ptrs: Dict[int, List[Tuple[Path, int]]],
    n_samples: int,
    rng: random.Random,
):
    """
    Proportional sampling by class to get exactly n_samples, preserving distribution.
    """
    # total available
    counts = {c: len(ptrs) for c, ptrs in class_to_ptrs.items()}
    total = sum(counts.values())
    if n_samples > total:
        raise ValueError(f"Requested {n_samples} samples but only {total} available")

    # compute quota per class: floor(share), then distribute leftovers by fractional part
    quotas = {}
    fracs = {}
    for c, cnt in counts.items():
        share = n_samples * (cnt / total) if total > 0 else 0.0
        base = int(share)
        frac = share - base
        quotas[c] = base
        fracs[c] = frac

    # adjust to get exactly n_samples
    current_total = sum(quotas.values())
    leftover = n_samples - current_total

    if leftover > 0:
        # add 1 to classes with largest fractional part
        for c, _ in sorted(fracs.items(), key=lambda x: x[1], reverse=True):
            if leftover == 0:
                break
            if quotas[c] < counts[c]:
                quotas[c] += 1
                leftover -= 1
    elif leftover < 0:
        # remove 1 from classes with smallest fractional part
        for c, _ in sorted(fracs.items(), key=lambda x: x[1]):
            if leftover == 0:
                break
            if quotas[c] > 0:
                quotas[c] -= 1
                leftover += 1

    assert sum(quotas.values()) == n_samples, "Quota adjustment failed"

    # actually sample pointers per class
    selected_ptrs: List[Tuple[Path, int]] = []
    for c, q in quotas.items():
        if q == 0:
            continue
        ptrs = class_to_ptrs[c]
        if q > len(ptrs):
            raise RuntimeError(f"Quota {q} exceeds available {len(ptrs)} for class {c}")
        selected_ptrs.extend(rng.sample(ptrs, q))

    rng.shuffle(selected_ptrs)
    return selected_ptrs


def materialize_samples(
    selected_ptrs: List[Tuple[Path, int]],
    num_classes: int,
):
    """
    Load the selected samples from original shards and stack into new tensors.
    """
    cache: Dict[Path, dict] = {}
    imgs = []
    labels = []

    for sp, idx in tqdm(selected_ptrs, desc="Gathering samples"):
        if sp not in cache:
            cache[sp] = torch.load(sp, map_location="cpu")
        d = cache[sp]
        imgs.append(d["images"][idx])
        labels.append(int(d["labels"][idx]))

    images = torch.stack(imgs, dim=0)  # [N,3,H,W] uint8
    labels_t = torch.tensor(labels, dtype=torch.long)  # [N]
    one_hot = F.one_hot(labels_t, num_classes=num_classes).to(torch.uint8)  # [N,C]
    return images, labels_t, one_hot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=str, default="torch_data",
                    help="Root with existing shards + meta.json")
    ap.add_argument("--dst_dir", type=str, default="torch_data_sampled",
                    help="Where to save sampled shards")
    ap.add_argument("--train_samples", type=int, default=8000)
    ap.add_argument("--val_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    meta = load_meta(src_root)
    num_classes = meta["num_classes"]
    image_size = meta["image_size"]
    classes = meta["classes"]

    # copy label mappings
    for fname in ["label_to_index.json", "index_to_label.json"]:
        src = src_root / fname
        if src.exists():
            with open(src, "r") as f_in, open(dst_root / fname, "w") as f_out:
                json.dump(json.load(f_in), f_out, indent=2)

    new_meta = {
        "image_size": image_size,
        "num_classes": num_classes,
        "classes": classes,
        "shard_size": max(args.train_samples, args.val_samples),
        "counts": {},
    }

    # --- Train split ---
    if args.train_samples > 0:
        train_src_dir = src_root / "train"
        train_dst_dir = dst_root / "train"
        train_dst_dir.mkdir(parents=True, exist_ok=True)

        print("Building class pointers for train split...")
        class_to_ptrs_train = build_class_pointers(train_src_dir, "train", num_classes)
        print("Stratified sampling train...")
        train_selected = stratified_sample_pointers(
            class_to_ptrs_train,
            args.train_samples,
            rng,
        )
        train_images, train_labels_t, train_one_hot = materialize_samples(
            train_selected, num_classes
        )

        out_path = train_dst_dir / "train_0000.pt"
        torch.save(
            {
                "images": train_images,
                "labels": train_labels_t,
                "one_hot": train_one_hot,
            },
            out_path,
        )
        new_meta["counts"]["train"] = {
            "ok": int(train_labels_t.numel()),
            "skipped": 0,
        }
        print(f"Saved sampled train shard to {out_path}")

    # --- Val split ---
    if args.val_samples > 0:
        val_src_dir = src_root / "val"
        val_dst_dir = dst_root / "val"
        val_dst_dir.mkdir(parents=True, exist_ok=True)

        print("Building class pointers for val split...")
        class_to_ptrs_val = build_class_pointers(val_src_dir, "val", num_classes)
        print("Stratified sampling val...")
        val_selected = stratified_sample_pointers(
            class_to_ptrs_val,
            args.val_samples,
            rng,
        )
        val_images, val_labels_t, val_one_hot = materialize_samples(
            val_selected, num_classes
        )

        out_path = val_dst_dir / "val_0000.pt"
        torch.save(
            {
                "images": val_images,
                "labels": val_labels_t,
                "one_hot": val_one_hot,
            },
            out_path,
        )
        new_meta["counts"]["val"] = {
            "ok": int(val_labels_t.numel()),
            "skipped": 0,
        }
        print(f"Saved sampled val shard to {out_path}")

    # no test sampling here, but keep key for consistency
    if "test" not in new_meta["counts"]:
        new_meta["counts"]["test"] = {"ok": 0, "skipped": 0}

    with open(dst_root / "meta.json", "w") as f:
        json.dump(new_meta, f, indent=2)

    print("Done.")
    print(json.dumps(new_meta, indent=2))


if __name__ == "__main__":
    main()
