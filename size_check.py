from pathlib import Path
from PIL import Image
from collections import Counter

root = Path("raw_data")  # root folder with designer subfolders
sizes = Counter()
count = 0

for img_path in root.rglob("*.png"):
    try:
        with Image.open(img_path) as im:
            sizes[im.size] += 1  # (width, height)
            count += 1
    except Exception as e:
        print(f"Error with {img_path}: {e}")

print("\nUnique image sizes:")
for (w, h), n in sizes.most_common():
    print(f"{w}x{h}: {n} images")

print(f"\nTotal images checked: {count}")
