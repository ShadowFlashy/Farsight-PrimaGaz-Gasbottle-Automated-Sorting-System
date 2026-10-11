"""Clean a Roboflow export so Ultralytics can use all images. Fixes 2 problems we ran into with CylinDeRS:

1. very long image names: Windows can't open files with a full path longer than 260 characters (unless "long paths"
   is turned on), Ultralytics then skips them with "No such file or directory". -> names get shortened (image + label)
2. label files that mix polygons (segmentation) and boxes: Ultralytics skips those images
   ("labels mix segment and detection rows"). -> polygons are turned into boxes

usage: python scripts/clean_export.py data/roboflow/cylinders
"""
import hashlib
import os
import re
import sys
from pathlib import Path


def long_path(p):
    return "\\\\?\\" + str(p)  # this prefix lets Windows open paths longer than 260 characters


def polygon_to_box(row):
    xs, ys = [float(v) for v in row[1::2]], [float(v) for v in row[2::2]]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return f"{row[0]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


root = Path(sys.argv[1]).resolve()
renamed = fixed = 0
for split in ("train", "valid", "val", "test"):
    img_dir, lbl_dir = root / split / "images", root / split / "labels"
    if not img_dir.exists():
        continue
    for name in os.listdir(long_path(img_dir)):
        stem, ext = os.path.splitext(name)
        if len(stem) <= 60:
            continue
        m = re.search(r"\.rf\.([0-9a-f]+)$", stem)  # roboflow names end with a unique hash
        tag = m.group(1)[:16] if m else hashlib.md5(stem.encode()).hexdigest()[:16]
        new_stem = f"{stem[:30]}_{tag}"
        os.rename(long_path(img_dir / name), long_path(img_dir / (new_stem + ext)))
        label = lbl_dir / (stem + ".txt")
        if os.path.exists(long_path(label)):
            os.rename(long_path(label), long_path(lbl_dir / (new_stem + ".txt")))
        renamed += 1
    for label in lbl_dir.glob("*.txt"):
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
        if any(len(r) > 5 for r in rows):
            label.write_text("\n".join(polygon_to_box(r) if len(r) > 5 else " ".join(r) for r in rows) + "\n")
            fixed += 1

for cache in root.rglob("*.cache"):  # old ultralytics caches still contain the old names / labels
    cache.unlink()
print(f"{root.name}: shortened {renamed} file names, turned polygons into boxes in {fixed} label files")
