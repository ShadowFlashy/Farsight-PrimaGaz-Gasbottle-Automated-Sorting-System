"""Copy a Roboflow export and put every box in one class (0 = gas_bottle), for datasets with several bottle classes.

usage: python scripts/to_one_class.py data/roboflow/juntar data/datasets/juntar
"""
import shutil
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists():
    shutil.rmtree(dst)
for split, new_split in (("train", "train"), ("valid", "val"), ("test", "test")):
    if not (src / split).exists():
        continue
    (dst / new_split / "labels").mkdir(parents=True)
    shutil.copytree(src / split / "images", dst / new_split / "images")
    boxes = 0
    for label in (src / split / "labels").glob("*.txt"):
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
        (dst / new_split / "labels" / label.name).write_text("".join(" ".join(["0"] + r[1:5]) + "\n" for r in rows))
        boxes += len(rows)
    print(f"{new_split}: {len(list((dst / new_split / 'images').iterdir()))} images, {boxes} boxes")
