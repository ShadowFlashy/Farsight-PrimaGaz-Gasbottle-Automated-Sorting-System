"""Build our Primagaz dataset from the two Roboflow exports with Primagaz bottles (gasoknok + Gas).

- all classes (gas_ok, gas_nok, gas bottle, damaged bottle, wrong bottle) become one class: 0 = gas_bottle
- the original Roboflow split puts the same bottle in train AND test (same bottle id in the file name), which makes
  test scores too optimistic. So we split again, by session + bottle id: 50% train / 15% val / 35% test.
- the Gas images (incl. other brands and damaged bottles) all go to test
- meta.csv keeps the original class of every box (could be used later for an OK / NOK model)

usage: python scripts/clean_export.py data/roboflow/gasoknok   (+ same for data/roboflow/gas)
       python scripts/prepare_primagaz.py   -> data/datasets/primagaz/{train,val,test}/{images,labels}
"""
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "roboflow"
OUT = ROOT / "data" / "datasets" / "primagaz"
NAMES = {"gasoknok": ["gas_nok", "gas_ok"], "gas": ["damaged bottle", "gas bottle", "wrong bottle"]}


def items(dataset):
    for split in ("train", "valid", "test"):
        for img in sorted((SRC / dataset / split / "images").iterdir()):
            yield img, SRC / dataset / split / "labels" / (img.stem + ".txt")


def group_of(img):
    """13_44_id_122_back_left_crap_2_jpg.rf.xxx -> ('13_44', '122'): same session + same bottle id = same group"""
    stem = img.name.split(".rf.")[0]
    m = re.match(r"^(\d+_\d+)_.*?id_([a-z0-9]+)", stem)
    return (m.group(1), m.group(2)) if m else ("single", stem)


def copy(img, label, split, dataset, meta):
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    name = f"{dataset}_{img.stem}"  # run clean_export.py first, so the names are short
    shutil.copy(img, OUT / split / "images" / (name + img.suffix))
    rows = [line.split() for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
    with open(OUT / split / "labels" / (name + ".txt"), "w") as f:
        for r in rows:
            f.write(" ".join(["0"] + r[1:5]) + "\n")  # every class -> gas_bottle, keep the box
            meta.append({"file": name + img.suffix, "split": split, "source": dataset,
                         "original_class": NAMES[dataset][int(r[0])]})


if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    meta = []

    groups = defaultdict(list)
    for img, label in items("gasoknok"):
        groups[group_of(img)].append((img, label))
    keys = sorted(groups)
    random.Random(0).shuffle(keys)
    n_train, n_val = int(0.5 * len(keys)), int(0.15 * len(keys))
    for i, k in enumerate(keys):
        split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for img, label in groups[k]:
            copy(img, label, split, "gasoknok", meta)

    for img, label in items("gas"):
        copy(img, label, "test", "gas", meta)

    with open(OUT / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)
    for split in ("train", "val", "test"):
        n_img = len(list((OUT / split / "images").iterdir()))
        n_box = sum(1 for m in meta if m["split"] == split)
        print(f"{split}: {n_img} images, {n_box} bottles")
    print(f"{len(keys)} bottle groups from gasoknok -> {n_train} train / {n_val} val / {len(keys) - n_train - n_val} test")
