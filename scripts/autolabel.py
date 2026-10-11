"""Pre-label images with one of our models ("model-assisted labeling"). The boxes are only a first guess, they still
have to be checked (and corrected) by a person, e.g. after uploading the images + labels to Roboflow or CVAT.

writes the labels in YOLO format next to the images folder (<images>/../labels/) and preview images with the boxes
drawn on them in --preview (to check them quickly)

usage: python scripts/autolabel.py data/commons/images --model runs/cyl_y11n/weights/best.pt --conf 0.4
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("images", help="folder with images")
    p.add_argument("--model", required=True)
    p.add_argument("--conf", type=float, default=0.4, help="higher = fewer wrong boxes but more missed bottles")
    p.add_argument("--preview", default="runs/autolabel_preview")
    a = p.parse_args()

    images = Path(a.images)
    labels = images.parent / "labels"
    labels.mkdir(exist_ok=True)
    Path(a.preview).mkdir(parents=True, exist_ok=True)
    model = YOLO(a.model)
    n_img = n_box = 0
    for r in model.predict(str(images), conf=a.conf, stream=True, verbose=False):
        name = Path(r.path).stem
        rows = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in r.boxes.xywhn.tolist()]
        (labels / f"{name}.txt").write_text("".join(row + "\n" for row in rows))
        r.save(filename=str(Path(a.preview) / f"{name}.jpg"))
        n_img += 1
        n_box += len(rows)
    print(f"{n_img} images, {n_box} boxes -> {labels}")
