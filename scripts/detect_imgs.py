"""Run a YOLO model on images and save the images with the boxes drawn on them.

usage: python scripts/detect_imgs.py <image or folder> [--model yolo11n.pt] [--out runs/detect_demo] [--conf 0.25]
e.g.   python scripts/detect_imgs.py data/commons/images                                       (COCO model: person, car...)
       python scripts/detect_imgs.py data/commons/images --model runs/cyl_y11n/weights/best.pt  (our gas bottle model)
"""
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="+", help="image(s) or a folder")
    p.add_argument("--model", default="yolo11n.pt")  # yolo11s.pt is bigger / better / slower
    p.add_argument("--out", default="runs/detect_demo")
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()

    model = YOLO(a.model)
    found = Counter()
    for src in a.source:
        for r in model.predict(src, conf=a.conf, stream=True, verbose=False):
            names = [f"{r.names[int(c)]} {s:.2f}" for c, s in zip(r.boxes.cls, r.boxes.conf)]
            found.update(r.names[int(c)] for c in r.boxes.cls)
            out = Path(a.out) / Path(r.path).name
            out.parent.mkdir(parents=True, exist_ok=True)
            r.save(filename=str(out))
            print(f"{Path(r.path).name}: {', '.join(names) or 'nothing found'}")
    print("total:", dict(found.most_common()))
