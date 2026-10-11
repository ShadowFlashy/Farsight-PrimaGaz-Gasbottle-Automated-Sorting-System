"""Train (fine-tune) a YOLO model on one of the dataset configs in configs/.

usage: python scripts/train.py configs/cylinders.yaml --model yolo11n.pt --epochs 60 --name cyl_y11n
results (weights, plots, results.csv) end up in runs/<name>/
"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":  # needed on Windows because the dataloader starts extra processes
    p = argparse.ArgumentParser()
    p.add_argument("data", help="dataset config, e.g. configs/cylinders.yaml")
    p.add_argument("--model", default="yolo11n.pt", help="start weights (COCO pretrained or one of our runs)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--name", required=True)
    a = p.parse_args()

    os.chdir(ROOT)  # dataset paths in configs/ are relative to the repo root
    model = YOLO(a.model)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        workers=a.workers,
        patience=15,  # stop when val mAP hasn't improved for 15 epochs
        project=str(ROOT / "runs"),
        name=a.name,
        exist_ok=True,
        seed=0,
    )
