"""Test one or more trained models on all our test sets and make one table (runs/eval/results.md + .csv).

usage: python scripts/evaluate.py models/gas_bottle_A_yolo11n.pt models/gas_bottle_B_yolo11n.pt
(ms per image is only a rough number: the test sets are small, so the warm-up of the GPU counts a lot)
"""
import csv
import os
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

# name: (dataset config, split)
TESTS = {
    "CylinDeRS test": ("configs/cylinders.yaml", "test"),
    "Primagaz test": ("configs/primagaz.yaml", "test"),
    "Juntar test (trucks from above)": ("configs/juntar.yaml", "test"),
}

if __name__ == "__main__":
    os.chdir(ROOT)  # dataset paths in configs/ are relative to the repo root
    rows = []
    for weights in sys.argv[1:]:
        w = Path(weights)
        model_name = w.parent.parent.name if w.parent.name == "weights" else w.stem  # runs/<name>/weights/best.pt
        for test, (cfg, split) in TESTS.items():
            m = YOLO(weights).val(data=cfg, split=split, imgsz=640, batch=16, verbose=False, plots=True,
                                  project=str(ROOT / "runs" / "eval"), name=f"{model_name}_{Path(cfg).stem}",
                                  exist_ok=True)
            rows.append({
                "model": model_name,
                "test set": test,
                "precision": round(float(m.box.mp), 3),
                "recall": round(float(m.box.mr), 3),
                "mAP50": round(float(m.box.map50), 3),
                "mAP50-95": round(float(m.box.map), 3),
                "ms per image": round(float(m.speed["inference"]), 1),
            })
            print(rows[-1])

    out = ROOT / "runs" / "eval"
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
