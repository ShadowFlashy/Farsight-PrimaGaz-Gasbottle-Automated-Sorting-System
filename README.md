# bottle-tracking

Detecting, tracking and counting (Primagaz) gas bottles with YOLO (Ultralytics).

- [docs/REPORT.md](docs/REPORT.md): our report (test setup, performance criteria, results, issues, findings, questions)
- [docs/DATA.md](docs/DATA.md): all data sources (Roboflow datasets, Commons photos, videos), how to download them, licenses
- `models/`: our trained gas bottle models (YOLO11n, 1 class `gas_bottle`)
  - `gas_bottle_A_yolo11n.pt`: trained on CylinDeRS
  - `gas_bottle_B_yolo11n.pt`: trained on all our data (best one)
- `configs/`: dataset configs for training / testing
- `data/` (not in git): all datasets, the Commons photos with our labels and the videos, see [docs/DATA.md](docs/DATA.md) how to get them. Credits of the Commons photos: `docs/commons_attribution.csv`
- `scripts/`:
  - `detect_imgs.py`: run a model on images
  - `track_videos.py`: track + count objects in a video (`--line` for a counting line)
  - `train.py`, `evaluate.py`: train a model, test models on all test sets
  - `autolabel.py`: pre-label images with a model (check them afterwards in Roboflow / CVAT)
  - `download_commons.py`, `download_roboflow.py`, `clean_export.py`, `prepare_primagaz.py`, `to_one_class.py`: get and prepare the data
  - `cut_clip.py`: cut a short clip out of a video

Quick try (needs `pip install -r requirements.txt` and the photos in `data/`, see [docs/DATA.md](docs/DATA.md)):

```
python scripts/detect_imgs.py data/commons/images --model models/gas_bottle_B_yolo11n.pt --out runs/detect_demo
```
