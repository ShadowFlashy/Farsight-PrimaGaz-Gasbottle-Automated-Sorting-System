"""Download the labeled Roboflow Universe datasets we use (YOLOv11 format) into data/roboflow/<name>/.

Needs your own (free) Roboflow API key: Roboflow -> Settings -> API Keys -> copy the Private API Key and put it in a
file called .env in the root of the repo:
    ROBOFLOW_API_KEY=your_key_here
(.env is in .gitignore, never commit your key)

usage: python scripts/download_roboflow.py [name ...]     (no names = all datasets)
"""
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "roboflow"

# name: (workspace, project, version)  -> see docs/DATA.md for what's in them
DATASETS = {
    "cylinders": ("klearchos-stavrothanasopoulos-konstantinos-gkountakos-6jwgj", "cylinders-iaq6n", 1),  # CylinDeRS
    "gasoknok": ("gas-by6ce", "gasoknok-vzwlk", 8),
    "gas": ("gas-by6ce", "gas-vm7vi", 1),
    "juntar": ("testgasbottles", "juntar", 2),
}


def api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        sys.exit("No Roboflow API key found, see the top of this script")
    return key


def export_link(workspace, project, version, key):
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov11"
    for _ in range(120):  # Roboflow first has to generate the export, this can take a few minutes
        r = requests.get(url, params={"api_key": key}, timeout=60)
        if r.status_code not in (200, 202):
            sys.exit(f"Roboflow error {r.status_code}: {r.text[:300]}")
        link = (r.json().get("export") or {}).get("link")
        if link:
            return link
        print("   export is being generated...")
        time.sleep(10)
    sys.exit("export took too long, try again later")


def download(name, key):
    target = OUT / name
    if target.exists():
        print(f"{name}: already downloaded ({target})")
        return
    print(f"{name}: requesting export")
    link = export_link(*DATASETS[name], key)
    zip_path = OUT / f"{name}.zip"
    with requests.get(link, stream=True, timeout=60) as r:
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r   {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="")
    print()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    zip_path.unlink()
    print(f"   -> {target}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(DATASETS)
    key = api_key()
    for n in names:
        download(n, key)
