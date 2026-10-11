"""Download freely licensed gas bottle photos from Wikimedia Commons (for labeling in CVAT / Roboflow).

Images go to data/commons/images, license + author info per image goes to data/commons/attribution.csv
(we need this to credit the photographers in the report). Afterwards we sorted the images by hand into
images/ other/ rejected/ (see docs/DATA.md), the decision is stored in attribution.csv.

The images are already in the repo, so you only need this script to download everything again.
usage: python scripts/download_commons.py
"""
import csv
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "commons"
API = "https://commons.wikimedia.org/w/api.php"
UA = "Farsight-PrimaGaz-student-project/0.1 (https://github.com/ShadowFlashy/Farsight-PrimaGaz-Gasbottle-Automated-Sorting-System)"
WIDTH = 1280  # images are downscaled to max 1280px wide, plenty for YOLO (imgsz 640)

# categories with LPG style bottles (like the Primagaz ones). Subcategories are followed unless excluded.
INCLUDE = [
    "Liquefied petroleum gas cylinders",
    "Repsol gas cylinders",
    "Gas cylinders",  # mixed category, only its own files + the country subcategories below
    "Gas cylinders in Germany",
    "Gas cylinders in India",
    "Gas cylinders in Israel",
    "Gas cylinders in Nepal",
    "Gas cylinders in Russia",
    "Gas cylinders in Ukraine",
]
# not the kind of bottle Primagaz handles (diving, welding, medical, camping cartridges, drawings...)
EXCLUDE = {
    "SVG gas cylinders", "Acetylene cylinders", "Carbon dioxide cylinders", "Diving cylinders",
    "Exploded gas cylinders", "Gas cartridge", "Gas cylinders with Lindal B188 valve", "Helium cylinders",
    "Hydrogen cylinders", "Oxygen cylinders", "Pierceable gas cylinders", "Repurposed gas cylinders",
    "Scuba cylinders", "Monument of GAZ-52/GAZ-53 (Tyumen)", "190g Propane disposable cylinders",
    "450g Propane disposable cylinders", "Fixed propane tanks", "Butane torches",
}
NO_RECURSE = {"Gas cylinders"}  # its subcategories are handled through INCLUDE / EXCLUDE
IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/tiff": ".jpg"}


def get(url, params=None, retries=5):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # rate limit / server hiccup -> wait and retry
            if attempt == retries - 1:
                raise
            print(f"   retry ({e})")
            time.sleep(2 ** attempt * 2)


def api(**params):
    return json.loads(get(API, {"format": "json", "formatversion": 2, **params}))


def category_files(cat, seen_cats, depth=0):
    """All files in a category (+ subcategories, max 2 levels deep) -> {title: category}."""
    if cat in seen_cats or cat in EXCLUDE:
        return {}
    seen_cats.add(cat)
    files, cont = {}, {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle="Category:" + cat, cmlimit=500,
                cmtype="file|subcat", **cont)
        for m in d["query"]["categorymembers"]:
            if m["ns"] == 6:
                files.setdefault(m["title"], cat)
            elif m["ns"] == 14 and depth < 2 and cat not in NO_RECURSE:
                sub = m["title"].removeprefix("Category:")
                for t, c in category_files(sub, seen_cats, depth + 1).items():
                    files.setdefault(t, c)
        if "continue" not in d:
            return files
        cont = d["continue"]


def clean(text):
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def main():
    if (OUT / "attribution.csv").exists():  # don't overwrite our review decisions
        print(f"already downloaded, see {OUT / 'attribution.csv'} (delete data/commons to download again)")
        return
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    seen_cats, files = set(), {}
    for cat in INCLUDE:
        for t, c in category_files(cat, seen_cats).items():
            files.setdefault(t, c)
    print(f"{len(files)} files found in {len(seen_cats)} categories")

    rows, skipped = [], []
    titles = sorted(files)
    for i in range(0, len(titles), 50):
        d = api(action="query", titles="|".join(titles[i:i + 50]), prop="imageinfo",
                iiprop="url|size|mime|extmetadata", iiurlwidth=WIDTH)
        for p in d["query"]["pages"]:
            info = (p.get("imageinfo") or [None])[0]
            if not info or info["mime"] not in IMAGE_TYPES:
                skipped.append((p["title"], info["mime"] if info else "?"))
                continue
            meta = info.get("extmetadata", {})
            url = info.get("thumburl") or info["url"]
            ext = Path(urllib.parse.urlparse(url).path).suffix.lower() or IMAGE_TYPES[info["mime"]]
            name = f"c{p['pageid']}{'.jpg' if ext == '.jpeg' else ext}"
            target = OUT / "images" / name
            if not target.exists():
                target.write_bytes(get(url))
                time.sleep(0.2)  # be nice to the Wikimedia servers
            rows.append({
                "file": name,
                "title": p["title"],
                "page": info["descriptionurl"],
                "author": clean(meta.get("Artist", {}).get("value")),
                "license": clean(meta.get("LicenseShortName", {}).get("value")),
                "license_url": clean(meta.get("LicenseUrl", {}).get("value")),
                "category": files[p["title"]],
                "orig_width": info["width"],
                "orig_height": info["height"],
            })
        print(f"   {len(rows)} images downloaded")

    with open(OUT / "attribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["file"]))
    print(f"done: {len(rows)} images, {len(skipped)} skipped (svg/video/pdf...)")
    for t, m in skipped:
        print(f"   skipped {m}: {t}")


if __name__ == "__main__":
    main()
