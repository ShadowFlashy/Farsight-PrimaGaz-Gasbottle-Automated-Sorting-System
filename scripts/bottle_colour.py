"""Colour of every line-crossing event (scripts/view_events.py): how much of the bottle is cyan.

For each event we take the track's box in a few frames around the crossing (event frame + `offsets`), keep the
central part of the box (less background), and measure the fraction of cyan pixels (HSV hue/saturation/value
ranges in the config). The median over those frames is the event's cyan_frac. Black bottles give ~0, cyan
bottles a large fraction; the threshold is chosen from the histogram that this script prints.

The video is read once from start to end (seeking is not frame-exact for these files).

Output: <events dir>/<recording>/<view>_colour.csv
usage: python scripts/bottle_colour.py 14_43 [--config configs/views.yaml]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def cyan_fraction(frame, box, c):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    # central part of the box: the bottle body, not the background next to it or the handle on top
    crop = frame[int(y1 + c["crop_y"][0] * h):int(y1 + c["crop_y"][1] * h),
                 int(x1 + c["crop_x"][0] * w):int(x1 + c["crop_x"][1] * w)]
    if crop.size == 0:
        return np.nan
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] >= c["hue"][0]) & (hsv[..., 0] <= c["hue"][1]) &
            (hsv[..., 1] >= c["min_sat"]) & (hsv[..., 2] >= c["min_val"]))
    return float(mask.mean())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("recording", help="e.g. 14_43")
    p.add_argument("--config", default="configs/views.yaml")
    a = p.parse_args()

    conf = yaml.safe_load(open(a.config))
    c, rec = conf["colour"], conf["recordings"][a.recording]
    events_dir = Path(conf["global"]["events"]) / a.recording

    for view in rec["views"]:
        with open(events_dir / f"{view}_events.csv", newline="") as f:
            events = list(csv.DictReader(f))
        boxes = {}  # (frame, track id) -> box
        with open(events_dir / f"{view}_tracks.csv", newline="") as f:
            for r in csv.DictReader(f):
                boxes[int(r["frame"]), int(r["track_id"])] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
        wanted = defaultdict(list)  # frame -> [(event index, track id)]
        for k, e in enumerate(events):
            for d in c["offsets"]:
                wanted[round(float(e["frame"])) + d].append((k, int(e["track_id"])))
        values = defaultdict(list)
        cap = cv2.VideoCapture(conf["video"].format(rec=a.recording, view=view))
        n, last = 0, max(wanted)
        while n <= last:
            if n in wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                for k, tid in wanted[n]:
                    if (n, tid) in boxes:
                        values[k].append(cyan_fraction(frame, boxes[n, tid], c))
            elif not cap.grab():  # grab without decoding the frames we don't need
                break
            n += 1
        cap.release()

        with open(events_dir / f"{view}_colour.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event", "frame", "track_id", "cyan_frac", "n_frames"])
            for k, e in enumerate(events):
                v = [x for x in values[k] if not np.isnan(x)]
                w.writerow([e["event"], e["frame"], e["track_id"], round(float(np.median(v)), 4) if v else "", len(v)])
        fr = np.array([np.median(values[k]) for k in range(len(events)) if values[k]])
        hist, edges = np.histogram(fr, bins=[0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
        print(f"{a.recording} {view}: {len(events)} events, cyan_frac histogram " +
              " ".join(f"[{lo:.2f}-{hi:.2f}):{h}" for lo, hi, h in zip(edges, edges[1:], hist)))
