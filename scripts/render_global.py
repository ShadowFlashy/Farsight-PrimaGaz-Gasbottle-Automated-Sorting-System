"""Video of all views of a recording side by side, every bottle labelled with its global id.

Uses the saved tracks (scripts/view_events.py) and the track -> global id map (scripts/global_ids.py), so no
model is run. A track that never crossed the counting line has no global id and is drawn thin and grey. The
counting line of each view is drawn in yellow. Views are ordered by delay (the order a bottle passes them).

usage: python scripts/render_global.py 14_43 [--start 0] [--seconds 60] [--width 900]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def colour(gid):
    """A fixed, clearly different colour per global id (BGR)."""
    hue = (gid * 47) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("recording", help="e.g. 14_43")
    p.add_argument("--config", default="configs/views.yaml")
    p.add_argument("--start", type=float, default=0, help="start time in video seconds")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--width", type=int, default=900, help="width of one view in the output")
    a = p.parse_args()

    conf = yaml.safe_load(open(a.config))
    g, rec = conf["global"], conf["recordings"][a.recording]
    events_dir, out_dir = Path(g["events"]) / a.recording, Path(g["out"]) / a.recording
    views = sorted(rec["views"], key=lambda v: rec["views"][v].get("delay_s", 0))

    gid_of = {}
    with open(out_dir / "track_map.csv", newline="") as f:
        for r in csv.DictReader(f):
            gid_of[r["view"], int(r["track_id"])] = int(r["global_id"])

    caps = {v: cv2.VideoCapture(conf["video"].format(rec=a.recording, view=v)) for v in views}
    fps = caps[views[0]].get(cv2.CAP_PROP_FPS)
    W, H = int(caps[views[0]].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[views[0]].get(cv2.CAP_PROP_FRAME_HEIGHT))
    first, last = round(a.start * fps), round((a.start + a.seconds) * fps)

    boxes = {v: defaultdict(list) for v in views}  # view -> frame -> [(track id, box)]
    for v in views:
        with open(events_dir / f"{v}_tracks.csv", newline="") as f:
            for r in csv.DictReader(f):
                n = int(r["frame"])
                if first <= n < last:
                    boxes[v][n].append((int(r["track_id"]), [int(float(r[k])) for k in ("x1", "y1", "x2", "y2")]))

    scale = a.width / W
    out = out_dir / f"demo_{a.start:.0f}s_{a.seconds:.0f}s.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (a.width * len(views), round(H * scale)))
    for v in views:  # skip to the start frame (grab does not decode, seeking is not frame-exact in these files)
        for _ in range(first):
            caps[v].grab()
    for n in range(first, last):
        panels = []
        for v in views:
            ok, frame = caps[v].read()
            if not ok:
                break
            x = int(rec["views"][v]["line"] * W)
            cv2.line(frame, (x, 0), (x, H), (0, 255, 255), 6)
            for tid, (x1, y1, x2, y2) in boxes[v][n]:
                gid = gid_of.get((v, tid))
                if gid is None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 3)
                    continue
                c = colour(gid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 10)
                cv2.rectangle(frame, (x1, y1), (x1 + 330, y1 + 110), c, -1)
                cv2.putText(frame, str(gid), (x1 + 15, y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 0), 8)
            cv2.rectangle(frame, (0, H - 110), (W, H), (0, 0, 0), -1)
            cv2.putText(frame, f"{v}  frame {n}  t {n / fps:.1f} s", (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                        (255, 255, 255), 5)
            panels.append(cv2.resize(frame, (a.width, round(H * scale)), interpolation=cv2.INTER_AREA))
        if len(panels) < len(views):
            break
        writer.write(cv2.hconcat(panels))
    writer.release()
    print(out)
