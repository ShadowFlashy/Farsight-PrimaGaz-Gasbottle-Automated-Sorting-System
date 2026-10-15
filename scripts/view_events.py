"""Line-crossing events per camera view, to match bottles across the views of one recording later.

For every view of a recording: run YOLO + ByteTrack on the first `seconds` of the video and save every tracked
box (<view>_tracks.csv). Then find the moments a track's box center crosses the counting line in the expected
direction (<view>_events.csv). Only the first crossing of a track counts, and only tracks seen in at least
`min_frames` frames. Lines, directions and settings are in configs/views.yaml.

usage: python scripts/view_events.py 14_43 [--config configs/views.yaml] [--views front back_left]
       python scripts/view_events.py 14_43 --from-tracks           (only redo the events, e.g. after moving a line)
       python scripts/view_events.py 14_43 --from-tracks --sweep   (events for line positions 0.10 ... 0.90)
       python scripts/view_events.py 14_43 --seconds 0 --out runs/events_full   (the whole video)
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

TRACK_FIELDS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]
EVENT_FIELDS = ["event", "frame", "time_s", "track_id", "track_frames", "conf"]


def track_view(video, cfg, out_csv):
    """Track the first cfg['seconds'] of a video (0 = all), write one row per tracked box. Returns (fps, width)."""
    fps, width = video_info(video)
    max_frames = round(cfg["seconds"] * fps) or float("inf")
    model = YOLO(cfg["model"])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRACK_FIELDS)
        for n, r in enumerate(model.track(str(video), tracker=cfg["tracker"], conf=cfg["conf"], stream=True,
                                          persist=True, verbose=False)):
            if n >= max_frames:
                break
            if r.boxes.id is not None:
                for tid, box, c in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    w.writerow([n, tid, *(round(v, 1) for v in box), round(c, 3)])
    return fps, width


def read_tracks(tracks_csv):
    """track id -> list of (frame, center x, conf), sorted by frame."""
    tracks = defaultdict(list)
    with open(tracks_csv, newline="") as f:
        for row in csv.DictReader(f):
            cx = (float(row["x1"]) + float(row["x2"])) / 2
            tracks[int(row["track_id"])].append((int(row["frame"]), cx, float(row["conf"])))
    return {tid: sorted(obs) for tid, obs in tracks.items()}


def crossings(tracks, line_x, direction, min_frames, fps):
    """First crossing of the line per track, in the given direction, sorted by time."""
    sign = 1 if direction == "left_to_right" else -1
    events = []
    for tid, obs in tracks.items():
        if len(obs) < min_frames:
            continue
        for (f0, x0, _), (f1, x1, c1) in zip(obs, obs[1:]):
            if sign * (x0 - line_x) < 0 <= sign * (x1 - line_x):  # was before the line, now on or after it
                frame = f0 + (f1 - f0) * (line_x - x0) / (x1 - x0) if x1 != x0 else f1  # interpolate
                events.append({"frame": round(frame, 2), "time_s": round(frame / fps, 3), "track_id": tid,
                               "track_frames": len(obs), "conf": c1})
                break
    events.sort(key=lambda e: e["frame"])
    for i, e in enumerate(events):
        e["event"] = i
    return events


def write_events(events, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        w.writeheader()
        w.writerows(events)


def video_info(video):
    cap = cv2.VideoCapture(str(video))
    fps, width = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return fps, width


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("recording", help="e.g. 14_43")
    p.add_argument("--config", default="configs/views.yaml")
    p.add_argument("--views", nargs="*", help="only these views (default: all views in the config)")
    p.add_argument("--from-tracks", action="store_true", help="reuse the saved <view>_tracks.csv, no tracking")
    p.add_argument("--sweep", action="store_true", help="print the number of events for lines at 0.10 ... 0.90")
    p.add_argument("--seconds", type=float, help="override events.seconds from the config (0 = whole video)")
    p.add_argument("--out", help="override events.out from the config")
    a = p.parse_args()

    conf = yaml.safe_load(open(a.config))
    cfg, rec = conf["events"], conf["recordings"][a.recording]
    if a.seconds is not None:
        cfg["seconds"] = a.seconds
    if a.out:
        cfg["out"] = a.out
    out_dir = Path(cfg["out"]) / a.recording
    out_dir.mkdir(parents=True, exist_ok=True)

    for view in a.views or rec["views"]:
        vcfg = rec["views"][view]
        video = Path(conf["video"].format(rec=a.recording, view=view))
        tracks_csv = out_dir / f"{view}_tracks.csv"
        if a.from_tracks:
            fps, width = video_info(video)
        else:
            fps, width = track_view(video, cfg, tracks_csv)
        tracks = read_tracks(tracks_csv)

        if a.sweep:
            counts = {x: len(crossings(tracks, x * width, vcfg["direction"], cfg["min_frames"], fps))
                      for x in np.round(np.arange(0.10, 0.91, 0.05), 2)}
            print(f"{a.recording} {view} events per line position: " + "  ".join(f"{x:.2f}:{n}" for x, n in counts.items()))

        events = crossings(tracks, vcfg["line"] * width, vcfg["direction"], cfg["min_frames"], fps)
        write_events(events, out_dir / f"{view}_events.csv")
        gaps = np.diff([e["time_s"] for e in events])
        gap_text = (f" | gap between events: min {gaps.min():.1f} s, median {np.median(gaps):.1f} s, "
                    f"max {gaps.max():.1f} s" if len(gaps) else "")
        print(f"{a.recording} {view}: {len(tracks)} track ids, {len(events)} events "
              f"(line {vcfg['line']}, {vcfg['direction']}){gap_text}")
