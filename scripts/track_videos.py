"""Track objects in a video with YOLO + a tracker (ByteTrack) and count them. The result video has the boxes, the ids
and the count on it.

2 ways of counting:
- unique ids: every object gets a track id, we count the ids that were seen in at least --min-frames frames
  (otherwise a box that flickers for 1 frame would also count). Every id switch = 1 bottle counted too many.
- line crossing (--line 0.5): only count ids whose center crosses a vertical line (here at 50% of the width).
  This is how a counter on a conveyor would work: every bottle passes the line once.

usage: python scripts/track_videos.py <video> [--model yolo11n.pt] [--classes 0 2] [--line 0.44] [--out runs/track_demo]
e.g.   python scripts/track_videos.py data/videos/flexspeed_workers.mp4 --classes 0              (COCO model, 0 = person)
       python scripts/track_videos.py data/videos/petrogal_checkweigher.mp4 --model runs/combined_y11n/weights/best.pt --line 0.44
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--model", default="yolo11n.pt")  # yolo11s.pt
    p.add_argument("--classes", type=int, nargs="*", help="only these class ids, e.g. 0 = person in COCO")
    p.add_argument("--tracker", default="bytetrack.yaml")  # botsort.yaml
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--min-frames", type=int, default=10, help="an id must be seen this many frames to be counted")
    p.add_argument("--line", type=float, help="count objects crossing a vertical line at this fraction of the width")
    p.add_argument("--out", default="runs/track_demo")
    a = p.parse_args()

    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    m = Path(a.model)
    model_name = m.parent.parent.name if m.parent.name == "weights" else m.stem
    out = Path(a.out) / f"{Path(a.video).stem}_{model_name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_x = a.line * w if a.line is not None else None
    seen = Counter()  # track id -> number of frames it was seen
    last_x = {}  # track id -> center x in the previous frame it was seen
    crossed = []  # (track id, time in s) of the objects that crossed the line
    times = []
    for n, r in enumerate(model.track(a.video, tracker=a.tracker, classes=a.classes, conf=a.conf, stream=True,
                                      persist=True, verbose=False)):
        if r.boxes.id is not None:
            for tid, (x1, _, x2, _) in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist()):
                seen[tid] += 1
                cx = (x1 + x2) / 2
                if line_x is not None and tid in last_x and (last_x[tid] - line_x) * (cx - line_x) < 0 \
                        and tid not in [c[0] for c in crossed]:  # changed side of the line
                    crossed.append((tid, n / fps))
                last_x[tid] = cx
        times.append(sum(r.speed.values()))  # preprocess + inference + postprocess in ms
        frame = r.plot(line_width=2)
        if line_x is not None:
            text = f"crossed the line: {len(crossed)}"
            cv2.line(frame, (int(line_x), 0), (int(line_x), h), (0, 255, 255), 3)
        else:
            text = f"objects counted: {sum(1 for k in seen.values() if k >= a.min_frames)}"
        cv2.rectangle(frame, (0, 0), (430, 50), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        writer.write(frame)
    writer.release()

    counted = sum(1 for k in seen.values() if k >= a.min_frames)
    print(f"{out}")
    print(f"frames: {len(times)} | ids: {len(seen)} | counted (>= {a.min_frames} frames): {counted} | "
          f"avg {sum(times) / len(times):.1f} ms per frame ({1000 * len(times) / sum(times):.0f} fps)")
    if line_x is not None:
        print(f"crossed the line: {len(crossed)} at " + ", ".join(f"{t:.1f}s (id {i})" for i, t in crossed))
