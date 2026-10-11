"""Cut a short clip out of a video (short videos keep tracking experiments fast).

usage: python scripts/cut_clip.py <video> <start m:ss> <end m:ss> <out.mp4>
e.g.   python scripts/cut_clip.py data/videos/UH-u0g3B8Pw.mp4 2:28 2:48 data/videos/petrogal_entry.mp4
"""
import sys

import cv2


def seconds(t):
    m, s = t.split(":")
    return int(m) * 60 + float(s)


video, start, end, out = sys.argv[1], seconds(sys.argv[2]), seconds(sys.argv[3]), sys.argv[4]
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
for _ in range(int((end - start) * fps)):
    ok, frame = cap.read()
    if not ok:
        break
    writer.write(frame)
writer.release()
print(f"{out}: {end - start:.0f} s")
