"""Match the global bottle ids (scripts/global_ids.py) with the rows of the ground-truth log.

Video times are converted to real seconds first (global.video_fps / global.real_fps).
The log has one row per bottle in passing order with a time stamp (14_43: Time.1, whole seconds; 14_55: Time,
whole minutes). A log time T stands for the interval [T, T + resolution). The video has no start time, so:
1. Offset o (log time = video time + o): for every o on a grid, align the bottles with the log rows whose time
   falls inside the video (+- log_max_dt) by dynamic programming (scripts/global_ids.align, distance from video
   time + o to the log interval). Score = alignment cost: distance / log_max_dt per match, + log_colour_cost if
   the colours differ (black / cyan; unknown matches anything), + gap_cost per video bottle or log row without a
   partner (log_gap_cost). Keep the lowest cost. (Counting only matches is not enough: an offset
   could then skip log rows for free.) With minute resolution many offsets fit equally well; the script
   prints the range.
2. With that offset: matched pairs, video bottles without a log row, log rows without a video bottle (only rows
   whose time falls inside the video).

Rows with only a weight (no Primagaz_status) are not bottles and are left out; the script says how many.

Output: <global.out>/<recording>/log_match.csv
usage: python scripts/log_align.py 14_43 [--config configs/views.yaml]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from global_ids import align
from gt_log import read_log


def hms(s):
    s = round(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def dist(start, tv, res):
    """Distance from a video time to the log interval [start, start + res)."""
    return max(start - tv, 0.0, tv - start - res)


def score(tv, starts, res, max_dt, gap_cost, log_col, video_col, colour_cost, edge_gap_cost):
    """Alignment of video times tv (already + offset) with the log rows inside the video's time span, with a
    cost for a colour mismatch. Returns (cost, pairs) with pairs = (log index or None, video index or None).
    A log row whose time interval is not completely inside the video may belong to a bottle before the start or
    after the end, so leaving it out costs only edge_gap_cost."""
    inside = np.flatnonzero((starts + res >= tv.min() - max_dt) & (starts <= tv.max() + max_dt))
    edge = (starts < tv.min()) | (starts + res > tv.max())
    # among edge rows, leaving out the outermost (first / last) rows is cheapest: a bottle before the start or
    # after the end of the video, not one in between
    rank = {i: min(k, len(inside) - 1 - k) for k, i in enumerate(inside)}
    edge_cost = {i: edge_gap_cost * (1 + 0.01 * rank[i]) for i in inside}
    a_gap = [edge_cost[i] if edge[i] else gap_cost for i in inside]
    pairs = align(list(starts[inside]), list(tv), max_dt, gap_cost, a_width=res,
                  a_label=[log_col[i] for i in inside], b_label=video_col, label_cost=colour_cost, a_gap=a_gap)
    pairs = [(inside[i] if i is not None else None, j) for i, j in pairs]
    cost = 0.0
    for i, j in pairs:
        if i is None:
            cost += gap_cost
        elif j is None:
            cost += edge_cost[i] if edge[i] else gap_cost
        else:
            cost += dist(starts[i], tv[j], res) / max_dt
            cost += colour_cost if None not in (log_col[i], video_col[j]) and log_col[i] != video_col[j] else 0.0
    return cost, pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("recording", help="e.g. 14_43")
    p.add_argument("--config", default="configs/views.yaml")
    a = p.parse_args()

    conf = yaml.safe_load(open(a.config))
    g, rec = conf["global"], conf["recordings"][a.recording]
    lg = rec["log"]
    out_dir = Path(g["out"]) / a.recording

    with open(out_dir / "bottles.csv", newline="") as f:
        rows_b = list(csv.DictReader(f))
    bottles = [(int(r["global_id"]), float(r["t_ref_s"])) for r in rows_b]
    # video seconds -> real seconds (the files play slower than the cameras recorded, see configs/views.yaml)
    t = np.array([b[1] for b in bottles]) * g["video_fps"] / g["real_fps"]
    video_col = [r.get("colour") or None for r in rows_b]

    rows = read_log(lg["file"].format(rec=a.recording))
    log = [r for r in rows if r["Primagaz_status"] is not None]
    print(f"{a.recording} log: {len(rows)} rows, {len(log)} bottles, {len(rows) - len(log)} rows with only a weight "
          f"left out (xlsx rows {[r['xlsx_row'] for r in rows if r['Primagaz_status'] is None]})")
    starts = np.array([r[lg["time_column"]] for r in log], dtype=float)
    res = lg["resolution_s"]
    # log colours other than black / cyan (purple, cyan-red) are treated as unknown: they match anything
    log_col = [r["color"] if r["color"] in ("black", "cyan") else None for r in log]
    args = (res, g["log_max_dt"], g["log_gap_cost"], log_col, video_col, g["log_colour_cost"], g["log_edge_gap_cost"])

    # coarse grid over every offset that puts the video somewhere over the log, then a fine grid around the best
    step = g["log_offset_step"]
    lo, hi = starts.min() - t.max(), starts.max() + res - t.min()
    coarse = [(score(t + o, starts, *args)[0], o) for o in np.arange(lo, hi, step[0])]
    best = min(c for c, _ in coarse)
    fine = [(score(t + x, starts, *args)[0], x)
            for c, o in coarse if c <= best + 2 * g["log_gap_cost"] for x in np.arange(o - step[0], o + step[0], step[1])]
    cost, o = min(fine)
    same = [x for c, x in fine if c <= cost + 1e-6]
    second = min(c for c, x in coarse + fine if abs(x - o) > 10)
    print(f"offset: video t = 0 s is log time {hms(o)} ({o:.2f} s after midnight), cost {cost:.2f}; same cost for "
          f"{hms(min(same))} - {hms(max(same))}; best cost more than 10 s away: {second:.2f}")

    cost, pairs = score(t + o, starts, *args)
    with open(out_dir / "log_match.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_id", "real_t_s", "video_colour", "log_time", "xlsx_row", "log_id", "log_color",
                    "dt_s"])
        for i, j in pairs:
            gid, tv = (bottles[j][0], t[j]) if j is not None else ("", None)  # tv in real seconds
            r = log[i] if i is not None else None
            dt = "" if r is None or tv is None else round(dist(starts[i], tv + o, res), 2)
            w.writerow([gid, "" if tv is None else round(tv, 2), (video_col[j] or "") if j is not None else "",
                        hms(starts[i]) if r else "",
                        r["xlsx_row"] if r else "", r["id"] if r else "", r["color"] if r else "", dt])

    video_only = [bottles[j][0] for i, j in pairs if i is None]
    log_only = [log[i] for i, j in pairs if j is None]
    matched = [(i, j) for i, j in pairs if i is not None and j is not None]
    d = np.array([dist(starts[i], t[j] + o, res) for i, j in matched])
    print(f"video: {len(bottles)} bottles | log rows inside the video's time span: {len(matched) + len(log_only)} | "
          f"matched: {len(matched)} "
          f"(distance to the log time: median {np.median(d):.2f} s, max {d.max():.2f} s)")
    col = [(video_col[j], log_col[i]) for i, j in matched if None not in (video_col[j], log_col[i])]
    print(f"colour: {len(col)} matches with a known colour on both sides, {sum(x != y for x, y in col)} differ")
    print(f"video bottles without a log row: {len(video_only)} {video_only}")
    print(f"log rows inside the video without a video bottle: {len(log_only)} "
          f"{[(r['xlsx_row'], r['id'], hms(r[lg['time_column']])) for r in log_only]}")
    ids = [log[i]["id"] for i, j in matched if log[i]["id"] is not None]
    print(f"log ids of the matched bottles: {ids[0]:.0f} ... {ids[-1]:.0f}, increasing: "
          f"{all(x < y for x, y in zip(ids, ids[1:]))} | matched rows without a log id: "
          f"{[log[i]['xlsx_row'] for i, j in matched if log[i]['id'] is None]}")
    print(f"-> {out_dir / 'log_match.csv'}")
