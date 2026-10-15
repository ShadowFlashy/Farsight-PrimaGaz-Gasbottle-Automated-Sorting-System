"""Give every bottle one global id over all views of a recording, from the line-crossing events of
scripts/view_events.py.

Bottles pass in single file, so every view sees the same sequence of line crossings, shifted by a delay.
1. Delay per view: the offset (scan over `delay_scan`) that puts the view's events closest to the reference
   view's events (median distance to the nearest event).
2. Match by order: the events of each view are aligned with the bottles found so far by dynamic programming
   (like Needleman-Wunsch). A match costs |time difference| / max_dt and is not allowed above max_dt, a missed
   or extra event costs gap_cost. So one missed or double detection in a view does not shift the rest.
   Views are added one by one (reference first, then by increasing delay); unmatched events become new bottles,
   so a bottle seen in only one view still gets an id.
3. Global id = rank of the bottle's time on the reference line, so ids increase with the passing order.

Colour (if scripts/bottle_colour.py was run): median cyan fraction over the views -> cyan / black / unknown.

Output in <global.out>/<recording>/: bottles.csv (one row per bottle) and track_map.csv (view, track id ->
global id; every track that created an event).

usage: python scripts/global_ids.py 14_43 [--config configs/views.yaml]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def read_events(path):
    with open(path, newline="") as f:
        return [{"event": int(r["event"]), "time_s": float(r["time_s"]), "frame": float(r["frame"]),
                 "track_id": int(r["track_id"])} for r in csv.DictReader(f)]


def read_colour(path):
    """event -> cyan fraction (scripts/bottle_colour.py), or {} if the file is not there."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {int(r["event"]): float(r["cyan_frac"]) for r in csv.DictReader(f) if r["cyan_frac"]}


def colour_label(cyan_frac, c):
    if cyan_frac is None or np.isnan(cyan_frac):
        return None
    return "cyan" if cyan_frac >= c["cyan_min"] else "black" if cyan_frac <= c["black_max"] else None


def estimate_delay(ref_times, times, scan, edge):
    """Offset d (s) so that times - d lies closest to ref_times. Returns (d, median residual)."""
    lo, hi = ref_times.min() + edge, ref_times.max() - edge
    best = (np.inf, 0.0)
    for d in np.arange(scan[0], scan[1] + scan[2] / 2, scan[2]):
        shifted = times - d
        shifted = shifted[(shifted > lo) & (shifted < hi)]  # skip the edges, where a view may miss events
        if len(shifted):
            res = np.median(np.abs(ref_times[None, :] - shifted[:, None]).min(axis=1))
            best = min(best, (res, round(float(d), 3)))
    return best[1], best[0]


def align(a, b, max_dt, gap_cost, a_width=0.0, a_label=None, b_label=None, label_cost=0.0, a_gap=None):
    """Order-preserving alignment of two sorted time lists. Returns a list of (i, j) with i or j None for a gap.
    a_width > 0: every a[i] is an interval [a[i], a[i] + a_width) (e.g. a log time with minute resolution).
    a_label / b_label: optional labels (e.g. colour); a match with two different labels costs label_cost extra,
    a label None (unknown) matches anything.
    a_gap: optional cost per a[i] for leaving it without a partner (default gap_cost)."""
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, :] = np.arange(m + 1) * gap_cost
    a_gap = [gap_cost] * n if a_gap is None else a_gap
    cost[:, 0] = np.concatenate([[0.0], np.cumsum(a_gap)])
    step = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 match, 1 gap in b (skip a[i]), 2 gap in a (skip b[j])
    step[1:, 0], step[0, 1:] = 1, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dt = max(a[i - 1] - b[j - 1], 0.0, b[j - 1] - a[i - 1] - a_width)
            la, lb = (a_label[i - 1] if a_label else None), (b_label[j - 1] if b_label else None)
            extra = label_cost if la is not None and lb is not None and la != lb else 0.0
            options = (cost[i - 1, j - 1] + dt / max_dt + extra if dt <= max_dt else np.inf,
                       cost[i - 1, j] + a_gap[i - 1], cost[i, j - 1] + gap_cost)
            step[i, j] = int(np.argmin(options))
            cost[i, j] = options[step[i, j]]
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        s = step[i, j]
        if s == 0:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif s == 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("recording", help="e.g. 14_43")
    p.add_argument("--config", default="configs/views.yaml")
    a = p.parse_args()

    conf = yaml.safe_load(open(a.config))
    g, rec = conf["global"], conf["recordings"][a.recording]
    events_dir = Path(g["events"]) / a.recording
    ref = rec["reference_view"]
    events = {v: read_events(events_dir / f"{v}_events.csv") for v in rec["views"]}
    ref_times = np.array([e["time_s"] for e in events[ref]])

    delays = {ref: 0.0}
    for v in rec["views"]:
        if v != ref:
            d, res = estimate_delay(ref_times, np.array([e["time_s"] for e in events[v]]), g["delay_scan"], g["edge_s"])
            delays[v] = d
            print(f"{a.recording} {v}: delay {d:+.2f} s vs {ref} (median residual {res:.3f} s; "
                  f"config delay_s {rec['views'][v].get('delay_s')})")

    # bottles: list of {"t": time on the reference line, "views": {view: event}}
    bottles = []
    for v in sorted(rec["views"], key=lambda v: delays[v]):
        times = [e["time_s"] - delays[v] for e in events[v]]
        pairs = align([b["t"] for b in bottles], times, g["max_dt"], g["gap_cost"])
        merged = []
        for i, j in pairs:
            b = bottles[i] if i is not None else {"views": {}}
            if j is not None:
                b["views"][v] = events[v][j]
            b["t"] = float(np.mean([e["time_s"] - delays[w] for w, e in b["views"].items()]))
            merged.append(b)
        bottles = sorted(merged, key=lambda b: b["t"])  # keeps the order; a gap can't reorder by > max_dt

    colours = {v: read_colour(events_dir / f"{v}_colour.csv") for v in rec["views"]}
    for b in bottles:
        fr = [colours[v][e["event"]] for v, e in b["views"].items() if e["event"] in colours[v]]
        b["cyan_frac"] = float(np.median(fr)) if fr else np.nan
        b["colour"] = colour_label(b["cyan_frac"], conf["colour"])

    views = list(rec["views"])
    out_dir = Path(g["out"]) / a.recording
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bottles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_id", "t_ref_s", "n_views", "cyan_frac", "colour"] +
                   [f"{v}_{k}" for v in views for k in ("track_id", "time_s")])
        for gid, b in enumerate(bottles, 1):
            w.writerow([gid, round(b["t"], 3), len(b["views"]),
                        "" if np.isnan(b["cyan_frac"]) else round(b["cyan_frac"], 4), b["colour"] or ""] +
                       [x for v in views for x in ((b["views"][v]["track_id"], b["views"][v]["time_s"])
                                                   if v in b["views"] else ("", ""))])
    with open(out_dir / "track_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["view", "track_id", "global_id"])
        for gid, b in enumerate(bottles, 1):
            for v, e in b["views"].items():
                w.writerow([v, e["track_id"], gid])

    n_views = np.array([len(b["views"]) for b in bottles])
    print(f"{a.recording}: {len(bottles)} bottles | seen in " +
          ", ".join(f"{k} views: {(n_views == k).sum()}" for k in range(len(views), 0, -1)) + " | colour: " +
          ", ".join(f"{k}: {sum(b['colour'] == k for b in bottles)}" for k in ("black", "cyan", None)))
    for v in views:  # colour agreement between a view and the bottle's colour (median over the views)
        lab = [(colour_label(colours[v].get(b["views"][v]["event"], np.nan), conf["colour"]), b["colour"])
               for b in bottles if v in b["views"]]
        print(f"  {v}: colour differs from the bottle's colour on {sum(x != y for x, y in lab)} of {len(lab)}")
    for v in views:
        if v == ref:
            continue
        r = np.array([(b["views"][v]["time_s"] - delays[v]) - (b["views"][ref]["time_s"])
                      for b in bottles if v in b["views"] and ref in b["views"]])
        q = np.array_split(r, 4)
        print(f"  {v} - {ref} after delay: n {len(r)}, residual median {np.median(r):+.3f} s, "
              f"max |res| {np.abs(r).max():.3f} s | median per quarter of the video: "
              + " ".join(f"{np.median(x):+.3f}" for x in q if len(x)))
    for b_gid, b in enumerate(bottles, 1):
        off = {v: e["time_s"] - delays[v] - b["views"][ref]["time_s"] for v, e in b["views"].items()
               if v != ref and ref in b["views"]}
        if any(abs(x) > g["review_dt"] for x in off.values()):
            print(f"  review: global id {b_gid} at {b['t']:.1f} s, off from the usual delay: "
                  + ", ".join(f"{v} {x:+.2f} s" for v, x in off.items()))
        if len(b["views"]) < len(views):
            print(f"  global id {b_gid} at {b['t']:.1f} s: only in {sorted(b['views'])}")
    print(f"-> {out_dir}")
