#!/usr/bin/env python3
"""
Evaluate the saved locator under live-play conditions on the TEST rounds.

Every test panorama is rendered into the 12 perspective screenshots the live script takes
(pitch -55/0/55 x 4 yaws, hfov 120, random start yaw), rebuilt with
SphericalImage.from_views (true heading known from the compass, car axis unknown) and
analysed by engine.locator.Locator exactly as in play.

  python3 tools/eval_live.py [--split test] [--n 0] [--workers 4]
"""
import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from calibrate import DATASET, load_records, split_of  # noqa: E402

_LOC = None


def render_views(sph, start_yaw, hfov=120.0, size=(960, 640)):
    views = []
    for pitch in (-55.0, 0.0, 55.0):
        for k in range(4):
            yaw = (start_yaw + 90.0 * k) % 360.0
            rel = (yaw - sph.heading + 180.0) % 360.0 - 180.0
            views.append({"image": sph.render_view(rel, pitch, hfov, size), "yaw": yaw, "pitch": pitch, "hfov": hfov})
    return views


def _work(r):
    global _LOC
    from engine.geo import geoguessr_score, haversine_km
    from engine.locator import Locator
    from engine.panorama import SphericalImage
    if _LOC is None:
        _LOC = Locator()
    sph = SphericalImage.from_equirect(os.path.join(DATASET, "panos", r["pano_id"] + ".jpg"), heading=r["heading"])
    rng = random.Random(r["pano_id"])
    res = _LOC.analyze_views(render_views(sph, rng.uniform(0, 360)))
    codes = [c["code"] for c in res["countries"]]
    d = float(haversine_km(r["lat"], r["lng"], res["guess"]["lat"], res["guess"]["lng"]))
    return {"label": r["label"], "codes": codes, "km": d, "points": float(geoguessr_score(d)),
            "ms": res["timing_ms"]["total"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    recs = [r for r in load_records() if split_of(r) == args.split and r.get("heading") is not None]
    if args.n:
        recs = recs[: args.n]
    with Pool(args.workers) as pool:
        out = pool.map(_work, recs, chunksize=2)
    rank = [o["codes"].index(o["label"]) if o["label"] in o["codes"] else 99 for o in out]
    res = {"split": args.split, "mode": "12 rendered views, car axis unknown", "n": len(out),
           "top1": float(np.mean([k == 0 for k in rank])), "top3": float(np.mean([k < 3 for k in rank])),
           "top5": float(np.mean([k < 5 for k in rank])), "mean_score": float(np.mean([o["points"] for o in out])),
           "median_km": float(np.median([o["km"] for o in out])),
           "analysis_ms": float(np.median([o["ms"] for o in out]))}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, "data", "model", "eval_live.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
