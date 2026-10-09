#!/usr/bin/env python3
"""
Evaluate the saved locator under live-play conditions on the TEST rounds.

Every test panorama is rendered into the perspective screenshots the live script takes on its
default 1112x740 canvas (play_live_visual.js planGrid at zoom 0: pitch -40/+40 x 5 yaws,
hfov 112.7, vfov 90, random start yaw), rebuilt with
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
STABILITY = False
WITH_MAP = False


def render_views(sph, start_yaw, hfov=112.715, size=(1112, 740)):
    views = []
    for pitch in (-40.0, 40.0):
        for k in range(5):
            yaw = (start_yaw + 72.0 * k) % 360.0
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
    mp = r.get("map") if WITH_MAP else None   # the map name the player sees
    res = _LOC.analyze_views(render_views(sph, rng.uniform(0, 360)), map_info=mp)
    codes = [c["code"] for c in res["countries"]]
    d = float(haversine_km(r["lat"], r["lng"], res["guess"]["lat"], res["guess"]["lng"]))
    from engine.geo import region_at
    reg = region_at(r["lat"], r["lng"])
    top_regions = [x["code"] for x in res["hints"][0]["regions"]] if res["hints"] else []
    from engine.geo import geoguessr_points, resolve_map
    rm = resolve_map(r.get("map")) or {}
    pts_map = geoguessr_points(d, rm["maxErrorDistance"]) if rm.get("maxErrorDistance") else float(geoguessr_score(d))
    out = {"label": r["label"], "codes": codes, "km": d, "points": float(geoguessr_score(d)), "points_map": float(pts_map),
           "ms": res["timing_ms"]["total"],
           "region_rank": top_regions.index(reg[0]) if reg and reg[0] in top_regions else 99}
    if STABILITY:  # a second capture of the same place from another start direction
        res2 = _LOC.analyze_views(render_views(sph, rng.uniform(0, 360)), map_info=mp)
        out["same_top"] = res2["countries"][0]["code"] == codes[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stability", action="store_true", help="also capture each place twice")
    ap.add_argument("--map", action="store_true", help="pass each round's map (as in the game) to the locator")
    args = ap.parse_args()
    global STABILITY, WITH_MAP
    STABILITY, WITH_MAP = args.stability, args.map
    recs = [r for r in load_records() if split_of(r) == args.split and r.get("heading") is not None]
    if args.n:
        recs = recs[: args.n]
    with Pool(args.workers) as pool:
        out = pool.map(_work, recs, chunksize=2)
    rank = [o["codes"].index(o["label"]) if o["label"] in o["codes"] else 99 for o in out]
    res = {"split": args.split, "mode": "10 rendered views (live grid 2x5, 112.7x90 deg), car axis unknown", "n": len(out),
           "top1": float(np.mean([k == 0 for k in rank])), "top3": float(np.mean([k < 3 for k in rank])),
           "top5": float(np.mean([k < 5 for k in rank])), "mean_score": float(np.mean([o["points"] for o in out])),
           "mean_points_map_formula": float(np.mean([o["points_map"] for o in out])), "with_map": WITH_MAP,
           "median_km": float(np.median([o["km"] for o in out])),
           "region_top1_if_country_right": float(np.mean([o["region_rank"] == 0 for o, k in zip(out, rank) if k == 0] or [0])),
           "region_top3_if_country_right": float(np.mean([o["region_rank"] < 3 for o, k in zip(out, rank) if k == 0] or [0])),
           "analysis_ms": float(np.median([o["ms"] for o in out]))}
    if STABILITY:
        res["same_top_country_two_captures"] = float(np.mean([o["same_top"] for o in out]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, "data", "model", "eval_live.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
