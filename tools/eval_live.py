#!/usr/bin/env python3
"""
Evaluate the saved locator under live-play conditions on held-out rounds.

Every panorama is rendered into the perspective screenshots the live script takes, rebuilt with
SphericalImage.from_views (true heading known from the compass, car axis unknown) and analysed by
engine.locator.Locator (country model, cards, region model, hints) exactly as in play.

  python3 tools/eval_live.py [--split test] [--n 0] [--workers 4] [--map] [--stability]
      the user's own World-map rounds (calib / test); writes data/model/eval_live.json
  python3 tools/eval_live.py --split world_test [--grid user|headless] [--no-map] [--out F] [--paired OTHER.json]
      the classic games the locator played on the official World map (tools/world_play.py, world_calib /
      world_test = halves by game): map info = the World map as the page sends it (id, slug, name, bounds,
      maxErrorDistance), points with the World formula, country top-k and log P(true country), region / area given
      the right country, 95% intervals by game (bootstrap); per-round rows kept for paired comparisons (--paired: the
      same rounds of an earlier output, e.g. another model).  Writes scratch/world/eval/eval_live_<split>[...].json
      (never data/model).  This is the No Move condition from the stored round panoramas; the Moving condition
      (several panoramas fused) is scored on the exact live captures: tools/world_play.py replay --server <url>.

Grids: user = the default 1112x740 canvas of the user's browser (play_live_visual.js planGrid at zoom 0: pitch
-40/+40 x 5 yaws, hfov 112.7, vfov 90); headless = the 1400x800 canvas of the headless world_play games (pitch -40/+40
x 4 yaws, hfov 120.5, vfov 90).  Random start yaw per panorama (seeded by its id).
"""
import argparse
import json
import math
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _tree_arg(argv):
    """--tree DIR (a copy of engine/ + data/, e.g. scratch/world/tree_* of tools/world_play.py --snapshot): the engine
    package and model files are imported from DIR before anything imports them from the working tree."""
    for i, a in enumerate(argv):
        if a == "--tree" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
        if a.startswith("--tree="):
            return os.path.abspath(a.split("=", 1)[1])
    return None


TREE = _tree_arg(sys.argv) if __name__ == "__main__" else None
if TREE:
    sys.path.insert(0, TREE)
    import engine  # noqa: E402  (the tree's package; later `from engine import ...` resolve inside it)
    if os.path.dirname(os.path.dirname(os.path.abspath(engine.__file__))) != TREE:
        sys.exit("--tree %s: engine imported from %s" % (TREE, engine.__file__))
from calibrate import DATASET, load_records, split_of  # noqa: E402

_LOC = None
STABILITY = False
WITH_MAP = False
GRID = "user"
MAP_OVERRIDE = None

# the official World map as the page's whitelisted metadata gives it in a classic game (scratch/world/live/*/meta.json)
WORLD_LIVE_MAP = {"id": "52d15ff6850ffb3b847cb3b4", "slug": "world", "name": "World",
                  "bounds": {"min": {"lat": -54.853378, "lng": -159.778792},
                             "max": {"lat": 78.655743, "lng": 178.4508514404297}},
                  "maxErrorDistance": 14916862}
WORLD_SPLITS = ("world_calib", "world_test")
GRIDS = {"user": {"yaws": 5, "hfov": 112.715, "size": (1112, 740)},
         "headless": {"yaws": 4, "hfov": 120.51, "size": (1400, 800)}}


def render_views(sph, start_yaw, hfov=112.715, size=(1112, 740), yaws=5):
    views = []
    for pitch in (-40.0, 40.0):
        for k in range(yaws):
            yaw = (start_yaw + 360.0 / yaws * k) % 360.0
            rel = (yaw - sph.heading + 180.0) % 360.0 - 180.0
            views.append({"image": sph.render_view(rel, pitch, hfov, size), "yaw": yaw, "pitch": pitch, "hfov": hfov})
    return views


def _rank(codes, code):
    return codes.index(code) if code in codes else 99


def _work(r):
    global _LOC
    from engine.geo import geoguessr_points, geoguessr_score, haversine_km, region_at, resolve_map
    from engine.locator import Locator
    from engine.panorama import SphericalImage
    if _LOC is None:
        _LOC = Locator()
    sph = SphericalImage.from_equirect(os.path.join(DATASET, "panos", r["pano_id"] + ".jpg"), heading=r["heading"])
    rng = random.Random(r["pano_id"])
    g = GRIDS[GRID]
    mp = (MAP_OVERRIDE or r.get("map")) if WITH_MAP else None   # the map the player is on
    res = _LOC.analyze_views(render_views(sph, rng.uniform(0, 360), g["hfov"], g["size"], g["yaws"]), map_info=mp)
    codes = [c["code"] for c in res["countries"]]
    d = float(haversine_km(r["lat"], r["lng"], res["guess"]["lat"], res["guess"]["lng"]))
    reg = region_at(r["lat"], r["lng"])
    hint = res["hints"][0] if res["hints"] else {}
    top_regions = [x["code"] for x in hint.get("regions") or []]
    rm = resolve_map(mp if mp is not None else r.get("map")) or {}
    pts_map = geoguessr_points(d, rm["maxErrorDistance"]) if rm.get("maxErrorDistance") else float(geoguessr_score(d))
    post = res.get("posterior") or {}
    out = {"pano_id": r["pano_id"], "game": r.get("game"), "label": r["label"], "codes": codes[:10], "km": round(d, 2),
           "points": float(geoguessr_score(d)), "points_map": float(pts_map), "ms": res["timing_ms"]["total"],
           "p_true": float(post.get(r["label"], 0.0)), "region": reg[0] if reg else None,
           "region_rank": top_regions.index(reg[0]) if reg and reg[0] in top_regions else 99,
           "region_rank_overall": _rank([x["code"] for x in res.get("regions") or []], reg[0] if reg else None),
           "walk": r.get("walk")}
    if hint.get("areas") and reg:   # GeoGuessr-style areas of the top country (engines with engine/areas.py)
        from engine.areas import country_areas
        area = next((a["id"] for a in country_areas(r["label"]) or [] if reg[0] in a.get("regions", ())), None)
        if area:
            out["area"] = area
            out["area_rank"] = _rank([a["id"] for a in hint["areas"]], area)
    if STABILITY:  # a second capture of the same place from another start direction
        res2 = _LOC.analyze_views(render_views(sph, rng.uniform(0, 360), g["hfov"], g["size"], g["yaws"]), map_info=mp)
        out["same_top"] = res2["countries"][0]["code"] == codes[0]
    return out


def boot_ci(values, groups, stat=np.mean, n=2000, seed=0):
    """95% interval of stat(values) by resampling whole groups (games); None for < 2 groups."""
    values = np.asarray(values, float)
    groups = np.asarray([str(g) for g in groups])
    keys = np.unique(groups)
    if len(keys) < 2:
        return None
    idx = {k: np.flatnonzero(groups == k) for k in keys}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        out.append(stat(values[pick]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def summarise(out, split, with_cis):
    rank = np.array([_rank(o["codes"], o["label"]) for o in out])
    games = [o.get("game") or o["pano_id"] for o in out]
    right = rank == 0
    res = {"split": split, "n": len(out), "games": len(set(games)),
           "top1": float(np.mean(rank == 0)), "top3": float(np.mean(rank < 3)), "top5": float(np.mean(rank < 5)),
           "mean_score": float(np.mean([o["points"] for o in out])),
           "mean_points_map_formula": float(np.mean([o["points_map"] for o in out])),
           "median_km": float(np.median([o["km"] for o in out])),
           "region_top1_if_country_right": float(np.mean([o["region_rank"] == 0 for o, k in zip(out, rank) if k == 0] or [0])),
           "region_top3_if_country_right": float(np.mean([o["region_rank"] < 3 for o, k in zip(out, rank) if k == 0] or [0])),
           "analysis_ms": float(np.median([o["ms"] for o in out]))}
    if "p_true" in out[0]:
        res["loglik_true_country"] = float(np.mean([math.log(max(o["p_true"], 1e-9)) for o in out]))
        res["region_top3_overall"] = float(np.mean([o["region_rank_overall"] < 3 for o in out]))
    ar = [o["area_rank"] for o, k in zip(out, rank) if k == 0 and "area_rank" in o]
    if ar:
        res["area_top1_if_country_right"] = float(np.mean([a == 0 for a in ar]))
        res["area_top2_if_country_right"] = float(np.mean([a < 2 for a in ar]))
        res["area_n"] = len(ar)
    if with_cis:
        res["ci95_by_game"] = {
            "points_map_formula": boot_ci([o["points_map"] for o in out], games),
            "top1": boot_ci(rank == 0, games), "top3": boot_ci(rank < 3, games),
            "loglik_true_country": boot_ci([math.log(max(o.get("p_true", 0.0), 1e-9)) for o in out], games)}
        walk = np.array([o.get("walk") or 0 for o in out])
        res["by_game_condition"] = {  # the condition the round's game was played in (same panoramas, No Move render)
            name: {"n": int(sel.sum()), "top1": float(np.mean(rank[sel] == 0)),
                   "points_map_formula": float(np.mean([o["points_map"] for o, s in zip(out, sel) if s]))}
            for name, sel in (("nomove_games", walk == 0), ("moving_games", walk > 0)) if sel.any()}
        res["right_country_n"] = int(right.sum())
        near = np.array([bool(o.get("near_train")) for o in out])
        if near.any():   # rounds with a training panorama (duel / Street View sample) within 1 km
            far = ~near
            res["without_training_panorama_within_1km"] = {
                "n": int(far.sum()), "top1": float(np.mean(rank[far] == 0)), "top3": float(np.mean(rank[far] < 3)),
                "points_map_formula": float(np.mean([o["points_map"] for o, f in zip(out, far) if f])),
                "points_ci95": boot_ci([o["points_map"] for o, f in zip(out, far) if f], [g for g, f in zip(games, far) if f])}
            res["with_training_panorama_within_1km"] = {
                "n": int(near.sum()), "top1": float(np.mean(rank[near] == 0)),
                "points_map_formula": float(np.mean([o["points_map"] for o, f in zip(out, near) if f]))}
    return res


def near_training(recs, km=1.0):
    """Pano ids of the held-out records with a training panorama (train split, image or features) within km: the
    saved model may hold such a reference when it was fitted before these rounds existed (load_records leaves
    training panoramas near held-out rounds out of a refit only)."""
    from calibrate import _within_km
    allr = load_records(exclude_own=False)
    tr = [r for r in allr if split_of(r) == "train"]
    if not tr or not recs:
        return set()
    tla, tln = np.array([t["lat"] for t in tr], float), np.array([t["lng"] for t in tr], float)
    out = set()
    for la, ln in (("lat", "lng"), ("round_lat", "round_lng")):
        sub = [r for r in recs if r.get(la) is not None and r.get(ln) is not None]
        hit = _within_km([r[la] for r in sub], [r[ln] for r in sub], tla, tln, km)
        out.update(r["pano_id"] for r, x in zip(sub, hit) if x)
    return out


def paired(out, other_path):
    """This output minus an earlier one on the same panoramas: points / top1 differences, 95% interval by game."""
    prev = {o["pano_id"]: o for o in json.load(open(other_path)).get("rows") or []}
    both = [(o, prev[o["pano_id"]]) for o in out if o["pano_id"] in prev]
    if not both:
        return {"other": other_path, "n": 0}
    games = [o.get("game") or o["pano_id"] for o, _ in both]
    dp = [o["points_map"] - p["points_map"] for o, p in both]
    d1 = [float(_rank(o["codes"], o["label"]) == 0) - float(_rank(p["codes"], p["label"]) == 0) for o, p in both]
    dl = [math.log(max(o["p_true"], 1e-9)) - math.log(max(p.get("p_true", 0.0), 1e-9)) for o, p in both]
    return {"other": os.path.relpath(other_path, ROOT), "n": len(both),
            "points_diff": round(float(np.mean(dp)), 1), "points_ci95": boot_ci(dp, games),
            "top1_diff": round(float(np.mean(d1)), 4), "top1_ci95": boot_ci(d1, games),
            "loglik_diff": round(float(np.mean(dl)), 4), "loglik_ci95": boot_ci(dl, games)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", help="test, calib, world_test or world_calib")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stability", action="store_true", help="also capture each place twice")
    ap.add_argument("--map", action="store_true", help="pass each round's map (as in the game) to the locator "
                    "(always on for the world splits: the official World map)")
    ap.add_argument("--no-map", action="store_true", help="world splits: no map info")
    ap.add_argument("--grid", default="user", choices=sorted(GRIDS))
    ap.add_argument("--out", default="", help="output JSON (world splits: default scratch/world/eval/eval_live_<split>.json)")
    ap.add_argument("--paired", default="", help="an earlier output of the same split: paired differences")
    ap.add_argument("--tree", default="", help="engine + model files from this copy (tools/world_play.py --snapshot "
                    "scratch/world/tree_*) instead of the working tree")
    args = ap.parse_args()
    global STABILITY, WITH_MAP, GRID, MAP_OVERRIDE
    world = args.split in WORLD_SPLITS
    STABILITY, GRID = args.stability, args.grid
    WITH_MAP = (world and not args.no_map) or (args.map and not args.no_map)
    MAP_OVERRIDE = WORLD_LIVE_MAP if world else None
    recs = [r for r in load_records(pixels=True) if split_of(r) == args.split and r.get("heading") is not None]
    if args.n:
        recs = recs[: args.n]
    if not recs:
        sys.exit("no panoramas of split %s on disk" % args.split)
    from world_play import model_fingerprint, same_tree
    fp0 = model_fingerprint(TREE or ROOT)
    with Pool(args.workers) as pool:
        out = pool.map(_work, recs, chunksize=2)
    fp1 = model_fingerprint(TREE or ROOT)
    if world:
        near = near_training(recs)
        for o in out:
            o["near_train"] = o["pano_id"] in near
    res = summarise(out, args.split, with_cis=world)
    res.update({"mode": "%d rendered views (live grid 2x%d, %.1fx90 deg, %dx%d), car axis unknown" % (
        2 * GRIDS[GRID]["yaws"], GRIDS[GRID]["yaws"], GRIDS[GRID]["hfov"], GRIDS[GRID]["size"][0], GRIDS[GRID]["size"][1]),
        "with_map": WITH_MAP, "map": (MAP_OVERRIDE or {}).get("name") if WITH_MAP and world else None, "grid": GRID})
    if STABILITY:
        res["same_top_country_two_captures"] = float(np.mean([o["same_top"] for o in out]))
    res["fingerprint"] = fp0
    res["tree"] = os.path.relpath(TREE, ROOT) if TREE else "working tree"
    if not same_tree(fp0, fp1):   # the model / engine files were rewritten while the workers ran: not one model
        res["warning"] = "model / engine files changed during the evaluation (%s -> %s): rerun with --tree" % (fp0, fp1)
        sys.stderr.write(res["warning"] + "\n")
    if world:
        if args.paired:
            res["paired"] = paired(out, args.paired)
    print(json.dumps(res, indent=1))
    if world:
        path = args.out or os.path.join(ROOT, "scratch", "world", "eval", "eval_live_%s%s%s.json" % (
            args.split, "" if GRID == "user" else "_" + GRID, "" if WITH_MAP else "_nomap"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(dict(res, rows=out), open(path, "w"), indent=1)
        print("saved", os.path.relpath(path, ROOT))
    else:
        json.dump(res, open(args.out or os.path.join(ROOT, "data", "model", "eval_live.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
