#!/usr/bin/env python3
"""
Empirical country priors for the map-aware locator -> data/model/priors.json.

  python3 tools/build_priors.py [--alpha 0.25] [--min-rounds 50] [--folds 5]

ranked_world  country counts of other players' public ranked duels (data/calibration/duel_rounds.json)
              without the user's own games and locations (any round within 1 km of a user round,
              CALIB and TEST alike, as tools/build_dataset.py duels does for the reference panoramas).
              The location filter only removes data, so it can only make TEST worse; keeping those
              rounds puts other players' copies of the TEST locations into the prior (+73 points on
              the 54 TEST rounds of World maps without their own counts, +29 on the 66 CALIB ones);
maps          per-map counts from the CALIB half (by game, tools/calibrate.split_of) of the user's
              rounds (data/calibration/history_rounds.json) for maps with >= --min-rounds such rounds,
              plus per-fold counts so the prior mix can be calibrated out-of-fold on CALIB, the dates
              of the counted games (from duel ObjectIds; unknown for Classic / Challenge tokens) and
              the map's updatedAt (data/maps.json) to flag counts older than the map's last edit.
TEST rounds are only used to remove duel rounds at their locations.  Counts cover every country (the model classes are applied at load),
so a refit of the model needs no rebuild.  The mixture weights ("params") come from
tools/train_model.py --calibrate-maps.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from calibrate import LABEL_FIX, split_of  # noqa: E402
from engine.geo import haversine_km, resolve_map  # noqa: E402
from engine.model import MODEL_DIR, PRIORS_FILE  # noqa: E402

CAL = os.path.join(ROOT, "data", "calibration")
DUELS = os.path.join(CAL, "duel_rounds.json")


def label(cc):
    cc = (cc or "").upper()
    return LABEL_FIX.get(cc, cc) or None


def game_split(r):
    """calib / test half of a user's round by game (same hash as calibrate.split_of), any map."""
    return split_of({"mode": "history", "map": None, "game": r["game"], "pano_id": r["pano_id"]})


def fold_of(game, k):
    """Fold inside a split; independent of the calib/test parity bit."""
    return (int(hashlib.md5(str(game).encode()).hexdigest(), 16) // 2) % k


def game_date(game):
    """Creation date of a game with a MongoDB ObjectId (duels), None for other tokens."""
    g = str(game)
    if len(g) != 24:
        return None
    try:
        return datetime.datetime.utcfromtimestamp(int(g[:8], 16)).date().isoformat()
    except ValueError:
        return None


def counts_of(codes):
    return dict(sorted(Counter(x for x in codes if x).items()))


def ranked_world(own):
    raw = open(DUELS, "rb").read()
    duel = json.loads(raw.decode("utf-8"))["rounds"]
    own_games = {r["game"] for r in own}
    own_panos = {r["pano_id"] for r in own}
    olat = np.array([r["lat"] for r in own])
    olng = np.array([r["lng"] for r in own])
    keep, seen, n_own, n_near = [], set(), 0, 0
    for r in duel:
        key = (r["game"], r["round"])
        if r["game"] in own_games:
            n_own += 1
        elif r.get("pano_id") in own_panos or float(np.min(haversine_km(r["lat"], r["lng"], olat, olng))) < 1.0:
            n_near += 1
        elif key not in seen:
            seen.add(key)
            keep.append(r)
    counts = counts_of(label(r.get("gg_country")) for r in keep)
    starts = sorted(r["start"] for r in keep if r.get("start"))
    return {"source": "public ranked duels of other players (data/calibration/duel_rounds.json)",
            "duel_file": {"rounds": len(duel), "sha1": hashlib.sha1(raw).hexdigest()[:12]},
            "n": sum(counts.values()), "removed_own_games": n_own, "removed_near_user_rounds": n_near,
            "removed_duplicates": len(duel) - n_own - n_near - len(keep),
            "dates": [starts[0][:10], starts[-1][:10]] if starts else None,
            "modes": dict(Counter(r.get("mode") for r in keep)), "counts": counts}


def per_map(own, min_rounds, k):
    by_map = {}
    for r in own:
        if not r.get("map") or game_split(r) != "calib":
            continue
        m = resolve_map(r["map"])
        if not m or not m.get("id"):
            continue
        by_map.setdefault(m["id"], (m, []))[1].append(r)
    out = {}
    for mid, (m, rows) in sorted(by_map.items(), key=lambda kv: -len(kv[1][1])):
        if len(rows) < min_rounds:
            print("  %-28s %4d calib rounds - below --min-rounds, uses %s"
                  % (m["name"], len(rows), "ranked_world" if m["world"] else "the model prior"))
            continue
        counts = counts_of(label(r.get("gg_country")) for r in rows)
        folds = [counts_of(label(r.get("gg_country")) for r in rows if fold_of(r["game"], k) == f) for f in range(k)]
        dates = sorted(d for d in (game_date(r["game"]) for r in rows) if d)
        out[mid] = {"name": m["name"], "source": "CALIB split of data/calibration/history_rounds.json",
                    "n": sum(counts.values()), "games": len({r["game"] for r in rows}),
                    "dates": [dates[0], dates[-1]] if dates else None, "dated_rounds": len(dates),
                    "map_updated": m.get("updatedAt"), "counts": counts, "fold_counts": folds}
        upd = str(m.get("updatedAt") or "")[:10]
        stale = bool(dates and upd > dates[-1])
        print("  %-28s %4d calib rounds, %d games, %d countries, games %s, map updated %s%s"
              % (m["name"], len(rows), out[mid]["games"], len(counts),
                 "%s..%s" % (dates[0], dates[-1]) if dates else "undated", upd or "?",
                 "  <- map edited after the counted rounds" if stale else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.25,
                    help="additive smoothing per class (best out-of-fold calib points of 0.1/0.25/0.5/1)")
    ap.add_argument("--min-rounds", type=int, default=50, help="calib rounds needed for a per-map prior")
    ap.add_argument("--folds", type=int, default=5, help="folds (by game) for out-of-fold calibration")
    args = ap.parse_args()
    own = json.load(open(os.path.join(CAL, "history_rounds.json")))
    path = os.path.join(MODEL_DIR, PRIORS_FILE)
    old = json.load(open(path)) if os.path.exists(path) else {}
    rw = ranked_world(own)
    print("ranked_world: %d rounds (removed: %d of the user's own games, %d at the user's locations), top %s"
          % (rw["n"], rw["removed_own_games"], rw["removed_near_user_rounds"], Counter(rw["counts"]).most_common(6)))
    out = {"alpha": args.alpha, "folds": args.folds, "ranked_world": rw,
           "maps": per_map(own, args.min_rounds, args.folds)}
    if old.get("params"):
        out["params"] = old["params"]
        print("kept the calibrated params - re-run tools/train_model.py --calibrate-maps --save")
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
