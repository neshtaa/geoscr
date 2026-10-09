#!/usr/bin/env python3
"""
Build the calibration dataset of official Street View panoramas.

Modes
  world     area-weighted random land points (mimics the GeoGuessr World map,
            which picks random locations proportionally to covered area)
  balanced  fixed quota of panoramas per country (for per-country likelihoods)
  history   every round of the user's own GeoGuessr games (duels, challenges,
            standard games) read from the GeoGuessr API - the real-game test set
  postmatch panoramas listed in data/geoguessr_postmatch_clues.json
  duels     rounds of public ranked duels (tools/crawl_duels.py); images kept - for many rounds use
            tools/stream_duels.py (features only, images deleted)

Output: <out>/panos/<pano_id>.jpg (2048x1024 equirectangular, centre = car heading)
        <out>/index.jsonl (one metadata record per panorama)
        <out>/world_hits.json (attempts/hits per country for the coverage prior)
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from calibrate import OWN_FILE, OWN_KM  # noqa: E402  (the user's own rounds; no duel location this close)
from engine.geo import _raster, country_at  # noqa: E402
from engine.streetview import download_panorama, get_metadata, search_pano  # noqa: E402

LOCK = threading.Lock()


def load_index(out):
    seen = {}
    path = os.path.join(out, "index.jsonl")
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                seen[r["pano_id"]] = r
            except Exception:
                pass
    return seen


def append_record(out, rec):
    with LOCK:
        with open(os.path.join(out, "index.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


def fetch_and_store(out, meta, mode, extra=None):
    path = os.path.join(out, "panos", meta["pano_id"] + ".jpg")
    if not os.path.exists(path):
        img = download_panorama(meta)
        img.save(path, quality=90)
    rec = dict(meta)
    rec["mode"] = mode
    rec["raster_country"] = country_at(meta["lat"], meta["lng"])
    if extra:
        rec.update(extra)
    append_record(out, rec)
    return rec


def retry(fn, *a, tries=3, **kw):
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def random_land_point(rng):
    while True:
        lat = math.degrees(math.asin(2 * rng.random() - 1))
        lng = rng.uniform(-180, 180)
        if lat < -60:  # Antarctica is not part of the World map rotation
            continue
        c = country_at(lat, lng, search_px=0)
        if c:
            return lat, lng, c


def country_cells():
    """Per-country arrays of raster cells (for uniform-in-area sampling)."""
    grid, codes, res = _raster()
    ys, xs = np.nonzero(grid)
    vals = grid[ys, xs]
    lats = 90.0 - (ys + 0.5) * res
    w = np.cos(np.radians(lats))
    order = np.argsort(vals, kind="stable")
    vals, ys, xs, w = vals[order], ys[order], xs[order], w[order]
    bounds = np.flatnonzero(np.diff(vals)) + 1
    cells = {}
    for seg in np.split(np.arange(len(vals)), bounds):
        if len(seg):
            cells[codes[vals[seg[0]]]] = (ys[seg], xs[seg], w[seg] / w[seg].sum())
    return cells, res


def run_world(args, seen):
    rng = random.Random(args.seed)
    hits_path = os.path.join(args.out, "world_hits.json")
    stats = json.load(open(hits_path)) if os.path.exists(hits_path) else {"attempts": {}, "hits": {}}
    pts = [random_land_point(rng) for _ in range(args.n)]

    def work(p):
        lat, lng, c = p
        try:
            meta = retry(search_pano, lat, lng, args.radius)
        except Exception:
            return None
        with LOCK:
            stats["attempts"][c] = stats["attempts"].get(c, 0) + 1
            if meta:
                stats["hits"][c] = stats["hits"].get(c, 0) + 1
        if not meta or meta["pano_id"] in seen:
            return None
        seen[meta["pano_id"]] = True
        try:
            return retry(fetch_and_store, args.out, meta, "world", {"sample_country": c})
        except Exception:
            return None

    done = 0
    with ThreadPoolExecutor(args.threads) as ex:
        for f in as_completed([ex.submit(work, p) for p in pts]):
            if f.result():
                done += 1
                if done % 100 == 0:
                    print(f"[world] stored {done}", flush=True)
                    with LOCK:
                        json.dump(stats, open(hits_path, "w"))
    json.dump(stats, open(hits_path, "w"))
    print(f"[world] stored {done} new panoramas")


def run_balanced(args, seen):
    rng = np.random.default_rng(args.seed)
    cells, res = country_cells()
    targets = [c.strip() for c in args.countries.split(",")] if args.countries else sorted(cells)
    have = {}
    for r in seen.values():
        if isinstance(r, dict) and r.get("mode") in ("world", "balanced"):
            cc = r.get("country_code") or r.get("raster_country")
            have[cc] = have.get(cc, 0) + 1

    def sample_country(cc):
        if cc not in cells:
            return 0
        ys, xs, p = cells[cc]
        got, misses, stored = have.get(cc, 0), 0, 0
        local = np.random.default_rng(abs(hash(cc)) % (2 ** 32) + args.seed)
        while got < args.quota and misses < args.max_misses:
            k = local.choice(len(p), p=p)
            lat = 90.0 - (ys[k] + local.random()) * res
            lng = -180.0 + (xs[k] + local.random()) * res
            try:
                meta = retry(search_pano, lat, lng, args.radius)
            except Exception:
                misses += 1
                continue
            if not meta or meta["pano_id"] in seen:
                misses += 1
                continue
            mc = (meta.get("country_code") or country_at(meta["lat"], meta["lng"]) or "").upper()
            if mc != cc:
                misses += 1
                continue
            seen[meta["pano_id"]] = True
            try:
                retry(fetch_and_store, args.out, meta, "balanced", {"sample_country": cc})
                got += 1
                stored += 1
            except Exception:
                misses += 1
        print(f"[balanced] {cc}: {got} panoramas ({misses} misses)", flush=True)
        return stored

    with ThreadPoolExecutor(args.threads) as ex:
        total = sum(ex.map(sample_country, targets))
    print(f"[balanced] stored {total} new panoramas")


def decode_pano(pid):
    if pid and len(pid) % 2 == 0 and all(c in "0123456789ABCDEFabcdef" for c in pid):
        try:
            return bytes.fromhex(pid).decode()
        except Exception:
            pass
    return pid


def history_rounds():
    """All rounds of the user's GeoGuessr games (feed -> duels / games / challenges)."""
    from gg_api import request
    tokens = {}
    pag = None
    for _ in range(60):
        url = "https://www.geoguessr.com/api/v4/feed/private" + (f"?paginationToken={pag}" if pag else "")
        st, d = request(url)
        if st != 200 or not d:
            break
        for e in d.get("entries", []):
            pl = e.get("payload")
            try:
                pl = json.loads(pl) if isinstance(pl, str) else pl
            except Exception:
                pass
            for it in (pl if isinstance(pl, list) else [pl]):
                sub = it.get("payload", it) if isinstance(it, dict) else None
                if isinstance(sub, str):
                    try:
                        sub = json.loads(sub)
                    except Exception:
                        sub = None
                if isinstance(sub, dict):
                    for k in ("gameId", "gameToken", "challengeToken"):
                        if sub.get(k):
                            tokens[sub[k]] = k
        pag = d.get("paginationToken")
        if not pag:
            break
    rounds = []
    for tok, kind in tokens.items():
        if kind == "gameId":
            st, d = request(f"https://game-server.geoguessr.com/api/duels/{tok}")
            if st != 200 or not d:
                continue
            map_name = ((d.get("options") or {}).get("map") or {}).get("name")
            for r in d.get("rounds", []):
                p = r.get("panorama") or {}
                if p.get("lat") is None:
                    continue
                rounds.append({"game": tok, "kind": "duel", "map": map_name, "round": r.get("roundNumber"),
                               "pano_id": decode_pano(p.get("panoId")), "lat": p["lat"], "lng": p["lng"],
                               "gg_country": (p.get("countryCode") or "").upper(),
                               "gg_heading": p.get("heading")})
        else:
            url = (f"https://www.geoguessr.com/api/v3/challenges/{tok}/game" if kind == "challengeToken"
                   else f"https://www.geoguessr.com/api/v3/games/{tok}")
            st, d = request(url)
            if st != 200 or not d:
                continue
            for i, r in enumerate(d.get("rounds", []), 1):
                rounds.append({"game": d.get("token", tok), "kind": "standard", "map": d.get("mapName"),
                               "round": i, "pano_id": decode_pano(r.get("panoId")), "lat": r["lat"],
                               "lng": r["lng"], "gg_country": (r.get("streakLocationCode") or "").upper(),
                               "gg_heading": r.get("heading")})
    return rounds


def write_json_atomic(obj, path):
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def merge_own_rounds(rounds, path=OWN_FILE):
    """Add the history rounds to OWN_FILE (the exclusion list of the duel crawl / stream and of
    calibrate.load_records; read by build_priors, build_clue_index, ...): union by (game, round), new rounds
    replace old copies.  Returns the number of rounds added."""
    try:
        with open(path) as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = []
    key = lambda r: (r.get("game"), r.get("round"))
    new = {key(r) for r in rounds}
    out = [r for r in old if key(r) not in new] + list(rounds)
    write_json_atomic(out, path)
    return len(out) - len(old)


def run_history(args, seen):
    if args.rounds_file:  # offline copy (e.g. in environments where geoguessr.com is blocked)
        rounds = json.load(open(args.rounds_file))
    else:
        rounds = history_rounds()
    print(f"[history] {len(rounds)} rounds in game history")
    json.dump(rounds, open(os.path.join(args.out, "history_rounds.json"), "w"))
    print(f"[history] {merge_own_rounds(rounds)} new rounds in {os.path.relpath(OWN_FILE, ROOT)}")

    def work(r):
        try:
            meta = retry(get_metadata, r["pano_id"]) if r.get("pano_id") else None
            if not meta:
                meta = retry(search_pano, r["lat"], r["lng"], 100)
            if not meta:
                return None
            prev = seen.get(meta["pano_id"])
            # a panorama known only as a training (duel / world / balanced) record becomes the user's round:
            # calibrate.load_records prefers the own record (and drops training records near own rounds)
            if prev is not None and not (isinstance(prev, dict) and prev.get("mode") in ("world", "balanced", "duel")):
                return None
            seen[meta["pano_id"]] = True
            extra = {k: r[k] for k in ("game", "kind", "map", "round", "gg_country", "gg_heading")}
            extra["round_lat"], extra["round_lng"] = r["lat"], r["lng"]
            return retry(fetch_and_store, args.out, meta, "history", extra)
        except Exception:
            return None

    with ThreadPoolExecutor(args.threads) as ex:
        n = sum(1 for r in ex.map(work, rounds) if r)
    print(f"[history] stored {n} panoramas")


DUELS_FILE = os.path.join(ROOT, "data", "calibration", "duel_rounds.json")


def round_key(r):
    """Identity of a crawled round: its panorama id, or game/round when the round has none."""
    return r.get("pano_id") or "%s/%s" % (r.get("game"), r.get("round"))


def duel_candidates(rounds, own, seen=(), skip=()):
    """Crawled duel rounds worth a panorama: not one of the user's own rounds (pano id) nor within OWN_KM of
    one, one round per panorama id, panorama not in `seen` (dataset index) or `skip` (round_key given up
    before).  Rounds without a panorama id are kept (the panorama is searched at the position)."""
    from engine.geo import haversine_km
    own_ids = {r.get("pano_id") for r in own} - {None}
    olat = np.array([r["lat"] for r in own], float)
    olng = np.array([r["lng"] for r in own], float)
    keep, dup = [], set()
    for r in rounds:
        pid = r.get("pano_id")
        if round_key(r) in skip or (pid is not None and (pid in own_ids or pid in dup or pid in seen)):
            continue
        if len(olat) and float(np.min(haversine_km(r["lat"], r["lng"], olat, olng))) < OWN_KM:
            continue
        if pid is not None:
            dup.add(pid)
        keep.append(r)
    return keep


def duel_meta(r):
    """Street View metadata of a duel round's panorama (fallback: nearest panorama within 100 m)."""
    meta = retry(get_metadata, r["pano_id"]) if r.get("pano_id") else None
    return meta or retry(search_pano, r["lat"], r["lng"], 100)


def duel_extra(r):
    """Index fields of a duel round (added to the Street View metadata)."""
    return {"game": r["game"], "round": r["round"], "gg_country": r["gg_country"],
            "gg_heading": r.get("gg_heading"), "duel_mode": r.get("mode"),
            "round_lat": r["lat"], "round_lng": r["lng"], "start": r.get("start")}


MAX_KEEP = 1000  # duels: new images kept without --keep-images (~0.45 MB each)


def run_duels(args, seen):
    """Rounds of public ranked duels (tools/crawl_duels.py) - real GeoGuessr location pools.  Keeps every image
    (~0.45 MB each); tools/stream_duels.py stores the features only."""
    rounds = json.load(open(DUELS_FILE))["rounds"]
    # never download a location of the user's own (calib/test) rounds
    keep = duel_candidates(rounds, json.load(open(OWN_FILE)))
    print(f"[duels] {len(rounds)} crawled rounds, {len(keep)} after removing the user's own locations")
    if args.n:
        keep = keep[: args.n]
    n_new = sum(1 for r in keep if r.get("pano_id") not in seen)
    if n_new > MAX_KEEP and not args.keep_images:
        sys.exit(f"[duels] {n_new} new panoramas would be kept as images (~{n_new * 0.45 / 1000:.1f} GB; limit "
                 f"{MAX_KEEP} without --keep-images): use tools/stream_duels.py (features only, images deleted), "
                 f"a smaller -n, or pass --keep-images")

    def work(r):
        try:
            meta = duel_meta(r)
            if not meta or meta["pano_id"] in seen:
                return None
            seen[meta["pano_id"]] = True
            return retry(fetch_and_store, args.out, meta, "duel", duel_extra(r))
        except Exception:
            return None

    done = 0
    with ThreadPoolExecutor(args.threads) as ex:
        for f in as_completed([ex.submit(work, r) for r in keep]):
            if f.result():
                done += 1
                if done % 500 == 0:
                    print(f"[duels] stored {done}", flush=True)
    print(f"[duels] stored {done} panoramas")


def run_postmatch(args, seen):
    pm = json.load(open(os.path.join(ROOT, "data/geoguessr_postmatch_clues.json"), encoding="utf-8"))
    panos = {}
    for clue in pm["clues_by_id"].values():
        for s in clue.get("sample_panoramas") or []:
            panos.setdefault(s["panoId"], {"game": s.get("gameId"), "round": s.get("round"),
                                           "clue_ids": []})["clue_ids"].append(clue["id"])

    def work(item):
        pid, extra = item
        if pid in seen:
            return None
        try:
            meta = retry(get_metadata, pid)
            if not meta:
                return None
            seen[pid] = True
            return retry(fetch_and_store, args.out, meta, "postmatch", extra)
        except Exception:
            return None

    with ThreadPoolExecutor(args.threads) as ex:
        n = sum(1 for r in ex.map(work, panos.items()) if r)
    print(f"[postmatch] stored {n} of {len(panos)} panoramas")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["world", "balanced", "history", "postmatch", "duels"])
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch/dataset"))
    ap.add_argument("-n", type=int, default=2000, help="world: number of random land points; duels: rounds (0 = all)")
    ap.add_argument("--quota", type=int, default=40, help="balanced: panoramas per country")
    ap.add_argument("--max-misses", type=int, default=60)
    ap.add_argument("--countries", default="", help="balanced: comma separated ISO codes")
    ap.add_argument("--radius", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds-file", default="", help="history: read rounds from this JSON instead of the GeoGuessr API")
    ap.add_argument("--keep-images", action="store_true", help="duels: allow storing > %d new images" % MAX_KEEP)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "panos"), exist_ok=True)
    seen = load_index(args.out)
    {"world": run_world, "balanced": run_balanced, "history": run_history,
     "postmatch": run_postmatch, "duels": run_duels}[args.mode](args, seen)


if __name__ == "__main__":
    main()
