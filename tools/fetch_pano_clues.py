#!/usr/bin/env python3
"""
Download GeoGuessr's own per-panorama clue placements (GET /api/v4/clues/{panoId}) of FINISHED
rounds.

This is how GeoGuessr's post-round analysis works: the clues are curated per location
(country, admin-1 regions via seterraRegionIds, and the camera heading/pitch/zoom where the
clue is visible) and looked up by the panorama id - the image itself is not analysed.
The placements are used here only OFFLINE, as labels for the hint ranking
(tools/build_clue_index.py); the live script never reads the panorama id.

  python3 tools/fetch_pano_clues.py                      rounds of data/calibration/history_rounds.json
  python3 tools/fetch_pano_clues.py --merge FILE.json    import a raw dump {pano_id: [api placement, ...]}
  python3 tools/fetch_pano_clues.py --duels --n 1500 [--months 2026-08,2026-09] [--modes StandardDuels]
                                                         rounds of finished public duels
                                                         (data/calibration/duel_rounds.json)
  python3 tools/fetch_pano_clues.py --translations       English text of clue keys that are missing
                                                         from data/geoguessr_all_clue_translations.json
  python3 tools/fetch_pano_clues.py --seterra            seterra region ids that are unmapped or mapped only by
                                                         a partial name match, with their titles
Coverage measured in Oct 2026: public Standard duels of Aug-Oct 2026 97% annotated, NoMove duels
53% (Aug) to 94% (Oct), NMPZ and Feb-May 2026 duels 0-15%, the user's standard 'World' games 100%.
Output: data/calibration/pano_clues.json {pano_id: [placement, ...]} (incremental, resumable).
GET only, >= 1.3 s between requests, stops at the first HTTP 429.
Needs the _ncfa cookie (data/session_cookie.txt) for the clue endpoint.
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gg_api import HEADERS, request  # noqa: E402

OUT = os.path.join(ROOT, "data", "calibration", "pano_clues.json")
HISTORY = os.path.join(ROOT, "data", "calibration", "history_rounds.json")
DUELS = os.path.join(ROOT, "data", "calibration", "duel_rounds.json")
TRANSLATIONS = os.path.join(ROOT, "data", "geoguessr_all_clue_translations.json")
PANOS = os.path.join(ROOT, "scratch", "dataset", "panos")
MIN_SLEEP = 1.3
KEEP = ("id", "heading", "pitch", "zoom", "title", "description", "image", "countryCode", "category", "type",
        "seterraGameId", "seterraRegionIds", "text")


class RateLimited(Exception):
    pass


def normalise(p):
    """API placement -> compact record (panoId is the dict key, the ignored fields are dropped)."""
    q = {k: p.get(k) for k in KEEP if p.get(k) is not None}
    for k, nd in (("heading", 2), ("pitch", 2), ("zoom", 3)):
        if isinstance(q.get(k), (int, float)):
            q[k] = round(float(q[k]), nd)
    return q


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(out):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, separators=(",", ":"), sort_keys=True)
    os.replace(tmp, OUT)


class NetworkDown(Exception):
    pass


_FAILS = [0]


def get_json(url, cookie=True):
    """(status, json); status None on a network error. Raises RateLimited on HTTP 429 and
    NetworkDown after 8 network errors in a row."""
    try:
        if cookie:
            st, d = request(url)
        else:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=20) as resp:
                st, d = resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as e:
        st, d = e.code, None
    except (OSError, ValueError):
        st, d = None, None
    _FAILS[0] = _FAILS[0] + 1 if st is None else 0
    if st == 429:
        raise RateLimited(url)
    if _FAILS[0] >= 8:
        raise NetworkDown(url)
    return st, d


def history_ids():
    return [r["pano_id"] for r in json.load(open(HISTORY))]


def duel_rounds(months=None, modes=None, min_age_h=24.0):
    """Rounds of finished public duels, minus every location of the user's own rounds (same rule
    as tools/build_dataset.py duels), one entry per panorama; downloaded panoramas first, the rest
    in a fixed pseudo-random order."""
    sys.path.insert(0, ROOT)
    import numpy as np
    from engine.geo import haversine_km
    own = json.load(open(HISTORY))
    own_ids = {r["pano_id"] for r in own}
    olat, olng = np.array([r["lat"] for r in own]), np.array([r["lng"] for r in own])
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - min_age_h * 3600))
    keep, seen = [], set()
    for r in json.load(open(DUELS))["rounds"]:
        pid, start = r.get("pano_id"), r.get("start") or ""
        if not pid or pid in own_ids or pid in seen or not start or start[:19] > cutoff:
            continue
        if months and start[:7] not in months or modes and r.get("mode") not in modes:
            continue
        if float(np.min(haversine_km(r["lat"], r["lng"], olat, olng))) < 1.0:
            continue
        seen.add(pid)
        keep.append(r)
    keep.sort(key=lambda r: (not os.path.exists(os.path.join(PANOS, r["pano_id"] + ".jpg")),
                             hashlib.md5(r["pano_id"].encode()).hexdigest()))
    return keep


def fetch(out, ids, sleep, strata=None):
    todo = [i for i in dict.fromkeys(ids) if i and i not in out]
    print("%d panoramas to query" % len(todo), flush=True)
    hit, done = collections.Counter(), collections.Counter()
    try:
        for k, pid in enumerate(todo, 1):
            t0 = time.time()
            st, d = get_json("https://www.geoguessr.com/api/v4/clues/" + urllib.parse.quote(pid, safe=""))
            if st in (401, 403):
                sys.exit("not authorised (HTTP %d): check data/session_cookie.txt" % st)
            if st == 200 and isinstance(d, list):
                out[pid] = [normalise(p) for p in d if p.get("panoId", pid) == pid]
                s = strata.get(pid) if strata else "all"
                done[s] += 1
                hit[s] += bool(out[pid])
            if k % 25 == 0 or k == len(todo):
                save(out)
                print("%d/%d panoramas, %d with clues" % (k, len(todo), sum(hit.values())), flush=True)
            time.sleep(max(0.0, max(sleep, MIN_SLEEP) - (time.time() - t0)))
    except RateLimited:
        print("HTTP 429 - stopping", flush=True)
    except NetworkDown:
        print("network unreachable - stopping (run again to resume)", flush=True)
    finally:
        save(out)
        for s in sorted(done):
            print("  %-28s %4d queried, %4d with clues (%.0f%%)" % (s, done[s], hit[s], 100.0 * hit[s] / done[s]))


def fetch_translations(out, sleep):
    known = json.load(open(TRANSLATIONS)) if os.path.exists(TRANSLATIONS) else {}
    have = {}
    for v in out.values():
        for p in v:
            for k, t in (p.get("text") or {}).items():
                have[p.get(k)] = t
    need = sorted({p[k] for v in out.values() for p in v for k in ("title", "description")
                   if p.get(k) and p[k] not in known and p[k] not in have})
    print("%d clue keys without text" % len(need), flush=True)
    try:
        for key in need:
            t0 = time.time()
            st, d = get_json("https://www.geoguessr.com/api/v3/translations/key/" + urllib.parse.quote(key, safe=""),
                             cookie=False)
            phr = [x.get("phrase") for x in ((d or {}).get("phrases") or []) if x.get("locale") == "en_US"]
            if st == 200 and phr and phr[0]:
                have[key] = phr[0]
            time.sleep(max(0.0, max(sleep, MIN_SLEEP) - (time.time() - t0)))
    except RateLimited:
        print("HTTP 429 - stopping", flush=True)
    except NetworkDown:
        print("network unreachable - stopping (run again to resume)", flush=True)
    for v in out.values():
        for p in v:
            txt = {k: have[p[k]] for k in ("title", "description") if p.get(k) in have}
            if txt:
                p["text"] = txt
    save(out)
    print("%d of %d keys resolved" % (len([k for k in need if k in have]), len(need)))


def seterra_report(out, sleep):
    """Print the seterra region ids that engine.hints maps to no admin-1 region of the raster, or only
    by a partial name match, with their titles from the Seterra quiz definition
    (GET /api/v4/seterra/{gameId}), to review and extend engine.hints.AREA_ALIASES / ISO_ALIASES."""
    sys.path.insert(0, ROOT)
    from engine.hints import ClueBase
    kb = ClueBase(index_path="")
    todo = collections.defaultdict(set)
    for v in out.values():
        for p in v:
            cc = (p.get("countryCode") or "").upper()
            for i in p.get("seterraRegionIds") or []:
                codes, how = kb.region_match(cc, i)
                if how in (None, "partial"):
                    todo[str(p.get("seterraGameId"))].add((cc, i, how or "unmapped", ",".join(codes)))
    try:
        for gid, items in sorted(todo.items()):
            t0 = time.time()
            st, d = get_json("https://www.geoguessr.com/api/v4/seterra/" + urllib.parse.quote(gid, safe=""))
            d = d if isinstance(d, dict) else {}
            titles = {x.get("gameItemId"): x.get("title") for x in d.get("items") or []}
            for cc, i, how, codes in sorted(items):
                print("\t".join(map(str, (gid, d.get("title"), cc, i, titles.get(i), how, codes))), flush=True)
            time.sleep(max(0.0, max(sleep, MIN_SLEEP) - (time.time() - t0)))
    except (RateLimited, NetworkDown) as e:
        print("stopped: %r" % (e,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", default="", help="raw dump {pano_id: [api placement, ...]} to import")
    ap.add_argument("--duels", action="store_true")
    ap.add_argument("--months", default="", help="duels: comma-separated YYYY-MM of the game start")
    ap.add_argument("--modes", default="", help="duels: StandardDuels,NoMoveDuels,NmpzDuels")
    ap.add_argument("--translations", action="store_true")
    ap.add_argument("--seterra", action="store_true", help="list unmapped seterra region ids with their titles")
    ap.add_argument("--n", type=int, default=0, help="query at most n new panoramas")
    ap.add_argument("--sleep", type=float, default=MIN_SLEEP)
    args = ap.parse_args()
    out = load()
    if args.merge:
        raw = json.load(open(args.merge))
        new = {pid: [normalise(p) for p in v if p.get("panoId", pid) == pid]
               for pid, v in raw.items() if isinstance(v, list) and pid not in out}
        out.update(new)
        save(out)
        print("merged %d panoramas (%d with clues), %d in total" % (len(new), sum(map(bool, new.values())), len(out)))
    elif args.translations:
        fetch_translations(out, args.sleep)
    elif args.seterra:
        seterra_report(out, args.sleep)
    else:
        strata = None
        if args.duels:
            rounds = duel_rounds(set(filter(None, args.months.split(","))), set(filter(None, args.modes.split(","))))
            strata = {r["pano_id"]: "%s %s" % (r.get("mode"), r["start"][:7]) for r in rounds}
            ids = [r["pano_id"] for r in rounds if r["pano_id"] not in out]
        else:
            ids = [i for i in history_ids() if i not in out]
        fetch(out, ids[: args.n] if args.n else ids, args.sleep, strata)


if __name__ == "__main__":
    main()
