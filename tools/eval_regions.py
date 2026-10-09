#!/usr/bin/env python3
"""
Fit, calibrate and evaluate the admin-1 region model (engine/regions.py).

  python3 tools/eval_regions.py --cv [--grid]   5-fold out-of-fold check on the ranked-duel train panoramas
                                                (structural parameters: K, tau, geo shrinkage, own rotation)
  python3 tools/eval_regions.py [--save]        fit on train, calibrate the exponents / prior / smoothing on
                                                CALIB, report CALIB and TEST -> data/model/regions.npz,
                                                data/model/eval_regions.json
  python3 tools/eval_regions.py --eval-only     saved model: CALIB / TEST report
  python3 tools/eval_regions.py --live          saved model on CALIB under live capture (10 rendered views, car axis
                                                unknown, as tools/eval_live.py; features cached in
                                                scratch/regions/live_calib.npz) -> eval_regions.json "calib_live"
  python3 tools/eval_regions.py --fetch-extra   extra region references: Street View panoramas 3-40 km from random
                                                ranked-duel train locations of the countries with many regions
                                                (<= 4 download threads, 4 feature processes, images deleted after
                                                feature extraction) -> scratch/regions/extra/; extra references
                                                whose features are not stored under the current module hashes are
                                                downloaded and extracted again
  python3 tools/eval_regions.py --check-extra-compat
                                                for a feature group whose extra-reference hash differs from the main
                                                cache (scratch/features): re-extract the group with the current module
                                                on --n main train panoramas and record the hash pair as equivalent if
                                                the cached features are reproduced (scratch/regions/extra/features/
                                                compat.json); load_extra drops all extras on an unverified mismatch
  --no-extra                                    fit without the extra references
  --refresh-cards                               rebuild the regional card snapshot (scratch/regions/card_sets.json)
                                                from data/calibration/pano_clues.json

train : world + balanced + ranked-duel Street View panoramas (region of each from engine.geo.region_index)
        + the extra references (region appearance only, not counted in the region prior)
        CV folds by location: panoramas within 1 km of each other share a fold, the extra references take
        the fold of the duel location they were sampled around, held-out duels with an extra reference of
        another fold within 1 km are not scored; the exponents of fold k are calibrated on fold k+1
calib : half of the user's real World-map rounds -> exponents, prior mix, smoothing
test  : the other half -> region top-k given the true country, unconditional region top-k with the
        GeoModel country posterior (without / with map info), both against
        engine.model.GeoModel.region_posterior, and GeoGuessr points with the guess placed on the
        region-weighted reference mass (engine.regions.location_weights).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from calibrate import DATASET, FEAT_DIR, label_of, load_records, module_hash, split_of  # noqa: E402
from train_model import group_matrix, round_maps  # noqa: E402
from engine.geo import _regions, country_at, geoguessr_points, haversine_km, region_index  # noqa: E402
from engine.model import MODEL_DIR, GeoModel  # noqa: E402
from engine.regions import (DEFAULT_PARAMS, REGIONS_FILE, RegionModel, location_weights,  # noqa: E402
                            region_posterior, sun_mixture)

FLOOR = 1e-4
AREA_MIN_REFS = 25
TOP_COUNTRIES = ["US", "BR", "RU", "CA", "AU", "ID", "MX", "IN", "ZA", "AR", "JP", "ES", "TR", "CL"]
EXTRA_DIR = os.path.join(ROOT, "scratch", "regions", "extra")
EXTRA_RATIO, EXTRA_CAP, EXTRA_MIN_EFF = 0.7, 500, 4.0  # extra refs per duel ref, cap per country, min eff. regions
EXTRA_KM = (3.0, 40.0)                                  # distance of the sampled point from the duel location
EXTRA_EXCLUDE_KM = 2.0                                  # no extra reference this close to a non-train round
FOLD_KM = 1.0                                           # CV: panoramas closer than this share a fold
CARDS_SNAPSHOT = os.path.join(ROOT, "scratch", "regions", "card_sets.json")
LIVE_FILE = os.path.join(ROOT, "scratch", "regions", "live_calib.npz")
COMPAT_FILE = os.path.join(EXTRA_DIR, "features", "compat.json")


def file_hash(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def main_hash(g):
    """Module hash the main feature cache scratch/features/<g>.npz was computed with."""
    return str(np.load(os.path.join(FEAT_DIR, g + ".npz"), allow_pickle=True)["hash"])


def load_compat():
    try:
        return json.load(open(COMPAT_FILE))
    except (OSError, ValueError):
        return {}


def extra_files(g):
    """Stored feature files of the extra references for group g: [(path, module hash)] (current file first,
    then the ones kept from older module versions)."""
    fdir = os.path.join(EXTRA_DIR, "features")
    out = []
    for f in [os.path.join(fdir, g + ".npz")] + sorted(glob.glob(os.path.join(fdir, g + ".*.npz"))):
        if os.path.exists(f):
            out.append((f, str(np.load(f, allow_pickle=True)["hash"])))
    return out


def load_extra(groups, verbose=True):
    """Extra references with all feature groups: (records, {group: (n, d)}, {group: module hash}).  Every group
    must be stored under the module hash of the main cache or under a hash verified equivalent to it
    (--check-extra-compat); otherwise the extra references are dropped (with a warning)."""
    path = os.path.join(EXTRA_DIR, "index.jsonl")
    if not os.path.exists(path):
        return [], {}, {}
    recs = {}
    for line in open(path):
        r = json.loads(line)
        recs.setdefault(r["pano_id"], r)
    recs = sorted(recs.values(), key=lambda r: r["pano_id"])
    compat = load_compat()
    mats, used, have = {}, {}, np.ones(len(recs), bool)
    for g in groups:
        hm = main_hash(g)
        ok = {hm} | {e["extra"] for e in compat.get(g, []) if e.get("main") == hm and e.get("equivalent")}
        f = [(p, h) for p, h in extra_files(g) if h in ok]
        if not f:
            if verbose:
                print("WARNING: extra references dropped: no %s features under the main cache's module hash %s "
                      "(stored: %s); run --check-extra-compat, or --fetch-extra after rebuilding the main cache"
                      % (g, hm, ", ".join(h for _, h in extra_files(g)) or "none"), flush=True)
            return [], {}, {}
        z = np.load(f[0][0], allow_pickle=True)
        used[g] = f[0][1]
        idx = {pid: i for i, pid in enumerate(z["ids"])}
        j = np.array([idx.get(r["pano_id"], -1) for r in recs])
        mats[g] = np.where((j >= 0)[:, None], z["X"][np.maximum(j, 0)].astype(np.float64), np.nan)
        have &= j >= 0
    rows = np.flatnonzero(have)
    return [recs[i] for i in rows], {g: m[rows] for g, m in mats.items()}, used


def location_folds(lat, lng, ids, k=5, km=FOLD_KM):
    """CV fold of every panorama: connected components of panoramas closer than km share the fold
    hash5(smallest pano id of the component)."""
    n = len(lat)
    la, ln = np.radians(lat), np.radians(lng)
    P = np.stack([np.cos(la) * np.cos(ln), np.cos(la) * np.sin(ln), np.sin(la)], 1) * 6371.0088
    cell = [tuple(c) for c in np.floor(P / km).astype(np.int64)]
    buckets = {}
    for i, c in enumerate(cell):
        buckets.setdefault(c, []).append(i)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    for c, mem in buckets.items():
        cand = np.array([j for o in offs for j in buckets.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]), ())])
        for i in mem:
            for j in cand[np.linalg.norm(P[cand] - P[i], axis=1) < km]:
                a, b = find(i), find(int(j))
                if a != b:
                    parent[max(a, b)] = min(a, b)
    root = np.array([find(i) for i in range(n)])
    first = {}
    for i in range(n):
        first[root[i]] = min(first.get(root[i], ids[i]), ids[i])
    return np.array([hash5(first[root[i]], k) for i in range(n)], int), root


def cross_fold_twins(D, km=FOLD_KM):
    """Train rows with a train panorama of another fold closer than km (the extra references can be)."""
    tr = np.flatnonzero(D["split"] == "train")
    la, ln = np.radians(D["lat"][tr]), np.radians(D["lng"][tr])
    P = np.stack([np.cos(la) * np.cos(ln), np.cos(la) * np.sin(ln), np.sin(la)], 1) * 6371.0088
    cell = [tuple(c) for c in np.floor(P / km).astype(np.int64)]
    buckets = {}
    for i, c in enumerate(cell):
        buckets.setdefault(c, []).append(i)
    f = D["fold"][tr]
    out = np.zeros(len(D["split"]), bool)
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    for c, mem in buckets.items():
        cand = np.array([j for o in offs for j in buckets.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]), ())])
        for i in mem:
            near = cand[np.linalg.norm(P[cand] - P[i], axis=1) < km]
            if (f[near] != f[i]).any():
                out[tr[i]] = True
    return out


def load_data(groups, extra=True):
    recs = load_records()
    mats, have = {}, np.ones(len(recs), bool)
    for g in groups:
        mats[g], h = group_matrix(g, recs)
        have &= h
    rows = np.flatnonzero(have)
    mats = {g: m[rows] for g, m in mats.items()}
    recs = [recs[i] for i in rows]
    n_main = len(recs)
    xr, xm, used = load_extra(groups) if extra else ([], {}, {})
    if xr:
        recs = recs + xr
        mats = {g: np.concatenate([m, xm[g]]) for g, m in mats.items()}
    lat = np.array([r["lat"] for r in recs])
    lng = np.array([r["lng"] for r in recs])
    split = np.array([split_of(r) for r in recs[:n_main]] + ["train"] * len(xr))
    # CV folds by location (train panoramas within FOLD_KM share a fold); extras follow their source duel
    fold = np.full(len(recs), -1)
    tr = np.flatnonzero(split[:n_main] == "train")
    fold[tr] = location_folds(lat[tr], lng[tr], [recs[i]["pano_id"] for i in tr])[0]
    pos = {recs[i]["pano_id"]: i for i in tr}
    for k, r in enumerate(xr):
        j = pos.get(r.get("src_pano"))
        fold[n_main + k] = fold[j] if j is not None else hash5(r["pano_id"])
    D = {"recs": recs, "X": mats, "lat": lat, "lng": lng, "region": region_index(lat, lng),
         "label": np.array([r["label"] for r in recs]), "split": split,
         "duel": np.array([r["mode"] == "duel" for r in recs]),
         "prior": np.array([r["mode"] != "extra" for r in recs]),
         "fold": fold, "extra_hashes": used, "n_main": n_main}
    D["sun"] = sun_mixture(mats) if "solar" in mats else None
    return D


def extra_quotas(D, ratio=EXTRA_RATIO, cap=EXTRA_CAP, min_eff=EXTRA_MIN_EFF):
    """{country: extra references}: ratio x its ranked-duel train panoramas (at most cap), for the countries
    whose duel panoramas spread over at least min_eff regions (exp of the region entropy)."""
    tr = (D["split"] == "train") & D["duel"]
    out = {}
    for cc in np.unique(D["label"][tr]):
        r = D["region"][tr & (D["label"] == cc)]
        c = np.bincount(r)[1:]
        q = c[c > 0] / max(c.sum(), 1)
        if len(q) and math.exp(-(q * np.log(q)).sum()) >= min_eff:
            out[str(cc)] = int(min(cap, round(ratio * len(r))))
    return out


def _destination(lat, lng, bearing, km):
    la, ln, b, d = math.radians(lat), math.radians(lng), math.radians(bearing), km / 6371.0
    la2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(b))
    ln2 = ln + math.atan2(math.sin(b) * math.sin(d) * math.cos(la), math.cos(d) - math.sin(la) * math.sin(la2))
    return math.degrees(la2), (math.degrees(ln2) + 540.0) % 360.0 - 180.0


def _extract_extra(groups, recs, workers):
    """Feature groups of the extra panoramas whose image is on disk -> EXTRA_DIR/features/<group>.npz
    (merged with the stored ones); the images are deleted afterwards."""
    import multiprocessing
    from calibrate import _extract_one, _init
    from engine import features
    fdir = os.path.join(EXTRA_DIR, "features")
    os.makedirs(fdir, exist_ok=True)
    todo = [r for r in recs if os.path.exists(os.path.join(EXTRA_DIR, "panos", r["pano_id"] + ".jpg"))]
    if not todo:
        return
    mods = [features.load(g) for g in groups]
    t0 = time.time()
    res = []
    jobs = [(os.path.join(EXTRA_DIR, "panos", r["pano_id"] + ".jpg"), r.get("heading")) for r in todo]
    ctx = multiprocessing.get_context("spawn")  # the download threads keep running in this process
    with ctx.Pool(workers, initializer=_init, initargs=(list(groups),)) as pool:
        for i, x in enumerate(pool.imap(_extract_one, jobs, chunksize=4)):
            res.append(x)
            if (i + 1) % 500 == 0:
                print("  features %d/%d %.1f pano/s" % (i + 1, len(jobs), (i + 1) / (time.time() - t0)), flush=True)
    for j, (g, m) in enumerate(zip(groups, mods)):
        f = os.path.join(fdir, g + ".npz")
        ids, X = [], []
        h = module_hash(m)
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            if str(z["hash"]) == h:
                ids, X = list(z["ids"]), list(z["X"])
            else:  # features of an older module version are kept (the images are gone), not overwritten
                old = os.path.join(fdir, "%s.%s.npz" % (g, z["hash"]))
                if not os.path.exists(old):
                    os.rename(f, old)
        new = {r["pano_id"] for r in todo}
        keep = [k for k, pid in enumerate(ids) if pid not in new]
        ids = [ids[k] for k in keep] + [r["pano_id"] for r in todo]
        X = [X[k] for k in keep] + [x[j] for x in res]
        np.savez_compressed(f, ids=np.array(ids), X=np.array(X, np.float32), names=np.array(m.FEATURE_NAMES), hash=h)
    for r in todo:
        os.remove(os.path.join(EXTRA_DIR, "panos", r["pano_id"] + ".jpg"))
    print("  features of %d extra panoramas in %.0fs, images deleted" % (len(todo), time.time() - t0), flush=True)


def cmd_fetch_extra(args, D, groups):
    """Extra region references near the GeoGuessr pools: for each country of extra_quotas, Street View
    panoramas found 3-40 km from random ranked-duel train locations (same country, not already in the
    dataset, >= 2 km from every non-train round), downloaded in batches of ~args.batch with <= 4 threads;
    the feature groups of a batch (args.workers processes) are extracted while the next batch downloads and
    the images are deleted after extraction.  Resumable (counts per country from the index)."""
    from concurrent.futures import ThreadPoolExecutor
    from engine import features
    from engine.streetview import download_panorama, search_pano
    os.makedirs(os.path.join(EXTRA_DIR, "panos"), exist_ok=True)
    idx_path = os.path.join(EXTRA_DIR, "index.jsonl")
    old = [json.loads(line) for line in open(idx_path)] if os.path.exists(idx_path) else []
    known = {r["pano_id"] for r in old}
    _extract_extra(groups, old, args.workers)  # images left by an interrupted run

    def current_ids():
        """Extra references with features of every group under the current module hashes."""
        out = None
        for g in groups:
            f = os.path.join(EXTRA_DIR, "features", g + ".npz")
            z = np.load(f, allow_pickle=True) if os.path.exists(f) else None
            ids = set(z["ids"]) if z is not None and str(z["hash"]) == module_hash(features.load(g)) else set()
            out = ids if out is None else out & ids
        return out or set()
    cur = current_ids()
    stale = [r for r in {r["pano_id"]: r for r in old}.values() if r["pano_id"] not in cur]
    if stale:  # features of an older module version: download the panorama again and re-extract
        print("re-extracting %d extra references (feature modules changed)" % len(stale), flush=True)

        def refetch(r):
            f = os.path.join(EXTRA_DIR, "panos", r["pano_id"] + ".jpg")
            try:
                if not os.path.exists(f):
                    download_panorama(r).save(f, quality=90)
            except Exception:
                pass
        for i in range(0, len(stale), max(1, args.batch)):
            b = stale[i:i + max(1, args.batch)]
            with ThreadPoolExecutor(min(4, args.threads)) as ex:
                list(ex.map(refetch, b))
            _extract_extra(groups, b, args.workers)
        cur = current_ids()
    old = [r for r in old if r["pano_id"] in cur]
    lock = threading.Lock()
    all_recs = load_records()
    seen = {r["pano_id"] for r in all_recs} | known
    held = [r for r in all_recs if split_of(r) != "train"]
    hlat, hlng = np.array([r["lat"] for r in held]), np.array([r["lng"] for r in held])
    tr = np.flatnonzero((D["split"] == "train") & D["duel"])
    have = {}
    for r in {r["pano_id"]: r for r in old}.values():
        have[r["sample_country"]] = have.get(r["sample_country"], 0) + 1
    need = {cc: q - have.get(cc, 0) for cc, q in extra_quotas(D).items() if q > have.get(cc, 0)}
    print("extra references to fetch: %d in %d countries (have %d)" % (sum(need.values()), len(need), len(old)),
          flush=True)

    def sample_country(task):
        cc, k, n = task
        rows = tr[D["label"][tr] == cc]
        rng = np.random.RandomState(int(hashlib_md5("%s-%d-%d" % (cc, k, len(old))), 16) % (2 ** 31))
        got, misses, out = 0, 0, []
        while got < n and misses < 4 * n + 20:
            i = rows[rng.randint(len(rows))]
            la, ln = _destination(D["lat"][i], D["lng"][i], rng.uniform(0, 360), rng.uniform(*EXTRA_KM))
            try:
                meta = search_pano(la, ln, 3000)
            except Exception:
                meta = None
            if not meta or meta["pano_id"] in seen:
                misses += 1
                continue
            rec = dict(meta, mode="extra", raster_country=country_at(meta["lat"], meta["lng"]), sample_country=cc,
                       src_pano=D["recs"][i]["pano_id"], src_fold=int(D["fold"][i]))
            if label_of(rec) != cc or (len(hlat) and haversine_km(meta["lat"], meta["lng"], hlat, hlng).min()
                                       < EXTRA_EXCLUDE_KM):
                misses += 1
                continue
            with lock:
                if meta["pano_id"] in seen:
                    continue
                seen.add(meta["pano_id"])
            try:
                download_panorama(meta).save(os.path.join(EXTRA_DIR, "panos", meta["pano_id"] + ".jpg"), quality=90)
            except Exception:
                misses += 1
                continue
            rec["label"] = cc
            with lock:
                with open(idx_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            out.append(rec)
            got += 1
        return out

    # chunks of <= 25 panoramas, round-robin over the countries (largest need first), so that the download
    # threads stay busy and an interrupted run has covered every country in proportion
    tasks = []
    for cc in need:
        tasks += [(cc, k, min(25, need[cc] - 25 * k)) for k in range((need[cc] + 24) // 25)]
    tasks.sort(key=lambda t: (t[1] / ((need[t[0]] + 24) // 25), -need[t[0]]))
    batches = [tasks[i:i + max(1, args.batch // 25)] for i in range(0, len(tasks), max(1, args.batch // 25))]
    ext = None  # feature extraction of the previous batch runs while the next one downloads
    for b in batches:
        t0 = time.time()
        with ThreadPoolExecutor(min(4, args.threads)) as ex:
            recs = [r for rs in ex.map(sample_country, b) for r in rs]
        print("batch of %d chunks (%s): %d panoramas in %.0fs" % (len(b), ",".join(sorted({t[0] for t in b})),
                                                                len(recs), time.time() - t0), flush=True)
        if ext is not None:
            ext.join()
        ext = threading.Thread(target=_extract_extra, args=(groups, recs, args.workers))
        ext.start()
    if ext is not None:
        ext.join()


def hashlib_md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def load_card_sets(D, exclude=("test",), refresh=False):
    """[(country, [region indices], placements)] of GeoGuessr's regional cards.  Read from the snapshot
    CARDS_SNAPSHOT (so that a refit is reproducible while the crawler keeps appending to
    data/calibration/pano_clues.json); built from pano_clues.json (read-only, retried while the crawler
    rewrites it) when missing or with refresh.  Placements on panoramas of the excluded splits are not counted."""
    _, codes, _, rcc, _ = _regions()
    cpos = {c: i for i, c in enumerate(codes)}
    if os.path.exists(CARDS_SNAPSHOT) and not refresh:
        snap = json.load(open(CARDS_SNAPSHOT))
        return [(cc, [cpos[x] for x in regs if x in cpos], n) for cc, regs, n in snap["cards"]]
    from engine.hints import ClueBase
    path = os.path.join(ROOT, "data", "calibration", "pano_clues.json")
    data = raw = None
    for _ in range(20):
        try:
            with open(path, "rb") as f:
                raw = f.read()
            data = json.loads(raw.decode("utf-8"))
            break
        except ValueError:
            time.sleep(1.5)
    if not data:
        return []
    skip = {r["pano_id"] for r, sp in zip(D["recs"], D["split"]) if sp in exclude}
    kb = ClueBase()
    cards = {}
    for pid, clues in data.items():
        if pid in skip or not isinstance(clues, list):
            continue
        for c in clues:
            ids = c.get("seterraRegionIds")
            cc = (c.get("countryCode") or "").upper()
            if not ids or not cc:
                continue
            key = (cc, c.get("id"))
            if key not in cards:
                regs = sorted({cpos[x] for x in kb._region_codes(cc, ids) if x in cpos and rcc[cpos[x]] == cc})
                cards[key] = [regs, 0]
            cards[key][1] += 1
    out = [(cc, regs, n) for (cc, _), (regs, n) in sorted(cards.items(), key=lambda t: (t[0][0], str(t[0][1])))
           if regs]
    os.makedirs(os.path.dirname(CARDS_SNAPSHOT), exist_ok=True)
    json.dump({"source": "data/calibration/pano_clues.json", "source_sha1": hashlib.sha1(raw).hexdigest()[:12],
               "panoramas": len(data), "excluded_splits": list(exclude), "built": time.strftime("%Y-%m-%d %H:%M"),
               "cards": [[cc, [codes[r] for r in regs], n] for cc, regs, n in out]}, open(CARDS_SNAPSHOT, "w"))
    return out


def hash5(s, k=5):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % k


def sub(X, rows):
    return {g: m[rows] for g, m in X.items()}


def fit_model(D, rows, params=None):
    return RegionModel(params).fit(sub(D["X"], rows), D["label"][rows], D["region"][rows], D["lat"][rows],
                                   D["lng"][rows], D["duel"][rows], D.get("cards", ()), D["prior"][rows])


def collect(model, D, rows):
    """{country: (rows, true region position or -1, raw components)} for rows of countries with a model."""
    out = {}
    for cc in np.unique(D["label"][rows]):
        if cc not in model.cidx:
            continue
        q = rows[D["label"][rows] == cc]
        regs = model.regions_of(cc)
        pos = {r: i for i, r in enumerate(regs)}
        ti = np.array([pos.get(r, -1) for r in D["region"][q]])
        e = model.embed(sub(D["X"], q), cc)
        out[cc] = (q, ti, model.components(e, cc, None if D["sun"] is None else D["sun"][q]))
    return out


def score(model, col, params=None, rows=None):
    """(log P(true region | true country) floored, rank of the true region): per collected row, or
    aligned with rows (rows of countries without a region model count as misses)."""
    L, R, Q = [], [], []
    for cc, (q, ti, comp) in col.items():
        P = model.combine(comp, cc, params)
        pt = np.where(ti >= 0, P[np.arange(len(ti)), np.maximum(ti, 0)], 0.0)
        L.append(np.log(np.maximum(pt, FLOOR)))
        R.append(np.where((ti >= 0) & (pt > 0), rank_of(P, pt), 999))
        Q.append(q)
    L, R, Q = np.concatenate(L), np.concatenate(R), np.concatenate(Q)
    if rows is None:
        return L, R
    pos = {r: k for k, r in enumerate(rows)}
    Lr, Rr = np.full(len(rows), np.log(FLOOR)), np.full(len(rows), 999)
    for l, r, q in zip(L, R, Q):
        Lr[pos[q]], Rr[pos[q]] = l, r
    return Lr, Rr


def rank_of(P, pt):
    """Rank of the true probability pt among P (rows of P against pt per row): regions above it plus half
    of the other regions tied with it (expected rank under random tie-breaking)."""
    P = np.atleast_2d(P)
    pt = np.reshape(pt, (-1, 1))
    tie = np.isclose(P, pt, rtol=1e-9, atol=0.0)
    r = (P > pt).sum(1) + 0.5 * np.maximum(tie.sum(1) - 1, 0)
    return r if len(r) > 1 else r[0]


def summary(L, R, sel=None):
    sel = np.ones(len(L), bool) if sel is None else sel
    L, R = L[sel], R[sel]
    if not len(L):
        return {"n": 0}
    return {"n": int(len(L)), "top1": round(float((R == 0).mean()), 4), "top3": round(float((R < 3).mean()), 4),
            "top5": round(float((R < 5).mean()), 4), "logp": round(float(L.mean()), 4)}


def paired(R, Rb, sel=None):
    """New minus GeoModel top1 / top3 rates with their paired standard errors."""
    sel = np.ones(len(R), bool) if sel is None else sel
    out = {}
    for k, name in ((1, "top1"), (3, "top3")):
        d = (R[sel] < k).astype(float) - (Rb[sel] < k).astype(float)
        out[name] = round(float(d.mean()), 4)
        out[name + "_se"] = round(float(d.std(ddof=1) / np.sqrt(max(len(d), 1))), 4) if len(d) > 1 else None
    return out


def fmt(s):
    return "n=%d top1 %.3f top3 %.3f top5 %.3f logp %.3f" % (s["n"], s["top1"], s["top3"], s["top5"], s["logp"]) \
        if s["n"] else "n=0"


def calibrate(model, items, iters=3):
    """Coordinate ascent of the exponents, prior mix and smoothing on the mean log P(true region | true
    country) of items = [(model, collected rows, rows)]."""
    p = json.loads(json.dumps(model.params))

    def obj(q):
        return float(np.mean(np.concatenate([score(m, c, q, r)[0] for m, c, r in items])))

    grids = {"w.prior": [0.4, 0.6, 0.8, 1.0, 1.2], "w.lda": [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7],
             "w.knn": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7], "w.sun": [0.0, 0.2, 0.4, 0.7, 1.0],
             "beta": [0.0, 0.1, 0.3, 0.6, 1.0], "alpha": [0.1, 0.3, 0.5, 1.0, 2.0, 4.0],
             "knn_s": [0.5, 1.0, 2.0, 4.0], "smooth": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
             "smooth_km": [100.0, 200.0, 400.0, 800.0], "smooth_card": [0.0, 0.05, 0.1, 0.2, 0.3]}
    best = start = obj(p)
    for _ in range(iters):
        for key, grid in grids.items():
            for v in grid:
                q = json.loads(json.dumps(p))
                if key.startswith("w."):
                    q["weights"][key[2:]] = v
                else:
                    q[key] = v
                s = obj(q)
                if s > best + 1e-6:
                    best, p = s, q
    return p, best, start


def cmd_cv(args, D, groups):
    """Out-of-fold check: fit on 4/5 of train (folds by location), score the held-out ranked-duel panoramas
    (true country) that have no training panorama within FOLD_KM; the exponents / prior / smoothing used for
    fold k are calibrated on the held-out rows of fold k+1 (cross-fitted, so the scores are out of sample)."""
    tr = np.flatnonzero(D["split"] == "train")
    twins = cross_fold_twins(D)
    held = D["duel"] & ~twins
    print("CV: %d held-out duel panoramas, %d not scored (training panorama of another fold within %.0f km)"
          % (int((D["duel"][tr]).sum()), int((D["duel"] & twins).sum()), FOLD_KM), flush=True)
    configs = [json.loads(args.params or "{}")]
    if args.grid:
        base = {"pre": "qn", "shrink": 0.1, "geo_sigma_km": 0.0, "own": 0.0, "K": 32}
        configs = [dict(base, **c) for c in ({"pre": ""}, {}, {"shrink": 0.3}, {"shrink": 0.5},
                                             {"shrink": 0.3, "K": 48}, {"shrink": 0.3, "own": 0.5},
                                             {"shrink": 0.3, "geo_sigma_km": 150.0},
                                             {"shrink": 0.3, "own": 0.5, "geo_sigma_km": 150.0})]
    out = []
    F = args.folds
    for cfg in configs:
        t0 = time.time()
        items = []
        for k in range(F):
            m = fit_model(D, tr[D["fold"][tr] != k], dict(DEFAULT_PARAMS, **cfg))
            q = tr[(D["fold"][tr] == k) & held[tr]]
            items.append((m, collect(m, D, q), q))
        ps = [calibrate(items[(k + 1) % F][0], [items[(k + 1) % F]], iters=2)[0] for k in range(F)]
        L, R = [np.concatenate(x) for x in zip(*[score(m, c, ps[k], q) for k, (m, c, q) in enumerate(items)])]
        s = summary(L, R)
        print("%-55s %s  %.0fs | fold 0: %s" % (json.dumps(cfg), fmt(s), time.time() - t0,
                                               json.dumps({k: ps[0][k] for k in ("weights", "beta", "alpha", "smooth",
                                                                                 "smooth_km", "smooth_card", "knn_s")})),
              flush=True)
        if args.by_country:
            lab = D["label"][np.concatenate([q for _, _, q in items])]
            print("    " + " ".join("%s %.3f/%.3f" % (cc, (R[lab == cc] == 0).mean(), (R[lab == cc] < 3).mean())
                                    for cc in TOP_COUNTRIES if (lab == cc).any()), flush=True)
        out.append((cfg, s, ps))
    return out


def baseline_region(geo, lp, d2, bounds=None):
    return geo.region_posterior(lp, d2, bounds=bounds)


def evaluate(model, geo, D, rows, label, ev, setups_map, verbose=True):
    """Region metrics given the true country / unconditional and GeoGuessr points, new vs GeoModel."""
    _, codes, _, rcc, _ = _regions()
    cidx = {c: i for i, c in enumerate(geo.classes)}
    true_reg = D["region"][rows]
    lab = D["label"][rows]
    # given the true country: P(region | true country) of both models over the regions of the country,
    # also summed over the areas of model.areas (tiny admin-1 units merged)
    col = collect(model, D, rows)
    pos = {r: k for k, r in enumerate(rows)}
    new, area_of = [None] * len(rows), {}
    for cc, (q, ti, comp) in col.items():
        P = model.combine(comp, cc)
        for k, r in enumerate(q):
            new[pos[r]] = P[k]
        area_of[cc] = model.areas(cc, AREA_MIN_REFS)
    res = {"new": ([], [], [], []), "geomodel": ([], [], [], [])}
    for k in range(len(rows)):
        cc, t = lab[k], true_reg[k]
        if cc not in model.cidx or cc not in cidx:
            for v in res.values():
                v[0].append(np.log(FLOOR)); v[1].append(999); v[2].append(np.log(FLOOR)); v[3].append(999)
            continue
        regs = model.regions_of(cc)
        lp = np.full(len(geo.classes), -1e9)
        lp[cidx[cc]] = 0.0
        pb = baseline_region(geo, lp, ev["d2"][k])
        pb = np.array([pb[r] if r < len(pb) else 0.0 for r in regs])
        pb = pb / pb.sum() if pb.sum() > 0 else pb  # the country's own regions (references abroad cut off)
        ti = int(np.flatnonzero(regs == t)[0]) if t in regs else -1
        for key, pv in (("new", new[k]), ("geomodel", pb)):
            pa = np.bincount(area_of[cc], weights=pv)
            for p_, t_, a, b in ((pv, ti, 0, 1), (pa, area_of[cc][ti] if ti >= 0 else -1, 2, 3)):
                pt = p_[t_] if t_ >= 0 else 0.0
                res[key][a].append(np.log(max(pt, FLOOR)))
                res[key][b].append(float(rank_of(p_, pt)) if pt > 0 else 999)
    res = {key: [np.array(x) for x in v] for key, v in res.items()}
    (L, R, La, Ra), (bl, br, bla, bra) = res["new"], res["geomodel"]
    multi = np.array([cc in area_of and len(area_of[cc]) > area_of[cc].max() + 1 for cc in lab])
    out = {"given_true_country": {"new": summary(L, R), "geomodel": summary(bl, br), "diff": paired(R, br),
                                  "by_country": {},
                                  "areas": {"min_refs": AREA_MIN_REFS, "new": summary(La, Ra),
                                            "geomodel": summary(bla, bra),
                                            "merged_countries_only": {"new": summary(La, Ra, multi),
                                                                      "geomodel": summary(bla, bra, multi)}}}}
    for cc in TOP_COUNTRIES:
        s_ = lab == cc
        if s_.sum():
            out["given_true_country"]["by_country"][cc] = {"new": summary(L, R, s_), "geomodel": summary(bl, br, s_)}
    # unconditional and points, without / with map info
    maps = round_maps(D["recs"], list(rows))
    for kind in ("no_map", "map"):
        lpm = ev["lp_" + kind]
        nl, nr, gl, gr, pts_new, pts_old, right = [], [], [], [], [], [], []
        for k, i in enumerate(rows):
            s = setups_map[k] if kind == "map" else {"scale_km": None, "bounds": None}
            lp, d2 = lpm[k], ev["d2"][k]
            post = np.exp(lp)
            mix, rp = region_posterior(sub(D["X"], [i]), post, model, geo.classes, by_country="both",
                                       bounds=s["bounds"])
            t = true_reg[k]
            code_t = codes[t] if t else None
            pt = mix.get(code_t, 0.0) if code_t else 0.0
            nl.append(np.log(max(pt, FLOOR)))
            nr.append(float(rank_of(np.array(list(mix.values())), pt)) if pt > 0 else 999)
            pb = baseline_region(geo, lp, d2, s["bounds"])
            ptb = pb[t] if 0 < t < len(pb) else 0.0
            gl.append(np.log(max(ptb, FLOOR)))
            gr.append(float(rank_of(pb, ptb)) if ptb > 0 else 999)
            right.append(lab[k] in cidx and int(np.argmax(post)) == cidx[lab[k]])
            me = (maps[k] or {}).get("maxErrorDistance")
            for w, acc in ((None, pts_old), (location_weights(geo, lp, d2, rp, bounds=s["bounds"]), pts_new)):
                g = geo.locate(lp, d2, w=w, score_scale_km=s["scale_km"], bounds=s["bounds"])
                d = float(haversine_km(D["lat"][i], D["lng"][i], g["lat"], g["lng"]))
                acc.append(float(geoguessr_points(d, me)))
        nl, nr, gl, gr, right = map(np.array, (nl, nr, gl, gr, right))
        pts_new, pts_old = np.array(pts_new), np.array(pts_old)
        diff = pts_new - pts_old
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        out[kind] = {"unconditional": {"new": summary(nl, nr), "geomodel": summary(gl, gr), "diff": paired(nr, gr)},
                     "if_country_right": {"new": summary(nl, nr, right), "geomodel": summary(gl, gr, right)},
                     "points": {"new": round(float(pts_new.mean()), 1), "geomodel": round(float(pts_old.mean()), 1),
                                "diff": round(float(diff.mean()), 1), "diff_se": round(float(se), 1),
                                "diff_if_country_right": round(float(diff[right].mean()), 1) if right.any() else None}}
    if verbose:
        g = out["given_true_country"]
        print("[%s] given the true country: new %s | GeoModel %s" % (label, fmt(g["new"]), fmt(g["geomodel"])))
        print("    paired diff top1 %+.3f +- %.3f, top3 %+.3f +- %.3f" % (g["diff"]["top1"], g["diff"]["top1_se"],
                                                                     g["diff"]["top3"], g["diff"]["top3_se"]))
        a = g["areas"]
        print("    areas (>= %d refs): new %s | GeoModel %s" % (AREA_MIN_REFS, fmt(a["new"]), fmt(a["geomodel"])))
        print("    areas, rounds in countries with merged regions: new %s | GeoModel %s"
              % (fmt(a["merged_countries_only"]["new"]), fmt(a["merged_countries_only"]["geomodel"])))
        for cc, v in g["by_country"].items():
            print("    %s  new %s | GeoModel %s" % (cc, fmt(v["new"]), fmt(v["geomodel"])))
        for kind in ("no_map", "map"):
            o = out[kind]
            print("[%s, %s] unconditional: new %s | GeoModel %s" % (label, kind, fmt(o["unconditional"]["new"]),
                                                                     fmt(o["unconditional"]["geomodel"])))
            dd = o["unconditional"]["diff"]
            print("    paired diff top1 %+.3f +- %.3f, top3 %+.3f +- %.3f" % (dd["top1"], dd["top1_se"], dd["top3"],
                                                                         dd["top3_se"]))
            print("    top country right: new %s | GeoModel %s" % (fmt(o["if_country_right"]["new"]),
                                                                  fmt(o["if_country_right"]["geomodel"])))
            p = o["points"]
            print("    points/round: region-weighted guess %.0f | GeoModel %.0f | diff %+.1f +- %.1f (country right %s)"
                  % (p["new"], p["geomodel"], p["diff"], p["diff_se"], p["diff_if_country_right"]))
    return out


def geo_evidence(geo, D, rows):
    """GeoModel country log posteriors (no map / map info), kernel distances and map setups."""
    ev = geo.evidence(sub(D["X"], rows))
    lp_no = geo.combine(ev)
    maps = round_maps(D["recs"], list(rows))
    cache, setups = {}, []
    for mp in maps:
        key = (mp or {}).get("id") or (mp or {}).get("name")
        if key not in cache:
            cache[key] = geo.map_setup(mp)
        setups.append(cache[key])
    lp_map = np.zeros_like(lp_no)
    for k, s in enumerate(setups):
        lp_map[k] = geo.combine({a: (v[k:k + 1] if isinstance(v, np.ndarray) else v) for a, v in ev.items()},
                                s["weights"], prior=s["prior"])[0]
    return {"lp_no_map": lp_no, "lp_map": lp_map, "d2": ev["_d2"]}, setups


def cmd_check_compat(args, groups):
    """Feature groups whose extra references were extracted with another module version than the main cache:
    re-extract the main cache's version check sample (args.n train panoramas, both car-axis regimes) with the
    current module; if the cached features are reproduced exactly, the (extra, main) hash pair is recorded as
    equivalent in COMPAT_FILE."""
    import multiprocessing
    from calibrate import _extract_one, _init
    from engine import features
    compat = load_compat()
    recs = sorted((r for r in load_records() if split_of(r) == "train"),
                  key=lambda r: hashlib.md5(r["pano_id"].encode()).hexdigest())
    for g in groups:
        hm, hc = main_hash(g), module_hash(features.load(g))
        stored = [h for _, h in extra_files(g)]
        if hm in stored:
            print("%s: extra features stored under the main cache's hash %s" % (g, hm))
            continue
        if hc not in stored:
            print("%s: cannot check, no extra features under the current module hash %s (stored %s)" % (g, hc, stored))
            continue
        z = np.load(os.path.join(FEAT_DIR, g + ".npz"), allow_pickle=True)
        idx = {pid: i for i, pid in enumerate(z["ids"])}
        rows = [r for r in recs if r["pano_id"] in idx][:args.n]
        jobs = [(os.path.join(DATASET, "panos", r["pano_id"] + ".jpg"), r.get("heading")) for r in rows]
        t0 = time.time()
        with multiprocessing.Pool(min(4, args.workers), initializer=_init, initargs=([g],)) as pool:
            A = np.array([x[0] for x in pool.imap(_extract_one, jobs, chunksize=4)], np.float32)
        B = z["X"][[idx[r["pano_id"]] for r in rows]].astype(np.float32)
        nan_mis = int((np.isnan(A) != np.isnan(B)).sum())
        both = np.isfinite(A) & np.isfinite(B)
        d = float(np.abs(A - B)[both].max()) if both.any() else 0.0
        eq = bool(nan_mis == 0 and d == 0.0)
        e = {"extra": hc, "main": hm, "n": len(rows), "max_abs_diff": d, "nan_mismatch": nan_mis, "equivalent": eq,
             "checked": time.strftime("%Y-%m-%d %H:%M")}
        compat[g] = [x for x in compat.get(g, []) if (x.get("extra"), x.get("main")) != (hc, hm)] + [e]
        print("%s: current module %s on %d main-cache panoramas (%.0fs): max |diff| %.3g, NaN mismatches %d -> %s"
              % (g, hc, len(rows), time.time() - t0, d, nan_mis, "equivalent to %s" % hm if eq else "NOT equivalent"))
    os.makedirs(os.path.dirname(COMPAT_FILE), exist_ok=True)
    json.dump(compat, open(COMPAT_FILE, "w"), indent=1)


def _live_one(job):
    """Features of one panorama as the live script captures it (eval_live.render_views: 2 x 5 views, random start
    yaw seeded by the pano id, true north known, car axis unknown)."""
    import random
    import calibrate
    from eval_live import render_views
    from engine.panorama import SphericalImage
    path, heading, pid = job
    sph = SphericalImage.from_equirect(path, heading=heading)
    live = SphericalImage.from_views(render_views(sph, random.Random(pid).uniform(0, 360)), width=2048, heading=0.0)
    out = []
    for m in calibrate._WORK_MODS:
        try:
            x = np.asarray(m.extract(live)["x"], np.float32)
        except Exception:
            x = np.full(len(m.FEATURE_NAMES), np.nan, np.float32)
        out.append(x)
    return out


def live_features(D, rows, groups, workers=4):
    """{group: (len(rows), d)} live-capture features of rows (cached in LIVE_FILE per module hash)."""
    import multiprocessing
    from calibrate import _init
    from engine import features
    hashes = {g: module_hash(features.load(g)) for g in groups}
    ids = [D["recs"][i]["pano_id"] for i in rows]
    cache = {}
    if os.path.exists(LIVE_FILE):
        z = np.load(LIVE_FILE, allow_pickle=True)
        if json.loads(str(z["hashes"])) == hashes:
            cache = {pid: [z[g][k] for g in groups] for k, pid in enumerate(z["ids"])}
    todo = [i for i, pid in zip(rows, ids) if pid not in cache]
    if todo:
        print("live capture features of %d panoramas (%d workers)" % (len(todo), workers), flush=True)
        jobs = [(os.path.join(DATASET, "panos", D["recs"][i]["pano_id"] + ".jpg"), D["recs"][i].get("heading"),
                 D["recs"][i]["pano_id"]) for i in todo]
        t0 = time.time()
        with multiprocessing.Pool(min(4, workers), initializer=_init, initargs=(list(groups),)) as pool:
            for k, x in enumerate(pool.imap(_live_one, jobs, chunksize=2)):
                cache[jobs[k][2]] = x
                if (k + 1) % 50 == 0:
                    print("  %d/%d %.2f pano/s" % (k + 1, len(jobs), (k + 1) / (time.time() - t0)), flush=True)
        allid = sorted(cache)
        os.makedirs(os.path.dirname(LIVE_FILE), exist_ok=True)
        np.savez_compressed(LIVE_FILE, ids=np.array(allid), hashes=np.array(json.dumps(hashes)),
                            **{g: np.array([cache[p][j] for p in allid], np.float32) for j, g in enumerate(groups)})
    return {g: np.array([cache[p][j] for p in ids], np.float64) for j, g in enumerate(groups)}


def provenance(D, groups):
    """Inputs of a fit / report: GeoModel files, feature caches, extra references, card snapshot."""
    return {"geomodel": {f: file_hash(os.path.join(MODEL_DIR, f)) for f in ("model.json", "priors.json", "model.npz")},
            "features_main": {g: main_hash(g) for g in groups}, "features_extra": D.get("extra_hashes", {}),
            "extra_references": int((~D["prior"]).sum()), "card_sets": file_hash(CARDS_SNAPSHOT),
            "regions_npz": file_hash(os.path.join(MODEL_DIR, REGIONS_FILE)), "cv_fold_km": FOLD_KM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="out-of-fold check on the duel train panoramas")
    ap.add_argument("--grid", action="store_true", help="with --cv: structural parameter grid")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--by-country", action="store_true", help="with --cv: top1/top3 of the main countries")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--params", default="", help="JSON overrides of the structural parameters")
    ap.add_argument("--calib-only", action="store_true", help="no TEST report (model selection)")
    ap.add_argument("--no-extra", action="store_true", help="without the extra references of --fetch-extra")
    ap.add_argument("--fetch-extra", action="store_true", help="download the extra region references")
    ap.add_argument("--check-extra-compat", action="store_true", help="verify extra features of another module hash")
    ap.add_argument("--live", action="store_true", help="saved model on CALIB under live capture")
    ap.add_argument("--refresh-cards", action="store_true", help="rebuild the regional card snapshot")
    ap.add_argument("--n", type=int, default=200, help="--check-extra-compat: panoramas to re-extract")
    ap.add_argument("--batch", type=int, default=1500, help="--fetch-extra: panoramas per download batch")
    ap.add_argument("--threads", type=int, default=4, help="--fetch-extra: download threads (<= 4)")
    ap.add_argument("--workers", type=int, default=4, help="feature processes (<= 4)")
    args = ap.parse_args()
    geo = GeoModel.load()
    if args.check_extra_compat:
        cmd_check_compat(args, geo.groups)
        return
    D = load_data(geo.groups, extra=not args.no_extra and not args.fetch_extra)
    if args.fetch_extra:
        cmd_fetch_extra(args, D, geo.groups)
        return
    D["cards"] = load_card_sets(D, refresh=args.refresh_cards)
    print("panoramas", {s: int((D["split"] == s).sum()) for s in ("train", "calib", "test")},
          "of them extra references", int((~D["prior"]).sum()),
          "regional cards", len(D["cards"]), "placements", sum(n for _, _, n in D["cards"]), flush=True)
    if args.cv:
        cmd_cv(args, D, geo.groups)
        return
    tr = np.flatnonzero(D["split"] == "train")
    ca = np.flatnonzero(D["split"] == "calib")
    te = np.flatnonzero(D["split"] == "test")
    out_path = os.path.join(MODEL_DIR, "eval_regions.json")
    if args.live:
        model = RegionModel.load()
        rows = np.array([i for i in ca if D["recs"][i].get("heading") is not None])
        Xl = live_features(D, rows, geo.groups, args.workers)
        Dl = dict(D, X={g: m.copy() for g, m in D["X"].items()})
        for g in geo.groups:
            Dl["X"][g][rows] = Xl[g]
        Dl["sun"] = D["sun"].copy() if D["sun"] is not None else None
        if Dl["sun"] is not None:
            Dl["sun"][rows] = sun_mixture(Xl)
        ev, setups = geo_evidence(geo, Dl, rows)
        res = evaluate(model, geo, Dl, rows, "CALIB live", ev, setups)
        res["note"] = ("live capture (10 rendered views, car axis unknown, no clue-card detections) of the CALIB "
                       "rounds; exponents calibrated on the full-panorama CALIB features")
        res["provenance"] = provenance(D, geo.groups)
        out = json.load(open(out_path)) if os.path.exists(out_path) else {}
        out["calib_live"] = res
        json.dump(out, open(out_path, "w"), indent=1)
        print("saved eval_regions.json calib_live")
        return
    if args.eval_only:
        model = RegionModel.load()
    else:
        t0 = time.time()
        model = fit_model(D, tr, dict(DEFAULT_PARAMS, **json.loads(args.params or "{}")))
        print("fit %.1fs: %d countries, %d regions" % (time.time() - t0, len(model.countries), len(model.reg_idx)))
        p, best, start = calibrate(model, [(model, collect(model, D, ca), ca)])
        model.params = p
        print("calibrated on CALIB: mean log P(true region | true country) %.3f (start %.3f)" % (best, start))
        print("params", json.dumps({k: v for k, v in p.items()}))
    out = {"params": model.params}
    for rows, label in ((ca, "calib"), (te, "TEST"))[:1 if args.calib_only else 2]:
        ev, setups = geo_evidence(geo, D, rows)
        out[label.lower()] = evaluate(model, geo, D, rows, label, ev, setups)
    out["calib"]["note"] = "in-sample: exponents / prior / smoothing calibrated on these rounds"
    if args.save:
        model.save()
        out["provenance"] = provenance(D, geo.groups)
        json.dump(out, open(out_path, "w"), indent=1)
        print("saved data/model/%s (%.1f MB), eval_regions.json"
              % (REGIONS_FILE, os.path.getsize(os.path.join(MODEL_DIR, REGIONS_FILE)) / 1e6))
    else:
        print("provenance", json.dumps(provenance(D, geo.groups)))


if __name__ == "__main__":
    main()
