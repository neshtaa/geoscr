#!/usr/bin/env python3
"""
Calibration / evaluation harness.

  python3 tools/calibrate.py features [--modules road,solar] [--workers 14] [--force]
      compute and cache feature vectors for every panorama of the dataset
      (scratch/features/<module>.npz); recomputes when the module source changed (or with --force).
      Image-less panoramas (below) keep their rows only if the recomputed panoramas reproduce the cache
      exactly (>= MIN_EQUIV_ROWS, both car-axis regimes); otherwise the rows go to scratch/features/stale/,
      a warning says how many training panoramas are lost and the exit status is 3.
  python3 tools/calibrate.py eval-group <module>
      standalone discriminative power of one feature group: country top-1/top-5
      accuracy and information gain (bits) on the real-game test set, trained on
      the world+balanced split, plus a per-feature ANOVA F ranking.
  python3 tools/calibrate.py stats
      dataset summary.

Splits: train = world + balanced panoramas, test = history (+ postmatch) panoramas
from the user's real GeoGuessr games.  Training records on or within OWN_KM (1 km) of one of the user's
rounds (data/calibration/history_rounds.json and every calib / test / other record) are left out by
load_records.

Streamed panoramas (tools/stream_duels.py) have an index record {"pano_deleted": true} and features in
the caches but no image: load_records keeps them (load_records(pixels=True) leaves them out), every
step that needs pixels skips them with a message, and `python3 tools/stream_duels.py --refresh`
downloads them again when a feature module changed.  The caches are written under a lock
(scratch/features/.lock) to a temp file + os.replace, so concurrent appends are merged, not lost.
"""
import argparse
import contextlib
import fcntl
import gc
import hashlib
import inspect
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATASET = os.path.join(ROOT, "scratch", "dataset")
FEAT_DIR = os.path.join(ROOT, "scratch", "features")
# exact re-extraction checks "module hash A reproduces the cache stored under hash B"
COMPAT_FILES = [os.path.join(FEAT_DIR, "compat.json"),                                # tools/stream_duels.py
                os.path.join(ROOT, "scratch", "regions", "extra", "features", "compat.json")]  # eval_regions.py
NO_PIXELS_MSG = ("panoramas without an image (pano_deleted: streamed by tools/stream_duels.py, features only)")
# the user's own rounds (tools/build_dataset.py history); no training record on or within OWN_KM of one
OWN_FILE = os.path.join(ROOT, "data", "calibration", "history_rounds.json")
OWN_KM = 1.0
# a changed module (or --force) keeps the rows of image-less panoramas only if it reproduces the cached rows
# of at least this many recomputed panoramas (both car-axis regimes) exactly
MIN_EQUIV_ROWS = 100

# GeoGuessr country codes that Street View metadata reports differently
LABEL_FIX = {"UK": "GB"}


def label_of(r):
    for k in ("country_code", "gg_country", "raster_country"):
        v = r.get(k)
        if v:
            v = v.upper()
            return LABEL_FIX.get(v, v)
    return None


def pano_path(pano_id, dataset=DATASET):
    return os.path.join(dataset, "panos", pano_id + ".jpg")


def has_pixels(r, dataset=DATASET):
    """The panorama image is on disk (False for the image-less records of tools/stream_duels.py)."""
    return not r.get("pano_deleted") and os.path.exists(pano_path(r["pano_id"], dataset))


def _within_km(lat, lng, plat, plng, km):
    """(n,) bool: point i is within km of one of the points (plat, plng) (latitude window, then haversine)."""
    from engine.geo import haversine_km
    lat, lng = np.asarray(lat, float), np.asarray(lng, float)
    out = np.zeros(len(lat), bool)
    if not len(plat) or not len(lat):
        return out
    o = np.argsort(plat)
    plat, plng = np.asarray(plat, float)[o], np.asarray(plng, float)[o]
    w = km / 110.0  # >= the latitude span of km anywhere (1 degree >= 110.57 km)
    lo, hi = np.searchsorted(plat, lat - w, "left"), np.searchsorted(plat, lat + w, "right")
    cand = np.flatnonzero(hi > lo)
    cnt = (hi - lo)[cand]
    ii = np.repeat(cand, cnt)  # every (point, reference in its latitude window) pair
    jj = np.repeat(lo[cand], cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    out[ii[haversine_km(lat[ii], lng[ii], plat[jj], plng[jj]) < km]] = True
    return out


def near_own(recs, own_file=None, km=OWN_KM):
    """Pano ids of training records (world / balanced / duel) that are one of the user's own rounds (pano id)
    or within km of one: the own rounds are those of own_file (tools/build_dataset.py history) and every
    calib / test / other record (panorama and round position).  load_records drops them, so a reference
    can never sit on a location of the evaluation rounds, whatever was crawled or streamed before."""
    try:
        with open(OWN_FILE if own_file is None else own_file) as f:
            own = json.load(f)
    except (OSError, ValueError):
        own = []
    held = [r for r in recs if split_of(r) != "train"]
    ids = {r.get("pano_id") for r in own} | {r["pano_id"] for r in held}
    pts = [(r["lat"], r["lng"]) for r in own if r.get("lat") is not None]
    for r in held:
        pts.append((r["lat"], r["lng"]))
        if r.get("round_lat") is not None:
            pts.append((r["round_lat"], r["round_lng"]))
    P = np.array(pts, float).reshape(-1, 2)
    tr = [r for r in recs if split_of(r) == "train"]
    out = {r["pano_id"] for r in tr if r["pano_id"] in ids}
    for la, ln in (("lat", "lng"), ("round_lat", "round_lng")):
        sub = [r for r in tr if r.get(la) is not None and r.get(ln) is not None]
        m = _within_km([r[la] for r in sub], [r[ln] for r in sub], P[:, 0], P[:, 1], km)
        out.update(r["pano_id"] for r, x in zip(sub, m) if x)
    return out


def load_records(dataset=DATASET, pixels=False, own_file=None, exclude_own=True):
    """Index records with a label, one per panorama: those whose image is on disk and the streamed
    ones ({"pano_deleted": true}: features in the caches, image deleted).  pixels=True: only records
    whose image is on disk (for steps that read the panorama).  A panorama with both a training record and
    a record of the user's own rounds is the user's round; training records on or within OWN_KM of the
    user's own rounds are left out (near_own; exclude_own=False keeps them)."""
    recs = {}
    with open(os.path.join(dataset, "index.jsonl")) as f:
        lines = f.readlines()
    gc_on = gc.isenabled()
    gc.disable()  # ~80k new dicts: the cyclic collector would rescan them over and over (no cycles here)
    try:
        for line in lines:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("pano_deleted"):
                if pixels:
                    continue
            elif not os.path.exists(pano_path(r["pano_id"], dataset)):
                continue
            r["label"] = label_of(r)
            if r["label"]:
                prev = recs.get(r["pano_id"])
                if prev is None or (split_of(prev) == "train" and split_of(r) != "train"):
                    recs[r["pano_id"]] = r
        out = list(recs.values())
        if exclude_own:
            drop = near_own(out, own_file)
            out = [r for r in out if r["pano_id"] not in drop]
    finally:
        if gc_on:
            gc.enable()
    out.sort(key=lambda r: r["pano_id"])
    return out


NON_WORLD_MAPS = {"Ukraine", "United Kingdom (Better Map)", "MLB and MiLB Stadiums 2019"}


def split_of(r):
    """train: random Street View samples + public ranked-duel rounds of other players;
    calib / test: halves (by game) of the user's real GeoGuessr World-map rounds;
    other: rounds from single-country maps."""
    if r["mode"] in ("world", "balanced", "duel"):  # duel = public ranked duels of other players
        return "train"
    if r.get("map") in NON_WORLD_MAPS:
        return "other"
    g = str(r.get("game") or r["pano_id"])
    return "calib" if int(hashlib.md5(g.encode()).hexdigest(), 16) % 2 == 0 else "test"


def in_eval(r):
    return split_of(r) in ("calib", "test")


def module_hash(mod):
    src = inspect.getsource(mod)
    for dep in getattr(mod, "DEPENDS", []):
        src += inspect.getsource(__import__(dep, fromlist=["x"]))
    return hashlib.sha1(src.encode()).hexdigest()[:12]


_WORK_MODS = None


def _init(names):
    global _WORK_MODS
    from engine import features
    _WORK_MODS = [features.load(n) for n in names]


def hide_car_axis(pano_id):
    """Half of the panoramas are processed as in live play, where the screenshots give the
    true heading (compass) but not the car's driving axis, so the model sees both regimes."""
    return int(hashlib.md5(pano_id.encode()).hexdigest(), 16) % 2 == 1


def _extract_one(args):
    path, heading = args
    from engine.panorama import SphericalImage
    if not os.path.exists(path):
        raise FileNotFoundError("%s: no image - %s are skipped by steps that need pixels "
                                "(select records with has_pixels / load_records(pixels=True))" % (path, NO_PIXELS_MSG))
    sph = SphericalImage.from_equirect(path, heading=heading)
    if hide_car_axis(os.path.basename(path)[:-4]):
        sph.car_heading = None
    out = []
    for m in _WORK_MODS:
        try:
            x = np.asarray(m.extract(sph)["x"], np.float32)
        except Exception as e:  # keep going; NaN row marks failure
            sys.stderr.write(f"[{m.NAME}] {os.path.basename(path)}: {e!r}\n")
            x = np.full(len(m.FEATURE_NAMES), np.nan, np.float32)
        out.append(x)
    return out


@contextlib.contextmanager
def cache_lock(feat_dir=FEAT_DIR):
    """Exclusive lock around a read-modify-write of the feature caches."""
    os.makedirs(feat_dir, exist_ok=True)
    with open(os.path.join(feat_dir, ".lock"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_cache(name, feat_dir=FEAT_DIR):
    """{"ids": [...], "X": (n, d) float32, "names": [...], "hash": str} of <feat_dir>/<name>.npz, or None."""
    path = os.path.join(feat_dir, name + ".npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as z:
        return {"ids": [str(p) for p in z["ids"]], "X": np.asarray(z["X"], np.float32),
                "names": [str(n) for n in z["names"]], "hash": str(z["hash"])}


def write_cache(name, ids, X, names, h, feat_dir=FEAT_DIR):
    """Atomic write of a feature cache: temp file in the same directory, fsync, os.replace (a crash leaves
    either the old or the new file, never a partial one).  Call under cache_lock()."""
    path = os.path.join(feat_dir, name + ".npz")
    tmp = "%s.tmp%d" % (path, os.getpid())
    X = np.asarray(X, np.float32).reshape(len(ids), len(names))
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, ids=np.array(ids), X=X, names=np.array(names), hash=h)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # exists, owned by someone else
        return True
    return True


def clean_tmp(feat_dir=FEAT_DIR, log=print):
    """Remove the temp files of cache writes killed half-way (<name>.npz.tmp<pid> of a process that is gone;
    `finally` does not run on SIGKILL).  Call under cache_lock(): every cache writer holds it."""
    removed = 0
    for d in (feat_dir, os.path.join(feat_dir, "stale")):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            head, sep, pid = f.rpartition(".npz.tmp")
            if not (head and sep and pid.isdigit()) or (int(pid) != os.getpid() and _pid_alive(int(pid))):
                continue
            try:
                removed += os.path.getsize(os.path.join(d, f))
                os.remove(os.path.join(d, f))
                log("removed %s (temp file of an interrupted cache write)" % os.path.join(d, f))
            except OSError:
                pass
    return removed


def equivalent_hashes(name, files=None):
    """{(module hash, cache hash)} pairs verified by an exact re-extraction (the module with the first hash
    reproduces the cache stored under the second)."""
    out = set()
    for f in (COMPAT_FILES if files is None else files):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        for e in d.get(name, []):
            a, b = e.get("module", e.get("extra")), e.get("cache", e.get("main"))
            if e.get("equivalent") and a and b:
                out.add((a, b))
    return out


def hash_ok(name, cache_hash, mod_hash, files=None):
    """Features computed by the module with mod_hash may be stored in a cache labelled cache_hash."""
    return cache_hash == mod_hash or (mod_hash, cache_hash) in equivalent_hashes(name, files)


class CacheMismatch(Exception):
    """A feature cache was built with another (not verified equivalent) module version."""


def upsert_features(rows, hashes, names, feat_dir=FEAT_DIR, compat_files=None, locked=False):
    """Add / replace rows of the feature caches: rows = {module: {pano_id: x}}, hashes = {module: hash of the
    module that computed them}, names = {module: FEATURE_NAMES}.  All caches are checked before anything is
    written (CacheMismatch when a cache's hash is neither the module hash nor verified equivalent to it,
    or its feature names differ); a missing cache is created.  Existing rows of other panoramas are kept;
    rows of the same panorama are replaced (re-running after a crash is idempotent).  locked=True: the
    caller holds cache_lock() (e.g. to append the index records under the same lock).  Returns
    {module: cache size}."""
    with (contextlib.nullcontext() if locked else cache_lock(feat_dir)):
        caches = {}
        for m in rows:
            c = read_cache(m, feat_dir)
            if c is not None:
                if list(c["names"]) != list(names[m]):
                    raise CacheMismatch("%s: cached feature names differ from the module's" % m)
                if not hash_ok(m, c["hash"], hashes[m], compat_files):
                    raise CacheMismatch("%s: cache built with module hash %s, rows computed with %s (not verified "
                                        "equivalent)" % (m, c["hash"], hashes[m]))
            caches[m] = c
        out = {}
        for m, new in rows.items():
            c = caches[m]
            if c is None:
                ids, X, label = [], np.zeros((0, len(names[m])), np.float32), hashes[m]
            else:
                ids, X, label = c["ids"], c["X"], c["hash"]  # keep the stored label (same or equivalent)
            pos = {p: i for i, p in enumerate(ids)}
            add = [p for p in new if p not in pos]
            X = np.concatenate([X, np.zeros((len(add), X.shape[1]), np.float32)]) if add else X.copy()
            ids = list(ids) + add
            pos.update({p: len(ids) - len(add) + k for k, p in enumerate(add)})
            for p, x in new.items():
                X[pos[p]] = np.asarray(x, np.float32)
            write_cache(m, ids, X, names[m], label, feat_dir)
            out[m] = len(ids)
        return out


def keep_stale(name, c, pids, feat_dir=FEAT_DIR):
    """Rows of image-less panoramas from a cache of an older module version -> <feat_dir>/stale/<name>.<hash>.npz
    (merged with what is there): they cannot be recomputed without downloading the images again."""
    pos = {p: i for i, p in enumerate(c["ids"])}
    keep = [p for p in pids if p in pos]
    if not keep:
        return None
    sdir = os.path.join(feat_dir, "stale")
    os.makedirs(sdir, exist_ok=True)
    tag = "stale/%s.%s" % (name, c["hash"])
    old = read_cache(tag, feat_dir)
    ids = list(old["ids"]) if old else []
    X = [old["X"]] if old else []
    have = set(ids)
    add = [p for p in keep if p not in have]
    ids += add
    X.append(c["X"][[pos[p] for p in add]])
    write_cache(tag, ids, np.concatenate(X), c["names"], c["hash"], feat_dir)
    return os.path.join(feat_dir, tag + ".npz")


def reproduces(results, j, c, min_rows=None):
    """Do the recomputed rows (results: {pano_id: [x per module]}, module j) equal the rows cached in c exactly
    (values and NaN pattern) on >= min_rows panoramas of both car-axis regimes?  -> (ok, compared, differing,
    max |diff|)."""
    min_rows = MIN_EQUIV_ROWS if min_rows is None else min_rows
    pos = {p: i for i, p in enumerate(c["ids"])}
    both = [p for p in results if p in pos]
    if not both:
        return False, 0, 0, 0.0
    A = np.array([results[p][j] for p in both], np.float32).reshape(len(both), -1)
    B = c["X"][[pos[p] for p in both]]
    if A.shape != B.shape:
        return False, len(both), len(both), float("inf")
    fa, fb = np.isfinite(A), np.isfinite(B)
    bad = ((np.isnan(A) != np.isnan(B)) | (fa & fb & (A != B))).any(1)
    d = float(np.abs(A - B)[fa & fb].max()) if (fa & fb).any() else 0.0
    regimes = {hide_car_axis(p) for p in both}
    ok = not bad.any() and len(both) >= min_rows and (len(regimes) == 2 or min_rows <= 1)
    return ok, len(both), int(bad.sum()), d


def _warn(msg):
    """A warning that must not get lost in the progress output (stdout and stderr)."""
    line = "!" * 100
    text = "\n".join([line] + ["WARNING: " + msg] + [line])
    print(text, flush=True)
    sys.stderr.write(text + "\n")


def cmd_features(args, dataset=DATASET, feat_dir=FEAT_DIR):
    """Returns 0, or 3 when features of image-less panoramas had to be dropped (moved to stale/)."""
    from engine import features
    names = args.modules.split(",") if args.modules else [m.NAME for m in features.available()]
    mods = [features.load(n) for n in names]
    recs = load_records(dataset, exclude_own=False)  # the caches cover every record (exclusion is a split matter)
    os.makedirs(feat_dir, exist_ok=True)
    with cache_lock(feat_dir):
        clean_tmp(feat_dir)
    todo_mods, cached = [], {}
    for m in mods:
        h = module_hash(m)
        c = read_cache(m.NAME, feat_dir)
        if c is not None and not args.force and c["names"] == list(m.FEATURE_NAMES) and hash_ok(m.NAME, c["hash"], h):
            cached[m.NAME] = dict(zip(c["ids"], c["X"]))
        todo_mods.append((m, h))
    # only panoramas missing from each module's cache
    if args.sample:
        tr = [r for r in recs if split_of(r) == "train"]
        tr.sort(key=lambda r: hashlib.md5(r["pano_id"].encode()).hexdigest())
        recs = [r for r in recs if in_eval(r)] + tr[: args.sample]
    need = [r for r in recs if any(r["pano_id"] not in cached.get(m.NAME, {}) for m, _ in todo_mods)]
    blind = [r for r in need if not has_pixels(r, dataset)]
    need = [r for r in need if has_pixels(r, dataset)]
    print(f"{len(recs)} panoramas, {len(need)} to process for modules {names}", flush=True)
    redo = [m.NAME for m, _ in todo_mods if m.NAME not in cached and os.path.exists(os.path.join(feat_dir, m.NAME + ".npz"))]
    if blind:
        print(f"{len(blind)} {NO_PIXELS_MSG} cannot be recomputed"
              + (f"; {redo} ({'--force' if args.force else 'module changed'}): their cached rows are kept only if "
                 f"the recomputed panoramas reproduce the cache exactly" if redo else ""), flush=True)
    t0 = time.time()
    results = {}
    if need:
        jobs = [(pano_path(r["pano_id"], dataset), r.get("heading")) for r in need]
        with Pool(args.workers, initializer=_init, initargs=([m.NAME for m, _ in todo_mods],)) as pool:
            for i, res in enumerate(pool.imap(_extract_one, jobs, chunksize=8)):
                results[need[i]["pano_id"]] = res
                if (i + 1) % 500 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"  {i + 1}/{len(need)}  {rate:.1f} pano/s", flush=True)
    status = 0
    with cache_lock(feat_dir):
        all_recs = load_records(dataset, exclude_own=False)  # re-read: stream_duels.py may have appended meanwhile
        blind_ids = [r["pano_id"] for r in all_recs if not has_pixels(r, dataset)]
        n_train = sum(1 for r in all_recs if split_of(r) == "train")
        for j, (m, h) in enumerate(todo_mods):
            c = read_cache(m.NAME, feat_dir)
            have = cached.get(m.NAME, {})
            same = c is not None and c["names"] == list(m.FEATURE_NAMES)
            current = same and hash_ok(m.NAME, c["hash"], h)
            if c is not None and m.NAME not in cached:  # module changed, or --force: verify before keeping rows
                ok, n_cmp, n_bad, d = reproduces(results, j, c) if same else (False, 0, 0, float("inf"))
                why = "--force" if args.force else "module changed (%s -> %s)" % (c["hash"], h)
                if ok:
                    print(f"[{m.NAME}] {why}: the recomputed features equal the cached ones on all {n_cmp} panoramas "
                          f"compared; the cached rows of the others (incl. image-less panoramas) are kept")
                    have = dict(zip(c["ids"], c["X"]))
                else:
                    in_c = set(c["ids"])
                    lost = [p for p in blind_ids if p in in_c]
                    kept = keep_stale(m.NAME, c, lost, feat_dir)
                    if lost:
                        status = 3
                        lost_set = set(lost)
                        n_tr = sum(1 for r in all_recs if r["pano_id"] in lost_set and split_of(r) == "train")
                        _warn(f"[{m.NAME}] {why}: the recomputed features differ from the cache on {n_bad} of "
                              f"{n_cmp} panoramas compared (max |diff| {d:.3g}; {MIN_EQUIV_ROWS} needed, both "
                              f"car-axis regimes). The {len(lost)} {NO_PIXELS_MSG} lose their {m.NAME} features "
                              f"(old rows moved to {kept}): train_model now trains on {n_train - n_tr} instead of "
                              f"{n_train} training panoramas until `python3 tools/stream_duels.py --refresh` "
                              f"downloads them again and recomputes them (~{len(lost) / 3.5 / 3600:.1f} h).")
            elif current:
                have = dict(have)
                for p, x in zip(c["ids"], c["X"]):  # rows appended by others since this run started
                    have.setdefault(p, x)
            label = c["hash"] if current else h  # same hash, or verified equivalent: keep the stored label
            ids, X = [], []
            for r in all_recs:
                pid = r["pano_id"]
                x = results[pid][j] if pid in results else have.get(pid)
                if x is None:
                    continue
                ids.append(pid)
                X.append(x)
            write_cache(m.NAME, ids, np.array(X, np.float32).reshape(len(ids), len(m.FEATURE_NAMES)),
                        list(m.FEATURE_NAMES), label, feat_dir)
            nan_rate = float(np.isnan(np.array(X)).mean()) if X else 0.0
            miss = sum(1 for p in blind_ids if p not in have)
            print(f"[{m.NAME}] saved {len(ids)} vectors x {len(m.FEATURE_NAMES)} features (NaN rate {nan_rate:.2%})"
                  + (f"; {miss} image-less panoramas without features: run `python3 tools/stream_duels.py "
                     f"--refresh`" if miss else ""))
    print(f"done in {time.time() - t0:.0f}s")
    return status


def load_features(names, recs=None):
    """Matrix (n, d) for the given modules aligned with recs; missing rows -> NaN (a warning when image-less
    panoramas miss rows: they can only be restored by `tools/stream_duels.py --refresh`)."""
    recs = recs if recs is not None else load_records()
    cols, fnames = [], []
    for n in names:
        z = np.load(os.path.join(FEAT_DIR, n + ".npz"), allow_pickle=True)
        Xz, fn = z["X"], list(z["names"])
        idx = {pid: i for i, pid in enumerate(z["ids"])}
        M = np.full((len(recs), Xz.shape[1]), np.nan, np.float32)
        lost = 0
        for i, r in enumerate(recs):
            j = idx.get(r["pano_id"])
            if j is not None:
                M[i] = Xz[j]
            elif r.get("pano_deleted"):
                lost += 1
        if lost:
            sys.stderr.write(f"WARNING: {lost} {NO_PIXELS_MSG} have no {n} features (module changed?): run "
                             f"`python3 tools/stream_duels.py --refresh`\n")
        cols.append(M)
        fnames += [f"{n}.{f}" for f in fn]
    return np.concatenate(cols, axis=1), fnames, recs


# ------------------------------------------------------------- quick models
def robust_scale(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    q1, q3 = np.nanpercentile(Xtr, 25, axis=0), np.nanpercentile(Xtr, 75, axis=0)
    sc = np.where((q3 - q1) > 1e-9, (q3 - q1) / 1.349, np.nanstd(Xtr, axis=0) + 1e-9)
    f = lambda X: np.nan_to_num(np.clip((X - med) / sc, -6, 6))
    return f(Xtr), f(Xte)


def lda_scores(Xtr, ytr, Xte, classes, shrink=0.3):
    d = Xtr.shape[1]
    mu = np.stack([Xtr[ytr == c].mean(0) for c in classes])
    R = Xtr - mu[np.searchsorted(classes, ytr)]
    S = np.cov(R.T) if d > 1 else np.array([[R.var()]])
    S = (1 - shrink) * S + shrink * np.eye(d) * np.trace(S) / d
    Si = np.linalg.pinv(S)
    A = Xte @ Si @ mu.T - 0.5 * np.sum(mu @ Si * mu, axis=1)[None, :]
    return A


def evaluate_scores(logits, yte, classes, prior):
    lp = logits + np.log(prior)[None, :]
    lp -= lp.max(1, keepdims=True)
    P = np.exp(lp)
    P /= P.sum(1, keepdims=True)
    idx = {c: i for i, c in enumerate(classes)}
    ok = np.array([y in idx for y in yte])
    yi = np.array([idx.get(y, 0) for y in yte])
    rank = (P > P[np.arange(len(yi)), yi][:, None]).sum(1)
    top1 = float(((rank == 0) & ok).mean())
    top5 = float(((rank < 5) & ok).mean())
    p_true = np.where(ok, P[np.arange(len(yi)), yi], 1e-6)
    p_prior = np.where(ok, prior[yi], 1e-6)
    gain = float(np.mean(np.log2(np.maximum(p_true, 1e-9)) - np.log2(np.maximum(p_prior, 1e-9))))
    return top1, top5, gain


def anova_f(X, y):
    classes = np.unique(y)
    gm = X.mean(0)
    ssb = sum((y == c).sum() * (X[y == c].mean(0) - gm) ** 2 for c in classes)
    ssw = sum(((X[y == c] - X[y == c].mean(0)) ** 2).sum(0) for c in classes)
    dfb, dfw = len(classes) - 1, len(X) - len(classes)
    return (ssb / dfb) / (ssw / dfw + 1e-12)


def cmd_eval_group(args):
    names = args.module.split(",")
    X, fnames, recs = load_features(names)
    y = np.array([r["label"] for r in recs])
    tr = np.array([split_of(r) == "train" for r in recs]) & ~np.isnan(X).all(1)
    te = np.array([in_eval(r) for r in recs]) & ~np.isnan(X).all(1)
    classes = np.array(sorted(c for c in set(y[tr]) if (y[tr] == c).sum() >= args.min_count))
    keep_tr = tr & np.isin(y, classes)
    Xtr, Xte = robust_scale(X[keep_tr], X[te])
    ytr, yte = y[keep_tr], y[te]
    world = np.array([r["mode"] == "world" for r in recs])
    cnt = np.array([((y == c) & world).sum() for c in classes], float) + 1.0
    prior = cnt / cnt.sum()
    A = lda_scores(Xtr, ytr, Xte, classes, shrink=args.shrink)
    t1, t5, g = evaluate_scores(A, yte, classes, prior)
    t1p, t5p, _ = evaluate_scores(np.zeros_like(A), yte, classes, prior)
    print(f"modules={names} dims={X.shape[1]} train={keep_tr.sum()} test={te.sum()} classes={len(classes)}")
    print(f"  prior only : top1 {t1p:.3f}  top5 {t5p:.3f}")
    print(f"  LDA+prior  : top1 {t1:.3f}  top5 {t5:.3f}  info gain {g:+.3f} bits/round")
    F = anova_f(Xtr, ytr)
    order = np.argsort(-F)
    print("  top features by ANOVA F:")
    for i in order[: min(25, len(order))]:
        print(f"    {fnames[i]:40s} F={F[i]:8.2f}  nan={np.isnan(X[tr][:, i]).mean():.2f}")


def cmd_stats(args):
    from collections import Counter
    recs = load_records()
    print(len(recs), Counter(r["mode"] for r in recs))
    every = load_records(exclude_own=False)
    drop = near_own(every)
    print("image-less (streamed) records", sum(1 for r in recs if r.get("pano_deleted")),
          "; training records left out on / within %.0f km of the user's own rounds:" % OWN_KM,
          dict(Counter(r["mode"] for r in every if r["pano_id"] in drop)))
    tr = Counter(r["label"] for r in recs if split_of(r) == "train")
    te = Counter(r["label"] for r in recs if in_eval(r))
    print("splits", Counter(split_of(r) for r in recs))
    print("train countries", len(tr), tr.most_common(40))
    print("test countries", len(te), te.most_common(40))
    print("test countries missing from train:", sorted(set(te) - set(tr)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("features")
    p.add_argument("--modules", default="")
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--force", action="store_true", help="recompute every panorama with an image; the rows of "
                   "image-less panoramas are kept only if the recomputed ones reproduce the cache exactly")
    p.add_argument("--sample", type=int, default=0, help="only eval panoramas + N train panoramas")
    p = sub.add_parser("eval-group")
    p.add_argument("module")
    p.add_argument("--shrink", type=float, default=0.3)
    p.add_argument("--min-count", type=int, default=8)
    sub.add_parser("stats")
    args = ap.parse_args()
    sys.exit({"features": cmd_features, "eval-group": cmd_eval_group, "stats": cmd_stats}[args.cmd](args) or 0)


if __name__ == "__main__":
    main()
