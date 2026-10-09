#!/usr/bin/env python3
"""
Moving games offline: simulate a player who walks along the road of a round and calibrate the fusion of
the captures (engine/fusion.py).

For every CALIB / TEST round of the user's history (the round's panorama is in the dataset):
  * walk: from the round's panorama the simulated player follows Street View's own navigation links
    (GetMetadata "links", the arrows a player clicks; time-machine entries of other capture dates are never
    links) as straight as the road allows, starting in the car's direction and turning back at a dead
    end.  Captures: the first step (one arrow click, ~10 m: the "short" move) and the panoramas nearest
    45 / 100 / 170 m from the start (straight line, within 50 m of the target, >= 25 m apart): the "walk".
    The capture dates are recorded (a walk can cross into another capture run, as a player's does);
  * the panoramas are downloaded with <= 6 threads (walk requests included) into a temp dir and DELETED
    after the round;
  * every panorama is rendered into the live grid exactly like tools/eval_live.py (2 x 5 views,
    112.7 x 90 deg, random start yaw, car axis unknown) and its locator inputs are kept: the feature
    vectors and the clue-card detector scores (scratch/moving/rounds/<pano_id>.npz).  These do not depend
    on the trained model, so `fit` / `report` recompute every capture's posterior (with and without the
    round's map), regions and guess under the CURRENT model exactly as engine.locator.Locator.analyze
    does; the files carry a features stamp (feature code + detector bank) and fit / report refuse rounds
    with another one;
  * for a third of the CALIB rounds the round's panorama is also rendered from another start yaw (a
    re-capture of the same panorama: the duplicate threshold);
  * for every 10th round the full Locator.analyze of the round's panorama is stored too (`check`
    compares the recomputation with it while the model is unchanged).

  python3 tools/moving_calib.py collect [--splits calib,test] [--per-split 300,0] [--workers 3] [--threads 6]
  python3 tools/moving_calib.py check     # recomputed posterior / guess == Locator.analyze (same model)
  python3 tools/moving_calib.py fit       # rho (w(n) = 1 / (1 + rho (n-1))), d2 mode, dup_ratio on CALIB
  python3 tools/moving_calib.py report    # TEST once: 1..4 panoramas fused vs the first one, without / with map
  python3 tools/moving_calib.py refresh-cards   # after a new clue-card detector bank: new card scores, same walks
  python3 tools/moving_calib.py collect --panos <id>,...   # given rounds only (e.g. the replay test round)
  python3 tools/moving_calib.py neighbours <pano_id>       # cached walk of a round (play_live_visual.js --moves)

fit and report write data/fusion.json ("params" read by engine/fusion.py, "calib", "test", each with the
model stamp it was computed under; report refuses to run when the model changed since fit).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # one core per worker process
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import collections
import glob
import hashlib
import json
import math
import random
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

OUT_DIR = os.path.join(ROOT, "scratch", "moving")
ROUNDS_DIR = os.environ.get("MOVING_ROUNDS_DIR") or os.path.join(OUT_DIR, "rounds")
REPLAY_DIR = os.path.join(OUT_DIR, "replay_rounds")
FUSION_JSON = os.environ.get("MOVING_FUSION_JSON") or os.path.join(ROOT, "data", "fusion.json")
MIN_M, MAX_M, SEP_M, TOL_M = 30.0, 200.0, 25.0, 50.0
TARGETS_M = (45.0, 100.0, 170.0)   # the walk: one panorama near each distance from the start
SHORT_MAX_M = 25.0                 # the first step of the walk is a "short" move when it is this close
SAME_SPOT_M = 3.0                  # a link this close is the same spot (another capture of it)
RENDER_VERSION = 1                 # LiveGrid / capture list format (part of the features stamp)
FORMAT = 2


# ----------------------------------------------------------------------------- stamps
def _file_hash(path):
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return "-"


def model_stamp():
    """The trained model and everything the locator loads besides the feature code (fit / report compare it)."""
    h = hashlib.sha1()
    for f in ("model.json", "model.npz", "priors.json", "regions.npz", "regions_params.json", "clue_index.npz"):
        h.update(("%s:%s|" % (f, _file_hash(os.path.join(ROOT, "data", "model", f)))).encode())
    return h.hexdigest()[:12]


CODE_FILES = [os.path.join("engine", "panorama.py"), os.path.join("engine", "clue_detect.py")]
BANK_FILE = os.path.join("data", "model", "clue_detectors.npz")


def _code_files():
    return sorted(glob.glob(os.path.join(ROOT, "engine", "features", "*.py"))) + [os.path.join(ROOT, f) for f in CODE_FILES]


def features_stamp():
    """What the stored per-capture inputs depend on: the feature modules, the sphere code, the clue-card
    detector bank and this file's rendering (RENDER_VERSION)."""
    h = hashlib.sha1(("render%d" % RENDER_VERSION).encode())
    for f in _code_files() + [os.path.join(ROOT, BANK_FILE)]:
        h.update(("%s:%s|" % (os.path.relpath(f, ROOT), _file_hash(f))).encode())
    return h.hexdigest()[:12]


def code_stamp():
    """The part of features_stamp() the feature vectors depend on (not the detector bank)."""
    h = hashlib.sha1(("render%d" % RENDER_VERSION).encode())
    for f in _code_files():
        h.update(("%s:%s|" % (os.path.relpath(f, ROOT), _file_hash(f))).encode())
    return h.hexdigest()[:12]


# ----------------------------------------------------------------------------- the walk
def _dist_m(lat1, lng1, lat2, lng2):
    from engine.geo import haversine_km
    return float(haversine_km(lat1, lng1, lat2, lng2)) * 1000.0


def _bearing(lat1, lng1, lat2, lng2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lng2 - lng1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def _angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def meta_block(pano_id):
    """The GetMetadata block of a panorama (engine.streetview), None if unavailable."""
    from engine import streetview as sv
    payload = [["apiv3", None, None, None, "US", None, None, None, None, None, [[0]]],
               ["en", "US"], [[[2, pano_id]]], [[1, 2, 3, 4, 8, 6]]]
    try:
        d = sv._rpc("GetMetadata", payload)
    except Exception:
        return None
    if not d or d[0] != [0] or len(d) < 2 or not d[1]:
        return None
    return d[1][0]


def walk_node(block):
    """{pano_id, lat, lng, date, meta (engine.streetview format, for the download), links: [(pano_id, lat, lng,
    heading of the arrow)]} of a GetMetadata block; links = the panorama's navigation arrows (never the
    time-machine entries of other dates)."""
    from engine.streetview import _parse_pano
    meta = _parse_pano(block) if block else None
    if not meta or meta.get("heading") is None:
        return None
    links = []
    try:
        s5 = block[5][0]
        items = s5[3][0] if len(s5) > 3 and s5[3] else []
        hist = {h[0] for h in (s5[8] if len(s5) > 8 and s5[8] else []) if isinstance(h, list) and h}
        for ln in (s5[6] if len(s5) > 6 and s5[6] else []):
            try:
                idx = ln[0]
                if idx in hist:
                    continue
                it = items[idx]
                links.append((it[0][1], float(it[2][0][2]), float(it[2][0][3]), float(ln[1][3])))
            except (IndexError, TypeError, ValueError):
                continue
    except (IndexError, TypeError):
        pass
    return {"pano_id": meta["pano_id"], "lat": float(meta["lat"]), "lng": float(meta["lng"]),
            "date": meta.get("date"), "meta": meta, "links": links}


def walk(start, heading, exclude=(), max_m=MAX_M + 10.0, max_steps=45, fetch=None):
    """The panoramas a player reaches clicking the arrow closest to the travel direction, from `start`
    towards `heading` (deg), until max_m from the start, a dead end or a turn of more than 100 deg.
    [node + {dist_m, steps}]."""
    fetch = fetch or (lambda pid: walk_node(meta_block(pid)))
    path, visited = [], {start["pano_id"]} | set(exclude)
    cur, travel = start, float(heading)
    for _ in range(max_steps):
        opts = [ln for ln in cur["links"] if ln[0] not in visited]
        if not opts:
            break
        ln = min(opts, key=lambda x: _angdiff(x[3], travel))
        if _angdiff(ln[3], travel) > 100.0:
            break
        visited.add(ln[0])
        nxt = fetch(ln[0])
        if nxt is None:
            break
        visited.add(nxt["pano_id"])
        step = _dist_m(cur["lat"], cur["lng"], nxt["lat"], nxt["lng"])
        travel = _bearing(cur["lat"], cur["lng"], nxt["lat"], nxt["lng"]) if step > 1.0 else ln[3]
        nxt = dict(nxt, dist_m=round(_dist_m(start["lat"], start["lng"], nxt["lat"], nxt["lng"]), 1), steps=len(path) + 1)
        path.append(nxt)
        cur = nxt
        if nxt["dist_m"] >= max_m:
            break
    return path


def choose_neighbours(r, pools, k=3, targets=TARGETS_M):
    """Up to k nodes 30-200 m from the round (its own panorama excluded, >= 25 m apart), one near each target
    distance (within TOL_M), taken from the first pool that has one (the forward walk before the walk back);
    sorted by distance."""
    out = []
    for t in targets[:k]:
        for pool in pools:
            best = None
            for n in pool:
                d = n["dist_m"]
                if n["pano_id"] == r["pano_id"] or not (MIN_M <= d <= MAX_M) or abs(d - t) > TOL_M:
                    continue
                if any(o["pano_id"] == n["pano_id"] or _dist_m(n["lat"], n["lng"], o["lat"], o["lng"]) < SEP_M for o in out):
                    continue
                if best is None or abs(d - t) < abs(best["dist_m"] - t):
                    best = n
            if best is not None:
                out.append(best)
                break
    return sorted(out, key=lambda n: n["dist_m"])


def find_neighbours(r, k=3, fetch=None):
    """(walk captures, short capture or None, info) of a round."""
    fetch = fetch or (lambda pid: walk_node(meta_block(pid)))
    start = fetch(r["pano_id"])
    if start is None or start["pano_id"] != r["pano_id"]:
        return [], None, {"src": "no metadata"}
    h = float(r.get("heading") if r.get("heading") is not None else start["meta"]["heading"])
    fwd = walk(start, h, fetch=fetch)
    pools, back = [fwd], []
    nb = choose_neighbours(r, pools, k)
    if len(nb) < k:   # dead end: the player turns round and walks the other way
        back = walk(start, h + 180.0, exclude={n["pano_id"] for n in fwd}, fetch=fetch)
        pools.append(back)
        nb = choose_neighbours(r, pools, k)
    chosen = {n["pano_id"] for n in nb}
    short = next((n for n in fwd + back if SAME_SPOT_M <= n["dist_m"] <= SHORT_MAX_M and n["pano_id"] not in chosen), None)
    info = {"src": "walk" if not back else "walk+back", "fwd_n": len(fwd), "back_n": len(back),
            "fwd_m": max([n["dist_m"] for n in fwd] or [0.0]), "back_m": max([n["dist_m"] for n in back] or [0.0]),
            "start_date": start.get("date")}
    return nb, short, info


def fetch_round(r, k, tmp, pool, fetch=None):
    """The round's walk and its captures downloaded into tmp (<= 6 Street View threads in total)."""
    from engine.streetview import download_panorama
    t0 = time.time()
    try:
        nb, short, info = find_neighbours(r, k, fetch=fetch)
    except Exception as e:
        return {"round": r, "neigh": [], "info": {"src": "error " + repr(e)[:80]}, "s": time.time() - t0}
    caps = ([dict(short, role="short")] if short else []) + [dict(n, role="walk") for n in nb]

    def get(n):
        try:
            meta = n["meta"]
            path = os.path.join(tmp, n["pano_id"].replace("/", "_") + ".jpg")
            download_panorama(meta).save(path, quality=92)
            return {"pano_id": n["pano_id"], "lat": n["lat"], "lng": n["lng"], "dist_m": n["dist_m"], "steps": n["steps"],
                    "heading": meta["heading"], "date": meta.get("date"), "path": path, "role": n["role"]}
        except Exception:
            return None

    got = [x for x in pool.map(get, caps) if x is not None]
    return {"round": r, "neigh": got, "info": info, "s": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- live-grid rendering
class LiveGrid:
    """tools/eval_live.render_views followed by SphericalImage.from_views (what Locator.analyze_views does)
    with the geometry computed once: turning the camera by yaw only shifts the azimuth, so the render
    maps are those of yaw 0 plus a column offset, and the back-projection maps of the 2 x 5 views are
    those of start yaw 0 rolled by whole sphere columns (the start yaw is drawn on the 2048-column grid).
    Same formulas as engine/panorama.py; the result equals the slow path up to float rounding."""

    def __init__(self, hfov=112.715, size=(1112, 740), pitches=(-40.0, 40.0), n_yaw=5, width=2048):
        from engine.panorama import _camera_basis, direction
        self.hfov, self.size, self.pitches, self.n_yaw, self.W = hfov, size, pitches, n_yaw, width
        iw, ih = size
        t = math.tan(math.radians(hfov) / 2)
        tv = t * ih / iw
        W, H = width, width // 2
        self.render = {}
        for pitch in pitches:  # render_view(yaw_rel=0, pitch): azimuth / elevation of every view pixel
            f, r, u = _camera_basis(0.0, pitch)
            xs = ((np.arange(iw) + 0.5) / iw * 2 - 1) * t
            ys = (1 - (np.arange(ih) + 0.5) / ih * 2) * tv
            d = f[None, None, :] + xs[None, :, None] * r[None, None, :] + ys[:, None, None] * u[None, None, :]
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            az = np.degrees(np.arctan2(d[..., 0], d[..., 1]))
            el = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
            py = (90.0 - el) / 180.0 * H - 0.5
            y0 = np.floor(py).astype(np.int64)
            # bilinear() rows of the view pixels (the panorama is 2048 x 1024 like the dataset's)
            self.render[pitch] = ((az + 180.0) / 360.0 * W - 0.5, py, np.clip(y0, 0, H - 1) * W,
                                  np.clip(y0 + 1, 0, H - 1) * W, (py - y0)[..., None])
        phi = (np.arange(W) + 0.5) / W * 360.0 - 180.0
        elg = 90.0 - (np.arange(H) + 0.5) / H * 180.0
        dd = direction(phi[None, :], elg[:, None])
        self.back = []
        for pitch in pitches:  # from_views, start yaw 0: the sphere pixels each view covers
            for k in range(n_yaw):
                f, r, u = _camera_basis(360.0 * k / n_yaw, pitch)
                zc = dd @ f
                ok = zc > 1e-3
                xc = np.where(ok, (dd @ r) / np.maximum(zc, 1e-3), 9.0)
                yc = np.where(ok, (dd @ u) / np.maximum(zc, 1e-3), 9.0)
                px = (xc / t + 1) * iw / 2 - 0.5
                py = (1 - yc / tv) * ih / 2 - 0.5
                inside = ok & (px >= 0) & (px <= iw - 1) & (py >= 0) & (py <= ih - 1)
                yy, xx = np.nonzero(inside)
                self.back.append((pitch, k, yy.astype(np.int32), xx.astype(np.int32), zc[yy, xx], px[yy, xx], py[yy, xx]))

    def start_yaw(self, seed):
        """eval_live's random start yaw, on the sphere's column grid."""
        u = random.Random(seed).uniform(0, 360)
        return (int(round(u * self.W / 360.0)) % self.W) * 360.0 / self.W

    def sphere(self, sph, start):
        """The live capture of sph (a full panorama with its heading) from start yaw `start`."""
        from engine.panorama import SphericalImage, bilinear
        W, H = self.W, self.W // 2
        if sph.rgb.shape[:2] != (H, W):
            raise ValueError("the panorama must be %dx%d" % (W, H))
        flat = sph.rgb.reshape(-1, 3)
        views = {}
        for pitch in self.pitches:
            px0, py0, r0, r1, fy = self.render[pitch]
            for k in range(self.n_yaw):
                yaw = (start + 360.0 * k / self.n_yaw) % 360.0
                rel = (yaw - sph.heading + 180.0) % 360.0 - 180.0
                # engine.panorama.bilinear(sph.rgb, px, py, wrap_x=True) with flat gathers (same arithmetic)
                xs = px0 + rel / 360.0 * W
                x0 = np.floor(xs).astype(np.int64)
                fx = (xs - x0)[..., None]
                x0m, x1m = x0 % W, (x0 + 1) % W
                a = flat[r0 + x0m].astype(np.float32)
                b = flat[r0 + x1m].astype(np.float32)
                c = flat[r1 + x0m].astype(np.float32)
                d = flat[r1 + x1m].astype(np.float32)
                out = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
                views[(pitch, k)] = np.clip(out, 0, 255).astype(np.uint8)
        s = int(round(start * W / 360.0)) % W
        rgb = np.zeros((H, W, 3), np.float32)
        best = np.full((H, W), -2.0, np.float32)
        for pitch, k, yy, xx, zc, px, py in self.back:
            xs = (xx + s) % W
            b = zc > best[yy, xs]
            if not b.any():
                continue
            rgb[yy[b], xs[b]] = bilinear(views[(pitch, k)], px[b], py[b])
            best[yy[b], xs[b]] = zc[b]
        return SphericalImage(np.clip(rgb, 0, 255).astype(np.uint8), heading=0.0, mask=best > -2.0,
                              car_heading=None, source="views")


# ----------------------------------------------------------------------------- analysis (worker)
_LOC = None
_LOC_STAMP = None
_GRID = None


def _grid():
    global _GRID
    if _GRID is None:
        _GRID = LiveGrid()
    return _GRID


def _loc():
    """The worker's locator and the stamp of the model files it loaded."""
    global _LOC, _LOC_STAMP
    if _LOC is None:
        from engine.locator import Locator
        while True:
            stamp = model_stamp()
            loc = Locator()
            if model_stamp() == stamp:
                break
        _LOC, _LOC_STAMP = loc, stamp
    return _LOC


def locator_inputs(loc, sph):
    """What engine.locator.Locator.analyze computes from the sphere before the model: the feature vectors
    ({group: (1, d)}, NaN for a failing module), the clue-card scores zt (None without a detector bank) and
    their coverage, and the input kind."""
    from engine.locator import input_kind
    X = {}
    for name, mod in loc.modules.items():
        try:
            x = np.asarray(mod.extract(sph)["x"], np.float64)
        except Exception:
            x = np.full(len(mod.FEATURE_NAMES), np.nan)
        X[name] = x[None, :]
    zt, cov = None, float("nan")
    if loc.cards is not None:
        try:
            det = loc.cards.detect(sph)
            zt, cov = np.asarray(loc.cards.zt(det["z"]), np.float32), float(det["coverage"])
        except Exception:
            zt = None
    return X, zt, cov, input_kind(sph)


def process_round(job):
    """Locator inputs of the round's panorama, its short move and its walk; writes the compact npz; deletes
    the downloaded jpgs."""
    from calibrate import DATASET
    from engine.panorama import SphericalImage
    r, neigh, dup, check = job["round"], job["neigh"], job.get("dup", False), job.get("check", False)
    loc = _loc()
    caps = [{"pano_id": r["pano_id"], "lat": r["lat"], "lng": r["lng"], "dist_m": 0.0, "steps": 0, "heading": r["heading"],
             "path": os.path.join(DATASET, "panos", r["pano_id"] + ".jpg"), "date": r.get("date"), "role": "own"}]
    caps += neigh
    rows, chk = [], None
    try:
        for i, c in enumerate(caps):
            t0 = time.time()
            sph = SphericalImage.from_equirect(c["path"], heading=c["heading"])
            yaws = [(_grid().start_yaw(c["pano_id"]), c["role"])]
            if i == 0 and dup:
                yaws.append((_grid().start_yaw(c["pano_id"] + "/dup"), "dup"))
            for yaw, role in yaws:
                rs = _grid().sphere(sph, yaw)
                X, zt, cov, kind = locator_inputs(loc, rs)
                if i == 0 and role == "own" and check:   # the full analysis, for `check`
                    res = loc.analyze(rs)
                    chk = {"post": res["posterior"], "guess": [res["guess"]["lat"], res["guess"]["lng"]],
                           "model": _LOC_STAMP, "kind": res["input"]["kind"]}
                rows.append({"cap": c, "role": role, "X": X, "zt": zt, "cov": cov, "kind": kind, "yaw": yaw,
                             "s": time.time() - t0})
    finally:
        for c in neigh:
            try:
                os.remove(c["path"])
            except OSError:
                pass
    groups = list(rows[0]["X"].keys())
    nz = max([len(row["zt"]) for row in rows if row["zt"] is not None] or [0])
    zt = np.full((len(rows), nz), np.nan, np.float32)
    for i, row in enumerate(rows):
        if row["zt"] is not None and nz:
            zt[i] = row["zt"]
    meta = {"round": {k: r.get(k) for k in ("pano_id", "lat", "lng", "label", "map", "game", "round", "heading")},
            "walk": job.get("info"), "features": features_stamp(), "feat_code": code_stamp(),
            "cards_bank": _file_hash(os.path.join(ROOT, BANK_FILE))[:12], "format": FORMAT,
            "dates": [row["cap"].get("date") for row in rows], "check": chk}
    out = {
        "groups": np.array(groups),
        "dims": np.array([rows[0]["X"][g].shape[1] for g in groups]),
        "X": np.stack([np.concatenate([row["X"][g][0] for g in groups]) for row in rows]).astype(np.float32),
        "zt": zt, "cov": np.array([row["cov"] for row in rows], np.float64),
        "kind": np.array([row["kind"] for row in rows]), "role": np.array([row["role"] for row in rows]),
        "pos": np.array([[row["cap"]["lat"], row["cap"]["lng"], row["cap"]["dist_m"]] for row in rows]),
        "steps": np.array([row["cap"].get("steps", 0) for row in rows]),
        "pano_ids": np.array([row["cap"]["pano_id"] for row in rows]),
        "yaw": np.array([row["yaw"] for row in rows]),
        "seconds": np.array([row["s"] for row in rows]),
        "meta": json.dumps(meta),
    }
    os.makedirs(ROUNDS_DIR, exist_ok=True)
    tmp = os.path.join(ROUNDS_DIR, ".%s.npz" % r["pano_id"])
    np.savez_compressed(tmp, **out)
    os.replace(tmp, os.path.join(ROUNDS_DIR, r["pano_id"] + ".npz"))
    return r["pano_id"], len(rows), float(np.sum(out["seconds"]))


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _hash_mod(s, k):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % k


def cmd_collect(args):
    from calibrate import load_records, split_of
    splits = args.splits.split(",")
    recs = [r for r in load_records(pixels=True) if split_of(r) in splits and r.get("heading") is not None]
    if args.panos:
        want = set(args.panos.split(","))
        recs = [r for r in recs if r["pano_id"] in want]
    recs.sort(key=lambda r: (splits.index(split_of(r)), hashlib.md5(r["pano_id"].encode()).hexdigest()))
    if args.per_split:  # the first N rounds of each split (random order by hash)
        lim = [int(x) or 10 ** 9 for x in args.per_split.split(",")]
        if len(lim) == len(splits):
            seen, keep = collections.Counter(), []
            for r in recs:
                s = split_of(r)
                if seen[s] < lim[splits.index(s)]:
                    keep.append(r)
                seen[s] += 1
            recs = keep
    done = set(f[:-4] for f in os.listdir(ROUNDS_DIR) if f.endswith(".npz")) if os.path.isdir(ROUNDS_DIR) else set()
    todo = [r for r in recs if r["pano_id"] not in done]
    if args.n:
        todo = todo[:args.n]
    print("%d rounds (%s), %d done, %d to do; features %s" % (len(recs), args.splits, len(done), len(todo), features_stamp()),
          flush=True)
    os.makedirs(os.path.join(ROOT, "scratch"), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="moving_", dir=os.path.join(ROOT, "scratch"))
    n_dl = max(1, min(6, args.threads) - args.walkers)
    dl = ThreadPoolExecutor(max_workers=n_dl)
    fetchers = ThreadPoolExecutor(max_workers=args.walkers)
    pool = Pool(args.workers, initializer=_init_worker, maxtasksperchild=40)
    sem = threading.BoundedSemaphore(args.inflight)
    t0, n_done, lock = time.time(), [0], threading.Lock()
    pending = []

    def finished(res):
        sem.release()
        with lock:
            n_done[0] += 1
            if n_done[0] % 10 == 0 or n_done[0] == len(todo):
                el = time.time() - t0
                print("[%d/%d] %.0f s elapsed, %.1f s/round; last %s (%d captures, %.0f s cpu)" %
                      (n_done[0], len(todo), el, el / n_done[0], res[0], res[1], res[2]), flush=True)

    def failed(e):
        sem.release()
        print("[!] round failed: %r" % (e,), flush=True)

    try:
        futs = []
        for r in todo:
            sem.acquire()
            job_flags = {"dup": split_of(r) == "calib" and _hash_mod(r["pano_id"], 3) == 0,
                         "check": _hash_mod(r["pano_id"] + "/check", args.check_every) == 0 if args.check_every else False}
            f = fetchers.submit(fetch_round, r, args.max_neigh, tmp, dl)

            def go(f, flags=job_flags):
                try:
                    job = f.result()
                except Exception as e:
                    failed(e)
                    return
                job.update(flags)
                pending.append(pool.apply_async(process_round, (job,), callback=finished, error_callback=failed))
            f.add_done_callback(go)
            futs.append(f)
        for f in futs:
            f.result()
        for _ in range(args.inflight):  # every round releases its slot when analysed or failed
            sem.acquire()
    finally:
        pool.close()
        pool.join()
        dl.shutdown()
        fetchers.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    print("done in %.0f s" % (time.time() - t0))


# ----------------------------------------------------------------------------- evaluation
def _load_round(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def round_files(split):
    """(npz paths of the stored rounds of a split, {features stamp: count})."""
    from calibrate import split_of
    out, stamps = [], collections.Counter()
    if not os.path.isdir(ROUNDS_DIR):
        return out, stamps
    for f in sorted(os.listdir(ROUNDS_DIR)):
        if not f.endswith(".npz") or f.startswith("."):
            continue
        path = os.path.join(ROUNDS_DIR, f)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        if split_of(dict(meta["round"], mode="history")) == split:
            out.append(path)
            stamps[meta.get("features") if meta.get("format") == FORMAT else "old format"] += 1
    return out, stamps


def require_stamps(stamps, what):
    """Exit unless all stored rounds were collected with the current feature code and detector bank."""
    cur = features_stamp()
    bad = {k: v for k, v in stamps.items() if k != cur}
    if bad:
        sys.exit("%s: %d stored rounds have other locator inputs than the current code (%s, current %s): "
                 "collect them again (delete scratch/moving/rounds/*.npz first)" % (what, sum(bad.values()), bad, cur))


def capture_states(d, model, cards, map_name):
    """engine.fusion capture states of every stored capture, {variant: [state]}, variant 0 without and 1 with
    the round's map, recomputed under the current model as engine.locator.Locator.analyze does: kind ->
    parameter set, map setup, evidence of the features + card scores, prior x evidence."""
    from engine.fusion import embedding, model_for
    from engine.geo import resolve_map
    groups, dims = [str(g) for g in d["groups"]], d["dims"]
    off = np.concatenate([[0], np.cumsum(dims)])
    mps = {0: None, 1: resolve_map(map_name) if map_name else None}
    out = {0: [], 1: []}
    for i in range(len(d["role"])):
        kind = str(d["kind"][i])
        m = model_for(model, kind)
        X = {g: d["X"][i, off[j]:off[j + 1]].astype(np.float64)[None, :] for j, g in enumerate(groups)}
        cl = None
        if cards is not None and d["zt"].shape[1] and np.isfinite(d["zt"][i]).all():
            cl = cards.country_evidence({"coverage": float(d["cov"][i]), "zt": d["zt"][i].astype(np.float64)}, m.classes)
        ev = m.evidence(X, cards_ll=cl)
        emb = embedding(m, X)
        classes = list(m.classes)
        for v in (0, 1):
            setup = m.map_setup(mps[v])
            lp = m.combine(ev, setup["weights"], prior=setup["prior"])[0]
            out[v].append(dict(emb, classes=classes, post=np.exp(lp), prior=np.asarray(setup["prior"], np.float64), X=X,
                               kind=kind, role=str(d["role"][i]), dist_m=float(d["pos"][i, 2]), steps=int(d["steps"][i]),
                               date=d["meta"]["dates"][i], i=i))
    return out


def sequences(caps):
    """{"walk": own + walk captures by distance, "short": own + the first step} of one variant's states."""
    own = [c for c in caps if c["role"] == "own"]
    walk_ = sorted([c for c in caps if c["role"] == "walk"], key=lambda c: c["dist_m"])
    short = [c for c in caps if c["role"] == "short"]
    return {"walk": own + walk_, "short": own + short if short else own}


def _same_month(a, b):
    return a is not None and b is not None and list(a)[:2] == list(b)[:2]


def _score(guess, r):
    from engine.geo import geoguessr_points, geoguessr_score, haversine_km, resolve_map
    d = float(haversine_km(r["lat"], r["lng"], guess[0], guess[1]))
    rm = resolve_map(r.get("map")) or {}
    pts_map = geoguessr_points(d, rm["maxErrorDistance"]) if rm.get("maxErrorDistance") else float(geoguessr_score(d))
    return d, float(geoguessr_score(d)), float(pts_map)


_EV = None


def _ev_init():
    global _EV
    from engine.locator import Locator
    from engine.regions import load_region_model
    loc = Locator()
    _EV = (loc.model, load_region_model(), loc.cards)


def _ev_round(job):
    """Rows of one stored round for every config {name, seq, rho, d2, variant, ns, need, points}: {name: {n: row}};
    row = {rank of the true country, ll = log P(true), km, points (World formula), points_map}; plus the walk's
    dates / distances and the duplicate ratios."""
    from engine.fusion import dup_ratio, fuse_core, fuse_country
    path, configs, want_dup = job
    model, rmodel, cards = _EV
    d = _load_round(path)
    classes = list(model.classes)
    r = d["meta"]["round"]
    yi = classes.index(r["label"]) if r["label"] in classes else None
    st = capture_states(d, model, cards, r.get("map"))
    seqs = {v: sequences([c for c in st[v] if c["role"] != "dup"]) for v in (0, 1)}
    wk = seqs[0]["walk"]
    out = {"round": r["pano_id"], "n_walk": len(wk), "has_short": len(seqs[0]["short"]) > 1, "rows": {},
           "walk_same_month": all(_same_month(c["date"], wk[0]["date"]) for c in wk[1:]) if len(wk) > 1 else None,
           "short_same_month": _same_month(seqs[0]["short"][-1]["date"], wk[0]["date"]) if len(seqs[0]["short"]) > 1 else None,
           "walk_dist": [c["dist_m"] for c in wk[1:]], "walk_steps": [c["steps"] for c in wk[1:]],
           "short_dist": seqs[0]["short"][-1]["dist_m"] if len(seqs[0]["short"]) > 1 else None,
           "walk_src": (d["meta"].get("walk") or {}).get("src")}
    for cfg in configs:
        cs = seqs[cfg["variant"]][cfg.get("seq", "walk")]
        if len(cs) < cfg["need"]:
            continue
        mp = r.get("map") if cfg["variant"] == 1 else None
        rows = {}
        for n in cfg["ns"]:
            if n > len(cs):
                continue
            sub = cs[:n]
            if cfg["points"]:
                core = fuse_core(model, sub, mp, cfg["rho"], cfg["d2"], rmodel)
                post, guess = core["post"], (core["guess"]["lat"], core["guess"]["lng"])
            else:
                post, guess = fuse_country(sub, cfg["rho"], classes)[1], None
            row = {"rank": int((post > post[yi]).sum()) if yi is not None else 999,
                   "ll": float(np.log(max(post[yi], 1e-12))) if yi is not None else math.log(1e-12)}
            if guess is not None:
                row["km"], row["points"], row["points_map"] = _score(guess, r)
            rows[n] = row
        out["rows"][cfg["name"]] = rows
    if want_dup and any(c["role"] == "dup" for c in st[0]):
        own = next(c for c in st[0] if c["role"] == "own")
        dupc = next(c for c in st[0] if c["role"] == "dup")
        out["dup"] = (dup_ratio(own, dupc), [dup_ratio(own, c) for c in st[0] if c["role"] in ("walk", "short")])
    return out


def run_eval(paths, configs, workers, want_dup=False):
    """{config name: {n: [rows over the rounds that have the config's `need` captures]}}, per-round results."""
    jobs = [(p, configs, want_dup) for p in paths]
    if workers > 1:
        with Pool(workers, initializer=_ev_init) as pool:
            res = pool.map(_ev_round, jobs, chunksize=4)
    else:
        _ev_init()
        res = [_ev_round(j) for j in jobs]
    agg = {c["name"]: {} for c in configs}
    for x in res:
        for name, rows in x["rows"].items():
            for n, row in rows.items():
                agg[name].setdefault(n, []).append(dict(row, _round=x["round"]))
    return agg, res


def summary(rows):
    rk = np.array([x["rank"] for x in rows])
    o = {"n_rounds": len(rows), "top1": float(np.mean(rk == 0)), "top3": float(np.mean(rk < 3)),
         "top5": float(np.mean(rk < 5)), "mean_log_p_true": float(np.mean([x["ll"] for x in rows]))}
    if rows and "points" in rows[0]:
        o.update(points=float(np.mean([x["points"] for x in rows])),
                 points_map_formula=float(np.mean([x["points_map"] for x in rows])),
                 median_km=float(np.median([x["km"] for x in rows])))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}


def _paired(a, b, key, n_boot=2000, seed=0):
    """Mean of b - a over paired rows with a 95% bootstrap interval; rows of the same round (several fused answers
    of one walk) are resampled together."""
    x = np.array([float(rb[key]) - float(ra[key]) for ra, rb in zip(a, b)], np.float64)
    if not len(x):
        return None
    rounds = [ra.get("_round", i) for i, ra in enumerate(a)]
    ids = {r: k for k, r in enumerate(dict.fromkeys(rounds))}
    g = np.array([ids[r] for r in rounds])
    S, N = np.bincount(g, weights=x), np.bincount(g).astype(np.float64)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(S), len(S))
        bs.append(S[idx].sum() / N[idx].sum())
    return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def _ind(rows):
    return [{"t": float(x["rank"] == 0), "t3": float(x["rank"] < 3), "ll": x["ll"], "points": x.get("points", 0.0),
             "_round": x.get("_round")} for x in rows]


def _paired_all(a, b):
    return {"points": _paired(a, b, "points"), "log_p_true": _paired(a, b, "ll"),
            "top1": _paired(_ind(a), _ind(b), "t"), "top3": _paired(_ind(a), _ind(b), "t3")}


def _load_json():
    try:
        return json.load(open(FUSION_JSON, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_json(js):
    tmp = FUSION_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(js, f, indent=1)
    os.replace(tmp, FUSION_JSON)


RHO_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RHO_POINTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def cmd_fit(args):
    paths, stamps = round_files("calib")
    require_stamps(stamps, "fit")
    stamp = model_stamp()
    print("CALIB rounds with captures: %d; model %s, features %s" % (len(paths), stamp, features_stamp()), flush=True)
    # 1. rho by the mean log P(true country) of the fused moves (country only), both variants: the walk
    # (n = 2..4) and the short move (n = 2) pooled, every fused answer one row
    cfgs = [{"name": "ll%d_%s_%g" % (v, s, rho), "seq": s, "rho": rho, "d2": "mean", "variant": v,
             "ns": (2, 3, 4) if s == "walk" else (2,), "need": 2, "points": False}
            for v in (0, 1) for s in ("walk", "short") for rho in RHO_GRID]
    agg, res = run_eval(paths, cfgs, args.workers, want_dup=True)
    fits = {}
    for v, name in ((0, "no_map"), (1, "map")):
        curves = {}
        for s in ("walk", "short", "pooled"):
            sc = []
            for rho in RHO_GRID:
                rows = [x for s2 in (("walk", "short") if s == "pooled" else (s,))
                        for n in (2, 3, 4) for x in agg["ll%d_%s_%g" % (v, s2, rho)].get(n, [])]
                sc.append(float(np.mean([x["ll"] for x in rows])) if rows else float("nan"))
            curves[s] = {"mean_log_p_true": [round(x, 4) for x in sc], "best": RHO_GRID[int(np.nanargmax(sc))],
                         "rows": len([x for s2 in (("walk", "short") if s == "pooled" else (s,)) for n in (2, 3, 4)
                                      for x in agg["ll%d_%s_%g" % (v, s2, RHO_GRID[0])].get(n, [])])}
            print("  %s %s: mean log P(true) by rho %s -> best %.2f" % (
                name, s, " ".join("%.2f:%.4f" % t for t in zip(RHO_GRID, sc)), curves[s]["best"]), flush=True)
        fits[name] = dict(curves, grid=RHO_GRID)
    # rho: the best mean log P(true) of all fused moves, averaged over the two variants
    both = np.mean([fits[k]["pooled"]["mean_log_p_true"] for k in ("no_map", "map")], axis=0)
    rho_ll = RHO_GRID[int(np.nanargmax(both))]
    # 2. points (fused regions + guess) on the walks with 3 neighbours, both variants: a coarse rho grid; the d2 mode
    rhos = sorted(set(RHO_POINTS + [rho_ll]))
    cfgs = [{"name": "p%d_%g" % (v, rho), "seq": "walk", "rho": rho, "d2": "mean", "variant": v, "ns": (1, 2, 3, 4), "need": 4,
             "points": True} for v in (0, 1) for rho in rhos]
    cfgs.append({"name": "p0_last", "seq": "walk", "rho": rho_ll, "d2": "last", "variant": 0, "ns": (1, 2, 3, 4), "need": 4,
                 "points": True})
    agg2, _ = run_eval(paths, cfgs, args.workers)
    pts = {}
    for c in cfgs:
        pts[c["name"]] = {str(n): summary(rows) for n, rows in sorted(agg2[c["name"]].items())}
        print("  %s: %s" % (c["name"], " | ".join("n%s %s" % (n, json.dumps({k: o[k] for k in ("top1", "top3", "points")}))
                                                  for n, o in pts[c["name"]].items())), flush=True)

    def pooled(rho, variants=(0, 1)):   # every fused answer (n = 2..4) of both variants (rows carry their round)
        return [x for v in variants for n in (2, 3, 4) for x in agg2["p%d_%g" % (v, rho)].get(n, [])]
    best_pts = max(rhos, key=lambda rho: np.mean([x["points"] for x in pooled(rho)]))
    choice = {"by_ll": rho_ll, "by_points": best_pts,
              "mean_points_n2to4_both_variants": {"%g" % r: round(float(np.mean([x["points"] for x in pooled(r)])), 1) for r in rhos}}
    rho = rho_ll
    if best_pts != rho_ll:
        diff = _paired(pooled(rho_ll), pooled(best_pts), "points")
        choice["points_gain_vs_ll_choice"] = diff
        if diff and diff[1] > 0:   # points clearly prefer another rho (rounds resampled as a whole)
            rho = best_pts
    d_last = _paired(pooled(rho_ll, (0,)), [x for n in (2, 3, 4) for x in agg2["p0_last"].get(n, [])], "points")
    choice["d2_last_minus_mean_points"] = d_last
    d2 = "last" if d_last and d_last[1] > 0 else "mean"
    # 3. re-captures of the same panorama vs the walk / short move (duplicate threshold); raw renders here, the
    # live path adds JPEG noise, so the threshold sits between the two distributions
    dups = [x["dup"] for x in res if x.get("dup")]
    same = np.array([p[0] for p in dups])
    other = np.array([y for p in dups for y in p[1]])
    dup = {"same_pano_n": int(len(same)), "neighbours_n": int(len(other))}
    thr = None
    if len(same) and len(other):
        hi, lo = float(np.percentile(same, 99)), float(np.percentile(other, 1))
        thr = float(math.sqrt(max(hi, 1e-6) * max(lo, 1e-6))) if hi < lo else None
        dup.update(same_median=round(float(np.median(same)), 5), same_p99=round(hi, 5), same_max=round(float(same.max()), 5),
                   neighbours_p1=round(lo, 5), neighbours_median=round(float(np.median(other)), 5),
                   neighbours_min=round(float(other.min()), 5), threshold=thr,
                   neighbours_below_threshold=int((other < thr).sum()) if thr else None)
        print("  dup ratio: same panorama median %.4f p99 %.4f | moves min %.4f p1 %.4f median %.4f -> %s"
              % (np.median(same), hi, other.min(), lo, np.median(other), thr), flush=True)
    params = {"rho": rho, "d2": d2, "max_captures": 8}
    if thr:
        params["dup_ratio"] = round(thr, 5)
    js = _load_json()
    js["params"] = params
    js["calib"] = {"rounds": len(paths), "walk_sizes": dict(collections.Counter(str(x["n_walk"]) for x in res)),
                   "with_short_move": int(sum(1 for x in res if x["has_short"])),
                   "rho_ll": fits, "points": pts, "choice": choice, "dup": dup, "model": stamp,
                   "features": features_stamp(), "fitted_at": time.strftime("%Y-%m-%d %H:%M")}
    js.pop("test", None)   # a TEST report belongs to the fit it was made with
    _save_json(js)
    print("params:", params, flush=True)


def cmd_report(args):
    js = _load_json()
    params = js.get("params")
    if not params or not js.get("calib"):
        sys.exit("run fit first")
    stamp = model_stamp()
    if js["calib"].get("model") != stamp:
        sys.exit("the model changed since fit (fit %s, now %s): run fit again before the report"
                 % (js["calib"].get("model"), stamp))
    paths, stamps = round_files(args.split)
    require_stamps(stamps, "report")
    if js["calib"].get("features") != features_stamp():
        sys.exit("the locator inputs changed since fit: collect and fit again")
    print("%s rounds with captures: %d; params %s; model %s" % (args.split.upper(), len(paths), params, stamp), flush=True)
    rho, d2 = params["rho"], params.get("d2", "mean")
    cfgs = []
    for v in (0, 1):
        cfgs.append({"name": "walk%d" % v, "seq": "walk", "rho": rho, "d2": d2, "variant": v, "ns": (1, 2, 3, 4), "need": 4,
                     "points": True})
        cfgs.append({"name": "product%d" % v, "seq": "walk", "rho": 0.0, "d2": d2, "variant": v, "ns": (2, 3, 4), "need": 4,
                     "points": True})
        cfgs.append({"name": "any%d" % v, "seq": "walk", "rho": rho, "d2": d2, "variant": v, "ns": (1, 2), "need": 2,
                     "points": True})
        cfgs.append({"name": "short%d" % v, "seq": "short", "rho": rho, "d2": d2, "variant": v, "ns": (1, 2), "need": 2,
                     "points": True})
    agg, res = run_eval(paths, cfgs, args.workers)
    by_round = {x["round"]: x for x in res}
    wk = [x for x in res if x["n_walk"] == 4]
    rep = {"rounds": len(paths), "params": params, "model": stamp, "features": features_stamp(),
           "reported_at": time.strftime("%Y-%m-%d %H:%M"),
           "walk_sizes": dict(collections.Counter(str(x["n_walk"]) for x in res)),
           "with_short_move": int(sum(1 for x in res if x["has_short"])),
           "walk": {"same_month_all_three": int(sum(1 for x in wk if x["walk_same_month"])), "rounds": len(wk),
                    "median_dist_m": [round(float(np.median([x["walk_dist"][i] for x in wk])), 1) for i in range(3)] if wk else None,
                    "median_steps": [float(np.median([x["walk_steps"][i] for x in wk])) for i in range(3)] if wk else None,
                    "src": dict(collections.Counter(x["walk_src"] for x in res)),
                    "short_median_dist_m": round(float(np.median([x["short_dist"] for x in res if x["short_dist"] is not None])), 1)
                    if any(x["short_dist"] is not None for x in res) else None,
                    "short_same_month": int(sum(1 for x in res if x["short_same_month"]))}}

    def subset(rows, same):
        return [x for x in rows if bool(by_round[x["_round"]]["walk_same_month"]) == same]
    for v, name in ((0, "no_map"), (1, "map")):
        a = agg["walk%d" % v]
        rep[name] = {str(n): summary(rows) for n, rows in sorted(a.items())}
        rep[name + "_naive_product"] = {str(n): summary(rows) for n, rows in sorted(agg["product%d" % v].items())}
        rep[name + "_rounds_with_2plus"] = {str(n): summary(rows) for n, rows in sorted(agg["any%d" % v].items())}
        rep[name + "_short_move"] = {str(n): summary(rows) for n, rows in sorted(agg["short%d" % v].items())}
        if 1 in a and 4 in a:
            rep[name + "_paired_4_vs_1"] = _paired_all(a[1], a[4])
            for same, tag in ((True, "same_month"), (False, "other_dates")):
                s1, s4 = subset(a[1], same), subset(a[4], same)
                rep[name + "_" + tag] = {"rounds": len(s1), "n1": summary(s1) if s1 else None, "n4": summary(s4) if s4 else None,
                                         "paired_4_vs_1": _paired_all(s1, s4) if s1 else None}
        sh = agg["short%d" % v]
        if 1 in sh and 2 in sh:
            rep[name + "_short_paired_2_vs_1"] = _paired_all(sh[1], sh[2])
        print(name, json.dumps(rep[name]), json.dumps(rep.get(name + "_paired_4_vs_1")), flush=True)
    js[args.split if args.split == "test" else args.split + "_report"] = rep
    _save_json(js)


def cmd_check(args):
    """Recomputed posterior / guess of the round's own capture (fuse_core, one capture) vs the full
    Locator.analyze: the one stored at collect (rounds collected under the current model), and with --now N
    a fresh Locator.analyze of N rounds' own panoramas (local) under the current model."""
    from engine.fusion import fuse_core
    from calibrate import DATASET
    from engine.panorama import SphericalImage
    _ev_init()
    model, rmodel, cards = _EV
    stamp = model_stamp()
    paths = [os.path.join(ROUNDS_DIR, f) for f in sorted(os.listdir(ROUNDS_DIR)) if f.endswith(".npz") and not f.startswith(".")]
    n, worst_p, worst_km, other = 0, 0.0, 0.0, 0
    fresh = 0
    for p in paths:
        d = _load_round(p)
        chk = d["meta"].get("check")
        if fresh < args.now and d["meta"].get("features") == features_stamp():
            r = d["meta"]["round"]
            i0 = [str(x) for x in d["role"]].index("own")
            sph = SphericalImage.from_equirect(os.path.join(DATASET, "panos", r["pano_id"] + ".jpg"), heading=r["heading"])
            res = _loc().analyze(_grid().sphere(sph, float(d["yaw"][i0])))
            chk = {"post": res["posterior"], "guess": [res["guess"]["lat"], res["guess"]["lng"]], "model": _LOC_STAMP}
            fresh += 1
        if not chk:
            continue
        if chk["model"] != stamp:
            other += 1
            continue
        st = capture_states(d, model, cards, None)[0]
        own = [c for c in st if c["role"] == "own"]
        core = fuse_core(model, own, None, 0.5, "mean", rmodel)
        ref = np.array([chk["post"].get(c, 0.0) for c in core["classes"]])
        dp = float(np.abs(core["post"] - ref).max())
        from engine.geo import haversine_km
        dk = float(haversine_km(core["guess"]["lat"], core["guess"]["lng"], chk["guess"][0], chk["guess"][1]))
        worst_p, worst_km, n = max(worst_p, dp), max(worst_km, dk), n + 1
    print(json.dumps({"checked": n, "fresh": fresh, "other_model": other, "max_abs_posterior_diff": round(worst_p, 6),
                      "max_guess_km": round(worst_km, 4)}))
    if n and (worst_p > 2e-3 or worst_km > 1.0):
        sys.exit("the recomputation differs from Locator.analyze")


def _refresh_round(job):
    """New clue-card scores (zt, coverage) of a stored round under the current detector bank: the captures are
    downloaded again by pano id (the round's own panorama is local), rendered from their stored start yaw and
    scanned; the feature vectors stay (their code is unchanged).  check: also recompute the features of one
    downloaded capture and compare (the download and rendering reproduce the stored capture)."""
    from calibrate import DATASET
    from engine.panorama import SphericalImage
    from engine.streetview import download_panorama, get_metadata
    path, tmp, check = job
    d = _load_round(path)
    loc = _loc()
    r = d["meta"]["round"]
    files, heads = {}, {}
    ids = [str(x) for x in d["pano_ids"]]

    def fetch(pid):
        if pid == r["pano_id"]:
            return pid, os.path.join(DATASET, "panos", pid + ".jpg"), r["heading"]
        meta = get_metadata(pid)
        if not meta or meta.get("heading") is None:
            raise RuntimeError("panorama %s is gone" % pid)
        f = os.path.join(tmp, pid.replace("/", "_") + ".jpg")
        download_panorama(meta).save(f, quality=92)
        return pid, f, meta["heading"]
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            for pid, f, h in ex.map(fetch, list(dict.fromkeys(ids))):
                files[pid], heads[pid] = f, h
        zt, cov, diff = d["zt"].copy(), d["cov"].copy(), None
        sph_cache = {}
        for i, pid in enumerate(ids):
            if pid not in sph_cache:
                sph_cache[pid] = SphericalImage.from_equirect(files[pid], heading=heads[pid])
            rs = _grid().sphere(sph_cache[pid], float(d["yaw"][i]))
            det = loc.cards.detect(rs)
            z = np.asarray(loc.cards.zt(det["z"]), np.float32)
            if zt.shape[1] != len(z):
                zt = np.full((len(ids), len(z)), np.nan, np.float32)
            zt[i], cov[i] = z, float(det["coverage"])
            if check and diff is None and str(d["role"][i]) == "walk":
                X, _, _, _ = locator_inputs(loc, rs)
                groups, dims = [str(g) for g in d["groups"]], d["dims"]
                x = np.concatenate([X[g][0] for g in groups]).astype(np.float32)
                ok = np.isfinite(x) & np.isfinite(d["X"][i])
                diff = float(np.abs(x[ok] - d["X"][i][ok]).max()) if ok.any() else 0.0
    finally:
        for pid, f in files.items():
            if pid != r["pano_id"]:
                try:
                    os.remove(f)
                except OSError:
                    pass
    d["meta"].update(features=features_stamp(), feat_code=code_stamp(), cards_bank=_file_hash(os.path.join(ROOT, BANK_FILE))[:12],
                     cards_refreshed=time.strftime("%Y-%m-%d %H:%M"))
    out = {k: v for k, v in d.items() if k != "meta"}
    out.update(zt=zt, cov=cov, meta=json.dumps(d["meta"]))
    t = os.path.join(ROUNDS_DIR, ".%s.npz" % r["pano_id"])
    np.savez_compressed(t, **out)
    os.replace(t, path)
    return r["pano_id"], diff


def cmd_refresh_cards(args):
    """After a new clue-card detector bank (data/model/clue_detectors.npz): recompute the stored rounds' card scores
    (zt) without collecting the walks again.  Only for rounds whose feature code is the current one (stored
    "feat_code", or for older files: no feature code file newer than the round file)."""
    cur, code = features_stamp(), code_stamp()
    newest_code = max(os.path.getmtime(f) for f in _code_files())
    todo, skip = [], []
    for f in sorted(os.listdir(ROUNDS_DIR)):
        if not f.endswith(".npz") or f.startswith("."):
            continue
        path = os.path.join(ROUNDS_DIR, f)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        if meta.get("features") == cur:
            continue
        same_code = meta.get("feat_code") == code if meta.get("feat_code") else os.path.getmtime(path) > newest_code
        (todo if same_code else skip).append(path)
    print("%d rounds to refresh (detector bank %s), %d need a full collect (feature code changed)" %
          (len(todo), _file_hash(os.path.join(ROOT, BANK_FILE))[:12], len(skip)), flush=True)
    if not todo:
        return
    tmp = tempfile.mkdtemp(prefix="moving_cards_", dir=os.path.join(ROOT, "scratch"))
    t0, diffs = time.time(), []
    try:
        jobs = [(p, tmp, _hash_mod(os.path.basename(p), 10) == 0) for p in todo]
        with Pool(args.workers, initializer=_init_worker, maxtasksperchild=60) as pool:
            for k, (pid, diff) in enumerate(pool.imap_unordered(_refresh_round, jobs), 1):
                if diff is not None:
                    diffs.append(diff)
                if k % 25 == 0 or k == len(jobs):
                    print("[%d/%d] %.0f s; feature check max |dX| %s" % (k, len(jobs), time.time() - t0,
                                                                         "%.2g" % max(diffs) if diffs else "-"), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if features_stamp() != cur:
        print("[!] the detector bank or feature code changed during the refresh: run refresh-cards again", flush=True)


def cmd_neighbours(args):
    """The cached walk of a finished round (play_live_visual.js --replay --moves): [{pano_id, dist_m, steps, role}]
    in walking order (the short move, then the walk)."""
    f = os.path.join(ROUNDS_DIR, args.pano + ".npz")
    if not os.path.exists(f):   # rounds collected only for the replay test (outside the CALIB / TEST sample)
        f = os.path.join(REPLAY_DIR, args.pano + ".npz")
    if not os.path.exists(f):
        sys.exit("no cached moving round for %s: MOVING_ROUNDS_DIR=scratch/moving/replay_rounds python3 tools/moving_calib.py "
                 "collect --panos=%s" % (args.pano, args.pano))
    with np.load(f) as z:
        ids, pos, role, steps = z["pano_ids"], z["pos"], z["role"], z["steps"]
    rows = [{"pano_id": str(i), "dist_m": float(p[2]), "steps": int(s), "role": str(r)}
            for i, p, r, s in zip(ids, pos, role, steps) if str(r) in ("short", "walk")]
    print(json.dumps(sorted(rows, key=lambda x: x["dist_m"])))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("collect")
    c.add_argument("--splits", default="calib,test")
    c.add_argument("--workers", type=int, default=3)
    c.add_argument("--threads", type=int, default=6, help="Street View threads in total (walk + downloads), <= 6")
    c.add_argument("--walkers", type=int, default=2, help="rounds walked at once (their requests count in --threads)")
    c.add_argument("--max-neigh", type=int, default=3)
    c.add_argument("--inflight", type=int, default=8)
    c.add_argument("--check-every", type=int, default=10)
    c.add_argument("--n", type=int, default=0)
    c.add_argument("--per-split", default="", help="e.g. 300,0: at most 300 calib rounds, all test rounds (0 = all)")
    c.add_argument("--panos", default="", help="only these rounds (pano ids, comma separated)")
    ck = sub.add_parser("check")
    ck.add_argument("--now", type=int, default=0, help="also analyse N rounds' own panoramas now")
    rc = sub.add_parser("refresh-cards")
    rc.add_argument("--workers", type=int, default=3)
    nb = sub.add_parser("neighbours")
    nb.add_argument("pano")
    ft = sub.add_parser("fit")
    ft.add_argument("--workers", type=int, default=3)
    rp = sub.add_parser("report")
    rp.add_argument("--split", default="test")
    rp.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    cmds = {"collect": cmd_collect, "check": cmd_check, "fit": cmd_fit, "report": cmd_report, "neighbours": cmd_neighbours,
            "refresh-cards": cmd_refresh_cards}
    if args.cmd in cmds:
        cmds[args.cmd](args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
