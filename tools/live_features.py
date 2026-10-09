#!/usr/bin/env python3
"""
Feature cache of the user's CALIB / TEST rounds as the locator sees them in live play.

Kinds of capture (--kind):
  grid   every history panorama rendered into the live grid exactly as tools/eval_live.py renders it (2 x 5
         views of 112.7 x 90 deg on the 1112x740 canvas, start yaw random.Random(pano_id).uniform(0, 360),
         true north known) - the rotating rounds;
  frame  the single frame of a no-rotate (NMPZ) round (play_live_visual.js captureSingle): the start view
         only (same seeded yaw, pitch 0, same FOV and canvas), ~14% of the sphere;
rebuilt with SphericalImage.from_views (width 2048, heading 0, car axis unknown, source "views", as
Locator.analyze_views) and measured by the six feature modules and (grid) the GeoGuessr card detectors
(a single frame shows < 90% of the searched windows, so the locator uses no card evidence there).

  python3 tools/live_features.py [--kind grid|frame] [--splits calib,test] [--workers 4] [--force] [--no-cards]

-> scratch/features_live/<module>.npz        ids, X, names, hash (the layout of scratch/features,
                                             tools/calibrate.py), render (render_hash: the rendering code)
   scratch/features_live/cards.npz           ids, Z (raw detector maxima, engine.clue_detect.ClueDetectors.detect),
                                             coverage (share of the searched windows visible), det_hash, render
   scratch/features_live/frame/<module>.npz  the same for --kind frame
Incremental: a module is recomputed when its source hash changes, the cards when the detector bank changes,
everything when the rendering changes (engine/panorama.py, eval_live.render_views, live_sphere / frame_views).
tools/train_model.py --views calibrates the "views" parameter set of the locator on the grid features
(--views --live-kind frame reports it on the single frames).
"""
import argparse
import hashlib
import inspect
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from calibrate import DATASET, load_records, module_hash, split_of  # noqa: E402

LIVE_DIR = os.path.join(ROOT, "scratch", "features_live")
KINDS = ("grid", "frame")
SAVE_EVERY = 100

_MODS, _BANK, _KIND = None, None, "grid"


def cache_dir(kind="grid"):
    return LIVE_DIR if kind == "grid" else os.path.join(LIVE_DIR, kind)


def cards_file(kind="grid"):
    return os.path.join(cache_dir(kind), "cards.npz")


def frame_views(sph, start_yaw):
    """The single frame of a no-rotate round: the start view at pitch 0 with the grid's FOV and canvas
    (the defaults of eval_live.render_views)."""
    from eval_live import render_views
    p = inspect.signature(render_views).parameters
    hfov, size = p["hfov"].default, p["size"].default
    rel = (start_yaw - sph.heading + 180.0) % 360.0 - 180.0
    return [{"image": sph.render_view(rel, 0.0, hfov, size), "yaw": start_yaw, "pitch": 0.0, "hfov": hfov}]


def live_sphere(path, heading, pano_id, kind="grid"):
    """The SphericalImage the live script would rebuild from its captures of this panorama
    (tools/eval_live.py: same grid, same seeded start yaw; kind "frame": its first view direction only)."""
    from eval_live import render_views
    from engine.panorama import SphericalImage
    sph = SphericalImage.from_equirect(path, heading=heading)
    yaw = random.Random(pano_id).uniform(0, 360)
    views = frame_views(sph, yaw) if kind == "frame" else render_views(sph, yaw)
    return SphericalImage.from_views(views, width=2048, heading=0.0)


def render_hash(kind="grid"):
    """Hash of the code the cached captures of kind come from: engine/panorama.py (equirect loading,
    render_view, from_views), eval_live.render_views (grid, FOV, canvas), live_sphere (+ frame_views)."""
    import engine.panorama as pano
    from eval_live import render_views
    h = hashlib.sha1(kind.encode())
    with open(pano.__file__, "rb") as f:
        h.update(f.read())
    for fn in (render_views, live_sphere) + ((frame_views,) if kind == "frame" else ()):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()[:12]


def _init(names, cards, kind):
    global _MODS, _BANK, _KIND
    _KIND = kind
    from engine import features
    _MODS = [features.load(n) for n in names]
    if cards:
        from engine.clue_detect import ClueDetectors
        _BANK = ClueDetectors.load()


def _one(job):
    path, heading, pid = job
    live = live_sphere(path, heading, pid, _KIND)
    xs = []
    for m in _MODS:
        try:
            x = np.asarray(m.extract(live)["x"], np.float32)
        except Exception as e:  # keep going; NaN row marks failure (as tools/calibrate.py)
            sys.stderr.write("[%s] %s: %r\n" % (m.NAME, pid, e))
            x = np.full(len(m.FEATURE_NAMES), np.nan, np.float32)
        xs.append(x)
    card = None
    if _BANK is not None:
        try:
            det = _BANK.detect(live)
            card = (np.asarray(det["z"], np.float32), float(det["coverage"]))
        except Exception as e:
            sys.stderr.write("[cards] %s: %r\n" % (pid, e))
    return pid, xs, card


def _save(path, **arrays):
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def bank_hash(bank):
    return str((bank.meta or {}).get("det_hash")) if bank is not None else None


def eval_records(splits=("calib", "test")):
    """The user's rounds of the splits whose panorama image is on disk and whose true heading is known."""
    return [r for r in load_records() if split_of(r) in splits and r.get("heading") is not None
            and os.path.exists(os.path.join(DATASET, "panos", r["pano_id"] + ".jpg"))]


def cmd_build(args):
    from engine import features
    kind = args.kind
    mods = [features.load(n) for n in (args.modules.split(",") if args.modules else features.MODULES)]
    recs = eval_records(tuple(args.splits.split(",")))
    out_dir, cf, rh = cache_dir(kind), cards_file(kind), render_hash(kind)
    os.makedirs(out_dir, exist_ok=True)
    cached, hashes = {}, {}
    for m in mods:
        hashes[m.NAME] = module_hash(m)
        f = os.path.join(out_dir, m.NAME + ".npz")
        cached[m.NAME] = {}
        if os.path.exists(f) and not args.force:
            z = np.load(f, allow_pickle=True)
            if (str(z["hash"]) == hashes[m.NAME] and list(z["names"]) == list(m.FEATURE_NAMES)
                    and _render_of(z) == rh):
                cached[m.NAME] = {str(p): x for p, x in zip(z["ids"], z["X"])}
    bank, cards, dh = None, None, None
    if not args.no_cards and kind == "grid":  # a single frame never reaches the card evidence's coverage
        from engine.clue_detect import ClueDetectors
        bank = ClueDetectors.load()
        dh = bank_hash(bank)
        cards = {}
        if bank is not None and os.path.exists(cf) and not args.force:
            z = np.load(cf, allow_pickle=True)
            if str(z["det_hash"]) == dh and _render_of(z) == rh:
                cards = {str(p): (zz, float(c)) for p, zz, c in zip(z["ids"], z["Z"], z["coverage"])}
        if bank is None:
            print("no detector bank (data/model/clue_detectors.npz) - cards skipped")
            cards = None
    ids = [r["pano_id"] for r in recs]
    todo_mods = [m for m in mods if any(p not in cached[m.NAME] for p in ids)]
    need_cards = cards is not None and any(p not in cards for p in ids)
    need = [r for r in recs if any(r["pano_id"] not in cached[m.NAME] for m in todo_mods)
            or (need_cards and r["pano_id"] not in cards)]
    print("%s: %d rounds (%s), %d to render; modules %s%s" % (
        kind, len(recs), args.splits, len(need), [m.NAME for m in todo_mods] or "-", " + cards" if need_cards else ""),
        flush=True)

    def save():
        for m in mods:
            got = sorted(cached[m.NAME])
            if not got:
                continue
            _save(os.path.join(out_dir, m.NAME + ".npz"), ids=np.array(got),
                  X=np.array([cached[m.NAME][p] for p in got], np.float32), names=np.array(m.FEATURE_NAMES),
                  hash=hashes[m.NAME], render=rh)
        if cards:
            got = sorted(cards)
            _save(cf, ids=np.array(got), Z=np.array([cards[p][0] for p in got], np.float32),
                  coverage=np.array([cards[p][1] for p in got], np.float32), det_hash=dh, render=rh)

    if need:
        jobs = [(os.path.join(DATASET, "panos", r["pano_id"] + ".jpg"), r.get("heading"), r["pano_id"]) for r in need]
        names = [m.NAME for m in mods]  # all modules: rendering dominates, the extra modules are cheap
        t0 = time.time()
        with Pool(min(args.workers, 6), initializer=_init, initargs=(names, need_cards, kind)) as pool:
            for k, (pid, xs, card) in enumerate(pool.imap_unordered(_one, jobs, chunksize=2)):
                for m, x in zip(mods, xs):
                    cached[m.NAME][pid] = x
                if card is not None and cards is not None:
                    cards[pid] = card
                if (k + 1) % 25 == 0:
                    print("  %d/%d  %.2f pano/s" % (k + 1, len(jobs), (k + 1) / (time.time() - t0)), flush=True)
                if (k + 1) % SAVE_EVERY == 0:
                    save()
        print("rendered and measured %d panoramas in %.0fs" % (len(need), time.time() - t0))
    save()
    for m in mods:
        X = np.array([cached[m.NAME][p] for p in ids if p in cached[m.NAME]])
        print("[%s] %d vectors x %d features (NaN rate %.2f%%)" % (m.NAME, len(X), X.shape[1] if X.ndim == 2 else 0,
                                                                   100.0 * float(np.isnan(X).mean()) if X.size else 0))
    if cards:
        cov = np.array([cards[p][1] for p in ids if p in cards])
        print("[cards] %d detections, coverage median %.3f, >= 0.9: %.1f%%" % (len(cov), np.median(cov),
                                                                              100.0 * float((cov >= 0.9).mean())))


# ----------------------------------------------------------------------------- loading
def _render_of(z):
    return str(z["render"]) if "render" in z.files else None


def _check(f, name, kind):
    """Why the cache file of feature group name is not usable (None when it is)."""
    from engine import features
    if not os.path.exists(f):
        return "%s missing" % f
    z = np.load(f, allow_pickle=True)
    if str(z["hash"]) != module_hash(features.load(name)):
        return "%s was computed with another version of the %s module" % (f, name)
    if _render_of(z) != render_hash(kind):
        return "%s was rendered by other code (engine/panorama.py / eval_live.render_views / live_sphere)" % f
    return None


def live_matrix(name, recs, kind="grid"):
    """(n, d) live features of group name aligned with recs (NaN rows where missing) and the (n,) mask of
    the rows found; raises when the cache is missing or computed with another module or rendering version."""
    f = os.path.join(cache_dir(kind), name + ".npz")
    why = _check(f, name, kind)
    if why:
        raise (FileNotFoundError if why.endswith("missing") else ValueError)(
            "%s - run tools/live_features.py --kind %s" % (why, kind))
    z = np.load(f, allow_pickle=True)
    idx = {str(p): i for i, p in enumerate(z["ids"])}
    M = np.full((len(recs), z["X"].shape[1]), np.nan, np.float64)
    have = np.zeros(len(recs), bool)
    for i, r in enumerate(recs):
        j = idx.get(r["pano_id"])
        if j is not None:
            M[i] = z["X"][j]
            have[i] = True
    return M, have


def cards_status(bank, kind="grid"):
    """None when the live card detections of kind exist for this detector bank and rendering, else why."""
    f = cards_file(kind)
    if bank is None:
        return "no detector bank"
    if not os.path.exists(f):
        return "%s missing" % f
    z = np.load(f, allow_pickle=True)
    if str(z["det_hash"]) != bank_hash(bank):
        return "%s is from other detectors" % f
    if _render_of(z) != render_hash(kind):
        return "%s was rendered by other code" % f
    return None


def live_status(groups, kind="grid", bank=None):
    """None when the live cache of every group (and, with a detector bank, the card detections) is present
    and current, else the reason."""
    for g in groups:
        why = _check(os.path.join(cache_dir(kind), g + ".npz"), g, kind)
        if why:
            return why
    return cards_status(bank, kind) if bank is not None else None


def live_cards(pids, classes, bank=None, kind="grid"):
    """(n, C) card log-likelihoods of the live detections as Locator.analyze computes them (zeros, i.e. no
    evidence, where less than MIN_EVIDENCE_COVERAGE of the windows is visible or the panorama is missing);
    None when the bank or the cache is missing or the cache comes from other detectors / rendering."""
    from engine.clue_detect import MIN_EVIDENCE_COVERAGE, ClueDetectors
    bank = bank or ClueDetectors.load()
    why = cards_status(bank, kind)
    if why:
        if bank is not None:
            sys.stderr.write("%s - live card evidence left out\n" % why)
        return None
    z = np.load(cards_file(kind), allow_pickle=True)
    row = {str(p): i for i, p in enumerate(z["ids"])}
    out = np.zeros((len(pids), len(classes)))
    have = [k for k, p in enumerate(pids) if p in row and z["coverage"][row[p]] >= MIN_EVIDENCE_COVERAGE]
    if have:
        out[have] = bank.country_loglik(bank.zt(z["Z"][[row[pids[k]] for k in have]].astype(np.float64)), classes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="grid", choices=KINDS, help="grid: rotating rounds, frame: NMPZ single frame")
    ap.add_argument("--splits", default="calib,test")
    ap.add_argument("--modules", default="", help="default: all feature modules")
    ap.add_argument("--workers", type=int, default=4, help="processes (<= 6)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-cards", action="store_true", help="skip the card detections (always skipped for frames)")
    cmd_build(ap.parse_args())


if __name__ == "__main__":
    main()
