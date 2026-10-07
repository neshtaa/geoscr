#!/usr/bin/env python3
"""
Calibration / evaluation harness.

  python3 tools/calibrate.py features [--modules road,solar] [--workers 14] [--force]
      compute and cache feature vectors for every panorama of the dataset
      (scratch/features/<module>.npz); recomputes when the module source changed.
  python3 tools/calibrate.py eval-group <module>
      standalone discriminative power of one feature group: country top-1/top-5
      accuracy and information gain (bits) on the real-game test set, trained on
      the world+balanced split, plus a per-feature ANOVA F ranking.
  python3 tools/calibrate.py stats
      dataset summary.

Splits: train = world + balanced panoramas, test = history (+ postmatch) panoramas
from the user's real GeoGuessr games.
"""
import argparse
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

# GeoGuessr country codes that Street View metadata reports differently
LABEL_FIX = {"UK": "GB"}


def label_of(r):
    for k in ("country_code", "gg_country", "raster_country"):
        v = r.get(k)
        if v:
            v = v.upper()
            return LABEL_FIX.get(v, v)
    return None


def load_records(dataset=DATASET):
    recs = {}
    for line in open(os.path.join(dataset, "index.jsonl")):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not os.path.exists(os.path.join(dataset, "panos", r["pano_id"] + ".jpg")):
            continue
        r["label"] = label_of(r)
        if r["label"]:
            recs.setdefault(r["pano_id"], r)
    out = list(recs.values())
    out.sort(key=lambda r: r["pano_id"])
    return out


NON_WORLD_MAPS = {"Ukraine", "United Kingdom (Better Map)", "MLB and MiLB Stadiums 2019"}


def split_of(r):
    """train: random Street View samples; calib / test: halves (by game) of the
    user's real GeoGuessr World-map rounds; other: rounds from single-country maps."""
    if r["mode"] in ("world", "balanced"):
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


def cmd_features(args):
    from engine import features
    names = args.modules.split(",") if args.modules else [m.NAME for m in features.available()]
    mods = [features.load(n) for n in names]
    recs = load_records()
    os.makedirs(FEAT_DIR, exist_ok=True)
    todo_mods, cached = [], {}
    for m in mods:
        path = os.path.join(FEAT_DIR, m.NAME + ".npz")
        h = module_hash(m)
        if os.path.exists(path) and not args.force:
            z = np.load(path, allow_pickle=True)
            if str(z["hash"]) == h and list(z["names"]) == list(m.FEATURE_NAMES):
                cached[m.NAME] = {pid: x for pid, x in zip(z["ids"], z["X"])}
        todo_mods.append((m, h))
    # only panoramas missing from each module's cache
    if args.sample:
        tr = [r for r in recs if split_of(r) == "train"]
        tr.sort(key=lambda r: hashlib.md5(r["pano_id"].encode()).hexdigest())
        recs = [r for r in recs if in_eval(r)] + tr[: args.sample]
    need = [r for r in recs if any(r["pano_id"] not in cached.get(m.NAME, {}) for m, _ in todo_mods)]
    print(f"{len(recs)} panoramas, {len(need)} to process for modules {names}", flush=True)
    t0 = time.time()
    results = {}
    if need:
        jobs = [(os.path.join(DATASET, "panos", r["pano_id"] + ".jpg"), r.get("heading")) for r in need]
        with Pool(args.workers, initializer=_init, initargs=([m.NAME for m, _ in todo_mods],)) as pool:
            for i, res in enumerate(pool.imap(_extract_one, jobs, chunksize=8)):
                results[need[i]["pano_id"]] = res
                if (i + 1) % 500 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"  {i + 1}/{len(need)}  {rate:.1f} pano/s", flush=True)
    for j, (m, h) in enumerate(todo_mods):
        ids, X = [], []
        for r in load_records():
            pid = r["pano_id"]
            x = results[pid][j] if pid in results else cached.get(m.NAME, {}).get(pid)
            if x is None:
                continue
            ids.append(pid)
            X.append(x)
        np.savez_compressed(os.path.join(FEAT_DIR, m.NAME + ".npz"), ids=np.array(ids), X=np.array(X, np.float32),
                            names=np.array(m.FEATURE_NAMES), hash=h)
        nan_rate = float(np.isnan(np.array(X)).mean()) if X else 0.0
        print(f"[{m.NAME}] saved {len(ids)} vectors x {len(m.FEATURE_NAMES)} features (NaN rate {nan_rate:.2%})")
    print(f"done in {time.time() - t0:.0f}s")


def load_features(names, recs=None):
    """Matrix (n, d) for the given modules aligned with recs; missing rows -> NaN."""
    recs = recs if recs is not None else load_records()
    cols, fnames = [], []
    for n in names:
        z = np.load(os.path.join(FEAT_DIR, n + ".npz"), allow_pickle=True)
        Xz, names = z["X"], list(z["names"])
        idx = {pid: i for i, pid in enumerate(z["ids"])}
        M = np.full((len(recs), Xz.shape[1]), np.nan, np.float32)
        for i, r in enumerate(recs):
            j = idx.get(r["pano_id"])
            if j is not None:
                M[i] = Xz[j]
        cols.append(M)
        fnames += [f"{n}.{f}" for f in names]
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
    p.add_argument("--force", action="store_true")
    p.add_argument("--sample", type=int, default=0, help="only eval panoramas + N train panoramas")
    p = sub.add_parser("eval-group")
    p.add_argument("module")
    p.add_argument("--shrink", type=float, default=0.3)
    p.add_argument("--min-count", type=int, default=8)
    sub.add_parser("stats")
    args = ap.parse_args()
    {"features": cmd_features, "eval-group": cmd_eval_group, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
