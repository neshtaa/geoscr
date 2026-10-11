#!/usr/bin/env python3
"""
Flag-detector spike (engine/flag_detect.py): does a closed-form colour-stripe detector find the national
flags GeoGuessr marks with its "flags" cards, and is it country evidence?

  python3 tools/flag_spike.py placements [--neg 900] [--workers 3]
      template score maps of the panoramas with a "flags" placement (data/calibration/pano_clues.json) and of
      annotated panoramas without one; per placement the best score inside the placement view
      (tan(hfov / 2) = 2^(1 - zoom), 16:9) vs. the same view size at a random heading of a panorama without
      flag cards -> scratch/flags/placements.npz, detection rate at fixed false-alarm rates per template
  python3 tools/flag_spike.py features --set train|calib|test|live_calib|live_test [--per-country 120]
      template maxima (FEATURE_NAMES) of training panoramas with pixels (at most --per-country per country,
      seeded), of the CALIB / TEST panoramas, or of their live-grid renderings (tools/live_features.py
      live_sphere) -> scratch/flags/features_<set>.npz (incremental)
  python3 tools/flag_spike.py evidence [--test]
      class-conditional Gaussian densities of the template maxima (engine.model.GroupGaussian, fitted on the
      training features) -> country log-likelihoods; weight of the "flags" source on CALIB live (2-fold CV by
      game, on top of the saved locator posterior), TEST only with --test
"""
import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from calibrate import load_records, pano_path, split_of  # noqa: E402
from engine import flag_detect as F  # noqa: E402
from engine.panorama import SphericalImage  # noqa: E402

OUT = os.path.join(ROOT, "scratch", "flags")
CLUES = os.path.join(ROOT, "data", "calibration", "pano_clues.json")
ASPECT = 9.0 / 16.0


def view_fov(zoom):
    """(hfov, vfov) of a GeoGuessr view at zoom (16:9; tan(hfov / 2) = 2^(1 - zoom))."""
    t = 2.0 ** (1.0 - float(zoom))
    return 2 * math.degrees(math.atan(t)), 2 * math.degrees(math.atan(t * ASPECT))


def in_view(maps, geo, az, el, zoom, margin=0.5):
    """(T,) best score per template among the cells inside the view centred on (az, el) (relative azimuth)."""
    hf, vf = view_fov(zoom)
    T, Hc, Wc = maps.shape
    ys = np.arange(Hc)
    xs = np.arange(Wc)
    caz, _ = F.cell_direction(geo, np.zeros(Wc, int), xs)
    _, cel = F.cell_direction(geo, ys, np.zeros(Hc, int))
    dx = np.abs((caz - az + 180.0) % 360.0 - 180.0) <= hf / 2 + margin
    dy = np.abs(cel - el) <= vf / 2 + margin
    if not dx.any() or not dy.any():
        return np.full(T, -np.inf)
    sub = maps[:, dy][:, :, dx]
    return sub.reshape(T, -1).max(1)


def _placement_job(job):
    pid, heading, views = job
    try:
        sph = SphericalImage.from_equirect(pano_path(pid), heading=heading)
        maps, geo, G = F.score_maps(sph)
    except Exception as e:  # noqa: BLE001
        return pid, None, repr(e)
    out = []
    for v in views:
        az = (float(v["heading"]) - float(heading or 0.0) + 180.0) % 360.0 - 180.0
        out.append(in_view(maps, geo, az, float(v.get("pitch", 0.0)), float(v["zoom"])))
    return pid, (np.array(out), F.features_from_maps(maps, G)), None


def cmd_placements(args):
    os.makedirs(OUT, exist_ok=True)
    clues = json.load(open(CLUES))
    recs = {r["pano_id"]: r for r in load_records(pixels=True, exclude_own=False)}
    flag_views, neg_panos = {}, []
    for pid, cards in clues.items():
        if pid not in recs or recs[pid].get("heading") is None:
            continue
        fl = [c for c in (cards or []) if c.get("type") == "flags" and c.get("zoom") is not None]
        if fl:
            flag_views[pid] = fl
        elif cards:
            neg_panos.append(pid)
    allf = [c for v in flag_views.values() for c in v]
    rs = random.Random(0)
    neg_panos.sort()
    rs.shuffle(neg_panos)
    jobs = [(pid, recs[pid]["heading"], v) for pid, v in sorted(flag_views.items())]
    negs = []
    for pid in neg_panos[:args.neg]:
        c = rs.choice(allf)
        views = [{"heading": rs.uniform(0, 360), "pitch": c.get("pitch", 0.0), "zoom": c["zoom"]}]
        jobs.append((pid, recs[pid]["heading"], views))
        negs.append(pid)
    print("placements: %d flag views on %d panoramas, %d negative panoramas" % (len(allf), len(flag_views), len(negs)),
          flush=True)
    rows, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for k, (pid, res, err) in enumerate(pool.imap_unordered(_placement_job, jobs, chunksize=4)):
            if res is None:
                print("  %s: %s" % (pid, err))
                continue
            sc, feat = res
            views = flag_views.get(pid)
            if views is not None:
                for v, s in zip(views, sc):
                    rows.append((pid, v["countryCode"], 1, float(v["zoom"]), float(v.get("pitch", 0.0)), s, feat,
                                 recs[pid]["label"]))
            else:
                rows.append((pid, recs[pid]["label"], 0, 0.0, 0.0, sc[0], feat, recs[pid]["label"]))
            if (k + 1) % 100 == 0:
                print("  %d/%d (%.0fs)" % (k + 1, len(jobs), time.time() - t0), flush=True)
    np.savez_compressed(os.path.join(OUT, "placements.npz"), pid=np.array([r[0] for r in rows]),
                        cc=np.array([r[1] for r in rows]), flag=np.array([r[2] for r in rows]),
                        zoom=np.array([r[3] for r in rows]), pitch=np.array([r[4] for r in rows]),
                        S=np.array([r[5] for r in rows]), X=np.array([r[6] for r in rows]),
                        label=np.array([r[7] for r in rows]), templates=np.array(F.TEMPLATE_NAMES))
    report_placements()


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks of ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


# which flags a template is meant for (for the detection table; the evidence model learns its own mapping)
INTENDED = {"KH": ["KH"], "KH_banner": ["KH"], "TH": ["TH"], "TH_banner": ["TH"],
            "RED_YELLOW_RED": ["VN", "ES"], "RED_YELLOW_RED_x": ["VN", "ES"], "WHITE_BLUE_RED": ["RU", "SI", "SK", "RS"],
            "RED_WHITE_BLUE": ["NL", "LU", "HR", "PY"], "BLUE_WHITE_RED_x": ["FR"], "RED_WHITE": ["ID", "MC"],
            "WHITE_RED": ["PL"], "RED_WHITE_RED_x": ["CA", "PE"], "RED_WHITE_RED": ["AT", "LV"],
            "YELLOW_BLUE_RED": ["CO", "EC"], "BLUE_YELLOW": ["UA"], "BLUE_RED": ["US", "MY", "AU", "NZ", "CL"],
            "RED_FIELD": ["TR", "VN", "CA", "DK", "CH", "PE"]}


def report_placements():
    z = np.load(os.path.join(OUT, "placements.npz"), allow_pickle=True)
    S, flag, cc = z["S"], z["flag"].astype(bool), z["cc"]
    S = np.where(np.isfinite(S), S, -1.0)
    neg = ~flag
    print("\n%d flag placement views, %d negative views (random heading, same zoom / pitch distribution)"
          % (flag.sum(), neg.sum()))
    print("%-18s %-14s %5s %6s %8s %8s %8s" % ("template", "flags of", "n", "AUC", "det@1%FA", "det@5%FA", "AUC all"))
    res = {}
    for ti, name in enumerate(z["templates"]):
        name = str(name)
        mine = flag & np.isin(cc, INTENDED.get(name, []))
        t1, t5 = np.percentile(S[neg, ti], 99), np.percentile(S[neg, ti], 95)
        a = auc(S[mine, ti], S[neg, ti])
        res[name] = {"n": int(mine.sum()), "auc": a, "det_fa1": float((S[mine, ti] > t1).mean()) if mine.any() else None,
                     "det_fa5": float((S[mine, ti] > t5).mean()) if mine.any() else None,
                     "auc_all_flags": auc(S[flag, ti], S[neg, ti])}
        print("%-18s %-14s %5d %6.3f %8s %8s %8.3f" % (
            name, ",".join(INTENDED.get(name, [])), mine.sum(), a,
            "%.2f" % res[name]["det_fa1"] if mine.any() else "-", "%.2f" % res[name]["det_fa5"] if mine.any() else "-",
            res[name]["auc_all_flags"]))
    # best template per flag view, any template: max over templates of the standardised score
    mu, sd = S[neg].mean(0), S[neg].std(0) + 1e-6
    Zs = (S - mu) / sd
    anyz = Zs.max(1)
    t1, t5 = np.percentile(anyz[neg], 99), np.percentile(anyz[neg], 95)
    print("any template (max z): AUC %.3f, det@1%%FA %.2f, det@5%%FA %.2f" % (
        auc(anyz[flag], anyz[neg]), (anyz[flag] > t1).mean(), (anyz[flag] > t5).mean()))
    for c in ("KH", "TH", "VN", "ID", "MY", "US", "TR", "CA", "RU"):
        s = flag & (cc == c)
        if s.any():
            print("   %s: n %d, any-template AUC %.3f det@5%%FA %.2f" % (c, s.sum(), auc(anyz[s], anyz[neg]),
                                                                    (anyz[s] > t5).mean()))
    json.dump(res, open(os.path.join(OUT, "placements_report.json"), "w"), indent=1)


def _feat_job(job):
    pid, heading, live = job
    try:
        if live:
            from live_features import live_sphere
            sph = live_sphere(pano_path(pid), heading, pid)
        else:
            sph = SphericalImage.from_equirect(pano_path(pid), heading=heading)
        return pid, F.scan(sph), None
    except Exception as e:  # noqa: BLE001
        return pid, None, repr(e)


def cmd_features(args):
    os.makedirs(OUT, exist_ok=True)
    recs = load_records(pixels=True)
    live = args.set.startswith("live_")
    split = args.set.replace("live_", "")
    if split == "train":
        by = {}
        for r in recs:
            if split_of(r) == "train":
                by.setdefault(r["label"], []).append(r)
        sel = []
        rs = random.Random(1)
        for c in sorted(by):
            v = sorted(by[c], key=lambda r: r["pano_id"])
            rs.shuffle(v)
            sel += v[:args.per_country]
    else:
        sel = [r for r in recs if split_of(r) == split and (not live or r.get("heading") is not None)]
    path = os.path.join(OUT, "features_%s.npz" % args.set)
    done = {}
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        done = {str(p): x for p, x in zip(z["ids"], z["X"])}
    todo = [(r["pano_id"], r.get("heading"), live) for r in sel if r["pano_id"] not in done]
    print("%s: %d panoramas, %d to do" % (args.set, len(sel), len(todo)), flush=True)
    t0 = time.time()

    def save():
        ids = sorted(done)
        np.savez_compressed(path, ids=np.array(ids), X=np.array([done[p] for p in ids]),
                            names=np.array(F.FEATURE_NAMES))

    with Pool(args.workers) as pool:
        for k, (pid, x, err) in enumerate(pool.imap_unordered(_feat_job, todo, chunksize=4)):
            if x is None:
                print("  %s: %s" % (pid, err))
                continue
            done[pid] = x
            if (k + 1) % 200 == 0:
                save()
                print("  %d/%d (%.0fs)" % (k + 1, len(todo), time.time() - t0), flush=True)
    save()
    print("saved %s (%d)" % (path, len(done)))


def load_features(name):
    z = np.load(os.path.join(OUT, "features_%s.npz" % name), allow_pickle=True)
    return {str(p): x for p, x in zip(z["ids"], z["X"])}


class FlagNB:
    """Bernoulli naive Bayes on "template fired" indicators: template t fires when its maximum exceeds the
    q-quantile of all training panoramas; P(fires | country) with a Beta prior of strength a around the overall
    rate.  log-likelihood of a panorama = sum_t log P(fire_t | c) (independence assumed; the exponent calibrated
    on CALIB absorbs the correlation)."""

    def __init__(self, q=0.95, a=10.0):
        self.q, self.a = q, a

    def fit(self, X, y, C):
        T = len(X[0]) - 3
        self.thr = np.nanpercentile(X[:, :T], 100 * self.q, axis=0)
        F = (X[:, :T] > self.thr).astype(float)
        base = F.mean(0)
        n = np.bincount(y, minlength=C).astype(float)
        k = np.zeros((C, T))
        np.add.at(k, y, F)
        self.p = (k + self.a * base[None, :]) / (n[:, None] + self.a)
        return self

    def loglik(self, X):
        T = self.p.shape[1]
        F = (np.nan_to_num(X[:, :T], nan=-9.0) > self.thr).astype(float)
        return F @ np.log(self.p).T + (1 - F) @ np.log(1 - self.p).T


def flag_model(classes, kind="gauss", ids=None):
    """Country model of the template maxima fitted on the training panoramas (features_train.npz; ids: a subset):
    "gauss" = engine.model.GroupGaussian (shared covariance), "nb" = FlagNB."""
    from engine.model import GroupGaussian
    recs = {r["pano_id"]: r for r in load_records(pixels=True)}
    tr = load_features("train")
    cidx = {c: i for i, c in enumerate(classes)}
    ids = [p for p in sorted(tr) if p in recs and recs[p]["label"] in cidx and (ids is None or p in ids)]
    X = np.array([tr[p] for p in ids])
    y = np.array([cidx[recs[p]["label"]] for p in ids])
    if kind == "nb":
        return FlagNB().fit(X, y, len(classes)), len(ids)
    return GroupGaussian(shrink=0.25, mean_tau=4.0, n_proj=8).fit(X, y, len(classes)), len(ids)


def cmd_evidence(args):
    """Country evidence of the flag template maxima on CALIB live (and TEST with --test): class-conditional
    Gaussians fitted on the training panoramas; exponent w on top of the saved locator's posterior (live evidence of
    scratch/features_live, official World prior / each round's own map prior), 2-fold CV by game on CALIB."""
    from build_priors import fold_of
    from engine.geo import resolve_map
    from engine.model import GeoModel
    from train_model import live_inputs, paired_diff, round_maps
    model = GeoModel.load()
    gg, n_train = flag_model(model.classes, args.kind)
    print("flag model (%s): %d training panoramas, %d classes" % (args.kind, n_train, len(model.classes)))
    recs = load_records()
    pidx = {r["pano_id"]: i for i, r in enumerate(recs)}
    out = {}
    for split in ("calib",) + (("test",) if args.test else ()):
        fl = load_features("live_" + split)
        rows = np.array([pidx[p] for p in sorted(fl) if p in pidx])
        ev, _ = live_inputs(model, recs, rows)
        cls = {c: i for i, c in enumerate(model.classes)}
        y = np.array([cls.get(recs[i]["label"], -1) for i in rows])
        ok = y >= 0
        Xf = np.array([fl[recs[i]["pano_id"]] for i in rows])
        lf = gg.loglik(Xf)
        lf = lf - lf.max(1, keepdims=True)
        games = np.array([str(recs[i].get("game")) for i in rows])
        world = model.map_setup(resolve_map("world"))
        cache, own = {}, []
        for mp in round_maps(recs, rows):
            key = (mp or {}).get("id")
            if key not in cache:
                cache[key] = model.map_setup(mp)
            own.append(cache[key])
        res = {}
        for name, setups in (("world_prior", [world] * len(rows)), ("own_map_prior", own)):
            base = np.stack([model.combine({k: (v[i:i + 1] if isinstance(v, np.ndarray) else v) for k, v in ev.items()},
                                           setups[i]["weights"], prior=setups[i]["prior"])[0] for i in range(len(rows))])
            grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

            def ll(w):
                lp = base + w * lf
                lp = lp - lp.max(1, keepdims=True)
                lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
                return lp[np.arange(len(y)), np.maximum(y, 0)], lp

            L = np.array([ll(w)[0] for w in grid])
            f2 = np.array([fold_of(g, 2) for g in games])
            cvw = np.zeros(len(y))
            ws = []
            for f in (0, 1):
                tr = ok & (f2 != f)
                k = int(np.argmax(L[:, tr].mean(1)))
                ws.append(grid[k])
                cvw[f2 == f] = L[k, f2 == f]
            d = paired_diff(cvw[ok], L[0, ok], games[ok])
            kbest = int(np.argmax(L[:, ok].mean(1)))
            lp0, lpb = ll(0.0)[1], ll(grid[kbest])[1]
            res[name] = {"grid": dict(zip(map(str, grid), [round(float(v), 4) for v in L[:, ok].mean(1)])),
                         "best_w_in_sample": grid[kbest], "cv_weights": ws, "cv_loglik_gain": d,
                         "top1_without": float((lp0.argmax(1) == y)[ok].mean()),
                         "top1_best_w_in_sample": float((lpb.argmax(1) == y)[ok].mean())}
            print("%s %s: loglik by w %s; 2-fold CV weights %s, CV gain %+.4f %s; top1 %.3f -> %.3f (w %.2f in-sample)" % (
                split, name, res[name]["grid"], ws, d["diff"], d["ci95"], res[name]["top1_without"],
                res[name]["top1_best_w_in_sample"], grid[kbest]))
        out[split] = res
    json.dump(out, open(os.path.join(OUT, "evidence_report_%s.json" % args.kind), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("placements", "report", "features", "evidence"))
    ap.add_argument("--neg", type=int, default=900)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--set", default="train")
    ap.add_argument("--per-country", type=int, default=120)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--kind", default="gauss", choices=("gauss", "nb"), help="evidence: country model of the maxima")
    args = ap.parse_args()
    if args.cmd == "placements":
        cmd_placements(args)
    elif args.cmd == "report":
        report_placements()
    elif args.cmd == "features":
        cmd_features(args)
    else:
        cmd_evidence(args)


if __name__ == "__main__":
    main()
