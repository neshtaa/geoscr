#!/usr/bin/env python3
"""
Fit, calibrate and evaluate the country/location model.

  python3 tools/train_model.py [--groups solar,road,...] [--save]

train  : world + balanced Street View samples (feature cache from tools/calibrate.py)
calib  : half of the user's real GeoGuessr World-map rounds -> evidence weights, prior mix
test   : the other half -> reported accuracy and mean GeoGuessr points per round
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from calibrate import FEAT_DIR, load_records, split_of  # noqa: E402
from engine.geo import geoguessr_score, haversine_km, region_index  # noqa: E402
from engine.model import GeoModel  # noqa: E402


def group_matrix(name, recs):
    z = np.load(os.path.join(FEAT_DIR, name + ".npz"), allow_pickle=True)
    Xz = z["X"]
    idx = {pid: i for i, pid in enumerate(z["ids"])}
    M = np.full((len(recs), Xz.shape[1]), np.nan, np.float64)
    have = np.zeros(len(recs), bool)
    for i, r in enumerate(recs):
        j = idx.get(r["pano_id"])
        if j is not None:
            M[i] = Xz[j]
            have[i] = True
    return M, have


def evaluate(model, ev, recs, rows, label, verbose=True):
    lp = model.combine(ev)
    cls = {c: i for i, c in enumerate(model.classes)}
    y = [recs[i]["label"] for i in rows]
    yi = np.array([cls.get(c, -1) for c in y])
    P = np.exp(lp)
    true_p = np.where(yi >= 0, P[np.arange(len(yi)), np.maximum(yi, 0)], 0.0)
    rank = (P > true_p[:, None]).sum(1)
    rank = np.where(yi >= 0, rank, 999)
    scores, dists, rrank, rrank_ok = [], [], [], []
    true_reg = region_index([recs[i]["lat"] for i in rows], [recs[i]["lng"] for i in rows])
    for k, i in enumerate(rows):
        g = model.locate(lp[k], ev["_d2"][k])
        d = float(haversine_km(recs[i]["lat"], recs[i]["lng"], g["lat"], g["lng"]))
        dists.append(d)
        scores.append(float(geoguessr_score(d)))
        pr = model.region_posterior(lp[k], ev["_d2"][k])
        t = true_reg[k]
        r = int((pr > pr[t]).sum()) if t and t < len(pr) and pr[t] > 0 else 999
        rrank.append(r)
        if rank[k] == 0:
            rrank_ok.append(r)
    rrank, rrank_ok = np.array(rrank), np.array(rrank_ok)
    res = {"n": len(rows), "top1": float((rank == 0).mean()), "top3": float((rank < 3).mean()),
           "top5": float((rank < 5).mean()), "mean_score": float(np.mean(scores)),
           "median_km": float(np.median(dists)), "loglik": float(np.mean(np.log(np.maximum(true_p, 1e-9)))),
           "region_top1": float((rrank == 0).mean()), "region_top3": float((rrank < 3).mean()),
           "region_top1_if_country_right": float((rrank_ok == 0).mean()) if len(rrank_ok) else 0.0,
           "region_top3_if_country_right": float((rrank_ok < 3).mean()) if len(rrank_ok) else 0.0}
    if verbose:
        print(f"[{label}] n={res['n']} top1={res['top1']:.3f} top3={res['top3']:.3f} top5={res['top5']:.3f} "
              f"points/round={res['mean_score']:.0f} median={res['median_km']:.0f} km loglik={res['loglik']:.3f} | "
              f"region top1={res['region_top1']:.3f} top3={res['region_top3']:.3f} "
              f"(country right: {res['region_top1_if_country_right']:.3f} / {res['region_top3_if_country_right']:.3f})")
    return res


def calibrate_location(model, ev, recs, rows):
    """Grid search of the within-country kernel: location (mean points) and region
    (mean log P(true region)) parameters on the CALIB rounds."""
    lp = model.combine(ev)
    true_reg = region_index([recs[i]["lat"] for i in rows], [recs[i]["lng"] for i in rows])
    best_loc, best_reg = (-1, None), (-1e9, None)
    for bw in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        for floor in (1e-3, 0.03, 0.1, 0.3):
            pts, rl = [], []
            for k, i in enumerate(rows):
                w = model.ref_weights(lp[k], ev["_d2"][k], bw, floor)
                g = model.locate(lp[k], ev["_d2"][k], w=w)
                pts.append(float(geoguessr_score(haversine_km(recs[i]["lat"], recs[i]["lng"], g["lat"], g["lng"]))))
                p = np.bincount(model.ref_region, weights=w, minlength=int(model.ref_region.max()) + 1)
                t = true_reg[k]
                rl.append(np.log(max(p[t] if t < len(p) else 0.0, 1e-4)))
            if np.mean(pts) > best_loc[0]:
                best_loc = (float(np.mean(pts)), {"bw": bw, "floor": floor})
            if np.mean(rl) > best_reg[0]:
                best_reg = (float(np.mean(rl)), {"bw": bw, "floor": floor})
    model.loc_params, model.region_params = best_loc[1], best_reg[1]
    print(f"location kernel {best_loc[1]} ({best_loc[0]:.0f} pts on calib), "
          f"region kernel {best_reg[1]} (log P(region) {best_reg[0]:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--min-count", type=int, default=6)
    args = ap.parse_args()
    from engine import features
    groups = args.groups.split(",") if args.groups else [m.NAME for m in features.available()
                                                         if os.path.exists(os.path.join(FEAT_DIR, m.NAME + ".npz"))]
    recs = load_records()
    mats, have = {}, np.ones(len(recs), bool)
    for g in groups:
        mats[g], h = group_matrix(g, recs)
        have &= h
    split = np.array([split_of(r) for r in recs])
    tr = np.flatnonzero((split == "train") & have)
    ca = np.flatnonzero((split == "calib") & have)
    te = np.flatnonzero((split == "test") & have)
    print(f"groups={groups} train={len(tr)} calib={len(ca)} test={len(te)}")
    t0 = time.time()
    model = GeoModel().fit({g: mats[g][tr] for g in groups}, [recs[i]["label"] for i in tr],
                           [recs[i]["lat"] for i in tr], [recs[i]["lng"] for i in tr],
                           [recs[i]["mode"] == "world" for i in tr], min_count=args.min_count)
    print(f"fit {time.time() - t0:.1f}s, classes={len(model.classes)}")

    def ev_for(rows):
        return model.evidence({g: mats[g][rows] for g in groups})

    ev_ca, ev_te = ev_for(ca), ev_for(te)
    cls = {c: i for i, c in enumerate(model.classes)}
    yca = np.array([cls.get(recs[i]["label"], -1) for i in ca])
    ok = yca >= 0
    ev_ca_ok = {k: (v[ok] if v is not None else None) for k, v in ev_ca.items()}
    evaluate(model, ev_te, recs, te, "test, default weights")
    ll = model.calibrate(ev_ca_ok, yca[ok])
    print("calibrated weights:", json.dumps({k: round(v, 3) for k, v in model.weights.items()}),
          "prior_mix", model.prior_mix, f"calib loglik {ll:.3f}")
    calibrate_location(model, ev_ca, recs, ca)
    evaluate(model, ev_ca, recs, ca, "calib")
    res = evaluate(model, ev_te, recs, te, "TEST")
    # prior-only reference
    zero = {k: (np.zeros_like(v) if isinstance(v, np.ndarray) and k != "_d2" else v) for k, v in ev_te.items()}
    evaluate(model, zero, recs, te, "test, prior only")
    if args.save:
        model.save()
        json.dump(res, open(os.path.join(ROOT, "data", "model", "eval.json"), "w"), indent=1)
        print("saved data/model")


if __name__ == "__main__":
    main()
