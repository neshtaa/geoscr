#!/usr/bin/env python3
"""
Fit, calibrate and evaluate the country/location model.

  python3 tools/train_model.py [--groups solar,road,...] [--save]
  python3 tools/train_model.py --eval-only [--save]         saved model on calib/test, without and with map
  python3 tools/train_model.py --calibrate-maps [--save]    refit only the map-aware prior on calib
  python3 tools/train_model.py --calibrate-cards [--save]   exponent of the GeoGuessr card detections
                                                            (tools/build_clue_detectors.py) on calib
  --no-cards                                                evaluate without the card evidence

Order after new data: train_model.py --save (refit + calibration), tools/build_priors.py when the
rounds behind the counts change, then train_model.py --calibrate-maps --save.

train  : world + balanced Street View samples (feature cache from tools/calibrate.py)
calib  : half of the user's real GeoGuessr World-map rounds -> evidence weights, prior mix,
         map-aware prior mix (data/model/priors.json from tools/build_priors.py)
test   : the other half -> reported accuracy and mean GeoGuessr points per round (each round
         scored with its map's maxErrorDistance, data/maps.json)
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
from build_clue_detectors import cached_loglik  # noqa: E402
from build_priors import fold_of, game_split  # noqa: E402
from calibrate import FEAT_DIR, load_records, split_of  # noqa: E402
from engine.geo import geoguessr_points, geoguessr_score, haversine_km, region_index, resolve_map  # noqa: E402
from engine.model import MODEL_DIR, PRIORS_FILE, GeoModel, mix_prior  # noqa: E402


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


def round_maps(recs, rows):
    """Map of every round (history 'map' name -> data/maps.json), None if unknown."""
    cache = {}
    for i in rows:
        name = recs[i].get("map")
        if name not in cache:
            cache[name] = resolve_map(name)
    return [cache[recs[i].get("map")] for i in rows]


def map_posterior(model, ev, setups):
    """Log posterior with a per-round prior and the evidence exponents of each round's setup."""
    lp = np.zeros_like(ev["knn"])
    groups = {}
    for k, s in enumerate(setups):
        groups.setdefault(id(s["weights"]), (s["weights"], []))[1].append(k)
    for w, idx in groups.values():
        sub = {key: (v[idx] if isinstance(v, np.ndarray) else v) for key, v in ev.items()}
        lp[idx] = model.combine(sub, w, prior=np.stack([setups[k]["prior"] for k in idx]))
    return lp


def summarize(r, sel=None):
    """Metrics of the per-round arrays of evaluate() (optionally a subset)."""
    sel = np.ones(len(r["rank"]), bool) if sel is None else np.asarray(sel, bool)
    rank, rrank = r["rank"][sel], r["rrank"][sel]
    ok = rrank[rank == 0]
    return {"n": int(sel.sum()), "top1": float((rank == 0).mean()), "top3": float((rank < 3).mean()),
            "top5": float((rank < 5).mean()), "mean_score": float(np.mean(r["score"][sel])),
            "mean_score_world_formula": float(np.mean(r["score_world"][sel])),
            "median_km": float(np.median(r["km"][sel])),
            "loglik": float(np.mean(np.log(np.maximum(r["true_p"][sel], 1e-9)))),
            "region_top1": float((rrank == 0).mean()), "region_top3": float((rrank < 3).mean()),
            "region_top1_if_country_right": float((ok == 0).mean()) if len(ok) else 0.0,
            "region_top3_if_country_right": float((ok < 3).mean()) if len(ok) else 0.0}


def evaluate(model, ev, recs, rows, label, use_map=False, use=("map", "ranked"), groups=None, verbose=True):
    """Country / region accuracy and GeoGuessr points (each round scored with its own map's
    maxErrorDistance).  use_map: the locator also gets the map (prior, score scale, bounds); use:
    the empirical priors allowed (engine.model.GeoModel.map_components).  groups: a label per row,
    adds "by_group" metrics."""
    maps = round_maps(recs, rows)
    cache = {}
    setups = []
    for mp in maps:
        key = (mp or {}).get("id") or (mp or {}).get("name")
        if key not in cache:
            cache[key] = model.map_setup(mp if use_map else None, use=use)
        setups.append(cache[key])
    lp = map_posterior(model, ev, setups)
    cls = {c: i for i, c in enumerate(model.classes)}
    yi = np.array([cls.get(recs[i]["label"], -1) for i in rows])
    P = np.exp(lp)
    true_p = np.where(yi >= 0, P[np.arange(len(yi)), np.maximum(yi, 0)], 0.0)
    rank = np.where(yi >= 0, (P > true_p[:, None]).sum(1), 999)
    r = {"rank": rank, "true_p": true_p, "score": [], "score_world": [], "km": [], "rrank": []}
    true_reg = region_index([recs[i]["lat"] for i in rows], [recs[i]["lng"] for i in rows])
    for k, i in enumerate(rows):
        s = setups[k]
        g = model.locate(lp[k], ev["_d2"][k], score_scale_km=s["scale_km"], bounds=s["bounds"])
        d = float(haversine_km(recs[i]["lat"], recs[i]["lng"], g["lat"], g["lng"]))
        r["km"].append(d)
        r["score"].append(float(geoguessr_points(d, (maps[k] or {}).get("maxErrorDistance"))))
        r["score_world"].append(float(geoguessr_score(d)))
        pr = model.region_posterior(lp[k], ev["_d2"][k], bounds=s["bounds"])
        t = true_reg[k]
        r["rrank"].append(int((pr > pr[t]).sum()) if t and t < len(pr) and pr[t] > 0 else 999)
    r = {k: np.asarray(v) for k, v in r.items()}
    res = dict(summarize(r), map_info=bool(use_map))
    if groups is not None:
        groups = np.asarray(groups)
        res["by_group"] = {g: summarize(r, groups == g) for g in dict.fromkeys(groups.tolist())}
    if verbose:
        print(f"[{label}] n={res['n']} top1={res['top1']:.3f} top3={res['top3']:.3f} top5={res['top5']:.3f} "
              f"points/round={res['mean_score']:.0f} (World formula {res['mean_score_world_formula']:.0f}) "
              f"median={res['median_km']:.0f} km loglik={res['loglik']:.3f} | "
              f"region top1={res['region_top1']:.3f} top3={res['region_top3']:.3f} "
              f"(country right: {res['region_top1_if_country_right']:.3f} / {res['region_top3_if_country_right']:.3f})")
        for g, v in res.get("by_group", {}).items():
            print(f"    {g:<34} n={v['n']:3d} top1={v['top1']:.3f} points/round={v['mean_score']:.0f} "
                  f"median={v['median_km']:.0f} km")
    return res


def calibrate_maps(model, ev, recs, rows, iters=3, only=None):
    """Map-aware prior on the CALIB rounds: evidence exponents, model-prior mix and the weights of
    the empirical priors by coordinate ascent on the mean log P(true country).  A CALIB round's
    per-map prior is built without its own fold of games (out-of-fold), so the mix is not fitted
    to rounds it has already seen.  'ranked' (World-type maps without their own prior) is fitted
    as if no round had a per-map prior.  only: refit just these exponents on top of the saved
    map parameters."""
    if not model.priors.get("ranked_world"):
        print("no data/model/priors.json - run tools/build_priors.py first; map-aware prior skipped")
        return None
    cls = {c: i for i, c in enumerate(model.classes)}
    yi = np.array([cls.get(recs[i]["label"], -1) for i in rows])
    ok = yi >= 0
    rows = [i for i, o in zip(rows, ok) if o]
    ev = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in ev.items()}
    yi = yi[ok]
    k_folds = int(model.priors.get("folds", 5))
    C = len(model.classes)
    kinds, world, R, E, W = [], [], [], [], []
    for i, mp in zip(rows, round_maps(recs, rows)):
        emp = model.map_entry(mp)
        oof = None
        if emp:
            f = emp["fold_counts"][fold_of(recs[i]["game"], k_folds)]
            oof = {c: n - f.get(c, 0) for c, n in emp["counts"].items()}
        kind, ranked, e, wb = model.map_components(mp, oof)
        kinds.append(kind)
        world.append(bool((mp or {}).get("world")))
        R.append(ranked if ranked is not None else np.zeros(C))
        E.append(e if e is not None else np.zeros(C))
        W.append(wb)
    kinds, world, R, E, W = np.array(kinds), np.array(world), np.array(R), np.array(E), np.array(W)
    print("map-aware calibration on %d calib rounds: %s" % (len(rows), dict(zip(*np.unique(kinds, return_counts=True)))))

    def prior(pm, mix, all_ranked=False):
        P = W * model.base_prior(pm)[None, :]
        P /= P.sum(1, keepdims=True)
        km = (kinds == "map") & (not all_ranked)
        kr = (kinds == "ranked") | ((kinds == "map") & world & all_ranked)
        P[km] = mix_prior("map", mix, P[km], R[km], E[km])
        P[kr] = mix_prior("ranked", mix, P[kr], R[kr])
        return P

    def obj(w, pm, mix, all_ranked=False):
        lp = model.combine(ev, w, prior=prior(pm, mix, all_ranked))
        return float(np.mean(lp[np.arange(len(yi)), yi]))

    keys = model.groups + ["knn"] + [k for k in ("glm", "sun", "cards") if k in ev]
    w, pm = dict(model.weights), model.prior_mix
    mix = {"map": 0.5, "map_ranked": 0.0, "ranked": 0.5}
    if only:
        p = model.map_params() or {}
        w, pm, mix = dict(p.get("weights", w)), p.get("prior_mix", pm), dict(p.get("mix", mix))
        keys = [k for k in keys if k in only]
    best = obj(w, pm, mix)
    start = best
    grid = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3, 1.7]
    mgrid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for _ in range(iters):
        for k in keys:
            for v in grid:
                s = obj(dict(w, **{k: v}), pm, mix)
                if s > best:
                    best, w = s, dict(w, **{k: v})
        if only:
            best_r = obj(w, pm, mix, True)
            break
        for v in [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]:
            s = obj(w, v, mix)
            if s > best:
                best, pm = s, v
        for name in ("map", "map_ranked"):
            for v in mgrid:
                m2 = dict(mix, **{name: v})
                if m2["map"] + m2["map_ranked"] > 1.0:
                    continue
                s = obj(w, pm, m2)
                if s > best:
                    best, mix = s, m2
        best_r = obj(w, pm, mix, True)
        for v in mgrid:
            s = obj(w, pm, dict(mix, ranked=v), True)
            if s > best_r:
                best_r, mix = s, dict(mix, ranked=v)
        best = obj(w, pm, mix)
    no_map = obj(model.weights, model.prior_mix, mix={"map": 0.0, "map_ranked": 0.0, "ranked": 0.0})
    params = {"weights": w, "prior_mix": pm, "mix": mix, "alpha": model.priors.get("alpha"),
              "model": model.fingerprint(), "calib_rounds": len(rows), "calib_loglik": round(best, 4),
              "calib_loglik_ranked_only": round(best_r, 4), "calib_loglik_model_prior_bounds": round(no_map, 4)}
    model.priors["params"] = params
    print("map params: weights %s prior_mix %s mix %s | calib loglik %.3f (start %.3f, model prior + bounds %.3f, "
          "ranked_world for all %.3f)" % (json.dumps({k: round(v, 3) for k, v in w.items()}), pm, mix, best, start,
                                          no_map, best_r))
    return params


def map_groups(model, recs, rows):
    """Evaluation group of each round: the name of a map with its own prior, else the prior kind."""
    out = []
    for mp in round_maps(recs, rows):
        kind = model.map_components(mp)[0]
        out.append(mp["name"] if kind == "map" else {"ranked": "World-type maps, ranked prior"}.get(
            kind, "other maps, model prior"))
    return out


def map_report(model, ev_ca, ev_te, recs, ca, te, ev_ot=None, ot=()):
    """Without / with map info on CALIB and TEST (per map group), the map-prior ablations and the
    single-country maps outside the calib/test splits (ev_ot / ot) -> data/model/eval_maps.json."""
    with_map = bool(model.map_params())
    out = {"note": "TEST = honest (parameters and per-map counts from CALIB games only); calib_map is "
                   "in-sample (its per-map counts include those rounds); other_maps: small n, indicative "
                   "(the box-share prior of single-country maps was designed after the UK rounds failed)"}
    abl = {}
    for ev, rows, label in ((ev_ca, ca, "calib"), (ev_te, te, "TEST")):
        groups = map_groups(model, recs, rows)
        key = label.lower()
        out[key + "_no_map"] = evaluate(model, ev, recs, rows, label + ", no map", groups=groups)
        if not with_map:
            continue
        out[key + "_map"] = evaluate(model, ev, recs, rows, label + ", map info", True, groups=groups)
        if key == "calib":
            out[key + "_map"]["in_sample"] = True
        abl[key + "_scale_bounds_only"] = evaluate(model, ev, recs, rows, label + ", map scale + bounds only", True,
                                                   use=(), groups=groups)
        abl[key + "_ranked_for_all"] = evaluate(model, ev, recs, rows, label + ", ranked prior for every map",
                                                True, use=("ranked",), groups=groups)
    if abl:
        out["ablation"] = abl
    if ev_ot is not None and len(ot):
        other = {}
        names = [recs[i].get("map") for i in ot]
        for name in dict.fromkeys(names):
            sel = [k for k, n in enumerate(names) if n == name]
            label = name
            if model.map_entry(resolve_map(name)):  # its own counts: score only the games they exclude
                sel = [k for k in sel if game_split(recs[ot[k]]) == "test"]
                label = name + " (test-half games)"
            if not sel:
                continue
            ev = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in ev_ot.items()}
            rows = [ot[k] for k in sel]
            other[label] = {"no_map": evaluate(model, ev, recs, rows, label + ", no map"),
                            "map": evaluate(model, ev, recs, rows, label + ", map info", True)}
        out["other_maps"] = other
    return out


def add_cards(model, ev, recs, rows):
    """ev["cards"]: card-detector log-likelihoods of the rows from the detection cache of
    tools/build_clue_detectors.py (left out when the bank or the cache is missing or stale)."""
    ll = cached_loglik([recs[i]["pano_id"] for i in rows], model.classes)
    if ll is not None:
        ev["cards"] = ll
    elif model.weights.get("cards", 0.0) > 0:
        sys.stderr.write("card evidence (weight %.2f in model.json) left out of this run\n" % model.weights["cards"])
    return ev


def calibrate_cards(model, ev, recs, rows):
    """Exponent of the GeoGuessr card evidence on the CALIB rounds with the other exponents and the
    priors fixed: 1-D grid on the mean log P(true country), with the model prior (model weights) and
    with the map-aware prior (priors.json params)."""
    if "cards" not in ev:
        print("no card detections (tools/build_clue_detectors.py) - card exponent not calibrated")
        return
    cls = {c: i for i, c in enumerate(model.classes)}
    yi = np.array([cls.get(recs[i]["label"], -1) for i in rows])
    ok = yi >= 0
    sub = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in ev.items()}
    grid = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3, 1.7]
    ll = np.array([model.combine(sub, dict(model.weights, cards=v))[np.arange(ok.sum()), yi[ok]] for v in grid])
    s = ll.mean(1)
    model.weights["cards"] = grid[int(np.argmax(s))]
    print("card exponent %.2f: calib loglik %.4f (without cards %.4f)" % (model.weights["cards"], max(s), s[0]))
    # the in-sample gain above is >= 0 by construction: 2-fold cross-validation over CALIB games
    fold = np.array([fold_of(recs[i]["game"], 2) for i in np.asarray(rows)[ok]])
    gain = np.zeros(ok.sum())
    for f in (0, 1):
        v = int(np.argmax(ll[:, fold != f].mean(1)))
        gain[fold == f] = ll[v, fold == f] - ll[0, fold == f]
        print("  fold %d: exponent %.2f fitted on the other fold" % (f, grid[v]))
    idx = np.random.RandomState(0).randint(0, len(gain), (4000, len(gain)))
    print("card evidence, 2-fold cross-validated calib loglik gain %+.4f [%+.4f, %+.4f]" % (
        gain.mean(), np.percentile(gain[idx].mean(1), 2.5), np.percentile(gain[idx].mean(1), 97.5)))
    if model.map_params():
        calibrate_maps(model, ev, recs, rows, only=("cards",))


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
    ap.add_argument("--eval-only", action="store_true", help="evaluate the saved model on calib/test, no fitting")
    ap.add_argument("--calibrate-maps", action="store_true",
                    help="saved model: refit the map-aware prior parameters on calib (data/model/priors.json)")
    ap.add_argument("--calibrate-cards", action="store_true",
                    help="saved model: calibrate the exponent of the card detections on calib")
    ap.add_argument("--no-cards", action="store_true", help="leave the card detections out")
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
    ot = np.flatnonzero((split == "other") & have)
    if args.eval_only or args.calibrate_maps or args.calibrate_cards:
        model = GeoModel.load()
        ev_ca, ev_te = (model.evidence({g: mats[g][rows] for g in model.groups}) for rows in (ca, te))
        ev_ot = model.evidence({g: mats[g][ot] for g in model.groups}) if len(ot) else None
        if not args.no_cards:
            add_cards(model, ev_ca, recs, ca)
            add_cards(model, ev_te, recs, te)
            if ev_ot is not None:
                add_cards(model, ev_ot, recs, ot)
        if args.calibrate_maps:
            calibrate_maps(model, ev_ca, recs, ca)
        if args.calibrate_cards:
            calibrate_cards(model, ev_ca, recs, ca)
        out = map_report(model, ev_ca, ev_te, recs, ca, te, ev_ot, ot)
        if args.save:
            if args.calibrate_maps or args.calibrate_cards:
                model.save_priors()
            if args.calibrate_cards:  # only the exponents change: model.json, not the fitted arrays
                mj = os.path.join(MODEL_DIR, "model.json")
                meta = json.load(open(mj))
                meta["weights"]["cards"] = model.weights.get("cards", 0.0)
                json.dump(meta, open(mj, "w"), indent=1)
            json.dump(out, open(os.path.join(MODEL_DIR, "eval_maps.json"), "w"), indent=1)
            json.dump(out["test_no_map"], open(os.path.join(MODEL_DIR, "eval.json"), "w"), indent=1)
            print("saved data/model/" + PRIORS_FILE + " params, eval_maps.json, eval.json")
        return
    t0 = time.time()
    model = GeoModel().fit({g: mats[g][tr] for g in groups}, [recs[i]["label"] for i in tr],
                           [recs[i]["lat"] for i in tr], [recs[i]["lng"] for i in tr],
                           [recs[i]["mode"] == "world" for i in tr], min_count=args.min_count)
    print(f"fit {time.time() - t0:.1f}s, classes={len(model.classes)}")

    def ev_for(rows):
        ev = model.evidence({g: mats[g][rows] for g in groups})
        return ev if args.no_cards else add_cards(model, ev, recs, rows)

    ev_ca, ev_te, ev_ot = ev_for(ca), ev_for(te), (ev_for(ot) if len(ot) else None)
    cls = {c: i for i, c in enumerate(model.classes)}
    yca = np.array([cls.get(recs[i]["label"], -1) for i in ca])
    ok = yca >= 0
    ev_ca_ok = {k: (v[ok] if v is not None else None) for k, v in ev_ca.items()}
    evaluate(model, ev_te, recs, te, "test, default weights")
    ll = model.calibrate(ev_ca_ok, yca[ok])
    print("calibrated weights:", json.dumps({k: round(v, 3) for k, v in model.weights.items()}),
          "prior_mix", model.prior_mix, f"calib loglik {ll:.3f}")
    calibrate_location(model, ev_ca, recs, ca)
    # prior-only reference
    zero = {k: (np.zeros_like(v) if isinstance(v, np.ndarray) and k != "_d2" else v) for k, v in ev_te.items()}
    evaluate(model, zero, recs, te, "test, prior only")
    # map-aware prior (per-map / ranked-duel country frequencies, score scale, bounds); the counts
    # (tools/build_priors.py) cover every country, so they need no rebuild after a refit
    pf = os.path.join(MODEL_DIR, PRIORS_FILE)
    model.priors = json.load(open(pf)) if os.path.exists(pf) else {}
    params = calibrate_maps(model, ev_ca, recs, ca)
    out = map_report(model, ev_ca, ev_te, recs, ca, te, ev_ot, ot)
    if args.save:
        model.save()
        if params:
            model.save_priors()
        json.dump(out, open(os.path.join(MODEL_DIR, "eval_maps.json"), "w"), indent=1)
        json.dump(out["test_no_map"], open(os.path.join(MODEL_DIR, "eval.json"), "w"), indent=1)
        print("saved data/model")


if __name__ == "__main__":
    main()
