#!/usr/bin/env python3
"""
Fit, calibrate and evaluate the country/location model.

  python3 tools/train_model.py [--groups solar,road,...] [--save]
  python3 tools/train_model.py --eval-only [--save]         saved model on calib/test, without and with map
  python3 tools/train_model.py --calibrate-maps [--save]    refit only the map-aware prior on calib
  python3 tools/train_model.py --calibrate-cards [--save]   exponent of the GeoGuessr card detections
                                                            (tools/build_clue_detectors.py) on calib
  --no-cards                                                evaluate without the card evidence
  python3 tools/train_model.py --views [--save]             saved model: calibrate the "views" parameter set
                                                            (screenshots / live play) on the CALIB rounds rendered
                                                            into the live grid (tools/live_features.py); report
                                                            CALIB / TEST live, "pano" vs "views" parameters
                                                            (-> data/model/eval_views.json)
  python3 tools/train_model.py --views --eval-only          report only, saved parameter sets
  python3 tools/train_model.py --views --calib-only         no TEST report (model selection)
  python3 tools/train_model.py --views --cv                 2-fold cross-validation inside CALIB (folds by game):
                                                            both sets recalibrated on one half, scored on the
                                                            other (-> data/model/eval_views_cv.json with --save)
  python3 tools/train_model.py --refit-glm 3e-4,1200 [--calib-only] [--save]
                                                            only the GLM refitted (ridge / iterations) + the
                                                            calibrations that depend on the posterior
  python3 tools/train_model.py --calibrate-maps --calib-only --save
                                                            map-aware prior recalibrated on CALIB (e.g. after
                                                            tools/build_priors.py), no TEST report
  python3 tools/train_model.py --world-points world_calib|world_test [--old-model DIR] [--workers 2] [--save]
                                                            the locator re-run on the saved live captures of the
                                                            World-map games (tools/world_play.py): No Move (start
                                                            capture) and Moving (fused) answers, points, evidence
                                                            -> scratch/world_eval/<split>.npz
  python3 tools/train_model.py --world --evidence scratch/world_eval [--world-test] [--save]
                                                            official World map: prior mixture of its per-map
                                                            counts on the world_calib answers (the current model's
                                                            evidence from --world-points) -> priors.json params
                                                            "maps"[World id]
  python3 tools/train_model.py --views --eval-only --live-kind frame [--save]
                                                            the saved sets on single NMPZ frames
                                                            (tools/live_features.py --kind frame;
                                                            -> data/model/eval_views_frame.json)

Order after new data: train_model.py --save (refit + calibration), tools/build_priors.py when the
rounds behind the counts change, then train_model.py --calibrate-maps --save.  The official World map after new
world_calib games (tools/world_play.py; when no game is being played), DIR = the model files the games were played
with (model.json, model.npz, priors.json):
    python3 tools/build_priors.py
    python3 tools/train_model.py --calibrate-maps --calib-only --save
    python3 tools/train_model.py --world-points world_calib --old-model DIR
    python3 tools/train_model.py --world --evidence scratch/world_eval --save
(world_test only once with final settings: --world-points world_test --old-model DIR --save).  The "views" set
(model.json "param_sets", data/model/regions_params.json) records what it was calibrated with (fitted
arrays, priors.json counts, regions.npz, clue_detectors.npz) and is ignored once any of them changes
(screenshots then use the panorama set): after a refit (--save), tools/build_priors.py, a regions.npz refit
(tools/eval_regions.py --save) or a detector rebuild (tools/build_clue_detectors.py) run
tools/live_features.py (incremental) and train_model.py --views --save.  A refit with a current live cache
recalibrates it by itself; --calibrate-maps / --calibrate-cards change only the panorama set.

train  : world + balanced Street View samples (feature cache from tools/calibrate.py)
calib  : half of the user's real GeoGuessr World-map rounds -> evidence weights, prior mix,
         map-aware prior mix (data/model/priors.json from tools/build_priors.py)
test   : the other half -> reported accuracy and mean GeoGuessr points per round (each round
         scored with its map's maxErrorDistance, data/maps.json)
"""
import argparse
import copy
import hashlib
import itertools
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
    old_maps = (model.priors.get("params") or {}).get("maps")
    if old_maps:  # per-map calibrations (the official World map, --world) are kept as a record; values whose
        # override_stamp no longer matches (new shared parameters / model fit) are ignored by GeoModel.map_override
        params["maps"] = old_maps
        stale = sorted(k for k, o in old_maps.items() if any(x in o for x in ("mix", "prior_mix", "evidence_scale"))
                       and {"fit": o.get("fit"), "shared": o.get("shared")} != model.override_stamp(params))
        print("kept the per-map parameters of %s%s - rerun tools/train_model.py --world-points world_calib and "
              "--world --evidence scratch/world_eval --save" % (", ".join(sorted(old_maps)), (
                  " (now inactive, calibrated with other shared parameters: %s)" % ", ".join(stale)) if stale else ""))
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


def calibrate_location(model, ev, recs, rows, regions=None):
    """Grid search of the within-country kernel: location (mean points) and region
    (mean log P(true region)) parameters on the CALIB rounds.  regions: (RegionModel, {group: (n, d)}
    features of the rows, region parameters) - the location kernel spreads the region-model posterior as
    the locator does (engine.regions.location_weights) instead of the country posterior alone."""
    lp = model.combine(ev)
    true_reg = region_index([recs[i]["lat"] for i in rows], [recs[i]["lng"] for i in rows])
    per = None
    if regions is not None and regions[0] is not None:
        from engine.regions import location_weights, region_posterior
        rm, X, rp = regions
        per = [region_posterior({g: v[k:k + 1] for g, v in X.items()}, np.exp(lp[k]), rm, model.classes,
                                by_country=True, params=rp) for k in range(len(rows))]
    best_loc, best_reg = (-1, None), (-1e9, None)
    for bw in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        for floor in (1e-3, 0.03, 0.1, 0.3):
            pts, rl = [], []
            for k, i in enumerate(rows):
                w = model.ref_weights(lp[k], ev["_d2"][k], bw, floor)
                if per is not None:
                    wl = location_weights(model, lp[k], ev["_d2"][k], per[k], bw=bw, floor=floor)
                    g = model.locate(lp[k], ev["_d2"][k], w=wl)
                else:
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


# ----------------------------------------------------------------------------- "views" parameter set
REGION_CALIBRATED = ("weights", "beta", "alpha", "knn_s", "smooth", "smooth_km", "smooth_card")


def live_inputs(model, recs, rows, cards=True, kind="grid"):
    """Live features of rows (tools/live_features.py, kind "grid" or "frame") -> (evidence, {group: (n, d)});
    rows missing from the live cache raise ValueError, as does a missing / stale card cache (grid) while
    the detector bank exists and cards are asked for.  Frames get no card evidence (as in the locator:
    a single frame shows < 90% of the searched windows)."""
    from live_features import cards_status, live_cards, live_matrix
    X = {}
    for g in model.groups:
        X[g], have = live_matrix(g, [recs[i] for i in rows], kind)
        if not have.all():
            raise ValueError("%d rounds missing from the live cache of %s - run tools/live_features.py --kind %s"
                             % (int((~have).sum()), g, kind))
    ev = model.evidence(X)
    if cards and kind == "grid":
        from engine.clue_detect import ClueDetectors
        bank = ClueDetectors.load()
        if bank is not None:
            why = cards_status(bank, kind)
            if why:
                raise ValueError("%s - run tools/live_features.py (or --no-cards)" % why)
            ev["cards"] = live_cards([recs[i]["pano_id"] for i in rows], model.classes, bank, kind)
    return ev, X


def region_frame(X, recs, rows):
    """tools/eval_regions.py data layout of the rows (live features) for its collect / score / calibrate."""
    from engine.regions import sun_mixture
    return {"X": X, "label": np.array([recs[i]["label"] for i in rows]),
            "region": region_index([recs[i]["lat"] for i in rows], [recs[i]["lng"] for i in rows]),
            "sun": sun_mixture(X) if "solar" in X else None}


def region_given_country(rmodel, D, params):
    """P(true region | true country) metrics of the region model with params (tools/eval_regions.score)."""
    from eval_regions import collect, score, summary
    idx = np.arange(len(D["label"]))
    L, R = score(rmodel, collect(rmodel, D, idx), params, idx)
    return summary(L, R), L, R


def calibrate_views(model, ev, X, recs, rows, rmodel=None):
    """The "views" parameter set on the CALIB rounds rendered into the live grid, with the procedures of the
    panorama set: evidence exponents + prior mix (GeoModel.calibrate, from the panorama set and from the
    fit defaults, the better CALIB log-likelihood kept), region-model exponents / prior / smoothing
    (tools/eval_regions.calibrate), location / region kernels (calibrate_location, location through the
    region model as in the locator), map-aware prior (calibrate_maps, out-of-fold per-map counts).
    Without card evidence in ev the set's card exponent is 0 (not tuned, so not used).  The set records the
    inputs it was calibrated with (GeoModel.input_hashes).  The model's own ("pano") parameters are not
    changed.  Sets model.param_sets["views"]; returns (the set, region-model params or None)."""
    cls = {c: i for i, c in enumerate(model.classes)}
    yi = np.array([cls.get(recs[i]["label"], -1) for i in rows])
    ok = yi >= 0
    ev_ok = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in ev.items()}
    vm = copy.copy(model)
    vm.priors = dict(model.priors)
    vm.param_set = "views"
    cards = "cards" in ev

    def start(w):
        return dict(w) if cards else dict(w, cards=0.0)

    def ll_of(w, pm):
        lp = vm.combine(ev_ok, w, pm)
        return float(np.mean(lp[np.arange(ok.sum()), yi[ok]]))

    ll_pano = ll_of(model.weights, model.prior_mix)
    defaults = dict({g: 0.3 for g in model.groups}, knn=0.5, glm=0.7, sun=0.0)
    best = None
    for name, w, pm in (("pano set", model.weights, model.prior_mix), ("fit defaults", defaults, 0.5)):
        vm.weights, vm.prior_mix = start(w), pm
        ll = vm.calibrate(ev_ok, yi[ok])
        print("views exponents from the %s: calib loglik %.4f %s prior_mix %s" % (
            name, ll, json.dumps({k: round(v, 3) for k, v in vm.weights.items()}), vm.prior_mix))
        if best is None or ll > best[0]:
            best = (ll, dict(vm.weights), vm.prior_mix, name)
    vm.weights, vm.prior_mix = best[1], best[2]
    print("views exponents: calib loglik %.4f (panorama parameters on the live features %.4f)%s" % (
        best[0], ll_pano, "" if cards else "; no card evidence - card exponent 0"))
    rp = None
    if rmodel is not None:
        from eval_regions import calibrate as calibrate_regions, collect
        D = region_frame(X, recs, rows)
        idx = np.arange(len(rows))
        rp, rbest, rstart = calibrate_regions(rmodel, [(rmodel, collect(rmodel, D, idx), idx)])
        print("views region model: calib mean log P(true region | true country) %.4f (panorama parameters %.4f)"
              % (rbest, rstart))
        print("    params", json.dumps({k: rp[k] for k in REGION_CALIBRATED}))
    calibrate_location(vm, ev, recs, rows, regions=(rmodel, X, rp) if rmodel is not None else None)
    vm.priors.pop("params", None)
    params = calibrate_maps(vm, ev, recs, rows)
    keys = ["model_fit", "priors_counts"] + (["regions_npz"] if rmodel is not None else []) + (
        ["clue_detectors_npz"] if cards else [])
    from live_features import render_hash
    ps = {"model": model.fingerprint(), "weights": vm.weights, "prior_mix": vm.prior_mix,
          "loc_params": vm.loc_params, "region_params": vm.region_params, "inputs": vm.input_hashes(keys),
          "calib": {"rounds": int(ok.sum()), "loglik": round(best[0], 4), "loglik_pano_params": round(ll_pano, 4),
                    "start": best[3], "features": "live grid renderings (tools/live_features.py, render %s)"
                                                  % render_hash("grid"), "cards": cards}}
    if params:
        ps["map"] = params
    old = (model.param_sets or {}).get("views") or {}
    for k in ("enabled", "disabled_reason"):   # a deliberate switch-off survives recalibration
        if k in old:
            ps[k] = old[k]
    model.param_sets = dict(model.param_sets, views=ps)
    return ps, rp


def calibrate_pano(model0, ev, X, recs, rows, rmodel=None):
    """The "pano" set recalibrated on rows of full-panorama features with the procedures behind the saved one
    (refit path of main / tools/eval_regions.py): GeoModel.calibrate from the fit defaults (exponents incl.
    cards, prior mix), calibrate_location (country kernel), calibrate_maps (out-of-fold per-map counts of
    model0.priors), region-model exponents / prior / smoothing (eval_regions.calibrate from the defaults).
    Returns (a model sharing model0's fitted arrays with that set, region-model params or None)."""
    from engine.regions import DEFAULT_PARAMS
    m = copy.copy(model0)
    m.priors, m.param_sets, m.param_set = dict(model0.priors), {}, "pano"
    m.weights, m.prior_mix = dict({g: 0.3 for g in m.groups}, knn=0.5, glm=0.7, sun=0.0), 0.5
    cls = {c: i for i, c in enumerate(m.classes)}
    yi = np.array([cls.get(recs[i]["label"], -1) for i in rows])
    ok = yi >= 0
    ll = m.calibrate({k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in ev.items()}, yi[ok])
    print("pano exponents: calib loglik %.4f %s prior_mix %s" % (
        ll, json.dumps({k: round(v, 3) for k, v in m.weights.items()}), m.prior_mix))
    calibrate_location(m, ev, recs, rows)
    m.priors.pop("params", None)
    calibrate_maps(m, ev, recs, rows)
    rp = None
    if rmodel is not None:
        from eval_regions import calibrate as calibrate_regions, collect
        rm = copy.copy(rmodel)
        rm.params = dict(rmodel.params, **{k: copy.deepcopy(DEFAULT_PARAMS[k]) for k in REGION_CALIBRATED})
        idx = np.arange(len(rows))
        rp, rbest, rstart = calibrate_regions(rm, [(rm, collect(rm, region_frame(X, recs, rows), idx), idx)])
        print("pano region model: calib mean log P(true region | true country) %.4f (defaults %.4f)" % (rbest, rstart))
    return m, rp


def evaluate_pipeline(model, ev, X, recs, rows, label, use_map=False, rmodel=None, rparams=None, verbose=True,
                      use=("map", "ranked")):
    """Locator.analyze on cached features (country posterior with the map setup, region-model posterior,
    guess on the region-weighted reference mass): the metrics of tools/eval_live.py - country top-k, points
    (World formula and each round's map formula), median km, region top-1/3 among the top country's regions
    when the country is right - plus the mean log P(true country); "per_round" arrays for paired tests
    (with the guesses "lat", "lng").  use: the empirical priors allowed with map info (GeoModel.map_components)."""
    from engine.geo import _regions
    from engine.regions import location_weights, region_posterior
    _, codes, _, rcc, _ = _regions()
    cpos = {c: j for j, c in enumerate(codes)}
    maps = round_maps(recs, rows)
    cache, setups = {}, []
    for mp in maps:
        key = (mp or {}).get("id") or (mp or {}).get("name")
        if key not in cache:
            cache[key] = model.map_setup(mp if use_map else None, use=use)
        setups.append(cache[key])
    lp_all = map_posterior(model, ev, setups)
    cls = {c: i for i, c in enumerate(model.classes)}
    true_reg = region_index([recs[i]["lat"] for i in rows], [recs[i]["lng"] for i in rows])
    out = {"rank": [], "true_p": [], "km": [], "points": [], "points_map": [], "region_rank": [], "lat": [],
           "lng": []}
    for k, i in enumerate(rows):
        s, lp, d2 = setups[k], lp_all[k], ev["_d2"][k]
        post = np.exp(lp)
        yi = cls.get(recs[i]["label"], -1)
        tp = float(post[yi]) if yi >= 0 else 0.0
        out["true_p"].append(tp)
        out["rank"].append(int((post > tp).sum()) if yi >= 0 else 999)
        where = {"score_scale_km": s["scale_km"], "bounds": s["bounds"]}
        top_cc = model.classes[int(np.argmax(post))]
        if rmodel is not None:
            mix, per = region_posterior({g: v[k:k + 1] for g, v in X.items()}, post, rmodel, model.classes,
                                        by_country="both", bounds=s["bounds"], params=rparams)
            g = model.locate(lp, d2, w=location_weights(model, lp, d2, per, bounds=s["bounds"]), **where)
            reg = sorted(((round(p, 5), c) for c, p in mix.items() if p > 0 and rcc[cpos[c]] == top_cc),
                         key=lambda t: -t[0])
            top_regions = [c for _, c in reg]
        else:
            g = model.locate(lp, d2, **where)
            pr = model.region_posterior(lp, d2, bounds=s["bounds"])
            top_regions = [codes[j] for j in np.argsort(-pr) if pr[j] > 0 and rcc[j] == top_cc]
        t = true_reg[k]
        out["region_rank"].append(top_regions[:3].index(codes[t]) if t and codes[t] in top_regions[:3] else 99)
        d = float(haversine_km(recs[i]["lat"], recs[i]["lng"], g["lat"], g["lng"]))
        out["lat"].append(g["lat"])
        out["lng"].append(g["lng"])
        out["km"].append(d)
        out["points"].append(float(geoguessr_score(d)))
        out["points_map"].append(float(geoguessr_points(d, (maps[k] or {}).get("maxErrorDistance"))))
    r = {k: np.asarray(v) for k, v in out.items()}
    ok = r["rank"] == 0
    res = {"n": len(rows), "top1": float((r["rank"] == 0).mean()), "top3": float((r["rank"] < 3).mean()),
           "top5": float((r["rank"] < 5).mean()), "mean_score": float(r["points"].mean()),
           "mean_points_map_formula": float(r["points_map"].mean()), "median_km": float(np.median(r["km"])),
           "loglik": float(np.mean(np.log(np.maximum(r["true_p"], 1e-9)))),
           "region_top1_if_country_right": float((r["region_rank"][ok] == 0).mean()) if ok.any() else 0.0,
           "region_top3_if_country_right": float((r["region_rank"][ok] < 3).mean()) if ok.any() else 0.0,
           "with_map": bool(use_map), "params": model.param_set}
    if verbose:
        print("[%s] n=%d top1=%.3f top3=%.3f top5=%.3f points/round World formula %.0f, map formula %.0f, median %.0f "
              "km, loglik %.3f | region (country right) top1=%.3f top3=%.3f" % (
                  label, res["n"], res["top1"], res["top3"], res["top5"], res["mean_score"],
                  res["mean_points_map_formula"], res["median_km"], res["loglik"],
                  res["region_top1_if_country_right"], res["region_top3_if_country_right"]), flush=True)
    res["per_round"] = r
    return res


def paired_diff(a, b, clusters=None, n_boot=4000, seed=0):
    """Mean of a - b with 95% bootstrap intervals: "ci95" resamples the clusters (games: the ~5 rounds of a
    game share map, player and day) when given, else the rounds; "ci95_rounds" always resamples rounds."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rs = np.random.RandomState(seed)
    m = d[rs.randint(0, len(d), (n_boot, len(d)))].mean(1)
    out = {"diff": round(float(d.mean()), 4), "ci95_rounds": [round(float(np.percentile(m, 2.5)), 4),
                                                              round(float(np.percentile(m, 97.5)), 4)]}
    if clusters is None:
        out["ci95"] = out["ci95_rounds"]
        return out
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    S, N = np.bincount(inv, weights=d), np.bincount(inv).astype(float)
    idx = np.random.RandomState(seed).randint(0, len(S), (n_boot, len(S)))
    m = S[idx].sum(1) / N[idx].sum(1)
    out["ci95"] = [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]
    out["clusters"] = int(len(S))
    return out


def games_of(recs, rows):
    return np.array([str(recs[i].get("game")) for i in rows])


def paired_metrics(pa, pb, games):
    """Paired differences (a - b) of per_round arrays: points (World / map formula), top1, top3, loglik."""
    return {"points": paired_diff(pa["points"], pb["points"], games),
            "points_map_formula": paired_diff(pa["points_map"], pb["points_map"], games),
            "top1": paired_diff(pa["rank"] == 0, pb["rank"] == 0, games),
            "top3": paired_diff(pa["rank"] < 3, pb["rank"] < 3, games),
            "loglik": paired_diff(np.log(np.maximum(pa["true_p"], 1e-9)), np.log(np.maximum(pb["true_p"], 1e-9)),
                                  games)}


def print_paired(name, d):
    print("  %-28s points %+.0f [%+.0f, %+.0f] (rounds [%+.0f, %+.0f]), top1 %+.3f [%+.3f, %+.3f], loglik %+.3f "
          "[%+.3f, %+.3f]  (95%% CI: %d games)" % (
              name, d["points"]["diff"], d["points"]["ci95"][0], d["points"]["ci95"][1], d["points"]["ci95_rounds"][0],
              d["points"]["ci95_rounds"][1], d["top1"]["diff"], d["top1"]["ci95"][0], d["top1"]["ci95"][1],
              d["loglik"]["diff"], d["loglik"]["ci95"][0], d["loglik"]["ci95"][1], d["points"].get("clusters", 0)))


PAIRS = (("views_no_map", "pano_no_map"), ("views_map", "pano_map"), ("pano_map", "pano_no_map"),
         ("views_map", "views_no_map"))


def views_report(model, rmodel, inputs, recs, splits, kind="grid"):
    """CALIB / TEST live features: "pano" vs "views" parameters, without / with map info (pipeline metrics),
    paired differences (game-level bootstrap), per map group, map-prior ablations, region metrics given the
    true country."""
    vm = model.for_input("views")
    rp_v = rmodel.params_for("views") if rmodel is not None else None
    rp_p = rmodel.params if rmodel is not None else None
    out = {"note": "%s of the user's rounds (tools/live_features.py --kind %s: eval_live.py capture, seeded start "
                   "yaw); CALIB is in-sample for BOTH parameter sets (pano: calibrated on the CALIB full panoramas, "
                   "views: on the CALIB live grid; per-map counts include CALIB rounds) - out-of-sample CALIB "
                   "comparisons: train_model.py --views --cv (eval_views_cv.json); TEST never used for fitting or "
                   "calibration.  ci95: bootstrap over games (clusters), ci95_rounds: over rounds"
                   % ({"grid": "live grid renderings", "frame": "single NMPZ frames"}[kind], kind),
           "kind": kind,
           "views_set": {k: v for k, v in (model.views_params() or {}).items() if k != "map"},
           "views_map": (model.views_params() or {}).get("map"),
           "views_regions": {k: rp_v[k] for k in REGION_CALIBRATED} if rp_v is not None and rp_v is not rp_p
           else None}
    for split in splits:
        rows = inputs[split]["rows"]
        ev, X = inputs[split]["ev"], inputs[split]["X"]
        games = games_of(recs, rows)
        res = {}
        for name, m, rp in (("pano", model, rp_p), ("views", vm, rp_v)):
            for use_map in (False, True):
                key = "%s_%s" % (name, "map" if use_map else "no_map")
                res[key] = evaluate_pipeline(m, ev, X, recs, rows, "%s %s, %s params, %s" % (
                    split, kind, name, "map info" if use_map else "no map"), use_map, rmodel, rp)
        pr = {k: v.pop("per_round") for k, v in res.items()}
        res["games"] = int(len(set(games)))
        diffs = {}
        for a, b in PAIRS:
            diffs["%s - %s" % (a, b)] = paired_metrics(pr[a], pr[b], games)
            print_paired(a + " - " + b, diffs["%s - %s" % (a, b)])
        res["paired"] = diffs
        # where map info gains / loses: per map group (own per-map counts / ranked-duel prior / model prior) and
        # the map-prior ablations of map_report (scale + bounds only, the ranked prior for every map)
        groups = np.array(map_groups(model, recs, rows))
        by = {}
        for gname in dict.fromkeys(groups.tolist()):
            sel = groups == gname
            by[gname] = {"n": int(sel.sum()), "games": int(len(set(games[sel])))}
            for name in ("pano", "views"):
                a, b = pr[name + "_map"], pr[name + "_no_map"]
                by[gname][name] = {"points_no_map": round(float(b["points_map"][sel].mean()), 1),
                                   "map_minus_no_map": paired_diff(a["points_map"][sel], b["points_map"][sel],
                                                                   games[sel]),
                                   "top1_map_minus_no_map": paired_diff(a["rank"][sel] == 0, b["rank"][sel] == 0,
                                                                        games[sel])}
            print("  %-34s n=%3d (%2d games) map - no map, points (map formula): pano %+.0f %s, views %+.0f %s" % (
                gname, sel.sum(), by[gname]["games"], by[gname]["pano"]["map_minus_no_map"]["diff"],
                by[gname]["pano"]["map_minus_no_map"]["ci95"], by[gname]["views"]["map_minus_no_map"]["diff"],
                by[gname]["views"]["map_minus_no_map"]["ci95"]))
        res["by_map_group"] = by
        abl = {}
        for name, m, rp in (("pano", model, rp_p), ("views", vm, rp_v)):
            for use, tag in (((), "scale_bounds_only"), (("ranked",), "ranked_for_all")):
                r = evaluate_pipeline(m, ev, X, recs, rows, "%s %s, %s params, map %s" % (split, kind, name, tag),
                                      True, rmodel, rp, use=use)
                prr = r.pop("per_round")
                r["points_minus_no_map"] = paired_diff(prr["points_map"], pr[name + "_no_map"]["points_map"], games)
                abl["%s_%s" % (name, tag)] = r
        res["map_ablation"] = abl
        if rmodel is not None:
            D = region_frame(X, recs, rows)
            rg = {}
            for name, rp in (("pano", rp_p), ("views", rp_v)):
                rg[name] = region_given_country(rmodel, D, rp)[0]
                print("  region given the true country, %s params: top1 %.3f top3 %.3f top5 %.3f logp %.3f" % (
                    name, rg[name]["top1"], rg[name]["top3"], rg[name]["top5"], rg[name]["logp"]))
            res["region_given_true_country"] = rg
        if split == "calib":
            res["in_sample"] = True
        out[split] = res
    return out


def save_param_sets(model, expect, path=MODEL_DIR):
    """Write model.param_sets (and fit_hash) into path/model.json without touching the rest; refuses (returns
    the reason) when the fitted model or an input of the set changed on disk since expect
    (GeoModel.input_hashes taken when the model was loaded)."""
    now = GeoModel.load(path).input_hashes(list(expect))
    bad = [k for k in expect if now.get(k) != expect[k]]
    if bad:
        return "changed on disk since the model was loaded: %s" % ", ".join(bad)
    mj = os.path.join(path, "model.json")
    meta = json.load(open(mj))  # only the parameter sets change, not the fitted arrays
    meta["param_sets"] = model.param_sets
    meta["fit_hash"] = model.fit_hash()
    tmp = mj + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, mj)
    return None


def cmd_views(args, recs, ca, te, mats=None):
    """--views: calibrate (unless --eval-only) and report the "views" parameter set on the live features;
    --cv: 2-fold cross-validation inside CALIB instead."""
    from engine.regions import load_region_model
    model = GeoModel.load()
    loaded = model.input_hashes()  # what is on disk now: --save refuses if it changes meanwhile
    rmodel = load_region_model()
    kind = args.live_kind
    if kind != "grid" and not args.eval_only:
        raise SystemExit("the views set is calibrated on the live grid; --live-kind %s needs --eval-only" % kind)
    inputs = {}
    for split, rows in (("calib", ca), ("test", te)):
        if split == "test" and (args.calib_only or args.cv):
            continue
        rows = np.array([i for i in rows if recs[i].get("heading") is not None])
        ev, X = live_inputs(model, recs, rows, cards=not args.no_cards, kind=kind)
        inputs[split] = {"rows": rows, "ev": ev, "X": X}
    if args.cv:
        frames = None
        try:  # single frames of the same rounds (tools/live_features.py --kind frame): also scored when cached
            rows = inputs["calib"]["rows"]
            frames = dict(zip(("ev", "X"), live_inputs(model, recs, rows, kind="frame")))
        except (OSError, ValueError) as e:
            print("no frame features in the CV (%s)" % e)
        out = views_cv(model, rmodel, inputs["calib"], recs, mats, cards=not args.no_cards, frames=frames)
        if args.save:
            json.dump(out, open(os.path.join(MODEL_DIR, "eval_views_cv.json"), "w"), indent=1)
            print("saved data/model/eval_views_cv.json")
        return out
    if not args.eval_only:
        c = inputs["calib"]
        ps, rp = calibrate_views(model, c["ev"], c["X"], recs, c["rows"], rmodel)
        if rp is not None:
            rmodel.param_sets = dict(rmodel.param_sets, views=rp)
    out = views_report(model, rmodel, inputs, recs, list(inputs), kind)
    if args.save:
        if not args.eval_only:
            from engine.clue_detect import ClueDetectors
            if not ps["calib"]["cards"] and ClueDetectors.load() is not None and not args.no_cards:
                raise SystemExit("not saved: no live card evidence although the detector bank exists")
            why = save_param_sets(model, loaded)
            if why:
                raise SystemExit("not saved (%s) - rerun tools/train_model.py --views --save" % why)
            save_region_views(rmodel, len(inputs["calib"]["rows"]))
            print("saved data/model/model.json param_sets (+ regions_params.json when it differs)")
        if not args.calib_only:  # the report file holds CALIB and TEST
            name = "eval_views.json" if kind == "grid" else "eval_views_%s.json" % kind
            json.dump(out, open(os.path.join(MODEL_DIR, name), "w"), indent=1)
            print("saved data/model/%s" % name)
    return out


def views_cv(model0, rmodel, calib, recs, mats, cards=True, frames=None):
    """2-fold cross-validation inside CALIB (halves of the 5 game folds of build_priors.fold_of): on half A both
    sets are recalibrated - "pano" on the full-panorama features (calibrate_pano), "views" on the live grid
    (calibrate_views, started from that half's pano set) - with the per-map counts of half B removed; both are
    scored on the live grid of half B (and on its single frames when frames = {"ev", "X"} of the same rows
    is given: "frame_*" arms).  TEST is not touched."""
    rows, ev_all, X_all = calib["rows"], calib["ev"], calib["X"]
    k5 = int(model0.priors.get("folds", 5))
    f5 = np.array([fold_of(recs[i]["game"], k5) for i in rows])
    halves = [(0, 2, 4), (1, 3)]
    keys = [a for a, _ in PAIRS[:2]] + [b for _, b in PAIRS[:2]]
    if frames is not None:
        keys += ["frame_" + k for k in keys]
    per = {k: {} for k in keys}
    sub = lambda ev, m: {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in ev.items()}  # noqa: E731
    model0 = copy.copy(model0)
    model0.param_sets = {}
    params = []
    for h, held in enumerate(halves):
        B = np.isin(f5, held)
        A = ~B
        pri = copy.deepcopy(model0.priors)
        for e in (pri.get("maps") or {}).values():  # per-map counts without the games of half B
            drop = {}
            for f in held:
                for c, n in e["fold_counts"][f].items():
                    drop[c] = drop.get(c, 0) + n
                e["fold_counts"][f] = {}
            e["counts"] = {c: n - drop.get(c, 0) for c, n in e["counts"].items() if n - drop.get(c, 0) > 0}
        base = copy.copy(model0)
        base.priors = pri
        Xp = {g: mats[g][rows[A]] for g in model0.groups}
        evp = model0.evidence(Xp)
        if cards:
            add_cards(model0, evp, recs, rows[A])
        print("== half %d: calibrate on %d rounds (%d games), score %d rounds (%d games)" % (
            h, A.sum(), len(set(games_of(recs, rows[A]))), B.sum(), len(set(games_of(recs, rows[B])))), flush=True)
        mp, rp_p = calibrate_pano(base, evp, Xp, recs, rows[A], rmodel)
        rm = rmodel
        if rmodel is not None:  # the views region parameters start from this half's pano parameters
            rm = copy.copy(rmodel)
            rm.params = rp_p
        _, rp_v = calibrate_views(mp, sub(ev_all, A), {g: v[A] for g, v in X_all.items()}, recs, rows[A], rm)
        mv = mp.for_input("views")
        params.append({"pano": {"weights": mp.weights, "prior_mix": mp.prior_mix, "loc_params": mp.loc_params,
                                "map": mp.priors.get("params"),
                                "regions": {k: rp_p[k] for k in REGION_CALIBRATED} if rp_p else None},
                       "views": dict({k: v for k, v in mp.param_sets["views"].items() if k != "inputs"},
                                     regions={k: rp_v[k] for k in REGION_CALIBRATED} if rp_v else None)})
        scored = [("", sub(ev_all, B), {g: v[B] for g, v in X_all.items()})]
        if frames is not None:
            scored.append(("frame_", sub(frames["ev"], B), {g: v[B] for g, v in frames["X"].items()}))
        for pre, evB, XB in scored:
            for name, m, rpar in (("pano", mp, rp_p), ("views", mv, rp_v)):
                for use_map in (False, True):
                    key = "%s%s_%s" % (pre, name, "map" if use_map else "no_map")
                    r = evaluate_pipeline(m, evB, XB, recs, rows[B], "half %d %s" % (h, key), use_map, rmodel, rpar)
                    for k, v in r.pop("per_round").items():
                        per[key].setdefault(k, np.zeros(len(rows), v.dtype))[B] = v
    games = games_of(recs, rows)
    out = {"note": "2-fold cross-validation inside CALIB live grid (folds by game): both parameter sets "
                   "recalibrated on one half (pano on its full-panorama features, views on its live grid, per-map "
                   "counts without the other half's games), scored on the other half; out of sample for both "
                   "arms.  ci95: bootstrap over games", "n": int(len(rows)), "games": int(len(set(games))),
           "params_by_half": params}
    print("\n== 2-fold CV on CALIB live grid (n=%d, %d games)" % (len(rows), len(set(games))))
    for k, v in per.items():
        out[k] = {"top1": float((v["rank"] == 0).mean()), "top3": float((v["rank"] < 3).mean()),
                  "top5": float((v["rank"] < 5).mean()), "mean_score": float(v["points"].mean()),
                  "mean_points_map_formula": float(v["points_map"].mean()),
                  "loglik": float(np.log(np.maximum(v["true_p"], 1e-9)).mean())}
        print("%-14s top1 %.3f top3 %.3f top5 %.3f points World %.0f map formula %.0f loglik %.3f" % (
            k, out[k]["top1"], out[k]["top3"], out[k]["top5"], out[k]["mean_score"], out[k]["mean_points_map_formula"],
            out[k]["loglik"]))
    out["paired"] = {}
    for pre in ("", "frame_") if frames is not None else ("",):
        for a, b in PAIRS:
            a, b = pre + a, pre + b
            out["paired"]["%s - %s" % (a, b)] = paired_metrics(per[a], per[b], games)
            print_paired(a + " - " + b, out["paired"]["%s - %s" % (a, b)])
    groups = np.array(map_groups(model0, recs, rows))
    out["by_map_group"] = {}
    for gname in dict.fromkeys(groups.tolist()):
        sel = groups == gname
        e = {"n": int(sel.sum()), "games": int(len(set(games[sel])))}
        for name in ("pano", "views"):
            e[name + "_map_minus_no_map"] = paired_diff(per[name + "_map"]["points_map"][sel],
                                                        per[name + "_no_map"]["points_map"][sel], games[sel])
        out["by_map_group"][gname] = e
        print("  %-34s n=%3d (%2d games) map - no map, points (map formula): pano %+.0f %s, views %+.0f %s" % (
            gname, e["n"], e["games"], e["pano_map_minus_no_map"]["diff"], e["pano_map_minus_no_map"]["ci95"],
            e["views_map_minus_no_map"]["diff"], e["views_map_minus_no_map"]["ci95"]))
    return out


def save_region_views(rmodel, n_calib, path=None):
    """regions_params.json: the region model's views parameters, tied to the regions.npz file (no file when
    the calibration kept the panorama parameters)."""
    from engine.regions import MODEL_DIR as RDIR, PARAMS_FILE, REGIONS_FILE, file_hash
    path = path or RDIR
    if rmodel is None or not rmodel.param_sets.get("views"):
        return
    f = os.path.join(path, PARAMS_FILE)
    if json.dumps(rmodel.param_sets["views"], sort_keys=True) == json.dumps(rmodel.params, sort_keys=True):
        print("region model: the views calibration kept the panorama parameters - no %s" % PARAMS_FILE)
        if os.path.exists(f):
            os.remove(f)
        return
    h = file_hash(os.path.join(path, REGIONS_FILE))
    json.dump({"views": {"params": rmodel.param_sets["views"], "regions_npz": h,
                         "calib_rounds": int(n_calib), "features": "live grid renderings (tools/live_features.py)"}},
              open(f, "w"), indent=1)


# ----------------------------------------------------------------------------- official World map (world_play games)
WORLD_GAMES = os.path.join(ROOT, "data", "calibration", "world_games.json")
WORLD_MAP_ID = "52d15ff6850ffb3b847cb3b4"
MIX_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
WORLD_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]
PM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def world_split_of(token):
    """world_calib / world_test of a World-map game without a stored "split": the parity rule of calibrate.split_of
    (mode world_play; = build_priors / tools/world_play.py split_of_game) - not fold_of, whose bit is independent of
    the parity."""
    return split_of({"mode": "world_play", "game": str(token), "pano_id": ""})


def world_rounds(path=WORLD_GAMES):
    """Rounds of the finished World-map calibration games (tools/world_play.py, data/calibration/world_games.json):
    [{game, split, walk, round, cc, lat, lng, points, dirs: capture directories oldest first (the start capture,
    then the moves), exclude: the game is flagged (its live answers came from an engine under edit)}]."""
    try:
        with open(path) as f:
            games = json.load(f).get("games", [])
    except (OSError, ValueError):
        return []
    out = []
    for g in games:
        if g.get("status") != "finished" or not g.get("token"):
            continue
        split = g.get("split") or world_split_of(g["token"])
        for r in g.get("rounds") or []:
            cc = (r.get("gg_country") or r.get("country") or "").upper()
            dirs = [d if os.path.isabs(d) else os.path.join(ROOT, d) for d in (r.get("dirs") or [])]
            out.append({"game": g["token"], "split": split, "walk": int(g.get("walk") or 0), "round": r.get("round"),
                        "cc": {"UK": "GB"}.get(cc, cc), "lat": r.get("lat"), "lng": r.get("lng"),
                        "points": r.get("points"), "dirs": [d for d in dirs if os.path.exists(os.path.join(d, "response.json"))],
                        "exclude": bool(g.get("exclude"))})
    return out


def response_evidence(path, classes):
    """(evidence relative to the prior the capture was analysed with: log posterior - log prior, (C,); that prior)
    of a saved live response (posterior / prior are stored to 4 significant digits)."""
    with open(os.path.join(path, "response.json")) as f:
        r = json.load(f)
    post, prior = r.get("posterior") or {}, r.get("prior") or {}
    p = np.array([float(post.get(c, 0.0)) for c in classes])
    q = np.array([float(prior.get(c, 0.0)) for c in classes])
    return np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(q, 1e-12)), q


def world_answers(model, rounds):
    """Saved live answers of the World-map rounds: y (class index, -1 unknown), E_start (the start capture: the
    No Move answer), E_final (the last capture: the fused Moving answer; = E_start without moves), P_old (the prior
    the game was played with) - rounds without a saved start response are dropped."""
    cls = {c: i for i, c in enumerate(model.classes)}
    keep, ys, es, ef, po = [], [], [], [], []
    for r in rounds:
        if not r["dirs"]:
            continue
        e0, q0 = response_evidence(r["dirs"][0], model.classes)
        e1, _ = response_evidence(r["dirs"][-1], model.classes)
        keep.append(r)
        ys.append(cls.get(r["cc"], -1))
        es.append(e0)
        ef.append(e1)
        po.append(q0)
    return keep, np.array(ys, int), np.array(es).reshape(-1, len(model.classes)), \
        np.array(ef).reshape(-1, len(model.classes)), np.array(po).reshape(-1, len(model.classes))


def world_priors(model, rounds, mix, params=None, oof=True, prior_mix=None):
    """(n, C) priors of the official World map for the rounds: its per-map counts (build_priors.py: the user's CALIB
    rounds on "World" + the world_calib games) without the round's own game fold when oof (the counted half),
    mixed with mix {"map", "map_ranked"} and the model prior of prior_mix (shared params otherwise)."""
    params = params or model.map_params() or {}
    world = resolve_map({"id": WORLD_MAP_ID})
    entry = model.map_entry(world)
    k = int(model.priors.get("folds", 5))
    cache, out = {}, []
    for r in rounds:
        f = fold_of(r["game"], k) if (oof and entry and r["split"] == "world_calib") else -1
        if f not in cache:
            counts = None
            if entry and f >= 0:
                fc = entry["fold_counts"][f]
                counts = {c: n - fc.get(c, 0) for c, n in entry["counts"].items()}
            kind, ranked, emp, wb = model.map_components(world, counts)
            pm = params.get("prior_mix", model.prior_mix) if prior_mix is None else prior_mix
            base = _norm_prior(model.base_prior(pm) * wb)
            if kind == "map":
                p = mix_prior("map", dict(params.get("mix") or {}, **mix), base, ranked, emp)
            elif kind == "ranked":
                p = mix_prior("ranked", params.get("mix") or {"ranked": 0.0}, base, ranked)
            else:
                p = base
            cache[f] = _norm_prior(p)
        out.append(cache[f])
    return np.array(out)


def _norm_prior(p):
    p = np.asarray(p, float)
    return p / p.sum(-1, keepdims=True)


def _lp_true(P, E, y, scale=1.0):
    lp = np.log(np.maximum(P, 1e-300)) + scale * E
    lp = lp - lp.max(1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    return lp[np.arange(len(y)), y], lp


def world_metrics(P, E, y, games, scale=1.0):
    tp, lp = _lp_true(P, E, y, scale)
    rank = (lp > tp[:, None]).sum(1)
    return {"n": int(len(y)), "games": int(len(set(games))), "top1": round(float((rank == 0).mean()), 4),
            "top3": round(float((rank < 3).mean()), 4), "top5": round(float((rank < 5).mean()), 4),
            "loglik": round(float(tp.mean()), 4)}, tp, rank


WORLD_CV_K, WORLD_CV_REPS = 5, 20


def world_families(shared):
    """Candidate World-map mixtures, nested families around the shared mix (shared = no own parameters): the
    model-prior blend only, the per-map weight only, both, per-map + ranked weights, all three."""
    g = [x for x in MIX_GRID if x <= 0.9 + 1e-9]

    def fam(keys):
        vals = {"map": g, "map_ranked": g, "prior_mix": PM_GRID}
        out = []
        for combo in itertools.product(*[vals[k] for k in keys]):
            c_ = dict(shared, **dict(zip(keys, combo)))
            if c_["map"] + c_["map_ranked"] <= 0.95 + 1e-9 and c_ not in out:
                out.append(c_)
        return out
    return [("shared", [dict(shared)]), ("prior_mix", fam(["prior_mix"])), ("map", fam(["map"])),
            ("map+prior_mix", fam(["map", "prior_mix"])), ("map+map_ranked", fam(["map", "map_ranked"])),
            ("full", fam(["map", "map_ranked", "prior_mix"]))]


def world_cv_folds(games, k=None, reps=None):
    """reps random assignments of the games to k folds (md5 of game + repetition: deterministic)."""
    k, reps = k or WORLD_CV_K, reps or WORLD_CV_REPS
    return [np.array([int(hashlib.md5(("%s|%d" % (g, rep)).encode()).hexdigest(), 16) % k for g in games])
            for rep in range(reps)]


def world_cv(L, folds):
    """Repeated k-fold CV of choosing the best row of L (candidates, answers) by the mean on the other folds: the
    out-of-fold value of every answer, averaged over the repetitions."""
    out = np.zeros(L.shape[1])
    for f in folds:
        cv_ = np.zeros(L.shape[1])
        for k in np.unique(f):
            tr = f != k
            cv_[~tr] = L[int(np.argmax(L[:, tr].mean(1))), ~tr]
        out += cv_
    return out / len(folds)


def cmd_world(args):
    """--world: the official World map's prior mixture on the world_calib games (answers of tools/world_play.py
    games: start capture = No Move answer, last capture = fused Moving answer; their evidence relative to the prior,
    from the saved live responses or, with --evidence, from --world-points re-analysis under the current model),
    out-of-fold per-map counts.  Nested families of World-specific parameters around the shared mixture (none, the
    model-prior blend, the per-map weight, ...; world_families) are compared by repeated 5-fold CV over the games; the
    best family whose CV gain over the shared mixture is clearly positive (lower 95% bound > 0, bootstrap over games)
    gives the override (its in-sample best mixture; none when no family qualifies), an evidence scale likewise.  World_test only with --world-test.  --save writes priors.json params
    "maps"[World id] (and data/model/eval_world.json)."""
    model = GeoModel.load()
    if args.priors:  # dry run on another priors.json (e.g. tools/build_priors.py --out)
        model.priors = json.load(open(args.priors))
    params = model.map_params()
    if not params:
        raise SystemExit("no calibrated map parameters (train_model.py --calibrate-maps --save first)")
    if not model.map_entry(resolve_map({"id": WORLD_MAP_ID})):
        print("WARNING: priors.json has no per-map counts for the official World map yet (tools/build_priors.py "
              "after world_calib games; >= --min-rounds) - the World map uses the ranked prior")
    allr = world_rounds()
    print("world_games.json: %d rounds of finished games (%s)" % (len(allr), dict(collections_counter(
        (r["split"], "moving" if r["walk"] else "no move", "excluded" if r["exclude"] else "ok") for r in allr))))
    out = {"note": "official World map (52d15ff6850ffb3b847cb3b4): live answers of the locator's own classic games "
                   "(tools/world_play.py), evidence = log posterior - log prior of the saved responses (start capture: "
                   "No Move answer; last capture: fused Moving answer); per-map counts out-of-fold on world_calib; "
                   "games flagged 'exclude' left out of the evidence (their counts are used)"}
    res = {}
    for split in ("world_calib",) + (("world_test",) if args.world_test else ()):
        rr = [r for r in allr if r["split"] == split and not r["exclude"]]
        rounds, y, Es, Ef, Pold = world_answers(model, rr)
        ev_file = os.path.join(args.evidence, "%s.npz" % split) if args.evidence else None
        if ev_file:  # evidence of the current model (cmd_world_points re-analysis) instead of the saved responses
            z = np.load(ev_file, allow_pickle=True)
            if list(z["classes"]) != list(model.classes):
                raise SystemExit("%s: other classes" % ev_file)
            pos = {(str(g), int(rd)): k for k, (g, rd) in enumerate(zip(z["game"], z["round"]))}
            sel = [i for i, r in enumerate(rounds) if (r["game"], int(r["round"])) in pos]
            idx = [pos[(rounds[i]["game"], int(rounds[i]["round"]))] for i in sel]
            print("%s: evidence of the current model from %s (%d of %d rounds)" % (split, ev_file, len(sel), len(rounds)))
            rounds = [rounds[i] for i in sel]
            y, Pold = y[sel], Pold[sel]
            Es, Ef = z["new_start_e"][idx].astype(np.float64), z["new_final_e"][idx].astype(np.float64)
            no_fused = ~np.isfinite(Ef).all(1)  # Moving rounds whose move images were pruned: start answer only
            if no_fused.any():
                print("%s: %d Moving rounds without a fused answer (pruned move-capture images): start answer only"
                      % (split, no_fused.sum()))
                rounds = [dict(r, no_fused=True) if nf else r for r, nf in zip(rounds, no_fused)]
                Ef = np.where(no_fused[:, None], Es, Ef)
        ok = y >= 0
        rounds = [r for r, o in zip(rounds, ok) if o]
        y, Es, Ef, Pold = y[ok], Es[ok], Ef[ok], Pold[ok]
        games = np.array([r["game"] for r in rounds])
        mov = np.array([r["walk"] > 0 and not r.get("no_fused") for r in rounds])  # rounds with a fused Moving answer
        res[split] = {"rounds": rounds, "y": y, "Es": Es, "Ef": Ef, "Pold": Pold, "games": games, "mov": mov,
                      "nomove": np.array([r["walk"] == 0 for r in rounds])}
        print("%s: %d rounds with saved answers (%d games; %d Moving rounds)" % (split, len(y), len(set(games)), mov.sum()))
    c = res["world_calib"]
    if not len(c["y"]):
        raise SystemExit("no world_calib answers yet")
    # observations: every start answer (No Move) and the fused answer of the Moving rounds
    E_obs = np.concatenate([c["Es"], c["Ef"][c["mov"]]])
    y_obs = np.concatenate([c["y"], c["y"][c["mov"]]])
    rr_obs = c["rounds"] + [r for r, mv in zip(c["rounds"], c["mov"]) if mv]

    shared = {"map": params["mix"].get("map", 0.0), "map_ranked": params["mix"].get("map_ranked", 0.0),
              "prior_mix": params.get("prior_mix", 0.0)}
    obs_games = np.array([r["game"] for r in rr_obs])
    scales = [1.0] + [s_ for s_ in WORLD_SCALES if s_ != 1.0]
    cache = {}

    def L_of(cand):  # (scales, answers) log P(true country), out-of-fold per-map counts
        k = tuple(sorted(cand.items()))
        if k not in cache:
            P = world_priors(model, rr_obs, {"map": cand["map"], "map_ranked": cand["map_ranked"]}, params,
                             prior_mix=cand["prior_mix"])
            cache[k] = np.array([_lp_true(P, E_obs, y_obs, sc)[0] for sc in scales])
        return cache[k]

    fams = world_families(shared)
    Lf = {name: np.array([L_of(c_) for c_ in cands]) for name, cands in fams}   # (cands, scales, answers)
    folds = world_cv_folds(obs_games)
    cv = {name: world_cv(L[:, 0], folds) for name, L in Lf.items()}
    L_shared = Lf["shared"][0, 0]
    # the World map without per-map counts (HEAD 89a558e: ranked kind) and the prior the games were played with
    world = resolve_map({"id": WORLD_MAP_ID})
    _, ranked, _, wb = model.map_components(world, use=("ranked",))
    base = _norm_prior(model.base_prior(params.get("prior_mix", model.prior_mix)) * wb)
    p_ranked = _norm_prior(mix_prior("ranked", params["mix"], base, ranked))
    L_ranked = _lp_true(np.tile(p_ranked, (len(y_obs), 1)), E_obs, y_obs)[0]
    played_tp = _lp_true(np.concatenate([c["Pold"], c["Pold"][c["mov"]]]), E_obs, y_obs)[0]
    flat = float(_lp_true(np.ones_like(E_obs), E_obs, y_obs)[0].mean())
    print("world_calib (%d answers, %d games): loglik flat prior %.4f | played with %.4f | ranked prior without per-map "
          "counts %.4f | shared mix over the per-map counts %.4f (out-of-fold counts)" % (
              len(y_obs), len(set(obs_games)), flat, played_tp.mean(), L_ranked.mean(), L_shared.mean()))
    d_counts = paired_diff(L_shared, L_ranked, obs_games)
    print("  per-map counts (shared mix) - ranked prior: %+.4f %s" % (d_counts["diff"], d_counts["ci95"]))
    cv_out = {}
    for name, _ in fams:
        d = paired_diff(cv[name], L_shared, obs_games)
        cv_out[name] = {"cv_loglik": round(float(cv[name].mean()), 4), "minus_shared": d}
        print("  family %-14s repeated %d-fold CV loglik %.4f, minus shared mix %+.4f %s" % (
            name, WORLD_CV_K, cv[name].mean(), d["diff"], d["ci95"]))
    # a World-specific family only when it beats the shared mixture clearly (lower 95% bound of the CV gain > 0):
    # ~20 games pick quite different mixtures fold to fold, so a mere CV mean edge is mostly noise
    ok_f = [n_ for n_ in cv if n_ == "shared" or cv_out[n_]["minus_shared"]["ci95"][0] > 0]
    family = max(ok_f, key=lambda n_: (float(cv[n_].mean()), n_ == "shared"))
    cands = dict(fams)[family]
    L = Lf[family]
    best_mix = cands[int(np.argmax(L[:, 0].mean(1)))]
    # evidence scale of the World map (with the chosen family): joint choice vs mixture only, kept only when its CV
    # gain is clearly positive
    cv_joint = world_cv(L.reshape(-1, L.shape[-1]), folds)
    d_scale = paired_diff(cv_joint, cv[family], obs_games)
    ij = int(np.argmax(L.reshape(-1, L.shape[-1]).mean(1)))
    use_scale = scales[ij % len(scales)] != 1.0 and d_scale["ci95"][0] > 0
    scale = scales[ij % len(scales)] if use_scale else 1.0
    if use_scale:
        best_mix = cands[ij // len(scales)]
    best = float(L_of(best_mix)[scales.index(scale)].mean())
    print("  -> family %s: mix %s, evidence scale %.2f (scale CV gain %+.4f %s), in-sample loglik %.4f" % (
        family, best_mix, scale, d_scale["diff"], d_scale["ci95"], best))
    # fit / shared: the override_stamp (model fit + shared map parameters) the values below belong to (map_override
    # ignores the values once either changes)
    over = {"family": family, "model": model.fingerprint(), **model.override_stamp(params),
            "calib_answers": int(len(y_obs)),
            "calib_rounds": int(len(c["y"])), "calib_games": int(len(set(c["games"]))),
            "calib_loglik": round(best, 4), "calib_loglik_shared": round(float(L_shared.mean()), 4),
            "calib_loglik_ranked_without_counts": round(float(L_ranked.mean()), 4),
            "calib_loglik_played": round(float(played_tp.mean()), 4), "counts_minus_ranked": d_counts,
            "cv": cv_out, "cv_scale_gain": d_scale,
            "counts_rounds": (model.map_entry(world) or {}).get("n"),
            "source": "tools/train_model.py --world (world_calib answers of tools/world_play.py games)"}
    if family != "shared":  # otherwise the shared mixture (and its later recalibrations) applies
        over.update(mix={"map": best_mix["map"], "map_ranked": best_mix["map_ranked"]}, prior_mix=best_mix["prior_mix"])
    if use_scale:
        over["evidence_scale"] = scale
    out["world_calib"] = over
    newp = dict(params, maps=dict(params.get("maps") or {}, **{WORLD_MAP_ID: over}))
    if args.world_test:
        t = res["world_test"]
        if len(t["y"]):
            t_out = {}
            Pnew = world_priors(model, t["rounds"], {"map": best_mix["map"], "map_ranked": best_mix["map_ranked"]},
                                params, oof=False, prior_mix=best_mix["prior_mix"])
            for name, E, sel in (("no_move_start", t["Es"], np.ones(len(t["y"]), bool)),
                                 ("no_move_games", t["Es"], t["nomove"]), ("moving_final", t["Ef"], t["mov"])):
                if not sel.any():
                    continue
                a, tpa, _ = world_metrics(Pnew[sel], E[sel], t["y"][sel], t["games"][sel], scale)
                b, tpb, _ = world_metrics(t["Pold"][sel], E[sel], t["y"][sel], t["games"][sel])
                t_out[name] = {"new_world_prior": a, "played_with": b,
                               "loglik_new_minus_played": paired_diff(tpa, tpb, t["games"][sel])}
                print("world_test %-14s new prior %s | played with %s | loglik %+.4f %s" % (
                    name, a, b, t_out[name]["loglik_new_minus_played"]["diff"],
                    t_out[name]["loglik_new_minus_played"]["ci95"]))
            out["world_test"] = t_out
    if args.save:
        model.priors["params"] = newp
        model.save_priors()
        print("saved priors.json params maps[%s] = %s" % (WORLD_MAP_ID, {k: over.get(k) for k in (
            "family", "mix", "prior_mix", "evidence_scale")}))
    if args.save:
        json.dump(out, open(os.path.join(MODEL_DIR, "eval_world.json"), "w"), indent=1)
    return out


def _points_job(job):
    """Worker of cmd_world_points: one round; the locator (working tree, features / card detections extracted once per
    capture, fusion.install_tap) re-run on its saved captures with every model variant; per variant the start answer
    (No Move) and the fused answer of all captures (Moving: engine.fusion.FusionStore, as the live server)."""
    r, mp, cls = job
    from engine import fusion
    from engine.geo import geoguessr_points as gp, haversine_km as hk
    from engine.panorama import SphericalImage
    from rebuild_views import load_capture
    loc = _WORKER["loc"]
    stores = {name: fusion.FusionStore(max_rounds=4) for name in _WORKER["models"]}
    start, final = {}, {}
    # tools/world_play.py prunes the move captures' view images of old games when its disk budget runs out (views.json
    # and response.json stay): such a round has only its start capture (No Move answer), no fused Moving answer
    complete = all(capture_has_images(d) for d in r["dirs"])
    for k, d in enumerate(r["dirs"] if complete else r["dirs"][:1]):
        _, views = load_capture(d)
        sph = SphericalImage.from_views(views, width=2048, heading=0.0)
        for name, m in _WORKER["models"].items():
            loc.model = m
            res = loc.analyze(sph, map_info=mp)
            if len(r["dirs"]) > 1:
                X, flat, det = fusion.tapped(loc, sph)
                res = stores[name].add(loc, "game:%s#%s" % (r["game"], r["round"]),
                                       fusion.capture_state(res, X, flat, det), map_info=mp)
            if k == 0:
                start[name] = res
            final[name] = res
    out = {}
    for name in _WORKER["models"]:
        rec = {}
        for tag, res in (("start", start[name]), ("final", final[name] if complete else None)):
            if res is None:  # fused answer unavailable (pruned move-capture images)
                rec[tag] = {"points": np.nan, "km": np.nan, "e": np.full(len(cls), np.nan, np.float32),
                            "post": np.full(len(cls), np.nan, np.float32)}
                continue
            g = res["guess"]
            d_km = float(hk(r["lat"], r["lng"], g["lat"], g["lng"]))
            post = np.array([float(res["posterior"].get(c, 0.0)) for c in cls])
            prior = np.array([float(res["prior"].get(c, 0.0)) for c in cls])
            rec[tag] = {"points": float(gp(d_km, mp["maxErrorDistance"])), "km": d_km,
                        "e": (np.log(np.maximum(post, 1e-12)) - np.log(np.maximum(prior, 1e-12))).astype(np.float32),
                        "post": post.astype(np.float32)}
        out[name] = rec
    return r["game"], r["round"], dict(out, complete=complete)


def capture_has_images(d):
    """True when every view image named in the capture's views.json exists."""
    try:
        with open(os.path.join(d, "views.json")) as f:
            v = json.load(f)
    except (OSError, ValueError):
        return False
    files = [x.get("file") for x in (v.get("views") or v.get("frames") or []) if isinstance(x, dict)]
    return bool(files) and all(f and os.path.exists(os.path.join(d, f)) for f in files)


_WORKER = {}


def _points_init(model_dirs):
    from engine import fusion
    from engine.locator import Locator
    _WORKER["loc"] = fusion.install_tap(Locator())
    _WORKER["models"] = {name: GeoModel.load(d) for name, d in model_dirs.items()}


def cmd_world_points(args):
    """--world-points SPLIT: the locator re-run on the saved live captures of the World-map games of SPLIT with the
    model files of --old-model (the version the games were played with: model.json, model.npz, priors.json) and the
    current data/model; per round and variant the start answer (No Move) and the fused answer (Moving): points with
    the World formula, country rank, evidence relative to the prior -> scratch/world_eval/<split>.npz (the evidence
    for cmd_world --evidence) and, with --save, data/model/eval_world_points_<split>.json (paired by game)."""
    from multiprocessing import Pool
    model_dirs = {"old": args.old_model, "new": MODEL_DIR} if args.old_model else {"new": MODEL_DIR}
    mp = resolve_map({"id": WORLD_MAP_ID})
    cls = list(GeoModel.load().classes)
    rounds = [r for r in world_rounds() if r["split"] == args.world_points and not r["exclude"] and r["dirs"]
              and r.get("lat") is not None]
    if args.limit:
        rounds = rounds[:args.limit]
    print("%s: %d rounds (%d captures), variants %s" % (args.world_points, len(rounds),
                                                         sum(len(r["dirs"]) for r in rounds), model_dirs), flush=True)
    res, t0 = {}, time.time()
    with Pool(args.workers, initializer=_points_init, initargs=(model_dirs,)) as pool:
        for k, (g, rd, out) in enumerate(pool.imap_unordered(_points_job, [(r, mp, cls) for r in rounds])):
            res[(g, rd)] = out
            if (k + 1) % 10 == 0:
                print("  %d/%d (%.0fs)" % (k + 1, len(rounds), time.time() - t0), flush=True)
    rows = [r for r in rounds if (r["game"], r["round"]) in res]
    cidx = {c: i for i, c in enumerate(cls)}
    arrays = {"game": np.array([r["game"] for r in rows]), "round": np.array([r["round"] for r in rows]),
              "complete": np.array([bool(res[(r["game"], r["round"])]["complete"]) for r in rows]),
              "y": np.array([cidx.get(r["cc"], -1) for r in rows]), "walk": np.array([r["walk"] for r in rows]),
              "played_points": np.array([float(r["points"] or 0) for r in rows]), "classes": np.array(cls)}
    for name in model_dirs:
        for tag in ("start", "final"):
            for key in ("points", "km", "e", "post"):
                arrays["%s_%s_%s" % (name, tag, key)] = np.array([res[(r["game"], r["round"])][name][tag][key]
                                                                  for r in rows])
    os.makedirs(WORLD_EVAL_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(WORLD_EVAL_DIR, "%s.npz" % args.world_points), **arrays)
    report = world_points_report(arrays, "locator (working tree) re-run on the saved captures of the %s games; old = %s, "
                                         "new = data/model; ci95 bootstrap over games" % (args.world_points, args.old_model))
    if args.save:
        json.dump(report, open(os.path.join(MODEL_DIR, "eval_world_points_%s.json" % args.world_points), "w"), indent=1)
    return report


WORLD_EVAL_DIR = os.path.join(ROOT, "scratch", "world_eval")


def world_points_report(a, note):
    games, y = a["game"], a["y"]
    mov = a["walk"] > 0
    comp = np.asarray(a["complete"], bool) if "complete" in a else np.ones(len(y), bool)
    names = [n for n in ("new", "old") if "%s_start_points" % n in a]
    report = {"note": note, "n": int(len(y)), "games": int(len(set(games.tolist()))),
              "moving_rounds_without_move_images": int((mov & ~comp).sum())}
    for tag, sel, label in (("start", np.ones(len(y), bool), "No Move answer (start capture), all rounds"),
                            ("start", ~mov, "No Move games"), ("start", mov, "Moving games, start capture"),
                            ("final", mov & comp, "Moving games (fused)")):
        if not sel.any():
            continue
        e = {"n": int(sel.sum()), "games": int(len(set(games[sel].tolist()))),
             "points_played": round(float(a["played_points"][sel].mean()), 1)}
        for n in names:
            P = a["%s_%s_post" % (n, tag)][sel]
            yy = y[sel]
            tp = np.where(yy >= 0, P[np.arange(len(yy)), np.maximum(yy, 0)], 0.0)
            rank = np.where(yy >= 0, (P > tp[:, None]).sum(1), 999)
            e[n] = {"points": round(float(a["%s_%s_points" % (n, tag)][sel].mean()), 1),
                    "top1": round(float((rank == 0).mean()), 4), "top3": round(float((rank < 3).mean()), 4),
                    "top5": round(float((rank < 5).mean()), 4),
                    "loglik": round(float(np.log(np.maximum(tp, 1e-9)).mean()), 4),
                    "median_km": round(float(np.median(a["%s_%s_km" % (n, tag)][sel])), 1)}
        if len(names) == 2:
            e["new_minus_old_points"] = paired_diff(a["new_%s_points" % tag][sel], a["old_%s_points" % tag][sel],
                                                    games[sel])
        report[label] = e
        print("%-44s n=%d %s | played %.0f%s" % (
            label, e["n"], " | ".join("%s: %.0f pts top1 %.3f top3 %.3f" % (n, e[n]["points"], e[n]["top1"], e[n]["top3"])
                                      for n in names), e["points_played"],
            (" | new-old %+.0f %s" % (e["new_minus_old_points"]["diff"], e["new_minus_old_points"]["ci95"]))
            if len(names) == 2 else ""), flush=True)
    return report


def collections_counter(it):
    from collections import Counter
    return Counter(it)


def cmd_refit_glm(args, recs, mats, tr, ca, te, ot):
    """--refit-glm LAM,ITERS: refit only the multinomial GLM of the saved model on the CURRENT train split (classes
    and standardisation of the saved model; the Gaussians, kNN references, embedding - hence the clue index, region
    model and card detectors - are unchanged).  The train split drifts from the one the saved kNN references were
    built on: tools/world_play.py turns duel training records that share a panorama with a World-map round into
    world_play records, and load_records drops training records within 1 km of them (78050 references, 78028 GLM
    rows on 2026-10-10), so a refit is not reproducible against the saved references and those references may still
    sit on world_* rounds until the next full fit.  Memory: peak RSS ~2.5 GB (the float64 design matrix of ~78k rows
    next to the feature caches), ~5 BLAS threads - run it alone, with OMP_NUM_THREADS=2..4.  Then recalibrate what depends on the posterior as a full refit does: evidence
    exponents and prior mix from the fit defaults (GeoModel.calibrate, card exponent included), the location /
    region kernels (calibrate_location) and the map-aware prior (calibrate_maps; per-map parameters kept, refit
    them with --world).  The views parameter set and regions_params.json are not touched."""
    from engine.model import SoftmaxGLM
    lam, iters = (float(v) for v in args.refit_glm.split(","))
    model = GeoModel.load()
    cidx = {c: i for i, c in enumerate(model.classes)}
    trk = np.array([i for i in tr if recs[i]["label"] in cidx])
    yi = np.array([cidx[recs[i]["label"]] for i in trk])
    t0 = time.time()
    model.glm = SoftmaxGLM(lam=lam, iters=int(iters)).fit(model._glm_input({g: mats[g][trk] for g in model.groups}),
                                                         yi, len(model.classes))
    model._fit_hash = None
    print("GLM refit (lambda %g, %d iterations) on %d training panoramas: %.0fs" % (lam, iters, len(trk), time.time() - t0))

    def ev_for(rows):
        ev = model.evidence({g: mats[g][rows] for g in model.groups})
        return ev if args.no_cards else add_cards(model, ev, recs, rows)

    ev_ca, ev_te, ev_ot = ev_for(ca), ev_for(te), (ev_for(ot) if len(ot) else None)
    yca = np.array([cidx.get(recs[i]["label"], -1) for i in ca])
    ok = yca >= 0
    model.weights = dict({g: 0.3 for g in model.groups}, knn=0.5, glm=0.7, sun=0.0)
    model.prior_mix = 0.5
    ll = model.calibrate({k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in ev_ca.items()}, yca[ok])
    print("calibrated weights:", json.dumps({k: round(v, 3) for k, v in model.weights.items()}),
          "prior_mix", model.prior_mix, "calib loglik %.3f" % ll)
    calibrate_location(model, ev_ca, recs, ca)
    params = calibrate_maps(model, ev_ca, recs, ca)
    if args.calib_only:  # no TEST numbers (report them once at the end: train_model.py --eval-only --save)
        groups = map_groups(model, recs, ca)
        evaluate(model, ev_ca, recs, ca, "calib, no map", groups=groups)
        evaluate(model, ev_ca, recs, ca, "calib, map info", True, groups=groups)
        if args.save:
            model.save()
            if params:
                model.save_priors()
            print("saved data/model (model.npz GLM, model.json, priors.json params; eval files not written: "
                  "--calib-only)")
        return model
    out = map_report(model, ev_ca, ev_te, recs, ca, te, ev_ot, ot)
    if args.save:
        model.save()
        if params:
            model.save_priors()
        json.dump(out, open(os.path.join(MODEL_DIR, "eval_maps.json"), "w"), indent=1)
        json.dump(out["test_no_map"], open(os.path.join(MODEL_DIR, "eval.json"), "w"), indent=1)
        print("saved data/model (model.npz GLM, model.json, priors.json params, eval_maps.json, eval.json)")


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
    ap.add_argument("--views", action="store_true",
                    help="saved model: calibrate the 'views' parameter set (screenshots, live play) on the CALIB "
                         "live-grid features of tools/live_features.py and report CALIB / TEST live")
    ap.add_argument("--calib-only", action="store_true",
                    help="with --views / --refit-glm / --calibrate-maps / --calibrate-cards: no TEST report (model selection)")
    ap.add_argument("--cv", action="store_true", help="with --views: 2-fold cross-validation inside CALIB")
    ap.add_argument("--live-kind", default="grid", choices=("grid", "frame"),
                    help="with --views: live features of the grid (rotating rounds) or of single NMPZ frames")
    ap.add_argument("--refit-glm", default="", metavar="LAM,ITERS",
                    help="saved model: refit only the GLM (ridge LAM, ITERS iterations) and recalibrate the exponents, "
                         "kernels and map-aware prior on calib")
    ap.add_argument("--world", action="store_true",
                    help="saved model: the official World map's prior mixture on the world_calib games "
                         "(tools/world_play.py live answers) -> priors.json params maps[World id] with --save")
    ap.add_argument("--world-test", action="store_true", help="with --world: also report world_test")
    ap.add_argument("--evidence", default="", help="with --world: directory of cmd_world_points outputs "
                    "(<split>.npz): the current model's evidence on the captures instead of the saved responses")
    ap.add_argument("--priors", default="", help="with --world: use this priors.json (dry run, never with --save)")
    ap.add_argument("--world-points", default="", choices=("", "world_calib", "world_test"),
                    help="re-run the locator on the saved captures of these World-map games with the model files of "
                         "--old-model and the current data/model; paired points (-> scratch/world_eval/<split>.npz, "
                         "data/model/eval_world_points_<split>.json with --save)")
    ap.add_argument("--old-model", default="", help="with --world-points: directory with the model.json, model.npz and "
                    "priors.json the games were played with (paired comparison)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.world_points:
        cmd_world_points(args)
        return
    if args.world:
        if args.priors and args.save:
            raise SystemExit("--priors is a dry run; --save writes data/model/priors.json from its own counts")
        cmd_world(args)
        return
    from engine import features
    groups = args.groups.split(",") if args.groups else [m.NAME for m in features.available()
                                                         if os.path.exists(os.path.join(FEAT_DIR, m.NAME + ".npz"))]
    recs = load_records()
    mats, have = {}, np.ones(len(recs), bool)
    for g in groups:
        mats[g], h = group_matrix(g, recs)
        lost = sum(1 for r, x in zip(recs, h) if r.get("pano_deleted") and not x)
        if lost:
            sys.stderr.write("WARNING: %d streamed panoramas (pano_deleted, no image) have no %s features and are "
                             "left out of training: run `python3 tools/stream_duels.py --refresh`\n" % (lost, g))
        have &= h
    split = np.array([split_of(r) for r in recs])
    tr = np.flatnonzero((split == "train") & have)
    ca = np.flatnonzero((split == "calib") & have)
    te = np.flatnonzero((split == "test") & have)
    print(f"groups={groups} train={len(tr)} calib={len(ca)} test={len(te)}")
    ot = np.flatnonzero((split == "other") & have)
    if args.views:
        cmd_views(args, recs, ca, te, mats)
        return
    if args.refit_glm:
        cmd_refit_glm(args, recs, mats, tr, ca, te, ot)
        return
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
        if args.calib_only:  # model selection / recalibration: CALIB numbers only, no TEST report or eval files
            groups = map_groups(model, recs, ca)
            evaluate(model, ev_ca, recs, ca, "calib, no map", groups=groups)
            if model.map_params():
                evaluate(model, ev_ca, recs, ca, "calib, map info", True, groups=groups)
            if args.save and (args.calibrate_maps or args.calibrate_cards):
                model.save_priors()
                if args.calibrate_cards:
                    mj = os.path.join(MODEL_DIR, "model.json")
                    meta = json.load(open(mj))
                    meta["weights"]["cards"] = model.weights.get("cards", 0.0)
                    json.dump(meta, open(mj, "w"), indent=1)
                print("saved data/model/" + PRIORS_FILE + " params (eval files not written: --calib-only)")
            return
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
    # the "views" set of the refitted model (the old one belongs to the old fit)
    from engine.clue_detect import ClueDetectors
    from live_features import live_status
    why = live_status(groups, bank=None if args.no_cards else ClueDetectors.load())
    rmodel = rows_v = None
    if why is None:
        from engine.regions import load_region_model
        rows_v = np.array([i for i in ca if recs[i].get("heading") is not None])
        try:
            ev_v, X_v = live_inputs(model, recs, rows_v, cards=not args.no_cards)
        except ValueError as e:  # rounds missing from the live cache
            why, rows_v = str(e), None
    if why is None:
        rmodel = load_region_model()
        rp = calibrate_views(model, ev_v, X_v, recs, rows_v, rmodel)[1]
        if rp is not None:
            rmodel.param_sets = dict(rmodel.param_sets, views=rp)
        print("views parameters calibrated (report: tools/train_model.py --views --eval-only)")
    else:
        print("no views parameter set (%s): screenshots use the panorama parameters until tools/live_features.py "
              "and tools/train_model.py --views --save" % why)
    if args.save:
        model.save()
        if rows_v is not None:
            save_region_views(rmodel, len(rows_v))
        if params:
            model.save_priors()
        json.dump(out, open(os.path.join(MODEL_DIR, "eval_maps.json"), "w"), indent=1)
        json.dump(out["test_no_map"], open(os.path.join(MODEL_DIR, "eval.json"), "w"), indent=1)
        print("saved data/model")


if __name__ == "__main__":
    main()
