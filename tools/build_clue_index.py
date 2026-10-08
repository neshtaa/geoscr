#!/usr/bin/env python3
"""
Build data/model/clue_index.npz, the labelled data behind the GeoGuessr card ranking in
engine/hints.py, and calibrate that ranking.

  python3 tools/build_clue_index.py [--groups solar,road,...] [--no-tune [--params-from INDEX]]
                                    [--baseline-rev 0c406f7]

Labels are GeoGuessr's own clue placements of finished rounds (data/calibration/pano_clues.json,
tools/fetch_pano_clues.py). Only rounds that are legal for fitting enter the index: public duel
rounds (minus anything within 1 km of the user's own rounds) and the user's CALIB rounds (minus
those within 1 km of a TEST round); the user's TEST rounds are used for evaluation only. Every
indexed panorama keeps its country, admin-1 region and clue cards; panoramas with a feature cache
(scratch/features, tools/calibrate.py) also keep a compact embedding: robust-standardised features
(median / IQR of the train split) projected on the leading principal or Fisher-discriminant axes.

The ranking (engine.hints.ClueBase.rank_cards: keyword score, region bonus, kNN vote, card
frequency, region-mixture vote) and the kNN kernel are grid-searched on CALIB (leave-one-out: each
round is scored without itself, so the CALIB numbers of the tuned ranking are optimistic);
precision@3 / recall@3 of the three shown cards against the real placements are then reported on
TEST given the true country for the tuned ranking and for each channel alone, with paired bootstrap
95% intervals of the differences; --baseline-rev also scores engine/hints.py of a git revision
(0c406f7 = the keyword + region ranking before the index). Region posteriors come from the saved
country model (data/model) given the true country.
"""
import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from calibrate import LABEL_FIX, load_features, load_records, split_of  # noqa: E402
from engine.geo import haversine_km, region_index, region_info  # noqa: E402
from engine.hints import (DEFAULT_RANK, INDEX_PATH, ClueBase, ClueIndex, clue_stem, observations,  # noqa: E402
                          region_probs_for)

CLUES = os.path.join(ROOT, "data", "calibration", "pano_clues.json")
HISTORY = os.path.join(ROOT, "data", "calibration", "history_rounds.json")
DUELS = os.path.join(ROOT, "data", "calibration", "duel_rounds.json")
GROUPS = "solar,road,landscape,vehicle,structure,texture"


def _label(cc):
    cc = (cc or "").upper()
    return LABEL_FIX.get(cc, cc) or None


def labelled_rounds(clues):
    """{pano_id: {"label", "split", "lat", "lng", "region"}} for every panorama with clue placements.
    CALIB rounds within 1 km of a TEST round become 'calib-near-test' and stay out of the index."""
    own = json.load(open(HISTORY))
    olat, olng = np.array([r["lat"] for r in own]), np.array([r["lng"] for r in own])
    split = [split_of({"mode": "history", "map": r.get("map"), "game": r.get("game"), "pano_id": r["pano_id"]})
             for r in own]
    is_test = np.array([s == "test" for s in split])
    out = {}
    for k, (r, s) in enumerate(zip(own, split)):
        if r["pano_id"] not in clues or (out.get(r["pano_id"]) or {}).get("split") == "test":
            continue
        if s == "calib" and is_test.any() and float(np.min(haversine_km(r["lat"], r["lng"], olat[is_test],
                                                                        olng[is_test]))) < 1.0:
            s = "calib-near-test"
        out[r["pano_id"]] = {"label": _label(r.get("gg_country")), "lat": r["lat"], "lng": r["lng"], "split": s}
    if os.path.exists(DUELS):
        for r in json.load(open(DUELS))["rounds"]:
            pid = r.get("pano_id")
            if pid not in clues or pid in out:
                continue
            if float(np.min(haversine_km(r["lat"], r["lng"], olat, olng))) < 1.0:
                continue  # a location of the user's own rounds: never in the index
            out[pid] = {"label": _label(r.get("gg_country")), "lat": r["lat"], "lng": r["lng"], "split": "train"}
    pids = list(out)
    for pid, i in zip(pids, region_index([out[p]["lat"] for p in pids], [out[p]["lng"] for p in pids])):
        out[pid]["region"] = region_info(int(i))[0] if i else ""
    return out


def card_sets(clues, rounds):
    """{pano_id: [clue stem, ...]} of the country cards placed on each panorama."""
    out = {}
    for pid, r in rounds.items():
        s = sorted({clue_stem(p.get("title")) or p["id"].lower() for p in clues[pid]
                    if (p.get("countryCode") or "").upper() == r["label"]})
        if s:
            out[pid] = s
    return out


def catalogue(clues, cards, rounds, legal):
    """Card records of every clue placed on an index panorama: identity, type, regions, English text,
    median camera pitch / zoom of the placements."""
    tr = json.load(open(os.path.join(ROOT, "data", "geoguessr_all_clue_translations.json")))
    pl = defaultdict(list)
    for pid in legal:
        cc = rounds[pid]["label"]
        for p in clues[pid]:
            if (p.get("countryCode") or "").upper() == cc:
                pl[(cc, clue_stem(p.get("title")) or p["id"].lower())].append(p)
    out = []
    for (cc, stem), ps in sorted(pl.items()):
        p0 = ps[0]
        txt = {}
        for p in ps:
            txt.update(p.get("text") or {})
        seterra = sorted({i for p in ps for i in (p.get("seterraRegionIds") or [])})
        out.append({"id": stem, "country": cc, "gg_id": p0.get("id"), "type": p0.get("type"),
                    "category": p0.get("category"), "image": p0.get("image"), "seterra": seterra,
                    "title": tr.get(p0.get("title")) or txt.get("title"),
                    "description": tr.get(p0.get("description")) or txt.get("description"),
                    "pitch": round(float(np.median([p.get("pitch", 0.0) for p in ps])), 2),
                    "zoom": round(float(np.median([p.get("zoom", 1.0) for p in ps])), 3),
                    "n": len(ps)})
    return out


# ------------------------------------------------------------------ embeddings
def robust_scaler(X):
    med = np.nanmedian(X, axis=0)
    q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
    sc = np.where((q3 - q1) > 1e-6 * (np.abs(med) + 1), (q3 - q1) / 1.349, np.nanstd(X, axis=0))
    sc = np.where(np.isfinite(sc) & (sc > 1e-9), sc, 1.0)
    return np.nan_to_num(med), sc


def standardise(X, med, sc):
    Z = np.clip((X - med) / sc, -6.0, 6.0)
    return np.where(np.isnan(Z), 0.0, Z)


def projection(Z, y, kind, dims, shrink=0.3):
    """'pca': leading principal axes (scaled to unit variance); 'lda': whitened Fisher-discriminant
    axes of the countries (shared shrunk covariance), weighted by sqrt(eigenvalue)."""
    Z = Z - Z.mean(0)
    if kind == "pca":
        w, V = np.linalg.eigh(Z.T @ Z / len(Z))
        o = np.argsort(-w)[:dims]
        return V[:, o] / np.sqrt(np.maximum(w[o], 1e-9))[None, :]
    classes, yi = np.unique(y, return_inverse=True)
    mu = np.stack([Z[yi == c].mean(0) for c in range(len(classes))])
    R = Z - mu[yi]
    S = R.T @ R / len(Z)
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S)) + 1e-6 * np.eye(len(S))
    L = np.linalg.cholesky(np.linalg.inv(S))
    cnt = np.bincount(yi).astype(float)
    Sb = (mu * cnt[:, None]).T @ mu / cnt.sum()
    ev, U = np.linalg.eigh(L.T @ Sb @ L)
    o = np.argsort(-ev)[:dims]
    return (L @ U[:, o]) * np.sqrt(np.maximum(ev[o], 1e-9) / max(ev[o].max(), 1e-9))[None, :]


def index_arrays(order, rounds, cards, cat, emb, med, sc, proj, fnames):
    cid = {(c["country"], c["id"]): i for i, c in enumerate(cat)}
    ptr, idx = [0], []
    for pid in order:
        idx += [cid[(rounds[pid]["label"], s)] for s in cards[pid]]
        ptr.append(len(idx))
    return {"feat_names": np.array(fnames), "med": med.astype(np.float32), "sc": sc.astype(np.float32),
            "proj": proj.astype(np.float32), "emb": emb.astype(np.float32),
            "country": np.array([rounds[p]["label"] for p in order]),
            "region": np.array([rounds[p]["region"] for p in order]), "pano_ids": np.array(order),
            "card_ptr": np.array(ptr, np.int32), "card_idx": np.array(idx, np.int32)}


# ------------------------------------------------------------------ evaluation
class EvalSet:
    """Candidate cards of the true country of every evaluation panorama as padded (n, m) matrices:
    keyword score, region bonus, is-a-true-card."""

    def __init__(self, kb, pids, rounds, cards, F, regions, index=None):
        rows = []
        for pid in pids:
            cc = rounds[pid]["label"]
            cand = kb.gg.get(cc, [])
            if cand:
                rows.append((pid, cc, cand))
        n, m = len(rows), max([len(r[2]) for r in rows] or [1])
        self.pid, self.cc = [r[0] for r in rows], [r[1] for r in rows]
        self.ids = [[[c["id"]] + c["aliases"] for c in r[2]] for r in rows]
        self.KW, self.REG, self.T = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m), bool)
        self.valid = np.zeros((n, m), bool)
        self.F = [F[pid] for pid in self.pid]
        self.rp, self.emb = [], [None] * n
        self.ntrue = np.zeros(n)
        for k, (pid, cc, cand) in enumerate(rows):
            tags = observations(F[pid])
            rp = region_probs_for(regions.get(pid), cc)
            self.rp.append(rp)
            self.valid[k, :len(cand)] = True
            truth = {kb.card_id(cc, s) for s in cards[pid]}
            self.ntrue[k] = len(truth)
            for j, c in enumerate(cand):
                self.KW[k, j] = kb._score((c.get("title") or "") + " " + (c.get("description") or ""), tags)[0]
                self.REG[k, j] = sum(rp.get(r, 0.0) for r in c["regions"]) if c["regions"] and rp else 0.0
                self.T[k, j] = c["id"] in truth
        self.S0 = DEFAULT_RANK["kw"] * self.KW + DEFAULT_RANK["region"] * self.REG
        self.reachable = float(np.mean(self.T.sum(1) / np.maximum(self.ntrue, 1))) if n else 0.0
        if index is not None:
            self.embed_with(index)

    def embed_with(self, index):
        self.emb = [index.embed(f) for f in self.F]

    def __len__(self):
        return len(self.pid)

    def channels(self, index, params, loo, with_regions=True):
        """kNN / frequency / region-vote scores of the candidates: {name: (n, m)}."""
        out = {k: np.zeros(self.KW.shape) for k in ("knn", "freq", "rvote")}
        for k, pid in enumerate(self.pid):
            sc = index.card_scores(self.cc[k], self.emb[k], self.rp[k] if with_regions else None,
                                   pid if loo else None, params)
            for name, d in sc.items():
                out[name][k, :len(self.ids[k])] = [max(d.get(i, 0.0) for i in ids) for ids in self.ids[k]]
        return out

    def hits(self, ch, w, n=3):
        """True cards among the n best-scored cards of every round (same order as ClueBase.rank_cards:
        score, then the default keyword + region score, then catalogue order)."""
        S = w["kw"] * self.KW + w["region"] * self.REG
        for name in ("knn", "freq", "rvote"):
            if w.get(name):
                S = S + w[name] * ch[name]
        S, S0 = np.where(self.valid, S, -np.inf), np.where(self.valid, self.S0, -np.inf)
        col = np.broadcast_to(np.arange(S.shape[1]), S.shape)
        top = np.lexsort((col, -S0, -S), axis=1)[:, :n]
        return np.take_along_axis(self.T, top, 1).sum(1).astype(float)

    def metrics(self, ch, w, n=3):
        """(precision@n, recall@n)."""
        h = self.hits(ch, w, n)
        return float(np.mean(h / float(n))), float(np.mean(h / np.maximum(self.ntrue, 1)))


def bootstrap(a, b, ntrue, n=3, reps=4000, seed=0):
    """Paired bootstrap over rounds of the precision@n / recall@n difference a - b (hits per round):
    [(mean, lo95, hi95) for precision, recall]."""
    rng = np.random.RandomState(seed)
    dp, dr = (a - b) / float(n), (a - b) / np.maximum(ntrue, 1)
    idx = rng.randint(0, len(a), (reps, len(a)))
    return [(float(d.mean()), float(np.percentile(d[idx].mean(1), 2.5)), float(np.percentile(d[idx].mean(1), 97.5)))
            for d in (dp, dr)]


def baseline_rev(rev, pids, rounds, cards, F, regions, kb):
    """{pano_id: hits@3} of the hint ranking of engine/hints.py at git revision rev (its own catalogue,
    keyword + region scoring) given the true country; shown cards are mapped to clue keys through their
    catalogue entry. Rounds of countries without cards in that catalogue are left out."""
    import importlib.util
    import subprocess
    import tempfile
    src = subprocess.check_output(["git", "-C", ROOT, "show", "%s:engine/hints.py" % rev])
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
        fh.write(src)
    try:
        spec = importlib.util.spec_from_file_location("engine._hints_rev", fh.name)
        H = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(H)
    finally:
        os.unlink(fh.name)
    H.DATA = os.path.join(ROOT, "data")
    old = H.ClueBase()
    pm = json.load(open(os.path.join(ROOT, "data", "geoguessr_postmatch_clues.json"))).get("clues_by_id", {})
    stem = {(c.get("id") or "").lower(): clue_stem(c.get("title_key")) for c in pm.values() if c.get("title_key")}

    def key_of(cc, shown):
        for c in old.gg[cc]:  # the shown card's catalogue entry -> its clue key
            if c.get("title") == shown["title"] and (c.get("description") or "").strip() == shown["text"]:
                k = (c.get("id") or "").lower()
                return kb.card_id(cc, stem.get(k, k))
        return None

    out = {}
    for pid in pids:
        cc = rounds[pid]["label"]
        if not old.gg.get(cc):
            continue
        res = old.for_country(cc, H.observations(F[pid]), region_probs=region_probs_for(regions.get(pid), cc))
        shown = {key_of(cc, c) for c in res["geoguessr"][:3]}
        out[pid] = float(len(shown & {kb.card_id(cc, s) for s in cards[pid]}))
    return out


def simplex(step=0.125):
    k = int(round(1 / step))
    return [(a * step, b * step, (k - a - b) * step) for a in range(k + 1) for b in range(k + 1 - a)]


def tune(ev, build, projs):
    """Staged grid search on CALIB (leave-one-out): kNN space and kernel, region-vote shrinkage,
    then the weights of the five scores. Objective: precision@3 + recall@3."""
    def obj(ch, w):
        return sum(ev.metrics(ch, w))

    best_knn = (-1.0, None, None)
    for key in projs:
        index = build(key, DEFAULT_RANK)
        ev.embed_with(index)
        for k, bw, beta in itertools.product((5, 10, 20, 40), (0.5, 1.0, 2.0), (0.5, 2.0)):
            p = dict(DEFAULT_RANK, k=k, bw=bw, beta=beta)
            ch = ev.channels(index, p, loo=True, with_regions=False)
            s = max(obj(ch, dict(kw=0, region=0, knn=a, freq=1 - a)) for a in (0.25, 0.5, 0.75, 1.0))
            if s > best_knn[0]:
                best_knn = (s, key, p)
    _, key, p = best_knn
    index = build(key, p)
    ev.embed_with(index)
    best = (-1.0, None)
    for beta_r in (0.5, 1.0, 2.0, 5.0):
        p = dict(p, beta_r=beta_r)
        ch = ev.channels(index, p, loo=True)
        for kw, region, (a, b, c) in itertools.product((0.0, 0.01, 0.03, 0.1, 0.3), (0.0, 0.1, 0.25, 0.5, 1.0),
                                                       simplex()):
            w = dict(kw=kw, region=region, knn=a, freq=b, rvote=c)
            s = obj(ch, w)
            if s > best[0] + 1e-9:
                best = (s, dict(p, **w))
    return key, best[1]


def region_posteriors(pids, Xg):
    """Admin-1 posterior given the TRUE country (within-country kernel of the saved model)."""
    from engine.model import GeoModel
    m = GeoModel.load()
    ev = m.evidence({g: Xg[g] for g in m.groups})
    cls = {c: i for i, c in enumerate(m.classes)}
    out = {}
    for k, (pid, cc) in enumerate(pids):
        if cc not in cls:
            continue
        lp = np.full(len(m.classes), -1e9)
        lp[cls[cc]] = 0.0
        pr = m.region_posterior(lp, ev["_d2"][k])
        out[pid] = [{"code": region_info(int(i))[0], "name": region_info(int(i))[1], "country": region_info(int(i))[2],
                     "probability": float(pr[i])} for i in np.flatnonzero(pr > 0)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=GROUPS)
    ap.add_argument("--no-tune", action="store_true", help="keep the ranking parameters of --params-from")
    ap.add_argument("--params-from", default=INDEX_PATH, help="index whose parameters --no-tune keeps")
    ap.add_argument("--baseline-rev", default="", help="also evaluate engine/hints.py of this git revision")
    ap.add_argument("--clues", default=CLUES, help="clue placements {pano_id: [placement, ...]}")
    ap.add_argument("--out", default=INDEX_PATH)
    args = ap.parse_args()
    t0 = time.time()
    old = ClueIndex.load(args.params_from) if args.no_tune else None
    if args.no_tune and old is None:
        sys.exit("--no-tune: no readable index at %s" % args.params_from)
    clues = json.load(open(args.clues))
    rounds = labelled_rounds(clues)
    cards = card_sets(clues, rounds)
    legal = sorted(p for p in cards if rounds[p]["split"] in ("train", "calib"))
    test = sorted(p for p in cards if rounds[p]["split"] == "test")
    cat = catalogue(clues, cards, rounds, legal)
    sp = Counter(rounds[p]["split"] for p in legal)
    print("labelled panoramas: %d train (duels), %d calib in the index; %d test held out, %d calib rounds near a "
          "test round dropped; %d cards" % (sp["train"], sp["calib"], len(test),
                                           sum(r["split"] == "calib-near-test" for r in rounds.values()), len(cat)))

    groups = args.groups.split(",")
    recs = load_records()
    X, fnames, recs = load_features(groups, recs)
    X = X.astype(np.float64)
    row = {r["pano_id"]: i for i, r in enumerate(recs) if not np.isnan(X[i]).all()}
    tr = np.array([i for i, r in enumerate(recs) if split_of(r) == "train" and r["pano_id"] in row])
    med, sc = robust_scaler(X[tr])
    Ztr = standardise(X[tr], med, sc)
    ytr = np.array([recs[i]["label"] for i in tr])
    F = {p: dict(zip(fnames, X[row[p]])) for p in legal + test if p in row}
    print("features: %d train panoramas fit the scaler; %d of %d index panoramas have features"
          % (len(tr), sum(p in row for p in legal), len(legal)))

    projs = {}
    for kind, dims in (("pca", 16), ("pca", 32), ("lda", 16), ("lda", 32)):
        projs[(kind, dims)] = projection(Ztr, ytr, kind, dims)

    def build(key, params):
        proj = projs[key]
        emb = np.full((len(legal), proj.shape[1]), np.nan)
        for k, p in enumerate(legal):
            if p in row:
                emb[k] = standardise(X[row[p]][None, :], med, sc)[0] @ proj
        meta = {"params": params, "cards": cat, "projection": "%s-%d" % key, "groups": groups}
        arrays = index_arrays(legal, rounds, cards, cat, emb, med, sc, proj, fnames)
        return ClueIndex(dict(arrays, meta=np.array(json.dumps(meta)))), arrays, meta

    ev_ids = [p for p in legal + test if rounds[p]["split"] in ("calib", "test") and p in row]
    Xg, off = {}, 0
    for g in groups:
        d = sum(1 for f in fnames if f.startswith(g + "."))
        Xg[g] = np.stack([X[row[p], off:off + d] for p in ev_ids]) if ev_ids else np.zeros((0, d))
        off += d
    try:
        regions = region_posteriors([(p, rounds[p]["label"]) for p in ev_ids], Xg)
    except Exception as e:  # the region bonus is then evaluated as absent
        print("region posteriors unavailable (%r)" % (e,))
        regions = {}

    calib = [p for p in legal if rounds[p]["split"] == "calib" and p in row]
    params, best_key = dict(DEFAULT_RANK), ("lda", 16)
    if old is not None:
        params = old.params
        kind, dims = old.info.get("projection", "lda-16").split("-")
        best_key = (kind, int(dims))
    elif calib:
        ev = EvalSet(ClueBase(index=build(best_key, params)[0]), calib, rounds, cards, F, regions)
        best_key, params = tune(ev, lambda key, p: build(key, p)[0], projs)
        print("calib (leave-one-out) best: projection %s-%d, %s" % (best_key[0], best_key[1], params))

    index, arrays, meta = build(best_key, params)
    kb = ClueBase(index=index)
    report = {}
    variants = (("keywords + region", DEFAULT_RANK), ("frequency only", dict(kw=0, region=0, freq=1)),
                ("kNN only", dict(kw=0, region=0, knn=1)), ("region vote only", dict(kw=0, region=0, rvote=1)),
                ("tuned", params))
    for name, pids, loo in (("calib", calib, True), ("test", [p for p in test if p in row], False)):
        ev = EvalSet(kb, pids, rounds, cards, F, regions, index)
        if not len(ev):
            continue
        ch = ev.channels(index, params, loo)
        res = {"n": len(ev), "reachable": ev.reachable, "ci_vs_tuned": {}}
        print("[%s] n=%d, %.0f%% of the true cards are in the catalogue%s" % (
            name, len(ev), 100 * ev.reachable, " (leave-one-out)" if loo else ""))
        hits = {v: ev.hits(ch, w) for v, w in variants}
        for v, _ in variants:
            res[v] = ev.metrics(ch, dict(variants)[v])
            ci = ""
            if v != "tuned":
                d = bootstrap(hits["tuned"], hits[v], ev.ntrue)
                res["ci_vs_tuned"][v] = d
                ci = "   tuned - this: P %+.3f [%+.3f, %+.3f]  R %+.3f [%+.3f, %+.3f]" % (d[0] + d[1])
            print("   %-20s precision@3 %.3f  recall@3 %.3f%s" % (v, res[v][0], res[v][1], ci))
        if args.baseline_rev and not loo:
            b = baseline_rev(args.baseline_rev, ev.pid, rounds, cards, F, regions, kb)
            k = np.array([i for i, p in enumerate(ev.pid) if p in b])
            hb = np.array([b[ev.pid[i]] for i in k])
            nt = ev.ntrue[k]
            d = bootstrap(hits["tuned"][k], hb, nt)
            res["git " + args.baseline_rev] = {"n": len(k), "old": (float(np.mean(hb / 3.0)), float(np.mean(hb / nt))),
                                               "tuned": (float(np.mean(hits["tuned"][k] / 3.0)),
                                                         float(np.mean(hits["tuned"][k] / nt))), "ci": d}
            print("   git %s hints.py (n=%d) precision@3 %.3f  recall@3 %.3f   tuned - this: P %+.3f [%+.3f, %+.3f]  "
                  "R %+.3f [%+.3f, %+.3f]" % ((args.baseline_rev, len(k), np.mean(hb / 3.0), np.mean(hb / nt))
                                              + d[0] + d[1]))
        report[name] = res
    meta["eval"] = report
    meta["built"] = time.strftime("%Y-%m-%d %H:%M")
    meta["n_index"] = {"train": sp["train"], "calib": sp["calib"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    print("saved %s (%d panoramas, %d cards) in %.0fs" % (args.out, len(legal), len(cat), time.time() - t0))


if __name__ == "__main__":
    main()
