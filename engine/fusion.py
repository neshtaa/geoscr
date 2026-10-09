"""
Fusion of several captures of one GeoGuessr round (moving games: the player walks to other panoramas
of the same place and the live helper re-captures what becomes visible).

Country posterior of n captures k = 1..n (each analysed alone by engine.locator.Locator.analyze, which
returns its full "posterior" and the "prior" it used):

    log P(c | 1..n) = log prior_n(c) + w(n) * sum_k [log P_k(c) - log prior_k(c)] + const

i.e. the prior times the product of the captures' evidence (likelihood ratios against their own prior),
each tempered by w(n) = 1 / (1 + rho (n - 1)).  Panoramas 30-200 m apart along the same road share most
of their evidence (same vegetation, road, architecture, camera generation), so a plain product
(rho = 0) over-counts; rho = 1 is the geometric mean (no gain from moving).  rho is fitted on the CALIB
rounds (tools/moving_calib.py: real neighbouring Street View panoramas of the user's rounds rendered into
the live grid); n = 1 gives the capture's own result unchanged.

Regions and the guess are fused the same way: inside each country the captures' region posteriors
(engine.regions, computed under the fused country posterior) are combined as
    log P(r | c) = w_prior log pi(r | c) + w(n) sum_k [log P_k(r | c) - w_prior log pi(r | c)]
(pi = the region model's prior and its calibrated exponent; a country without a usable region prior gets the
captures' geometric mean), the location kernel uses the mean of the
captures' embedding distances to the reference panoramas, and the guess is the score-optimal point of
engine.model.GeoModel.locate under the fused country / region posterior (the same calls as the locator).
Hints are built from the fused posterior; the observations are the union of the captures' observations
(strongest strength per tag), the clue-index embedding is the mean of the captures'.

Re-captures of the same panorama (the player returned to the start, undid a move, or pressed Alt+G twice)
must not count twice: a capture whose embedding lies within dup_ratio x the location-kernel scale of an
earlier one replaces it (threshold set on CALIB from re-renderings of the same panorama vs neighbours).

The live server keeps the captures per round (FusionStore: client-supplied round key = game token +
round number, no location data; LRU + TTL bound).  Nothing here reads a location.
"""

import collections
import json
import math
import os
import threading
import time

import numpy as np

from .geo import region_info, resolve_map

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PARAMS_PATH = os.path.join(DATA_DIR, "fusion.json")
# defaults until tools/moving_calib.py writes data/fusion.json
DEFAULT_PARAMS = {"rho": 0.5, "dup_ratio": 0.05, "max_captures": 8, "d2": "mean"}

_PARAMS = None


def load_params(path=PARAMS_PATH):
    """Fusion parameters: data/fusion.json "params" (tools/moving_calib.py) over the defaults."""
    global _PARAMS
    if _PARAMS is None or path != PARAMS_PATH:
        p = dict(DEFAULT_PARAMS)
        try:
            with open(path, encoding="utf-8") as f:
                p.update((json.load(f) or {}).get("params") or {})
        except (OSError, ValueError):
            pass
        if path != PARAMS_PATH:
            return p
        _PARAMS = p
    return dict(_PARAMS)


def weight(n, rho):
    """Per-capture evidence exponent w(n) = 1 / (1 + rho (n - 1)); n w(n) = the effective number of captures."""
    return 1.0 / (1.0 + float(rho) * max(int(n) - 1, 0))


def _logp(v):
    return np.log(np.maximum(np.asarray(v, np.float64), 1e-300))


def _norm_log(lp):
    lp = lp - lp.max()
    p = np.exp(lp)
    return p / p.sum()


def model_for(loc_model, kind):
    """The GeoModel with the parameter set of an input kind, as the locator picks it."""
    f = getattr(loc_model, "for_input", None)
    return f(kind) if f else loc_model


def region_params(rmodel, kind):
    if rmodel is None:
        return None
    f = getattr(rmodel, "params_for", None)
    return f(kind) if f else rmodel.params


# --------------------------------------------------------------------------- feature tap
class _TapModule:
    """A feature module whose last extraction (per sphere object) is kept and reused: the locator's
    features of a capture without changing engine/locator.py, and a second analysis of the same sphere
    (e.g. with map info) without extracting again."""

    def __init__(self, mod):
        self._mod = mod
        self.last = None

    def __getattr__(self, k):
        return getattr(self._mod, k)

    def extract(self, sph):
        if self.last is not None and self.last[0] is sph:
            r = self.last[1]
        else:
            try:
                r = self._mod.extract(sph)
            except Exception as e:  # the locator turns a failing module into NaN features
                r = e
            self.last = (sph, r)
        if isinstance(r, Exception):
            raise r
        return r


class _TapCards:
    def __init__(self, cards):
        self._cards = cards
        self.last = None

    def __getattr__(self, k):
        return getattr(self._cards, k)

    def detect(self, sph):
        if self.last is not None and self.last[0] is sph:
            r = self.last[1]
        else:
            try:
                r = self._cards.detect(sph)
            except Exception as e:
                r = e
            self.last = (sph, r)
        if isinstance(r, Exception):
            raise r
        return dict(r)


def install_tap(loc):
    """Wrap the locator's feature modules and card detectors (idempotent)."""
    if getattr(loc, "_fusion_tap", False):
        return loc
    loc.modules = {k: (m if isinstance(m, _TapModule) else _TapModule(m)) for k, m in loc.modules.items()}
    if getattr(loc, "cards", None) is not None and not isinstance(loc.cards, _TapCards):
        loc.cards = _TapCards(loc.cards)
    loc._fusion_tap = True
    return loc


def last_sphere(loc):
    """The sphere of the locator's latest analysis (None before the first one or without the tap)."""
    for mod in loc.modules.values():
        last = getattr(mod, "last", None)
        if last is not None:
            return last[0]
    return None


def tapped(loc, sph=None):
    """(X {group: (1, d)}, flat {'module.feature': value}, card detections or None) of the locator's
    analysis of sph (default: its latest analysis; NaN features for a module that failed or did not run)."""
    sph = last_sphere(loc) if sph is None else sph
    X, flat = {}, {}
    for name, mod in loc.modules.items():
        last = getattr(mod, "last", None)
        r = last[1] if last is not None and last[0] is sph else None
        names = list(mod.FEATURE_NAMES)
        if isinstance(r, dict):
            x = np.asarray(r["x"], np.float64)
        else:
            x = np.full(len(names), np.nan)
        X[name] = x[None, :]
        flat.update({"%s.%s" % (name, k): float(v) for k, v in zip(names, x)})
    det = None
    cl = getattr(loc, "cards", None)
    last = getattr(cl, "last", None)
    if last is not None and last[0] is sph and isinstance(last[1], dict):
        det = last[1]
    return X, flat, det


# --------------------------------------------------------------------------- capture state
def capture_state(result, X, flat=None, det=None, model=None):
    """Compact state of one capture: the analysis' posterior / prior, its features, observations and
    the embedding used to recognise re-captures of the same panorama."""
    post = result.get("posterior")
    prior = result.get("prior")
    if not post or not prior:
        raise ValueError("the locator result has no full posterior / prior (engine/locator.py too old)")
    classes = list(post.keys())
    st = {"classes": classes, "post": np.array([post[c] for c in classes], np.float64),
          "prior": np.array([prior.get(c, 0.0) for c in classes], np.float64),
          "X": {g: np.asarray(v, np.float64).reshape(1, -1) for g, v in X.items()},
          "flat": flat or {}, "det": det, "kind": _kind(result),
          "result": result, "t": time.time()}
    if model is not None:
        st.update(embedding(model_for(model, st["kind"]), st["X"]))
    return st


def _kind(result):
    """The parameter set the locator used for a capture (its result's input kind; older locators: by source)."""
    inp = result.get("input") or {}
    return inp.get("kind") or ("views" if inp.get("source") == "views" else "pano")


def embedding(m, X):
    """{"emb": (D,) GeoModel discriminant embedding, "d2": (n_refs,) squared distances to the reference
    panoramas (the location kernel's input), "h2": their scale (median of the 200 smallest)}."""
    emb = m._embed(X)
    d2 = np.maximum((emb ** 2).sum(1)[:, None] + (m.ref_emb ** 2).sum(1)[None, :] - 2.0 * emb @ m.ref_emb.T, 0.0)[0]
    k = min(200, len(d2))
    return {"emb": emb[0], "h2": max(float(np.median(np.partition(d2, k - 1)[:k])), 1e-6), "d2": d2.astype(np.float32)}


def dup_ratio(a, b):
    """Squared embedding distance between two captures relative to the location-kernel scale."""
    if a.get("emb") is None or b.get("emb") is None:
        return float("inf")
    d2 = float(((a["emb"] - b["emb"]) ** 2).sum())
    return d2 / max(min(a["h2"], b["h2"]), 1e-9)


def _vec(st, key, classes):
    if st["classes"] == classes:
        return st[key]
    pos = {c: i for i, c in enumerate(st["classes"])}
    return np.array([st[key][pos[c]] if c in pos else 0.0 for c in classes])


def fuse_country(caps, rho, classes=None):
    """(classes, fused posterior (C,)) of the captures (oldest first); the prior is the latest capture's."""
    classes = classes or caps[-1]["classes"]
    w = weight(len(caps), rho)
    lp = _logp(_vec(caps[-1], "prior", classes))
    for c in caps:
        lp = lp + w * (_logp(_vec(c, "post", classes)) - _logp(_vec(c, "prior", classes)))
    return classes, _norm_log(lp)


def fuse_regions(per_list, rmodel, rparams, w):
    """{country: {region code: P}} fused over the captures' per-country region posteriors."""
    from .regions import _regions
    codes = _regions()[1]
    pos = {c: i for i, c in enumerate(codes)}
    wp = float(((rparams or {}).get("weights") or {}).get("prior", 1.0))
    out = {}
    for cc in per_list[-1]:
        dists = [p.get(cc) for p in per_list if p.get(cc)]
        if not dists:
            continue
        keys = sorted(set().union(*[d.keys() for d in dists]))
        P = np.array([[d.get(k, 0.0) for k in keys] for d in dists])
        pi = None
        if rmodel is not None and cc in getattr(rmodel, "cidx", {}):
            try:
                pr = rmodel.prior(cc, rparams.get("beta"), rparams.get("alpha"))
                ridx = list(rmodel.regions_of(cc))
                pm = {codes[r]: pr[j] for j, r in enumerate(ridx)}
                pi = np.array([pm.get(k, 0.0) for k in keys])
            except Exception:
                pi = None
        if pi is None or not np.all(pi > 0):
            # no usable region prior (no region model: every capture has the same area distribution): the
            # captures' geometric mean, symmetric in the captures (no evidence is counted twice or negatively)
            lp = _logp(P).mean(0)
        else:
            base = _logp(pi)
            lp = wp * base + w * (_logp(P) - wp * base[None]).sum(0)
        lp[(P <= 0).all(0)] = -np.inf  # regions masked out (map bounds) stay out
        if not np.isfinite(lp).any():
            continue
        lp = lp - lp[np.isfinite(lp)].max()
        q = np.where(np.isfinite(lp), np.exp(lp), 0.0)
        d = {k: float(v) for k, v in zip(keys, q / q.sum()) if k in pos}
        if d:
            out[cc] = d
    return out


def fuse_core(model, caps, map_info=None, rho=0.5, d2_mode="mean", rmodel=None, top_countries=12):
    """Fused country posterior, regions, guess and in-country point of the captures (oldest first).
    model: the locator's GeoModel (the input kind's parameter set is picked here).  Returns
    {classes, post, lp, regions (list), per, guess, top_point, setup, w}."""
    from .regions import location_weights, region_list, region_posterior
    kind = caps[-1]["kind"]
    m = model_for(model, kind)
    mp = resolve_map(map_info)
    setup = m.map_setup(mp)
    where = {"score_scale_km": setup["scale_km"], "bounds": setup["bounds"]}
    classes, post = fuse_country(caps, rho, list(m.classes))
    lp = _logp(post)
    w = weight(len(caps), rho)
    for c in caps:
        if c.get("d2") is None:
            c.update(embedding(m, c["X"]))
    D = np.stack([np.asarray(c["d2"], np.float64) for c in caps])
    d2 = D.mean(0) if d2_mode == "mean" else D[-1]
    order = np.argsort(-post)
    lp_top = np.full_like(lp, -1e9)
    lp_top[order[0]] = 0.0
    out = {"classes": classes, "post": post, "lp": lp, "setup": setup, "w": w, "kind": kind}
    if rmodel is not None:
        rp = region_params(rmodel, kind)
        per_list = [region_posterior(c["X"], post, classes=classes, by_country=True, bounds=setup["bounds"],
                                     params=rp, top_countries=top_countries) for c in caps]
        per = fuse_regions(per_list, rmodel, rp, w) if len(caps) > 1 else per_list[-1]
        # the mixture over the countries, as engine.regions.region_posterior forms it (the top countries with
        # P >= 1e-4; one without regions keeps its share outside the mixture)
        items = [(classes[i], float(post[i])) for i in np.argsort(-post)[:top_countries]]
        items = [t for t in items if t[1] >= 1e-4] or items[:1]
        mix, tot = {}, sum(pc for _, pc in items)
        for cc, pc in items:
            for k, q in (per.get(cc) or {}).items():
                mix[k] = mix.get(k, 0.0) + pc * q
        mix = {k: v / tot for k, v in mix.items()} if tot > 0 else mix
        out["regions"] = region_list(mix)
        out["per"] = per
        out["guess"] = m.locate(lp, d2, w=location_weights(m, lp, d2, per, bounds=setup["bounds"]), **where)
        out["top_point"] = m.locate(lp_top, d2, w=location_weights(m, lp_top, d2, per, bounds=setup["bounds"]), **where)
    else:
        out["guess"] = m.locate(lp, d2, **where)
        out["top_point"] = m.locate(lp_top, d2, **where)
        pr = m.region_posterior(lp, d2, bounds=setup["bounds"])
        regions = []
        for i in np.argsort(-pr)[:400]:
            if pr[i] <= 0:
                break
            code, name, cc = region_info(int(i))
            regions.append({"code": code, "name": name, "country": cc, "probability": round(float(pr[i]), 5)})
        out["regions"] = regions
        out["per"] = None
    return out


# --------------------------------------------------------------------------- hints
def union_observations(caps):
    """Observations of all captures, the strongest strength per tag (the latest capture's order first)."""
    best = collections.OrderedDict()
    for c in reversed(caps):
        for o in (c["result"].get("observations") or []):
            t = o.get("tag") or o.get("text")
            if t not in best or float(o.get("strength") or 0) > float(best[t].get("strength") or 0):
                best[t] = dict(o)
    return sorted(best.values(), key=lambda o: -float(o.get("strength") or 0))


def _merged_detections(cards, cc, caps):
    """Card detections of country cc over the captures: presence ("first") from any capture, the best
    probability per card; a direction only from the latest capture (the camera is there now)."""
    best = {}
    for i, c in enumerate(caps):
        det = c.get("det")
        if det is None:
            continue
        try:
            for d in cards.detected_cards(cc, det):
                d = dict(d)
                if i != len(caps) - 1:
                    d["direction"] = False
                if d["stem"] not in best or d["prob"] > best[d["stem"]]["prob"]:
                    best[d["stem"]] = d
        except Exception:
            continue
    return sorted(best.values(), key=lambda d: -d["prob"])


def fused_hints(loc, caps, classes, post, regions, n_countries=3, n_cards=3):
    from .hints import TAGS, clue_base, observations, region_probs_for
    kb = clue_base()
    tags = collections.OrderedDict()
    for c in caps:
        for t, s in observations(c["flat"]):
            tags[t] = max(s, tags.get(t, 0.0))
    tag_list = sorted(tags.items(), key=lambda ts: -ts[1])
    embs = [kb.index.embed(c["flat"]) for c in caps] if kb.index is not None else []
    embs = [e for e in embs if e is not None]
    emb = np.mean(embs, axis=0) if embs else None
    side_tag = next((t for t, _ in tag_list if t in ("drive_left", "drive_right")), None)
    names = {r["code"]: r["name"] for r in (regions or [])}
    cards_bank = getattr(loc, "cards", None)
    out = []
    for i in np.argsort(-post)[:n_countries]:
        cc, p = classes[i], float(post[i])
        rp = region_probs_for(regions, cc)
        det = _merged_detections(cards_bank, cc, caps) if cards_bank is not None else None
        h = kb.for_country(cc, tag_list, n_gg=n_cards, region_probs=rp, emb=emb, detected=det)
        h["regions"] = [{"code": k, "name": names[k], "probability": round(v, 3)}
                        for k, v in sorted(rp.items(), key=lambda kv: -kv[1])[:3]]
        h["probability"] = round(p, 3)
        if side_tag and h["driving_side"]:
            h["driving_side_consistent"] = (h["driving_side"] == side_tag.split("_")[1])
        out.append(h)
    obs = [{"tag": t, "text": TAGS[t][0], "strength": s} for t, s in tag_list]
    if not any(c["flat"] for c in caps):  # no features tapped: the captures' own observations
        obs = union_observations(caps)
    return obs, out


# --------------------------------------------------------------------------- full fused result
def _summary(res):
    return {"countries": [{"code": c["code"], "probability": c["probability"]} for c in (res.get("countries") or [])[:3]],
            "guess": res.get("guess")}


def fuse_result(loc, caps, map_info=None, params=None, top_k=5):
    """The locator result format for the fused captures (oldest first); one capture: its own result."""
    params = dict(load_params(), **(params or {}))
    latest = caps[-1]["result"]
    if len(caps) == 1:
        return dict(latest)
    t0 = time.time()
    from .regions import load_region_model
    core = fuse_core(loc.model, caps, map_info, params["rho"], params.get("d2", "mean"), load_region_model())
    classes, post = core["classes"], core["post"]
    order = np.argsort(-post)
    kb_name = None
    try:
        from .hints import clue_base
        kb_name = clue_base().country_name
    except Exception:
        pass
    countries = [{"code": classes[i], "name": kb_name(classes[i]) if kb_name else classes[i],
                  "probability": round(float(post[i]), 4)} for i in order[:top_k]]
    regions = core["regions"]
    obs, hints = fused_hints(loc, caps, classes, post, regions)
    g, tp = core["guess"], core["top_point"]
    res = dict(latest)
    res.update({
        "countries": countries,
        "regions": regions[:5],
        "top_country_point": {"lat": round(tp["lat"], 5), "lng": round(tp["lng"], 5),
                              "expected_score_if_country_right": int(round(tp["expected_score"]))},
        "guess": {"lat": round(g["lat"], 5), "lng": round(g["lng"], 5), "expected_score": int(round(g["expected_score"]))},
        "observations": obs,
        "hints": hints,
        "posterior": {classes[i]: float("%.4g" % post[i]) for i in range(len(post))},
    })
    res["timing_ms"] = dict(latest.get("timing_ms") or {}, fusion=int((time.time() - t0) * 1000))
    return res


# --------------------------------------------------------------------------- per-round store
class FusionStore:
    """Captures per round key (client-supplied: game token + round number, no location data), bounded by
    max_rounds (least recently used dropped), ttl_s and max_captures per round (oldest dropped)."""

    def __init__(self, max_rounds=32, ttl_s=3 * 3600, max_captures=None):
        self.max_rounds, self.ttl_s = int(max_rounds), float(ttl_s)
        self.max_captures = max_captures
        self.rounds = collections.OrderedDict()
        self.lock = threading.Lock()

    def _expire(self, now):
        for k in [k for k, v in self.rounds.items() if now - v["t"] > self.ttl_s]:
            del self.rounds[k]
        while len(self.rounds) > self.max_rounds:
            self.rounds.popitem(last=False)

    def get(self, key):
        with self.lock:
            self._expire(time.time())
            r = self.rounds.get(key)
            return list(r["caps"]) if r else []

    def add(self, loc, key, cap, map_info=None, params=None, reset=False):
        """Adds a capture to the round and returns the fused result with "fusion" (n, w, whether the
        capture replaced a re-capture of the same panorama, whether it changed the top country, the
        captures' own top countries) and "capture" (the capture's own result)."""
        params = dict(load_params(), **(params or {}))
        maxc = int(self.max_captures or params.get("max_captures", 8))
        now = time.time()
        with self.lock:
            self._expire(now)
            r = self.rounds.pop(key, None)
            if r is None or reset:
                r = {"caps": [], "t": now, "top": None, "seq": 0}
            self.rounds[key] = r
            r["t"] = now
            r["seq"] += 1
            cap["seq"] = r["seq"]
            replaced = None
            if cap.get("emb") is None:
                cap.update(embedding(model_for(loc.model, cap["kind"]), cap["X"]))
            for i, old in enumerate(r["caps"]):
                q = dup_ratio(cap, old)
                if q < float(params["dup_ratio"]):
                    replaced = {"seq": old["seq"], "ratio": round(q, 4)}
                    del r["caps"][i]
                    break
            r["caps"].append(cap)
            while len(r["caps"]) > maxc:
                r["caps"].pop(0)
            caps = list(r["caps"])
            prev_top = r["top"]
        try:
            res = fuse_result(loc, caps, map_info, params)
        except Exception as e:  # never lose the capture's own answer
            res = dict(cap["result"])
            res["warnings"] = list(res.get("warnings") or []) + ["fusion failed: %r" % (e,)]
        top = res["countries"][0]["code"] if res.get("countries") else None
        with self.lock:
            if key in self.rounds:
                self.rounds[key]["top"] = top
        single = cap["result"]
        res["fusion"] = {"n": len(caps), "w": round(weight(len(caps), params["rho"]), 4), "rho": params["rho"],
                         "seq": cap["seq"], "replaced": replaced, "top": top, "prev_top": prev_top,
                         "changed_top": None if prev_top is None else top != prev_top,
                         "captures": [dict(_summary(c["result"]), seq=c["seq"]) for c in caps]}
        # the capture's own analysis (without its hints and full posterior: the HUD shows the fused result)
        res["capture"] = {k: v for k, v in single.items() if k not in ("hints", "posterior", "prior", "contributions")}
        return res


__all__ = ["FusionStore", "fuse_core", "fuse_country", "fuse_regions", "fuse_result", "capture_state", "install_tap",
           "tapped", "weight", "load_params", "embedding", "dup_ratio", "union_observations"]
