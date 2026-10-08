"""
Calibrated Bayesian country model + score-optimal guess.

Everything here is classical statistics / linear algebra (no neural networks):

  P(country | image) ∝ prior(country) · Π_g P(features_g | country)^w_g · LR_knn(country)^w_knn · P(sun | country)^w_sun

* every feature group g (one engine.features module) gets a Gaussian class-conditional
  density with a shared, shrinkage-regularised covariance; class means are shrunk
  towards the global mean (empirical Bayes) so rare countries stay sane; missing
  features are marginalised exactly (Gaussian marginal on the observed dims);
* a k-nearest-neighbour likelihood ratio in the Fisher-discriminant embedding
  captures multi-modal countries (deserts and forests of the same country);
* the sun term is analytic solar geometry (engine.features.solar.latitude_likelihood)
  averaged over the latitudes where the country is covered;
* the exponents w (one per evidence source) and the prior mix are calibrated on the
  user's real GeoGuessr rounds (calib split) by maximising log-likelihood;
* the location is a mixture over reference panoramas (country posterior spread by
  within-country similarity); the guess is the point maximising the expected
  GeoGuessr score 5000·exp(-10 d / D) with the map's maxErrorDistance D;
* map-aware prior (data/model/priors.json, tools/build_priors.py): a mixture of the map's
  empirical country frequencies (the user's CALIB rounds on that map, or public ranked duels
  for other World-type maps) with the model prior.  The map bounds remove the countries with
  (almost) nothing inside them; on maps smaller than World-type ones the world-level priors are
  conditioned on the box (times each country's share inside it).  The guess stays inside the bounds.
"""

import hashlib
import json
import math
import os
import sys

import numpy as np

from .geo import (WORLD_SCORE_SCALE_KM, clip_to_bounds, country_area_inside, haversine_km, in_bounds,
                  parse_bounds, score_scale_km)

SUN_LAT_GRID = np.arange(-60.0, 80.1, 1.0)
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "model")
PRIORS_FILE = "priors.json"
BOUNDS_MARGIN_DEG = 0.5   # location kernel: reference panoramas this far outside the bounds still count
MASK_MIN_FRAC = 0.01      # share of its area or references inside the bounds a country needs to be kept


# ----------------------------------------------------------------------------- utils
def _logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def _softmax(a):
    a = a - a.max(-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(-1, keepdims=True)


def _norm(p):
    p = np.asarray(p, float)
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-300)


def mix_prior(kind, mix, base, ranked=None, emp=None):
    """Prior of a map kind: 'map' = per-map empirical + ranked_world + model prior, 'ranked' =
    ranked_world + model prior, anything else the model prior.  The components are already
    restricted to the map bounds and normalised.  Row-wise on (n, C) arrays."""
    if kind == "map":
        return mix["map"] * emp + mix["map_ranked"] * ranked + (1.0 - mix["map"] - mix["map_ranked"]) * base
    if kind == "ranked":
        return mix["ranked"] * ranked + (1.0 - mix["ranked"]) * base
    return base


class GroupGaussian:
    """Shared-covariance Gaussian class densities for one feature group (NaN aware)."""

    def __init__(self, shrink=0.25, mean_tau=4.0, n_proj=16):
        self.shrink, self.mean_tau, self.n_proj = shrink, mean_tau, n_proj

    def fit(self, X, yi, n_classes):
        med = np.nanmedian(X, axis=0)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        sd = np.nanstd(X, axis=0)
        sc = np.where((q3 - q1) > 1e-6 * (np.abs(med) + 1), (q3 - q1) / 1.349, sd)
        sc = np.where(np.isfinite(sc) & (sc > 1e-9), sc, 1.0)
        self.med, self.sc = np.nan_to_num(med), sc
        Z = self.transform(X)
        M = ~np.isnan(Z)
        Z0 = np.where(M, Z, 0.0)
        d = Z.shape[1]
        gmean = Z0.sum(0) / np.maximum(M.sum(0), 1)
        mu = np.zeros((n_classes, d))
        for c in range(n_classes):
            rows = yi == c
            cnt = M[rows].sum(0)
            s = Z0[rows].sum(0)
            mu[c] = (s + self.mean_tau * gmean) / (cnt + self.mean_tau)
        R = np.where(M, Z - mu[yi], 0.0)
        cnt = M.T.astype(np.float64) @ M.astype(np.float64)
        S = (R.T @ R) / np.maximum(cnt - 1, 1)
        S = 0.5 * (S + S.T)
        diag = np.diag(np.diag(S))
        S = (1 - self.shrink) * S + self.shrink * diag
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, 1e-3 * max(w.max(), 1e-6))
        self.S = (V * w) @ V.T
        self.mu = mu
        self.gmean = gmean
        # Fisher discriminant directions for the kNN embedding
        Si = np.linalg.inv(self.S)
        counts = np.bincount(yi, minlength=n_classes).astype(float)
        D = mu - gmean
        Sb = (D * counts[:, None]).T @ D / max(counts.sum(), 1)
        L = np.linalg.cholesky(Si)
        Mb = L.T @ Sb @ L
        ev, U = np.linalg.eigh(0.5 * (Mb + Mb.T))
        order = np.argsort(-ev)[: min(self.n_proj, d)]
        self.proj = L @ U[:, order]  # z = x_std @ proj  (whitened discriminant coordinates)
        self.proj_ev = ev[order]
        return self

    def transform(self, X):
        return np.clip((X - self.med) / self.sc, -8, 8)

    def loglik(self, X):
        """(n, C) log-likelihoods up to a per-row constant; exact marginal over observed dims."""
        Z = self.transform(X)
        M = ~np.isnan(Z)
        out = np.zeros((len(Z), len(self.mu)))
        if len(Z) == 0:
            return out
        keys = np.packbits(M, axis=1)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        for k in np.unique(inv):
            rows = np.flatnonzero(inv == k)
            o = M[rows[0]]
            if not o.any():
                continue
            Si = np.linalg.inv(self.S[np.ix_(o, o)])
            z = Z[np.ix_(rows, o)]
            mu = self.mu[:, o]
            # -(1/2)(z-mu)^T Si (z-mu) = z^T Si mu - 1/2 mu^T Si mu - 1/2 z^T Si z
            a = z @ Si @ mu.T - 0.5 * np.sum((mu @ Si) * mu, axis=1)[None, :]
            out[rows] = a - 0.5 * np.sum((z @ Si) * z, axis=1)[:, None]
        return out

    def embed(self, X):
        Z = np.nan_to_num(self.transform(X))  # missing -> median
        return Z @ self.proj

    def to_dict(self):
        return {k: getattr(self, k) for k in ("med", "sc", "mu", "S", "gmean", "proj", "proj_ev")}

    @classmethod
    def from_dict(cls, d):
        g = cls()
        for k, v in d.items():
            setattr(g, k, np.asarray(v))
        return g


class SoftmaxGLM:
    """Multinomial logit (a generalised linear model) on the robust-standardised features of
    all groups plus missing-value indicators, ridge penalty, class-balanced likelihood (so its
    output acts as a likelihood ratio and the prior is applied separately).  Fitted by
    full-batch Nesterov gradient descent on the convex penalised log-likelihood."""

    def __init__(self, lam=3e-3, iters=400, lr=0.5):
        self.lam, self.iters, self.lr = lam, iters, lr

    def design(self, Z):
        M = np.isnan(Z)
        return np.hstack([np.clip(np.where(M, 0.0, Z), -5, 5), M[:, self.mcols].astype(float), np.ones((len(Z), 1))])

    def fit(self, Z, yi, n_classes):
        self.mcols = np.isnan(Z).mean(0) > 0.02
        F = self.design(Z)
        n, d = F.shape
        cnt = np.bincount(yi, minlength=n_classes).astype(float)
        ws = 1.0 / cnt[yi]
        ws /= ws.sum()
        Y = np.zeros((n, n_classes))
        Y[np.arange(n), yi] = 1.0
        W = np.zeros((d, n_classes))
        V = np.zeros_like(W)
        for _ in range(self.iters):
            Wn = W + 0.9 * V
            A = F @ Wn
            A -= A.max(1, keepdims=True)
            P = np.exp(A)
            P /= P.sum(1, keepdims=True)
            g = F.T @ ((P - Y) * ws[:, None]) + self.lam * Wn
            V = 0.9 * V - self.lr * g
            W = W + V
        self.W = W
        return self

    def loglik(self, Z):
        A = self.design(Z) @ self.W
        return A - A.max(1, keepdims=True)


class GeoModel:
    def __init__(self):
        self.groups = []
        self.classes = []
        self.weights = {}
        self.prior_mix = 0.5
        self.loc_params = {"bw": 1.0, "floor": 1e-3}
        self.region_params = {"bw": 1.0, "floor": 0.05}
        self.priors = {}
        self._inside, self._share = {}, {}

    # ------------------------------------------------------------------ training
    def fit(self, X_by_group, y, lat, lng, is_world, min_count=6, knn_k=40):
        y = np.asarray(y)
        classes = sorted(c for c in set(y) if (y == c).sum() >= min_count)
        self.classes = classes
        cidx = {c: i for i, c in enumerate(classes)}
        keep = np.array([c in cidx for c in y])
        yi = np.array([cidx[c] for c in y[keep]])
        self.groups = list(X_by_group)
        self.gauss = {}
        for g in self.groups:
            self.gauss[g] = GroupGaussian().fit(X_by_group[g][keep], yi, len(classes))
        self.ref_yi = yi
        self.ref_lat, self.ref_lng = np.asarray(lat)[keep], np.asarray(lng)[keep]
        self.ref_emb = self._embed({g: X_by_group[g][keep] for g in self.groups})
        self.glm = SoftmaxGLM().fit(self._glm_input({g: X_by_group[g][keep] for g in self.groups}), yi, len(classes))
        world = np.asarray(is_world)[keep]
        cw = np.bincount(yi[world], minlength=len(classes)).astype(float)
        self.prior_world = (cw + 0.5) / (cw + 0.5).sum()
        self.prior_uniform = np.full(len(classes), 1.0 / len(classes))
        self.train_freq = np.bincount(yi, minlength=len(classes)).astype(float)
        self.knn_k = knn_k
        self._index_refs()
        # latitude histogram of each country's reference panoramas (for the analytic sun term)
        H = np.zeros((len(classes), len(SUN_LAT_GRID)))
        b = np.clip(np.round(self.ref_lat - SUN_LAT_GRID[0]).astype(int), 0, len(SUN_LAT_GRID) - 1)
        np.add.at(H, (yi, b), 1.0)
        H = np.apply_along_axis(lambda h: np.convolve(h, np.ones(5) / 5.0, mode="same"), 1, H) + 1e-3
        self.lat_hist = H / H.sum(1, keepdims=True)
        self.weights = {g: 0.3 for g in self.groups}
        self.weights.update({"knn": 0.5, "glm": 0.7, "sun": 0.0})
        return self

    def _glm_input(self, X_by_group):
        return np.concatenate([self.gauss[g].transform(X_by_group[g]) for g in self.groups], axis=1)

    def _embed(self, X_by_group):
        parts = []
        for g in self.groups:
            e = self.gauss[g].embed(X_by_group[g])
            ev = np.sqrt(np.maximum(self.gauss[g].proj_ev, 0) + 1e-9)
            parts.append(e * (ev / (ev.max() + 1e-9))[None, :])
        return np.concatenate(parts, axis=1)

    # ---------------------------------------------------------------- evidence
    def sun_loglik(self, Xsolar):
        """Analytic sun term log E_{lat ~ country}[ conf * L(lat | sun) + (1 - conf) ] from the
        solar features (true azimuth from sun_az_cos/sin, elevation, detector confidence)."""
        from .features import solar
        n = solar.FEATURE_NAMES
        out = np.zeros((len(Xsolar), len(self.classes)))
        for i, x in enumerate(Xsolar):
            c, s_, el, conf = x[n.index("sun_az_cos")], x[n.index("sun_az_sin")], x[n.index("sun_el")], x[n.index("sun_conf")]
            if not (np.isfinite(c) and np.isfinite(s_) and np.isfinite(el) and np.isfinite(conf)):
                continue
            az = math.degrees(math.atan2(s_, c)) % 360.0
            L = np.asarray(solar.latitude_likelihood(az, el, SUN_LAT_GRID), float)
            L = L / max(L.mean(), 1e-12)
            mix = conf * L + (1.0 - conf)
            out[i] = np.log(self.lat_hist @ mix + 1e-9)
        return out

    def evidence(self, X_by_group, sun_ll=None):
        """Per-source (n, C) log-likelihood matrices (before weighting)."""
        ev = {}
        if sun_ll is None and "solar" in X_by_group and hasattr(self, "lat_hist"):
            sun_ll = self.sun_loglik(X_by_group["solar"])
        for g in self.groups:
            ll = self.gauss[g].loglik(X_by_group[g])
            ev[g] = ll - ll.max(1, keepdims=True)
        emb = self._embed(X_by_group)
        ev["knn"], ev["_d2"] = self._knn_loglr(emb)
        if getattr(self, "glm", None) is not None:
            ev["glm"] = self.glm.loglik(self._glm_input(X_by_group))
        if sun_ll is not None:
            ev["sun"] = sun_ll
        return ev

    def _knn_loglr(self, emb, exclude=None):
        d2 = np.maximum((emb ** 2).sum(1)[:, None] + (self.ref_emb ** 2).sum(1)[None, :]
                        - 2.0 * emb @ self.ref_emb.T, 0.0)
        if exclude is not None:
            d2[exclude] = np.inf
        k = min(self.knn_k, d2.shape[1] - 1)
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        dk = np.take_along_axis(d2, idx, 1)
        h2 = np.maximum(np.median(dk, axis=1, keepdims=True), 1e-6)
        w = np.exp(-0.5 * dk / h2)
        C = len(self.classes)
        votes = np.zeros((len(emb), C))
        for j in range(k):
            np.add.at(votes, (np.arange(len(emb)), self.ref_yi[idx[:, j]]), w[:, j])
        base = self.train_freq / self.train_freq.sum()
        p = (votes + 1.0 * base[None, :]) / (votes.sum(1, keepdims=True) + 1.0)
        return np.log(p) - np.log(base)[None, :], d2

    def base_prior(self, prior_mix=None):
        pm = self.prior_mix if prior_mix is None else prior_mix
        return pm * self.prior_world + (1 - pm) * self.prior_uniform

    def prior(self, map_info=None):
        return self.map_setup(map_info)["prior"]

    def combine(self, ev, weights=None, prior_mix=None, prior=None):
        """Log posterior (n, C); prior = (C,) or per-row (n, C) vector, else the model prior."""
        w = weights or self.weights
        if prior is None:
            prior = self.base_prior(prior_mix)
        lp = np.log(np.maximum(np.atleast_2d(prior), 1e-300))
        for g in self.groups:
            lp = lp + w.get(g, 0.0) * ev[g]
        lp = lp + w.get("knn", 0.0) * ev["knn"]
        if "glm" in ev:
            lp = lp + w.get("glm", 0.0) * ev["glm"]
        if "sun" in ev and ev["sun"] is not None:
            lp = lp + w.get("sun", 0.0) * np.nan_to_num(ev["sun"])
        return lp - _logsumexp(lp)[:, None]

    def calibrate(self, ev, yi, iters=4):
        """Coordinate ascent on the mean log-likelihood of the true class (calib split)."""
        keys = self.groups + ["knn"] + [k for k in ("glm", "sun") if k in ev]
        w = dict(self.weights)
        pm = self.prior_mix

        def obj(w_, pm_):
            lp = self.combine(ev, w_, pm_)
            return float(np.mean(lp[np.arange(len(yi)), yi]))

        best = obj(w, pm)
        grid = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3, 1.7]
        for _ in range(iters):
            for k in keys:
                for v in grid:
                    w2 = dict(w)
                    w2[k] = v
                    s = obj(w2, pm)
                    if s > best:
                        best, w = s, w2
            for v in [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]:
                s = obj(w, v)
                if s > best:
                    best, pm = s, v
        self.weights, self.prior_mix = w, pm
        return best

    # ---------------------------------------------------------------- map-aware prior
    def empirical_prior(self, counts):
        """Smoothed class frequencies from {country code: rounds} (codes outside the classes ignored)."""
        a = float(self.priors.get("alpha", 0.25))
        cidx = {c: i for i, c in enumerate(self.classes)}
        v = np.zeros(len(self.classes))
        for cc, n in counts.items():
            if cc in cidx:
                v[cidx[cc]] += max(float(n), 0.0)
        return (v + a) / (v.sum() + a * len(v))

    def refs_inside(self, bounds, margin=BOUNDS_MARGIN_DEG):
        b = parse_bounds(bounds)
        if (b, margin) not in self._inside:
            self._inside[(b, margin)] = in_bounds(self.ref_lat, self.ref_lng, b, margin)
        return self._inside[(b, margin)]

    def bounds_share(self, bounds):
        """(C,) share of each class inside the bounds (strict box): the larger of its land-area
        share and its reference-panorama share."""
        b = parse_bounds(bounds)
        if b not in self._share:
            C = len(self.classes)
            n_all = np.bincount(self.ref_yi, minlength=C).astype(float)
            n_in = np.bincount(self.ref_yi[self.refs_inside(b, 0.0)], minlength=C)
            area = country_area_inside(b)
            self._share[b] = np.maximum(n_in / np.maximum(n_all, 1.0), [area.get(c, 0.0) for c in self.classes])
        return self._share[b]

    def bounds_factors(self, bounds):
        """(support, share) prior factors of the map bounds.  support: 1 for the classes with at
        least MASK_MIN_FRAC of their area or references inside (relative to the best class, so a
        city box keeps its country), ~0 for the rest.  share: ~P(inside the bounds | class), which
        conditions a world-level prior on a smaller map (a neighbour that only touches the box
        keeps almost nothing).  Both all ones without bounds."""
        C = len(self.classes)
        if parse_bounds(bounds) is None:
            return np.ones(C), np.ones(C)
        s = self.bounds_share(bounds)
        thr = min(MASK_MIN_FRAC, 0.1 * float(s.max()))
        return np.where((s >= thr) & (s > 0), 1.0, 1e-9), np.maximum(s, 1e-9)

    def map_components(self, map_info, emp_counts=None, use=("map", "ranked")):
        """(kind, ranked_world prior, per-map prior, bounds factor of the world-level priors) of a
        map resolved by engine.geo.resolve_map, the priors restricted to the bounds and normalised.
        The map's own counts already describe its bounds (support only); the ranked-duel and model
        priors are world-level: support on World-type maps, conditioned on the box on other maps.
        emp_counts overrides the stored per-map counts (out-of-fold calibration); use limits the
        empirical priors ('map': per-map counts, 'ranked': ranked_world for World-type maps)."""
        if not map_info:
            return "none", None, None, np.ones(len(self.classes))
        support, share = self.bounds_factors(map_info.get("bounds"))
        wb = support if map_info.get("world") else share
        rc = (self.priors.get("ranked_world") or {}).get("counts")
        ranked = _norm(self.empirical_prior(rc) * wb) if rc else None
        if emp_counts is None:
            emp_counts = self.map_entry(map_info).get("counts")
        emp = None
        if "map" in use and emp_counts and ranked is not None:
            emp = _norm(self.empirical_prior(emp_counts) * support)
        if emp is not None:
            return "map", ranked, emp, wb
        ranked_ok = ranked is not None and map_info.get("world") and "ranked" in use
        return ("ranked" if ranked_ok else "base"), ranked, emp, wb

    def map_entry(self, map_info):
        return (self.priors.get("maps") or {}).get((map_info or {}).get("id") or "") or {}

    def fingerprint(self):
        return hashlib.sha1(json.dumps([self.classes, self.groups]).encode()).hexdigest()[:12]

    def map_params(self):
        """Calibrated map-aware parameters (priors.json "params"), None if missing or fitted for a
        model with other classes / feature groups."""
        p = self.priors.get("params")
        if p and p.get("model") not in (None, self.fingerprint()):
            if not getattr(self, "_warned", False):
                sys.stderr.write("priors.json params were calibrated for another model - map priors off "
                                 "(run tools/train_model.py --calibrate-maps --save)\n")
                self._warned = True
            return None
        return p

    def counts_info(self, map_info):
        """Rounds and dates behind a map's own counts; stale when the map was edited after them."""
        e = self.map_entry(map_info)
        if not e:
            return None
        dates = e.get("dates")
        upd = str((map_info or {}).get("updatedAt") or e.get("map_updated") or "")[:10]
        return {"rounds": e.get("n"), "dates": dates, "map_updated": upd or None,
                "stale": upd > dates[1] if dates and upd else None}

    def map_setup(self, map_info=None, use=("map", "ranked")):
        """Prior, evidence exponents, score scale and bounds for a map resolved by
        engine.geo.resolve_map.  No map: model prior, World scale, no bounds.  The empirical
        mixtures need the parameters calibrated by tools/train_model.py (priors.json "params")."""
        kind, ranked, emp, wb = self.map_components(map_info, use=use)
        params = self.map_params()
        if kind in ("map", "ranked") and not params:
            kind = "base"
        if kind in ("map", "ranked"):
            weights = params["weights"]
            prior = mix_prior(kind, params["mix"], _norm(self.base_prior(params["prior_mix"]) * wb), ranked, emp)
        else:
            weights, prior = self.weights, self.base_prior() * wb
        return {"kind": kind, "weights": weights, "prior": _norm(prior),
                "counts": self.counts_info(map_info) if kind == "map" else None,
                "scale_km": score_scale_km((map_info or {}).get("maxErrorDistance")),
                "bounds": parse_bounds((map_info or {}).get("bounds"))}

    # ---------------------------------------------------------------- location
    def ref_weights(self, logpost_row, d2_row, bw=None, floor=None, top_countries=12, bounds=None):
        """Posterior mass over reference panoramas: P(country) spread inside each country by a
        Gaussian kernel on the embedding distance (bandwidth bw x the median of the 200 nearest
        distances, plus a uniform floor share inside the country).  Panoramas outside the map
        bounds (plus a small margin) get no weight."""
        bw = self.loc_params.get("bw", 1.0) if bw is None else bw
        floor = self.loc_params.get("floor", 1e-3) if floor is None else floor
        post = np.exp(logpost_row)
        w = np.zeros(len(self.ref_yi))
        h2 = max(float(np.median(np.sort(d2_row)[:200])), 1e-6) * bw
        kern = np.exp(-0.5 * (d2_row - d2_row.min()) / h2)
        inside = self.refs_inside(bounds) if parse_bounds(bounds) else None
        for c in np.argsort(-post)[:top_countries]:
            rows = self.ref_rows[c]
            if inside is not None:
                rows = rows[inside[rows]]
            if not len(rows):
                continue
            kc = kern[rows]
            kc = kc / max(kc.sum(), 1e-300)
            w[rows] = post[c] * ((1 - floor) * kc + floor / len(rows))
        if inside is not None and w.sum() <= 0:  # no reference inside the bounds: clip later
            return self.ref_weights(logpost_row, d2_row, bw, floor, top_countries)
        return w / max(w.sum(), 1e-300)

    def locate(self, logpost_row, d2_row, support=2500, n_cand=300, w=None, score_scale_km=None, bounds=None):
        """Guess maximising the expected GeoGuessr score (scale = maxErrorDistance / 10, World map
        by default) under the reference-panorama mixture, inside the map bounds."""
        scale = score_scale_km or WORLD_SCORE_SCALE_KM
        if w is None:
            w = self.ref_weights(logpost_row, d2_row, bounds=bounds)
        sup = np.argsort(-w)[:support]
        ws = w[sup] / w[sup].sum()
        cand = sup[:n_cand]
        D = haversine_km(self.ref_lat[cand][:, None], self.ref_lng[cand][:, None],
                         self.ref_lat[sup][None, :], self.ref_lng[sup][None, :])
        exp_score = (5000.0 * np.exp(-D / scale)) @ ws
        b = int(np.argmax(exp_score))
        lat, lng = clip_to_bounds(self.ref_lat[cand[b]], self.ref_lng[cand[b]], bounds)
        es = float(exp_score[b])
        if (lat, lng) != (float(self.ref_lat[cand[b]]), float(self.ref_lng[cand[b]])):
            es = float(5000.0 * np.exp(-haversine_km(lat, lng, self.ref_lat[sup], self.ref_lng[sup]) / scale) @ ws)
        return {"lat": lat, "lng": lng, "expected_score": es}

    def region_posterior(self, logpost_row, d2_row, bounds=None):
        """P(admin-1 region) = sum of the reference weights of its panoramas (region kernel
        parameters calibrated separately from the location ones)."""
        rp = self.region_params
        w = self.ref_weights(logpost_row, d2_row, rp.get("bw", 1.0), rp.get("floor", 0.05), bounds=bounds)
        p = np.bincount(self.ref_region, weights=w, minlength=int(self.ref_region.max()) + 1)
        p[0] = 0.0
        return p / max(p.sum(), 1e-300)

    def _index_refs(self):
        from .geo import region_index
        self.ref_rows = [np.flatnonzero(self.ref_yi == c) for c in range(len(self.classes))]
        self.ref_region = region_index(self.ref_lat, self.ref_lng)
        self._inside, self._share = {}, {}

    # ------------------------------------------------------------- persistence
    def save(self, path=MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        arrays = {"ref_yi": self.ref_yi, "ref_lat": self.ref_lat, "ref_lng": self.ref_lng,
                  "ref_emb": self.ref_emb.astype(np.float32), "prior_world": self.prior_world,
                  "prior_uniform": self.prior_uniform, "train_freq": self.train_freq, "lat_hist": self.lat_hist}
        arrays["glm.W"] = self.glm.W
        arrays["glm.mcols"] = self.glm.mcols
        for g in self.groups:
            for k, v in self.gauss[g].to_dict().items():
                arrays[f"g.{g}.{k}"] = v
        np.savez_compressed(os.path.join(path, "model.npz"), **arrays)
        json.dump({"groups": self.groups, "classes": self.classes, "weights": self.weights,
                   "prior_mix": self.prior_mix, "knn_k": self.knn_k,
                   "loc_params": self.loc_params, "region_params": self.region_params},
                  open(os.path.join(path, "model.json"), "w"), indent=1)

    def save_priors(self, path=MODEL_DIR):
        json.dump(self.priors, open(os.path.join(path, PRIORS_FILE), "w"), indent=1)

    @classmethod
    def load(cls, path=MODEL_DIR):
        m = cls()
        meta = json.load(open(os.path.join(path, "model.json")))
        z = np.load(os.path.join(path, "model.npz"))
        m.groups, m.classes, m.weights = meta["groups"], meta["classes"], meta["weights"]
        m.prior_mix, m.knn_k = meta["prior_mix"], meta["knn_k"]
        m.loc_params = meta.get("loc_params", {})
        m.region_params = meta.get("region_params", {})
        pf = os.path.join(path, PRIORS_FILE)
        m.priors = json.load(open(pf)) if os.path.exists(pf) else {}
        for k in ("ref_yi", "ref_lat", "ref_lng", "ref_emb", "prior_world", "prior_uniform", "train_freq", "lat_hist"):
            setattr(m, k, z[k])
        m.ref_emb = m.ref_emb.astype(np.float64)
        m.glm = None
        if "glm.W" in z.files:
            m.glm = SoftmaxGLM()
            m.glm.W, m.glm.mcols = z["glm.W"], z["glm.mcols"].astype(bool)
        m.gauss = {}
        for g in m.groups:
            m.gauss[g] = GroupGaussian.from_dict({k.split(".", 2)[2]: z[k] for k in z.files if k.startswith(f"g.{g}.")})
        m._index_refs()
        return m
