"""
Admin-1 region posterior inside a country (classical discriminant analysis, numpy only).

  P(region | image, country) ∝ prior(region)^w_prior · N(z | mu_region, I)^w_lda · LR_knn(region)^w_knn
                               · P(sun | region)^w_sun

* z: the feature groups of engine.model (every column Gaussianised through its reference quantiles,
  robust-standardised, missing -> median) projected on the K leading within-country
  between-region discriminant directions: generalised eigenvectors of the
  between-region scatter (region means around their country mean, pooled over the countries)
  against the pooled within-region covariance (shrunk towards its diagonal).  Inside z the
  within-region covariance is the identity, so the class densities are N(z | mu_region, I)
  (reduced-rank LDA).  Countries with enough reference panoramas re-rotate the leading
  global subspace towards their own between-region scatter (mix 'own').
* mu_region: the region mean shrunk (empirical Bayes, tau pseudo-panoramas) towards a geographic
  neighbourhood mean (Gaussian kernel on the distance from the region centroid to the reference
  panoramas of the country), itself shrunk towards the country mean, so tiny admin-1 units borrow
  from their neighbours.
* prior: region frequencies in GeoGuessr's public ranked-duel pools (+ the random Street View
  samples with weight beta), smoothed towards the area share of the region (90%, + 10% uniform).
  Extra references (prior_mask False: Street View panoramas sampled 3-40 km from duel locations,
  tools/eval_regions.py --fetch-extra) only describe the region appearance (weight extra_w in the
  means) and do not enter the prior.
* kNN: Gaussian-kernel votes of the k nearest reference panoramas of the country in z, as a
  likelihood ratio against the region's share of the references.
* sun: the analytic solar latitude likelihood (engine.features.solar) averaged over the
  latitudes of each region's reference panoramas.
* region smoothing: a share of the posterior is spread over neighbouring regions (distance kernel
  between region centroids, weighted by the prior) and over the regions that GeoGuessr's regional
  clue cards group together (co-occurrence in the cards' seterraRegionIds, data/calibration/pano_clues.json).
* exponents and smoothing are calibrated on the CALIB rounds by tools/eval_regions.py, which also
  fits the model (data/model/regions.npz).
* country labels that the region raster files under another country (the French overseas
  departments RE, MQ, GP, GF, YT: FR-RE, ...) own those regions (raster_regions); the parent
  country does not.

region_posterior() takes the country probabilities of engine.model.GeoModel and returns
P(region) = sum_c P(c) P(region | c, image) over the likely countries.  location_weights() turns
it into GeoModel.locate() weights over the GeoModel's reference panoramas.

Parameter sets by input kind (as engine.model.GeoModel.for_input): the exponents / prior / smoothing in
regions.npz serve full panoramas ("pano"); data/model/regions_params.json may hold a "views" set calibrated
on the CALIB rounds rendered into the live grid (tools/train_model.py --views), valid only for the
regions.npz it was calibrated with (file hash).  RegionModel.params_for(kind) picks the set.
"""

import json
import math
import os
import sys

import numpy as np

from .geo import _regions, haversine_km, in_bounds, parse_bounds
from .model import file_hash

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "model")
REGIONS_FILE = "regions.npz"
PARAMS_FILE = "regions_params.json"   # parameter sets of other input kinds ({"views": {"params", "regions_npz"}})
SUN_LAT_GRID = np.arange(-60.0, 80.1, 1.0)
QN_LEVELS = 129
# country labels without admin-1 units of their own in the region raster -> their ISO 3166-2 unit
ALIAS = {"GF": "FR-GF", "GP": "FR-GP", "MQ": "FR-MQ", "RE": "FR-RE", "YT": "FR-YT", "SJ": "NO-21"}
# structural parameters chosen by 5-fold out-of-fold CV on the ranked-duel train panoramas (folds by location,
# tools/eval_regions.py --cv --grid); the exponents / prior / smoothing below are starting values, calibrated on CALIB
DEFAULT_PARAMS = {"pre": "qn", "K": 32, "K_big": 128, "shrink": 0.3, "tau": 5.0, "tau_geo": 5.0, "geo_sigma_km": 150.0,
                  "own": 0.0, "own_min": 150, "extra_w": 1.0, "beta": 0.3, "alpha": 0.5, "knn_k": 30, "knn_s": 2.0,
                  "sun_smooth": 5, "weights": {"prior": 1.0, "lda": 0.3, "knn": 0.0, "sun": 0.0},
                  "smooth": 0.0, "smooth_km": 200.0, "smooth_card": 0.0}


def _std_params(X):
    med = np.nanmedian(X, axis=0)
    q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
    sd = np.nanstd(X, axis=0)
    sc = np.where((q3 - q1) > 1e-6 * (np.abs(med) + 1), (q3 - q1) / 1.349, sd)
    sc = np.where(np.isfinite(sc) & (sc > 1e-9), sc, 1.0)
    return np.nan_to_num(med), sc


def _norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _qn_params(X, levels=QN_LEVELS):
    """Per column: the distinct reference quantiles and the normal scores they map to (tied quantiles
    share the mean score), flattened with offsets."""
    lv = np.linspace(0.0, 1.0, levels)
    zq = np.array([_norm_ppf(min(max(v, 0.5 / levels), 1.0 - 0.5 / levels)) for v in lv])
    Q = np.nanpercentile(X, lv * 100.0, axis=0)
    xs, zs, off = [], [], [0]
    for j in range(X.shape[1]):
        q = Q[:, j]
        if not np.isfinite(q).all():
            q = np.zeros(1)
            z = np.zeros(1)
        else:
            q, inv = np.unique(q, return_inverse=True)
            z = np.bincount(inv, weights=zq) / np.bincount(inv)
        xs.append(q)
        zs.append(z)
        off.append(off[-1] + len(q))
    return np.concatenate(xs), np.concatenate(zs), np.array(off)


_CENTROIDS = None


def raster_regions(cc):
    """Region indices of country label cc: its admin-1 units in the raster, or its ALIAS unit
    (RE -> FR-RE) for a label the raster files under another country."""
    _, codes, _, rcc, _ = _regions()
    own = [i for i in range(1, len(codes)) if rcc[i] == cc]
    if own or cc not in ALIAS or ALIAS[cc] not in codes:
        return own
    return [codes.index(ALIAS[cc])]


def region_owner(labels):
    """(n_regions,) country label owning each raster region: the raster country, except the ALIAS units
    of the given labels, which belong to the label (FR-RE -> RE when RE is a label)."""
    _, codes, _, rcc, _ = _regions()
    own = np.array(rcc, dtype=object)
    for c in labels:
        if c in ALIAS and ALIAS[c] in codes and c not in set(rcc[1:]):
            own[codes.index(ALIAS[c])] = c
    return own


def region_centroids():
    """(lat, lng, km2) of every region index of the raster (cos-latitude weighted)."""
    global _CENTROIDS
    if _CENTROIDS is None:
        grid, codes, _, _, res = _regions()
        h, w = grid.shape
        nr = len(codes)
        lat_c = 90.0 - (np.arange(h) + 0.5) * res
        lng_r = np.radians(-180.0 + (np.arange(w) + 0.5) * res)
        cl, sl = np.cos(lng_r), np.sin(lng_r)
        area, sx, sy, sz = np.zeros(nr), np.zeros(nr), np.zeros(nr), np.zeros(nr)
        for y in range(h):
            row = grid[y]
            nz = row > 0
            if not nz.any():
                continue
            r = row[nz]
            cw = math.cos(math.radians(lat_c[y]))
            area += cw * np.bincount(r, minlength=nr)[:nr]
            sx += cw * cw * np.bincount(r, weights=cl[nz], minlength=nr)[:nr]
            sy += cw * cw * np.bincount(r, weights=sl[nz], minlength=nr)[:nr]
            sz += cw * math.sin(math.radians(lat_c[y])) * np.bincount(r, minlength=nr)[:nr]
        _CENTROIDS = (np.degrees(np.arctan2(sz, np.hypot(sx, sy))), np.degrees(np.arctan2(sy, sx)),
                      area * (res * 111.32) ** 2)
    return _CENTROIDS


class RegionModel:
    def __init__(self, params=None):
        self.params = dict(DEFAULT_PARAMS, **(params or {}))
        self.params["weights"] = dict(DEFAULT_PARAMS["weights"], **self.params.get("weights", {}))
        self.qn = None
        self.param_sets = {}

    def params_for(self, kind):
        """Exponents / prior / smoothing for an input kind ("views": screenshots and live captures); the model's
        own parameters ("pano") when the kind has no set of its own."""
        return self.param_sets.get(kind) or self.params

    # ------------------------------------------------------------------ training
    def fit(self, X_by_group, country, region, lat, lng, duel, card_sets=(), prior_mask=None):
        """X_by_group: {group: (n, d)} features of the reference panoramas, country: (n,) labels,
        region: (n,) region indices of engine.geo.region_index (0 or a region the label does not own
        (region_owner) = not used), duel: (n,) True for panoramas of real GeoGuessr pools (ranked duels).
        card_sets: [(country, [region indices], placements)] of GeoGuessr's regional cards.
        prior_mask: (n,) False for references that only describe the region appearance (not counted
        in the region prior)."""
        p = self.params
        _, codes, _, _, _ = _regions()
        self.groups = list(X_by_group)
        self.dims = [int(X_by_group[g].shape[1]) for g in self.groups]
        X = np.concatenate([np.asarray(X_by_group[g], np.float64) for g in self.groups], axis=1)
        country, region = np.asarray(country), np.asarray(region)
        lat, lng, duel = np.asarray(lat, float), np.asarray(lng, float), np.asarray(duel, bool)
        pm = np.ones(len(country), bool) if prior_mask is None else np.asarray(prior_mask, bool)
        rcc = region_owner(sorted(set(country)))
        ok = (region > 0) & (rcc[region] == country)
        X, country, region, lat, lng, duel, pm = X[ok], country[ok], region[ok], lat[ok], lng[ok], duel[ok], pm[ok]
        self.qn = _qn_params(X) if p.get("pre") == "qn" else None
        X = self._pre(X)
        self.med, self.sc = _std_params(X)
        Z = self.transform(X, pre=False)
        Kb = max(p["K"], p["K_big"])
        self.proj, self.proj_ev = self._projection(Z, country, region, Kb, p["shrink"])
        E = Z @ self.proj
        clat, clng, km2 = region_centroids()
        self.countries = sorted(set(country))
        regs, mus, ndu, shares, hist, cnt, call, refs, ref_reg, ref_rows = [], [], [], [], [], [], [], [], [], []
        ref_off, reg_off = [0], [0]
        self.rot = {}
        for c in self.countries:
            rows = np.flatnonzero(country == c)
            rc = np.flatnonzero(rcc == c)
            pos = np.full(len(codes), -1)
            pos[rc] = np.arange(len(rc))
            ri = pos[region[rows]]
            wr = np.where(pm[rows], 1.0, p.get("extra_w", 1.0))  # weight of the appearance-only references
            nall = np.bincount(ri, minlength=len(rc)).astype(float)
            n = np.bincount(ri[pm[rows]], minlength=len(rc)).astype(float)
            nd = np.bincount(ri[duel[rows] & pm[rows]], minlength=len(rc)).astype(float)
            U = self._rotation(E[rows], ri, len(rc)) if p["own"] > 0 and len(rows) >= p["own_min"] else None
            e = E[rows][:, :p["K"]] if U is None else E[rows] @ U
            if U is not None:
                self.rot[c] = U
            mc = e.mean(0)
            if p["geo_sigma_km"] > 0:
                d = haversine_km(clat[rc][:, None], clng[rc][:, None], lat[rows][None], lng[rows][None])
                kg = np.exp(-0.5 * (d / p["geo_sigma_km"]) ** 2) * wr[None]
                tgt = (kg @ e + p["tau_geo"] * mc) / (kg.sum(1) + p["tau_geo"])[:, None]
            else:
                tgt = np.repeat(mc[None], len(rc), 0)
            S = np.zeros((len(rc), e.shape[1]))
            np.add.at(S, ri, e * wr[:, None])
            mus.append((S + p["tau"] * tgt) / (np.bincount(ri, weights=wr, minlength=len(rc)) + p["tau"])[:, None])
            ndu.append(nd)
            shares.append(0.9 * km2[rc] / max(km2[rc].sum(), 1e-12) + 0.1 / len(rc))
            H = np.zeros((len(rc), len(SUN_LAT_GRID)))
            b = np.clip(np.round(lat[rows] - SUN_LAT_GRID[0]).astype(int), 0, len(SUN_LAT_GRID) - 1)
            np.add.at(H, (ri, b), 1.0)
            hist.append(H)
            regs.append(rc)
            cnt.append(n)
            call.append(nall)
            refs.append(e)
            ref_reg.append(ri)
            ref_rows.append(rows)
            ref_off.append(ref_off[-1] + len(rows))
            reg_off.append(reg_off[-1] + len(rc))
        self.reg_idx = np.concatenate(regs)
        self.reg_lat, self.reg_lng = clat[self.reg_idx], clng[self.reg_idx]
        self.reg_off = np.array(reg_off)
        self._card_cooccurrence(card_sets)
        self.ref_off = np.array(ref_off)
        self.mu = np.concatenate(mus)
        self.count = np.concatenate(cnt)
        self.count_duel = np.concatenate(ndu)
        self.count_all = np.concatenate(call)
        self.share = np.concatenate(shares)
        self.hist = np.concatenate(hist).astype(np.float32)
        self.ref_emb = np.concatenate(refs)
        self.ref_reg = np.concatenate(ref_reg).astype(np.int32)
        rr = np.concatenate(ref_rows)
        self.ref_lat, self.ref_lng, self.ref_duel = lat[rr], lng[rr], duel[rr]
        self._index()
        return self

    @staticmethod
    def _projection(Z, country, region, K, shrink):
        ur, inv = np.unique(region, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        S = np.zeros((len(ur), Z.shape[1]))
        np.add.at(S, inv, Z)
        M = S / cnt[:, None]
        R = Z - M[inv]
        Sw = R.T @ R / max(len(Z) - len(ur), 1)
        Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
        uc, cinv = np.unique(country, return_inverse=True)
        Sc = np.zeros((len(uc), Z.shape[1]))
        np.add.at(Sc, cinv, Z)
        Mc = Sc / np.bincount(cinv)[:, None]
        first = np.zeros(len(ur), int)
        first[inv] = cinv
        Dm = M - Mc[first]
        Sb = (Dm * cnt[:, None]).T @ Dm / cnt.sum()
        w, V = np.linalg.eigh(Sw)
        Wh = V / np.sqrt(np.maximum(w, 1e-4 * w.max()))
        ev, U = np.linalg.eigh(Wh.T @ Sb @ Wh)
        o = np.argsort(-ev)[:K]
        return Wh @ U[:, o], ev[o]

    def _rotation(self, e, ri, R):
        """Directions (K_big, K) of a country: its own between-region scatter mixed with the global
        one (diag of the global eigenvalues) inside the leading global subspace."""
        p = self.params
        cnt = np.bincount(ri, minlength=R).astype(float)
        S = np.zeros((R, e.shape[1]))
        np.add.at(S, ri, e)
        has = cnt > 0
        Dm = S[has] / cnt[has, None] - e.mean(0)
        B = (Dm * cnt[has, None]).T @ Dm / len(e)
        M = p["own"] * B + (1 - p["own"]) * np.diag(self.proj_ev)
        ev, U = np.linalg.eigh(0.5 * (M + M.T))
        return U[:, np.argsort(-ev)[:p["K"]]]

    def _card_cooccurrence(self, card_sets):
        co, off = [], [0]
        by = {}
        for cc, regs, n in card_sets:
            by.setdefault(cc, []).append((regs, n))
        for i, cc in enumerate(self.countries):
            rc = self.reg_idx[self.reg_off[i]:self.reg_off[i + 1]]
            pos = {r: k for k, r in enumerate(rc)}
            A = np.zeros((len(rc), len(rc)), np.float32)
            for regs, n in by.get(cc, []):
                ix = [pos[r] for r in regs if r in pos]
                if len(ix) > 1:
                    A[np.ix_(ix, ix)] += n / len(ix)
            if A.any():
                co.append(A.ravel())
            off.append(off[-1] + (A.size if A.any() else 0))
        self.card_co = np.concatenate(co) if co else np.zeros(0, np.float32)
        self.card_off = np.array(off)

    def _index(self):
        self.cidx = {c: i for i, c in enumerate(self.countries)}
        self._kern = {}

    def prior(self, cc, beta=None, alpha=None):
        """P(region | country): duel-pool counts + beta x the random-sample counts, smoothed towards the
        area shares with alpha x sqrt(total count) pseudo-counts."""
        p = self.params
        beta = p["beta"] if beta is None else beta
        alpha = p["alpha"] if alpha is None else alpha
        i = self.cidx[cc]
        sl = slice(self.reg_off[i], self.reg_off[i + 1])
        n, nd = self.count[sl], self.count_duel[sl]
        c = nd + beta * (n - nd)
        q = c + alpha * max(1.0, math.sqrt(c.sum())) * self.share[sl]
        return q / max(q.sum(), 1e-300)

    def _pre(self, X):
        """Quantile Gaussianisation of every column (NaN stays NaN)."""
        X = np.asarray(X, np.float64)
        if self.qn is None:
            return X
        xs, zs, off = self.qn
        Y = np.zeros_like(X)
        for j in range(X.shape[1]):
            a, b = off[j], off[j + 1]
            if b - a > 1:
                Y[:, j] = np.interp(X[:, j], xs[a:b], zs[a:b])
        Y[np.isnan(X)] = np.nan
        return Y

    def transform(self, X, pre=True):
        X = self._pre(X) if pre else np.asarray(X, np.float64)
        return np.nan_to_num(np.clip((X - self.med) / self.sc, -8, 8))

    def embed_all(self, X_by_group):
        """(n, K_big) global discriminant coordinates of feature rows (a missing group counts as missing values)."""
        n = max(len(np.atleast_2d(v)) for v in X_by_group.values()) if X_by_group else 1
        X = np.concatenate([np.atleast_2d(np.asarray(X_by_group[g], np.float64)) if g in X_by_group
                            else np.full((n, d), np.nan) for g, d in zip(self.groups, self.dims)], axis=1)
        return self.transform(X) @ self.proj

    def embed(self, X_by_group, cc, E=None):
        """(n, K) discriminant coordinates of feature rows in country cc's space (E: embed_all rows)."""
        E = self.embed_all(X_by_group) if E is None else E
        U = self.rot.get(cc)
        return E[:, :self.params["K"]] if U is None else E @ U

    # ---------------------------------------------------------------- evidence
    def regions_of(self, cc):
        i = self.cidx[cc]
        return self.reg_idx[self.reg_off[i]:self.reg_off[i + 1]]

    def components(self, e, cc, sun_mix=None):
        """Raw (n, R_c) terms of country cc for embedded rows e: LDA log-density, kNN kernel votes and
        the sun log-likelihood (with sun_mix)."""
        p = self.params
        i = self.cidx[cc]
        sl = slice(self.reg_off[i], self.reg_off[i + 1])
        mu = self.mu[sl]
        a = -0.5 * ((e[:, None, :] - mu[None]) ** 2).sum(-1)
        out = {"lda": a - a.max(1, keepdims=True)}
        rs = slice(self.ref_off[i], self.ref_off[i + 1])
        re, rr = self.ref_emb[rs], self.ref_reg[rs]
        V = np.zeros((len(e), len(mu)))
        k = min(int(p["knn_k"]), len(re))
        if k >= 2:
            if ("n2", cc) not in self._kern:
                self._kern[("n2", cc)] = (re ** 2).sum(1)
            d2 = np.maximum((e ** 2).sum(1)[:, None] + self._kern[("n2", cc)][None] - 2 * e @ re.T, 0.0)
            idx = np.argpartition(d2, k - 1, 1)[:, :k]
            dk = np.take_along_axis(d2, idx, 1)
            w = np.exp(-0.5 * dk / np.maximum(np.median(dk, 1, keepdims=True), 1e-6))
            cell = (np.arange(len(e))[:, None] * len(mu) + rr[idx]).ravel()
            V = np.bincount(cell, weights=w.ravel(), minlength=V.size).reshape(V.shape)
        out["knn_votes"] = V
        if sun_mix is not None:
            out["sun"] = np.log(np.maximum(np.atleast_2d(sun_mix) @ self.sun_hist(cc).T, 0.0) + 1e-9)
        return out

    def sun_hist(self, cc):
        """(R_c, len(SUN_LAT_GRID)) latitude distribution of each region's references (smoothed; regions
        without references borrow the country's)."""
        if ("sun", cc) not in self._kern:
            i = self.cidx[cc]
            H = self.hist[self.reg_off[i]:self.reg_off[i + 1]].astype(np.float64)
            m = int(self.params["sun_smooth"])
            if m > 1:
                H = np.apply_along_axis(lambda h: np.convolve(h, np.ones(m) / m, mode="same"), 1, H)
            tot = H.sum(0) + 1e-3
            H = H + tot[None] / tot.sum()
            self._kern[("sun", cc)] = H / H.sum(1, keepdims=True)
        return self._kern[("sun", cc)]

    def combine(self, comp, cc, params=None):
        """(n, R_c) P(region | country cc) from the raw components; params overrides self.params
        (weights, beta, alpha, knn_s, smooth, smooth_km)."""
        p = dict(self.params, **(params or {}))
        w = p["weights"]
        prior = self.prior(cc, p["beta"], p["alpha"])
        lp = np.repeat(w.get("prior", 1.0) * np.log(prior)[None], len(comp["lda"]), 0)
        if w.get("lda"):
            lp = lp + w["lda"] * comp["lda"]
        if w.get("knn"):  # vote share against the reference share of the region
            V = comp["knn_votes"]
            s = p["knn_s"]
            i = self.cidx[cc]
            base = self.count_all[self.reg_off[i]:self.reg_off[i + 1]] + 0.1
            base = base / base.sum()
            lp = lp + w["knn"] * (np.log((V + s * base[None]) / (V.sum(1, keepdims=True) + s)) - np.log(base)[None])
        if w.get("sun") and "sun" in comp:
            lp = lp + w["sun"] * comp["sun"]
        lp = lp - lp.max(1, keepdims=True)
        P = np.exp(lp)
        P /= P.sum(1, keepdims=True)
        if p["smooth"] > 0 and P.shape[1] > 1:
            P = (1 - p["smooth"]) * P + p["smooth"] * P @ self.smoothing_kernel(cc, p["smooth_km"], prior)
        C = self.card_kernel(cc, prior) if p["smooth_card"] > 0 else None
        if C is not None:
            P = (1 - p["smooth_card"]) * P + p["smooth_card"] * P @ C
        return P

    def card_kernel(self, cc, prior):
        """(R, R) row-stochastic: region r -> the regions GeoGuessr's regional cards put together with r
        (co-occurrence in the cards' region sets, weighted by placements and the prior); None without cards."""
        i = self.cidx[cc]
        a, b = self.card_off[i], self.card_off[i + 1]
        if b <= a:
            return None
        R = self.reg_off[i + 1] - self.reg_off[i]
        A = self.card_co[a:b].reshape(R, R).astype(np.float64) * prior[None]
        A[np.diag_indices(R)] += 1e-3 * prior
        return A / A.sum(1, keepdims=True)

    def smoothing_kernel(self, cc, km, prior):
        """(R, R) row-stochastic: region r -> region r' with weight prior(r') exp(-d^2 / 2 km^2), d the
        distance between the centroids."""
        if ("d", cc) not in self._kern:
            i = self.cidx[cc]
            sl = slice(self.reg_off[i], self.reg_off[i + 1])
            la, ln = self.reg_lat[sl], self.reg_lng[sl]
            self._kern[("d", cc)] = haversine_km(la[:, None], ln[:, None], la[None], ln[None])
        K = np.exp(-0.5 * (self._kern[("d", cc)] / km) ** 2) * prior[None]
        return K / K.sum(1, keepdims=True)

    def areas(self, cc, min_refs=25):
        """Admin-1 regions of cc merged into areas with at least min_refs reference panoramas: the area
        with the fewest references joins the nearest one (reference-weighted centroids) until all
        have enough.  Returns (R_c,) area labels (0..A-1)."""
        i = self.cidx[cc]
        sl = slice(self.reg_off[i], self.reg_off[i + 1])
        n = self.count[sl] + 1e-3
        la, ln = np.radians(self.reg_lat[sl]), np.radians(self.reg_lng[sl])
        v = np.stack([np.cos(la) * np.cos(ln), np.cos(la) * np.sin(ln), np.sin(la)], 1)
        lab = np.arange(len(n))
        while True:
            ids = np.unique(lab)
            if len(ids) < 2:
                break
            cnt = np.array([n[lab == a].sum() for a in ids])
            k = int(np.argmin(cnt))
            if cnt[k] >= min_refs:
                break
            cen = np.array([(v[lab == a] * n[lab == a, None]).sum(0) for a in ids])
            cen /= np.linalg.norm(cen, axis=1, keepdims=True)
            d = cen @ cen[k]
            d[k] = -2.0
            lab[lab == ids[k]] = ids[int(np.argmax(d))]
        return np.unique(lab, return_inverse=True)[1]

    def country_posterior(self, X_by_group, cc, sun_mix=None, E=None, params=None):
        """(region indices of cc, (n, R_c) P(region | cc, image)); None if cc has no region model.  params:
        overrides of self.params (e.g. params_for("views"))."""
        if cc not in self.cidx:
            return None
        e = self.embed(X_by_group, cc, E)
        if sun_mix is None and (params or self.params)["weights"].get("sun"):
            sun_mix = sun_mixture(X_by_group)
        return self.regions_of(cc), self.combine(self.components(e, cc, sun_mix), cc, params)

    # ------------------------------------------------------------- persistence
    def save(self, path=MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        _, codes, _, _, _ = _regions()
        rot_c = sorted(self.rot)
        np.savez_compressed(
            os.path.join(path, REGIONS_FILE),
            meta=np.array(json.dumps({"params": self.params, "groups": self.groups, "dims": self.dims,
                                      "countries": self.countries, "rot_countries": rot_c})),
            med=self.med, sc=self.sc, proj=self.proj.astype(np.float32), proj_ev=self.proj_ev,
            qn_x=self.qn[0] if self.qn else np.zeros(0), qn_z=self.qn[1] if self.qn else np.zeros(0),
            qn_off=self.qn[2] if self.qn else np.zeros(0, int),
            rot=np.array([self.rot[c] for c in rot_c], np.float32).reshape(len(rot_c), len(self.proj_ev),
                                                                         int(self.params["K"])),
            reg_code=np.array([codes[i] for i in self.reg_idx]), reg_lat=self.reg_lat, reg_lng=self.reg_lng,
            reg_off=self.reg_off, ref_off=self.ref_off, card_co=self.card_co, card_off=self.card_off,
            mu=self.mu.astype(np.float32), count=self.count, count_duel=self.count_duel, count_all=self.count_all,
            share=self.share, hist=self.hist,
            ref_emb=self.ref_emb.astype(np.float32), ref_reg=self.ref_reg, ref_lat=self.ref_lat.astype(np.float32),
            ref_lng=self.ref_lng.astype(np.float32), ref_duel=self.ref_duel)

    @classmethod
    def load(cls, path=MODEL_DIR):
        f = path if path.endswith(".npz") else os.path.join(path, REGIONS_FILE)
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        m = cls(meta["params"])
        m.groups, m.dims, m.countries = meta["groups"], meta["dims"], meta["countries"]
        m.med, m.sc, m.proj_ev = z["med"], z["sc"], z["proj_ev"]
        m.qn = (z["qn_x"], z["qn_z"], z["qn_off"]) if "qn_off" in z.files and len(z["qn_off"]) else None
        m.proj = z["proj"].astype(np.float64)
        m.rot = {c: z["rot"][i].astype(np.float64) for i, c in enumerate(meta["rot_countries"])}
        _, codes, _, _, _ = _regions()
        cpos = {c: i for i, c in enumerate(codes)}
        m.reg_idx = np.array([cpos.get(str(c), 0) for c in z["reg_code"]])
        for k in ("reg_lat", "reg_lng", "reg_off", "ref_off", "count", "count_duel", "share", "hist", "ref_reg",
                  "card_co", "card_off", "ref_duel"):
            setattr(m, k, z[k])
        m.count_all = z["count_all"] if "count_all" in z.files else m.count
        m.mu, m.ref_emb = z["mu"].astype(np.float64), z["ref_emb"].astype(np.float64)
        m.ref_lat, m.ref_lng = z["ref_lat"].astype(np.float64), z["ref_lng"].astype(np.float64)
        m._index()
        m.param_sets = load_param_sets(f)
        return m


def load_param_sets(npz_path):
    """{kind: params} of PARAMS_FILE next to npz_path whose regions_npz hash matches that file (a set
    calibrated for another regions.npz is dropped with a warning)."""
    f = os.path.join(os.path.dirname(npz_path), PARAMS_FILE)
    if not os.path.exists(f):
        return {}
    try:
        sets = json.load(open(f))
    except ValueError:
        return {}
    h, out = None, {}
    for kind, e in sets.items():
        if not isinstance(e, dict) or not e.get("params"):
            continue
        h = h or file_hash(npz_path)
        if e.get("regions_npz") != h:
            sys.stderr.write("%s: the %s region parameters were calibrated for another %s - %s inputs use the "
                             "panorama parameters (run tools/train_model.py --views --save)\n"
                             % (PARAMS_FILE, kind, REGIONS_FILE, kind))
            continue
        p = dict(DEFAULT_PARAMS, **e["params"])
        p["weights"] = dict(DEFAULT_PARAMS["weights"], **e["params"].get("weights", {}))
        out[kind] = p
    return out


def sun_mixture(X_by_group):
    """(n, len(SUN_LAT_GRID)) conf * L(lat | sun) + (1 - conf) from the solar features (ones without
    a usable sun), the latitude likelihood normalised to mean 1."""
    from .features import solar
    n = solar.FEATURE_NAMES
    Xs = np.atleast_2d(np.asarray(X_by_group["solar"], np.float64)) if "solar" in X_by_group else None
    if Xs is None:
        return None
    out = np.ones((len(Xs), len(SUN_LAT_GRID)))
    for i, x in enumerate(Xs):
        c, s, el, conf = (x[n.index(k)] for k in ("sun_az_cos", "sun_az_sin", "sun_el", "sun_conf"))
        if not (np.isfinite(c) and np.isfinite(s) and np.isfinite(el) and np.isfinite(conf)):
            continue
        L = np.maximum(np.nan_to_num(np.asarray(
            solar.latitude_likelihood(math.degrees(math.atan2(s, c)) % 360.0, el, SUN_LAT_GRID), float)), 0.0)
        conf = min(max(float(conf), 0.0), 1.0)
        out[i] = conf * L / max(L.mean(), 1e-12) + (1.0 - conf)
    return out


_MODEL = None


def load_region_model(path=MODEL_DIR):
    """Cached RegionModel, None when data/model/regions.npz is missing."""
    global _MODEL
    if _MODEL is None:
        f = path if path.endswith(".npz") else os.path.join(path, REGIONS_FILE)
        if not os.path.exists(f):
            return None
        _MODEL = RegionModel.load(path)
    return _MODEL


def _country_items(country_probs, classes=None):
    if isinstance(country_probs, dict):
        return [(str(c), float(p)) for c, p in country_probs.items()]
    return [(str(c), float(p)) for c, p in zip(classes, np.ravel(country_probs))]


def _bounds_mask(model, cc, regs, bounds, margin=0.5):
    """Regions of cc with their centroid or a reference panorama inside the map bounds (+ margin deg)."""
    clat, clng = region_centroids()[:2] if model is None or cc not in model.cidx else (None, None)
    if clat is not None:
        return in_bounds(clat[regs], clng[regs], bounds, margin)
    i = model.cidx[cc]
    sl, rs = slice(model.reg_off[i], model.reg_off[i + 1]), slice(model.ref_off[i], model.ref_off[i + 1])
    m = in_bounds(model.reg_lat[sl], model.reg_lng[sl], bounds, margin)
    ins = in_bounds(model.ref_lat[rs], model.ref_lng[rs], bounds, margin)
    m[np.unique(model.ref_reg[rs][ins])] = True
    return m


def region_posterior(features_by_group, country_probs, model=None, classes=None, top_countries=12,
                     min_prob=1e-4, by_country=False, bounds=None, params=None):
    """P(admin-1 region) = sum_c P(c) P(region | c, image) over the top countries.

    features_by_group: {group: (d,) or (1, d)} feature vectors of one image (engine.features);
    country_probs: {country code: probability} or a (C,) vector aligned with classes (e.g.
    exp(GeoModel.combine(...))[0] with GeoModel.classes); model: RegionModel (default
    data/model/regions.npz).  Countries without a region model keep their mass on their raster
    regions (raster_regions) in proportion to area.  Returns {ISO 3166-2 code: probability}: the
    mixture over the top_countries countries with P(country) >= min_prob, normalised by their total
    probability (a country without any region keeps its share outside the result, which then sums to
    less than 1); by_country=True: {country: {code: P(region | country)}}; by_country="both": the
    pair (mixture, per country) - the per-country form feeds location_weights.  bounds: map bounds
    (engine.geo.parse_bounds forms); regions with neither their centroid nor a reference panorama
    inside get no mass (unless that empties the country).  params: overrides of the model's exponents /
    prior / smoothing (RegionModel.params_for(kind) for a non-panorama input)."""
    model = model or load_region_model()
    _, codes, _, _, _ = _regions()
    items = sorted(_country_items(country_probs, classes), key=lambda t: -t[1])
    items = [t for t in items[:top_countries] if t[1] >= min_prob] or items[:1]
    w = (params or model.params)["weights"] if model is not None else {}
    sun = sun_mixture(features_by_group) if model is not None and w.get("sun") else None
    E = model.embed_all(features_by_group) if model is not None else None
    out, per = {}, {}
    for cc, pc in items:
        res = model.country_posterior(features_by_group, cc, sun, E, params) if model is not None else None
        if res is None:
            regs = raster_regions(cc)
            if not regs:
                continue
            km2 = region_centroids()[2][regs]
            pr = km2 / max(km2.sum(), 1e-12)
        else:
            regs, pr = res[0], res[1][0]
        if parse_bounds(bounds) is not None:
            m = _bounds_mask(model, cc, np.asarray(regs), bounds)
            if m.any() and pr[m].sum() > 0:
                pr = np.where(m, pr, 0.0) / pr[m].sum()
        per[cc] = {codes[r]: float(q) for r, q in zip(regs, pr)}
        for r, q in zip(regs, pr):
            out[codes[r]] = out.get(codes[r], 0.0) + pc * float(q)
    tot = sum(pc for _, pc in items)
    mix = {k: v / tot for k, v in out.items()} if tot > 0 else out
    if by_country == "both":
        return mix, per
    return per if by_country else mix


def region_vector(probs):
    """{code: p} -> (n_regions,) array over engine.geo region indices (the GeoModel.region_posterior
    layout; index 0 = no region)."""
    _, codes, _, _, _ = _regions()
    pos = {c: i for i, c in enumerate(codes)}
    v = np.zeros(len(codes))
    for code, p in probs.items():
        i = pos.get(code, 0)
        if i:
            v[i] += p
    return v


def region_list(probs, limit=400):
    """[{code, name, country, probability}] sorted by probability, the format of Locator.analyze()
    'regions' (engine.hints.region_probs_for reads it)."""
    _, codes, names, rcc, _ = _regions()
    pos = {c: i for i, c in enumerate(codes)}
    out = []
    for code, p in sorted(probs.items(), key=lambda t: -t[1])[:limit]:
        if p <= 0:
            break
        i = pos.get(code, 0)
        out.append({"code": code, "name": names[i], "country": rcc[i], "probability": round(float(p), 5)})
    return out


def location_weights(geo_model, logpost_row, d2_row, region_probs, top_countries=12, bw=None, floor=None,
                     bounds=None):
    """GeoModel.locate() weights over its reference panoramas: P(country) · P(region | country), the
    region's mass spread over its references by the GeoModel location kernel (bandwidth bw x the
    median of the 200 nearest distances, plus a uniform floor share inside the region).
    region_probs: region_posterior(..., by_country=True).  Countries without region probabilities
    (or whose likely regions have no reference inside the bounds) keep GeoModel.ref_weights."""
    lp = geo_model.loc_params
    bw = lp.get("bw", 1.0) if bw is None else bw
    floor = lp.get("floor", 1e-3) if floor is None else floor
    base = geo_model.ref_weights(logpost_row, d2_row, bw, floor, top_countries, bounds=bounds)
    _, codes, _, _, _ = _regions()
    post = np.exp(logpost_row)
    k = min(200, len(d2_row))
    h2 = max(float(np.median(np.partition(d2_row, k - 1)[:k])), 1e-6) * bw
    kern = np.exp(-0.5 * (d2_row - d2_row.min()) / h2)
    inside = geo_model.refs_inside(bounds) if parse_bounds(bounds) else None
    w = np.array(base)
    for c in np.argsort(-post)[:top_countries]:
        rp = (region_probs or {}).get(geo_model.classes[c])
        rows = geo_model.ref_rows[c]
        if inside is not None:
            rows = rows[inside[rows]]
        if not rp or not len(rows):
            continue
        rr = geo_model.ref_region[rows]
        wc = np.zeros(len(rows))
        for r in np.unique(rr):
            q = rp.get(codes[r], 0.0) if r else 0.0
            if q <= 0:
                continue
            m = rr == r
            kc = kern[rows[m]]
            wc[m] = q * ((1 - floor) * kc / max(kc.sum(), 1e-300) + floor / m.sum())
        bs = base[rows].sum()
        if wc.sum() <= 0 or bs <= 0:
            continue
        w[geo_model.ref_rows[c]] = 0.0
        w[rows] = wc / wc.sum() * bs
    return w / max(w.sum(), 1e-300)
