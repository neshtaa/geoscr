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
  GeoGuessr score 5000·exp(-d / 1491.7 km).
"""

import json
import math
import os

import numpy as np

from .geo import WORLD_SCORE_SCALE_KM, haversine_km

SUN_LAT_GRID = np.arange(-60.0, 80.1, 1.0)
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "model")


# ----------------------------------------------------------------------------- utils
def _logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def _softmax(a):
    a = a - a.max(-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(-1, keepdims=True)


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

    def prior(self):
        return self.prior_mix * self.prior_world + (1 - self.prior_mix) * self.prior_uniform

    def combine(self, ev, weights=None, prior_mix=None):
        w = weights or self.weights
        pm = self.prior_mix if prior_mix is None else prior_mix
        prior = pm * self.prior_world + (1 - pm) * self.prior_uniform
        lp = np.log(prior)[None, :] + 0.0
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

    # ---------------------------------------------------------------- location
    def locate(self, logpost_row, d2_row, top_countries=12, support=2500, n_cand=300):
        """Mixture over reference panoramas -> guess maximising expected GeoGuessr score."""
        post = np.exp(logpost_row)
        C = len(self.classes)
        w = np.zeros(len(self.ref_yi))
        h2 = max(float(np.median(np.sort(d2_row)[:200])), 1e-6)
        kern = np.exp(-0.5 * (d2_row - d2_row.min()) / h2)
        for c in np.argsort(-post)[:top_countries]:
            rows = self.ref_yi == c
            if not rows.any():
                continue
            kc = kern[rows] + 1e-3 * kern[rows].max()
            w[rows] = post[c] * kc / kc.sum()
        sup = np.argsort(-w)[:support]
        ws = w[sup] / w[sup].sum()
        cand = sup[:n_cand]
        D = haversine_km(self.ref_lat[cand][:, None], self.ref_lng[cand][:, None],
                         self.ref_lat[sup][None, :], self.ref_lng[sup][None, :])
        exp_score = (5000.0 * np.exp(-D / WORLD_SCORE_SCALE_KM)) @ ws
        b = int(np.argmax(exp_score))
        return {"lat": float(self.ref_lat[cand[b]]), "lng": float(self.ref_lng[cand[b]]),
                "expected_score": float(exp_score[b]),
                "support": [(float(self.ref_lat[i]), float(self.ref_lng[i]), float(ws[j])) for j, i in enumerate(sup[:50])]}

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
                   "prior_mix": self.prior_mix, "knn_k": self.knn_k},
                  open(os.path.join(path, "model.json"), "w"), indent=1)

    @classmethod
    def load(cls, path=MODEL_DIR):
        m = cls()
        meta = json.load(open(os.path.join(path, "model.json")))
        z = np.load(os.path.join(path, "model.npz"))
        m.groups, m.classes, m.weights = meta["groups"], meta["classes"], meta["weights"]
        m.prior_mix, m.knn_k = meta["prior_mix"], meta["knn_k"]
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
        return m
