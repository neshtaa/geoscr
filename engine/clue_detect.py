"""
GeoGuessr clue-card detectors: classical sliding-window template detectors trained on GeoGuessr's own
clue placements (tools/build_clue_detectors.py) and their use as country / region evidence and hints.

Every placement says "card c is visible in panorama P at heading h, pitch p, zoom z": a labelled view.
A view is the central square of the 16:9 GeoGuessr view, tan(fov / 2) = 2^(1 - zoom) * 9 / 16.

Descriptor (HOG-style, Dalal & Triggs / Felzenszwalb): every zoom channel resamples the elevation band
of its placements from the sphere so that one cell = fov / CELLS degrees = CELL_PX pixels; per cell
9 unsigned gradient-orientation bins (block-normalised, truncated), the gradient energy and the mean
L*a*b* colour.  A window = CELLS x CELLS cells (the view of one placement), stride one cell, x wraps.

Detector (exemplar-LDA against a shared background, Hariharan, Malik & Ramanan 2012): background
windows of training panoramas give a mean per window row (elevation) and a pooled covariance S; the
template of a card is the whitened mean deviation of its placement windows from the row means,
u = T (mu_card - mu_row) / |.|, T = top-k eigen-whitening of the shrunk S; the score of a window,
u . T (x - mu_row), is ~N(0, 1) on background windows.  Detection keeps per detector the best window
inside the pitch range of its placements.  Folded into one matrix product per channel:
score = X G - O[row], G = T' U, O = mu_row T' U.

Detections become (fitted on out-of-fold detections of training panoramas):
  * z~ = max score standardised per detector;
  * P(card present | country, detections) = logistic(a0 + a1 logit(card frequency) + a2 z~);
  * P(the best window shows the card | country, detections) = logistic(b0 + b1 logit(freq) + b2 z~)
    (present and within half a window of one of its placements): a card's direction is shown only
    when this reaches the threshold calibrated on CALIB (off when no threshold is precise enough);
  * a per-country log-likelihood: Gaussian class densities of the vector of all z~ with a shared
    shrunk covariance (engine.model.GroupGaussian), i.e. linear discriminants z A_c - b_c; only used
    when the searched windows are (nearly) all visible - fewer windows lower every max;
  * a soft region update (Jeffrey's rule) for cards tied to admin-1 regions (seterraRegionIds).
"""

import json
import math
import os
import sys

import numpy as np
from PIL import Image

from .features.texture import _lab

DETECTORS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "model",
                              "clue_detectors.npz")
ASPECT = 9.0 / 16.0  # the window = the central square of a 16:9 GeoGuessr view
CELLS = 6            # cells per window side
CELL_PX = 8          # pixels per cell on the channel level
NBINS = 9            # unsigned orientation bins
CF = NBINS + 4       # per cell: orientations, gradient energy, L, a, b
DIM = CELLS * CELLS * CF
MIN_COVER = 0.8      # share of valid (mask) pixels a window needs
MIN_EVIDENCE_COVERAGE = 0.9  # share of the searched windows that must be visible for the country evidence
# zoom channels (placement zooms 1.5 / 2 / 3.5; 2.5 -> 2, 3 -> 3.5) with the pitch range of the window
# centres (~1st..99th percentile of the placement pitches)
CHANNELS = ({"zoom": 1.5, "pitch": [-18.0, 18.0]},
            {"zoom": 2.0, "pitch": [-20.0, 17.0]},
            {"zoom": 3.5, "pitch": [-20.0, 17.0]})
# uint8 quantisation of the cell features (cache and runtime): value = lo + q / 255 * (hi - lo)
Q_LO = np.array([0.0] * NBINS + [0.0, 0.0, -2.0, -2.0], np.float32)
Q_HI = np.array([0.5] * NBINS + [5.0, 1.0, 2.0, 2.0], np.float32)


def crop_fov(zoom):
    """Field of view (deg) of the square window for a GeoGuessr zoom."""
    return math.degrees(2.0 * math.atan(2.0 ** (1.0 - float(zoom)) * ASPECT))


def channel_of(zoom, channels=CHANNELS):
    zs = np.array([c["zoom"] for c in channels])
    return int(np.argmin(np.abs(zs - float(zoom))))


def grid(ch):
    """Cell grid of a channel: cell size (deg), cells around, top elevation, cell rows, window rows."""
    fov = crop_fov(ch["zoom"])
    n_az = int(round(360.0 / (fov / CELLS)))
    cell = 360.0 / n_az
    lo, hi = ch["pitch"]
    n_wrows = int(math.ceil((hi - lo) / cell)) + 1
    return {"fov": fov, "cell": cell, "n_az": n_az, "top": hi + CELLS / 2.0 * cell,
            "n_rows": n_wrows + CELLS - 1, "n_wrows": n_wrows}


def window_pitch(g, i):
    """Elevation (deg) of the centre of window row i."""
    return g["top"] - (np.asarray(i) + CELLS / 2.0) * g["cell"]


def window_yaw(g, j):
    """Relative longitude (deg) of the centre of window column j."""
    return (-180.0 + (np.asarray(j) + CELLS / 2.0) * g["cell"] + 180.0) % 360.0 - 180.0


def window_at(g, yaw, pitch):
    """(row, col) of the window centred nearest to relative yaw / pitch (deg); row may be out of range."""
    i = int(round((g["top"] - pitch) / g["cell"] - CELLS / 2.0))
    j = int(round((yaw + 180.0) / g["cell"] - CELLS / 2.0)) % g["n_az"]
    return i, j


# ----------------------------------------------------------------------------- cells
def _box3(E):
    """3 x 3 box mean, x wraps, y clamps."""
    P = np.pad(E, ((1, 1), (0, 0)), mode="edge")
    V = P[:-2] + P[1:-1] + P[2:]
    return (V + np.roll(V, 1, 1) + np.roll(V, -1, 1)) / 9.0


def channel_cells(rgb, mask, ch):
    """Quantised cell features (n_rows, n_az, CF) uint8 and valid share per cell (or None) of one channel
    from an equirectangular rgb (h, w, 3) uint8 (+ mask (h, w) bool or None)."""
    g = grid(ch)
    h, w = rgb.shape[:2]
    R, A = g["n_rows"], g["n_az"]
    y0 = (90.0 - g["top"]) / 180.0 * h
    y1 = (90.0 - (g["top"] - R * g["cell"])) / 180.0 * h
    size = (A * CELL_PX, R * CELL_PX)
    lv = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR, box=(0.0, y0, float(w), y1)))
    L, a, b = _lab(lv)
    L = L.astype(np.float32)
    gx = np.roll(L, -1, 1) - np.roll(L, 1, 1)
    gy = np.empty_like(L)
    gy[1:-1] = L[:-2] - L[2:]
    gy[0], gy[-1] = gy[1], gy[-2]
    mag = 0.5 * np.hypot(gx, gy)
    o = (np.arctan2(gy, gx) % np.pi) / (np.pi / NBINS) - 0.5
    o0 = np.floor(o)
    f = (o - o0).astype(np.float32)
    o0 = o0.astype(np.int64) % NBINS
    cid = (np.arange(R * CELL_PX) // CELL_PX)[:, None] * A + (np.arange(A * CELL_PX) // CELL_PX)[None, :]
    base = cid * NBINS
    n = R * A * NBINS
    hist = (np.bincount((base + o0).ravel(), (mag * (1 - f)).ravel(), n) +
            np.bincount((base + (o0 + 1) % NBINS).ravel(), (mag * f).ravel(), n)).reshape(R, A, NBINS)
    E = (hist * hist).sum(-1)
    eps = float(CELL_PX * CELL_PX)
    hn = np.minimum(hist / np.sqrt(_box3(E) + eps * eps)[..., None], Q_HI[0])

    def cellmean(X):
        return X.reshape(R, CELL_PX, A, CELL_PX).mean((1, 3))

    feats = np.concatenate([hn, np.log1p(cellmean(mag))[..., None], (cellmean(L) / 100.0)[..., None],
                            (cellmean(a) / 40.0)[..., None], (cellmean(b) / 40.0)[..., None]], axis=-1)
    q = np.clip(np.round((feats - Q_LO) / (Q_HI - Q_LO) * 255.0), 0, 255).astype(np.uint8)
    valid = None
    if mask is not None:
        m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.BILINEAR,
                                                                            box=(0.0, y0, float(w), y1)))
        valid = cellmean(m.astype(np.float32) / 255.0)
    return q, valid


def sphere_cells(sph, channels=CHANNELS):
    """[(cells uint8, valid or None)] of every channel of a SphericalImage."""
    rgb = sph.rgb if sph.w >= 2048 else sph.resized(2048).rgb
    mask = None if sph.mask.all() else (sph.mask if sph.w >= 2048 else sph.resized(2048).mask)
    return [channel_cells(rgb, mask, ch) for ch in channels]


def dequant(q):
    return Q_LO + q.astype(np.float32) * ((Q_HI - Q_LO) / 255.0)


def windows(cells, rows=None):
    """Window descriptors (n_wrows * n_az, DIM) float32 of a channel's cells (uint8 or float); rows: only
    these window rows (an index array) -> (len(rows) * n_az, DIM)."""
    C = dequant(cells) if cells.dtype == np.uint8 else cells
    C = np.concatenate([C, C[:, :CELLS - 1]], axis=1)
    V = np.lib.stride_tricks.sliding_window_view(C, (CELLS, CELLS), axis=(0, 1))  # (wr, A, CF, 6, 6)
    V = V[:, :cells.shape[1]]
    if rows is not None:
        V = V[np.asarray(rows)]
    return np.ascontiguousarray(V).reshape(-1, DIM)


def window_valid(valid):
    """Valid share of every window (n_wrows, n_az) from the per-cell shares."""
    V = np.concatenate([valid, valid[:, :CELLS - 1]], axis=1)
    return np.lib.stride_tricks.sliding_window_view(V, (CELLS, CELLS)).mean((-1, -2))[:, :valid.shape[1]]


def best_windows(cells, valid, G, O, r0, r1):
    """Best score of every detector of a channel: (z (D,), row (D,), col (D,)), windows restricted to the
    rows r0[d] <= row <= r1[d]; -inf where no valid window."""
    R, A = cells.shape[0] - CELLS + 1, cells.shape[1]
    S = (windows(cells) @ G).reshape(R, A, -1)
    if valid is not None:
        S[window_valid(valid) < MIN_COVER] = -np.inf
    col = S.argmax(1)                                     # (R, D)
    M = np.take_along_axis(S, col[:, None, :], 1)[:, 0] - O
    rr = np.arange(R)[:, None]
    M[(rr < r0[None, :]) | (rr > r1[None, :])] = -np.inf
    row = M.argmax(0)
    k = np.arange(M.shape[1])
    return M[row, k], row, col[row, k]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------- detector bank
class ClueDetectors:
    """Detector bank (data/model/clue_detectors.npz, tools/build_clue_detectors.py).

    meta: channels, detectors [{country, cards [stem, ...], type, channel, rows [r0, r1], n, group, freq}],
          cards {country: {stem: {"freq", "det", "regions", "type"}}}, presence / country / region params.
    arrays: G<c> (DIM, D_c) and O<c> (window rows, D_c) of every channel (detectors of a channel in bank
          order), zmu, zsd (max-score standardisation), country model cm_classes, cm_med, cm_sc (D,),
          cm_A (D, C), cm_b (C,): loglik = clip((z~ - med) / sc, -8, 8) A - b."""

    def __init__(self, z):
        self.meta = json.loads(str(z["meta"]))
        self.channels = self.meta["channels"]
        self.dets = self.meta["detectors"]
        D = len(self.dets)
        self.chan = np.array([d["channel"] for d in self.dets], int)
        self.G = [np.asarray(z["G%d" % c], np.float32) for c in range(len(self.channels))]
        self.O = [np.asarray(z["O%d" % c], np.float32) for c in range(len(self.channels))]
        self.idx = [np.flatnonzero(self.chan == c) for c in range(len(self.channels))]
        rows = np.array([d["rows"] for d in self.dets], int).reshape(-1, 2)
        self.r0, self.r1 = rows[:, 0], rows[:, 1]
        self.zmu = np.asarray(z["zmu"], float) if "zmu" in z else np.zeros(D)
        self.zsd = np.asarray(z["zsd"], float) if "zsd" in z else np.ones(D)
        self.cards = self.meta.get("cards", {})
        self.presence = self.meta.get("presence")
        self.country = None
        if "cm_A" in z:
            self.country = {"classes": [str(c) for c in z["cm_classes"]], "A": np.asarray(z["cm_A"], float),
                            "b": np.asarray(z["cm_b"], float), "med": np.asarray(z["cm_med"], float),
                            "sc": np.asarray(z["cm_sc"], float)}
        self.region = self.meta.get("region", {"weight": 0.0})
        self.by_country = {}
        for i, d in enumerate(self.dets):
            self.by_country.setdefault(d["country"], []).append(i)

    @classmethod
    def load(cls, path=DETECTORS_PATH):
        if not path or not os.path.exists(path):
            return None
        try:
            with np.load(path) as z:
                return cls(z)
        except Exception as e:  # unreadable or outdated bank: no card evidence
            sys.stderr.write("clue detectors %s not loaded (%r) - no card evidence\n" % (path, e))
            return None

    # -------------------------------------------------------------- detection
    def scores(self, cells):
        """Best score and its window of every detector from the channel cells [(cells, valid)]:
        (z (D,), row (D,), col (D,))."""
        D = len(self.dets)
        z = np.full(D, -np.inf)
        row, col = np.zeros(D, int), np.zeros(D, int)
        for c, (cl, valid) in enumerate(cells):
            k = self.idx[c]
            if not len(k):
                continue
            z[k], row[k], col[k] = best_windows(cl, valid, self.G[c], self.O[c], self.r0[k], self.r1[k])
        return z, row, col

    def detect(self, sph):
        """Best standardised score and direction of every detector on a SphericalImage:
        {"z": (D,), "yaw": (D,) relative longitude, "pitch": (D,), "heading": true azimuth of yaw 0 or None,
        "coverage": share of the searched windows that are visible}."""
        cells = sphere_cells(sph, self.channels)
        z, row, col = self.scores(cells)
        yaw, pitch = np.full(len(z), np.nan), np.full(len(z), np.nan)
        for c, ch in enumerate(self.channels):
            k = self.idx[c]
            g = grid(ch)
            yaw[k], pitch[k] = window_yaw(g, col[k]), window_pitch(g, row[k])
        return {"z": z, "yaw": yaw, "pitch": pitch, "heading": sph.heading, "coverage": self.coverage(cells)}

    def coverage(self, cells):
        """Share of the windows inside the detectors' rows (mean over channels) with enough valid pixels."""
        s = []
        for c, (cl, valid) in enumerate(cells):
            k = self.idx[c]
            if not len(k):
                continue
            if valid is None:
                s.append(1.0)
                continue
            wv = window_valid(valid)[self.r0[k].min():self.r1[k].max() + 1]
            s.append(float((wv >= MIN_COVER).mean()))
        return float(np.mean(s)) if s else 0.0

    def zt(self, z):
        """Max scores standardised per detector over training panoramas ((D,) or (n, D)); -inf -> 0."""
        z = np.asarray(z, float)
        with np.errstate(invalid="ignore"):
            out = (z - self.zmu) / self.zsd
        return np.where(np.isfinite(out), out, 0.0)

    # -------------------------------------------------------------- presence
    def card_presence(self, cc, zt_row):
        """{stem: P(card present | country cc, detections)} for the cards of country cc."""
        a = (self.presence or {}).get("a", [0.0, 1.0, 0.0, 0.0])
        out = {}
        for stem, c in (self.cards.get(cc) or {}).items():
            d = c.get("det")
            x = a[0] + a[1] * _logit(c["freq"])
            if d is not None:
                x += (a[3] if self.dets[d].get("group") else a[2]) * zt_row[d]
            out[stem] = float(_sigmoid(x))
        return out

    # -------------------------------------------------------------- country evidence
    def country_evidence(self, det, classes):
        """(1, C) card log-likelihood of one detection (detect() + "zt"); None when less than
        MIN_EVIDENCE_COVERAGE of the searched windows is visible (the country model is fitted on the
        maxima of full panoramas: over fewer windows every max drops and would read as absent cards)."""
        if det.get("coverage", 1.0) < MIN_EVIDENCE_COVERAGE:
            return None
        return self.country_loglik(np.asarray(det["zt"])[None, :], classes)

    def country_loglik(self, ZT, classes):
        """(n, C) log-likelihood of the card detections for the given class list (rows max 0); countries
        outside the country model get the mean of the others."""
        ZT = np.atleast_2d(ZT)
        out = np.zeros((len(ZT), len(classes)))
        cm = self.country
        if cm is None:
            return out
        L = np.clip((ZT - cm["med"]) / cm["sc"], -8, 8) @ cm["A"] - cm["b"]
        ci = {c: i for i, c in enumerate(cm["classes"])}
        k = [ci.get(c, -1) for c in classes]
        out[:] = L.mean(1, keepdims=True)
        have = [j for j, i in enumerate(k) if i >= 0]
        out[:, have] = L[:, [k[j] for j in have]]
        return out - out.max(1, keepdims=True)

    # -------------------------------------------------------------- regions
    def region_factors(self, cc, zt_row, region_probs):
        """Soft update of P(region | country cc) by the regional cards: {region: factor}
        (Jeffrey's rule per card, q = P(card present), factor q 1[r in R] / P(R) + 1 - q,
        tempered by the calibrated exponent)."""
        w = float(self.region.get("weight", 0.0))
        if w <= 0 or not region_probs:
            return {}
        pres = self.card_presence(cc, zt_row)
        f = {r: 0.0 for r in region_probs}
        for stem, c in (self.cards.get(cc) or {}).items():
            R = [r for r in c.get("regions") or [] if r in region_probs]
            pr = sum(region_probs[r] for r in R)
            if not R or pr <= 0 or pr >= 0.999 or c.get("det") is None:
                continue
            q = pres[stem]
            for r in f:
                f[r] += math.log(q * (1.0 / pr if r in R else 0.0) + 1.0 - q)
        return {r: math.exp(w * v) for r, v in f.items()}

    def update_regions(self, cc, zt_row, region_probs):
        """region_probs {code: p} of one country -> the renormalised updated posterior."""
        fac = self.region_factors(cc, zt_row, region_probs)
        if not fac:
            return dict(region_probs)
        p = {r: v * fac.get(r, 1.0) for r, v in region_probs.items()}
        s = sum(p.values())
        return {r: v / s for r, v in p.items()} if s > 0 else dict(region_probs)

    # -------------------------------------------------------------- hints
    def direction_prob(self, cc, zt_row):
        """{stem: P(the card's best window shows it | country cc, detections)} for the cards of country cc
        with their own detector (out-of-fold calibrated; empty without a direction model)."""
        b = ((self.presence or {}).get("direction") or {}).get("a")
        if not b:
            return {}
        out = {}
        for stem, c in (self.cards.get(cc) or {}).items():
            d = c.get("det")
            if d is not None and not self.dets[d].get("group"):
                out[stem] = float(_sigmoid(b[0] + b[1] * _logit(c["freq"]) + b[2] * zt_row[d]))
        return out

    def detected_cards(self, cc, det, min_prob=None, min_dir=None):
        """Cards of country cc worth a mention: [{stem, prob, score, p_dir, heading, pitch, true_north, zoom,
        first, direction}], most probable first.  first: P(present) >= the calibrated hint threshold (or
        min_prob) - the card leads the hints; direction: P(best window shows the card | cc) >= the
        calibrated direction threshold (or min_dir) - the window is worth pointing at; cards that are
        neither are left out.  Both thresholds are > 1 (off) unless they beat the plain ranking on CALIB."""
        zt = det["zt"] if "zt" in det else self.zt(det["z"])
        pres = self.card_presence(cc, zt)
        pdir = self.direction_prob(cc, zt)
        p_ = self.presence or {}
        thr = p_.get("hint_min_prob", 1.01) if min_prob is None else min_prob
        dmin = (p_.get("direction") or {}).get("min_prob", 1.01) if min_dir is None else min_dir
        out = []
        for stem, p in pres.items():
            d = (self.cards[cc][stem] or {}).get("det")
            if d is None or not np.isfinite(det["z"][d]):
                continue
            q = pdir.get(stem, 0.0)
            if p < thr and q < dmin:
                continue
            yaw = float(det["yaw"][d])
            hd = det.get("heading")
            out.append({"stem": stem, "prob": p, "score": float(zt[d]), "p_dir": q, "first": bool(p >= thr),
                        "direction": bool(q >= dmin),
                        "heading": round(((hd + yaw) % 360.0) if hd is not None else yaw, 1),
                        "true_north": hd is not None, "pitch": round(float(det["pitch"][d]), 1),
                        "zoom": self.channels[self.dets[d]["channel"]]["zoom"], "group": bool(self.dets[d].get("group"))})
        out.sort(key=lambda r: -r["prob"])
        return out


_BANK = False


def clue_detectors():
    """The shared detector bank (None without data/model/clue_detectors.npz)."""
    global _BANK
    if _BANK is False:
        _BANK = ClueDetectors.load()
    return _BANK
