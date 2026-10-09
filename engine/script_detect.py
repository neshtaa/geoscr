"""
Writing detector (prototype): text lines on signs and their script family, closed-form.

  view rendering     gnomonic crop of an equirectangular panorama (the zoom-5 tile renderer of
                     the spike is tools/script_spike.render_from_tiles: offline only, it needs
                     the panorama's metadata, i.e. its location)
  text candidates    local-contrast binarisation (both polarities, two window sizes) ->
                     8-connected components (run-length union-find, numpy only) -> geometric
                     filters (size, aspect, fill, stroke width, holes)
  text lines         components chained by vertical overlap, similar height / stroke width and a
                     gap below 1.2 x height (also vertical columns of square glyphs); a line needs
                     >= 3 components (2 if large); scored by a logit over the line statistics
  line statistics    projection profile, gradient orientations, holes, satellites (dots,
                     diacritics), run lengths, stroke width, coherence (sd.FEATURES)
  writing descriptor ink / horizontal-edge / vertical-edge profiles in zones anchored on the
                     body band (Devanagari headline, Arabic baseline, Thai / Latin marks), the
                     mean glyph shape (HOG of every component's raster + its size and position)
                     and a histogram over a k-means codebook of glyph shapes
  line classifier    ridge multinomial logit over LINE_CLASSES (scripts + Digits + Noise),
                     fitted by tools/script_spike.py on font-rendered synthetic signs + real
                     placements (multiple-instance) + control-crop lines (Noise)
  view posterior     crop_loglik: a view's best text lines under every script hypothesis
                     (local script with prob q, else Latin / digits / noise) + prior

Spike result (scratch/script_spike/eval_*.json): useful only at ~0.03 deg/px and when the
camera points at a sign; see ScriptReader.  numpy + Pillow only.
"""

import math

import numpy as np
from PIL import Image

from engine.panorama import _camera_basis, bilinear

SCRIPTS = ["Latin", "Cyrillic", "Greek", "CJK", "Hangul", "Thai", "Indic", "Hebrew", "Arabic"]
LINE_CLASSES = SCRIPTS + ["Digits", "Noise"]  # line classifier: + digit strings, non-text lines
DIGITS, NOISE = len(SCRIPTS), len(SCRIPTS) + 1


# ---------------------------------------------------------------------------- rendering
def view_angles(yaw, pitch, t_h, iw, ih):
    """True azimuth / elevation (deg) of every pixel of a pinhole view; t_h = tan(hfov / 2)."""
    f, r, u = _camera_basis(yaw, pitch)
    tv = t_h * ih / iw
    xs = ((np.arange(iw) + 0.5) / iw * 2 - 1) * t_h
    ys = (1 - (np.arange(ih) + 0.5) / ih * 2) * tv
    d = f[None, None, :] + xs[None, :, None] * r[None, None, :] + ys[:, None, None] * u[None, None, :]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    az = np.degrees(np.arctan2(d[..., 0], d[..., 1]))
    el = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
    return az, el


def render_from_equirect(rgb, pano_heading, yaw, pitch, t_h, iw, ih):
    """The same crop from an equirectangular panorama array (centre column = pano_heading)."""
    H, W = rgb.shape[:2]
    az, el = view_angles(yaw, pitch, t_h, iw, ih)
    phi = (az - pano_heading + 180.0) % 360.0 - 180.0
    px = (phi + 180.0) / 360.0 * W - 0.5
    py = (90.0 - el) / 180.0 * H - 0.5
    out = bilinear(rgb, px, py, wrap_x=True)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- connected components
def label_components(mask):
    """8-connected components of a boolean image.

    Returns (labels image int32 with -1 = background, number of components, runs (row, start,
    exclusive end, component)).  Runs of every row
    are joined to the overlapping runs of the row above; the run graph is resolved by vectorised
    hooking + pointer jumping (a handful of passes), so no per-pixel Python loop is needed."""
    h, w = mask.shape
    m = np.zeros((h, w + 2), np.int8)
    m[:, 1:-1] = mask
    d = np.diff(m, axis=1)
    sy, sx = np.nonzero(d == 1)
    _, ex = np.nonzero(d == -1)  # exclusive end; same order as the starts (row-major)
    n = sx.size
    labels = np.full((h, w), -1, np.int32)
    if n == 0:
        return labels, 0, None
    row_first = np.searchsorted(sy, np.arange(h + 1))
    stride = w + 2
    skey = sy.astype(np.int64) * stride + sx
    ekey = sy.astype(np.int64) * stride + ex
    has_prev = (sy > 0)
    b = np.nonzero(has_prev)[0]
    yb = sy[b] - 1
    # runs a of the previous row with end >= start_b and start <= end_b (8-connectivity)
    lo = np.searchsorted(ekey, yb.astype(np.int64) * stride + sx[b], "left")
    hi = np.searchsorted(skey, yb.astype(np.int64) * stride + ex[b], "right")
    lo = np.maximum(lo, row_first[yb])
    hi = np.minimum(hi, row_first[yb + 1])
    cnt = np.maximum(hi - lo, 0)
    ea = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    eb = np.repeat(b, cnt)
    parent = np.arange(n)
    while ea.size:
        pa, pb = parent[ea], parent[eb]
        diff = pa != pb
        if not diff.any():
            break
        ea, eb, pa, pb = ea[diff], eb[diff], pa[diff], pb[diff]
        np.minimum.at(parent, np.maximum(pa, pb), np.minimum(pa, pb))
        while True:
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp
    roots, comp = np.unique(parent, return_inverse=True)
    lens = ex - sx
    rows = np.repeat(sy, lens)
    cols = np.repeat(sx, lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
    labels[rows, cols] = np.repeat(comp, lens)
    return labels, roots.size, (sy, sx, ex, comp)


def component_stats(labels, n, runs):
    """Per component: area, bbox, boundary pixel count and number of holes (8/4 Euler number)."""
    sy, sx, ex, comp = runs
    order = np.argsort(comp, kind="stable")
    first = np.searchsorted(comp[order], np.arange(n))
    area = np.bincount(comp, weights=ex - sx, minlength=n)
    x0 = np.minimum.reduceat(sx[order], first)
    x1 = np.maximum.reduceat(ex[order], first) - 1
    y0 = np.minimum.reduceat(sy[order], first)
    y1 = np.maximum.reduceat(sy[order], first)
    fg = labels >= 0
    p = np.pad(fg, 1)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    border = fg & ~inner
    bl = labels[border]
    boundary = np.bincount(bl, minlength=n).astype(np.float64)
    # bit-quad Euler number (8-connected foreground): E = (Q1 - Q3 - 2 QD) / 4
    q = np.pad(labels, 1, constant_values=-1)
    a, b_, c, d = q[:-1, :-1], q[:-1, 1:], q[1:, :-1], q[1:, 1:]
    fa, fb, fc, fd = a >= 0, b_ >= 0, c >= 0, d >= 0
    s = fa.astype(np.int8) + fb + fc + fd
    lq = np.maximum(np.maximum(a, b_), np.maximum(c, d))
    diag = (s == 2) & (fa == fd)
    e = np.zeros(lq.shape, np.int8)
    e[s == 1] = 1
    e[s == 3] = -1
    e[diag] = -2
    sel = (lq >= 0) & (e != 0)
    euler = np.bincount(lq[sel], weights=e[sel], minlength=n) / 4.0
    holes = np.maximum(1.0 - euler, 0.0)
    return {"area": area, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "boundary": boundary, "holes": holes}


# ------------------------------------------------------------------- text candidates
def box_mean(a, r):
    """Mean over a (2r+1)^2 window (edge-replicated), via an integral image."""
    k = 2 * r + 1
    p = np.pad(a.astype(np.float64), r, mode="edge")
    s = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    s[1:, 1:] = p.cumsum(0).cumsum(1)
    return ((s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k)).astype(np.float32)


def to_gray(img):
    a = np.asarray(img.convert("RGB"), np.float32) if isinstance(img, Image.Image) else img.astype(np.float32)
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


WINDOWS = (7, 20)        # half widths of the local mean windows (px)
NIBLACK_K = 0.25
MIN_STD = 9.0            # local contrast gate (grey levels)
MIN_H, MAX_H_FRAC = 7, 0.45


def candidate_masks(gray):
    """(name, mask) pairs: dark-on-light and light-on-dark strokes at two window sizes."""
    g2 = gray * gray
    out = []
    for r in WINDOWS:
        m = box_mean(gray, r)
        s = np.sqrt(np.maximum(box_mean(g2, r) - m * m, 0))
        gate = s > MIN_STD
        out.append(("dark%d" % r, gate & (gray < m - NIBLACK_K * s)))
        out.append(("light%d" % r, gate & (gray > m + NIBLACK_K * s)))
    return out


def char_candidates(labels, n, runs, shape):
    """Component statistics + the geometric filter of character-like components."""
    st = component_stats(labels, n, runs)
    ih, iw = shape
    w = (st["x1"] - st["x0"] + 1).astype(np.float64)
    h = (st["y1"] - st["y0"] + 1).astype(np.float64)
    fill = st["area"] / (w * h)
    sw = 2.0 * st["area"] / np.maximum(st["boundary"], 1)   # mean stroke width
    st.update(w=w, h=h, fill=fill, sw=sw)
    ok = ((h >= MIN_H) & (h <= MAX_H_FRAC * ih) & (w <= 0.6 * iw) & (w >= 2) & (st["area"] >= 18)
          & (fill > 0.08) & (fill < 0.95) & (w / h < 12) & (w / h > 0.06) & (sw / h < 0.4)
          & (st["holes"] <= 6) & (st["x0"] > 0) & (st["y0"] > 0) & (st["x1"] < iw - 1) & (st["y1"] < ih - 1))
    st["ok"] = ok
    return st


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def group_lines(st, max_cand=2500):
    """Chain character candidates into horizontal text lines -> list of component index arrays.

    Neighbours: height ratio <= 2, vertical overlap >= 0.6 x the smaller height, horizontal gap
    <= 1.2 x the larger height, stroke width ratio <= 2.5."""
    idx = np.nonzero(st["ok"])[0]
    if idx.size < 2:
        return []
    if idx.size > max_cand:  # keep the most character-like (mid fill, thin strokes)
        score = -np.abs(st["fill"][idx] - 0.4) - st["sw"][idx] / st["h"][idx]
        idx = idx[np.argsort(-score)[:max_cand]]
    x0, x1 = st["x0"][idx].astype(float), st["x1"][idx].astype(float)
    y0, y1 = st["y0"][idx].astype(float), st["y1"][idx].astype(float)
    h, sw = st["h"][idx], st["sw"][idx]
    order = np.argsort(x0, kind="stable")
    xs0 = x0[order]
    uf = _UF(idx.size)
    reach = x1 + 2.4 * h + 1  # the partner's height may be up to 2x larger
    for a in range(idx.size):
        i = order[a]
        stop = np.searchsorted(xs0, reach[i], "right")
        if stop <= a + 1:
            continue
        j = order[a + 1:stop]
        hi, hj = h[i], h[j]
        hm, hn = np.maximum(hi, hj), np.minimum(hi, hj)
        gap = x0[j] - x1[i]
        ov = np.minimum(y1[i], y1[j]) - np.maximum(y0[i], y0[j]) + 1
        good = ((hm <= 2.0 * hn) & (gap <= 1.2 * hm) & (gap >= -0.5 * hn) & (ov >= 0.6 * hn)
                & (np.maximum(sw[i], sw[j]) <= 2.5 * np.maximum(np.minimum(sw[i], sw[j]), 0.5)))
        for jj in j[good]:
            uf.union(i, jj)
    groups = {}
    for k in range(idx.size):
        groups.setdefault(uf.find(k), []).append(k)
    lines = []
    for g in groups.values():
        if len(g) >= 3 or (len(g) == 2 and h[g].min() >= 14):
            lines.append(idx[np.array(g)])
    return lines


# ------------------------------------------------------------------ line statistics
NBINS_PROF, NBINS_ORI = 10, 8
FEATURES = (["log_n", "h_cv", "top_dev", "bot_dev", "w_cv", "sw_cv", "sw_rel", "fill_mean", "fill_std",
             "holes_mean", "holes_frac", "aspect_mean", "aspect_std", "coverage", "gap_mean", "gap_cv",
             "log_aspect", "contrast", "bg_noise", "fg_noise", "log_h", "headline", "baseline",
             "long_h", "long_v", "run_h", "run_v", "sat_above", "sat_below", "sat_inside", "wide_frac",
             "comp_per_glyph", "euler_area", "ori_entropy", "hv_ratio", "polarity", "ascender", "descender",
             "holes2_frac", "short_frac", "coherence", "straight_frac", "vertical"]
            + ["prof%d" % i for i in range(NBINS_PROF)] + ["ori%d" % i for i in range(NBINS_ORI)])


def _runs(b):
    """Lengths of the foreground runs along the rows of a boolean array."""
    m = np.zeros((b.shape[0], b.shape[1] + 2), np.int8)
    m[:, 1:-1] = b
    d = np.diff(m, axis=1)
    return np.nonzero(d == -1)[1] - np.nonzero(d == 1)[1]


def line_features(gray, gx, gy, labels, st, comps, polarity, vertical=False):
    x0, x1, y0, y1 = st["x0"][comps], st["x1"][comps], st["y0"][comps], st["y1"][comps]
    h, w, sw = st["h"][comps], st["w"][comps], st["sw"][comps]
    X0, X1, Y0, Y1 = int(x0.min()), int(x1.max()), int(y0.min()), int(y1.max())
    LW, LH = X1 - X0 + 1, Y1 - Y0 + 1
    hm = float(np.median(h))
    sub = labels[Y0:Y1 + 1, X0:X1 + 1]
    B = np.isin(sub, comps)
    g = gray[Y0:Y1 + 1, X0:X1 + 1]
    fg, bg = g[B], g[~B]
    c = abs(float(fg.mean()) - float(bg.mean())) if bg.size else 0.0
    order = np.argsort(x0)
    gaps = (x0[order][1:] - x1[order][:-1]).astype(float)
    gaps = np.maximum(gaps, 0)
    rp = B.sum(1).astype(float)
    cum = np.concatenate([[0], np.cumsum(rp)])
    edges = np.linspace(0, LH, NBINS_PROF + 1)
    prof = np.diff(np.interp(edges, np.arange(LH + 1), cum))
    prof /= max(prof.sum(), 1)
    k = max(int(math.ceil(0.35 * LH)), 1)
    headline = rp[:k].max() / LW
    baseline = rp[-k:].max() / LW
    rh, rv = _runs(B), _runs(B.T)
    area = max(B.sum(), 1)
    long_h = rh[rh > 0.5 * hm].sum() / area
    long_v = rv[rv > 0.6 * hm].sum() / area
    # gradient orientation on the stroke boundary
    p = np.pad(B, 1)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    edge = B & ~inner
    ex, ey = gx[Y0:Y1 + 1, X0:X1 + 1][edge], gy[Y0:Y1 + 1, X0:X1 + 1][edge]
    mag = np.hypot(ex, ey)
    ang = np.mod(np.arctan2(ey, ex), np.pi)
    ori = np.bincount(np.minimum((ang / np.pi * NBINS_ORI).astype(int), NBINS_ORI - 1), weights=mag,
                      minlength=NBINS_ORI)
    ori = ori / max(ori.sum(), 1e-6)
    ent = -float((ori * np.log(np.maximum(ori, 1e-9))).sum())
    hv = (ori[0] + ori[NBINS_ORI // 2]) / max(ori[NBINS_ORI // 4] + ori[3 * NBINS_ORI // 4], 1e-3)
    # structure-tensor coherence on the stroke boundary: straight strokes ~1, curves lower
    r = max(int(round(float(np.median(sw)) / 2)), 1)
    sgx, sgy = gx[Y0:Y1 + 1, X0:X1 + 1], gy[Y0:Y1 + 1, X0:X1 + 1]
    jxx, jyy, jxy = box_mean(sgx * sgx, r)[edge], box_mean(sgy * sgy, r)[edge], box_mean(sgx * sgy, r)[edge]
    coh = ((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum((jxx + jyy) ** 2, 1e-6)
    coherence = float((coh * mag).sum() / max(mag.sum(), 1e-6))
    straight = float((mag * (coh > 0.8)).sum() / max(mag.sum(), 1e-6))
    # satellites: small components around the line (dots, diacritics, tone marks)
    my0, my1 = float(np.median(y0)), float(np.median(y1))
    ax0, ax1, ay0, ay1 = st["x0"], st["x1"], st["y0"], st["y1"]
    cy = (ay0 + ay1) / 2.0
    near = ((ax0 >= X0 - 0.3 * hm) & (ax1 <= X1 + 0.3 * hm) & (ay0 >= Y0 - 0.8 * hm) & (ay1 <= Y1 + 0.8 * hm)
            & (st["h"] < 0.5 * hm) & (st["area"] >= 3))
    near[comps] = False
    glyphs = max(LW / max(hm, 1), 1.0)
    sat_above = float((near & (cy < my0)).sum()) / glyphs
    sat_below = float((near & (cy > my1)).sum()) / glyphs
    sat_inside = float((near & (cy >= my0) & (cy <= my1)).sum()) / glyphs
    return np.array([
        math.log(len(comps)), h.std() / h.mean(), y0.std() / hm, y1.std() / hm, w.std() / w.mean(),
        sw.std() / sw.mean(), float(np.median(sw)) / hm, st["fill"][comps].mean(), st["fill"][comps].std(),
        st["holes"][comps].mean(), (st["holes"][comps] > 0).mean(), (w / h).mean(), (w / h).std(),
        w.sum() / LW, gaps.mean() / hm if gaps.size else 0.0,
        gaps.std() / max(gaps.mean(), 0.5) if gaps.size else 0.0, math.log(LW / LH),
        c / 255.0, (bg.std() if bg.size else 0.0) / max(c, 1.0), fg.std() / max(c, 1.0), math.log(hm),
        headline, baseline, long_h, long_v, rh.mean() / hm, rv.mean() / hm, sat_above, sat_below, sat_inside,
        (w > 1.5 * h).mean(), len(comps) / glyphs, (1 - st["holes"][comps]).sum() / area * hm * hm, ent, hv,
        float(polarity), (y0 < my0 - 0.2 * hm).mean(), (y1 > my1 + 0.2 * hm).mean(),
        (st["holes"][comps] >= 2).mean(), (h < 0.75 * hm).mean(), coherence, straight, float(vertical)]
        + list(prof) + list(ori)), (X0, Y0, X1, Y1)


# ------------------------------------------------------------------ glyph shapes
GLYPH, SUPER, NB_GLYPH, CELL = 16, 3, 4, 4   # glyph raster side, supersampling, HOG bins / cell
NZONE = (4, 8, 4)                            # zone rows above / inside / below the body band
GLYPH_DIM = NB_GLYPH * (GLYPH // CELL) ** 2 + 6


def glyph_rasters(labels, st, comps):
    """Every component's own mask in a square of side max(w, h) around its centre -> (n, G, G)."""
    x0, x1, y0, y1 = st["x0"][comps], st["x1"][comps], st["y0"][comps], st["y1"][comps]
    side = np.maximum(x1 - x0 + 1, y1 - y0 + 1).astype(np.float64)
    m = GLYPH * SUPER
    u = (np.arange(m) + 0.5) / m - 0.5
    px = np.floor((x0 + x1 + 1)[:, None] / 2.0 + u[None, :] * side[:, None]).astype(np.int64)
    py = np.floor((y0 + y1 + 1)[:, None] / 2.0 + u[None, :] * side[:, None]).astype(np.int64)
    h, w = labels.shape
    okx, oky = (px >= 0) & (px < w), (py >= 0) & (py < h)
    lab = labels[np.clip(py, 0, h - 1)[:, :, None], np.clip(px, 0, w - 1)[:, None, :]] == comps[:, None, None]
    lab &= oky[:, :, None] & okx[:, None, :]
    return lab.reshape(len(comps), GLYPH, SUPER, GLYPH, SUPER).mean((2, 4))


def glyph_vectors(labels, st, comps, my0, my1):
    """Shape vector per component: HOG of its raster (4 x 4 cells, 4 unsigned orientations, L2)
    + log aspect, log height, top / bottom offset from the body band, holes, fill."""
    R = glyph_rasters(labels, st, comps)
    gy, gx = np.gradient(R, axis=(1, 2))
    mag = np.hypot(gx, gy)
    a = np.mod(np.arctan2(gy, gx), np.pi) / (np.pi / NB_GLYPH)
    b0 = np.floor(a).astype(np.int64) % NB_GLYPH
    f = a - np.floor(a)
    nc = GLYPH // CELL
    O = np.zeros((len(comps), NB_GLYPH, GLYPH, GLYPH))
    for k in range(NB_GLYPH):
        O[:, k] = mag * np.where(b0 == k, 1 - f, 0) + mag * np.where((b0 + 1) % NB_GLYPH == k, f, 0)
    hog = O.reshape(len(comps), NB_GLYPH, nc, CELL, nc, CELL).sum((3, 5)).reshape(len(comps), -1)
    hog /= np.maximum(np.linalg.norm(hog, axis=1, keepdims=True), 1e-6)
    hb = max(my1 - my0 + 1, 1.0)
    h, w = st["h"][comps], st["w"][comps]
    sc = np.stack([np.log(w / h), np.log(h / hb), (st["y0"][comps] - my0) / hb, (st["y1"][comps] - my1) / hb,
                   np.minimum(st["holes"][comps], 2) / 2.0, st["fill"][comps]], 1)
    return np.hstack([hog, 0.5 * sc])


def zone_profiles(labels, st, comps, my0, my1, X0, X1, extra=None):
    """Ink, horizontal-edge and vertical-edge row profiles of the line's components (+ extra:
    satellites) over a band anchored on the body (median top / bottom of the components):
    NZONE rows above, inside and below it, each profile normalised to sum 1."""
    hb = my1 - my0 + 1.0
    t0, t1 = my0 - 0.5 * hb, my1 + 1 + 0.5 * hb
    r0, r1 = max(int(np.floor(t0)), 0), min(int(np.ceil(t1)), labels.shape[0])
    sub = labels[r0:r1, X0:X1 + 1]
    B = np.isin(sub, comps if extra is None else np.concatenate([comps, extra]))
    ink = B.sum(1).astype(np.float64)
    he = np.zeros(len(ink))
    he[:-1] = (B[1:] != B[:-1]).sum(1)
    ve = (B[:, 1:] != B[:, :-1]).sum(1).astype(np.float64)
    edges = np.concatenate([np.linspace(t0, my0, NZONE[0] + 1)[:-1], np.linspace(my0, my1 + 1, NZONE[1] + 1)[:-1],
                            np.linspace(my1 + 1, t1, NZONE[2] + 1)]) - r0
    out = []
    for prof in (ink, he, ve):
        cum = np.concatenate([[0], np.cumsum(prof)])
        z = np.diff(np.interp(edges, np.arange(len(cum)), cum))
        out.append(z / max(z.sum(), 1e-6))
    return np.concatenate(out)


def _iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0]) + 1
    iy = min(a[3], b[3]) - max(a[1], b[1]) + 1
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / float((a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter)


def script_vectors(labels, st, comps, codebook=None):
    """Writing descriptor of a line: zone profiles, mean glyph shape vector and (with a codebook
    of glyph shapes) the normalised histogram of the nearest codewords, square-rooted.
    codebook = (pooled codebook, word codebook): also the word index of every glyph.
    -> (descriptor, glyph shape vectors, glyph words or None)"""
    x0, x1, y0, y1 = st["x0"][comps], st["x1"][comps], st["y0"][comps], st["y1"][comps]
    my0, my1 = float(np.median(y0)), float(np.median(y1))
    hm = float(np.median(st["h"][comps]))
    X0, X1, Y0, Y1 = int(x0.min()), int(x1.max()), int(y0.min()), int(y1.max())
    ax0, ax1, ay0, ay1 = st["x0"], st["x1"], st["y0"], st["y1"]
    near = ((ax0 >= X0 - 0.3 * hm) & (ax1 <= X1 + 0.3 * hm) & (ay0 >= Y0 - 0.8 * hm) & (ay1 <= Y1 + 0.8 * hm)
            & (st["h"] < 0.5 * hm) & (st["area"] >= 3))
    near[comps] = False
    zp = zone_profiles(labels, st, comps, my0, my1, max(X0 - int(0.3 * hm), 0), X1 + int(0.3 * hm),
                       np.nonzero(near)[0])
    gv = glyph_vectors(labels, st, comps, my0, my1)
    parts = [zp, gv.mean(0)]
    cb, words = (codebook, None) if not isinstance(codebook, tuple) else codebook
    if cb is not None:
        hist = np.bincount(nearest(gv, cb), minlength=len(cb)) / float(len(comps))
        parts.append(np.sqrt(hist))
    return np.concatenate(parts), gv, (nearest(gv, words) if words is not None else None)


def detect_lines(img, max_side=None, codebook=None, window=None, glyphs=False):
    """All candidate text lines of an image -> dict: feats (n, F) line statistics, script (n, S)
    writing descriptors, boxes (n, 4), ncomp (n,), [glyphs: per-line glyph shape vectors],
    [words: per-line glyph word indices, when codebook = (pooled, words)].

    Overlapping lines of the four candidate masks (IoU > 0.5) are reduced to the one with the
    most components before the line statistics are measured.  window = (fx, fy): keep only lines
    whose centre lies in the central fx x fy fraction of the image."""
    gray = to_gray(img)
    if max_side and max(gray.shape) > max_side:
        s = max_side / float(max(gray.shape))
        gray = to_gray(Image.fromarray(gray.astype(np.uint8)).resize(
            (int(gray.shape[1] * s), int(gray.shape[0] * s)), Image.BILINEAR))
    gy, gx = np.gradient(gray)
    masks, cands = [], []
    for name, mask in candidate_masks(gray):
        labels, n, runs = label_components(mask)
        if n == 0:
            continue
        st = char_candidates(labels, n, runs, gray.shape)
        mi = len(masks)
        masks.append((labels, st, name.startswith("dark")))
        for comps in group_lines(st):
            box = (int(st["x0"][comps].min()), int(st["y0"][comps].min()),
                   int(st["x1"][comps].max()), int(st["y1"][comps].max()))
            cands.append((len(comps), mi, False, comps, box))
        # vertical columns of square-ish glyphs (CJK / Hangul signs), measured on the transposed image
        sq = st["ok"] & (st["w"] >= 0.6 * st["h"]) & (st["w"] <= 1.6 * st["h"])
        tst = dict(st, x0=st["y0"], x1=st["y1"], y0=st["x0"], y1=st["x1"], w=st["h"], h=st["w"], ok=sq)
        for comps in group_lines(tst):
            if len(comps) >= 3:
                box = (int(st["x0"][comps].min()), int(st["y0"][comps].min()),
                       int(st["x1"][comps].max()), int(st["y1"][comps].max()))
                cands.append((len(comps), mi, True, comps, box))
    if window is not None:
        ih, iw = gray.shape
        cands = [c for c in cands if abs((c[4][0] + c[4][2] + 1) / 2.0 / iw - 0.5) < window[0] / 2
                 and abs((c[4][1] + c[4][3] + 1) / 2.0 / ih - 0.5) < window[1] / 2]
    keep = []
    for c in sorted(cands, key=lambda c: -c[0]):
        if all(_iou(c[4], k[4]) <= 0.5 for k in keep):
            keep.append(c)
    feats, svec, gls, words, boxes, ncomp = [], [], [], [], [], []
    for nc, mi, vert, comps, box in keep:
        labels, st, dark = masks[mi]
        if vert:
            st = dict(st, x0=st["y0"], x1=st["y1"], y0=st["x0"], y1=st["x1"], w=st["h"], h=st["w"])
            f, _ = line_features(gray.T, gy.T, gx.T, labels.T, st, comps, dark, True)
            sv, gv, wd = script_vectors(labels.T, st, comps, codebook)
        else:
            f, _ = line_features(gray, gx, gy, labels, st, comps, dark)
            sv, gv, wd = script_vectors(labels, st, comps, codebook)
        feats.append(f)
        svec.append(sv)
        gls.append(gv)
        words.append(wd)
        boxes.append(box)
        ncomp.append(nc)
    cb = codebook[0] if isinstance(codebook, tuple) else codebook
    S = 3 * sum(NZONE) + GLYPH_DIM + (0 if cb is None else len(cb))
    out = {"feats": np.array(feats).reshape(-1, len(FEATURES)), "script": np.array(svec).reshape(-1, S),
           "boxes": np.array(boxes, int).reshape(-1, 4), "ncomp": np.array(ncomp, int)}
    if glyphs:
        out["glyphs"] = gls
    if isinstance(codebook, tuple):
        out["words"] = words
    return out


def nearest(X, C, chunk=20000):
    """Index of the nearest centre (rows of C) of every row of X."""
    cc = (C * C).sum(1)
    return np.concatenate([np.argmin(cc[None, :] - 2 * X[i:i + chunk] @ C.T, 1)
                           for i in range(0, len(X), chunk)]) if len(X) else np.zeros(0, int)


def kmeans(X, k, iters=25, seed=0):
    """Plain Lloyd k-means (k-means++ seeding) -> centres (k, d)."""
    rng = np.random.RandomState(seed)
    C = [X[rng.randint(len(X))]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        a = nearest(X, C)
        for j in range(k):
            m = a == j
            C[j] = X[m].mean(0) if m.any() else X[rng.randint(len(X))]
    return C


# ------------------------------------------------------------------------- classifiers
def expand(feats, center, scale):
    """Robust-standardised features and their squares (clipped), plus a bias column."""
    z = np.clip((np.asarray(feats, np.float64) - center) / scale, -5, 5)
    return np.hstack([z, z * z / 5.0, np.ones((len(z), 1))])


def softmax(s):
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def fit_logit(X, y, k, weights=None, lam=1.0, iters=25):
    """Ridge multinomial logit by Newton's method (bias = last column, not penalised)."""
    n, d = X.shape
    w = np.ones(n) if weights is None else weights
    W = np.zeros((d, k))
    Y = np.eye(k)[y]
    pen = np.full(d, lam)
    pen[-1] = 1e-6
    P_ = np.kron(np.eye(k), np.diag(pen))
    for _ in range(iters):
        P = softmax(X @ W)
        G = X.T @ ((P - Y) * w[:, None]) + pen[:, None] * W
        H = np.zeros((d * k, d * k))
        for a in range(k):
            for b in range(a, k):
                c = w * P[:, a] * ((a == b) - P[:, b])
                blk = (X * c[:, None]).T @ X
                H[a * d:(a + 1) * d, b * d:(b + 1) * d] = blk
                H[b * d:(b + 1) * d, a * d:(a + 1) * d] = blk
        step = np.linalg.solve(H + P_ + 1e-8 * np.eye(d * k), G.T.reshape(-1))
        W -= step.reshape(k, d).T
        if np.abs(step).max() < 1e-5:
            break
    return W


def fit_logit_lbfgs(X, y, k, weights=None, lam=1.0, iters=300, mem=10, W0=None):
    """Ridge multinomial logit by L-BFGS (bias = last column, not penalised) -> W (d, k)."""
    n, d = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    Y = np.eye(k)[y]
    pen = np.full((d, 1), lam)
    pen[-1] = 0.0

    def fg(v):
        W = v.reshape(d, k)
        S = X @ W
        S -= S.max(1, keepdims=True)
        lse = np.log(np.exp(S).sum(1))
        f = float((w * (lse - (S * Y).sum(1))).sum() + 0.5 * (pen * W * W).sum())
        P = np.exp(S - lse[:, None])
        g = X.T @ ((P - Y) * w[:, None]) + pen * W
        return f, g.reshape(-1)

    v = np.zeros(d * k) if W0 is None else W0.reshape(-1).copy()
    f, g = fg(v)
    sk, yk = [], []
    for _ in range(iters):
        q = g.copy()
        al = []
        for s_, y_ in reversed(list(zip(sk, yk))):
            a = s_ @ q / (y_ @ s_)
            al.append(a)
            q -= a * y_
        if sk:
            q *= (sk[-1] @ yk[-1]) / (yk[-1] @ yk[-1])
        for (s_, y_), a in zip(zip(sk, yk), reversed(al)):
            q += s_ * (a - y_ @ q / (y_ @ s_))
        dvec = -q
        gd = g @ dvec
        if gd >= 0:
            dvec, gd, sk, yk = -g, -(g @ g), [], []
        t = 1.0
        while True:
            f2, g2 = fg(v + t * dvec)
            if f2 <= f + 1e-4 * t * gd or t < 1e-10:
                break
            t *= 0.5
        s_, y_ = t * dvec, g2 - g
        v, f_old, f, g = v + s_, f, f2, g2
        if y_ @ s_ > 1e-12:
            sk.append(s_)
            yk.append(y_)
            if len(sk) > mem:
                sk.pop(0)
                yk.pop(0)
        if abs(f_old - f) < 1e-7 * max(abs(f), 1.0):
            break
    return v.reshape(d, k)


# ------------------------------------------------------------------------- inference
def line_design(F, S, m, extra=None):
    """Design matrix of the line classifier: robust-standardised line statistics and their
    squares ('stats'), standardised writing descriptors ('writing'), both ('all'); + bias."""
    parts = []
    if m["feats"] in ("stats", "all"):
        z = np.clip((F - m["cf"]) / m["sf"], -5, 5)
        parts += [z, z * z / 5.0]
    if m["feats"] in ("writing", "all"):
        parts.append(np.clip((S - m["cs"]) / m["ss"], -5, 5))
    if extra is not None:
        parts.append(extra)
    parts.append(np.ones((len(F), 1)))
    return np.hstack(parts)


def crop_loglik(P, q):
    """Log-likelihood of a view's text lines under every script hypothesis (SCRIPTS), with the
    line posteriors of the class-balanced line classifier (LINE_CLASSES) as scaled likelihoods.
    Script s != Latin: a line is in s (prob q) or Latin / digits (1 - q) or noise; Latin: every
    line is Latin / digits or noise."""
    lat = P[:, 0] + P[:, DIGITS]
    noise = P[:, NOISE]
    ll = np.empty(len(SCRIPTS))
    ll[0] = np.log(lat + noise + 1e-9).sum()
    ll[1:] = np.log(q * P[:, 1:len(SCRIPTS)] + ((1 - q) * lat + noise)[:, None] + 1e-9).sum(0)
    return ll


class ScriptReader:
    """Prototype: the text lines of a pinhole view (~0.03 deg/px) and a posterior over SCRIPTS.

    Model file written by `tools/script_spike.py eval --save`: glyph codebooks, text-line scorer
    and its threshold, line classifier, script prior and the mixing weight q.  Calibrated on the
    central 20 x 12 deg window of 0.03 deg/px crops only: on whole 1112 px frames (window=None)
    the same threshold fires on 20% of views not aimed at a sign (11% in the window), and other
    resolutions need their own model (0.13 deg/px, the live frame centre: at chance).  "posterior"
    includes the script mix of GeoGuessr's sign placements as prior; a locator that has its own
    country prior must use "loglik" (no prior) instead."""

    def __init__(self, path):
        z = np.load(path)
        self.z = {k: z[k] for k in z.files}
        self.codebook = (self.z["codebook"], self.z["wordbook"])
        self.text = {"c": self.z["text_c"], "s": self.z["text_s"], "W": self.z["text_W"]}
        self.line = {"feats": str(self.z["feats"]), "cf": self.z["script_cf"], "sf": self.z["script_sf"],
                     "cs": self.z["script_cs"], "ss": self.z["script_ss"], "W": self.z["script_W"]}
        self.thr, self.q = float(self.z["text_thr"]), float(self.z["q"])
        self.logprior = self.z["logprior"] * float(self.z["pw"])

    def read(self, img, window=None, k=3):
        """-> None when no text line passes the threshold, else {"posterior": {script: p},
        "loglik": {script: log-likelihood of the lines, no prior}, "lines": [(box, text score,
        line class, p)]} for the k best text lines."""
        o = detect_lines(img, codebook=self.codebook, window=window)
        if not len(o["feats"]):
            return None
        F = o["feats"].astype(np.float64)
        z = expand(F, self.text["c"], self.text["s"]) @ self.text["W"]
        ts = z[:, 1] - z[:, 0]
        top = [i for i in np.argsort(-ts)[:k] if ts[i] > self.thr]
        if not top:
            return None
        P = softmax(line_design(F[top], o["script"][top].astype(np.float64), self.line) @ self.line["W"])
        ll = crop_loglik(P, self.q)
        post = softmax((ll + self.logprior)[None, :])[0]
        return {"posterior": dict(zip(SCRIPTS, post.round(4).tolist())),
                "loglik": dict(zip(SCRIPTS, (ll - ll.max()).round(3).tolist())),
                "lines": [(tuple(int(v) for v in o["boxes"][i]), round(float(ts[i]), 2),
                           LINE_CLASSES[int(np.argmax(P[j]))], round(float(P[j].max()), 3)) for j, i in enumerate(top)]}
