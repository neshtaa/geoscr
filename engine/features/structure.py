"""
Built environment, utility poles / overhead wires, GIST-like texture and colour
descriptors.  Pure closed-form image mathematics on the equirectangular sphere
(numpy only); every statistic is computed over valid (``sph.mask``) pixels only
and everything is ROTATION INVARIANT (pooled over azimuth), so neither
``sph.heading`` nor ``sph.car_heading`` is needed: the same vector comes out for
a full panorama, for a canvas rebuilt from screenshots and (with more NaNs) for a
single screenshot.

Notation: Y = Rec.601 luma (0..255), el = elevation (deg).  The canvas is 2:1
equirectangular, so a world-vertical line is exactly an image column.

1. GIST-like oriented gradient energy  (``g{s}_b{band}_*``)
   Y is analysed at three scales (1024x512, 512x256, 256x128 = 0.35 / 0.7 /
   1.4 deg per pixel; the coarser scales are 2x2 means over valid pixels).
   Gradients are central differences; the longitudinal derivative is divided by
   cos(el) so it is measured per degree of arc (equirectangular stretch
   compensation).  The gradient magnitude m is split by a triangular-tuned
   steerable orientation filter bank on theta = atan2(gy, gx) mod pi with
   channels at 0 / 45 / 90 / 135 deg.  Energies are pooled in 6 elevation bands
   (60..30, 30..10, 10..2, 2..-5, -5..-20, -20..-45) x 16 azimuth sectors
   (22.5 deg); per band and scale:
     e_mean / e_std / e_max : mean, std and max over sectors of log(mean m + 0.5)
                              (rotation-invariant texture-energy summary),
     fv / fh                : pooled share of the energy in the vertical-edge
                              (theta = 0) and horizontal-edge (theta = 90)
                              channels (the diagonal share is 1 - fv - fh).
   Sectors with < 30 % valid pixels are skipped; no valid sector -> NaN.

2. Built environment  (``b_*``, 1024 px, el 40..-6)
   Edge pixels: m > 8.  STRAIGHT edge pixel: quantise theta into 8 bins
   (22.5 deg); the edge pixel needs >= 3 of the 4 pixels at -6, -3, +3, +6 px
   ALONG ITS OWN EDGE DIRECTION to be edges of the same (+-1) bin, i.e. a
   collinear segment of ~4.5 deg (walls, windows, eaves, poles - not foliage or
   clouds).  Families: vertical lines (bins within 34 deg of vertical) and
   horizontal / oblique lines (the rest).
     b_sv_dens / b_so_dens : straight vertical / other straight line density, el 2..30
     b_straight            : straight / all edge pixels, el 0..30
     b_built_frac          : share of the el 0..20 band classified as built: a
                             7x7 deg box holding straight lines of BOTH families
                             (>= 1 % each, >= 4 % together, >= 30 % of all its
                             edges straight) and less than half sky
     b_built_sect          : share of 16 azimuth sectors with a built share > 15 %
   Palette of built, non-sky, non-vegetation pixels in el 0..20 (CIELAB, hue
   h = atan2(b*, a*), chroma C = hypot(a*, b*)): p_terracotta (h 10..65 deg,
   C >= 22, L >= 42), p_brick (same hues, C > 14, L < 42), p_white (L > 78,
   C < 10), p_grey (40 < L <= 78, C < 8), p_dark (L <= 40, C < 10), p_colour
   (C > 25 outside the red family: painted metal roofs / walls).

3. Sky and skyline  (``sky_*``, 512 px)
   Sky-like pixel: blue (B - R > 6, B >= G - 8) or bright neutral cloud
   (max - min < 45, Y > max(0.5 x 95th pct of the upper hemisphere, 100)), local
   gradient < 90, el > -3.  Seeds: sky-like pixels with <= 4 non-sky pixels
   above them in their column; then a geodesic propagation down the rows
   through horizontal runs of sky-like pixels (sky under a bridge girder or
   between branches stays sky; white walls below a roof edge do not).
     sky_frac (share of el > 0), sky_hband (share of el 2..20), skyline_med /
     skyline_iqr (elevation of the lowest sky pixel per column), skyline_rough
     (mean |difference| of the skyline between adjacent columns, clipped at 10
     deg - jagged tree tops vs flat roofs), sky_open (columns whose skyline is
     below 3 deg = open horizon).

4. Utility poles  (``pole_*``, 1024 px, el 45..-3)
   Dark vertical ridge filter: for centre widths 1, 3, 5, 7 px the centre mean
   C is compared with two flank boxes (width max(2, h), 1 px gap):
   R = (min(Left, Right) - C) / (max(Left, Right) + 5); step edges give R <= 0,
   only thin dark lines respond.  Both flanks must be sky.  A pole column has
   a 12-row (4.2 deg) window with >= 80 % ridge pixels (R > 0.12, +-1 px
   lateral tolerance), few other ridge pixels within +-10 px (isolated, not a
   tree crown), is not floating (the ridge does not end while the sky goes on
   below it) and has open sky above its top.  World-vertical poles are exactly
   image columns, so this is a geometric test.  Adjacent columns are grouped.
     pole_n360  : poles per 360 deg (count / valid azimuth share)
     pole_top   : median elevation of the detected tops (how tall / close)
     pole_width : median angular width (deg) of the best ridge scale
     pole_thick : median width_rad / tan(top_el) ~ diameter / (height - camera
                  height): a distance-free thickness
     pole_rel_l : median centre / flank luma (dark wood ~0.3, light concrete
                  ~0.7);  pole_a / pole_b: median CIELAB a*, b* of pole pixels.

5. Overhead wires  (``wire_*``, 2048 px, el 65..3)
   Dark horizontal ridge on the BLUE channel (against blue sky and white clouds
   alike a wire is dark in blue, while the blue gaps between cloud puffs are
   not): 1-px centre vs rows +-2..3 and 3-px centre vs rows +-3..4, both flanks
   sky, R > 0.05 and stronger than a 5-px-wide ridge (thin line).  Rejected:
   pixels in vertical-ridge clutter (branches; > 8 % of a 9x9 window at half
   resolution) or with more than 8 other ridge pixels 2..5 rows above/below.
   Chains: dynamic programming over columns at half resolution, each step
   moving at most 1 row (slope <= 45 deg); a wire pixel lies on a chain of
   >= 24 columns (8.4 deg) - catenaries are long smooth curves.
     wire_cols  : share of sky columns crossed by at least one wire
     wire_count : mean number of distinct wires per sky column
     wire_smax  : max over 16 sectors of the crossed-column share
     wire_pix   : wire pixels per 1000 sky pixels

6. Colour histogram  (``h_*``, ``c{band}_*``, 512 px)
   CIELAB (D65) of valid pixels with el >= -55 (car and nadir blur excluded),
   each weighted by cos(el) (true solid angle).  3 L levels (< 40, 40..70,
   > 70) x 4 a* bins (edges -6, 2, 10) x 4 b* bins (edges -4, 8, 20) = 48 bins,
   normalised and square-rooted (Hellinger).  Plus per band (upper el 10..60,
   horizon -5..10, ground -45..-5) the weighted mean L*, a*, b* and std of L*.
"""
import math

import numpy as np

NAME = "structure"

GIST_BANDS = [(60, 30), (30, 10), (10, 2), (2, -5), (-5, -20), (-20, -45)]
GIST_SCALES = (1024, 512, 256)
N_SECT = 16
EDGE_T = 8.0
COLOR_BANDS = [("up", 60, 10), ("hz", 10, -5), ("gr", -5, -45)]
L_EDGES = np.array([40.0, 70.0], np.float32)
A_EDGES = np.array([-6.0, 2.0, 10.0], np.float32)
B_EDGES = np.array([-4.0, 8.0, 20.0], np.float32)
PALETTE = ["terracotta", "brick", "white", "grey", "dark", "colour"]
POLE_HW = (1, 2, 3, 4)       # ridge half-widths at 1024 px (0.35 .. 2.5 deg wide)
POLE_T = 0.12
POLE_K = 12                  # 4.2 deg vertical support
WIRE_T = 0.05
WIRE_W = 2048
WIRE_CLUTTER = 0.08
WIRE_L = 24                  # chain >= 24 columns at half resolution (1024) = 8.4 deg


def _names():
    n = []
    for s in range(len(GIST_SCALES)):
        for bi in range(len(GIST_BANDS)):
            for f in ("e_mean", "e_std", "e_max", "fv", "fh"):
                n.append("g%d_b%d_%s" % (s, bi, f))
    n += ["b_sv_dens", "b_so_dens", "b_straight", "b_built_frac", "b_built_sect"]
    n += ["p_" + p for p in PALETTE]
    n += ["sky_frac", "sky_hband", "skyline_med", "skyline_iqr", "skyline_rough", "sky_open"]
    n += ["pole_n360", "pole_top", "pole_width", "pole_thick", "pole_rel_l", "pole_a", "pole_b"]
    n += ["wire_cols", "wire_count", "wire_smax", "wire_pix"]
    for li in range(3):
        for ai in range(4):
            for bi in range(4):
                n.append("h_L%da%db%d" % (li, ai, bi))
    for name, _, _ in COLOR_BANDS:
        n += ["c%s_L" % name, "c%s_a" % name, "c%s_b" % name, "c%s_Lstd" % name]
    return n


FEATURE_NAMES = _names()

# ----------------------------------------------------------------- helpers
_LUMA_W = np.array([0.299, 0.587, 0.114], np.float32)
_v = np.arange(256, dtype=np.float64) / 255.0
_SRGB_LIN = np.where(_v > 0.04045, ((_v + 0.055) / 1.055) ** 2.4, _v / 12.92).astype(np.float32)
_XYZ_M = (np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
          / np.array([0.95047, 1.0, 1.08883])[:, None]).T.astype(np.float32)
del _v


def _luma(rgb_u8):
    return rgb_u8.astype(np.float32) @ _LUMA_W


def _lab(rgb_u8):
    """CIELAB (D65) of uint8 sRGB pixels (..., 3) -> float32 (..., 3)."""
    xyz = _SRGB_LIN[rgb_u8] @ _XYZ_M
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _elev(h):
    return (90.0 - (np.arange(h) + 0.5) / h * 180.0).astype(np.float32)


def _row(h, el):
    return int(np.clip(round((90.0 - el) / 180.0 * h), 0, h))


def _down2(g, m):
    """2x2 block average of image g over valid pixels; block valid if >= 3 of 4 valid."""
    h, w = g.shape
    mf = m.astype(np.float32)
    gs = (g * mf).reshape(h // 2, 2, w // 2, 2).sum((1, 3))
    ms = mf.reshape(h // 2, 2, w // 2, 2).sum((1, 3))
    return gs / np.maximum(ms, 1.0), ms >= 3


def _box_axis(a, k, axis):
    """Sum of a over a centred window of length k (odd) along axis (zero padded)."""
    a = a.astype(np.float32)
    r = k // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad), axis=axis)
    n = a.shape[axis]
    if axis == 0:
        return c[k:k + n] - c[:n]
    return c[:, k:k + n] - c[:, :n]


def _box_wrap_x(a, k):
    """Centred horizontal window sum with azimuth wrap-around."""
    r = k // 2
    p = np.concatenate([a[:, -r - 1:], a, a[:, :r]], 1).astype(np.float32)
    c = np.cumsum(p, 1)
    n = a.shape[1]
    return c[:, k:k + n] - c[:, :n]


def _box2(a, k):
    return _box_wrap_x(_box_axis(a, k, 0), k)


def _shift_or(b, axis, d=1):
    out = b.copy()
    for s in range(1, d + 1):
        out |= np.roll(b, s, axis) | np.roll(b, -s, axis)
    return out


def _sector_sums(a, r0, r1):
    """Sum of a[r0:r1] per azimuth sector -> (N_SECT,)."""
    return a[r0:r1].sum(0).reshape(N_SECT, -1).sum(1)


# ------------------------------------------------------------ GIST energy
def _gradients(g, m, r0, r1):
    """Central-difference gradients for rows r0..r1 of g (sphere-corrected gx,
    gy > 0 = brighter upwards) and their validity."""
    h = g.shape[0]
    el = _elev(h)[r0:r1]
    a, b = max(r0 - 1, 0), min(r1 + 1, h)
    G = g[a:b]
    M = m[a:b]
    o = r0 - a
    n = r1 - r0
    gc = G[o:o + n]
    gx = (np.roll(gc, -1, 1) - np.roll(gc, 1, 1)) * (0.5 / np.maximum(np.cos(np.radians(el)), 0.25))[:, None]
    up = G[o - 1:o - 1 + n] if o >= 1 else gc
    dn = G[o + 1:o + 1 + n] if o + n < G.shape[0] else gc
    gy = (up - dn) * 0.5
    mc = M[o:o + n]
    v = mc & np.roll(mc, 1, 1) & np.roll(mc, -1, 1)
    v &= (M[o - 1:o - 1 + n] if o >= 1 else False) | False if o < 1 else v & M[o - 1:o - 1 + n]
    if o + n < M.shape[0]:
        v &= M[o + 1:o + 1 + n]
    else:
        v[-1] = False
    if o < 1:
        v[0] = False
    return gx.astype(np.float32), gy.astype(np.float32), v


def _gist_pool(mag, w0, w2, v, r_off, h, out):
    w = mag.shape[1]
    vf = v.astype(np.float32)
    cell = w // N_SECT
    e0 = mag * w0
    e2 = mag * w2
    for top, bot in GIST_BANDS:
        r0, r1 = _row(h, top) - r_off, _row(h, bot) - r_off
        cnt = _sector_sums(vf, r0, r1)
        ok = cnt >= 0.3 * cell * (r1 - r0)
        if not ok.any():
            out += [np.nan] * 5
            continue
        e = _sector_sums(mag, r0, r1)[ok]
        le = np.log(e / cnt[ok] + 0.5)
        tot = e.sum() + 1e-6
        out += [float(le.mean()), float(le.std()) if ok.sum() > 1 else np.nan, float(le.max()),
                float(_sector_sums(e0, r0, r1)[ok].sum() / tot), float(_sector_sums(e2, r0, r1)[ok].sum() / tot)]


def _gist_rows(h):
    return _row(h, GIST_BANDS[0][0]), _row(h, GIST_BANDS[-1][1])


def _orient_w(phi):
    """triangular tuning weights for the 0 and 90 deg channels from phi in [0, pi/2]."""
    t = phi * (4.0 / np.pi)
    return np.clip(1.0 - t, 0.0, 1.0), np.clip(t - 1.0, 0.0, 1.0)


# ------------------------------------------------------------------- sky
def _sky(rgb_u8, m):
    """Sky mask on a (h, w, 3) uint8 canvas (see docstring)."""
    h, w = m.shape
    el = _elev(h)
    rgb = rgb_u8.astype(np.float32)
    Y = rgb @ _LUMA_W
    up = m & (el[:, None] > 0)
    if up.sum() < 50:
        return np.zeros_like(m)
    ref = float(np.percentile(Y[up], 95))
    last = _row(h, -3)
    Yc = Y[:last + 1]
    gy = np.zeros_like(Yc)
    gy[1:-1] = np.abs(Yc[2:] - Yc[:-2])
    gx = np.abs(np.roll(Yc, -1, 1) - np.roll(Yc, 1, 1))
    grad = _box2(gx + gy, 3)[:last] / 9.0
    r, g, b = rgb[:last, :, 0], rgb[:last, :, 1], rgb[:last, :, 2]
    rng = rgb_u8[:last].max(-1).astype(np.float32) - rgb_u8[:last].min(-1)
    Yl = Y[:last]
    blue = (b - r > 6) & (b >= g - 8) & (Yl > 60)
    cloud = (rng < 45) & (Yl > max(0.5 * ref, 100.0)) & (b >= r - 10)
    skyish = (blue | cloud) & (grad < 90) & m[:last]
    bad = (m[:last] & ~skyish).astype(np.int16)
    seed = skyish & (np.cumsum(bad, axis=0) <= 4)
    sky = np.zeros((h, w), bool)
    prev = np.zeros(w, bool)
    for y in range(last):
        sk = skyish[y]
        extra = sk & ~seed[y]
        if not extra.any():
            row = sk
        else:
            start = sk & ~np.roll(sk, 1)
            if not start.any():          # the whole row is one sky run
                row = sk if (prev.any() or seed[y].any()) else seed[y]
            else:
                lab = np.cumsum(start)
                if sk[0] and not start[0]:   # run wrapping around the 360 deg seam
                    lab[lab == 0] = lab[-1]
                lab = lab * sk
                hit = np.bincount(lab, weights=(prev | seed[y]) & sk, minlength=int(lab.max()) + 1) > 0
                hit[0] = False
                row = hit[lab]
        sky[y] = row
        prev = row
    return sky


# ---------------------------------------------------------- built + palette
def _straight_edges(th, edge, nb=8, dists=(-6, -3, 3, 6), need=3):
    """Edge pixels with >= ``need`` collinear, similarly oriented (+-1 bin) edge
    pixels at the given offsets along their own edge direction.  Returns the
    straight-edge map and the orientation bin (0 = vertical edge)."""
    b = (np.floor(th * (nb / np.pi) + 0.5).astype(np.int8)) % nb
    bins = [edge & (b == k) for k in range(nb)]
    out = np.zeros(edge.shape, bool)
    for k in range(nb):
        ek = bins[k]
        if not ek.any():
            continue
        tol = bins[k] | bins[(k - 1) % nb] | bins[(k + 1) % nb]
        t = k * np.pi / nb
        cnt = np.zeros(edge.shape, np.int8)
        for d in dists:
            dx = int(round(-math.sin(t) * d))
            dy = int(round(-math.cos(t) * d))
            cnt += np.roll(np.roll(tol, dy, 0), dx, 1)
        out |= ek & (cnt >= need)
    return out, b


def _built(rgb1, th, mag, v, sky1, g_off, h, out):
    """th, mag, v: orientation / magnitude / validity rows starting at g_off."""
    w = mag.shape[1]
    b0, b1 = _row(h, 40), _row(h, -6)
    G = slice(b0 - g_off, b1 - g_off)
    vv = v[G]
    edge = (mag[G] > EDGE_T) & vv
    st, ob = _straight_edges(th[G], edge)
    sv = st & ((ob == 0) | (ob == 1) | (ob == 7))     # within ~34 deg of vertical
    so = st & ~sv                                      # horizontal / oblique lines
    vf = vv.astype(np.float32)

    def rows(top, bot):
        return slice(_row(h, top) - b0, _row(h, bot) - b0)

    def dens(a, top, bot):
        r = rows(top, bot)
        n = vf[r].sum()
        return float(a[r].sum() / n) if n > 200 else np.nan

    sv_d, so_d = dens(sv, 30, 2), dens(so, 30, 2)
    r = rows(30, 0)
    ne = edge[r].sum()
    straight = float(st[r].sum() / ne) if ne > 100 else np.nan
    skyb = sky1[b0:b1]
    k = 21
    nv_, no_, ne_, den, skyd = _box2(sv, k), _box2(so, k), _box2(edge, k), _box2(vf, k), _box2(skyb, k)
    den1 = np.maximum(den, 1.0)
    built = ((nv_ >= 0.01 * den1) & (no_ >= 0.01 * den1) & (nv_ + no_ >= 0.04 * den1)
             & (nv_ + no_ >= 0.3 * np.maximum(ne_, 1.0)) & (skyd < 0.5 * den1) & vv & ~skyb)
    r = rows(20, 0)
    bb, vb = built[r], vf[r]
    nv = vb.sum()
    if nv > 500:
        bf = float(bb.sum() / nv)
        cs = _sector_sums(vb, 0, vb.shape[0])
        bs = _sector_sums(bb.astype(np.float32), 0, vb.shape[0])
        okc = cs > 0.3 * vb.shape[0] * (w // N_SECT)
        bsect = float((bs[okc] / cs[okc] > 0.15).mean()) if okc.any() else np.nan
    else:
        bf, bsect = np.nan, np.nan
    out += [sv_d, so_d, straight, bf, bsect]
    pal = [np.nan] * len(PALETTE)
    if nv > 500 and bb.sum() > 150:
        lab = _lab(rgb1[b0 + r.start:b0 + r.stop][bb])
        L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
        C = np.hypot(a, b)
        hue = np.degrees(np.arctan2(b, a))
        use = ~((a < -5) & (b > 3))                    # drop vegetation
        n = use.sum()
        if n > 150:
            red = (hue > 10) & (hue < 65) & (C > 14)
            cls = [red & (C >= 22) & (L >= 42), red & (L < 42), (L > 78) & (C < 10),
                   (L > 40) & (L <= 78) & (C < 8), (L <= 40) & (C < 10), (C > 25) & ~red]
            pal = [float((c & use).sum() / n) for c in cls]
    out += pal


def _skyline(sky, m, out):
    h, w = sky.shape
    el = _elev(h)
    up = m & (el[:, None] > 0)
    nup = up.sum()
    if nup < 200:
        out += [np.nan] * 6
        return
    sf = float((sky & up).sum() / nup)
    r0, r1 = _row(h, 20), _row(h, 2)
    nb = m[r0:r1].sum()
    shb = float(sky[r0:r1].sum() / nb) if nb > 100 else np.nan
    colok = m[:_row(h, 30)].any(0) & m[_row(h, 30):_row(h, -5)].all(0)
    has = sky.any(0) & colok
    if has.sum() < 20:
        out += [sf, shb, np.nan, np.nan, np.nan, np.nan]
        return
    last = h - 1 - np.argmax(sky[::-1], axis=0)
    skl = np.where(has, el[last], np.nan)
    val = skl[has]
    d1 = np.abs(np.roll(skl, -1) - skl)
    d1 = d1[~np.isnan(d1)]
    out += [sf, shb, float(np.median(val)), float(np.subtract(*np.percentile(val, [75, 25]))),
            float(np.minimum(d1, 10.0).mean()) if len(d1) > 10 else np.nan, float((val < 3).mean())]


# ------------------------------------------------------ poles and wires
def _hmeans(cs, a, b, n, pad):
    """mean over x+a .. x+b (inclusive) from padded cumsum cs (leading zero column)."""
    return (cs[:, pad + b + 1:pad + b + 1 + n] - cs[:, pad + a:pad + a + n]) * (1.0 / (b - a + 1))


def _poles(rgb1, g1, m1, sky1, out):
    """Pole detector on the 1024 canvas (see module docstring)."""
    h, w = m1.shape
    el = _elev(h)
    p0, p1 = _row(h, 45), _row(h, -3)
    Y = g1[p0:p1]
    M = m1[p0:p1]
    S = sky1[p0:p1]
    n = w
    pad = 16
    Yp = np.concatenate([Y[:, -pad:], Y, Y[:, :pad]], 1)
    cs = np.concatenate([np.zeros((Y.shape[0], 1), np.float32), np.cumsum(Yp, 1)], 1)
    Mp = np.concatenate([M[:, -pad:], M, M[:, :pad]], 1).astype(np.float32)
    cm = np.concatenate([np.zeros((Y.shape[0], 1), np.float32), np.cumsum(Mp, 1)], 1)
    best = np.zeros(Y.shape, np.float32)
    bscale = np.zeros(Y.shape, np.int8)
    for i, hw in enumerate(POLE_HW):
        f = max(2, hw)
        a0, a1 = -(hw - 1), hw - 1
        C = _hmeans(cs, a0, a1, n, pad)
        Lf = _hmeans(cs, a0 - 1 - f, a0 - 2, n, pad)
        Rf = _hmeans(cs, a1 + 2, a1 + 1 + f, n, pad)
        full = _hmeans(cm, a0 - 1 - f, a1 + 1 + f, n, pad) > 0.999
        off = a1 + 1 + f
        skyok = np.roll(S, off, 1) & np.roll(S, -off, 1)
        R = (np.minimum(Lf, Rf) - C) / (np.maximum(Lf, Rf) + 5.0)
        R = np.where(full & skyok, R, 0.0)
        upd = R > best
        best = np.where(upd, R, best)
        bscale = np.where(upd, i, bscale)
    P = best > POLE_T
    K = POLE_K
    Pd = _shift_or(P, 1)
    c = np.concatenate([np.zeros((1, n), np.float32), np.cumsum(Pd, 0, dtype=np.float32)], 0)
    win = c[K:] - c[:-K]                     # vertical window starting at row r
    cP = np.concatenate([np.zeros((1, n), np.float32), np.cumsum(P, 0, dtype=np.float32)], 0)
    wP = cP[K:] - cP[:-K]
    near = _box_wrap_x(wP, 21) - _box_wrap_x(wP, 5)
    passw = (win >= 0.8 * K) & (near <= 0.08 * 16 * K)
    col = passw.any(0)
    widths_deg = (2 * np.array(POLE_HW, np.float32) - 1) * 360.0 / w
    poles = []
    if col.any():
        idx = np.nonzero(col)[0]
        groups = []
        cur = [idx[0]]
        for x in idx[1:]:
            if x - cur[-1] <= 2:
                cur.append(x)
            else:
                groups.append(cur)
                cur = [x]
        groups.append(cur)
        if len(groups) > 1 and groups[0][0] + w - groups[-1][-1] <= 2:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
        strength = passw.sum(0)
        for gcols in groups:
            gcols = np.array(gcols)
            x = int(gcols[np.argmax(strength[gcols])])
            rws = np.nonzero(passw[:, x])[0]
            top_r = rws.min()
            bot_r = rws.max() + K - 1
            below = slice(bot_r + 2, bot_r + 6)
            if bot_r + 6 <= S.shape[0] and (S[below, (x - 4) % w] & S[below, (x + 4) % w]).mean() > 0.5 \
                    and not P[below, x].any() and not P[below, (x + 1) % w].any():
                continue                       # floating in the sky: cloud / wire, not a pole
            if top_r >= 4:
                above = [S[top_r - 4:top_r - 1, (x + d) % w].mean() > 0.5 for d in (-4, 0, 4)]
                if sum(above) < 2:
                    continue                   # starts inside clutter (tree crown)
            pr = np.arange(top_r, bot_r + 1)
            xs = x + np.argmax(np.stack([best[pr, (x + d) % w] for d in (-1, 0, 1)]), 0) - 1
            ok = P[pr, xs % w]
            if ok.sum() < 5:
                continue
            ys, xs = pr[ok], xs[ok] % w
            poles.append({"x": x, "top": float(el[p0 + top_r]), "bot": float(el[p0 + bot_r]),
                          "width": float(np.median(widths_deg[bscale[ys, xs]])),
                          "rel": float(np.median(1.0 - best[ys, xs])),
                          "ys": p0 + ys, "xs": xs})
    azfrac = float((M.mean(0) > 0.5).mean())
    if azfrac < 0.05 or S.sum() < 100:
        out += [np.nan] * 7
    elif not poles:
        out += [0.0] + [np.nan] * 6
    else:
        tops = np.array([p["top"] for p in poles])
        wid = np.array([p["width"] for p in poles])
        thick = np.radians(wid) / np.tan(np.radians(np.maximum(tops, 2.0)))
        lab = _lab(np.concatenate([rgb1[p["ys"], p["xs"]] for p in poles], 0))
        out += [len(poles) / azfrac, float(np.median(tops)), float(np.median(wid)), float(np.median(thick)),
                float(np.median([p["rel"] for p in poles])), float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))]
    return poles


def _chain_dp(W):
    """Length of the longest gap-free chain (<= 1 row change per column) of W
    pixels through each pixel.  W: (n_cols, n_rows) bool; columns wrap."""
    n, nr = W.shape
    ext = min(64, n)
    idx = np.concatenate([np.arange(n - ext, n), np.arange(n), np.arange(ext)])
    Wx = W[idx]
    live = Wx.any(1)
    res = []
    for order in (range(len(idx)), range(len(idx) - 1, -1, -1)):
        F = np.zeros(Wx.shape, np.int16)
        prev = None
        for i in order:
            if not live[i]:
                prev = None
                continue
            wcol = Wx[i]
            if prev is None:
                cur = wcol.astype(np.int16)
            else:
                m = prev.copy()
                np.maximum(m[1:], prev[:-1], out=m[1:])
                np.maximum(m[:-1], prev[1:], out=m[:-1])
                cur = (m + 1) * wcol
            F[i] = cur
            prev = cur
        res.append(F[ext:ext + n])
    return res[0] + res[1] - 1


def _wires(rgbw, mw, skyw, ctxw, h, out):
    """Overhead wire detector (see module docstring).  rgbw / mw / skyw / ctxw:
    the canvas at WIRE_W (rows el 65..3 are used)."""
    w = mw.shape[1]
    sc = w / 2048.0
    q0, q1 = _row(h, 65), _row(h, 3)
    q1 = q0 + ((q1 - q0) // 2) * 2
    a, b = q0 - 6, q1 + 6
    Z = rgbw[a:b, :, 2].astype(np.float32)          # blue channel with 6 rows margin
    nr = q1 - q0

    def rows(d0, d1):                                # mean of rows y+d0 .. y+d1
        s = Z[6 + d0:6 + d0 + nr].copy()
        for d in range(d0 + 1, d1 + 1):
            s += Z[6 + d:6 + d + nr]
        return s * (1.0 / (d1 - d0 + 1))

    Yw = Z[6:6 + nr]
    U, B = rows(-3, -2), rows(2, 3)
    den = np.maximum(U, B) + 5.0
    N1 = np.minimum(U, B) - Yw
    N2 = np.minimum(rows(-4, -3), rows(3, 4)) - rows(-1, 1)
    N3 = np.minimum(rows(-6, -4), rows(4, 6)) - rows(-2, 2)
    Nt = np.maximum(N1, N2)
    Sw = skyw[q0:q1]
    skyok = np.zeros_like(Sw)
    skyok[4:-4] = Sw[:-8] & Sw[8:]
    ridge = Nt > WIRE_T * den
    Wp = ridge & (Nt > 1.25 * N3) & skyok & mw[q0:q1] & ctxw[q0:q1]
    # clutter: thin VERTICAL ridges that are not on a horizontal ridge (branches)
    Yp = np.concatenate([Yw[:, -3:], Yw, Yw[:, :3]], 1)
    Lf = Yp[:, 0:w] + Yp[:, 1:w + 1]
    Rf = Yp[:, 5:w + 5] + Yp[:, 6:w + 6]
    Vp = (np.minimum(Lf, Rf) * 0.5 - Yw) > WIRE_T * (np.maximum(Lf, Rf) * 0.5 + 5.0)
    Vp &= ~_shift_or(ridge, 0, 1)
    Wh = Wp.reshape(nr // 2, 2, w // 2, 2).any((1, 3))
    Vh = Vp.reshape(nr // 2, 2, w // 2, 2).any((1, 3))
    Sh = Sw.reshape(nr // 2, 2, w // 2, 2).all((1, 3))
    clutter = _box2(Vh, 9) * (1.0 / 81.0)
    around = _box_wrap_x(_box_axis(Wh, 11, 0) - _box_axis(Wh, 3, 0), 5)
    Wh &= (clutter < WIRE_CLUTTER) & (around <= 8)
    if Wh.any():
        Lc = _chain_dp(np.ascontiguousarray(Wh.T)).T
        wire = Wh & (Lc >= int(round(WIRE_L * sc)))
    else:
        wire = Wh
    skycols = Sh.sum(0) >= 8 * sc
    if skycols.sum() < 16 * sc:
        out += [np.nan] * 4
        return wire, q0
    crossed = wire.any(0)
    runs = (wire[1:] & ~wire[:-1]).sum(0) + wire[0]
    sc_n = skycols.reshape(N_SECT, -1).sum(1)
    sc_c = (crossed & skycols).reshape(N_SECT, -1).sum(1)
    oks = sc_n >= 4 * sc
    out += [float(crossed[skycols].mean()), float(runs[skycols].mean()),
            float((sc_c[oks] / sc_n[oks]).max()) if oks.any() else np.nan,
            float(1000.0 * (wire & Sh).sum() / max(Sh.sum(), 1))]
    return wire, q0


# --------------------------------------------------------------- colour
def _colour(rgb5, m5, out):
    h = m5.shape[0]
    el = _elev(h)
    r1 = _row(h, -55)
    lab = _lab(rgb5[:r1])
    wgt = (np.cos(np.radians(el[:r1]))[:, None] * m5[:r1]).astype(np.float32)
    tot = wgt.sum()
    if tot < 50:
        out += [np.nan] * (48 + 4 * len(COLOR_BANDS))
        return None
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    idx = np.searchsorted(L_EDGES, L.ravel()) * 16 + np.searchsorted(A_EDGES, a.ravel()) * 4 \
        + np.searchsorted(B_EDGES, b.ravel())
    hist = np.bincount(idx, weights=wgt.ravel(), minlength=48)[:48] / tot
    out += np.sqrt(hist).tolist()
    for _, top, bot in COLOR_BANDS:
        r0, rr = _row(h, top), _row(h, bot)
        ww = wgt[r0:rr]
        sw = ww.sum()
        if sw < 0.002 * tot or sw < 20:
            out += [np.nan] * 4
            continue
        mL = float((L[r0:rr] * ww).sum() / sw)
        out += [mL, float((a[r0:rr] * ww).sum() / sw), float((b[r0:rr] * ww).sum() / sw),
                float(np.sqrt((((L[r0:rr] - mL) ** 2) * ww).sum() / sw))]
    return hist


# ---------------------------------------------------------------- extract
def extract(sph, debug=None):
    out = []
    s1 = sph.resized(1024)
    rgb1, m1 = s1.rgb, s1.mask
    g1 = _luma(rgb1)
    h1 = m1.shape[0]
    # --- GIST, scale 0 (also feeds the built-environment detector)
    r0, r1 = _gist_rows(h1)
    gx, gy, v = _gradients(g1, m1, r0, r1)
    mag = np.sqrt(gx * gx + gy * gy) * v
    th = np.arctan2(gy, gx) % np.pi
    phi = np.minimum(th, np.pi - th)
    w0, w2 = _orient_w(phi)
    _gist_pool(mag, w0, w2, v, r0, h1, out)
    g, m = g1, m1
    for _ in GIST_SCALES[1:]:
        g, m = _down2(g, m)
        hh = m.shape[0]
        a, b = _gist_rows(hh)
        gxs, gys, vs = _gradients(g, m, a, b)
        mags = np.sqrt(gxs * gxs + gys * gys) * vs
        w0s, w2s = _orient_w(np.arctan2(np.abs(gys), np.abs(gxs)))
        _gist_pool(mags, w0s, w2s, vs, a, hh, out)
    # --- sky (512) and built environment (1024)
    s5 = sph.resized(512)
    sky5 = _sky(s5.rgb, s5.mask)
    sky1 = np.repeat(np.repeat(sky5, 2, 0), 2, 1) & m1
    _built(rgb1, th, mag, v, sky1, r0, h1, out)
    _skyline(sky5, s5.mask, out)
    # --- poles (1024) and wires (2048)
    poles = _poles(rgb1, g1, m1, sky1, out)
    sw = sph.resized(WIRE_W)
    fw = WIRE_W // 512
    skyw = np.repeat(np.repeat(sky5, fw, 0), fw, 1) & sw.mask
    ctx5 = _box2(sky5, 9) >= 0.6 * 81
    ctxw = np.repeat(np.repeat(ctx5, fw, 0), fw, 1)
    wire, wq0 = _wires(sw.rgb, sw.mask, skyw, ctxw, sw.h, out)
    hist = _colour(s5.rgb, s5.mask, out)
    x = np.array(out, np.float32)
    assert len(x) == len(FEATURE_NAMES), (len(x), len(FEATURE_NAMES))
    if debug is not None:
        debug.update({"poles": poles, "wire": wire, "wire_q0": wq0, "sky5": sky5})
    return {"x": x, "evidence": _evidence(x, poles, hist)}


def _hist_desc(i):
    li, ai, bi = i // 16, (i // 4) % 4, i % 4
    return "%s, %s a*, %s b*" % (["dark", "mid", "bright"][li], ["green", "neutral", "warm", "red"][ai],
                                 ["blue", "neutral", "yellowish", "yellow"][bi])


def _evidence(x, poles, hist):
    f = dict(zip(FEATURE_NAMES, x.tolist()))
    ev = {}
    bf = f["b_built_frac"]
    if not np.isnan(bf):
        if bf > 0.25:
            lvl = "urban"
        elif bf > 0.10:
            lvl = "suburban / village"
        elif bf > 0.03:
            lvl = "rural, scattered buildings"
        else:
            lvl = "no buildings"
        conf = float(np.clip(abs(bf - 0.1) / 0.15 + 0.4, 0.4, 0.95))
        ev["built_environment"] = {"level": lvl, "built_fraction": round(bf, 3),
                                   "sectors_with_buildings": None if np.isnan(f["b_built_sect"]) else round(f["b_built_sect"], 2),
                                   "confidence": round(conf, 2)}
        pal = {p: f["p_" + p] for p in PALETTE}
        if not any(np.isnan(v) for v in pal.values()) and bf > 0.03:
            top = max(pal, key=pal.get)
            if pal[top] > 0.12:
                label = {"terracotta": "terracotta / orange roofs", "brick": "red-brown brick",
                         "white": "white plaster walls", "grey": "grey concrete", "dark": "dark roofs / walls",
                         "colour": "coloured (painted metal) roofs"}[top]
                ev["building_palette"] = {"dominant": label,
                                          "fractions": {k: round(v, 3) for k, v in pal.items()},
                                          "confidence": round(float(np.clip(pal[top] * 1.5, 0.2, 0.9)), 2)}
    n = f["pole_n360"]
    if not np.isnan(n):
        if n >= 1:
            rel = f["pole_rel_l"]
            tone = None
            if not np.isnan(rel):
                tone = "dark (wooden?)" if rel < 0.45 else ("light (concrete?)" if rel > 0.65 else "medium tone")
            ev["utility_poles"] = {"count_per_360": round(n, 1), "tone": tone,
                                   "median_top_elevation": round(f["pole_top"], 1),
                                   "median_width_deg": round(f["pole_width"], 2),
                                   "positions_deg": [round((p["x"] + 0.5) / 1024.0 * 360 - 180, 1) for p in poles][:20],
                                   "confidence": round(float(np.clip(0.4 + 0.1 * n, 0.4, 0.9)), 2)}
        else:
            ev["utility_poles"] = {"count_per_360": 0, "confidence": 0.5}
    wc = f["wire_cols"]
    if not np.isnan(wc):
        lvl = "dense overhead wires" if wc > 0.35 else ("some overhead wires" if wc > 0.08 else "no overhead wires")
        ev["overhead_wires"] = {"level": lvl, "crossed_sky_fraction": round(wc, 3),
                                "wires_per_column": round(f["wire_count"], 2),
                                "confidence": round(float(np.clip(0.5 + abs(wc - 0.1) * 2, 0.5, 0.9)), 2)}
    if not np.isnan(f["sky_frac"]):
        ev["sky"] = {"sky_fraction_upper": round(f["sky_frac"], 3),
                     "skyline_median_el": None if np.isnan(f["skyline_med"]) else round(f["skyline_med"], 1),
                     "open_horizon_fraction": None if np.isnan(f["sky_open"]) else round(f["sky_open"], 2)}
    if hist is not None:
        ev["dominant_colours"] = [{"bin": _hist_desc(i), "share": round(float(hist[i]), 3)}
                                  for i in np.argsort(-hist)[:3]]
    return ev
