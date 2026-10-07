"""
texture: generic, rotation-invariant scene descriptors per elevation band.

Classical colour / texture statistics (no learning inside the extractor):
  * joint CIE L*a*b* histogram, 4 x 4 x 4 bins, weighted by pixel solid angle (cos el),
    square-root (Hellinger) normalised;
  * uniform local binary patterns (8 neighbours, radius 1 on the 1024 canvas and on the
    512 canvas = two scales): 10-bin histogram of the number of brighter neighbours
    (9 = non-uniform pattern), square-root normalised;
  * gradient orientation histogram (8 bins over 0..180 deg, magnitude weighted, relative to
    the image axes: vertical / horizontal / oblique structure) and the mean gradient.
Bands: upper (el 10..40: canopy, buildings, hills), horizon (-5..10), ground (-40..-10).
Only valid (mask) pixels are used; a band covered by < 5 % valid pixels gives NaN.
"""
import numpy as np

NAME = "texture"
BANDS = (("up", 40.0, 10.0), ("hor", 10.0, -5.0), ("gnd", -10.0, -40.0))
L_EDGES = np.array([35.0, 55.0, 75.0])
A_EDGES = np.array([-9.0, -2.0, 5.0])
B_EDGES = np.array([-8.0, 3.0, 15.0])
FEATURE_NAMES = []
for _b, _t, _l in BANDS:
    FEATURE_NAMES += ["%s_lab%d%d%d" % (_b, i, j, k) for i in range(4) for j in range(4) for k in range(4)]
    FEATURE_NAMES += ["%s_lbp1_%d" % (_b, i) for i in range(10)]
    FEATURE_NAMES += ["%s_lbp2_%d" % (_b, i) for i in range(10)]
    FEATURE_NAMES += ["%s_ori%d" % (_b, i) for i in range(8)] + ["%s_gmag" % _b]

_c = np.arange(256, dtype=np.float64) / 255.0
_LIN = np.where(_c > 0.04045, ((_c + 0.055) / 1.055) ** 2.4, _c / 12.92).astype(np.float32)
_M = (np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
      / np.array([[0.95047], [1.0], [1.08883]])).astype(np.float32)
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
# uniform-LBP lookup: pattern -> number of ones (0..8) if <= 2 circular transitions else 9
_LUT = np.zeros(256, np.int64)
for _p in range(256):
    _bits = [(_p >> _i) & 1 for _i in range(8)]
    _tr = sum(_bits[_i] != _bits[(_i + 1) % 8] for _i in range(8))
    _LUT[_p] = sum(_bits) if _tr <= 2 else 9


def _lab(rgb):
    xyz = _LIN[rgb] @ _M.T
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return 116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])


def _lbp(L):
    """Uniform LBP code per pixel (x wraps around), edges rows get code from clamped rows."""
    P = np.pad(L, ((1, 1), (0, 0)), mode="edge")
    c = P[1:-1]
    code = np.zeros(L.shape, np.int64)
    for i, (dy, dx) in enumerate(_NB):
        nb = np.roll(P[1 + dy:P.shape[0] - 1 + dy], -dx, axis=1)
        code |= ((nb > c + 1.0).astype(np.int64) << i)
    return _LUT[code]


def _band_feats(s, L, A, B, lbp1, lbp2, el, el2, gy, gx):
    out = []
    for name, top, bot in BANDS:
        rows = (el <= top) & (el >= bot)
        m = s.mask & rows[:, None]
        if m.sum() < 0.05 * rows.sum() * s.w or m.sum() < 200:
            out += [np.nan] * (64 + 10 + 10 + 9)
            continue
        w = np.repeat(np.cos(np.radians(el))[:, None], s.w, 1)[m]
        li = np.searchsorted(L_EDGES, L[m])
        ai = np.searchsorted(A_EDGES, A[m])
        bi = np.searchsorted(B_EDGES, B[m])
        h = np.bincount(li * 16 + ai * 4 + bi, weights=w, minlength=64)
        out += list(np.sqrt(h / max(h.sum(), 1e-9)))
        h = np.bincount(lbp1[m], weights=w, minlength=10)
        out += list(np.sqrt(h / max(h.sum(), 1e-9)))
        rows2 = (el2 <= top) & (el2 >= bot)
        m2 = s.m2 & rows2[:, None]
        if m2.sum() > 50:
            h = np.bincount(lbp2[m2], minlength=10).astype(float)
            out += list(np.sqrt(h / max(h.sum(), 1e-9)))
        else:
            out += [np.nan] * 10
        mag = np.hypot(gx[m], gy[m])
        ori = (np.degrees(np.arctan2(gy[m], gx[m])) % 180.0 / 22.5).astype(np.int64) % 8
        h = np.bincount(ori, weights=mag * w, minlength=8)
        out += list(h / max(h.sum(), 1e-9))
        out.append(float(np.sum(mag * w) / max(w.sum(), 1e-9)))
    return out


def extract(sph):
    s = sph.resized(1024)
    L, A, B = _lab(s.rgb)
    el = s.elevations()
    lbp1 = _lbp(L)
    s2 = sph.resized(512)
    L2, _, _ = _lab(s2.rgb)
    lbp2 = _lbp(L2)
    s.m2 = s2.mask
    gx = (np.roll(L, -1, axis=1) - np.roll(L, 1, axis=1)) * 0.5
    gy = np.zeros_like(L)
    gy[1:-1] = (L[:-2] - L[2:]) * 0.5
    # gradients across the mask border are not image structure
    edge = s.mask & np.roll(s.mask, 1, 1) & np.roll(s.mask, -1, 1)
    edge[1:-1] &= s.mask[:-2] & s.mask[2:]
    gx = np.where(edge, gx, 0.0)
    gy = np.where(edge, gy, 0.0)
    x = _band_feats(s, L, A, B, lbp1, lbp2, el, s2.elevations(), gy, gx)
    del s.m2
    return {"x": np.array(x, np.float32), "evidence": {}}
