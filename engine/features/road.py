"""
Road surface and lane markings from an inverse-perspective (bird's-eye) map.

(full description of features and maths: see the end of development; draft)
"""

import math

import numpy as np

NAME = "road"

FEATURE_NAMES = [
    "ipm_valid", "road_L", "road_a", "road_b", "road_chroma", "road_texture",
    "surf_asphalt", "surf_concrete", "surf_dirt", "surf_sand", "paved",
    "road_width_m", "road_center_off_m",
    "markings", "no_markings", "yellow_center", "white_center", "yellow_edge", "white_edge",
    "yellow_any", "double_center", "dashed_center", "dashed_edge", "center_off_m",
    "lane_width_m", "n_lines", "right_hand_traffic",
    "curb", "sidewalk", "verge_a", "verge_b", "verge_green",
]

H_CAM = 2.5           # Street View camera height above the road (m)
DX, DY = 0.10, 0.05   # IPM cell size forward / lateral (m)
XR, YR = 20.0, 10.0   # IPM half extents (m)
R_MIN = 3.5           # nadir / car exclusion radius (m)
R_FINE = 8.0          # closer cells are sampled from a 2x2-averaged canvas (anti-aliasing)
NX, NY = int(round(2 * XR / DX)), int(round(2 * YR / DY))
XS = (-XR + (np.arange(NX) + 0.5) * DX).astype(np.float32)
YS = (-YR + (np.arange(NY) + 0.5) * DY).astype(np.float32)
J0 = NY // 2          # first lateral index with y > 0
PAD = 20              # lateral padding for box sums

_GEOM = {}

# sRGB -> linear lookup table
_LIN = np.arange(256, dtype=np.float64) / 255.0
_LIN = np.where(_LIN > 0.04045, ((_LIN + 0.055) / 1.055) ** 2.4, _LIN / 12.92).astype(np.float32)
_M_XYZ = (np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
          / np.array([0.95047, 1.0, 1.08883])[:, None]).astype(np.float32)


def _lab_from_linear(lin):
    """Linear RGB (..., 3) 0..1 -> CIE L*a*b* (D65)."""
    xyz = lin @ _M_XYZ.T
    f = np.where(xyz > 0.008856, np.cbrt(np.maximum(xyz, 0)), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _srgb_from_linear(lin):
    c = np.clip(lin, 0, 1)
    return np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c) * 255.0


def _sig(z):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))


# ---------------------------------------------------------------- IPM
def _grid():
    if "grid" not in _GEOM:
        X, Y = np.meshgrid(XS, YS, indexing="ij")
        R = np.hypot(X, Y)
        ang = np.degrees(np.arctan2(Y, X))            # left of the axis = positive
        el = -np.degrees(np.arctan2(H_CAM, R))
        _GEOM["grid"] = (X, Y, R, ang, el)
    return _GEOM["grid"]


def _bilin_index(phi, el, w, h, row0=0):
    px = ((phi + 180.0) % 360.0) / 360.0 * w - 0.5
    py = (90.0 - el) / 180.0 * h - 0.5
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    fx = (px - x0).astype(np.float32)[:, None]
    fy = (py - y0).astype(np.float32)[:, None]
    x0m, x1m = x0 % w, (x0 + 1) % w
    y0m, y1m = np.clip(y0, 0, h - 1) - row0, np.clip(y0 + 1, 0, h - 1) - row0
    return (y0m * w + x0m, y0m * w + x1m, y1m * w + x0m, y1m * w + x1m, fx, fy)


def _geom(w, h, car):
    key = (w, h, round(car, 3))
    g = _GEOM.get(key)
    if g is not None:
        return g
    X, Y, R, ang, el = _grid()
    phi = (car - ang).ravel()
    elr = el.ravel()
    Rr = R.ravel()
    near = np.nonzero((Rr >= R_MIN) & (Rr < R_FINE))[0]
    far = np.nonzero(Rr >= R_FINE)[0]
    allv = np.nonzero(Rr >= R_MIN)[0]
    w2, h2 = w // 2, h // 2
    py2 = (90.0 - elr[near]) / 180.0 * h2 - 0.5
    b0 = max(0, int(np.floor(py2.min())))
    b1 = min(h2 - 1, int(np.floor(py2.max())) + 1)
    g = {
        "near": near, "far": far, "all": allv,
        "near_ix": _bilin_index(phi[near], elr[near], w2, h2, row0=b0), "band": (b0, b1, w2),
        "far_ix": _bilin_index(phi[far], elr[far], w, h),
        "mask_ix": (np.clip(np.rint((90.0 - elr[allv]) / 180.0 * h - 0.5), 0, h - 1).astype(np.int64) * w
                    + np.rint(((phi[allv] + 180.0) % 360.0) / 360.0 * w - 0.5).astype(np.int64) % w),
    }
    if len(_GEOM) > 8:
        for k in [k for k in _GEOM if k != "grid"]:
            del _GEOM[k]
    _GEOM[key] = g
    return g


def _gather(flat, ix):
    i00, i01, i10, i11, fx, fy = ix
    a, b, c, d = flat[i00], flat[i01], flat[i10], flat[i11]
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy


def build_ipm(sph, car):
    """IPM linear-RGB (NX, NY, 3) float32 and validity (NX, NY) for car axis ``car`` (relative longitude)."""
    g = _geom(sph.w, sph.h, car)
    lin = np.zeros((NX * NY, 3), np.float32)
    lin[g["far"]] = _gather_lut(sph.rgb, g["far_ix"])
    b0, b1, w2 = g["band"]
    band = sph.rgb[2 * b0:2 * (b1 + 1), :2 * w2]
    band = _LIN[band].reshape(b1 + 1 - b0, 2, w2, 2, 3).mean(axis=(1, 3))
    lin[g["near"]] = _gather(band.reshape(-1, 3), g["near_ix"])
    valid = np.zeros(NX * NY, bool)
    valid[g["all"]] = sph.mask.ravel()[g["mask_ix"]]
    return lin.reshape(NX, NY, 3), valid.reshape(NX, NY)


def _gather_lut(rgb, ix):
    flat = rgb.reshape(-1, 3)
    i00, i01, i10, i11, fx, fy = ix
    a, b, c, d = _LIN[flat[i00]], _LIN[flat[i01]], _LIN[flat[i10]], _LIN[flat[i11]]
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy


# ---------------------------------------------------------------- helpers
def _lat_cumsum(A):
    """Padded lateral cumulative sums of A (..., NX, NY)."""
    S = np.zeros(A.shape[:-1] + (NY + 2 * PAD + 1,), np.float32)
    np.cumsum(A, axis=-1, dtype=np.float32, out=S[..., PAD + 1:PAD + 1 + NY])
    S[..., PAD + 1 + NY:] = S[..., PAD + NY:PAD + NY + 1]
    return S


def _win(S, lo, hi):
    """Window sums over lateral offsets [lo, hi] (cells) for every position."""
    return S[..., PAD + hi + 1:PAD + hi + 1 + NY] - S[..., PAD + lo:PAD + lo + NY]


def _box2d(A, V, r):
    """Box mean over (2r+1)^2 valid cells."""
    def cs2(M):
        M = np.cumsum(np.cumsum(M, 0, dtype=np.float32), 1, dtype=np.float32)
        return np.pad(M, ((1, 0), (1, 0)))
    S, C = cs2(A * V), cs2(V.astype(np.float32))
    n0, n1 = A.shape
    i = np.arange(n0)
    j = np.arange(n1)
    i0, i1 = np.clip(i - r, 0, n0), np.clip(i + r + 1, 0, n0)
    j0, j1 = np.clip(j - r, 0, n1), np.clip(j + r + 1, 0, n1)
    s = S[i1][:, j1] - S[i0][:, j1] - S[i1][:, j0] + S[i0][:, j0]
    c = C[i1][:, j1] - C[i0][:, j1] - C[i1][:, j0] + C[i0][:, j0]
    return s / np.maximum(c, 1)


def _structure_axis(L, V):
    """Dominant road-axis angle (deg, left positive, mod 180) from the doubled-angle gradient mean."""
    gx = np.zeros_like(L)
    gy = np.zeros_like(L)
    gx[1:-1] = (L[2:] - L[:-2]) / (2 * DX)
    gy[:, 1:-1] = (L[:, 2:] - L[:, :-2]) / (2 * DY)
    ok = V.copy()
    ok[1:-1] &= V[2:] & V[:-2]
    ok[:, 1:-1] &= V[:, 2:] & V[:, :-2]
    ok[[0, -1]] = False
    ok[:, [0, -1]] = False
    if ok.sum() < 500:
        return None, 0.0
    gx, gy = gx[ok], gy[ok]
    m2 = gx * gx + gy * gy
    cap = np.percentile(m2, 99) + 1e-9
    wgt = np.minimum(m2, cap) / (m2 + 1e-9)
    c2 = float(np.sum(wgt * (gx * gx - gy * gy)))
    s2 = float(np.sum(wgt * 2 * gx * gy))
    coh = math.hypot(c2, s2) / (float(np.sum(wgt * m2)) + 1e-9)
    grad_dir = 0.5 * math.degrees(math.atan2(s2, c2))    # gradient orientation (x -> y)
    axis = (grad_dir + 90.0 + 90.0) % 180.0 - 90.0        # features run perpendicular to gradients
    return axis, coh


# ---------------------------------------------------------------- lines
THETAS = np.radians(np.arange(-14, 15, 1.0)).astype(np.float32)
TANS = np.tan(THETAS)


def _hough_lines(cand, half, max_lines=8, min_votes=22):
    """Hough peaks (theta, y0, votes) for one half-plane (half = +1 front / -1 rear),
    removing the votes of each accepted line's points before searching the next one."""
    rows = (XS > 0) if half > 0 else (XS < 0)
    ii, jj = np.nonzero(cand & rows[:, None])
    if len(ii) < min_votes:
        return []
    x = XS[ii]
    y = YS[jj]
    nT = len(THETAS)
    y0 = y[:, None] - x[:, None] * TANS[None, :]
    yb = np.rint((y0 + YR) / DY - 0.5).astype(np.int64)
    ok = (yb >= 0) & (yb < NY)
    flat = np.where(ok, np.arange(nT)[None, :] * NY + yb, nT * NY)
    acc = np.bincount(flat.ravel(), minlength=nT * NY + 1)[:nT * NY].reshape(nT, NY).astype(np.float32)
    alive = np.ones(len(ii), bool)
    out = []
    for _ in range(max_lines):
        acc3 = acc.copy()
        acc3[:, 1:] += acc[:, :-1]
        acc3[:, :-1] += acc[:, 1:]
        k = int(np.argmax(acc3))
        t, j = divmod(k, NY)
        v = float(acc3[t, j])
        if v < min_votes:
            break
        out.append((float(THETAS[t]), float(YS[j]), v))
        on = alive & (np.abs(yb[:, t] - j) <= 2)
        if on.any():
            rem = np.bincount(flat[on].ravel(), minlength=nT * NY + 1)[:nT * NY].reshape(nT, NY)
            acc -= rem
            alive &= ~on
    return out


def _line_rows(theta, y0, half):
    rows = np.nonzero((XS > 0) if half > 0 else (XS < 0))[0]
    yl = y0 + XS[rows] * math.tan(theta)
    j = np.rint((yl + YR) / DY - 0.5).astype(np.int64)
    inb = (j >= 1) & (j < NY - 1)
    return rows[inb], j[inb]


def _walk_line(theta, y0, half, cand, V, T_L, T_b, lab, road0):
    """Sample a line through the IPM; returns stats dict or None."""
    rows, j = _line_rows(theta, y0, half)
    if len(rows) < 10:
        return None
    vis = V[rows, j]
    nvis = int(vis.sum())
    if nvis < 20:
        return None
    hit = (cand[rows, j - 1] | cand[rows, j] | cand[rows, j + 1]) & vis
    nh = int(hit.sum())
    if nh < 5:
        return None
    tl = np.stack([T_L[rows, j + d] for d in (-1, 0, 1)], 1)
    tb = np.stack([T_b[rows, j + d] for d in (-1, 0, 1)], 1)
    best = np.argmax(np.maximum(tl, 1.5 * tb), 1)
    k = np.arange(len(best))
    jb = j + best - 1
    # candidate density in flanking bands 0.25-1.0 m on both sides: chance level of the hit rate
    offs = np.concatenate([np.arange(-20, -4), np.arange(5, 21)])
    jj = j[:, None] + offs[None, :]
    okj = (jj >= 0) & (jj < NY)
    jj = np.clip(jj, 0, NY - 1)
    vv = V[rows[:, None], jj] & okj & vis[:, None]
    rho = float((cand[rows[:, None], jj] & vv).sum()) / max(float(vv.sum()), 1.0)
    # surface on each side of the line: fraction of road-coloured cells (paint lies on a road)
    rd = road0[rows[:, None], jj] & vv
    nside = vv[:, :16].sum(), vv[:, 16:].sum()
    flank = (float(rd[:, :16].sum()) / max(float(nside[0]), 1.0), float(rd[:, 16:].sum()) / max(float(nside[1]), 1.0))
    chance = 1.0 - (1.0 - rho) ** 3
    hv = hit[vis].astype(np.int8)
    d = np.diff(np.concatenate([[0], hv, [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    runs = (ends - starts) * DX
    gaps = (starts[1:] - ends[:-1]) * DX if len(starts) > 1 else np.zeros(0)
    x = XS[rows]
    return {
        "theta": theta, "y0": y0, "half": half,
        "nvis": nvis, "nhit": nh, "fill": nh / max(nvis, 1), "chance": chance, "flank": flank,
        "len_vis": nvis * DX, "len_hit": nh * DX,
        "runs": runs, "gaps": gaps,
        "tL": float(np.median(tl[k, best][hit])),
        "tb": float(np.median(tb[k, best][hit])),
        "lab": np.median(lab[rows[hit], jb[hit]], axis=0),
        "near_hits": int((hit & (np.abs(x) < 10.0)).sum()),
        "rows": rows, "j": j,
    }


def _double_profile(ln, Lc, bc, V, yellow):
    """Across-line profile within 3.5-10 m: two maxima 0.15-0.45 m apart with a dip between."""
    offs = np.arange(-10, 11)          # +-0.5 m
    acc = np.zeros(len(offs), np.float32)
    cnt = np.zeros(len(offs), np.float32)
    ch = bc if yellow else Lc
    for seg in ln["segs"]:
        sel = np.abs(XS[seg["rows"]]) < 10.0
        rows, j = seg["rows"][sel], seg["j"][sel]
        if len(rows) == 0:
            continue
        jj = j[:, None] + offs[None, :]
        okj = (jj >= 0) & (jj < NY)
        jj = np.clip(jj, 0, NY - 1)
        vals = ch[rows[:, None], jj]
        vv = V[rows[:, None], jj] & okj
        acc += (vals * vv).sum(0)
        cnt += vv.sum(0)
    if cnt.min() < 8:
        return 0.0, None
    p = acc / np.maximum(cnt, 1)
    best, sep = 0.0, None
    base = min(p[0:3].mean(), p[-3:].mean())
    for a in range(2, len(p) - 2):
        if not (p[a] >= p[a - 1] and p[a] >= p[a + 1]):
            continue
        for b in range(a + 3, min(a + 10, len(p) - 1)):
            if not (p[b] >= p[b - 1] and p[b] >= p[b + 1]):
                continue
            hgt = min(p[a], p[b]) - base
            if hgt <= 2.0:
                continue
            depth = (min(p[a], p[b]) - p[a + 1:b].min()) / hgt
            if depth > best:
                best, sep = depth, (b - a) * DY
    return float(best), sep


# ---------------------------------------------------------------- road extent
def _lat_profile(M, V, theta, half, xlo=R_MIN, xhi=12.0):
    """Mean of M over valid cells per lateral bin in the frame sheared by theta (rows xlo..xhi)."""
    rows = np.nonzero(((XS > xlo) & (XS < xhi)) if half > 0 else ((XS < -xlo) & (XS > -xhi)))[0]
    shift = np.rint(XS[rows] * math.tan(theta) / DY).astype(np.int64)
    jj = np.arange(NY)[None, :] + shift[:, None]
    ok = (jj >= 0) & (jj < NY)
    jj = np.clip(jj, 0, NY - 1)
    vv = V[rows[:, None], jj] & ok
    num = (M[rows[:, None], jj] * vv).sum(0)
    den = vv.sum(0).astype(np.float32)
    return num, den


def _walk_edge(P, ok, start, step, gap_cells=16):
    """Walk from index start in direction step until P < 0.5 for gap_cells; returns edge index,
    or None when the profile leaves the visible area / grid first."""
    j = start
    last_road = start
    run = 0
    while 0 <= j < NY:
        if not ok[j]:
            return None
        if P[j] >= 0.5:
            last_road = j
            run = 0
        else:
            run += 1
            if run >= gap_cells:
                return last_road
        j += step
    return None


def _fill_nan_1d(v):
    ok = np.isfinite(v)
    if not ok.any():
        return v
    idx = np.arange(len(v))
    return np.interp(idx, idx[ok], v[ok]).astype(np.float32)


# ---------------------------------------------------------------- main
def extract(sph, debug=False):
    out = {k: np.nan for k in FEATURE_NAMES}
    ev = {}
    car_known = sph.car_heading is not None
    car = float(sph.car_heading) if car_known else 0.0
    lin, V = build_ipm(sph, car)
    if not car_known and V.sum() > 2000:
        ax, coh = _structure_axis(_lab_from_linear(lin)[..., 0], V)
        if ax is not None:
            car = car - ax        # rotate the grid so that +x runs along the road axis
            lin, V = build_ipm(sph, car)
            ev["axis_estimated"] = {"rel_longitude": round((car + 180.0) % 360.0 - 180.0, 1),
                                    "coherence": round(coh, 2)}
    out["ipm_valid"] = float(V.mean())
    if V.sum() < 1500:
        ev["road"] = "ground not visible"
        return {"x": np.array([out[k] for k in FEATURE_NAMES], np.float32), "evidence": ev}
    lab = _lab_from_linear(lin)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    Vf = V.astype(np.float32)

    # ---- lateral top-hat (paint must be brighter / yellower than both sides)
    # each side has a near window (0.20-0.40 m) and a far window (0.45-0.80 m); the darker of the
    # two is the side's background, so the second line of a double line does not mask the first
    S = _lat_cumsum(np.stack([L * Vf, B * Vf, Vf]))
    c3 = _win(S, -1, 1)
    nc = c3[2]
    Lc, bc = c3[0] / np.maximum(nc, 1), c3[1] / np.maximum(nc, 1)
    sides = []
    for lo, hi in ((4, 8), (9, 16)):
        for sg in (1, -1):
            w = _win(S, lo, hi) if sg > 0 else _win(S, -hi, -lo)
            n = np.maximum(w[2], 1)
            sides.append((w[0] / n, w[1] / n, w[2] >= 3))
    (Ll1, bl1, ol1), (Lr1, br1, or1), (Ll2, bl2, ol2), (Lr2, br2, or2) = sides
    Ll = np.where(ol2, np.minimum(Ll1, Ll2), Ll1)
    Lr = np.where(or2, np.minimum(Lr1, Lr2), Lr1)
    bl = np.where(ol2, np.minimum(bl1, bl2), bl1)
    br = np.where(or2, np.minimum(br1, br2), br1)
    okT = V & ol1 & or1 & (nc >= 2)
    T_L = np.where(okT, Lc - np.maximum(Ll, Lr), 0).astype(np.float32)
    T_b = np.where(okT, bc - np.maximum(bl, br), 0).astype(np.float32)
    tlv, tbv = T_L[okT], T_b[okT]
    sL = float(np.median(np.abs(tlv))) * 1.48 + 0.5 if len(tlv) else 3.0
    sb = float(np.median(np.abs(tbv))) * 1.48 + 0.3 if len(tbv) else 2.0
    thL = max(7.0, 4.0 * sL)
    thb = max(5.0, 4.0 * sb)
    # paint is never green: white paint is near-neutral, yellow paint has a* >= -4
    not_green = ~((A < -5.0) & (B > 4.0))
    cand = okT & not_green & ((T_L > thL) | ((T_b > thb) & (T_L > -2.0) & (A > -4.0)))

    # ---- road reference colour (unpainted cells near the car)
    X, Y, R, _, _ = _grid()
    lane0 = slice(J0 - 16, J0 + 16)              # |y| < 0.8 m
    refm = V[:, lane0] & ~cand[:, lane0]
    ref = np.zeros_like(V)
    ref[:, lane0] = refm
    ref &= np.abs(X) <= 9.0
    if ref.sum() < 80:
        ref = V & (R <= 9.0) & ~cand
    if ref.sum() < 80:
        ref = V & ~cand
    Lr0, ar0, br0 = (float(np.median(c[ref])) for c in (L, A, B))
    chroma = math.hypot(ar0, br0)
    hue = math.degrees(math.atan2(br0, ar0))
    hp = np.abs(L - _box2d(L, Vf, 2))
    tex_sel = ref & (R < 8.0)
    texture = float(np.mean(hp[tex_sel])) if tex_sel.sum() > 50 else float(np.mean(hp[ref]))
    out.update(road_L=Lr0, road_a=ar0, road_b=br0, road_chroma=chroma, road_texture=texture)

    # surface memberships
    grey = math.exp(-(max(chroma - 6.0, 0.0) / 5.0) ** 2)
    m_asph = grey * _sig((60.0 - Lr0) / 4.0)
    m_conc = grey * _sig((Lr0 - 60.0) / 4.0)
    hd = lambda h0: (hue - h0 + 180.0) % 360.0 - 180.0
    m_dirt = _sig((chroma - 13.0) / 3.0) * math.exp(-(hd(52.0) / 20.0) ** 2) * _sig((ar0 - 5.0) / 2.0)
    m_sand = _sig((chroma - 6.0) / 2.5) * math.exp(-(hd(78.0) / 16.0) ** 2) * _sig((Lr0 - 48.0) / 5.0) \
        * (1.0 - _sig((chroma - 24.0) / 3.0))
    tot = m_asph + m_conc + m_dirt + m_sand + 1e-6
    out.update(surf_asphalt=m_asph / tot, surf_concrete=m_conc / tot, surf_dirt=m_dirt / tot, surf_sand=m_sand / tot)

    # per-row reference (handles shadows / illumination changes along the road)
    def row_ref(C):
        Cm = np.where(refm, C[:, lane0], np.nan)
        cnt = refm.sum(1)
        with np.errstate(all="ignore"):
            v = np.nanmedian(np.where(cnt[:, None] >= 6, Cm, np.nan), axis=1)
        v = _fill_nan_1d(v)
        k = np.ones(11, np.float32) / 11
        return np.convolve(np.pad(v, 5, mode="edge"), k, mode="valid").astype(np.float32)
    if refm.sum(1).max() >= 6:
        rL, ra, rb = row_ref(L), row_ref(A), row_ref(B)
    else:
        rL, ra, rb = (np.full(NX, v, np.float32) for v in (Lr0, ar0, br0))
    dist = np.sqrt(((L - rL[:, None]) / 2.0) ** 2 + (A - ra[:, None]) ** 2 + (B - rb[:, None]) ** 2)
    spread = float(np.median(dist[ref])) if ref.any() else 3.0
    tau = max(6.0, min(12.0, 3.0 * spread))
    road0 = dist < tau
    paint = (T_L > 0.5 * thL) | (T_b > 0.5 * thb)
    paint[:, 2:-2] = paint[:, 2:-2] | paint[:, :-4] | paint[:, 4:] | paint[:, 1:-3] | paint[:, 3:-1]
    roadlike = ((dist < tau) | paint).astype(np.float32)

    # ---- lane lines per half plane
    halves = []
    for half in (1, -1):
        rows_h = (XS > 0) if half > 0 else (XS < 0)
        if V[rows_h].sum() < 600:
            continue
        segs = []
        for th, y0, v in _hough_lines(cand, half):
            if abs(y0) > 9.0:
                continue
            s = _walk_line(th, y0, half, cand, V, T_L, T_b, lab, road0)
            if s is None:
                continue
            if s["len_hit"] < 1.5 or s["fill"] - s["chance"] < 0.1 or s["near_hits"] < 4 or max(s["flank"]) < 0.6:
                continue
            segs.append(s)
        halves.append((half, segs))

    axis_theta = {}
    for half, segs in halves:
        if segs:
            w = np.array([s["len_hit"] for s in segs])
            th = np.array([s["theta"] for s in segs])
            axis_theta[half] = float(np.sum(w * th) / np.sum(w))
        else:
            axis_theta[half] = 0.0

    lines = []
    for half, segs in halves:
        for s in segs:
            match = None
            for ln in lines:
                if half not in ln["halves"] and abs(ln["y0"] - s["y0"]) < 0.35:
                    match = ln
                    break
            if match is None:
                lines.append({"y0": s["y0"], "halves": [half], "segs": [s]})
            else:
                n0 = sum(q["len_hit"] for q in match["segs"])
                match["y0"] = (match["y0"] * n0 + s["y0"] * s["len_hit"]) / (n0 + s["len_hit"])
                match["halves"].append(half)
                match["segs"].append(s)

    # ---- per-line statistics: colour, confidence, dash pattern
    for ln in lines:
        segs = ln["segs"]
        w = np.array([s["len_hit"] for s in segs])
        ln["len_hit"] = float(w.sum())
        ln["len_vis"] = float(sum(s["len_vis"] for s in segs))
        ln["fill"] = ln["len_hit"] / max(ln["len_vis"], 1e-6)
        ln["tL"] = float(np.sum(w * np.array([s["tL"] for s in segs])) / w.sum())
        ln["tb"] = float(np.sum(w * np.array([s["tb"] for s in segs])) / w.sum())
        ln["lab"] = np.sum(w[:, None] * np.array([s["lab"] for s in segs]), 0) / w.sum()
        ln["gaps"] = np.concatenate([s["gaps"] for s in segs])
        tb, tL = ln["tb"], ln["tL"]
        yel = _sig((tb - 4.5) / 1.5) * _sig((tb / max(abs(tL), 2.0) - 0.3) / 0.08)
        la, lb = float(ln["lab"][1]), float(ln["lab"][2])
        yel *= _sig((lb - 1.2 * abs(la) - 6.0) / 3.0)
        ln["yellow"] = yel
        ln["conf"] = _sig((ln["len_hit"] - 2.5) / 1.0) * _sig((max(tL, 2.5 * tb) - 9.0) / 3.0)
        long_gaps = int(np.sum(ln["gaps"] > 1.2))
        ln["dashed"] = _sig((0.62 - ln["fill"]) / 0.08) * (1.0 if long_gaps >= 1 else 0.4)

    # ---- group parallel lines < 0.55 m apart with the same colour into markings (double lines)
    lines.sort(key=lambda q: q["y0"])
    marks = []
    for ln in lines:
        if ln["conf"] < 0.2:
            continue
        m = marks[-1] if marks else None
        if m is not None and ln["y0"] - m["ys"][-1] < 0.55 and abs(ln["yellow"] - m["members"][-1]["yellow"]) < 0.5:
            m["members"].append(ln)
            m["ys"].append(ln["y0"])
        else:
            marks.append({"members": [ln], "ys": [ln["y0"]]})
    for m in marks:
        mem = m["members"]
        w = np.array([q["conf"] * q["len_hit"] for q in mem]) + 1e-6
        m["y0"] = float(np.sum(w * np.array(m["ys"])) / w.sum())
        m["conf"] = max(q["conf"] for q in mem)
        m["yellow"] = float(np.sum(w * np.array([q["yellow"] for q in mem])) / w.sum())
        best = mem[int(np.argmax(w))]
        m["dashed"], m["fill"] = best["dashed"], best["fill"]
        m["halves"] = sorted(set(h for q in mem for h in q["halves"]))
        m["segs"] = [sg for q in mem for sg in q["segs"]]
        m["sep"] = max(m["ys"]) - min(m["ys"])
        for q in mem:
            q["mark"] = m
    good = [m for m in marks if m["conf"] > 0.3]

    # ---- carriageway extent per half: road-likeness profile walked outward, extended to the
    #      outermost painted marking
    profiles = {}
    ext = {}
    for half, segs in halves:
        num, den = _lat_profile(roadlike, V, axis_theta[half], half)
        okp = den >= 8
        P = np.where(okp, num / np.maximum(den, 1), 0.0)
        P = np.convolve(P, np.ones(3) / 3.0, mode="same")
        profiles[half] = (P, okp)
        jl = _walk_edge(P, okp, J0, +1)
        jr = _walk_edge(P, okp, J0 - 1, -1)
        el = YS[jl] + DY / 2 if jl is not None else np.nan
        er = YS[jr] - DY / 2 if jr is not None else np.nan
        mL = [m["y0"] for m in good if half in m["halves"] and m["y0"] > 0.3]
        mR = [m["y0"] for m in good if half in m["halves"] and m["y0"] < -0.3]
        if mL:
            el = max(el, max(mL) + 0.1) if np.isfinite(el) else max(mL) + 0.1
        if mR:
            er = min(er, min(mR) - 0.1) if np.isfinite(er) else min(mR) - 0.1
        ext[half] = (float(el), float(er))
    els = [e[0] for e in ext.values() if np.isfinite(e[0])]
    ers = [e[1] for e in ext.values() if np.isfinite(e[1])]
    yl = float(np.median(els)) if els else np.nan
    yr = float(np.median(ers)) if ers else np.nan
    if np.isfinite(yl) and np.isfinite(yr):
        out["road_width_m"] = yl - yr
        out["road_center_off_m"] = 0.5 * (yl + yr)
    else:
        out["road_width_m"] = 2 * YR       # an edge out of range: wide surface

    # ---- roles: outermost marking on a side with < 2.6 m of road beyond it is an edge line,
    #      the others separate lanes; the centre line is the separator nearest the carriageway middle
    for m in good:
        lim = yl if m["y0"] > 0 else yr
        m["beyond"] = abs(lim - m["y0"]) if np.isfinite(lim) else np.nan
        m["role"] = "lane"
    for side in (1, -1):
        sm = [m for m in good if m["y0"] * side > 0.3]
        if sm:
            o = max(sm, key=lambda q: q["y0"] * side)
            if not (o["beyond"] >= 2.6):
                o["role"] = "edge"
    centre = None
    seps = [m for m in good if m["role"] == "lane" and abs(m["y0"]) >= 0.5]
    if seps:
        mid = out["road_center_off_m"] if np.isfinite(out["road_center_off_m"]) else 0.0
        centre = min(seps, key=lambda q: abs(q["y0"] - mid) - 0.7 * q["yellow"] - 0.3 * q["conf"])
        centre["role"] = "center"
    edge_marks = [m for m in good if m["role"] == "edge"]

    allm = max([ln["conf"] for ln in lines], default=0.0)
    out["n_lines"] = float(sum(m["conf"] for m in good))
    out["markings"] = allm
    out["no_markings"] = 1.0 - allm
    out["yellow_any"] = max([ln["conf"] * ln["yellow"] for ln in lines], default=0.0)
    if centre is not None:
        c = centre["conf"]
        out["yellow_center"] = c * centre["yellow"]
        out["white_center"] = c * (1 - centre["yellow"])
        out["dashed_center"] = centre["dashed"]
        out["center_off_m"] = centre["y0"]
        dbl, sep = _double_profile(centre, Lc, bc, V, centre["yellow"] > 0.5)
        d1 = _sig((dbl - 0.35) / 0.08)
        d2 = 1.0 if (len(centre["members"]) >= 2 and 0.17 <= centre["sep"] <= 0.55) else 0.0
        centre["double"] = max(d1, d2)
        out["double_center"] = c * centre["double"]
        others = [q for q in good if q is not centre and (q["y0"] - centre["y0"]) * centre["y0"] < 0]
        if others:
            nb = min(others, key=lambda q: abs(q["y0"] - centre["y0"]))
            out["lane_width_m"] = abs(nb["y0"] - centre["y0"])
        elif np.isfinite(yl) and np.isfinite(yr):
            out["lane_width_m"] = abs((yr if centre["y0"] > 0 else yl) - centre["y0"])
    else:
        out["yellow_center"] = 0.0
        out["white_center"] = 0.0
        out["double_center"] = 0.0
    if edge_marks:
        e = max(edge_marks, key=lambda q: q["conf"])
        out["yellow_edge"] = max(q["conf"] * q["yellow"] for q in edge_marks)
        out["white_edge"] = max(q["conf"] * (1 - q["yellow"]) for q in edge_marks)
        out["dashed_edge"] = e["dashed"]
    else:
        out["yellow_edge"] = 0.0
        out["white_edge"] = 0.0
    markings = allm

    tex_p = _sig((6.0 - texture) / 1.5)
    out["paved"] = max(markings, (m_asph + m_conc) / tot * tex_p)

    # ---- driving side: the car keeps to its side, so more carriageway lies on the side of the
    #      oncoming lane (left in right-hand traffic)
    dsd = {}
    if car_known:
        asy = []
        for half, (el, er) in ext.items():
            if np.isfinite(el) and np.isfinite(er) and el - er >= 3.0:
                asy.append(0.5 * (el + er) / (el - er))
        if asy:
            agree = len(asy) == 1 or (asy[0] * asy[1] > 0)
            dsd["asym"] = float(np.mean(asy))
            dsd["asym_w"] = (1.0 if len(asy) == 2 else 0.6) if agree else 0.25
        if centre is not None and 0.8 < abs(centre["y0"]) < 4.5:
            dsd["centre"] = math.copysign(1.0, centre["y0"]) * centre["conf"]
        if dsd:
            z = 10.0 * dsd.get("asym", 0.0) * dsd.get("asym_w", 0.0) + 1.5 * dsd.get("centre", 0.0)
            out["right_hand_traffic"] = _sig(z)

    # ---- curb / sidewalk / verge
    verge_a, verge_b, green, curb, side = [], [], [], [], []
    sel_v = V & (np.abs(X) > R_MIN) & (np.abs(X) < 12.0)
    for ye, sgn in ((yl, 1), (yr, -1)):
        if not np.isfinite(ye):
            continue
        dy = (Y - ye) * sgn
        band = sel_v & (dy > 0.5) & (dy < 3.0)
        if band.sum() > 50:
            verge_a.append(float(np.median(A[band])))
            verge_b.append(float(np.median(B[band])))
            green.append(float(np.mean((A[band] < -6) & (B[band] > 8))))
        sw = sel_v & (dy > 0.2) & (dy < 2.2)
        if sw.sum() > 50:
            C = np.hypot(A[sw], B[sw])
            side.append(float(np.mean((C < 9) & (L[sw] > Lr0 + 6))))
        inner = sel_v & (dy > -0.6) & (dy < -0.15)
        outer_b = sel_v & (dy > 0.15) & (dy < 0.6)
        if inner.sum() > 30 and outer_b.sum() > 30:
            step = abs(float(np.median(L[outer_b]) - np.median(L[inner])))
            curb.append(_sig((step - 8.0) / 3.0))
    if verge_a:
        out["verge_a"] = float(np.mean(verge_a))
        out["verge_b"] = float(np.mean(verge_b))
        out["verge_green"] = float(np.mean(green))
    if side:
        out["sidewalk"] = float(max(side))
    if curb:
        out["curb"] = float(max(curb))

    # ---- evidence
    surf = {"asphalt": out["surf_asphalt"], "concrete": out["surf_concrete"],
            "dirt": out["surf_dirt"], "sand_gravel": out["surf_sand"]}
    st = max(surf, key=surf.get)
    ev["surface"] = {"type": st, "confidence": round(float(surf[st]), 2), "paved": round(float(out["paved"]), 2),
                     "lab": [round(Lr0, 1), round(ar0, 1), round(br0, 1)], "texture": round(texture, 2)}
    if out["road_width_m"] < 2 * YR:
        ev["road_width_m"] = round(float(out["road_width_m"]), 1)
    rl = {"confidence": round(float(markings), 2)}
    if centre is not None:
        rl["center"] = "yellow" if centre["yellow"] > 0.5 else "white"
        rl["center_style"] = ("double " if centre.get("double", 0) > 0.5 else "") + \
            ("dashed" if centre["dashed"] > 0.5 else "solid")
        rl["center_offset_m"] = round(centre["y0"], 2)
    if edge_marks:
        e = max(edge_marks, key=lambda q: q["conf"])
        rl["edge"] = "yellow" if e["yellow"] > 0.5 else "white"
        rl["edge_style"] = "dashed" if e["dashed"] > 0.5 else "solid"
    if markings < 0.3:
        rl["none"] = True
    ev["road_lines"] = rl
    ev["lines"] = [{"offset_m": round(m["y0"], 2), "color": "yellow" if m["yellow"] > 0.5 else "white",
                    "style": ("double " if m.get("double", 0) > 0.5 else "") + ("dashed" if m["dashed"] > 0.5 else "solid"),
                    "role": m["role"], "fill": round(m["fill"], 2), "confidence": round(m["conf"], 2)}
                   for m in sorted(good, key=lambda q: q["y0"], reverse=True)]
    if np.isfinite(out["right_hand_traffic"]):
        p = float(out["right_hand_traffic"])
        ev["driving_side"] = {"side": "right" if p >= 0.5 else "left", "confidence": round(abs(p - 0.5) * 2, 2)}
    if np.isfinite(out["curb"]):
        ev["curb"] = round(float(out["curb"]), 2)
    if np.isfinite(out["sidewalk"]):
        ev["sidewalk"] = round(float(out["sidewalk"]), 2)
    res = {"x": np.array([out[k] for k in FEATURE_NAMES], np.float32), "evidence": ev}
    if debug:
        res["_debug"] = {"lines": lines, "marks": marks, "ext": ext, "dsd": dsd, "edges": (yl, yr), "rgb": _srgb_from_linear(lin), "V": V,
                         "cand": cand, "car": car, "roadlike": roadlike, "T_L": T_L, "T_b": T_b}
    return res
