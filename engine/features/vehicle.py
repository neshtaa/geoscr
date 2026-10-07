"""
vehicle: Google camera vehicle, camera generation and image quality.

(draft v2 - full docstring written once the feature set is final)
"""
import math

import numpy as np

NAME = "vehicle"

# ----------------------------------------------------------------------------- colour
_M_RGB2XYZ = (np.array([[0.4124, 0.3576, 0.1805],
                        [0.2126, 0.7152, 0.0722],
                        [0.0193, 0.1192, 0.9505]]) / np.array([[0.95047], [1.0], [1.08883]])).astype(np.float32)
_LIN_LUT = np.where(np.arange(256) / 255.0 > 0.04045, ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4,
                    np.arange(256) / 255.0 / 12.92).astype(np.float32)
_Y_W = np.array([0.299, 0.587, 0.114], np.float32)


_XYZ_ROWS = [tuple(float(v) for v in row) for row in _M_RGB2XYZ]


def _lab(rgb):
    """sRGB (..., 3) (uint8 or float 0..255) -> CIE L*a*b* (D65), float32."""
    if rgb.dtype == np.uint8:
        lin = _LIN_LUT[rgb]
    else:
        c = np.clip(rgb, 0, 255).astype(np.float32) / 255.0
        lin = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92).astype(np.float32)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    f = []
    for m0, m1, m2 in _XYZ_ROWS:
        t = m0 * r + m1 * g + m2 * b
        f.append(np.where(t > 0.008856, np.cbrt(np.maximum(t, 0.008856)), 7.787 * t + 16.0 / 116.0))
    return np.stack([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])], -1).astype(np.float32)


# --------------------------------------------------------------------- spherical sampling
_GRID_CACHE = {}


def _gnomonic_grid(n, rmax_deg):
    """n x n gnomonic (rectilinear) view of half-angle rmax centred on a pole.

    Returns r (angular distance from the pole, deg), theta (azimuth of the pixel
    clockwise from image 'up', deg) and plane coordinates u (right), v (up) in
    tan units.  For the nadir seen from above, up = reference direction (the car's
    front), right = theta 90 (the car's right-hand side).
    """
    key = (n, rmax_deg)
    if key not in _GRID_CACHE:
        t = math.tan(math.radians(rmax_deg))
        c = ((np.arange(n) + 0.5) / n * 2.0 - 1.0) * t
        u = np.repeat(c[None, :], n, 0)
        v = np.repeat(-c[:, None], n, 1)
        rho = np.hypot(u, v)
        r = np.degrees(np.arctan(rho))
        th = np.degrees(np.arctan2(u, v)) % 360.0
        _GRID_CACHE[key] = tuple(a.astype(np.float32) for a in (r, th, u, v))
    return _GRID_CACHE[key]


_IDX_CACHE = {}


def _sample_sphere(s, el, phi, key=None):
    """Bilinear sample of SphericalImage s at elevation / relative-longitude arrays.
    Returns rgb float32 (..., 3) and the validity mask (nearest neighbour).  ``key``
    caches the interpolation indices (geometry depends only on the canvas size)."""
    ck = None if key is None else key + (s.w, s.h)
    if ck is not None and ck in _IDX_CACHE:
        i00, i01, i10, i11, w00, w01, w10, w11, inn = _IDX_CACHE[ck]
    else:
        xs = (((phi + 180.0) % 360.0) / 360.0 * s.w - 0.5).astype(np.float64)
        ys = np.clip((90.0 - el) / 180.0 * s.h - 0.5, 0, s.h - 1).astype(np.float64)
        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        fx = (xs - x0).astype(np.float32)
        fy = (ys - y0).astype(np.float32)
        x0m, x1m = x0 % s.w, (x0 + 1) % s.w
        y1 = np.minimum(y0 + 1, s.h - 1)
        i00, i01, i10, i11 = y0 * s.w + x0m, y0 * s.w + x1m, y1 * s.w + x0m, y1 * s.w + x1m
        w00, w01, w10, w11 = (1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy
        inn = np.clip(np.round(ys).astype(np.int64), 0, s.h - 1) * s.w + np.round(xs).astype(np.int64) % s.w
        if ck is not None:
            _IDX_CACHE[ck] = (i00, i01, i10, i11, w00, w01, w10, w11, inn)
    flat = s.rgb.reshape(-1, 3)
    out = (np.take(flat, i00, 0) * w00[..., None] + np.take(flat, i01, 0) * w01[..., None]
           + np.take(flat, i10, 0) * w10[..., None] + np.take(flat, i11, 0) * w11[..., None])
    m = np.take(s.mask.reshape(-1), inn)
    return out.astype(np.float32), m


def _pole_view(s, n, rmax, pole, ref_phi):
    r, th, u, v = _gnomonic_grid(n, rmax)
    el = pole * (90.0 - r)
    phi = ref_phi + (th if pole < 0 else -th)
    rgb, m = _sample_sphere(s, el, phi, key=("pole", n, rmax, pole, round(float(ref_phi), 3)))
    return rgb, m & (r <= rmax), r, th, u, v


# ------------------------------------------------------------------- small filters
def _grad(L):
    gx = np.zeros_like(L)
    gy = np.zeros_like(L)
    gx[:, 1:-1] = 0.5 * (L[:, 2:] - L[:, :-2])
    gy[1:-1, :] = 0.5 * (L[2:, :] - L[:-2, :])
    return gx, gy


def _erode(m, k=1):
    out = m.copy()
    for _ in range(k):
        e = out.copy()
        e[1:, :] &= out[:-1, :]
        e[:-1, :] &= out[1:, :]
        e[:, 1:] &= out[:, :-1]
        e[:, :-1] &= out[:, 1:]
        out = e
    return out


def _box(a, kr, kc=None, wrap_cols=False):
    """Box mean filter (kr rows x kc cols, odd sizes), edge-replicated rows,
    edge-replicated or wrapped columns; sums of shifted slices (small kernels)."""
    kc = kr if kc is None else kc
    pr, pc = kr // 2, kc // 2
    a = a.astype(np.float32)
    b = np.concatenate([a[:1]] * pr + [a] + [a[-1:]] * pr, 0) if pr else a
    acc = b[0:b.shape[0] - 2 * pr].copy()
    for k in range(1, kr):
        acc += b[k:b.shape[0] - 2 * pr + k]
    if pc:
        if wrap_cols:
            b = np.concatenate([acc[:, -pc:], acc, acc[:, :pc]], 1)
        else:
            b = np.concatenate([acc[:, :1]] * pc + [acc] + [acc[:, -1:]] * pc, 1)
    else:
        b = acc
    out = b[:, 0:b.shape[1] - 2 * pc].copy()
    for k in range(1, kc):
        out += b[:, k:b.shape[1] - 2 * pc + k]
    return out / float(kr * kc)


def _f(x):
    return float(x) if x is not None and np.isfinite(x) else float("nan")


# ------------------------------------------------------------------------- constants
NADIR_N = 200
NADIR_RMAX = 45.0       # gnomonic nadir view covers el <= -45
CORE_R = 20.0           # core   : el <= -70
RING_R = 35.0           # ring   : -70 < el <= -55
REF_R = (41.0, 45.0)    # reference ground texture: -49 .. -45 deg
SECT_R = (8.0, 40.0)    # car-relative sectors (front/back/left/right)
MARK_THETA = np.array([75.0, 120.0, 240.0, 285.0])   # gen-3 roof-mount feet (deg from car front)
MARK_R = np.arange(32.0, 44.01, 0.5)

COLOURS = ["white", "black", "grey", "red", "brown", "green", "blue"]
SECTORS = ["front", "right", "back", "left"]

FEATURE_NAMES = (
    ["nad_L", "nad_a", "nad_b", "nad_Lsd", "nad_asd", "nad_bsd",
     "ring_L", "ring_a", "ring_b", "nad_ring_dE"]
    + ["nad_frac_" + c for c in COLOURS]
    + ["ring_frac_black",
       "nad_tex", "ring_tex", "nad_tex_ratio", "nad_smooth_frac", "nad_edge_frac",
       "blur_el", "blur_el_sd", "blur_elong", "blur_el_front", "blur_el_back", "blur_el_side",
       "nad_tangential", "nad_line_peak", "nad_across_frac", "nad_sector_range",
       "mark_score", "mark_score_ri"]
    + ["sec_%s_%s" % (s, q) for s in SECTORS for q in ("L", "C", "tex")]
    + ["zen_L", "zen_a", "zen_b", "zen_Lsd", "zen_tex", "zen_black", "zen_sky_dL"]
    + ["sky_blue_frac", "sky_blue_C", "sky_blue_hue", "sky_white_a", "sky_white_b", "sky_clip"]
    + ["q_lap_grad", "q_hf_ratio", "q_grad_norm", "q_chroma_noise",
       "q_clip_hi", "q_clip_lo", "q_cast_a", "q_cast_b", "q_white_a", "q_white_b",
       "q_chroma", "q_chroma_p90", "q_L_p1", "q_L_p99", "q_L_med",
       "pipe_block_h", "pipe_block_v"]
)


# ------------------------------------------------------------------------- nadir
def _sector_of(th, ref_known):
    """0 front, 1 right, 2 back, 3 left (90-deg sectors centred on the car axes)."""
    return (((th + 45.0) % 360.0) // 90.0).astype(np.int64)


def _nadir(sph, out, ctx):
    ref_known = sph.car_heading is not None
    ref = sph.car_heading if ref_known else 0.0
    rgb, m, r, th, u, v = _pole_view(sph, NADIR_N, NADIR_RMAX, -1, ref)
    core_all = r <= CORE_R
    core = m & core_all
    if core.sum() < 0.5 * core_all.sum():
        return
    ring = m & (r > CORE_R) & (r <= RING_R)
    lab = _lab(rgb)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(A, B)
    lc = lab[core]
    mc = lc.mean(0)
    out["nad_L"], out["nad_a"], out["nad_b"] = [float(x) for x in mc]
    out["nad_Lsd"], out["nad_asd"], out["nad_bsd"] = [float(x) for x in lc.std(0)]
    if ring.sum() > 200:
        mr = lab[ring].mean(0)
        out["ring_L"], out["ring_a"], out["ring_b"] = [float(x) for x in mr]
        out["nad_ring_dE"] = float(np.linalg.norm(mc - mr))
        out["ring_frac_black"] = float(((L[ring] < 22) & (C[ring] < 15)).mean())
    # colour classes of the core (vehicle body / blur smudge / ground)
    hue = np.degrees(np.arctan2(B, A)) % 360.0
    Lc, Cc, Hc = L[core], C[core], hue[core]
    chrom = Cc >= 12.0
    fr = {
        "white": (Lc >= 72) & ~chrom,
        "black": (Lc < 25) & ~chrom,
        "grey": (Lc >= 25) & (Lc < 72) & ~chrom,
        "red": chrom & ((Hc < 45) | (Hc >= 330)),
        "brown": chrom & (Hc >= 45) & (Hc < 100),
        "green": chrom & (Hc >= 100) & (Hc < 200),
        "blue": chrom & (Hc >= 200) & (Hc < 330),
    }
    for k in COLOURS:
        out["nad_frac_" + k] = float(fr[k].mean())
    ctx["nad_colour_fracs"] = {k: float(fr[k].mean()) for k in COLOURS}
    # texture: gradient magnitude per degree of arc
    gx, gy = _grad(L)
    pix_deg = math.degrees(2.0 * math.tan(math.radians(NADIR_RMAX)) / NADIR_N)
    G = np.hypot(gx, gy) / (pix_deg * np.cos(np.radians(r)) ** 1.5)
    me = _erode(m, 1)
    cm, rm = core & me, ring & me
    refm = me & (r >= REF_R[0]) & (r <= REF_R[1])
    if cm.sum() < 200:
        return
    out["nad_tex"] = float(G[cm].mean())
    if rm.sum() > 200:
        out["ring_tex"] = float(G[rm].mean())
    gref = float(np.median(G[refm])) if refm.sum() > 200 else float("nan")
    if np.isfinite(gref):
        out["nad_tex_ratio"] = float(np.median(G[cm]) / (gref + 1.0))
        out["nad_smooth_frac"] = float((G[cm] < 0.25 * gref + 0.5).mean())
    out["nad_edge_frac"] = float((G[cm] > 30.0).mean())
    ctx["gref"] = gref
    # radial blur-boundary profile, 16 azimuth sectors x 1.5 deg radial bins
    nsec, dr = 16, 1.5
    nb = int(NADIR_RMAX / dr)
    sec = ((th % 360.0) / (360.0 / nsec)).astype(np.int64) % nsec
    rbin = np.clip((r / dr).astype(np.int64), 0, nb - 1)
    idx = (sec * nb + rbin)[me]
    sums = np.bincount(idx, weights=G[me], minlength=nsec * nb).reshape(nsec, nb)
    cnts = np.bincount(idx, minlength=nsec * nb).reshape(nsec, nb)
    prof = np.where(cnts > 3, sums / np.maximum(cnts, 1), np.nan)
    if np.isfinite(gref) and gref > 1.0:
        bnd = np.full(nsec, np.nan)
        rc = (np.arange(nb) + 0.5) * dr
        for k in range(nsec):
            p = prof[k] / gref
            ok = np.isfinite(p)
            if ok.sum() < nb // 2:
                continue
            # boundary = first radius from which texture stays >= 45 % of the ground reference
            low = ok & (p < 0.45)
            last = np.nonzero(low & (rc < REF_R[0]))[0]
            bnd[k] = rc[last[-1]] + dr / 2 if len(last) else 0.0
        okb = np.isfinite(bnd)
        if okb.sum() >= nsec // 2:
            bl = -90.0 + bnd
            out["blur_el"] = float(np.nanmedian(bl))
            out["blur_el_sd"] = float(np.nanstd(bl))
            ang = np.radians((np.arange(nsec) + 0.5) * 360.0 / nsec)
            c2 = np.mean(np.cos(2 * ang[okb]) * bnd[okb])
            s2 = np.mean(np.sin(2 * ang[okb]) * bnd[okb])
            out["blur_elong"] = float(2 * np.hypot(c2, s2) / (np.mean(bnd[okb]) + 3.0))
            if ref_known:
                out["blur_el_front"] = _f(np.nanmean(bl[[15, 0]]))
                out["blur_el_back"] = _f(np.nanmean(bl[[7, 8]]))
                out["blur_el_side"] = _f(np.nanmean(bl[[3, 4, 11, 12]]))
            ctx["blur_el"] = out["blur_el"]
    # radial vs tangential gradient energy: pinwheel smear has only tangential structure
    rho = np.hypot(u, v) + 1e-6
    gv = -gy
    gr = (gx * u + gv * v) / rho
    gt = (-gx * v + gv * u) / rho
    pm = me & (r >= 3) & (r <= 25)
    if pm.sum() > 200:
        er, et = float(np.sum(gr[pm] ** 2)), float(np.sum(gt[pm] ** 2))
        out["nad_tangential"] = et / (er + et + 1e-6)
    # orientation-constrained Hough: share of edge energy on the single strongest line
    hm = me & (r <= 35)
    gm = np.hypot(gx, gy)
    if hm.sum() > 500:
        thr = max(3.0, float(np.percentile(gm[hm], 85)))
        e = hm & (gm > thr)
        if e.sum() > 50:
            ang = np.arctan2(gv[e], gx[e]) % np.pi
            ab = (ang / np.pi * 36).astype(np.int64) % 36
            dist = u[e] * np.cos(ang) + v[e] * np.sin(ang)
            tmax = math.tan(math.radians(35.0))
            db = np.clip(((dist / tmax + 1.0) * 40).astype(np.int64), 0, 79)
            acc = np.bincount(ab * 80 + db, weights=gm[e], minlength=36 * 80).reshape(36, 80)
            acc = acc + np.roll(acc, 1, 0) + np.roll(acc, -1, 0)
            out["nad_line_peak"] = float(acc.max() / (gm[e].sum() + 1e-6))
            ia, idd = np.unravel_index(int(acc.argmax()), acc.shape)
            ctx["line"] = (float(ia * 5.0), float((idd / 40.0 - 1.0) * tmax))
        if ref_known:
            out["nad_across_frac"] = float(np.sum(gv[hm] ** 2) / (np.sum(gm[hm] ** 2) + 1e-6))
    # car-relative sectors (vehicle body shows up front/back, ground on the sides)
    sec4 = _sector_of(th, ref_known)
    sm = me & (r >= SECT_R[0]) & (r <= SECT_R[1])
    secL = np.full(4, np.nan)
    for k in range(4):
        q = sm & (sec4 == k)
        if q.sum() < 100:
            continue
        secL[k] = L[q].mean()
        if ref_known:
            out["sec_%s_L" % SECTORS[k]] = float(secL[k])
            out["sec_%s_C" % SECTORS[k]] = float(C[q].mean())
            out["sec_%s_tex" % SECTORS[k]] = float(np.median(G[q]) / (gref + 1.0)) if np.isfinite(gref) else float("nan")
    if np.isfinite(secL).sum() >= 3:
        out["nad_sector_range"] = float(np.nanmax(secL) - np.nanmin(secL))
    ctx["nad_lab"] = (L, A, B, C, core, me, r, th, sec4, sm)
    _markers(sph, out, ctx, ref, ref_known)


def _markers(sph, out, ctx, ref, ref_known):
    """Gen-3 roof-mount feet: four small bright metal bars seen at ~37-40 deg from
    the nadir, at 75/120/240/285 deg from the car's front (fixed relative to the car)."""
    th = np.arange(0.0, 360.0, 1.0)
    el = -90.0 + MARK_R[:, None] + 0.0 * th[None, :]
    phi = ref + th[None, :] + 0.0 * MARK_R[:, None]
    rgb, m = _sample_sphere(sph, el, phi, key=("marks", round(float(ref), 3)))
    if m.mean() < 0.9:
        return
    L = _lab(rgb)[..., 0]
    top = L - _box(L, 13, 13, wrap_cols=True)   # bright compact detail vs 6.5 x 13 deg surround
    resp = top.max(0)
    resp = resp - np.median(resp)
    scale = float(np.median(np.abs(resp))) + 1.0
    sm = np.maximum.reduce([np.roll(resp, k) for k in (-2, -1, 0, 1, 2)])   # +-2 deg tolerance
    ti = MARK_THETA.astype(np.int64)
    if ref_known:
        out["mark_score"] = float(np.min(sm[ti]) / scale)
        ctx["mark_score"] = out["mark_score"]
    # rotation invariant: best rotation of the 4-foot template
    rot = np.min(np.stack([np.roll(sm, -t) for t in ti]), 0)
    out["mark_score_ri"] = float(rot.max() / scale)


# ------------------------------------------------------------------------- zenith / sky
def _zenith(sph, out, ctx):
    rgb, m, r, th, u, v = _pole_view(sph, 64, 15.0, 1, 0.0)
    if m.sum() < 0.5 * (r <= 15.0).sum():
        return
    lab = _lab(rgb)
    L = lab[..., 0]
    out["zen_L"] = float(L[m].mean())
    out["zen_a"] = float(lab[..., 1][m].mean())
    out["zen_b"] = float(lab[..., 2][m].mean())
    out["zen_Lsd"] = float(L[m].std())
    gx, gy = _grad(L)
    me = _erode(m, 1)
    if me.sum() > 100:
        out["zen_tex"] = float(np.hypot(gx, gy)[me].mean())
    out["zen_black"] = float((L[m] < 8).mean())
    ctx["zen_black"] = out["zen_black"]


def _global_lab(sph, ctx):
    """Lab of the 512-wide canvas above el -45 (excludes the vehicle), stride 2."""
    if "glab" not in ctx:
        s = sph.resized(512)
        sl = s.band(89, -45)
        rgb = s.rgb[sl][::2, ::2]
        m = s.mask[sl][::2, ::2]
        el = s.elevations()[sl][::2]
        ctx["glab"] = (rgb, _lab(rgb), m, np.repeat(el[:, None], rgb.shape[1], 1))
    return ctx["glab"]


def _sky(sph, out, ctx):
    rgb, lab, mm, el = _global_lab(sph, ctx)
    sk = mm & (el >= 30) & (el <= 85)
    if sk.sum() < 200:
        return
    L, A, B = lab[..., 0][sk], lab[..., 1][sk], lab[..., 2][sk]
    C = np.hypot(A, B)
    if "zen_L" in out:
        out["zen_sky_dL"] = out["zen_L"] - float(L.mean())
    blue = (B < -8) & (L > 35) & (B < -0.5 * np.abs(A))
    out["sky_blue_frac"] = float(blue.mean())
    if blue.sum() > 30:
        out["sky_blue_C"] = float(C[blue].mean())
        out["sky_blue_hue"] = float(np.degrees(np.arctan2(-A[blue].mean(), -B[blue].mean())))
        ctx["sky_blue_C"] = out["sky_blue_C"]
    white = (L > 78) & (C < 18)
    if white.sum() > 30:
        out["sky_white_a"] = float(A[white].mean())
        out["sky_white_b"] = float(B[white].mean())
        ctx["sky_white_ab"] = (out["sky_white_a"], out["sky_white_b"])
    out["sky_clip"] = float((rgb[sk].min(1) >= 250).mean())


# ------------------------------------------------------------------------- image quality
def _quality(sph, out, ctx):
    sl = sph.band(12, -12)
    rgb = sph.rgb[sl]
    m = _erode(sph.mask[sl], 3)
    if m.sum() < 5000:
        return
    Y = rgb.astype(np.float32) @ _Y_W
    gx, gy = _grad(Y)
    g = np.hypot(gx, gy)
    lap = np.zeros_like(Y)
    lap[1:-1, 1:-1] = Y[2:, 1:-1] + Y[:-2, 1:-1] + Y[1:-1, 2:] + Y[1:-1, :-2] - 4.0 * Y[1:-1, 1:-1]
    mg = float(g[m].mean())
    out["q_lap_grad"] = float(np.abs(lap[m]).mean() / (mg + 1e-3))
    b3 = _box(Y, 3)
    b9 = _box(b3, 7)
    e1 = float(np.mean((Y - b3)[m] ** 2))
    e2 = float(np.mean((b3 - b9)[m] ** 2))
    out["q_hf_ratio"] = e1 / (e2 + 1e-3)
    out["q_grad_norm"] = mg / (float(Y[m].std()) + 1.0)
    ctx["q_lap_grad"] = out["q_lap_grad"]
    flat = m & (g < np.percentile(g[m], 30))
    if flat.sum() > 1000:
        rf = rgb.astype(np.float32)
        cb, cr = rf[..., 2] - Y, rf[..., 0] - Y
        hc = np.hypot(cb - _box(cb, 3), cr - _box(cr, 3))
        out["q_chroma_noise"] = float(hc[flat].mean() / (np.abs(Y - b3)[flat].mean() + 0.05))
    if sph.source == "pano":
        # JPEG 8x8 grid strength: only meaningful on panoramas decoded from Google tiles
        dh = np.abs(np.diff(Y, axis=1))
        dv = np.abs(np.diff(Y, axis=0))
        mh = m[:, 1:] & m[:, :-1]
        mv = m[1:, :] & m[:-1, :]
        colpos = (np.arange(dh.shape[1]) % 8 == 7)[None, :]
        rowpos = ((np.arange(dv.shape[0]) + sl.start) % 8 == 7)[:, None]
        out["pipe_block_h"] = float(dh[mh & colpos].mean() / (dh[mh & ~colpos].mean() + 1e-3))
        if (mv & rowpos).sum() > 100:
            out["pipe_block_v"] = float(dv[mv & rowpos].mean() / (dv[mv & ~rowpos].mean() + 1e-3))
    # global tone / colour (512 canvas above el -45, excludes the vehicle)
    rgb2, lab2, m2, _ = _global_lab(sph, ctx)
    if m2.sum() < 1000:
        return
    px = rgb2[m2]
    lab = lab2[m2]
    out["q_clip_hi"] = float((px.max(1) >= 252).mean())
    out["q_clip_lo"] = float((px.min(1) <= 3).mean())
    out["q_cast_a"] = float(lab[:, 1].mean())
    out["q_cast_b"] = float(lab[:, 2].mean())
    Lv = lab[:, 0]
    p1, p50, p98, p99 = np.percentile(Lv, [1, 50, 98, 99])
    hi = Lv >= p98
    out["q_white_a"] = float(lab[hi, 1].mean())
    out["q_white_b"] = float(lab[hi, 2].mean())
    C = np.hypot(lab[:, 1], lab[:, 2])
    out["q_chroma"] = float(C.mean())
    out["q_chroma_p90"] = float(np.percentile(C, 90))
    out["q_L_p1"], out["q_L_med"], out["q_L_p99"] = float(p1), float(p50), float(p99)
    ctx["q_chroma"] = out["q_chroma"]


def extract(sph):
    out, ctx = {}, {}
    for fn in (_nadir, _zenith, _sky, _quality):
        fn(sph, out, ctx)
    x = np.array([out.get(k, np.nan) for k in FEATURE_NAMES], np.float32)
    return {"x": x, "evidence": {}}
