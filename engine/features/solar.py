"""
Solar module: sun position + sky appearance (closed-form image maths only).

The extractor works on an equirectangular ``SphericalImage`` (see engine/panorama.py).
Every statistic is computed over valid (``sph.mask``) pixels only; features that
need the true orientation of the canvas are NaN when ``sph.heading`` is None.

1. SUN POSITION (three cues, best one reported)
   a) Sun disk.  On the 512x256 canvas, pixels whose minimum channel is >= 238
      and elevation > -3 deg are "saturated".  The mask is dilated by one pixel
      (to merge fragments of a disk seen through foliage) and split into
      8-connected components (run-length union-find, horizontal wrap-around).
      For each component the solid angle Omega = sum(cos(el) dphi dtheta), the
      spherical centroid c (normalised sum of unit direction vectors weighted by
      cos(el)), the equivalent cap radius r_eq = acos(1 - Omega / 2pi) and the
      maximal angular distance r_max of its pixels from c are computed.
      Candidates with 0.3 < r_eq < 12 deg are scored from the radial luminance
      profile around c: halo rings h1 = [r+0.5, r+4], h2 = [r+4, r+10] and the
      far ring [r+12, r+28] deg: a real sun has a bright, isotropic halo that
      falls off radially (Y_h1 > Y_h2 > Y_far), a mostly non-saturated far ring
      (rejects overexposed overcast sky) and a compact core (r_eq / r_max).
      A logistic combination of these terms gives ``sun_disk_conf``.
   b) Camera shadow.  The shadow of the Street View camera head is always at
      the ANTISOLAR point (phi + 180, -el): the ray from the sun through the
      camera continues to the ground.  The shadow of the camera mast is the
      ground line from the nadir to that point, i.e. a vertical dark stripe in
      equirectangular coordinates ending in a dark blob.  A matched filter
      (dark blob darker than its ring, plus darker stripe below it) is
      evaluated below the horizon; a hit gives the sun at (phi_s - 180, -el_s).
      It also confirms disk detections (shadow present at the antisolar point).
   c) Solar aureole.  When no disk/shadow is found, sky luminance residuals
      (after removing the median elevation profile of sky brightness) are
      averaged in 36 azimuth sectors; the first circular harmonic gives the
      azimuth of the brightest sky sector (forward scattering around the sun).
      Elevation is unknown for this cue.

2. ORIENTATION + LATITUDE.  With ``sph.heading`` the relative longitude becomes
   the true azimuth A.  Features cos(A) (> 0: sun towards the north, i.e. the
   observer is south of the subsolar point) and sin(A).
   ``latitude_likelihood(A, h, lat_grid)`` integrates over solar declination
   (day of year uniform -> delta = 23.44 sin(2 pi (d - 81) / 365), an arcsine
   density, smoothed by 1 deg) and hour angle (uniform over daylight,
   |H| <= H0 with cos H0 = -tan(phi) tan(delta)).  The integral is evaluated by
   a change of variables (delta, H) -> (A, h): both are spherical coordinates
   related by a rotation, so cos(delta) d(delta) dH = cos(h) dh dA and the
   density of the sun position at latitude phi is
        p_phi(A, h) = p(delta) / (2 H0) * cos(h) / cos(delta),
        sin(delta) = sin(phi) sin(h) + cos(phi) cos(h) cos(A).
   The Gaussian kernel N(A_obs - A; sigma_A) N(h_obs - h; sigma_h) is applied by
   Gauss-Hermite-like quadrature around the observation (sigma_A is widened by
   sigma_h * tan(h) because azimuth is ill-conditioned near the zenith).
   Features: maximum-likelihood latitude (restricted to -56..72 deg),
   posterior P(lat < 0) under a flat prior on -56..72, and log likelihood
   (per-degree density, mixed with a uniform density by 1 - confidence) at the
   latitude band centres -45 .. 60.

3. SKY APPEARANCE (no heading needed).  Sky pixels: el > -2, low texture
   (5x5 mean |grad L*| small) and sky colour (blue: b* < -4, or neutral bright:
   C*ab < 14 & L* > 55, or saturated), kept when connected to the upper sky.
   Features: sky fraction of the valid upper hemisphere; mean chromaticity
   r, g, b = R/(R+G+B) ...; mean L*, a*, b*; blueness (B - R)/(B + R);
   clear fraction (b* < -8), cloud fraction (C* < 10 or saturated), overcast
   score (grey, C* < 8 and L* < 88), saturated fraction, haze (RMS contrast of
   L* in el 0..5 over el 5..15), horizon chroma ratio (sky chroma el 0..10 over
   20..40), vertical gradients of L* and b* (el 45..90 minus el 5..20) and sky
   texture.
"""

import math

import numpy as np

NAME = "solar"

LAT_BANDS = [-45, -30, -15, 0, 15, 30, 45, 60]
_BAND_NAMES = ["m45", "m30", "m15", "0", "p15", "p30", "p45", "p60"]

FEATURE_NAMES = (
    ["sun_conf", "sun_disk_conf", "shadow_conf", "aureole_conf", "sun_el", "sun_az_cos", "sun_az_sin",
     "sun_lat_map", "sun_p_south"]
    + ["sun_lat_ll_" + b for b in _BAND_NAMES]
    + ["sky_frac", "sky_r", "sky_g", "sky_b", "sky_L", "sky_a", "sky_bstar", "sky_blueness",
       "sky_clear_frac", "sky_cloud_frac", "sky_overcast", "sky_sat_frac", "sky_haze_contrast",
       "sky_horizon_chroma", "sky_grad_L", "sky_grad_b", "sky_texture"]
)

_W = 512                 # working resolution for sun / sky analysis
_SAT = 238               # min-channel threshold of a saturated pixel (after bilinear downsample)
_OBLIQ = 23.44
_LAT_LO, _LAT_HI = -56.0, 72.0   # inhabited Street View latitude range


# ---------------------------------------------------------------- geometry
_GEOM = {}


def _geom(w):
    if w not in _GEOM:
        h = w // 2
        phi = (np.arange(w) + 0.5) / w * 360.0 - 180.0
        el = 90.0 - (np.arange(h) + 0.5) / h * 180.0
        pr, er = np.radians(phi), np.radians(el)
        ce = np.cos(er)
        dirs = np.empty((h, w, 3), np.float32)
        dirs[..., 0] = np.sin(pr)[None, :] * ce[:, None]
        dirs[..., 1] = np.cos(pr)[None, :] * ce[:, None]
        dirs[..., 2] = np.sin(er)[:, None]
        dA = (2 * math.pi / w) * (math.pi / h) * ce   # solid angle of a pixel per row
        _GEOM[w] = {"phi": phi, "el": el, "dirs": dirs, "dA": dA.astype(np.float64), "cos_el": ce}
    return _GEOM[w]


def _vec_to_angles(v):
    v = v / (np.linalg.norm(v) + 1e-12)
    el = math.degrees(math.asin(max(-1.0, min(1.0, float(v[2])))))
    phi = math.degrees(math.atan2(float(v[0]), float(v[1])))
    return phi, el


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


# -------------------------------------------------------- image primitives
def _box(a, r):
    """Separable box mean of radius r; wraps in x, clamps in y."""
    if r <= 0:
        return a
    k = 2 * r + 1
    p = np.concatenate([a[:, -r:], a, a[:, :r]], axis=1)
    c = np.cumsum(p, axis=1, dtype=np.float64)
    c = np.concatenate([np.zeros((a.shape[0], 1)), c], axis=1)
    b = (c[:, k:] - c[:, :-k]) / k
    p = np.concatenate([np.repeat(b[:1], r, 0), b, np.repeat(b[-1:], r, 0)], axis=0)
    c = np.cumsum(p, axis=0)
    c = np.concatenate([np.zeros((1, a.shape[1])), c], axis=0)
    return ((c[k:] - c[:-k]) / k).astype(np.float32)


def _dilate(m):
    d = m | np.roll(m, 1, axis=1) | np.roll(m, -1, axis=1)
    out = d.copy()
    out[1:] |= d[:-1]
    out[:-1] |= d[1:]
    return out


def _label(mask):
    """8-connected components with horizontal wrap-around (run-length union-find).

    Returns (labels int32 (h, w) with 0 = background, number of components)."""
    h, w = mask.shape
    m = mask.astype(np.int8)
    d = np.diff(np.pad(m, ((0, 0), (1, 1))), axis=1)
    ry, rs = np.nonzero(d == 1)
    _, re = np.nonzero(d == -1)
    n = len(rs)
    lab = np.zeros((h, w), np.int32)
    if n == 0:
        return lab, 0
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    row_start = np.searchsorted(ry, np.arange(h + 1)).tolist()
    rs_l, re_l = rs.tolist(), re.tolist()
    for y in range(h - 1):
        a0, a1, b1 = row_start[y], row_start[y + 1], row_start[y + 2]
        i, j = a0, a1
        while i < a1 and j < b1:
            if rs_l[j] <= re_l[i] and rs_l[i] <= re_l[j]:
                union(i, j)
            if re_l[i] < re_l[j]:
                i += 1
            else:
                j += 1
    # horizontal wrap (same row and diagonal neighbours across the seam)
    left = {}
    right = {}
    for y in range(h):
        a0, a1 = row_start[y], row_start[y + 1]
        if a1 > a0:
            if rs_l[a0] == 0:
                left[y] = a0
            if re_l[a1 - 1] == w:
                right[y] = a1 - 1
    for y, i in left.items():
        for yy in (y - 1, y, y + 1):
            j = right.get(yy)
            if j is not None:
                union(i, j)
    roots = np.array([find(i) for i in range(n)])
    uniq, comp = np.unique(roots, return_inverse=True)
    lens = re - rs
    starts = ry * w + rs
    idx = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)
    lab.flat[idx] = np.repeat(comp + 1, lens)
    return lab, len(uniq)


def _srgb_to_lab(rgb):
    """rgb float (..., 3) in 0..255 -> CIE L*a*b* (D65)."""
    c = rgb / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    X = lin @ np.array([0.4124, 0.3576, 0.1805]) / 0.95047
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    Z = lin @ np.array([0.0193, 0.1192, 0.9505]) / 1.08883

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(X), f(Y), f(Z)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


# ------------------------------------------------------ latitude likelihood
def _build_decl_table():
    # day-of-year uniform -> delta; density on a fine grid, smoothed by 1 deg
    step = 0.05
    grid = np.arange(-40.0, 40.0 + 1e-9, step)
    d = np.arange(0.0, 365.0, 0.01)
    delta = _OBLIQ * np.sin(2 * np.pi * (d - 81.0) / 365.0)
    hist, _ = np.histogram(delta, bins=len(grid), range=(grid[0] - step / 2, grid[-1] + step / 2))
    k = np.arange(-80, 81) * step
    ker = np.exp(-0.5 * (k / 1.0) ** 2)
    ker /= ker.sum()
    dens = np.convolve(hist.astype(np.float64), ker, mode="same")
    dens /= dens.sum() * math.radians(step)          # per radian of delta
    cdf = np.cumsum(dens) * math.radians(step)
    return grid, dens, cdf


_DECL_GRID, _DECL_DENS, _DECL_CDF = _build_decl_table()
_QN = 9
_QU = np.linspace(-2.6, 2.6, _QN)
_QW = np.exp(-0.5 * _QU ** 2)
_QW /= _QW.sum()


def latitude_likelihood(az_true_deg, el_deg, lat_grid_deg, sigma_az=8.0, sigma_el=4.0):
    """Likelihood of observer latitude given an observed sun position.

    az_true_deg: true azimuth of the sun (deg clockwise from north) or None
                 (azimuth unknown -> marginalised).
    el_deg:      sun elevation (deg) or None (marginalised over 0..90).
    lat_grid_deg: 1-D array of latitudes.
    Returns an array (len(lat_grid),) normalised to sum 1 (uniform if the
    observation is impossible everywhere).  Both az and el may also be 1-D
    arrays of equal length N (or None) -> returns (N, len(lat_grid)).

    p(obs | phi) = int int p(delta) p(H | delta, phi) K(obs - sun(phi, delta, H)) dH ddelta
    computed in horizontal coordinates: p_phi(A, h) = p(delta) / (2 H0) cos h / cos delta,
    with p(delta) the day-of-year-uniform declination density and H uniform on daylight.
    """
    lat = np.atleast_1d(np.asarray(lat_grid_deg, np.float64))
    scalar = np.ndim(az_true_deg) == 0 and np.ndim(el_deg) == 0
    n = 1
    for v in (az_true_deg, el_deg):
        if v is not None and np.ndim(v) > 0:
            n = len(v)
    out = np.empty((n, len(lat)))
    for i in range(n):
        a = None if az_true_deg is None else float(np.ravel(az_true_deg)[i if np.ndim(az_true_deg) else 0])
        e = None if el_deg is None else float(np.ravel(el_deg)[i if np.ndim(el_deg) else 0])
        out[i] = _lat_like_one(a, e, lat, float(sigma_az), float(sigma_el))
    return out[0] if scalar else out


def _lat_like_one(az, el, lat, sigma_az, sigma_el):
    if az is not None and not np.isfinite(az):
        az = None
    if el is not None and not np.isfinite(el):
        el = None
    if el is not None:
        hs = el + _QU * sigma_el
        wh = _QW
    else:
        hs = np.arange(1.0, 90.0, 2.0)
        wh = np.full(len(hs), 1.0 / len(hs))
    if az is not None:
        sa = sigma_az if el is None else math.hypot(sigma_az, sigma_el * math.tan(math.radians(min(abs(el), 87.0))))
        sa = min(sa, 180.0)
        As = az + _QU * sa
        wa = _QW
    else:
        As = np.arange(0.0, 360.0, 5.0)
        wa = np.full(len(As), 1.0 / len(As))
    hr = np.radians(hs)[None, :, None]
    ar = np.radians(As)[None, None, :]
    pr = np.radians(lat)[:, None, None]
    sd = np.sin(pr) * np.sin(hr) + np.cos(pr) * np.cos(hr) * np.cos(ar)
    sd = np.clip(sd, -1.0, 1.0)
    dl = np.degrees(np.arcsin(sd))
    pdl = np.interp(dl, _DECL_GRID, _DECL_DENS, left=0.0, right=0.0)
    cosd = np.sqrt(np.maximum(1.0 - sd * sd, 1e-12))
    x = -np.tan(pr) * sd / cosd
    H0 = np.arccos(np.clip(x, -1.0, 1.0))
    # renormalise by the fraction of days that have daylight (polar night)
    lim = 90.0 - np.abs(lat)
    day_frac = np.where(lat >= 0, 1.0 - np.interp(-lim, _DECL_GRID, _DECL_CDF, left=0.0, right=1.0),
                        np.interp(lim, _DECL_GRID, _DECL_CDF, left=0.0, right=1.0))
    dens = pdl / (2.0 * np.maximum(H0, 1e-3)) * np.cos(hr) / cosd
    dens = np.where(hr > 0, dens, 0.0)
    L = np.einsum("lij,i,j->l", dens, wh, wa) / np.maximum(day_frac, 1e-6)
    s = L.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(len(lat), 1.0 / len(lat))
    return L / s


_LAT_GRID = np.arange(-89.5, 90.0, 1.0)
_INHAB = (_LAT_GRID >= _LAT_LO) & (_LAT_GRID <= _LAT_HI)


def _lat_features(az, el, conf):
    """MAP latitude, P(south) and log densities at band centres (mixture with uniform)."""
    L = latitude_likelihood(az, el, _LAT_GRID)
    Li = np.where(_INHAB, L, 0.0)
    Li = Li / max(Li.sum(), 1e-300)
    lat_map = float(_LAT_GRID[int(np.argmax(Li))])
    p_south = float(Li[_LAT_GRID < 0].sum())
    u = 1.0 / (_LAT_HI - _LAT_LO)
    mix = conf * Li + (1.0 - conf) * u * _INHAB
    ll = [float(np.log(np.interp(b, _LAT_GRID, mix) + 1e-4)) for b in LAT_BANDS]
    return lat_map, p_south, ll, L


# ----------------------------------------------------------------- sky mask
def _sky(rgb, valid, g):
    """Sky mask + Lab image on the upper rows (el > -2)."""
    el = g["el"]
    nrow = int(np.searchsorted(-el, 2.0))     # rows with el > -2
    up = rgb[:nrow].astype(np.float32)
    lab = _srgb_to_lab(up).astype(np.float32)
    L = lab[..., 0]
    gx = np.abs(np.roll(L, -1, axis=1) - L)
    gy = np.zeros_like(L)
    gy[:-1] = np.abs(L[1:] - L[:-1])
    tex = _box(gx + gy, 2)
    C = np.hypot(lab[..., 1], lab[..., 2])
    mn = up.min(2)
    sat = mn >= _SAT
    blue = (lab[..., 2] < -4) & (L > 35) & (lab[..., 1] < 20)
    neutral = (C < 14) & (L > 55)
    cand = valid[:nrow] & (tex < 7.0) & (blue | neutral | sat)
    lab_ids, n = _label(cand)
    sky = np.zeros_like(cand)
    if n:
        vrows = np.nonzero(valid[:nrow].any(1))[0]
        top_el = el[vrows[0]] if len(vrows) else 90.0
        need = min(40.0, top_el - 5.0)
        ids = lab_ids[cand]
        rows = np.nonzero(cand)[0]
        maxel = np.full(n + 1, -99.0)
        np.maximum.at(maxel, ids, el[rows])
        area = np.bincount(ids, weights=g["cos_el"][rows], minlength=n + 1)
        keep = (maxel >= need) & (area > 30)
        keep[0] = False
        sky = keep[lab_ids] & cand
    return {"nrow": nrow, "lab": lab, "sky": sky, "tex": tex, "sat": sat, "C": C}


# ----------------------------------------------------------------- sun disk
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _nz(v, d):
    return d if (v is None or not np.isfinite(v)) else float(v)


def _ring_stats(c, r0, P, g, rows):
    """Statistics of rings around direction c (core radius r0 deg).  P: per-pixel planes."""
    dirs = g["dirs"][rows]
    cosd = dirs @ c.astype(np.float32)
    ang = np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))
    out = {}
    v = P["valid"][rows]
    Yr, Sr, Kr, Br = P["Y"][rows], P["sat"][rows], P["sky"][rows], P["bstar"][rows]
    for name, lo, hi in (("core", -1.0, r0), ("h1", r0 + 0.5, r0 + 4.0), ("h2", r0 + 4.0, r0 + 10.0),
                         ("far", r0 + 12.0, r0 + 28.0)):
        ring = (ang >= lo) & (ang < hi)
        rv = ring & v
        nr = int(ring.sum())
        nv = int(rv.sum())
        out[name + "_vf"] = nv / max(nr, 1)
        out[name + "_Y"] = float(Yr[rv].mean()) if nv else float("nan")
        out[name + "_sat"] = float(Sr[rv].mean()) if nv else float("nan")
        out[name + "_sky"] = float(Kr[rv].mean()) if nv else float("nan")
        if name in ("h2", "far"):
            ks = rv & Kr & ~Sr
            out[name + "_bstar"] = float(Br[ks].mean()) if ks.sum() >= 5 else float("nan")
        if name == "h1" and nv >= 8:
            # isotropy: position angle around c, 8 sectors
            e = np.cross(c, [0.0, 0.0, 1.0])
            if np.linalg.norm(e) < 1e-6:
                e = np.array([1.0, 0.0, 0.0])
            e = e / np.linalg.norm(e)
            n2 = np.cross(c, e)
            dv = dirs[rv]
            pa = np.arctan2(dv @ n2.astype(np.float32), dv @ e.astype(np.float32))
            sec = ((pa + np.pi) / (2 * np.pi) * 8).astype(int) % 8
            cnt = np.bincount(sec, minlength=8)
            sm = np.bincount(sec, weights=Yr[rv], minlength=8)
            ok = cnt >= 2
            means = sm[ok] / cnt[ok]
            out["iso"] = float(means.min() / max(means.max(), 1.0)) if ok.sum() >= 4 else float("nan")
    return out


def _sun_disk(P, g):
    el, h = g["el"], len(g["el"])
    nrow = int(np.searchsorted(-el, 3.0))     # el > -3
    sat = P["sat"][:nrow] & P["valid"][:nrow]
    if sat.sum() < 2:
        return []
    lab, n = _label(_dilate(sat))
    ys, xs = np.nonzero(sat)
    ids = lab[ys, xs]
    w = g["cos_el"][ys]
    dv = g["dirs"][ys, xs].astype(np.float64)
    area = np.bincount(ids, weights=w, minlength=n + 1) * (2 * math.pi / len(g["phi"])) * (math.pi / h)
    cx = np.bincount(ids, weights=w * dv[:, 0], minlength=n + 1)
    cy = np.bincount(ids, weights=w * dv[:, 1], minlength=n + 1)
    cz = np.bincount(ids, weights=w * dv[:, 2], minlength=n + 1)
    cen = np.stack([cx, cy, cz], 1)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-12
    cosd = np.einsum("ij,ij->i", dv, cen[ids])
    mincos = np.ones(n + 1)
    np.minimum.at(mincos, ids, cosd)
    req = np.degrees(np.arccos(np.clip(1.0 - area / (2 * math.pi), -1, 1)))
    rmax = np.degrees(np.arccos(np.clip(mincos, -1, 1)))
    order = np.argsort(-area)
    total_sat = float(area[1:].sum())
    cands = []
    for k in order[:6]:
        if k == 0 or req[k] < 0.3 or req[k] > 15.0:
            continue
        c = cen[k]
        phi_c, el_c = _vec_to_angles(c)
        if el_c < -2.0:
            continue
        r0 = max(req[k], 0.5)
        reach = r0 + 28.0
        r0w = int(np.searchsorted(-el, -min(el_c + reach, 90.0)))
        r1w = int(np.searchsorted(-el, -max(el_c - reach, -90.0)))
        rows = slice(r0w, max(r1w, r0w + 1))
        st = _ring_stats(c, r0, P, g, rows)
        st.update({"phi": phi_c, "el": el_c, "req": float(req[k]), "rmax": float(rmax[k]),
                   "compact": float(req[k] / max(rmax[k], 0.35)), "c": c,
                   "area_share": float(area[k] / max(total_sat, 1e-12))})
        cands.append(st)
    return cands


def _disk_score(st):
    h1, far = _nz(st.get("h1_Y"), 0.0), _nz(st.get("far_Y"), 255.0)
    far_sat = _nz(st.get("far_sat"), 1.0)
    h2_sat = _nz(st.get("h2_sat"), 1.0)
    iso = _nz(st.get("iso"), 0.3)
    falloff = (h1 - far) / 255.0
    x = (-4.0 + 9.0 * falloff + 4.0 * (h1 > 225) + 2.5 * iso - 5.0 * far_sat - 2.0 * h2_sat
         + 1.5 * min(st["compact"], 1.0) - 0.15 * max(st["req"] - 6.0, 0.0))
    return _sigmoid(x)


# ------------------------------------------------------------ camera shadow
_SHADOW_K = 2.3          # angular radius of the camera-head shadow = K * sin(sun elevation) deg
_SW = 1024


def _shadow_contrast(Y, phi, el, search=2.5):
    """Darkness of a camera-head-sized blob near (phi, el) (el < 0) relative to all four sides.

    Y: luminance (h, w) at _SW resolution.  Returns (contrast, phi, el) of the best position
    within +-search deg, contrast = 1 - mean(inner) / min(mean(left, right, top, bottom))."""
    h, w = Y.shape
    px = 360.0 / w
    e = -el
    if not (3.0 <= e <= 82.0):
        return float("nan"), phi, el
    r = max(0.7, _SHADOW_K * math.sin(math.radians(e)))
    ce = max(math.cos(math.radians(el)), 0.15)
    a = max(1, int(round(0.75 * r / px)))
    b = max(1, min(60, int(round(r / (px * ce)))))
    ga, gb = max(1, a // 2), max(1, b // 2)
    sy = int(math.ceil(search / px)) + 1
    sx = int(math.ceil(search / (px * ce))) + 1
    yc = int((90.0 - el) / 180.0 * h)
    xc = int((phi + 180.0) / 360.0 * w)
    padY = 3 * a + ga + sy + 2
    padX = 3 * b + gb + sx + 2
    ys = np.clip(np.arange(yc - padY, yc + padY + 1), 0, h - 1)
    xs = np.arange(xc - padX, xc + padX + 1) % w
    patch = Y[np.ix_(ys, xs)].astype(np.float64)
    I = np.zeros((patch.shape[0] + 1, patch.shape[1] + 1))
    I[1:, 1:] = np.cumsum(np.cumsum(patch, 0), 1)
    cy = np.arange(padY - sy, padY + sy + 1)[:, None]
    cx = np.arange(padX - sx, padX + sx + 1)[None, :]

    def box(dy0, dy1, dx0, dx1):
        y0, y1, x0, x1 = cy + dy0, cy + dy1, cx + dx0, cx + dx1
        return (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / float((dy1 - dy0) * (dx1 - dx0))

    inner = box(-a, a + 1, -b, b + 1)
    sides = np.minimum(np.minimum(box(-a, a + 1, -3 * b - gb, -b - gb), box(-a, a + 1, b + 1 + gb, 3 * b + 1 + gb)),
                       np.minimum(box(-3 * a - ga, -a - ga, -b, b + 1), box(a + 1 + ga, 3 * a + 1 + ga, -b, b + 1)))
    C = 1.0 - inner / np.maximum(sides, 1.0)
    k = int(np.argmax(C))
    iy, ix = np.unravel_index(k, C.shape)
    return float(C[iy, ix]), phi + (ix - sx) * px * ce / ce * 1.0, el - (iy - sy) * px


# ------------------------------------------------------------------- aureole
def _aureole(P, g, sk):
    """Azimuth of the brightest sky sector after removing the elevation profile."""
    el = g["el"]
    nrow = sk["nrow"]
    sky = sk["sky"] & ~P["sat"][:nrow]
    rows = (el[:nrow] > 3.0) & (el[:nrow] < 60.0)
    m = sky & rows[:, None]
    if m.sum() < 400:
        return None
    L = sk["lab"][..., 0]
    ys, xs = np.nonzero(m)
    v = L[ys, xs]
    eb = np.clip(((el[ys] - 3.0) / 3.0).astype(int), 0, 19)
    med = np.zeros(20)
    for b in np.unique(eb):
        med[b] = np.median(v[eb == b])
    res = v - med[eb]
    nsec = 36
    sec = (xs * nsec // len(g["phi"])).astype(int)
    cnt = np.bincount(sec, minlength=nsec)
    sm = np.bincount(sec, weights=res, minlength=nsec)
    ok = cnt >= 15
    if ok.sum() < 12:
        return None
    mean = np.where(ok, sm / np.maximum(cnt, 1), 0.0)
    ang = np.radians((np.arange(nsec) + 0.5) / nsec * 360.0 - 180.0)
    wts = ok.astype(float)
    z = np.sum(wts * mean * np.exp(1j * ang)) / wts.sum()
    spread = float(np.sqrt(np.sum(wts * mean ** 2) / wts.sum())) + 1e-6
    amp = float(abs(z))
    return {"phi": float(np.degrees(np.angle(z))), "amp": amp, "ratio": amp / spread,
            "coverage": float(ok.mean())}


# ------------------------------------------------------------------- extract
def _nan_vec():
    return np.full(len(FEATURE_NAMES), np.nan, np.float32)


def _planes(s, g):
    rgb = s.rgb
    valid = s.mask
    sk = _sky(rgb, valid, g)
    h, w = valid.shape
    nrow = sk["nrow"]
    Y = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)
    sat = rgb.min(2) >= _SAT
    sky = np.zeros((h, w), bool)
    sky[:nrow] = sk["sky"]
    bstar = np.zeros((h, w), np.float32)
    bstar[:nrow] = sk["lab"][..., 2]
    return {"valid": valid, "Y": Y, "sat": sat, "sky": sky, "bstar": bstar}, sk


def _sky_features(P, sk, g, F):
    el = g["el"]
    nrow = sk["nrow"]
    valid = P["valid"][:nrow]
    up = valid & (el[:nrow] > 0)[:, None]
    nup = int(up.sum())
    if nup < 200:
        return
    sky = sk["sky"] & up
    ns = int(sky.sum())
    F["sky_frac"] = ns / nup
    lab = sk["lab"]
    sat = P["sat"][:nrow]
    # haze: RMS contrast of L* near the horizon vs higher up (all valid pixels)
    L = lab[..., 0]
    b0 = valid & ((el[:nrow] >= 0) & (el[:nrow] < 5))[:, None]
    b1 = valid & ((el[:nrow] >= 5) & (el[:nrow] < 15))[:, None]
    if b0.sum() > 100 and b1.sum() > 100:
        F["sky_haze_contrast"] = float(L[b0].std() / (L[b1].std() + 1.0))
    if ns < 100:
        return
    w = g["cos_el"][:nrow][:, None] * np.ones((1, sky.shape[1]), np.float32)
    ws = w[sky]
    wsum = ws.sum()

    def wmean(a):
        return float((a[sky] * ws).sum() / wsum)

    rgbf = P["rgb_up"].astype(np.float32)
    tot = rgbf.sum(2) + 1.0
    F["sky_r"] = wmean(rgbf[..., 0] / tot)
    F["sky_g"] = wmean(rgbf[..., 1] / tot)
    F["sky_b"] = wmean(rgbf[..., 2] / tot)
    F["sky_blueness"] = wmean((rgbf[..., 2] - rgbf[..., 0]) / (rgbf[..., 2] + rgbf[..., 0] + 1.0))
    F["sky_L"] = wmean(L)
    F["sky_a"] = wmean(lab[..., 1])
    F["sky_bstar"] = wmean(lab[..., 2])
    C = sk["C"]
    clear = (lab[..., 2] < -8) & ~sat
    cloud = (C < 10) | sat
    grey = (C < 8) & (L < 88) & ~sat
    F["sky_clear_frac"] = wmean(clear.astype(np.float32))
    F["sky_cloud_frac"] = wmean(cloud.astype(np.float32))
    F["sky_overcast"] = wmean(grey.astype(np.float32))
    F["sky_sat_frac"] = wmean(sat.astype(np.float32))
    F["sky_texture"] = wmean(sk["tex"])
    hor = sky & ((el[:nrow] >= 0) & (el[:nrow] < 10))[:, None]
    mid = sky & ((el[:nrow] >= 20) & (el[:nrow] < 40))[:, None]
    if hor.sum() > 30 and mid.sum() > 30:
        F["sky_horizon_chroma"] = float((C[hor].mean() + 1.0) / (C[mid].mean() + 1.0))
    zen = sky & (el[:nrow] >= 45)[:, None]
    low = sky & ((el[:nrow] >= 5) & (el[:nrow] < 20))[:, None]
    if zen.sum() > 30 and low.sum() > 30:
        F["sky_grad_L"] = float(L[zen].mean() - L[low].mean())
        F["sky_grad_b"] = float(lab[..., 2][zen].mean() - lab[..., 2][low].mean())


def _disk_candidates(sph, P, g, Ysh):
    cands = _sun_disk(P, g)
    for c in cands:
        c["score"] = _disk_score(c)
        ap = c["phi"] + 180.0 if c["phi"] < 0 else c["phi"] - 180.0
        sc, sp, se = _shadow_contrast(Ysh, ap, -c["el"])
        c["shadow"] = sc
    cands.sort(key=lambda s: -s["score"])
    return cands


def extract(sph):
    x = _nan_vec()
    ev = {}
    s = sph.resized(_W) if sph.w != _W else sph
    g = _geom(_W)
    P, sk = _planes(s, g)
    P["rgb_up"] = s.rgb[: sk["nrow"]]
    F = dict()
    s2 = sph.resized(_SW) if sph.w != _SW else sph
    Ysh = (s2.rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))

    # ---------------------------------------------------------- sky
    _sky_features(P, sk, g, F)

    # ---------------------------------------------------------- sun disk
    cands = _disk_candidates(sph, P, g, Ysh)
    best = cands[0] if cands else None
    upper_valid = P["valid"][: int(np.searchsorted(-g["el"], 0.0))]
    measurable = upper_valid.mean() > 0.02
    disk_conf = best["score"] if best else (0.0 if measurable else float("nan"))
    F["sun_disk_conf"] = disk_conf

    sun_phi = sun_el = None
    sun_conf = 0.0 if measurable else float("nan")
    cue = None
    if best is not None and best["score"] >= 0.5:
        sun_phi, sun_el, sun_conf, cue = best["phi"], best["el"], best["score"], "disk"

    F["shadow_conf"] = 0.0 if measurable else float("nan")
    F["aureole_conf"] = 0.0 if measurable else float("nan")
    F["sun_conf"] = sun_conf
    if sun_phi is not None:
        F["sun_el"] = sun_el
        ev["sun"] = {"cue": cue, "rel_longitude": round(sun_phi, 1), "elevation": round(sun_el, 1),
                     "confidence": round(sun_conf, 3)}
        if sph.heading is not None:
            az = (sph.heading + sun_phi) % 360.0
            F["sun_az_cos"] = math.cos(math.radians(az))
            F["sun_az_sin"] = math.sin(math.radians(az))
            lat_map, p_south, ll, _ = _lat_features(az, sun_el, sun_conf)
            F["sun_lat_map"] = lat_map
            F["sun_p_south"] = p_south
            for b, v in zip(_BAND_NAMES, ll):
                F["sun_lat_ll_" + b] = v
            ev["sun"]["true_azimuth"] = round(az, 1)
            ev["sun"]["direction"] = "north" if math.cos(math.radians(az)) > 0 else "south"
            ev["latitude_from_sun"] = {"map_lat": lat_map, "p_south": round(p_south, 3)}
    for i, n in enumerate(FEATURE_NAMES):
        if n in F and F[n] is not None:
            x[i] = F[n]
    return {"x": x, "evidence": ev, "_cands": cands}
