"""
Landscape: vegetation, soil, snow, water, terrain relief and the overall
"climate look" of a panorama.  Pure closed-form colour/geometry maths
(numpy only, no learned models).

Everything is computed on the 512 x 256 equirectangular downsample
(0.70 deg per pixel) and only over pixels where ``sph.mask`` is true.

Per-pixel quantities
  linear RGB      sRGB decoded with a 256-entry lookup table
  CIE L*a*b*      D65 white, standard sRGB matrix, cube-root companding
  HSV             hue H (deg), saturation S = (max-min)/max, value V = max
  ExG             excess green on normalised chromaticity, 2g - r - b
                  (r,g,b = R,G,B / (R+G+B)) = 3g - 1
  texture T       local mean (3x3 box) of |dL*/dx| + |dL*/dy|  (L* units per pixel)

Sky / skyline
  A pixel is "sky" when it is above el -4 deg, smooth (T below a threshold) and
  either blue (H 180-262, S > 0.08, L* > 35) or a bright low-saturation
  white/grey (cloud, overcast).  For every column the skyline is the first row
  (scanning down from the zenith) that starts a run of 3 non-sky pixels
  (2.1 deg), so thin wires do not stop the scan.  The per-column skyline
  elevation is median-filtered over 5 columns (3.5 deg) to drop poles.
  Columns whose first valid pixel is already non-sky (tree canopy overhead,
  canvas starting below the sky) are "censored" at that elevation.

Pixel classes (non-sky pixels; first matching rule wins)
  snow     L* > 78, S < 0.10, b* < 6, smooth (T < 7)          -- on ground/horizon
  water    below el -1, blue hue 175-255, S > 0.12, smooth     -- sea / lakes
  green    hue 62-170, ExG > 0.035, S > 0.10, V > 0.05         -- living vegetation
             dark green (conifer-like): green and L* < 32 and hue > 85
             light green (pasture/broadleaf in sun): green and L* >= 45
  autumn   hue < 38 or > 340, S > 0.40, V > 0.25, textured     -- canopy/horizon only
  dry      hue 30-62, S 0.16-0.65, L* 35-88, textured (T > 3)  -- straw, dry grass
  red      hue < 28 or > 345, S 0.22-0.85, L* 22-72, a* > 9    -- laterite / red soil
  sand     hue 22-62, S 0.08-0.45, L* > 58, smooth             -- desert sand / beige
  brown    hue 15-50, S 0.12-0.55, L* 18-58                    -- brown earth
  grey     S < 0.12, L* 25-80                                  -- rock, gravel, concrete
  black    L* < 22, S < 0.35                                   -- volcanic soil / deep shade

Elevation bands
  canopy   el  +5 .. +30   (things rising above the horizon: trees, hills)
  horizon  el  -5 .. +5
  ground   el -35 .. -5    restricted to the "side" region: the car's own road
           strip is removed with a wedge |phi - axis| < atan(2.2 tan(-el))
           (road half-width ~5.5 m seen from a 2.5 m camera), min 12 deg, in
           front of and behind the car.  The axis is sph.car_heading when known;
           otherwise it is estimated as the azimuth whose front/back wedges are
           the most asphalt-like (grey, smooth).

Features (NaN = not measurable, e.g. band not covered by the mask)
  veg_<band>_{green,dark,light,dry}   class fractions of valid band pixels
  veg_<band>_exg                       mean ExG over non-sky band pixels
  veg_autumn                           autumn-foliage fraction (canopy+horizon)
  veg_hue / veg_sat / veg_L            mean hue, saturation, L* of green pixels
  soil_{red,brown,sand,grey,black,green,dry} class fractions on the ground side
  soil_a / soil_b / soil_L             mean a*, b*, L* of bare ground (not green, not snow)
  soil_redness                         mean of clip(a* - 0.35 b*, 0, 40) over bare ground
  snow_ground / snow_horizon           snow fraction on ground side / non-sky horizon band
  water_frac                           water fraction beside the road below the horizon (el -15..-1)
  sky_{mean,std,p10,p50,p90,max}       skyline elevation statistics (deg, capped at 60)
  sky_gt{3,8,15,30}                    fraction of columns whose skyline exceeds X deg
  sky_open                             fraction of columns with skyline < 1.5 deg (open horizon)
  sky_rough                            mean |s(x+1) - s(x)| (deg/0.7 deg) over raised columns
  sky_censored                         fraction of columns censored (canopy overhead)
  far_frac / far_el / far_max          distant-terrain columns: raised skyline (>1.5 deg) whose
                                       band just below the skyline is smooth, not green and low
                                       contrast to the sky (hazy mountains/hills)
  near_veg_frac / built_frac           raised columns whose sub-skyline band is vegetation /
                                       grey textured (buildings)
  sub_{L,a,b,tex,contrast}             mean colour / texture / sky-contrast of the sub-skyline band
  skyc_{blue,L,sat,haze}               sky colour: blue fraction, mean L*, mean S,
                                       L*(el 2..8) - L*(el 35..60)
  lab_<band>_{L,a,b}_{mean,std}        L*a*b* statistics of non-sky pixels per band
  ab_<i><j>                            3x3 a*/b* histogram (fractions) of non-sky pixels in the
                                       horizon + ground-side bands; a* bins <-4|-4..6|>6,
                                       b* bins <8|8..22|>22
"""

import math

import numpy as np

NAME = "landscape"
WORK_W = 512

BANDS = (("canopy", 30.0, 5.0), ("horizon", 5.0, -5.0), ("ground", -5.0, -35.0))
SOIL_CLASSES = ("red", "brown", "sand", "grey", "black", "green", "dry")

FEATURE_NAMES = []
for _b, _t, _l in BANDS:
    FEATURE_NAMES += ["veg_%s_%s" % (_b, k) for k in ("green", "dark", "light", "dry", "exg")]
FEATURE_NAMES += ["veg_autumn", "veg_hue", "veg_sat", "veg_L"]
FEATURE_NAMES += ["soil_%s" % c for c in SOIL_CLASSES]
FEATURE_NAMES += ["soil_a", "soil_b", "soil_L", "soil_redness", "snow_ground", "snow_horizon", "water_frac"]
FEATURE_NAMES += ["sky_mean", "sky_std", "sky_p10", "sky_p50", "sky_p90", "sky_max",
                  "sky_gt3", "sky_gt8", "sky_gt15", "sky_gt30", "sky_open", "sky_rough", "sky_censored",
                  "far_frac", "far_el", "far_max", "near_veg_frac", "built_frac",
                  "sub_L", "sub_a", "sub_b", "sub_tex", "sub_contrast"]
FEATURE_NAMES += ["skyc_blue", "skyc_L", "skyc_sat", "skyc_haze"]
for _b, _t, _l in BANDS:
    FEATURE_NAMES += ["lab_%s_%s_%s" % (_b, c, s) for c in "Lab" for s in ("mean", "std")]
FEATURE_NAMES += ["ab_%d%d" % (i, j) for i in range(3) for j in range(3)]
_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

# ----------------------------------------------------------------- colour maths
_c = np.arange(256, dtype=np.float64) / 255.0
_LIN = np.where(_c > 0.04045, ((_c + 0.055) / 1.055) ** 2.4, _c / 12.92).astype(np.float32)
_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
_M = (_M / np.array([[0.95047], [1.0], [1.08883]], np.float32)).astype(np.float32)


def _lab(rgb):
    lin = _LIN[rgb]
    xyz = lin @ _M.T
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return L, a, b


def _hsv(rgbf):
    r, g, b = rgbf[..., 0], rgbf[..., 1], rgbf[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    c = mx - mn
    cs = np.where(c > 1e-6, c, 1.0)
    h = np.where(mx == r, ((g - b) / cs) % 6.0, np.where(mx == g, (b - r) / cs + 2.0, (r - g) / cs + 4.0))
    h = np.where(c > 1e-6, h * 60.0, 0.0)
    s = np.where(mx > 1e-6, c / np.maximum(mx, 1e-6), 0.0)
    return h.astype(np.float32), s.astype(np.float32), (mx / 255.0).astype(np.float32)


def _box3(a):
    """3x3 box mean with edge padding (x wraps around: the panorama is a cylinder)."""
    p = np.concatenate([a[:, -1:], a, a[:, :1]], axis=1)
    p = np.concatenate([p[:1], p, p[-1:]], axis=0)
    s = p[:-2] + p[1:-1] + p[2:]
    return (s[:, :-2] + s[:, 1:-1] + s[:, 2:]) / 9.0


def _texture(L, valid):
    gx = np.abs(np.roll(L, -1, axis=1) - L)
    gy = np.zeros_like(L)
    gy[:-1] = np.abs(L[1:] - L[:-1])
    # do not let invalid canvas borders look like edges
    vx = valid & np.roll(valid, -1, axis=1)
    vy = np.zeros_like(valid)
    vy[:-1] = valid[1:] & valid[:-1]
    g = np.where(vx, gx, 0.0) + np.where(vy, gy, 0.0)
    return _box3(g.astype(np.float32))


def _hue_in(h, lo, hi):
    return (h >= lo) & (h <= hi)


# -------------------------------------------------------------- pixel analysis
def analyse(sph):
    """Per-pixel maps on the working canvas (also used by debug scripts)."""
    s = sph.resized(WORK_W) if sph.w != WORK_W else sph
    rgb = s.rgb
    valid = s.mask
    H, W = valid.shape
    rgbf = rgb.astype(np.float32)
    L, A, B = _lab(rgb)
    hue, sat, val = _hsv(rgbf)
    tot = rgbf.sum(-1)
    g_ch = rgbf[..., 1] / np.maximum(tot, 1.0)
    exg = np.where(tot > 15, 3.0 * g_ch - 1.0, 0.0).astype(np.float32)
    tex = _texture(L, valid)
    el = (90.0 - (np.arange(H) + 0.5) / H * 180.0).astype(np.float32)
    phi = ((np.arange(W) + 0.5) / W * 360.0 - 180.0).astype(np.float32)
    EL = el[:, None]

    # ---- sky
    blue = _hue_in(hue, 180, 262) & (sat > 0.08) & (L > 35)
    white = (sat < 0.16) & (L > 55)
    dark_cloud = (EL > 12.0) & (sat < 0.12) & (L > 40) & (tex < 4.0)
    # cloud edges are textured but bright; allow them well above the horizon
    blue_strong = _hue_in(hue, 195, 250) & (sat > 0.15) & (L > 40)
    smooth_sky = ((tex < 8.0) | ((EL > 3.0) & (tex < 25.0) & ((L > 68) | blue_strong))
                  | ((L > 90) & (sat < 0.10)))
    sky = valid & (EL > -4.0) & (((blue | white) & smooth_sky) | dark_cloud)

    # ---- classes (non-sky)
    ns = valid & ~sky
    cls = np.zeros((H, W), np.int8)  # 0 other
    smooth = tex < 7.0
    snow = ns & (L > 78) & (sat < 0.10) & (B < 6) & smooth
    water = (ns & ~snow & (EL < -1.0) & (EL > -15.0) & _hue_in(hue, 185, 240) & (sat > 0.25)
             & (tex < 2.5) & (L > 25) & (L < 75))
    green = ns & ~snow & ~water & _hue_in(hue, 62, 170) & (exg > 0.035) & (sat > 0.10) & (val > 0.05)
    rest = ns & ~snow & ~water & ~green
    redh = (hue < 38) | (hue > 340)
    autumn = rest & (EL > -5.0) & redh & (sat > 0.50) & (val > 0.25) & (tex > 6.0)
    rest2 = rest & ~autumn
    dry = rest2 & _hue_in(hue, 30, 62) & (sat > 0.16) & (sat < 0.65) & (L > 35) & (L < 88) & (tex > 3.0)
    rest3 = rest2 & ~dry
    red = rest3 & ((hue < 28) | (hue > 345)) & (sat > 0.22) & (sat < 0.85) & (L > 22) & (L < 72) & (A > 9)
    rest4 = rest3 & ~red
    sand = rest4 & _hue_in(hue, 22, 62) & (sat > 0.08) & (sat < 0.45) & (L > 58) & smooth
    rest5 = rest4 & ~sand
    brown = rest5 & _hue_in(hue, 15, 50) & (sat > 0.12) & (sat < 0.55) & (L > 18) & (L < 58)
    rest6 = rest5 & ~brown
    grey = rest6 & (sat < 0.12) & (L > 25) & (L < 80)
    black = rest6 & ~grey & (L < 22) & (sat < 0.35)
    for k, m in enumerate((snow, water, green, autumn, dry, red, sand, brown, grey, black), start=1):
        cls[m] = k
    return {
        "s": s, "valid": valid, "L": L, "A": A, "B": B, "hue": hue, "sat": sat, "val": val, "exg": exg,
        "tex": tex, "el": el, "phi": phi, "sky": sky, "cls": cls,
    }


CLASS_NAMES = ("other", "snow", "water", "green", "autumn", "dry", "red", "sand", "brown", "grey", "black")
_CI = {n: i for i, n in enumerate(CLASS_NAMES)}


# ------------------------------------------------------------------- skyline
def skyline(an):
    sky, valid, el = an["sky"], an["valid"], an["el"]
    H, W = sky.shape
    nonsky = valid & ~sky
    run = nonsky[:-2] & nonsky[1:-1] & nonsky[2:]
    has = run.any(0)
    first = np.argmax(run, axis=0)
    cols = np.arange(W)
    ok = has & valid[first, cols]
    # censored: no valid sky pixel above the first non-sky run
    sky_above = np.cumsum(sky, axis=0)
    above = np.where(first > 0, sky_above[np.maximum(first - 1, 0), cols], 0)
    censored = ok & (above == 0)
    s = np.where(ok, el[first], np.nan).astype(np.float32)
    s = np.minimum(s, 60.0)
    # 5-column circular median (poles, wires)
    stack = np.stack([np.roll(s, k) for k in (-2, -1, 0, 1, 2)])
    with np.errstate(all="ignore"):
        sm = np.where(ok, np.nanmedian(stack, axis=0), np.nan)
    return sm, first, ok, censored


def _road_axis(an):
    """Relative longitude of the road axis estimated from asphalt-likeness."""
    valid, el, sat, tex, L = an["valid"], an["el"], an["sat"], an["tex"], an["L"]
    rows = (el < -15) & (el > -40)
    m = valid[rows]
    asph = (sat[rows] < 0.15) & (tex[rows] < 6) & (L[rows] > 20) & (L[rows] < 75) & m
    num = asph.sum(0).astype(np.float64)
    den = m.sum(0).astype(np.float64)
    W = valid.shape[1]
    half = W // 18  # +-10 deg window
    k = np.ones(2 * half + 1)
    numc = np.convolve(np.concatenate([num[-half:], num, num[:half]]), k, "valid")
    denc = np.convolve(np.concatenate([den[-half:], den, den[:half]]), k, "valid")
    score = numc / np.maximum(denc, 1.0) * (denc > 0)
    both = score + np.roll(score, W // 2)
    if not np.any(denc > 0):
        return None, 0.0
    i = int(np.argmax(both))
    return float(an["phi"][i]), float(both[i] / 2.0)


def side_mask(an, axis, top=-5.0, bottom=-35.0):
    el, phi = an["el"], an["phi"]
    EL = el[:, None]
    half = np.degrees(np.arctan(2.2 * np.tan(np.radians(np.clip(-EL, 0, 89)))))
    half = np.maximum(half, 12.0)
    d = np.abs((phi[None, :] - axis + 180.0) % 360.0 - 180.0)
    road = (d < half) | (d > 180.0 - half)
    band = (EL <= top) & (EL >= bottom)
    return band & ~road


# ------------------------------------------------------------------- extract
def _frac(m, base):
    n = base.sum()
    return float((m & base).sum()) / n if n > 0 else np.nan


def _mean(v, m, min_n=1):
    n = m.sum()
    return float(v[m].mean()) if n >= min_n else np.nan


def extract(sph):
    an = analyse(sph)
    valid, el, cls, sky = an["valid"], an["el"], an["cls"], an["sky"]
    L, A, B, hue, sat, exg, tex = an["L"], an["A"], an["B"], an["hue"], an["sat"], an["exg"], an["tex"]
    H, W = valid.shape
    EL = el[:, None]
    x = np.full(len(FEATURE_NAMES), np.nan, np.float32)
    ev = {}

    def put(name, v):
        x[_IDX[name]] = v

    is_ = {n: cls == i for n, i in _CI.items()}
    ns = valid & ~sky

    # road axis / ground side region
    if sph.car_heading is not None:
        axis, axis_conf = float(sph.car_heading), 1.0
    else:
        axis, axis_conf = _road_axis(an)
    if axis is None:
        side = (EL <= -5.0) & (EL >= -35.0) & valid
    else:
        side = side_mask(an, axis) & valid

    band_masks = {}
    for name, top, bot in BANDS:
        if name == "ground":
            bm = side
        else:
            bm = (EL <= top) & (EL >= bot) & valid
        band_masks[name] = bm
        enough = bm.sum() >= 150
        if enough:
            put("veg_%s_green" % name, _frac(is_["green"], bm))
            put("veg_%s_dark" % name, _frac(is_["green"] & (L < 32) & (hue > 85), bm))
            put("veg_%s_light" % name, _frac(is_["green"] & (L >= 45), bm))
            put("veg_%s_dry" % name, _frac(is_["dry"], bm))
            put("veg_%s_exg" % name, _mean(exg, bm & ns, 30) if (bm & ns).sum() >= 30 else 0.0)

    upper = (band_masks["canopy"] | band_masks["horizon"])
    if upper.sum() >= 150:
        put("veg_autumn", _frac(is_["autumn"], upper))
    gmask = is_["green"] & (band_masks["canopy"] | band_masks["horizon"] | band_masks["ground"])
    if gmask.sum() >= 40:
        put("veg_hue", _mean(hue, gmask))
        put("veg_sat", _mean(sat, gmask))
        put("veg_L", _mean(L, gmask))

    # soil on the ground side
    if side.sum() >= 150:
        for c in SOIL_CLASSES:
            put("soil_%s" % c, _frac(is_[c], side))
        bare = side & ~is_["green"] & ~is_["snow"] & ~is_["water"]
        if bare.sum() >= 40:
            put("soil_a", _mean(A, bare))
            put("soil_b", _mean(B, bare))
            put("soil_L", _mean(L, bare))
            put("soil_redness", _mean(np.clip(A - 0.35 * B, 0, 40), bare))
        put("snow_ground", _frac(is_["snow"], side))
    hz = band_masks["horizon"] & ns
    if hz.sum() >= 100:
        put("snow_horizon", _frac(is_["snow"], hz))
    wb = (EL < -1.0) & (EL > -15.0) & valid
    if axis is not None:
        wb &= side_mask(an, axis, top=-1.0, bottom=-15.0)
    if wb.sum() >= 150:
        put("water_frac", _frac(is_["water"], wb))

    # skyline
    sl, first, ok, censored = skyline(an)
    n_ok = int(ok.sum())
    if n_ok >= 16:
        v = sl[ok]
        put("sky_mean", float(v.mean()))
        put("sky_std", float(v.std()))
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        put("sky_p10", p10)
        put("sky_p50", p50)
        put("sky_p90", p90)
        put("sky_max", float(v.max()))
        for t in (3, 8, 15, 30):
            put("sky_gt%d" % t, float((v > t).mean()))
        put("sky_open", float((v < 1.5).mean()))
        nb = ok & np.roll(ok, -1)
        raised = nb & ((sl > 1.5) | (np.roll(sl, -1) > 1.5))
        d = np.abs(np.roll(sl, -1) - sl)
        put("sky_rough", float(d[raised].mean()) if raised.sum() >= 4 else 0.0)
        put("sky_censored", float(censored[ok].mean()))

        # band just below the skyline: rows first+1 .. first+5 (0.7-3.5 deg)
        cols = np.arange(W)
        rr = np.clip(first[None, :] + np.arange(1, 6)[:, None], 0, H - 1)
        vm = valid[rr, cols]
        def colmean(arr):
            a = arr[rr, cols]
            return (a * vm).sum(0) / np.maximum(vm.sum(0), 1)
        subL, subA, subB = colmean(L), colmean(A), colmean(B)
        subT, subG = colmean(tex), colmean(exg)
        subS = colmean(sat)
        ra = np.clip(first[None, :] - np.arange(2, 6)[:, None], 0, H - 1)
        sm_ = sky[ra, cols]
        skyL = (L[ra, cols] * sm_).sum(0) / np.maximum(sm_.sum(0), 1)
        has_sky = sm_.sum(0) > 0
        contrast = np.where(has_sky, skyL - subL, np.nan)
        raised = ok & (sl > 1.5) & ~censored & (vm.sum(0) >= 3)
        far = raised & has_sky & (subT < 5.5) & (subG < 0.06) & (contrast < 32) & (subS < 0.35)
        vegc = raised & (subG >= 0.06)
        built = raised & ~far & ~vegc & (subS < 0.15) & (subT >= 5.5)
        put("far_frac", float(far[ok].mean()))
        put("far_el", float(sl[far].mean()) if far.sum() >= 3 else 0.0)
        put("far_max", float(sl[far].max()) if far.sum() >= 3 else 0.0)
        put("near_veg_frac", float(vegc[ok].mean()))
        put("built_frac", float(built[ok].mean()))
        if raised.sum() >= 8:
            put("sub_L", float(subL[raised].mean()))
            put("sub_a", float(subA[raised].mean()))
            put("sub_b", float(subB[raised].mean()))
            put("sub_tex", float(subT[raised].mean()))
            cr = raised & has_sky
            put("sub_contrast", float(contrast[cr].mean()) if cr.sum() >= 4 else np.nan)
        ev_terrain = {
            "skyline_mean_deg": round(float(v.mean()), 1),
            "skyline_p90_deg": round(float(p90), 1),
            "open_horizon": round(float((v < 1.5).mean()), 2),
            "distant_relief_frac": round(float(far[ok].mean()), 2),
            "distant_relief_max_deg": round(float(sl[far].max()), 1) if far.sum() >= 3 else 0.0,
        }
        ev["terrain"] = ev_terrain

    # sky colour
    skym = sky & (EL > 5.0) & (EL < 60.0)
    if skym.sum() >= 150:
        put("skyc_blue", _frac(_hue_in(hue, 185, 255) & (sat > 0.15), skym))
        put("skyc_L", _mean(L, skym))
        put("skyc_sat", _mean(sat, skym))
        lo = sky & (EL > 2.0) & (EL < 8.0)
        hi = sky & (EL > 35.0) & (EL < 60.0)
        if lo.sum() >= 30 and hi.sum() >= 30:
            put("skyc_haze", _mean(L, lo) - _mean(L, hi))

    # Lab stats per band (non-sky)
    for name, _t, _b in BANDS:
        m = band_masks[name] & ns
        if m.sum() >= 100 and m.sum() >= 0.02 * max(band_masks[name].sum(), 1):
            for c, arr in (("L", L), ("a", A), ("b", B)):
                vv = arr[m]
                put("lab_%s_%s_mean" % (name, c), float(vv.mean()))
                put("lab_%s_%s_std" % (name, c), float(vv.std()))

    # a*/b* histogram (horizon + ground side, non-sky)
    m = (band_masks["horizon"] | band_masks["ground"]) & ns
    if m.sum() >= 150:
        ai = np.digitize(A[m], [-4.0, 6.0])
        bi = np.digitize(B[m], [8.0, 22.0])
        hist = np.bincount(ai * 3 + bi, minlength=9).astype(np.float32) / m.sum()
        for i in range(3):
            for j in range(3):
                put("ab_%d%d" % (i, j), hist[i * 3 + j])

    ev.update(_evidence(x))
    return {"x": x, "evidence": ev}


def _g(x, n):
    v = x[_IDX[n]]
    return None if np.isnan(v) else float(v)


def _evidence(x):
    ev = {}
    return ev
