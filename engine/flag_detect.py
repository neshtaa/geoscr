"""
National-flag detector (closed form): small, saturated, roughly rectangular colour patches above the ground
whose stripe layout matches a flag design.

  1. the elevation band EL_TOP..EL_BOT of the sphere (flags hang on poles, houses, gates) in CIE L*a*b*;
  2. soft colour memberships per pixel (absolute L*a*b* ramps; hazy Street View colours are washed out, so
     the ramps start low):
       red    a* high, hue angle -19 .. 50 deg (not orange, not purple)
       blue   b* low, not purple, darker than a bright sky (L* < 72)
       yellow b* high, a* moderate
       white  L* high, low chroma
     averaged over CELL x CELL pixel cells (masked pixels count as no colour);
  3. windows of WIN_H cells high with the aspect of a landscape flag (3:2) or of a hanging vertical banner
     (2:5), stride a sixth of the window side (x wraps); every template is a stripe layout along
     one axis, e.g. Cambodia blue | red (x2) | blue, Thailand red | white | blue (x2) | white | red, Vietnam
     red | yellow | red; its score in a window is the mean over its stripes of the membership of the
     expected colour minus the strongest other flag colour, minus RING_W x the expected colours' share of the
     ring around the window (a flag stands out from its surroundings: the inside of a red roof or of a blue
     sky scores low, a corner of it at most 1 - RING_W / 2);
  4. per template the maximum over windows and sizes (and its direction) -> FEATURE_NAMES; score_maps()
     gives the per-cell maps (max over the windows centred on a cell) for the placement study.

The feature vector is not used by the country model directly: tools/flag_spike.py measures detection at
GeoGuessr's "flags" card placements, fits class-conditional densities of the template maxima on training
panoramas and calibrates a country evidence source ("flags") on CALIB.
"""

import numpy as np

from .features.texture import _lab

NAME = "flags"
EL_TOP, EL_BOT = 40.0, -6.0
WIDTH = 2048           # the sphere is resampled to this width first (the live sphere is 2048 wide)
CELL = 2               # pixels per cell side (0.35 deg at 2048 px)
WIN_H = (3, 4, 5, 6, 8, 10, 12, 15, 19, 24)   # window heights in cells (~25% steps)
RING_W = 1.0           # weight of the ring term
ASPECTS = {"h": 1.5, "v": 0.4}  # landscape flag, hanging vertical banner (width / height)
COLOURS = ("R", "B", "Y", "W")

# template: (name, axis of the stripes: "y" = horizontal stripes top -> bottom, "x" = vertical stripes
# left -> right, ((colour, relative width), ...), window shapes)
TEMPLATES = [
    ("KH", "y", (("B", 1), ("R", 2), ("B", 1)), ("h",)),
    ("KH_banner", "x", (("B", 1), ("R", 2), ("B", 1)), ("v",)),
    ("TH", "y", (("R", 1), ("W", 1), ("B", 2), ("W", 1), ("R", 1)), ("h",)),
    ("TH_banner", "x", (("R", 1), ("W", 1), ("B", 2), ("W", 1), ("R", 1)), ("v",)),
    ("RED_YELLOW_RED", "y", (("R", 1), ("Y", 1), ("R", 1)), ("h",)),      # VN (star), ES
    ("RED_YELLOW_RED_x", "x", (("R", 1), ("Y", 1), ("R", 1)), ("h", "v")),
    ("WHITE_BLUE_RED", "y", (("W", 1), ("B", 1), ("R", 1)), ("h",)),      # RU, SI, SK, RS
    ("RED_WHITE_BLUE", "y", (("R", 1), ("W", 1), ("B", 1)), ("h",)),      # NL, LU, HR, PY
    ("BLUE_WHITE_RED_x", "x", (("B", 1), ("W", 1), ("R", 1)), ("h",)),    # FR
    ("RED_WHITE", "y", (("R", 1), ("W", 1)), ("h",)),                     # ID, MC
    ("WHITE_RED", "y", (("W", 1), ("R", 1)), ("h",)),                     # PL
    ("RED_WHITE_RED_x", "x", (("R", 1), ("W", 2), ("R", 1)), ("h",)),     # CA, PE
    ("RED_WHITE_RED", "y", (("R", 1), ("W", 1), ("R", 1)), ("h",)),       # AT, LV
    ("YELLOW_BLUE_RED", "y", (("Y", 2), ("B", 1), ("R", 1)), ("h",)),     # CO, EC
    ("BLUE_YELLOW", "y", (("B", 1), ("Y", 1)), ("h",)),                   # UA
    ("BLUE_RED", "y", (("B", 1), ("R", 1)), ("h",)),                      # canton flags (US, MY, AU) top
    ("RED_FIELD", "y", (("R", 1),), ("h", "v")),                          # TR, VN, CN, party banners
]
TEMPLATE_NAMES = [t[0] for t in TEMPLATES]
FEATURE_NAMES = ["flag_" + n for n in TEMPLATE_NAMES] + ["flag_red_cells", "flag_blue_cells", "flag_yellow_cells"]


def _ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _box_mean(A, k):
    """Mean over a k x k box (x wraps, y clamped) of a 2-D array."""
    r = k // 2
    P = np.pad(A, ((r, r), (0, 0)), mode="edge")
    P = np.concatenate([P[:, -r:], P, P[:, :r]], axis=1)
    I = np.zeros((P.shape[0] + 1, P.shape[1] + 1))
    I[1:, 1:] = P.cumsum(0).cumsum(1)
    h, w = A.shape
    return (I[k:k + h, k:k + w] - I[:h, k:k + w] - I[k:k + h, :w] + I[:h, :w]) / float(k * k)


def memberships(rgb, mask=None):
    """(H, W, 4) soft memberships R, B, Y, W of an RGB band (uint8)."""
    L, a, b = _lab(np.ascontiguousarray(rgb))
    L, a, b = L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)
    chroma = np.hypot(a, b)
    red = _ramp(a, 16.0, 30.0) * (b < 1.2 * a) * (b > -0.35 * a)        # hue angle -19 .. 50 deg
    blue = _ramp(-b, 16.0, 28.0) * _ramp(72.0 - L, 0.0, 10.0) * (-b > 0.9 * a)
    yellow = _ramp(b, 32.0, 50.0) * (a < 25.0)
    white = _ramp(L, 76.0, 88.0) * _ramp(14.0 - chroma, 0.0, 6.0)
    M = np.stack([red, blue, yellow, white], -1)
    if mask is not None:
        M *= mask[..., None]
    return M


def _cells(M):
    h, w = M.shape[0] // CELL * CELL, M.shape[1] // CELL * CELL
    return M[:h, :w].reshape(h // CELL, CELL, w // CELL, CELL, M.shape[2]).mean((1, 3))


def _integral(G, pad):
    P = np.concatenate([G, G[:, :pad]], axis=1)
    I = np.zeros((P.shape[0] + 1, P.shape[1] + 1, P.shape[2]), np.float32)
    I[1:, 1:] = P.cumsum(0, dtype=np.float64).cumsum(1)
    return I


def _boxes(I, ys, h, xs, w):
    """(ny, nx, k) box means, rows ys..ys+h-1, cols xs..xs+w-1 (ys, xs: evenly spaced 1-D start arrays; basic
    slicing with steps, no gathers)."""
    sy = int(ys[1] - ys[0]) if len(ys) > 1 else 1
    sx = int(xs[1] - xs[0]) if len(xs) > 1 else 1
    y0, x0, ny, nx = int(ys[0]), int(xs[0]), len(ys), len(xs)
    r0, r1 = slice(y0, y0 + sy * (ny - 1) + 1, sy), slice(y0 + h, y0 + h + sy * (ny - 1) + 1, sy)
    c0, c1 = slice(x0, x0 + sx * (nx - 1) + 1, sx), slice(x0 + w, x0 + w + sx * (nx - 1) + 1, sx)
    return (I[r1, c1] - I[r0, c1] - I[r1, c0] + I[r0, c0]) * (1.0 / float(h * w))


def _band(sph):
    s = sph.resized(WIDTH) if sph.w != WIDTH else sph
    r0, r1 = s.row_of(EL_TOP), s.row_of(EL_BOT)
    return s, r0, r1


def score_maps(sph):
    """(maps (T, Hc, Wc): per template the best score of the windows centred on each cell (-inf where none),
    cell grid geometry {r0, deg, w, h}, cell memberships (Hc, Wc, 4))."""
    s, r0, r1 = _band(sph)
    mask = s.mask[r0:r1].astype(np.float64)
    G = _cells(memberships(s.rgb[r0:r1], mask)).astype(np.float32)
    vis = _cells(mask[..., None]).astype(np.float32)
    Hc, Wc = G.shape[:2]
    pad = int(max(WIN_H) * max(ASPECTS.values())) + 2 * max(WIN_H) + 4
    I = _integral(G, pad)
    V = _integral(vis, pad)
    maps = np.full((len(TEMPLATES), Hc, Wc), -np.inf, np.float32)
    ci = {c: i for i, c in enumerate(COLOURS)}
    for hc in WIN_H:
        ring = max(1, hc // 3)
        for shp, asp in ASPECTS.items():
            tpl = [(ti, t) for ti, t in enumerate(TEMPLATES) if shp in t[3]]
            if not tpl:
                continue
            wc = max(2, int(round(hc * asp)))
            # stride: about a sixth of the window side (one cell for small windows)
            ys = np.arange(ring, Hc - hc - ring + 1, max(1, hc // 6))
            if not len(ys):
                continue
            xs = np.arange(ring, Wc + ring, max(1, wc // 6))
            # frame around the window, visibility
            big = _boxes(I, ys - ring, hc + 2 * ring, xs - ring, wc + 2 * ring)
            inner = _boxes(I, ys, hc, xs, wc)
            a_big, a_in = (hc + 2 * ring) * (wc + 2 * ring), hc * wc
            ring_m = (big * a_big - inner * a_in) / float(a_big - a_in)
            ok = _boxes(V, ys - ring, hc + 2 * ring, xs - ring, wc + 2 * ring)[..., 0] > 0.95
            cy = ys + hc // 2
            cx = (xs + wc // 2) % Wc
            stripe_cache = {}
            for ti, (name, axis, stripes, _) in tpl:
                n = hc if axis == "y" else wc
                if n < len(stripes):
                    continue
                tot = float(sum(wt for _, wt in stripes))
                edges = tuple(np.round(np.cumsum([0.0] + [wt / tot * n for _, wt in stripes])).astype(int))
                score = np.zeros((len(ys), len(xs)), np.float32)
                expected = []
                for (col, _), e0, e1 in zip(stripes, edges[:-1], edges[1:]):
                    e1 = max(e1, e0 + 1)
                    key = (axis, e0, e1)
                    if key not in stripe_cache:
                        stripe_cache[key] = (_boxes(I, ys + e0, e1 - e0, xs, wc) if axis == "y"
                                             else _boxes(I, ys, hc, xs + e0, e1 - e0))
                    j = ci[col]
                    if key + (j,) not in stripe_cache:
                        m = stripe_cache[key]
                        o = [k for k in range(len(COLOURS)) if k != j]
                        stripe_cache[key + (j,)] = m[..., j] - np.maximum(np.maximum(m[..., o[0]], m[..., o[1]]),
                                                                           m[..., o[2]])
                    score += stripe_cache[key + (j,)] * (1.0 / len(stripes))
                    expected.append(j)
                ex = sorted(set(expected))
                rmax = ring_m[..., ex[0]]
                for j in ex[1:]:
                    rmax = np.maximum(rmax, ring_m[..., j])
                score -= RING_W * rmax
                score = np.where(ok, score, -np.inf)
                cur = maps[ti][np.ix_(cy, cx)]
                maps[ti][np.ix_(cy, cx)] = np.maximum(cur, score)
    geo = {"r0": r0, "deg": 360.0 / s.w * CELL, "w": s.w, "h": s.h}
    return maps, geo, G


def cell_direction(geo, yy, xx):
    """(relative azimuth, elevation) of the centre of cell (yy, xx)."""
    el = 90.0 - (geo["r0"] + (yy + 0.5) * CELL) / geo["h"] * 180.0
    az = ((xx + 0.5) * CELL / geo["w"] * 360.0 - 180.0 + 180.0) % 360.0 - 180.0
    return az, el


def features_from_maps(maps, G):
    m = maps.reshape(len(maps), -1).max(1)
    x = np.where(np.isfinite(m), m, np.nan).astype(np.float64)
    return np.concatenate([x, [float((G[..., k] > 0.5).mean()) for k in range(3)]])


def scan(sph, return_where=False):
    """Template maxima (FEATURE_NAMES order); with return_where also {template: (score, az_rel, el)}."""
    maps, geo, G = score_maps(sph)
    x = features_from_maps(maps, G)
    if not return_where:
        return x
    where = {}
    for ti, name in enumerate(TEMPLATE_NAMES):
        k = int(np.argmax(maps[ti]))
        if np.isfinite(maps[ti].flat[k]):
            yy, xx = divmod(k, maps.shape[2])
            az, el = cell_direction(geo, yy, xx)
            where[name] = (float(maps[ti].flat[k]), az, el)
    return x, where


def extract(sph):
    x, where = scan(sph, return_where=True)
    best = max(where.items(), key=lambda kv: kv[1][0]) if where else None
    ev = {}
    if best is not None:
        ev = {"best_template": best[0], "score": round(best[1][0], 3), "azimuth_rel": round(best[1][1], 1),
              "elevation": round(best[1][2], 1)}
    return {"x": x, "evidence": ev}
