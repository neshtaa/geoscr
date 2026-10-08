"""
Geographic helpers: lat/lng -> country lookup (rasterised Natural Earth),
great-circle distance, the GeoGuessr scoring curve and map metadata (score scale, bounds).
"""

import json
import os

import numpy as np
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EARTH_R = 6371.0088
# GeoGuessr: score = round(5000 * exp(-10 * d / D)), D = the map's maxErrorDistance (metres);
# D = 14916.862 km for the official World map
WORLD_MAX_ERROR_M = 14916862
WORLD_SCORE_SCALE_KM = WORLD_MAX_ERROR_M / 10000.0

_RASTER = None


def _raster():
    global _RASTER
    if _RASTER is None:
        meta = json.load(open(os.path.join(DATA_DIR, "world_countries.json")))
        grid = np.array(Image.open(os.path.join(DATA_DIR, "world_countries.png")))
        _RASTER = (grid, meta["codes"], meta["resolution_deg"])
    return _RASTER


def country_at(lat, lng, search_px=3):
    """ISO-3166 alpha-2 code at lat/lng (None over the ocean)."""
    grid, codes, res = _raster()
    h, w = grid.shape
    y = min(h - 1, max(0, int((90.0 - lat) / res)))
    x = int((lng + 180.0) / res) % w
    v = grid[y, x]
    if v == 0 and search_px:
        y0, y1 = max(0, y - search_px), min(h, y + search_px + 1)
        xs = [(x + d) % w for d in range(-search_px, search_px + 1)]
        win = grid[y0:y1][:, xs]
        nz = np.argwhere(win > 0)
        if len(nz):
            cy, cx = y - y0, search_px
            k = np.argmin((nz[:, 0] - cy) ** 2 + (nz[:, 1] - cx) ** 2)
            v = win[nz[k][0], nz[k][1]]
    return codes[v] if v else None


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance; accepts scalars or numpy arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lng2) - np.asarray(lng1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def positive_float(v):
    """float(v) for a finite positive number (booleans excluded), else None."""
    if isinstance(v, bool):
        return None
    try:
        d = float(v)
    except (TypeError, ValueError):
        return None
    return d if d > 0 and np.isfinite(d) else None


def score_scale_km(max_error_distance_m=None):
    """Distance (km) over which the score falls by a factor e: maxErrorDistance / 10 (World map
    when the value is missing or invalid)."""
    d = positive_float(max_error_distance_m)
    return d / 10000.0 if d else WORLD_SCORE_SCALE_KM


def geoguessr_score(distance_km, scale_km=None):
    """Continuous score; scale_km = score_scale_km(map maxErrorDistance), World map if None."""
    return 5000.0 * np.exp(-np.asarray(distance_km) / (scale_km or WORLD_SCORE_SCALE_KM))


def geoguessr_points(distance_km, max_error_distance_m=None):
    """Points as the game client computes them (Math.round, 5000 within 25 m); exact when the map's
    maxErrorDistance is given.  Without it the official World map is assumed (the client itself
    falls back to 20,037,580 m, but here a missing value means an unknown map)."""
    d = np.asarray(distance_km, float)
    return np.where(d <= 0.025, 5000.0, np.floor(geoguessr_score(d, score_scale_km(max_error_distance_m)) + 0.5))


_REGIONS = None


def _regions():
    global _REGIONS
    if _REGIONS is None:
        z = np.load(os.path.join(DATA_DIR, "world_regions.npz"))
        _REGIONS = (z["grid"], [str(c) for c in z["codes"]], [str(n) for n in z["names"]],
                    [str(c) for c in z["countries"]], float(z["resolution_deg"]))
    return _REGIONS


def region_index(lat, lng, search_px=3):
    """Index into region_info() of the admin-1 region at lat/lng (0 = none). Vectorised."""
    grid, _, _, _, res = _regions()
    h, w = grid.shape
    lat, lng = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lng, float))
    y = np.clip(((90.0 - lat) / res).astype(int), 0, h - 1)
    x = ((lng + 180.0) / res).astype(int) % w
    v = grid[y, x].astype(int)
    for k in np.flatnonzero(v == 0):
        if not search_px:
            break
        y0, y1 = max(0, y[k] - search_px), min(h, y[k] + search_px + 1)
        xs = [(x[k] + d) % w for d in range(-search_px, search_px + 1)]
        win = grid[y0:y1][:, xs]
        nz = np.argwhere(win > 0)
        if len(nz):
            j = np.argmin((nz[:, 0] - (y[k] - y0)) ** 2 + (nz[:, 1] - search_px) ** 2)
            v[k] = win[nz[j][0], nz[j][1]]
    return v


def region_info(i):
    """(ISO 3166-2 code, English name, country code) of a region index."""
    _, codes, names, countries, _ = _regions()
    return codes[i], names[i], countries[i]


def region_at(lat, lng):
    i = int(region_index(lat, lng)[0])
    return region_info(i) if i else None


# ------------------------------------------------------------------ maps
_AREA = {}


def parse_bounds(b):
    """GeoGuessr bounds {"min": {"lat", "lng"}, "max": {"lat", "lng"}} (or [[lat, lng], [lat, lng]])
    -> (lat_min, lng_min, lat_max, lng_max), None if missing or invalid."""
    if not b:
        return None
    try:
        if isinstance(b, dict):
            v = (b["min"]["lat"], b["min"]["lng"], b["max"]["lat"], b["max"]["lng"])
        elif len(b) == 4:
            v = tuple(b)
        else:
            v = (b[0][0], b[0][1], b[1][0], b[1][1])
        v = tuple(float(x) for x in v)
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if not all(np.isfinite(v)) or v[0] > v[2] or v[1] > v[3]:
        return None
    return (max(v[0], -90.0), max(v[1], -180.0), min(v[2], 90.0), min(v[3], 180.0))


def bounds_dict(bounds):
    b = parse_bounds(bounds)
    return None if b is None else {"min": {"lat": b[0], "lng": b[1]}, "max": {"lat": b[2], "lng": b[3]}}


def in_bounds(lat, lng, bounds, margin_deg=0.0):
    """Vectorised: points inside the box widened by margin_deg (all True without bounds)."""
    b = parse_bounds(bounds)
    lat, lng = np.asarray(lat, float), np.asarray(lng, float)
    if b is None:
        return np.ones(np.broadcast(lat, lng).shape, bool)
    m = margin_deg
    return (lat >= b[0] - m) & (lat <= b[2] + m) & (lng >= b[1] - m) & (lng <= b[3] + m)


def clip_to_bounds(lat, lng, bounds):
    b = parse_bounds(bounds)
    if b is None:
        return float(lat), float(lng)
    return float(np.clip(lat, b[0], b[2])), float(np.clip(lng, b[1], b[3]))


def country_area_inside(bounds):
    """{country code: share of its land area (cos-latitude weighted raster) inside the bounds}."""
    b = parse_bounds(bounds)
    if b is None:
        return {}
    if b in _AREA:
        return _AREA[b]
    grid, codes, res = _raster()
    h, w = grid.shape
    coslat = np.cos(np.radians(90.0 - (np.arange(h) + 0.5) * res))
    if None not in _AREA:
        tot = np.zeros(len(codes))
        for y in range(h):
            tot += coslat[y] * np.bincount(grid[y], minlength=len(codes))[: len(codes)]
        _AREA[None] = tot
    y0, y1 = int(np.clip((90.0 - b[2]) / res, 0, h - 1)), int(np.clip((90.0 - b[0]) / res, 0, h - 1))
    x0, x1 = int(np.clip((b[1] + 180.0) / res, 0, w - 1)), int(np.clip((b[3] + 180.0) / res, 0, w - 1))
    ins = np.zeros(len(codes))
    for y in range(y0, y1 + 1):
        ins += coslat[y] * np.bincount(grid[y, x0:x1 + 1], minlength=len(codes))[: len(codes)]
    frac = ins / np.maximum(_AREA[None], 1e-12)
    _AREA[b] = {codes[i]: float(frac[i]) for i in range(1, len(codes)) if codes[i] and _AREA[None][i] > 0}
    return _AREA[b]


def is_world_map(info):
    """World-type map: bounds spanning most of the globe, or (without bounds) 'world' in the name."""
    b = parse_bounds((info or {}).get("bounds"))
    if b is not None:
        return (b[2] - b[0]) >= 60.0 and (b[3] - b[1]) >= 180.0
    return "world" in str((info or {}).get("name") or "").lower()


_MAPS = None


def load_maps():
    """data/maps.json (tools/fetch_maps.py): [{id, slug, name, maxErrorDistance, bounds, ...}]."""
    global _MAPS
    if _MAPS is None:
        path = os.path.join(DATA_DIR, "maps.json")
        _MAPS = json.load(open(path, encoding="utf-8")).get("maps", []) if os.path.exists(path) else []
    return _MAPS


def find_map(key):
    """Registry entry by id, slug, name or former name (case-insensitive), None if unknown."""
    k = str(key or "").strip().lower()
    if not k:
        return None
    for field in ("id", "slug", "name", "aliases"):
        for m in load_maps():
            v = m.get(field) or ""
            if k in [str(x).strip().lower() for x in (v if isinstance(v, list) else [v])]:
                return m
    return None


def resolve_map(info):
    """Map description used by the locator.  info: None, an id / slug / name string, or a dict with
    any of id, slug, name, bounds, maxErrorDistance.  A given id or slug is authoritative (the name
    is only used without them); the caller's bounds and maxErrorDistance (in-game values) win over
    data/maps.json; invalid values fall back to the registry.  Returns None or
    {id, slug, name, bounds, maxErrorDistance, world, known, updatedAt}."""
    if info is None or info == "" or info == {}:
        return None
    if isinstance(info, dict):
        keys = [info.get("id"), info.get("slug")] if info.get("id") or info.get("slug") else [info.get("name")]
        reg = next((m for m in map(find_map, keys) if m), None)
    else:
        reg = find_map(info)
        info = {} if reg else {"name": str(info)}
    reg = reg or {}
    out = {}
    for k in ("id", "slug", "name"):
        out[k] = reg.get(k) or info.get(k)
    out["bounds"] = bounds_dict(info.get("bounds")) or bounds_dict(reg.get("bounds"))
    out["maxErrorDistance"] = positive_float(info.get("maxErrorDistance")) or positive_float(reg.get("maxErrorDistance"))
    out["world"] = is_world_map(out)
    out["known"] = bool(reg)
    out["updatedAt"] = reg.get("updatedAt") or info.get("updatedAt")
    return out
