"""
Geographic helpers: lat/lng -> country lookup (rasterised Natural Earth),
great-circle distance and the GeoGuessr World-map scoring curve.
"""

import json
import os

import numpy as np
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EARTH_R = 6371.0088
# GeoGuessr: score = 5000 * exp(-10 * d / D) with D = 14916.862 km for the World map
WORLD_SCORE_SCALE_KM = 1491.6862

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


def geoguessr_score(distance_km):
    return 5000.0 * np.exp(-np.asarray(distance_km) / WORLD_SCORE_SCALE_KM)
