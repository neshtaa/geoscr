"""
GeoGuessr-style areas: admin-1 regions of the region raster grouped per country the way players name parts of a
country (US census divisions, Brazilian macro-regions, Russian federal districts, Indian zones, Indonesian
islands, Slovenian statistical regions, ...).

data/areas.json (written by tools/make_areas.py, sources per country) lists the areas of the countries where the
grouping matters; every raster unit of a listed country belongs to exactly one area or to the country's "hidden"
list (units the raster files under a country they do not belong to, e.g. Crimea under Russia: in no area and no
region list of that country).  A country that is not listed and has at least AUTO_MIN_UNITS (17) units gets
automatic compass areas: its units are clustered by position (deterministic weighted k-means on the unit centroids,
weights sqrt(km2); the largest k <= AUTO_MAX_AREAS whose clusters get distinct compass names) and each cluster is
named after its direction from the country's centre ("Північ", "Південний схід", "Центр", ...; only when no k gives
distinct names does a repeated direction get the name of its largest unit in brackets).  Units without a raster
pixel join the automatic area whose centre is nearest to their label point (data/areas.json unit_points), or none.
Other countries have no areas (each unit is its own region, nothing to group).

  country_areas(cc)       [{id, name, en, regions: [codes]}] or None
  hidden_units(cc)        raster units filed under cc that are not shown as part of it
  area_labels(cc, regs)   (len(regs),) area index of raster region indices (-1: unit outside every area)
  group_regions(probs, cc, top_regions=3)
                          {code: P(region | cc)} -> [{id, name, en, probability, size, regions: [{code, name,
                          probability}] (the area's most probable units)}], most probable area first (size: the
                          area's number of units); None when the country has no areas
  region_name(code)       display name of a raster unit (data/areas.json region_names, else the raster's name)
"""

import json
import math
import os

import numpy as np

from .geo import _regions

AREAS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "areas.json")
AUTO_MIN_UNITS = 17   # countries with at least this many units and no curated areas get compass areas
AUTO_MAX_AREAS = 6
CENTRE_FRACTION = 0.35  # an automatic area this close to the centre (x the country's RMS unit distance) is "Центр"
DIRECTIONS = [("Північ", "North"), ("Північний схід", "Northeast"), ("Схід", "East"), ("Південний схід", "Southeast"),
              ("Південь", "South"), ("Південний захід", "Southwest"), ("Захід", "West"),
              ("Північний захід", "Northwest")]

_DATA = None
_CACHE = {}


def load(path=AREAS_FILE):
    """The parsed data/areas.json ({} when missing or unreadable), cached."""
    global _DATA
    if _DATA is None or path != AREAS_FILE:
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = {}
        if path != AREAS_FILE:
            return d
        _DATA = d
    return _DATA


def region_name(code):
    """Display name of raster unit code (the curated override, else the raster's name, else the code)."""
    names = load().get("region_names") or {}
    if code in names:
        return names[code]
    _, codes, rnames, _, _ = _regions()
    pos = _code_pos()
    i = pos.get(code)
    return rnames[i] if i is not None else code


def _code_pos():
    if "pos" not in _CACHE:
        _CACHE["pos"] = {c: i for i, c in enumerate(_regions()[1])}
    return _CACHE["pos"]


def units_of(cc):
    """Raster region indices filed under country cc."""
    if "units" not in _CACHE:
        _, codes, _, rcc, _ = _regions()
        by = {}
        for i in range(1, len(codes)):
            by.setdefault(rcc[i], []).append(i)
        _CACHE["units"] = by
    return _CACHE["units"].get(cc, [])


def hidden_units(cc):
    """Raster unit codes filed under cc that are not part of it (data/areas.json countries.<cc>.hidden)."""
    return set(((load().get("countries") or {}).get(cc) or {}).get("hidden") or ())


def country_areas(cc):
    """[{id, name, en, regions: [codes]}] of country cc (curated or automatic), None without areas."""
    key = ("areas", cc)
    if key not in _CACHE:
        cur = ((load().get("countries") or {}).get(cc) or {}).get("areas")
        if cur:
            out = [{"id": a["id"], "name": a["name"], "en": a.get("en", a["name"]), "regions": list(a["regions"])}
                   for a in cur if a.get("regions")]
        else:
            out = auto_areas(cc)
        if out is not None and len(out) < 2:
            out = None
        _CACHE[key] = out
    return _CACHE[key]


def auto_areas(cc, min_units=AUTO_MIN_UNITS, max_areas=AUTO_MAX_AREAS):
    """Compass areas of a country without curated ones (None with fewer than min_units units)."""
    from .regions import region_centroids
    units = units_of(cc)
    if len(units) < min_units:
        return None
    _, codes, names, _, _ = _regions()
    clat, clng, km2 = region_centroids()
    u = np.array(units)
    has = km2[u] > 0
    if has.sum() < 2:
        return None
    la, ln = np.radians(clat[u]), np.radians(clng[u])
    V = np.stack([np.cos(la) * np.cos(ln), np.cos(la) * np.sin(ln), np.sin(la)], 1)
    w = np.sqrt(np.maximum(km2[u], 0.0))
    c0 = _mean_dir(V[has], w[has])
    rms = math.sqrt(float((w[has] * _ang(V[has], c0) ** 2).sum() / w[has].sum())) or 1e-9
    k0 = int(min(max_areas, max(3, round(math.sqrt(len(units) / 2.0)))))
    best = None
    for k in range(k0, 2, -1):   # the most areas whose clusters get distinct compass names
        lab = _kmeans(V[has], w[has], k)
        centres = np.array([_mean_dir(V[has][lab == j], w[has][lab == j]) for j in range(k)])
        dirs = [_direction(centres[j], c0, rms) for j in range(k)]
        if best is None:
            best = (k, lab, centres, dirs)
        if len(set(dirs)) == k:
            best = (k, lab, centres, dirs)
            break
    k, lab, centres, dirs = best
    # units without pixels (too small for the raster): the area with the nearest centre to their label point
    full = np.full(len(u), -1)
    full[has] = lab
    pts = load().get("unit_points") or {}
    for j in np.flatnonzero(~has):
        p = pts.get(codes[u[j]])
        if p:
            a, b = math.radians(p[0]), math.radians(p[1])
            full[j] = int(np.argmax(centres @ np.array([math.cos(a) * math.cos(b), math.cos(a) * math.sin(b),
                                                       math.sin(a)])))
    out = []
    for j in range(k):
        m = full == j
        if not m.any():
            continue
        uk, en = dirs[j]
        big = u[m][int(np.argmax(w[m]))]
        out.append({"id": "%s:auto%d" % (cc, j), "name": uk, "en": en,
                    "regions": [codes[i] for i in u[m]], "_big": region_name(codes[big])})
    seen = {}
    for a in out:
        seen[a["name"]] = seen.get(a["name"], 0) + 1
    for a in out:
        big = a.pop("_big")
        if seen[a["name"]] > 1:
            a["name"] = "%s (%s)" % (a["name"], big)
            a["en"] = "%s (%s)" % (a["en"], big)
    order = np.argsort([-sum(km2[_code_pos()[r]] for r in a["regions"]) for a in out], kind="stable")
    return [out[i] for i in order]


def _direction(c, c0, rms):
    """(Ukrainian, English) compass name of a cluster centre c seen from the country's centre c0."""
    if float(_ang(c[None], c0)[0]) < CENTRE_FRACTION * rms:
        return "Центр", "Centre"
    return DIRECTIONS[int(((_bearing(c0, c) + 22.5) % 360.0) // 45.0)]


def _ang(V, c):
    return np.arccos(np.clip(V @ c, -1.0, 1.0))


def _mean_dir(V, w):
    m = (V * w[:, None]).sum(0)
    return m / max(np.linalg.norm(m), 1e-12)


def _bearing(a, b):
    """Initial bearing (deg) from unit vector a to unit vector b."""
    la1, ln1 = math.asin(a[2]), math.atan2(a[1], a[0])
    la2, ln2 = math.asin(b[2]), math.atan2(b[1], b[0])
    y = math.sin(ln2 - ln1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(ln2 - ln1)
    return math.degrees(math.atan2(y, x)) % 360.0


def _kmeans(V, w, k, iters=50):
    """Deterministic weighted spherical k-means: farthest-point initialisation from the unit farthest from the
    weighted centre, then Lloyd iterations."""
    c0 = _mean_dir(V, w)
    C = [V[int(np.argmax(_ang(V, c0)))]]
    while len(C) < min(k, len(V)):
        d = np.min(np.stack([_ang(V, c) for c in C]), 0)
        C.append(V[int(np.argmax(d * np.sqrt(w)))])
    C = np.array(C)
    lab = np.zeros(len(V), int)
    for _ in range(iters):
        new = np.argmax(V @ C.T, 1)
        if _ > 0 and (new == lab).all():
            break
        lab = new
        for j in range(len(C)):
            if (lab == j).any():
                C[j] = _mean_dir(V[lab == j], w[lab == j])
    return lab


def area_labels(cc, regs):
    """(len(regs),) index into country_areas(cc) of every raster region index in regs (-1: not in an area);
    None when the country has no areas."""
    areas = country_areas(cc)
    if areas is None:
        return None
    pos = _code_pos()
    lab = {}
    for j, a in enumerate(areas):
        for c in a["regions"]:
            if c in pos:
                lab[pos[c]] = j
    return np.array([lab.get(int(r), -1) for r in regs])


def group_regions(probs, cc, top_regions=3, min_prob=0.0):
    """{code: P(region | cc)} -> the country's areas with their probability and their top_regions most probable
    units, most probable area first (areas below min_prob left out); None when cc has no areas.  Units of the
    country missing from probs count as 0; probs of units outside every area are left out."""
    areas = country_areas(cc)
    if areas is None:
        return None
    out = []
    for a in areas:
        rs = sorted(((probs.get(c, 0.0), c) for c in a["regions"]), key=lambda t: -t[0])
        p = sum(q for q, _ in rs)
        if p <= min_prob:
            continue
        out.append({"id": a["id"], "name": a["name"], "en": a["en"], "probability": p, "size": len(a["regions"]),
                    "regions": [{"code": c, "name": region_name(c), "probability": q}
                                for q, c in rs[:top_regions] if q > 0]})
    out.sort(key=lambda a: -a["probability"])
    return out
