#!/usr/bin/env python3
"""
Rasterise Natural Earth 10m admin-0 map units into a compact lat/lng -> country
lookup grid (data/world_countries.png + data/world_countries.json).

Usage:
  curl -o scratch/ne/ne_10m_admin_0_map_units.geojson \
    https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_map_units.geojson
  python3 tools/make_country_raster.py scratch/ne/ne_10m_admin_0_map_units.geojson
"""
import json
import os
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = 0.05  # degrees per pixel
W, H = int(360 / RES), int(180 / RES)

# Map units without an ISO code that should still resolve to a country.
FALLBACK = {"N. Cyprus": "CY", "Cyprus U.N. Buffer Zone": "CY", "Somaliland": "SO",
            "Korean DMZ (south)": "KR", "Korean DMZ (north)": "KP", "Akrotiri": "CY",
            "Dhekelia": "CY", "USNB Guantanamo Bay": "CU", "Siachen Glacier": "IN",
            "Southern Patagonian Ice Field": "AR", "Bir Tawil": "EG"}


def ring_area(ring):
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def to_px(ring):
    return [((lng + 180.0) / RES, (90.0 - lat) / RES) for lng, lat in ring]


def main(path):
    fc = json.load(open(path, encoding="utf-8"))
    codes = []
    polys = []  # (area, code_idx, exterior, holes)
    for f in fc["features"]:
        p = f["properties"]
        code = p.get("ISO_A2_EH")
        if not code or code == "-99":
            code = FALLBACK.get(p.get("NAME"))
        if not code:
            continue
        if code not in codes:
            codes.append(code)
        idx = codes.index(code) + 1
        g = f["geometry"]
        parts = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for part in parts:
            polys.append((ring_area(part[0]), idx, part[0], part[1:]))
    assert len(codes) < 255, len(codes)
    polys.sort(key=lambda t: -t[0])  # large first, enclaves drawn last
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for _, idx, ext, holes in polys:
        draw.polygon(to_px(ext), fill=idx)
        for hole in holes:
            draw.polygon(to_px(hole), fill=0)
        # islands smaller than a pixel still get one pixel
        lng, lat = ext[0]
        x, y = int((lng + 180) / RES) % W, min(H - 1, int((90 - lat) / RES))
        if img.getpixel((x, y)) == 0:
            img.putpixel((x, y), idx)
    img.save(os.path.join(ROOT, "data", "world_countries.png"), optimize=True)
    json.dump({"resolution_deg": RES, "codes": ["--"] + codes},
              open(os.path.join(ROOT, "data", "world_countries.json"), "w"))
    print("codes:", len(codes))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "scratch/ne/ne_10m_admin_0_map_units.geojson"))
