#!/usr/bin/env python3
"""
Rasterise Natural Earth 10m admin-1 (states / provinces) into a lat/lng -> region lookup
grid (data/world_regions.npz: uint16 grid at 0.05 deg + ISO 3166-2 codes and names).

  curl -o scratch/ne/admin1.geojson \
    https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson
  python3 tools/make_region_raster.py scratch/ne/admin1.geojson
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from make_country_raster import H, RES, W, ring_area, to_px  # noqa: E402


def main(path):
    fc = json.load(open(path, encoding="utf-8"))
    codes, names, countries, polys = ["--"], [""], [""], []
    for f in fc["features"]:
        p = f["properties"]
        cc = (p.get("iso_a2") or "").upper()
        code = p.get("iso_3166_2") or ""
        if not code or code.startswith("-") or "-" not in code or code.endswith("~"):
            code = "%s-%s" % (cc, p.get("adm1_code"))
        if not f.get("geometry"):
            continue
        codes.append(code.upper())
        names.append(p.get("name_en") or p.get("name") or code)
        countries.append(cc)
        idx = len(codes) - 1
        g = f["geometry"]
        parts = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for part in parts:
            polys.append((ring_area(part[0]), idx, part[0], part[1:]))
    polys.sort(key=lambda t: -t[0])
    img = Image.new("I", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for _, idx, ext, holes in polys:
        draw.polygon(to_px(ext), fill=idx)
        for hole in holes:
            draw.polygon(to_px(hole), fill=0)
        lng, lat = ext[0]
        x, y = int((lng + 180) / RES) % W, min(H - 1, int((90 - lat) / RES))
        if img.getpixel((x, y)) == 0:
            img.putpixel((x, y), idx)
    grid = np.asarray(img, dtype=np.int64).astype(np.uint16)
    np.savez_compressed(os.path.join(ROOT, "data", "world_regions.npz"), grid=grid, codes=np.array(codes),
                        names=np.array(names), countries=np.array(countries), resolution_deg=RES)
    print("regions:", len(codes) - 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "scratch/ne/admin1.geojson"))
