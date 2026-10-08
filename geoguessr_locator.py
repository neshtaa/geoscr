#!/usr/bin/env python3
"""
GeoGuessr locator - purely mathematical (no AI / neural networks).

  python3 geoguessr_locator.py --image pano.jpg [--heading 123]       equirect panorama or a screenshot
  python3 geoguessr_locator.py --views views.json                      in-game screenshots with camera angles
  python3 geoguessr_locator.py --pano <pano_id>                        official Street View panorama (test)
  python3 geoguessr_locator.py --latlng 50.45,30.52                    nearest panorama to a point (test)
  add --map "<name|id|slug>" for the map being played (prior, score scale, bounds; data/maps.json)
  add --json for machine-readable output

views.json: [{"image": "v0.jpg", "yaw": 0, "pitch": 0, "hfov": 100}, ...]  (yaw = true azimuth, deg)
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from engine.geo import load_maps, resolve_map  # noqa: E402
from engine.locator import distance_report, get_locator  # noqa: E402

GROUP_UK = {"solar": "сонце/небо", "road": "дорога", "landscape": "ландшафт", "vehicle": "авто/камера",
            "structure": "забудова/стовпи", "knn": "схожі панорами", "sun": "положення сонця"}


def print_result(res, truth=None):
    line = "=" * 64
    print(line)
    mp = res.get("map")
    if mp:
        print("Карта: %s (бали: D = %.0f км, апріорі: %s)" % (mp["name"] or mp["id"], mp["score_scale_km"] * 10, mp["prior"]))
        pc = mp.get("prior_counts") or {}
        if pc.get("stale"):
            print("  увага: карту змінено %s, після ігор, з яких взято частоти (%s..%s)"
                  % (pc["map_updated"], pc["dates"][0], pc["dates"][1]))
    print("Найімовірніші країни:")
    for i, c in enumerate(res["countries"], 1):
        bar = "#" * int(round(c["probability"] * 30))
        print("  %d. %-28s %5.1f%%  %s" % (i, "%s (%s)" % (c["name"], c["code"]), 100 * c["probability"], bar))
    if res.get("regions"):
        print("\nНайімовірніші регіони:")
        for r in res["regions"]:
            print("  - %-34s %5.1f%%" % ("%s (%s)" % (r["name"], r["code"]), 100 * r["probability"]))
    g = res["guess"]
    print("\nТочка здогадки: %.4f, %.4f   (очікувано ~%d балів)" % (g["lat"], g["lng"], g["expected_score"]))
    t = res.get("top_country_point")
    if t:
        print("Точка всередині %s: %.4f, %.4f   (~%d балів, якщо країна вірна)"
              % (res["countries"][0]["code"], t["lat"], t["lng"], t["expected_score_if_country_right"]))
    if res["observations"]:
        print("\nЩо видно на зображенні:")
        for o in res["observations"][:10]:
            print("  - %s  (%.0f%%)" % (o["text"], 100 * o["strength"]))
    for h in res["hints"]:
        print("\n%s - %s (%.1f%%)" % (h["country_code"], h["country"], 100 * h["probability"]))
        if h.get("driving_side"):
            ok = h.get("driving_side_consistent")
            mark = "" if ok is None else ("  [збігається]" if ok else "  [НЕ збігається]")
            print("  Рух: %s%s" % ({"left": "лівосторонній", "right": "правосторонній"}.get(h["driving_side"], h["driving_side"]), mark))
        if h.get("regions"):
            print("  Регіони в межах країни: " + ", ".join("%s %.0f%%" % (r["name"], 100 * r["probability"]) for r in h["regions"]))
        contrib = res["contributions"].get(h["country_code"], {})
        if contrib:
            parts = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:4]
            print("  Внесок доказів: " + ", ".join("%s %+.1f" % (GROUP_UK.get(k, k), v) for k, v in parts))
        for c in h["geoguessr"]:
            m = (" <- " + ", ".join(c["matched"])) if c["matched"] else ""
            print("  [GeoGuessr] %s: %s%s" % (c["title"], c["text"][:220], m))
        for t in h["plonkit"]:
            m = (" <- " + ", ".join(t["matched"])) if t["matched"] else ""
            print("  [Plonk It] %s%s" % (t["text"][:220], m))
    if truth:
        print("\nПеревірка: справжня країна %s (ранг %s), регіон %s (ранг %s), похибка %.0f км, %d балів"
              % (truth["true_country"], truth["rank_of_true_country"], truth["true_region"],
                 truth["rank_of_true_region"], truth["distance_km"], truth["points"]))
    print("(%d мс)" % res["timing_ms"]["total"])
    print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--views")
    src.add_argument("--pano")
    src.add_argument("--latlng")
    ap.add_argument("--heading", type=float, default=None,
                    help="true azimuth of the image centre (deg); unknown if omitted")
    ap.add_argument("--hfov", type=float, default=None, help="horizontal field of view of a screenshot")
    ap.add_argument("--radius", type=int, default=1000)
    ap.add_argument("--map", default=None, help="map being played: name, id or slug (data/maps.json)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    mp = resolve_map(args.map)
    if mp and not mp["known"]:
        sys.stderr.write("map %r is not in data/maps.json (known: %s); using its name only\n"
                         % (args.map, ", ".join(m["name"] for m in load_maps())))
    loc = get_locator()
    truth = None
    if args.image:
        res = loc.analyze_image(args.image, heading=args.heading, hfov=args.hfov, map_info=mp)
    elif args.views:
        base = os.path.dirname(os.path.abspath(args.views))
        views = json.load(open(args.views))
        for v in views:
            v["image"] = os.path.join(base, v["image"])
        res = loc.analyze_views(views, map_info=mp)
    elif args.pano:
        res = loc.analyze_pano(pano_id=args.pano, map_info=mp)
    else:
        lat, lng = (float(v) for v in args.latlng.split(","))
        res = loc.analyze_pano(lat=lat, lng=lng, radius=args.radius, map_info=mp)
    if "panorama" in res:
        truth = distance_report(res, res["panorama"]["lat"], res["panorama"]["lng"])
        res["check"] = truth
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    else:
        print_result(res, truth)


if __name__ == "__main__":
    main()
