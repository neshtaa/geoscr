#!/usr/bin/env python3
"""
Download GeoGuessr's own per-panorama clue annotations (GET /api/v4/clues/{panoId}).

This is how GeoGuessr's post-round analysis works: the clues are curated per location
(country, admin-1 regions via seterraRegionIds, and the camera heading/pitch/zoom where the
clue is visible) and looked up by the panorama id - the image itself is not analysed.
The annotations are used here only OFFLINE, as labels for calibrating the locator's
detectors and hint ranking; the live script never reads the panorama id.

  python3 tools/fetch_pano_clues.py                    rounds of data/calibration/history_rounds.json
  python3 tools/fetch_pano_clues.py --dataset --n 200  probe random dataset panoramas (does the
                                                       endpoint annotate panoramas outside games?)
Output: data/calibration/pano_clues.json {pano_id: [clue, ...]} (incremental).
Needs network access to www.geoguessr.com and the _ncfa cookie (data/session_cookie.txt).
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gg_api import request  # noqa: E402

OUT = os.path.join(ROOT, "data", "calibration", "pano_clues.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="store_true")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if args.dataset:
        from calibrate import load_records
        ids = [r["pano_id"] for r in load_records() if r["mode"] in ("world", "balanced")]
        random.Random(0).shuffle(ids)
    else:
        ids = [r["pano_id"] for r in json.load(open(os.path.join(ROOT, "data/calibration/history_rounds.json")))]
    ids = [i for i in dict.fromkeys(ids) if i and i not in out]
    if args.n:
        ids = ids[: args.n]
    hit = 0
    for k, pid in enumerate(ids, 1):
        st, d = request("https://www.geoguessr.com/api/v4/clues/" + pid)
        if st == 401 or st == 403:
            sys.exit("not authorised (HTTP %d): check data/session_cookie.txt" % st)
        out[pid] = d if (st == 200 and isinstance(d, list)) else []
        hit += bool(out[pid])
        if k % 50 == 0 or k == len(ids):
            json.dump(out, open(OUT, "w"))
            print("%d/%d panoramas, %d with clues" % (k, len(ids), hit), flush=True)
        time.sleep(args.sleep)
    json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    main()
