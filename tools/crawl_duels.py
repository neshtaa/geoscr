#!/usr/bin/env python3
"""
Collect real GeoGuessr rounds from public duel histories (GET only) for calibration.

Every player's last 40 duels are public (GET /api/v4/game-history/{userId}, the data behind
profile pages); each round carries the panorama id, the true position and country. Starting
from the user's own duel opponents the crawler walks breadth-first through players and stores
the rounds (no player ids or nicknames are kept). The map of each duel comes from the public
static result file /static-content/game-results/duels/{gameId}.json.

These rounds are used OFFLINE only: per-map country priors, more calibration rounds, and
reference panoramas from the real GeoGuessr location pools. Rounds that coincide with the
user's own calib/test rounds are removed later (tools/train_model.py), never used for testing.

  python3 tools/crawl_duels.py --users 80 [--label-maps] [--sleep 0.7]
Output: data/calibration/duel_rounds.json (incremental, resumable).
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gg_api import HEADERS, request  # noqa: E402

OUT = os.path.join(ROOT, "data", "calibration", "duel_rounds.json")


def static_duel(game_id, timeout=20):
    url = "https://www.geoguessr.com/static-content/game-results/duels/%s.json" % game_id
    req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"], "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=80, help="number of player histories to read")
    ap.add_argument("--label-maps", action="store_true", help="fetch the map of every new duel")
    ap.add_argument("--sleep", type=float, default=0.7)
    args = ap.parse_args()
    state = json.load(open(OUT)) if os.path.exists(OUT) else {"rounds": [], "games": {}, "visited": [], "queue": []}
    games = state["games"]                      # gameId -> {"mode", "map", "map_name", "start"}
    visited = set(state["visited"])
    queue = deque(state["queue"])
    have = {(r["game"], r["round"]) for r in state["rounds"]}
    if not visited and not queue:
        st, me = request("https://www.geoguessr.com/api/v3/profiles/me")
        queue.append(("me", (me or {}).get("id")))
    n_users = 0
    while queue and n_users < args.users:
        key, uid = queue.popleft()
        if uid in visited:
            continue
        visited.add(uid)
        st, d = request("https://www.geoguessr.com/api/v4/game-history/%s" % key)
        time.sleep(args.sleep)
        if st != 200 or not d:
            continue
        n_users += 1
        new = 0
        for e in d.get("entries") or []:
            duel = e.get("duel") or {}
            gid = e.get("gameId") or duel.get("gameId")
            for p in e.get("players") or []:
                if p.get("id") and p["id"] not in visited:
                    queue.append((p["id"], p["id"]))
            if not gid or not duel.get("rounds"):
                continue
            g = games.setdefault(gid, {"mode": duel.get("gameMode"), "map": None, "map_name": None,
                                       "start": (duel["rounds"][0] or {}).get("startTime")})
            for r in duel["rounds"]:
                if r.get("correctLat") is None or (gid, r.get("roundNumber")) in have:
                    continue
                have.add((gid, r.get("roundNumber")))
                state["rounds"].append({"game": gid, "round": r.get("roundNumber"), "pano_id": r.get("panoId"),
                                        "lat": r["correctLat"], "lng": r["correctLng"],
                                        "gg_country": (r.get("correctCountryCode") or "").upper(),
                                        "gg_heading": r.get("heading"), "start": r.get("startTime"),
                                        "mode": duel.get("gameMode")})
                new += 1
            if args.label_maps and g["map"] is None:
                s = static_duel(gid)
                time.sleep(args.sleep / 2)
                if s:
                    o = s.get("options") or {}
                    g["map"], g["map_name"] = o.get("mapSlug"), o.get("mapName")
                    g["movement"] = o.get("mode")
        print("user %d: +%d rounds (total %d, %d games, queue %d)"
              % (n_users, new, len(state["rounds"]), len(games), len(queue)), flush=True)
        state["visited"], state["queue"] = sorted(visited), list(queue)[:5000]
        if n_users % 5 == 0:
            json.dump(state, open(OUT, "w"))
    state["visited"], state["queue"] = sorted(visited), list(queue)[:5000]
    json.dump(state, open(OUT, "w"))
    for r in state["rounds"]:
        g = games.get(r["game"]) or {}
        r["map"], r["map_name"] = g.get("map"), g.get("map_name")
    json.dump(state, open(OUT, "w"))


if __name__ == "__main__":
    main()
