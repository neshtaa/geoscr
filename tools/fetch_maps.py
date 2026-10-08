#!/usr/bin/env python3
"""
Map metadata for the map-aware locator (GET only, >= 1.2 s between requests, stops at HTTP 429).

  python3 tools/fetch_maps.py [--refresh] [--no-history]

For the ranked / official maps below and every map name in data/calibration/history_rounds.json
it stores id, slug, name, maxErrorDistance (score = round(5000 * exp(-10 * d / D))), bounds (the
bounding box of the map's locations, shown to the player on the guess map), coordinateCount and
updatedAt in data/maps.json.  Names are resolved through the user's own finished games
(GET /api/v4/guess-history/range gives each game's mapSlug), then GET /api/v3/search/map?q=.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gg_api import request  # noqa: E402
from engine.geo import is_world_map  # noqa: E402

API = "https://www.geoguessr.com/api"
OUT = os.path.join(ROOT, "data", "maps.json")
HISTORY = os.path.join(ROOT, "data", "calibration", "history_rounds.json")
KNOWN = ["world", "66014417ff2366aa9a7504df", "696fe47c5b07bed052077a95", "698ef135b77fb917bcdf8425",
         "698f47ed7f653e99dffa51bb", "643dbc7ccc47d3a344307998", "668ea3252973190af74233b5", "ukraine"]
MIN_INTERVAL = 1.2


class RateLimited(Exception):
    pass


class Client:
    def __init__(self):
        self.last, self.n = 0.0, 0

    def get(self, url):
        wait = self.last + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        st, data = request(url)
        self.last, self.n = time.time(), self.n + 1
        print("  GET %s -> %s" % (url, st), flush=True)
        if st == 429:
            raise RateLimited(url)
        return st, data


def entry(d):
    size = d.get("mapSize") or {}
    e = {"id": d["id"], "slug": d.get("slug") or d["id"], "name": d.get("name"),
         "maxErrorDistance": d.get("maxErrorDistance"), "bounds": d.get("bounds"),
         "coordinateCount": size.get("coordinateCount"), "updatedAt": d.get("updatedAt")}
    e["world"] = is_world_map(e)
    return e


def slugs_from_history(client, names_by_game):
    """{map name: slug} from the user's own finished games."""
    st, d = client.get(API + "/v4/guess-history/range?" + urllib.parse.urlencode(
        {"from": "1970-01-01T00:00:00.000Z", "to": time.strftime("%Y-%m-%dT00:00:00.000Z", time.gmtime(time.time() + 86400)),
         "limit": 5000}))
    votes = defaultdict(Counter)
    items = (d or {}).get("items") or [] if st == 200 else []
    for it in items:
        name = names_by_game.get(it.get("gameId"))
        if name and it.get("mapSlug"):
            votes[name][it["mapSlug"]] += 1
    return {name: c.most_common(1)[0][0] for name, c in votes.items()}


def search_slug(client, name):
    st, d = client.get(API + "/v3/search/map?" + urllib.parse.urlencode({"page": 0, "count": 20, "q": name}))
    hits = [m for m in (d or []) if st == 200 and (m.get("name") or "").strip().lower() == name.strip().lower()]
    hits.sort(key=lambda m: -(m.get("numberOfGamesPlayed") or 0))
    return hits[0]["id"] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch maps already in data/maps.json")
    ap.add_argument("--no-history", action="store_true", help="resolve names by search only")
    args = ap.parse_args()
    old = json.load(open(OUT)).get("maps", []) if os.path.exists(OUT) else []
    maps = {m["id"]: m for m in old}
    by_key = {k.lower(): m["id"] for m in old for k in [m["id"], m["slug"], m["name"] or ""] + m.get("aliases", [])}
    rounds = json.load(open(HISTORY))
    names = sorted({r["map"] for r in rounds if r.get("map")})
    names_by_game = {r["game"]: r["map"] for r in rounds if r.get("map")}
    client = Client()
    unresolved, slug_of = [], {}
    try:
        targets = list(KNOWN)
        todo = [n for n in names if n.lower() not in by_key or args.refresh]
        hist = slugs_from_history(client, names_by_game) if todo and not args.no_history else {}
        for n in todo:
            slug_of[n] = hist.get(n) or search_slug(client, n)
            if slug_of[n]:
                targets.append(slug_of[n])
            else:
                unresolved.append(n)
        for key in dict.fromkeys(targets):
            if key.lower() in by_key and not args.refresh:
                continue
            st, d = client.get(API + "/maps/" + urllib.parse.quote(key))
            if st == 200 and d and d.get("id"):
                e = entry(d)
                if maps.get(e["id"], {}).get("aliases"):
                    e["aliases"] = maps[e["id"]]["aliases"]
                maps[e["id"]] = e
                for k in (e["id"], e["slug"], e["name"] or ""):
                    by_key[k.lower()] = e["id"]
            else:
                unresolved.append(key)
    except RateLimited as e:
        print("HTTP 429 at %s - stopping, partial result saved" % e)
    for n, slug in slug_of.items():  # maps renamed since the user played them keep the old name
        m = maps.get(by_key.get((slug or "").lower()))
        if m and n.lower() != (m["name"] or "").lower() and n not in m.setdefault("aliases", []):
            m["aliases"].append(n)
    out = sorted(maps.values(), key=lambda m: (not m["world"], m["name"] or ""))
    json.dump({"source": "GET /api/maps/{id}", "fetched": time.strftime("%Y-%m-%d"), "maps": out},
              open(OUT, "w"), indent=1, ensure_ascii=False)
    print("%d maps -> %s (%d requests)%s" % (len(out), OUT, client.n,
                                             "; unresolved: " + ", ".join(unresolved) if unresolved else ""))


if __name__ == "__main__":
    main()
