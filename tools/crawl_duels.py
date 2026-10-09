#!/usr/bin/env python3
"""
Collect real GeoGuessr rounds from public duel histories (GET only) for calibration.

Every player's last 40 duels are public (GET /api/v4/game-history/{userId}, the data behind
profile pages); each round carries the panorama id, the true position and country. Starting
from the user's own duel opponents the crawler walks breadth-first through players and stores
the rounds (no nicknames are kept; player ids only as the crawl queue). The map of each duel comes
from the public static result file /static-content/game-results/duels/{gameId}.json.

These rounds are used OFFLINE only: per-map country priors, more calibration rounds, and
reference panoramas from the real GeoGuessr location pools. Rounds that coincide with the
user's own calib/test rounds are removed later (tools/build_dataset.py duel_candidates), never
used for testing.

Politeness: GET only, requests >= 1.5 s apart (start to start), stop at the first HTTP 429 or 401
(cookie expired) and after 5 other failing statuses in a row (403 / 5xx: e.g. a Cloudflare block); players
that were not read stay in the queue (the state is saved; rerun later to continue).

  python3 tools/crawl_duels.py --users 250 [--label-maps] [--sleep 1.5]
Output: data/calibration/duel_rounds.json (incremental, resumable, written atomically).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gg_api import HEADERS, request  # noqa: E402

OUT = os.path.join(ROOT, "data", "calibration", "duel_rounds.json")
MIN_SLEEP = 1.5      # seconds between two GeoGuessr requests (start to start)
MAX_NET_ERRORS = 5   # consecutive network failures before giving up
MAX_BAD_STATUS = 5   # consecutive failing HTTP statuses (other than 404 / 401 / 429) before giving up


class RateLimited(Exception):
    """HTTP 429 (or 401: not logged in): stop crawling."""


class Throttle:
    """At most one request per `gap` seconds (measured between request starts)."""

    def __init__(self, gap, clock=time.monotonic, sleep=time.sleep):
        self.gap, self.clock, self.sleep, self.last = gap, clock, sleep, None

    def wait(self):
        if self.last is not None:
            dt = self.gap - (self.clock() - self.last)
            if dt > 0:
                self.sleep(dt)
        self.last = self.clock()


def static_duel(game_id, timeout=20):
    """(status, json) of the public static result of a duel."""
    url = "https://www.geoguessr.com/static-content/game-results/duels/%s.json" % game_id
    req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"], "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return None, None


def save_state(state, path=OUT):
    """Write the crawl state atomically (temp file + os.replace): a crash never leaves a partial file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def add_history(state, d, have, visited, queue):
    """Merge one game-history response into the state; returns the number of new rounds."""
    games, new = state["games"], 0
    for e in d.get("entries") or []:
        duel = e.get("duel") or {}
        gid = e.get("gameId") or duel.get("gameId")
        for p in e.get("players") or []:
            if p.get("id") and p["id"] not in visited:
                queue.append((p["id"], p["id"]))
        if not gid or not duel.get("rounds"):
            continue
        games.setdefault(gid, {"mode": duel.get("gameMode"), "map": None, "map_name": None,
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
    return new


def crawl(state, users, label_maps=False, gap=MIN_SLEEP, get=request, static=static_duel, throttle=None,
          save=save_state, log=lambda m: print(m, flush=True)):
    """Read up to `users` non-empty player histories (breadth first from the saved queue).  Returns the
    stop reason: 'done', 'queue empty', 'HTTP 429', 'HTTP 401', 'HTTP <status> x5' or 'network'.  A player is
    marked visited only when its history was read (200) or does not exist (404); on another failing status it
    goes to the back of the queue (visited after a second failure in the same run)."""
    throttle = throttle or Throttle(max(gap, MIN_SLEEP))
    games = state["games"]                      # gameId -> {"mode", "map", "map_name", "start"}
    visited = set(state["visited"])
    queue = deque(tuple(q) for q in state["queue"])
    have = {(r["game"], r["round"]) for r in state["rounds"]}
    n_users, n_req, net_err, bad, reason = 0, 0, 0, 0, "done"
    failed = set()  # players with a failing status in this run

    def checkpoint():
        state["visited"], state["queue"] = sorted(visited), [list(q) for q in list(queue)[:5000]]
        save(state)

    def call(fn, *a):
        nonlocal n_req
        throttle.wait()
        n_req += 1
        st, d = fn(*a)
        if st in (429, 401):
            raise RateLimited(st)
        return st, d

    try:
        if not visited and not queue:
            st, me = call(get, "https://www.geoguessr.com/api/v3/profiles/me")
            queue.append(("me", (me or {}).get("id")))
        while n_users < users:
            if not queue:
                reason = "queue empty"
                break
            key, uid = queue.popleft()
            if uid in visited:
                continue
            try:
                st, d = call(get, "https://www.geoguessr.com/api/v4/game-history/%s" % key)
            except RateLimited:
                queue.appendleft((key, uid))  # not read: keep it for the next run
                raise
            except Exception as e:  # network error: retry this player later
                net_err += 1
                queue.appendleft((key, uid))
                log("network error %r (%d in a row)" % (e, net_err))
                if net_err >= MAX_NET_ERRORS:
                    reason = "network"
                    break
                continue
            net_err = 0
            if st not in (200, 404):  # 403 / 5xx / ...: not read, try again later (Cloudflare, outage)
                bad += 1
                log("HTTP %s for a player history (%d in a row)" % (st, bad))
                if uid in failed:
                    visited.add(uid)
                else:
                    failed.add(uid)
                    queue.append((key, uid))
                if bad >= MAX_BAD_STATUS:
                    reason = "HTTP %s x%d" % (st, bad)
                    break
                continue
            bad = 0
            visited.add(uid)
            if st != 200 or not d or not d.get("entries"):
                continue
            n_users += 1
            new = add_history(state, d, have, visited, queue)
            if label_maps:
                for e in d.get("entries") or []:
                    gid = e.get("gameId") or (e.get("duel") or {}).get("gameId")
                    g = games.get(gid)
                    if g is None or g["map"] is not None:
                        continue
                    st, s = call(static, gid)
                    if s:
                        o = s.get("options") or {}
                        g["map"], g["map_name"] = o.get("mapSlug"), o.get("mapName")
                        g["movement"] = o.get("mode")
            log("user %d: +%d rounds (total %d, %d games, queue %d, %d requests)"
                % (n_users, new, len(state["rounds"]), len(games), len(queue), n_req))
            if n_users % 5 == 0:
                checkpoint()
    except RateLimited as e:
        reason = "HTTP %s" % e.args[0]
        log("%s - stopping (state saved; rerun later%s)"
            % (reason, " with a fresh _ncfa cookie" if e.args[0] == 401 else " to continue"))
    for r in state["rounds"]:
        g = games.get(r["game"]) or {}
        r["map"], r["map_name"] = g.get("map"), g.get("map_name")
    checkpoint()
    return reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=80, help="number of (non-empty) player histories to read")
    ap.add_argument("--label-maps", action="store_true", help="fetch the map of every new duel (1 GET per duel)")
    ap.add_argument("--sleep", type=float, default=MIN_SLEEP, help="seconds between requests (>= %.1f)" % MIN_SLEEP)
    args = ap.parse_args()
    if args.sleep < MIN_SLEEP:
        print("--sleep %.2f raised to %.1f s" % (args.sleep, MIN_SLEEP))
    state = json.load(open(OUT)) if os.path.exists(OUT) else {"rounds": [], "games": {}, "visited": [], "queue": []}
    n0, ids0 = len(state["rounds"]), {r.get("pano_id") for r in state["rounds"]}
    t0 = time.time()
    reason = crawl(state, args.users, args.label_maps, max(args.sleep, MIN_SLEEP))
    new_ids = {r.get("pano_id") for r in state["rounds"][n0:]} - ids0
    print("stopped: %s; +%d rounds (%d new panorama ids), %d rounds in total, %.0f s"
          % (reason, len(state["rounds"]) - n0, len(new_ids), len(state["rounds"]), time.time() - t0))
    if reason.startswith("HTTP") or reason == "network":
        sys.exit(2)


if __name__ == "__main__":
    main()
