#!/usr/bin/env python3
"""
World-map calibration games: classic single-player games on the official World map
(id 52d15ff6850ffb3b847cb3b4, slug "world"), played by the locator, as calibration / live-evaluation data.

  python3 tools/world_play.py play --games 120 --hours 6 [--port 8097] [--walk 3] [--gap 20] [--validate 4]
                                  [--snapshot [--refresh-min 15] | --freeze <commit>] [--clues sample|all|none]
      Loop: create a game as the web client does (POST /api/v3/games {map: "world", type: "standard",
      timeLimit: 0, forbidMoving / forbidZooming / forbidRotating: false, rounds: 5}), play it with
      `node play_live_visual.js --game <t> --submit --headless --exit-when-done --summary <f> [--test-walk K]`
      against an own locator server (LOCATOR_URL, started here on --port when nothing answers there), and only
      once the game is finished read it (GET /api/v3/games/<t>): per round the truth (lat, lng, decoded pano id,
      heading, country, admin-1 region), our guess, points, distance, panoramas fused and the locator's top
      countries / regions of the start capture and of the submitted (fused) answer, from the saved
      response.json files.  Games alternate: no walking (No Move quality) / --test-walk K (Moving).
      The server runs the working tree, a snapshot of it (--snapshot: the current saved model, moved to a newer
      saved model between games once it stopped changing) or a frozen commit (--freeze); every game records the
      model / engine fingerprint it was played with.
      After every game GeoGuessr's clue placements of its panoramas are appended to
      data/calibration/pano_clues.json (--clues sample: the first round of each game, as classic World games came
      back without placements); every --collect-every games (and at the end) the round panoramas are
      downloaded (tools/build_dataset.py world_play, kept as World-map references) and their features computed
      (tools/calibrate.py features code path, current module versions only); progress numbers go to
      scratch/world/progress.json.  --validate N: stop after N games (end-to-end check).
  python3 tools/world_play.py collect      clues + panoramas + features of the finished games (resumable)
  python3 tools/world_play.py features     features of the world_play panoramas only
  python3 tools/world_play.py report       scratch/world/progress.json from data/calibration/world_games.json
  python3 tools/world_play.py rebuild      recompute the answer digests from the saved captures (no network)
  python3 tools/world_play.py replay --server http://127.0.0.1:8098 [--split world_test] [--no-map]
      re-run any locator server on the exact live captures of the finished games (start capture = No Move,
      a Moving round's captures fused in order) and score them against the truth: how a changed model would
      have played these World-map rounds live (local requests only)

Data: data/calibration/world_games.json {"map": ..., "games": [{token, n, walk, status, split, rounds: [...]}]}
(status created -> played -> finished, or abandoned); captures of each game are moved to
scratch/world/live/<token>/ (views, response.json, result_r<k>.jpg screenshots), logs to scratch/world/logs/.
Splits: world_calib / world_test = halves of the games by md5(token) (tools/calibrate.py split_of, mode
world_play): calibrate on world_calib, report on world_test.

Fair play: during a game nothing here or in play_live_visual.js reads the round's location (no GET of the
game, no clue / location endpoint, no pano id or position): the page only captures what the player sees and
--submit posts the locator's guess.  The game is read only after the last guess's response said it is
finished (state / guess count from the page's whitelisted metadata); an unfinished game is resumed by the
page like a player or abandoned, never read.  Only classic single-player games are created; multiplayer
modes are never touched.
Pacing: >= 1.5 s between this script's GeoGuessr requests, >= --gap s between games; stops at the first
HTTP 429 and at any 401 / 403.  Crash-safe: world_games.json is rewritten atomically after every change and
a rerun resumes (unfinished games are continued, finished ones fetched, missing clues / panoramas added).
Memory: waits while less than --min-avail-mb is available (earlyoom); disk: stops at --max-disk-gb for
this workstream's files (captures + panoramas), dropping the move captures' view images first.
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

SITE = "https://www.geoguessr.com"
WORLD_MAP = {"id": "52d15ff6850ffb3b847cb3b4", "slug": "world", "name": "World", "maxErrorDistance": 14916862}
GAMES_FILE = os.path.join(ROOT, "data", "calibration", "world_games.json")
CLUES_FILE = os.path.join(ROOT, "data", "calibration", "pano_clues.json")
WORK = os.path.join(ROOT, "scratch", "world")
LIVE_OUT = os.path.join(ROOT, "scratch", "live")
DATASET = os.path.join(ROOT, "scratch", "dataset")
NODE_SCRIPT = os.path.join(ROOT, "play_live_visual.js")
MIN_GAP_S = 1.5
LABEL_FIX = {"UK": "GB"}


class Stop(Exception):
    """A reason to stop the whole run (HTTP 429 / 401 / 403, disk, time)."""


# ------------------------------------------------------------------ state file

def load_state(path=GAMES_FILE):
    try:
        with open(path) as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}
    st.setdefault("map", dict(WORLD_MAP))
    st.setdefault("games", [])
    return st


def save_state(st, path=GAMES_FILE):
    st["updated_at"] = now_iso()
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def split_of_game(token):
    """world_calib / world_test by game: the hash rule of tools/calibrate.py split_of (mode world_play)."""
    return "world_calib" if int(hashlib.md5(str(token).encode()).hexdigest(), 16) % 2 == 0 else "world_test"


def next_walk(games, walk):
    """Alternate No Move (0) and Moving (walk) games: the condition with fewer played games goes next."""
    done = [g for g in games if g.get("status") in ("created", "played", "finished")]
    n_move = sum(1 for g in done if g.get("walk"))
    return walk if walk and n_move < len(done) - n_move else 0


# ------------------------------------------------------------------ GeoGuessr API (paced)

class Api:
    """The script's own GeoGuessr requests: >= MIN_GAP_S apart, Stop at the first 429 and at 401 / 403."""

    def __init__(self, request=None, gap=MIN_GAP_S, sleep=time.sleep, clock=time.time):
        if request is None:
            from gg_api import request
        self._request, self.gap, self._sleep, self._clock = request, gap, sleep, clock
        self.last = 0.0
        self.calls = []

    def call(self, method, url, data=None):
        wait = self.last + self.gap - self._clock()
        if wait > 0:
            self._sleep(wait)
        self.last = self._clock()
        st, d = self._request(url, data=data, method=method)
        self.calls.append((method, url, st))
        if st == 429:
            raise Stop("HTTP 429 from GeoGuessr (%s %s): stopping" % (method, url))
        if st in (401, 403):
            raise Stop("HTTP %s from GeoGuessr (%s %s): check data/session_cookie.txt / the subscription" % (st, method, url))
        return st, d


def create_body(rounds=5):
    """The web client's startGame body (static_chunks_58447 module 683110: {map: mapSlug, type, timeLimit,
    forbidMoving, forbidZooming, forbidRotating, rounds})."""
    return {"map": WORLD_MAP["slug"], "type": "standard", "timeLimit": 0, "forbidMoving": False,
            "forbidZooming": False, "forbidRotating": False, "rounds": rounds}


def create_game(api):
    st, d = api.call("POST", SITE + "/api/v3/games", create_body())
    if st != 200 or not isinstance(d, dict) or not d.get("token"):
        msg = d.get("message") if isinstance(d, dict) else None   # never the game object itself
        raise RuntimeError("could not create a game: HTTP %s %s" % (st, msg or ""))
    # keep only settings; a new game's response holds its first round, which is never read here
    keep = {k: d.get(k) for k in ("token", "type", "mode", "state", "roundCount", "timeLimit", "forbidMoving",
                                  "forbidZooming", "forbidRotating", "map", "mapName")}
    if keep["type"] != "standard" or keep["mode"] not in (None, "standard") or keep["map"] not in (WORLD_MAP["id"], "world"):
        raise Stop("unexpected game created: %r" % keep)
    return keep


def decode_pano(pid):
    if pid and len(pid) % 2 == 0 and all(c in "0123456789ABCDEFabcdef" for c in pid):
        try:
            return bytes.fromhex(pid).decode()
        except Exception:
            pass
    return pid


def fetch_finished(api, game):
    """GET /api/v3/games/<t> of a game whose last guess was answered with state finished (summary).  Refuses
    any other game, and drops the response unread unless the server also says finished."""
    if game.get("status") != "played" or not (game.get("summary") or {}).get("finished"):
        raise ValueError("game %s is not known to be finished: never read" % game.get("token"))
    st, d = api.call("GET", SITE + "/api/v3/games/" + urllib.parse.quote(game["token"], safe=""))
    if st != 200 or not isinstance(d, dict):
        raise RuntimeError("GET game %s: HTTP %s" % (game["token"], st))
    if d.get("state") != "finished":
        raise RuntimeError("game %s: state %r, not finished: response dropped" % (game["token"], d.get("state")))
    return d


# ------------------------------------------------------------------ round records

def _top(lst, n=5, key="code"):
    return [[x.get(key), round(float(x.get("probability", 0.0)), 5)] for x in (lst or [])[:n]]


def response_digest(resp, max_err_m=WORLD_MAP["maxErrorDistance"], truth=None, truth_cc=None):
    """Countries / regions / guess of one response.json (+ offline points against the truth, and the prior /
    posterior probability and rank of the true country)."""
    if not resp:
        return None
    out = {"countries": _top(resp.get("countries")), "regions": _top(resp.get("regions")),
           "guess": [resp["guess"]["lat"], resp["guess"]["lng"]] if resp.get("guess") else None,
           "n_fused": (resp.get("fusion") or {}).get("n", 1),
           "replaced": bool((resp.get("fusion") or {}).get("replaced"))}
    h = (resp.get("hints") or [{}])[0] or {}
    out["top_country_regions"] = _top(h.get("regions"))
    if h.get("areas"):   # GeoGuessr-style areas of the top country (engine/areas.py), most probable first
        out["top_country_areas"] = _top(h.get("areas"), key="id")
    cap = resp.get("capture") or {}
    if cap.get("countries") and _top(cap.get("countries")) != out["countries"]:
        out["capture_countries"] = _top(cap.get("countries"))   # this capture alone (moving: before fusion)
    pm = resp.get("map") or {}
    out["prior"] = pm.get("prior")
    post, prior = resp.get("posterior") or {}, resp.get("prior") or {}
    if truth_cc and post:
        p = float(post.get(truth_cc, 0.0))
        out["p_true"] = round(p, 6)
        out["rank_true"] = sum(1 for v in post.values() if float(v) > p)
        out["prior_true"] = round(float(prior.get(truth_cc, 0.0)), 6) if prior else None
        if out["countries"]:   # the top country's prior: did the prior carry the answer?
            out["prior_top"] = round(float(prior.get(out["countries"][0][0], 0.0)), 6) if prior else None
    if truth and out["guess"]:
        from engine.geo import geoguessr_points, haversine_km
        d = float(haversine_km(truth[0], truth[1], out["guess"][0], out["guess"][1]))
        out["km"] = round(d, 1)
        out["points"] = int(geoguessr_points(d, max_err_m))
    return out


def capture_dirs(root, token, rnd):
    """Capture directories of a round: <token>_r<k> (start), <token>_r<k>_m<j> (moves), in order."""
    out = []
    for d in glob.glob(os.path.join(root, "%s_r%d" % (token, rnd))) + glob.glob(os.path.join(root, "%s_r%d_m*" % (token, rnd))):
        tail = os.path.basename(d)[len("%s_r%d" % (token, rnd)):]
        out.append((int(tail[2:]) if tail.startswith("_m") and tail[2:].isdigit() else 0, d))
    return [d for _, d in sorted(out)]


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def truth_country(r):
    from engine.geo import country_at
    gg = (r.get("streakLocationCode") or "").upper()
    gg = LABEL_FIX.get(gg, gg)
    raster = country_at(r["lat"], r["lng"])
    return gg or raster, raster


def truth_area(cc, region):
    """Id of the area (engine/areas.py) of country cc that holds admin-1 unit `region`, None when cc has no areas or
    the unit is in none of them."""
    if not cc or not region:
        return None
    try:
        from engine.areas import country_areas
        for a in country_areas(cc) or []:
            if region in a.get("regions") or ():
                return a["id"]
    except Exception:
        pass
    return None


def round_records(game_json, summary, dirs_of):
    """Per-round records of a finished game: truth from the finished game, our guess / points from GeoGuessr's
    guesses, the locator's answers from the captures (dirs_of(round) -> capture directories in order)."""
    from engine.geo import region_at
    max_err = ((game_json.get("map") or {}) if isinstance(game_json.get("map"), dict) else {}).get(
        "maxErrorDistance") or WORLD_MAP["maxErrorDistance"]
    guesses = (game_json.get("player") or {}).get("guesses") or []
    srounds = {r.get("round"): r for r in (summary or {}).get("rounds") or []}
    out = []
    for i, r in enumerate(game_json.get("rounds") or [], 1):
        g = guesses[i - 1] if i - 1 < len(guesses) else {}
        cc, raster = truth_country(r)
        reg = region_at(r["lat"], r["lng"])
        truth = (r["lat"], r["lng"])
        dirs = dirs_of(i)
        resps = [read_json(os.path.join(d, "response.json")) for d in dirs]
        resps = [x for x in resps if x]
        views = read_json(os.path.join(dirs[0], "views.json")) if dirs else None
        s = srounds.get(i) or {}
        rec = {"round": i, "lat": r["lat"], "lng": r["lng"], "pano_id": decode_pano(r.get("panoId")),
               "heading": r.get("heading"), "pitch": r.get("pitch"), "zoom": r.get("zoom"),
               "gg_country": (r.get("streakLocationCode") or "").upper(), "country": cc, "raster_country": raster,
               "region": reg[0] if reg else None, "region_name": reg[1] if reg else None,
               "area": truth_area(cc, reg[0] if reg else None),
               "guess": {"lat": g.get("lat"), "lng": g.get("lng")} if g else None,
               "points": g.get("roundScoreInPoints"), "distance_m": g.get("distanceInMeters"),
               "time_s": g.get("time"), "timed_out": g.get("timedOut"), "steps": g.get("stepsCount"),
               "n_captures": len(resps), "n_fused": (resps[-1].get("fusion") or {}).get("n", 1) if resps else None,
               # captures the server took for a re-capture of an earlier panorama (replaced it, not fused)
               "n_replaced": sum(1 for x in resps[1:] if (x.get("fusion") or {}).get("replaced")),
               "walked": s.get("steps"),
               "start": response_digest(resps[0], max_err, truth, cc) if resps else None,
               # the submitted (fused) answer when the round had more than one capture (else = start)
               "final": response_digest(resps[-1], max_err, truth, cc) if len(resps) > 1 else None,
               "capture": {"mode": views.get("mode"), "views": len(views.get("views") or []),
                           "fov": (views.get("fov") or {}).get("method"), "zoom_locked": views.get("zoom_locked")}
               if views else None,
               "dirs": [os.path.relpath(d, ROOT) for d in dirs], "summary_points": s.get("points"),
               # the round's panorama never loaded in the page (GeoGuessr "Failed to load"): a blind guess
               "blind": bool(s.get("blind"))}
        out.append(rec)
    return out


def rebuild_digests(st, root=ROOT):
    """Recomputes the per-round answer digests (start / final) from the saved capture directories of every
    finished game, with the truth already stored (no network).  Returns the number of rounds updated."""
    n = 0
    for g in st["games"]:
        if g.get("status") != "finished":
            continue
        for r in g.get("rounds") or []:
            dirs = [os.path.join(root, d) for d in r.get("dirs") or []]
            resps = [x for x in (read_json(os.path.join(d, "response.json")) for d in dirs) if x]
            if not resps:
                continue
            truth = (r["lat"], r["lng"])
            r["area"] = truth_area(r.get("country"), r.get("region"))
            r["start"] = response_digest(resps[0], WORLD_MAP["maxErrorDistance"], truth, r.get("country"))
            r["final"] = response_digest(resps[-1], WORLD_MAP["maxErrorDistance"], truth, r.get("country")) \
                if len(resps) > 1 else None
            r["n_replaced"] = sum(1 for x in resps[1:] if (x.get("fusion") or {}).get("replaced"))
            n += 1
    return n


def post_json(url, body, timeout=300):
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def capture_body(d, fusion_key=None, reset=False, use_map=True):
    """The /api/predict body of a saved capture, as the page sent it (views + the game's map settings)."""
    import base64
    v = read_json(os.path.join(d, "views.json")) or {}
    views = []
    for x in v.get("views") or []:
        f = os.path.join(d, x["file"])
        if not os.path.exists(f):
            return None   # view images dropped to save disk
        with open(f, "rb") as fh:
            views.append({"image_b64": base64.b64encode(fh.read()).decode(), "yaw": x["yaw"], "pitch": x["pitch"],
                          "hfov": x["hfov"], "vfov": x.get("vfov")})
    body = {"views": views}
    m = (read_json(os.path.join(d, "meta.json")) or {}).get("map")
    if use_map and m:
        body["map"] = {k: m[k] for k in ("id", "slug", "name", "bounds", "maxErrorDistance") if m.get(k) is not None}
    if fusion_key:
        body["fusion"] = {"round": fusion_key, "reset": bool(reset)}
    return body


def replay_eval(st, server, splits=("world_test",), use_map=True, root=ROOT, log=print, post=post_json, tag=None,
                threads=1):
    """Re-runs a locator server on the saved live captures of the finished games of `splits` (the exact views
    the page captured; a Moving round's captures fused in order under a fresh round key) and scores the start
    capture (No Move) and the fused answer against the truth.  Only local requests; `threads` rounds at a time
    (the captures of a round in order).  -> {"rows", "summary"}."""
    from concurrent.futures import ThreadPoolExecutor
    tag = tag or ("r%d" % int(time.time()))[:24]
    tasks = [(g, r) for g in st["games"]
             if g.get("status") == "finished" and not g.get("exclude") and g.get("split") in splits
             for r in g.get("rounds") or []]

    def one(task):
        g, r = task
        dirs = [os.path.join(root, d) for d in r.get("dirs") or []]
        key = "replay:%s#%d@%s" % (g["token"], r["round"], tag)
        out = []
        for i, d in enumerate(dirs):
            body = capture_body(d, key, reset=(i == 0), use_map=use_map)
            if body is None:
                break
            out.append(post(server.rstrip("/") + "/api/predict", body))
        if not out:
            return None
        truth = (r["lat"], r["lng"])
        return {"game": g["token"], "round": r["round"], "walk": g.get("walk", 0), "country": r.get("country"),
                "region": r.get("region"), "area": r.get("area") or truth_area(r.get("country"), r.get("region")),
                "pano_id": r.get("pano_id"),
                "start": response_digest(out[0], WORLD_MAP["maxErrorDistance"], truth, r.get("country")),
                "final": response_digest(out[-1], WORLD_MAP["maxErrorDistance"], truth, r.get("country")),
                # all captures of the round replayed (False: a move capture's view images were dropped to save disk;
                # its final answer would be an earlier capture's, so the round is left out of the summary)
                "captures": len(out), "of": len(dirs), "complete": len(out) == len(dirs)}

    rows = []
    with ThreadPoolExecutor(max(1, int(threads))) as ex:
        for k, row in enumerate(ex.map(one, tasks)):
            if row:
                rows.append(row)
            if (k + 1) % 25 == 0 or k + 1 == len(tasks):
                log("replay: %d of %d rounds" % (k + 1, len(tasks)))
    st2 = {"games": []}
    for row in rows:   # the report's aggregation, per condition (rounds with every capture only)
        if row.get("complete", True):
            st2["games"].append({"token": row["game"], "status": "finished", "walk": row["walk"],
                                 "rounds": [dict(row, points=(row["final"] or {}).get("points"))]})
    rep = report(st2)
    keep = ("rounds", "points_submitted", "points_start_capture_only", "country_start", "country_final",
            "region_overall_final", "region_given_right_country_final", "region_given_right_country_start",
            "area_given_right_country_final",
            "logp_true_start", "logp_true_final", "wrong_top1", "moving_gain_points_final_minus_start")
    summary = {k: rep.get(k) for k in keep}
    summary["incomplete_rounds_left_out"] = sum(1 for r in rows if not r.get("complete", True))
    summary["games"] = {"nomove": len({r["game"] for r in rows if not r["walk"] and r.get("complete", True)}),
                        "moving": len({r["game"] for r in rows if r["walk"] and r.get("complete", True)})}
    return {"rows": rows, "summary": summary, "splits": list(splits), "server": server, "use_map": use_map}


def _boot_ci(values, groups, n=2000, seed=0):
    """95% interval of the mean of values, resampling whole groups (games); None for < 2 groups."""
    import random
    by = collections.OrderedDict()
    for v, g in zip(values, groups):
        by.setdefault(g, []).append(float(v))
    keys = list(by)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = c = 0.0
        for k in (rng.choice(keys) for _ in keys):
            s += sum(by[k])
            c += len(by[k])
        means.append(s / c)
    means.sort()
    return [round(means[int(0.025 * n)], 4), round(means[min(n - 1, int(0.975 * n))], 4)]


def replay_paired(rows, other_rows):
    """Replay rows of one server minus those of another on the same rounds (game, round; complete rounds only):
    points, top-1 country and log P(true country) of the start capture (No Move games) and of the fused answer
    (Moving games); 95% intervals by game."""
    prev = {(r["game"], r["round"]): r for r in other_rows if r.get("complete", True)}
    out = {}
    for cond, sel, which in (("nomove", lambda r: not r["walk"], "start"), ("moving", lambda r: r["walk"], "final"),
                             ("moving_start", lambda r: r["walk"], "start")):
        pairs = [(r, prev[(r["game"], r["round"])]) for r in rows
                 if r.get("complete", True) and sel(r) and (r["game"], r["round"]) in prev
                 and r.get(which) and prev[(r["game"], r["round"])].get(which)]
        if not pairs:
            continue
        games = [a["game"] for a, _ in pairs]

        def top1(d, cc):
            return float(bool(d.get("countries")) and d["countries"][0][0] == cc)
        dp = [(a[which].get("points") or 0) - (b[which].get("points") or 0) for a, b in pairs]
        d1 = [top1(a[which], a["country"]) - top1(b[which], b["country"]) for a, b in pairs]
        dl = [math.log(max(a[which].get("p_true") or 0.0, 1e-6)) - math.log(max(b[which].get("p_true") or 0.0, 1e-6))
              for a, b in pairs]
        out[cond] = {"n": len(pairs), "games": len(set(games)), "answer": which,
                     "points_diff": round(sum(dp) / len(dp), 1), "points_ci95": _boot_ci(dp, games),
                     "top1_diff": round(sum(d1) / len(d1), 4), "top1_ci95": _boot_ci(d1, games),
                     "logp_diff": round(sum(dl) / len(dl), 4), "logp_ci95": _boot_ci(dl, games)}
    return out


# ------------------------------------------------------------------ locator server / node runner

def health(url, timeout=5):
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=timeout) as r:
            return json.loads(r.read().decode()).get("ok") is True
    except Exception:
        return False


ENGINE_DATA = ("areas.json", "world_regions.npz", "world_countries.png", "maps.json", "country_rules.json")


def model_fingerprint(root=ROOT, head=True):
    """Model data + engine sources of the server's tree (sha1), the other data files the engine reads ("data": areas,
    region / country rasters, maps, country rules) and git HEAD (or the frozen copy's commit)."""
    h = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(root, "data", "model", "*.json")) + [os.path.join(root, "data", "fusion.json")]):
        if os.path.basename(p).startswith("eval"):
            continue
        try:
            h.update(open(p, "rb").read())
        except OSError:
            pass
    for p in sorted(glob.glob(os.path.join(root, "data", "model", "*.npz"))):
        with open(p, "rb") as f:
            h.update(hashlib.sha1(f.read()).digest())
    e = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(root, "engine", "**", "*.py"), recursive=True)):
        with open(p, "rb") as f:
            e.update(f.read())
    x = hashlib.sha1()
    for name in ENGINE_DATA:
        try:
            with open(os.path.join(root, "data", name), "rb") as f:
                x.update(name.encode() + hashlib.sha1(f.read()).digest())
        except OSError:
            pass
    out = {"model": h.hexdigest()[:12], "engine": e.hexdigest()[:12], "data": x.hexdigest()[:12]}
    if not head:
        return out
    frozen = read_json(os.path.join(root, "FROZEN.json"))
    if frozen:
        out["head"] = "frozen " + str(frozen.get("commit"))
    else:
        try:
            out["head"] = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True,
                                         text=True, timeout=10).stdout.strip()
        except Exception:
            out["head"] = None
    return out


def same_tree(a, b):
    return bool(a and b) and all(a.get(k) == b.get(k) for k in ("model", "engine", "data"))


def tree_quiet_s(root=ROOT):
    """Seconds since the last change of a file the locator server loads (engine/, web/, data/model/, data/*); the
    evaluation reports data/model/eval*.json are not loaded (and not in model_fingerprint), so they do not count."""
    newest = 0.0
    for pat in ("engine/**/*.py", "web/**/*", "data/model/*", "data/*.json", "data/*.npz", "data/*.png"):
        for f in glob.glob(os.path.join(root, pat), recursive=True):
            if os.path.basename(f).startswith("eval") and os.path.dirname(f).endswith(os.path.join("data", "model")):
                continue
            try:
                newest = max(newest, os.path.getmtime(f))
            except OSError:
                pass
    return time.time() - newest


def snapshot_tree(dest_root=None):
    """A copy of the working tree's engine/, web/ and data/ - the current saved model and engine, uncommitted edits
    included - for the run's own locator server (scratch/world/tree_<model>_<engine>_<data>/), so the games are not
    affected by edits made while they are played; FROZEN.json names it.  None when the files changed during the copy
    (another workstream writing them: try again later)."""
    dest_root = dest_root or WORK
    fp = model_fingerprint(ROOT, head=False)
    dest = os.path.join(dest_root, "tree_%s_%s_%s" % (fp["model"], fp["engine"], fp["data"]))
    if read_json(os.path.join(dest, "FROZEN.json")):
        return dest
    tmp = dest + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp*", "*.part")
    for sub in ("engine", "web", "data"):
        shutil.copytree(os.path.join(ROOT, sub), os.path.join(tmp, sub), ignore=ign, symlinks=True)
    hud = os.path.join(ROOT, "scratch", "hud_img")
    if os.path.isdir(hud):
        os.makedirs(os.path.join(tmp, "scratch"), exist_ok=True)
        os.symlink(hud, os.path.join(tmp, "scratch", "hud_img"))
    if not same_tree(model_fingerprint(tmp, head=False), fp) or not same_tree(model_fingerprint(ROOT, head=False), fp):
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                          timeout=10).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "engine", "web", "data"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
    with open(os.path.join(tmp, "FROZEN.json"), "w") as f:
        json.dump({"commit": head + ("+worktree" if dirty else ""), "snapshot": True, "made_at": now_iso(),
                   "fingerprint": fp}, f)
    os.rename(tmp, dest)
    return dest


def smoke_test(url, st, root=ROOT, post=None):
    """One saved start capture of a finished game through the server (local request): True when it answers with
    countries and a guess (an engine / model under edit may load and still fail on a panorama)."""
    post = post or post_json
    for g in reversed(st.get("games") or []):
        for r in (g.get("rounds") or []) if g.get("status") == "finished" else []:
            for d in (r.get("dirs") or [])[:1]:
                body = capture_body(os.path.join(root, d), use_map=True)
                if body is None:
                    continue
                try:
                    res = post(url.rstrip("/") + "/api/predict", body)
                except Exception:
                    return False
                return bool(res.get("countries")) and bool(res.get("guess"))
    return True   # nothing to test with


def freeze_tree(commit, dest):
    """A frozen copy of engine/, web/ and data/ at `commit` (git archive) for the run's own locator server, so a
    long run is not affected by engine / model edits in the working tree; FROZEN.json names the commit."""
    if read_json(os.path.join(dest, "FROZEN.json")):
        return dest
    os.makedirs(dest, exist_ok=True)
    arch = subprocess.run(["git", "-C", ROOT, "archive", commit, "engine", "web", "data"], capture_output=True, timeout=300)
    if arch.returncode:
        raise Stop("git archive %s failed: %s" % (commit, arch.stderr.decode()[-200:]))
    subprocess.run(["tar", "-x", "-C", dest], input=arch.stdout, check=True, timeout=300)
    hud = os.path.join(ROOT, "scratch", "hud_img")
    if os.path.isdir(hud) and not os.path.exists(os.path.join(dest, "scratch", "hud_img")):
        os.makedirs(os.path.join(dest, "scratch"), exist_ok=True)
        os.symlink(hud, os.path.join(dest, "scratch", "hud_img"))
    full = subprocess.run(["git", "-C", ROOT, "rev-parse", commit], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(dest, "FROZEN.json"), "w") as f:
        json.dump({"commit": full[:12], "made_at": now_iso()}, f)
    return dest


def start_server(port, log, root=ROOT):
    """The run's own locator server (web/server.py of `root`) on port; (process, url); process None when
    something already answers there (then the model it runs is not known: fingerprint "external")."""
    url = "http://127.0.0.1:%d" % port
    if health(url):
        log("locator server already answers on %s (not started here: model fingerprint unknown)" % url)
        return None, url
    os.makedirs(os.path.join(WORK, "logs"), exist_ok=True)
    out = open(os.path.join(WORK, "logs", "server_%d.log" % port), "a")
    p = subprocess.Popen([sys.executable, os.path.join(root, "web", "server.py"), "--port", str(port)], cwd=root,
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(240):
        if p.poll() is not None:
            raise Stop("locator server on port %d exited (%s)" % (port, p.returncode))
        if health(url):
            log("locator server started on %s (pid %d)" % (url, p.pid))
            return p, url
        time.sleep(1)
    raise Stop("locator server on port %d did not come up" % port)


def mem_available_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def wait_memory(min_mb, log, max_wait_s=900):
    t0 = time.time()
    while mem_available_mb() < min_mb:
        if time.time() - t0 > max_wait_s:
            raise Stop("less than %d MB available for %d s" % (min_mb, max_wait_s))
        log("waiting for memory: %d MB available, %d needed" % (mem_available_mb(), min_mb))
        time.sleep(30)


def kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(20):
        if p.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_node(token, walk, server_url, summary_path, log_path, max_minutes, walk_clicks=1):
    args = ["node", NODE_SCRIPT, "--game", token, "--submit", "--headless", "--exit-when-done",
            "--summary", summary_path, "--max-minutes", str(max_minutes)]
    if walk:
        args += ["--test-walk", str(walk), "--test-walk-clicks", str(walk_clicks)]
    env = dict(os.environ, LOCATOR_URL=server_url)
    with open(log_path, "a") as out:
        out.write("\n==== %s %s\n" % (now_iso(), " ".join(args[1:])))
        out.flush()
        p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=max_minutes * 60 + 120)
        except subprocess.TimeoutExpired:
            kill_group(p)
            return None
        finally:
            kill_group(p)   # chrome children of a node process that exited
    return p.returncode


def move_captures(token, dest_root=None, src=LIVE_OUT):
    """scratch/live/<token>_r* -> scratch/world/live/<token>/ (returns the destination)."""
    dest = os.path.join(dest_root or os.path.join(WORK, "live"), token)
    os.makedirs(dest, exist_ok=True)
    for d in glob.glob(os.path.join(src, token + "_r*")):
        target = os.path.join(dest, os.path.basename(d))
        if os.path.exists(target):  # a resumed run captured the round again: keep both
            target += "_%d" % int(time.time())
        shutil.move(d, target)
    return dest


# ------------------------------------------------------------------ clues, panoramas, features

def fetch_clues(api, pano_ids, log):
    """GeoGuessr's clue placements of finished rounds -> data/calibration/pano_clues.json (format of
    tools/fetch_pano_clues.py).  Returns the number of panoramas queried."""
    import fetch_pano_clues as fpc
    out = fpc.load()
    todo = [p for p in dict.fromkeys(pano_ids) if p and p not in out]
    n = 0
    try:
        for pid in todo:
            st, d = api.call("GET", SITE + "/api/v4/clues/" + urllib.parse.quote(pid, safe=""))
            if st == 200 and isinstance(d, list):
                out[pid] = [fpc.normalise(p) for p in d if p.get("panoId", pid) == pid]
                n += 1
    finally:
        if n:
            fpc.save(out)
    if todo:
        log("clues: %d of %d panoramas queried" % (n, len(todo)))
    return n


def download_panoramas(log):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_dataset.py"), "world_play", "-n", "0",
                        "--threads", "4"], cwd=ROOT, capture_output=True, text=True, timeout=3600)
    for line in (r.stdout + r.stderr).strip().splitlines()[-4:]:
        log("dataset: " + line)
    return r.returncode


def world_records(dataset=DATASET):
    recs = []
    try:
        with open(os.path.join(dataset, "index.jsonl")) as f:
            for line in f:
                if '"world_play"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("mode") == "world_play" and os.path.exists(os.path.join(dataset, "panos", r["pano_id"] + ".jpg")):
                    recs.append(r)
    except OSError:
        pass
    return recs


def compute_features(log, workers=2):
    """Features of the world_play panoramas through tools/calibrate.py's code path (_init / _extract_one /
    upsert_features), only for modules whose cache was built by the current module version: a changed module
    needs the full `calibrate.py features` run anyway, which then includes these panoramas."""
    from multiprocessing import Pool
    import calibrate as C
    from engine import features
    recs = world_records()
    mods, hashes, names, cached = [], {}, {}, {}
    for m in features.available():
        h = C.module_hash(m)
        c = C.read_cache(m.NAME)
        if c is None or c["names"] != list(m.FEATURE_NAMES) or not C.hash_ok(m.NAME, c["hash"], h):
            log("features: %s cache is not of the current module version: skipped (run calibrate.py features)" % m.NAME)
            continue
        mods.append(m)
        hashes[m.NAME], names[m.NAME] = h, list(m.FEATURE_NAMES)
        cached[m.NAME] = set(c["ids"])
    need = [r for r in recs if any(r["pano_id"] not in cached[m.NAME] for m in mods)]
    if not need or not mods:
        return 0
    wait_memory(3000, log)
    jobs = [(C.pano_path(r["pano_id"]), r.get("heading")) for r in need]
    rows = {m.NAME: {} for m in mods}
    with Pool(workers, initializer=C._init, initargs=([m.NAME for m in mods],)) as pool:
        for r, res in zip(need, pool.imap(C._extract_one, jobs, chunksize=2)):
            for m, x in zip(mods, res):
                rows[m.NAME][r["pano_id"]] = x
    C.upsert_features(rows, hashes, names)
    log("features: %d world_play panoramas x %d modules" % (len(need), len(mods)))
    return len(need)


# ------------------------------------------------------------------ disk

def du_bytes(path):
    total = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def disk_used(dataset=DATASET):
    panos = sum(os.path.getsize(os.path.join(dataset, "panos", r["pano_id"] + ".jpg")) for r in world_records(dataset))
    return du_bytes(WORK) + panos


def prune_move_views(st, log, keep_bytes, quality=80, delete_bytes=None, keep_screens=10):
    """Makes room (oldest games first; the start captures, views.json and response.json stay exact): the move
    captures' view images re-encoded as JPEG quality 80 until keep_bytes (game field move_views_quality); then, while
    still over delete_bytes (default keep_bytes), the page screenshots (hud_r*, result_r*) of all but the last
    keep_screens games (screens_deleted) and the move captures' view images (move_views_deleted) are deleted.
    Returns the bytes freed."""
    from PIL import Image
    delete_bytes = keep_bytes if delete_bytes is None else delete_bytes
    freed = 0
    games = st["games"]
    for step, target in (("requality", keep_bytes), ("screens", delete_bytes), ("delete", delete_bytes)):
        for gi, g in enumerate(games):
            if disk_used() <= target:
                break
            if step == "screens":
                if g.get("screens_deleted") or gi >= len(games) - keep_screens:
                    continue
                files = glob.glob(os.path.join(WORK, "live", g["token"], "*_r*", "hud_r*.jpg")) + \
                    glob.glob(os.path.join(WORK, "live", g["token"], "*_r*", "result_r*.jpg"))
            else:
                if g.get("move_views_deleted") or (step == "requality" and g.get("move_views_quality")):
                    continue
                files = glob.glob(os.path.join(WORK, "live", g["token"], "*_r*_m*", "view_*.jpg"))
            for f in sorted(files):
                n0 = os.path.getsize(f)
                if step == "requality":
                    Image.open(f).convert("RGB").save(f + ".tmp", "JPEG", quality=quality)
                    os.replace(f + ".tmp", f)
                else:
                    os.remove(f)
                freed += n0 - (os.path.getsize(f) if os.path.exists(f) else 0)
            if g.get("status") in ("finished", "abandoned"):
                g[{"requality": "move_views_quality", "screens": "screens_deleted", "delete": "move_views_deleted"}[step]] = \
                    quality if step == "requality" else True
    if freed:
        save_state(st)
        log("disk: %.0f MB freed in move-capture view images / screenshots" % (freed / 1e6))
    return freed


# ------------------------------------------------------------------ report

def _mean_ci(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return {"n": len(xs), "mean": round(m, 1)}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    half = 1.96 * sd / math.sqrt(len(xs))
    return {"n": len(xs), "mean": round(m, 1), "ci95": [round(m - half, 1), round(m + half, 1)]}


def _rank(pairs, code):
    codes = [p[0] for p in pairs or []]
    return codes.index(code) if code in codes else None


def _acc(rows, which, field, truth_key, ks=(1, 3)):
    out = {}
    ok = [r for r in rows if r.get(which)]
    if not ok:
        return None
    for k in ks:
        hit = sum(1 for r in ok if _rank(r[which].get(field), r.get(truth_key)) is not None
                  and _rank(r[which].get(field), r.get(truth_key)) < k)
        out["top%d" % k] = round(hit / len(ok), 4)
    out["n"] = len(ok)
    return out


def _region_given_country(rows, which, truth="region", field="top_country_regions"):
    ok = [r for r in rows if r.get(which) and r.get(truth) and r[which].get("countries")
          and r[which]["countries"][0][0] == r.get("country")]
    if truth == "area":   # answers made by an engine with areas only
        ok = [r for r in ok if r[which].get(field) is not None]
    if not ok:
        return None
    out = {"n": len(ok)}
    ranks = [_rank(r[which].get(field), r[truth]) for r in ok]
    for k in (1, 3):
        out["top%d" % k] = round(sum(1 for x in ranks if x is not None and x < k) / len(ok), 4)
    return out


def report(st, extra=None):
    games = st["games"]
    # games flagged "exclude" (e.g. validation runs against a server of another engine state) keep their truth
    # (dataset, clues) but stay out of the numbers
    fin = [g for g in games if g.get("status") == "finished" and not g.get("exclude")]
    every = [dict(r, walk=g.get("walk", 0), split=g.get("split"), final=r.get("final") or r.get("start"))
             for g in fin for r in g.get("rounds") or []]
    rows = [r for r in every if not r.get("blind")]   # blind guesses (panorama never loaded) are not answers
    clues = {}
    try:
        clues = read_json(CLUES_FILE) or {}
    except (OSError, ValueError):
        pass
    out = {"updated_at": now_iso(),
           "games": {"total": len(games), "finished": len(fin),
                     "nomove": sum(1 for g in fin if not g.get("walk")), "moving": sum(1 for g in fin if g.get("walk")),
                     "abandoned": sum(1 for g in games if g.get("status") == "abandoned"),
                     "excluded": sum(1 for g in games if g.get("status") == "finished" and g.get("exclude")),
                     "in_progress": sum(1 for g in games if g.get("status") in ("created", "played"))},
           "rounds": len(rows), "blind_rounds": len(every) - len(rows)}
    by = {"nomove": [r for r in rows if not r["walk"]], "moving": [r for r in rows if r["walk"]], "all": rows}
    out["points_submitted"] = {k: _mean_ci([r.get("points") for r in v]) for k, v in by.items()}
    out["points_start_capture_only"] = {k: _mean_ci([(r.get("start") or {}).get("points") for r in v]) for k, v in by.items()}
    out["country_start"] = {k: _acc(v, "start", "countries", "country") for k, v in by.items()}
    out["country_final"] = {k: _acc(v, "final", "countries", "country") for k, v in by.items()}
    out["region_overall_final"] = {k: _acc(v, "final", "regions", "region") for k, v in by.items()}
    out["region_given_right_country_final"] = {k: _region_given_country(v, "final") for k, v in by.items()}
    out["region_given_right_country_start"] = {k: _region_given_country(v, "start") for k, v in by.items()}
    out["area_given_right_country_final"] = {k: _region_given_country(v, "final", "area", "top_country_areas")
                                             for k, v in by.items()}
    for which in ("start", "final"):
        out["logp_true_" + which] = {k: _mean_ci([math.log(max(r[which]["p_true"], 1e-6)) for r in v
                                                  if r.get(which) and r[which].get("p_true") is not None]) for k, v in by.items()}
    # wrong top country that the map prior favoured over the true one (the Cambodia -> India pattern)
    wrong = [r for r in rows if r.get("final") and r["final"].get("rank_true") not in (0, None)
             and r["final"].get("prior_top") is not None and r["final"].get("prior_true") is not None]
    out["wrong_top1"] = {"n": len(wrong), "prior_favoured_top_over_truth": sum(
        1 for r in wrong if r["final"]["prior_top"] > r["final"]["prior_true"])}
    out["n_fused_moving"] = dict(collections.Counter(r.get("n_fused") for r in by["moving"]))
    out["moving_captures"] = {"rounds": len(by["moving"]), "captures": sum(r.get("n_captures") or 0 for r in by["moving"]),
                              "replaced_as_same_panorama": sum(r.get("n_replaced") or 0 for r in by["moving"]),
                              "steps_walked": _mean_ci([r.get("steps") for r in by["moving"]])}
    for k, v in by.items():   # what walking changed: final (fused) minus start capture, same rounds
        d = [(r["final"] or {}).get("points", 0) - (r["start"] or {}).get("points", 0) for r in v
             if r.get("start") and r.get("final") and r["start"].get("points") is not None]
        if k == "moving":
            out["moving_gain_points_final_minus_start"] = _mean_ci(d)
    out["splits"] = dict(collections.Counter(g.get("split") for g in fin))
    cc = collections.Counter(r.get("country") for r in rows)
    out["world_country_distribution"] = cc.most_common()
    out["countries_seen"] = len(cc)
    with_clue = [r for r in rows if r.get("pano_id") in clues]
    out["clues"] = {"rounds_queried": len(with_clue), "rounds_with_placements": sum(1 for r in with_clue if clues[r["pano_id"]]),
                    "placements": sum(len(clues[r["pano_id"]]) for r in with_clue)}
    out["label_disagreements_gg_vs_raster"] = sum(1 for r in rows if r.get("gg_country") and r.get("raster_country")
                                                  and LABEL_FIX.get(r["gg_country"], r["gg_country"]) != r["raster_country"])
    wr = world_records()
    out["dataset"] = {"panoramas": len(wr), "by_split": dict(collections.Counter(
        "world_calib" if split_of_game(r.get("game")) == "world_calib" else "world_test" for r in wr))}
    out["models"] = dict(collections.Counter(json.dumps(g.get("model"), sort_keys=True) for g in fin))
    out["per_game"] = [{"token": g["token"], "walk": g.get("walk"), "split": g.get("split"),
                        "total": sum((r.get("points") or 0) for r in g.get("rounds") or []),
                        "countries": [r.get("country") for r in g.get("rounds") or []],
                        "top1": [((r.get("final") or r.get("start") or {}).get("countries") or [[None]])[0][0]
                                 for r in g.get("rounds") or []]}
                       for g in fin]
    if extra:
        out.update(extra)
    return out


def write_progress(st, extra=None):
    os.makedirs(WORK, exist_ok=True)
    rep = report(st, extra)
    tmp = os.path.join(WORK, "progress.json.tmp")
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, os.path.join(WORK, "progress.json"))
    return rep


# ------------------------------------------------------------------ play loop

class Runner:
    def __init__(self, args, api=None, log=None):
        self.args = args
        self.api = api or Api(gap=max(MIN_GAP_S, args.api_gap))
        os.makedirs(os.path.join(WORK, "logs"), exist_ok=True)
        self.logf = open(os.path.join(WORK, "logs", "world_play.log"), "a")
        self.log = log or self._log
        self.server = None
        self.url = None
        self.fingerprint = None
        self.tree = None        # --snapshot: the copy of the working tree the server runs
        self.rejected = None    # a working-tree state whose snapshot failed the smoke test
        self.waiting_for = None

    def _log(self, msg):
        line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def tree_root(self):
        """The tree the run's locator server runs: a frozen commit (--freeze), a snapshot of the working tree
        (--snapshot: the current saved model), else the working tree itself."""
        if self.args.freeze:
            return freeze_tree(self.args.freeze, os.path.join(WORK, "head_" + self.args.freeze))
        if not self.args.snapshot:
            return ROOT
        if not self.tree and self.args.refresh_min > 0 and tree_quiet_s() < self.args.refresh_min * 60:
            # the working tree is being edited: start from the newest earlier snapshot (moved on by maybe_refresh)
            prev = sorted((d for d in glob.glob(os.path.join(WORK, "tree_*")) if read_json(os.path.join(d, "FROZEN.json"))),
                          key=lambda d: read_json(os.path.join(d, "FROZEN.json")).get("made_at") or "")
            if prev:
                self.tree = prev[-1]
                self.log("working tree changed in the last %g min: starting from the snapshot %s"
                         % (self.args.refresh_min, os.path.relpath(self.tree, ROOT)))
        for _ in range(10):
            if self.tree:
                return self.tree
            self.tree = snapshot_tree()
            if not self.tree:
                self.log("snapshot: the working tree changed during the copy: again in 60 s")
                time.sleep(60)
        raise Stop("could not snapshot the working tree (files kept changing)")

    def ensure_server(self):
        if self.url and health(self.url):
            return
        if self.server is not None and self.server.poll() is None:
            kill_group(self.server)
        root = self.tree_root()
        self.server, self.url = start_server(self.args.port, self.log, root)
        self.fingerprint = model_fingerprint(root) if self.server is not None else {"external": self.url}
        self.log("locator tree %s, fingerprint %s" % (os.path.relpath(root, ROOT), self.fingerprint))
        if self.server is not None and not smoke_test(self.url, load_state()):
            kill_group(self.server)
            self.server = self.url = None
            raise Stop("locator server of %s failed the smoke test (a saved capture)" % os.path.relpath(root, ROOT))

    def maybe_refresh(self):
        """--snapshot with --refresh-min M: between games, when the working tree's model / engine differ from the
        snapshot and none of their files changed for M minutes (a refit finished), the server moves to a new snapshot;
        a snapshot whose server fails to start or to answer a saved capture is rejected (the old one stays)."""
        if not (self.args.snapshot and self.args.refresh_min > 0 and self.tree):
            return
        now = model_fingerprint(ROOT, head=False)
        cur = (read_json(os.path.join(self.tree, "FROZEN.json")) or {}).get("fingerprint")
        if same_tree(now, cur) or same_tree(now, self.rejected):
            return
        if tree_quiet_s() < self.args.refresh_min * 60:
            if not same_tree(now, self.waiting_for):
                self.waiting_for = now
                self.log("working tree changed (%s): the server moves to it once it is unchanged for %g min"
                         % (now, self.args.refresh_min))
            return
        new = snapshot_tree()
        if not new or new == self.tree:
            return
        old = self.tree
        if self.server is not None:
            kill_group(self.server)
        self.server = self.url = None
        self.tree = new
        try:
            self.ensure_server()
            self.log("locator server moved to the working tree's new state %s" % os.path.relpath(new, ROOT))
            for d in glob.glob(os.path.join(WORK, "tree_*")):   # keep the new and the previous copy (an evaluation
                if os.path.realpath(d) not in (os.path.realpath(new), os.path.realpath(old)):   # may still use it)
                    shutil.rmtree(d, ignore_errors=True)
        except Stop as e:
            self.log("new working-tree state rejected (%s): staying with %s" % (e, os.path.relpath(old, ROOT)))
            self.rejected = now
            if self.server is not None:
                kill_group(self.server)
            self.server = self.url = None
            self.tree = old
            shutil.rmtree(new, ignore_errors=True)
            self.ensure_server()

    def check_disk(self, st):
        limit = self.args.max_disk_gb * 1e9
        used = disk_used()
        if used > 0.85 * limit:
            prune_move_views(st, self.log, int(0.75 * limit), delete_bytes=int(0.92 * limit))
            used = disk_used()
        if used > limit:
            raise Stop("disk: %.2f GB used by this workstream (limit %.2f GB)" % (used / 1e9, self.args.max_disk_gb))
        return used

    def play_one(self, st, g):
        """Runs (or resumes) the game in node until finished; status -> played or (after 2 runs) abandoned."""
        tok = g["token"]
        summary = os.path.join(WORK, "summaries", tok + ".json")
        os.makedirs(os.path.dirname(summary), exist_ok=True)
        while g.get("status") == "created":
            if g.get("attempts", 0) >= self.args.max_attempts:
                g["status"] = "abandoned"
                self.log("game %s abandoned after %d runs (never read)" % (tok, g["attempts"]))
                save_state(st)
                return
            wait_memory(self.args.min_avail_mb, self.log)
            self.ensure_server()
            g["attempts"] = g.get("attempts", 0) + 1
            g.setdefault("models", []).append(self.fingerprint)
            g["model"] = self.fingerprint
            save_state(st)
            t0 = time.time()
            self.log("game %d %s: %s, run %d" % (g["n"], tok, "Moving (walk %d)" % g["walk"] if g["walk"] else "No Move", g["attempts"]))
            code = run_node(tok, g["walk"], self.url, summary, os.path.join(WORK, "logs", tok + ".log"),
                            self.args.max_minutes, g.get("walk_clicks", 1))
            s = read_json(summary) or {}
            g["summary"] = {k: s.get(k) for k in ("finished", "end", "total")}
            g["summary"]["rounds"] = s.get("rounds") or []
            g["play_s"] = round(g.get("play_s", 0) + time.time() - t0)
            self.log("game %s: node exit %s, %s, total %s" % (tok, code, s.get("end"), s.get("total")))
            if s.get("finished"):
                g["status"] = "played"
            save_state(st)
            end = str(s.get("end") or "")
            if "429" in end or "401" in end or "403" in end:
                raise Stop("node: " + end)

    def finish_one(self, st, g):
        tok = g["token"]
        try:
            d = fetch_finished(self.api, g)
        except (RuntimeError, ValueError) as e:  # left as played: fetched by a later run
            self.log("game %s: %s" % (tok, e))
            return
        dest = move_captures(tok)
        g["rounds"] = round_records(d, g.get("summary"), lambda i: capture_dirs(dest, tok, i))
        g["total"] = sum((r.get("points") or 0) for r in g["rounds"])
        g["status"] = "finished"
        g["finished_at"] = now_iso()
        save_state(st)
        tops = ", ".join("%s>%s %d" % (r["country"], ((r.get("final") or r.get("start") or {}).get("countries") or [[None]])[0][0],
                                         r.get("points") or 0) for r in g["rounds"])
        self.log("game %s finished: %d points (%s)" % (tok, g["total"], tops))

    def clue_ids(self, st):
        """Panoramas whose GeoGuessr clue placements are fetched (--clues): all rounds, or (sample) the first round
        of each finished game and all rounds of a game where a placement was found, or none.  The classic World games
        came back without placements (0 of 195 rounds) while the user's earlier World game had them: the sample
        notices if that changes without a request per round."""
        if self.args.clues == "none":
            return []
        have = (read_json(CLUES_FILE) or {}) if self.args.clues == "sample" else {}
        ids = []
        for g in st["games"]:
            rs = [r for r in g.get("rounds") or [] if r.get("pano_id")] if g.get("status") == "finished" else []
            if self.args.clues == "all" or any(have.get(r["pano_id"]) for r in rs):
                ids += [r["pano_id"] for r in rs]
            elif rs:
                ids.append(rs[0]["pano_id"])
        return ids

    def collect(self, st, panoramas=True):
        fetch_clues(self.api, self.clue_ids(st), self.log)
        if panoramas:
            download_panoramas(self.log)
            try:
                compute_features(self.log, workers=self.args.workers)
            except Exception as e:  # features can be rebuilt later; the games go on
                self.log("features: %r" % (e,))

    def run(self):
        st = load_state()
        t_end = time.time() + self.args.hours * 3600
        played_now = fails = 0
        try:
            self.check_disk(st)
            # resume what a crash left behind
            resumed = False
            for g in st["games"]:
                if g.get("status") == "created":
                    self.play_one(st, g)
                    resumed = True
                if g.get("status") == "played":
                    self.finish_one(st, g)
            if resumed:
                self.collect(st, panoramas=False)
                time.sleep(self.args.gap)
            while played_now < self.args.games and time.time() < t_end:
                if os.path.exists(os.path.join(WORK, "STOP")):
                    self.log("STOP file: stopping")
                    break
                self.check_disk(st)
                wait_memory(self.args.min_avail_mb, self.log)
                self.maybe_refresh()
                self.ensure_server()
                walk = next_walk(st["games"], self.args.walk)
                try:
                    meta = create_game(self.api)
                except RuntimeError as e:
                    fails += 1
                    self.log("create: %s" % e)
                    if fails >= 3:
                        raise Stop("3 failed game creations in a row")
                    time.sleep(self.args.gap)
                    continue
                fails = 0
                g = {"n": len(st["games"]) + 1, "token": meta["token"], "walk": walk,
                     "walk_clicks": self.args.walk_clicks if walk else 0, "status": "created",
                     "split": split_of_game(meta["token"]), "created_at": now_iso(), "settings": meta, "attempts": 0}
                st["games"].append(g)
                save_state(st)
                self.play_one(st, g)
                if g["status"] == "played":
                    self.finish_one(st, g)
                played_now += 1
                last = played_now >= self.args.games or time.time() >= t_end
                self.collect(st, panoramas=last or played_now % self.args.collect_every == 0)
                write_progress(st, {"run": {"played_this_run": played_now, "hours_left": round((t_end - time.time()) / 3600, 2),
                                            "disk_mb": round(disk_used() / 1e6), "fingerprint": self.fingerprint}})
                if not last:
                    time.sleep(self.args.gap)
        except Stop as e:
            self.log("STOP: %s" % e)
            st = load_state()
        finally:
            try:
                write_progress(load_state(), {"run": {"played_this_run": played_now, "ended_at": now_iso(),
                                                      "disk_mb": round(disk_used() / 1e6), "fingerprint": self.fingerprint}})
            except Exception as e:
                self.log("progress: %r" % (e,))
            if self.server is not None and not self.args.keep_server:
                kill_group(self.server)
        return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["play", "collect", "features", "report", "rebuild", "replay"])
    ap.add_argument("--server", default="", help="replay: locator server URL to evaluate (e.g. http://127.0.0.1:8098)")
    ap.add_argument("--split", default="world_test", help="replay: world_calib, world_test or both (comma-separated)")
    ap.add_argument("--no-map", action="store_true", help="replay: send no map settings")
    ap.add_argument("--threads", type=int, default=1, help="replay: rounds sent at a time")
    ap.add_argument("--out", default="", help="replay: output JSON (default scratch/world/replay_<split>.json)")
    ap.add_argument("--paired", default="", help="replay: an earlier replay output of the same split (another "
                    "server): paired differences on the same rounds")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--validate", type=int, default=0, help="play: stop after N games (end-to-end check)")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--walk", type=int, default=3, help="steps of the Moving games (--test-walk)")
    ap.add_argument("--walk-clicks", type=int, default=4, help="moves per step before the player stops "
                    "(--test-walk-clicks; ~10 m each, 4 ~ the 45 m spacing of tools/moving_calib.py)")
    ap.add_argument("--gap", type=float, default=20.0, help="seconds between games (>= 20)")
    ap.add_argument("--api-gap", type=float, default=MIN_GAP_S)
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--keep-server", action="store_true")
    ap.add_argument("--freeze", default="", help="run the locator server from a frozen copy of this commit "
                    "(git archive of engine/ web/ data/ -> scratch/world/head_<commit>/), e.g. 89a558e")
    ap.add_argument("--snapshot", action="store_true", help="run the locator server from a copy of the working tree's "
                    "engine/ web/ data/ (the current saved model) -> scratch/world/tree_<fingerprint>/")
    ap.add_argument("--refresh-min", type=float, default=15.0, help="--snapshot: move to a new snapshot between games "
                    "when the working tree's model / engine changed and stayed unchanged this many minutes (0: never)")
    ap.add_argument("--clues", default="sample", choices=("all", "sample", "none"),
                    help="GeoGuessr clue placements of the finished rounds: all, sample (first round per game, all "
                    "rounds of games with placements), none")
    ap.add_argument("--max-minutes", type=float, default=20.0, help="per run of one game")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--collect-every", type=int, default=10, help="games between panorama / feature batches")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-avail-mb", type=int, default=2500)
    ap.add_argument("--max-disk-gb", type=float, default=1.9)
    args = ap.parse_args()
    args.gap = max(20.0, args.gap)
    if args.validate:
        args.games = args.validate
    if args.port in (8080,):
        sys.exit("port 8080 is the user's locator server: use 8097 / 8098 / 8099")
    if args.freeze and args.snapshot:
        sys.exit("--freeze and --snapshot exclude each other")
    if args.cmd == "play":
        Runner(args).run()
    elif args.cmd == "collect":
        r = Runner(args)
        try:
            r.collect(load_state())
        except Stop as e:
            r.log("STOP: %s" % e)
        write_progress(load_state())
    elif args.cmd == "features":
        compute_features(print, workers=args.workers)
    elif args.cmd == "replay":
        if not args.server:
            sys.exit("replay needs --server (a locator server; never the user's 8080 while they play)")
        splits = tuple(x.strip() for x in args.split.split(",") if x.strip())
        res = replay_eval(load_state(), args.server, splits, use_map=not args.no_map, threads=args.threads)
        try:
            import urllib.request
            with urllib.request.urlopen(args.server.rstrip("/") + "/api/health", timeout=10) as r:
                res["server_health"] = json.loads(r.read().decode())
        except Exception:
            pass
        if args.paired:   # an earlier replay of the same split on another server (e.g. the frozen baseline)
            res["summary"]["paired"] = dict(replay_paired(res["rows"], (read_json(args.paired) or {}).get("rows") or []),
                                            other=os.path.relpath(os.path.abspath(args.paired), ROOT))
        out = args.out or os.path.join(WORK, "replay_%s.json" % "_".join(splits))
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res["summary"], indent=1))
        print("saved", out)
    elif args.cmd == "rebuild":   # not while a play run is writing world_games.json
        st = load_state()
        print("%d rounds updated" % rebuild_digests(st))
        save_state(st)
        write_progress(st)
    else:
        rep = write_progress(load_state())
        print(json.dumps({k: rep[k] for k in rep if k not in ("per_game", "world_country_distribution")}, indent=1))


if __name__ == "__main__":
    main()
