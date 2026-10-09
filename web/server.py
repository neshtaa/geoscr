#!/usr/bin/env python3
"""
Local HTTP server for the pure-math locator (used by play_live_visual.js and the web page).

  python3 web/server.py [--port 8080]

POST /api/predict  JSON, one of:
  {"views": [{"image_b64": "...", "yaw": 0, "pitch": 0, "hfov": 100}, ...]}   yaw = true azimuth (deg)
  {"image_b64": "...", "heading": 123.0, "hfov": 100}                          panorama (2:1) or screenshot
  {"pano_id": "..."} | {"lat": 50.4, "lng": 30.5}                               official Street View (tests)
  optional "map": {"id", "name", "bounds": {"min": {"lat", "lng"}, "max": {...}}, "maxErrorDistance"}
      (all optional; missing fields from data/maps.json) -> map prior, score scale and bounds;
      invalid bounds / maxErrorDistance are dropped and listed in the result's "warnings";
  optional "debug_dir": save the reconstructed sphere there as sphere.jpg (inside the project
      or the temp directory; relative paths are relative to the project)
POST /api/evaluate_round  {"lat", "lng", "guess_lat", "guess_lng", "pred_code", "map"?}
GET  /api/maps     known maps (data/maps.json)
GET  /api/health
GET  /hud/<file>   the live HUD files (web/hud/: hud.js, hud.css)
GET  /hud/img?u=<image_url>   a clue-card image from the offline cache scratch/hud_img/
      (tools/fetch_hud_images.py); never fetched from the network here
CORS: only the Chrome extension (chrome-extension://...) and local pages (http://localhost:*,
http://127.0.0.1:*); other web origins, www.geoguessr.com included, get no CORS headers.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import sys
import tempfile
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.geo import (country_at, geoguessr_points, haversine_km, load_maps, parse_bounds,  # noqa: E402
                        positive_float, resolve_map)
from engine.locator import distance_report, get_locator  # noqa: E402

LOCK = threading.Lock()
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
HUD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hud")
HUD_TYPES = {".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml"}
HUD_IMG_DIR = os.path.join(ROOT, "scratch", "hud_img")
HUD_IMG_TYPES = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _img(b64):
    if "," in b64[:100]:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def map_from_request(req, warnings=None):
    """The optional "map" of a request -> map_info for the locator (None when absent).
    A string is taken as an id / slug / name.  Invalid bounds (also boxes crossing the
    antimeridian) and maxErrorDistance values are dropped (data/maps.json values apply instead)
    with a note in warnings, so a round keeps its prediction."""
    m = req.get("map")
    if m is None or m == "" or m == {}:
        return None
    if isinstance(m, str):
        return m
    if not isinstance(m, dict):
        raise ValueError("map must be an object {id, name, bounds, maxErrorDistance}")
    warnings = [] if warnings is None else warnings
    out = {k: str(m[k]) for k in ("id", "slug", "name") if m.get(k)}
    if m.get("bounds") is not None:
        if parse_bounds(m["bounds"]) is None:
            warnings.append("map.bounds ignored: expected {min: {lat, lng}, max: {lat, lng}} with min <= max")
        else:
            out["bounds"] = m["bounds"]
    if m.get("maxErrorDistance") is not None:
        d = positive_float(m["maxErrorDistance"])
        if d is None:
            warnings.append("map.maxErrorDistance ignored: expected a positive number of metres")
        else:
            out["maxErrorDistance"] = d
    return out or None


def debug_dir_from_request(req):
    d = req.get("debug_dir")
    if not d:
        return None
    path = os.path.realpath(os.path.join(ROOT, str(d)))
    allowed = [os.path.realpath(ROOT), os.path.realpath(tempfile.gettempdir())]
    if not any(path == a or path.startswith(a + os.sep) for a in allowed):
        raise ValueError("debug_dir must be inside the project or the temp directory")
    return path


def predict(req):
    loc = get_locator()
    warnings = []
    kw = {"map_info": map_from_request(req, warnings), "debug_dir": debug_dir_from_request(req)}
    if req.get("views"):
        views = [{"image": _img(v["image_b64"]), "yaw": float(v.get("yaw", 0.0)), "pitch": float(v.get("pitch", 0.0)),
                  "hfov": float(v.get("hfov", 100.0)), "true_north": bool(v.get("true_north", True))}
                 for v in req["views"]]
        res = loc.analyze_views(views, **kw)
    elif req.get("image_b64"):
        h = req.get("heading")
        res = loc.analyze_image(_img(req["image_b64"]), heading=None if h is None else float(h),
                                hfov=req.get("hfov"), pitch=float(req.get("pitch", 0.0)), **kw)
    elif req.get("pano_id") or req.get("lat") is not None:
        res = loc.analyze_pano(pano_id=req.get("pano_id"), lat=req.get("lat"), lng=req.get("lng"), **kw)
        res["check"] = distance_report(res, res["panorama"]["lat"], res["panorama"]["lng"])
    else:
        raise ValueError("expected views, image_b64, pano_id or lat/lng")
    if warnings:
        res["warnings"] = warnings
    return res


def evaluate_round(req):
    lat, lng = float(req["lat"]), float(req["lng"])
    true_cc = country_at(lat, lng)
    out = {"true_code": true_cc, "pred_code": req.get("pred_code"), "is_match": true_cc == req.get("pred_code")}
    if req.get("guess_lat") is not None:
        d = float(haversine_km(lat, lng, float(req["guess_lat"]), float(req["guess_lng"])))
        mp = resolve_map(map_from_request(req))
        out.update(distance_km=round(d, 1), points=int(geoguessr_points(d, (mp or {}).get("maxErrorDistance"))))
    return out


def hud_file(path):
    """/hud/<name> -> (bytes, content type) of a file directly in web/hud/, else None."""
    name = path.split("?", 1)[0][len("/hud/"):]
    ext = os.path.splitext(name)[1]
    if not name or "/" in name or "\\" in name or name.startswith(".") or ext not in HUD_TYPES:
        return None
    full = os.path.join(HUD_DIR, name)
    if not os.path.isfile(full):
        return None
    with open(full, "rb") as f:
        return f.read(), HUD_TYPES[ext]


def hud_image(path):
    """/hud/img?u=<url> -> (bytes, content type) of the cached image (tools/fetch_hud_images.py), else None."""
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
    url = (q.get("u") or [""])[0]
    if not url.startswith("https://"):
        return None
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:24]
    for ext, ctype in HUD_IMG_TYPES.items():
        full = os.path.join(HUD_IMG_DIR, key + ext)
        if os.path.isfile(full):
            with open(full, "rb") as f:
                return f.read(), ctype
    return None


def cors_origin(origin):
    """The origin to allow (the extension's service worker, pages on this machine) or None."""
    if origin.startswith("chrome-extension://"):
        return origin
    u = urllib.parse.urlsplit(origin)
    return origin if u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1") else None


def list_maps():
    return {"maps": [{k: m.get(k) for k in ("id", "slug", "name", "world", "maxErrorDistance", "bounds")}
                     for m in load_maps()]}


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        allow = cors_origin(self.headers.get("Origin") or "")
        if allow:
            self.send_header("Access-Control-Allow-Origin", allow)
        self.send_header("Vary", "Origin")

    def _send(self, code, body, ctype="application/json", cache=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", cache)
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        if cors_origin(self.headers.get("Origin") or ""):
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, open(PAGE, "rb").read(), "text/html")
        if self.path.startswith("/hud/img?"):
            f = hud_image(self.path)
            return self._send(200, f[0], f[1], cache="max-age=86400") if f else self._send(404, {"error": "not cached"})
        if self.path.startswith("/hud/"):
            f = hud_file(self.path)
            return self._send(200, f[0], f[1], cache="no-cache") if f else self._send(404, {"error": "not found"})
        if self.path == "/api/maps":
            return self._send(200, list_maps())
        if self.path == "/api/health":
            m = get_locator().model
            return self._send(200, {"ok": True, "countries": len(m.classes), "groups": m.groups})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            if self.path == "/api/predict":
                with LOCK:  # feature extraction is CPU bound; one request at a time
                    return self._send(200, predict(req))
            if self.path == "/api/evaluate_round":
                return self._send(200, evaluate_round(req))
            self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self._send(400, {"error": repr(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write("[server] " + fmt % args + "\n")


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    get_locator()  # load the model before accepting requests
    print("Locator server on http://%s:%d" % (args.host, args.port))
    Server((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
