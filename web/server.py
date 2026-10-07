#!/usr/bin/env python3
"""
Local HTTP server for the pure-math locator (used by play_live_visual.js and the web page).

  python3 web/server.py [--port 8080]

POST /api/predict  JSON, one of:
  {"views": [{"image_b64": "...", "yaw": 0, "pitch": 0, "hfov": 100}, ...]}   yaw = true azimuth (deg)
  {"image_b64": "...", "heading": 123.0, "hfov": 100}                          panorama (2:1) or screenshot
  {"pano_id": "..."} | {"lat": 50.4, "lng": 30.5}                               official Street View (tests)
POST /api/evaluate_round  {"lat", "lng", "guess_lat", "guess_lng", "pred_code"}
GET  /api/health
"""
import argparse
import base64
import io
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.geo import country_at, geoguessr_score, haversine_km  # noqa: E402
from engine.locator import distance_report, get_locator  # noqa: E402

LOCK = threading.Lock()
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


def _img(b64):
    if "," in b64[:100]:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def predict(req):
    loc = get_locator()
    if req.get("views"):
        views = [{"image": _img(v["image_b64"]), "yaw": float(v.get("yaw", 0.0)), "pitch": float(v.get("pitch", 0.0)),
                  "hfov": float(v.get("hfov", 100.0)), "true_north": bool(v.get("true_north", True))}
                 for v in req["views"]]
        return loc.analyze_views(views)
    if req.get("image_b64"):
        h = req.get("heading")
        return loc.analyze_image(_img(req["image_b64"]), heading=None if h is None else float(h),
                                 hfov=req.get("hfov"), pitch=float(req.get("pitch", 0.0)))
    if req.get("pano_id") or req.get("lat") is not None:
        res = loc.analyze_pano(pano_id=req.get("pano_id"), lat=req.get("lat"), lng=req.get("lng"))
        res["check"] = distance_report(res, res["panorama"]["lat"], res["panorama"]["lng"])
        return res
    raise ValueError("expected views, image_b64, pano_id or lat/lng")


def evaluate_round(req):
    lat, lng = float(req["lat"]), float(req["lng"])
    true_cc = country_at(lat, lng)
    out = {"true_code": true_cc, "pred_code": req.get("pred_code"), "is_match": true_cc == req.get("pred_code")}
    if req.get("guess_lat") is not None:
        d = float(haversine_km(lat, lng, float(req["guess_lat"]), float(req["guess_lng"])))
        out.update(distance_km=round(d, 1), points=int(round(float(geoguessr_score(d)))))
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, open(PAGE, "rb").read(), "text/html")
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
