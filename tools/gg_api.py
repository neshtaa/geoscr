"""Small authenticated GeoGuessr API helper (cookie from data/session_cookie.txt)."""
import json
import os
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json",
    "Origin": "https://www.geoguessr.com",
    "Referer": "https://www.geoguessr.com/",
}


def cookie():
    path = os.path.join(ROOT, "data", "session_cookie.txt")
    if os.path.exists(path):
        return open(path).read().strip()
    return os.environ.get("GEOGUESSR_COOKIE", "")


def request(url, data=None, method=None, timeout=20):
    headers = dict(HEADERS)
    headers["Cookie"] = "_ncfa=" + cookie()
    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "null")
        except Exception:
            return e.code, None
