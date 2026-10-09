"""Offline cache of the GeoGuessr clue-card images shown by the live HUD (web/hud/hud.js).

The HUD never loads an image from www.geoguessr.com inside the game page: the extension's service
worker and play_live_visual.js fetch GET /hud/img?u=<image_url> from the local server, which serves
only files of this cache (scratch/hud_img/<sha1(url)[:24]>.<ext>), and hand the HUD data: URLs.

    python3 tools/fetch_hud_images.py              # every card image of the hint base (~900, ~25 MB)
    python3 tools/fetch_hud_images.py --limit 50

Run it between games, not during a round. Requests: GET only, no cookie, >= 1.3 s apart, stops at the
first HTTP 429. Already cached images are skipped, so it can be re-run after the hint base grows.
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CACHE = os.path.join(ROOT, "scratch", "hud_img")
EXTS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
GG_PREFIX = "https://www.geoguessr.com/images/resize:fit:600:600/plain/"
THUMB_PREFIX = "https://www.geoguessr.com/images/resize:fit:320:320/plain/"   # the HUD shows <= 380 px


def image_key(url):
    """Cache name of an image URL (web/server.py hud_image uses the same key)."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:24]


def cached(url):
    k = image_key(url)
    return any(os.path.exists(os.path.join(CACHE, k + e)) for e in EXTS.values())


def card_urls():
    from engine.hints import ClueBase
    cb = ClueBase()
    urls = set()
    for cards in cb.gg.values():
        for c in cards:
            u = c.get("image_url")
            if u and u.startswith(GG_PREFIX):
                urls.add(u)
    return sorted(urls)


def fetch(url):
    req = urllib.request.Request(THUMB_PREFIX + url[len(GG_PREFIX):],
                                 headers={"User-Agent": "geoscr-hud-cache/1", "Accept": "image/jpeg,image/png,image/webp"})
    with urllib.request.urlopen(req, timeout=30) as r:
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip()
        return ctype, r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    todo = [u for u in card_urls() if not cached(u)]
    if args.limit:
        todo = todo[:args.limit]
    print("%d images to fetch into %s" % (len(todo), os.path.relpath(CACHE, ROOT)), flush=True)
    last, fails, got, size = 0.0, 0, 0, 0
    for i, url in enumerate(todo):
        wait = last + 1.3 - time.time()
        if wait > 0:
            time.sleep(wait)
        last = time.time()
        try:
            ctype, data = fetch(url)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print("HTTP 429: stopping", flush=True)
                break
            print("HTTP %d %s" % (e.code, url), flush=True)
            fails += 1
            continue
        except Exception as e:  # noqa: BLE001 - network errors: count and go on
            print("error %s: %s" % (url, e), flush=True)
            fails += 1
            if fails >= 10:
                print("too many errors: stopping", flush=True)
                break
            continue
        ext = EXTS.get(ctype)
        if not ext or not data:
            print("not an image (%s): %s" % (ctype, url), flush=True)
            continue
        out = os.path.join(CACHE, image_key(url) + ext)
        with open(out + ".part", "wb") as f:
            f.write(data)
        os.replace(out + ".part", out)
        got += 1
        size += len(data)
        if got % 50 == 0:
            print("%d/%d, %.1f MB" % (i + 1, len(todo), size / 1e6), flush=True)
    print("fetched %d images (%.1f MB), %d errors" % (got, size / 1e6, fails), flush=True)


if __name__ == "__main__":
    main()
