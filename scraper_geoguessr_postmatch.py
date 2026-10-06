"""
GeoGuessr Match Clues Scraper
Harvests official post-match breakdown clues from real duels, competitive games, and showcase matches.
Resolves localized clue titles, descriptions, categories, and image links.
"""

import urllib.request
import urllib.parse
import json
import os
import sys
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.geoguessr.com",
    "Referer": "https://www.geoguessr.com/"
}

TRANSLATION_CACHE = {}

def get_cookie():
    cookie_file = "data/session_cookie.txt"
    if os.path.exists(cookie_file):
        with open(cookie_file, "r") as f:
            return f.read().strip()
    return os.environ.get("GEOGUESSR_COOKIE", "")

def fetch_json(url, cookie=None):
    headers = dict(HEADERS)
    if cookie:
        headers["Cookie"] = f"_ncfa={cookie}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"HTTP {e.code} for {url}", file=sys.stderr)
        return None
    except Exception as e:
        return None

def resolve_translation(key):
    if not key:
        return ""
    if key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[key]
    
    url = f"https://www.geoguessr.com/api/v3/translations/key/{urllib.parse.quote(key)}"
    data = fetch_json(url)
    if data and "phrases" in data:
        for p in data["phrases"]:
            if p.get("language") == "en":
                phrase = p.get("phrase", "")
                TRANSLATION_CACHE[key] = phrase
                return phrase
        if data["phrases"]:
            phrase = data["phrases"][0].get("phrase", "")
            TRANSLATION_CACHE[key] = phrase
            return phrase
    
    TRANSLATION_CACHE[key] = key
    return key

def harvest_game_ids(cookie, max_pages=3):
    game_ids = set()

    # 1. Showcase duels (curated pro duels)
    showcase = fetch_json("https://www.geoguessr.com/api/v4/showcase-duels/on-air", cookie)
    if showcase and "gameId" in showcase:
        game_ids.add(showcase["gameId"])
        print(f"[+] Found showcase on-air duel: {showcase['gameId']}")

    # 2. Feed private (user matches)
    for p in range(max_pages):
        feed = fetch_json(f"https://www.geoguessr.com/api/v4/feed/private?page={p}", cookie)
        if not feed or "entries" not in feed:
            break
        for entry in feed.get("entries", []):
            payload = entry.get("payload")
            if isinstance(payload, str):
                try: payload = json.loads(payload)
                except Exception: pass
            
            if isinstance(payload, list):
                for item in payload:
                    sub = item.get("payload") if isinstance(item, dict) else None
                    if isinstance(sub, str):
                        try: sub = json.loads(sub)
                        except Exception: pass
                    if isinstance(sub, dict) and "gameId" in sub:
                        game_ids.add(sub["gameId"])
            elif isinstance(payload, dict):
                gid = payload.get("gameId")
                if gid:
                    game_ids.add(gid)

    print(f"[+] Harvested {len(game_ids)} distinct duel game IDs from match history.")
    return list(game_ids)

def scrape_match_clues(out_path="data/geoguessr_postmatch_clues.json", max_games=40):
    cookie = get_cookie()
    if not cookie:
        print("[!] No session cookie found in data/session_cookie.txt or GEOGUESSR_COOKIE.", file=sys.stderr)
        return False

    print("[*] Validating user profile...")
    profile = fetch_json("https://www.geoguessr.com/api/v3/profiles/me", cookie)
    nick = profile.get("user", {}).get("nick", "User") if profile else "User"
    print(f"[+] Logged in as: {nick}")

    all_game_ids = harvest_game_ids(cookie, max_pages=3)
    target_games = all_game_ids[:max_games]
    
    all_clues = {}
    total_panos = 0
    total_clues_found = 0

    print(f"[*] Extracting round clues across {len(target_games)} matches...")
    
    for i, gid in enumerate(target_games, 1):
        duel_url = f"https://game-server.geoguessr.com/api/duels/{gid}"
        duel_data = fetch_json(duel_url, cookie)
        if not duel_data:
            continue

        rounds = duel_data.get("rounds", [])
        for r in rounds:
            r_num = r.get("roundNumber")
            pano = r.get("panorama", {})
            hex_pano = pano.get("panoId")
            if not hex_pano:
                continue

            try:
                raw_pano = bytes.fromhex(hex_pano).decode()
            except Exception:
                raw_pano = hex_pano

            country = (pano.get("countryCode") or "XX").upper()
            lat = pano.get("lat")
            lng = pano.get("lng")
            total_panos += 1

            # Fetch clues for this panorama
            clues_url = f"https://www.geoguessr.com/api/v4/clues/{raw_pano}"
            clues_list = fetch_json(clues_url, cookie)
            if not clues_list:
                continue

            for c in clues_list:
                cid = c.get("id")
                if not cid:
                    continue

                if cid not in all_clues:
                    title_raw = c.get("title") or ""
                    desc_raw = c.get("description") or ""
                    title_en = resolve_translation(title_raw)
                    desc_en = resolve_translation(desc_raw)
                    
                    img_path = c.get("image")
                    full_img_url = f"https://www.geoguessr.com/images/resize:fit:600:600/plain/{img_path}" if img_path else None

                    clue_country = (c.get("countryCode") or country or "GLOBAL").upper()

                    all_clues[cid] = {
                        "id": cid,
                        "title": title_en,
                        "description": desc_en,
                        "title_key": title_raw,
                        "description_key": desc_raw,
                        "countryCode": clue_country,
                        "category": c.get("category"),
                        "type": c.get("type"),
                        "heading": c.get("heading"),
                        "pitch": c.get("pitch"),
                        "zoom": c.get("zoom"),
                        "image_path": img_path,
                        "image_url": full_img_url,
                        "seterraRegionIds": c.get("seterraRegionIds", []),
                        "sample_panoramas": []
                    }
                    total_clues_found += 1
                    print(f"   [+] New Official Clue #{len(all_clues)}: [{clue_country}] {title_en} ({c.get('type')})")

                if len(all_clues[cid]["sample_panoramas"]) < 5:
                    all_clues[cid]["sample_panoramas"].append({
                        "panoId": raw_pano,
                        "lat": lat,
                        "lng": lng,
                        "round": r_num,
                        "gameId": gid
                    })

        if i % 5 == 0 or i == len(target_games):
            print(f"   --> Progress: {i}/{len(target_games)} games scanned | {len(all_clues)} unique clues | {total_panos} panoramas")

    # Group clues by country code
    clues_by_country = {}
    for cid, c in all_clues.items():
        cc = (c.get("countryCode") or "GLOBAL").upper()
        if cc not in clues_by_country:
            clues_by_country[cc] = []
        clues_by_country[cc].append(c)

    results = {
        "metadata": {
            "source": "GeoGuessr Official Post-Match Game Analysis",
            "account": nick,
            "total_games_scanned": len(target_games),
            "total_panoramas_inspected": total_panos,
            "total_unique_clues": len(all_clues),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "clues_by_id": all_clues,
        "clues_by_country": clues_by_country
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n========================================================")
    print(f"🎉 [УСПІХ] Базу післяматчевих підказок GeoGuessr збережено!")
    print(f"📊 Всього унікальних підказок: {len(all_clues)}")
    print(f"🌍 Охоплено країн: {len(clues_by_country)}")
    print(f"💾 Файл: {out_path}")
    print(f"========================================================")
    return True

if __name__ == "__main__":
    scrape_match_clues()
