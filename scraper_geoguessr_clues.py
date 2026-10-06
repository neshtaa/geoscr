"""
GeoGuessr Post-Match Clues and Knowledge Base Scraper
Extracts official location clues, panorama placements, and comparison data.
Supports:
1. Direct session cookie (_ncfa)
2. Interactive browser session on local DISPLAY to pass Cloudflare Turnstile
"""

import urllib.request
import json
import os
import sys
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.geoguessr.com",
    "Referer": "https://www.geoguessr.com/"
}

def fetch_endpoint(url, ncfa_cookie):
    headers = dict(HEADERS)
    if ncfa_cookie:
        headers["Cookie"] = f"_ncfa={ncfa_cookie}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} for {url}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return None

def dump_all_clues(ncfa_cookie, out_path="data/geoguessr_postmatch_clues.json"):
    print(f"[*] Validating session cookie...")
    profile = fetch_endpoint("https://www.geoguessr.com/api/v3/profiles/me", ncfa_cookie)
    if not profile:
        print("[!] Error: Session cookie is invalid or expired (401 Unauthorized).", file=sys.stderr)
        return False

    user_nick = profile.get("user", {}).get("nick", "Unknown")
    is_pro = profile.get("user", {}).get("isProUser", False)
    print(f"[+] Authenticated successfully as '{user_nick}' (Pro: {is_pro})")

    results = {
        "user": {"nick": user_nick, "isPro": is_pro},
        "clues": [],
        "placements": [],
        "learning_hub": []
    }

    print("[*] Fetching all clue definitions from /api/v4/clues ...")
    clues_data = fetch_endpoint("https://www.geoguessr.com/api/v4/clues", ncfa_cookie)
    if clues_data:
        results["clues"] = clues_data.get("payload", clues_data) if isinstance(clues_data, dict) else clues_data
        print(f"[+] Retrieved {len(results['clues'])} official clue definitions!")

    print("[*] Fetching all panorama placements from /api/v4/clues/panorama ...")
    placements_data = fetch_endpoint("https://www.geoguessr.com/api/v4/clues/panorama", ncfa_cookie)
    if placements_data:
        results["placements"] = placements_data.get("payload", placements_data) if isinstance(placements_data, dict) else placements_data
        print(f"[+] Retrieved {len(results['placements'])} panorama clue placements!")

    print("[*] Fetching learning hub clues from /api/v4/clues/learning-hub ...")
    hub_data = fetch_endpoint("https://www.geoguessr.com/api/v4/clues/learning-hub", ncfa_cookie)
    if hub_data:
        results["learning_hub"] = hub_data.get("payload", hub_data) if isinstance(hub_data, dict) else hub_data
        print(f"[+] Retrieved {len(results['learning_hub'])} learning hub clue entries!")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Saved full GeoGuessr clues database to {out_path}")
    return True

if __name__ == "__main__":
    cookie = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEOGUESSR_COOKIE")
    if not cookie:
        print("Usage: python3 scraper_geoguessr_clues.py <_ncfa_cookie_value>")
        print("Or set environment variable GEOGUESSR_COOKIE")
        sys.exit(1)
    
    success = dump_all_clues(cookie)
    sys.exit(0 if success else 1)
