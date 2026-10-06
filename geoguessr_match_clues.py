"""
GeoGuessr Match Clues Extractor
Fetches official post-match round clues and breakdown from a GeoGuessr duel/game URL.
Requires user session cookie (_ncfa) or public game token.
"""

import urllib.request
import json
import re
import sys
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Content-Type": "application/json"
}

def extract_game_token(input_str):
    """Extracts game or duel token from a URL or raw ID."""
    m = re.search(r"duels/([a-zA-Z0-9\-]+)", input_str)
    if m:
        return m.group(1)
    m = re.search(r"game/([a-zA-Z0-9\-]+)", input_str)
    if m:
        return m.group(1)
    m = re.search(r"results/([a-zA-Z0-9\-]+)", input_str)
    if m:
        return m.group(1)
    return input_str.strip()

def get_duel_result(token, ncfa_cookie=None):
    """Fetches duel game summary data."""
    url = f"https://www.geoguessr.com/api/v4/game-results/duels/{token}"
    headers = dict(HEADERS)
    if ncfa_cookie:
        headers["Cookie"] = f"_ncfa={ncfa_cookie}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching duel results: {e}", file=sys.stderr)
        return None

def get_round_clues(pano_id, ncfa_cookie=None):
    """Fetches GeoGuessr official location clues for a specific panorama ID."""
    url = f"https://www.geoguessr.com/api/v4/clues/{pano_id}"
    headers = dict(HEADERS)
    if ncfa_cookie:
        headers["Cookie"] = f"_ncfa={ncfa_cookie}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching clues for pano {pano_id}: {e}", file=sys.stderr)
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 geoguessr_match_clues.py <duel_url_or_token> [ncfa_cookie]")
        print("Example: python3 geoguessr_match_clues.py https://www.geoguessr.com/duels/abc-123/summary")
        return

    token = extract_game_token(sys.argv[1])
    cookie = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("GEOGUESSR_COOKIE")
    
    print(f"Fetching match analysis for token: {token} ...")
    result = get_duel_result(token, ncfa_cookie=cookie)
    if not result:
        print("Could not retrieve game result. Note: private duels require providing your _ncfa session cookie.")
        return

    print("Match loaded successfully!")
    rounds = result.get("rounds", [])
    print(f"Total rounds: {len(rounds)}")
    
    for r in rounds:
        r_num = r.get("roundNumber", "?")
        pano = r.get("panorama", {})
        pano_id = pano.get("panoId")
        lat = pano.get("lat")
        lng = pano.get("lng")
        country = r.get("countryCode", "??").upper()
        print(f"\n--- Round {r_num}: {country} ({lat}, {lng}) | Pano: {pano_id} ---")
        
        if pano_id:
            clues = get_round_clues(pano_id, ncfa_cookie=cookie)
            if clues:
                print(f"Official Clues for Round {r_num}:")
                for c in clues:
                    print(f"  • [{c.get('category', 'Clue')}] {c.get('title')}: {c.get('description')}")
            else:
                print("  No official clues returned (or session cookie required).")

if __name__ == "__main__":
    main()
