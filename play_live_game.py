#!/usr/bin/env python3
"""
GeoGuessr Live Game Runner & Real-Condition Verification
Executes real rounds on GeoGuessr (World Map challenges / live games),
captures Street View panoramas with Puppeteer, applies computer vision & rule matching,
submits live guesses to the GeoGuessr game server, and verifies real accuracy.
"""

import os
import sys
import json
import time
import subprocess
import urllib.request
from engine.vision_extractor import analyze_panorama_image
from engine.offline_engine import OfflineGeoLocator
from engine.rules_matcher import GeoKnowledgeBase

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.geoguessr.com",
    "Referer": "https://www.geoguessr.com/"
}

# Verified public World Map replayable challenges on GeoGuessr
WORLD_CHALLENGES = [
    "fJF1YS0kaazGyiVz",  # GeoBettr World - Replayable
    "CQjV6Hcbn9gRzsrX",  # GeoBettr World - Replayable
    "KVcwOOnGHaKdrMc3",  # GeoBettr World - Replayable
    "iHrcmeuhRXTEZi7c",  # GeoBettr World - Replayable
    "yvcakNlXsdAyUHkw",  # GeoBettr World - Replayable
]

def get_cookie():
    cookie_file = "data/session_cookie.txt"
    if os.path.exists(cookie_file):
        with open(cookie_file, "r") as f:
            return f.read().strip()
    return os.environ.get("GEOGUESSR_COOKIE", "")

def fetch_json(url, cookie=None, data=None):
    headers = dict(HEADERS)
    if cookie:
        headers["Cookie"] = f"_ncfa={cookie}"
    if data is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(data).encode("utf-8") if isinstance(data, dict) else data
        req = urllib.request.Request(url, data=payload, headers=headers)
    else:
        req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"[!] HTTP Error {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[!] Request error: {e}", file=sys.stderr)
        return None

def start_challenge_game(challenge_token, cookie):
    url = f"https://www.geoguessr.com/api/v3/challenges/{challenge_token}"
    res = fetch_json(url, cookie=cookie, data={})
    return res

def get_game_state(game_token, cookie):
    url = f"https://www.geoguessr.com/api/v3/games/{game_token}"
    return fetch_json(url, cookie=cookie)

def submit_guess(game_token, lat, lng, cookie):
    url = f"https://www.geoguessr.com/api/v3/games/{game_token}"
    payload = {
        "token": game_token,
        "lat": float(lat),
        "lng": float(lng),
        "timedOut": False
    }
    return fetch_json(url, cookie=cookie, data=payload)

def capture_streetview_screenshot(game_token, output_img="scratch/live_round.jpg"):
    os.makedirs("scratch", exist_ok=True)
    # Call node script to capture the Street View canvas with drag rotation
    node_code = f"""
const puppeteer = require('puppeteer');
const fs = require('fs');

async function run() {{
  const cookie = fs.readFileSync('data/session_cookie.txt', 'utf8').trim();
  const browser = await puppeteer.launch({{
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  }});
  const page = await browser.newPage();
  await page.setViewport({{width: 1280, height: 720}});
  await page.setCookie({{name: '_ncfa', value: cookie, domain: '.geoguessr.com'}});

  await page.goto('https://www.geoguessr.com/game/{game_token}', {{waitUntil: 'networkidle2', timeout: 35000}});
  await new Promise(r => setTimeout(r, 2000));

  // If previous round result modal is visible, click to advance
  try {{
    const nextBtn = await page.$("[data-qa='close-round-result']") || await page.$("[data-qa='play-next-round']");
    if (nextBtn) {{
      await nextBtn.click();
      await new Promise(r => setTimeout(r, 3500));
    }}
  }} catch(e) {{}}

  const canvasEl = await page.$("[data-qa='panorama-canvas']") || await page.$("canvas.widget-scene-canvas");
  if (canvasEl) {{
    await canvasEl.screenshot({{path: '{output_img}'}});
  }} else {{
    await page.screenshot({{path: '{output_img}'}});
  }}

  // Perform mouse drag to sample rotated view (180 degrees)
  if (canvasEl) {{
    const box = await canvasEl.boundingBox();
    if (box) {{
      const cy = box.y + box.height / 2;
      const cx = box.x + box.width / 2;
      await page.mouse.move(cx, cy);
      await page.mouse.down();
      await page.mouse.move(cx - 350, cy, {{steps: 15}});
      await page.mouse.up();
      await new Promise(r => setTimeout(r, 600));
      await canvasEl.screenshot({{path: 'scratch/live_round_rotated.jpg'}});
    }}
  }}
  await browser.close();
}}

run().catch(err => {{
  console.error(err);
  process.exit(1);
}});
"""
    tmp_js = "scratch/temp_capture.js"
    with open(tmp_js, "w") as f:
        f.write(node_code)

    proc = subprocess.run(["node", tmp_js], capture_output=True, text=True, timeout=45)
    if proc.returncode != 0:
        print(f"[!] Puppeteer error: {proc.stderr}")
        return False
    return os.path.exists(output_img)

def run_live_match(challenge_index=0):
    cookie = get_cookie()
    if not cookie:
        print("[!] No active session cookie found in data/session_cookie.txt")
        return

    challenge_token = WORLD_CHALLENGES[challenge_index % len(WORLD_CHALLENGES)]

    print("=" * 68)
    print("🎮 GEOGUESSR REAL-CONDITION LIVE GAME EXECUTION")
    print(f"🌍 Map: World Map Challenge ({challenge_token})")
    print("=" * 68)

    print("[*] Initializing game session with GeoGuessr server...")
    game = start_challenge_game(challenge_token, cookie)
    if not game or "token" not in game:
        print("[!] Failed to start live game on GeoGuessr server.")
        return

    game_token = game["token"]
    map_name = game.get("mapName", "World")
    total_rounds = game.get("roundCount", 5)

    print(f"[+] Live game created! Token: {game_token}")
    print(f"[+] Map: {map_name} | Total Rounds: {total_rounds}\n")

    locator = OfflineGeoLocator()
    kb = GeoKnowledgeBase()

    locator = OfflineGeoLocator()
    kb = GeoKnowledgeBase()

    round_scores = []
    player = game.get("player", {})
    already_played = len(player.get("guesses", []))
    total_score = sum(int(g.get("roundScoreInPoints", 0)) for g in player.get("guesses", []))

    if already_played > 0:
        print(f"[*] Resuming game from round {already_played + 1} (already completed {already_played} rounds, current score: {total_score:,} pts)")

    for r_num in range(already_played + 1, total_rounds + 1):
        print(f"────────────────────────────────────────────────────────────────────")
        print(f"📍 ROUND {r_num} / {total_rounds}")
        print(f"────────────────────────────────────────────────────────────────────")

        # 1. Capture real panorama from the live game browser
        t0 = time.time()
        print("[1/4] Capturing 360° Street View panorama via headless browser...")
        img_path = f"scratch/live_r{r_num}.jpg"
        captured = capture_streetview_screenshot(game_token, output_img=img_path)
        if not captured:
            print("[!] Could not capture panorama screenshot. Using fallback.")
            img_path = "scratch/live_game_round1.jpg"

        capture_time = round((time.time() - t0), 1)
        print(f"      Captured in {capture_time}s")

        # 2. Extract visual features and run geolocation deduction
        t_inf0 = time.time()
        print("[2/4] Running Computer Vision & Plonk It Intelligence Engine...")

        # Analyze front and rotated views
        cv_front = analyze_panorama_image(img_path)
        rotated_path = "scratch/live_round_rotated.jpg"
        combined_clues = list(cv_front.get("clues", []))
        merged_params = dict(cv_front.get("detected_parameters", {}))

        if os.path.exists(rotated_path):
            cv_rot = analyze_panorama_image(rotated_path)
            for c in cv_rot.get("clues", []):
                if c not in combined_clues:
                    combined_clues.append(c)

            # Check sun hemisphere between North (front) and South (rot)
            f_params = cv_front.get("detected_parameters", {})
            r_params = cv_rot.get("detected_parameters", {})
            f_max = f_params.get("max_sky_brightness", 0)
            r_max = r_params.get("max_sky_brightness", 0)
            f_angle = f_params.get("sun_angle")
            r_angle = r_params.get("sun_angle")

            if f_angle == "high" or r_angle == "high":
                if "sun overhead" not in combined_clues:
                    combined_clues.append("sun overhead")
            elif f_max > 225 and f_max >= r_max + 12:
                # Facing North with bright sun -> Southern Hemisphere
                combined_clues.append("sun visible to the north")
            elif r_max > 225 and r_max >= f_max + 12:
                # Facing South with bright sun -> Northern Hemisphere
                combined_clues.append("sun visible to the south")

        pred = locator.predict_from_features(
            clues_list=[],
            cv_clues=combined_clues,
            detected_params=merged_params
        )
        inf_time_ms = round((time.time() - t_inf0) * 1000, 1)

        top = pred["top_prediction"]
        pred_country = top["country"]
        pred_cc = top["country_code"]
        conf = top["confidence_percent"]
        gps = top.get("gps_estimate", {"lat": 0.0, "lng": 0.0})

        print(f"      Inference completed in {inf_time_ms} ms")
        print(f"      🎯 Top Prediction: {pred_country} ({pred_cc}) [{conf}% confidence]")
        print(f"      📍 Predicted Region: {top.get('region', 'General')}")
        print(f"      🔍 Visual Signals: {', '.join(pred.get('identified_clues', [])[:3])}")
        print(f"      📌 Guess Coordinates: lat={gps['lat']}, lng={gps['lng']}")

        alts = [f"{a['country']} ({a['country_code']}) {a['confidence_percent']}%" for a in pred.get("alternative_candidates", [])[:2]]
        if alts:
            print(f"      🥈 Alternatives: {', '.join(alts)}")

        # 3. Submit guess to real GeoGuessr server
        print("[3/4] Submitting guess to official GeoGuessr server...")
        guess_res = submit_guess(game_token, gps["lat"], gps["lng"], cookie)
        if not guess_res:
            print("[!] Failed to submit guess to GeoGuessr server.")
            continue

        player_data = guess_res.get("player", {})
        last_guess = player_data.get("guesses", [])[-1] if player_data.get("guesses") else {}
        round_pts = int(last_guess.get("roundScoreInPoints", 0))
        dist_m = float(last_guess.get("distanceInMeters", 0))
        dist_km = round(dist_m / 1000.0, 1)

        total_score += round_pts

        # 4. Extract true location for this round
        finished_round = guess_res.get("rounds", [])[r_num - 1] if len(guess_res.get("rounds", [])) >= r_num else {}
        true_lat = finished_round.get("lat")
        true_lng = finished_round.get("lng")
        true_cc = (finished_round.get("streakLocationCode") or "").upper()

        status_icon = "✅" if (true_cc and pred_cc == true_cc) else ("🎯 (Near)" if dist_km < 300 else "📍")
        print(f"[4/4] 🏆 AUTHORITATIVE RESULT FROM GEOGUESSR:")
        print(f"      • Round Score:     {round_pts:,} / 5,000 points {status_icon}")
        print(f"      • Distance Error:  {dist_km:,} km")
        print(f"      • True Location:   {true_cc} (lat={true_lat}, lng={true_lng})")
        print(f"      • Total Score So Far: {total_score:,} / {r_num * 5000:,} pts\n")

        round_scores.append({
            "round": r_num,
            "predicted_country": pred_country,
            "predicted_code": pred_cc,
            "confidence": conf,
            "score_points": round_pts,
            "distance_km": dist_km,
            "true_lat": true_lat,
            "true_lng": true_lng
        })

        time.sleep(2)

    print("=" * 68)
    print("📊 LIVE GAME FINAL SCOREBOARD & SUMMARY REPORT")
    print("=" * 68)
    print(f"• Total Score:   {total_score:,} / 25,000 points")
    avg_dist = round(sum(r['distance_km'] for r in round_scores) / len(round_scores), 1) if round_scores else 0
    print(f"• Average Distance: {avg_dist:,} km")
    print("\nRound Breakdown:")
    for r in round_scores:
        print(f"  R{r['round']:02d}: Pred: {r['predicted_country']} ({r['predicted_code']}) [{r['confidence']}%] | Score: {r['score_points']:,} pts | Error: {r['distance_km']:,} km")
    print("=" * 68)

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_live_match(idx)
