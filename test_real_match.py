#!/usr/bin/env python3
"""
Real-Condition Match Verification & Accuracy Benchmark
Tests the GeoGuessr Vision Locator against real rounds played on GeoGuessr.
Extracts ground truth locations, feeds visual clues and panorama parameters,
and verifies predictions.
"""

import urllib.request
import json
import os
import sys
import time
from engine.offline_engine import OfflineGeoLocator
from engine.rules_matcher import GeoKnowledgeBase

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.geoguessr.com",
    "Referer": "https://www.geoguessr.com/"
}

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
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

def run_benchmark(num_games=10):
    cookie = get_cookie()
    if not cookie:
        print("[!] No session cookie found. Cannot authenticate with GeoGuessr.")
        return

    print("=" * 66)
    print("🌍 GEOGUESSR REAL-CONDITION MATCH VERIFICATION & BENCHMARK")
    print("=" * 66)

    # Initialize locator
    locator = OfflineGeoLocator()
    kb = GeoKnowledgeBase()

    # Load match history from feed
    print("[*] Loading real match history from GeoGuessr...")
    game_ids = []
    for page in range(3):
        feed = fetch_json(f"https://www.geoguessr.com/api/v4/feed/private?page={page}", cookie)
        if not feed:
            break
        for entry in feed.get("entries", []):
            payload = entry.get("payload")
            if isinstance(payload, str):
                try: payload = json.loads(payload)
                except: pass
            if isinstance(payload, list):
                for item in payload:
                    sub = item.get("payload") if isinstance(item, dict) else None
                    if isinstance(sub, str):
                        try: sub = json.loads(sub)
                        except: pass
                    if isinstance(sub, dict) and "gameId" in sub:
                        game_ids.append(sub["gameId"])
            elif isinstance(payload, dict) and "gameId" in payload:
                game_ids.append(payload["gameId"])

    unique_games = list(dict.fromkeys(game_ids))[:num_games]
    print(f"[+] Loaded {len(unique_games)} real duel matches for testing.\n")

    total_rounds = 0
    correct_top1 = 0
    correct_top3 = 0
    round_results = []

    start_time = time.time()

    for g_idx, gid in enumerate(unique_games, 1):
        duel_url = f"https://game-server.geoguessr.com/api/duels/{gid}"
        duel = fetch_json(duel_url, cookie)
        if not duel or "rounds" not in duel:
            continue

        print(f"--- [Match #{g_idx}: {gid}] ---")

        for r in duel["rounds"]:
            r_num = r.get("roundNumber", 1)
            pano = r.get("panorama", {})
            hex_p = pano.get("panoId")
            if not hex_p:
                continue

            try:
                raw_p = bytes.fromhex(hex_p).decode()
            except Exception:
                raw_p = hex_p

            true_cc = (pano.get("countryCode") or "").upper()
            lat = pano.get("lat")
            lng = pano.get("lng")

            if not true_cc:
                continue

            # Fetch official round clues
            clues_url = f"https://www.geoguessr.com/api/v4/clues/{raw_p}"
            clues_data = fetch_json(clues_url, cookie) or []

            # Extract clue textual features
            clue_texts = []
            driving_side = None

            for c in clues_data:
                cid = c.get("id", "")
                title = c.get("title", "")
                ctype = c.get("type", "")

                if "drives-on-left" in cid or "drive-left" in cid:
                    driving_side = "left"
                elif "drives-on-right" in cid or "drive-right" in cid:
                    driving_side = "right"

                # Check if we have localized translation in postmatch_kb
                master_id = title.replace("clue.", "").replace("-title", "")
                cached = (
                    locator.postmatch_kb.get("all_clues", {}).get(master_id) or
                    locator.postmatch_kb.get("clues_by_id", {}).get(master_id) or
                    locator.postmatch_kb.get("all_clues", {}).get(cid) or
                    locator.postmatch_kb.get("clues_by_id", {}).get(cid)
                )
                if cached:
                    clue_texts.append(cached.get("id", master_id))
                    clue_texts.append(cached.get("title", title))
                    if cached.get("description"):
                        clue_texts.append(cached.get("description"))
                else:
                    clue_texts.append(title.replace("clue.", "").replace("-title", "").replace("-", " "))

            # Clean and deduplicate clue texts
            input_clues = [t.strip() for t in clue_texts if t.strip() and len(t.strip()) > 3]

            # Run prediction: If clues exist, predict from features. If not, analyze panorama image!
            t0 = time.time()
            if input_clues or driving_side:
                prediction = locator.predict_from_features(
                    driving_side=driving_side,
                    clues_list=input_clues
                )
            else:
                # Download panorama thumbnail and run computer vision
                tmp_pano = f"/tmp/bench_pano_{raw_p}.jpg"
                thumb_url = f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail?panoid={raw_p}&cb_client=maps_sv.tactile&w=600&h=400"
                try:
                    req = urllib.request.Request(thumb_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=5) as r_img, open(tmp_pano, "wb") as f_img:
                        f_img.write(r_img.read())
                    prediction = locator.predict_image_heuristics(tmp_pano)
                except Exception:
                    prediction = locator.predict_from_features()
                finally:
                    if os.path.exists(tmp_pano):
                        try: os.remove(tmp_pano)
                        except: pass

            elapsed_ms = round((time.time() - t0) * 1000, 1)

            top = prediction["top_prediction"]
            pred_cc = top["country_code"]
            conf = top["confidence_percent"]

            alts = [a["country_code"] for a in prediction.get("alternative_candidates", [])]
            top3 = [pred_cc] + alts[:2]

            is_top1 = (pred_cc == true_cc)
            is_top3 = (true_cc in top3)

            total_rounds += 1
            if is_top1:
                correct_top1 += 1
            if is_top3:
                correct_top3 += 1

            status_icon = "✅" if is_top1 else ("⚠️ (Top 3)" if is_top3 else "❌")
            print(f"  Round {r_num:02d}: True: {true_cc} | Pred: {pred_cc} ({conf}%) | {status_icon} [{elapsed_ms}ms]")
            if not is_top1:
                print(f"           Alternatives: {top3[1:]}")
                if input_clues:
                    print(f"           Clues: {input_clues[:3]}")

            round_results.append({
                "gameId": gid,
                "round": r_num,
                "true_country": true_cc,
                "predicted_country": pred_cc,
                "confidence": conf,
                "is_top1": is_top1,
                "is_top3": is_top3,
                "elapsed_ms": elapsed_ms
            })

    total_time = round(time.time() - start_time, 2)
    top1_acc = round((correct_top1 / total_rounds) * 100, 1) if total_rounds else 0
    top3_acc = round((correct_top3 / total_rounds) * 100, 1) if total_rounds else 0
    avg_speed = round(sum(r["elapsed_ms"] for r in round_results) / total_rounds, 1) if total_rounds else 0

    print("\n" + "=" * 66)
    print("📊 BENCHMARK SUMMARY & ACCURACY REPORT")
    print("=" * 66)
    print(f"• Total Matches Tested:        {len(unique_games)}")
    print(f"• Total Panoramas / Rounds:    {total_rounds}")
    print(f"• Top-1 Exact Accuracy:        {correct_top1}/{total_rounds} ({top1_acc}%)")
    print(f"• Top-3 Regional Accuracy:     {correct_top3}/{total_rounds} ({top3_acc}%)")
    print(f"• Average Inference Speed:     {avg_speed} ms per round")
    print(f"• Total Benchmark Time:        {total_time}s")
    print("=" * 66 + "\n")

    return {
        "total_rounds": total_rounds,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "avg_speed_ms": avg_speed,
        "results": round_results
    }

if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_benchmark(num)
