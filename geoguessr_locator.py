#!/usr/bin/env python3
"""
GeoGuessr Panorama Locator & Plonk It Intelligence Engine
Determines the country and region of street panoramas in seconds.
"""

import sys
import os
import argparse
import json
import urllib.request
import tempfile
from engine.rules_matcher import GeoKnowledgeBase
from engine.vlm_engine import analyze_image_with_gemini
from engine.offline_engine import OfflineGeoLocator

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║               🌍 GEOGUESSR VISION LOCATOR 🌍                 ║
║   Powered by Plonk It & GeoGuessr Learning Hub (136 Maps)    ║
╚══════════════════════════════════════════════════════════════╝
"""

def print_result(result):
    print("\n" + "="*62)
    engine = result.get("engine_used", "Local Computer Vision & Rules")
    print(f"⚙️  Engine: {engine}")
    
    top = result.get("top_prediction")
    if top:
        print(f"🥇 [MOST PROBABLE CANDIDATE] - {top.get('confidence_percent', 0)}% Confidence")
        print(f"   Country: {top.get('country')} ({top.get('country_code', '')})")
        print(f"   Region / Province: {top.get('region', 'N/A')}")
        if top.get("gps_estimate"):
            gps = top["gps_estimate"]
            print(f"   GPS Estimate: {gps.get('lat', 'N/A')}, {gps.get('lng', 'N/A')}")
    else:
        print("❌ Could not determine location.")

    alts = result.get("alternative_candidates", [])
    if alts:
        print(f"\n🥈 [ALTERNATIVE CANDIDATES]")
        for idx, alt in enumerate(alts, 1):
            conf = alt.get("confidence_percent", 0)
            why = f" - {alt['why_considered']}" if "why_considered" in alt else ""
            print(f"   {idx}. {alt.get('country')} ({alt.get('country_code', '')}) [{conf}%] | Region: {alt.get('region', 'General')}{why}")

    params = result.get("detected_parameters", {})
    if params:
        print(f"\n🔬 [PIXEL ANALYSIS / DETECTED ATTRIBUTES]")
        for k, v in params.items():
            print(f"   • {k.replace('_', ' ').capitalize()}: {v}")

    clues = result.get("identified_clues", [])
    if clues:
        print(f"\n🔎 [IDENTIFIED VISUAL CLUES]")
        for c in clues:
            print(f"   • {c}")

    official = top.get("official_clues", []) if top else []
    if official:
        print(f"\n🏆 [OFFICIAL GEOGUESSR POST-MATCH CLUES - {top.get('country')}]")
        for c in official[:4]:
            ctype = f" [{c.get('type')}]" if c.get('type') else ""
            print(f"   • {c.get('title')}{ctype}: {c.get('description', '')}")

    reasoning = result.get("plonkit_meta_reasoning")
    if reasoning:
        print(f"\n💡 [PLONK IT & GEOGUESSR REASONING]")
        print(f"   {reasoning}")

    print("="*62 + "\n")

def handle_image_analysis(image_path, api_key=None, clues=""):
    print(f"Analyzing panorama: {image_path} ...")
    
    # Priority 1: Gemini multimodal API if key is present
    has_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if has_api_key:
        try:
            print("Running Multimodal Vision deduction (GeoGuessr Grandmaster Model)...")
            res = analyze_image_with_gemini(image_path, api_key=has_api_key)
            res["engine_used"] = "🤖 Gemini Multimodal Grandmaster"
            print_result(res)
            return res
        except Exception as e:
            print(f"Warning: Multimodal API call failed ({e}). Falling back to local computer vision...")

    # Priority 2: Local computer vision & rule matching
    print("Running local Computer Vision & Plonk It rule matcher...")
    offline = OfflineGeoLocator()
    res = offline.predict_image_heuristics(image_path, user_clues=clues)
    print_result(res)
    return res

def handle_country_lookup(code):
    kb = GeoKnowledgeBase()
    data = kb.get_country(code)
    if not data:
        print(f"Country '{code}' not found in Plonk It knowledge base.")
        return

    print("\n" + "="*62)
    print(f"📘 PLONK IT GUIDE: {data['title']} ({data['code']})")
    print(f"Continents: {', '.join(data.get('continents', []))}")
    print("="*62)

    # GeoGuessr official metadata
    gg_path = "data/geoguessr_kb.json"
    if os.path.exists(gg_path):
        try:
            with open(gg_path, "r", encoding="utf-8") as f:
                gg = json.load(f).get("countries", {}).get(code.upper(), {})
                if gg:
                    print("\n[🏛️ GeoGuessr Official Learning Hub Profile]")
                    print(f" • Capital: {gg.get('capital')}")
                    print(f" • Driving Side: {gg.get('driving_side')}")
                    print(f" • Domain: {gg.get('domain')}")
                    print(f" • Currency: {gg.get('currency')}")
                    print(f" • Calling Code: {gg.get('calling_code')}")
        except Exception:
            pass

    print("\n[🎯 Identification Clues]")
    for tip in data.get("tips", [])[:8]:
        print(f" • {tip['text']}")

    print("\n[📍 Regional Clues]")
    for reg in data.get("regional_clues", [])[:6]:
        print(f" • {reg['text']}")

    print("\n[🌟 Spotlight Metas]")
    for spot in data.get("spotlights", [])[:4]:
        print(f" • {spot['text']}")

    gg_clues = data.get("official_geoguessr_clues", [])
    if gg_clues:
        print(f"\n[🏆 Official GeoGuessr Post-Match Clues ({len(gg_clues)} total)]")
        for c in gg_clues[:6]:
            print(f" • [{c.get('type', 'clue')}] {c.get('title')}: {c.get('description', '')}")

    print("="*62 + "\n")

def handle_compare(code1, code2):
    kb = GeoKnowledgeBase()
    c1 = kb.get_country(code1)
    c2 = kb.get_country(code2)
    if not c1 or not c2:
        print(f"Could not find one of the countries: {code1}, {code2}")
        return

    print("\n" + "="*62)
    print(f"⚔️ COMPARISON: {c1['title']} vs {c2['title']}")
    print("="*62)

    diffs = kb.get_differentiating_clues(code1, code2)
    diffs2 = kb.get_differentiating_clues(code2, code1)
    all_diffs = diffs + diffs2

    if all_diffs:
        for d in all_diffs[:10]:
            print(f"\n[💡 {d['focus']} Distinctive Meta]:")
            print(f"  {d['tip']}")
    else:
        print(f"No direct comparison tips found between {c1['title']} and {c2['title']}.")
        if c1.get("tips"):
            print("Tip 1 from", c1['title'], ":", c1['tips'][0]['text'])
        if c2.get("tips"):
            print("Tip 1 from", c2['title'], ":", c2['tips'][0]['text'])
    print("="*62 + "\n")

def handle_search(query):
    kb = GeoKnowledgeBase()
    results = kb.search_clues(query, top_k=6)
    print(f"\n🔍 Search results for: '{query}'")
    print("="*62)
    if not results:
        print("No matching clues found.")
        return
    for r in results:
        print(f"[{r['country']} ({r['code']})] - {r.get('section', 'General')}")
        print(f"  {r['text']}")
        print("-" * 50)
    print("="*62 + "\n")

def start_web_server(port=8080):
    from web.server import run_server
    print(f"Starting GeoGuessr Vision Web UI on http://localhost:{port} ...")
    run_server(port=port)

def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="GeoGuessr Panorama Locator & Plonk It Intelligence")
    parser.add_argument("--image", "-i", type=str, help="Path to panorama or screenshot image file")
    parser.add_argument("--url", "-u", type=str, help="URL of image to download and geolocate")
    parser.add_argument("--api-key", "-k", type=str, help="Gemini API Key (optional, or set GEMINI_API_KEY env var)")
    parser.add_argument("--clues", "-c", type=str, default="", help="Observed clues (comma-separated, e.g. 'yellow plates, snorkel')")
    parser.add_argument("--country", type=str, help="Look up Plonk It guides for country (e.g. UA, JP, ZA)")
    parser.add_argument("--compare", nargs=2, metavar=("C1", "C2"), help="Compare two countries (e.g. UA RU, AU NZ)")
    parser.add_argument("--search", "-s", type=str, help="Search the Plonk It knowledge base for specific clues")
    parser.add_argument("--web", "-w", action="store_true", help="Launch interactive Web UI")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Web UI port (default: 8080)")

    args = parser.parse_args()

    if args.web:
        start_web_server(args.port)
        return

    if args.country:
        handle_country_lookup(args.country)
        return

    if args.compare:
        handle_compare(args.compare[0], args.compare[1])
        return

    if args.search:
        handle_search(args.search)
        return

    if args.clues and not args.image and not args.url:
        print(f"Evaluating clues: {args.clues} ...")
        clues_list = [c.strip() for c in args.clues.split(",") if c.strip()]
        offline = OfflineGeoLocator()
        res = offline.predict_from_features(clues_list=clues_list)
        print_result(res)
        return

    if args.url:
        print(f"Downloading image from {args.url} ...")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            urllib.request.urlretrieve(args.url, tmp.name)
            handle_image_analysis(tmp.name, api_key=args.api_key, clues=args.clues)
        return

    if args.image:
        handle_image_analysis(args.image, api_key=args.api_key, clues=args.clues)
        return

    parser.print_help()

if __name__ == "__main__":
    main()
