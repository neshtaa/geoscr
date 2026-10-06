#!/usr/bin/env python3
"""
GeoGuessr Internal Knowledge Base Scraper
Extracts official country guides, learning hub meta, onboarding tips,
and quiz geography data directly from GeoGuessr's frontend bundles and APIs.
"""

import urllib.request
import re
import json
import os
import sys

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def fetch_learning_hub_countries():
    """Extracts GeoGuessr's built-in 114 country guide database from Learning Hub bundle."""
    print("Fetching GeoGuessr Learning Hub bundle...")
    url = "https://www.geoguessr.com/_next/static/chunks/79671-1d0b1d0acaacf185.js"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        txt = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
    except Exception as e:
        print(f"Error fetching chunk: {e}", file=sys.stderr)
        return {}

    idx = txt.find("se:{currency:\"Swedish Krona (SEK)\"")
    if idx == -1:
        print("Could not find start of country dict", file=sys.stderr)
        return {}

    start = txt.rfind("{", 0, idx)
    sub = txt[start:start+35000]

    pattern = r"([a-z]{2}):\{currency:\"([^\"]+)\",drivingSide:\"([^\"]+)\",domain:\"([^\"]+)\",capital:\"([^\"]+)\",language:\[([^\]]*)\],callingCode:\"([^\"]+)\"\}"
    matches = re.findall(pattern, sub)

    countries = {}
    for code, curr, ds, dom, cap, raw_lang, cc in matches:
        langs = [l.replace("learning-hub.lang-", "").strip('"') for l in raw_lang.split(",") if l.strip()]
        side = ds.replace("learning-hub.driving-side-", "")
        countries[code.upper()] = {
            "code": code.upper(),
            "capital": cap,
            "currency": curr,
            "driving_side": side,
            "domain": dom,
            "calling_code": cc,
            "languages": langs
        }
    print(f"Extracted {len(countries)} official country profiles from GeoGuessr Learning Hub.")
    return countries

def fetch_onboarding_tutorial_meta():
    """Extracts GeoGuessr's official tutorial locations and core lessons."""
    print("Fetching GeoGuessr onboarding tips metadata...")
    url = "https://www.geoguessr.com/_next/static/chunks/15268.e33a0f8e8173d802.js"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        txt = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
    except Exception as e:
        print(f"Error fetching onboarding chunk: {e}", file=sys.stderr)
        return []

    # Extract let h=[{stepNumber:1...
    m = re.search(r"let h=(\[\{.*?\}\]);", txt)
    if not m:
        return []

    raw_json = m.group(1)
    # Convert JS object syntax to valid JSON
    raw_json = re.sub(r"([a-zA-Z0-9_]+):", r'"\1":', raw_json)
    raw_json = raw_json.replace("!0", "true").replace("!1", "false")
    try:
        steps = json.loads(raw_json)
        # Add human explanations for keys
        lesson_names = {
            "onboarding.tip-driving-side-title": "Driving Side Identification",
            "onboarding.tip-script-language-title": "Alphabet & Script Clues",
            "onboarding.tip-hemisphere-title": "Sun Position & Hemisphere Detection"
        }
        for s in steps:
            s["lesson_name"] = lesson_names.get(s.get("titleKey"), s.get("titleKey"))
        print(f"Extracted {len(steps)} onboarding tutorial locations with GPS coordinates and panoIds.")
        return steps
    except Exception as e:
        print(f"Error parsing onboarding steps: {e}", file=sys.stderr)
        return []

def main():
    os.makedirs("data", exist_ok=True)
    countries = fetch_learning_hub_countries()
    onboarding_steps = fetch_onboarding_tutorial_meta()

    geoguessr_kb = {
        "source": "GeoGuessr Official Client Bundles & Learning Hub",
        "total_countries": len(countries),
        "countries": countries,
        "tutorial_lessons": onboarding_steps
    }

    out_path = "data/geoguessr_kb.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geoguessr_kb, f, ensure_ascii=False, indent=2)
    print(f"Saved GeoGuessr internal knowledge base to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
