#!/usr/bin/env python3
"""
Plonk It Knowledge Base Scraper
Extracts all country guides, meta tips, regional clues, and tags from plonkit.net.
"""

import urllib.request
import json
import re
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.plonkit.net"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

def get_country_list():
    """Fetches the list of all available countries and regions from Plonk It."""
    print("Fetching guide list from Plonk It...")
    req = urllib.request.Request(f"{BASE_URL}/guide", headers=HEADERS)
    html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
    
    match = re.search(r'<script id="__PRELOADED_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
    if not match:
        raise ValueError("Could not find __PRELOADED_DATA__ in /guide")
    
    raw_data = json.loads(match.group(1))
    guides = raw_data.get("data", [])
    
    valid_countries = []
    for g in guides:
        code = g.get("code", "")
        # Filter out meta categories like maps or beginner's guide
        if code and not code.startswith("XX-"):
            valid_countries.append({
                "title": g.get("title"),
                "slug": g.get("slug"),
                "code": code,
                "continents": g.get("cat", []),
                "updatedAt": g.get("updatedAt")
            })
    print(f"Found {len(valid_countries)} country guides to parse.")
    return valid_countries

CACHE_DIR = "data/cache"

def parse_country(country_meta, max_retries=4):
    """Fetches and parses a single country's complete guide with caching and retries."""
    slug = country_meta["slug"]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{slug}.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    url = f"{BASE_URL}/{slug}"
    req = urllib.request.Request(url, headers=HEADERS)
    
    for attempt in range(max_retries):
        try:
            time.sleep(0.4) # polite rate limiting
            html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = (attempt + 1) * 3
                # print(f"Rate limited on {slug}, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                if attempt == max_retries - 1:
                    print(f"Error fetching {slug}: {e}", file=sys.stderr)
                    return None
                time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching {slug}: {e}", file=sys.stderr)
                return None
            time.sleep(1)
    else:
        return None

    match = re.search(r'<script id="__PRELOADED_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
    if not match:
        print(f"No preloaded data found for {slug}", file=sys.stderr)
        return None

    try:
        raw = json.loads(match.group(1))["data"]["public"]
    except Exception as e:
        print(f"Error parsing JSON for {slug}: {e}", file=sys.stderr)
        return None

    country_info = {
        "title": raw.get("title", country_meta["title"]),
        "slug": slug,
        "code": raw.get("code", country_meta["code"]),
        "continents": raw.get("cat", country_meta["continents"]),
        "hero_image": raw.get("heroImage"),
        "tips": [],
        "regional_clues": [],
        "spotlights": [],
        "all_clues": [],
        "tags_summary": {}
    }

    for step in raw.get("steps", []):
        stitle = step.get("title", "")
        stitle_lower = stitle.lower()
        
        category = "other"
        if "identifying" in stitle_lower:
            category = "identifying"
        elif "regional" in stitle_lower:
            category = "regional"
        elif "spotlight" in stitle_lower:
            category = "spotlight"

        for item in step.get("items", []):
            item_text = []
            if "text" in item:
                if isinstance(item["text"], list):
                    item_text.extend(item["text"])
                else:
                    item_text.append(str(item["text"]))
            
            data = item.get("data", {})
            if isinstance(data, dict) and "text" in data:
                if isinstance(data["text"], list):
                    item_text.extend(data["text"])
                else:
                    item_text.append(str(data["text"]))

            clean_text = " ".join([t for t in item_text if t]).strip()
            tags = item.get("tags") or []
            
            image_url = None
            if isinstance(data, dict):
                img = data.get("image", {})
                if isinstance(img, dict):
                    image_url = img.get("imageUrl")
            if not image_url and item.get("imageUrl"):
                image_url = item.get("imageUrl")

            if clean_text:
                entry = {
                    "section": stitle,
                    "category": category,
                    "tags": tags,
                    "text": clean_text,
                    "image_url": f"{BASE_URL}{image_url}" if image_url and image_url.startswith("/") else image_url
                }
                country_info["all_clues"].append(entry)
                
                for t in tags:
                    country_info["tags_summary"].setdefault(t, []).append(clean_text)

                if category == "identifying":
                    country_info["tips"].append(entry)
                elif category == "regional":
                    country_info["regional_clues"].append(entry)
                elif category == "spotlight":
                    country_info["spotlights"].append(entry)

    # Save to per-country cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(country_info, f, ensure_ascii=False)
    except Exception as e:
        pass

    return country_info

def extract_structured_rules(kb):
    """
    Extracts high-level quick-reference rules across countries:
    Driving side, camera gens, license plates, bollards, poles, etc.
    """
    rules = {
        "driving_side": {"left": [], "right": []},
        "camera_gens": {},
        "car_metas": {},
        "bollards": {},
        "poles": {},
        "plates": {},
        "by_country": {}
    }

    # Left-hand driving countries in GeoGuessr
    left_hand_countries = {
        "GB", "IE", "AU", "NZ", "ZA", "BW", "SZ", "LS", "KE", "UG", 
        "JP", "MY", "SG", "TH", "ID", "LK", "BD", "BT", "IN", "HK", 
        "MO", "MT", "CY", "BM", "VI", "CX", "CC", "IM", "JE"
    }

    for code, cdata in kb.items():
        title = cdata["title"]
        code = cdata["code"]
        all_text = " ".join([c["text"] for c in cdata["all_clues"]]).lower()

        # Driving side
        is_left = code in left_hand_countries or "drives on the left" in all_text or "drive on the left" in all_text or "left-hand traffic" in all_text
        if is_left:
            rules["driving_side"]["left"].append({"code": code, "country": title})
        else:
            rules["driving_side"]["right"].append({"code": code, "country": title})

        country_summary = {
            "title": title,
            "code": code,
            "continents": cdata["continents"],
            "driving_side": "left" if is_left else "right",
            "car_meta": [],
            "poles": [],
            "bollards": [],
            "plates": [],
            "top_clues": [c["text"] for c in cdata["tips"][:5]]
        }

        # Extract specific tags
        for clue in cdata["all_clues"]:
            text = clue["text"]
            tags = clue["tags"]
            if "pole" in tags or "pole" in text.lower():
                country_summary["poles"].append(text[:200])
            if "bollard" in tags or "bollard" in text.lower():
                country_summary["bollards"].append(text[:200])
            if "plate" in text.lower() or "licence" in text.lower() or "license" in text.lower():
                country_summary["plates"].append(text[:200])
            if "google car" in text.lower() or "antenna" in text.lower() or "roof rack" in text.lower() or "snorkel" in text.lower():
                country_summary["car_meta"].append(text[:200])

        rules["by_country"][code] = country_summary

    return rules

def main():
    start_time = time.time()
    countries = get_country_list()
    
    os.makedirs("data", exist_ok=True)
    
    kb = {}
    total = len(countries)
    completed = 0
    
    print(f"Scraping {total} countries using multi-threading (with rate-limiting & cache)...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_country = {executor.submit(parse_country, c): c for c in countries}
        for future in as_completed(future_to_country):
            c = future_to_country[future]
            completed += 1
            res = future.result()
            if res:
                kb[res["code"]] = res
                print(f"[{completed}/{total}] Parsed: {res['title']} ({res['code']}) - {len(res['all_clues'])} clues")
            else:
                print(f"[{completed}/{total}] Failed: {c['title']}")

    # Save full knowledge base
    output_kb_file = "data/plonkit_kb.json"
    with open(output_kb_file, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full knowledge base to {output_kb_file} ({os.path.getsize(output_kb_file) / 1024:.1f} KB)")

    # Extract and save structured quick rules
    rules = extract_structured_rules(kb)
    output_rules_file = "data/country_rules.json"
    with open(output_rules_file, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    print(f"Saved structured rules to {output_rules_file} ({os.path.getsize(output_rules_file) / 1024:.1f} KB)")

    elapsed = time.time() - start_time
    print(f"Completed scraping {len(kb)} countries in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    main()
