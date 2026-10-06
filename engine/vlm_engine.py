"""
Multimodal Vision-Language Geolocation Engine
Analyzes Street View / GeoGuessr panoramas using multimodal models
grounded in the Plonk It knowledge base taxonomy.
"""

import os
import json
import base64
import urllib.request
import urllib.parse

SYSTEM_PROMPT = """You are a World-Class GeoGuessr Grandmaster and Geolocation Expert.
Your task is to analyze the provided street-level panorama image and determine the EXACT country and region/province with high precision.

You MUST systematically analyze these visual clues in order:
1. DRIVING SIDE: Left or Right side of the road.
2. SUN POSITION & HEMISPHERE: Is the sun in the North (Southern Hemisphere), South (Northern Hemisphere), or directly overhead (Equator)?
3. GOOGLE CAR & CAMERA META:
   - Camera Generation: Gen 2 (circular blur, low res), Gen 3 (standard 2048x1024), Gen 4 (crisp 4K, 11-blade blue halo or clean blur).
   - Google Car features: Car color (red car in Ukraine/Belgium, white, black, silver), roof rack/bars (Kyrgyzstan, Mongolia, Guatemala, Dominican Rep), snorkel (Kenya, Mongolia), visible antenna (short/long, ball, flag).
4. ROAD INFRASTRUCTURE:
   - Road Lines: White vs yellow center line; dashed or continuous; yellow outer lines (South Africa, Israel, etc.).
   - Utility Poles: Square concrete (post-soviet / Eastern Europe), ladder poles (France, Spain, Poland, Romania), holey concrete poles (Poland, Hungary, Romania), wooden poles, painted white base.
   - Bollards (Guide posts): Red rectangular reflector (Poland, Ukraine Zakarpattia, etc.), yellow reflector, round vs rectangular, wooden.
   - Guardrails: Black-and-white painted (Ukraine, Russia), wooden, metal W-beam.
5. SIGNS, LANGUAGE & ALPHABET:
   - Script: Latin, Cyrillic (note unique letters: Ukrainian 'і, ї, є, ґ', Russian 'ы, э, ъ', Serbian 'ђ, ћ', etc.), Greek, Arabic, Thai, Khmer, Japanese, Korean, Hebrew.
   - Chevrons: White on blue, white on red, black on yellow.
   - Pedestrian crossing signs: Striped pattern, silhouette style.
6. LICENSE PLATES:
   - European standard (long white with blue Euroband on left).
   - Yellow plates: UK rear, Netherlands both, Israel both, Luxembourg both, France older, Colombia taxis/commercial.
   - Small square or short US-style plates: Americas, Japan (yellow for Kei cars), Philippines.
7. LANDSCAPE, SOIL & VEGETATION:
   - Soil: Red soil (Brazil, Kryvyi Rih Ukraine, parts of Africa/Australia), dry arid steppes, lush tropical, boreal pine forests.
   - Architecture: Soviet prefab blocks, European tiled roofs, American suburban wooden houses, African brick/tin.

OUTPUT FORMAT:
You MUST respond ONLY with a valid JSON object with the following exact keys:
{
  "top_prediction": {
    "country": "Country Name",
    "country_code": "ISO 2-letter code",
    "region": "Specific Region, State, Oblast, or Province",
    "confidence_percent": 85,
    "gps_estimate": {"lat": 0.0, "lng": 0.0}
  },
  "alternative_candidates": [
    {
      "country": "Alternative Country",
      "country_code": "Code",
      "region": "Possible Region",
      "confidence_percent": 10,
      "why_considered": "Explanation of shared features and why it's less likely"
    }
  ],
  "identified_clues": [
    "Specific visual clue 1 observed in the image",
    "Specific visual clue 2 observed in the image",
    "Specific visual clue 3 observed in the image"
  ],
  "plonkit_meta_reasoning": "Detailed 2-3 sentence explanation referencing specific meta rules (bollards, car, camera gen, poles, language) that prove the top prediction."
}
"""

def encode_image(image_path):
    """Encodes a local image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image_with_gemini(image_path, api_key=None):
    """
    Calls Google Gemini multimodal API (e.g. gemini-1.5-flash or gemini-2.0-flash)
    using urllib to avoid external heavy dependencies.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")

    b64_data = encode_image(image_path)
    
    # Determine mime type
    mime_type = "image/jpeg"
    lower_path = image_path.lower()
    if lower_path.endswith(".png"):
        mime_type = "image/png"
    elif lower_path.endswith(".webp"):
        mime_type = "image/webp"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data
                        }
                    },
                    {"text": "Analyze this panorama/street view screenshot now and provide the JSON output."}
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            candidate = res_data["candidates"][0]
            text = candidate["content"]["parts"][0]["text"]
            return json.loads(text)
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        raise RuntimeError(f"Gemini API error ({e.code}): {error_msg}")
    except Exception as e:
        raise RuntimeError(f"Inference error: {e}")
