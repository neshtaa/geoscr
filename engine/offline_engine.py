"""
Offline / Local Heuristic and Computer Vision Geolocation Engine
Extracts visual features directly from image pixels (Pillow / NumPy)
and cross-references them with the Plonk It and GeoGuessr knowledge bases.
"""

import os
import json
import math
import re
import unicodedata
from .rules_matcher import GeoKnowledgeBase
from .vision_extractor import analyze_panorama_image, compare_panoramas


# GeoGuessr Geographic / Meta Profiles
RED_SOIL_COUNTRIES = {
    "AU": 20, "BR": 20, "NG": 18, "ZA": 18, "BW": 18, "KE": 16, "UG": 16,
    "GH": 16, "SN": 14, "MG": 16, "US": 16, "KH": 14, "LK": 10, "UY": 8, "AR": 8,
    "ES": 16, "PT": 14, "IT": 14, "GR": 14, "TR": 14, "IN": 16, "BD": 14, "CO": 10, "PY": 12, "BO": 12
}
YELLOW_LINES_COUNTRIES = {
    "US": 18, "CA": 16, "MX": 14, "BR": 12, "CO": 12, "AR": 12, "CL": 12, "PE": 10,
    "ZA": 14, "BW": 14, "SZ": 14, "LS": 14, "NA": 14, "JP": 10, "PH": 8, "TH": 10, "MY": 8, "TW": 8,
    "NO": 10, "FI": 10, "IS": 10, "IL": 10
}
WHITE_LINES_EUROPE = {
    "UA": 12, "PL": 12, "DE": 12, "FR": 12, "ES": 12, "IT": 12, "RO": 12, 
    "CZ": 12, "SK": 12, "AT": 12, "CH": 14, "HU": 12, "BG": 12, "LT": 12, "LV": 12, "EE": 12,
    "GR": 12, "TR": 12, "PT": 12, "HR": 12, "SI": 12, "RS": 12, "MK": 12, "AL": 12, "ME": 12,
    "NL": 12, "BE": 12, "DK": 12, "SE": 12, "NO": 12, "FI": 12, "IE": 10, "GB": 10, "LU": 12, "IS": 10
}
ARID_COUNTRIES = {
    "JO": 15, "AE": 15, "QA": 15, "OM": 15, "EG": 14, "TN": 12, "BW": 14, "NA": 14, "ZA": 12, 
    "AU": 16, "US": 16, "CL": 12, "PE": 12, "AR": 10, "MX": 10, "ES": 10, "TR": 10, "GR": 8, "MN": 12, "IL": 12
}
TROPICAL_LUSH_COUNTRIES = {
    "BD": 24, "IN": 22, "LK": 22, "TH": 20, "MY": 20, "ID": 20, "PH": 20, "VN": 20,
    "CR": 20, "CO": 22, "BR": 18, "EC": 18, "NG": 16, "GH": 16, "TW": 16
}
LUSH_COUNTRIES = {
    "GB": 16, "IE": 16, "CZ": 16, "SK": 16, "PL": 16, "AT": 16, "DE": 16, "FR": 16, "HU": 14, "CH": 14,
    "US": 12, "CA": 10, "JP": 12, "NZ": 12, "AU": 10,
    "CR": 12, "BR": 14, "ID": 14, "MY": 14, "PH": 12, "TH": 14, "EC": 14, "CO": 16, "UA": 10, "VN": 12, "TW": 12,
    "BD": 12, "IN": 12, "LK": 12, "ES": 16, "PT": 14, "IT": 16, "GR": 14, "TR": 14
}
UNPAVED_COUNTRIES = {
    "BR": 18, "MN": 15, "BW": 14, "MG": 14, "KG": 14, "BO": 14, "PE": 12, "LA": 12,
    "SN": 12, "KH": 12, "ZA": 10, "NG": 16, "UG": 15, "GH": 14, "CO": 14, "PY": 14, "AR": 12
}


SOUTHERN_HEMISPHERE = {"AU", "NZ", "ZA", "BW", "SZ", "LS", "NA", "AR", "CL", "UY", "PY", "BO", "PE", "MG", "ID", "BR", "EC", "KE", "UG", "RW", "TZ"}
NORTHERN_HEMISPHERE = {"US", "CA", "MX", "GB", "IE", "FR", "ES", "PT", "DE", "IT", "NL", "BE", "DK", "NO", "SE", "FI", "EE", "LV", "LT", "PL", "CZ", "SK", "AT", "CH", "HU", "RO", "BG", "GR", "TR", "UA", "RU", "KZ", "MN", "JP", "KR", "TW", "TH", "VN", "PH", "MY", "IN", "BD", "LK", "IL", "JO", "AE", "QA", "OM", "AD", "IS", "SI", "HR", "RS", "MK", "AL", "ME"}
EQUATORIAL_COUNTRIES = {"EC", "CO", "BR", "ID", "MY", "SG", "KE", "UG", "GH", "SN", "NG", "LK", "TH", "VN", "PH", "MX", "IN", "PA", "CR", "GT", "KH", "LA"}
HIGH_LATITUDE_NORTH = {
    "NO", "SE", "FI", "IS", "EE", "LV", "LT", "RU", "CA", "GB", "IE",
    "DK", "NL", "BE", "DE", "PL", "CZ", "SK", "AT", "CH", "HU", "UA",
    "RO", "BG", "KZ", "MN"
}

COUNTRY_CENTERS = {
    "US": (37.0902, -95.7129), "CA": (56.1304, -106.3468), "MX": (23.6345, -102.5528),
    "BR": (-14.2350, -51.9253), "AR": (-38.4161, -63.6167), "CL": (-35.6751, -71.5430),
    "CO": (4.5709, -74.2973), "PE": (-9.1900, -75.0152), "EC": (-1.8312, -78.1834),
    "UY": (-32.5228, -55.7658), "BO": (-16.2902, -63.5887), "PY": (-23.4425, -58.4438),
    "GB": (55.3781, -3.4360), "IE": (53.1424, -7.6921), "FR": (46.2276, 2.2137),
    "ES": (40.4637, -3.7492), "PT": (39.3999, -8.2245), "DE": (51.1657, 10.4515),
    "IT": (41.8719, 12.5674), "NL": (52.1326, 5.2913), "BE": (50.5039, 4.4699),
    "CH": (46.8182, 8.2275), "AT": (47.5162, 14.5501), "PL": (51.9194, 19.1451),
    "CZ": (49.8175, 15.4730), "SK": (48.6690, 19.6990), "HU": (47.1625, 19.5033),
    "RO": (45.9432, 24.9668), "BG": (42.7339, 25.4858), "GR": (39.0742, 21.8243),
    "UA": (48.3794, 31.1656), "RU": (61.5240, 105.3188), "TR": (38.9637, 35.2433),
    "DK": (56.2639, 9.5018), "SE": (60.1282, 18.6435), "NO": (60.4720, 8.4689),
    "FI": (61.9241, 25.7482), "EE": (58.5953, 25.0136), "LV": (56.8796, 24.6032),
    "LT": (55.1694, 23.8813), "IS": (64.9631, -19.0208), "SI": (46.1512, 14.9955),
    "HR": (45.1000, 15.2000), "RS": (44.0165, 21.0059), "MK": (41.6086, 21.7453),
    "AL": (41.1533, 20.1683), "ME": (42.7087, 19.3744), "AD": (42.5063, 1.5218),
    "ZA": (-30.5595, 22.9375), "BW": (-22.3285, 24.6849), "SZ": (-26.5225, 31.4659),
    "LS": (-29.6099, 28.2336), "NA": (-22.9576, 18.4904), "KE": (-0.0236, 37.9062),
    "UG": (1.3733, 32.2903), "GH": (7.9465, -1.0232), "SN": (14.4974, -14.4524),
    "NG": (9.0820, 8.6753), "EG": (26.8206, 30.8025), "TN": (33.8869, 9.5375),
    "JP": (36.2048, 138.2529), "KR": (35.9078, 127.7669), "TW": (23.6978, 120.9605),
    "TH": (15.8700, 100.9925), "VN": (14.0583, 108.2772), "MY": (4.2105, 101.9758),
    "ID": (-0.7893, 113.9213), "PH": (12.8797, 121.7740), "SG": (1.3521, 103.8198),
    "KH": (12.5657, 104.9910), "LA": (19.8563, 102.4955), "IN": (20.5937, 78.9629),
    "BD": (23.6850, 90.3563), "LK": (7.8731, 80.7718), "BT": (27.5142, 90.4336),
    "MN": (46.8625, 103.8467), "KG": (41.2044, 74.7661), "JO": (30.5852, 36.2384),
    "IL": (31.0461, 34.8516), "AE": (23.4241, 53.8478), "QA": (25.3548, 51.1839),
    "OM": (21.4735, 55.9754), "SA": (23.8859, 45.0792), "KW": (29.3117, 47.4818),
    "AU": (-25.2744, 133.7751), "NZ": (-40.9006, 174.8860),
    "CR": (9.7489, -83.7534), "GT": (15.7835, -90.2308), "PA": (8.5380, -80.7821),
    "DO": (18.7357, -70.1627), "PR": (18.2208, -66.5901), "CW": (12.1696, -68.9900),
    "MG": (-18.7669, 46.8691), "RW": (-1.9403, 29.8739), "TZ": (-6.3690, 34.8888),
    "LU": (49.8153, 6.1296), "MT": (35.9375, 14.3754), "CY": (35.1264, 33.4299),
    "HK": (22.3193, 114.1694), "MO": (22.1987, 113.5439), "GU": (13.4443, 144.7937)
}

# GeoGuessr World Map Street View Coverage & Probability Priors
COVERAGE_PRIORS = {
    "US": 0.9, "CA": 0.8, "BR": 0.8, "AU": 0.75, "JP": 0.75, "DE": 0.75,
    "FR": 0.65, "GB": 0.65, "ES": 0.7, "IT": 0.7, "MX": 0.7, "ID": 0.65,
    "TH": 0.6, "ZA": 0.6, "PL": 0.70, "UA": 0.55, "AR": 0.55, "CL": 0.55,
    "CO": 0.55, "TR": 0.55, "SE": 0.5, "NO": 0.5, "FI": 0.5, "NZ": 0.5,
    "MY": 0.5, "PH": 0.5, "RO": 0.5, "CZ": 0.70, "SK": 0.65, "AT": 0.65, "HU": 0.55,
    "NL": 0.5, "BE": 0.5, "DK": 0.5, "PT": 0.5, "GR": 0.45, "IE": 0.45,
    "PE": 0.4, "EC": 0.4, "KE": 0.35, "UG": 0.35, "GH": 0.35, "SN": 0.35,
    "NG": 0.35, "BW": 0.35, "LK": 0.35, "BD": 0.35, "TW": 0.35, "KR": 0.35,
    "UY": 0.3, "BO": 0.3, "PY": 0.3, "BG": 0.35, "SK": 0.65, "HR": 0.35,
    "RS": 0.3, "SI": 0.35, "EE": 0.35, "LV": 0.35, "LT": 0.35, "IS": 0.3,
    "CH": 0.3, "MN": 0.25, "KG": 0.25, "CR": 0.25, "GT": 0.25, "PA": 0.25,
    "JO": 0.2, "AE": 0.2, "QA": 0.2, "OM": 0.2, "TN": 0.2, "SZ": 0.2,
    "LS": 0.2, "NA": 0.2, "MK": 0.2, "AL": 0.2, "ME": 0.2, "AD": 0.15,
    "SM": 0.15, "MC": 0.15, "LI": 0.15, "LU": 0.15, "MT": 0.15, "CY": 0.15,
    "FO": 0.1, "GL": 0.1, "BM": 0.1, "CW": 0.1, "GU": 0.1, "MP": 0.1
}

DEMONYMS = {
    "IE": ["ireland", "irish"],
    "VN": ["vietnam", "vietnamese"],
    "ES": ["spain", "spanish"],
    "AD": ["andorra", "andorran"],
    "BR": ["brazil", "brazilian", "portuguese"],
    "CA": ["canada", "canadian", "quebec"],
    "MX": ["mexico", "mexican"],
    "US": ["united states", "american", "usa", "cascadia"],
    "JP": ["japan", "japanese", "hokkaido"],
    "SE": ["sweden", "swedish"],
    "NO": ["norway", "norwegian"],
    "DK": ["denmark", "danish"],
    "FI": ["finland", "finnish"],
    "DE": ["germany", "german"],
    "FR": ["france", "french"],
    "IT": ["italy", "italian"],
    "PL": ["poland", "polish"],
    "UA": ["ukraine", "ukrainian"],
    "RU": ["russia", "russian"],
    "KZ": ["kazakhstan", "kazakh"],
    "CO": ["colombia", "colombian"],
    "EC": ["ecuador", "ecuadorian", "esmeraldas"],
    "PE": ["peru", "peruvian"],
    "CL": ["chile", "chilean", "atacama"],
    "AR": ["argentina", "argentine", "argentinean"],
    "ZA": ["south africa", "south african", "afrikaans"],
    "KE": ["kenya", "kenyan"],
    "ID": ["indonesia", "indonesian", "java", "sumatra", "bali"],
    "MY": ["malaysia", "malaysian"],
    "TH": ["thailand", "thai"],
    "PH": ["philippines", "filipino", "philippine"],
    "AU": ["australia", "australian", "eucalyptus"],
    "NZ": ["new zealand", "kiwi", "waikato"],
    "GR": ["greece", "greek"],
    "TR": ["turkey", "turkish"],
    "RO": ["romania", "romanian"],
    "BG": ["bulgaria", "bulgarian"],
    "EE": ["estonia", "estonian"],
    "LV": ["latvia", "latvian"],
    "LT": ["lithuania", "lithuanian"],
    "MK": ["north macedonia", "macedonian", "macedonia"],
    "SK": ["slovakia", "slovak"],
    "CZ": ["czech", "czechia"],
    "HU": ["hungary", "hungarian"],
    "NL": ["netherlands", "dutch"],
    "BE": ["belgium", "belgian"],
    "CH": ["switzerland", "swiss"],
    "AT": ["austria", "austrian"],
    "PT": ["portugal", "portuguese"],
    "BD": ["bangladesh", "bengali", "bangla"],
    "LK": ["sri lanka", "sinhala", "sri lankan"],
    "IN": ["india", "indian", "hindi"],
    "TW": ["taiwan", "taiwanese"],
    "KR": ["south korea", "korean"],
    "AE": ["uae", "emirates", "emirati", "arabic"],
    "SI": ["slovenia", "slovenian"]
}

CV_GENERIC_CLUES = {
    "red soil", "yellow road lines", "white road lines", "concrete road pavement", "concrete pavement",
    "dominant white road markings with yellow curb lines",
    "arid dry desert landscape", "lush green vegetation", "temperate vegetation", "unpaved dirt road",
    "open fields / grassland", "low sun angle / temperate or subpolar latitude",
    "high sun angle / tropical latitude", "red google car"
}

GENERIC_CLUE_TITLES = {
    "architecture", "country flag", "licence plates", "chevrons", "kilometre markers",
    "bollards", "directional signs", "road lines", "red soil", "phone area codes",
    "pedestrian crossing signs", "utility poles", "chevron signs", "language",
    "bus stop signs", "topography", "telephone area codes", "stop signs",
    "striped signposts", "corn fields", "trident pole tops", "direction signs",
    "tea plantations", "area codes", "grey ev camera car", "vegetation", "nature",
    "soil", "houses", "landscape", "roads", "trees", "sun", "sky", "fields"
}

def normalize_text(text):
    if not text:
        return ""
    norm = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8')
    return re.sub(r'[^a-z0-9]+', ' ', norm.lower()).strip()

STOPWORDS = {
    "country", "state", "road", "border", "which", "there", "where", "angle",
    "latitude", "temperate", "subpolar", "tropical", "fields", "grassland",
    "vegetation", "landscape", "coverage", "street", "google", "lines",
    "white", "yellow", "asphalt", "unpaved", "paved", "driving", "right",
    "left", "found", "using", "common", "across", "along", "places", "often", "rural"
}

class OfflineGeoLocator:
    def __init__(self, kb_path="data/plonkit_kb.json", rules_path="data/country_rules.json", geoguessr_kb_path="data/geoguessr_kb.json", postmatch_kb_path=None):
        self.kb = GeoKnowledgeBase(kb_path, rules_path)
        self.geoguessr_kb = {}
        if os.path.exists(geoguessr_kb_path):
            try:
                with open(geoguessr_kb_path, "r", encoding="utf-8") as f:
                    self.geoguessr_kb = json.load(f)
            except Exception:
                pass
        self.postmatch_kb = {}
        
        target_postmatch = postmatch_kb_path
        if not target_postmatch:
            if os.path.exists("data/geoguessr_master_clues.json"):
                target_postmatch = "data/geoguessr_master_clues.json"
            else:
                target_postmatch = "data/geoguessr_postmatch_clues.json"
                
        if target_postmatch and os.path.exists(target_postmatch):
            try:
                with open(target_postmatch, "r", encoding="utf-8") as f:
                    self.postmatch_kb = json.load(f)
                if "all_clues" in self.postmatch_kb and "clues_by_id" not in self.postmatch_kb:
                    self.postmatch_kb["clues_by_id"] = self.postmatch_kb["all_clues"]
                elif "clues_by_id" in self.postmatch_kb and "all_clues" not in self.postmatch_kb:
                    self.postmatch_kb["all_clues"] = self.postmatch_kb["clues_by_id"]
            except Exception:
                pass

    def predict_from_features(self, driving_side=None, continent=None, clues_list=None, cv_clues=None, detected_params=None):
        """
        Calculates ranked country candidates from visual clues and detected features.
        Directly cross-references official GeoGuessr complete clue catalog and Plonk It guides.
        """
        if isinstance(driving_side, dict):
            data = driving_side
            driving_side = data.get("driving_side")
            continent = data.get("continent")
            clues_list = data.get("clues") or data.get("clues_list")
            cv_clues = data.get("cv_clues") or clues_list
            detected_params = data.get("detected_parameters") or data.get("detected_params")

        clues_list = clues_list or []
        cv_clues = cv_clues or []
        detected_params = detected_params or {}

        # Distinct explicit user/game clues vs generic physical computer vision signals
        explicit_clues = [c for c in clues_list if c.lower() not in CV_GENERIC_CLUES]
        all_cv = list(cv_clues) + [c for c in clues_list if c.lower() in CV_GENERIC_CLUES and c not in cv_clues]
        combined_clues = explicit_clues + all_cv

        # Detect sun position / hemisphere signals
        sun_south = any("sun visible to the south" in c.lower() or "sun-to-south" in c.lower() or "sun to south" in c.lower() for c in combined_clues)
        sun_north = any("sun visible to the north" in c.lower() or "sun-to-north" in c.lower() or "sun to north" in c.lower() for c in combined_clues)
        sun_overhead = any("sun overhead" in c.lower() or "sun visible overhead" in c.lower() or "high sun angle" in c.lower() for c in combined_clues)
        sun_low = any("low sun angle" in c.lower() for c in combined_clues)

        active_driving_side = driving_side or (detected_params.get("driving_side") if isinstance(detected_params, dict) else None)
        if not active_driving_side or not isinstance(active_driving_side, str):
            if any("driving on the right" in c.lower() for c in combined_clues):
                active_driving_side = "right"
            elif any("driving on the left" in c.lower() for c in combined_clues):
                active_driving_side = "left"
            else:
                active_driving_side = None

        scored_countries = []

        for code, country in self.kb.kb.items():
            base_score = 1.0 + COVERAGE_PRIORS.get(code, 0.2) * 1.5
            multiplier = 1.0
            matched = []
            matched_official = []
            all_text = " ".join([c["text"] for c in country.get("all_clues", [])]).lower()
            gg_info = self.geoguessr_kb.get("countries", {}).get(code, {})
            postmatch_clues = self.postmatch_kb.get("clues_by_country", {}).get(code, [])

            # 1. Driving side check
            if active_driving_side and isinstance(active_driving_side, str):
                country_side = gg_info.get("driving_side") or ("left" if code in [c["code"] for c in self.kb.rules.get("driving_side", {}).get("left", [])] else "right")
                if country_side == active_driving_side.lower():
                    base_score += 12.0
                    matched.append(f"driving on {country_side}")
                else:
                    multiplier *= 0.02

            # 1b. High Precision Vehicle Meta / Hardware Fingerprints
            has_kenya_snorkel = any("kenya" in c.lower() and "snorkel" in c.lower() for c in combined_clues) or (isinstance(detected_params, dict) and detected_params.get("car_meta") == "kenya snorkel")
            if has_kenya_snorkel:
                if code == "KE":
                    base_score += 150.0
                    matched.append("Google Car Snorkel (Kenya 100% Unique Meta)")
                elif code in ["UG", "TZ", "RW"]:
                    base_score += 15.0
                else:
                    multiplier *= 0.01

            has_red_car_clue = any("red google car" in c.lower() or "red car hood" in c.lower() for c in combined_clues) or (isinstance(detected_params, dict) and detected_params.get("car_meta") == "red car hood")
            if has_red_car_clue:
                if code == "UA":
                    base_score += 90.0
                    matched.append("Red Google car hood (Ukraine Gen 2/3 meta)")
                elif code == "RU":
                    base_score += 65.0
                    matched.append("Red Google car hood (Russia Gen 2/3 meta)")
                elif code == "BE":
                    base_score += 15.0
                else:
                    multiplier *= 0.10

            # 2. Hemisphere / Sun Position
            if sun_south:
                if code in NORTHERN_HEMISPHERE:
                    base_score += 12.0
                    matched.append("Sun to south (Northern Hemisphere)")
                elif code in SOUTHERN_HEMISPHERE and code not in EQUATORIAL_COUNTRIES:
                    multiplier *= 0.04
            elif sun_north:
                if code in SOUTHERN_HEMISPHERE:
                    base_score += 18.0
                    matched.append("Sun to north (Southern Hemisphere)")
                elif code in NORTHERN_HEMISPHERE and code not in EQUATORIAL_COUNTRIES:
                    multiplier *= 0.04
            elif sun_overhead:
                if code in EQUATORIAL_COUNTRIES:
                    base_score += 20.0
                    matched.append("High sun angle / tropical latitude (+20)")
                elif code in WHITE_LINES_EUROPE or code in HIGH_LATITUDE_NORTH:
                    multiplier *= 0.20

            elif sun_low:
                if code in HIGH_LATITUDE_NORTH:
                    base_score += 10.0
                    matched.append("Low sun angle / temperate latitude")
                elif code in EQUATORIAL_COUNTRIES:
                    multiplier *= 0.25

            # 3. GeoGuessr Official Master Clues Matching (ONLY against explicit clues)
            for pm_clue in postmatch_clues:
                c_id_norm = normalize_text(pm_clue.get("id", ""))
                c_title_raw = pm_clue.get("title", "")
                c_title_norm = normalize_text(c_title_raw)
                c_desc_norm = normalize_text(pm_clue.get("description", ""))
                c_type = pm_clue.get("type", "") or pm_clue.get("category", "")

                for user_clue in explicit_clues:
                    u_norm = normalize_text(user_clue)
                    if not u_norm or u_norm in [
                        "vehicles drive on the right", "vehicles drive on the left", 
                        "traffic here appears to drive on the right side of the road",
                        "drives on right", "drives on left", "sun to north", "sun to south"
                    ]:
                        continue

                    # Exact ID match (e.g. "br parana pine" vs "br-parana-pine")
                    if c_id_norm and (c_id_norm == u_norm or c_id_norm in u_norm or (len(u_norm) > 6 and u_norm in c_id_norm)):
                        base_score += 60.0
                        matched.append(f"Official Clue ID: {pm_clue.get('title')} ({c_type})")
                        matched_official.append(pm_clue)
                        continue

                    # Exact Title match
                    if c_title_norm and c_title_norm == u_norm:
                        if c_title_norm in GENERIC_CLUE_TITLES:
                            base_score += 15.0
                        else:
                            base_score += 50.0
                        matched.append(f"Official Clue (Exact): {pm_clue.get('title')} ({c_type})")
                        matched_official.append(pm_clue)
                        continue

                    # Substring match if title is not generic
                    if c_title_norm and c_title_norm not in GENERIC_CLUE_TITLES:
                        if c_title_norm in u_norm or (len(c_title_norm) > 6 and u_norm in c_title_norm):
                            base_score += 35.0
                            matched.append(f"Official Clue: {pm_clue.get('title')} ({c_type})")
                            matched_official.append(pm_clue)
                            continue

                    # Word overlap with clue title or description
                    title_words = [w for w in c_title_norm.split() if len(w) > 4 and w not in STOPWORDS]
                    if len(title_words) >= 2 and all(w in u_norm for w in title_words[:2]):
                        base_score += 25.0
                        matched.append(f"Official Clue Match: {pm_clue.get('title')}")
                        matched_official.append(pm_clue)

            # 4. Demonym & Country Name Specific Signals
            demonyms = DEMONYMS.get(code, [])
            for d in demonyms:
                d_norm = normalize_text(d)
                for user_clue in explicit_clues:
                    u_norm = normalize_text(user_clue)
                    if re.search(r'\b' + re.escape(d_norm) + r'\b', u_norm):
                        base_score += 120.0
                        matched.append(f"Country identifier '{d}'")
                        break

            # Country code prefix matching (e.g. "vn-", "es-", "mk-", "br-", "us-")
            cc_lower = code.lower()
            cc_matches = 0
            for user_clue in explicit_clues:
                u_norm = normalize_text(user_clue)
                if u_norm.startswith(cc_lower + " ") or u_norm == cc_lower or u_norm.startswith(cc_lower + "-"):
                    cc_matches += 1
            if cc_matches > 0:
                base_score += min(130.0, 50.0 + cc_matches * 30.0)
                matched.append(f"Country code identifier [{code}] ({cc_matches} clues)")

            # 5. Continent & Regional Context Matching
            CONTINENT_KEYWORDS = {
                "north america": ["US", "CA", "MX"],
                "south america": ["BR", "AR", "CL", "CO", "PE", "EC", "UY", "BO", "PY"],
                "europe": ["UA", "PL", "DE", "FR", "ES", "IT", "RO", "CZ", "SK", "AT", "HU", "BG", "LT", "LV", "EE", "GB", "IE", "NL", "BE", "DK", "NO", "SE", "FI", "CH", "PT", "GR", "SI", "HR", "RS", "MK", "AL", "ME", "AD"],
                "africa": ["ZA", "KE", "UG", "GH", "SN", "BW", "NG", "NA", "SZ", "LS", "EG", "TN"],
                "asia": ["JP", "KR", "TW", "TH", "VN", "ID", "MY", "PH", "IN", "BD", "LK", "SG", "KH", "LA"],
                "oceania": ["AU", "NZ"]
            }
            has_explicit_country_id = any(
                m.startswith("Country code identifier") or m.startswith("Country identifier")
                for m in matched
            )
            for cont_name, cont_codes in CONTINENT_KEYWORDS.items():
                cont_pattern = r'\b' + re.escape(cont_name) + r'\b'
                is_cont_context = (continent and cont_name in continent.lower()) or any(
                    re.search(cont_pattern, normalize_text(c)) and not re.search(cont_pattern + r'\s+(influenced|style|architecture|settler|heritage|type)', normalize_text(c))
                    for c in combined_clues
                )
                if is_cont_context:
                    if code in cont_codes:
                        base_score += 25.0
                        matched.append(f"Continent meta ({cont_name})")
                    elif not has_explicit_country_id:
                        multiplier *= 0.15

            # 6. Distinctive Meta Clues & Pixel Signals
            has_red_soil = any("red soil" in c.lower() for c in combined_clues) or float(detected_params.get("red_soil_pct", 0)) > 3.5
            has_yellow_lines = any("yellow road line" in c.lower() for c in combined_clues)
            has_white_lines = any("white road line" in c.lower() for c in combined_clues)
            has_concrete = any("concrete road pavement" in c.lower() or "concrete pavement" in c.lower() for c in combined_clues)
            has_arid = any("arid" in c.lower() or "desert" in c.lower() for c in combined_clues)
            has_lush = any("lush" in c.lower() for c in combined_clues)
            has_unpaved = any("unpaved" in c.lower() or "dirt" in c.lower() for c in combined_clues)
            has_red_car = any("red google car" in c.lower() or "red car" in c.lower() for c in combined_clues)
            has_snorkel = any("snorkel" in c.lower() for c in combined_clues)
            has_tape = any("black tape" in c.lower() or "tape on roof rack" in c.lower() for c in combined_clues)

            # Red Soil scoring
            if has_red_soil:
                if code in RED_SOIL_COUNTRIES:
                    pts = RED_SOIL_COUNTRIES[code]
                    base_score += pts * 2.0
                    matched.append(f"red soil biome (+{pts})")
                elif "red soil" in all_text:
                    base_score += 3.0
                    matched.append("local red soil meta")
                elif code in WHITE_LINES_EUROPE and has_white_lines:
                    # European countries frequently have red clay / terra rossa soil alongside white road lines
                    base_score += 8.0
                else:
                    multiplier *= 0.3

            # Concrete pavement scoring
            if has_concrete:
                if code in ["US", "CA"]:
                    base_score += 20.0
                    matched.append("North American concrete pavement standard")
                elif code in ["MX", "BR", "AR", "CL"]:
                    base_score += 8.0
                    matched.append("Latin American concrete pavement")
                elif code in ["ES", "GR", "IT", "TR", "PT", "HR", "BE"]:
                    base_score += 12.0
                    matched.append("Mediterranean / European concrete or sun-bleached pavement")

            # Yellow road lines scoring
            if has_yellow_lines:
                if code in YELLOW_LINES_COUNTRIES:
                    pts = YELLOW_LINES_COUNTRIES[code]
                    base_score += pts * 2.0
                    matched.append(f"yellow line standard (+{pts})")
                elif code in WHITE_LINES_EUROPE and code not in ["NO", "FI", "IS", "GB", "IE"]:
                    multiplier *= 0.1
                elif "yellow" in all_text:
                    base_score += 2.0

                has_yellow_center = any("yellow center line" in c.lower() for c in combined_clues)
                if has_yellow_center:
                    if code in ["US", "CA"]:
                        base_score += 70.0
                        matched.append("North American yellow center line with white edge lines")
                    elif code in ["MX", "BR", "CO", "AR", "CL"]:
                        base_score += 25.0
                        matched.append("American continent yellow center line")
                    elif code in ["GB", "IE"]:
                        multiplier *= 0.15
                    elif code in ["ZA", "BW", "SZ", "LS", "NA"]:
                        multiplier *= 0.15
                elif any("white center line with yellow edge markings" in c.lower() or "yellow edge markings with white road lines" in c.lower() for c in combined_clues):
                    is_southern_africa = (
                        active_driving_side == "left" or
                        sun_north or
                        "Southern Hemisphere" in str(detected_params.get("hemisphere_deduction", ""))
                    )
                    if is_southern_africa and code in ["ZA", "BW", "SZ", "LS", "NA"]:
                        base_score += 70.0
                        matched.append("Southern African standard: white center line with yellow edge markings")
                    elif not is_southern_africa and code in ["IE", "GB"]:
                        pts = 75.0 if code == "IE" else 45.0
                        base_score += pts
                        matched.append("Irish/UK standard: yellow road edge markings with white road lines")
                    elif not is_southern_africa and code in ["US", "CA"]:
                        multiplier *= 0.20
                    elif code in WHITE_LINES_EUROPE and code not in ["IE", "GB"]:
                        multiplier *= 0.15
                elif any("dominant white road markings with yellow curb lines" in c.lower() for c in combined_clues):
                    if code in ["GB", "IE"]:
                        base_score += 55.0
                        matched.append("UK / Ireland road marking standard: white road lines with yellow curb restrictions")
                    elif code in ["US", "CA"]:
                        base_score += 6.0
                elif has_white_lines:
                    # Southern African standard: outer yellow lines with white center markings
                    is_southern_africa_env = (
                        active_driving_side == "left" or 
                        sun_north or 
                        (not sun_south and has_red_soil and active_driving_side != "right")
                    )
                    if is_southern_africa_env and code in ["ZA", "BW", "SZ", "LS", "NA"]:
                        base_score += 55.0
                        matched.append("Southern African standard: yellow edge lines with white center lines")
                    elif active_driving_side == "left" and code in ["JP"]:
                        base_score += 15.0
                    elif code in ["US", "CA"]:
                        base_score += 42.0
                        matched.append("North American yellow center line with white edge lines")
                    elif code in ["MX", "BR", "CO", "AR", "CL"]:
                        base_score += 24.0
                        matched.append("American continent yellow center line")
                    elif code in ["GB", "IE"]:
                        base_score += 35.0
                        matched.append("UK / Ireland white road lines with yellow edge markings")
                elif not has_red_soil and code in ["US", "CA"]:
                    base_score += 16.0
                    matched.append("North American yellow center line (non-tropical soil)")

            # North American MUTCD diamond yellow warning sign / hazard marker
            has_yellow_sign = any("yellow hazard sign" in c.lower() or "warning marker" in c.lower() for c in combined_clues) or (detected_params.get("road_sign") == "North American yellow sign")
            if has_yellow_sign:
                if code in ["US", "CA"]:
                    base_score += 70.0
                    matched.append("North American MUTCD diamond yellow warning sign standard")
                elif code == "MX":
                    base_score += 45.0
                    matched.append("North American / Mexican yellow diamond sign standard")
                elif code in ["AU", "NZ", "IE", "BR"]:
                    base_score += 15.0
                elif code in WHITE_LINES_EUROPE and code not in ["SE", "PL", "IS"]:
                    # European countries use triangular warning signs with red borders, not MUTCD yellow diamonds
                    multiplier *= 0.08

            # Arid Mediterranean / Anatolian landscape & terra cotta architecture
            has_mediterranean = any("arid mediterranean" in c.lower() or "anatolian" in c.lower() for c in combined_clues)
            if has_mediterranean:
                if code in ["TR", "GR", "ES", "IT", "PT", "CY", "HR", "ME", "AL"]:
                    pts = 55.0 if code in ["TR", "GR", "ES"] else 35.0
                    base_score += pts
                    matched.append(f"Mediterranean / Anatolian arid landscape & architecture (+{pts})")
                elif code in ["JO", "IL", "TN", "EG"]:
                    base_score += 30.0
                    matched.append("Levantine / North African arid landscape")
                else:
                    multiplier *= 0.20

            # White road lines scoring
            if has_white_lines and not has_yellow_lines:
                if has_concrete and code in ["US", "CA"]:
                    base_score += 8.0
                    matched.append("concrete multi-lane white road markings")

                if has_yellow_sign:
                    if code in ["US", "CA"]:
                        base_score += 25.0
                        matched.append("North American white road markings")
                elif code in WHITE_LINES_EUROPE:
                    base_score += 30.0
                    matched.append("European white road marking standard")
                elif (active_driving_side == "left" or sun_north) and code in ["AU", "NZ"]:
                    base_score += 22.0
                    matched.append("Oceanic white road markings (LHT standard)")
                elif code in ["RU", "KZ", "TR", "MN", "KG"]:
                    base_score += 18.0
                    matched.append("Eurasian white road markings")
                elif code in ["IN", "LK", "BD", "TH", "MY", "ID", "PH", "VN", "KH", "JP", "TW", "KR"]:
                    base_score += 18.0
                    matched.append("Asian white road markings standard")
                elif sun_north and code in ["ZA", "BW", "NA"]:
                    base_score += 18.0
                    matched.append("Southern African white road markings")
                elif code in ["CL", "AR", "UY"]:
                    base_score += 14.0
                    matched.append("Southern Cone white road markings")
                elif code in ["US", "CA"]:
                    base_score += 4.0
                    matched.append("white road markings (North American standard)")
                elif code in ["MX", "BR", "CO", "PE"]:
                    base_score += 4.0
                    matched.append("Latin American white road markings")
            elif has_white_lines:
                if code in WHITE_LINES_EUROPE:
                    base_score += 6.0
                    matched.append("white road markings")
                elif code in ["AU", "NZ", "JP", "GB", "IE"]:
                    base_score += 6.0
                    matched.append("white road markings")
                elif code in ["US", "CA", "MX", "BR", "CO"]:
                    base_score += 2.0

            # Biome & Vegetation scoring
            has_temperate = any("temperate vegetation" in c.lower() for c in combined_clues)
            if has_temperate:
                if code in [
                    "DE", "FR", "PL", "UA", "GB", "CZ", "SK", "AT", "CH", "US", "CA", "NZ", 
                    "JP", "CL", "AR", "RO", "HU", "IT", "ES", "GR", "TR", "HR", "SI", "SE", 
                    "NO", "FI", "IE", "NL", "BE", "DK", "EE", "LV", "LT", "BG", "RS", "MK"
                ]:
                    base_score += 10.0
                    matched.append("temperate biome")
                elif code in ["CR", "TH", "MY", "ID", "PH", "VN", "SG"]:
                    multiplier *= 0.25
                elif code in ["AE", "QA", "OM", "EG", "JO", "SA"]:
                    multiplier *= 0.1

            # Arid / Desert scoring
            if has_arid:
                if code in ARID_COUNTRIES:
                    pts = ARID_COUNTRIES[code]
                    base_score += pts * 2.5
                    matched.append(f"arid desert environment (+{pts})")
                else:
                    multiplier *= 0.2

            # Lush green scoring
            if has_lush:
                is_tropical_env = (
                    sun_overhead or 
                    any("tropical" in c.lower() for c in combined_clues) or 
                    any("equatorial" in c.lower() for c in combined_clues) or
                    "Equatorial" in str(detected_params.get("hemisphere_deduction", ""))
                )
                if is_tropical_env and code in TROPICAL_LUSH_COUNTRIES:
                    pts = TROPICAL_LUSH_COUNTRIES[code]
                    base_score += pts * 2.2
                    matched.append(f"tropical lush foliage (+{pts})")
                elif code in LUSH_COUNTRIES:
                    pts = LUSH_COUNTRIES[code]
                    base_score += pts * 1.5
                    matched.append(f"lush foliage (+{pts})")
                elif code in ["AE", "QA", "OM", "EG", "JO", "SA", "MN", "KG", "KZ", "BW", "NA"]:
                    multiplier *= 0.1

            # Dirt road scoring
            if has_unpaved and not (has_yellow_lines or has_white_lines):
                if code in UNPAVED_COUNTRIES:
                    pts = UNPAVED_COUNTRIES[code]
                    base_score += pts * 2.5
                    matched.append(f"common unpaved coverage (+{pts})")

            # Concrete curb and gutter / wide suburban road
            has_curb_gutter = any("concrete curb" in c.lower() for c in combined_clues)
            if has_curb_gutter:
                if code in ["US", "CA"]:
                    base_score += 20.0
                    matched.append("North American wide suburban concrete curb and gutter")
                elif code in ["AU", "NZ"]:
                    base_score += 10.0
                    matched.append("Oceanic suburban concrete curb and gutter")
                elif code in ["CO", "MX", "BR", "AR", "CL", "PE"]:
                    base_score += 7.0
                    matched.append("Latin American suburban concrete curb")

            # Distinctive Red Soil / Outback Terrain
            has_outback_soil = any("outback terrain" in c.lower() or "distinctive red soil" in c.lower() for c in combined_clues)
            if has_outback_soil:
                if has_unpaved and code in ["BR", "NG", "UG", "KE", "GH", "MG", "SN", "BO", "PY", "CO"]:
                    base_score += 65.0
                    matched.append("Tropical / Southern Hemisphere unpaved red laterite dirt road")
                elif not has_unpaved and (sun_south or not sun_north) and code in ["US", "MX"]:
                    base_score += 58.0
                    matched.append("North American red desert sandstone / arid terrain")
                elif code == "AU":
                    base_score += 55.0
                    matched.append("Iconic Australian red outback dirt & ferric soil")
                elif code in ["NG", "UG", "KE", "GH", "MG", "SN", "ZA", "BW", "BR", "NA"]:
                    base_score += 40.0
                    matched.append("African / Southern Hemisphere red soil terrain")
                elif not has_unpaved and code in ["US", "MX"]:
                    base_score += 48.0
                    matched.append("North American red desert sandstone / arid terrain")
                else:
                    multiplier *= 0.05

            # British red sandstone architecture / UK urban
            has_uk_sandstone = any("british red sandstone" in c.lower() for c in combined_clues)
            if has_uk_sandstone:
                if code in ["GB", "IE"]:
                    base_score += 35.0
                    matched.append("British Victorian red sandstone architecture / UK urban aesthetic")
                elif code in ["US", "CA", "NL", "BE", "DE"]:
                    base_score += 14.0
                    matched.append("North American / European brick architecture")
                else:
                    multiplier *= 0.40

            # Post-Soviet panel block architecture / Eurasian urban
            has_post_soviet = any("post-soviet panel" in c.lower() for c in combined_clues)
            if has_post_soviet:
                if code in ["MN", "RU", "KZ", "KG", "UA", "BY", "BG"]:
                    base_score += 42.0
                    matched.append("Post-Soviet concrete panel block architecture & Eurasian urban layout")
                else:
                    multiplier *= 0.1

            # North American / American wooden utility poles & overhead wiring
            has_na_poles = any("north american utility poles" in c.lower() for c in combined_clues)
            if has_na_poles:
                if code in ["US", "CA"]:
                    base_score += 18.0
                    matched.append("North American wooden utility pole infrastructure")
                elif code in ["CO", "MX", "BR", "PE"]:
                    base_score += 14.0
                    matched.append("American wooden / concrete utility pole infrastructure")
                elif any("dominant white road markings with yellow curb lines" in c.lower() for c in combined_clues):
                    # Do not penalize UK/IE when yellow curb restrictions are present
                    pass
                elif code in ["GB", "IE", "DE", "FR", "NL", "BE", "DK", "SE", "NO"]:
                    multiplier *= 0.65

            # Red Google car
            if has_red_car:
                if code == "UA":
                    base_score += 55.0
                    matched.append("Gen 3 red Google car (unique to Ukraine)")
                elif code == "BE":
                    base_score += 15.0
                    matched.append("Red car meta (rare in Belgium)")
                else:
                    multiplier *= 0.05

            # Snorkel meta
            if has_snorkel:
                if code == "KE":
                    base_score += 65.0
                    matched.append("Google Car Snorkel (iconic Kenya meta)")
                elif code == "MN":
                    base_score += 25.0
                    matched.append("Snorkel meta (Mongolia Gen 2)")
                elif code == "CW":
                    base_score += 20.0
                else:
                    multiplier *= 0.1

            # Black tape on roof rack
            if has_tape:
                if code == "GH":
                    base_score += 65.0
                    matched.append("Black tape on roof rack (unique Ghana meta)")
                else:
                    multiplier *= 0.1

            # Custom text clues matching against Plonk It Guide (ONLY explicit user clues)
            for clue in explicit_clues:
                clue_lower = clue.lower()
                if clue_lower in CV_GENERIC_CLUES:
                    continue
                if clue_lower in all_text:
                    base_score += 8.0
                    matched.append(clue)
                else:
                    words = [w for w in clue_lower.split() if len(w) > 4 and w not in STOPWORDS]
                    matches = sum(1 for w in words if w in all_text)
                    if matches >= 2:
                        base_score += matches * 2.5
                        matched.append(f"Plonk It guide meta match ({matches} terms)")

            # Apply final constraint multiplier
            final_score = base_score * multiplier

            # Deduplicate matched list
            seen_m = set()
            unique_matched = []
            for m in matched:
                if m not in seen_m:
                    seen_m.add(m)
                    unique_matched.append(m)

            if unique_matched:
                if not matched_official and postmatch_clues:
                    matched_official = postmatch_clues[:3]

                scored_countries.append({
                    "country": country["title"],
                    "code": code,
                    "continents": country.get("continents", []),
                    "score": final_score,
                    "matched_features": unique_matched,
                    "official_clues": matched_official[:4],
                    "gg_info": gg_info
                })

        if not scored_countries:
            for code, country in list(self.kb.kb.items())[:5]:
                postmatch = self.postmatch_kb.get("clues_by_country", {}).get(code, [])[:3]
                scored_countries.append({
                    "country": country["title"],
                    "code": code,
                    "continents": country.get("continents", []),
                    "score": 1.0,
                    "matched_features": ["General geographic profile"],
                    "official_clues": postmatch,
                    "gg_info": {}
                })

        scored_countries.sort(key=lambda x: x["score"], reverse=True)

        top_candidates = scored_countries[:5]
        # Softmax / exponential normalization for realistic confidence spread
        scores = [c["score"] for c in top_candidates]
        max_s = max(scores)
        exp_scores = [math.exp((s - max_s) / 8.0) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [round((e / sum_exp) * 100, 1) for e in exp_scores]

        top = top_candidates[0]
        top_coords = COUNTRY_CENTERS.get(top["code"], (0.0, 0.0))
        top_country_data = self.kb.get_country(top["code"])
        region_hint = "General Region"
        if top_country_data and top_country_data.get("regional_clues"):
            region_hint = top_country_data["regional_clues"][0]["section"]

        alternatives = []
        for idx, c in enumerate(top_candidates[1:5], 1):
            c_data = self.kb.get_country(c["code"])
            c_coords = COUNTRY_CENTERS.get(c["code"], (0.0, 0.0))
            reg = "General"
            if c_data and c_data.get("regional_clues"):
                reg = c_data["regional_clues"][0]["section"]
            alternatives.append({
                "country": c["country"],
                "country_code": c["code"],
                "region": reg,
                "confidence_percent": probs[idx],
                "gps_estimate": {"lat": c_coords[0], "lng": c_coords[1]},
                "why_considered": f"Matched features: {', '.join(c['matched_features'][:3])}",
                "official_clues": c.get("official_clues", [])[:2]
            })

        # Reasoning
        reasoning = f"Analysis based on Plonk It database and GeoGuessr Learning Hub. {top['country']} scored highest with matching features: {', '.join(top['matched_features'][:4])}."
        if top["gg_info"]:
            info = top["gg_info"]
            reasoning += f" Official GeoGuessr profile: Driving side: {info.get('driving_side', 'N/A')}, Capital: {info.get('capital', 'N/A')}, Domain: {info.get('domain', 'N/A')}."

        return {
            "top_prediction": {
                "country": top["country"],
                "country_code": top["code"],
                "region": region_hint,
                "confidence_percent": probs[0],
                "gps_estimate": {"lat": top_coords[0], "lng": top_coords[1]},
                "matched_features": top["matched_features"],
                "official_clues": top.get("official_clues", [])
            },
            "alternative_candidates": alternatives,
            "identified_clues": combined_clues,
            "detected_parameters": detected_params,
            "plonkit_meta_reasoning": reasoning,
            "engine_used": "Local Computer Vision & Plonk It Intelligence"
        }

    def predict_image_heuristics(self, image_path, south_image_path=None, user_clues=""):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if south_image_path and os.path.exists(south_image_path):
            cv_result = compare_panoramas(image_path, south_image_path)
        else:
            cv_result = analyze_panorama_image(image_path)

        visual_clues = list(cv_result.get("clues", []))

        user_clues_list = []
        if user_clues:
            for c in user_clues.split(","):
                clean = c.strip()
                if clean:
                    user_clues_list.append(clean)

        return self.predict_from_features(
            clues_list=user_clues_list,
            cv_clues=visual_clues,
            detected_params=cv_result.get("detected_parameters", {})
        )

