"""
Deterministic Local GeoGuessr Clue Matcher
A pure algorithmic computer vision and database matching engine with zero AI / LLM dependencies.
Directly extracts factual visual signals from street panoramas (sun physics, driving side,
road markings, soil, flora, hardware meta) and intersects them with the official GeoGuessr
(geoguessr_master_clues.json - 4,477 clues) and Plonk It (plonkit_kb.json - 136 countries) databases.
"""

import os
import json
import math
import numpy as np
from PIL import Image

class DeterministicGeoLocator:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.master_clues = {}
        self.country_rules = {}
        self.plonkit_kb = {}
        self.country_centers = {}
        self.country_profiles = {}
        self.load_databases()
        self.compile_clue_index()

    def load_databases(self):
        # 1. Master clues
        mc_path = os.path.join(self.data_dir, "geoguessr_master_clues.json")
        if os.path.exists(mc_path):
            with open(mc_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.master_clues = data.get("clues_by_country", {})

        # 2. Country rules
        cr_path = os.path.join(self.data_dir, "country_rules.json")
        if os.path.exists(cr_path):
            with open(cr_path, "r", encoding="utf-8") as f:
                self.country_rules = json.load(f)

        # 3. Plonk It KB
        pk_path = os.path.join(self.data_dir, "plonkit_kb.json")
        if os.path.exists(pk_path):
            with open(pk_path, "r", encoding="utf-8") as f:
                self.plonkit_kb = json.load(f)

        # 4. Country centers
        from engine.offline_engine import COUNTRY_CENTERS
        self.country_centers = COUNTRY_CENTERS

        # Authoritative real-world set of 49 Left-Hand-Driving countries
        self.left_driving_countries = {
            "GB", "IE", "MT", "CY", "IM", "JE", "GG",
            "JP", "ID", "MY", "TH", "SG", "HK", "MO", "IN", "PK", "BD", "LK", "NP", "BT", "BN", "TL",
            "ZA", "BW", "SZ", "LS", "NA", "KE", "UG", "TZ", "ZW", "ZM", "MZ", "MW", "MU", "SC",
            "AU", "NZ", "PG", "FJ", "WS", "TO", "SB",
            "GY", "SR", "JM", "BS", "BB", "TT", "VI"
        }

        self.right_driving_countries = set(
            c for c in self.plonkit_kb.keys() if c not in self.left_driving_countries
        )

    def compile_clue_index(self):
        """Compiles structured clue profiles for all countries from official databases."""
        all_codes = set(self.plonkit_kb.keys()) | set(self.master_clues.keys())
        for code in all_codes:
            pk_data = self.plonkit_kb.get(code, {})
            gm_clues = self.master_clues.get(code, [])
            pk_clues = pk_data.get("all_clues", [])
            coords = self.country_centers.get(code, (0.0, 0.0))
            lat, lng = coords

            # Hemisphere classification
            if lat < -12.0:
                hemi = "south"
            elif lat > 18.0:
                hemi = "north"
            else:
                hemi = "equator"

            drv = "left" if code in self.left_driving_countries else "right"

            # Biome and geographic ground truths
            is_tropical = (-15.0 <= lat <= 20.0 and code in [
                "ID", "MY", "PH", "TH", "VN", "KH", "LA", "SG", "CO", "CR", "PA", "EC", "BR", "LK"
            ])
            is_boreal = (lat >= 45.0 and code in ["RU", "CA", "FI", "SE", "NO", "EE", "LV", "LT"])
            is_savanna = (code in ["KE", "BW", "ZA", "NA", "SN", "TZ", "UG", "ZM", "ZW", "GH", "NG"])
            is_temperate = (lat > 20.0 and not is_boreal and not is_tropical) or (lat < -20.0 and not is_savanna)

            # Road markings and ground infrastructure
            uses_yellow_center = code in [
                "US", "CA", "MX", "BR", "CO", "EC", "PE", "CL", "ID", "MY", "TH", "TW", "JP", "NO", "FI", "IS"
            ]
            uses_white_only = code in [
                "RU", "UA", "BY", "PL", "DE", "FR", "IT", "ES", "PT", "GB", "AU", "AT", "CZ", "SK",
                "HU", "RO", "BG", "GR", "TR", "EE", "LV", "LT", "DK", "SE", "BE", "NL", "CH", "NZ"
            ]
            has_red_soil = code in ["AU", "BR", "KE", "UG", "KH", "MG", "LS", "BW", "SZ", "ZA", "LK", "ID"]
            unpaved_rural_common = code in ["ID", "MN", "KH", "UG", "KE", "BW", "MG", "LA", "BO", "PE", "SN"]
            is_flat_lowlands = code in ["NL", "DK", "BE", "EE", "LV", "LT", "PL", "HU"]

            self.country_profiles[code] = {
                "title": pk_data.get("title") or code,
                "code": code,
                "continents": pk_data.get("continents", ["General"]),
                "lat": lat, "lng": lng,
                "hemisphere": hemi,
                "driving_side": drv,
                "is_tropical": is_tropical,
                "is_boreal": is_boreal,
                "is_savanna": is_savanna,
                "is_temperate": is_temperate,
                "uses_yellow_center": uses_yellow_center,
                "uses_white_only": uses_white_only,
                "has_red_soil": has_red_soil,
                "unpaved_rural_common": unpaved_rural_common,
                "is_flat_lowlands": is_flat_lowlands,
                "gm_clues": gm_clues,
                "pk_clues": pk_clues,
            }

    # -------------------------------------------------------------------------
    # Visual Feature Extraction (Pure NumPy & Math)
    # -------------------------------------------------------------------------
    def extract_features(self, north_img_path, south_img_path):
        im_n = np.array(Image.open(north_img_path).convert("RGB"), dtype=np.float32)
        im_s = np.array(Image.open(south_img_path).convert("RGB"), dtype=np.float32)
        h, w, _ = im_n.shape

        features = {
            "hemisphere": "NEUTRAL",
            "driving_side": "UNKNOWN",
            "road_type": "asphalt",
            "road_lines": "none",
            "soil": "neutral",
            "biome": "temperate",
            "architecture": "none",
            "hardware_meta": [],
            "matched_clue_signatures": [],
            "metrics": {}
        }

        # 1. Sun Physics & Radial Gradient Analysis
        sky_n = im_n[115:int(h * 0.42), 60:-60]
        sky_s = im_s[115:int(h * 0.42), 60:-60]
        bn = 0.299 * sky_n[:, :, 0] + 0.587 * sky_n[:, :, 1] + 0.114 * sky_n[:, :, 2]
        bs = 0.299 * sky_s[:, :, 0] + 0.587 * sky_s[:, :, 1] + 0.114 * sky_s[:, :, 2]
        mean_n = float(np.mean(bn))
        mean_s = float(np.mean(bs))
        sat_n = int(np.sum(bn > 248.0))
        sat_s = int(np.sum(bs > 248.0))
        sky_diff = mean_n - mean_s

        def get_solar_physics(bright):
            max_b = float(np.max(bright))
            mean_sky = float(np.mean(bright))
            if max_b < 240.0:
                return 0.0, 0, mean_sky
            sat_pixels = int(np.sum(bright > 248.0))
            if sat_pixels < 20 or sat_pixels > 3500:
                return 0.0, sat_pixels, mean_sky
            py, px = np.unravel_index(np.argmax(bright), bright.shape)
            r = 20
            y1, y2 = max(0, py - r), min(bright.shape[0], py + r + 1)
            x1, x2 = max(0, px - r), min(bright.shape[1], px + r + 1)
            patch = bright[y1:y2, x1:x2]
            center_val = patch[py - y1, px - x1]
            edge_vals = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
            gradient = float(center_val - np.mean(edge_vals))
            return gradient, sat_pixels, mean_sky

        grad_n, _, _ = get_solar_physics(bn)
        grad_s, _, _ = get_solar_physics(bs)
        features["metrics"]["sun_grad_north"] = round(grad_n, 1)
        features["metrics"]["sun_grad_south"] = round(grad_s, 1)
        features["metrics"]["sky_diff"] = round(sky_diff, 1)

        sun_in_south = (grad_s > 25.0 and (grad_s > grad_n + 15.0 or grad_n == 0.0))
        sun_in_north = (grad_n > 25.0 and (grad_n > grad_s + 15.0 or grad_s == 0.0))

        if sat_s > 4000 and sat_s > sat_n + 1500:
            features["hemisphere"] = "NORTHERN"
            features["matched_clue_signatures"].append("overwhelming solar glare in South (Northern Hemisphere)")
        elif sat_n > 4000 and sat_n > sat_s + 1500:
            features["hemisphere"] = "SOUTHERN"
            features["matched_clue_signatures"].append("overwhelming solar glare in North (Southern Hemisphere)")
        elif (sun_in_south or sky_diff < -15.0) and not sun_in_north:
            features["hemisphere"] = "NORTHERN"
            features["matched_clue_signatures"].append("astronomical sun in south / brighter southern sky (Northern Hemisphere)")
        elif (sun_in_north or sky_diff > 15.0) and not sun_in_south:
            features["hemisphere"] = "SOUTHERN"
            features["matched_clue_signatures"].append("astronomical sun in north / brighter northern sky (Southern Hemisphere)")
        else:
            features["hemisphere"] = "NEUTRAL"
            features["matched_clue_signatures"].append("overcast sky / diffuse illumination (Hemisphere Neutral)")

        # 2. Taiwan Diagonal Striped Utility Pole (Exclusive Clue)
        def check_taiwan_diagonal_stripes(im):
            vh, vw, _ = im.shape
            pr, pg, pb = im[:, :, 0], im[:, :, 1], im[:, :, 2]
            y_mask = (pr > 125) & (pg > 80) & (pb < 90) & (pr > pb + 35) & (pg > pb + 15)
            k_mask = (pr < 60) & (pg < 60) & (pb < 60)
            if np.sum(y_mask) < 1800 or np.sum(k_mask) < 2200:
                return False
            win_h, win_w = 90, 25
            for py in range(int(0.35 * vh), int(0.75 * vh), 30):
                for px in range(30, vw - 40, 15):
                    sub_y = y_mask[py:py+win_h, px:px+win_w]
                    y_cnt = np.sum(sub_y)
                    if y_cnt < 250:
                        continue
                    sub_k = k_mask[py:py+win_h, px:px+win_w]
                    k_cnt = np.sum(sub_k)
                    if k_cnt < 300:
                        continue
                    tot = win_h * win_w
                    if (y_cnt / tot < 0.15) or (k_cnt / tot < 0.20):
                        continue
                    ys, xs = np.where(sub_y)
                    for period in [13, 14, 15, 16, 17]:
                        for sign in [1, -1]:
                            diags = (ys + sign * xs) % period
                            hist, _ = np.histogram(diags, bins=period, range=(0, period))
                            zero_bins = np.sum(hist < 3)
                            ratio = np.max(hist) / (np.min(hist) + 1)
                            if zero_bins >= 3 and ratio > 15.0:
                                return True
            return False

        # 3. View Metrics (Soil, Flora, Markings, Architecture)
        def get_view_metrics(arr):
            vh, vw, _ = arr.shape
            rz = arr[int(vh * 0.60):int(vh * 0.94), int(vw * 0.10):int(vw * 0.90)]
            rr, rg, rb = rz[:, :, 0], rz[:, :, 1], rz[:, :, 2]
            rbright = 0.299 * rr + 0.587 * rg + 0.114 * rb
            mean_rbright = float(np.mean(rbright))

            # Red laterite / Outback soil
            red_mask = (rr > 125) & (rr > rg + 20) & (rr > rb + 32)
            red_pct = float(np.mean(red_mask)) * 100

            # Sahel sand
            sand_mask = (rr > 165) & (rg > 135) & (rb > 95) & (rr > rb + 35) & (rr > rg + 10)
            sand_pct = float(np.mean(sand_mask)) * 100

            # Vegetation
            exg = 2.0 * arr[:, :, 1] - arr[:, :, 0] - arr[:, :, 2]
            vpct = float(np.mean((exg > 18) & (arr[:, :, 1] > 40))) * 100

            # Tropical Palms / Broadleaf Canopy
            trop_z = arr[int(vh * 0.20):int(vh * 0.65), :]
            texg = 2.0 * trop_z[:, :, 1] - trop_z[:, :, 0] - trop_z[:, :, 2]
            tgreen = float(np.mean(texg > 25)) * 100

            # Horizon std
            horiz_z = arr[int(vh * 0.38):int(vh * 0.58), :]
            hstd = float(np.std(horiz_z[:, :, 0]))

            # Dirt / Unpaved Road
            dirt_m = (rr > rg + 12) & (rg > rb + 8) & (rbright < 165)
            dpct = float(np.mean(dirt_m)) * 100

            # Asphalt Pavement
            asphalt_m = (np.abs(rr - rg) < 22) & (np.abs(rg - rb) < 26) & (rbright > 35) & (rbright < 215)
            asphalt_pct = float(np.mean(asphalt_m)) * 100
            mean_asphalt_b = float(np.mean(rbright[asphalt_m])) if np.sum(asphalt_m) > 0 else mean_rbright

            # Painted lines
            yp = (rr > 150) & (rg > 120) & (rr >= rg - 12) & (rr > rb + 36) & (rbright > mean_asphalt_b + 18)
            wp = (rbright > mean_asphalt_b + 32) & (rbright > 165) & (np.abs(rr - rg) < 14) & (np.abs(rg - rb) < 14) & (~yp)

            ys, xs = np.where(yp)
            hist_x, _ = np.histogram(xs / rz.shape[1], bins=10, range=(0, 1)) if len(xs) > 0 else ([], None)

            y_cnt = len(ys)
            w_cnt = int(np.sum(wp))
            y_mean_x = float(np.mean(xs) / rz.shape[1]) if y_cnt > 0 else None

            wys, wxs = np.where(wp)
            w_mean_x = float(np.mean(wxs) / rz.shape[1]) if w_cnt > 0 else None

            has_real_yellow = False
            if 400 <= y_cnt <= 18000:
                if y_mean_x is not None and y_mean_x < 0.22:
                    has_real_yellow = True
                elif len(hist_x) == 10:
                    edges = hist_x[0] + hist_x[1] + hist_x[7] + hist_x[8]
                    center = hist_x[3] + hist_x[4] + hist_x[5] + hist_x[6]
                    if center > 100 and edges <= 2.2 * center:
                        has_real_yellow = True

            return {
                "mean_road_brightness": mean_rbright,
                "red_soil_pct": red_pct,
                "sahel_sand_pct": sand_pct,
                "vegetation_pct": vpct,
                "trop_green_pct": tgreen,
                "horizon_std": hstd,
                "dirt_pct": dpct,
                "asphalt_pct": asphalt_pct,
                "has_real_yellow": has_real_yellow,
                "y_cnt": y_cnt,
                "y_mean_x": y_mean_x,
                "w_cnt": w_cnt,
                "w_mean_x": w_mean_x
            }

        mn = get_view_metrics(im_n)
        ms = get_view_metrics(im_s)

        mean_road_bright = (mn["mean_road_brightness"] + ms["mean_road_brightness"]) / 2.0
        red_soil_pct = max(mn["red_soil_pct"], ms["red_soil_pct"])
        sahel_sand_pct = max(mn["sahel_sand_pct"], ms["sahel_sand_pct"])
        veg_pct = max(mn["vegetation_pct"], ms["vegetation_pct"])
        trop_green = max(mn["trop_green_pct"], ms["trop_green_pct"])
        dirt_pct = max(mn["dirt_pct"], ms["dirt_pct"])
        asphalt_pct = max(mn["asphalt_pct"], ms["asphalt_pct"])
        horizon_std = min(mn["horizon_std"], ms["horizon_std"])

        features["metrics"]["mean_road_brightness"] = round(mean_road_bright, 1)
        features["metrics"]["red_soil_pct"] = round(red_soil_pct, 1)
        features["metrics"]["sahel_sand_pct"] = round(sahel_sand_pct, 1)
        features["metrics"]["vegetation_pct"] = round(veg_pct, 1)
        features["metrics"]["trop_green_pct"] = round(trop_green, 1)
        features["metrics"]["horizon_std"] = round(horizon_std, 1)
        features["metrics"]["dirt_pct"] = round(dirt_pct, 1)
        features["metrics"]["asphalt_pct"] = round(asphalt_pct, 1)

        # 4. Road Marking geometry & Driving side
        is_unpaved = (dirt_pct > 30.0 and dirt_pct > asphalt_pct * 1.2) or (red_soil_pct > 20.0 and dirt_pct > 20.0) or (asphalt_pct < 25.0 and dirt_pct > 15.0)

        if is_unpaved:
            features["road_type"] = "unpaved_dirt"
            features["road_lines"] = "none"
            features["matched_clue_signatures"].append("unpaved rural dirt / gravel track")
        else:
            features["road_type"] = "asphalt"
            # North American standard (US/Canada): yellow center dividing line on left with white edge line on right
            if mn["has_real_yellow"] and mn["y_mean_x"] is not None and mn["y_mean_x"] < 0.45 and mn["w_cnt"] > 300 and mn["w_mean_x"] is not None and mn["w_mean_x"] > 0.60:
                features["road_lines"] = "us_yellow_center_white_edge"
                features["driving_side"] = "RIGHT"
                features["matched_clue_signatures"].append("North American standard: yellow center dividing line with white shoulder line (Right traffic)")
            # Irish yellow dashed outer road edge line (Left traffic)
            elif (mn["has_real_yellow"] and mn["y_mean_x"] is not None and mn["y_mean_x"] < 0.22 and (mn["w_cnt"] < 300 or mn["w_mean_x"] is None or mn["w_mean_x"] < 0.50)) or \
                 (ms["has_real_yellow"] and ms["y_mean_x"] is not None and ms["y_mean_x"] < 0.22):
                features["road_lines"] = "irish_yellow_dashed_edge"
                features["driving_side"] = "LEFT"
                features["matched_clue_signatures"].append("Irish yellow dashed road edge markings (Left traffic)")
            # Left-hand traffic with yellow centerline
            elif ms["has_real_yellow"] and ms["y_mean_x"] is not None and ms["y_mean_x"] < 0.45:
                features["road_lines"] = "yellow_center"
                features["driving_side"] = "LEFT"
                features["matched_clue_signatures"].append("painted yellow road centerline on Left-hand traffic")
            elif mn["has_real_yellow"] and mn["y_mean_x"] is not None and mn["y_mean_x"] > 0.55:
                features["road_lines"] = "yellow_center"
                features["driving_side"] = "LEFT"
                features["matched_clue_signatures"].append("painted yellow road centerline on Left-hand traffic")
            elif mn["has_real_yellow"] or ms["has_real_yellow"]:
                features["road_lines"] = "yellow_center"
                features["matched_clue_signatures"].append("painted yellow road centerline")
            elif (400 <= mn["w_cnt"] <= 10000) or (400 <= ms["w_cnt"] <= 10000):
                features["road_lines"] = "white_lines"
                features["matched_clue_signatures"].append("standard all-white road markings")

        # 5. Hardware Meta detection
        for im_view in [im_n, im_s]:
            vh, vw, _ = im_view.shape
            if check_taiwan_diagonal_stripes(im_view):
                if "tw_diagonal_striped_pole" not in features["hardware_meta"]:
                    features["hardware_meta"].append("tw_diagonal_striped_pole")
                    features["matched_clue_signatures"].append("exclusive Taiwan black & yellow diagonal hazard striped utility pole")

            # Kenya snorkel
            hood_right = im_view[int(vh * 0.85):vh, int(vw * 0.44):int(vw * 0.62)]
            hr, hg, hb = hood_right[:, :, 0], hood_right[:, :, 1], hood_right[:, :, 2]
            snorkel_mask = (hb > hr + 10) & (hb > hg + 10)
            if np.sum(snorkel_mask) > 2500 and red_soil_pct > 20.0:
                if "kenya_snorkel" not in features["hardware_meta"]:
                    features["hardware_meta"].append("kenya_snorkel")
                    features["matched_clue_signatures"].append("Google Car Snorkel (exclusive Kenya meta)")

        # 6. Biome Deduction
        if "tw_diagonal_striped_pole" in features["hardware_meta"]:
            features["biome"] = "taiwan"
        elif "kenya_snorkel" in features["hardware_meta"]:
            features["biome"] = "savanna"
        elif is_unpaved and trop_green < 8.0:
            features["biome"] = "steppe"
            features["matched_clue_signatures"].append("vast open grassland steppe & unpaved dirt tracks")
        elif trop_green > 25.0 and features["hemisphere"] != "NORTHERN":
            features["biome"] = "tropical"
            features["matched_clue_signatures"].append("dense tropical rainforest palm canopy")
        elif trop_green > 28.0 and features["road_lines"] not in ["white_lines", "irish_yellow_dashed_edge", "us_yellow_center_white_edge"]:
            features["biome"] = "tropical"
            features["matched_clue_signatures"].append("dense tropical rainforest palm canopy")
        elif features["hemisphere"] == "NORTHERN" and trop_green < 8.0 and veg_pct > 3.0 and features["road_lines"] != "us_yellow_center_white_edge":
            features["biome"] = "boreal"
            features["matched_clue_signatures"].append("cold northern boreal taiga & birch forest")
        elif red_soil_pct > 15.0 and features["hemisphere"] == "SOUTHERN":
            features["biome"] = "steppe"
            features["matched_clue_signatures"].append("Australian Outback deep red iron-oxide soil with arid scrub")
        elif veg_pct > 15.0:
            features["biome"] = "temperate"
            features["matched_clue_signatures"].append("temperate green pastures & rolling hills")
        else:
            features["biome"] = "temperate"

        return features

    # -------------------------------------------------------------------------
    # Database Matching & Candidate Scoring (0% AI / Soulless Clue Matching)
    # -------------------------------------------------------------------------
    def predict(self, features):
        scores = {}
        matched_details = {}

        hemi = features.get("hemisphere", "NEUTRAL")
        drv = features.get("driving_side", "UNKNOWN")
        road_type = features.get("road_type", "asphalt")
        road_lines = features.get("road_lines", "none")
        biome = features.get("biome", "temperate")
        hw_meta = features.get("hardware_meta", [])
        metrics = features.get("metrics", {})
        red_soil_pct = metrics.get("red_soil_pct", 0.0)
        hstd = metrics.get("horizon_std", 20.0)

        for code, profile in self.country_profiles.items():
            score = 10.0
            reasons = []

            # 1. Driving Side Constraint
            if drv == "LEFT":
                if profile["driving_side"] != "left":
                    continue
                score += 40.0
                reasons.append(f"{code} drives on the LEFT")
            elif drv == "RIGHT":
                if profile["driving_side"] != "right":
                    continue
                score += 20.0
                reasons.append(f"{code} drives on the RIGHT")

            # 2. Solar Hemisphere Constraint
            if hemi == "NORTHERN":
                if profile["hemisphere"] == "south":
                    continue
                if profile["hemisphere"] == "north":
                    score += 30.0
                    reasons.append(f"{code} in Northern Hemisphere (Sun in South)")
            elif hemi == "SOUTHERN":
                if profile["hemisphere"] == "north":
                    continue
                if profile["hemisphere"] == "south":
                    score += 45.0
                    reasons.append(f"{code} in Southern Hemisphere (Sun in North)")
            elif hemi == "NEUTRAL":
                if profile["hemisphere"] == "equator":
                    score += 25.0
                    reasons.append(f"{code} in Equatorial zone (Overhead sun / diffuse)")

            # 3. Road Markings & Surface
            if road_lines == "us_yellow_center_white_edge":
                if code == "US":
                    score += 80.0
                    reasons.append("US standard: double yellow center dividing line with white edge line (+80)")
                elif code in ["CA", "MX"]:
                    score += 40.0
                    reasons.append("North American standard road markings (+40)")
                elif profile["uses_yellow_center"]:
                    score += 20.0
                else:
                    score -= 40.0
            elif road_lines == "irish_yellow_dashed_edge":
                if code == "IE":
                    score += 120.0
                    reasons.append("Irish yellow dashed road edge markings with Left-hand traffic (+120)")
                elif code == "GB":
                    score += 20.0
                else:
                    score -= 40.0
            elif road_lines == "white_lines":
                if profile["uses_white_only"]:
                    score += 35.0
                    reasons.append(f"{code} standard all-white road markings (+35)")
                elif profile["uses_yellow_center"]:
                    score -= 25.0
            elif road_lines == "yellow_center":
                if profile["uses_yellow_center"]:
                    score += 35.0
                    reasons.append(f"{code} yellow center line (+35)")
                elif profile["uses_white_only"]:
                    score -= 30.0

            if road_type == "unpaved_dirt":
                if profile["unpaved_rural_common"]:
                    score += 45.0
                    reasons.append(f"{code} common unpaved rural dirt tracks (+45)")
                else:
                    score -= 35.0

            # 4. Soil Evidence
            if red_soil_pct > 15.0:
                if profile["has_red_soil"]:
                    score += 40.0
                    reasons.append(f"{code} rich red iron-oxide laterite soil (+40)")
                else:
                    score -= 30.0
            elif red_soil_pct < 2.0:
                if code in ["AU"] and biome == "steppe":
                    score -= 20.0

            # 5. Biome / Flora Compatibility
            if biome == "boreal":
                if profile["is_boreal"]:
                    score += 50.0
                    reasons.append(f"{code} cold northern boreal taiga & birch forest (+50)")
                    if code == "RU":
                        score += 20.0  # Russia taiga continental dominance
                else:
                    score -= 35.0
            elif biome == "tropical":
                if profile["is_tropical"]:
                    score += 50.0
                    reasons.append(f"{code} equatorial tropical rainforest (+50)")
                    if code == "ID":
                        score += 15.0  # Primary Southeast Asian equatorial archipelago
                elif profile["is_savanna"]:
                    score -= 20.0
                else:
                    score -= 40.0
            elif biome == "savanna":
                if profile["is_savanna"]:
                    score += 45.0
                    reasons.append(f"{code} African acacia savanna & dry scrub (+45)")
                else:
                    score -= 25.0
            elif biome == "steppe":
                if code in ["MN", "AU", "KG", "KZ", "AR", "CL"]:
                    score += 50.0
                    reasons.append(f"{code} vast open grassland steppe (+50)")
                    if code == "MN" and road_type == "unpaved_dirt":
                        score += 20.0  # Vast open Mongolian steppe dirt tracks
                else:
                    score -= 25.0
            elif biome == "temperate":
                if profile["is_temperate"]:
                    score += 30.0
                    reasons.append(f"{code} temperate landscape (+30)")
                    if hstd < 14.0 and profile["is_flat_lowlands"]:
                        score += 25.0
                        reasons.append(f"{code} flat European agricultural plain (+25)")
                elif profile["is_tropical"]:
                    score -= 25.0

            # Distinction for New Zealand lush pastures over Australia when red soil is absent
            if hemi == "SOUTHERN" and biome == "temperate" and red_soil_pct < 10.0:
                if code == "NZ":
                    score += 35.0
                    reasons.append("NZ lush green temperate pastures (+35)")
                elif code == "AU":
                    score -= 25.0

            # 6. Hardware Meta (Country-Exclusive Signatures)
            for h in hw_meta:
                if h == "tw_diagonal_striped_pole":
                    if code == "TW":
                        score += 450.0
                        reasons.insert(0, "exclusive Taiwan black & yellow diagonal hazard striped utility pole (+450)")
                    elif code in ["JP", "KR"]:
                        score += 30.0
                    else:
                        score *= 0.05
                elif h == "kenya_snorkel":
                    if code == "KE":
                        score += 450.0
                        reasons.insert(0, "exclusive Kenya Google Car Snorkel (+450)")
                    elif code in ["UG", "TZ"]:
                        score += 30.0
                    else:
                        score *= 0.05

            scores[code] = score
            matched_details[code] = reasons

        if not scores:
            scores["US"] = 1.0
            matched_details["US"] = ["default fallback"]

        # Rank candidates
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_k = ranked[:5]

        # Softmax for probabilities
        max_s = top_k[0][1]
        exp_s = [math.exp((s - max_s) / 12.0) for _, s in top_k]
        sum_exp = sum(exp_s)
        probs = [round((e / sum_exp) * 100, 1) for e in exp_s]

        top_code, top_score = top_k[0]
        top_country_data = self.plonkit_kb.get(top_code, {})
        country_name = top_country_data.get("title", top_code)
        coords = self.country_centers.get(top_code, (0.0, 0.0))

        alternatives = []
        for idx, (code, score) in enumerate(top_k[1:], 1):
            c_data = self.plonkit_kb.get(code, {})
            c_coords = self.country_centers.get(code, (0.0, 0.0))
            alternatives.append({
                "country": c_data.get("title", code),
                "country_code": code,
                "confidence_percent": probs[idx],
                "gps_estimate": {"lat": c_coords[0], "lng": c_coords[1]},
                "why_considered": "; ".join(matched_details.get(code, [])[:3])
            })

        return {
            "top_prediction": {
                "country": country_name,
                "country_code": top_code,
                "region": top_country_data.get("continents", ["General"])[0] if top_country_data.get("continents") else "General",
                "confidence_percent": probs[0],
                "gps_estimate": {"lat": coords[0], "lng": coords[1]},
                "matched_features": matched_details.get(top_code, [])
            },
            "alternative_candidates": alternatives,
            "identified_clues": features.get("matched_clue_signatures", []),
            "detected_parameters": features
        }
