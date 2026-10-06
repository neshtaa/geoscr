"""
Computer Vision Feature Extractor for Street View & GeoGuessr Panoramas
Analyzes image pixels (sky, sun, road lines, soil, vegetation biome, car blur)
to extract factual visual clues without requiring external cloud APIs.
"""

import os
import math
import numpy as np
from PIL import Image

def analyze_panorama_image(image_path):
    """
    Performs comprehensive visual inspection of a panorama or Street View image.
    Returns a dictionary of detected visual characteristics and matching clues.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        width, height = img_rgb.size
        target_w = 800
        target_h = int(height * (800 / width))
        resized = img_rgb.resize((target_w, target_h), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)

    h, w, _ = arr.shape
    features = {
        "aspect_ratio": round(w / h, 2),
        "is_panorama": (w / h) >= 1.7,
        "clues": [],
        "detected_parameters": {}
    }

    # -------------------------------------------------------------
    # 1. Sky & Sun Analysis (Top 35% of the image)
    # -------------------------------------------------------------
    sky_zone = arr[:int(h * 0.35), :]
    sky_r = sky_zone[:, :, 0]
    sky_g = sky_zone[:, :, 1]
    sky_b = sky_zone[:, :, 2]
    sky_brightness = 0.299 * sky_r + 0.587 * sky_g + 0.114 * sky_b

    max_bright = np.max(sky_brightness)
    features["detected_parameters"]["max_sky_brightness"] = round(float(max_bright), 1)

    blue_ratio = np.mean(sky_b / (sky_r + sky_g + 1e-5))
    sky_std = float(np.std(sky_brightness))

    if blue_ratio > 0.62:
        features["detected_parameters"]["sky_condition"] = "clear blue"
    elif sky_std < 22 and np.mean(sky_brightness) > 175:
        features["detected_parameters"]["sky_condition"] = "overcast"
    else:
        features["detected_parameters"]["sky_condition"] = "partly cloudy"

    mean_sky_b = float(np.mean(sky_brightness))
    sat_sky_pct = float(np.sum(sky_brightness > 246) / sky_brightness.size) * 100
    features["detected_parameters"]["mean_sky_brightness"] = round(mean_sky_b, 1)
    features["detected_parameters"]["sat_sky_pct"] = round(sat_sky_pct, 1)

    # Precise Sun detection: check for distinct compact solar cluster (not diffuse overcast clouds)
    bright_cluster = sky_brightness > 235
    bright_pct = float(np.sum(bright_cluster) / sky_brightness.size) * 100

    if max_bright > 225 and (0.01 <= bright_pct <= 10.0):
        sun_y_idx, sun_x_idx = np.where(bright_cluster)
        x_spread = np.max(sun_x_idx) - np.min(sun_x_idx)
        x_std = float(np.std(sun_x_idx))

        # Direct sun disk has concentrated X spread (< 220px, std < 70)
        # Broadly scattered clouds have wide X spread across the horizon
        if x_spread < 220 and x_std < 70:
            sun_y = float(np.mean(sun_y_idx))
            sun_x = float(np.mean(sun_x_idx))
            sun_norm_y = sun_y / (h * 0.35)
            sun_norm_x = sun_x / w
            features["detected_parameters"]["sun_position"] = {
                "x_ratio": round(float(sun_norm_x), 2),
                "y_ratio": round(float(sun_norm_y), 2)
            }
            if sun_norm_y < 0.35:
                features["clues"].append("high sun angle / tropical latitude")
                features["detected_parameters"]["sun_angle"] = "high"
            elif sun_norm_y > 0.55:
                features["clues"].append("low sun angle / temperate or subpolar latitude")
                features["detected_parameters"]["sun_angle"] = "low"
            else:
                features["detected_parameters"]["sun_angle"] = "moderate"
        else:
            features["detected_parameters"]["sky_condition"] = "overcast diffuse"

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # 2. Vehicle Meta & Camera Artifacts (Snorkel, Red car, Roof rack)
    # -------------------------------------------------------------
    # A. Ukraine / Russia red car hood in Gen 2/3 (broadened to bottom corners)
    car_zone = arr[int(h * 0.86):h, int(w * 0.20):int(w * 0.80)]
    c_r = car_zone[:, :, 0].astype(np.float32)
    c_g = car_zone[:, :, 1].astype(np.float32)
    c_b = car_zone[:, :, 2].astype(np.float32)

    red_car_paint = (c_r > 150) & (c_r > c_g + 45) & (c_r > c_b + 45) & (c_g < 100) & (c_b < 100)
    has_red_car = bool(np.sum(red_car_paint) > 50)
    if has_red_car:
        features["clues"].append("red Google car (Ukraine / Russia meta)")
        features["detected_parameters"]["google_car"] = "Red Car (Ukraine / Russia Meta)"
        features["detected_parameters"]["car_meta"] = "red car hood"

    # B. Kenya Google car snorkel (chrome/metallic curved intake tube on front right hood)
    hood_center = arr[int(h * 0.89):h, int(w * 0.40):int(w * 0.54)]
    if hood_center.size > 0:
        hr = hood_center[:, :, 0].astype(np.float32)
        hg = hood_center[:, :, 1].astype(np.float32)
        hb = hood_center[:, :, 2].astype(np.float32)
        hbright = (hr + hg + hb) / 3.0
        metallic_snorkel = (hb > hr + 7) & (hbright > 135) & (hbright < 245)
        if np.sum(metallic_snorkel) > 1200:
            features["clues"].append("kenya google car snorkel (Kenya)")
            features["detected_parameters"]["google_car"] = "Kenya Snorkel (Kenya)"
            features["detected_parameters"]["car_meta"] = "kenya snorkel"

    # Organic Ferric Red Soil
    ground_zone = arr[int(h * 0.50):, :]
    gr_r = ground_zone[:, :, 0].astype(np.int16)
    gr_g = ground_zone[:, :, 1].astype(np.int16)
    gr_b = ground_zone[:, :, 2].astype(np.int16)

    soil_mask = (gr_r > 130) & (gr_r > gr_g + 20) & (gr_r > gr_b + 35) & (gr_g > gr_b + 8)
    soil_pct = float(np.sum(soil_mask) / soil_mask.size) * 100
    features["detected_parameters"]["red_soil_pct"] = round(soil_pct, 2)

    if soil_pct > 18.0:
        features["clues"].append("distinctive red soil / outback terrain")
        features["detected_parameters"]["soil"] = "distinctive red outback terrain"
    elif soil_pct > 3.5:
        features["clues"].append("red soil")
        features["detected_parameters"]["soil"] = "red soil"

    # -------------------------------------------------------------
    # 3. Road Surface & Markings
    # Focus directly on asphalt/driving corridor; eliminate horizon and vehicle hood
    # -------------------------------------------------------------
    road_zone = arr[int(h * 0.62):int(h * 0.94), int(w * 0.08):int(w * 0.92)]
    road_r = road_zone[:, :, 0].astype(np.float32)
    road_g = road_zone[:, :, 1].astype(np.float32)
    road_b = road_zone[:, :, 2].astype(np.float32)
    road_bright = 0.299 * road_r + 0.587 * road_g + 0.114 * road_b
    mean_road_bright = float(np.mean(road_bright))
    features["detected_parameters"]["road_mean_brightness"] = round(mean_road_bright, 1)

    max_rgb = np.maximum(np.maximum(road_r, road_g), road_b)
    min_rgb = np.minimum(np.minimum(road_r, road_g), road_b)
    sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-5), 0)

    # 1. Unpaved dirt road detection (MUST precede line detection to prevent dirt triggering paint lines)
    dirt_mask = (road_r > road_g + 18) & (road_g > road_b + 12) & (sat > 0.28)
    dirt_pct = float(np.sum(dirt_mask) / dirt_mask.size) * 100
    features["detected_parameters"]["dirt_pct"] = round(dirt_pct, 1)

    road_type = "asphalt"
    is_concrete = False
    is_dirt = False
    if dirt_pct > 26.0:
        is_dirt = True
        features["clues"].append("unpaved dirt road")
        features["detected_parameters"]["road_type"] = "dirt"
        road_type = "dirt"
    elif mean_road_bright > 192 and float(np.mean(sat)) < 0.10:
        is_concrete = True
        features["clues"].append("concrete road pavement")
        features["detected_parameters"]["road_surface"] = "concrete"
        features["detected_parameters"]["road_type"] = "concrete"
    else:
        features["detected_parameters"]["road_surface"] = "asphalt"
        features["detected_parameters"]["road_type"] = "asphalt"

    # 2. Road line detection (suppressed completely on dirt roads)
    if is_dirt:
        has_yellow = False
        has_white = False
        yellow_pixel_pct = 0.0
        white_pixel_pct = 0.0
    else:
        # Yellow traffic paint: Red must be >= Green - 5 (Yellow/Orange hue, not green!)
        # With significant blue absorption and brighter than the road surface
        yellow_line_mask = (
            (road_r > 160) & (road_g > 130) &
            (road_r >= road_g - 5) &
            (road_r > road_b + 38) & (road_g > road_b + 18) &
            (sat > 0.22) & (road_bright > mean_road_bright + 8)
        )
        yellow_pixel_pct = float(np.sum(yellow_line_mask) / yellow_line_mask.size) * 100

        if is_concrete:
            white_line_mask = (road_bright > mean_road_bright + 30) & (road_bright > 195) & (sat < 0.12) & (~yellow_line_mask)
        else:
            white_line_mask = (road_bright > mean_road_bright + 35) & (road_bright > 165) & (sat < 0.14) & (~yellow_line_mask)

        white_pixel_pct = float(np.sum(white_line_mask) / white_line_mask.size) * 100

        has_yellow = (not has_red_car) and (yellow_pixel_pct > 0.35) and (yellow_pixel_pct > white_pixel_pct * 0.08)
        has_white = (not has_red_car) and (white_pixel_pct > 0.35) and (white_pixel_pct > yellow_pixel_pct * 0.08)

    features["detected_parameters"]["yellow_line_pct"] = round(yellow_pixel_pct, 2)
    features["detected_parameters"]["white_line_pct"] = round(white_pixel_pct, 2)

    # 3. Lane layout and relative position
    if has_yellow and has_white:
        y_idx, x_idx = np.where(yellow_line_mask)
        wy_idx, wx_idx = np.where(white_line_mask)
        rw = road_zone.shape[1]
        x_norm = x_idx / rw if len(x_idx) > 0 else np.array([0.5])
        wx_norm = wx_idx / rw if len(wx_idx) > 0 else np.array([0.5])
        mean_y_x = float(np.mean(x_norm))
        mean_w_x = float(np.mean(wx_norm))

        # Check whether yellow is in the inner roadway or outer roadside edges
        # Outer edges: x < 0.25 (left shoulder/curb) or x > 0.75 (right shoulder/curb)
        y_inner_ratio = float(np.sum((x_norm >= 0.25) & (x_norm <= 0.75)) / len(x_norm))

        if y_inner_ratio < 0.28:
            # Yellow lines are on the outer edges/curbs, white lines are on the roadway/center
            # (Standard in Ireland [yellow dashed edges], South Africa [yellow outer edges], UK [yellow curb restrictions])
            features["clues"].append("white center line with yellow edge markings")
            features["clues"].append("yellow edge markings with white road lines")
            features["clues"].append("yellow road lines")
            features["clues"].append("white road lines")
            features["detected_parameters"]["road_lines"] = "white center line with yellow edge markings"
            features["detected_parameters"]["yellow_line_position"] = "edges"
        elif mean_y_x < 0.48 and mean_w_x > 0.50:
            features["clues"].append("yellow center line with white road markings")
            features["clues"].append("yellow road lines")
            features["clues"].append("white road lines")
            features["detected_parameters"]["road_lines"] = "yellow center line with white markings"
            features["detected_parameters"]["yellow_line_position"] = "center"
        elif mean_w_x < 0.48 and mean_y_x > 0.50:
            features["clues"].append("white center line with yellow edge markings")
            features["clues"].append("yellow road lines")
            features["clues"].append("white road lines")
            features["detected_parameters"]["road_lines"] = "white center line with yellow edge markings"
            features["detected_parameters"]["yellow_line_position"] = "edges"
        elif abs(mean_y_x - 0.5) < abs(mean_w_x - 0.5):
            features["clues"].append("yellow center line with white road markings")
            features["clues"].append("yellow road lines")
            features["clues"].append("white road lines")
            features["detected_parameters"]["road_lines"] = "yellow center line with white markings"
            features["detected_parameters"]["yellow_line_position"] = "center"
        else:
            features["clues"].append("yellow road lines")
            features["clues"].append("white road lines")
            features["detected_parameters"]["road_lines"] = "yellow and white"
    elif has_yellow:
        features["clues"].append("yellow road lines")
        features["detected_parameters"]["road_lines"] = "yellow"
    elif has_white:
        features["clues"].append("white road lines")
        features["detected_parameters"]["road_lines"] = "white"

    # 4. North American MUTCD yellow hazard markers / warning signs
    marker_zone = arr[int(h * 0.40):int(h * 0.72), :]
    mz_r = marker_zone[:, :, 0].astype(np.float32)
    mz_g = marker_zone[:, :, 1].astype(np.float32)
    mz_b = marker_zone[:, :, 2].astype(np.float32)
    yellow_sign = (mz_r > 125) & (mz_g > 90) & (mz_b < 45) & (mz_r > mz_b + 65) & (mz_g > mz_b + 35)
    if np.sum(yellow_sign) > 60:
        features["clues"].append("North American yellow hazard sign / warning marker")
        features["detected_parameters"]["road_sign"] = "North American yellow sign"

    # -------------------------------------------------------------
    # 5. Vegetation & Biome (Excess Green Index: 2*G - R - B)
    # -------------------------------------------------------------
    mid_zone = arr[int(h * 0.20):int(h * 0.65), :]
    m_r = mid_zone[:, :, 0]
    m_g = mid_zone[:, :, 1]
    m_b = mid_zone[:, :, 2]
    
    exg = 2.0 * m_g - m_r - m_b
    green_mask = (exg > 15) & (m_g > 45)
    veg_pct = float(np.sum(green_mask) / green_mask.size) * 100
    features["detected_parameters"]["vegetation_pct"] = round(veg_pct, 1)

    if veg_pct > 32.0:
        features["clues"].append("lush green vegetation")
        features["detected_parameters"]["biome"] = "lush green foliage"
    elif dirt_pct > 25.0 and veg_pct < 5.0:
        features["clues"].append("arid dry desert landscape")
        features["detected_parameters"]["biome"] = "arid / dry desert"
    elif veg_pct < 9.0 and soil_pct < 15.0 and dirt_pct < 20.0:
        features["clues"].append("arid Mediterranean / Anatolian landscape")
        features["detected_parameters"]["biome"] = "arid Mediterranean"
    elif veg_pct > 9.0:
        features["clues"].append("temperate vegetation")
        features["detected_parameters"]["biome"] = "temperate rural"
    # -------------------------------------------------------------
    # 5. Architecture, Utility Infrastructure & Urban Metas
    # -------------------------------------------------------------
    facade_zone = arr[int(h * 0.10):int(h * 0.45), :]
    f_r, f_g, f_b = facade_zone[:, :, 0], facade_zone[:, :, 1], facade_zone[:, :, 2]

    # British red sandstone / brick Victorian tenements (strictly dense urban facades)
    red_sandstone_mask = (f_r > 125) & (f_r > f_g + 18) & (f_r > f_b + 35) & (f_g > f_b + 8) & (f_b < 120)
    sandstone_pct = float(np.sum(red_sandstone_mask) / red_sandstone_mask.size) * 100
    if sandstone_pct > 8.0 and soil_pct < 6.0 and veg_pct < 18.0:
        features["clues"].append("british red sandstone architecture / uk urban")
        features["detected_parameters"]["architecture"] = "british red sandstone"

    # Soviet / Eastern Bloc concrete panel blocks (repeated grey balconies/facade)
    max_c = np.maximum(np.maximum(f_r, f_g), f_b)
    min_c = np.minimum(np.minimum(f_r, f_g), f_b)
    panel_sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-5), 0)
    panel_mask = (max_c > 115) & (max_c < 200) & (panel_sat < 0.12)
    panel_pct = float(np.sum(panel_mask) / panel_mask.size) * 100
    
    # Require sharp edge density (balconies, windows, rooflines) to distinguish buildings from smooth sky
    diff_y = np.abs(facade_zone[1:, :, 0] - facade_zone[:-1, :, 0])
    diff_x = np.abs(facade_zone[:, 1:, 0] - facade_zone[:, :-1, 0])
    edge_pct = float(np.mean((diff_y[:, :-1] + diff_x[:-1, :]) > 14)) * 100

    if panel_pct > 28.0 and edge_pct > 1.8 and soil_pct < 6.0 and veg_pct < 16.0:
        features["clues"].append("post-soviet panel block architecture / eurasian urban")
        features["detected_parameters"]["architecture"] = "post-soviet panel block"

    return features


def compare_panoramas(north_image_path, south_image_path):
    """
    Analyzes dual perspective Street View frames (snapped North and rotated South).
    Extracts conclusive hemisphere orientation (Sun North vs Sun South)
    and combines road and vegetation features across the full panorama.
    """
    fn = analyze_panorama_image(north_image_path)
    fs = analyze_panorama_image(south_image_path)

    combined_clues = []
    seen = set()

    for c in fn.get("clues", []) + fs.get("clues", []):
        # Exclude tentative individual sun angles from individual frames
        if "sun" in c.lower():
            continue
        if c not in seen:
            seen.add(c)
            combined_clues.append(c)

    params_n = fn.get("detected_parameters", {})
    params_s = fs.get("detected_parameters", {})

    # If any frame detects paved road or painted road markings, suppress unpaved dirt road
    has_paved_road = (
        "yellow road lines" in combined_clues or 
        "white road lines" in combined_clues or 
        "concrete road pavement" in combined_clues or
        params_n.get("road_type") == "asphalt" or 
        params_s.get("road_type") == "asphalt"
    )
    if has_paved_road and "unpaved dirt road" in combined_clues:
        combined_clues.remove("unpaved dirt road")

    # Clean up mutually exclusive center line clues
    has_na_lines = "yellow center line with white road markings" in combined_clues
    has_za_lines = "white center line with yellow edge markings" in combined_clues
    if has_na_lines and has_za_lines:
        if "yellow center line with white road markings" in fn.get("clues", []):
            combined_clues.remove("white center line with yellow edge markings")
        else:
            combined_clues.remove("yellow center line with white road markings")

    max_n = params_n.get("max_sky_brightness", 0.0)
    max_s = params_s.get("max_sky_brightness", 0.0)
    sun_n = params_n.get("sun_position")
    sun_s = params_s.get("sun_position")

    cond_n = params_n.get("sky_condition", "")
    cond_s = params_s.get("sky_condition", "")
    is_diffuse = "diffuse" in cond_n or "diffuse" in cond_s or "overcast" in cond_n or "overcast" in cond_s

    mean_bn = params_n.get("mean_sky_brightness", 0.0)
    mean_bs = params_s.get("mean_sky_brightness", 0.0)
    sat_n = params_n.get("sat_sky_pct", 0.0)
    sat_s = params_s.get("sat_sky_pct", 0.0)

    # Hemisphere deduction based on astronomical sun position and glare differential
    if sun_n and not sun_s:
        combined_clues.append("sun visible to the north")
        hemisphere = "Southern Hemisphere (Sun clearly visible to the North)"
    elif sun_s and not sun_n:
        combined_clues.append("sun visible to the south")
        hemisphere = "Northern Hemisphere (Sun clearly visible to the South)"
    elif sat_n > 8.0 and sat_n > sat_s * 2.5:
        combined_clues.append("sun visible to the north")
        hemisphere = "Southern Hemisphere (Solar bloom clearly in the North)"
    elif sat_s > 8.0 and sat_s > sat_n * 2.5:
        combined_clues.append("sun visible to the south")
        hemisphere = "Northern Hemisphere (Solar bloom clearly in the South)"
    elif (mean_bn - mean_bs) >= 22.0:
        combined_clues.append("sun visible to the north")
        hemisphere = "Southern Hemisphere (Highest sky illumination to the North)"
    elif (mean_bs - mean_bn) >= 22.0:
        combined_clues.append("sun visible to the south")
        hemisphere = "Northern Hemisphere (Highest sky illumination to the South)"
    elif not is_diffuse and max_n > 230 and (max_n - max_s) >= 35:
        combined_clues.append("sun visible to the north")
        hemisphere = "Southern Hemisphere (Highest solar glare to the North)"
    elif not is_diffuse and max_s > 230 and (max_s - max_n) >= 35:
        combined_clues.append("sun visible to the south")
        hemisphere = "Northern Hemisphere (Highest solar glare to the South)"
    elif (sat_n > 35.0 and sat_s > 35.0) or (sun_n and sun_s and sun_n.get("y_ratio", 1.0) < 0.25 and sun_s.get("y_ratio", 1.0) < 0.25):
        combined_clues.append("high sun angle / tropical latitude")
        hemisphere = "Equatorial / Tropical zone"
    else:
        hemisphere = "Overcast / Diffuse illumination (Neutral Hemisphere)"


    merged_params = {**params_n, **params_s}
    merged_params["hemisphere_deduction"] = hemisphere
    merged_params["north_sky_max"] = max_n
    merged_params["south_sky_max"] = max_s
    if params_n.get("driving_side"):
        merged_params["driving_side"] = params_n["driving_side"]
        side_clue = f"driving on the {params_n['driving_side']} side"
        if side_clue not in combined_clues:
            combined_clues.append(side_clue)

    return {
        "clues": combined_clues,
        "detected_parameters": merged_params,
        "is_panorama": True
    }

