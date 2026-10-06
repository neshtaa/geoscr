"""
Country Adjacency, Regional Clusters, and Knowledge Base Clue Overlap Matrix
Used for evaluating whether a GeoGuessr prediction is the exact country
or a legitimate regional neighbor sharing the same visual clues.
"""

NEIGHBORS = {
    # Americas - North
    "US": ["CA", "MX"],
    "CA": ["US"],
    "MX": ["US", "GT", "BZ"],
    "GT": ["MX", "BZ", "HN", "SV"],
    "PA": ["CR", "CO"],
    "CR": ["NI", "PA"],
    
    # Americas - South
    "BR": ["AR", "BO", "PY", "UY", "PE", "CO", "VE", "GY", "SR", "GF"],
    "AR": ["CL", "BO", "PY", "BR", "UY"],
    "CL": ["AR", "BO", "PE"],
    "PE": ["EC", "CO", "BR", "BO", "CL"],
    "BO": ["PE", "BR", "PY", "AR", "CL"],
    "EC": ["CO", "PE"],
    "CO": ["EC", "PE", "BR", "VE", "PA"],
    "UY": ["AR", "BR"],
    "PY": ["AR", "BO", "BR"],

    # Europe - Western & Central
    "FR": ["BE", "LU", "DE", "CH", "IT", "ES", "AD", "MC", "GB"],
    "DE": ["DK", "PL", "CZ", "AT", "CH", "FR", "LU", "BE", "NL"],
    "CH": ["FR", "DE", "AT", "IT", "LI"],
    "AT": ["DE", "CZ", "SK", "HU", "SI", "IT", "CH", "LI"],
    "IT": ["FR", "CH", "AT", "SI", "SM", "VA"],
    "BE": ["NL", "DE", "LU", "FR", "GB"],
    "NL": ["DE", "BE", "GB"],
    "GB": ["IE", "FR", "NL", "BE"],
    "IE": ["GB"],
    "ES": ["PT", "FR", "AD", "GI"],
    "PT": ["ES"],

    # Europe - Northern / Nordic / Baltic
    "SE": ["NO", "FI", "DK"],
    "NO": ["SE", "FI", "RU"],
    "FI": ["SE", "NO", "RU", "EE"],
    "DK": ["DE", "SE", "NO"],
    "EE": ["LV", "RU", "FI"],
    "LV": ["EE", "LT", "RU", "BY"],
    "LT": ["LV", "PL", "BY", "RU"],
    "IS": ["NO", "FO", "GB"],

    # Europe - Central / Eastern / Balkans
    "PL": ["DE", "CZ", "SK", "UA", "BY", "LT", "RU"],
    "CZ": ["DE", "PL", "SK", "AT"],
    "SK": ["CZ", "PL", "UA", "HU", "AT"],
    "HU": ["SK", "UA", "RO", "RS", "HR", "SI", "AT"],
    "RO": ["UA", "MD", "BG", "RS", "HU"],
    "BG": ["RO", "RS", "MK", "GR", "TR"],
    "GR": ["AL", "MK", "BG", "TR", "CY"],
    "TR": ["GR", "BG", "GE", "AM", "AZ", "IR", "IQ", "SY"],
    "UA": ["RU", "BY", "PL", "SK", "HU", "RO", "MD"],
    "HR": ["SI", "HU", "RS", "BA", "ME"],
    "SI": ["IT", "AT", "HU", "HR"],
    "RS": ["HU", "RO", "BG", "MK", "XK", "ME", "BA", "HR"],
    "MK": ["RS", "XK", "AL", "GR", "BG"],
    "AL": ["ME", "XK", "MK", "GR"],
    "ME": ["BA", "RS", "XK", "AL", "HR"],

    # Africa - Southern
    "ZA": ["NA", "BW", "ZW", "MZ", "SZ", "LS"],
    "BW": ["NA", "ZA", "ZW", "ZM"],
    "SZ": ["ZA", "MZ"],
    "LS": ["ZA"],
    "NA": ["AO", "ZM", "BW", "ZA"],

    # Africa - Eastern & Western
    "KE": ["ET", "SO", "SS", "UG", "TZ"],
    "UG": ["SS", "KE", "TZ", "RW", "CD"],
    "SN": ["MR", "ML", "GN", "GW", "GM"],
    "GH": ["CI", "BF", "TG"],
    "NG": ["BJ", "NE", "TD", "CM"],

    # Asia - East & Southeast
    "JP": ["KR", "TW"],
    "KR": ["KP", "JP", "CN"],
    "TW": ["JP", "PH"],
    "TH": ["MM", "LA", "KH", "MY"],
    "MY": ["TH", "ID", "BN", "SG"],
    "ID": ["MY", "TL", "PG", "SG"],
    "PH": ["TW", "MY", "ID"],
    "KH": ["TH", "LA", "VN"],
    "LA": ["MM", "CN", "VN", "KH", "TH"],
    "VN": ["CN", "LA", "KH"],
    "BT": ["IN", "CN", "NP"],
    "IN": ["PK", "CN", "NP", "BT", "BD", "MM", "LK"],
    "BD": ["IN", "MM"],
    "LK": ["IN"],

    # Eurasia / Post-Soviet / Central Asia
    "RU": ["MN", "KZ", "UA", "BY", "FI", "NO", "EE", "LV", "LT", "PL", "GE", "AZ", "CN", "KP"],
    "MN": ["RU", "CN", "KZ"],
    "KZ": ["RU", "KG", "UZ", "TM", "CN", "MN"],
    "KG": ["KZ", "UZ", "TJ", "CN"],

    # Middle East & North Africa
    "AE": ["OM", "SA", "QA"],
    "QA": ["SA", "AE"],
    "OM": ["AE", "SA", "YE"],
    "SA": ["JO", "IQ", "KW", "QA", "AE", "OM", "YE"],
    "JO": ["IL", "SA", "SY", "IQ", "PS"],
    "IL": ["JO", "EG", "LB", "SY", "PS"],
    "EG": ["IL", "LY", "SD"],

    # Oceania
    "AU": ["NZ"],
    "NZ": ["AU"]
}

REGIONAL_CLUSTERS = {
    "southern_africa": {"ZA", "BW", "SZ", "LS", "NA"},
    "east_africa": {"KE", "UG", "RW", "TZ"},
    "west_africa": {"SN", "GH", "NG"},
    "southern_cone": {"AR", "CL", "UY", "PY", "BR"},
    "andean": {"PE", "BO", "EC", "CO", "CL"},
    "north_america": {"US", "CA", "MX"},
    "central_america": {"GT", "CR", "PA", "MX"},
    "middle_east": {"AE", "QA", "OM", "SA", "KW", "BH", "JO", "IL", "LB"},
    "arabian_peninsula": {"AE", "QA", "OM", "SA", "KW", "BH", "YE"},
    "nordic": {"SE", "NO", "FI", "DK", "IS"},
    "baltic": {"EE", "LV", "LT"},
    "baltic_sea": {"EE", "LV", "LT", "PL", "SE", "FI", "DE", "RU"},
    "northern_europe": {"SE", "NO", "FI", "DK", "IS", "EE", "LV", "LT", "GB", "IE"},
    "western_europe": {"FR", "DE", "BE", "NL", "LU", "CH", "AT", "GB", "IE"},
    "mediterranean": {"ES", "PT", "IT", "GR", "TR", "HR", "SI", "AL", "MK", "ME", "CY", "MT"},
    "eastern_europe": {"PL", "CZ", "SK", "HU", "RO", "BG", "UA", "RU"},
    "central_europe": {"CZ", "SK", "PL", "AT", "HU", "DE", "SI", "CH", "RO", "UA"},
    "post_soviet": {"RU", "UA", "BY", "KZ", "KG", "MN"},
    "eurasian_steppe": {"MN", "KZ", "KG", "RU"},
    "southeast_asia": {"TH", "MY", "ID", "PH", "KH", "LA", "VN", "SG"},
    "east_asia": {"JP", "KR", "TW"},
    "south_asia": {"IN", "BD", "LK", "BT"},
    "oceania": {"AU", "NZ"}
}

def is_correct_or_neighbor(true_code, pred_code):
    """
    Returns True if pred_code is the exact true_code,
    a direct geographical neighbor, or belongs to the same regional clue cluster.
    """
    if not true_code or not pred_code:
        return False
    true_c = true_code.upper().strip()
    pred_c = pred_code.upper().strip()
    
    if true_c == pred_c:
        return True, "EXACT"

    # Check direct adjacency
    if pred_c in NEIGHBORS.get(true_c, []):
        return True, f"NEIGHBOR (Borders {true_c})"

    # Check shared regional cluster
    for cluster_name, cluster_set in REGIONAL_CLUSTERS.items():
        if true_c in cluster_set and pred_c in cluster_set:
            return True, f"REGIONAL CLUSTER ({cluster_name})"

    return False, "DIFFERENT REGION"
