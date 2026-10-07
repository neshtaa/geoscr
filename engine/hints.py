"""
GeoGuessr-style hints: what was measured in the image + the matching clue cards from the
two scraped knowledge bases (GeoGuessr's own clue catalogue and Plonk It).

No language model is involved.  Measured feature values are turned into observation tags
by fixed thresholds; every tag carries English keywords, and the clue texts of the top
candidate countries are ranked by keyword overlap (tag strength x keyword hits).
"""

import json
import os
import re

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# id: (Ukrainian description, keywords searched in clue texts)
TAGS = {
    "drive_left": ("Рух лівосторонній", ["left side", "drive on the left", "left-hand", "drives on the left", "driving side"]),
    "drive_right": ("Рух правосторонній", ["right side", "drive on the right", "right-hand", "driving side"]),
    "yellow_center": ("Жовта центральна лінія", ["yellow centre", "yellow center", "yellow middle", "yellow line", "yellow lines", "yellow road"]),
    "white_center": ("Біла центральна лінія", ["white centre", "white center", "white middle", "white line", "white lines"]),
    "double_center": ("Подвійна центральна лінія", ["double", "two lines", "double yellow", "double white"]),
    "dashed_center": ("Переривчаста центральна лінія", ["dashed", "broken line", "dashes"]),
    "yellow_edge": ("Жовті крайові лінії", ["yellow outer", "yellow edge", "yellow outside", "yellow side line"]),
    "white_edge": ("Білі крайові лінії", ["white outer", "white edge", "white outside", "white side line"]),
    "no_markings": ("Розмітки немає", ["no road lines", "no lines", "unmarked", "without lines", "no markings"]),
    "unpaved": ("Ґрунтова дорога", ["dirt road", "unpaved", "gravel", "dirt", "unsealed"]),
    "red_soil": ("Червоний ґрунт", ["red soil", "red dirt", "red earth", "reddish", "laterite", "red sand"]),
    "sand": ("Пісок / пустельний ландшафт", ["desert", "sand", "arid", "dry landscape"]),
    "dry": ("Суха трава, посушливий клімат", ["dry", "arid", "savanna", "savannah", "steppe", "yellow grass"]),
    "lush": ("Густа зелена рослинність", ["lush", "tropical", "jungle", "dense vegetation", "rainforest", "green"]),
    "conifer": ("Темні хвойні ліси", ["coniferous", "pine", "spruce", "fir", "boreal", "taiga"]),
    "snow": ("Сніг", ["snow", "winter", "snowy"]),
    "mountains": ("Гори / рельєф на горизонті", ["mountain", "mountains", "hilly", "hills", "alps", "relief"]),
    "flat": ("Рівнина, відкритий горизонт", ["flat", "plains", "open landscape", "prairie"]),
    "water": ("Вода поруч (море / озеро)", ["sea", "coast", "lake", "ocean", "beach", "coastal"]),
    "urban": ("Міська забудова", ["city", "urban", "buildings", "apartment"]),
    "rural": ("Сільська місцевість, мало будинків", ["rural", "countryside", "village", "farmland"]),
    "terracotta": ("Теракотові дахи", ["terracotta", "red roof", "orange roof", "tiled roof", "clay tile"]),
    "brick": ("Цегляні будинки", ["brick"]),
    "white_walls": ("Білі стіни будинків", ["white house", "white walls", "whitewashed", "white buildings"]),
    "poles": ("Стовпи ЛЕП уздовж дороги", ["pole", "poles", "utility pole", "electricity pole"]),
    "wooden_poles": ("Темні (дерев'яні?) стовпи", ["wooden pole", "wooden poles", "wood pole"]),
    "concrete_poles": ("Світлі (бетонні?) стовпи", ["concrete pole", "concrete poles", "cement"]),
    "wires": ("Багато повітряних дротів", ["wires", "power lines", "cables", "overhead"]),
    "curb": ("Бордюри / тротуари", ["curb", "kerb", "sidewalk", "pavement"]),
    "sun_north": ("Сонце на півночі -> ймовірно Південна півкуля", ["southern hemisphere", "sun in the north"]),
    "sun_south": ("Сонце на півдні -> ймовірно Північна півкуля", ["northern hemisphere", "sun in the south"]),
    "car_white": ("Світлий/білий Google-автомобіль у надирі", ["white car", "white google car", "white vehicle"]),
    "car_black": ("Темний Google-автомобіль у надирі", ["black car", "black google car", "dark car"]),
    "car_blur": ("Розмита \"пляма\" замість авто (сучасна генерація камери)", ["blur", "blurred", "car is blurred", "gen 4", "generation 4"]),
    "gen3_mount": ("Видно кріплення камери (Gen 3)", ["generation 3", "gen 3", "roof rack", "rack"]),
}


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _f(F, k):
    v = F.get(k)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def observations(F):
    """Feature dict {'module.feature': value} -> list of (tag, strength 0..1)."""
    out = []

    def add(tag, s):
        if s is not None and s > 0.15:
            out.append((tag, round(min(1.0, s), 2)))

    rht = _f(F, "road.right_hand_traffic")
    if rht is not None:
        add("drive_right" if rht >= 0.5 else "drive_left", abs(rht - 0.5) * 2)
    for k in ("yellow_center", "white_center", "double_center", "yellow_edge", "white_edge"):
        v = _f(F, "road." + k)
        if v is not None and v > 0.5:
            add(k, v)
    if (_f(F, "road.dashed_center") or 0) > 0.6 and max(_f(F, "road.yellow_center") or 0, _f(F, "road.white_center") or 0) > 0.5:
        add("dashed_center", _f(F, "road.dashed_center"))
    paved, mark = _f(F, "road.paved"), _f(F, "road.markings")
    if mark is not None and mark < 0.2 and (paved or 0) > 0.6:
        add("no_markings", 1 - mark)
    if paved is not None and paved < 0.4:
        add("unpaved", 1 - paved)
    red = _f(F, "landscape.soil_red")
    if red is not None and red > 0.12:
        add("red_soil", red * 3)
    sand = _f(F, "landscape.soil_sand")
    if sand is not None and sand > 0.2:
        add("sand", sand * 2)
    dry = _f(F, "landscape.veg_horizon_dry")
    if dry is not None and dry > 0.15:
        add("dry", dry * 2.5)
    green = _f(F, "landscape.veg_canopy_green")
    if green is not None and green > 0.45:
        add("lush", green)
    dark = _f(F, "landscape.veg_canopy_dark")
    if dark is not None and dark > 0.25:
        add("conifer", dark * 2)
    snow = max(_f(F, "landscape.snow_ground") or 0, _f(F, "landscape.snow_horizon") or 0)
    if snow > 0.08:
        add("snow", snow * 4)
    far = _f(F, "landscape.far_frac")
    if far is not None and far > 0.15:
        add("mountains", far * 2.5)
    op = _f(F, "landscape.sky_open")
    if op is not None and op > 0.5:
        add("flat", op)
    water = _f(F, "landscape.water_frac")
    if water is not None and water > 0.04:
        add("water", water * 8)
    built = _f(F, "structure.b_built_frac")
    if built is not None:
        if built > 0.25:
            add("urban", 0.5 + built)
        elif built < 0.03:
            add("rural", 0.7)
        if built > 0.05:
            for k, tag in (("p_terracotta", "terracotta"), ("p_brick", "brick"), ("p_white", "white_walls")):
                v = _f(F, "structure." + k)
                if v is not None and v > 0.2:
                    add(tag, v * 2)
    n = _f(F, "structure.pole_n360")
    if n is not None and n >= 2:
        add("poles", 0.3 + 0.1 * n)
        rel = _f(F, "structure.pole_rel_l")
        if rel is not None:
            if rel < 0.45:
                add("wooden_poles", 0.6)
            elif rel > 0.65:
                add("concrete_poles", 0.6)
    wc = _f(F, "structure.wire_cols")
    if wc is not None and wc > 0.35:
        add("wires", wc)
    curb = _f(F, "road.curb")
    if curb is not None and curb > 0.8:
        add("curb", curb * 0.6)
    cz, conf = _f(F, "solar.sun_az_cos"), _f(F, "solar.sun_conf")
    if cz is not None and conf is not None and conf > 0.5 and abs(cz) > 0.5:
        add("sun_north" if cz > 0 else "sun_south", conf * abs(cz))
    w, b = _f(F, "vehicle.nad_frac_white"), _f(F, "vehicle.nad_frac_black")
    if w is not None and w > 0.45:
        add("car_white", w)
    if b is not None and b > 0.45:
        add("car_black", b)
    sm = _f(F, "vehicle.nad_smooth_frac")
    if sm is not None and sm > 0.7:
        add("car_blur", sm * 0.8)
    ms = _f(F, "vehicle.mark_score_ri")
    if ms is not None and ms > 6:
        add("gen3_mount", min(1.0, ms / 12))
    out.sort(key=lambda t: -t[1])
    return out


class ClueBase:
    def __init__(self):
        self.gg = {}
        self.plonkit = {}
        self.rules = {}
        p = os.path.join(DATA, "geoguessr_master_clues.json")
        if os.path.exists(p):
            for cc, clues in _load(p).get("clues_by_country", {}).items():
                self.gg[cc.upper()] = [c for c in clues if (c.get("title") or c.get("description"))]
        p = os.path.join(DATA, "geoguessr_postmatch_clues.json")
        if os.path.exists(p):
            seen = {(c.get("id") or "").lower() for v in self.gg.values() for c in v}
            for c in _load(p).get("clues_by_id", {}).values():
                if (c.get("id") or "").lower() not in seen and c.get("countryCode"):
                    self.gg.setdefault(c["countryCode"].upper(), []).append(
                        {"id": c["id"], "title": c.get("title"), "description": c.get("description"),
                         "category": c.get("type") or c.get("category"), "image_url": c.get("image_url")})
        p = os.path.join(DATA, "plonkit_kb.json")
        if os.path.exists(p):
            self.plonkit = _load(p)
        p = os.path.join(DATA, "country_rules.json")
        if os.path.exists(p):
            self.rules = _load(p).get("by_country", {})

    def country_name(self, cc):
        for src in (self.plonkit, self.rules):
            if cc in src and src[cc].get("title"):
                return src[cc]["title"]
        return cc

    @staticmethod
    def _score(text, tags):
        t = " " + re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()) + " "
        s, why = 0.0, []
        for tag, strength in tags:
            hits = sum(1 for kw in TAGS[tag][1] if " " + kw + " " in t)
            if hits:
                s += strength * (1.0 + 0.3 * (hits - 1))
                why.append(tag)
        return s, why

    def for_country(self, cc, tags, n_gg=3, n_plonkit=2):
        cc = cc.upper()
        res = {"country_code": cc, "country": self.country_name(cc), "geoguessr": [], "plonkit": []}
        side = (self.rules.get(cc) or {}).get("driving_side")
        res["driving_side"] = side
        ranked = []
        for i, c in enumerate(self.gg.get(cc, [])):
            s, why = self._score((c.get("title") or "") + " " + (c.get("description") or ""), tags)
            ranked.append((s, -i, c, why))
        ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
        for s, _, c, why in ranked[:n_gg]:
            res["geoguessr"].append({"title": c.get("title"), "text": (c.get("description") or "").strip(),
                                     "category": c.get("category"), "image_url": c.get("image_url"),
                                     "matched": [TAGS[t][0] for t in why]})
        tips = (self.plonkit.get(cc) or {}).get("tips") or []
        ranked = []
        for i, t in enumerate(tips):
            s, why = self._score(t.get("text"), tags)
            ranked.append((s, -i, t, why))
        ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
        for s, _, t, why in ranked[:n_plonkit]:
            res["plonkit"].append({"text": re.sub(r"\*\*", "", (t.get("text") or "")).strip()[:400],
                                   "section": t.get("section"), "image_url": t.get("image_url"),
                                   "matched": [TAGS[w][0] for w in why]})
        return res


_BASE = None


def clue_base():
    global _BASE
    if _BASE is None:
        _BASE = ClueBase()
    return _BASE


def build_hints(F, top_countries, n_countries=3):
    """F: {'module.feature': value}; top_countries: [(code, prob), ...]."""
    tags = observations(F)
    kb = clue_base()
    obs = [{"tag": t, "text": TAGS[t][0], "strength": s} for t, s in tags]
    side_tag = next((t for t, _ in tags if t in ("drive_left", "drive_right")), None)
    cards = []
    for cc, p in top_countries[:n_countries]:
        c = kb.for_country(cc, tags)
        c["probability"] = round(float(p), 3)
        if side_tag and c["driving_side"]:
            c["driving_side_consistent"] = (c["driving_side"] == side_tag.split("_")[1])
        cards.append(c)
    return {"observations": obs, "countries": cards}
