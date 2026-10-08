"""
GeoGuessr-style hints: what was measured in the image + the matching clue cards from the
two scraped knowledge bases (GeoGuessr's own clue catalogue and Plonk It).

No language model is involved.  Measured feature values are turned into observation tags
by fixed thresholds; every tag carries English keywords.  The GeoGuessr cards of a candidate
country are ranked by a calibrated sum of: keyword overlap with the observations (tag strength
x keyword hits), the posterior of the card's admin-1 regions, a kernel-weighted vote of the most
similar panoramas that GeoGuessr itself annotated with clue placements (data/model/clue_index.npz,
tools/build_clue_index.py), and the card's frequency among the country's annotated panoramas,
overall and inside the likely regions.  Without the index, and for countries or cards without
annotated panoramas, keywords + regions decide.
"""

import json
import math
import os
import re
from collections import Counter

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# id: (Ukrainian description, keywords searched in clue texts)
TAGS = {
    "drive_left": ("Рух лівосторонній", ["left side", "drive on the left", "left-hand", "drives on the left", "driving side"]),
    "drive_right": ("Рух правосторонній", ["right side", "drive on the right", "right-hand", "driving side"]),
    "yellow_center": ("Жовта центральна лінія", ["yellow centre", "yellow center", "yellow middle", "yellow line", "yellow lines", "yellow road"]),
    "white_center": ("Біла центральна лінія", ["white centre", "white center", "white middle", "white line", "white lines"]),
    "double_center": ("Подвійна центральна лінія", ["double", "two lines", "double yellow", "double white"]),
    "dashed_center": ("Переривчаста центральна лінія", ["dashed centre", "dashed center", "dashed middle", "dashed yellow", "dashed white", "broken centre", "broken center"]),
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
    "rural": ("Майже немає будівель", ["rural", "countryside", "village", "farmland"]),
    "terracotta": ("Теракотові дахи", ["terracotta", "red roof", "orange roof", "tiled roof", "clay tile"]),
    "brick": ("Цегляні будинки", ["brick"]),
    "white_walls": ("Білі стіни будинків", ["white house", "white walls", "whitewashed", "white buildings"]),
    "poles": ("Стовпи ЛЕП уздовж дороги", ["pole", "poles", "utility pole", "electricity pole"]),
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
    if dry is not None and dry > 0.35:
        add("dry", dry * 1.6)
    green = _f(F, "landscape.veg_canopy_green")
    if green is not None and green > 0.45:
        add("lush", green)
    dark = _f(F, "landscape.veg_canopy_dark")
    if dark is not None and dark > 0.2:
        add("conifer", dark * 2.5)
    snow = max(_f(F, "landscape.snow_ground") or 0, _f(F, "landscape.snow_horizon") or 0)
    if snow > 0.15:
        add("snow", snow * 3)
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
        if built > 0.08:
            add("urban", 0.5 + 2 * built)
        elif built < 0.002:
            add("rural", 0.6)
        if built > 0.03:
            for k, tag, thr in (("p_terracotta", "terracotta", 0.25), ("p_brick", "brick", 0.3), ("p_white", "white_walls", 0.5)):
                v = _f(F, "structure." + k)
                if v is not None and v > thr:
                    add(tag, v * 1.5)
    n = _f(F, "structure.pole_n360")
    if n is not None and n >= 2:
        add("poles", 0.3 + 0.1 * n)
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


# GeoGuessr areas that group several admin-1 units of the Natural Earth raster
AREA_ALIASES = {
    "AREA_AUSTRIA_KARNTEN": ["AT-2"], "AREA_AUSTRIA_STEIERMARK": ["AT-6"], "AREA_AUSTRIA_TIROL": ["AT-7"],
    "AREA_GERMANY_BAYERN": ["DE-BY"], "AREA_GERMANY_NIEDERSACHSEN": ["DE-NI"],
    "AREA_GERMANY_NORDRHEINWESTFALEN": ["DE-NW"],
    "AREA_ANDALUSIA": ["ES-AL", "ES-CA", "ES-CO", "ES-GR", "ES-H", "ES-J", "ES-MA", "ES-SE"],
    "AREA_ARAGON": ["ES-HU", "ES-TE", "ES-Z"], "AREA_BASQUECOUNTRY": ["ES-BI", "ES-SS", "ES-VI"],
    "AREA_CANARIANISLANDS": ["ES-GC", "ES-TF"], "AREA_CASTILELAMANCHA": ["ES-AB", "ES-CR", "ES-CU", "ES-GU", "ES-TO"],
    "AREA_CATALONIA": ["ES-B", "ES-GI", "ES-L", "ES-T"], "AREA_EXTREMADURA": ["ES-BA", "ES-CC"],
    "AREA_GALICIA": ["ES-C", "ES-LU", "ES-OR", "ES-PO"],
    "AREA_BRETAGNE": ["FR-22", "FR-29", "FR-35", "FR-56"], "AREA_NORMANDIE": ["FR-14", "FR-27", "FR-50", "FR-61", "FR-76"],
    "AREA_PROVENCEALPESCOTEDAZUR": ["FR-04", "FR-05", "FR-06", "FR-13", "FR-83", "FR-84"],
    "AREA_LANGUEDOCROUSSILLONMIDIPYRENEES": ["FR-09", "FR-11", "FR-12", "FR-30", "FR-31", "FR-32", "FR-34", "FR-46",
                                             "FR-48", "FR-65", "FR-66", "FR-81", "FR-82"],
    "AREA_JAPAN_TOHOKU": ["JP-02", "JP-03", "JP-04", "JP-05", "JP-06", "JP-07"],
    "AREA_JAPAN_KANTO": ["JP-08", "JP-09", "JP-10", "JP-11", "JP-12", "JP-13", "JP-14"],
    "AREA_JAPAN_CHUBU": ["JP-15", "JP-16", "JP-17", "JP-18", "JP-19", "JP-20", "JP-21", "JP-22", "JP-23"],
    # Norway's 2020 counties over the pre-2020 ones of the raster
    "AREA_NORWAY_VIKEN": ["NO-01", "NO-02", "NO-06"], "AREA_NORWAY_INNLANDET": ["NO-04", "NO-05"],
    "AREA_NORWAY_TRONDELAG": ["NO-16", "NO-17"],
    "AREA_SWITZERLAND_GRAUBUNDEN": ["CH-GR"], "AREA_NETHERLANDS_NOORDHOLLAND": ["NL-NH"],
    "AREA_NETHERLANDS_ZUIDHOLLAND": ["NL-ZH"], "AREA_ROMANIA_VILCEA": ["RO-VL"],
    "AREA_UK_SCOTLAND": ["GB-" + c for c in ("ABE ABD ANS AGB CLK DGY DND EAY EDU ELN ERW EDH ELS FAL FIF GLG HLD IVC "
                                             "MLN MRY NAY NLK ORK PKN RFW SCB ZET SAY SLK STG WDU WLN").split()],
    "AREA_SCOTLAND_NAHEILEANANSIAR": ["GB-ELS"],
    "AREA_AUSTRIA_NIEDEROSTERREICH": ["AT-3"], "AREA_AUSTRIA_OBEROSTERREICH": ["AT-4"], "AREA_AUSTRIA_WIEN": ["AT-9"],
    "AREA_CZECH_KRALOVEHRADECKY": ["CZ-KR"], "AREA_CZECH_USTECKY": ["CZ-US"],
    "AREA_GERMANY_MECKLENBURGVORPOMMERN": ["DE-MV"], "AREA_INDIA_ORISSA": ["IN-OR"],
    "AREA_JAPAN_KYUSHU": ["JP-40", "JP-41", "JP-42", "JP-43", "JP-44", "JP-45", "JP-46"],
    # Slovak historical regions -> the self-governing regions that contain them
    "AREA_SLOVAKIA_ABOV": ["SK-KI"], "AREA_SLOVAKIA_DOLNY_ZEMPLIN": ["SK-KI"], "AREA_SLOVAKIA_HORNY_ZEMPLIN": ["SK-PV"],
    "AREA_NORWAY_VESTLAND": ["NO-12", "NO-14"], "AREA_NORWAY_AGDER": ["NO-09", "NO-10"],
    "AREA_NORWAY_VESTFOLDOGTELEMARK": ["NO-07", "NO-08"], "AREA_NORWAY_TROMSOGFINNMARK": ["NO-19", "NO-20"],
    "AREA_GERMANY_SACHSEN": ["DE-SN"], "AREA_GERMANY_SACHSENANHALT": ["DE-ST"], "AREA_GERMANY_THURINGEN": ["DE-TH"],
    "AREA_GERMANY_RHEINLANDPFALZ": ["DE-RP"], "AREA_SWITZERLAND_SANKTGALLEN": ["CH-SG"],
    "AREA_MEXICO_DISTRITOFEDERAL": ["MX-DIF"], "CITY_BRUSSELS": ["BE-BRU"], "CITY_VIENNA": ["AT-9"],
    "CITY_INNSBRUCK": ["AT-7"],
    "AREA_JAPAN_HOKKAIDO": ["JP-01"], "AREA_JAPAN_KINKI": ["JP-24", "JP-25", "JP-26", "JP-27", "JP-28", "JP-29", "JP-30"],
    "AREA_JAPAN_CHUGOKU": ["JP-31", "JP-32", "JP-33", "JP-34", "JP-35"],
    "AREA_JAPAN_SHIKOKU": ["JP-36", "JP-37", "JP-38", "JP-39"], "AREA_JAPAN_OKINAWA": ["JP-47"],
    "AREA_UK_WALES": ["GB-" + c for c in ("AGY BGW BGE CAY CRF CMN CGN CWY DEN FLN GWN MTY MON NTL NWP PEM POW RCT SWA "
                                          "TOF VGL WRX").split()],
    "AREA_UK_NORTHERNIRELAND": ["GB-" + c for c in ("ANT ARD ARM BLA BLY BNB BFS CKF CSR CLR CKT CGV DRY DOW DGN FER "
                                                    "LRN LMV LSB MFT MYL NYM NTA NDN OMH STB").split()],
    # Spanish autonomous communities whose name is also one of their provinces
    "AREA_CASTILELEON": ["ES-AV", "ES-BU", "ES-LE", "ES-P", "ES-SA", "ES-SG", "ES-SO", "ES-VA", "ES-ZA"],
    "AREA_VALENCIA": ["ES-A", "ES-CS", "ES-V"],
    # seterra keeps the pre-2002 name 'Northern Province' of Limpopo (Northern Cape is AREA_SAFRICA_NORTHERNCAPE)
    "AREA_SAFRICA_NORTHERN": ["ZA-LP"], "AREA_INDIA_JAMMUKASHMIR": ["IN-JK"],
}
# France: the 2016 regions (seterra ids keep the pre-2016 names) -> departments of the raster
for _reg, _dep in (("ALSACECHAMPAGNEARDENNELORRAINE", "08 10 51 52 54 55 57 67 68 88"),
                   ("AQUITAINELIMOUSINPOITOUCHARENTES", "16 17 19 23 24 33 40 47 64 79 86 87"),
                   ("AUVERGNERHONEALPES", "01 03 07 15 26 38 42 43 63 69 73 74"),
                   ("BOURGOGNEFRANCHECOMTE", "21 25 39 58 70 71 89 90"), ("CENTREVALDELOIRE", "18 28 36 37 41 45"),
                   ("CORSE", "2A 2B"), ("ILEDEFRANCE", "75 77 78 91 92 93 94 95"),
                   ("NORDPASDECALAISPICARDIE", "02 59 60 62 80"), ("PAYSDELALOIRE", "44 49 53 72 85")):
    AREA_ALIASES["AREA_" + _reg] = ["FR-" + c for c in _dep.split()]
# Poland: voivodeships (seterra spells Warmińsko-Mazurskie 'WARMINSKOMARZURSKIE')
for _reg, _code in (("DOLNOSLASKIE", "DS"), ("KUJAWSKOPOMORSKIE", "KP"), ("LODZKIE", "LD"), ("LUBELSKIE", "LU"),
                    ("LUBUSKIE", "LB"), ("MALOPOLSKIE", "MA"), ("MAZOWIECKIE", "MZ"), ("OPOLSKIE", "OP"),
                    ("PODKARPACKIE", "PK"), ("PODLASKIE", "PD"), ("POMORSKIE", "PM"), ("SLASKIE", "SL"),
                    ("SWIETOKRZYSKIE", "SK"), ("WARMINSKOMARZURSKIE", "WN"), ("WARMINSKOMAZURSKIE", "WN"),
                    ("WIELKOPOLSKIE", "WP"), ("ZACHODNIOPOMORSKIE", "ZP")):
    AREA_ALIASES["AREA_POLAND2_" + _reg] = ["PL-" + _code]
# Italian regions -> the provinces of the raster
for _reg, _prov in (("ABRUZZO", "AQ CH PE TE"), ("BASILICATA", "MT PZ"), ("CALABRIA", "CS CZ KR RC VV"),
                    ("CAMPANIA", "AV BN CE NA SA"), ("EMILIAROMAGNA", "BO FC FE MO PC PR RA RE RN"),
                    ("FRIULIVENEZIAGIULIA", "GO PN TS UD"), ("LAZIO", "FR LT RI RM VT"), ("LIGURIA", "GE IM SP SV"),
                    ("LOMBARDIA", "BG BS CO CR LC LO MB MI MN PV SO VA"), ("MARCHE", "AN AP FM MC PU"),
                    ("MOLISE", "CB IS"), ("PIEMONTE", "AL AT BI CN NO TO VB VC"), ("PUGLIA", "BA BR BT FG LE TA"),
                    ("SARDEGNA", "CA CI NU OG OR OT SS VS"), ("SICILIA", "AG CL CT EN ME PA RG SR TP"),
                    ("TOSCANA", "AR FI GR LI LU MS PI PO PT SI"), ("TRENTINOALTOADIGE", "BZ TN"), ("UMBRIA", "PG TR"),
                    ("VALLEDAOSTA", "AO"), ("VENETO", "BL PD RO TV VE VI VR")):
    AREA_ALIASES["AREA_ITALY_" + _reg] = ["IT-" + c for c in _prov.split()]
# ISO 3166-2 units newer than the raster -> the units they were split from
ISO_ALIASES = {"ID-KU": ["ID-KI"], "PH-DVO": ["PH-DAS"], "OM-BS": ["OM-BA"], "OM-SJ": ["OM-SH"],
               # Ghana's 2018 regions -> the 10 former ones
               "GH-AF": ["GH-BA"], "GH-BO": ["GH-BA"], "GH-BE": ["GH-BA"], "GH-NE": ["GH-NP"], "GH-SV": ["GH-NP"],
               "GH-OT": ["GH-TV"], "GH-WN": ["GH-WP"]}
# Nepal: the 7 provinces of 2015 -> the 14 former zones (approximate: some zones are split)
for _prov, _zones in (("P1", "ME KO SA"), ("P2", "SA JA NA"), ("P3", "BA JA NA"), ("P4", "GA DH"), ("P5", "LU RA BH"),
                      ("P6", "KA BH"), ("P7", "SE MA")):
    ISO_ALIASES["NP-" + _prov] = ["NP-" + z for z in _zones.split()]
# Kenya: 47 counties (ISO 3166-2:KE-01..47) -> the 8 former provinces of the raster
for _prov, _counties in (("KE-200", "13 15 29 35 36"), ("KE-300", "14 19 21 28 39 40"),
                         ("KE-400", "06 09 18 22 23 25 26 41"), ("KE-500", "07 24 46"), ("KE-600", "08 16 17 27 34 38"),
                         ("KE-700", "01 02 05 10 12 20 31 32 33 37 42 43 44 47"), ("KE-800", "03 04 11 45"),
                         ("KE-110", "30")):
    ISO_ALIASES.update({"KE-" + c: [_prov] for c in _counties.split()})

_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss", "đ": "d", "Đ": "D", "ł": "l",
                           "Ł": "L", "þ": "th", "Þ": "TH", "ð": "d", "Ð": "D", "ı": "i", "œ": "oe", "Œ": "OE"})

IMAGE_BASE = "https://www.geoguessr.com/images/resize:fit:600:600/plain/"
INDEX_PATH = os.path.join(DATA, "model", "clue_index.npz")
# ranking weights without the clue index: keyword overlap + regional bonus
DEFAULT_RANK = {"kw": 1.0, "region": 3.0, "knn": 0.0, "freq": 0.0, "rvote": 0.0, "k": 15, "bw": 1.0, "beta": 2.0,
                "beta_r": 2.0}
MIN_SUPPORT = 5  # labelled panoramas of a country before a card is called frequent / seen on similar ones

# what to look at for each GeoGuessr clue type
LOOK_OBJECTS = {
    "pole": "стовпи", "bollards": "придорожні стовпчики", "road-signs": "дорожні знаки",
    "license-plates": "номерні знаки авто", "language": "написи й вивіски", "architecture": "будинки",
    "nature": "рослинність і рельєф", "road": "дорогу й розмітку", "flags": "прапори", "vehicles": "автомобілі",
}


def clue_stem(key):
    """Translation key 'clue.br-ladder-poles-title' -> clue identity 'br-ladder-poles'."""
    k = re.sub(r"^clue\.", "", (key or "").strip().lower())
    return re.sub(r"-(title|description|desc)$", "", k)


def zoom_fov(zoom):
    """GeoGuessr's Street View zoom -> horizontal field of view in degrees."""
    return math.degrees(2.0 * math.atan(2.0 ** (1.0 - float(zoom))))


def look_text(ctype, pitch, zoom):
    """Typical camera direction of a clue (median placement pitch/zoom) as a short instruction."""
    obj = LOOK_OBJECTS.get(ctype)
    if obj is None or pitch is None or zoom is None:
        return None
    where = "вниз " if pitch <= -10 else "трохи вниз " if pitch <= -5 else "вгору " if pitch >= 10 else ""
    if zoom >= 2.75:
        return "Наблизьте камеру й подивіться %sна %s" % (where, obj)
    if not where and zoom < 1.75:
        return "Загальний план: %s" % obj
    return "Подивіться %sна %s" % (where, obj)


class ClueIndex:
    """GeoGuessr's own clue placements on labelled panoramas (tools/build_clue_index.py): per
    panorama its country, admin-1 region, clue cards and (when features were cached) a robust-
    standardised feature embedding; plus the card catalogue and the calibrated ranking weights."""

    def __init__(self, z):
        meta = json.loads(str(z["meta"]))
        self.params = dict(DEFAULT_RANK, **meta.get("params", {}))
        self.cards = meta["cards"]
        self.info = {k: v for k, v in meta.items() if k not in ("cards", "params")}
        self.feat_names = [str(s) for s in z["feat_names"]]
        self.med, self.sc = z["med"].astype(np.float64), z["sc"].astype(np.float64)
        self.proj = z["proj"].astype(np.float64)
        self.emb = z["emb"].astype(np.float64)
        self.has_emb = ~np.isnan(self.emb).any(1)
        self.country = [str(c) for c in z["country"]]
        self.region = [str(c) for c in z["region"]] if "region" in z else [""] * len(self.country)
        self.pano_ids = [str(p) for p in z["pano_ids"]]
        ptr, idx = z["card_ptr"], z["card_idx"]
        self.row_cards = [idx[ptr[i]:ptr[i + 1]].tolist() for i in range(len(self.country))]
        self.row_of = {p: i for i, p in enumerate(self.pano_ids)}
        self.rows = {}
        for i, cc in enumerate(self.country):
            self.rows.setdefault(cc, []).append(i)
        self.counts = {cc: Counter(c for i in rows for c in self.row_cards[i]) for cc, rows in self.rows.items()}

    @classmethod
    def load(cls, path=INDEX_PATH):
        if not path or not os.path.exists(path):
            return None
        try:
            with np.load(path) as z:
                return cls(z)
        except Exception:  # unreadable or outdated index: hints fall back to keywords + regions
            return None

    def embed(self, F):
        x = np.array([_f(F, n) for n in self.feat_names], np.float64)  # None -> NaN
        if np.isnan(x).mean() > 0.5:
            return None
        z = np.clip((x - self.med) / self.sc, -6.0, 6.0)
        return np.where(np.isnan(z), 0.0, z) @ self.proj

    def card_scores(self, cc, emb=None, region_probs=None, exclude=None, params=None):
        """Card scores inside country cc, {"freq"|"knn"|"rvote": {card id: score}}:
        freq  - share of the country's labelled panoramas that carry the card;
        knn   - kernel-weighted vote of the k panoramas nearest to emb, shrunk towards freq;
        rvote - sum_r P(region r) * P(card | r), P(card | r) from the panoramas of region r shrunk
                towards freq (regions without labels contribute freq).
        exclude: a pano id left out (leave-one-out tuning)."""
        p = params or self.params
        rows = self.rows.get(cc, [])
        x = self.row_of.get(exclude, -1)
        cnt = self.counts.get(cc, Counter())
        if x in rows:
            rows = [i for i in rows if i != x]
            cnt = cnt - Counter(self.row_cards[x])
        out = {"freq": {}, "knn": {}, "rvote": {}}
        if not rows:
            return out
        ids = {c: self.cards[c]["id"] for c in cnt}
        freq = {c: k / float(len(rows)) for c, k in cnt.items()}
        out["freq"] = {ids[c]: f for c, f in freq.items()}
        out["knn"] = out["rvote"] = out["freq"]
        r = [i for i in rows if self.has_emb[i]]
        if emb is not None and r:
            d2 = ((self.emb[r] - emb) ** 2).sum(1)
            o = np.argsort(d2)[: int(p["k"])]
            h2 = max(float(np.median(d2[o])) * p["bw"], 1e-9)
            w = np.exp(-0.5 * (d2[o] - d2[o].min()) / h2)
            votes = Counter()
            for wj, j in zip(w, o):
                for c in self.row_cards[r[j]]:
                    votes[c] += wj
            tot = float(w.sum()) + p["beta"]
            out["knn"] = {ids[c]: (votes[c] + p["beta"] * f) / tot for c, f in freq.items()}
        if region_probs:
            by_reg = {}
            for i in rows:
                if self.region[i] in region_probs:
                    by_reg.setdefault(self.region[i], []).append(i)
            covered = sum(region_probs[g] for g in by_reg)
            rv = {c: (1.0 - covered) * f for c, f in freq.items()}
            for g, ri in by_reg.items():
                n_rc = Counter(c for i in ri for c in self.row_cards[i])
                for c, f in freq.items():
                    rv[c] += region_probs[g] * (n_rc[c] + p["beta_r"] * f) / (len(ri) + p["beta_r"])
            out["rvote"] = {ids[c]: v for c, v in rv.items()}
        return out

    def support(self, cc, region_probs=None, exclude=None):
        """(labelled panoramas of country cc, expected number of them inside the likely regions)."""
        rows = [i for i in self.rows.get(cc, []) if self.pano_ids[i] != exclude]
        n_reg = sum(region_probs.get(self.region[i], 0.0) for i in rows) if region_probs else 0.0
        return len(rows), n_reg


class ClueBase:
    def __init__(self, index_path=INDEX_PATH, index=None):
        self.gg = {}
        self._cards, self._by_text = {}, {}
        self.plonkit = {}
        self.rules = {}
        p = os.path.join(DATA, "plonkit_kb.json")  # country names first: _region_codes uses them
        if os.path.exists(p):
            self.plonkit = _load(p)
        p = os.path.join(DATA, "country_rules.json")
        if os.path.exists(p):
            self.rules = _load(p).get("by_country", {})
        p = os.path.join(DATA, "geoguessr_master_clues.json")
        if os.path.exists(p):
            for cc, clues in _load(p).get("clues_by_country", {}).items():
                for c in clues:
                    if c.get("title") or c.get("description"):
                        self._merge(cc.upper(), c["id"].lower(), c.get("title"), c.get("description"),
                                    {"category": c.get("category"), "image_url": c.get("image_url")})
        p = os.path.join(DATA, "geoguessr_postmatch_clues.json")
        if os.path.exists(p):
            for c in _load(p).get("clues_by_id", {}).values():
                if c.get("countryCode"):
                    self._merge(c["countryCode"].upper(), clue_stem(c.get("title_key")) or c["id"].lower(),
                                c.get("title"), c.get("description"),
                                {"gg_id": c["id"], "type": c.get("type"), "image_url": c.get("image_url"),
                                 "seterra": c.get("seterraRegionIds")})
        self.index = index if index is not None else ClueIndex.load(index_path)
        for c in (self.index.cards if self.index else []):
            self._merge(c["country"], c["id"], c.get("title"), c.get("description"),
                        {"gg_id": c.get("gg_id"), "type": c.get("type"), "seterra": c.get("seterra"),
                         "image_url": IMAGE_BASE + c["image"] if c.get("image") else None,
                         "pitch": c.get("pitch"), "zoom": c.get("zoom")})

    def _merge(self, cc, key, title, description, info):
        """Add a card or complete an existing one. Same clue = same translation key, or the same
        title and description inside a country (GeoGuessr keeps a few clues under two keys). The card
        id is the first key that carries a GeoGuessr clue id (placements use those keys); the other
        keys become 'aliases'."""
        card = self._cards.get((cc, key))
        if card is None:
            if not (title or description):
                return
            text = (cc, self._norm(title), self._norm(description))
            card = self._by_text.get(text) if len(text[2]) >= 20 else None
            if card is None:
                card = {"id": key, "title": title, "description": description, "category": info.get("category"),
                        "image_url": None, "regions": [], "aliases": []}
                self._by_text[text] = card
                self.gg.setdefault(cc, []).append(card)
            elif info.get("gg_id") and not card.get("gg_id"):
                card["aliases"].append(card["id"])
                card["id"] = key
            else:
                card["aliases"].append(key)
            self._cards[(cc, key)] = card
        if info.get("type") and card["category"] in (None, "general"):
            card["category"] = info["type"]
        if info.get("seterra") and not card["regions"]:
            card["regions"] = self._region_codes(cc, info["seterra"])
        card["image_url"] = card["image_url"] or info.get("image_url")
        if info.get("gg_id") and not card.get("gg_id"):
            card["gg_id"] = info["gg_id"]
        if info.get("pitch") is not None and info.get("zoom") is not None:
            card["view"] = {"pitch": info["pitch"], "zoom": info["zoom"]}

    def card_id(self, cc, key):
        """Id of the card shown for clue key in country cc (keys of duplicate clues map to one card)."""
        c = self._cards.get((cc, key))
        return c["id"] if c else key

    @staticmethod
    def _norm(t):
        import unicodedata
        t = unicodedata.normalize("NFKD", (t or "").translate(_TRANSLIT)).encode("ascii", "ignore").decode()
        return re.sub(r"[^A-Z]", "", t.upper())

    def _region_codes(self, cc, ids):
        """GeoGuessr seterra ids ('ISO-ID-JI', 'AREA_BRAZIL_PARANA') -> ISO 3166-2 codes of the raster.
        Whole-country areas ('AREA_ITALY', 'AREA_THEUNITEDSTATES') are not regions and give nothing."""
        return [c for i in ids for c in self.region_match(cc, i)[0]]

    def region_match(self, cc, i):
        """One seterra id -> (codes, how); how is 'alias', 'iso', 'exact', 'partial', 'country' or None."""
        if not hasattr(self, "_rnames"):
            self._rnames, self._rcodes = {}, set()
            try:
                from .geo import _regions
                _, codes, names, countries, _ = _regions()
                self._rcodes = set(codes)
                for code, name, c in zip(codes, names, countries):
                    self._rnames.setdefault(c, {})[self._norm(name)] = code
            except Exception:
                pass
        if i in AREA_ALIASES:
            return list(AREA_ALIASES[i]), "alias"
        if i.startswith("ISO-"):
            code = i[4:].upper()
            if code in ISO_ALIASES:
                return list(ISO_ALIASES[code]), "alias"
            if re.match(r"US-02\d{3}$|US-2\d{3}$", code):  # Alaska boroughs (FIPS 02xxx)
                return ["US-AK"], "alias"
            if code in self._rcodes or not self._rcodes:
                return [code], "iso"
            if code[-1:].isdigit():  # Sri Lanka 'LK-3' (province) -> its districts 'LK-31', 'LK-32', ...
                sub = sorted(c for c in self._rcodes if c.startswith(code) and len(c) == len(code) + 1)
                return sub, "iso" if sub else None
            return [], None
        if not i.startswith("AREA_"):
            return [], None
        parts = i.split("_", 2)
        key = self._norm(parts[-1])
        if len(parts) == 2:  # 'AREA_<COUNTRY>' is the whole country (seterra flag quiz)
            name = self._norm(self.country_name(cc))
            k = key[3:] if key.startswith("THE") else key
            if len(k) >= 4 and (name.startswith(k) or name.startswith(key)):
                return [], "country"
        names = self._rnames.get(cc, {})
        if key in names:
            return [names[key]], "exact"
        # partial name match: the longest region name inside the id ('BLEKINGELAN' -> 'BLEKINGE'),
        # else the only region name that contains the id ('MADRID' -> 'COMMUNITYOFMADRID')
        inner = sorted((k for k in names if len(k) >= 4 and k in key), key=len)
        if key and inner:
            return [names[inner[-1]]], "partial"
        outer = [k for k in names if key and key in k]
        if len(outer) == 1:
            return [names[outer[0]]], "partial"
        return [], None

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

    def rank_cards(self, cc, tags, region_probs=None, emb=None, exclude=None, params=None):
        """GeoGuessr cards of country cc, best first: [(score, card, reasons, seen)].
        score = weighted keyword overlap with the observations + regional bonus + kNN vote of similar
        labelled panoramas + frequency of the card in the country and in the likely regions (weights
        calibrated by tools/build_clue_index.py). Ties, e.g. all cards of a country without labelled
        panoramas, go by the default keyword + region score, then by catalogue order.
        reasons: what raised the card - observations / likely region worth >= 10% of its score; cards of
        similar panoramas or of the likely regions above the country rate; a frequent card of the country
        (votes need MIN_SUPPORT labelled panoramas). seen: observations / likely region that agree with
        the card but count for less."""
        p = params or (self.index.params if self.index else DEFAULT_RANK)
        sc = self.index.card_scores(cc, emb, region_probs, exclude, p) if self.index else {}
        n, n_reg = self.index.support(cc, region_probs, exclude) if self.index else (0, 0.0)
        knn, freq, rv = sc.get("knn", {}), sc.get("freq", {}), sc.get("rvote", {})
        ranked = []
        for i, c in enumerate(self.gg.get(cc, [])):
            kw, hit = self._score((c.get("title") or "") + " " + (c.get("description") or ""), tags)
            pr = sum(region_probs.get(r, 0.0) for r in c["regions"]) if c["regions"] and region_probs else 0.0
            ids = [c["id"]] + c["aliases"]
            k, f, v = (max(d.get(j, 0.0) for j in ids) for d in (knn, freq, rv))
            s = p["kw"] * kw + p["region"] * pr
            s = s + p["knn"] * k
            s = s + p["freq"] * f
            s = s + p["rvote"] * v
            s0 = DEFAULT_RANK["kw"] * kw + DEFAULT_RANK["region"] * pr
            # with few labelled panoramas the default keyword + region score orders most cards
            wk, wr, tot = (p["kw"], p["region"], s) if n >= MIN_SUPPORT else \
                (DEFAULT_RANK["kw"], DEFAULT_RANK["region"], s0)
            why, seen = [], []
            (why if wk * kw >= 0.1 * tot > 0 else seen).extend(hit)
            if pr > 0.15:
                (why if wr * pr >= 0.1 * tot > 0 else seen).append("_region")
            votes = []
            if p["knn"] > 0 and emb is not None and n >= MIN_SUPPORT and k >= 0.25 and k > 1.25 * f:
                votes.append("_similar")
            if p["rvote"] > 0 and n_reg >= 3 and v >= 0.25 and v > 1.25 * f:
                votes.append("_regional")
            if not votes and p["freq"] + p["rvote"] > 0 and n >= MIN_SUPPORT and max(f, v) >= 0.2:
                votes.append("_frequent")
            why += votes
            ranked.append((s, s0, -i, c, why, seen))
        ranked.sort(key=lambda r: r[:3], reverse=True)
        return [(s, c, why, seen) for s, _, _, c, why, seen in ranked]

    def for_country(self, cc, tags, n_gg=3, n_plonkit=2, region_probs=None, emb=None):
        cc = cc.upper()
        res = {"country_code": cc, "country": self.country_name(cc), "geoguessr": [], "plonkit": []}
        side = (self.rules.get(cc) or {}).get("driving_side")
        res["driving_side"] = side
        for s, c, why, seen in self.rank_cards(cc, tags, region_probs, emb)[:n_gg]:
            card = {"id": c["id"], "title": c.get("title"), "text": (c.get("description") or "").strip(),
                    "category": c.get("category"), "image_url": c.get("image_url"), "regions": c.get("regions") or [],
                    "matched": [REASONS.get(t) or TAGS[t][0] for t in why],
                    "seen": [REASONS.get(t) or TAGS[t][0] for t in seen]}
            v = c.get("view")
            if v:
                card["look"] = look_text(c.get("category"), v["pitch"], v["zoom"])
                card["view"] = {"pitch": round(v["pitch"], 1), "zoom": round(v["zoom"], 2),
                                "fov": round(zoom_fov(v["zoom"]))}
            res["geoguessr"].append(card)
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


REASONS = {"_region": "Імовірний регіон", "_similar": "Є на схожих панорамах",
           "_regional": "Часта в імовірному регіоні", "_frequent": "Часта картка країни"}

_BASE = None


def clue_base():
    global _BASE
    if _BASE is None:
        _BASE = ClueBase()
    return _BASE


def region_probs_for(regions, cc):
    """Admin-1 posterior restricted to country cc and renormalised: {code: probability}."""
    rp = {r["code"]: r["probability"] for r in (regions or []) if r["country"] == cc}
    tot = sum(rp.values())
    return {k: v / tot for k, v in rp.items()} if tot > 0 else {}


def build_hints(F, top_countries, n_countries=3, regions=None, n_cards=3, kb=None):
    """F: {'module.feature': value}; top_countries: [(code, prob), ...];
    regions: [{"code", "name", "country", "probability"}, ...] (posterior over admin-1 regions);
    n_cards: GeoGuessr cards per country; kb: a ClueBase (default: the shared one)."""
    tags = observations(F)
    kb = kb or clue_base()
    emb = kb.index.embed(F) if kb.index is not None else None
    obs = [{"tag": t, "text": TAGS[t][0], "strength": s} for t, s in tags]
    side_tag = next((t for t, _ in tags if t in ("drive_left", "drive_right")), None)
    names = {r["code"]: r["name"] for r in (regions or [])}
    cards = []
    for cc, p in top_countries[:n_countries]:
        rp = region_probs_for(regions, cc)
        c = kb.for_country(cc, tags, n_gg=n_cards, region_probs=rp, emb=emb)
        c["regions"] = [{"code": k, "name": names[k], "probability": round(v, 3)}
                        for k, v in sorted(rp.items(), key=lambda kv: -kv[1])[:3]]
        c["probability"] = round(float(p), 3)
        if side_tag and c["driving_side"]:
            c["driving_side_consistent"] = (c["driving_side"] == side_tag.split("_")[1])
        cards.append(c)
    return {"observations": obs, "countries": cards}
