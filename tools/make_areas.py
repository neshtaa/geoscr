#!/usr/bin/env python3
"""
Writes data/areas.json: GeoGuessr-style areas that group the admin-1 units of the region raster
(data/world_regions.npz, Natural Earth 10m admin-1) per country, read by engine/areas.py.

  python3 tools/make_areas.py [scratch/ne/admin1.geojson]

The Natural Earth admin-1 GeoJSON (the file tools/make_region_raster.py rasterised; download command there) gives
  * the "region" attribute of the countries grouped from it (NE_REGION below: Slovenia's statistical regions,
    Latvia's planning regions, North Macedonia's statistical regions, Uganda's four regions, ...);
  * the label point of every raster unit without pixels (unit_points: engine/areas.py places such units of a
    country with automatic areas in the nearest area);
  * the names of units whose raster (name_en) name is wrong (region_names).
Every other grouping is written out below with its source.  Checks (exit status 1 on a problem): every raster unit
of a listed country is in exactly one area or in the country's "hidden" list, and at least half of a country's areas
group two or more units (a one-to-one "grouping" only repeats the region list in the HUD).
"""
import collections
import json
import os
import re
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.geo import _regions, country_at  # noqa: E402

_, CODES, NAMES, RCC, _ = _regions()
NE_DEFAULT = os.path.join(ROOT, "scratch", "ne", "admin1.geojson")


def P(cc, s):
    return ["%s-%s" % (cc, x) for x in s.split()]


A = OrderedDict()
SRC = {}
HIDDEN = {}
NE_AREAS = OrderedDict()   # cc -> (source, [(uk, en, [NE region values])], {code: NE region value} overrides)


def country(cc, source, areas, note=None, hidden=None):
    A[cc] = [{"id": "%s:%s" % (cc, en.replace(" ", "")), "name": uk, "en": en, "regions": regs}
             for uk, en, regs in areas]
    SRC[cc] = source if note is None else source + " " + note
    if hidden:
        HIDDEN[cc] = list(hidden)


def ne_country(cc, source, areas, fix=None):
    """Areas from the Natural Earth "region" attribute: areas = [(uk, en, [NE region values])]; fix = {raster code:
    NE region value} for units whose attribute is empty."""
    NE_AREAS[cc] = (source, areas, dict(fix or {}))


# ----------------------------------------------------------------------------------------------- Americas
country("US", "US Census Bureau census regions and divisions (Alaska and Hawaii split from the Pacific division).", [
    ("Нова Англія", "New England", P("US", "CT ME MA NH RI VT")),
    ("Сер.-атлантичні штати", "Mid-Atlantic", P("US", "NJ NY PA")),
    ("Великі озера", "East North Central", P("US", "IL IN MI OH WI")),
    ("Великі рівнини", "West North Central", P("US", "IA KS MN MO NE ND SD")),
    ("Південна Атлантика", "South Atlantic", P("US", "DE DC FL GA MD NC SC VA WV")),
    ("Теннессі–Алабама", "East South Central", P("US", "AL KY MS TN")),
    ("Техас і сусіди", "West South Central", P("US", "AR LA OK TX")),
    ("Гірські штати", "Mountain", P("US", "AZ CO ID MT NV NM UT WY")),
    ("Західне узбережжя", "Pacific", P("US", "CA OR WA")),
    ("Аляска", "Alaska", P("US", "AK")),
    ("Гаваї", "Hawaii", P("US", "HI")),
])
country("MX", "Common five-way split of the 32 states (north / west-Bajio / centre / south-Gulf / Yucatan Peninsula), "
        "after the INEGI-style macro-regions used by GeoGuessr guides.", [
    ("Північ", "North", P("MX", "BCN BCS SON CHH COA NLE TAM SIN DUR")),
    ("Захід і Бахіо", "West and Bajio", P("MX", "JAL COL MIC NAY AGU ZAC SLP GUA QUE")),
    ("Центр (Мехіко)", "Centre", P("MX", "DIF MEX MOR PUE TLA HID")),
    ("Південь і затока", "South and Gulf", P("MX", "GRO OAX CHP VER TAB")),
    ("Юкатан", "Yucatan Peninsula", P("MX", "CAM YUC ROO MEX99")),
])
country("BR", "IBGE macro-regions (Norte, Nordeste, Centro-Oeste, Sudeste, Sul).", [
    ("Північ (Норте)", "North", P("BR", "AC AM AP PA RO RR TO")),
    ("Північний Схід (Нордесте)", "Northeast", P("BR", "AL BA CE MA PB PE PI RN SE")),
    ("Центр-Захід", "Centre-West", P("BR", "DF GO MS MT")),
    ("Південний Схід", "Southeast", P("BR", "ES MG RJ SP")),
    ("Південь (Сул)", "South", P("BR", "PR RS SC")),
])
country("AR", "Geographic regions of Argentina (NOA, NEA, Cuyo, Pampas / Centro, Patagonia).", [
    ("Північний Захід (NOA)", "Northwest", P("AR", "Y A T K G F")),
    ("Північний Схід (NEA)", "Northeast", P("AR", "P H N W")),
    ("Куйо", "Cuyo", P("AR", "M J D")),
    ("Пампа і Центр", "Pampas and Centre", P("AR", "B C X S E L")),
    ("Патагонія", "Patagonia", P("AR", "Q R U Z V")),
])
country("CL", "Chile's natural macro-zones (Norte Grande, Norte Chico, Zona Central, Zona Sur, Zona Austral).", [
    ("Крайня північ", "Norte Grande", P("CL", "AP TA AN")),
    ("Норте-Чіко", "Norte Chico", P("CL", "AT CO")),
    ("Центр", "Zona Central", P("CL", "VS RM LI ML")),
    ("Південь", "Zona Sur", P("CL", "NB BI AR LR LL")),
    ("Крайній Південь", "Zona Austral", P("CL", "AI MA")),
])
country("CO", "Natural regions of Colombia (departments by their main region).", [
    ("Карибське узбережжя", "Caribbean", P("CO", "ATL BOL CES COR LAG MAG SUC SAP")),
    ("Анди", "Andes", P("CO", "ANT CAL RIS QUI SAN NSA BOY CUN TOL HUI")),
    ("Тихоокеанський захід", "Pacific and southwest", P("CO", "CHO VAC CAU NAR COL99")),
    ("Льянос (Орінокія)", "Orinoquia", P("CO", "ARA CAS MET VID")),
    ("Амазонія", "Amazon", P("CO", "AMA CAQ GUA GUV PUT VAU")),
])
country("PE", "Macro-regions of Peru (north / centre incl. Lima / south / Amazon).", [
    ("Північ", "North", P("PE", "TUM PIU LAM LAL CAJ AMA")),
    ("Центр і Ліма", "Centre and Lima", P("PE", "LIM CAL ANC HUC PAS JUN HUV ICA")),
    ("Південь", "South", P("PE", "ARE MOQ TAC PUN CUS APU AYA")),
    ("Амазонія", "Amazon", P("PE", "LOR UCA MDD SAM")),
])
country("EC", "Natural regions of Ecuador (Costa, Sierra, Oriente, Galapagos).", [
    ("Узбережжя (Коста)", "Coast", P("EC", "E M G SE O R SD")),
    ("Анди (Сьєрра)", "Highlands", P("EC", "C I P X T B H F A L")),
    ("Амазонія (Оріенте)", "Amazon", P("EC", "U N D Y S Z")),
    ("Галапагоси", "Galapagos", P("EC", "W")),
])
country("BO", "Bolivia's three physiographic zones (Altiplano, valleys, lowlands).", [
    ("Альтіплано", "Altiplano", P("BO", "L O P")),
    ("Долини", "Valleys", P("BO", "C H T")),
    ("Низовини (Льянос)", "Lowlands", P("BO", "S B N")),
])
country("PY", "Eastern (Oriental) and western (Chaco) Paraguay.", [
    ("Схід (Орієнталь)", "Eastern", P("PY", "ASU 1 2 3 4 5 6 7 8 9 10 11 12 13 14")),
    ("Чако", "Chaco", P("PY", "15 16 19")),
])
country("UY", "Montevideo metro / coast / interior split common in guides.", [
    ("Монтевідео і південь", "Montevideo and south", P("UY", "MO CA SJ FD")),
    ("Узбережжя (схід)", "East coast", P("UY", "MA RO LA TT")),
    ("Річка Уругвай (захід)", "West (Uruguay River)", P("UY", "CO SO RN PA SA AR")),
    ("Північ і центр", "Interior", P("UY", "RV TA CL DU FS")),
])
country("GT", "Regions of Guatemala grouped (centre / west highlands / north / east / Pacific coast).", [
    ("Центр", "Centre", P("GT", "GU SA CM")),
    ("Західне високогір'я", "Western highlands", P("GT", "QZ TO SO SM HU QC")),
    ("Північ (Петен і Верапаси)", "North", P("GT", "PE AV BV")),
    ("Схід", "East", P("GT", "IZ ZA CQ PR JA JU SR")),
    ("Тихоокеанське узбережжя", "Pacific coast", P("GT", "ES SU RE")),
])
# ----------------------------------------------------------------------------------------------- Europe
country("RU", "Federal districts of Russia.", [
    ("Центральний ФО", "Central", P("RU", "BEL BRY VLA VOR IVA KLU KOS KRS LIP MOS MOW ORL RYA SMO TAM TVE TUL YAR")),
    ("Північно-Західний ФО", "Northwestern", P("RU", "ARK VLG KGD KR KO LEN MUR NEN NGR PSK SPE")),
    ("Південний ФО", "Southern", P("RU", "AD KL KDA AST VGG ROS")),
    ("Північно-Кавказький ФО", "North Caucasian", P("RU", "DA IN KB KC SE CE STA")),
    ("Приволзький ФО", "Volga", P("RU", "BA ME MO TA UD CU PER KIR NIZ ORE PNZ SAM SAR ULY")),
    ("Уральський ФО", "Ural", P("RU", "KGN SVE TYU CHE KHM YAN RUS99")),
    ("Сибірський ФО", "Siberian", P("RU", "AL ALT TY KK KYA IRK KEM NVS OMS TOM")),
    ("Далекосхідний ФО", "Far Eastern", P("RU", "BU ZAB SA KAM PRI KHA AMU MAG SAK YEV CHU")),
], note="RU-RUS99, an unnamed raster sliver at 68.8N 67.3E (Yamal), counts in the Ural district.  Hidden: Crimea "
        "(UA-43, UA-40) is Ukraine; the Natural Earth raster files it under Russia, so it appears in no area and in "
        "no region list of Russia.", hidden=["UA-43", "UA-40"])
country("UA", "Common macro-regions of Ukraine (west / north / centre / east / south).", [
    ("Захід", "West", P("UA", "07 56 46 61 26 21 77 68")),
    ("Північ і Київ", "North and Kyiv", P("UA", "30 32 18 74 59")),
    ("Центр", "Centre", P("UA", "05 71 35 53")),
    ("Схід", "East", P("UA", "63 12 14 09")),
    ("Південь", "South", P("UA", "51 48 65 23")),
])
country("FR", "The 13 metropolitan regions of 2016 (from the departments of the raster); overseas departments are "
        "their own GeoGuessr countries.", [
    ("Іль-де-Франс", "Ile-de-France", P("FR", "75 77 78 91 92 93 94 95")),
    ("О-де-Франс", "Hauts-de-France", P("FR", "02 59 60 62 80")),
    ("Гранд-Ест", "Grand Est", P("FR", "08 10 51 52 54 55 57 67 68 88")),
    ("Нормандія", "Normandy", P("FR", "14 27 50 61 76")),
    ("Бретань", "Brittany", P("FR", "22 29 35 56")),
    ("Луара", "Pays de la Loire", P("FR", "44 49 53 72 85")),
    ("Центр — Долина Луари", "Centre-Val de Loire", P("FR", "18 28 36 37 41 45")),
    ("Бургундія — Франш-Конте", "Bourgogne-Franche-Comte", P("FR", "21 25 39 58 70 71 89 90")),
    ("Нова Аквітанія", "Nouvelle-Aquitaine", P("FR", "16 17 19 23 24 33 40 47 64 79 86 87")),
    ("Окситанія", "Occitanie", P("FR", "09 11 12 30 31 32 34 46 48 65 66 81 82")),
    ("Овернь — Рона — Альпи", "Auvergne-Rhone-Alpes", P("FR", "01 03 07 15 26 38 42 43 63 69 73 74")),
    ("Прованс — Лазурний берег", "Provence-Alpes-Cote d'Azur", P("FR", "04 05 06 13 83 84")),
    ("Корсика", "Corsica", P("FR", "2A 2B")),
    ("Заморські території", "Overseas", P("FR", "GF MQ GP RE YT CLP00")),
])
country("ES", "Autonomous communities of Spain (from the provinces of the raster).", [
    ("Галісія", "Galicia", P("ES", "C LU OR PO")), ("Астурія", "Asturias", P("ES", "O")),
    ("Кантабрія", "Cantabria", P("ES", "S")), ("Країна Басків", "Basque Country", P("ES", "BI SS VI")),
    ("Наварра", "Navarre", P("ES", "NA")), ("Ла-Ріоха", "La Rioja", P("ES", "LO")),
    ("Арагон", "Aragon", P("ES", "HU TE Z")), ("Каталонія", "Catalonia", P("ES", "B GI L T")),
    ("Кастилія і Леон", "Castile and Leon", P("ES", "AV BU LE P SA SG SO VA ZA")),
    ("Мадрид", "Madrid", P("ES", "M")),
    ("Кастилія — Ла-Манча", "Castile-La Mancha", P("ES", "AB CR CU GU TO")),
    ("Валенсія", "Valencia", P("ES", "A CS V")), ("Мурсія", "Murcia", P("ES", "MU")),
    ("Естремадура", "Extremadura", P("ES", "BA CC")),
    ("Андалусія", "Andalusia", P("ES", "AL CA CO GR H J MA SE")),
    ("Балеарські острови", "Balearic Islands", P("ES", "PM")),
    ("Канарські острови", "Canary Islands", P("ES", "GC TF")),
    ("Сеута і Мелілья", "Ceuta and Melilla", P("ES", "CE ML")),
])
country("PT", "NUTS-2 regions of Portugal approximated by districts (Santarem counted in the Centre).", [
    ("Північ", "North", P("PT", "16 03 13 17 04")),
    ("Центр", "Centre", P("PT", "01 18 09 06 05 10 14")),
    ("Лісабон", "Lisbon", P("PT", "11 15")),
    ("Алентежу", "Alentejo", P("PT", "12 07 02")),
    ("Алгарве", "Algarve", P("PT", "08")),
    ("Азори", "Azores", P("PT", "20")), ("Мадейра", "Madeira", P("PT", "30")),
])
country("IT", "The 20 regions of Italy (from the provinces of the raster).", [
    ("Валле-д'Аоста", "Aosta Valley", P("IT", "AO")),
    ("П'ємонт", "Piedmont", P("IT", "AL AT BI CN NO TO VB VC")),
    ("Лігурія", "Liguria", P("IT", "GE IM SP SV")),
    ("Ломбардія", "Lombardy", P("IT", "BG BS CO CR LC LO MB MI MN PV SO VA")),
    ("Трентіно — Альто-Адідже", "Trentino-South Tyrol", P("IT", "BZ TN")),
    ("Венето", "Veneto", P("IT", "BL PD RO TV VE VI VR")),
    ("Фріулі — Венеція-Джулія", "Friuli-Venezia Giulia", P("IT", "GO PN TS UD")),
    ("Емілія-Романья", "Emilia-Romagna", P("IT", "BO FC FE MO PC PR RA RE RN")),
    ("Тоскана", "Tuscany", P("IT", "AR FI GR LI LU MS PI PO PT SI")),
    ("Умбрія", "Umbria", P("IT", "PG TR")), ("Марке", "Marche", P("IT", "AN AP FM MC PU")),
    ("Лаціо", "Lazio", P("IT", "FR LT RI RM VT")), ("Абруццо", "Abruzzo", P("IT", "AQ CH PE TE")),
    ("Молізе", "Molise", P("IT", "CB IS")), ("Кампанія", "Campania", P("IT", "AV BN CE NA SA")),
    ("Апулія", "Apulia", P("IT", "BA BR BT FG LE TA")), ("Базиліката", "Basilicata", P("IT", "MT PZ")),
    ("Калабрія", "Calabria", P("IT", "CS CZ KR RC VV")),
    ("Сицилія", "Sicily", P("IT", "AG CL CT EN ME PA RG SR TP")),
    ("Сардинія", "Sardinia", P("IT", "CA CI NU OG OR OT SS VS")),
])
GB_SCO = "ABE ABD ANS AGB CLK DGY DND EAY EDU ELN ERW EDH ELS FAL FIF GLG HLD IVC MLN MRY NAY NLK ORK PKN RFW SCB ZET SAY SLK STG WDU WLN"
GB_WLS = "AGY BGW BGE CAY CRF CMN CGN CWY DEN FLN GWN MTY MON NTL NWP PEM POW RCT SWA TOF VGL WRX"
GB_NIR = "ANT ARD ARM BLA BLY BNB BFS CKF CSR CLR CKT CGV DRY DOW DGN FER LRN LMV LSB MFT MYL NYM NTA NDN OMH STB"
country("GB", "The four nations; England in four groups of ONS regions (North East + North West + Yorkshire and the "
        "Humber, East + West Midlands, London + South East + East of England, South West).", [
    ("Шотландія", "Scotland", P("GB", GB_SCO)),
    ("Уельс", "Wales", P("GB", GB_WLS)),
    ("Північна Ірландія", "Northern Ireland", P("GB", GB_NIR)),
    ("Північ Англії", "Northern England", P("GB", "NTY STY SND DUR HPL RCC STT DAL MDB NBL GAT NET CMA CHW CHE WRL HAL "
                                                 "KWL LIV SFT LAN BPL BBD SHN WGN SKP WRT SLF BOL TRF MAN OLD RCH TAM BUR "
                                                 "NYK ERY KHL NLN NEL YOR DNC ROT SHF CLD KIR LDS BRD WKF BNS")),
    ("Мідлендс", "Midlands", P("GB", "LIN RUT NTT NTH LEC DBY NGM LCE DER SHR HEF STE TFW STS WOR WAR SOL COV BIR SAW "
                                     "DUD WLL WLV")),
    ("Лондон і південний схід", "London and South East", P("GB", "WND MRT WSM KEC HNS EAL HMF RIC LND TWH ENF BNE WFT "
                                                                  "RDB HAV BEX STN HIL BEN HRW CMD ISL LBH SWK CRY LEW HRY "
                                                                  "KTT NWM GRE HCK BDG BRY KEN MDW ESX BNH WSX HAM POR STH "
                                                                  "IOW MIK BKM OXF WBK WOK BRC WNM SLG RDG SRY NFK SFK ESS "
                                                                  "SOS THR HRT CAM LUT CBF BDF PTE")),
    ("Південний захід Англії", "South West England", P("GB", "GLS DOR BMH POL DEV TOB PLY CON SOM NSM BST SGC IOS SWD "
                                                             "BAS WIL")),
])
country("IE", "The four provinces of Ireland.", [
    ("Ленстер", "Leinster", P("IE", "D WW WX KK MH KE CW LS OY WH LD LH")),
    ("Манстер", "Munster", P("IE", "WD CO KY LK CE TA")),
    ("Коннахт", "Connacht", P("IE", "G MO SO LM RN")),
    ("Ольстер (Ірландія)", "Ulster (Republic)", P("IE", "DL CN MN")),
])
country("NO", "The traditional regions (landsdeler) of Norway over the pre-2020 counties of the raster; Svalbard and "
        "Bouvet Island apart.", [
    ("Північна Норвегія", "Northern Norway", P("NO", "18 19 20")),
    ("Трьонделаг", "Trondelag", P("NO", "16 17")),
    ("Західна Норвегія (фіорди)", "Western Norway", P("NO", "11 12 14 15")),
    ("Сьорланн (південь)", "Southern Norway", P("NO", "09 10")),
    ("Східна Норвегія", "Eastern Norway", P("NO", "01 02 03 04 05 06 07 08")),
    ("Шпіцберген", "Svalbard", P("NO", "21")), ("Острів Буве", "Bouvet Island", P("NO", "BVT00")),
])
country("SE", "The three lands of Sweden (Norrland, Svealand, Gotaland) over the counties.", [
    ("Норрланд", "Norrland", P("SE", "BD AC Z Y X")),
    ("Свеаланд", "Svealand", P("SE", "W S C AB D U T")),
    ("Йоталанд", "Gotaland", P("SE", "O E H K M N I F G")),
])
country("FI", "The provinces of Finland 1997-2009 (Southern, Western, Eastern, Oulu, Lapland) over the regions.", [
    ("Південна Фінляндія", "Southern Finland", P("FI", "18 09 02 16 06")),
    ("Західна Фінляндія", "Western Finland", P("FI", "19 17 11 08 03 12 07")),
    ("Східна Фінляндія", "Eastern Finland", P("FI", "04 15 13")),
    ("Оулу і Кайнуу", "Oulu", P("FI", "14 05")),
    ("Лапландія", "Lapland", P("FI", "10")),
])
country("PL", "Macro-regions of Poland (NUTS-1, 2018) over the voivodeships.", [
    ("Мазовія", "Masovia", P("PL", "MZ")),
    ("Сілезія і Мала Польща", "South", P("PL", "SL MA")),
    ("Схід", "East", P("PL", "LU PK PD SK")),
    ("Північний захід", "Northwest", P("PL", "WP ZP LB")),
    ("Південний захід", "Southwest", P("PL", "DS OP")),
    ("Північ", "North", P("PL", "KP PM WN")),
    ("Центр (Лодзь)", "Centre", P("PL", "LD")),
])
country("DE", "German states grouped as north / east / west / south (common GeoGuessr split).", [
    ("Північ", "North", P("DE", "SH HH HB NI MV")),
    ("Схід", "East", P("DE", "BE BB ST SN TH")),
    ("Захід", "West", P("DE", "NW HE RP SL")),
    ("Південь", "South", P("DE", "BY BW")),
])
country("RO", "Development regions of Romania (NUTS-2).", [
    ("Північний захід", "North-West", P("RO", "BH BN CJ MM SM SJ")),
    ("Центр (Трансільванія)", "Centre", P("RO", "AB BV CV HR MS SB")),
    ("Північний схід (Молдова)", "North-East", P("RO", "BC BT IS NT SV VS")),
    ("Південний схід (Добруджа)", "South-East", P("RO", "BR BZ CT GL TL VN")),
    ("Південь (Мунтенія)", "South-Muntenia", P("RO", "AG CL DB GR IL PH TR")),
    ("Бухарест-Ілфов", "Bucharest-Ilfov", P("RO", "B IF")),
    ("Південний захід (Олтенія)", "South-West Oltenia", P("RO", "DJ GJ MH OT VL")),
    ("Захід (Банат)", "West", P("RO", "AR CS HD TM")),
])
country("HU", "Regions of Hungary (NUTS-2) over the counties and the cities with county rights of the raster.", [
    ("Будапешт і Пешт", "Central Hungary", P("HU", "BU PE ED")),
    ("Центральне Задунав'я", "Central Transdanubia", P("HU", "FE KE VE SF TB VM DU")),
    ("Західне Задунав'я", "Western Transdanubia", P("HU", "GS VA ZA SN GY SH ZE NK")),
    ("Південне Задунав'я", "Southern Transdanubia", P("HU", "BA SO TO PS KV SS")),
    ("Північна Угорщина", "Northern Hungary", P("HU", "BZ HE NO EG MI ST")),
    ("Північний Альфельд", "Northern Great Plain", P("HU", "HB JN SZ DE NY SK")),
    ("Південний Альфельд", "Southern Great Plain", P("HU", "BK BE CS SD HV KM BC")),
])
country("BG", "Planning regions of Bulgaria (NUTS-2).", [
    ("Північний захід", "North-West", P("BG", "05 06 12 15 11")),
    ("Північний центр", "North-Central", P("BG", "04 07 17 18 19")),
    ("Північний схід", "North-East", P("BG", "03 08 27 25")),
    ("Південний схід", "South-East", P("BG", "02 20 28 24")),
    ("Південний захід (Софія)", "South-West", P("BG", "22 23 14 10 01")),
    ("Південний центр", "South-Central", P("BG", "16 13 21 09 26")),
])
country("HR", "Croatia: Adriatic, Zagreb and the north, Central Croatia, Slavonia.", [
    ("Адріатика", "Adriatic", P("HR", "18 08 09 13 15 17 19")),
    ("Загреб і північ", "Zagreb and north", P("HR", "21 01 02 05 20 06")),
    ("Центральна Хорватія", "Central Croatia", P("HR", "04 03 07")),
    ("Славонія", "Slavonia", P("HR", "10 11 12 14 16")),
], note="HR-11 (Pozega-Slavonia) may be missing from the raster.")
country("CH", "The seven Grossregionen of Switzerland (NUTS-2).", [
    ("Регіон Женевського озера", "Lake Geneva region", P("CH", "GE VD VS")),
    ("Еспас-Міттелланд", "Espace Mittelland", P("CH", "BE FR SO NE JU")),
    ("Північний захід", "Northwestern Switzerland", P("CH", "BS BL AG")),
    ("Цюрих", "Zurich", P("CH", "ZH")),
    ("Східна Швейцарія", "Eastern Switzerland", P("CH", "GL SH AR AI SG GR TG")),
    ("Центральна Швейцарія", "Central Switzerland", P("CH", "LU UR SZ OW NW ZG")),
    ("Тічино", "Ticino", P("CH", "TI")),
])
country("BE", "Belgium's regions (Flanders, Wallonia, Brussels).", [
    ("Фландрія", "Flanders", P("BE", "VWV VOV VAN VLI VBR")),
    ("Валлонія", "Wallonia", P("BE", "WHT WNA WLX WLG WBR")),
    ("Брюссель", "Brussels", P("BE", "BRU")),
])
country("RS", "Statistical regions of Serbia.", [
    ("Воєводина", "Vojvodina", P("RS", "01 02 03 04 05 06 07")),
    ("Белград", "Belgrade", P("RS", "00")),
    ("Шумадія і Західна Сербія", "Sumadija and Western Serbia", P("RS", "08 09 12 13 16 17 18 19")),
    ("Південна і Східна Сербія", "Southern and Eastern Serbia", P("RS", "10 11 14 15 20 21 22 23 24")),
])
country("GR", "Greece's 13 regions grouped as north (Macedonia, Thrace, Epirus) / centre (Thessaly, Central Greece, "
        "Attica) / Peloponnese and west / islands.", [
    ("Північна Греція", "North", P("GR", "A B C D 69")),
    ("Центр і Аттика", "Centre and Attica", P("GR", "E H A1")),
    ("Пелопоннес і захід", "Peloponnese and West", P("GR", "J G")),
    ("Острови", "Islands", P("GR", "M L K F")),
])
country("TR", "The seven geographical regions of Turkey (1941 Geography Congress) over the provinces (plate codes).", [
    ("Мармурове море", "Marmara", P("TR", "10 11 16 17 22 34 39 41 54 59 77")),
    ("Егейське узбережжя", "Aegean", P("TR", "03 09 20 35 43 45 48 64")),
    ("Середземномор'я", "Mediterranean", P("TR", "01 07 15 31 32 33 46 80")),
    ("Центральна Анатолія", "Central Anatolia", P("TR", "06 18 26 38 40 42 50 51 58 66 68 70 71")),
    ("Чорноморське узбережжя", "Black Sea", P("TR", "05 08 14 19 28 29 37 52 53 55 57 60 61 67 69 74 78 81")),
    ("Східна Анатолія", "Eastern Anatolia", P("TR", "04 12 13 23 24 25 30 36 44 49 62 65 75 76")),
    ("Південно-Східна Анатолія", "Southeastern Anatolia", P("TR", "02 21 27 47 56 63 72 73 79")),
])
# ----------------------------------------------------------------------------------------------- Asia, Oceania
country("JP", "The traditional regions of Japan (Okinawa apart from Kyushu).", [
    ("Хоккайдо", "Hokkaido", P("JP", "01")),
    ("Тохоку", "Tohoku", P("JP", "02 03 04 05 06 07")),
    ("Канто", "Kanto", P("JP", "08 09 10 11 12 13 14")),
    ("Тюбу", "Chubu", P("JP", "15 16 17 18 19 20 21 22 23")),
    ("Кансай (Кінкі)", "Kansai", P("JP", "24 25 26 27 28 29 30")),
    ("Тюґоку", "Chugoku", P("JP", "31 32 33 34 35")),
    ("Сікоку", "Shikoku", P("JP", "36 37 38 39")),
    ("Кюсю", "Kyushu", P("JP", "40 41 42 43 44 45 46")),
    ("Окінава", "Okinawa", P("JP", "47")),
])
country("KR", "Traditional provinces of South Korea with their metropolitan cities.", [
    ("Сеул і столичний регіон", "Seoul Capital Area", P("KR", "11 28 41")),
    ("Канвон", "Gangwon", P("KR", "42")),
    ("Чхунчхон", "Chungcheong", P("KR", "43 44 30 50")),
    ("Чолла", "Jeolla", P("KR", "45 46 29")),
    ("Кьонсан", "Gyeongsang", P("KR", "47 48 26 27 31")),
    ("Чеджу", "Jeju", P("KR", "49")),
])
country("TW", "Taiwan's north / centre / south / east and the outlying islands.", [
    ("Північ", "North", P("TW", "TPE TPQ KEE TAO HSZ HSQ ILA")),
    ("Центр", "Centre", P("TW", "MIA TXG CHA NAN YUN")),
    ("Південь", "South", P("TW", "CYI CYQ TNN KHH PIF")),
    ("Схід", "East", P("TW", "HUA TTT")),
    ("Острови", "Outlying islands", P("TW", "KIN PEN")),
])
country("IN", "Zonal councils of India (States Reorganisation Act 1956), Andaman, Nicobar and Lakshadweep with the "
        "south.", [
    ("Північ", "North", P("IN", "JK KAS00 LA HP PB CH HR DL RJ")),
    ("Центр", "Central", P("IN", "UP UT MP CT")),
    ("Схід", "East", P("IN", "BR JH OR WB")),
    ("Захід", "West", P("IN", "GJ MH GA DH")),
    ("Південь", "South", P("IN", "AP TG KA KL TN PY LD AN")),
    ("Північний схід", "Northeast", P("IN", "AS AR MN ML MZ NL TR SK")),
])
country("ID", "Indonesia's main islands / island groups.", [
    ("Суматра", "Sumatra", P("ID", "AC SU SB RI KR JA SS BE LA BB")),
    ("Ява", "Java", P("ID", "BT JK JB JT YO JI")),
    ("Балі і Нуса-Тенгара", "Bali and Nusa Tenggara", P("ID", "BA NB NT")),
    ("Калімантан", "Kalimantan", P("ID", "KB KT KS KI")),
    ("Сулавесі", "Sulawesi", P("ID", "SA GO ST SR SN SG")),
    ("Молукки і Папуа", "Maluku and Papua", P("ID", "MA MU PA PB")),
])
country("PH", "The island groups of the Philippines, Luzon in north / centre-south, Bicol and MIMAROPA.", [
    ("Північний Лусон", "Northern Luzon", P("PH", "ILN ILS LUN PAN ABR APA KAL MOU IFU BEN CAG ISA QUI NUV BTN")),
    ("Центр. і Півд. Лусон", "Central and Southern Luzon", P("PH", "AUR NUE TAR ZMB BAN PAM BUL MNL RIZ LAG CAV "
                                                                           "BTG QUE")),
    ("Бікол", "Bicol", P("PH", "CAN CAS ALB SOR CAT MAS")),
    ("Міндоро і Палаван", "MIMAROPA", P("PH", "MDR MDC MAD ROM PLW")),
    ("Вісайї", "Visayas", P("PH", "AKL ANT CAP ILI GUI NEC BCD NER CEB MDE LAP BOH SIG LEY SLE BIL WSA EAS NSA")),
    ("Мінданао", "Mindanao", P("PH", "ZAN ZAS ZSI MSC MSR LAN CAM BUK LAS AGN AGS SUN SUR DAV COM DAS DAO NCO SUK SAR "
                                     "SCO MAG BAS SLU TAW")),
])
country("TH", "The six geographic regions of Thailand (National Research Council 1977).", [
    ("Північ", "North", P("TH", "50 51 52 53 54 55 56 57 58")),
    ("Північний схід (Ісан)", "Northeast (Isan)", P("TH", "30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 "
                                                           "49")),
    ("Центр і Бангкок", "Central", P("TH", "10 11 12 13 14 15 16 17 18 19 26 60 61 62 64 65 66 67 72 73 74 75")),
    ("Схід", "East", P("TH", "20 21 22 23 24 25 27")),
    ("Захід", "West", P("TH", "63 70 71 76 77")),
    ("Південь", "South", P("TH", "80 81 82 83 84 85 86 90 91 92 93 94 95 96")),
])
country("VN", "Vietnam's three parts (North, Centre incl. the Central Highlands, South).", [
    ("Північ", "North", P("VN", "01 02 03 04 05 06 07 09 13 14 18 20 53 54 56 61 63 66 67 68 69 70 71 HN HP")),
    ("Центр", "Centre", P("VN", "21 22 23 24 25 26 27 29 31 32 34 36 40 DN 28 30 33 35 72")),
    ("Південь", "South", P("VN", "37 39 41 43 44 45 46 47 49 50 51 52 55 57 58 59 73 SG CT")),
])
country("KH", "The four natural regions of Cambodia with Phnom Penh in the plain.", [
    ("Рівнина і Пномпень", "Central plain", P("KH", "12 8 3 14 20 21")),
    ("Тонлесап", "Tonle Sap", P("KH", "1 2 4 6 15 17 24")),
    ("Узбережжя", "Coast", P("KH", "7 9 18 23")),
    ("Плато і гори", "Plateau and mountains", P("KH", "5 10 11 13 16 19 22")),
], note="KH-5 ('Kam') is Kampong Speu in the raster.")
country("MY", "Peninsular Malaysia (north / east coast / centre-south) and East Malaysia.", [
    ("Північ півострова", "Northern Peninsula", P("MY", "09 02 07 08")),
    ("Східне узбережжя", "East Coast", P("MY", "03 11 06")),
    ("Центр і південь", "Central and Southern Peninsula", P("MY", "10 14 16 05 04 01")),
    ("Сабах", "Sabah", P("MY", "12 15")),
    ("Саравак", "Sarawak", P("MY", "13")),
])
country("LK", "Provinces of Sri Lanka over the districts.", [
    ("Західна", "Western", P("LK", "11 12 13")),
    ("Центральна", "Central", P("LK", "21 22 23")),
    ("Південна", "Southern", P("LK", "31 32 33")),
    ("Північна", "Northern", P("LK", "41 42 43 44 45")),
    ("Східна", "Eastern", P("LK", "51 52 53")),
    ("Північно-Західна", "North Western", P("LK", "61 62")),
    ("Північно-Центральна", "North Central", P("LK", "71 72")),
    ("Ува", "Uva", P("LK", "81 82")),
    ("Сабарагамува", "Sabaragamuwa", P("LK", "91 92")),
])
country("NP", "Nepal's development regions over the former zones.", [
    ("Далекий захід", "Far-Western", P("NP", "MA SE")),
    ("Середній захід", "Mid-Western", P("NP", "KA BH RA")),
    ("Захід", "Western", P("NP", "DH GA LU")),
    ("Центр (Катманду)", "Central", P("NP", "BA JA NA")),
    ("Схід", "Eastern", P("NP", "ME KO SA")),
])
country("MN", "The regions of Mongolia (Western, Khangai, Central, Eastern, Ulaanbaatar).", [
    ("Захід", "Western", P("MN", "071 065 043 046 057")),
    ("Хангай", "Khangai", P("MN", "073 069 067 035 055 041")),
    ("Центр", "Central", P("MN", "037 063 059 064 053 049 047")),
    ("Схід", "Eastern", P("MN", "061 039 051")),
    ("Улан-Батор", "Ulaanbaatar", P("MN", "1")),
])
country("KZ", "Kazakhstan's west / north / centre / east / south.", [
    ("Захід", "West", P("KZ", "ATY MAN ZAP AKT")),
    ("Північ і Астана", "North", P("KZ", "KUS SEV AKM PAV AST")),
    ("Центр", "Centre", P("KZ", "KAR")),
    ("Схід", "East", P("KZ", "VOS")),
    ("Південь", "South", P("KZ", "ALA ZHA YUZ KZY KAB00")),
])
country("NZ", "North Island / South Island; Chatham and the subantarctic islands apart.", [
    ("Північний острів", "North Island", P("NZ", "NTL AUK WKO BOP GIS HKB TKI MWT WGN ZL5476 ZL5471")),
    ("Південний острів", "South Island", P("NZ", "TAS NSN MBH WTC CAN OTA STL")),
    ("Віддалені острови", "Outlying islands", P("NZ", "CIT ZL5477 ZL5478 ZL5479 ZL5480")),
])
# ----------------------------------------------------------------------------------------------- Africa
country("NG", "The six geopolitical zones of Nigeria.", [
    ("Північний захід", "North West", P("NG", "JI KD KN KT KE SO ZA")),
    ("Північний схід", "North East", P("NG", "AD BA BO GO TA YO")),
    ("Північний центр", "North Central", P("NG", "BE KO KW NA NI PL FC")),
    ("Південний захід", "South West", P("NG", "EK LA OG ON OS OY")),
    ("Південний схід", "South East", P("NG", "AB AN EB EN IM")),
    ("Південь (дельта)", "South South", P("NG", "AK BY CR DE ED RI")),
])
country("TN", "Tunisia: north / centre (Sahel) / south.", [
    ("Північ", "North", P("TN", "11 12 13 14 21 22 23 31 32 33 34")),
    ("Центр і Сахель", "Centre and Sahel", P("TN", "41 42 43 51 52 53 61")),
    ("Південь", "South", P("TN", "71 72 73 81 82 83")),
])
country("BW", "Botswana's districts with their towns.", [
    ("Південний схід (Габороне)", "South-East", P("BW", "SE GA KL KW LO JW SO")),
    ("Центр і північний схід", "Central and North-East", P("BW", "CE NE FR SP ST")),
    ("Калахарі і Окаванго", "West and North", P("BW", "NW GH KG")),
])
country("MT", "Malta and Gozo (with Comino).", [
    ("Гозо", "Gozo", P("MT", "10 13 14 16 22 36 37 42 45 50 52 61 62 65 66")),
    ("Мальта", "Malta", None),
])


# ----------------------------------------------------------------------------------------------- hand-curated (no
# grouping attribute in Natural Earth)
country("ME", "Statistical regions of Montenegro (Monstat: Coastal, Central, Northern) over the 21 municipalities of "
        "the raster.", [
    ("Узбережжя", "Coastal", P("ME", "02 05 08 10 19 20")),
    ("Центр (Подгориця)", "Central", P("ME", "06 07 12 16")),
    ("Північ", "Northern", P("ME", "01 03 04 09 11 13 14 15 17 18 21")),
])
country("DO", "Development regions of the Dominican Republic (Decree 710-04) grouped by macro-region: Cibao (Norte, "
        "Sur, Nordeste, Noroeste); the south-west (Valdesia, El Valle, Enriquillo); Ozama (Santo Domingo with the "
        "Distrito Nacional) apart from the east (Higuamo, Yuma).", [
    ("Сібао (північ)", "Cibao", P("DO", "25 18 09 13 28 24 06 14 20 19 27 15 05 26")),
    ("Південний захід", "Southwest", P("DO", "17 21 31 02 22 07 04 03 10 16")),
    ("Санто-Домінго", "Santo Domingo (Ozama)", P("DO", "01 32")),
    ("Схід", "East", P("DO", "23 30 29 12 11 08")),
])
country("VE", "Administrative regions of Venezuela (CORDIPLAN regionalisation: Capital, Central, Centro-Occidental, Zulia, Andes, "
        "Llanos, Nor-Oriental, Insular, Guayana).", [
    ("Каракас і столичний", "Capital", P("VE", "A M X")),
    ("Центр", "Central", P("VE", "D G H")),
    ("Центр-Захід", "Centro-Occidental", P("VE", "I K P U")),
    ("Сулія", "Zulia", P("VE", "V")),
    ("Анди", "Andes", P("VE", "E L S T")),
    ("Льянос", "Llanos", P("VE", "C J")),
    ("Північний схід", "Nor-Oriental", P("VE", "B N R")),
    ("Острови", "Insular", P("VE", "O W VEN99")),
    ("Гвіана", "Guayana", P("VE", "F Z Y")),
], note="VE-VEN99 is Isla de Aves (Federal Dependencies).")

# ----------------------------------------------------------------------------------------------- from the Natural
# Earth admin-1 "region" attribute
ne_country("SI", "The 12 statistical regions of Slovenia (Natural Earth admin-1 'region', pre-2015 names: "
           "Spodnjeposavska = Posavska, Notranjsko-kraska = Primorsko-notranjska) over the municipalities.", [
    ("Любляна і центр", "Central Slovenia", ["Osrednjeslovenska"]),
    ("Гореньська", "Upper Carniola", ["Gorenjska"]),
    ("Горишка", "Gorizia", ["Goriška"]),
    ("Узбережжя і Карст", "Coastal-Karst", ["Obalno-kraška"]),
    ("Нотраньська", "Inner Carniola", ["Notranjsko-kraška"]),
    ("Південно-Східна Словенія", "Southeast Slovenia", ["Jugovzhodna Slovenija"]),
    ("Посавська", "Lower Sava", ["Spodnjeposavska"]),
    ("Засавська", "Central Sava", ["Zasavska"]),
    ("Савинська (Целє)", "Savinja", ["Savinjska"]),
    ("Корошка", "Carinthia", ["Koroška"]),
    ("Подравська (Марибор)", "Drava", ["Podravska"]),
    ("Помур'я", "Mura", ["Pomurska"]),
])
ne_country("LV", "The five planning regions of Latvia (Natural Earth admin-1 'region') over the 2009-2021 "
           "municipalities and republican cities.", [
    ("Рига і околиці", "Riga region", ["Riga"]),
    ("Відземе", "Vidzeme", ["Vidzeme"]),
    ("Курземе", "Kurzeme", ["Kurzeme"]),
    ("Земгале", "Zemgale", ["Zemgale"]),
    ("Латгале", "Latgale", ["Latgale"]),
])
ne_country("MK", "The eight statistical regions of North Macedonia (Natural Earth admin-1 'region'; 'Greater Skopje' "
           "and 'Skopje' together are the Skopje region; units without the attribute assigned from the State "
           "Statistical Office list).", [
    ("Скоп'є", "Skopje", ["Skopje", "Greater Skopje"]),
    ("Полог", "Polog", ["Polog"]),
    ("Південний захід", "Southwestern", ["Southwestern"]),
    ("Пелагонія", "Pelagonia", ["Pelagonia"]),
    ("Вардар", "Vardar", ["Vardar"]),
    ("Південний схід", "Southeastern", ["Southeastern"]),
    ("Схід", "Eastern", ["Eastern"]),
    ("Північний схід", "Northeastern", ["Northeastern"]),
], fix={"MK-73": "Southeastern", "MK-18": "Southeastern", "MK-07": "Southeastern", "MK-35": "Polog",
        "MK-75": "Polog", "MK-76": "Polog", "MK-16": "Polog", "MK-71": "Northeastern", "MK-57": "Southwestern",
        "MK-28": "Southwestern", "MK-61": "Southwestern", "MK-53": "Pelagonia", "MK-67": "Vardar",
        "MK-85": "Skopje", "MK-33": "Eastern"})
ne_country("UG", "The four regions of Uganda (Natural Earth admin-1 'region') over the districts.", [
    ("Центр (Кампала)", "Central", ["Central"]),
    ("Схід", "Eastern", ["Eastern"]),
    ("Північ", "Northern", ["Northern"]),
    ("Захід", "Western", ["Western"]),
])
ne_country("BT", "Western / Central / Eastern / Southern Bhutan (Natural Earth admin-1 'region') over the 20 "
           "dzongkhags.", [
    ("Захід (Тхімпху)", "Western", ["Western"]),
    ("Центр", "Central", ["Central"]),
    ("Схід", "Eastern", ["Eastern"]),
    ("Південь", "Southern", ["Southern"]),
])
ne_country("HK", "Hong Kong Island, Kowloon and the New Territories (Natural Earth admin-1 'region') over the 18 "
           "districts.", [
    ("Острів Гонконг", "Hong Kong Island", ["Hong Kong Island"]),
    ("Коулун", "Kowloon", ["Kowloon"]),
    ("Нові Території", "New Territories", ["The New Territories"]),
])
ne_country("XK", "The seven districts of Kosovo (Natural Earth admin-1 'region') over the municipalities.", [
    ("Приштина", "Pristina", ["Pristina"]),
    ("Митровиця", "Mitrovica", ["Kosovska Mitrovica"]),
    ("Печ (Пея)", "Peja (Pec)", ["Peć"]),
    ("Джяковиця (Гякова)", "Gjakova (Dakovica)", ["Đakovica"]),
    ("Призрен", "Prizren", ["Prizren"]),
    ("Урошевац (Феризай)", "Ferizaj (Urosevac)", ["Uroševac"]),
    ("Гнилане (Гїлан)", "Gjilan (Gnjilane)", ["Gnjilane"]),
])
ne_country("CN", "The traditional six regions of China (Natural Earth admin-1 'region'; Fujian with East China, the "
           "Paracel Islands with South Central China as administered from Hainan).", [
    ("Північний Китай", "North China", ["North China"]),
    ("Північний схід", "Northeast China", ["Northeast China"]),
    ("Східний Китай", "East China", ["East China"]),
    ("Центр і південь", "South Central China", ["South Central China"]),
    ("Південний захід", "Southwest China", ["Southwest China"]),
    ("Північний захід", "Northwest China", ["Northwest China"]),
], fix={"CN-FJ": "East China", "CN-PFA00": "South Central China"})


def ne_features(path):
    """[(raster code, properties)] of the Natural Earth admin-1 features (codes derived as
    tools/make_region_raster.py derives them)."""
    fc = json.load(open(path, encoding="utf-8"))
    out = []
    for f in fc["features"]:
        p = f["properties"]
        if not f.get("geometry"):
            continue
        cc = (p.get("iso_a2") or "").upper()
        if not re.match(r"^[A-Z]{2}$", cc) and p.get("latitude") is not None:
            cc = country_at(p["latitude"], p["longitude"]) or cc
        code = (p.get("iso_3166_2") or "").upper()
        if not re.match(r"^[A-Z]{2}-[A-Z0-9]{1,3}$", code):
            code = "%s-%s" % (cc, re.sub(r"[^A-Z0-9]", "", str(p.get("adm1_code") or "").upper())[-6:])
        out.append((code.upper(), p))
    return out


def build_ne_areas(feats):
    """country() entries of NE_AREAS from the features' "region" attribute (several features of one raster unit:
    the attribute of the largest)."""
    attr = {}
    for code, p in feats:
        a = p.get("region")
        size = float(p.get("area_sqkm") or 0.0)
        if a and (code not in attr or size > attr[code][1]):
            attr[code] = (a, size)
    ok = True
    for cc, (source, areas, fix) in NE_AREAS.items():
        units = [CODES[i] for i in range(1, len(CODES)) if RCC[i] == cc]
        val = {u: fix.get(u) or (attr.get(u) or (None,))[0] for u in units}
        groups = []
        for uk, en, values in areas:
            groups.append((uk, en, sorted(u for u in units if val[u] in values)))
        lost = sorted(u for u in units if not any(val[u] in v for _, _, v in areas))
        if lost:
            print("NE %s: units without a known region: %s" % (cc, [(u, val[u], NAMES[CODES.index(u)]) for u in lost]))
            ok = False
        country(cc, source, groups)
    return ok


def unit_points(feats):
    """{raster code: [lat, lng]} label points of the raster units without a pixel (too small for the 0.05 deg
    grid), from the largest feature of the unit."""
    from engine.regions import region_centroids
    km2 = region_centroids()[2]
    pos = {c: i for i, c in enumerate(CODES)}
    best = {}
    for code, p in feats:
        i = pos.get(code)
        if not i or km2[i] > 0 or p.get("latitude") is None:
            continue
        size = float(p.get("area_sqkm") or 0.0)
        if code not in best or size > best[code][0]:
            best[code] = (size, [round(float(p["latitude"]), 4), round(float(p["longitude"]), 4)])
    return OrderedDict((c, best[c][1]) for c in sorted(best))


# raster names (Natural Earth name_en, from Wikidata) that name another place: the feature's own "name" instead
NAME_FROM_NE = ("XK-OS5902", "XK-OS5907", "XK-OS5908")


def main(ne_path=NE_DEFAULT):
    feats = ne_features(ne_path)
    bad = not build_ne_areas(feats)
    out = OrderedDict()
    out["_doc"] = ("GeoGuessr-style areas that group the admin-1 units of the region raster (data/world_regions.npz, "
                   "Natural Earth admin-1) per country, as players name them (US census divisions, Brazilian "
                   "macro-regions, Russian federal districts, ...).  Written by tools/make_areas.py (edit the "
                   "groupings there, not here).  countries.<CC>: {source, areas: [{id, name (Ukrainian), en, "
                   "regions: [ISO 3166-2 codes of the raster]}], hidden: [codes]}; every raster unit of a listed "
                   "country is in exactly one area or in its hidden list (units the raster files under the country "
                   "that do not belong to it: shown in no area and no region list of it), and at least half of a "
                   "country's areas group two or more units (checked by test_regions_areas.py).  Countries not "
                   "listed: engine/areas.py groups their units automatically into compass-named areas when they have "
                   "at least 17 (AUTO_MIN_UNITS) units, otherwise they have no areas (the HUD lists their regions).  "
                   "region_names: display names replacing the raster's.  unit_points: label points (lat, lng) of the "
                   "raster units without a pixel (engine/areas.py puts them into the nearest automatic area).")
    out["countries"] = OrderedDict()
    own = {}
    for i in range(1, len(CODES)):
        own.setdefault(RCC[i], []).append(CODES[i])
    for cc, areas in A.items():
        units = set(own.get(cc, []))
        hidden = set(HIDDEN.get(cc, []))
        # the Malta rest area: every unit not in Gozo
        for a in areas:
            if a["regions"] is None:
                rest = units - hidden - {r for b in areas if b["regions"] for r in b["regions"]}
                a["regions"] = sorted(rest)
        seen = {}
        for a in areas:
            for r in a["regions"]:
                if r in seen or r in hidden:
                    print("DUP", cc, r, seen.get(r, "hidden"), a["en"])
                    bad = True
                seen[r] = a["en"]
        missing = units - set(seen) - hidden
        extra = set(seen) - units
        if missing:
            print("MISSING", cc, sorted(missing), [NAMES[CODES.index(m)] for m in sorted(missing)])
            bad = True
        if extra:
            print("NOT IN RASTER", cc, sorted(extra))
            for a in areas:
                a["regions"] = [r for r in a["regions"] if r in units]
        if hidden - units:
            print("HIDDEN NOT IN RASTER", cc, sorted(hidden - units))
            bad = True
        areas = [a for a in areas if a["regions"]]
        multi = sum(1 for a in areas if len(a["regions"]) > 1)
        if 2 * multi < len(areas):
            print("ONE-TO-ONE", cc, "only %d of %d areas group several units" % (multi, len(areas)))
            bad = True
        entry = OrderedDict([("source", SRC[cc]), ("areas", areas)])
        if hidden:
            entry["hidden"] = sorted(hidden)
        out["countries"][cc] = entry
    names = OrderedDict([
        ("US-DC", "District of Columbia"), ("BR-DF", "Distrito Federal"), ("RU-MOS", "Moscow Oblast"),
        ("RU-MOW", "Moscow"), ("RU-YEV", "Jewish Autonomous Oblast"), ("RU-ALT", "Altai Krai"),
        ("RU-RUS99", "Russia (other)"), ("MX-DIF", "Mexico City"), ("MX-MEX99", "Mexico (other)"),
        ("AR-C", "Ciudad de Buenos Aires"), ("CO-CUN", "Cundinamarca"), ("CO-COL99", "Colombia (other)"),
        ("IN-KAS00", "Kashmir (disputed)"), ("KH-5", "Kampong Speu"), ("KH-12", "Phnom Penh"),
        ("UA-30", "Kyiv City"), ("UA-32", "Kyiv Oblast"), ("HR-21", "City of Zagreb"), ("HR-01", "Zagreb County"),
        ("TW-HSZ", "Hsinchu City"), ("TW-HSQ", "Hsinchu County"), ("TW-CYI", "Chiayi City"),
        ("TW-CYQ", "Chiayi County"), ("IT-GE", "Genoa"), ("IT-BZ", "South Tyrol"), ("IT-TN", "Trentino"),
        ("GB-WRL", "Wirral"), ("GB-SHN", "St Helens"), ("GB-LND", "City of London"), ("PH-MNL", "Metro Manila"),
        ("PH-ZMB", "Zambales"), ("PH-LAN", "Lanao del Norte"), ("BG-22", "Sofia City"), ("BG-23", "Sofia Province"),
        ("VN-39", "Đồng Nai"), ("VN-53", "Bắc Kạn"), ("VN-66", "Hưng Yên"), ("CZ-PL", "Plzeň"),
        ("HU-VM", "Veszprém (city)"), ("KE-110", "Nairobi"), ("PE-CUS", "Cusco"), ("NP-KO", "Kosi Zone"),
        ("AE-ARE344", "Fujairah (other)"), ("AE-ARE347", "Sayh Mudayrah"), ("DK-84", "Capital Region"),
        ("VE-VEN99", "Isla de Aves"), ("RU-AL", "Altai Republic"), ("UZ-TK", "Tashkent (city)"),
        ("UZ-TO", "Tashkent Region"), ("AZ-LA", "Lankaran (city)"), ("BY-MI", "Minsk Region"),
    ])
    names["HK-KG5155"] = "Eastern District"   # the raster's name_en is a hill in it (Siu Ma Shan)
    ne_feat = {}
    for code, p in feats:
        ne_feat.setdefault(code, p)
    for code in NAME_FROM_NE:
        names[code] = ne_feat[code]["name"]
    # units of one country that share a raster name (Wikidata name_en of a later merger, e.g. 21 Latvian
    # municipalities all called "Valmiera"): the feature's own Natural Earth name
    dup = collections.Counter((RCC[i], NAMES[i]) for i in range(1, len(CODES)))
    for i in range(1, len(CODES)):
        c = CODES[i]
        p = ne_feat.get(c) or {}
        if dup[(RCC[i], NAMES[i])] < 2 or c in names or not p.get("name") or p["name"] == NAMES[i]:
            continue
        names[c] = p["name"] + (" novads" if RCC[i] == "LV" and p.get("type_en") == "Municipality" else "")
    out["region_names"] = names
    out["unit_points"] = unit_points(feats)
    if bad:
        print("problems above: data/areas.json not written")
        sys.exit(1)
    path = os.path.join(ROOT, "data", "areas.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("countries", len(out["countries"]), "areas", sum(len(v["areas"]) for v in out["countries"].values()),
          "unit points", len(out["unit_points"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else NE_DEFAULT)
