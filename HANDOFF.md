# Handoff: GeoGuessr pure-math locator (work in progress)

## Task (from the user, Ukrainian)
"There was a task to build a script that recognises the location the way GeoGuessr's own
analysis does. The script must be fully mathematical and must not use AI functions. Two
knowledge bases were scraped: GeoGuessr's own clue base and Plonk It. The script is very
bad at this stage; bring it to a working state. You may calibrate it in GeoGuessr games on
the World map. It should end up roughly like GeoGuessr's own recognition, which gives hints
right away."

Constraints: no neural networks, pretrained models, LLM/VLM calls or ML libraries. Only
closed-form image maths (colour spaces, gradients, projections, geometry, astronomy) plus
classical statistics calibrated on data. The runtime depends on numpy + Pillow only. Keep the
code Python 3.8 compatible, because the user's machine runs 3.8. The old Gemini path
(`engine/vlm_engine.py`) violates the constraint and must be removed from all entry points.

## What exists now (commit on branch `claude/sharp-keller-ztbilq`)
- `engine/streetview.py`: key-less Street View client. `search_pano(lat, lng, radius)` and
  `get_metadata(pano_id)` use the internal Maps RPC
  (`maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/...`, JSON+protobuf).
  `download_panorama(meta)` stitches tiles from `streetviewpixels-pa.googleapis.com` with
  `cb_client=apiv3`. Only official coverage (image type 2) is requested. Metadata gives country
  code, car heading (= centre column azimuth), capture date and the tile pyramid.
- `engine/geo.py`: lat/lng -> country lookup from the `data/world_countries.png/json` raster
  (Natural Earth 10m map units at 0.05 deg, built by `tools/make_country_raster.py`), haversine,
  and the GeoGuessr World score `5000*exp(-d/1491.6862 km)`.
- `engine/panorama.py`: `SphericalImage`, an equirect canvas + validity mask + true heading.
  `from_equirect` (panos), `from_views` (back-projects in-game screenshots, pinhole model;
  12 views = pitch -55/0/55 x 4 yaws at hfov 120 cover the full sphere, round-trip error is about
  1 grey level), `from_screenshot` (single image, heading unknown) and `render_view`.
- `engine/features/`: feature-module registry (`MODULES = solar, road, landscape, vehicle,
  structure`). The interface is `NAME`, `FEATURE_NAMES`, `extract(sph) -> {"x", "evidence"}`.
  The five modules were written by parallel agents that were STOPPED MID-WORK (the user asked to
  stop everything). They are syntactically valid but UNREVIEWED and their metrics are unknown.
  Review each one adversarially: mask handling, `heading`/`car_heading` None, screenshot
  robustness, runtime <= 250 ms, and visual checks of detections. `baseline.py` is only a
  pipeline smoke test (Lab stats per band). It reached top1 0.092 / top5 0.281 vs prior
  0.058 / 0.178.
- `tools/build_dataset.py`: dataset builder with modes `world` (area-weighted random land points,
  like the World map), `balanced` (per-country quota), `history` (the user's real GeoGuessr
  rounds), `postmatch`. Panoramas go to `scratch/dataset/panos/<id>.jpg` (2048x1024, about
  420 KB each) and records to `scratch/dataset/index.jsonl`.
- `tools/calibrate.py`: feature cache (`features --modules X --workers N [--sample K]`,
  incremental, keyed by module source hash), `eval-group X` (standalone LDA top1/top5 and
  info gain vs prior, ANOVA ranking) and `stats`. Splits: train = world+balanced; calib/test =
  halves (by game) of the user's real World-map rounds; maps "Ukraine", "United Kingdom (Better
  Map)" and "MLB..." are excluded.
- `engine/model.py` + `tools/train_model.py`: per-group NaN-aware shared-covariance Gaussian
  class densities (shrinkage, empirical-Bayes means), kNN likelihood ratio in the Fisher
  embedding, analytic sun term via `solar.latitude_likelihood`, evidence weights + prior mix
  calibrated on the calib split, and location as a mixture over reference panoramas with the
  guess maximising the expected GeoGuessr score. NOT RUN YET; expect bugs.
- `data/calibration/history_rounds.json`: 1330 rounds from the user's games (duels and
  challenges; pano ids already hex-decoded, lat/lng, GeoGuessr country code). About 890 are on
  World-type maps.

## Rebuild the dataset (scratch/ is git-ignored; the local copy was not pushed)
In the cloud geoguessr.com is blocked; the Street View endpoints above work. Locally it ran at
about 600 panos/min.
```
python3 tools/build_dataset.py history --rounds-file data/calibration/history_rounds.json --threads 16
python3 tools/build_dataset.py world -n 12000 --radius 3000 --threads 24 --seed 1        # about 2300 panos
python3 tools/build_dataset.py balanced --quota 60 --radius 5000 --max-misses 80 --threads 24 --seed 2 \
  --countries AD,AE,AL,AR,AS,AT,AU,BA,BD,BE,BG,BM,BO,BR,BT,BW,BY,CA,CC,CH,CL,CN,CO,CR,CW,CX,CY,CZ,DE,DK,DO,EC,EE,EG,ES,FI,FK,FO,FR,GB,GE,GH,GI,GL,GR,GT,GU,HK,HR,HU,ID,IE,IL,IM,IN,IQ,IS,IT,JE,JO,JP,KE,KG,KH,KR,KZ,LA,LB,LI,LK,LS,LT,LU,LV,MC,ME,MG,MK,ML,MN,MO,MP,MQ,MT,MX,MY,NA,NG,NL,NO,NP,NZ,OM,PA,PE,PH,PK,PL,PM,PR,PT,PY,QA,RE,RO,RS,RU,RW,SE,SG,SI,SJ,SK,SM,SN,ST,SZ,TH,TN,TR,TW,TZ,UA,UG,US,UY,VI,VN,VU,XK,ZA
```
The full dataset is about 8.8k panos and 3.7 GB. Check `df -h` first. The user's local disk
filled up, so be frugal (consider `--quota 40`).

## Facts learned
- The user's GeoGuessr account is free: "No free games left" for new games, and creating
  challenges needs a paid plan. The daily challenge (World, 5 rounds, 180 s) is playable once
  per day. Live play is only possible on the user's machine (cookie in
  `data/session_cookie.txt`, git-ignored; geoguessr.com is reachable there).
- Duel API: `game-server.geoguessr.com/api/duels/{id}`. Rounds hold `panorama.panoId` as hex
  of the ASCII pano id, plus `lat`, `lng`, `countryCode` and the start `heading`.
- The World-sample distribution is very US/IN/BR-heavy, while the user's maps (A Moving World,
  The World) are more balanced. Tune the prior mix on calib.
- Live capture plan: in the page, hook `google.maps.StreetViewPanorama` via
  `evaluateOnNewDocument` and use `setPov` ONLY to point the camera (never read position). Hide
  overlays with CSS, take 12 canvas screenshots, read heading from the compass/POV, rebuild with
  `SphericalImage.from_views`, run the engine, show a HUD with hints and submit the guess.
  Calibrate the FOV by registration of overlapping views.

## Next steps
1. Rebuild the dataset; run `calibrate.py features` for all modules; review and fix each module;
   `eval-group` each.
2. `tools/train_model.py --save`; iterate on features, model and the sun term. Report TEST
   top1/top3/top5 and points/round. Compare with the old engine (`engine/offline_engine.py`)
   on rendered views as the "before" number.
3. Hints: map module `evidence` + top countries to GeoGuessr clue cards
   (`data/geoguessr_master_clues.json`, `data/geoguessr_postmatch_clues.json`) and Plonk It tips
   (`data/plonkit_kb.json`), GeoGuessr-style.
4. Wire into `geoguessr_locator.py` (`--image`, `--pano`, `--latlng` for tests), `web/server.py`
   `/api/predict` (views + headings) and `play_live_visual.js`. Remove `vlm_engine` and the old
   engines from the entry points. Update the README (Ukrainian) and tests.
5. Commit and push to `claude/sharp-keller-ztbilq`. Do not open a PR unless asked.
