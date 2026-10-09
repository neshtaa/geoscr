# Handoff: GeoGuessr pure-math locator

## Task (from the user, Ukrainian)
Build a script that recognises the location the way GeoGuessr's own analysis does: fully
mathematical, no AI functions (no neural networks, pretrained models, LLM/VLM calls or ML
libraries), using the two scraped knowledge bases (GeoGuessr clue catalogue, Plonk It), giving
hints right away. Runtime: numpy + Pillow only, Python 3.8 compatible.

## State (branch `claude/sharp-keller-ztbilq`)
TEST = 386 of the user's real World-map rounds (half of the games by hash; never used for fitting or
calibration). Live mode = panorama rendered into the live grid of 10 views (car axis unknown), full
locator incl. cards + region model (`tools/train_model.py --views --eval-only`, `tools/eval_live.py`).

| | top1 | top3 | top5 | pts/round (World formula) |
|---|---|---|---|---|
| prior only | 4.3% | 10% | 16% | 209 |
| first working version (17k random Street View panoramas) | 17.9% | 34.0% | 46.0% | 1617 |
| live mode, 1 panorama (No Move) | 31.9% | 56.2% | 63.0% | 2276 |
| live mode, 1 panorama + map info | 34.5% | 56.0% | 63.7% | 2279 |
| Moving, 4 panoramas fused (`engine/fusion.py`, 366 rounds) | 44.0% | 62.6% | 72.1% | 2661 |

Moving: 1 -> 4 panoramas 2341 -> 2661 (+320 [+159, +484]) on simulated walks along Street View
links (`tools/moving_calib.py`, rho fitted on CALIB). Region given the right country: top3 ~54%.
Real games (one panorama per round, --submit): daily 2026-10-04 17,664 (26.7k model).

Data: 78k training panoramas = 7k random Street View (world/balanced) + 71k panoramas of public ranked
duels (`tools/crawl_duels.py`, 260 player histories, GET only; `tools/stream_duels.py` keeps only the
features of 51k of them, records marked pano_deleted; own rounds within 1 km excluded in load_records).
GeoGuessr clue placements: 33k placements on 8.9k panoramas (`data/calibration/pano_clues.json`).
The "views" parameter set (calibrated on live-grid CALIB features) is disabled (`enabled: false` in
model.json): 2-fold CV inside CALIB showed no gain.

## GeoGuessr's own analysis
Not image recognition: the client calls `GET /api/v4/clues/{panoId}` and the server returns
curated cards for that panorama (country, `seterraRegionIds` = ISO 3166-2 / AREA_* ids,
heading/pitch/zoom of the view showing the clue). Offline labels only, never in live play:
`tools/fetch_pano_clues.py` (history rounds; `--duels --n N [--months 2026-08,...]` finished
public duels; `--translations` card texts; `--seterra` unmapped / partially matched region ids)
-> `data/calibration/pano_clues.json`; after `calibrate.py features` and `train_model.py --save`,
`python3 tools/build_clue_index.py [--baseline-rev 0c406f7]` -> `data/model/clue_index.npz`
(index = duel + CALIB rounds, TEST held out). It ranks the hint cards by card frequency in the
country + kNN of similar annotated panoramas + region vote; keyword + region ranking without the
index or for countries without annotations. TEST (127 rounds, true country): precision@3 0.404,
recall@3 0.457 vs 0.131 / 0.148 for 0c406f7 (+0.27 [0.22, 0.33]); the gain is the country-level
card frequency, the kNN / region parts are within noise of it (frequency only 0.396 / 0.449).

## What was fixed / learned
- `solar`: the hand-tuned disk score accepted bright clouds (hemisphere accuracy 67%, worse than
  the 77% base rate). Replaced by a logistic detector (`tools/calibrate_sun.py`) labelled by
  astronomy: a candidate is "real" if within 3 deg of a solar position possible at the pano's
  latitude and capture month. Hemisphere accuracy of accepted detections -> 92%.
  `train_model.py` previously looked for a non-existent `sun_az_true` feature, so the sun term
  never ran; it now lives in `GeoModel.sun_loglik`.
- `road`: the structure-tensor axis estimate always returned ~0 deg (IPM sampling artefacts
  dominate). New `estimate_axis` maximises the lateral correlation ratio over candidate axes
  (median error 3.5 deg). Driving side needs the car's forward direction, so `side_*` and
  `right_hand_traffic` are NaN when the car axis is unknown (live play). Driving side accuracy
  with known axis ~72% balanced (centre-line cue 83%).
- `calibrate.py` processes half of the panoramas with `car_heading=None` (`hide_car_axis`) so
  the model sees both regimes; `load_features` re-read the npz per row (minutes) - fixed.
- `texture` (new): Lab joint histograms, uniform LBP, gradient orientations per band; the
  strongest single group.
- Model: per-group Gaussians + kNN + multinomial logit (GLM) + sun term; exponents and prior
  mix calibrated on CALIB. More data helped clearly (half -> full train: calib top5 37% -> 45%).
- Regions: within-country kernel over reference panoramas; tried Fisher / landscape / texture
  spaces and sun-latitude weighting on CALIB - all ~8% top1 / ~23% top3 given the true country
  (barely above reference density), so regions are limited by ~150 references per country.
  Location and region kernels are calibrated separately (`calibrate_location`).
- Street View download: `_parse_pano` now returns None for removed panoramas so history rounds
  fall back to a 100 m search.

## Clue-card detectors (`engine/clue_detect.py`, `tools/build_clue_detectors.py`)
Classical sliding-window detectors trained on GeoGuessr's own placements (card X visible in
panorama P at heading/pitch/zoom): per zoom channel (1.5 / 2 / 3.5) HOG-style cells (9 orientations,
gradient energy, mean Lab), 6x6-cell windows = one placement view; one whitened-LDA template per
card with >= 3 duel-round placements and >= 1 in each cross-fitting fold, rarer cards share a
(country, type) template; 2024 detectors, 4.7 MB, ~0.2 s per sphere. Only public duel rounds train
detectors, presence / direction / country models (out-of-fold scores); CALIB sets the thresholds and
the weight; TEST is only scored.
- Detectors are weak: present-vs-absent AUC 0.64 (train OOF) / 0.63 (TEST); the best window lies
  within half a window of a placement for 15% (train) / 21% (TEST) of present cards.
- Country evidence "cards" (weight 0.05 on CALIB, 2-fold CV loglik gain on CALIB +0.037
  [+0.019, +0.054]). TEST, with - without cards (paired bootstrap): map info loglik +0.008
  [-0.014, +0.030], top1 30.1 vs 29.5%, top3 55.4 vs 54.1%, points -9 [-89, +72]; no map loglik
  +0.019 [-0.003, +0.042], top1 29.5 vs 27.7%, top3 54.1 vs 52.1% (+2.1 [+0.5, +3.9]), points +92
  [+11, +176] (not seen on CALIB: -8 [-67, +51], nor with map info). Live grid (eval_live
  rendering, no map, paired): CALIB top3 +2.7 [+1.0, +4.6], top5 +2.9 [+1.0, +5.0], points +13
  [-31, +62]; TEST top1 33.4 vs 31.1% (+2.3 [+0.5, +4.2]), top3 +0.8 [-1.6, +3.1], points +0
  [-84, +83]; +0.17 s per round. In short: a small log-likelihood gain, country accuracy +1-3
  points in some conditions, points within noise.
  Skipped when < 90% of the searched windows are visible (one frame: maxima drop and would read
  as "no cards").
- Hints: "detected cards first" and the detector's direction are both off (thresholds 1.01): the
  order did not beat the clue-index ranking on CALIB (bootstrap lower bound <= 0) and no threshold on
  P(direction right) gave >= 50% correct directions (best 14-17% on CALIB). The code paths stay
  (`hints[i].geoguessr[j].detected` = {heading, pitch, score, probability = P(direction right |
  country), true_north}) and switch on by themselves if a better bank passes the CALIB bars.
- Region update by regional cards: any positive exponent lowered CALIB log P(true region) -> 0.
- Rebuild order (after `train_model.py --save` / `build_clue_index.py`): `build_clue_detectors.py
  cells` (freezes the labels in scratch/cards/clues_used.json), `fit`, `score`, `calibrate`, then
  `train_model.py --calibrate-cards --save`; rerun the last one after every `train_model.py --save`
  (it needs scratch/cards/scores.npz; without it the card evidence is left out with a warning).
  Cache scratch/cards/ ~1.4 GB, of which cells/ (1.2 GB) is only needed for fit / score.

## Rebuild
See README "Навчання й калібрування". Dataset in `scratch/` (git-ignored), ~17k panoramas (~7 GB).
Feature extraction ~9 pano/s on 4 cores; `train_model.py` ~2 min.

## Ideas not done
- Script recognition spike (`engine/script_detect.py`, `tools/script_spike.py`): classical text
  detection + glyph statistics on GeoGuessr language / sign placements at 0.03 deg/px; non-Latin
  recall 130/631, CALIB gain indistinguishable from 0 -> dropped (not integrated).
- Text/sign/plate recognition is out of scope for closed-form maths. Bollards, poles etc. have
  classical template detectors now (above), but they localise the card only ~1 time in 5; the
  limit looks like 2048-px panoramas + one template per card (multi-component / part models and
  higher-resolution tiles around candidate windows are the untried next steps).
- Live script: tested on real single-player challenges (`--challenge <t> --submit`; see State).
  Fixed on the real site: single in-game frame (NMPZ) -> road axis estimator gave up below 40%
  azimuth coverage (now 15%); Street View navigation arrows were drawn on the road in the captures
  (now `linksControl` off during capture, restored after). Earlier capture checks: `--replay`
  (`tools/rebuild_views.py` registration: measured FOV exact, s = 1.000). The FOV comes from
  Google's projection matrix or a two-frame registration (tan(hfov/2) = 2^(1-zoom), vertical FOV
  capped at 90; the old `180 / 2^zoom` was right only at zoom 1). Not yet exercised on a live
  round: round detection, `--submit` and the ticket prompt on `/game` pages.
  `tools/eval_live.py` now renders the live grid (2 x 5 views, 112.7 x 90 deg); the "12 rendered
  views" row above is from the old grid (a paired run of both grids differed by +30 +- 44 pts).
