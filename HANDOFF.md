# Handoff: GeoGuessr pure-math locator

## Task (from the user, Ukrainian)
Build a script that recognises the location the way GeoGuessr's own analysis does: fully
mathematical, no AI functions (no neural networks, pretrained models, LLM/VLM calls or ML
libraries), using the two scraped knowledge bases (GeoGuessr clue catalogue, Plonk It), giving
hints right away. Runtime: numpy + Pillow only, Python 3.8 compatible.

## State (branch `claude/sharp-keller-ztbilq`)
Working end to end. TEST = 530 of the user's real World-map rounds (half of the games, never used
for fitting or calibration):

| | top1 | top3 | top5 | pts/round |
|---|---|---|---|---|
| prior only | 4.3% | 10% | 16% | 209 |
| old `deterministic_locator` (removed) | 3.4% | 6.6% | - | 450 |
| new, full panorama | 18.9% | 35.3% | 44.7% | 1653 |
| new, 12 rendered views, car axis unknown (`tools/eval_live.py`) | 17.9% | 34.0% | 46.0% | 1617 |

Model trained on 17.1k panoramas (world 2.3k, balanced quota 150 per country, 118 classes).
The first version on 10k panoramas gave 1515 pts/round; more data is still the clearest lever.

Pipeline: `engine/locator.py` (SphericalImage -> 6 feature modules -> `engine/model.py` ->
score-optimal guess -> `engine/hints.py`). Entry points: `geoguessr_locator.py` (CLI),
`web/server.py` (+ `web/index.html`), `play_live_visual.js` (live HUD, `setPov` only).
Gemini (`vlm_engine`) and all old heuristic engines are deleted.

Regions (admin-1) when the country is right: top1 14.7%, top3 31.6% (live mode); the same top
country from two captures with different start yaw: 94.2%.

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

## Rebuild
See README "Навчання й калібрування". Dataset in `scratch/` (git-ignored), ~17k panoramas (~7 GB).
Feature extraction ~9 pano/s on 4 cores; `train_model.py` ~2 min.

## Ideas not done
- Text/sign/plate recognition is out of scope for closed-form maths; bollards and pole types
  would need dedicated detectors.
- Live script: capture checked on the real site only through `--replay` of a finished game
  (`tools/rebuild_views.py` registration: measured FOV exact, s = 1.000). The FOV comes from
  Google's projection matrix or a two-frame registration (tan(hfov/2) = 2^(1-zoom), vertical FOV
  capped at 90; the old `180 / 2^zoom` was right only at zoom 1). Not yet exercised on a live
  round: round detection, `--submit` and the ticket prompt on `/game` pages.
  `tools/eval_live.py` now renders the live grid (2 x 5 views, 112.7 x 90 deg); the "12 rendered
  views" row above is from the old grid (a paired run of both grids differed by +30 +- 44 pts).
