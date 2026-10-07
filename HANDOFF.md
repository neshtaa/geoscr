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
| new, full panorama | 18.9% | 35.3% | 44.7% | 1643 |
| new, 12 rendered views, car axis unknown (`tools/eval_live.py`) | 17.9% | 34.0% | 46.0% | 1560 |

Model trained on 17.1k panoramas (world 2.3k, balanced quota 150 per country, 118 classes).
The first version on 10k panoramas gave 1515 pts/round; more data is still the clearest lever.

Pipeline: `engine/locator.py` (SphericalImage -> 6 feature modules -> `engine/model.py` ->
score-optimal guess -> `engine/hints.py`). Entry points: `geoguessr_locator.py` (CLI),
`web/server.py` (+ `web/index.html`), `play_live_visual.js` (live HUD, `setPov` only).
Gemini (`vlm_engine`) and all old heuristic engines are deleted.

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
- Street View download: `_parse_pano` now returns None for removed panoramas so history rounds
  fall back to a 100 m search.

## Rebuild
See README "Навчання й калібрування". Dataset in `scratch/` (git-ignored), ~17k panoramas (~7 GB).
Feature extraction ~9 pano/s on 4 cores; `train_model.py` ~2 min.

## Ideas not done
- Text/sign/plate recognition is out of scope for closed-form maths; bollards and pole types
  would need dedicated detectors.
- Live script is untested here (geoguessr.com is blocked in the cloud); the FOV formula
  `fov = 180 / 2^zoom` and the overlay hiding should be checked on the user's machine
  (rebuild one round with `SphericalImage.from_views` and look at it).
