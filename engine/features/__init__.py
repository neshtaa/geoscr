"""
Feature extractor registry.

Every module in this package exposes:
  NAME           short identifier, e.g. "road"
  FEATURE_NAMES  list of names for the numeric vector
  extract(sph)   -> {"x": np.ndarray(len(FEATURE_NAMES)) (NaN = not measurable),
                     "evidence": {...human readable detections for hints...}}
where ``sph`` is an engine.panorama.SphericalImage.  Extractors use only
closed-form image mathematics (colour spaces, gradients, projections, geometry).
"""

import importlib

MODULES = ["solar", "road", "landscape", "vehicle", "structure", "texture"]


def load(name):
    return importlib.import_module("engine.features." + name)


def available():
    out = []
    for n in MODULES:
        try:
            out.append(load(n))
        except ModuleNotFoundError:
            pass
    return out
