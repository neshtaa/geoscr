"""
Baseline colour statistics (used only to exercise the model pipeline):
Lab mean/std per elevation band, rotation invariant.
"""
import numpy as np

NAME = "baseline"
BANDS = [(60, 20), (20, 5), (5, -5), (-5, -20), (-20, -45)]
FEATURE_NAMES = [f"b{i}_{s}_{c}" for i in range(len(BANDS)) for s in ("mean", "std") for c in "Lab"]


def rgb_to_lab(rgb):
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def extract(sph):
    s = sph.resized(512)
    lab = rgb_to_lab(s.float_rgb())
    x = []
    for top, bot in BANDS:
        sl = s.band(top, bot)
        m = s.mask[sl]
        v = lab[sl][m]
        if len(v) < 50:
            x += [np.nan] * 6
        else:
            x += list(v.mean(0)) + list(v.std(0))
    return {"x": np.array(x, np.float32), "evidence": {}}
