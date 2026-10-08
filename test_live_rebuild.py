#!/usr/bin/env python3
"""
Tests for tools/rebuild_views.py (FOV registration of live captures):
  python3 -m unittest test_live_rebuild -v
"""
import json
import os
import sys
import tempfile
import unittest

from PIL import Image

from engine.panorama import SphericalImage

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rebuild_views  # noqa: E402

PANO = os.path.join(ROOT, "test_japan_pano.jpg")


def _sphere():
    img = Image.open(PANO).convert("RGB").resize((2048, 1024))
    return SphericalImage.from_equirect(img, heading=0.0, width=2048)


class TestRebuildViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sph = _sphere()

    def test_probe_recovers_hfov(self):
        true = 106.26
        a = self.sph.render_view(10.0, 0.0, true, (700, 466))
        b = self.sph.render_view(35.0, 0.0, true, (700, 466))
        got = rebuild_views.probe_fov(a, (10.0, 0.0), b, (35.0, 0.0))
        self.assertAlmostEqual(got["hfov"], true, delta=1.0)

    def test_probe_with_pitch(self):
        true = 90.0
        a = self.sph.render_view(-20.0, 0.0, true, (600, 400))
        b = self.sph.render_view(-20.0, 25.0, true, (600, 400))
        got = rebuild_views.probe_fov(a, (-20.0, 0.0), b, (-20.0, 25.0))
        self.assertAlmostEqual(got["hfov"], true, delta=1.0)

    def test_registration_finds_wrong_fov(self):
        true, recorded = 118.0, 118.0 / 1.06
        views, meta = [], []
        with tempfile.TemporaryDirectory() as d:
            k = 0
            for pitch in (-42.0, 0.0, 42.0):
                for yaw in (0.0, 90.0, 180.0, 270.0):
                    name = "view_%02d.jpg" % k
                    self.sph.render_view(yaw, pitch, true, (480, 300)).save(os.path.join(d, name), quality=92)
                    meta.append({"file": name, "yaw": yaw, "pitch": pitch, "hfov": recorded, "vfov": None})
                    k += 1
            with open(os.path.join(d, "views.json"), "w") as f:
                json.dump({"views": meta}, f)
            _, views = rebuild_views.load_capture(d)
        best, rows = rebuild_views.register_scale(views, 0.97, 1.13, 0.02, width=512)
        self.assertAlmostEqual(best, 1.06, delta=0.015)


if __name__ == "__main__":
    unittest.main()
