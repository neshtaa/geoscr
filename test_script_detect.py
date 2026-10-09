#!/usr/bin/env python3
"""
Tests for the writing-detector prototype (engine/script_detect.py):
    python3 -m unittest test_script_detect -v
"""
import unittest
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine import script_detect as sd

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def flood(mask, nbrs):
    """Brute-force connected components (BFS) -> labels (-1 = background), count."""
    h, w = mask.shape
    lab = np.full((h, w), -1, int)
    n = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and lab[y, x] < 0:
                lab[y, x] = n
                q = deque([(y, x)])
                while q:
                    a, b = q.popleft()
                    for dy, dx in nbrs:
                        u, v = a + dy, b + dx
                        if 0 <= u < h and 0 <= v < w and mask[u, v] and lab[u, v] < 0:
                            lab[u, v] = n
                            q.append((u, v))
                n += 1
    return lab, n


def same_partition(a, b):
    """True when two label images define the same components (labels may be permuted)."""
    if not np.array_equal(a < 0, b < 0):
        return False
    fg = a >= 0
    pairs = set(zip(a[fg].tolist(), b[fg].tolist()))
    return len(pairs) == len(set(a[fg].tolist())) == len(set(b[fg].tolist()))


def brute_holes(comp):
    """Holes of one 8-connected component = 4-connected background regions not touching the border."""
    p = np.pad(comp, 1)
    lab, n = flood(~p, N4)
    return n - 1  # the outer background is a single region (the padding connects it)


class TestComponents(unittest.TestCase):
    def test_labels_match_flood_fill(self):
        rng = np.random.RandomState(0)
        for i in range(150):
            h, w = rng.randint(1, 40), rng.randint(1, 40)
            mask = rng.rand(h, w) < rng.uniform(0.1, 0.7)
            labels, n, _ = sd.label_components(mask)
            ref, nref = flood(mask, N8)
            self.assertEqual(n, nref)
            self.assertTrue(same_partition(labels, ref), "mask %d" % i)

    def test_holes_match_brute_force(self):
        rng = np.random.RandomState(1)
        for i in range(150):
            h, w = rng.randint(3, 30), rng.randint(3, 30)
            mask = rng.rand(h, w) < rng.uniform(0.3, 0.8)
            labels, n, runs = sd.label_components(mask)
            if n == 0:
                continue
            st = sd.component_stats(labels, n, runs)
            for c in range(n):
                self.assertEqual(int(st["holes"][c]), brute_holes(labels == c), "mask %d comp %d" % (i, c))
                self.assertEqual(int(st["area"][c]), int((labels == c).sum()))

    def test_ring_and_letters(self):
        m = np.zeros((12, 30), bool)
        m[2:9, 2:9] = True
        m[4:7, 4:7] = False      # 'O': one hole
        m[2:9, 12:20] = True
        m[3:5, 14:18] = False
        m[6:8, 14:18] = False    # 'B'-like: two holes
        m[2:9, 23] = True        # 'l': none
        labels, n, runs = sd.label_components(m)
        st = sd.component_stats(labels, n, runs)
        self.assertEqual(n, 3)
        self.assertEqual(sorted(st["holes"].astype(int).tolist()), [0, 1, 2])

    def test_empty(self):
        labels, n, runs = sd.label_components(np.zeros((5, 7), bool))
        self.assertEqual(n, 0)
        self.assertTrue((labels == -1).all())


class TestLines(unittest.TestCase):
    def test_detects_a_text_line(self):
        img = Image.new("L", (110, 30), 235)
        ImageDraw.Draw(img).text((6, 8), "HOTEL 24", fill=20, font=ImageFont.load_default())
        img = img.resize((440, 120), Image.NEAREST).convert("RGB")
        o = sd.detect_lines(img)
        self.assertGreaterEqual(len(o["feats"]), 1)
        self.assertEqual(o["feats"].shape[1], len(sd.FEATURES))
        best = o["boxes"][int(np.argmax(o["ncomp"]))]
        self.assertLess(best[1], 70)
        self.assertGreater(best[3], 40)

    def test_crop_loglik(self):
        P = np.full((2, len(sd.LINE_CLASSES)), 0.01)
        P[:, sd.SCRIPTS.index("Thai")] = 1.0
        P /= P.sum(1, keepdims=True)
        ll = sd.crop_loglik(P, 0.3)
        self.assertEqual(int(np.argmax(ll)), sd.SCRIPTS.index("Thai"))
        P = np.full((2, len(sd.LINE_CLASSES)), 0.01)
        P[:, sd.NOISE] = 1.0
        ll = sd.crop_loglik(P / P.sum(1, keepdims=True), 0.3)
        self.assertLess(ll.max() - ll.min(), 0.2)  # noise lines carry no script evidence


if __name__ == "__main__":
    unittest.main()
