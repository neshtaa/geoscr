"""Tests of tools/world_play.py (World-map calibration games), the world_play dataset mode and its split.

  python3 -m unittest test_world_play -v
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import build_dataset as BD  # noqa: E402
import calibrate as C  # noqa: E402
import world_play as W  # noqa: E402


class FakeTime:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def fake_api(responses):
    ft = FakeTime()
    log = []

    def request(url, data=None, method=None):
        log.append((method, url, data))
        ft.t += 0.1
        return responses.pop(0)

    return W.Api(request=request, sleep=ft.sleep, clock=ft.clock), log, ft


def finished_game(token="AbCdEf123", state="finished"):
    hexid = "CAoSLEFGMVFpcE1fZm9v".encode().hex()
    rounds = [{"lat": 12.85, "lng": 102.61, "panoId": hexid, "heading": 33.0, "pitch": 0, "zoom": 0,
               "streakLocationCode": "kh", "startTime": "2026-10-10T10:00:00Z"},
              {"lat": 50.45, "lng": 30.52, "panoId": None, "heading": 120, "pitch": 0, "zoom": 0,
               "streakLocationCode": "ua", "startTime": "2026-10-10T10:01:00Z"}]
    guesses = [{"lat": 23.0, "lng": 77.0, "roundScoreInPoints": 1200, "distanceInMeters": 2.6e6, "time": 30,
                "timedOut": False, "stepsCount": 0},
               {"lat": 50.0, "lng": 30.0, "roundScoreInPoints": 4800, "distanceInMeters": 6.0e4, "time": 25,
                "timedOut": False, "stepsCount": 3}]
    return {"token": token, "type": "standard", "mode": "standard", "state": state, "map": W.WORLD_MAP["id"],
            "mapName": "World", "roundCount": 2, "rounds": rounds, "player": {"guesses": guesses}}


def response(top, prob, guess, regions=(), fused=1):
    return {"countries": [{"code": top, "probability": prob}, {"code": "XX", "probability": 1 - prob}],
            "regions": [{"code": c, "country": top, "probability": p} for c, p in regions],
            "hints": [{"regions": [{"code": c, "probability": p} for c, p in regions]}],
            "guess": {"lat": guess[0], "lng": guess[1], "expected_score": 1000},
            "fusion": {"n": fused}, "map": {"prior": "ranked"}}


class TestApi(unittest.TestCase):
    def test_create_body_is_the_clients_start_game_body(self):
        b = W.create_body()
        self.assertEqual(b, {"map": "world", "type": "standard", "timeLimit": 0, "forbidMoving": False,
                             "forbidZooming": False, "forbidRotating": False, "rounds": 5})

    def test_pacing_and_stops(self):
        api, log, ft = fake_api([(200, {}), (200, {}), (429, None)])
        api.call("GET", "u1")
        api.call("GET", "u2")
        self.assertTrue(ft.slept and min(ft.slept) >= W.MIN_GAP_S - 0.11)
        with self.assertRaises(W.Stop):
            api.call("GET", "u3")
        for code in (401, 403):
            api, _, _ = fake_api([(code, None)])
            with self.assertRaises(W.Stop):
                api.call("POST", "x", {})

    def test_create_game_keeps_settings_only(self):
        g = finished_game(state="started")
        g["round"] = 1
        api, log, _ = fake_api([(200, g)])
        meta = W.create_game(api)
        self.assertEqual(log[0][0], "POST")
        self.assertTrue(log[0][1].endswith("/api/v3/games"))
        self.assertEqual(log[0][2], W.create_body())
        s = json.dumps(meta)
        self.assertNotIn("rounds", s)
        self.assertNotIn("12.85", s)
        self.assertEqual(meta["token"], "AbCdEf123")
        bad = dict(g, type="challenge")
        api, _, _ = fake_api([(200, bad)])
        with self.assertRaises(W.Stop):
            W.create_game(api)
        api, _, _ = fake_api([(400, {"message": "InvalidSettings"})])
        with self.assertRaises(RuntimeError):
            W.create_game(api)

    def test_never_reads_a_game_not_known_finished(self):
        for g in ({"token": "t", "status": "created"}, {"token": "t", "status": "played", "summary": {"finished": False}},
                  {"token": "t", "status": "played"}, {"token": "t", "status": "abandoned", "summary": {"finished": True}}):
            api, log, _ = fake_api([(200, finished_game())])
            with self.assertRaises(ValueError):
                W.fetch_finished(api, g)
            self.assertEqual(log, [])
        g = {"token": "t", "status": "played", "summary": {"finished": True}}
        api, log, _ = fake_api([(200, finished_game(state="started"))])
        with self.assertRaises(RuntimeError):
            W.fetch_finished(api, g)
        api, log, _ = fake_api([(200, finished_game())])
        self.assertEqual(W.fetch_finished(api, g)["state"], "finished")
        self.assertEqual(log[0][:2], ("GET", W.SITE + "/api/v3/games/t"))


class TestRounds(unittest.TestCase):
    def test_splits_and_alternation(self):
        for tok in ("a", "b", "x78UXZE1PAnz4oLR", "AbCdEf123"):
            rec = {"mode": "world_play", "game": tok, "pano_id": "p"}
            self.assertEqual(C.split_of(rec), W.split_of_game(tok))
            self.assertIn(C.split_of(rec), ("world_calib", "world_test"))
            hist = C.split_of({"mode": "history", "game": tok, "pano_id": "p", "map": "World"})
            self.assertEqual(hist, W.split_of_game(tok)[6:])  # same hash rule as calib / test
        self.assertFalse(C.in_eval({"mode": "world_play", "game": "a", "pano_id": "p"}))
        games = []
        seq = []
        for _ in range(6):
            w = W.next_walk(games, 3)
            seq.append(w)
            games.append({"walk": w, "status": "finished"})
        self.assertEqual(seq, [0, 3, 0, 3, 0, 3])
        self.assertEqual(W.next_walk([{"walk": 0, "status": "abandoned"}], 3), 0)
        self.assertEqual(W.next_walk([{"walk": 0, "status": "finished"}], 0), 0)

    def test_capture_dirs_order(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("T_r1", "T_r1_m1", "T_r1_m10", "T_r1_m2", "T_r10", "T_r2"):
                os.makedirs(os.path.join(d, n))
            self.assertEqual([os.path.basename(x) for x in W.capture_dirs(d, "T", 1)], ["T_r1", "T_r1_m1", "T_r1_m2", "T_r1_m10"])
            self.assertEqual([os.path.basename(x) for x in W.capture_dirs(d, "T", 3)], [])

    def test_round_records(self):
        with tempfile.TemporaryDirectory() as d:
            def put(name, resp, views=None):
                os.makedirs(os.path.join(d, name))
                json.dump(resp, open(os.path.join(d, name, "response.json"), "w"))
                if views:
                    json.dump(views, open(os.path.join(d, name, "views.json"), "w"))
            put("G_r1", response("IN", 0.9, (23.0, 77.0), [("IN-MP", 0.3)]),
                {"mode": "sweep", "views": [{}] * 10, "fov": {"method": "registration"}})
            put("G_r2", response("RU", 0.5, (55.0, 37.0)))
            put("G_r2_m1", response("UA", 0.6, (50.0, 30.0), [("UA-30", 0.5)], fused=2))
            summary = {"rounds": [{"round": 1, "points": 1200}, {"round": 2, "points": 4800}]}
            recs = W.round_records(finished_game("G"), summary, lambda i: W.capture_dirs(d, "G", i))
        self.assertEqual(len(recs), 2)
        a, b = recs
        self.assertEqual(a["pano_id"], "CAoSLEFGMVFpcE1fZm9v")
        self.assertEqual(a["country"], "KH")
        self.assertEqual(a["points"], 1200)
        self.assertEqual(a["start"]["countries"][0][0], "IN")
        self.assertEqual(a["capture"], {"mode": "sweep", "views": 10, "fov": "registration", "zoom_locked": None})
        self.assertEqual(a["n_captures"], 1)
        self.assertIsNone(b["pano_id"])
        self.assertEqual(b["country"], "UA")
        self.assertEqual(b["region"], "UA-30")
        self.assertEqual(b["n_captures"], 2)
        self.assertEqual(b["n_fused"], 2)
        self.assertEqual(b["start"]["countries"][0][0], "RU")
        self.assertEqual(b["final"]["countries"][0][0], "UA")
        self.assertEqual(b["steps"], 3)
        self.assertLess(b["start"]["points"], b["final"]["points"])  # offline points of each answer
        self.assertGreater(b["final"]["points"], 4500)

    def test_report(self):
        with tempfile.TemporaryDirectory() as d:
            def put(name, resp):
                os.makedirs(os.path.join(d, name))
                json.dump(resp, open(os.path.join(d, name, "response.json"), "w"))
            put("G_r1", response("KH", 0.6, (12.8, 102.6), [("KH-24", 0.4)]))
            put("G_r2", response("UA", 0.9, (50.4, 30.5), [("UA-30", 0.9)]))
            recs = W.round_records(finished_game("G"), {}, lambda i: W.capture_dirs(d, "G", i))
        st = {"games": [{"token": "G", "walk": 0, "status": "finished", "split": "world_calib", "rounds": recs},
                        {"token": "H", "walk": 3, "status": "abandoned"}]}
        rep = W.report(st)
        self.assertEqual(rep["games"]["finished"], 1)
        self.assertEqual(rep["games"]["abandoned"], 1)
        self.assertEqual(rep["rounds"], 2)
        self.assertEqual(rep["country_final"]["nomove"]["top1"], 1.0)
        self.assertEqual(rep["points_submitted"]["nomove"]["mean"], 3000.0)
        self.assertIsNone(rep["points_submitted"]["moving"])
        self.assertEqual(rep["region_given_right_country_final"]["nomove"]["n"], 2)
        self.assertEqual(dict(rep["world_country_distribution"]), {"KH": 1, "UA": 1})


class TestRebuild(unittest.TestCase):
    def test_true_country_probabilities_and_rebuild(self):
        resp = response("IN", 0.9, (23.0, 77.0))
        resp["posterior"] = {"IN": 0.76, "KH": 0.11, "TH": 0.13}
        resp["prior"] = {"IN": 0.05, "KH": 0.005, "TH": 0.02}
        d = W.response_digest(resp, truth=(12.85, 102.61), truth_cc="KH")
        self.assertEqual(d["rank_true"], 2)
        self.assertAlmostEqual(d["p_true"], 0.11)
        self.assertAlmostEqual(d["prior_true"], 0.005)
        self.assertAlmostEqual(d["prior_top"], 0.05)
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "g", "G_r1"))
            json.dump(resp, open(os.path.join(root, "g", "G_r1", "response.json"), "w"))
            st = {"games": [{"token": "G", "status": "finished", "walk": 0, "rounds": [
                {"round": 1, "lat": 12.85, "lng": 102.61, "country": "KH", "region": None, "dirs": ["g/G_r1"]}]},
                {"token": "H", "status": "created"}]}
            self.assertEqual(W.rebuild_digests(st, root), 1)
        r = st["games"][0]["rounds"][0]
        self.assertEqual(r["start"]["rank_true"], 2)
        self.assertIsNone(r["final"])
        rep = W.report(st)
        self.assertEqual(rep["wrong_top1"], {"n": 1, "prior_favoured_top_over_truth": 1})
        self.assertLess(rep["logp_true_start"]["all"]["mean"], -2.0)


class TestReplay(unittest.TestCase):
    def test_replay_posts_the_saved_views_in_order_under_one_round_key(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("G_r1", "G_r1_m1"):
                d = os.path.join(root, name)
                os.makedirs(d)
                open(os.path.join(d, "view_00.jpg"), "wb").write(b"jpeg")
                json.dump({"views": [{"file": "view_00.jpg", "yaw": 10, "pitch": 0, "hfov": 120, "vfov": 90}]},
                          open(os.path.join(d, "views.json"), "w"))
                json.dump({"map": dict(W.WORLD_MAP, extra=1)}, open(os.path.join(d, "meta.json"), "w"))
            st = {"games": [{"token": "G", "status": "finished", "walk": 3, "split": "world_test", "rounds": [
                {"round": 1, "lat": 50.45, "lng": 30.52, "country": "UA", "region": "UA-30", "dirs": ["G_r1", "G_r1_m1"]}]},
                {"token": "H", "status": "finished", "walk": 0, "split": "world_calib", "rounds": [
                    {"round": 1, "lat": 1, "lng": 1, "dirs": ["G_r1"]}]}]}
            sent = []
            answers = [response("RU", 0.6, (55.7, 37.6)), response("UA", 0.7, (50.4, 30.5), fused=2)]

            def post(url, body):
                sent.append((url, body))
                return answers[len(sent) - 1]
            res = W.replay_eval(st, "http://127.0.0.1:9", ("world_test",), root=root, post=post, tag="t1", log=lambda m: None)
        self.assertEqual(len(sent), 2)
        self.assertTrue(sent[0][0].endswith("/api/predict"))
        self.assertEqual(sent[0][1]["fusion"], {"round": "replay:G#1@t1", "reset": True})
        self.assertEqual(sent[1][1]["fusion"], {"round": "replay:G#1@t1", "reset": False})
        self.assertNotIn("extra", sent[0][1]["map"])
        self.assertEqual(sent[0][1]["views"][0]["yaw"], 10)
        row = res["rows"][0]
        self.assertEqual(row["start"]["countries"][0][0], "RU")
        self.assertEqual(row["final"]["countries"][0][0], "UA")
        self.assertEqual(res["summary"]["country_final"]["moving"]["top1"], 1.0)
        self.assertEqual(res["summary"]["country_start"]["moving"]["top1"], 0.0)


class TestReplayIncompleteAndPaired(unittest.TestCase):
    def test_incomplete_rounds_left_out_and_paired_differences(self):
        with tempfile.TemporaryDirectory() as root:
            for name, img in (("G_r1", True), ("G_r1_m1", True), ("G_r2", True), ("G_r2_m1", False), ("N_r1", True)):
                d = os.path.join(root, name)
                os.makedirs(d)
                if img:
                    open(os.path.join(d, "view_00.jpg"), "wb").write(b"jpeg")
                json.dump({"views": [{"file": "view_00.jpg", "yaw": 0, "pitch": 0, "hfov": 120, "vfov": 90}]},
                          open(os.path.join(d, "views.json"), "w"))
            st = {"games": [
                {"token": "G", "status": "finished", "walk": 3, "split": "world_test", "rounds": [
                    {"round": 1, "lat": 50.45, "lng": 30.52, "country": "UA", "dirs": ["G_r1", "G_r1_m1"]},
                    {"round": 2, "lat": 50.45, "lng": 30.52, "country": "UA", "dirs": ["G_r2", "G_r2_m1"]}]},
                {"token": "N", "status": "finished", "walk": 0, "split": "world_test", "rounds": [
                    {"round": 1, "lat": 50.45, "lng": 30.52, "country": "UA", "dirs": ["N_r1"]}]}]}

            def run(answers):
                it = iter(answers)
                return W.replay_eval(st, "http://127.0.0.1:9", ("world_test",), root=root, post=lambda u, b: next(it),
                                     tag="t", log=lambda m: None)
            # new server: G r1 start RU then fused UA; G r2 start UA (its move capture has no image); N r1 UA
            new = run([response("RU", 0.6, (55.7, 37.6)), response("UA", 0.7, (50.4, 30.5), fused=2),
                       response("UA", 0.6, (50.4, 30.5)), response("UA", 0.8, (50.4, 30.5))])
            old = run([response("RU", 0.6, (55.7, 37.6)), response("RU", 0.7, (55.7, 37.6), fused=2),
                       response("UA", 0.6, (50.4, 30.5)), response("RU", 0.8, (55.7, 37.6))])
        g2 = [r for r in new["rows"] if r["game"] == "G" and r["round"] == 2][0]
        self.assertFalse(g2["complete"])
        self.assertEqual((g2["captures"], g2["of"]), (1, 2))
        self.assertEqual(new["summary"]["incomplete_rounds_left_out"], 1)
        self.assertEqual(new["summary"]["rounds"], 2)
        self.assertEqual(new["summary"]["country_final"]["moving"]["n"], 1)
        pd = W.replay_paired(new["rows"], old["rows"])
        self.assertEqual(pd["moving"]["n"], 1)
        self.assertEqual(pd["moving"]["top1_diff"], 1.0)
        self.assertGreater(pd["moving"]["points_diff"], 0)
        self.assertEqual(pd["moving_start"]["top1_diff"], 0.0)
        self.assertEqual(pd["nomove"]["top1_diff"], 1.0)
        self.assertIsNone(pd["nomove"]["points_ci95"])   # one game: no interval by game


class TestAreasSnapshotClues(unittest.TestCase):
    def test_true_area_and_area_metric(self):
        self.assertEqual(W.truth_area("KH", "KH-24"), "KH:TonleSap")
        self.assertIsNone(W.truth_area("KH", None))
        self.assertIsNone(W.truth_area(None, "KH-24"))
        resp = response("KH", 0.8, (12.8, 102.6), [("KH-24", 0.4)])
        resp["hints"][0]["areas"] = [{"id": "KH:TonleSap", "probability": 0.6}, {"id": "KH:Coast", "probability": 0.2}]
        d = W.response_digest(resp)
        self.assertEqual(d["top_country_areas"], [["KH:TonleSap", 0.6], ["KH:Coast", 0.2]])
        old = W.response_digest(response("KH", 0.8, (12.8, 102.6), [("KH-24", 0.4)]))
        self.assertNotIn("top_country_areas", old)   # engine without areas
        rows = [{"country": "KH", "region": "KH-24", "area": "KH:TonleSap", "walk": 0, "start": d, "final": d},
                {"country": "KH", "region": "KH-7", "area": "KH:Coast", "walk": 0, "start": old, "final": old}]
        st = {"games": [{"token": "G", "walk": 0, "status": "finished", "rounds": rows}]}
        rep = W.report(st)
        self.assertEqual(rep["area_given_right_country_final"]["all"], {"n": 1, "top1": 1.0, "top3": 1.0})

    def test_snapshot_of_the_working_tree(self):
        with tempfile.TemporaryDirectory() as root:
            for rel, txt in (("engine/a.py", "x = 1"), ("web/server.py", "pass"), ("data/model/model.json", "{}"),
                             ("data/areas.json", "{}"), ("data/fusion.json", "{}"), ("engine/__pycache__/a.pyc", "c")):
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                open(os.path.join(root, rel), "w").write(txt)
            work = os.path.join(root, "scratch", "world")
            old_root = W.ROOT
            W.ROOT = root
            try:
                fp = W.model_fingerprint(root, head=False)
                t = W.snapshot_tree(work)
                self.assertTrue(t and os.path.isdir(t))
                self.assertEqual(W.snapshot_tree(work), t)   # same state: same copy
                fz = json.load(open(os.path.join(t, "FROZEN.json")))
                self.assertTrue(fz["snapshot"])
                self.assertTrue(W.same_tree(fz["fingerprint"], fp))
                self.assertTrue(W.same_tree(W.model_fingerprint(t, head=False), fp))
                self.assertFalse(os.path.exists(os.path.join(t, "engine", "__pycache__")))
                self.assertTrue(W.model_fingerprint(t)["head"].startswith("frozen"))
                open(os.path.join(root, "data", "areas.json"), "w").write('{"a": 1}')   # areas are part of it
                fp2 = W.model_fingerprint(root, head=False)
                self.assertFalse(W.same_tree(fp, fp2))
                self.assertEqual((fp["model"], fp["engine"]), (fp2["model"], fp2["engine"]))
                self.assertNotEqual(W.snapshot_tree(work), t)
                self.assertLess(W.tree_quiet_s(root), 60)
            finally:
                W.ROOT = old_root

    def test_clue_sample(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pano_clues.json")
            json.dump({"B1": [{"id": "x"}]}, open(path, "w"))
            old = W.CLUES_FILE
            W.CLUES_FILE = path
            try:
                st = {"games": [{"token": "A", "status": "finished", "rounds": [{"pano_id": "A1"}, {"pano_id": "A2"}]},
                                {"token": "B", "status": "finished", "rounds": [{"pano_id": "B1"}, {"pano_id": "B2"}]},
                                {"token": "C", "status": "created", "rounds": []},
                                {"token": "D", "status": "played"}]}
                r = W.Runner.__new__(W.Runner)
                for mode, want in (("sample", ["A1", "B1", "B2"]), ("all", ["A1", "A2", "B1", "B2"]), ("none", [])):
                    r.args = argparse.Namespace(clues=mode)
                    self.assertEqual(r.clue_ids(st), want)
            finally:
                W.CLUES_FILE = old


class TestResume(unittest.TestCase):
    def test_a_rerun_finishes_what_a_crash_left_and_creates_nothing(self):
        """created -> node run (walk kept) -> played -> read once finished; played -> read; a game past its runs
        -> abandoned unread; --games 0 creates no game; every GeoGuessr request is a GET of a finished game."""
        import argparse
        with tempfile.TemporaryDirectory() as work:
            st = {"map": dict(W.WORLD_MAP), "games": [
                {"n": 1, "token": "Gcreated", "walk": 3, "walk_clicks": 4, "status": "created", "attempts": 1},
                {"n": 2, "token": "Gplayed", "walk": 0, "status": "played", "summary": {"finished": True, "rounds": []}},
                {"n": 3, "token": "Gstuck", "walk": 0, "status": "created", "attempts": 2}]}
            saved = []
            runs = []

            def run_node(token, walk, url, summary, log_path, max_minutes, walk_clicks=1):
                runs.append((token, walk, walk_clicks))
                os.makedirs(os.path.dirname(summary), exist_ok=True)
                json.dump({"finished": True, "end": "game finished", "total": 6000, "rounds": []}, open(summary, "w"))
                return 0

            api, log, _ = fake_api([(200, finished_game("Gcreated")), (200, finished_game("Gplayed"))])
            args = argparse.Namespace(api_gap=W.MIN_GAP_S, games=0, hours=1.0, gap=0.0, walk=3, walk_clicks=4, freeze="",
                                      snapshot=False, refresh_min=0.0, clues="none", max_minutes=1.0, max_attempts=2,
                                      collect_every=10, workers=1, min_avail_mb=0, max_disk_gb=100.0, keep_server=False, port=8097)
            patched = {"WORK": work, "LIVE_OUT": os.path.join(work, "live_in"), "load_state": lambda path=None: st,
                       "save_state": lambda s, path=None: saved.append(json.dumps(s)), "run_node": run_node,
                       "wait_memory": lambda *a, **k: None}
            old = {k: getattr(W, k) for k in patched}
            old_ensure = W.Runner.ensure_server
            for k, v in patched.items():
                setattr(W, k, v)

            def ensure(self):
                self.url, self.fingerprint = "http://127.0.0.1:9", {"model": "m", "engine": "e", "data": "d"}
            W.Runner.ensure_server = ensure
            try:
                W.Runner(args, api=api, log=lambda m: None).run()
            finally:
                for k, v in old.items():
                    setattr(W, k, v)
                W.Runner.ensure_server = old_ensure
        self.assertEqual(runs, [("Gcreated", 3, 4)])
        self.assertEqual([g["status"] for g in st["games"]], ["finished", "finished", "abandoned"])
        self.assertEqual(st["games"][0]["attempts"], 2)
        self.assertEqual(st["games"][0]["model"], {"model": "m", "engine": "e", "data": "d"})
        self.assertEqual([g["rounds"][0]["country"] for g in st["games"][:2]], ["KH", "KH"])
        self.assertEqual([(m, u) for m, u, _ in log], [("GET", W.SITE + "/api/v3/games/Gcreated"),
                                                       ("GET", W.SITE + "/api/v3/games/Gplayed")])
        self.assertTrue(saved)


class TestPrune(unittest.TestCase):
    def test_requality_first_then_screens_then_move_views(self):
        from PIL import Image
        import numpy as np
        with tempfile.TemporaryDirectory() as work:
            st = {"games": []}
            for t in ("A", "B", "C"):
                for sub in ("%s_r1" % t, "%s_r1_m1" % t):
                    d = os.path.join(work, "live", t, sub)
                    os.makedirs(d)
                    Image.fromarray((np.random.default_rng(1).random((200, 300, 3)) * 255).astype("uint8")).save(
                        os.path.join(d, "view_00.jpg"), quality=95)
                    open(os.path.join(d, "hud_r1.jpg"), "wb").write(b"x" * 5000)
                st["games"].append({"token": t, "status": "finished"})
            old_work, old_used, old_save = W.WORK, W.disk_used, W.save_state
            W.WORK, W.save_state = work, lambda st: None
            W.disk_used = lambda: W.du_bytes(work)
            try:
                start = W.du_bytes(work)
                start_view = os.path.getsize(os.path.join(work, "live", "A", "A_r1", "view_00.jpg"))
                W.prune_move_views(st, lambda m: None, keep_bytes=0, delete_bytes=start)   # requality only
                self.assertTrue(all(g.get("move_views_quality") == 80 for g in st["games"]))
                self.assertFalse(any(g.get("move_views_deleted") or g.get("screens_deleted") for g in st["games"]))
                self.assertEqual(os.path.getsize(os.path.join(work, "live", "A", "A_r1", "view_00.jpg")), start_view)
                W.prune_move_views(st, lambda m: None, keep_bytes=0, delete_bytes=0, keep_screens=1)
                self.assertEqual([bool(g.get("screens_deleted")) for g in st["games"]], [True, True, False])
                self.assertTrue(all(g.get("move_views_deleted") for g in st["games"]))
                self.assertTrue(os.path.exists(os.path.join(work, "live", "A", "A_r1", "view_00.jpg")))   # start kept
                self.assertFalse(os.path.exists(os.path.join(work, "live", "A", "A_r1_m1", "view_00.jpg")))
                self.assertTrue(os.path.exists(os.path.join(work, "live", "C", "C_r1", "hud_r1.jpg")))
            finally:
                W.WORK, W.disk_used, W.save_state = old_work, old_used, old_save


class TestEvalLiveWorld(unittest.TestCase):
    def rows(self, shift=0.0):
        out = []
        for i in range(8):
            right = i % 2 == 0
            out.append({"pano_id": "P%d" % i, "game": "G%d" % (i // 2), "label": "KH", "codes": ["KH", "IN"] if right
                        else ["IN", "TH", "KH"], "km": 100.0, "points": 3000.0 + shift, "points_map": 3000.0 + shift,
                        "ms": 1000, "p_true": 0.6 if right else 0.1, "region": "KH-24", "region_rank": 0 if right else 99,
                        "region_rank_overall": 1, "walk": 3 if i < 4 else 0, "area": "KH:TonleSap",
                        "area_rank": 0 if i == 0 else 1})
        return out

    def test_world_summary_and_paired(self):
        import eval_live as E
        self.assertEqual(E.WORLD_LIVE_MAP["id"], W.WORLD_MAP["id"])
        self.assertEqual(E.WORLD_LIVE_MAP["maxErrorDistance"], W.WORLD_MAP["maxErrorDistance"])
        res = E.summarise(self.rows(), "world_test", with_cis=True)
        self.assertEqual((res["n"], res["games"]), (8, 4))
        self.assertEqual((res["top1"], res["top3"]), (0.5, 1.0))
        self.assertEqual(res["region_top1_if_country_right"], 1.0)
        self.assertEqual((res["area_top1_if_country_right"], res["area_n"]), (0.25, 4))
        self.assertAlmostEqual(res["loglik_true_country"], (4 * __import__("math").log(0.6) + 4 * __import__("math").log(0.1)) / 8)
        self.assertEqual(res["by_game_condition"]["moving_games"]["n"], 4)
        self.assertEqual(len(res["ci95_by_game"]["top1"]), 2)
        with tempfile.TemporaryDirectory() as d:
            prev = os.path.join(d, "prev.json")
            json.dump({"rows": self.rows(shift=-100.0)}, open(prev, "w"))
            pr = E.paired(self.rows(), prev)
        self.assertEqual((pr["n"], pr["points_diff"], pr["top1_diff"]), (8, 100.0, 0.0))
        views = E.render_views(type("S", (), {"heading": 0.0, "render_view": lambda self, *a: a})(), 10.0,
                               **{k: E.GRIDS["headless"][k] for k in ("hfov", "size", "yaws")})
        self.assertEqual([v["yaw"] for v in views[:4]], [10.0, 100.0, 190.0, 280.0])
        self.assertEqual(len(views), 8)


class TestDatasetMode(unittest.TestCase):
    def test_world_play_rounds_and_history_filter(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "world_games.json")
            st = {"games": [
                {"token": "A", "walk": 3, "status": "finished", "rounds": [
                    {"round": 1, "lat": 1.0, "lng": 2.0, "pano_id": "P1", "heading": 10, "gg_country": "kh"},
                    {"round": 2, "lat": 3.0, "lng": 4.0, "pano_id": "P1", "heading": 10, "gg_country": "kh"}]},
                {"token": "B", "status": "created"},
                {"token": "C", "status": "abandoned"}]}
            json.dump(st, open(path, "w"))
            rounds = BD.world_play_rounds(path)
            self.assertEqual([(r["game"], r["round"]) for r in rounds], [("A", 1)])
            ex = BD.world_play_extra(rounds[0])
            self.assertEqual(ex["gg_country"], "KH")
            self.assertEqual(ex["walk"], 3)
            self.assertEqual((ex["round_lat"], ex["round_lng"]), (1.0, 2.0))
            self.assertEqual(BD.world_play_tokens(path), {"A", "B", "C"})
            self.assertEqual(BD.world_play_tokens(os.path.join(d, "missing.json")), set())

    def test_world_rounds_are_held_out_of_training(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "panos"))
            recs = [{"pano_id": "W1", "mode": "world_play", "game": "A", "lat": 10.0, "lng": 20.0, "gg_country": "KH",
                     "round_lat": 10.0, "round_lng": 20.0},
                    {"pano_id": "D1", "mode": "duel", "lat": 10.001, "lng": 20.001, "gg_country": "KH"},
                    {"pano_id": "D2", "mode": "duel", "lat": 30.0, "lng": 20.0, "gg_country": "KH"}]
            with open(os.path.join(d, "index.jsonl"), "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                    open(os.path.join(d, "panos", r["pano_id"] + ".jpg"), "w").close()
            own = os.path.join(d, "own.json")
            json.dump([], open(own, "w"))
            out = {r["pano_id"]: C.split_of(r) for r in C.load_records(d, own_file=own)}
        self.assertNotIn("D1", out)  # a training panorama within 1 km of a World-map round is left out
        self.assertEqual(out["D2"], "train")
        self.assertIn(out["W1"], ("world_calib", "world_test"))


if __name__ == "__main__":
    unittest.main()
