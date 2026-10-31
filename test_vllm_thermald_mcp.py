import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import vllm_thermald as daemon
import vllm_thermald_mcp as mcp

HERE = Path(__file__).parent
NOW = 1_800_000_000.0


def stamp(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def status(**overrides):
    base = {
        "updated": NOW,
        "daemon_started": NOW - 100,
        "container": "vllm",
        "container_status": "running",
        "gpu_c": 60,
        "board_c": 65,
        "power": "40 W",
        "util": "50 %",
        "hold": False,
        "counts": {"pause": 0},
        "controller": {"restart_wait_s": 0, "running_for_s": 90, "paused_for_s": None},
        "config": asdict(daemon.Config()),
    }
    return {**base, **overrides}


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write_status(self, **overrides):
        (self.dir / "status.json").write_text(json.dumps(status(**overrides)))

    def write_csv(self, rows, name="thermal.csv"):
        with open(self.dir / name, "a") as f:
            for t, gpu, board, st in rows:
                f.write(f"{stamp(t)},{gpu},{board},40.5 W,50 %,False,False,{st}\n")


class StateTests(unittest.TestCase):
    def check(self, expected, **overrides):
        self.assertEqual(mcp.derive_state(status(**overrides), NOW), expected)

    def test_states(self):
        self.check("normal")
        self.check("paused_cooling", container_status="paused")
        self.check("waiting_for_cool", container_status="exited")
        self.check("cooldown", container_status="exited", controller={"restart_wait_s": 30})
        self.check("held_off", container_status="exited", hold=True)
        self.check("unavailable", container_status="missing")
        self.check("daemon_down", updated=NOW - daemon.STALE_AFTER_S - 1)

    def test_headroom_uses_nearest_sensor(self):
        s = status(gpu_c=60, board_c=85)
        self.assertEqual(mcp.derive_headroom(s, "normal"), {"level": "low", "degrees_to_pause": 7})

    def test_headroom_levels(self):
        self.assertEqual(mcp.derive_headroom(status(gpu_c=65, board_c=50), "normal")["level"], "ample")
        self.assertEqual(mcp.derive_headroom(status(gpu_c=66, board_c=50), "normal")["level"], "moderate")

    def test_no_headroom_when_paused(self):
        self.assertEqual(mcp.derive_headroom(status(), "paused_cooling")["level"], "none")

    def test_headroom_is_unknown_with_a_failed_sensor(self):
        self.assertEqual(mcp.derive_headroom(status(gpu_c=None, board_c=50), "normal")["level"], "unknown")

    def test_docker_failure_is_not_protected(self):
        s = status(container_status="unknown")
        self.assertIn("Docker", mcp.protection_issues(s, mcp.derive_state(s, NOW))[0])

    def test_missing_gpu_sensor_is_not_protected(self):
        s = status(gpu_c=None)
        self.assertIn("GPU", mcp.protection_issues(s, "normal")[0])

    def test_healthy_has_no_protection_issues(self):
        self.assertEqual(mcp.protection_issues(status(), "normal"), [])


class ToolTests(Base):
    def test_status_without_daemon_reports_unprotected(self):
        got = mcp.thermal_status(self.dir, NOW)
        self.assertEqual(got["state"], "daemon_down")
        self.assertFalse(got["protected"])

    def test_stale_status_reports_unprotected(self):
        self.write_status(updated=NOW - 60)
        self.assertFalse(mcp.thermal_status(self.dir, NOW)["protected"])

    def test_status_reports_trend(self):
        self.write_status()
        self.write_csv([(NOW - 30, 60, 65, "running"), (NOW - 0, 62, 66, "running")])
        got = mcp.thermal_status(self.dir, NOW)
        self.assertEqual(got["trend_last_minute"]["gpu_c_per_min"], 4.0)

    def test_policy_reports_daemons_active_config_not_defaults(self):
        cfg = asdict(daemon.Config(pause_gpu=80, resume_gpu=65))
        self.write_status(config=cfg)
        self.assertEqual(mcp.thermal_policy(self.dir)["active_thresholds_c"]["pause"]["gpu"], 80)

    def test_status_marks_docker_failure_unprotected(self):
        self.write_status(container_status="unknown")
        self.assertFalse(mcp.thermal_status(self.dir, NOW)["protected"])

    def test_policy_includes_owner_observations_when_valid(self):
        self.write_status()
        path = self.dir / "obs.json"
        self.assertIsNone(mcp.thermal_policy(self.dir, path)["owner_observations"])
        path.write_text('{"note": "runs hot"}')
        self.assertEqual(mcp.thermal_policy(self.dir, path)["owner_observations"], {"note": "runs hot"})
        for bad in ("not json", "[1]", "x" * (mcp.OBSERVATIONS_MAX_BYTES + 1)):
            path.write_text(bad)
            self.assertIsNone(mcp.thermal_policy(self.dir, path)["owner_observations"])

    def test_non_object_status_file_is_down(self):
        (self.dir / "status.json").write_text("[1]")
        self.assertEqual(mcp.thermal_status(self.dir, NOW)["state"], "daemon_down")

    def test_policy_without_daemon(self):
        self.assertIn("error", mcp.thermal_policy(self.dir))

    def test_policy_states_cover_every_derived_state(self):
        self.write_status()
        states = mcp.thermal_policy(self.dir)["states"]
        for overrides in (
            {}, {"container_status": "paused"}, {"container_status": "exited"},
            {"container_status": "exited", "controller": {"restart_wait_s": 1}},
            {"container_status": "exited", "hold": True}, {"container_status": "missing"}, {"updated": 0},
        ):
            self.assertIn(mcp.derive_state(status(**overrides), NOW), states)

    def test_history_summarizes_and_excludes_old(self):
        self.write_csv([(NOW - 7200, 99, 99, "running"), (NOW - 100, 50, 60, "running"), (NOW - 50, 70, 80, "paused")])
        got = mcp.thermal_history(self.dir, NOW, minutes=30)
        self.assertEqual(got["samples"], 2)
        self.assertEqual(got["gpu_c"]["max"], 70)
        self.assertEqual(got["samples_paused"], 1)

    def test_history_finds_logging_gaps(self):
        self.write_csv([(NOW - 500, 50, 60, "running"), (NOW - 100, 50, 60, "running")])
        self.assertEqual(mcp.thermal_history(self.dir, NOW)["gaps_over_60s"][0]["seconds"], 400)

    def test_history_includes_rotated_file(self):
        self.write_csv([(NOW - 200, 50, 60, "running")], "thermal.csv.1")
        self.write_csv([(NOW - 100, 70, 60, "running")])
        self.assertEqual(mcp.thermal_history(self.dir, NOW)["samples"], 2)

    def test_history_limits_series_points(self):
        self.write_csv([(NOW - i, 50, 60, "running") for i in range(600, 0, -1)])
        self.assertLessEqual(len(mcp.thermal_history(self.dir, NOW, minutes=30, points=5)["series"]), 5)

    def test_history_empty_and_corrupt_lines(self):
        (self.dir / "thermal.csv").write_text("garbage\n1,2\n")
        self.assertEqual(mcp.thermal_history(self.dir, NOW)["samples"], 0)

    def test_events_newest_last_and_limited(self):
        log = daemon.EventLog(self.dir / "events.jsonl")
        for i in range(5):
            log.write("pause", reason=str(i))
        with open(self.dir / "events.jsonl", "a") as f:
            f.write("not json\n")
        got = mcp.thermal_events(self.dir, limit=2)["events"]
        self.assertEqual([e["reason"] for e in got], ["3", "4"])


class ParserTests(unittest.TestCase):
    def test_metrics_sums_labels_and_derives_rates(self):
        got = mcp.parse_metrics(
            '# HELP x\n'
            'vllm:num_requests_running{engine="0"} 2.0\n'
            'vllm:num_requests_running{engine="1"} 1.0\n'
            'vllm:generation_tokens_total{m="a"} 500.0\n'
            'vllm:request_decode_time_seconds_sum{m="a"} 10.0\n'
            'vllm:prompt_tokens_total 100.0\nvllm:prompt_tokens_cached_total 25.0\n'
        )
        self.assertEqual(got["requests_running"], 3.0)
        self.assertEqual(got["mean_generation_tokens_per_s"], 50.0)
        self.assertEqual(got["prompt_cache_hit_fraction"], 0.25)

    def test_metrics_empty(self):
        self.assertIsNone(mcp.parse_metrics("")["mean_generation_tokens_per_s"])

    def test_flags_parse_and_omit_secrets(self):
        got = mcp.parse_flags(["vllm", "serve", "m", "--port", "8000", "--async-scheduling", "--api-key", "abc", "--hf-token", "x"])
        self.assertEqual(got, {"port": "8000", "async-scheduling": True})

    def test_flags_handle_equals_form(self):
        self.assertEqual(mcp.parse_flags(["--port=8001", "--api-key=abc"]), {"port": "8001"})

    def test_flags_keep_harmless_names_containing_token(self):
        self.assertEqual(mcp.parse_flags(["--tokenizer", "x", "--max-tokens", "5"]), {"tokenizer": "x", "max-tokens": "5"})

    def test_flags_omit_secret_looking_values(self):
        self.assertEqual(mcp.parse_flags(["--extra", "hf_abcdefghijklmnop", "--port", "1"]), {"port": "1"})

    def test_http_ignores_proxy_environment(self):
        with mock.patch.dict(os.environ, {"http_proxy": "http://proxy.invalid:1"}):
            opener = mcp.urllib.request.build_opener(mcp.urllib.request.ProxyHandler({}))
        self.assertEqual([h for h in opener.handlers if isinstance(h, mcp.urllib.request.ProxyHandler)], [])
        self.assertEqual([h for h in mcp._opener.handlers if isinstance(h, mcp.urllib.request.ProxyHandler)], [])

    def test_docker_inspect_survives_a_pipe_in_the_command(self):
        out = '{"Status":"running"}\n["bash","-c","a | b"]\n"img"\n'
        with mock.patch.object(mcp.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=out)):
            self.assertEqual(mcp._docker_inspect("c")["cmd"], ["bash", "-c", "a | b"])


class ControllerSnapshotTests(unittest.TestCase):
    def test_snapshot_reports_durations(self):
        ctl = daemon.Controller(daemon.Config())
        ctl.step(daemon.Reading(gpu=40, board=40), "running", 100)
        got = ctl.snapshot(130)
        self.assertEqual(got["running_for_s"], 30)
        self.assertIsNone(got["paused_for_s"])
        self.assertEqual(got["restart_wait_s"], 0)


class ProtocolTests(Base):
    def rpc(self, method, params=None, request_id=1):
        return mcp.handle({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}, self.dir)

    def test_initialize_echoes_supported_version(self):
        got = self.rpc("initialize", {"protocolVersion": "2025-03-26"})["result"]
        self.assertEqual(got["protocolVersion"], "2025-03-26")
        self.assertIn("tools", got["capabilities"])

    def test_initialize_offers_latest_for_unknown_version(self):
        got = self.rpc("initialize", {"protocolVersion": "1999-01-01"})["result"]
        self.assertEqual(got["protocolVersion"], mcp.PROTOCOL_VERSIONS[0])

    def test_notification_gets_no_response(self):
        self.assertIsNone(mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, self.dir))

    def test_unknown_method(self):
        self.assertEqual(self.rpc("nope")["error"]["code"], -32601)

    def test_all_tools_are_read_only(self):
        tools = self.rpc("tools/list")["result"]["tools"]
        self.assertEqual({t["name"] for t in tools}, set(mcp.TOOLS))
        for tool in tools:
            self.assertTrue(tool["annotations"]["readOnlyHint"])

    def test_call_returns_json_text(self):
        got = self.rpc("tools/call", {"name": "thermal_status"})["result"]
        self.assertFalse(got["isError"])
        self.assertEqual(json.loads(got["content"][0]["text"])["state"], "daemon_down")

    def test_rejects_bad_arguments(self):
        for args in ({"minutes": 0}, {"minutes": 99999}, {"minutes": "5"}, {"minutes": True}, {"state_dir": "/etc"}):
            got = self.rpc("tools/call", {"name": "thermal_history", "arguments": args})
            self.assertEqual(got["error"]["code"], -32602, args)

    def test_unknown_tool(self):
        self.assertEqual(self.rpc("tools/call", {"name": "hold"})["error"]["code"], -32602)

    def test_tool_fault_is_reported_without_leaking_detail(self):
        self.write_status()
        original = mcp.TOOLS["thermal_status"]
        def boom(**_):
            raise RuntimeError("secret detail")
        mcp.TOOLS["thermal_status"] = (boom, original[1], original[2])
        self.addCleanup(mcp.TOOLS.__setitem__, "thermal_status", original)
        got = self.rpc("tools/call", {"name": "thermal_status"})["result"]
        self.assertTrue(got["isError"])
        self.assertNotIn("secret", got["content"][0]["text"])

    def test_malformed_requests_get_errors_rather_than_raising(self):
        cases = [
            ('{"id":1,"method":"tools/call","params":{"name":"thermal_history","arguments":[1]}}', -32602),
            ('{"id":1,"method":"tools/call","params":[1]}', -32602),
            ('{"id":1,"method":"tools/call","params":{"name":[]}}', -32602),
            ('{"id":1,"method":"tools/call","params":{"name":{}}}', -32602),
            ('{"id":1,"method":["x"]}', -32601),
            ("[1,2]", -32600),
            ('"text"', -32600),
            ("{bad", -32700),
        ]
        for line, code in cases:
            self.assertEqual(mcp.process_line(line, self.dir)["error"]["code"], code, line)

    def test_unexpected_fault_in_handle_is_contained(self):
        with mock.patch.object(mcp, "handle", side_effect=RuntimeError("x")):
            self.assertEqual(mcp.process_line('{"id":7}', self.dir), {
                "jsonrpc": "2.0", "id": 7, "error": {"code": -32603, "message": "internal error"},
            })

    def test_stdio_survives_invalid_utf8(self):
        (self.dir / "vllm-thermald").mkdir()
        request = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}).encode()
        env = {**os.environ, "XDG_STATE_HOME": str(self.dir), "PYTHONDONTWRITEBYTECODE": "1"}
        out = subprocess.run(
            [sys.executable, str(HERE / "vllm_thermald_mcp.py")],
            input=b"\xff\xfe garbage\n" + request + b"\n", capture_output=True, env=env, timeout=20,
        ).stdout.decode().splitlines()
        self.assertEqual(json.loads(out[-1])["id"], 2)

    def test_stdio_session_survives_bad_input(self):
        (self.dir / "vllm-thermald").mkdir()
        lines = [
            "not json",
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "thermal_status"}}),
        ]
        env = {**os.environ, "XDG_STATE_HOME": str(self.dir), "PYTHONDONTWRITEBYTECODE": "1"}
        out = subprocess.run(
            [sys.executable, str(HERE / "vllm_thermald_mcp.py")],
            input="\n".join(lines) + "\n", capture_output=True, text=True, env=env, timeout=20,
        ).stdout.splitlines()
        self.assertEqual(json.loads(out[0])["error"]["code"], -32700)
        self.assertEqual(json.loads(out[1])["result"]["serverInfo"]["name"], "vllm-thermald")
        self.assertEqual(len(out), 3)
        self.assertEqual(json.loads(json.loads(out[2])["result"]["content"][0]["text"])["state"], "daemon_down")


if __name__ == "__main__":
    unittest.main()
