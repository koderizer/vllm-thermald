import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vllm_thermald import __main__ as cli
from vllm_thermald import service


def listing(*rows):
    return "\n".join("\t".join(row) for row in rows)


class ChooseContainerTests(unittest.TestCase):
    def test_picks_the_only_running_vllm_container(self):
        out = listing(("old", "vllm/vllm-openai:v1", "exited"), ("new", "nvcr.io/nvidia/vllm:2", "running"), ("db", "postgres", "running"))
        self.assertEqual(service.choose_container(out), "new")

    def test_picks_a_lone_stopped_container(self):
        self.assertEqual(service.choose_container(listing(("a", "vllm/x", "exited"))), "a")

    def test_matches_on_name_too(self):
        self.assertEqual(service.choose_container(listing(("my-vllm", "custom", "running"))), "my-vllm")

    def test_refuses_to_guess(self):
        for out in (
            listing(("a", "vllm/x", "running"), ("b", "vllm/x", "running")),
            listing(("a", "vllm/x", "exited"), ("b", "vllm/x", "exited")),
        ):
            with self.assertRaisesRegex(service.SetupError, "--container"):
                service.choose_container(out)

    def test_none_found(self):
        with self.assertRaisesRegex(service.SetupError, "no vLLM container"):
            service.choose_container(listing(("db", "postgres", "running")))


class ResolveContainerTests(unittest.TestCase):
    def test_argument_wins_over_config(self):
        self.assertEqual(service.resolve_container("a", {"container": "b"}), "a")

    def test_existing_config_is_kept(self):
        self.assertEqual(service.resolve_container(None, {"container": "b"}), "b")


class ConfigTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "sub" / "config.json"

    def test_write_keeps_custom_limits_and_is_private(self):
        self.path.parent.mkdir()
        self.path.write_text('{"pause_gpu": 83}')
        service.write_config(self.path, service._read_config_dict(self.path), "x")
        self.assertEqual(json.loads(self.path.read_text()), {"pause_gpu": 83, "container": "x"})
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_write_refuses_invalid_config(self):
        with self.assertRaisesRegex(service.SetupError, "invalid config"):
            service.write_config(self.path, {"pause_gpu": 999}, "x")
        with self.assertRaisesRegex(service.SetupError, "invalid config"):
            service.write_config(self.path, {"typo": 1}, "x")
        with self.assertRaisesRegex(service.SetupError, "invalid config"):
            service.write_config(self.path, {}, "--help")

    def test_unreadable_config_is_reported(self):
        self.path.parent.mkdir()
        for content in ("not json", "[1]"):
            self.path.write_text(content)
            with self.assertRaises(service.SetupError):
                service._read_config_dict(self.path)

    def test_missing_config_is_empty(self):
        self.assertEqual(service._read_config_dict(self.path), {})


class UnitTests(unittest.TestCase):
    def test_renders_python_path_and_hardening(self):
        unit = service.render_unit("/home/me/.local/share/uv/tools/vllm-thermald/bin/python")
        self.assertIn("ExecStart=/home/me/.local/share/uv/tools/vllm-thermald/bin/python -u -m vllm_thermald run", unit)
        for setting in ("NoNewPrivileges=yes", "ProtectSystem=strict", "WatchdogSec=60", "Restart=always"):
            self.assertIn(setting, unit)

    def test_refuses_paths_that_could_inject_unit_settings(self):
        for path in ("/a b/python", "/a\nExecStartPre=/bin/evil", "/a;b", "/a%h"):
            with self.assertRaises(service.SetupError):
                service.render_unit(path)


class InstallTests(unittest.TestCase):
    def test_preflight_failure_stops_before_changing_anything(self):
        with mock.patch.object(service, "checks", return_value=[("docker access", False, "denied")]), \
                mock.patch.object(service, "write_config") as write:
            with self.assertRaisesRegex(service.SetupError, "docker access: denied"):
                service.install("x")
        write.assert_not_called()

    def test_missing_container_stops_before_changing_anything(self):
        with mock.patch.object(service, "checks", return_value=[]), \
                mock.patch.object(service, "_read_config_dict", return_value={}), \
                mock.patch.object(service.Docker, "status", return_value="missing"), \
                mock.patch.object(service, "write_config") as write:
            with self.assertRaisesRegex(service.SetupError, "not found"):
                service.install("x")
        write.assert_not_called()


class CliTests(unittest.TestCase):
    def test_setup_error_is_a_clean_exit(self):
        with mock.patch.object(service, "doctor", side_effect=service.SetupError("nope")), \
                mock.patch("sys.stderr") as stderr:
            self.assertEqual(cli.main(["doctor"]), 1)
        self.assertTrue(stderr.write.called)

    def test_hold_and_release(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cli.daemon, "STATE_DIR", Path(tmp)), \
                mock.patch("builtins.print"):
            cli.main(["hold"])
            self.assertTrue((Path(tmp) / "hold").exists())
            cli.main(["release"])
            self.assertFalse((Path(tmp) / "hold").exists())

    def test_every_command_is_wired(self):
        for command in ("uninstall", "doctor", "status", "hold", "release", "check-config", "run", "mcp", "install"):
            with self.assertRaises(SystemExit) as ctx, mock.patch("sys.stdout"):
                cli.main([command, "--help"])
            self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
