import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vllm_thermald import daemon as vllm_thermald
from vllm_thermald.daemon import KILL, PAUSE, START, UNPAUSE, Config, Controller, Reading, parse_nvidia

COOL = Reading(gpu=45, board=50)
WARM = Reading(gpu=75, board=80)  # between resume and pause
HOT = Reading(gpu=85, board=80)
CRITICAL = Reading(gpu=90, board=80)
NO_SENSORS = Reading()


def action(ctl, reading, status, now, hold=False):
    decision = ctl.step(reading, status, now, hold)
    return decision.action if decision else None


class RunningTests(unittest.TestCase):
    def setUp(self):
        self.ctl = Controller(Config())

    def test_normal_temperature_takes_no_action(self):
        self.assertIsNone(action(self.ctl, WARM, "running", 0))

    def test_pauses_at_gpu_threshold(self):
        self.assertEqual(action(self.ctl, HOT, "running", 0), PAUSE)

    def test_pauses_at_board_threshold(self):
        self.assertEqual(action(self.ctl, Reading(gpu=60, board=92), "running", 0), PAUSE)

    def test_kills_at_critical_gpu(self):
        self.assertEqual(action(self.ctl, CRITICAL, "running", 0), KILL)

    def test_kills_at_critical_board(self):
        self.assertEqual(action(self.ctl, Reading(gpu=60, board=96), "running", 0), KILL)

    def test_hardware_slowdown_pauses_only_when_sustained(self):
        slow = Reading(gpu=70, board=75, hw_slowdown=True)
        self.assertIsNone(action(self.ctl, slow, "running", 0))
        self.assertIsNone(action(self.ctl, slow, "running", 1))
        self.assertEqual(action(self.ctl, slow, "running", 2), PAUSE)

    def test_hardware_slowdown_count_resets(self):
        slow = Reading(gpu=70, board=75, hw_slowdown=True)
        action(self.ctl, slow, "running", 0)
        action(self.ctl, slow, "running", 1)
        action(self.ctl, COOL, "running", 2)
        self.assertIsNone(action(self.ctl, slow, "running", 3))

    def test_one_failed_sensor_still_protects(self):
        self.assertEqual(action(self.ctl, Reading(gpu=None, board=92), "running", 0), PAUSE)

    def test_failed_pause_is_retried(self):
        self.assertEqual(action(self.ctl, HOT, "running", 0), PAUSE)
        self.assertEqual(action(self.ctl, HOT, "running", 1), PAUSE)


class PausedTests(unittest.TestCase):
    def setUp(self):
        self.ctl = Controller(Config())
        self.ctl.step(COOL, "paused", 0)

    def test_stays_paused_for_minimum_time_even_when_cool(self):
        self.assertIsNone(action(self.ctl, COOL, "paused", 29))

    def test_unpauses_once_cool_after_minimum_time(self):
        self.assertEqual(action(self.ctl, COOL, "paused", 30), UNPAUSE)

    def test_stays_paused_between_resume_and_pause(self):
        self.assertIsNone(action(self.ctl, WARM, "paused", 60))

    def test_both_sensors_must_be_cool_to_unpause(self):
        self.assertIsNone(action(self.ctl, Reading(gpu=50, board=76), "paused", 60))

    def test_no_kill_on_lagging_reading_right_after_pause(self):
        self.assertIsNone(action(self.ctl, CRITICAL, "paused", 4))

    def test_kills_if_still_critical_after_grace(self):
        self.assertEqual(action(self.ctl, CRITICAL, "paused", 5), KILL)

    def test_kills_if_not_cooling(self):
        self.assertEqual(action(self.ctl, WARM, "paused", 600), KILL)

    def test_cool_at_max_pause_unpauses_rather_than_kills(self):
        self.assertEqual(action(self.ctl, COOL, "paused", 600), UNPAUSE)

    def test_hold_prevents_unpause(self):
        self.assertIsNone(action(self.ctl, COOL, "paused", 60, hold=True))

    def test_hold_does_not_prevent_kill(self):
        self.assertEqual(action(self.ctl, CRITICAL, "paused", 60, hold=True), KILL)


class StartTests(unittest.TestCase):
    def setUp(self):
        self.ctl = Controller(Config())

    def test_starts_stopped_container_when_cool(self):
        self.assertEqual(action(self.ctl, COOL, "exited", 0), START)

    def test_waits_until_cool(self):
        self.assertIsNone(action(self.ctl, Reading(gpu=61, board=50), "exited", 0))
        self.assertIsNone(action(self.ctl, Reading(gpu=50, board=66), "exited", 1))

    def test_hold_prevents_start(self):
        self.assertIsNone(action(self.ctl, COOL, "exited", 0, hold=True))

    def test_ignores_missing_and_unknown(self):
        self.assertIsNone(action(self.ctl, COOL, "missing", 0))
        self.assertIsNone(action(self.ctl, COOL, "unknown", 0))

    def test_cooldown_after_kill_doubles(self):
        ctl = self.ctl
        ctl.step(COOL, "running", 0)
        self.assertEqual(action(ctl, CRITICAL, "running", 10000), KILL)
        self.assertIsNone(action(ctl, COOL, "exited", 10001))
        self.assertIsNone(action(ctl, COOL, "exited", 10300))
        self.assertEqual(action(ctl, COOL, "exited", 10301), START)
        action(ctl, COOL, "running", 10302)
        self.assertEqual(action(ctl, CRITICAL, "running", 10303), KILL)
        self.assertIsNone(action(ctl, COOL, "exited", 10304))
        self.assertIsNone(action(ctl, COOL, "exited", 10903))
        self.assertEqual(action(ctl, COOL, "exited", 10904), START)

    def test_cooldown_is_capped(self):
        ctl = self.ctl
        for i in range(12):
            ctl.step(COOL, "running", i * 10)
            ctl.step(COOL, "exited", i * 10 + 1)
        self.assertEqual(ctl.cooldown, ctl.cfg.max_cooldown_s)

    def test_early_exit_backs_off_to_prevent_restart_loop(self):
        self.ctl.step(COOL, "running", 0)
        self.assertIsNone(action(self.ctl, COOL, "exited", 60))
        self.assertEqual(action(self.ctl, COOL, "exited", 360), START)

    def test_clean_exit_after_long_run_restarts_immediately(self):
        self.ctl.step(COOL, "running", 0)
        self.assertEqual(action(self.ctl, COOL, "exited", 700), START)

    def test_stable_run_resets_cooldown(self):
        ctl = self.ctl
        ctl.step(COOL, "running", 0)
        ctl.step(COOL, "exited", 1)
        self.assertEqual(ctl.cooldown, 600)
        ctl.step(COOL, "running", 400)
        ctl.step(COOL, "running", 400 + 3600)
        self.assertEqual(ctl.cooldown, 300)

    def test_pause_does_not_count_as_exit_or_reset_uptime(self):
        ctl = self.ctl
        ctl.step(COOL, "running", 0)
        ctl.step(HOT, "running", 700)
        ctl.step(HOT, "paused", 701)
        ctl.step(COOL, "running", 760)
        self.assertEqual(ctl.running_since, 0)


class SensorLossTests(unittest.TestCase):
    def test_kills_after_sustained_total_sensor_loss(self):
        ctl = Controller(Config())
        self.assertIsNone(action(ctl, NO_SENSORS, "running", 0))
        self.assertIsNone(action(ctl, NO_SENSORS, "running", 29))
        self.assertEqual(action(ctl, NO_SENSORS, "running", 30), KILL)

    def test_sensor_recovery_resets_timer(self):
        ctl = Controller(Config())
        action(ctl, NO_SENSORS, "running", 0)
        action(ctl, COOL, "running", 20)
        self.assertIsNone(action(ctl, NO_SENSORS, "running", 40))
        self.assertIsNone(action(ctl, NO_SENSORS, "running", 69))

    def test_never_starts_without_sensors(self):
        self.assertIsNone(action(Controller(Config()), NO_SENSORS, "exited", 0))


class ReviewRegressionTests(unittest.TestCase):
    def setUp(self):
        self.ctl = Controller(Config())

    def test_transient_unknown_does_not_erase_kill_backoff(self):
        ctl = self.ctl
        ctl.step(COOL, "running", 0)
        self.assertEqual(action(ctl, CRITICAL, "running", 10000), KILL)
        self.assertIsNone(action(ctl, COOL, "unknown", 10001))
        self.assertIsNone(action(ctl, COOL, "exited", 10002))
        self.assertIsNone(action(ctl, COOL, "exited", 10301))
        self.assertEqual(action(ctl, COOL, "exited", 10302), START)

    def test_transient_unknown_does_not_reset_pause_clock(self):
        ctl = self.ctl
        ctl.step(WARM, "paused", 0)
        ctl.step(WARM, "unknown", 599)
        self.assertEqual(action(ctl, WARM, "paused", 600), KILL)

    def test_transient_unknown_does_not_reset_uptime(self):
        ctl = self.ctl
        ctl.step(COOL, "running", 0)
        ctl.step(COOL, "unknown", 500)
        ctl.step(COOL, "running", 501)
        self.assertEqual(ctl.running_since, 0)

    def test_old_kill_does_not_penalize_later_clean_exit(self):
        ctl = self.ctl
        ctl.step(COOL, "running", 0)
        self.assertEqual(action(ctl, CRITICAL, "running", 1000), KILL)  # the kill fails
        ctl.step(COOL, "running", 1001)
        self.assertEqual(action(ctl, COOL, "exited", 5000), START)

    def test_missing_gpu_never_counts_as_cool(self):
        no_gpu = Reading(gpu=None, board=50)
        self.assertIsNone(action(self.ctl, no_gpu, "exited", 0))
        ctl = Controller(Config())
        ctl.step(no_gpu, "paused", 0)
        self.assertIsNone(action(ctl, no_gpu, "paused", 60))

    def test_missing_board_never_counts_as_cool(self):
        self.assertIsNone(action(self.ctl, Reading(gpu=40, board=None), "exited", 0))

    def test_one_missing_sensor_pauses_after_timeout(self):
        no_gpu = Reading(gpu=None, board=50)
        self.assertIsNone(action(self.ctl, no_gpu, "running", 0))
        self.assertIsNone(action(self.ctl, no_gpu, "running", 29))
        self.assertEqual(action(self.ctl, no_gpu, "running", 30), PAUSE)

    def test_paused_with_missing_sensor_is_killed_at_max_pause(self):
        no_gpu = Reading(gpu=None, board=50)
        self.ctl.step(no_gpu, "paused", 0)
        self.assertEqual(action(self.ctl, no_gpu, "paused", 600), KILL)

    def test_failed_start_backs_off(self):
        ctl = self.ctl
        self.assertEqual(action(ctl, COOL, "exited", 0), START)
        self.assertIsNone(action(ctl, COOL, "exited", 1))
        self.assertIsNone(action(ctl, COOL, "exited", 300))
        self.assertEqual(action(ctl, COOL, "exited", 301), START)
        self.assertIsNone(action(ctl, COOL, "exited", 302))
        self.assertIsNone(action(ctl, COOL, "exited", 901))
        self.assertEqual(action(ctl, COOL, "exited", 902), START)

    def test_container_dying_between_polls_backs_off(self):
        ctl = self.ctl
        ctl.step(COOL, "exited", 0)
        self.assertIsNone(action(ctl, COOL, "exited", 1))

    def test_successful_start_clears_pending(self):
        ctl = self.ctl
        ctl.step(COOL, "exited", 0)
        ctl.step(COOL, "running", 1)
        self.assertFalse(ctl.start_pending)

    def test_hw_slowdown_blocks_unpause_and_ends_in_kill(self):
        slow = Reading(gpu=70, board=75, hw_slowdown=True)
        self.ctl.step(slow, "paused", 0)
        self.assertIsNone(action(self.ctl, slow, "paused", 30))
        self.assertEqual(action(self.ctl, slow, "paused", 600), KILL)

    def test_restarting_container_is_killed_when_critical(self):
        self.assertEqual(action(self.ctl, CRITICAL, "restarting", 0), KILL)
        self.assertIsNone(action(Controller(Config()), COOL, "restarting", 0))

    def test_restored_backoff_blocks_start(self):
        self.ctl.restore_backoff(100, 1200, now=0)
        self.assertIsNone(action(self.ctl, COOL, "exited", 99))
        self.assertEqual(action(self.ctl, COOL, "exited", 100), START)
        self.assertEqual(self.ctl.cooldown, 1200)

    def test_restored_backoff_is_clamped(self):
        self.ctl.restore_backoff(10 ** 9, 10 ** 9, now=0)
        self.assertEqual(self.ctl.not_before, 3600)
        self.assertEqual(self.ctl.cooldown, 3600)
        self.ctl.restore_backoff(-5, 0, now=0)
        self.assertEqual(self.ctl.not_before, 0)
        self.assertEqual(self.ctl.cooldown, 300)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_backoff_survives_a_new_controller_and_clock(self):
        old = Controller(Config())
        old.step(COOL, "running", 0)
        old.step(COOL, "exited", 1)
        vllm_thermald.save_backoff(self.dir, old, mono_now=1, wall_now=5000)
        new = Controller(Config())
        vllm_thermald.restore_backoff(self.dir, new, mono_now=9000, wall_now=5100)
        self.assertEqual(new.not_before, 9200)
        self.assertEqual(new.cooldown, 600)

    def test_damaged_backoff_is_ignored(self):
        for content in ("not json", "[]", '{"restart_at": "x", "cooldown": 1}', "{}"):
            (self.dir / "backoff.json").write_text(content)
            ctl = Controller(Config())
            vllm_thermald.restore_backoff(self.dir, ctl, 0, 0)
            self.assertEqual(ctl.not_before, 0, content)

    def test_only_one_daemon_can_hold_the_lock(self):
        first = vllm_thermald.acquire_lock(self.dir)
        self.addCleanup(first.close)
        self.assertIsNone(vllm_thermald.acquire_lock(self.dir))

    def test_bad_config_falls_back_to_defaults(self):
        path = self.dir / "config.json"
        for content in ('{"pause_gpuu": 80}', "not json", "[]", '{"pause_gpu": "hot"}', '{"pause_gpu": 95}'):
            path.write_text(content)
            cfg, error = vllm_thermald.load_config_or_defaults(path)
            self.assertEqual(cfg, Config(), content)
            self.assertIn("config.json", error)

    def test_bad_config_keeps_the_configured_container(self):
        path = self.dir / "config.json"
        path.write_text('{"container": "my-vllm", "pause_gpu": 999}')
        cfg, error = vllm_thermald.load_config_or_defaults(path)
        self.assertEqual(cfg, Config(container="my-vllm"))
        self.assertIsNotNone(error)

    def test_bad_config_with_invalid_container_uses_default_container(self):
        path = self.dir / "config.json"
        path.write_text('{"container": "--help"}')
        self.assertEqual(vllm_thermald.load_config_or_defaults(path)[0], Config())

    def test_missing_config_is_not_an_error(self):
        self.assertEqual(vllm_thermald.load_config_or_defaults(self.dir / "none.json"), (Config(), None))

    def test_identical_events_are_rate_limited(self):
        log = vllm_thermald.EventLog(self.dir / "events.jsonl")
        for _ in range(3):
            log.write("pause", reason="hot", ok=False)
        log.write("pause", reason="hot", ok=True)
        self.assertEqual(len((self.dir / "events.jsonl").read_text().splitlines()), 2)

    def test_prepare_state_dir_is_private(self):
        target = self.dir / "state"
        vllm_thermald.prepare_state_dir(target)
        self.assertEqual(target.stat().st_mode & 0o777, 0o700)


class ShutdownTests(unittest.TestCase):
    def daemon(self, status, reading, hold=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = vllm_thermald.Daemon(Config(), Path(tmp.name))
        if hold:
            d.hold_file.touch()
        d.docker = mock.Mock()
        d.docker.status.return_value = status
        patcher = mock.patch.object(vllm_thermald, "read_sensors", return_value=reading)
        patcher.start()
        self.addCleanup(patcher.stop)
        return d

    def test_unpauses_when_cool(self):
        d = self.daemon("paused", COOL)
        d.shutdown()
        d.docker.act.assert_called_once_with(UNPAUSE)

    def test_respects_hold(self):
        d = self.daemon("paused", COOL, hold=True)
        d.shutdown()
        d.docker.act.assert_not_called()

    def test_stays_paused_when_hot(self):
        d = self.daemon("paused", WARM)
        d.shutdown()
        d.docker.act.assert_not_called()

    def test_stays_paused_without_sensors(self):
        d = self.daemon("paused", NO_SENSORS)
        d.shutdown()
        d.docker.act.assert_not_called()

    def test_uses_current_status_rather_than_last_loop(self):
        d = self.daemon("paused", COOL)
        d.last_status = "running"
        d.shutdown()
        d.docker.act.assert_called_once_with(UNPAUSE)


class TickTests(unittest.TestCase):
    def test_failed_action_is_not_counted_and_loop_continues(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = vllm_thermald.Daemon(Config(), Path(tmp.name))
        d.docker = mock.Mock()
        d.docker.status.return_value = "running"
        d.docker.act.return_value = (False, "docker broke")
        with mock.patch.object(vllm_thermald, "read_sensors", return_value=HOT):
            d.tick()
        self.assertEqual(d.counts[PAUSE], 0)
        self.assertTrue((Path(tmp.name) / "status.json").exists())


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        Config()

    def test_rejects_wrong_types(self):
        for kwargs in ({"pause_gpu": "85"}, {"pause_gpu": True}, {"container": 5}, {"poll_s": None}):
            with self.assertRaises(ValueError, msg=kwargs):
                Config(**kwargs)

    def test_rejects_unusable_timing(self):
        for kwargs in ({"poll_s": 0}, {"min_pause_s": -1}, {"hw_slowdown_samples": 0}, {"cooldown_s": 7200}):
            with self.assertRaises(ValueError, msg=kwargs):
                Config(**kwargs)

    def test_rejects_option_like_or_invalid_container_names(self):
        for name in ("--help", "-f", "a b", "", "x;y", "../x"):
            with self.assertRaises(ValueError, msg=name):
                Config(container=name)

    def test_vllm_url_must_be_loopback_http(self):
        for url in ("http://example.com:8000", "https://127.0.0.1", "file:///etc/passwd", "http://10.0.0.1"):
            with self.assertRaises(ValueError, msg=url):
                Config(vllm_url=url)
        Config(vllm_url="http://localhost:9000")

    def test_rejects_resume_above_pause(self):
        with self.assertRaises(ValueError):
            Config(resume_gpu=86)

    def test_rejects_kill_below_pause(self):
        with self.assertRaises(ValueError):
            Config(kill_board=90)


class ParseTests(unittest.TestCase):
    def test_parses_real_output(self):
        got = parse_nvidia("53, Not Active, Active, 5.67 W, 0 %\n")
        self.assertEqual(got["gpu"], 53)
        self.assertFalse(got["hw_slowdown"])
        self.assertTrue(got["sw_slowdown"])

    def test_not_available_temperature(self):
        self.assertIsNone(parse_nvidia("[N/A], Not Active, Not Active, [N/A], [N/A]\n")["gpu"])

    def test_malformed_output(self):
        self.assertEqual(parse_nvidia("Failed to initialize NVML\n"), {})

    def test_empty_output_does_not_raise(self):
        self.assertEqual(parse_nvidia(""), {})

    def test_impossible_temperature_is_missing(self):
        self.assertIsNone(parse_nvidia("999, Not Active, Not Active, 1 W, 0 %\n")["gpu"])


if __name__ == "__main__":
    unittest.main()
