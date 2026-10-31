#!/usr/bin/env python3
"""Thermal manager for a vLLM Docker container.

Pausing is the normal response to heat: `docker pause` removes GPU load within
seconds, keeps the model loaded, and in-flight requests resume afterwards.
Killing is the last resort, because a restart reloads the model and runs
FlashInfer autotune, which is the hottest phase of the container's life.

    running --(hot)--> paused --(cool)--> running
       |                  |
       | (critical)       | (critical, or will not cool)
       v                  v
     killed --(cooldown elapsed and cold)--> running

The daemon fails safe: when a sensor is unavailable it treats temperatures as
unknown rather than cool, and an invalid config falls back to the defaults
rather than leaving the container unprotected.

Commands: run | status | hold | release | check-config
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import glob
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

log = logging.getLogger("vllm-thermald")

STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "vllm-thermald"
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "vllm-thermald"
CONFIG_PATH = CONFIG_DIR / "config.json"

PAUSE, UNPAUSE, KILL, START = "pause", "unpause", "kill", "start"
HOT_LOG_GPU_C, HOT_LOG_BOARD_C = 75, 85
CSV_MAX_BYTES = 20_000_000
EVENT_LOG_MAX_BYTES = 1_000_000
STALE_AFTER_S = 30  # longer than the worst-case loop: sensor + inspect + action timeouts
SENSOR_TIMEOUT_S = 5
DOCKER_STATUS_TIMEOUT_S = 5
DOCKER_ACTION_TIMEOUT_S = {PAUSE: 10, UNPAUSE: 10, KILL: 10, START: 30}
KILL_ATTRIBUTION_S = 30  # an exit this soon after a kill request is attributed to it
EVENT_REPEAT_S = 60  # identical events are written at most this often
MIN_VALID_C, MAX_VALID_C = -20, 150  # readings outside this range are treated as missing

UP_STATUSES = ("running", "paused")
UNCERTAIN_STATUSES = ("unknown", "restarting", "removing")  # say nothing about whether it exited
CONTAINER_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True)
class Config:
    """Defaults come from a run that died at GPU 89C / board 96C with thermal slowdown active."""

    container: str = "vllm-spark-128gb-qwen38-27b"
    vllm_url: str = "http://127.0.0.1:8000"
    poll_s: float = 1.0
    # Pause when either sensor reaches these.
    pause_gpu: int = 85
    pause_board: int = 92
    # Unpause when both sensors are at or below these.
    resume_gpu: int = 70
    resume_board: int = 75
    # Kill when either sensor reaches these.
    kill_gpu: int = 90
    kill_board: int = 96
    # (Re)start a stopped container only when both sensors are at or below these.
    start_gpu: int = 60
    start_board: int = 65
    hw_slowdown_samples: int = 3  # consecutive hardware-slowdown readings before pausing
    min_pause_s: float = 30  # avoids rapid pause/unpause cycling
    max_pause_s: float = 600  # still hot after this long means pausing is not working
    pause_grace_s: float = 5  # sensors lag; do not kill on the reading right after a pause
    sensor_loss_s: float = 30  # a sensor missing this long is treated as unsafe
    cooldown_s: float = 300  # wait after a kill or a failed run; doubles each time
    max_cooldown_s: float = 3600
    early_exit_s: float = 600  # exiting sooner than this counts as a failed run
    stable_s: float = 3600  # running this long resets the cooldown

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name in ("container", "vllm_url"):
                if not isinstance(value, str):
                    raise ValueError(f"{field.name} must be a string")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be a number")
            elif value < 0:
                raise ValueError(f"{field.name} must not be negative")
        if self.poll_s <= 0:
            raise ValueError("poll_s must be positive")
        if self.hw_slowdown_samples < 1:
            raise ValueError("hw_slowdown_samples must be at least 1")
        if self.cooldown_s > self.max_cooldown_s:
            raise ValueError("cooldown_s must not exceed max_cooldown_s")
        if not CONTAINER_NAME.match(self.container):
            raise ValueError(f"invalid container name: {self.container!r}")
        url = urlsplit(self.vllm_url)
        if url.scheme != "http" or url.hostname not in LOOPBACK_HOSTS:
            raise ValueError("vllm_url must be an http:// loopback address")
        ordered = (
            self.start_gpu <= self.resume_gpu < self.pause_gpu < self.kill_gpu
            and self.start_board <= self.resume_board < self.pause_board < self.kill_board
        )
        if not ordered:
            raise ValueError("thresholds must satisfy start <= resume < pause < kill")


@dataclass(frozen=True)
class Reading:
    gpu: Optional[int] = None
    board: Optional[int] = None
    hw_slowdown: bool = False
    sw_slowdown: bool = False
    power: str = ""
    util: str = ""


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


class Controller:
    """Decides what to do from sensors and container status. Performs no I/O.

    State is derived from observed container status rather than from the
    actions it requested, so a failed docker command is simply retried.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.was_up = False
        self.paused_since: Optional[float] = None
        self.running_since: Optional[float] = None
        self.not_before = 0.0
        self.cooldown = cfg.cooldown_s
        self.hw_count = 0
        self.sensor_missing_since: Optional[float] = None
        self.last_kill_at: Optional[float] = None
        self.start_pending = False

    def _at_or_above(self, r: Reading, gpu: int, board: int) -> bool:
        return (r.gpu is not None and r.gpu >= gpu) or (r.board is not None and r.board >= board)

    def at_or_below(self, r: Reading, gpu: int, board: int) -> bool:
        """True only when both sensors confirm it. A missing sensor is never treated as cool."""
        return r.gpu is not None and r.board is not None and r.gpu <= gpu and r.board <= board

    def _backoff(self, now: float) -> None:
        self.not_before = now + self.cooldown
        self.cooldown = min(self.cooldown * 2, self.cfg.max_cooldown_s)

    def _observe(self, status: str, now: float) -> None:
        if status in UNCERTAIN_STATUSES:
            return  # keep every timer: a transient Docker failure must not erase a backoff
        up = status in UP_STATUSES
        if up:
            self.start_pending = False
            if self.running_since is None:
                self.running_since = now
        if status == "running" and self.running_since is not None:
            if now - self.running_since >= self.cfg.stable_s:
                self.cooldown = self.cfg.cooldown_s
        if status != "paused":
            self.paused_since = None
        elif self.paused_since is None:
            self.paused_since = now
        if not up and (self.was_up or self.start_pending):
            ran_for = now - self.running_since if self.running_since is not None else 0.0
            killed = self.last_kill_at is not None and now - self.last_kill_at <= KILL_ATTRIBUTION_S
            if killed or self.start_pending or ran_for < self.cfg.early_exit_s:
                self._backoff(now)
            self.start_pending = False
        if not up:
            self.running_since = None
        self.was_up = up

    def snapshot(self, now: float) -> dict:
        """Elapsed-time state as durations, since monotonic timestamps mean nothing to readers."""
        since = lambda t: None if t is None else round(now - t, 1)
        return {
            "paused_for_s": since(self.paused_since),
            "running_for_s": since(self.running_since),
            "restart_wait_s": round(max(0.0, self.not_before - now), 1),
            "next_cooldown_s": self.cooldown,
            "hw_slowdown_consecutive": self.hw_count,
            "sensor_missing_for_s": since(self.sensor_missing_since),
        }

    def restore_backoff(self, remaining_s: float, cooldown_s: float, now: float) -> None:
        """Reinstates a persisted backoff, clamped so a damaged file cannot disable vLLM for long."""
        self.not_before = now + max(0.0, min(remaining_s, self.cfg.max_cooldown_s))
        self.cooldown = max(self.cfg.cooldown_s, min(cooldown_s, self.cfg.max_cooldown_s))

    def _kill(self, reason: str, now: float) -> Decision:
        self.last_kill_at = now
        return Decision(KILL, reason)

    def step(self, r: Reading, status: str, now: float, hold: bool = False) -> Optional[Decision]:
        self._observe(status, now)
        self.hw_count = self.hw_count + 1 if r.hw_slowdown else 0
        cfg = self.cfg

        missing = r.gpu is None or r.board is None
        if not missing:
            self.sensor_missing_since = None
        elif self.sensor_missing_since is None:
            self.sensor_missing_since = now
        sensors_lost = missing and now - self.sensor_missing_since >= cfg.sensor_loss_s

        if status == "unknown":
            return None  # Docker cannot act right now either
        if r.gpu is None and r.board is None:
            if sensors_lost and status in ("running", "paused", "restarting"):
                return self._kill(f"no temperature sensors for {cfg.sensor_loss_s:.0f}s", now)
            return None

        temps = f"gpu={r.gpu}C board={r.board}C"
        critical = self._at_or_above(r, cfg.kill_gpu, cfg.kill_board)

        if status == "restarting":
            return self._kill(f"critical {temps}", now) if critical else None

        if status == "running":
            if critical:
                return self._kill(f"critical {temps}", now)
            if self._at_or_above(r, cfg.pause_gpu, cfg.pause_board):
                return Decision(PAUSE, f"hot {temps}")
            if self.hw_count >= cfg.hw_slowdown_samples:
                return Decision(PAUSE, f"hardware thermal slowdown {temps}")
            if sensors_lost:
                return Decision(PAUSE, f"sensor unavailable for {cfg.sensor_loss_s:.0f}s {temps}")
            return None

        if status == "paused":
            paused_for = now - self.paused_since
            if paused_for >= cfg.pause_grace_s and critical:
                return self._kill(f"critical while paused {temps}", now)
            cool = self.at_or_below(r, cfg.resume_gpu, cfg.resume_board) and not r.hw_slowdown
            if paused_for >= cfg.max_pause_s and not cool:
                return self._kill(f"not cooling after {paused_for:.0f}s paused {temps}", now)
            if cool and paused_for >= cfg.min_pause_s and not hold:
                return Decision(UNPAUSE, f"cooled {temps}")
            return None

        if status in ("exited", "created") and not hold and now >= self.not_before:
            if self.at_or_below(r, cfg.start_gpu, cfg.start_board):
                self.start_pending = True
                return Decision(START, f"cool {temps}")
        return None


def _valid_temp(value: int) -> Optional[int]:
    return value if MIN_VALID_C <= value <= MAX_VALID_C else None


def parse_nvidia(out: str) -> dict:
    lines = out.splitlines()
    parts = [p.strip() for p in lines[0].split(",")] if lines else []
    if len(parts) < 5:
        return {}
    gpu = _valid_temp(int(parts[0])) if parts[0].lstrip("-").isdigit() else None
    return {
        "gpu": gpu,
        "hw_slowdown": parts[1] == "Active",
        "sw_slowdown": parts[2] == "Active",
        "power": parts[3],
        "util": parts[4],
    }


def read_board_temp() -> Optional[int]:
    temps = []
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            temp = _valid_temp(int(Path(path).read_text()) // 1000)
        except (OSError, ValueError):
            continue
        if temp is not None:
            temps.append(temp)
    return max(temps) if temps else None


def read_sensors() -> Reading:
    query = (
        "temperature.gpu,clocks_event_reasons.hw_thermal_slowdown,"
        "clocks_event_reasons.sw_thermal_slowdown,power.draw,utilization.gpu"
    )
    fields: dict = {"board": read_board_temp()}
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=SENSOR_TIMEOUT_S, check=True,
        ).stdout
        fields.update(parse_nvidia(out))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        log.debug("nvidia-smi failed: %s", exc)
    return Reading(**fields)


class Docker:
    def __init__(self, container: str) -> None:
        self.container = container

    def _run(self, timeout: float, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)

    def status(self) -> str:
        """Container status, `missing` if absent, or `unknown` if Docker cannot answer."""
        try:
            p = self._run(DOCKER_STATUS_TIMEOUT_S, "inspect", "-f", "{{.State.Status}}", self.container)
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        if p.returncode == 0:
            return p.stdout.strip() or "unknown"
        return "missing" if "no such" in p.stderr.lower() else "unknown"

    def act(self, action: str) -> tuple:
        """Returns (succeeded, error text)."""
        try:
            p = self._run(DOCKER_ACTION_TIMEOUT_S[action], action, self.container)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        return p.returncode == 0, p.stderr.strip()


def _append_synced(path: Path, line: str) -> None:
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _rotate(path: Path, suffix: str, max_bytes: int) -> None:
    if path.exists() and path.stat().st_size > max_bytes:
        path.replace(path.with_suffix(suffix))


class ThermalLog:
    """Synced CSV, so the readings just before a sudden power loss survive it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.last = 0.0

    def write(self, r: Reading, status: str, now: float) -> None:
        hot = (r.gpu or 0) >= HOT_LOG_GPU_C or (r.board or 0) >= HOT_LOG_BOARD_C
        if not hot and now - self.last < 5:
            return
        self.last = now
        _rotate(self.path, ".csv.1", CSV_MAX_BYTES)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        _append_synced(self.path, f"{stamp},{r.gpu},{r.board},{r.power},{r.util},{r.hw_slowdown},{r.sw_slowdown},{status}\n")


class EventLog:
    """Synced JSONL of daemon actions. A write failure must never affect protection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.last_key: Optional[tuple] = None
        self.last_time = 0.0

    def write(self, event: str, r: Optional[Reading] = None, **fields) -> None:
        now = time.time()
        key = (event, fields.get("ok"), fields.get("reason"))
        if key == self.last_key and now - self.last_time < EVENT_REPEAT_S:
            return  # a retrying action must not push older events out of the log
        self.last_key, self.last_time = key, now
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "event": event, **fields}
        if r is not None:
            record.update(gpu_c=r.gpu, board_c=r.board, power=r.power, util=r.util)
        try:
            _rotate(self.path, ".jsonl.1", EVENT_LOG_MAX_BYTES)
            _append_synced(self.path, json.dumps(record) + "\n")
        except OSError as exc:
            log.error("cannot write event: %s", exc)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def write_status(
    state_dir: Path, r: Reading, status: str, ctl: Controller, hold: bool, counts: dict, now: float, started: float
) -> None:
    _write_json(state_dir / "status.json", {
        "updated": time.time(),
        "daemon_started": started,
        "daemon_pid": os.getpid(),
        "container": ctl.cfg.container,
        "container_status": status,
        "gpu_c": r.gpu,
        "board_c": r.board,
        "power": r.power,
        "util": r.util,
        "hw_slowdown": r.hw_slowdown,
        "sw_slowdown": r.sw_slowdown,
        "hold": hold,
        "counts": counts,
        "controller": ctl.snapshot(now),
        "config": dataclasses.asdict(ctl.cfg),
    })


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Raises ValueError on any invalid config."""
    try:
        if not path.exists():
            return Config()
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("config must be a JSON object")
        return Config(**data)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_config_or_defaults(path: Path = CONFIG_PATH) -> tuple:
    """A bad config must not leave vLLM unprotected. Returns (config, error or None)."""
    try:
        return load_config(path), None
    except ValueError as exc:
        return _defaults_for_same_container(path), str(exc)


def _defaults_for_same_container(path: Path) -> Config:
    """Default thresholds, but still the configured container: the wrong one would leave it unprotected."""
    try:
        return Config(container=json.loads(path.read_text())["container"])
    except (OSError, ValueError, KeyError, TypeError):
        return Config()


def save_backoff(state_dir: Path, ctl: Controller, mono_now: float, wall_now: float) -> None:
    _write_json(state_dir / "backoff.json", {
        "restart_at": wall_now + max(0.0, ctl.not_before - mono_now),
        "cooldown": ctl.cooldown,
    })


def restore_backoff(state_dir: Path, ctl: Controller, mono_now: float, wall_now: float) -> None:
    """Keeps a pending backoff across daemon restarts and reboots."""
    try:
        data = json.loads((state_dir / "backoff.json").read_text())
        ctl.restore_backoff(float(data["restart_at"]) - wall_now, float(data["cooldown"]), mono_now)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("ignoring damaged backoff state: %s", exc)


def acquire_lock(state_dir: Path):
    """Two daemons would fight over the container. Returns the open lock file, or None if held."""
    lock = open(state_dir / "daemon.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        return None
    return lock


def sd_notify(message: str) -> None:
    """Tells systemd the daemon is alive, so a hung loop gets restarted. No-op outside systemd."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(message.encode(), address)
    except OSError as exc:
        log.debug("sd_notify failed: %s", exc)


def prepare_state_dir(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_dir.chmod(0o700)


class Daemon:
    def __init__(self, cfg: Config, state_dir: Path) -> None:
        self.cfg = cfg
        self.state_dir = state_dir
        self.docker = Docker(cfg.container)
        self.ctl = Controller(cfg)
        self.csv = ThermalLog(state_dir / "thermal.csv")
        self.events = EventLog(state_dir / "events.jsonl")
        self.hold_file = state_dir / "hold"
        self.counts = {PAUSE: 0, UNPAUSE: 0, KILL: 0, START: 0}
        self.started = time.time()
        self.last_status: Optional[str] = None
        self.last_sensors: Optional[tuple] = None
        self.last_backoff: Optional[tuple] = None
        self.last_summary = time.monotonic()

    def _note_changes(self, r: Reading, status: str) -> None:
        if status != self.last_status:
            log.info("container status: %s -> %s", self.last_status, status)
            if status == "missing":
                log.error("container %s does not exist; create it with the vLLM launch script", self.cfg.container)
            self.events.write("container_status", r, previous=self.last_status, status=status)
            self.last_status = status
        sensors = (r.gpu is not None, r.board is not None)
        if sensors != self.last_sensors:
            if not all(sensors):
                log.error("temperature sensor unavailable: gpu=%s board=%s", r.gpu, r.board)
            elif self.last_sensors is not None:
                log.info("temperature sensors recovered")
            if self.last_sensors is not None or not all(sensors):
                self.events.write("sensors", r, gpu_ok=sensors[0], board_ok=sensors[1])
            self.last_sensors = sensors

    def _persist(self, r: Reading, status: str, hold: bool, now: float) -> None:
        wall = time.time()
        try:
            self.csv.write(r, status, wall)
            write_status(self.state_dir, r, status, self.ctl, hold, self.counts, now, self.started)
            backoff = (self.ctl.not_before, self.ctl.cooldown)
            if backoff != self.last_backoff:
                save_backoff(self.state_dir, self.ctl, now, wall)
                self.last_backoff = backoff
        except OSError as exc:
            log.error("cannot write state: %s", exc)

    def tick(self) -> None:
        now = time.monotonic()
        r = read_sensors()
        status = self.docker.status()
        hold = self.hold_file.exists()
        self._note_changes(r, status)
        decision = self.ctl.step(r, status, now, hold)
        if decision:
            ok, error = self.docker.act(decision.action)
            if ok:
                self.counts[decision.action] += 1
            level = logging.WARNING if ok else logging.ERROR
            log.log(level, "%s %s: %s (%s %s) %s", decision.action.upper(), "ok" if ok else "FAILED",
                    decision.reason, r.power, r.util, error)
            self.events.write(decision.action, r, reason=decision.reason, ok=ok)
        self._persist(r, status, hold, now)
        if now - self.last_summary >= 300:
            log.info("gpu=%sC board=%sC %s %s status=%s counts=%s", r.gpu, r.board, r.power, r.util, status, self.counts)
            self.last_summary = now

    def shutdown(self) -> None:
        """Nothing else would unfreeze the container, unless it is hot or held: then leave it to the next daemon."""
        r = read_sensors()
        status = self.docker.status()
        if status == "paused" and not self.hold_file.exists() and self.ctl.at_or_below(
            r, self.cfg.resume_gpu, self.cfg.resume_board
        ):
            log.info("exiting: unpausing container")
            self.docker.act(UNPAUSE)
        log.warning("exiting: vLLM has no thermal protection until the daemon runs again")
        self.events.write("daemon_stopped", r)


def run(state_dir: Path = STATE_DIR, config_path: Path = CONFIG_PATH) -> int:
    os.umask(0o077)
    prepare_state_dir(state_dir)
    lock = acquire_lock(state_dir)
    if lock is None:
        log.error("another vllm-thermald is already running")
        return 1
    cfg, error = load_config_or_defaults(config_path)
    daemon = Daemon(cfg, state_dir)
    restore_backoff(state_dir, daemon.ctl, time.monotonic(), time.time())
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.append(True))
    log.info("started: %s", dataclasses.asdict(cfg))
    daemon.events.write("daemon_started")
    if error:
        log.error("invalid config, using defaults: %s", error)
        daemon.events.write("config_error", reason=error)
    sd_notify("READY=1")
    while not stopping:
        try:
            daemon.tick()
        except Exception:  # one bad iteration must never end the protection loop
            log.exception("tick failed")
        sd_notify("WATCHDOG=1")
        time.sleep(cfg.poll_s)
    sd_notify("STOPPING=1")
    daemon.shutdown()
    lock.close()
    return 0


def show_status(state_dir: Path = STATE_DIR) -> int:
    try:
        s = json.loads((state_dir / "status.json").read_text())
        age = time.time() - s["updated"]
    except (OSError, ValueError, KeyError, TypeError):
        print("no status yet; is the daemon running?")
        return 1
    stale = age > STALE_AFTER_S
    print(f"container : {s.get('container')} ({s.get('container_status')})")
    print(f"sensors   : gpu {s.get('gpu_c')}C, board {s.get('board_c')}C, {s.get('power')}, util {s.get('util')}")
    print(f"hold      : {'on' if s.get('hold') else 'off'}")
    print(f"actions   : {s.get('counts')}")
    print(f"updated   : {age:.0f}s ago" + ("  ** STALE: daemon may be down **" if stale else ""))
    return 1 if stale else 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "status", "hold", "release", "check-config"], help=(
        "hold keeps vLLM from being started or unpaused (use it before stopping vLLM on purpose); "
        "heat protection stays active. release undoes it. check-config validates the config file."))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    if args.command == "run":
        return run()
    if args.command == "status":
        return show_status()
    if args.command == "check-config":
        try:
            print(json.dumps(dataclasses.asdict(load_config()), indent=2))
            return 0
        except ValueError as exc:
            print(f"invalid: {exc}", file=sys.stderr)
            return 1
    prepare_state_dir(STATE_DIR)
    hold_file = STATE_DIR / "hold"
    if args.command == "hold":
        hold_file.touch()
    else:
        hold_file.unlink(missing_ok=True)
    print(f"hold {'on' if args.command == 'hold' else 'off'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
