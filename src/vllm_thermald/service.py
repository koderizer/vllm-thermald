# Copyright 2026 koderizer
# SPDX-License-Identifier: Apache-2.0
"""One-command setup: preflight checks, container detection, and the systemd user service."""

from __future__ import annotations

import getpass
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .daemon import CONFIG_PATH, STATE_DIR, Config, Docker, read_sensors

UNIT_NAME = "vllm-thermald.service"
UNIT_PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd/user" / UNIT_NAME
SAFE_UNIT_PATH = re.compile(r"^[\w./+-]+$")

UNIT = """\
[Unit]
Description=Thermal manager for the vLLM container
StartLimitIntervalSec=0

[Service]
Type=notify
NotifyAccess=main
ExecStart={python} -u -m vllm_thermald run
Restart=always
RestartSec=2
WatchdogSec=60
# Signal only the daemon on stop, so its final docker and nvidia-smi calls can finish.
KillMode=mixed
TimeoutStopSec=30
NoNewPrivileges=yes
UMask=0077
ProtectSystem=strict
PrivateTmp=yes
StateDirectory=vllm-thermald
StateDirectoryMode=0700

[Install]
WantedBy=default.target
"""


class SetupError(Exception):
    """A problem the user can fix, reported without a traceback."""


def _run(*cmd: str, timeout: float = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(cmd, 1, "", str(exc))


def checks() -> list:
    """(name, passed, detail) for everything the daemon needs."""
    sensors = read_sensors()
    docker = _run("docker", "ps", "-q")
    systemd = _run("systemctl", "--user", "show-environment")
    return [
        ("systemd user session", systemd.returncode == 0, systemd.stderr.strip() or "ok"),
        ("docker access", docker.returncode == 0, docker.stderr.strip()[:120] or "ok"),
        ("GPU temperature", sensors.gpu is not None, f"{sensors.gpu}C" if sensors.gpu is not None else "nvidia-smi gave no reading"),
        ("board temperature", sensors.board is not None, f"{sensors.board}C" if sensors.board is not None else "no /sys/class/thermal reading"),
    ]


def doctor() -> int:
    results = checks()
    for name, passed, detail in results:
        print(f"[{'ok' if passed else 'FAIL'}] {name}: {detail}")
    return 0 if all(passed for _, passed, _ in results) else 1


def choose_container(listing: str) -> str:
    """Picks the vLLM container from `docker ps -a` lines of name, image, state."""
    found = []
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and "vllm" in f"{parts[0]} {parts[1]}".lower():
            found.append(parts)
    running = [name for name, _, state in found if state == "running"]
    if len(running) == 1:
        return running[0]
    if not running and len(found) == 1:
        return found[0][0]
    if not found:
        raise SetupError("no vLLM container found. Create it first, then pass --container NAME if it isn't detected.")
    options = ", ".join(name for name, _, _ in found)
    raise SetupError(f"more than one possible vLLM container ({options}). Choose one with --container NAME.")


def _read_config_dict(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise SetupError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SetupError(f"{path} must be a JSON object")
    return data


def resolve_container(requested: Optional[str], config: dict) -> str:
    if requested:
        return requested
    if config.get("container"):
        return config["container"]
    listing = _run("docker", "ps", "-a", "--format", "{{.Names}}\t{{.Image}}\t{{.State}}")
    return choose_container(listing.stdout)


def write_config(path: Path, config: dict, container: str) -> None:
    merged = {**config, "container": container}
    try:
        Config(**merged)
    except (TypeError, ValueError) as exc:
        raise SetupError(f"invalid config: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(".tmp")
    tmp.touch(mode=0o600)
    tmp.write_text(json.dumps(merged, indent=2) + "\n")
    tmp.replace(path)


def render_unit(python: str) -> str:
    if not SAFE_UNIT_PATH.match(python):
        raise SetupError(f"the Python path has characters unsafe in a unit file: {python}")
    return UNIT.format(python=python)


def _systemctl(*args: str) -> None:
    p = _run("systemctl", "--user", *args)
    if p.returncode != 0:
        raise SetupError(f"systemctl --user {' '.join(args)} failed: {p.stderr.strip()}")


def install(requested: Optional[str] = None) -> int:
    failed = [f"{name}: {detail}" for name, passed, detail in checks() if not passed]
    if failed:
        raise SetupError("preflight failed:\n  " + "\n  ".join(failed))
    config = _read_config_dict(CONFIG_PATH)
    container = resolve_container(requested, config)
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$", container) or Docker(container).status() == "missing":
        raise SetupError(f"container not found: {container}")
    write_config(CONFIG_PATH, config, container)

    UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIT_PATH.unlink(missing_ok=True)  # an older install may have left a symlink here
    UNIT_PATH.write_text(render_unit(sys.executable))
    _systemctl("daemon-reload")
    _systemctl("enable", UNIT_NAME)
    _systemctl("restart", UNIT_NAME)
    time.sleep(2)
    _systemctl("is-active", UNIT_NAME)

    print(f"Protecting container: {container}")
    print(f"Limits: {CONFIG_PATH} | State: {STATE_DIR}")
    print("After this, the daemon starts vLLM whenever it is stopped and cool, including at boot.")
    print("Run `vllm-thermald hold` before stopping vLLM on purpose.")
    if _run("loginctl", "enable-linger", getpass.getuser()).returncode != 0:
        print(f"Warning: to start at boot without a login, run: sudo loginctl enable-linger {getpass.getuser()}")
    return 0


def uninstall() -> int:
    _run("systemctl", "--user", "disable", "--now", UNIT_NAME)
    UNIT_PATH.unlink(missing_ok=True)
    _systemctl("daemon-reload")
    print(f"Removed the service. Config and history remain in {CONFIG_PATH.parent} and {STATE_DIR}.")
    return 0
