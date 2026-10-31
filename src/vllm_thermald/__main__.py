# Copyright 2026 koderizer
# SPDX-License-Identifier: Apache-2.0
"""The vllm-thermald command."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from typing import Optional

from . import __version__, daemon, mcp_server, service


def show_status() -> int:
    try:
        s = json.loads((daemon.STATE_DIR / "status.json").read_text())
        age = time.time() - s["updated"]
    except (OSError, ValueError, KeyError, TypeError):
        print("no status yet; is the daemon running?")
        return 1
    stale = age > daemon.STALE_AFTER_S
    print(f"container : {s.get('container')} ({s.get('container_status')})")
    print(f"sensors   : gpu {s.get('gpu_c')}C, board {s.get('board_c')}C, {s.get('power')}, util {s.get('util')}")
    print(f"hold      : {'on' if s.get('hold') else 'off'}")
    print(f"actions   : {s.get('counts')}")
    print(f"updated   : {age:.0f}s ago" + ("  ** STALE: daemon may be down **" if stale else ""))
    return 1 if stale else 0


def set_hold(on: bool) -> int:
    daemon.prepare_state_dir(daemon.STATE_DIR)
    hold_file = daemon.STATE_DIR / "hold"
    hold_file.touch() if on else hold_file.unlink(missing_ok=True)
    print(f"hold {'on' if on else 'off'}")
    return 0


def check_config() -> int:
    try:
        print(json.dumps(dataclasses.asdict(daemon.load_config()), indent=2))
        return 0
    except ValueError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm-thermald", description="Keeps a vLLM container from overheating.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install", help="set up and start the service")
    install.add_argument("--container", help="vLLM container name (detected if omitted)")
    commands.add_parser("uninstall", help="stop and remove the service")
    commands.add_parser("doctor", help="check this machine has what the daemon needs")
    commands.add_parser("status", help="show temperatures and actions")
    commands.add_parser("hold", help="stop vLLM being started or unpaused; heat protection stays active")
    commands.add_parser("release", help="undo hold")
    commands.add_parser("check-config", help="validate the config file")
    commands.add_parser("run", help="run the daemon in the foreground (the service does this)")
    commands.add_parser("mcp", help="run the read-only MCP server on stdio")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    try:
        if args.command == "install":
            return service.install(args.container)
        actions = {
            "uninstall": service.uninstall, "doctor": service.doctor, "status": show_status,
            "hold": lambda: set_hold(True), "release": lambda: set_hold(False), "check-config": check_config,
            "run": daemon.run, "mcp": mcp_server.serve,
        }
        return actions[args.command]()
    except service.SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
