# Copyright 2026 koderizer
# SPDX-License-Identifier: Apache-2.0
"""Read-only MCP server (stdio) exposing thermal and vLLM readings to an agent.

It only reads the daemon's state files, so a fault here cannot affect protection.
No tool can change limits, hold, or control the container: an agent must not be
able to disable its own heat protection.
"""

from __future__ import annotations

import calendar
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from . import __version__
from .daemon import CONFIG_DIR, STALE_AFTER_S, STATE_DIR

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
OBSERVATIONS_PATH = CONFIG_DIR / "observations.json"
OBSERVATIONS_MAX_BYTES = 64_000
HTTP_MAX_BYTES = 8_000_000
HISTORY_TAIL_BYTES = 4_000_000
TREND_TAIL_BYTES = 64_000
SERIES_POINTS = 30
GAP_S = 60
SECRET_FLAG = re.compile(r"token$|api.?key|secret|password|credential|^auth", re.I)
SECRET_VALUE = re.compile(r"hf_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_-]{10,}|nvapi-[A-Za-z0-9_-]{10,}")

INSTRUCTIONS = (
    "Physical temperature and state of the machine running the LLM you are thinking with. "
    "This is hardware temperature, not sampling temperature. Read thermal_policy once to learn "
    "the rules, then check thermal_status before long or heavy work. Read-only: you cannot change "
    "limits or control the daemon; ask a human for that."
)

# Without this, an http_proxy variable would send these local requests to the proxy.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def read_status(state_dir: Path = STATE_DIR) -> Optional[dict]:
    try:
        data = json.loads((state_dir / "status.json").read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def derive_state(s: dict, now: float) -> str:
    if now - s.get("updated", 0) > STALE_AFTER_S:
        return "daemon_down"
    container = s.get("container_status")
    if container == "paused":
        return "paused_cooling"
    if container == "running":
        return "normal"
    if s.get("hold"):
        return "held_off"
    if container in ("exited", "created"):
        return "cooldown" if s.get("controller", {}).get("restart_wait_s", 0) > 0 else "waiting_for_cool"
    return "unavailable"


def protection_issues(s: dict, state: str) -> list:
    """Reasons the daemon may be unable to protect vLLM right now."""
    if state == "daemon_down":
        return ["the daemon is not updating, so nothing is watching the temperature"]
    checks = (
        (s.get("container_status") == "unknown", "Docker is not answering, so vLLM cannot be paused or killed"),
        (s.get("gpu_c") is None, "the GPU temperature is unavailable"),
        (s.get("board_c") is None, "the board temperature is unavailable"),
    )
    return [message for failed, message in checks if failed]


def derive_headroom(s: dict, state: str) -> dict:
    """Degrees before the daemon pauses vLLM, using whichever sensor is closest."""
    if state == "daemon_down" or s.get("gpu_c") is None or s.get("board_c") is None:
        return {"level": "unknown", "degrees_to_pause": None}
    cfg = s.get("config", {})
    if state != "normal" or "pause_gpu" not in cfg or "pause_board" not in cfg:
        return {"level": "none", "degrees_to_pause": None}
    margin = min(cfg["pause_gpu"] - s["gpu_c"], cfg["pause_board"] - s["board_c"])
    return {"level": "ample" if margin >= 20 else "moderate" if margin >= 10 else "low", "degrees_to_pause": margin}


def _tail_lines(path: Path, max_bytes: int) -> list:
    try:
        with open(path, "rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    lines = data.decode(errors="replace").splitlines()
    return lines[1:] if size > max_bytes else lines  # the first line may be cut off


def _int(value: str) -> Optional[int]:
    return int(value) if value.lstrip("-").isdigit() else None


def _power(value: str) -> Optional[float]:
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return None


def load_samples(state_dir: Path, since: float, names: tuple = ("thermal.csv.1", "thermal.csv"),
                 max_bytes: int = HISTORY_TAIL_BYTES) -> list:
    """CSV samples at or after `since`, oldest first."""
    rows = []
    for name in names:
        for line in _tail_lines(state_dir / name, max_bytes):
            parts = line.split(",")
            if len(parts) != 8:
                continue
            try:
                t = float(calendar.timegm(time.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ")))
            except ValueError:
                continue
            if t >= since:
                rows.append({
                    "t": t, "gpu": _int(parts[1]), "board": _int(parts[2]), "power": _power(parts[3]),
                    "hw": parts[5] == "True", "sw": parts[6] == "True", "status": parts[7],
                })
    return rows


def _stats(values: list) -> Optional[dict]:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"min": min(values), "max": max(values), "mean": round(sum(values) / len(values), 1)}


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _peak(group: list, key: str):
    return max((x[key] for x in group if x[key] is not None), default=None)


def summarize(rows: list, now: float, minutes: int) -> dict:
    if not rows:
        return {"window_minutes": minutes, "samples": 0}
    start, width = now - minutes * 60, minutes * 60 / SERIES_POINTS
    buckets: dict = {}
    for row in rows:
        buckets.setdefault(min(int((row["t"] - start) / width), SERIES_POINTS - 1), []).append(row)
    return {
        "window_minutes": minutes,
        "samples": len(rows),
        "gpu_c": _stats([x["gpu"] for x in rows]),
        "board_c": _stats([x["board"] for x in rows]),
        "power_w": _stats([x["power"] for x in rows]),
        "samples_with_sw_slowdown": sum(x["sw"] for x in rows),
        "samples_with_hw_slowdown": sum(x["hw"] for x in rows),
        "samples_paused": sum(x["status"] == "paused" for x in rows),
        "gaps_over_60s": [
            {"from": _iso(a["t"]), "to": _iso(b["t"]), "seconds": round(b["t"] - a["t"])}
            for a, b in zip(rows, rows[1:])
            if b["t"] - a["t"] > GAP_S
        ],
        "series": [
            {"t": _iso(g[0]["t"]), "gpu_max": _peak(g, "gpu"), "board_max": _peak(g, "board"),
             "power_max": _peak(g, "power")}
            for _, g in sorted(buckets.items())
        ],
    }


def trend(rows: list) -> Optional[dict]:
    """Degrees per minute over the supplied samples."""
    if len(rows) < 2 or rows[-1]["t"] - rows[0]["t"] < 10:
        return None
    minutes = (rows[-1]["t"] - rows[0]["t"]) / 60
    rate = lambda key: (
        round((rows[-1][key] - rows[0][key]) / minutes, 1)
        if rows[-1][key] is not None and rows[0][key] is not None else None
    )
    return {"gpu_c_per_min": rate("gpu"), "board_c_per_min": rate("board"), "over_seconds": round(minutes * 60)}


def load_events(state_dir: Path, limit: int) -> list:
    events = []
    for name in ("events.jsonl.1", "events.jsonl"):
        for line in _tail_lines(state_dir / name, HISTORY_TAIL_BYTES):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events[-limit:]


def load_observations(path: Path = OBSERVATIONS_PATH) -> Optional[dict]:
    """Optional notes from the machine's owner about this hardware."""
    try:
        if path.stat().st_size > OBSERVATIONS_MAX_BYTES:
            return None
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# --- tools ---------------------------------------------------------------------------


def thermal_status(state_dir: Path = STATE_DIR, now: Optional[float] = None, **_) -> dict:
    now = time.time() if now is None else now
    s = read_status(state_dir)
    if s is None:
        return {
            "state": "daemon_down",
            "protected": False,
            "protection_issues": ["the daemon has written no status"],
            "headroom": {"level": "unknown", "degrees_to_pause": None},
        }
    state = derive_state(s, now)
    issues = protection_issues(s, state)
    controller = s.get("controller", {})
    return {
        "state": state,
        "protected": not issues,
        "protection_issues": issues,
        "headroom": derive_headroom(s, state),
        "gpu_c": s.get("gpu_c"),
        "board_c": s.get("board_c"),
        "power": s.get("power"),
        "gpu_util": s.get("util"),
        "trend_last_minute": trend(load_samples(state_dir, now - 60, ("thermal.csv",), TREND_TAIL_BYTES)),
        "sw_thermal_slowdown": s.get("sw_slowdown"),
        "hw_thermal_slowdown": s.get("hw_slowdown"),
        "container_status": s.get("container_status"),
        "hold": s.get("hold"),
        "restart_wait_s": controller.get("restart_wait_s"),
        "actions_since_daemon_start": s.get("counts"),
        "reading_age_s": round(now - s.get("updated", 0), 1),
    }


def thermal_policy(state_dir: Path = STATE_DIR, observations_path: Path = OBSERVATIONS_PATH, **_) -> dict:
    s = read_status(state_dir)
    cfg = (s or {}).get("config")
    if not cfg:
        return {"error": "the daemon has written no configuration; it may be down"}
    return {
        "subject": "Physical hardware temperature of the machine running this LLM, not sampling temperature.",
        "active_limits_c": {
            "pause": {"gpu": cfg["pause_gpu"], "board": cfg["pause_board"], "rule": "either sensor at or above"},
            "resume": {"gpu": cfg["resume_gpu"], "board": cfg["resume_board"], "rule": "both sensors at or below"},
            "kill": {"gpu": cfg["kill_gpu"], "board": cfg["kill_board"], "rule": "either sensor at or above"},
            "start": {"gpu": cfg["start_gpu"], "board": cfg["start_board"], "rule": "both sensors at or below"},
        },
        "timing_s": {key: cfg[key] for key in (
            "min_pause_s", "max_pause_s", "sensor_loss_s", "cooldown_s", "max_cooldown_s", "early_exit_s",
        )},
        "states": {
            "normal": "vLLM is running. You can only observe the machine in this state.",
            "paused_cooling": "vLLM is frozen and the model stays loaded. Requests resume afterwards.",
            "cooldown": "vLLM was killed or failed, and a restart is waiting on a backoff timer that doubles each time.",
            "waiting_for_cool": "vLLM is stopped until both sensors reach the start limits.",
            "held_off": "A human placed a hold. vLLM will not be started or unpaused automatically.",
            "daemon_down": "The daemon is not updating. Nothing is protecting the machine; tell a human.",
            "unavailable": "The container is missing, or Docker is not answering.",
        },
        "transitions": [
            "normal -> paused_cooling: pause limit, sustained hardware slowdown, or one sensor missing for sensor_loss_s",
            "paused_cooling -> normal: resume limits confirmed, no hardware slowdown, and min_pause_s elapsed",
            "normal or paused_cooling -> cooldown: kill limit, not cool at max_pause_s, both sensors missing, "
            "or an exit sooner than early_exit_s",
            "cooldown -> waiting_for_cool -> normal: backoff elapsed, then start limits confirmed; model reload follows",
        ],
        "headroom": "Degrees to the nearest pause limit. ample >= 20, moderate 10-19, low < 10. A heuristic of this "
                    "tool, not a daemon rule.",
        "what_this_means_for_you": [
            "While paused you are frozen and cannot call tools. You experience a pause only as a time gap, so "
            "after an unexplained delay check thermal_events.",
            "A pause keeps your request and context. A kill loses in-flight requests, and a restart takes several "
            "minutes and is itself the hottest phase, so avoiding kills matters most.",
            "If protected is false, heavy work is unsafe: the daemon may be unable to act. Tell a human.",
            "These tools are read-only. To change limits or take vLLM offline, ask a human.",
        ],
        "owner_observations": load_observations(observations_path),
    }


def thermal_history(state_dir: Path = STATE_DIR, now: Optional[float] = None, minutes: int = 30, **_) -> dict:
    now = time.time() if now is None else now
    return summarize(load_samples(state_dir, now - minutes * 60), now, minutes)


def thermal_events(state_dir: Path = STATE_DIR, limit: int = 20, **_) -> dict:
    return {"events": load_events(state_dir, limit)}


def parse_metrics(text: str) -> dict:
    totals: dict = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        name_part, _, value = line.rpartition(" ")
        name = name_part.split("{")[0]
        try:
            totals[name] = totals.get(name, 0.0) + float(value)
        except ValueError:
            continue
    get = lambda name: totals.get(f"vllm:{name}")
    generated, decode = get("generation_tokens_total"), get("request_decode_time_seconds_sum")
    prompt, cached = get("prompt_tokens_total"), get("prompt_tokens_cached_total")
    return {
        "requests_running": get("num_requests_running"),
        "requests_waiting": get("num_requests_waiting"),
        "kv_cache_usage_fraction": get("kv_cache_usage_perc"),
        "generation_tokens_total": generated,
        "prompt_tokens_total": prompt,
        "preemptions_total": get("num_preemptions_total"),
        "mean_generation_tokens_per_s": round(generated / decode, 1) if generated and decode else None,
        "prompt_cache_hit_fraction": round(cached / prompt, 3) if cached is not None and prompt else None,
    }


def parse_flags(cmd: list) -> dict:
    """`--flag value` pairs from the serve command. Environment variables are never read."""
    flags: dict = {}
    for i, arg in enumerate(cmd):
        if not arg.startswith("--"):
            continue
        name, has_inline, inline = arg[2:].partition("=")
        if has_inline:
            value = inline
        else:
            following = cmd[i + 1] if i + 1 < len(cmd) else None
            value = True if following is None or following.startswith("--") else following
        if SECRET_FLAG.search(name) or (isinstance(value, str) and SECRET_VALUE.search(value)):
            continue
        flags[name] = value
    return flags


def _http(base_url: str, path: str, timeout: float = 2) -> Optional[str]:
    try:
        with _opener.open(base_url + path, timeout=timeout) as resp:
            return resp.read(HTTP_MAX_BYTES).decode(errors="replace")
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _docker_inspect(container: str) -> Optional[dict]:
    template = "{{json .State}}\n{{json .Config.Cmd}}\n{{json .Config.Image}}"
    try:
        p = subprocess.run(
            ["docker", "inspect", "-f", template, container], capture_output=True, text=True, timeout=5,
        )
        if p.returncode != 0:
            return None
        state, cmd, image = (json.loads(line) for line in p.stdout.strip().split("\n"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return {"state": state, "cmd": cmd or [], "image": image}


def vllm_status(state_dir: Path = STATE_DIR, **_) -> dict:
    s = read_status(state_dir) or {}
    container = s.get("container")
    info = _docker_inspect(container) if container else None
    if info is None:
        return {"serving": False, "container": container, "error": "cannot read the container state"}
    state = info["state"]
    out = {
        "serving": False,
        "container": container,
        "container_status": state.get("Status"),
        "started_at": state.get("StartedAt"),
        "image": info["image"],
        "serve_settings": parse_flags(info["cmd"]),
    }
    if state.get("Status") != "running":
        return out
    base_url = s.get("config", {}).get("vllm_url", "http://127.0.0.1:8000")
    if _http(base_url, "/health") is None:
        return {**out, "detail": "the container is running but not answering; it is probably loading the model"}
    out["serving"] = True
    models = _http(base_url, "/v1/models")
    if models:
        try:
            out["models"] = [
                {"id": m.get("id"), "max_model_len": m.get("max_model_len")} for m in json.loads(models)["data"]
            ]
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
    metrics = _http(base_url, "/metrics")
    if metrics:
        out["metrics_since_process_start"] = parse_metrics(metrics)
    return out


def _schema(prop: Optional[str] = None, description: str = "", default: int = 0, maximum: int = 0) -> dict:
    properties = {}
    if prop:
        properties[prop] = {
            "type": "integer", "description": description, "default": default, "minimum": 1, "maximum": maximum,
        }
    return {"type": "object", "properties": properties, "additionalProperties": False}


TOOLS: dict = {
    "thermal_status": (thermal_status, _schema(), (
        "Current GPU and board temperature, power, daemon state, whether protection is working, degrees of headroom "
        "before vLLM is paused, and the last minute's trend. Check before long or heavy work.")),
    "thermal_policy": (thermal_policy, _schema(), (
        "The daemon's active limits, state machine, what each state means for you, and the owner's notes about this "
        "machine. Read once to understand the rules.")),
    "thermal_history": (thermal_history, _schema("minutes", "How far back to look.", 30, 1440), (
        "Temperature and power over time, slowdown counts and a downsampled series. Use it to learn how your work "
        "affects temperature. Readings are every 5s when cool and 1s when hot, so means lean hot. A gap means the "
        "daemon was not logging: machine off, crashed, or daemon down.")),
    "thermal_events": (thermal_events, _schema("limit", "Maximum events, newest last.", 20, 200), (
        "Log of daemon actions: pause, unpause, kill, start, container_status, sensors, config_error, daemon_started, "
        "daemon_stopped. Check after an unexplained delay, because you cannot observe a pause while it happens. A "
        "daemon_started following other events with no daemon_stopped between suggests a crash or power loss; the "
        "first entry implies nothing, as it only marks where logging began.")),
    "vllm_status": (vllm_status, _schema(), (
        "The serving vLLM engine: container state, serve settings, model and context limit, running and waiting "
        "requests, KV cache usage and throughput.")),
}


def _rpc_result(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _rpc_error(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _call_tool(request_id, params: dict, state_dir: Path) -> dict:
    name, args = params.get("name"), params.get("arguments")
    args = {} if args is None else args
    if not isinstance(name, str) or name not in TOOLS:
        return _rpc_error(request_id, -32602, f"unknown tool: {name!r}")
    if not isinstance(args, dict):
        return _rpc_error(request_id, -32602, "arguments must be an object")
    func, schema, _ = TOOLS[name]
    for key, value in args.items():
        prop = schema["properties"].get(key)
        if prop is None or isinstance(value, bool) or not isinstance(value, int):
            return _rpc_error(request_id, -32602, f"invalid argument: {key}")
        if not prop["minimum"] <= value <= prop["maximum"]:
            return _rpc_error(request_id, -32602, f"{key} must be {prop['minimum']}-{prop['maximum']}")
    try:
        text = json.dumps(func(state_dir=state_dir, **args))
        return _rpc_result(request_id, {"content": [{"type": "text", "text": text}], "isError": False})
    except Exception as exc:  # a tool fault must be reported to the agent, not end the server
        print(f"{name} failed: {exc!r}", file=sys.stderr, flush=True)
        message = f"{name} failed: {type(exc).__name__}"
        return _rpc_result(request_id, {"content": [{"type": "text", "text": message}], "isError": True})


def handle(message: dict, state_dir: Path = STATE_DIR) -> Optional[dict]:
    """One JSON-RPC message in, one response out, or None for a notification."""
    if "id" not in message:
        return None
    method, request_id = message.get("method"), message.get("id")
    params = message.get("params")
    params = {} if params is None else params
    if not isinstance(params, dict):
        return _rpc_error(request_id, -32602, "params must be an object")
    if method == "initialize":
        requested = params.get("protocolVersion")
        return _rpc_result(request_id, {
            "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "vllm-thermald", "version": __version__},
            "instructions": INSTRUCTIONS,
        })
    if method == "ping":
        return _rpc_result(request_id, {})
    if method == "tools/list":
        return _rpc_result(request_id, {"tools": [
            {
                "name": name,
                "description": description,
                "inputSchema": schema,
                "annotations": {"readOnlyHint": True, "destructiveHint": False},
            }
            for name, (_, schema, description) in TOOLS.items()
        ]})
    if method == "tools/call":
        return _call_tool(request_id, params, state_dir)
    return _rpc_error(request_id, -32601, f"method not found: {method!r}")


def process_line(line: str, state_dir: Path = STATE_DIR) -> Optional[dict]:
    """Never raises: one malformed message must not end the agent's tools for the session."""
    try:
        message = json.loads(line)
    except ValueError:
        return _rpc_error(None, -32700, "parse error")
    if not isinstance(message, dict):
        return _rpc_error(None, -32600, "invalid request")
    try:
        return handle(message, state_dir)
    except Exception as exc:
        print(f"request failed: {exc!r}", file=sys.stderr, flush=True)
        return _rpc_error(message.get("id"), -32603, "internal error")


def serve(state_dir: Path = STATE_DIR) -> int:
    sys.stdin.reconfigure(errors="replace")
    for line in sys.stdin:
        if not line.strip():
            continue
        response = process_line(line, state_dir)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0
