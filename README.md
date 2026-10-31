# vllm-thermald

Keeps a [vLLM](https://github.com/vllm-project/vllm) Docker container from overheating an NVIDIA
DGX Spark, and lets AI agents read the temperature of the hardware they run on. It has no dependencies
beyond Python 3.9+, Docker and `nvidia-smi`.

**This reduces the risk of an overheating shutdown but can't guarantee against one.** It won't help if a
shutdown is caused by power rather than heat, and temperatures can rise quickly between readings. It's
tested on one DGX Spark, so check the limits against your own readings.

## Quick start

With your vLLM container already created, on the Spark:

```bash
uv tool install vllm-thermald    # or: pipx install vllm-thermald
vllm-thermald install
```

That checks the machine, finds your running vLLM container, and starts a service that also runs at boot.
**From then on, the daemon starts vLLM whenever it's stopped and cool.** To stop vLLM on purpose, run
`vllm-thermald hold` first.

## How it works

It checks temperatures every second and acts when a sensor crosses a limit:

| Situation | GPU | Board | Action |
|---|---|---|---|
| Hot (either sensor) | 85°C | 92°C | `docker pause` |
| Cooled (both sensors, after at least 30s) | 70°C | 75°C | `docker unpause` |
| Critical (either sensor), or not cool after 10 minutes paused | 90°C | 96°C | `docker kill` |
| Stopped and cold (both sensors) | 60°C | 65°C | `docker start` |

**Why it pauses first:** a pause removes GPU load within seconds, keeps the model loaded, and in-flight
requests continue afterwards. A restart reloads the model and runs autotune, the hottest phase, so simply
turning vLLM off and on could cause the crash it's trying to avoid.

**It fails safe:**

- A missing sensor never counts as cool. If one is missing for 30 seconds vLLM is paused, and if both are
  missing it's killed.
- After a kill or failed run, restarts wait 5 minutes, doubling up to an hour. This survives daemon
  restarts and reboots, but a sudden power loss doesn't start a wait.
- An invalid config logs an error and uses the default limits, rather than leaving vLLM unprotected.
- systemd restarts the daemon if it crashes or hangs.

## Commands

| Command | What it does |
|---|---|
| `vllm-thermald install [--container NAME]` | Set up and start the service |
| `vllm-thermald uninstall` | Stop and remove the service, keeping config and history |
| `vllm-thermald doctor` | Check this machine has what the daemon needs |
| `vllm-thermald status` | Show temperatures and actions |
| `vllm-thermald hold` / `release` | Stop vLLM being started or unpaused, or undo that. Heat protection stays active |
| `vllm-thermald check-config` | Validate the config file |

Logs: `journalctl --user -u vllm-thermald -f`

## Configure

Edit `~/.config/vllm-thermald/config.json`, setting only what you want to change, then run
`systemctl --user restart vllm-thermald`. Limits must satisfy start ≤ resume < pause < kill, and every
option is listed in `Config` in [`daemon.py`](src/vllm_thermald/daemon.py).

```json
{"container": "my-vllm", "pause_gpu": 83, "resume_gpu": 68}
```

History is in `~/.local/state/vllm-thermald/`: `thermal.csv` holds readings, synced to disk so they survive
a sudden power loss, and `events.jsonl` holds every action with its reason.

## MCP server for agents

`vllm-thermald mcp` is a stdio [MCP](https://modelcontextprotocol.io) server. It only reads the daemon's
files, so a fault in it can't affect protection.

| Tool | Returns |
|---|---|
| `thermal_status` | Temperatures, power, state, whether protection is working, headroom, trend |
| `thermal_policy` | Active limits, state machine, and what each state means for the agent |
| `thermal_history` | Temperature and power over time, slowdown counts, logging gaps |
| `thermal_events` | Log of daemon actions |
| `vllm_status` | Serve settings, model, requests, cache use, throughput |

Add it to any MCP client with the command `vllm-thermald` and the argument `mcp`. For example, with
[Hermes Agent](https://github.com/NousResearch/hermes-agent):

```bash
hermes mcp add thermald --command vllm-thermald --args mcp
```

While vLLM is paused, an agent running on it is frozen and can't call anything. It sees a pause only as a
gap in time, so it should check `thermal_events` after an unexplained delay.

To give the agent notes about your machine, put a JSON object in
`~/.config/vllm-thermald/observations.json`. `thermal_policy` returns it, so write only what the agent
should read.

## Security

- **No agent control:** all MCP tools are read-only, so an agent can't change limits or disable its own
  protection.
- **Local only:** the MCP server uses stdio with no network port, and `vllm_url` must be a loopback address.
- **Secrets:** tools read the container's command but never its environment. Flags and values that look like
  credentials are left out on a best-effort basis, so avoid passing secrets as command-line flags.
- **Docker access:** the daemon needs Docker, which is effectively root on the host. Anyone who can edit its
  config or files can control this container.

Please report vulnerabilities privately through
[GitHub security advisories](https://github.com/koderizer/vllm-thermald/security/advisories/new).

## Develop

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## License

[Apache 2.0](LICENSE). An independent project, not affiliated with or endorsed by NVIDIA, the vLLM project,
or Nous Research. All trademarks belong to their owners.
