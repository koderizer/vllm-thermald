# vllm-thermald

Keeps a vLLM Docker container from overheating its machine, and lets an AI agent read the
temperature of the hardware it runs on. Built for an NVIDIA DGX Spark, with no dependencies
beyond Python 3.9+, Docker and `nvidia-smi`.

**This reduces the risk of an overheating shutdown but cannot guarantee against one.** It won't
help if a shutdown is caused by power rather than heat, and temperatures can rise quickly between
readings. Fix any cooling problem first.

## How it works

The daemon checks temperatures every second and acts when a sensor crosses a limit.

| Situation | GPU | Board | Action |
|---|---|---|---|
| Hot (either sensor) | 85°C | 92°C | `docker pause` |
| Cooled (both sensors, after at least 30s) | 70°C | 75°C | `docker unpause` |
| Critical (either sensor), or not cool after 10 minutes paused | 90°C | 96°C | `docker kill` |
| Stopped and cold (both sensors) | 60°C | 65°C | `docker start` |

**Why it pauses first:** a pause removes GPU load within seconds, keeps the model loaded, and
in-flight requests continue afterwards. A restart reloads the model and runs FlashInfer autotune,
which is the hottest phase, so simply turning vLLM off and on could cause the crash it's avoiding.

**It fails safe:**

- A missing sensor is never treated as cool. If one is missing for 30 seconds, vLLM is paused. If both
  are missing, it's killed.
- After a kill or failed run, restarts wait 5 minutes, doubling up to an hour. This survives daemon
  restarts and reboots.
- An invalid config logs an error and falls back to the defaults, rather than leaving vLLM unprotected.
- systemd restarts the daemon if it crashes or hangs, and a lock stops two copies from running.

## Install

The container must already exist, created by your vLLM launch script. The default name is
`vllm-spark-128gb-qwen38-27b`; to use another, set `container` in the config below.

```bash
./install.sh
```

After installation the daemon **owns the container's state**: it starts vLLM whenever it's stopped and
cool, including at boot.

## Use

```bash
./vllm_thermald.py status          # temperatures and action counts
./vllm_thermald.py hold            # run before stopping vLLM on purpose
./vllm_thermald.py release         # hand control back to the daemon
./vllm_thermald.py check-config    # validate the config file
journalctl --user -u vllm-thermald -f
```

Hold stops the daemon from starting or unpausing vLLM. Heat protection stays active.

To remove it: `systemctl --user disable --now vllm-thermald`, then delete
`~/.config/systemd/user/vllm-thermald.service`.

## Configure

Optional file: `~/.config/vllm-thermald/config.json`. Set only what you want to change, then restart
the service. See `Config` in `vllm_thermald.py` for every option.

```json
{"container": "my-vllm", "pause_gpu": 83, "resume_gpu": 68}
```

Thresholds must satisfy start ≤ resume < pause < kill. The defaults come from one machine, so check
them against your own readings.

## Files

In `~/.local/state/vllm-thermald/`, private to your user:

| File | Contents |
|---|---|
| `thermal.csv` | Readings, synced to disk so they survive a sudden power loss |
| `events.jsonl` | Every pause, unpause, kill and start, with the reason and readings |
| `status.json` | Current state |
| `backoff.json` | Pending restart backoff |

## MCP server for agents

`vllm_thermald_mcp.py` is a stdio MCP server. It runs separately and only reads the files above, so a
fault in it can't affect protection.

| Tool | Returns |
|---|---|
| `thermal_status` | Temperatures, power, state, whether protection is working, headroom, trend |
| `thermal_policy` | Active limits, state machine, and what each state means for the agent |
| `thermal_history` | Temperature and power over time, slowdown counts, logging gaps |
| `thermal_events` | Log of daemon actions |
| `vllm_status` | Serve settings, model, requests, cache use, throughput |

Register it with [Hermes Agent](https://github.com/NousResearch/hermes-agent):

```bash
hermes mcp add thermald --command /usr/bin/python3 --args "$PWD/vllm_thermald_mcp.py"
```

While vLLM is paused, an agent running on it is frozen and can't call anything. It sees a pause only
as a gap in time, so it should check `thermal_events` after an unexplained delay.

To give the agent notes about your machine, put a JSON object in
`~/.config/vllm-thermald/observations.json`. It's returned by `thermal_policy`, so write only what you
want the agent to read.

## Security

- **No agent control:** all MCP tools are read-only, so an agent can't change limits or disable its own
  protection.
- **Local only:** the MCP server uses stdio with no network port, and `vllm_url` must be a loopback
  address.
- **Secrets:** tools read the container's command but never its environment. Flags and values that look
  like credentials are left out, on a best-effort basis, so avoid passing secrets as command-line flags.
- **Docker access:** the daemon needs Docker, and access to Docker is effectively root on the host. Anyone
  who can edit the config or the installed files can control this container.

## Test

```bash
python3 -m unittest
```
