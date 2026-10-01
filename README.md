# Group 2 — Agentic Harness Project

*AI Engineering Lab, SoSe 2026*

## Overview

This is an agent that manages an inventory — the kind of record you'd want on hand for an insurance claim. Give it a receipt, a purchase note, or just describe an item in conversation, and it pulls out the details that matter and keeps a running list in `inventory.md`. It's built around the ReAct pattern: the agent reasons about what to do, takes an action, observes the result, and repeats until it has an answer, running on a hosted LLM through InnKube.

## Week 1

- The core ReAct loop, verified against the real InnKube API
- A command-line entrypoint for multi-turn conversation
- JSON-based configuration, with the access token kept in an environment variable, never in a file
- File tools the agent can use inside its workspace: create, navigate, search, read, read_many, modify — deliberately no delete, and no writing outside the workspace boundary
- Docker containerization (unit tests run separately with `pytest tests/unit_tests/` — they do not gate the build)

## Week 2

**MCP integration** — connects to configured MCP servers, discovers tools at runtime, and invokes them mid-conversation. Servers are enabled/disabled via `config.json`, no code changes needed. A failed server is skipped without crashing the agent.

- **Product Recall** — checks the official US CPSC recall database for safety recalls on inventory items. No login required.
- **OCR** — extracts text from photographed or scanned receipts (Tesseract).
- **QR generation** — produces a scannable PNG for an inventory item, a link, a contact card, or guest Wi-Fi credentials. Codes are written to `workspace/qr_codes/`; the tool takes a bare filename and refuses anything path-shaped, so it cannot write elsewhere.
## Week 3

**Monitoring and subagents** — Use prometheus to scrap metrics from hooks, and use streamlit to show it in a dashboard.
**Permission management** —
**Sub-agents** —

**Long-term memory** — two systems:

- **Receipt memory (JSON)** — deduplicates receipts by file content hash. Viewing a receipt is never blocked; adding it twice is.
- **General memory (ChromaDB)** — stores freeform facts across sessions, matched by semantic search, with a `forget` function to remove entries.

**Also included:**

- CLI tables render as aligned grids instead of raw pipe characters.
- Extended test suite covering OCR, the recall tool, QR payload formats and generation, both memory systems, and MCP error handling — plus an end-to-end test that has the agent generate a QR code, then decodes the image with an independent decoder and asserts the payload round-trips.
- **Additional:** a Streamlit dashboard — see below.

## Project Layout

```
agent/            the harness itself — loop, tool registry, config, both memory systems, table formatting
mcp_servers/       the three self-hosted MCP servers: recall/, ocr/ and qr/
tests/             unit tests and end-to-end tests
samples/receipts/  sample receipts (text and real photos) for setup and testing
diagrams/          architecture diagrams
config_example.json, dashboard.py, Dockerfile, docker-compose.yml, requirements.txt
docker/entrypoint.sh  container entrypoint: repairs state-dir ownership, drops to the non-root agent user
prometheus.yml        scrape config (target agent:9000 is hand-pinned, see Setup)
.dockerignore, .containerignore, .gitattributes
```

## Setup

Before running anything, copy `config_example.json` to `config.json` and `.env_example` to `.env`, and fill in the one field:

```
INNKUBE_TOKEN=
```

There is nothing else to configure. The metrics port is fixed at 9000 in three places on purpose — `agent/main.py`, `docker-compose.yml` and `prometheus.yml` — because Prometheus reads plain YAML and cannot follow an environment variable. A port variable would only let those three drift apart and turn a typo into a dashboard that is silently empty while everything else looks healthy.

## Run agent with dashboard and prometheus

This path is the same on **Linux, macOS and Windows**, and only needs Docker/Podman (or any docker-compatible engine):

1. Build the image (once):

   ```
   docker build -t inventory-agent .
   ```

2. Start the background services:

   ```
   docker compose up -d dashboard prometheus
   ```

3. Start the agent (interactive):

   ```
   docker compose run --rm -it --use-aliases agent
   ```
   In case in linux using podman:

   podman compose run --rm agent

   `--use-aliases` is load-bearing: it gives the container the `agent` network alias that `prometheus.yml` scrapes. Without it, `docker compose run` generates a name like `group-2-agent-run-x7k2`, the target `agent:9000` resolves to nothing, and the dashboard looks healthy but empty.

4. Stop everything:

   ```
   docker compose down
   ```

   Add `-v` only if you also want to wipe the stored memory database.

Dashboard: <http://localhost:8501> — Prometheus: <http://localhost:9090> — metrics: <http://localhost:9000/metrics>.

### Where the data lives

| State | Config key (inside container) | On the host | Survives `docker compose down`? |
|---|---|---|---|
| ChromaDB memory + run log | `ltm_db_path` = `data/chroma`, `data/run_log.jsonl` → `/app/data` | **named volume** `memory-data` (`docker volume inspect group-2_memory-data`) | yes — wiped only by `down -v` |
| Workspace: `inventory.md`, receipts, QR codes | `workspace_root` = `./workspace` → `/app/workspace` | `./workspace` directory in the repo | yes |
| Config + token | `config.json`, `.env` | repo root (never copied into the image: excluded in `.dockerignore`) | yes — untouched by compose |

The memory database must stay a named volume; the workspace must stay a bind mount (so files are visible on the host). Flipping either of those is what caused the original ChromaDB permission failures.

## Running in a Container

Docker is the better-tested path — the image already handles Tesseract, `poppler-utils`, and all Python dependencies, so there's nothing extra to install.

Set `INNKUBE_TOKEN` as an environment variable first, the same way as in [Setup](#setup) — `docker compose` reads it from your shell automatically. (If you'd rather not export it every session, put it in `.env`; Compose reads that file too.)

```
docker compose run --rm --build -it --use-aliases agent   # first time
docker compose run --rm -it --use-aliases agent           # after that
```

Compose keeps the memory database in a *named volume*, so nothing has to exist on the host beforehand.

The image's entrypoint (`docker/entrypoint.sh`) starts as root for a few milliseconds only to repair ownership of `/app/data` and `/app/workspace` if a stale volume or a host-owned bind mount would otherwise make ChromaDB fail, then drops to the non-root `agent` user with `gosu` before a single line of the application runs. Nothing of ours executes as root.

### Linux

Everything above works on Linux with Docker unchanged. For **rootless Podman**, skip `podman compose` unless `podman-compose` is already installed — unlike `docker compose`, it is a separate tool. Building and running directly is more reliable:

```
podman build -t inventory-agent .
podman volume create inventory-memory   # once
podman run --rm -it --userns=keep-id --env-file .env \
  -v inventory-memory:/app/data \
  -v "$(pwd)/workspace:/app/workspace:Z" \
  inventory-agent
```

For the dashboard and Prometheus alongside it:

```
podman compose up -d dashboard prometheus   # if podman-compose is installed
```

Each of those pieces is load-bearing, and each fails differently if you leave it out:

- **Named volume for `/app/data`** — never bind-mount the data directory. A bind mount hands the directory's ownership to the host, and SQLite does not tolerate Docker Desktop's file-sharing layer; both made ChromaDB fail to open its database on other machines. The entrypoint repairs ownership at startup either way, but a named volume seeded from the image avoids the problem before it starts.
- **`--userns=keep-id`** — rootless Podman maps your host user to *root* inside the container, which leaves the image's non-root `agent` user (uid 1000) on an unrelated subuid that does not own your files. Without it the `workspace` bind mount is not writable by the agent.
- **`:Z`** — relabels the mount for SELinux (Fedora, RHEL, CentOS), which otherwise blocks the container from reading it. `:Z` marks a directory private to this container; use `:z` if something else needs it too. On distributions without SELinux the suffix does nothing.
- **`--env-file .env`** — reads the token from `.env` (see [Setup](#setup)) and keeps it out of your shell history. If you would rather use an exported variable, `-e INNKUBE_TOKEN="$INNKUBE_TOKEN"` works as well.

### macOS

Docker Desktop on macOS follows the [universal path](#run-agent-with-dashboard-and-prometheus) above — nothing changes. With Podman, use the same command as Linux, minus `:Z` unless you are on a SELinux-enabled setup (you are not, on a stock Mac): `-v "$(pwd)/workspace:/app/workspace"`.

### Windows (PowerShell)

The universal path above works as-is with Docker Desktop. If you use Podman on Windows:

```
podman run --rm -it -e INNKUBE_TOKEN=$env:INNKUBE_TOKEN `
  -v inventory-memory:/app/data `
  -v ${PWD}/workspace:/app/workspace inventory-agent
```

Note that `$env:VAR` is PowerShell syntax — in bash it expands to nothing useful, and the agent will start normally and then fail its first request with a 401.

Podman on Windows runs inside a WSL2 virtual machine, so bind mounts behave a little differently from Docker Desktop, and neither `--userns=keep-id` nor `:Z` applies there. If local files aren't showing up inside the container, check that `podman machine` is actually running and that the mounted path is somewhere the VM can see.

### Troubleshooting: `Permission denied (os error 13)` from ChromaDB

This is the one error that has bitten every machine this repo has been run on. It is always a permissions problem on `/app/data`, never a missing database — the directory is created automatically.

| Where it happens | Cause | Fix |
|---|---|---|
| Any OS, first run on a fresh machine | A data directory bind-mounted from the host, owned by a different uid (or by root) | Use the named volume (default in `docker-compose.yml`); do not mount `./data` |
| Docker Desktop (Windows/macOS) | `./data` bind mount crossing the file-sharing layer — SQLite/ChromaDB cannot work there | Same: named volume |
| An old checkout with an old volume | The volume was seeded by a previous image and is root-owned; `compose down` keeps it, only `down -v` removes it | The entrypoint now repairs this automatically at startup. If it still fails: `docker compose down -v` |
| Rootless Podman, `workspace` not writable | Missing `--userns=keep-id` | Add the flag |

To confirm which volume a container is actually using — `docker volume inspect group-2_memory-data` (Podman: `podman volume inspect group-2_memory-data`) prints its mountpoint and, on Linux, its owner. If the startup check fails, the message itself names the unwritable path, the uid it runs as, and the matching row above.

If the entrypoint prints `entrypoint: '/app/data' is not writable by 'agent' -- repairing ownership`, that is it doing its job — it is not an error.

## Additional: Dashboard

A Streamlit dashboard sits alongside the CLI — same underlying agent, same `ToolRegistry`/`react_step` pipeline, just a visual window onto it rather than a second implementation. It's not the primary interface; the CLI is.

```
pip install -r requirements.txt
streamlit run dashboard.py
```

From there you can upload a receipt and process it, check a product for safety recalls, search the inventory, and browse the receipt images on file — all backed by the same tools the CLI uses.

## Tests

```
pytest tests/unit_tests/ -v
pytest tests/e2e_tests/ -v
```
The unit tests need no token and no network. The end-to-end tests call the real LLM, so they need `INNKUBE_TOKEN` — either exported, or placed in a `.env` file at the repo root, which the test suite reads automatically. Without it they skip rather than fail.

The unit tests do **not** run as part of the Docker build — an earlier version gated the build on them, and that gate was removed along with the multi-stage build. Run them yourself before building if you want the same guarantee.

## Try It Out

With sample receipts in place:

> Look through the receipts folder and build a complete inventory from what's there.

A few things worth trying specifically:

- *"Process the receipt image 'Receipt 1.jpg' and add it to the inventory."*
- Ask it to process that same receipt again — it recognizes the duplicate and skips it, but still tells you what's already on record.
- *"Show me what's on Receipt 1.jpg."* — this always works, even for something already added.
- *"Check if the Digital Blood Pressure Monitor has any recalls."*
- *"Generate a QR code named asset-tag.png that encodes 'Dell XPS 13 - DXP-13-45927'."* — then look in `workspace/qr_codes/` and scan it with a phone.
- *"Make a QR code for the guest Wi-Fi, network GuestNet, password hunter2."* — scanning it joins the network; special characters in either field are escaped correctly.
- *"Generate a QR code at ../../escape.png."* — it refuses, because the tool takes a filename and not a path.
- *"Remember that I prefer metric units."* — then, in a new session: *"What units do I prefer?"*
- *"Forget that I prefer metric units."* — then ask again; it should no longer know.

### Complex scenarios

The above each scenarios one feature in isolation. These chain several together, the way a real session preparing an insurance claim actually would:

- *"Go through everything already on record, and for anything worth over 100 EUR or USD that's electronics, check whether it has a safety recall. Then give me a short claim-ready summary: item, value, and recall status."* — the agent reads the inventory, filters by both category and price threshold, then calls the confirmation-gated recall tool once per qualifying item, approving each in turn.
- *"Use the inventory-auditor to find the exact price and serial number recorded for [an item], then use the qr-labeller to generate an asset tag QR code encoding that price and serial number together."* — chains two different sub-agents in one request, the second acting on what the first found; both show up in the dashboard's Sub-agent Activity panel, attributed to the same parent run.
- *"Check [an item] for recalls, and also forget that I prefer metric units."* — mixes a require-user-confirmation outcome with a deny outcome in the same turn. The recall check is approved and runs; `forget_fact` is blocked immediately with no confirmation prompt at all, proving a `deny` rule never consults the user, even when everything else in the same request is being approved.

Automated versions of all three run as `tests/e2e_tests/test_complex_scenarios.py`.

- **NOTE** — The dashboard is open in localhost:8501