# UEOA - Enterprise Agent Orchestrator

UEOA is a platform-agnostic **registry, gateway and orchestrator** for enterprise agents and tools. Agents are described
by a **capability contract** and an **A2A agent card**. The gateway enforces entitlements, JSON Schemas, payload-bound
approval for writes, and audit. The planner turns a goal into a validated workflow; the run controller executes it,
pausing for human input and approvals.

## Layers

| Package | Role |
|---|---|
| `src/ueoa` | **Core backend**, domain-agnostic: contracts, registry, gateway, planner, validator, run controller, REST API, MCP server, web UI. Starts empty. |
| `src/ueoa_sdk` | **Agent SDK**: hosts Python functions as A2A agents and provides the LLM client. |
| `frontend` | **React console** source and build configuration. |

## Quick start (Windows, Python 3.12)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m uvicorn ueoa.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** (API docs at `/docs`). The core starts with an empty registry; register your own
capabilities and agent implementations through the API. Configure `UEOA_LLM_*` environment variables to enable planning.
Set `UEOA_DATA_DIR` before starting the API to persist hosted-agent definitions, workflow templates, and Skills catalog
entries between restarts:

```powershell
$env:UEOA_DATA_DIR = ".ueoa-data"
.\.venv\Scripts\python.exe -m uvicorn ueoa.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

The web console is a bundled React application backed by the same API. To rebuild after changing its source:

```powershell
cd frontend
npm ci
npm run build
cd ..
```

Use the **Orchestrate** page to plan and run goals against the capabilities and agents you register.

## MCP endpoint

`http://127.0.0.1:8000/mcp` (Streamable HTTP, JSON). Each read capability your role may use becomes a tool, plus
`ueoa_plan` and `ueoa_run_goal`. VS Code `.vscode/mcp.json`:

```json
{"servers": {"ueoa": {"type": "http", "url": "http://127.0.0.1:8000/mcp", "headers": {"X-Demo-Role": "user"}}}}
```

The Build → MCP page lets an administrator register Streamable HTTP MCP endpoint URLs in the managed server catalog.
The catalog is informational; adding an endpoint there does not import its tools into the capability registry. MCP
tools declared by registered agent cards can be invoked by the gateway over Streamable HTTP.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest              # default suite (no network)
.\.venv\Scripts\python.exe -m ruff check src tests
```

## Security note

- The `X-Demo-Role` header is a **local demo identity, not authentication**. Bind to `127.0.0.1` only; do not expose
  the API on a network until Entra ID sign-in is added.
- Requests are limited to trusted `Host` headers, local origins (or `UEOA_ALLOWED_ORIGINS`) and a body size limit.
  Outbound calls only go to `UEOA_ALLOWED_HOSTS`. Secrets are referenced by name (`UEOA_SECRET_*`), never stored in cards.
- `UEOA_ENTRA_STATIC_TOKEN` is a development-only fixed token and is sent to loopback hosts only.
- Registry capabilities/implementations, runs, approvals, and the audit are in memory and are lost on restart.
- Hosted-agent definitions, workflow templates, and Skills entries use the JSON document store only when
  `UEOA_DATA_DIR` is configured; without it, those objects are also in memory.
