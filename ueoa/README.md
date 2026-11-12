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
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn ueoa.api.app:create_app --factory --host 127.0.0.1 --port 8766
```

Open **http://127.0.0.1:8766** (API docs at `/docs`). Keep the server running in this PowerShell window. Azure OpenAI
configuration is optional: without it, the console and API run normally, but AI workflow planning and agent generation
are unavailable. To enable them, set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, and
`AZURE_API_VERSION` before starting the server. Hosted agents use the application address that receives their
registration request, so agents work
when the server runs on a non-default local port. Set `UEOA_PUBLIC_BASE_URL` when the app is behind a proxy or must be
reachable at a different address. The app loads `.env` from the repository root; variables already set in the process
environment take precedence.
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

## Persistence (PostgreSQL)

Set `UEOA_STORE_URL` to enable PostgreSQL; an empty URL keeps the current in-memory backend. The local database
helper requires Docker Desktop to be installed and running. On PowerShell systems that block script execution, invoke
the helper with `-ExecutionPolicy Bypass` for this process only:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\pg-start.ps1
$env:UEOA_STORE_URL = "postgresql+asyncpg://ueoa:ueoa-dev-only@localhost:5432/ueoa"
.\.venv\Scripts\python.exe -m pip install -e ".[db]"
.\.venv\Scripts\python.exe -m ueoa.persistence.migrate upgrade
.\.venv\Scripts\python.exe -m uvicorn ueoa.api.app:create_app --factory --host 127.0.0.1 --port 8766
```

PostgreSQL stores capability contracts, implementation cards and operator status/priority, workflow templates and
versions, runs and node results, pending/decided approvals, and gateway audit history. Catalog capabilities and
seed implementations are upserted at startup; approved, deprecated, and revoked implementation decisions are kept
instead of being overwritten by catalog seeds. Implementations, workflows, pending approvals, and recent runs are
reloaded on restart. Cards persist only secret-reference names, never secret values. Hosted-agent definitions and
Skills remain in the existing document store and require `UEOA_DATA_DIR` to survive restarts.

Use the **Orchestrate** page to plan and run goals against the capabilities and agents you register.
In Orchestrator chat, select a saved workflow and describe the task in the composer. The chat goal is bound to the
workflow's best matching declared text input (for example `prompt`, `task_description`, or `source_text`). Set defaults
in the workflow for any other values it needs; chat sends the goal directly without per-run JSON input. The workflow
must include an executable agent/tool step to produce an answer.
The **Create → Prompt Builder** page uses Azure OpenAI to refine task requirements; choose **Use prompt in workflow**
to hand the result to the visual workflow generator. Generation only uses capabilities with approved implementations.
If none are approved, register an agent implementation and approve it before generating executable workflow steps.

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
- With the default memory backend, registry capabilities/implementations, runs, approvals, and audit history are lost
  on restart. Set `UEOA_STORE_URL` to retain them in PostgreSQL.
- Hosted-agent definitions, workflow templates, and Skills entries use the JSON document store only when
  `UEOA_DATA_DIR` is configured; without it, those objects are also in memory.
