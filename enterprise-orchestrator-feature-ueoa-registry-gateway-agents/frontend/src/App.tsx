import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  Activity, AlertCircle, ArrowRight, Bot, Check, ChevronRight, CircleHelp,
  Clock3, Copy, Database, FileText, GitBranch, LayoutDashboard, LoaderCircle,
  Menu, Play, Plus, RefreshCw, Send, Server, ShieldCheck, Sparkles, Square, Terminal, X, Trash2,
} from "lucide-react";
import WorkflowStudio, { type RunResult, type Workflow, type WorkflowDocument, type WorkflowSummary, type Validation } from "./WorkflowStudio";

type JsonRecord = Record<string, unknown>;
type Health = {
  status: string;
  planner: boolean;
  capabilities: number;
  implementations: number;
  approved: number;
  default_role: string;
};
type Capability = {
  id: string;
  description: string;
  effects: string;
  implementations: string[];
};
type Implementation = {
  key: string;
  name: string;
  kind: string;
  implementation: string;
  binding: string;
  owner: string;
  implements: string[];
  status: string;
  priority: number;
  conformant: boolean;
  checks: { check: string; passed: boolean; detail: string }[];
};
type RunSummary = {
  id: string;
  status: string;
  goal: string;
  created_at: string;
  finished_at: string | null;
  nodes: Record<string, number>;
};
type RunDetail = {
  id: string;
  status: string;
  goal: string;
  plan: JsonRecord | null;
  workflow: { nodes: { id: string; capability?: string; type?: string; depends_on?: string[] }[] };
  nodes: Record<string, {
    state: string; implementation?: string; error?: string; output?: unknown;
    items?: { index: number; state: string; error?: string; output?: unknown }[];
    started_at?: string | null; finished_at?: string | null; duration_ms?: number;
  }>;
  outputs: unknown;
  error: string | null;
  events: { at: string; type: string; node?: string | null; detail?: string }[];
  created_at: string;
  finished_at?: string | null;
};
type Approval = {
  run_id: string;
  goal: string;
  node: string;
  item: number | null;
  capability: string;
  payload: JsonRecord;
  payload_hash: string;
  requested_at: string;
  requested_by: string;
};
type AuditRecord = {
  at: string;
  principal: { role: string; name: string };
  capability: string;
  implementation: string | null;
  status: string;
  duration_ms: number;
  error: string | null;
};
type Skill = {
  id: string;
  name: string;
  description: string;
  instructions: string;
  tags: string[];
  builtin: boolean;
  owner: string;
  updated_at: string;
};
type McpServer = {
  id: string;
  name: string;
  url: string;
  description: string;
  transport: string;
  owner: string;
  registered_at: string;
};
type ChatAttachment = { name: string; size: number; mediaType: string };
type EncodedAttachment = {
  document_id: string;
  text?: string;
  content_base64?: string;
  media_type?: string;
};
type StagedAttachment = ChatAttachment & { id: string; encoded: EncodedAttachment };
type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  attachments?: ChatAttachment[];
  selectedCapabilities?: string[];
  workflowId?: string;
  stage?: "planning" | "running" | "completed" | "failed" | "cancelled";
  statusText?: string;
  plan?: JsonRecord;
  run?: RunDetail;
  error?: string;
  fallbackWorkflows?: string[];
};
type ChatRequest = {
  text: string;
  inputs: JsonRecord;
  attachmentMeta: ChatAttachment[];
  documents: EncodedAttachment[];
  capabilities: string[];
};
type PromptBrief = {
  description: string;
  audience: string;
  inputs: string;
  steps: string;
  constraints: string;
  output: string;
};
type OverviewStats = {
  total_agents: number;
  active_agents: number;
  total_workflows: number;
  total_skills: number;
  registered_mcp_servers: number;
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  recent_activities: { type: string; id: string; label: string; status: string; at: string }[];
};
type HostedAgent = {
  id: string;
  name: string;
  template: string;
  version: string;
  capability: string;
  fields: string[];
  implementation: { key: string; status: string } | null;
};
type Data = {
  health: Health | null;
  capabilities: Capability[];
  implementations: Implementation[];
  agents: HostedAgent[];
  workflows: WorkflowSummary[];
  runs: RunSummary[];
  approvals: Approval[];
  audit: AuditRecord[];
  skills: Skill[];
  mcpServers: McpServer[];
  overview: OverviewStats | null;
};
type Page = "Overview" | "Prompt Builder" | "Orchestrator" | "Agent Registry" | "Workflow" | "Saved workflows" | "Workflows" | "Skills" | "MCP Registry" | "Runs" | "Approvals" | "Audit";

const EMPTY: Data = {
  health: null,
  capabilities: [],
  implementations: [],
  agents: [],
  workflows: [],
  runs: [],
  approvals: [],
  audit: [],
  skills: [],
  mcpServers: [],
  overview: null,
};
const NAV: { label: Page; group: string; icon: typeof LayoutDashboard }[] = [
  { label: "Overview", group: "Platform", icon: LayoutDashboard },
  { label: "Prompt Builder", group: "Create", icon: Sparkles },
  { label: "Workflow", group: "Create", icon: GitBranch },
  { label: "Saved workflows", group: "Create", icon: FileText },
  { label: "Orchestrator", group: "Run", icon: Terminal },
  { label: "Agent Registry", group: "Build", icon: Bot },
  { label: "Skills", group: "Build", icon: CircleHelp },
  { label: "MCP Registry", group: "Build", icon: Server },
  { label: "Runs", group: "Run", icon: Activity },
  { label: "Approvals", group: "Govern", icon: ShieldCheck },
  { label: "Audit", group: "Govern", icon: Database },
];

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function pageFromLocation(): Page {
  if (window.location.pathname === "/create/workflow") return "Workflow";
  if (window.location.pathname === "/create/saved-workflows") return "Saved workflows";
  if (window.location.pathname === "/create/prompt") return "Prompt Builder";
  return "Overview";
}

async function api<T>(
  path: string,
  role: string,
  options: { method?: string; body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  const response = await fetch(path, {
    method: options.method ?? "GET",
    signal: options.signal,
    headers: {
      "X-Demo-Role": role,
      ...(options.body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    ...(options.body === undefined ? {} : { body: JSON.stringify(options.body) }),
  });
  const text = await response.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!response.ok) {
    const detail = typeof body === "object" && body !== null && "detail" in body
      ? JSON.stringify((body as { detail: unknown }).detail)
      : String(body || response.statusText);
    throw new ApiError(response.status, detail);
  }
  return body as T;
}

function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function formatTime(value: string | null | undefined): string {
  if (!value) return "In progress";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function statusTone(value: string): string {
  const status = value.toLowerCase();
  if (["approved", "completed", "succeeded", "ok", "pass"].some((part) => status.includes(part))) return "good";
  if (["draft", "waiting", "pending", "running"].some((part) => status.includes(part))) return "warn";
  if (["failed", "error", "denied", "rejected"].some((part) => status.includes(part))) return "bad";
  return "neutral";
}

function App() {
  const [page, setPage] = useState<Page>(pageFromLocation);
  const [studioWorkflowId, setStudioWorkflowId] = useState<string | null>(() =>
    new URLSearchParams(window.location.search).get("workflow"),
  );
  const [studioPrompt, setStudioPrompt] = useState<string | null>(() =>
    new URLSearchParams(window.location.search).get("prompt"),
  );
  const [mobileMenu, setMobileMenu] = useState(false);
  const [role, setRole] = useState(() => localStorage.getItem("ueoa-role") || "user");
  const [roles, setRoles] = useState<string[]>([]);
  const [data, setData] = useState<Data>(EMPTY);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [notice, setNotice] = useState<{ text: string; tone: "good" | "bad" } | null>(null);
  const [inputsText, setInputsText] = useState("{}");
  const [selectedRun, setSelectedRun] = useState<RunDetail | null>(null);
  const [cancellingRunId, setCancellingRunId] = useState<string | null>(null);
  const [draftGoal, setDraftGoal] = useState("");
  const [draft, setDraft] = useState<JsonRecord | null>(null);
  const [workflowName, setWorkflowName] = useState("");
  const [registerUrl, setRegisterUrl] = useState("");
  const [busy, setBusy] = useState(false);

  const announce = useCallback((text: string, tone: "good" | "bad" = "good") => {
    setNotice({ text, tone });
    window.setTimeout(() => setNotice(null), 5000);
  }, []);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const base = await Promise.all([
        api<string[]>("/api/v1/roles", role),
        api<Health>("/api/v1/health", role),
      ]);
      const [availableRoles, health] = base;
      setRoles(availableRoles);
      const activeRole = availableRoles.includes(role)
        ? role
        : availableRoles.includes(health.default_role)
          ? health.default_role
          : availableRoles[0] || role;
      if (activeRole !== role) setRole(activeRole);

      const requests = await Promise.allSettled([
        api<Capability[]>("/api/v1/capabilities", activeRole),
        api<Implementation[]>("/api/v1/implementations", activeRole),
        api<JsonRecord[]>("/api/v1/agents", activeRole),
        api<WorkflowSummary[]>("/api/v1/workflows", activeRole),
        api<RunSummary[]>("/api/v1/runs", activeRole),
        api<Approval[]>("/api/v1/approvals", activeRole),
        api<AuditRecord[]>("/api/v1/audit", activeRole),
        api<Skill[]>("/api/v1/skills", activeRole),
        api<McpServer[]>("/api/v1/mcp-servers", activeRole),
        api<OverviewStats>("/api/v1/overview", activeRole),
      ]);
      const next: Data = { ...EMPTY, health };
      const labels = ["capabilities", "implementations", "agents", "workflows", "runs", "approvals", "audit", "skills", "mcpServers", "overview"] as const;
      const failures: string[] = [];
      requests.forEach((result, index) => {
        if (result.status === "fulfilled") {
          Object.assign(next, { [labels[index]]: result.value });
        } else {
          failures.push(`${labels[index]}: ${result.reason instanceof Error ? result.reason.message : String(result.reason)}`);
        }
      });
      setData(next);
      setLoading(false);
      if (failures.length) announce(`Some console data could not be loaded. ${failures.join(" · ")}`, "bad");
    } catch (error) {
      setLoading(false);
      announce(error instanceof Error ? error.message : String(error), "bad");
    } finally {
      setRefreshing(false);
    }
  }, [announce, role]);

  useEffect(() => {
    localStorage.setItem("ueoa-role", role);
  }, [role]);
  useEffect(() => {
    const syncRoute = () => {
      setPage(pageFromLocation());
      const params = new URLSearchParams(window.location.search);
      setStudioWorkflowId(params.get("workflow"));
      setStudioPrompt(params.get("prompt"));
    };
    window.addEventListener("popstate", syncRoute);
    return () => window.removeEventListener("popstate", syncRoute);
  }, []);
  useEffect(() => {
    void refresh().catch((error: unknown) => {
      setLoading(false);
      setRefreshing(false);
      announce(error instanceof Error ? error.message : String(error), "bad");
    });
  }, [refresh, announce]);

  const activeRun = selectedRun && ["pending", "running", "waiting"].includes(selectedRun.status);
  useEffect(() => {
    if (!selectedRun || !activeRun) return;
    const timer = window.setTimeout(() => {
      void api<RunDetail>(`/api/v1/runs/${encodeURIComponent(selectedRun.id)}?wait_s=1`, role)
        .then((run) => setSelectedRun(run))
        .catch((error: unknown) => announce(error instanceof Error ? error.message : String(error), "bad"));
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [activeRun, announce, role, selectedRun]);
  useEffect(() => {
    if (selectedRun && !activeRun) void refresh();
  }, [activeRun, refresh, selectedRun?.id]);

  const groups = useMemo(() => [...new Set(NAV.map((item) => item.group))], []);
  const navigate = (next: Page, workflowId?: string, prompt?: string) => {
    setPage(next);
    const query = new URLSearchParams();
    if (workflowId) query.set("workflow", workflowId);
    if (prompt) query.set("prompt", prompt);
    const suffix = query.size ? `?${query.toString()}` : "";
    const path = next === "Workflow"
      ? `/create/workflow${suffix}`
      : next === "Saved workflows" ? "/create/saved-workflows"
        : next === "Prompt Builder" ? "/create/prompt" : "/";
    setStudioWorkflowId(next === "Workflow" ? workflowId || null : null);
    setStudioPrompt(next === "Workflow" ? prompt || null : null);
    if (window.location.href !== new URL(path, window.location.origin).href) window.history.pushState({}, "", path);
    setMobileMenu(false);
  };

  async function submitRun(workflowId?: string) {
    let parsedInputs: JsonRecord;
    try {
      const parsed: unknown = JSON.parse(inputsText);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        throw new Error("Inputs must be a JSON object.");
      }
      parsedInputs = parsed as JsonRecord;
    } catch (error) {
      announce(error instanceof Error ? error.message : "Inputs must be valid JSON.", "bad");
      return;
    }
    if (!workflowId) {
      announce("Select a saved workflow before starting a run.", "bad");
      return;
    }
    setBusy(true);
    try {
      const run = await api<RunDetail>("/api/v1/runs", role, {
        method: "POST",
        body: { workflow_id: workflowId, inputs: parsedInputs },
      });
      setSelectedRun(run);
      setPage("Runs");
      announce(`Run ${run.id} started.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    } finally {
      setBusy(false);
    }
  }

  async function loadRun(id: string) {
    try {
      setSelectedRun(await api<RunDetail>(`/api/v1/runs/${encodeURIComponent(id)}`, role));
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    }
  }

  async function cancelRun(id: string) {
    setCancellingRunId(id);
    try {
      const run = await api<RunDetail>(`/api/v1/runs/${encodeURIComponent(id)}/cancel`, role, { method: "POST" });
      setSelectedRun(run);
      announce(`Run ${id} stopped.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    } finally {
      setCancellingRunId(null);
    }
  }

  async function registerImplementation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    try {
      const result = await api<Implementation>("/api/v1/implementations", role, {
        method: "POST",
        body: { url: registerUrl.trim() },
      });
      setRegisterUrl("");
      announce(`Registered ${result.key} as ${result.status}.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    } finally {
      setBusy(false);
    }
  }

  async function registerMcpServer(server: Pick<McpServer, "name" | "url" | "description">) {
    setBusy(true);
    try {
      const result = await api<McpServer>("/api/v1/mcp-servers", role, {
        method: "POST",
        body: server,
      });
      announce(`Registered MCP server ${result.name}.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function removeMcpServer(server: McpServer) {
    try {
      await api(`/api/v1/mcp-servers/${encodeURIComponent(server.id)}`, role, { method: "DELETE" });
      announce(`Removed MCP server ${server.name}.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    }
  }

  async function approveImplementation(key: string) {
    try {
      await api(`/api/v1/implementations/${encodeURIComponent(key)}/approve`, role, {
        method: "POST",
        body: {},
      });
      announce(`Approved ${key}.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    }
  }

  async function previewAgent(definition: JsonRecord) {
    return api<JsonRecord>("/api/v1/agents/preview", role, {
      method: "POST",
      body: { definition },
    });
  }

  async function generateAgentConfiguration(name: string, description: string) {
    return api<JsonRecord>("/api/v1/agents/generate", role, {
      method: "POST",
      body: { name, description },
    });
  }

  async function saveAgent(definition: JsonRecord, editingId?: string) {
    const saved = await api<JsonRecord>(
      editingId ? `/api/v1/agents/${encodeURIComponent(editingId)}` : "/api/v1/agents",
      role,
      { method: editingId ? "PUT" : "POST", body: { definition } },
    );
    const registrationError = saved.registration_error;
    if (typeof registrationError === "string" && registrationError) {
      announce(`Agent saved, but registry registration failed: ${registrationError}`, "bad");
    } else {
      announce(`Agent ${editingId ? "updated" : "created"} successfully.`);
    }
    await refresh();
    return saved;
  }

  async function retryAgentRegistration(agentId: string) {
    try {
      await api(`/api/v1/agents/${encodeURIComponent(agentId)}/register`, role, { method: "POST" });
      announce(`Agent ${agentId} registered and approved.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    }
  }

  async function removeAgent(agentId: string) {
    try {
      await api(`/api/v1/agents/${encodeURIComponent(agentId)}`, role, { method: "DELETE" });
      announce(`Removed agent ${agentId}.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    }
  }

  async function saveSkill(skill: Pick<Skill, "name" | "description" | "instructions" | "tags">, editingId?: string) {
    const skillId = editingId || skill.name.trim().toLowerCase().replace(/\s+/g, "-");
    try {
      await api(`/api/v1/skills${editingId ? `/${encodeURIComponent(skillId)}` : ""}`, role, {
        method: editingId ? "PUT" : "POST",
        body: skill,
      });
      announce(`Skill ${editingId ? "updated" : "registered"}.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
      throw error;
    }
  }

  async function removeSkill(skillId: string) {
    try {
      await api(`/api/v1/skills/${encodeURIComponent(skillId)}`, role, { method: "DELETE" });
      announce(`Removed skill ${skillId}.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    }
  }

  async function getAgent(agentId: string) {
    return api<JsonRecord>(`/api/v1/agents/${encodeURIComponent(agentId)}`, role);
  }

  async function draftWorkflow() {
    if (!draftGoal.trim()) {
      announce("Describe the workflow you want to draft.", "bad");
      return;
    }
    setBusy(true);
    try {
      const result = await api<JsonRecord>("/api/v1/workflows/draft", role, {
        method: "POST",
        body: { goal: draftGoal.trim() },
      });
      setDraft(result);
      const workflow = result.workflow as { id?: string } | undefined;
      setWorkflowName(workflow?.id || "");
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    } finally {
      setBusy(false);
    }
  }

  async function saveWorkflow() {
    const workflow = draft?.workflow;
    const workflowId = workflowName.trim();
    if (!workflow || !workflowId) {
      announce("Create a valid workflow draft before saving.", "bad");
      return;
    }
    setBusy(true);
    try {
      const validation = await api<{ ok: boolean; issues: unknown[] }>("/api/v1/workflows/validate", role, {
        method: "POST",
        body: { workflow },
      });
      if (!validation.ok) {
        announce(`Workflow validation failed: ${JSON.stringify(validation.issues)}`, "bad");
        return;
      }
      await api(`/api/v1/workflows/${encodeURIComponent(workflowId)}`, role, {
        method: "PUT",
        body: { workflow, name: workflowId, description: draftGoal },
      });
      setDraft(null);
      setDraftGoal("");
      announce(`Saved workflow ${workflowId}.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    } finally {
      setBusy(false);
    }
  }

  async function loadStudioWorkflow(id: string): Promise<WorkflowDocument> {
    return api<WorkflowDocument>(`/api/v1/workflows/${encodeURIComponent(id)}`, role);
  }

  async function validateStudioWorkflow(workflow: Workflow): Promise<Validation> {
    return api<Validation>("/api/v1/workflows/validate", role, { method: "POST", body: { workflow } });
  }

  async function saveStudioWorkflow(id: string, workflow: Workflow, name: string, description: string): Promise<WorkflowDocument> {
    const saved = await api<WorkflowDocument>(`/api/v1/workflows/${encodeURIComponent(id)}`, role, {
      method: "PUT",
      body: { workflow, name, description },
    });
    await refresh();
    return saved;
  }

  async function generateStudioWorkflow(goal: string): Promise<{
    ok: boolean;
    workflow: Workflow | null;
    rationale?: string;
    issues?: string[];
    considered?: string[];
  }> {
    return api<{
      ok: boolean;
      workflow: Workflow | null;
      rationale?: string;
      issues?: string[];
      considered?: string[];
    }>("/api/v1/workflows/draft", role, {
      method: "POST",
      body: { goal },
    });
  }

  async function generateProfessionalPrompt(brief: PromptBrief): Promise<string> {
    const result = await api<{ prompt: string }>("/api/v1/workflows/prompt", role, {
      method: "POST",
      body: brief,
    });
    return result.prompt;
  }

  async function deleteSavedWorkflow(id: string) {
    try {
      await api(`/api/v1/workflows/${encodeURIComponent(id)}`, role, { method: "DELETE" });
      announce(`Deleted workflow draft ${id}.`);
      await refresh();
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    }
  }

  async function runStudioWorkflow(workflow: Workflow, inputs: JsonRecord): Promise<RunResult> {
    const run = await api<RunDetail>("/api/v1/runs", role, {
      method: "POST",
      body: { workflow, inputs, wait_s: 5 },
    });
    setSelectedRun(run);
    await refresh();
    return run;
  }

  async function decideApproval(approval: Approval, decision: "approve" | "reject") {
    try {
      await api(`/api/v1/runs/${encodeURIComponent(approval.run_id)}/approvals/${encodeURIComponent(approval.node)}`, role, {
        method: "POST",
        body: {
          decision,
          payload_hash: approval.payload_hash,
          ...(approval.item === null ? {} : { item: approval.item }),
        },
      });
      announce(`Approval ${decision === "approve" ? "approved" : "rejected"} for ${approval.node}.`);
      await refresh();
      if (selectedRun?.id === approval.run_id) await loadRun(approval.run_id);
    } catch (error) {
      announce(error instanceof Error ? error.message : String(error), "bad");
    }
  }

  function viewAudit() {
    navigate("Audit");
    window.requestAnimationFrame(() => document.getElementById("audit")?.scrollIntoView({ behavior: "smooth" }));
  }

  return (
    <div className="app-shell">
      {notice && <div className={`toast ${notice.tone}`} role="status">{notice.text}</div>}
      <header className="topbar">
        <button className="icon-button mobile-toggle" aria-label="Toggle navigation" onClick={() => setMobileMenu(!mobileMenu)}>
          {mobileMenu ? <X size={18} /> : <Menu size={18} />}
        </button>
        <button className="brand" onClick={() => navigate("Overview")}>
          <span className="brand-mark">UO</span>
          <span><b>UEOA</b><small>Enterprise Orchestrator</small></span>
        </button>
        <div className="topbar-right">
          <span className={`health ${data.health?.status === "ok" ? "online" : ""}`}>
            <i />{data.health?.status === "ok" ? "API online" : loading ? "Connecting" : "API unavailable"}
          </span>
          <label className="role-picker"><span>Role</span>
            <select value={role} onChange={(event) => setRole(event.target.value)}>
              {(roles.length ? roles : [role]).map((availableRole) =>
                <option key={availableRole} value={availableRole}>{availableRole}</option>)}
            </select>
          </label>
          <button className="icon-button" aria-label="Refresh data" onClick={() => void refresh()} disabled={refreshing}>
            <RefreshCw size={17} className={refreshing ? "spin" : ""} />
          </button>
        </div>
      </header>

      <div className="layout">
        <aside className={`sidebar ${mobileMenu ? "open" : ""}`}>
          {groups.map((group) => (
            <div className="nav-group" key={group}>
              <div className="nav-label">{group}</div>
              {NAV.filter((item) => item.group === group).map(({ label, icon: Icon }) => (
                <button key={label} className={`nav-item ${page === label ? "selected" : ""}`} onClick={() => navigate(label)}>
                  <Icon size={16} /><span>{label}</span>
                  {label === "Approvals" && data.approvals.length > 0 && <span className="nav-count">{data.approvals.length}</span>}
                </button>
              ))}
            </div>
          ))}
          <div className="sidebar-foot">
            <span className="sidebar-dot" /> Connected to this UEOA instance
          </div>
        </aside>

        <main className="content">
          <div className="page-heading">
            <div><p className="eyebrow">CONTROL PLANE</p><h1>{page}</h1></div>
            <div className="heading-meta">
              <span>{data.health?.approved ?? 0} approved implementations</span>
              <span className="dot-separator">·</span>
              <span>{data.health?.capabilities ?? 0} capabilities</span>
            </div>
          </div>

          {loading && <div className="loading"><LoaderCircle className="spin" size={22} /> Loading live project data…</div>}
          {!loading && page === "Overview" && <Overview data={data} onNavigate={navigate} onSelectRun={(id) => {
            navigate("Runs");
            void loadRun(id);
          }} />}
          {!loading && page === "Orchestrator" && (
            <Orchestrator
              data={data} role={role} selectedRun={selectedRun}
              onSelectedRun={setSelectedRun} onClearRun={() => setSelectedRun(null)}
              onCancelRun={cancelRun} cancellingRunId={cancellingRunId}
              onViewAudit={viewAudit} onViewApprovals={() => navigate("Approvals")}
            />
          )}
          {!loading && page === "Workflow" && (
            <WorkflowStudio
              capabilities={data.capabilities}
              agents={data.agents}
              skills={data.skills}
              mcpServers={data.mcpServers}
              liveRun={selectedRun}
              initialWorkflowId={studioWorkflowId}
              initialPrompt={studioPrompt}
              onLoad={loadStudioWorkflow}
              onValidate={validateStudioWorkflow}
              onSave={saveStudioWorkflow}
              onDraft={generateStudioWorkflow}
              onRun={runStudioWorkflow}
            />
          )}
          {!loading && page === "Prompt Builder" && (
            <PromptBuilder
              onGenerate={generateProfessionalPrompt}
              onUsePrompt={(prompt) => navigate("Workflow", undefined, prompt)}
            />
          )}
          {!loading && page === "Saved workflows" && (
            <SavedWorkflows
              workflows={data.workflows}
              onEdit={(id) => navigate("Workflow", id)}
              onDelete={(id) => void deleteSavedWorkflow(id)}
              onCreate={() => navigate("Workflow")}
            />
          )}
          {!loading && page === "Agent Registry" && (
            <Registry
              data={data} role={role} registerUrl={registerUrl} setRegisterUrl={setRegisterUrl}
              onRegister={registerImplementation} onApprove={approveImplementation} busy={busy}
              onPreviewAgent={previewAgent} onSaveAgent={saveAgent} onDeleteAgent={removeAgent} onGetAgent={getAgent}
              onGenerateAgent={generateAgentConfiguration} onRetryAgentRegistration={retryAgentRegistration}
            />
          )}
          {!loading && page === "Workflows" && (
            <Workflows
              data={data} inputsText={inputsText} setInputsText={setInputsText}
              draftGoal={draftGoal} setDraftGoal={setDraftGoal} draft={draft}
              workflowName={workflowName} setWorkflowName={setWorkflowName} busy={busy}
              onDraft={draftWorkflow} onSave={saveWorkflow} onRun={(id) => void submitRun(id)}
            />
          )}
          {!loading && page === "Skills" && (
            <Skills data={data} role={role} onSave={saveSkill} onDelete={removeSkill} />
          )}
          {!loading && page === "MCP Registry" && (
            <MCP servers={data.mcpServers} role={role} busy={busy} onRegister={registerMcpServer} onDelete={removeMcpServer} />
          )}
          {!loading && page === "Runs" && (
            <Runs data={data} selectedRun={selectedRun} onSelect={loadRun}
              onCancelRun={cancelRun} cancellingRunId={cancellingRunId} />
          )}
          {!loading && page === "Approvals" && (
            <Approvals data={data} onDecide={decideApproval} />
          )}
          {!loading && page === "Audit" && <Audit data={data} />}
        </main>
      </div>
    </div>
  );
}

function Overview({ data, onNavigate, onSelectRun }: {
  data: Data; onNavigate: (page: Page) => void; onSelectRun: (id: string) => void;
}) {
  const active = data.runs.filter((run) => ["pending", "running", "waiting"].includes(run.status)).length;
  const stats = data.overview;
  return (
    <div className="stack">
      <section className="welcome-card">
        <div><span className="welcome-label">UEOA PLATFORM</span><h2>Orchestrate your agents with confidence.</h2>
          <p>Monitor registered implementations, run workflows, and review approvals from one place.</p></div>
        <button className="primary-button" onClick={() => onNavigate("Orchestrator")}><Play size={15} /> New run</button>
      </section>
      <div className="stat-grid">
        <Stat icon={Bot} label="Total agents" value={stats?.total_agents ?? data.agents.length} detail={`${stats?.active_agents ?? data.health?.approved ?? 0} active`} />
        <Stat icon={Check} label="Active agents" value={stats?.active_agents ?? data.health?.approved ?? 0} detail="Approved and available" />
        <Stat icon={GitBranch} label="Total workflows" value={stats?.total_workflows ?? data.workflows.length} detail="Saved templates" />
        <Stat icon={CircleHelp} label="Total skills" value={stats?.total_skills ?? data.skills.length} detail="Defaults and custom" />
        <Stat icon={Database} label="Registered MCP servers" value={stats?.registered_mcp_servers ?? 0} detail="MCP registry not configured" />
        <Stat icon={Activity} label="Total runs" value={stats?.total_runs ?? data.runs.length} detail={`${active} currently active`} />
        <Stat icon={Check} label="Successful runs" value={stats?.successful_runs ?? 0} detail="Completed successfully" />
        <Stat icon={AlertCircle} label="Failed runs" value={stats?.failed_runs ?? 0} detail="Failed or cancelled" />
      </div>
      <div className="two-column">
        <section className="panel">
          <div className="panel-heading"><div><h2>Recent runs</h2><p>Live execution history for the selected role</p></div>
            <button className="text-button" onClick={() => onNavigate("Runs")}>View all <ArrowRight size={14} /></button></div>
          <RunList runs={data.runs.slice(0, 5)} onSelect={onSelectRun} />
        </section>
        <section className="panel">
          <div className="panel-heading"><div><h2>Platform health</h2><p>Current API-reported state</p></div><span className={`status ${statusTone(data.health?.status || "offline")}`}>{data.health?.status || "offline"}</span></div>
          <div className="health-list">
            <HealthRow label="Planner" value={data.health?.planner ? "Configured" : "Unavailable"} good={Boolean(data.health?.planner)} />
            <HealthRow label="Default role" value={data.health?.default_role || "—"} />
            <HealthRow label="Approved implementations" value={String(data.health?.approved ?? 0)} />
          </div>
          <section className="panel">
            <div className="panel-heading"><div><h2>Recent activities</h2><p>Latest runs and gateway invocations visible to your role</p></div></div>
            {stats?.recent_activities.length ? <div className="activity-list">{stats.recent_activities.map((item) =>
              <div className="activity-row" key={`${item.type}-${item.id}`}>
                <span className="activity-description"><b>{item.label}</b><small>{item.type} · {formatTime(item.at)}</small></span>
                <Status value={item.status} />
              </div>)}</div> : <EmptyState text="No recent platform activity." />}
          </section>
        </section>
      </div>
      <section className="panel">
        <div className="panel-heading"><div><h2>Capability catalog</h2><p>Contracts available on this instance</p></div><span className="subtle-count">{data.capabilities.length} total</span></div>
        {data.capabilities.length ? <div className="table-wrap"><table><thead><tr><th>Capability</th><th>Effect</th><th>Implementations</th></tr></thead>
          <tbody>{data.capabilities.slice(0, 8).map((cap) => <tr key={cap.id}><td><b>{cap.id}</b><small className="cell-detail">{cap.description}</small></td>
            <td><Status value={cap.effects} /></td><td>{cap.implementations.length ? cap.implementations.join(", ") : <span className="muted">None</span>}</td></tr>)}</tbody></table></div>
          : <EmptyState text="No capabilities are registered yet." /> }
      </section>
    </div>
  );
}

const CHAT_STORAGE_KEY = "ueoa-orchestrator-thread";
const CHAT_TERMINAL_FAILURES = new Set(["failed", "cancelled", "completed_rejected"]);
const TEXT_MEDIA_TYPES: Record<string, string> = {
  txt: "text/plain",
  md: "text/markdown",
  csv: "text/csv",
  json: "application/json",
};

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function loadChatMessages(): { messages: ChatMessage[]; warning: string } {
  try {
    const raw: unknown = JSON.parse(window.sessionStorage.getItem(CHAT_STORAGE_KEY) || "[]");
    if (!Array.isArray(raw)) return { messages: [], warning: "Saved chat history had an invalid format and was cleared." };
    const messages = raw.filter((message): message is ChatMessage =>
      isRecord(message) && typeof message.id === "string" &&
      (message.role === "user" || message.role === "assistant") && typeof message.text === "string");
    return { messages, warning: "" };
  } catch {
    return { messages: [], warning: "Chat history could not be loaded from this browser session." };
  }
}

function newChatId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function apiErrorText(error: unknown): string {
  if (!(error instanceof Error)) return String(error);
  if (error instanceof ApiError) {
    try {
      const detail: unknown = JSON.parse(error.message);
      if (typeof detail === "string") return detail;
      if (isRecord(detail)) {
        const reason = typeof detail.rationale === "string" ? detail.rationale : "";
        const issues = Array.isArray(detail.issues)
          ? detail.issues.map((issue) => isRecord(issue) && typeof issue.message === "string" ? issue.message : "").filter(Boolean)
          : [];
        return [reason, ...issues].filter(Boolean).join(" ") || error.message;
      }
    } catch {
      return error.message;
    }
  }
  return error.message;
}

function progressStatus(status: string): string {
  if (["succeeded", "completed"].includes(status)) return "completed";
  if (status.startsWith("waiting")) return "waiting";
  return status;
}

function markdownInline(source: string, keyPrefix: string): React.ReactNode[] {
  const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  const result: React.ReactNode[] = [];
  let cursor = 0;
  let index = 0;
  for (const match of source.matchAll(pattern)) {
    const text = match[0];
    const start = match.index ?? 0;
    if (start > cursor) result.push(source.slice(cursor, start));
    if (text.startsWith("**")) {
      result.push(<strong key={`${keyPrefix}-${index}`}>{markdownInline(text.slice(2, -2), `${keyPrefix}-${index}`)}</strong>);
    } else if (text.startsWith("*")) {
      result.push(<em key={`${keyPrefix}-${index}`}>{markdownInline(text.slice(1, -1), `${keyPrefix}-${index}`)}</em>);
    } else if (text.startsWith("`")) {
      result.push(<code key={`${keyPrefix}-${index}`}>{text.slice(1, -1)}</code>);
    } else {
      const label = text.match(/^\[([^\]]+)\]\([^)]+\)$/)?.[1] || text;
      result.push(<span className="markdown-link" key={`${keyPrefix}-${index}`}>{label}</span>);
    }
    cursor = start + text.length;
    index += 1;
  }
  if (cursor < source.length) result.push(source.slice(cursor));
  return result;
}

function Markdown({ source }: { source: string }) {
  const lines = source.split(/\r?\n/);
  const blocks: React.ReactNode[] = [];
  let index = 0;
  const isTableRow = (line: string) => line.trim().startsWith("|") && line.trim().endsWith("|");
  const isTableRule = (line: string) => /^\s*\|?[\s:|-]+\|?\s*$/.test(line) && line.includes("-");
  const isBlockStart = (line: string) =>
    /^#{1,6}\s/.test(line) || /^```/.test(line) || /^\s*>/.test(line) ||
    /^\s*[-*+]\s/.test(line) || /^\s*\d+[.)]\s/.test(line);

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }
    if (line.startsWith("```")) {
      const code: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) code.push(lines[index++]);
      if (index < lines.length) index += 1;
      blocks.push(<pre className="markdown-code" key={`code-${index}`}><code>{code.join("\n")}</code></pre>);
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = heading[1].length;
      const children = markdownInline(heading[2], `heading-${index}`);
      const key = `heading-${index++}`;
      if (level === 1) blocks.push(<h1 key={key}>{children}</h1>);
      else if (level === 2) blocks.push(<h2 key={key}>{children}</h2>);
      else if (level === 3) blocks.push(<h3 key={key}>{children}</h3>);
      else if (level === 4) blocks.push(<h4 key={key}>{children}</h4>);
      else if (level === 5) blocks.push(<h5 key={key}>{children}</h5>);
      else blocks.push(<h6 key={key}>{children}</h6>);
      continue;
    }
    if (/^\s*>/.test(line)) {
      const quote: string[] = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) quote.push(lines[index++].replace(/^\s*>\s?/, ""));
      blocks.push(<blockquote key={`quote-${index}`}>{quote.map((text, item) =>
        <p key={`quote-${index}-${item}`}>{markdownInline(text, `quote-${index}-${item}`)}</p>)}</blockquote>);
      continue;
    }
    if (isTableRow(line) && index + 1 < lines.length && isTableRule(lines[index + 1])) {
      const cells = (row: string) => row.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
      const headers = cells(line);
      index += 2;
      const rows: string[][] = [];
      while (index < lines.length && isTableRow(lines[index])) rows.push(cells(lines[index++]));
      blocks.push(<div className="markdown-table-wrap" key={`table-${index}`}><table><thead><tr>{headers.map((cell, cellIndex) =>
        <th key={`th-${cellIndex}`}>{markdownInline(cell, `th-${index}-${cellIndex}`)}</th>)}</tr></thead>
        <tbody>{rows.map((row, rowIndex) => <tr key={`tr-${rowIndex}`}>{headers.map((_, cellIndex) =>
          <td key={`td-${rowIndex}-${cellIndex}`}>{markdownInline(row[cellIndex] || "", `td-${index}-${rowIndex}-${cellIndex}`)}</td>)}</tr>)}</tbody>
      </table></div>);
      continue;
    }
    const listMatch = line.match(/^\s*(?:[-*+]\s+|\d+[.)]\s+)/);
    if (listMatch) {
      const ordered = /^\s*\d/.test(line);
      const items: string[] = [];
      while (index < lines.length && /^\s*(?:[-*+]\s+|\d+[.)]\s+)/.test(lines[index])) {
        items.push(lines[index++].replace(/^\s*(?:[-*+]\s+|\d+[.)]\s+)/, ""));
      }
      const listItems = items.map((item, itemIndex) =>
        <li key={`list-${index}-${itemIndex}`}>{markdownInline(item, `list-${index}-${itemIndex}`)}</li>);
      blocks.push(ordered ? <ol key={`ol-${index}`}>{listItems}</ol> : <ul key={`ul-${index}`}>{listItems}</ul>);
      continue;
    }
    const paragraph: string[] = [];
    while (index < lines.length && lines[index].trim() && !isBlockStart(lines[index]) &&
      !(isTableRow(lines[index]) && index + 1 < lines.length && isTableRule(lines[index + 1]))) {
      paragraph.push(lines[index++]);
    }
    if (paragraph.length) {
      blocks.push(<p key={`p-${index}`}>{paragraph.map((text, lineIndex) =>
        <span key={`line-${index}-${lineIndex}`}>{lineIndex > 0 && <br />}{markdownInline(text, `p-${index}-${lineIndex}`)}</span>)}</p>);
    } else {
      index += 1;
    }
  }
  return <div className="markdown-content">{blocks}</div>;
}

function Orchestrator({ data, role, selectedRun, onSelectedRun, onClearRun, onCancelRun, cancellingRunId, onViewAudit, onViewApprovals }: {
  data: Data; role: string; selectedRun: RunDetail | null;
  onSelectedRun: (run: RunDetail) => void; onClearRun: () => void;
  onCancelRun: (id: string) => void; cancellingRunId: string | null;
  onViewAudit: () => void; onViewApprovals: () => void;
}) {
  const loaded = useMemo(loadChatMessages, []);
  const [messages, setMessages] = useState<ChatMessage[]>(loaded.messages);
  const [staged, setStaged] = useState<StagedAttachment[]>([]);
  const [selectedCapabilities, setSelectedCapabilities] = useState<string[]>([]);
  const [chatText, setChatText] = useState("");
  const [runInputText, setRunInputText] = useState("{}");
  const [mode, setMode] = useState("");
  const [sending, setSending] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [composerError, setComposerError] = useState("");
  const [persistenceWarning, setPersistenceWarning] = useState(loaded.warning);
  const [copyStatus, setCopyStatus] = useState("");
  const threadRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const requestByMessage = useRef<Map<string, ChatRequest>>(new Map());
  const restoreAttempted = useRef<Set<string>>(new Set());
  const workflowChoices = useMemo(() => {
    const latestPublished = new Map<string, WorkflowSummary>();
    for (const workflow of data.workflows) {
      if (!workflow.published) continue;
      const family = workflow.version_ref?.split("@", 1)[0] || workflow.id;
      const current = latestPublished.get(family);
      if (!current || (workflow.version || "").localeCompare(current.version || "", undefined, { numeric: true }) > 0) {
        latestPublished.set(family, workflow);
      }
    }
    const latestIds = new Set([...latestPublished.values()].map((workflow) => workflow.id));
    return data.workflows
      .filter((workflow) => !workflow.published || latestIds.has(workflow.id))
      .sort((left, right) =>
        Number(left.builtin) - Number(right.builtin) ||
        Number(left.published) - Number(right.published) ||
        (left.name || left.id).localeCompare(right.name || right.id));
  }, [data.workflows]);

  useEffect(() => {
    try {
      const history = messages.map((message) => message.run
        ? { ...message, run: { ...message.run, inputs: undefined } }
        : message);
      window.sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(history));
      setPersistenceWarning("");
    } catch {
      setPersistenceWarning("This conversation could not be saved in session storage. Large results may exceed browser storage limits.");
    }
  }, [messages]);
  useEffect(() => {
    if (threadRef.current) threadRef.current.scrollTop = threadRef.current.scrollHeight;
  }, [messages]);
  useEffect(() => {
    if (!selectedRun) return;
    setMessages((current) => {
      let changed = false;
      const next = current.map((message) => {
        if (message.role !== "assistant" || message.run?.id !== selectedRun.id) return message;
        changed = true;
        const completed = selectedRun.status === "completed";
        const cancelled = selectedRun.status === "cancelled";
        const failed = CHAT_TERMINAL_FAILURES.has(selectedRun.status) && !cancelled;
        const stage: ChatMessage["stage"] = completed ? "completed" : cancelled ? "cancelled" : failed ? "failed" : "running";
        return {
          ...message,
          run: selectedRun,
          stage,
          statusText: completed ? "Completed" : cancelled ? "Stopped" : failed ? "Run failed" : selectedRun.status === "waiting" ? "Waiting for input or approval" : "Running...",
          error: failed ? selectedRun.error || Object.values(selectedRun.nodes).find((node) => node.error)?.error || "The run failed." :
            cancelled ? undefined : message.error,
        };
      });
      return changed ? next : current;
    });
  }, [selectedRun]);
  useEffect(() => {
    const activeMessage = [...messages].reverse().find((message) =>
      message.role === "assistant" && message.run &&
      ["pending", "running", "waiting"].includes(message.run.status));
    const runId = activeMessage?.run?.id;
    if (!activeMessage || !runId || selectedRun?.id === runId || restoreAttempted.current.has(runId)) return;
    restoreAttempted.current.add(runId);
    void api<RunDetail>(`/api/v1/runs/${encodeURIComponent(runId)}`, role)
      .then(onSelectedRun)
      .catch((error: unknown) => updateMessage(activeMessage.id, (message) => ({
        ...message, statusText: "Could not reload run status", error: apiErrorText(error),
      })));
  }, [messages, onSelectedRun, role, selectedRun?.id]);
  useEffect(() => {
    if (!menuOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [menuOpen]);
  useEffect(() => {
    const currentWorkflow = mode.startsWith("workflow:") ? mode.slice("workflow:".length) : "";
    if (!workflowChoices.some((workflow) => workflow.id === currentWorkflow)) {
      setMode(workflowChoices[0] ? `workflow:${workflowChoices[0].id}` : "");
    }
  }, [mode, workflowChoices]);

  function updateMessage(id: string, update: (message: ChatMessage) => ChatMessage) {
    setMessages((current) => current.map((message) => message.id === id ? update(message) : message));
  }

  function clearChat() {
    setMessages([]);
    setStaged([]);
    setSelectedCapabilities([]);
    setChatText("");
    setRunInputText("{}");
    setComposerError("");
    requestByMessage.current.clear();
    onClearRun();
    try {
      window.sessionStorage.removeItem(CHAT_STORAGE_KEY);
      setPersistenceWarning("");
    } catch {
      setPersistenceWarning("The conversation was cleared, but browser storage could not be updated.");
    }
  }

  async function addFiles(files: FileList | File[]) {
    const added: StagedAttachment[] = [];
    const errors: string[] = [];
    for (const file of Array.from(files)) {
      const extension = file.name.split(".").pop()?.toLowerCase() || "";
      const isPdf = extension === "pdf";
      if (!isPdf && !["txt", "md", "csv", "json"].includes(extension)) {
        errors.push(`${file.name}: supported attachments are PDF, TXT, MD, CSV, and JSON.`);
        continue;
      }
      const limit = (isPdf ? 4 : 2) * 1024 * 1024;
      if (file.size > limit) {
        errors.push(`${file.name} exceeds the ${isPdf ? "4 MB" : "2 MB"} attachment limit.`);
        continue;
      }
      try {
        let encoded: EncodedAttachment;
        if (isPdf) {
          const bytes = new Uint8Array(await file.arrayBuffer());
          let binary = "";
          for (let offset = 0; offset < bytes.length; offset += 0x8000) {
            binary += String.fromCharCode(...bytes.subarray(offset, Math.min(offset + 0x8000, bytes.length)));
          }
          encoded = { document_id: file.name, content_base64: btoa(binary), media_type: "application/pdf" };
        } else {
          encoded = { document_id: file.name, text: await file.text() };
        }
        const mediaType = isPdf ? "application/pdf" : TEXT_MEDIA_TYPES[extension];
        added.push({ id: newChatId(), name: file.name, size: file.size, mediaType, encoded });
      } catch (error) {
        errors.push(`${file.name} could not be read: ${apiErrorText(error)}`);
      }
    }
    if (added.length) {
      setStaged((current) => [...current, ...added]);
    }
    setComposerError(errors.join(" "));
  }

  async function sendMessage(selectedMode = mode, resend?: ChatRequest) {
    if (sending) return;
    const workflowId = selectedMode.startsWith("workflow:") ? selectedMode.slice("workflow:".length) : "";
    const workflow = data.workflows.find((item) => item.id === workflowId);
    if (!workflowId) {
      setComposerError("Choose a saved workflow before starting a run.");
      return;
    }
    if (!workflow) {
      setComposerError(`Workflow '${workflowId}' is no longer available.`);
      return;
    }

    const text = resend ? resend.text : chatText.trim();
    if (!text.trim()) {
      setComposerError("Type your goal before sending it to the selected workflow.");
      return;
    }
    const capabilities = resend?.capabilities ?? selectedCapabilities;
    const documents = resend?.documents ?? staged.map((item) => item.encoded);
    const attachmentMeta = resend?.attachmentMeta ?? staged.map(({ name, size, mediaType }) => ({ name, size, mediaType }));
    if (Object.values(selectedRun?.nodes || {}).some((node) => ["pending", "running", "waiting_input", "waiting_approval"].includes(node.state)) &&
      ["pending", "running", "waiting"].includes(selectedRun?.status || "")) {
      setComposerError("Wait for the current run to finish before starting another run.");
      return;
    }
    let additionalInputs: JsonRecord = {};
    if (!resend) {
      try {
        const parsed: unknown = JSON.parse(runInputText);
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
          throw new Error("Additional workflow inputs must be a JSON object.");
        }
        additionalInputs = parsed as JsonRecord;
      } catch (error) {
        setComposerError(error instanceof Error ? error.message : "Additional workflow inputs must be valid JSON.");
        return;
      }
    }
    const runInputs = {
      ...additionalInputs,
      ...(resend?.inputs ?? {}),
      ...(capabilities.length ? { capabilities } : {}),
      ...(documents.length ? { documents } : {}),
    };
    const displayText = text;
    const userMessage: ChatMessage = {
      id: newChatId(), role: "user", text: displayText,
      ...(attachmentMeta.length ? { attachments: attachmentMeta } : {}),
      ...(capabilities.length ? { selectedCapabilities: capabilities } : {}),
      ...(workflowId ? { workflowId } : {}),
    };
    const assistantId = newChatId();
    const assistantMessage: ChatMessage = {
      id: assistantId, role: "assistant", text: "", stage: "running",
      statusText: "Processing your goal with the saved workflow...",
      ...(workflowId ? { workflowId } : {}),
    };
    const request: ChatRequest = { text, inputs: runInputs, attachmentMeta, documents, capabilities };
    requestByMessage.current.set(assistantId, request);
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setStaged([]);
    setSelectedCapabilities([]);
    if (!resend) {
      setChatText("");
      setRunInputText("{}");
    }
    setComposerError("");
    setSending(true);

    try {
      const run = await api<RunDetail>("/api/v1/runs", role, {
        method: "POST",
        body: { workflow_id: workflowId, goal: text, inputs: runInputs, wait_s: 0 },
      });
      onSelectedRun(run);
      updateMessage(assistantId, (message) => ({
        ...message, run, stage: run.status === "completed" ? "completed" :
          run.status === "cancelled" ? "cancelled" :
            CHAT_TERMINAL_FAILURES.has(run.status) ? "failed" : "running",
        statusText: run.status === "completed" ? "Completed" :
          run.status === "cancelled" ? "Stopped" :
            CHAT_TERMINAL_FAILURES.has(run.status) ? "Run failed" : "Running...",
        ...(workflowId ? { plan: { workflow: run.workflow, rationale: run.plan?.rationale || `Running saved workflow for: ${text}` } } : {}),
      }));
    } catch (error) {
      const reason = apiErrorText(error);
      updateMessage(assistantId, (message) => ({
        ...message, stage: "failed", statusText: "Run could not start", error: reason,
      }));
    } finally {
      setSending(false);
    }
  }

  async function copyResult(run: RunDetail) {
    const result = resultMarkdown(run.outputs) ?? resultText(run.outputs) ?? pretty(run.outputs);
    try {
      await navigator.clipboard.writeText(result);
      setCopyStatus("Result copied.");
    } catch {
      setCopyStatus("Clipboard access was denied by the browser.");
    }
  }

  function rerunWithWorkflow(message: ChatMessage, workflowId: string) {
    const request = requestByMessage.current.get(message.id);
    setMode(`workflow:${workflowId}`);
    if (request) void sendMessage(`workflow:${workflowId}`, request);
    else setComposerError("The original attachment data is no longer available. Attach the files again and retry.");
  }

  return (
    <section className="orchestrator-shell" aria-label="Orchestrator chat">
      <div className="orchestrator-toolbar">
        <div><h2>Orchestrator</h2><p>Run saved workflows through your registered agents.</p></div>
        <button className="secondary-button compact" onClick={clearChat} aria-label="Start a new chat"><X size={14} /> New chat</button>
      </div>
      <div className="orchestrator-thread" ref={threadRef} role="log" aria-live="polite" aria-label="Conversation">
        {!messages.length && <div className="chat-welcome"><div className="chat-avatar"><Bot size={19} /></div>
          <h3>Choose a workflow to get started</h3><p>Add files or capabilities from the plus menu, then run one of your saved workflows.</p></div>}
        {messages.map((message) => <article key={message.id} className={`chat-message ${message.role}`}>
          {message.role === "assistant" && <div className="chat-avatar" aria-hidden="true"><Bot size={16} /></div>}
          <div className="chat-message-body">
            {message.role === "assistant" && <div className="chat-speaker">UEOA <span>{message.statusText}</span></div>}
            {message.role === "user" && message.workflowId && <span className="chat-workflow-note">Workflow: {message.workflowId}</span>}
            {message.role === "user" && <p className="chat-user-text">{message.text}</p>}
            {message.selectedCapabilities && <div className="chat-capability-list">{message.selectedCapabilities.map((capability) =>
              <span className="chat-capability-chip" key={capability}>{capability}</span>)}</div>}
            {message.attachments && <div className="chat-attachment-list">{message.attachments.map((file, index) =>
              <span className="chat-attachment" key={`${file.name}-${index}`}><FileText size={13} /> {file.name}
                <small>{(file.size / 1024).toFixed(0)} KB · {file.mediaType}</small></span>)}</div>}
            {message.role === "assistant" && message.plan && <PlanSummary plan={message.plan} />}
            {message.role === "assistant" && message.run && <RunProgress run={message.run} onViewApprovals={onViewApprovals}
              onCancelRun={onCancelRun} cancelling={cancellingRunId === message.run.id} />}
            {message.role === "assistant" && message.stage === "failed" && message.error && <div className="chat-error"><AlertCircle size={15} />{message.error}</div>}
            {message.role === "assistant" && message.fallbackWorkflows && message.fallbackWorkflows.length > 0 &&
              <div className="workflow-fallback"><p>Try a saved workflow instead:</p>{message.fallbackWorkflows.map((id) =>
                <button className="workflow-chip" key={id} onClick={() => rerunWithWorkflow(message, id)}>
                    <Play size={13} /> Run {id}{message.attachments?.length ? " with these attachments" : ""}
                </button>)}</div>}
            {message.role === "assistant" && message.run && <AssistantResult
              run={message.run} onCopy={() => void copyResult(message.run as RunDetail)} onViewAudit={onViewAudit}
            />}
          </div>
        </article>)}
      </div>
      {(persistenceWarning || composerError || copyStatus) && <div className={`chat-notice ${composerError ? "bad" : ""}`} role="status">
        {composerError || persistenceWarning || copyStatus}
      </div>}
      <div className="chat-composer">
        {(staged.length > 0 || selectedCapabilities.length > 0) && <div className="staged-files" aria-label="Items waiting to send">
          {selectedCapabilities.map((capability) => <span className="staged-file capability-staged" key={capability}>
            <Bot size={13} /><span>{capability}<small>Capability</small></span>
            <button type="button" aria-label={`Remove capability ${capability}`} onClick={() =>
              setSelectedCapabilities((current) => current.filter((item) => item !== capability))}><X size={13} /></button>
          </span>)}
          {staged.map((file) =>
            <span className="staged-file" key={file.id}><FileText size={13} /><span>{file.name}<small>{(file.size / 1024).toFixed(0)} KB</small></span>
              <button type="button" aria-label={`Remove ${file.name}`} onClick={() => setStaged((current) => current.filter((item) => item !== file))}><X size={13} /></button></span>)}
        </div>}
        <div className="composer-row">
          <div className="attachment-menu-wrap">
            <button type="button" className="composer-icon-button" aria-label="Add attachment" aria-expanded={menuOpen} onClick={() => setMenuOpen((open) => !open)}><Plus size={19} /></button>
            {menuOpen && <div className="attachment-menu" role="menu">
              <button type="button" role="menuitem" onClick={() => { setMenuOpen(false); fileInputRef.current?.click(); }}>Upload files <small>PDF, TXT, MD, CSV, JSON</small></button>
              <div className="capability-picker" role="group" aria-label="Add capabilities">
                <b>Add capabilities</b>
                {data.capabilities.length ? data.capabilities.map((capability) => {
                  const selected = selectedCapabilities.includes(capability.id);
                  return <button type="button" className="capability-picker-option" key={capability.id}
                    aria-pressed={selected} onClick={() => setSelectedCapabilities((current) =>
                      selected ? current.filter((item) => item !== capability.id) : [...current, capability.id])}>
                    <span>{capability.id}<small>{capability.description}</small></span>
                    <span className="capability-picker-action">{selected ? "Added" : "Add"}</span>
                  </button>;
                }) : <small className="capability-picker-empty">No capabilities are available.</small>}
              </div>
              <label className="additional-input-editor">
                <b>Workflow inputs (JSON)</b>
                <textarea value={runInputText} onChange={(event) => setRunInputText(event.target.value)} rows={5}
                  aria-label="Additional workflow inputs" placeholder='{"customer_id":"123"}' spellCheck={false} />
                <small>Extra values your saved workflow expects.</small>
              </label>
            </div>}
            <input ref={fileInputRef} type="file" className="visually-hidden" accept=".pdf,.txt,.md,.csv,.json" multiple
              aria-label="Choose attachments" onChange={(event) => {
                if (event.target.files) void addFiles(event.target.files);
                event.target.value = "";
              }} />
          </div>
          <div className="composer-form">
            <textarea className="goal-composer" aria-label="Workflow goal"
              placeholder="Describe what you want the selected workflow to do…"
              value={chatText} rows={2} onChange={(event) => setChatText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }} />
            <div className="composer-controls">
              <select aria-label="Saved workflow" value={mode} onChange={(event) => setMode(event.target.value)}>
                {!workflowChoices.length && <option value="">No saved workflows</option>}
                {workflowChoices.map((workflow) => <option value={`workflow:${workflow.id}`} key={workflow.id}>
                  {workflow.name || workflow.id}{workflow.published ? ` · v${workflow.version || "published"}` : ""}
                </option>)}
              </select>
              <button type="button" className="primary-button send-button" disabled={sending || !mode || !chatText.trim()}
                onClick={() => void sendMessage()}>
                {sending ? <LoaderCircle size={15} className="spin" /> : <Send size={15} />} Send
              </button>
            </div>
          </div>
        </div>
        <p className="composer-hint">Type a goal, choose a saved workflow, and press Enter or Send. Use + for files, capabilities, and extra workflow inputs.</p>
      </div>
    </section>
  );
}

function PlanSummary({ plan }: { plan: JsonRecord }) {
  const workflow = isRecord(plan.workflow) ? plan.workflow : {};
  const nodes = Array.isArray(workflow.nodes) ? workflow.nodes.filter(isRecord) : [];
  const rationale = typeof plan.rationale === "string" ? plan.rationale : "Plan generated from the available capabilities.";
  const routing = isRecord(plan.routing) && Array.isArray(plan.routing.selected)
    ? plan.routing.selected.filter(isRecord) : [];
  return <div className="chat-plan">
    <p>{rationale}</p>
    {nodes.length > 0 && <ol className="plan-step-list">{nodes.map((node, index) => {
      const id = typeof node.id === "string" ? node.id : `step-${index + 1}`;
      const capability = typeof node.capability === "string" ? node.capability : String(node.type || "workflow step");
      const route = routing.find((entry) => entry.node === id);
      const implementation = route && typeof route.implementation === "string" ? route.implementation : "Implementation selected at runtime";
      const dependencies = Array.isArray(node.depends_on) ? node.depends_on.filter((value): value is string => typeof value === "string") : [];
      return <li key={id}><div><b>{id}</b><code>{capability}</code><small>{implementation}</small>
        {dependencies.length > 0 && <small className="plan-dependencies">← depends on {dependencies.join(", ")}</small>}</div>
      </li>;
    })}</ol>}
  </div>;
}

function RunProgress({ run, onViewApprovals, onCancelRun, cancelling }: {
  run: RunDetail; onViewApprovals: () => void; onCancelRun: (id: string) => void; cancelling: boolean;
}) {
  const nodes = run.workflow.nodes.map((node) => {
    const result = run.nodes[node.id];
    const elapsed = result?.duration_ms ?? (result?.started_at
      ? Math.max(0, (result.finished_at ? new Date(result.finished_at).getTime() : Date.now()) -
        new Date(result.started_at).getTime()) : null);
    const failed = result?.state === "failed" || Boolean(result?.error);
    return <article className="progress-node" key={node.id}>
      <div className="progress-node-heading"><div><b>{node.id}</b><small>{node.capability || node.type || "workflow node"} · {result?.implementation || "not routed"}</small></div>
        <Status value={progressStatus(result?.state || "pending")} /></div>
      <div className="progress-node-meta">
        {result?.items && <span>{result.items.length} items · {result.items.filter((item) => progressStatus(item.state) === "completed").length} completed</span>}
        {elapsed !== null && <span>{Math.round(elapsed)} ms</span>}
        {failed && <span className="error-text">{result?.error || "Node failed"}</span>}
      </div>
      {(result?.output !== undefined || result?.items?.length) && <details className="progress-output">
        <summary>Node output</summary><pre className="json-output">{pretty({ output: result.output, items: result.items })}</pre>
      </details>}
    </article>;
  });
  const waiting = run.status === "waiting" || run.status === "waiting_input" || run.status === "waiting_approval";
  const active = ["pending", "running", "waiting"].includes(run.status);
  return <div className="run-progress">
    <div className="run-progress-heading"><b>Run progress</b><div className="run-progress-actions">
      <Status value={run.status} />
      {active && <button className="danger-button compact" onClick={() => onCancelRun(run.id)} disabled={cancelling}
        aria-label={`Stop run ${run.id}`}><Square size={13} /> {cancelling ? "Stopping…" : "Stop run"}</button>}
    </div></div>
    {nodes}
    {waiting && <div className="waiting-action"><span>Run is waiting for input or approval.</span>
    <button className="text-button" onClick={onViewApprovals}>View approvals <ArrowRight size={13} /></button>
    </div>}
  </div>;
}

function resultMarkdown(outputs: unknown): string | null {
  if (!isRecord(outputs)) return null;
  if (typeof outputs.report_markdown === "string") return outputs.report_markdown;
  function findMarkdown(value: unknown): string | null {
    if (Array.isArray(value)) {
      for (const item of value) {
        const result = findMarkdown(item);
        if (result !== null) return result;
      }
    } else if (isRecord(value)) {
      for (const [name, item] of Object.entries(value)) {
        if (name.endsWith("_markdown") && typeof item === "string") return item;
        const result = findMarkdown(item);
        if (result !== null) return result;
      }
    }
    return null;
  }
  return findMarkdown(outputs);
}

function resultText(outputs: unknown): string | null {
  if (!isRecord(outputs)) return null;
  for (const key of ["code_snippet", "result", "answer", "text"]) {
    if (typeof outputs[key] === "string") return outputs[key];
  }
  const textValues = Object.values(outputs).filter((value): value is string => typeof value === "string");
  return textValues.length === 1 ? textValues[0] : null;
}

function AssistantResult({ run, onCopy, onViewAudit }: { run: RunDetail; onCopy: () => void; onViewAudit: () => void }) {
  const outputs = isRecord(run.outputs) ? run.outputs : {};
  const markdown = resultMarkdown(run.outputs);
  const text = resultText(run.outputs);
  const codeSnippet = typeof outputs.code_snippet === "string" ? outputs.code_snippet : null;
  const highlights = Array.isArray(outputs.highlights) ? outputs.highlights : [];
  const actions = Array.isArray(outputs.action_items) ? outputs.action_items : [];
  const completed = run.status === "completed";
  const failedNode = run.workflow.nodes.find((node) => {
    const result = run.nodes[node.id];
    return result?.state === "failed" || Boolean(result?.error);
  });
  const durationMs = run.finished_at
    ? Math.max(0, new Date(run.finished_at).getTime() - new Date(run.created_at).getTime())
    : 0;
  const failed = CHAT_TERMINAL_FAILURES.has(run.status);
  const failureReason = failedNode
    ? `${failedNode.id}: ${run.nodes[failedNode.id]?.error || run.error || "Node failed."}`
    : run.error || "The run failed.";
  return <div className="assistant-result">
    {completed && <div className="result-body">
      {markdown ? <Markdown source={markdown} /> : codeSnippet !== null
        ? <>
          {typeof outputs.explanation === "string" && <section><h4>Explanation</h4><p>{outputs.explanation}</p></section>}
          <section><h4>Python code</h4><pre className="markdown-code"><code>{codeSnippet}</code></pre></section>
          {typeof outputs.example_usage === "string" && <section><h4>Example usage</h4><pre className="markdown-code"><code>{outputs.example_usage}</code></pre></section>}
        </>
        : text !== null ? <pre className="markdown-code"><code>{text}</code></pre>
          : <pre className="json-output">{pretty(run.outputs)}</pre>}
      {highlights.length > 0 && <section><h4>Highlights</h4><ul>{highlights.map((item, index) => <li key={index}>{String(item)}</li>)}</ul></section>}
      {actions.length > 0 && <section><h4>Action items</h4><ul>{actions.map((item, index) => <li key={index}>{String(item)}</li>)}</ul></section>}
    </div>}
    {failed && <div className="chat-error"><AlertCircle size={15} />{failureReason}</div>}
    {completed && <div className="result-footer"><span>Run <code>{run.id}</code> · {durationMs} ms</span>
      <button type="button" className="text-button" onClick={onViewAudit}>View in audit</button>
      <button type="button" className="secondary-button compact" onClick={onCopy} aria-label="Copy result"><Copy size={13} /> Copy result</button>
    </div>}
  </div>;
}

function Registry({ data, role, registerUrl, setRegisterUrl, onRegister, onApprove, busy,
  onPreviewAgent, onSaveAgent, onDeleteAgent, onGetAgent, onGenerateAgent, onRetryAgentRegistration,
}: {
  data: Data; role: string; registerUrl: string; setRegisterUrl: (value: string) => void;
  onRegister: (event: FormEvent<HTMLFormElement>) => void; onApprove: (key: string) => void; busy: boolean;
  onPreviewAgent: (definition: JsonRecord) => Promise<JsonRecord>;
  onSaveAgent: (definition: JsonRecord, editingId?: string) => Promise<JsonRecord>;
  onDeleteAgent: (agentId: string) => void;
  onGetAgent: (agentId: string) => Promise<JsonRecord>;
  onGenerateAgent: (name: string, description: string) => Promise<JsonRecord>;
  onRetryAgentRegistration: (agentId: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [registrationMode, setRegistrationMode] = useState<"describe" | "automatic" | "manual">("describe");
  const [editorOpen, setEditorOpen] = useState(false);
  const [preview, setPreview] = useState<JsonRecord | null>(null);
  const [detailAgent, setDetailAgent] = useState<JsonRecord | null>(null);
  const [detailError, setDetailError] = useState("");
  const [editingId, setEditingId] = useState<string | undefined>();
  const [editorBusy, setEditorBusy] = useState(false);
  const [editorError, setEditorError] = useState("");
  const [generationName, setGenerationName] = useState("");
  const [generationPrompt, setGenerationPrompt] = useState("");
  const [generationError, setGenerationError] = useState("");
  const [generationBusy, setGenerationBusy] = useState(false);
  const [agentName, setAgentName] = useState("");
  const [agentId, setAgentId] = useState("");
  const [agentDescription, setAgentDescription] = useState("");
  const [agentOwner, setAgentOwner] = useState("platform");
  const [capabilityId, setCapabilityId] = useState("");
  const [capabilityDescription, setCapabilityDescription] = useState("");
  const [useWhen, setUseWhen] = useState("");
  const [instructions, setInstructions] = useState("");
  const [timeoutSeconds, setTimeoutSeconds] = useState(90);
  const [inputSchema, setInputSchema] = useState(previewSchema);
  const [outputSchema, setOutputSchema] = useState(previewOutputSchema);
  const hostedAgents = data.agents.filter((agent) =>
    `${agent.id} ${agent.name} ${agent.capability} ${agent.template}`.toLowerCase().includes(query.toLowerCase()));
  const records = data.implementations.filter((item) =>
    `${item.key} ${item.name} ${item.owner} ${item.implements.join(" ")}`.toLowerCase().includes(query.toLowerCase()));

  useEffect(() => {
    setPreview(null);
  }, [agentName, agentId, agentDescription, agentOwner, capabilityId, capabilityDescription, useWhen, instructions, timeoutSeconds, inputSchema, outputSchema]);

  function resetEditor() {
    setEditingId(undefined);
    setAgentName("");
    setAgentId("");
    setAgentDescription("");
    setAgentOwner("platform");
    setCapabilityId("");
    setCapabilityDescription("");
    setUseWhen("");
    setInstructions("");
    setTimeoutSeconds(90);
    setInputSchema(previewSchema);
    setOutputSchema(previewOutputSchema);
    setPreview(null);
    setEditorError("");
    setEditorOpen(true);
  }

  async function editAgent(id: string) {
    setEditorBusy(true);
    setEditorError("");
    setDetailError("");
    try {
      const response = await onGetAgent(id);
      const def = response.definition as JsonRecord;
      if (def.template !== "instruction") {
        setDetailError("This agent uses the extractor template and cannot be edited in the instruction-agent form.");
        return;
      }
      const capability = def.capability as JsonRecord;
      setEditingId(id);
      setAgentId(String(def.id || id));
      setAgentName(String(def.name || ""));
      setAgentDescription(String(def.description || ""));
      setAgentOwner(String(def.owner || "platform"));
      setCapabilityId(String(capability.id || ""));
      setCapabilityDescription(String(capability.description || ""));
      setUseWhen(String(capability.use_when || ""));
      setInstructions(String(def.instructions || ""));
      setTimeoutSeconds(Number(def.timeout_s) || 90);
      setInputSchema(pretty(def.input_schema));
      setOutputSchema(pretty(def.output_schema));
      setPreview(null);
      setEditorOpen(true);
    } catch (error) {
      setEditorError(error instanceof Error ? error.message : String(error));
    } finally {
      setEditorBusy(false);
    }
  }

  function buildDefinition(): JsonRecord {
    const parsedInput: unknown = JSON.parse(inputSchema);
    const parsedOutput: unknown = JSON.parse(outputSchema);
    if (typeof parsedInput !== "object" || parsedInput === null || Array.isArray(parsedInput)) {
      throw new Error("Input schema must be a JSON Schema object.");
    }
    if (typeof parsedOutput !== "object" || parsedOutput === null || Array.isArray(parsedOutput)) {
      throw new Error("Output schema must be a JSON Schema object.");
    }
    return {
      id: agentId.trim(),
      name: agentName.trim(),
      description: agentDescription.trim(),
      owner: agentOwner.trim(),
      version: editingId ? incrementPatchVersion(data.agents.find((agent) => agent.id === editingId)?.version) : "1.0.0",
      template: "instruction",
      capability: {
        id: capabilityId.trim(),
        description: capabilityDescription.trim(),
        use_when: useWhen.trim(),
        do_not_use_when: "",
        examples: [],
      },
      instructions: instructions.trim(),
      fields: [],
      input_schema: parsedInput,
      output_schema: parsedOutput,
      timeout_s: timeoutSeconds,
      approve: true,
      languages: ["en"],
    };
  }

  async function previewDefinition(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setEditorBusy(true);
    setEditorError("");
    try {
      setPreview(await onPreviewAgent(buildDefinition()));
    } catch (error) {
      setPreview(null);
      setEditorError(error instanceof Error ? error.message : String(error));
    } finally {
      setEditorBusy(false);
    }
  }

  async function generateDefinition(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setGenerationBusy(true);
    setGenerationError("");
    try {
      const generated = await onGenerateAgent(generationName, generationPrompt);
      const capability = generated.capability as JsonRecord;
      setEditingId(undefined);
      setAgentId(String(generated.id || ""));
      setAgentName(String(generated.name || generationName));
      setAgentDescription(String(generated.description || generationPrompt));
      setAgentOwner(String(generated.owner || "platform"));
      setCapabilityId(String(capability.id || ""));
      setCapabilityDescription(String(capability.description || ""));
      setUseWhen(String(capability.use_when || ""));
      setInstructions(String(generated.instructions || ""));
      setTimeoutSeconds(Number(generated.timeout_s) || 90);
      setInputSchema(pretty(generated.input_schema));
      setOutputSchema(pretty(generated.output_schema));
      setPreview(null);
      setEditorError("");
      setEditorOpen(true);
      setRegistrationMode("manual");
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : String(error));
    } finally {
      setGenerationBusy(false);
    }
  }

  async function submitDefinition() {
    setEditorBusy(true);
    setEditorError("");
    try {
      await onSaveAgent(buildDefinition(), editingId);
      setEditorOpen(false);
      setPreview(null);
    } catch (error) {
      setEditorError(error instanceof Error ? error.message : String(error));
    } finally {
      setEditorBusy(false);
    }
  }

  return (
    <div className="stack">
      <section className="panel"><div className="panel-heading"><div><h2>Registered implementations</h2><p>Registry state, conformance, and approved capability bindings</p></div>
        <span className="subtle-count">{data.implementations.length} total</span></div>
        <label className="search-field"><CircleHelp size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter by name, owner, or capability" /></label>
        <h3 className="section-title">Hosted agents</h3>
        {hostedAgents.length ? <div className="implementation-list">{hostedAgents.map((agent) => <article className="implementation-card" key={agent.id}>
          <div className="implementation-top"><div><h3>{agent.name}</h3><code>{agent.id} · v{agent.version}</code></div>
            <Status value={agent.implementation?.status || "not registered"} /></div>
          <div className="implementation-meta"><span>Capability: {agent.capability}</span><span>Template: {agent.template}</span>
            {agent.fields.length > 0 && <span>Fields: {agent.fields.join(", ")}</span>}</div>
          <div className="form-actions">
            {role === "admin" && agent.implementation?.status !== "approved" && <button className="secondary-button compact" onClick={() => onRetryAgentRegistration(agent.id)}>
              <Activity size={14} /> Retry registration
            </button>}
            <button className="text-button" onClick={() => void (async () => {
              setDetailError("");
              try {
                setDetailAgent(await onGetAgent(agent.id));
              } catch (error) {
                setDetailError(error instanceof Error ? error.message : String(error));
              }
            })()}>View details</button>
            {role === "admin" && agent.template === "instruction" && <button className="secondary-button compact" onClick={() => void editAgent(agent.id)} disabled={editorBusy}>Edit</button>}
            {role === "admin" && <button className="danger-button compact" onClick={() => {
              if (window.confirm(`Delete hosted agent "${agent.name}"? Its registered implementations will be revoked.`)) onDeleteAgent(agent.id);
            }}>Delete</button>}
          </div>
        </article>)}</div> : <EmptyState text="No hosted agents have been created." />}
        {detailError && <div className="error-box">{detailError}</div>}
        {detailAgent && <div className="preview-result"><div className="panel-heading"><div><b>{String((detailAgent.definition as JsonRecord | undefined)?.name || "Agent details")}</b>
          <span>Definition, generated contract, card, and current registry status</span></div>
          <button className="icon-button" aria-label="Close agent details" onClick={() => setDetailAgent(null)}>×</button></div>
          <pre className="json-output">{pretty(detailAgent)}</pre></div>}
        <h3 className="section-title">Registered implementations</h3>
        {records.length ? <div className="implementation-list">{records.map((item) => <article className="implementation-card" key={item.key}>
          <div className="implementation-top"><div><h3>{item.name}</h3><code>{item.key}</code></div><Status value={item.status} /></div>
          <div className="implementation-meta"><span>{item.kind} · {item.binding}</span><span>Owner: {item.owner}</span><span>Priority: {item.priority}</span></div>
          <div className="tag-row">{item.implements.map((cap) => <span className="tag" key={cap}>{cap}</span>)}</div>
          <details className="check-details"><summary>{item.checks.filter((check) => check.passed).length}/{item.checks.length} conformance checks passed</summary>
            <ul>{item.checks.map((check) => <li key={check.check}><span className={check.passed ? "check-ok" : "check-fail"}>{check.passed ? "Passed" : "Failed"}</span> {check.check} <small>{check.detail}</small></li>)}</ul>
          </details>
          {role === "admin" && item.status === "draft" && <button className="secondary-button compact" onClick={() => onApprove(item.key)}><Check size={14} /> Approve</button>}
        </article>)}</div> : <EmptyState text="No registry implementations match this filter." />}
      </section>
      <section className="panel"><div className="panel-heading"><div><h2>Register an agent</h2><p>Use a published A2A card or create a model-backed hosted agent.</p></div></div>
        <div className="tab-row" role="tablist" aria-label="Agent registration method">
          <button type="button" className={`tab-button ${registrationMode === "describe" ? "selected" : ""}`} onClick={() => setRegistrationMode("describe")}><Sparkles size={14} /> Describe with AI</button>
          <button type="button" className={`tab-button ${registrationMode === "automatic" ? "selected" : ""}`} onClick={() => setRegistrationMode("automatic")}>Automatic · A2A discovery</button>
          <button type="button" className={`tab-button ${registrationMode === "manual" ? "selected" : ""}`} onClick={() => setRegistrationMode("manual")}>Manual · Hosted agent</button>
        </div>
        {registrationMode === "describe" ? <form className="register-form agent-generate-form" onSubmit={(event) => void generateDefinition(event)}>
          <p className="muted registration-help">Name the agent and describe its role, goals, and behavior. The configured server-side LLM generates a draft configuration for review; nothing is saved until you confirm.</p>
          {role !== "admin" && <div className="error-box">Switch to the admin role to generate or register agents.</div>}
          <label className="field-label" htmlFor="agent-generation-name">Agent name</label>
          <input id="agent-generation-name" value={generationName} onChange={(event) => setGenerationName(event.target.value)} required maxLength={80} placeholder="e.g. Sales Research Assistant" />
          <label className="field-label" htmlFor="agent-generation-prompt">Describe the agent</label>
          <textarea id="agent-generation-prompt" className="goal-input" value={generationPrompt} onChange={(event) => setGenerationPrompt(event.target.value)} required maxLength={2000}
            placeholder="Describe its role, goals, expected inputs, output, and behavior…" />
          {generationError && <div className="error-box">{generationError}</div>}
          <div className="form-actions"><span className="muted">Uses the LLM configured on the server. API keys stay server-side.</span>
            <button type="submit" className="primary-button" disabled={generationBusy || role !== "admin"}>{generationBusy ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />} Generate agent draft</button>
          </div>
        </form> : registrationMode === "automatic" ? <form className="register-form" onSubmit={(event) => void onRegister(event)}>
          <p className="muted registration-help">Fetches <code>/.well-known/agent-card.json</code> (or a provided JSON card URL), validates its UEOA metadata and conformance, then registers it. The API allow-list and SSRF checks apply.</p>
          <label className="field-label" htmlFor="agent-url">Agent endpoint or card URL</label>
          <div className="inline-form"><input id="agent-url" type="url" required value={registerUrl} onChange={(event) => setRegisterUrl(event.target.value)} placeholder="https://agent.example.com" />
            <button type="submit" className="primary-button" disabled={busy || role !== "admin"}>{busy ? <LoaderCircle size={15} className="spin" /> : <Bot size={15} />} Discover &amp; register</button></div>
          <p className="muted">Admin role required. Approval remains a separate action when the card is registered as a draft.</p>
        </form> : <div className="register-form">
          <p className="muted registration-help">Creates a UEOA-hosted instruction agent, derives its capability contract from your JSON Schemas, previews the generated card, and saves only when you confirm. Hosted agents require an LLM configured on the server.</p>
          {role !== "admin" && <div className="error-box">Switch to the admin role to create, edit, or delete agents.</div>}
          {editorOpen ? <form className="agent-editor" onSubmit={(event) => void previewDefinition(event)}>
            <div className="editor-grid">
              <Field label="Agent name" value={agentName} onChange={setAgentName} required />
              <Field label="Agent ID (lowercase, hyphens)" value={agentId} onChange={setAgentId} required disabled={Boolean(editingId)} />
              <Field label="Owner / team" value={agentOwner} onChange={setAgentOwner} required />
              <Field label="Capability ID (for example text.summarize)" value={capabilityId} onChange={setCapabilityId} required disabled={Boolean(editingId)} />
              <Field label="Timeout (seconds, 10–300)" value={String(timeoutSeconds)} onChange={(value) => setTimeoutSeconds(Number(value))} type="number" min="10" max="300" required />
              <Field label="Capability description" value={capabilityDescription} onChange={setCapabilityDescription} required />
              <label className="editor-field wide"><span>Agent description</span><textarea value={agentDescription} onChange={(event) => setAgentDescription(event.target.value)} required maxLength={500} /></label>
              <label className="editor-field wide"><span>Use this capability when</span><textarea value={useWhen} onChange={(event) => setUseWhen(event.target.value)} required maxLength={1000} /></label>
              <label className="editor-field wide"><span>Instructions / system prompt</span><textarea value={instructions} onChange={(event) => setInstructions(event.target.value)} required maxLength={4000} /></label>
              <label className="editor-field"><span>Input JSON Schema</span><textarea className="code-input small-code" value={inputSchema} onChange={(event) => setInputSchema(event.target.value)} required spellCheck={false} readOnly={Boolean(editingId)} /></label>
              <label className="editor-field"><span>Output JSON Schema</span><textarea className="code-input small-code" value={outputSchema} onChange={(event) => setOutputSchema(event.target.value)} required spellCheck={false} readOnly={Boolean(editingId)} /></label>
            </div>
            {editorError && <div className="error-box">{editorError}</div>}
            <div className="form-actions"><button type="button" className="secondary-button" onClick={() => { setEditorOpen(false); setPreview(null); }}>Cancel</button>
              <button type="submit" className="secondary-button" disabled={editorBusy || role !== "admin"}>{editorBusy ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />} Preview contract</button>
              {preview && <button type="button" className="primary-button" onClick={() => void submitDefinition()} disabled={editorBusy || role !== "admin"}><Check size={15} /> {editingId ? "Update agent" : "Create agent"}</button>}
            </div>
            {preview && <div className="preview-result"><b>Contract preview ready</b><span>Review the generated capability contract and A2A card below. Save is an explicit action.</span>
              <details><summary>Generated capability contract and card</summary><pre className="json-output">{pretty({ contract: preview.contract, card: preview.card })}</pre></details>
            </div>}
          </form> : <button className="primary-button" onClick={resetEditor} disabled={role !== "admin"}><Bot size={15} /> Create hosted agent</button>}
        </div>}
      </section>
    </div>
  );
}

const previewSchema = pretty({
  type: "object",
  additionalProperties: false,
  required: ["text"],
  properties: { text: { type: "string", default: "sample input" } },
});
const previewOutputSchema = pretty({
  type: "object",
  additionalProperties: false,
  required: ["result"],
  properties: { result: { type: "string" } },
});

function incrementPatchVersion(version: string | undefined): string {
  const parts = (version || "1.0.0").split(".").map(Number);
  if (parts.length !== 3 || parts.some((part) => !Number.isInteger(part))) return "1.0.1";
  return `${parts[0]}.${parts[1]}.${parts[2] + 1}`;
}

function Field({ label, value, onChange, required, type = "text", pattern, min, max, disabled }: {
  label: string; value: string; onChange: (value: string) => void; required?: boolean;
  type?: string; pattern?: string; min?: string; max?: string; disabled?: boolean;
}) {
  return <label className="editor-field"><span>{label}</span>
    <input type={type} value={value} onChange={(event) => onChange(event.target.value)} required={required}
      pattern={pattern} min={min} max={max} disabled={disabled} /></label>;
}

function Skills({ data, role, onSave, onDelete }: {
  data: Data;
  role: string;
  onSave: (skill: Pick<Skill, "name" | "description" | "instructions" | "tags">, editingId?: string) => Promise<void>;
  onDelete: (skillId: string) => void;
}) {
  const [editingId, setEditingId] = useState<string | undefined>();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [instructions, setInstructions] = useState("");
  const [tagsText, setTagsText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const editableSkills = data.skills.filter((skill) => !skill.builtin);

  function reset() {
    setEditingId(undefined);
    setName("");
    setDescription("");
    setInstructions("");
    setTagsText("");
    setError("");
  }

  function edit(skill: Skill) {
    setEditingId(skill.id);
    setName(skill.name);
    setDescription(skill.description);
    setInstructions(skill.instructions);
    setTagsText(skill.tags.join(", "));
    setError("");
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await onSave({
        name: name.trim(),
        description: description.trim(),
        instructions: instructions.trim(),
        tags: tagsText.split(",").map((tag) => tag.trim()).filter(Boolean),
      }, editingId);
      reset();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <section className="panel">
        <div className="panel-heading"><div><h2>Default skills</h2><p>Reusable instruction profiles included with the platform.</p></div>
          <span className="subtle-count">{data.skills.filter((skill) => skill.builtin).length} defaults</span></div>
        <div className="skill-list">{data.skills.filter((skill) => skill.builtin).map((skill) =>
          <article className="skill-card" key={skill.id}>
            <div className="implementation-top"><div><h3>{skill.name}</h3><code>{skill.id}</code></div><Status value="default" /></div>
            <p>{skill.description}</p><div className="tag-row">{skill.tags.map((tag) => <span className="tag" key={tag}>{tag}</span>)}</div>
          </article>)}</div>
      </section>
      <section className="panel">
        <div className="panel-heading"><div><h2>Custom skills</h2><p>Store reusable instructions and tags for future agent and workflow bindings.</p></div>
          <span className="subtle-count">{editableSkills.length} custom</span></div>
        {editableSkills.length ? <div className="skill-list">{editableSkills.map((skill) =>
          <article className="skill-card" key={skill.id}>
            <div className="implementation-top"><div><h3>{skill.name}</h3><code>{skill.id}</code></div><Status value="custom" /></div>
            <p>{skill.description}</p><details className="check-details"><summary>Instructions</summary><p>{skill.instructions}</p></details>
            <div className="tag-row">{skill.tags.map((tag) => <span className="tag" key={tag}>{tag}</span>)}</div>
            {role === "admin" && <div className="form-actions"><button className="secondary-button compact" onClick={() => edit(skill)}>Edit</button>
              <button className="danger-button compact" onClick={() => {
                if (window.confirm(`Delete skill "${skill.name}"?`)) onDelete(skill.id);
              }}>Delete</button></div>}
          </article>)}</div> : <EmptyState text="No custom skills have been registered." />}
      </section>
      <section className="panel">
        <div className="panel-heading"><div><h2>{editingId ? "Edit skill" : "Register a custom skill"}</h2><p>Custom skill changes are stored through the platform API.</p></div></div>
        {role === "admin" ? <form className="skill-form" onSubmit={(event) => void submit(event)}>
          <label className="editor-field"><span>Name</span><input value={name} onChange={(event) => setName(event.target.value)} required maxLength={120} /></label>
          <label className="editor-field"><span>Description</span><textarea value={description} onChange={(event) => setDescription(event.target.value)} required maxLength={2000} /></label>
          <label className="editor-field"><span>Reusable instructions</span><textarea value={instructions} onChange={(event) => setInstructions(event.target.value)} required maxLength={10000} /></label>
          <label className="editor-field"><span>Tags (comma-separated)</span><input value={tagsText} onChange={(event) => setTagsText(event.target.value)} /></label>
          {error && <div className="error-box">{error}</div>}
          <div className="form-actions">{editingId && <button type="button" className="secondary-button" onClick={reset}>Cancel edit</button>}
            <button type="submit" className="primary-button" disabled={busy}>{busy ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />}
              {editingId ? "Save skill" : "Register skill"}</button></div>
        </form> : <EmptyState text="Switch to the admin role to create or edit custom skills." />}
      </section>
    </div>
  );
}

function MCP({ servers, role, busy, onRegister, onDelete }: {
  servers: McpServer[];
  role: string;
  busy: boolean;
  onRegister: (server: Pick<McpServer, "name" | "url" | "description">) => Promise<void>;
  onDelete: (server: McpServer) => void;
}) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [description, setDescription] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    try {
      await onRegister({ name: name.trim(), url: url.trim(), description: description.trim() });
      setName("");
      setUrl("");
      setDescription("");
    } catch {
      // The parent displays the API error and the entered values stay available for correction.
    }
  }

  return (
    <div className="stack">
      <section className="panel">
        <div className="panel-heading"><div><h2>Registered MCP servers</h2>
          <p>Streamable HTTP endpoints saved in this UEOA instance.</p></div>
          <span className="subtle-count">{servers.length} registered</span></div>
        {servers.length ? <div className="implementation-list">{servers.map((server) =>
          <article className="implementation-card" key={server.id}>
            <div className="implementation-top"><div><h3>{server.name}</h3><code>{server.id}</code></div>
              <Status value="registered" /></div>
            <div className="mcp-server-url"><span>Endpoint</span><code>{server.url}</code></div>
            {server.description && <p className="mcp-description">{server.description}</p>}
            <div className="implementation-meta"><span>Transport: {server.transport}</span>
              <span>Registered by: {server.owner}</span><span>Added: {formatTime(server.registered_at)}</span></div>
            {role === "admin" && <div className="form-actions"><button className="danger-button compact" onClick={() => {
              if (window.confirm(`Remove MCP server "${server.name}" from the catalog?`)) onDelete(server);
            }}>Remove</button></div>}
          </article>)}</div> : <EmptyState text="No MCP servers have been registered yet." />}
      </section>
      <section className="panel">
        <div className="panel-heading"><div><h2>Register an MCP server</h2>
          <p>Add the server name and its Streamable HTTP MCP endpoint URL.</p></div></div>
        {role === "admin" ? <form className="mcp-form" onSubmit={(event) => void submit(event)}>
          <label className="editor-field"><span>Server name</span>
            <input value={name} onChange={(event) => setName(event.target.value)} required maxLength={120} placeholder="e.g. Internal tools" /></label>
          <label className="editor-field"><span>MCP endpoint URL</span>
            <input type="url" value={url} onChange={(event) => setUrl(event.target.value)} required placeholder="https://mcp.example.com/mcp" /></label>
          <label className="editor-field"><span>Description (optional)</span>
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={2000} placeholder="What this server provides" /></label>
          <p className="muted">This saves the endpoint to the catalog. Do not include credentials in the URL.</p>
          <div className="form-actions"><button type="submit" className="primary-button" disabled={busy}>
            {busy ? <LoaderCircle size={15} className="spin" /> : <Server size={15} />} Register MCP server</button></div>
        </form> : <EmptyState text="Switch to the admin role to register or remove MCP servers." />}
      </section>
    </div>
  );
}

function PromptBuilder({ onGenerate, onUsePrompt }: {
  onGenerate: (brief: PromptBrief) => Promise<string>;
  onUsePrompt: (prompt: string) => void;
}) {
  const [brief, setBrief] = useState<PromptBrief>({
    description: "",
    audience: "",
    inputs: "",
    steps: "",
    constraints: "",
    output: "",
  });
  const [prompt, setPrompt] = useState("");
  const [working, setWorking] = useState(false);
  const [notice, setNotice] = useState("");
  const [copied, setCopied] = useState(false);
  const update = (key: keyof PromptBrief, value: string) => setBrief((current) => ({ ...current, [key]: value }));

  async function generate() {
    if (brief.description.trim().length < 10) {
      setNotice("Describe the work in at least 10 characters so the prompt can be specific.");
      return;
    }
    setWorking(true);
    setNotice("");
    setCopied(false);
    try {
      setPrompt(await onGenerate(brief));
      setNotice("Professional prompt generated. Review it, then use it to create a workflow.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setWorking(false);
    }
  }

  async function copyPrompt() {
    try {
      await navigator.clipboard.writeText(prompt);
      setCopied(true);
      setNotice("Prompt copied.");
    } catch {
      setNotice("Clipboard access was denied. Select and copy the prompt text manually.");
    }
  }

  return (
    <div className="prompt-builder">
      <section className="panel prompt-builder-intro">
        <p className="eyebrow">CREATE / PROMPT BUILDER</p>
        <h2>Turn your idea into a professional workflow brief</h2>
        <p>Describe the task and requirements. Azure OpenAI will organize them into a clear prompt you can send to the workflow generator.</p>
      </section>
      <div className="prompt-builder-grid">
        <section className="panel">
          <div className="panel-heading"><div><h2>Describe your work</h2><p>Specific requirements help create a better workflow.</p></div></div>
          <label className="field-label" htmlFor="prompt-purpose">What should the workflow accomplish? *</label>
          <textarea id="prompt-purpose" className="goal-input" value={brief.description}
            onChange={(event) => update("description", event.target.value)}
            placeholder="Describe the job, desired result, and how success should be measured." />
          <label className="field-label" htmlFor="prompt-audience">Who or what is it for?</label>
          <textarea id="prompt-audience" className="prompt-input" value={brief.audience}
            onChange={(event) => update("audience", event.target.value)}
            placeholder="Users, customers, teams, systems, or audience." />
          <label className="field-label" htmlFor="prompt-inputs">Inputs and references</label>
          <textarea id="prompt-inputs" className="prompt-input" value={brief.inputs}
            onChange={(event) => update("inputs", event.target.value)}
            placeholder="Files, fields, records, or context the workflow receives." />
          <label className="field-label" htmlFor="prompt-steps">Process or key steps</label>
          <textarea id="prompt-steps" className="prompt-input" value={brief.steps}
            onChange={(event) => update("steps", event.target.value)}
            placeholder="Checks, decisions, analysis, or ordering requirements." />
          <label className="field-label" htmlFor="prompt-constraints">Rules and constraints</label>
          <textarea id="prompt-constraints" className="prompt-input" value={brief.constraints}
            onChange={(event) => update("constraints", event.target.value)}
            placeholder="Permissions, policies, exclusions, or safety requirements." />
          <label className="field-label" htmlFor="prompt-output">Expected output</label>
          <textarea id="prompt-output" className="prompt-input" value={brief.output}
            onChange={(event) => update("output", event.target.value)}
            placeholder="Format, fields, tone, and level of detail." />
          <div className="form-actions">
            <button type="button" className="primary-button" disabled={working || brief.description.trim().length < 10} onClick={() => void generate()}>
              {working ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
              {working ? "Creating prompt…" : "Create professional prompt"}
            </button>
          </div>
          {notice && <p className={`prompt-notice ${notice.includes("denied") ? "bad" : ""}`} role="status">{notice}</p>}
        </section>
        <section className="panel prompt-result-panel">
          <div className="panel-heading"><div><h2>Workflow prompt</h2><p>Review or edit the prompt before sending it to the workflow studio.</p></div></div>
          <textarea className="prompt-result" aria-label="Generated professional prompt" value={prompt}
            onChange={(event) => setPrompt(event.target.value)} placeholder="Your professional prompt will appear here." />
          <div className="form-actions">
            <button type="button" className="secondary-button" disabled={!prompt.trim()} onClick={() => void copyPrompt()}>
              <Copy size={14} /> {copied ? "Copied" : "Copy prompt"}
            </button>
            <button type="button" className="primary-button" disabled={!prompt.trim()} onClick={() => onUsePrompt(prompt.trim())}>
              <GitBranch size={14} /> Use prompt in workflow
            </button>
          </div>
          <div className="prompt-tip"><CircleHelp size={15} /><span>The workflow generator can only use capabilities that are registered and approved in this project. It will show which capabilities are missing instead of inventing steps.</span></div>
        </section>
      </div>
    </div>
  );
}

function SavedWorkflows({ workflows, onEdit, onDelete, onCreate }: {
  workflows: WorkflowSummary[];
  onEdit: (id: string) => void;
  onDelete: (id: string) => void;
  onCreate: () => void;
}) {
  const saved = workflows.filter((workflow) => !workflow.builtin && !workflow.published);
  const published = workflows.filter((workflow) => workflow.published);
  return (
    <section className="panel saved-workflows">
      <div className="panel-heading">
        <div><h2>Saved workflows</h2><p>Continue editing drafts you saved, or create a new workflow.</p></div>
        <button className="primary-button compact" onClick={onCreate}><Plus size={14} /> Create workflow</button>
      </div>
      {!saved.length
        ? <div className="empty-state"><GitBranch size={22} /><b>No saved workflows yet</b><span>Create a workflow, then choose Save draft or Save workflow.</span></div>
        : <div className="saved-workflow-list">{saved.map((workflow) => (
          <article className="saved-workflow-card" key={workflow.id}>
            <div className="saved-workflow-info">
              <div><GitBranch size={16} /><h3>{workflow.name || workflow.id}</h3>
                <span className={`status ${workflow.valid ? "good" : "warn"}`}>{workflow.valid ? "Ready to test" : "Draft · needs work"}</span>
              </div>
              <p>{workflow.description || "No description provided."}</p>
              <small>{workflow.nodes ?? 0} steps · Updated {formatTime(workflow.updated_at)} · {workflow.id}{workflow.version ? `@${workflow.version}` : ""}</small>
            </div>
            <div className="saved-workflow-actions">
              <button className="secondary-button compact" onClick={() => onEdit(workflow.id)}><GitBranch size={14} /> Continue editing</button>
              <button className="icon-button danger-icon" aria-label={`Delete ${workflow.name || workflow.id}`} title="Delete workflow" onClick={() => {
                if (window.confirm(`Delete workflow "${workflow.name || workflow.id}"? This cannot be undone.`)) onDelete(workflow.id);
              }}><Trash2 size={15} /></button>
            </div>
          </article>
        ))}</div>}
      {published.length > 0 && <div className="published-workflow-info">
        <b>Published versions ({published.length})</b>
        <span>Published snapshots are immutable and remain runnable after their editable draft is deleted.</span>
        <ul>{published.map((workflow) => <li key={workflow.id}>
          {workflow.name || workflow.id} · {workflow.version_ref || `v${workflow.version || "published"}`}
        </li>)}</ul>
      </div>}
    </section>
  );
}

function Workflows({ data, inputsText, setInputsText, draftGoal, setDraftGoal, draft, workflowName, setWorkflowName, busy, onDraft, onSave, onRun }: {
  data: Data; inputsText: string; setInputsText: (value: string) => void; draftGoal: string; setDraftGoal: (value: string) => void; draft: JsonRecord | null;
  workflowName: string; setWorkflowName: (value: string) => void; busy: boolean;
  onDraft: () => void; onSave: () => void; onRun: (id: string) => void;
}) {
  const hasDraftWorkflow = typeof draft?.workflow === "object" && draft.workflow !== null;
  return (
    <div className="stack">
      <section className="panel"><div className="panel-heading"><div><h2>Workflow templates</h2><p>Saved templates reported by the workflow API</p></div><span className="subtle-count">{data.workflows.length} saved</span></div>
        {data.workflows.length ? <div className="workflow-list">{data.workflows.map((workflow) => <article className="workflow-card" key={workflow.id}>
          <div className="workflow-card-main"><div className="workflow-icon"><GitBranch size={17} /></div><div><h3>{workflow.name || workflow.id}</h3><code>{workflow.id}</code>
            <p>{workflow.description || "No description provided."}</p><div className="tag-row"><span className="tag">{workflow.nodes} nodes</span>
              <span className="tag">{(workflow.capabilities || []).length} capabilities</span><Status value={workflow.valid ? "valid" : "invalid"} /></div></div></div>
          <button className="secondary-button compact" onClick={() => onRun(workflow.id)}><Play size={14} /> Run</button>
        </article>)}</div> : <EmptyState text="No workflow templates have been saved." /> }
        <label className="field-label" htmlFor="workflow-inputs">Inputs for workflow run</label>
        <textarea id="workflow-inputs" className="code-input small-code" value={inputsText} onChange={(event) => setInputsText(event.target.value)} spellCheck={false} />
      </section>
      <section className="panel"><div className="panel-heading"><div><h2>Draft a workflow</h2><p>Uses the configured planner, then validates and saves against the live registry.</p></div>
        <span className={`status ${data.health?.planner ? "good" : "warn"}`}>{data.health?.planner ? "Planner configured" : "Planner unavailable"}</span></div>
        <label className="field-label" htmlFor="draft-goal">Workflow goal</label>
        <textarea id="draft-goal" className="goal-input" value={draftGoal} onChange={(event) => setDraftGoal(event.target.value)} placeholder="Describe a repeatable workflow…" />
        <div className="form-actions"><span className="muted">Planner response is validated before it can be saved.</span>
          <button className="primary-button" disabled={busy || !data.health?.planner} onClick={onDraft}>{busy ? <LoaderCircle size={15} className="spin" /> : <GitBranch size={15} />} Draft workflow</button>
        </div>
        {draft && <div className="draft-result"><div className="panel-heading"><div><h3>Draft result</h3><p>{String(draft.rationale || "")}</p></div><Status value={draft.ok ? "valid" : "needs review"} /></div>
          {hasDraftWorkflow && <><label className="field-label" htmlFor="workflow-id">Template ID</label>
            <div className="inline-form"><input id="workflow-id" value={workflowName} onChange={(event) => setWorkflowName(event.target.value)} />
              <button className="secondary-button" disabled={busy || !draft.ok} onClick={onSave}><Check size={15} /> Validate &amp; save</button></div>
            <pre className="json-output">{pretty(draft.workflow)}</pre></>}
          {Array.isArray(draft.issues) && draft.issues.length > 0 && <pre className="json-output issue-output">{pretty(draft.issues)}</pre>}
        </div>}
      </section>
    </div>
  );
}

function Runs({ data, selectedRun, onSelect, onCancelRun, cancellingRunId }: {
  data: Data; selectedRun: RunDetail | null; onSelect: (id: string) => void;
  onCancelRun: (id: string) => void; cancellingRunId: string | null;
}) {
  const active = selectedRun && ["pending", "running", "waiting"].includes(selectedRun.status);
  return (
    <div className="two-column runs-layout">
      <section className="panel"><div className="panel-heading"><div><h2>Run history</h2><p>Runs visible to your selected role</p></div><span className="subtle-count">{data.runs.length} total</span></div>
        <RunList runs={data.runs} onSelect={onSelect} selectedId={selectedRun?.id} />
      </section>
      <section className="panel run-detail"><div className="panel-heading"><div><h2>Run details</h2><p>{selectedRun ? selectedRun.id : "Select a run to inspect its live state"}</p></div>
        {selectedRun && <div className="run-progress-actions"><Status value={selectedRun.status} />
          {active && <button className="danger-button compact" onClick={() => onCancelRun(selectedRun.id)}
            disabled={cancellingRunId === selectedRun.id} aria-label={`Stop run ${selectedRun.id}`}>
            <Square size={13} /> {cancellingRunId === selectedRun.id ? "Stopping…" : "Stop run"}
          </button>}</div>}</div>
        {selectedRun ? <><div className="run-goal">{selectedRun.goal}</div>
          <div className="timeline">{selectedRun.workflow.nodes.map((node) => {
            const result = selectedRun.nodes[node.id];
            return <div className="timeline-item" key={node.id}><span className={`timeline-marker ${statusTone(result?.state || "pending")}`} />
              <div className="timeline-content"><b>{node.id}</b><small>{node.capability || node.type || "workflow node"} · {result?.implementation || "not routed"}</small>
                <Status value={result?.state || "pending"} />{result?.error && <p className="error-text">{result.error}</p>}</div></div>;
          })}</div>
          {selectedRun.outputs !== null && selectedRun.outputs !== undefined && <><h3 className="section-title">Outputs</h3><pre className="json-output">{pretty(selectedRun.outputs)}</pre></>}
          {selectedRun.error && <div className="error-box"><AlertCircle size={16} />{selectedRun.error}</div>}
          <div className="run-timestamp"><Clock3 size={14} /> Started {formatTime(selectedRun.created_at)}</div>
        </> : <EmptyState text="No run selected." />}
      </section>
    </div>
  );
}

function Approvals({ data, onDecide }: { data: Data; onDecide: (approval: Approval, decision: "approve" | "reject") => void }) {
  return (
    <section className="panel"><div className="panel-heading"><div><h2>Pending approvals</h2><p>Review the exact payload and its hash before making a decision.</p></div>
      <span className="subtle-count">{data.approvals.length} pending</span></div>
      {data.approvals.length ? <div className="approval-list">{data.approvals.map((approval) => <article className="approval-card" key={`${approval.run_id}-${approval.node}-${approval.item}`}>
        <div className="approval-head"><div><h3>{approval.capability}</h3><p>{approval.goal}</p></div><Status value="pending approval" /></div>
        <div className="approval-meta"><span>Run <code>{approval.run_id}</code></span><span>Node <code>{approval.node}</code></span>
          {approval.item !== null && <span>Item {approval.item}</span>}<span>Requested by {approval.requested_by}</span></div>
        <pre className="json-output">{pretty(approval.payload)}</pre>
        <div className="payload-hash">Payload SHA-256 <code>{approval.payload_hash}</code></div>
        <div className="form-actions"><span className="muted">Approvals are bound to this payload hash.</span>
          <button className="danger-button" onClick={() => onDecide(approval, "reject")}>Reject</button>
          <button className="primary-button" onClick={() => onDecide(approval, "approve")}><Check size={15} /> Approve</button></div>
      </article>)}</div> : <EmptyState text="There are no pending approvals for this role." />}
    </section>
  );
}

function Audit({ data }: { data: Data }) {
  return <section className="panel" id="audit"><div className="panel-heading"><div><h2>Gateway audit</h2><p>Invocation records returned by the API for your selected role</p></div><span className="subtle-count">{data.audit.length} records</span></div>
    {data.audit.length ? <div className="table-wrap"><table><thead><tr><th>Time</th><th>Principal</th><th>Capability</th><th>Implementation</th><th>Status</th><th>Duration</th><th>Reason</th></tr></thead>
      <tbody>{data.audit.map((record, index) => <tr key={`${record.at}-${record.capability}-${index}`}><td>{formatTime(record.at)}</td><td>{record.principal.role}</td>
        <td><code>{record.capability}</code></td><td>{record.implementation || "—"}</td><td><Status value={record.status} /></td>
        <td>{record.duration_ms} ms</td><td className="error-text">{record.error || "—"}</td></tr>)}</tbody></table></div>
      : <EmptyState text="No gateway invocations have been recorded for this role." />}
  </section>;
}

function RunList({ runs, onSelect, selectedId }: { runs: RunSummary[]; onSelect: (id: string) => void; selectedId?: string }) {
  return runs.length ? <div className="run-list">{runs.map((run) => <button key={run.id} className={`run-row ${run.id === selectedId ? "active" : ""}`} onClick={() => onSelect(run.id)}>
    <span className={`run-dot ${statusTone(run.status)}`} /><span className="run-row-main"><b>{run.goal}</b><small>{formatTime(run.created_at)} · {run.id}</small></span>
    <Status value={run.status} /><ChevronRight size={15} />
  </button>)}</div> : <EmptyState text="No runs have been recorded yet." />;
}

function Stat({ icon: Icon, label, value, detail }: { icon: typeof Bot; label: string; value: number; detail: string }) {
  return <section className="stat-card"><div className="stat-icon"><Icon size={17} /></div><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-detail">{detail}</div></section>;
}

function Status({ value }: { value: string }) {
  return <span className={`status ${statusTone(value)}`}>{value.replaceAll("_", " ")}</span>;
}

function HealthRow({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return <div className="health-row"><span>{label}</span><b>{good === undefined ? value : <><i className={`health-indicator ${good ? "good" : "warn"}`} />{value}</>}</b></div>;
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state"><span><CircleHelp size={18} /></span><p>{text}</p></div>;
}

export default App;
