import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import {
  AlertCircle, ArrowDownToLine, ArrowUpFromLine, Bot, Check, ChevronDown, ChevronRight,
  CircleHelp, Grip, LoaderCircle, Play, Plus, Search, Settings2,
  Sparkles, Trash2, Wrench, ZoomIn, ZoomOut,
} from "lucide-react";

type Capability = { id: string; description: string; implementations: string[] };
type Agent = { id: string; name: string; version: string; capability: string };
type Skill = { id: string; name: string; description: string; instructions: string };
type McpServer = { id: string; name: string; description: string };
export type WorkflowSummary = { id: string; name: string; description?: string; version?: string; version_ref?: string; valid: boolean; published?: boolean; builtin?: boolean; updated_at: string; nodes?: number; capabilities?: string[] };
type StepKind = "agent" | "skill" | "mcp_tool";
type Step = {
  id: string;
  type: "capability";
  kind: StepKind;
  capability: string;
  skill_id?: string;
  depends_on: string[];
  inputs: Record<string, unknown>;
  canvas_position: { x: number; y: number };
};
export type Workflow = {
  id: string;
  version: string;
  goal: string;
  allowed_tasks: string[];
  inputs: Record<string, Record<string, unknown>>;
  nodes: Step[];
  outputs: Record<string, string>;
};
export type Validation = { ok: boolean; issues: { message: string; node?: string | null }[]; warnings?: { message: string }[] };
export type WorkflowDocument = {
  workflow: Workflow;
  name: string;
  description: string;
  builtin?: boolean;
};
export type RunResult = {
  id: string;
  status: string;
  nodes: Record<string, { state: string; error?: string; output?: unknown }>;
  outputs: unknown;
};
type PaletteItem = { id: string; label: string; description: string; kind: StepKind; capability: string; skillId?: string };

type Props = {
  capabilities: Capability[];
  agents: Agent[];
  skills: Skill[];
  mcpServers: McpServer[];
  liveRun: RunResult | null;
  initialWorkflowId?: string | null;
  initialPrompt?: string | null;
  onLoad: (id: string) => Promise<WorkflowDocument>;
  onValidate: (workflow: Workflow) => Promise<Validation>;
  onSave: (id: string, workflow: Workflow, name: string, description: string) => Promise<WorkflowDocument>;
  onDraft: (goal: string) => Promise<{
    ok: boolean;
    workflow: Workflow | null;
    rationale?: string;
    issues?: string[];
    considered?: string[];
  }>;
  onRun: (workflow: Workflow, inputs: Record<string, unknown>) => Promise<RunResult>;
};

const START_INPUT = { id: "input", label: "Input", x: 45, y: 275 };
const END_OUTPUT = { id: "output", label: "Output", x: 1120, y: 275 };

function slug(value: string): string {
  return value.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "new-workflow";
}

function stepId(value: string): string {
  const cleaned = value.toLowerCase().trim().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40);
  return cleaned && /^[a-z]/.test(cleaned) ? cleaned : `step_${cleaned || "node"}`;
}

function toExpression(value: unknown): unknown {
  if (typeof value === "string") {
    return value.replace(/\{\{input(?:\.([a-zA-Z0-9_.-]+))?\}\}/g, (_match, path: string | undefined) =>
      path ? `$inputs.input.${path}` : "$inputs.input")
      .replace(/\{\{steps\.([a-z][a-z0-9_]*)\.output(?:\.([a-zA-Z0-9_.-]+))?\}\}/g, (_match, id: string, path: string | undefined) =>
        path ? `$nodes.${id}.output.${path}` : `$nodes.${id}.output`);
  }
  if (Array.isArray(value)) return value.map(toExpression);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, toExpression(item)]));
  }
  return value;
}

function showExpression(value: unknown): unknown {
  if (typeof value === "string") {
    return value.replace(/\$inputs\.input(?:\.([a-zA-Z0-9_.-]+))?/g, (_match, path: string | undefined) =>
      path ? `{{input.${path}}}` : "{{input}}")
      .replace(/\$nodes\.([a-z][a-z0-9_]*)\.output(?:\.([a-zA-Z0-9_.-]+))?/g, (_match, id: string, path: string | undefined) =>
        path ? `{{steps.${id}.output.${path}}}` : `{{steps.${id}.output}}`);
  }
  if (Array.isArray(value)) return value.map(showExpression);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, showExpression(item)]));
  }
  return value;
}

function initialWorkflow(): Workflow {
  return {
    id: "new-workflow",
    version: "0.1.0",
    goal: "",
    allowed_tasks: [],
    inputs: { input: { type: "object", properties: {}, additionalProperties: true } },
    nodes: [],
    outputs: {},
  };
}

function isValidGraph(workflow: Workflow, capabilities: Capability[]): string[] {
  const problems: string[] = [];
  if (!workflow.id || !/^[a-z][a-z0-9-]*$/.test(workflow.id)) problems.push("Workflow name must produce a valid lowercase ID.");
  if (!workflow.allowed_tasks.length) problems.push("spec.allowed_tasks needs at least one task.");
  if (!workflow.nodes.length) problems.push("Add at least one Agent, Skill, or MCP tool step.");
  const ids = new Set(workflow.nodes.map((node) => node.id));
  if (ids.size !== workflow.nodes.length) problems.push("Step IDs must be unique.");
  for (const node of workflow.nodes) {
    if (!node.capability || !capabilities.some((capability) => capability.id === node.capability)) {
      problems.push(`${node.id}: choose an available execution capability.`);
    }
    for (const dependency of node.depends_on) {
      if (!ids.has(dependency) || dependency === node.id) problems.push(`${node.id}: invalid connection to ${dependency}.`);
    }
    const visit = (id: string, path: Set<string>): boolean => {
      if (path.has(id)) return true;
      const step = workflow.nodes.find((candidate) => candidate.id === id);
      if (!step) return false;
      const next = new Set(path).add(id);
      return step.depends_on.some((dependency) => visit(dependency, next));
    };
    if (visit(node.id, new Set())) problems.push(`${node.id}: workflow connections contain a cycle.`);
    const refs = JSON.stringify(node.inputs).matchAll(/\{\{steps\.([a-z][a-z0-9_]*)\.output(?:\.[a-zA-Z0-9_.-]+)?\}\}/g);
    for (const [, referenced] of refs) {
      if (!ids.has(referenced)) problems.push(`${node.id}: reference to missing step "${referenced}".`);
      else if (!node.depends_on.includes(referenced)) problems.push(`${node.id}: connect step "${referenced}" before using its output.`);
    }
  }
  return [...new Set(problems)];
}

export default function WorkflowStudio(props: Props) {
  const { capabilities, agents, skills, mcpServers } = props;
  const [workflow, setWorkflow] = useState<Workflow>(initialWorkflow);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [draftPrompt, setDraftPrompt] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [runOpen, setRunOpen] = useState(false);
  const [inputsText, setInputsText] = useState('{"input":{}}');
  const [inputSchemaText, setInputSchemaText] = useState(JSON.stringify(initialWorkflow().inputs, null, 2));
  const [outputText, setOutputText] = useState("");
  const [validation, setValidation] = useState<Validation | null>(null);
  const [notice, setNotice] = useState("");
  const [working, setWorking] = useState(false);
  const [runResult, setRunResult] = useState<RunResult | null>(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [connectingFrom, setConnectingFrom] = useState<string | null>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const interaction = useRef<{ kind: "move" | "pan"; id?: string; x: number; y: number; originX: number; originY: number } | null>(null);

  const selectedNode = workflow.nodes.find((node) => node.id === selected) || null;
  const graphIssues = useMemo(() => isValidGraph(workflow, capabilities), [workflow, capabilities]);
  const palette: Record<string, PaletteItem[]> = useMemo(() => {
    const agentsItems = agents.map((agent) => ({
      id: `agent:${agent.id}`, label: agent.name, description: agent.capability, kind: "agent" as const, capability: agent.capability,
    }));
    const skillItems = skills.map((skill) => ({
      id: `skill:${skill.id}`, label: skill.name, description: skill.description, kind: "skill" as const,
      capability: capabilities[0]?.id || "", skillId: skill.id,
    }));
    const mcpItems = capabilities.map((capability) => ({
      id: `mcp:${capability.id}`, label: capability.id, description: capability.description, kind: "mcp_tool" as const,
      capability: capability.id,
    }));
    return { Agents: agentsItems, Skills: skillItems, "MCP tools": mcpItems };
  }, [agents, capabilities, skills]);
  const shownPalette = Object.entries(palette).map(([category, items]) => [
    category,
    items.filter((item) => `${item.label} ${item.description}`.toLowerCase().includes(query.toLowerCase())),
  ] as const);

  function applyWorkflow(doc: WorkflowDocument) {
    setWorkflow({
      ...doc.workflow,
      id: doc.workflow.id || slug(doc.name),
      version: doc.workflow.version || "0.1.0",
      goal: doc.workflow.goal || doc.description || "",
      allowed_tasks: doc.workflow.allowed_tasks || [],
      inputs: doc.workflow.inputs || initialWorkflow().inputs,
      outputs: doc.workflow.outputs || {},
      nodes: doc.workflow.nodes.map((node, index) => ({
        ...node,
        kind: node.kind || "agent",
        canvas_position: node.canvas_position || { x: 340 + (index % 3) * 235, y: 160 + Math.floor(index / 3) * 155 },
      })),
    });
    setName(doc.name || doc.workflow.id);
    setDescription(doc.description || doc.workflow.goal);
    setInputSchemaText(JSON.stringify(doc.workflow.inputs, null, 2));
    const firstOutput = Object.values(doc.workflow.outputs || {})[0] || "";
    setOutputText(String(showExpression(firstOutput)));
    setSelected(null);
    setValidation(null);
  }

  useEffect(() => {
    if (props.initialWorkflowId) {
      void loadWorkflowById(props.initialWorkflowId);
      return;
    }
    const blank = initialWorkflow();
    setWorkflow(blank);
    setName("");
    setDescription("");
    setDraftPrompt("");
    setInputSchemaText(JSON.stringify(blank.inputs, null, 2));
    setOutputText("");
    setSelected(null);
    setValidation(null);
  }, [props.initialWorkflowId]);

  useEffect(() => {
    if (!props.initialPrompt) return;
    setDraftPrompt(props.initialPrompt);
    setDescription(props.initialPrompt);
    setWorkflow((current) => ({ ...current, goal: props.initialPrompt || current.goal }));
  }, [props.initialPrompt]);

  async function loadWorkflowById(id: string) {
    setWorking(true);
    try {
      applyWorkflow(await props.onLoad(id));
      setNotice(`Loaded ${id}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setWorking(false);
    }
  }

  function addItem(item: PaletteItem, position?: { x: number; y: number }) {
    const id = stepId(item.label);
    let uniqueId = id;
    let suffix = 2;
    while (workflow.nodes.some((node) => node.id === uniqueId)) uniqueId = `${id}_${suffix++}`;
    const dependency = workflow.nodes.at(-1)?.id;
    const next: Step = {
      id: uniqueId,
      type: "capability",
      kind: item.kind,
      capability: item.capability,
      ...(item.skillId ? { skill_id: item.skillId } : {}),
      depends_on: dependency ? [dependency] : [],
      inputs: { input: "{{input}}" },
      canvas_position: position || { x: 350 + (workflow.nodes.length % 3) * 230, y: 150 + Math.floor(workflow.nodes.length / 3) * 150 },
    };
    setWorkflow((current) => ({
      ...current,
      id: slug(name || current.id),
      nodes: [...current.nodes, next],
      allowed_tasks: [...new Set([...current.allowed_tasks, item.capability].filter(Boolean))],
    }));
    setSelected(uniqueId);
    setValidation(null);
  }

  function updateNode(id: string, update: Partial<Step>) {
    setWorkflow((current) => {
      const nodes = current.nodes.map((node) => node.id === id ? { ...node, ...update } : node);
      return {
        ...current,
        nodes,
        allowed_tasks: [...new Set([...current.allowed_tasks, ...nodes.map((node) => node.capability).filter(Boolean)])],
      };
    });
    setValidation(null);
  }

  function removeNode(id: string) {
    setWorkflow((current) => ({
      ...current,
      nodes: current.nodes.filter((node) => node.id !== id).map((node) => ({ ...node, depends_on: node.depends_on.filter((dep) => dep !== id) })),
      allowed_tasks: current.allowed_tasks.filter((task) => current.nodes.some((node) => node.id !== id && node.capability === task)),
    }));
    setSelected(null);
    setValidation(null);
  }

  function makeWorkflow(): Workflow {
    let inputs: Record<string, Record<string, unknown>>;
    try {
      const parsed: unknown = JSON.parse(inputSchemaText);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw new Error("Input schema must be a JSON object.");
      inputs = parsed as Record<string, Record<string, unknown>>;
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : "Input schema must be valid JSON.");
    }
    return {
      ...workflow,
      id: slug(name || workflow.id),
      goal: description || name,
      inputs,
      nodes: workflow.nodes.map((node) => ({ ...node, inputs: toExpression(node.inputs) as Record<string, unknown> })),
      outputs: outputText.trim()
        ? { result: toExpression(outputText.trim()) as string }
        : workflow.nodes.length ? { result: `$nodes.${workflow.nodes.at(-1)?.id}.output` } : {},
    };
  }

  async function validate(candidate: Workflow): Promise<boolean> {
    const localIssues = isValidGraph(candidate, capabilities);
    if (localIssues.length) {
      setValidation({ ok: false, issues: localIssues.map((message) => ({ message })) });
      return false;
    }
    try {
      const result = await props.onValidate(candidate);
      setValidation(result);
      return result.ok;
    } catch (error) {
      setValidation({ ok: false, issues: [{ message: error instanceof Error ? error.message : String(error) }] });
      return false;
    }
  }

  async function save(): Promise<WorkflowDocument | null> {
    let candidate: Workflow;
    try {
      candidate = makeWorkflow();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
      return null;
    }
    await validate(candidate);
    setWorking(true);
    try {
      const saved = await props.onSave(candidate.id, candidate, name || candidate.id, description || candidate.goal);
      applyWorkflow(saved);
      setNotice(`Saved ${candidate.id}@${candidate.version}.`);
      return saved;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
      return null;
    } finally {
      setWorking(false);
    }
  }

  async function createFromDescription() {
    if (!draftPrompt.trim()) {
      setNotice("Describe what the workflow should do.");
      return;
    }
    setWorking(true);
    try {
      const result = await props.onDraft(draftPrompt.trim());
      const draft = result.workflow;
      if (!result.ok || !draft) {
        const reasons = [result.rationale, ...(result.issues || [])].filter(Boolean).join(" ");
        const available = result.considered?.length ? ` Available workflow capabilities: ${result.considered.join(", ")}.` : "";
        const readiness = result.considered?.length
          ? ""
          : " No approved read-capability is available. In Build → Agent Registry, create an agent with Describe with AI or retry registration for an existing agent, then generate the workflow again.";
        setNotice(`${reasons || "No executable workflow could be generated."}${readiness}${available}`);
        return;
      }
      applyWorkflow({
        workflow: {
          ...draft,
          id: draft.id || slug(draftPrompt),
          version: draft.version || "0.1.0",
          goal: draft.goal || draftPrompt.trim(),
          allowed_tasks: draft.allowed_tasks?.length
            ? draft.allowed_tasks
            : [...new Set((draft.nodes || []).map((node) => node.capability).filter(Boolean))],
          inputs: draft.inputs || initialWorkflow().inputs,
          nodes: (draft.nodes || []).map((node, index) => ({
            ...node,
            kind: node.kind || "agent",
            canvas_position: node.canvas_position || { x: 340 + (index % 3) * 235, y: 160 + Math.floor(index / 3) * 155 },
          })),
          outputs: draft.outputs || {},
        },
        name: draft.id || slug(draftPrompt),
        description: draft.goal || draftPrompt.trim(),
      });
      setNotice(result.rationale || "AI created a workflow draft. Review it, then save or test it.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setWorking(false);
    }
  }

  async function runWorkflow() {
    let candidate: Workflow;
    let inputs: Record<string, unknown>;
    try {
      candidate = makeWorkflow();
      const parsed: unknown = JSON.parse(inputsText);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw new Error("Run input must be a JSON object.");
      inputs = parsed as Record<string, unknown>;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
      return;
    }
    if (!await validate(candidate)) return;
    setWorking(true);
    setRunResult(null);
    try {
      setRunResult(await props.onRun(candidate, inputs));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setWorking(false);
    }
  }

  function arrange() {
    setWorkflow((current) => ({
      ...current,
      nodes: current.nodes.map((node, index) => ({
        ...node,
        canvas_position: { x: 330 + index * 235, y: 275 },
      })),
    }));
  }

  function beginMove(event: ReactPointerEvent<HTMLDivElement>, node: Step) {
    event.stopPropagation();
    interaction.current = {
      kind: "move", id: node.id, x: event.clientX, y: event.clientY,
      originX: node.canvas_position.x, originY: node.canvas_position.y,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setSelected(node.id);
  }

  function beginPan(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.target instanceof Element && event.target.closest(".studio-step-node, .studio-virtual-node")) return;
    interaction.current = { kind: "pan", x: event.clientX, y: event.clientY, originX: offset.x, originY: offset.y };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function movePointer(event: ReactPointerEvent<HTMLDivElement>) {
    const current = interaction.current;
    if (!current) return;
    const dx = (event.clientX - current.x) / zoom;
    const dy = (event.clientY - current.y) / zoom;
    if (current.kind === "move" && current.id) {
      updateNode(current.id, { canvas_position: { x: Math.max(0, current.originX + dx), y: Math.max(0, current.originY + dy) } });
    } else {
      setOffset({ x: current.originX + dx, y: current.originY + dy });
    }
  }

  function endPointer() {
    interaction.current = null;
  }

  function dropItem(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const item = JSON.parse(event.dataTransfer.getData("application/json")) as PaletteItem;
    const bounds = stageRef.current?.getBoundingClientRect();
    if (!bounds) return;
    addItem(item, { x: (event.clientX - bounds.left - offset.x) / zoom - 80, y: (event.clientY - bounds.top - offset.y) / zoom - 35 });
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.key === "Delete" || event.key === "Backspace") && selected && !/INPUT|TEXTAREA|SELECT/.test((event.target as HTMLElement).tagName)) {
        removeNode(selected);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selected]);

  const edgeNodes = workflow.nodes;
  const edges = edgeNodes.flatMap((node) => node.depends_on.flatMap((dependency) => {
    const from = edgeNodes.find((candidate) => candidate.id === dependency);
    return from ? [{ from, to: node }] : [];
  }));
  const issues = [...graphIssues, ...(validation?.issues.map((issue) => issue.message) || [])];
  const taskText = workflow.allowed_tasks.join(", ");

  useEffect(() => {
    if (props.liveRun && runResult?.id === props.liveRun.id) setRunResult(props.liveRun);
  }, [props.liveRun, runResult?.id]);

  return (
    <div className="workflow-studio">
      <div className={`studio-readiness ${graphIssues.length ? "warning" : "ready"}`}>
        <AlertCircle size={17} />
        <span>{graphIssues.length ? "To see the preview: add at least one allowed task and an executable step in Workflow settings." : "Workflow is ready to validate against the live registry."}</span>
        <button className="secondary-button compact" onClick={() => setSettingsOpen(true)}><Settings2 size={14} /> Workflow settings</button>
      </div>
      <div className="studio-toolbar">
        <div><p className="eyebrow">CREATE / WORKFLOW</p><h2>{name || "Untitled workflow"}</h2></div>
        <div className="studio-toolbar-actions">
          <button className="secondary-button compact" onClick={() => void save()} disabled={working}><Check size={14} /> Save draft</button>
          <button className="primary-button compact" onClick={() => setRunOpen(true)} disabled={working}><Play size={14} /> Run / Test</button>
        </div>
      </div>
      <section className="studio-ai-draft">
        <div><Sparkles size={16} /><b>Create workflow with AI</b><span>Describe the process and review the generated flow on the canvas.</span></div>
        <textarea value={draftPrompt} onChange={(event) => setDraftPrompt(event.target.value)} rows={2} placeholder="Example: Triage a customer support request, classify its urgency, and draft a response." />
        <button className="primary-button compact" disabled={working || !draftPrompt.trim()} onClick={() => void createFromDescription()}>
          {working ? <LoaderCircle size={14} className="spin" /> : <Sparkles size={14} />} Generate workflow
        </button>
      </section>
      <div className="studio-three-pane">
        <aside className="studio-palette">
          <div className="studio-pane-title">BUILDING BLOCKS <span>{capabilities.length + skills.length + agents.length}</span></div>
          <label className="studio-search"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search agents, skills, MCP tools..." /></label>
          {shownPalette.map(([category, items]) => <section className="palette-section" key={category}>
            <button className="palette-category" onClick={() => setCollapsed((state) => ({ ...state, [category]: !state[category] }))}>
              {collapsed[category] ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
              <span>{category}</span><b>{items.length}</b>
            </button>
            {!collapsed[category] && <div className="palette-items">{items.map((item) => <button
              className="palette-item" key={item.id} draggable onDragStart={(event) => event.dataTransfer.setData("application/json", JSON.stringify(item))}
              onClick={() => addItem(item)} title={item.description}>
              {item.kind === "agent" ? <Bot size={15} /> : item.kind === "skill" ? <Sparkles size={15} /> : <Wrench size={15} />}
              <span><b>{item.label}</b><small>{item.description || item.capability || "Registry item"}</small></span><Plus size={13} />
            </button>)}</div>}
          </section>)}
          {!mcpServers.length && <p className="studio-muted">MCP server registry is empty. Tools exposed as registered capabilities appear above.</p>}
        </aside>
        <section className="studio-canvas-column">
          <div className="studio-canvas-toolbar">
            <button className="secondary-button compact" onClick={() => setSettingsOpen(true)}><Settings2 size={14} /> Workflow settings</button>
            <span className={`status ${graphIssues.length ? "warn" : "good"}`}>{graphIssues.length ? "tasks missing" : "ready"}</span>
            <button className="secondary-button compact" onClick={arrange}>Auto-arrange</button>
            <div className="studio-zoom"><button aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(.55, value - .1))}><ZoomOut size={14} /></button><span>{Math.round(zoom * 100)}%</span><button aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(1.5, value + .1))}><ZoomIn size={14} /></button></div>
          </div>
          <div className="studio-canvas" onWheel={(event) => { if (event.ctrlKey || event.metaKey) { event.preventDefault(); setZoom((value) => Math.max(.55, Math.min(1.5, value + (event.deltaY < 0 ? .05 : -.05)))); } }}>
            <div className="studio-stage" ref={stageRef} onPointerDown={beginPan} onPointerMove={movePointer} onPointerUp={endPointer}
              onPointerCancel={endPointer} onDragOver={(event) => event.preventDefault()} onDrop={dropItem}>
              <div className="studio-world" style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})` }}>
                <svg className="studio-edges" width="1400" height="650">
                  {edges.map(({ from, to }) => <path key={`${from.id}-${to.id}`} d={`M ${from.canvas_position.x + 185} ${from.canvas_position.y + 48} C ${from.canvas_position.x + 255} ${from.canvas_position.y + 48}, ${to.canvas_position.x - 70} ${to.canvas_position.y + 48}, ${to.canvas_position.x} ${to.canvas_position.y + 48}`} />)}
                  {workflow.nodes.filter((node) => !node.depends_on.length).map((node) => <path key={`input-${node.id}`} d={`M ${START_INPUT.x + 175} ${START_INPUT.y + 48} C ${START_INPUT.x + 245} ${START_INPUT.y + 48}, ${node.canvas_position.x - 70} ${node.canvas_position.y + 48}, ${node.canvas_position.x} ${node.canvas_position.y + 48}`} />)}
                  {workflow.nodes.filter((node) => !workflow.nodes.some((candidate) => candidate.depends_on.includes(node.id))).map((node) => <path key={`output-${node.id}`} d={`M ${node.canvas_position.x + 185} ${node.canvas_position.y + 48} C ${node.canvas_position.x + 255} ${node.canvas_position.y + 48}, ${END_OUTPUT.x - 70} ${END_OUTPUT.y + 48}, ${END_OUTPUT.x} ${END_OUTPUT.y + 48}`} />)}
                </svg>
                <article className="studio-virtual-node" style={{ left: START_INPUT.x, top: START_INPUT.y }}>
                  <div><ArrowUpFromLine size={16} /><b>{START_INPUT.label}</b></div><code>{"{{input}}"}</code><span>Workflow input</span><i className="studio-port output-port" />
                </article>
                <article className="studio-virtual-node" style={{ left: END_OUTPUT.x, top: END_OUTPUT.y }}>
                  <div><ArrowDownToLine size={16} /><b>{END_OUTPUT.label}</b></div><code>{outputText || "{{steps.last.output}}"}</code><span>Workflow output</span><i className="studio-port input-port" />
                </article>
                {workflow.nodes.map((node) => {
                  const capability = capabilities.find((item) => item.id === node.capability);
                  const icon = node.kind === "agent" ? <Bot size={16} /> : node.kind === "skill" ? <Sparkles size={16} /> : <Wrench size={16} />;
                  return <div key={node.id} className={`studio-step-node ${selected === node.id ? "selected" : ""}`}
                    style={{ left: node.canvas_position.x, top: node.canvas_position.y }}
                    onPointerDown={(event) => beginMove(event, node)} onClick={() => setSelected(node.id)}>
                    <button className="studio-port input-port" aria-label={`Connect to ${node.id}`} onPointerDown={(event) => event.stopPropagation()} onClick={(event) => {
                      event.stopPropagation();
                      if (connectingFrom && connectingFrom !== node.id) {
                        updateNode(node.id, { depends_on: [...new Set([...node.depends_on, connectingFrom])] });
                        setConnectingFrom(null);
                      }
                    }} />
                    <div className="studio-node-heading">{icon}<b>{node.id}</b><button aria-label={`Delete ${node.id}`} onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); removeNode(node.id); }}><Trash2 size={13} /></button></div>
                    <code>{capability ? `${capability.id}@^1.0.0` : node.capability || "Select capability"}</code>
                    <p>{capability?.description || "Configure this workflow step."}</p>
                    <small>{node.kind === "mcp_tool" ? "MCP tool" : node.kind === "skill" ? "Skill" : "Agent"}</small>
                    <button className="studio-port output-port" aria-label={`Connect from ${node.id}`} onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); setConnectingFrom(node.id); }} />
                  </div>;
                })}
              </div>
            </div>
            <div className="studio-canvas-hint"><Grip size={13} /> Drag nodes to arrange · Drag blocks here or click to add · Select a node to configure</div>
          </div>
        </section>
        <aside className="studio-inspector">
          <div className="studio-pane-title">STEP INSPECTOR</div>
          {selectedNode ? <div className="studio-inspector-content">
            <h3>{selectedNode.id}</h3>
            <label className="studio-field"><span>Step id</span><input value={selectedNode.id} onChange={(event) => {
              const nextId = stepId(event.target.value);
              setWorkflow((current) => ({
                ...current,
                nodes: current.nodes.map((node) => ({
                  ...node,
                  id: node.id === selectedNode.id ? nextId : node.id,
                  depends_on: node.depends_on.map((dependency) => dependency === selectedNode.id ? nextId : dependency),
                })),
              }));
              setSelected(nextId);
            }} /></label>
            <div className="studio-reference"><span>Referenced as</span><code>{`{{steps.${selectedNode.id}.output}}`}</code></div>
            <label className="studio-field"><span>Step type</span><select value={selectedNode.kind} onChange={(event) => updateNode(selectedNode.id, { kind: event.target.value as StepKind })}>
              <option value="agent">Agent</option><option value="skill">Skill</option><option value="mcp_tool">MCP tool</option>
            </select></label>
            {selectedNode.kind === "agent" && <label className="studio-field"><span>Agent</span><select value={selectedNode.capability} onChange={(event) => updateNode(selectedNode.id, { capability: event.target.value })}>
              <option value="">Select an agent</option>{agents.map((agent) => <option key={agent.id} value={agent.capability}>{agent.name} · {agent.capability}@^{agent.version}</option>)}
              {capabilities.filter((item) => !agents.some((agent) => agent.capability === item.id)).map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
            </select></label>}
            {selectedNode.kind === "skill" && <>
              <label className="studio-field"><span>Skill</span><select value={selectedNode.skill_id || ""} onChange={(event) => updateNode(selectedNode.id, { skill_id: event.target.value })}>
                <option value="">Select a skill</option>{skills.map((skill) => <option value={skill.id} key={skill.id}>{skill.name}</option>)}
              </select></label>
              <div className="studio-preview"><b>Context / instructions</b><p>{skills.find((skill) => skill.id === selectedNode.skill_id)?.instructions || "Select a skill to preview its instructions."}</p></div>
              <label className="studio-field"><span>Execution capability</span><select value={selectedNode.capability} onChange={(event) => updateNode(selectedNode.id, { capability: event.target.value })}>
                <option value="">Select a registered capability</option>{capabilities.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
              </select></label>
            </>}
            {selectedNode.kind === "mcp_tool" && <label className="studio-field"><span>MCP tool</span><select value={selectedNode.capability} onChange={(event) => updateNode(selectedNode.id, { capability: event.target.value })}>
              <option value="">Select a registered tool</option>{capabilities.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.implementations.length} implementations</option>)}
            </select></label>}
            {selectedNode.kind === "agent" && <div className="studio-preview"><b>Does</b><p>{capabilities.find((item) => item.id === selectedNode.capability)?.description || "Choose a registered agent to see its capabilities."}</p></div>}
            {selectedNode.depends_on.length > 0 && <div className="studio-connections"><b>Connected from</b>{selectedNode.depends_on.map((dependency) => <div key={dependency}>
              <code>{dependency}</code><button className="text-button" onClick={() => updateNode(selectedNode.id, { depends_on: selectedNode.depends_on.filter((item) => item !== dependency) })}>Remove</button>
            </div>)}</div>}
            <label className="studio-field"><span>Description / input mapping (JSON)</span><textarea rows={5} value={JSON.stringify(showExpression(selectedNode.inputs), null, 2)} onChange={(event) => {
              try {
                const parsed: unknown = JSON.parse(event.target.value);
                if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) updateNode(selectedNode.id, { inputs: parsed as Record<string, unknown> });
              } catch { /* Keep the last valid mapping until the JSON is corrected. */ }
            }} /></label>
            <div className="studio-binding-help">Bind values with <code>{"{{input}}"}</code> or <code>{"{{steps.step_id.output}}"}</code>.</div>
            <button className="danger-button compact" onClick={() => removeNode(selectedNode.id)}><Trash2 size={14} /> Delete step</button>
          </div> : <div className="studio-empty-inspector"><CircleHelp size={20} /><p>Select a canvas step to configure its type, registry item, and input bindings.</p></div>}
        </aside>
      </div>
      <footer className="studio-publish-bar">
        <div><b>Workflow draft: {slug(name || workflow.id)}@{workflow.version}</b><small> · Save to continue editing later</small>
          {issues.length > 0 && <span className="studio-inline-error">{issues[0]}</span>}
          {validation?.warnings?.map((warning) => <span className="studio-inline-warning" key={warning.message}>{warning.message}</span>)}
        </div>
        <button className="primary-button" disabled={working} onClick={() => void save()}>{working ? <LoaderCircle size={14} className="spin" /> : <Check size={14} />} Save workflow</button>
      </footer>
      {notice && <div className="studio-toast" role="status">{notice}</div>}

      {settingsOpen && <div className="studio-modal-backdrop" role="presentation" onClick={(event) => { if (event.target === event.currentTarget) setSettingsOpen(false); }}>
        <section className="studio-modal" role="dialog" aria-modal="true" aria-labelledby="workflow-settings-title">
          <div className="panel-heading"><div><h2 id="workflow-settings-title">Workflow settings</h2><p>Metadata, allowed tasks, and input/output contracts.</p></div><button className="icon-button" onClick={() => setSettingsOpen(false)} aria-label="Close settings">×</button></div>
          <label className="studio-field"><span>Workflow name</span><input value={name} onChange={(event) => { setName(event.target.value); setWorkflow((current) => ({ ...current, id: slug(event.target.value) })); }} placeholder="support-triage" /></label>
          <label className="studio-field"><span>Version</span><input value={workflow.version} onChange={(event) => setWorkflow((current) => ({ ...current, version: event.target.value }))} placeholder="0.1.0" /></label>
          <label className="studio-field"><span>Description / goal</span><textarea rows={2} value={description} onChange={(event) => setDescription(event.target.value)} placeholder="What this workflow resolves" /></label>
          <label className="studio-field"><span>spec.allowed_tasks (comma-separated)</span><input value={taskText} onChange={(event) => setWorkflow((current) => ({ ...current, allowed_tasks: [...new Set(event.target.value.split(",").map((task) => task.trim()).filter(Boolean))] }))} placeholder="support.answer" /></label>
          <label className="studio-field"><span>Input schema (workflow input definitions JSON)</span><textarea rows={7} className="studio-code" value={inputSchemaText} onChange={(event) => setInputSchemaText(event.target.value)} /></label>
          <label className="studio-field"><span>Output template</span><input value={outputText} onChange={(event) => setOutputText(event.target.value)} placeholder="{{steps.last_step.output}}" /></label>
          <div className="form-actions"><button className="primary-button" onClick={() => setSettingsOpen(false)}>Done</button></div>
        </section>
      </div>}

      {runOpen && <div className="studio-modal-backdrop" role="presentation" onClick={(event) => { if (event.target === event.currentTarget) setRunOpen(false); }}>
        <section className="studio-modal studio-run-modal" role="dialog" aria-modal="true" aria-labelledby="workflow-run-title">
          <div className="panel-heading"><div><h2 id="workflow-run-title">Run / Test workflow</h2><p>Supply workflow input and inspect per-step results.</p></div><button className="icon-button" onClick={() => setRunOpen(false)} aria-label="Close run panel">×</button></div>
          <label className="studio-field"><span>Input JSON</span><textarea rows={7} className="studio-code" value={inputsText} onChange={(event) => setInputsText(event.target.value)} /></label>
          <button className="primary-button" disabled={working} onClick={() => void runWorkflow()}>{working ? <LoaderCircle size={15} className="spin" /> : <Play size={15} />} Run workflow</button>
          {runResult && <div className="studio-run-result"><div className="panel-heading"><b>Run {runResult.status}</b><code>{runResult.id}</code></div>
            {Object.entries(runResult.nodes).map(([id, node]) => <article className="studio-run-node" key={id}><div><b>{id}</b><span className={`status ${node.state === "succeeded" ? "good" : node.state === "failed" ? "bad" : "warn"}`}>{node.state}</span></div>
              {node.error && <p className="studio-inline-error">{node.error}</p>}{node.output !== undefined && <pre>{JSON.stringify(node.output, null, 2)}</pre>}</article>)}
            <details><summary>Workflow output</summary><pre>{JSON.stringify(runResult.outputs, null, 2)}</pre></details>
          </div>}
        </section>
      </div>}
    </div>
  );
}
