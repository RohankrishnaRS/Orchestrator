import { Check, FileJson, LoaderCircle, Plus, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { FormEvent } from "react";

type Props = {
  role: string;
  onRegistered?: () => Promise<void> | void;
};

type Mode = "json" | "form";

const EXAMPLE = {
  id: "poc.clause.extract",
  version: "1.0.0",
  description: "Find requested clause types in one contract and return the exact clause text with its location.",
  effects: "read",
  languages: ["en"],
  selection_doc: {
    use_when: "A task asks whether a contract contains specific clauses, or needs the verbatim text of named clauses.",
    do_not_use_when: "The task needs a risk opinion, a compliance verdict or a written report.",
    examples: { en: ["Does this NDA contain a confidentiality clause?"] },
  },
  input_schema: {
    type: "object",
    additionalProperties: false,
    required: ["document"],
    properties: {
      document: {
        type: "object",
        anyOf: [{ required: ["sample"] }, { required: ["text"] }],
        properties: { sample: { type: "string" }, document_id: { type: "string" }, text: { type: "string" } },
      },
      clause_types: { type: "array", minItems: 1, items: { type: "string" } },
    },
  },
  output_schema: {
    type: "object",
    required: ["document_id", "clauses", "found_count", "missing"],
    properties: {
      document_id: { type: "string" },
      clauses: { type: "array", items: { type: "object", required: ["clause_type", "found", "text", "verified"] } },
      found_count: { type: "integer", minimum: 0 },
      missing: { type: "array", items: { type: "string" } },
      summary: { type: "string" },
      mode: { enum: ["llm", "offline"] },
    },
  },
  samples: [
    {
      document: { document_id: "sample-nda", text: "1. Confidentiality\nEach party shall keep Confidential Information strictly confidential.\n" },
      clause_types: ["confidentiality", "indemnification"],
    },
  ],
};

const pretty = (value: unknown) => JSON.stringify(value, null, 2);
const ID_RE = /^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$/;
const SEMVER_RE = /^\d+\.\d+\.\d+$/;

function parseJson(text: string, label: string): { value?: unknown; error?: string } {
  try {
    return { value: JSON.parse(text) };
  } catch (error) {
    return { error: `${label}: ${error instanceof Error ? error.message : String(error)}` };
  }
}

function validateContract(contract: Record<string, unknown>): string[] {
  const issues: string[] = [];
  if (typeof contract.id !== "string" || !ID_RE.test(contract.id)) issues.push("id must look like domain.action (lowercase, dot-separated)");
  if (typeof contract.version !== "string" || !SEMVER_RE.test(contract.version)) issues.push("version must be semver, e.g. 1.0.0");
  if (typeof contract.description !== "string" || contract.description.trim().length < 10) issues.push("description is required (10+ chars)");
  if (!["read", "write"].includes(String(contract.effects))) issues.push("effects must be 'read' or 'write'");
  for (const key of ["input_schema", "output_schema"]) {
    const schema = contract[key];
    if (typeof schema !== "object" || schema === null || (schema as { type?: unknown }).type !== "object") issues.push(`${key} must be a JSON Schema object with type "object"`);
  }
  if (!Array.isArray(contract.samples) || contract.samples.length === 0) issues.push("at least one sample input is required (used for conformance checks)");
  return issues;
}

async function postCapability(role: string, contract: unknown): Promise<Record<string, unknown>> {
  const response = await fetch("/api/v1/capabilities", {
    method: "POST",
    headers: { "X-Demo-Role": role, "Content-Type": "application/json" },
    body: JSON.stringify(contract),
  });
  const text = await response.text();
  let body: unknown = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }
  if (!response.ok) {
    const detail = typeof body === "object" && body !== null && "detail" in body ? (body as { detail: unknown }).detail : body;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return (body ?? {}) as Record<string, unknown>;
}

export default function RegisterCapability({ role, onRegistered }: Props) {
  const [mode, setMode] = useState<Mode>("json");
  const [jsonText, setJsonText] = useState(pretty(EXAMPLE));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  // form mode
  const [id, setId] = useState("");
  const [version, setVersion] = useState("1.0.0");
  const [description, setDescription] = useState("");
  const [effects, setEffects] = useState<"read" | "write">("read");
  const [useWhen, setUseWhen] = useState("");
  const [doNotUseWhen, setDoNotUseWhen] = useState("");
  const [inputSchema, setInputSchema] = useState(pretty(EXAMPLE.input_schema));
  const [outputSchema, setOutputSchema] = useState(pretty(EXAMPLE.output_schema));
  const [samples, setSamples] = useState<string[]>([pretty(EXAMPLE.samples[0])]);

  const formContract = useMemo(() => {
    const input = parseJson(inputSchema, "Input schema");
    const output = parseJson(outputSchema, "Output schema");
    const parsedSamples = samples.map((sample, index) => parseJson(sample, `Sample ${index + 1}`));
    const errors = [input.error, output.error, ...parsedSamples.map((s) => s.error)].filter(Boolean) as string[];
    const contract = {
      id: id.trim(),
      version: version.trim(),
      description: description.trim(),
      effects,
      languages: ["en"],
      selection_doc: { use_when: useWhen.trim(), do_not_use_when: doNotUseWhen.trim(), examples: { en: [] as string[] } },
      input_schema: input.value ?? {},
      output_schema: output.value ?? {},
      samples: parsedSamples.map((s) => s.value).filter((s) => s !== undefined),
    };
    return { contract, errors };
  }, [id, version, description, effects, useWhen, doNotUseWhen, inputSchema, outputSchema, samples]);

  const currentContract = (): { contract?: Record<string, unknown>; issues: string[] } => {
    if (mode === "json") {
      const parsed = parseJson(jsonText, "Contract JSON");
      if (parsed.error) return { issues: [parsed.error] };
      if (typeof parsed.value !== "object" || parsed.value === null || Array.isArray(parsed.value)) return { issues: ["Contract must be a JSON object"] };
      const contract = parsed.value as Record<string, unknown>;
      return { contract, issues: validateContract(contract) };
    }
    const issues = [...formContract.errors, ...validateContract(formContract.contract as Record<string, unknown>)];
    return { contract: formContract.contract as Record<string, unknown>, issues };
  };

  const validation = currentContract();

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);
    if (validation.issues.length || !validation.contract) {
      setError(validation.issues.join("; "));
      return;
    }
    setBusy(true);
    try {
      const created = await postCapability(role, validation.contract);
      setResult(created);
      await onRegistered?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div><h2>Register a capability</h2><p>Define the contract an agent implements: id, schemas and samples. Agents are matched to it by <code>implements</code>.</p></div>
      </div>
      <div className="tab-row" role="tablist" aria-label="Capability registration method">
        <button type="button" className={`tab-button ${mode === "json" ? "selected" : ""}`} onClick={() => setMode("json")}><FileJson size={14} /> Paste contract JSON</button>
        <button type="button" className={`tab-button ${mode === "form" ? "selected" : ""}`} onClick={() => setMode("form")}>Form</button>
      </div>
      <form className="register-form" onSubmit={(event) => void submit(event)}>
        {role !== "admin" && <div className="error-box">Switch to the admin role to register capabilities.</div>}
        {mode === "json" ? (
          <>
            <label className="field-label" htmlFor="capability-json">Capability contract (JSON)</label>
            <textarea id="capability-json" className="code-input" value={jsonText} onChange={(event) => setJsonText(event.target.value)} spellCheck={false} rows={18} />
            <p className="muted">Same shape as a catalog <code>capabilities/*.yaml</code> entry: id, version, description, effects, selection_doc, input_schema, output_schema, samples.</p>
          </>
        ) : (
          <div className="editor-grid">
            <label className="editor-field"><span>Capability ID (e.g. poc.clause.extract)</span><input value={id} onChange={(event) => setId(event.target.value)} required /></label>
            <label className="editor-field"><span>Version</span><input value={version} onChange={(event) => setVersion(event.target.value)} required /></label>
            <label className="editor-field"><span>Effects</span>
              <select value={effects} onChange={(event) => setEffects(event.target.value as "read" | "write")}><option value="read">read</option><option value="write">write (needs approval)</option></select></label>
            <label className="editor-field wide"><span>Description</span><textarea value={description} onChange={(event) => setDescription(event.target.value)} required maxLength={500} /></label>
            <label className="editor-field wide"><span>Use this capability when</span><textarea value={useWhen} onChange={(event) => setUseWhen(event.target.value)} maxLength={1000} /></label>
            <label className="editor-field wide"><span>Do not use when</span><textarea value={doNotUseWhen} onChange={(event) => setDoNotUseWhen(event.target.value)} maxLength={1000} /></label>
            <label className="editor-field"><span>Input JSON Schema</span><textarea className="code-input small-code" value={inputSchema} onChange={(event) => setInputSchema(event.target.value)} spellCheck={false} /></label>
            <label className="editor-field"><span>Output JSON Schema</span><textarea className="code-input small-code" value={outputSchema} onChange={(event) => setOutputSchema(event.target.value)} spellCheck={false} /></label>
            <div className="editor-field wide">
              <span>Sample inputs (used by conformance checks)</span>
              {samples.map((sample, index) => (
                <div key={index} className="inline-form">
                  <textarea className="code-input small-code" value={sample} onChange={(event) => setSamples((current) => current.map((s, i) => (i === index ? event.target.value : s)))} spellCheck={false} />
                  <button type="button" className="secondary-button compact" aria-label="Remove sample" disabled={samples.length === 1} onClick={() => setSamples((current) => current.filter((_, i) => i !== index))}><Trash2 size={14} /></button>
                </div>
              ))}
              <button type="button" className="secondary-button compact" onClick={() => setSamples((current) => [...current, "{\n  \n}"])}><Plus size={14} /> Add sample</button>
            </div>
          </div>
        )}
        {validation.issues.length > 0 && <div className="error-box"><b>Fix before registering:</b><ul>{validation.issues.map((issue) => <li key={issue}>{issue}</li>)}</ul></div>}
        {error && <div className="error-box">{error}</div>}
        <div className="form-actions">
          <span className="muted">Admin role required. The contract is registered immediately; agents implementing it can then be discovered.</span>
          <button type="submit" className="primary-button" disabled={busy || role !== "admin" || validation.issues.length > 0}>
            {busy ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />} Register capability
          </button>
        </div>
        {result && (
          <div className="preview-result"><b>Registered {String(result.id)} v{String(result.version)}</b>
            <span>Now register the agent that implements it (Register an agent → Automatic · A2A discovery).</span>
            <details><summary>Stored contract</summary><pre className="json-output">{pretty(result)}</pre></details>
          </div>
        )}
      </form>
    </section>
  );
}