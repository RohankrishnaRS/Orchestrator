export type Page =
  | "Overview"
  | "Prompt Builder"
  | "Orchestrator"
  | "Agent Registry"
  | "Capabilities"
  | "Workflow"
  | "Saved workflows"
  | "Workflows"
  | "Skills"
  | "MCP Registry"
  | "Runs"
  | "Approvals"
  | "Audit";

const PAGE_PATHS: Record<Page, string> = {
  Overview: "/",
  "Prompt Builder": "/create/prompt",
  Orchestrator: "/orchestrator",
  "Agent Registry": "/agent-registry",
  Capabilities: "/capabilities",
  Workflow: "/create/workflow",
  "Saved workflows": "/create/saved-workflows",
  Workflows: "/workflows",
  Skills: "/skills",
  "MCP Registry": "/mcp-registry",
  Runs: "/runs",
  Approvals: "/approvals",
  Audit: "/audit",
};

export function pathForPage(page: Page): string {
  return PAGE_PATHS[page];
}

export function pageForPath(pathname: string): Page {
  for (const [page, path] of Object.entries(PAGE_PATHS)) {
    if (path === pathname) return page as Page;
  }
  return "Overview";
}
