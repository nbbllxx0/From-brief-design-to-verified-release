import { create } from "zustand";
import type { Report, Selection, Spec } from "./types";

type Conn = "connecting" | "open" | "closed";
export type Tool = "select" | "support" | "load" | "keepin" | "keepout";

export interface Rung { h: number; grid: [number, number, number]; n_elem: number }
export interface JobState {
  id: string; h: number; status: "queued" | "running" | "converged" | "not_converged" | "cancelled" | "failed" | "error" | "skipped";
  it: number; c_phys?: number; vol?: number; compute_s?: number; certified?: boolean; warm?: boolean; message?: string;
  spec?: string;                            // hash of the spec it solves
  history: number[];
}
export interface Frame { job: string; shape: [number, number, number]; data: Uint8Array; final?: boolean }
/** Released surface of a finished job (rho = 0.5 isosurface, pooled for display), positions in spec units. */
export interface Surface { job: string; pos: Float32Array; idx: Uint32Array; pool: number; triangles: number }
export interface ChatMsg { role: "user" | "assistant"; text: string; run?: string }
export interface AgentStep { id: string; tool: string; args?: unknown; ok?: boolean; summary?: string }
export interface Card { id: string; op: "add" | "remove" | "change" | "set"; list?: string; item_id?: string; field?: string;
  before?: any; after?: any; text: string }
export interface AgentRunState {
  id: string; mode: "propose" | "act"; running: boolean; t0: number; steps: AgentStep[]; messages: string[];
  cards: Card[]; proposal: Spec | null; decided: Record<string, "accepted" | "rejected">; error?: string;
  timing?: Record<string, number>; usage?: Record<string, number>;
}
export interface ClarifyPreview { ok: boolean; compliance_phys?: number; png_b64?: string; error?: string; compute_s?: number }
export interface Clarify { qid: string; question: string; options: { label: string; item_kind?: string; item?: any; params?: any }[];
  previews: ClarifyPreview[] | null; previews_s?: number }
export interface Exports { stl?: string; triangles?: number; mb?: number; report?: string; markdown?: string; busy?: string }
/** One session-log record (agentto.tools): a tool call, or the late result of a solve. */
export interface LogRec {
  type: "call" | "result" | "session" | "answer"; seq?: number; spec_hash?: string; t?: string; actor?: string; tool?: string; ok?: boolean;
  error?: string; args?: Record<string, unknown>; result?: Record<string, any>; meta?: { label?: string; origin?: string };
  answer?: { choice: number | null; label?: string | null; comment?: string; by?: string };   // type "answer"
}

interface State {
  theme: "light" | "dark";
  tool: Tool;
  conn: Conn;
  rttMs: number | null;
  worker: "off" | "starting" | "ready" | "dead";
  source: string;
  spec: Spec | null;
  specOrigin: "local" | "server";          // local edits are synced to the session as set_spec calls
  editLabel: string;
  client: string | null;
  log: LogRec[];
  brief: string | null;
  report: Report | null;
  selection: Selection;
  rungs: Rung[];
  jobs: Record<string, JobState>;
  rungJob: Record<string, string>;          // h -> job id (latest)
  frame: Frame | null;
  surface: Surface | null;
  exports: Record<string, Exports>;
  chat: ChatMsg[];
  agentMode: "propose" | "act";
  agentRun: AgentRunState | null;
  ghostCard: string | null;                 // card under the pointer: drawn in the viewport
  clarify: Clarify | null;                  // an open ask_clarification question
  setTheme: (t: "light" | "dark") => void;
  setTool: (t: Tool) => void;
  setConn: (c: Conn, rttMs?: number | null) => void;
  load: (source: string, spec: Spec, brief?: string | null) => void;
  editSpec: (fn: (s: Spec) => Spec, label?: string) => void;
  serverSpec: (spec: Spec | null) => void;
  setReport: (r: Report) => void;
  select: (s: Selection) => void;
  set: (p: Partial<State>) => void;
  patchJob: (id: string, p: Partial<JobState>) => void;
}

const initialTheme = (): "light" | "dark" => {
  try {
    const t = localStorage.getItem("agentto.theme");
    if (t === "light" || t === "dark") return t;
  } catch { /* storage unavailable */ }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
};

export const useStore = create<State>((set) => ({
  theme: initialTheme(),
  tool: "select",
  conn: "connecting",
  rttMs: null,
  worker: "off",
  source: "",
  spec: null,
  specOrigin: "server",
  editLabel: "",
  client: null,
  log: [],
  brief: null,
  report: null,
  selection: null,
  rungs: [],
  jobs: {},
  rungJob: {},
  frame: null,
  surface: null,
  exports: {},
  chat: [],
  agentMode: "propose",
  agentRun: null,
  ghostCard: null,
  clarify: null,
  setTheme: (theme) => {
    try { localStorage.setItem("agentto.theme", theme); } catch { /* ignore */ }
    set({ theme });
  },
  setTool: (tool) => set({ tool }),
  setConn: (conn, rttMs = null) => set({ conn, rttMs }),
  load: (source, spec, brief = null) =>
    set({ source, spec, brief, report: null, selection: null, rungs: [], rungJob: {}, frame: null, surface: null, specOrigin: "local",
      editLabel: `load ${source}` }),
  editSpec: (fn, label = "edit") =>
    set((s) => (s.spec ? { spec: fn(structuredClone(s.spec)), rungJob: {}, specOrigin: "local", editLabel: label } : {})),
  serverSpec: (spec) => set((s) => ({ spec: spec ? { ...spec, passive: spec.passive ?? [] } : null, specOrigin: "server",
    rungJob: {}, selection: s.selection })),
  setReport: (report) => set({ report }),
  select: (selection) => set({ selection }),
  set: (p) => set(p),
  patchJob: (id, p) => set((s) => {
    const prev = s.jobs[id] ?? { id, h: 0, status: "queued", it: 0, history: [] };
    return { jobs: { ...s.jobs, [id]: { ...prev, ...p } } };
  }),
}));
