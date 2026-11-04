import { useStore } from "./store";
import type { AgentRunState, Card, LogRec } from "./store";
import type { Report, Spec } from "./types";

export async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json() as Promise<T>;
}

export async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.json() as Promise<T>;
}

export const validate = (spec: Spec) => postJSON<Report>("/api/validate", spec);

let ws: WebSocket | null = null;
let refN = 0;
const pending = new Map<string, (r: ToolResult) => void>();
export interface ToolResult { ok: boolean; seq?: number; result?: any; error?: string; spec_hash?: string }

export function send(msg: unknown) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

/** One call on the session's tool contract (logged with actor "user"). */
export function callTool(name: string, args: Record<string, unknown> = {}, label?: string): Promise<ToolResult> {
  const ref = `r${++refN}`;
  return new Promise((resolve) => {
    pending.set(ref, resolve);
    send({ type: "tool", name, args, ref, meta: label ? { label } : undefined });
  });
}

// GUI edits reach the session as whole-spec commits, 300 ms after the last change (one log record per pause)
let syncTimer: number | undefined;
function flushSync() {
  window.clearTimeout(syncTimer);
  syncTimer = undefined;
  const st = useStore.getState();
  if (st.spec && st.specOrigin === "local") {
    useStore.setState({ specOrigin: "server" });
    void callTool("set_spec", { spec: st.spec }, st.editLabel);
  }
}
useStore.subscribe((s, prev) => {
  if (s.spec !== prev.spec && s.specOrigin === "local") {
    window.clearTimeout(syncTimer);
    syncTimer = window.setTimeout(flushSync, 300);
  }
});

/** Solve rung h through the session: it warm-starts from this spec's design on the next coarser rung, queueing
 *  behind that rung if it is still running. The job appears in the store from the broadcast log record. */
/** optimize: re-open the warm start (about 60 iterations) instead of only certifying the upsampled design. */
export function solveRung(h: number, optimize = false) {
  flushSync();
  void callTool("solve", { h, wait: false, ...(optimize ? { optimize: true } : {}) }).then((r) => {
    if (!r.ok) useStore.getState().set({ report: { ...(useStore.getState().report ?? { ok: false }),
      findings: [...(useStore.getState().report?.findings ?? []), { code: "SOLVE", severity: "error", message: r.error ?? "solve failed", ids: [] }] } });
  });
}

export function cancelSolve() { void callTool("cancel"); }

/** Fetch the preview surface of a finished job and show it in the viewport (replaces the voxel frame). */
export async function showSurface(job: string) {
  const r = await fetch(`/api/mesh/${job}`);
  if (!r.ok) throw new Error(`mesh ${job}: ${r.status}`);
  const buf = await r.arrayBuffer();
  const head = new DataView(buf, 0, 12);
  const nv = head.getUint32(0, true), nt = head.getUint32(4, true);
  const info = JSON.parse(r.headers.get("X-Mesh-Info") ?? "{}");
  const pos = new Float32Array(buf, 12, nv * 3);
  const idx = new Uint32Array(buf, 12 + nv * 12, nt * 3);
  useStore.getState().set({ surface: { job, pos, idx, pool: info.pool ?? 1, triangles: nt } });
}

const patchExports = (job: string, p: Record<string, unknown>) => {
  const s = useStore.getState();
  s.set({ exports: { ...s.exports, [job]: { ...s.exports[job], ...p } } });
};
const fileName = (path: string) => path.split(/[\\/]/).pop() ?? path;

/** Export the design as STL, then write the report (which records the STL); both are logged tool calls. */
export async function exportDesign(job: string) {
  patchExports(job, { busy: "STL" });
  const a = await callTool("export_stl", { job }, "export STL");
  if (a.ok) patchExports(job, { stl: fileName(a.result.path), triangles: a.result.triangles, mb: a.result.bytes / 1e6 });
  patchExports(job, { busy: "report" });
  const b = await callTool("export_report", { job }, "export report");
  patchExports(job, b.ok ? { report: fileName(b.result.json), markdown: fileName(b.result.markdown), busy: undefined }
    : { busy: undefined });
}
export function undo() { flushSync(); void callTool("undo", {}, "undo"); }
export function redo() { flushSync(); void callTool("redo", {}, "redo"); }

// ---- assistant ----------------------------------------------------------------------------------------------

/** Send a message to the assistant with the GUI context (selection, current findings, latest result). */
export function sendAgent(message: string) {
  flushSync();
  const s = useStore.getState();
  const sel = s.selection;
  const last = [...s.log].reverse().find((r) => r.type === "result")?.result;
  const context = {
    selection: sel ? (sel.kind === "domain" || sel.kind === "material" ? { kind: sel.kind } : sel) : null,
    findings: (s.report?.findings ?? []).map((f) => `${f.code} ${f.severity}: ${f.message}`),
    result: last ? { status: last.status, n_elem: last.n_elem, compliance_phys: last.compliance_phys, certified: last.certified } : null,
  };
  send({ type: "agent", message, mode: s.agentMode, context });
}
export function stopAgent() { send({ type: "agent_stop" }); }
export function revertAgent(run: string) { flushSync(); send({ type: "agent_revert", run }); }
export function answerClarify(qid: string, choice: number | null, label?: string, comment?: string) {
  send({ type: "clarify_answer", qid, choice, label, comment });
}
export function acceptCards(run: string, cards: string[]) {
  flushSync();
  send({ type: "agent_accept", run, cards });
}
export function rejectCards(cards: string[]) {
  const r = useStore.getState().agentRun;
  if (!r) return;
  const decided = { ...r.decided };
  cards.forEach((c) => { decided[c] = "rejected"; });
  useStore.getState().set({ agentRun: { ...r, decided } });
}

function onAgent(m: Record<string, any>) {
  const s = useStore.getState();
  const cur = s.agentRun;
  const patch = (p: Partial<AgentRunState>) => {
    const r = useStore.getState().agentRun;
    if (r && r.id === m.run) useStore.getState().set({ agentRun: { ...r, ...p } });
  };
  switch (m.kind) {
    case "start":
      s.set({ agentRun: { id: m.run, mode: m.mode, running: true, t0: Date.now(), steps: [], messages: [], cards: [],
        proposal: null, decided: {} } });
      break;
    case "tool": if (cur) patch({ steps: [...cur.steps, { id: m.id, tool: m.tool, args: m.args }] }); break;
    case "tool_done":
      if (cur) patch({ steps: cur.steps.map((x) => (x.id === m.id ? { ...x, ok: m.ok, summary: m.summary } : x)) });
      break;
    case "message": if (cur) patch({ messages: [...cur.messages, m.text] }); break;
    case "proposal": patch({ cards: m.cards as Card[], proposal: m.spec }); break;
    case "error":
      if (m.run && cur?.id === m.run) patch({ error: m.message });
      else s.set({ chat: [...s.chat, { role: "assistant", text: `⚠ ${m.message}` }] });
      break;
    case "done":
      patch({ running: false, timing: m.timing, usage: m.usage, ...(m.stderr ? { error: m.stderr } : {}) });
      s.set({ chat: [...useStore.getState().chat, { role: "assistant", text: m.final || "(no reply)", run: m.run }] });
      break;
  }
}

/** Session socket: reconnects with backoff, measures round-trip time, and routes worker events into the store. */
export function connectSession(): () => void {
  let timer: number | undefined;
  let retry = 500;
  let stopped = false;
  let frameHeader: { job: string; shape: [number, number, number]; final?: boolean } | null = null;
  const st = () => useStore.getState();

  const onEvent = (m: Record<string, any>) => {
    const s = st();
    switch (m.type) {
      case "hello": s.set({ worker: m.worker, client: m.client }); break;
      case "tool_result": { const f = pending.get(m.ref); if (f) { pending.delete(m.ref); f(m as ToolResult); } break; }
      case "spec":
        // agent edits, undo, solves from other clients; our own whole-spec commits are already shown
        if (!(m.origin === s.client && m.tool === "set_spec")) {
          s.serverSpec(m.spec);
          // undo/redo back to a solved spec: show that spec's rung jobs on the ladder again (its log record came first)
          const hash = [...st().log].reverse().find((r) => r.type === "call" && r.ok)?.spec_hash;
          const rungJob: Record<string, string> = {};
          if (hash) Object.values(st().jobs).filter((j) => j.spec === hash && !["cancelled", "failed", "error", "skipped"].includes(j.status))
            .forEach((j) => { rungJob[String(j.h)] = j.id; });
          st().set({ rungJob });
        }
        break;
      case "log": {
        const r = m.record as LogRec;
        s.set({ log: [...s.log.slice(-199), r] });
        if (r.type === "call" && r.tool === "solve" && r.ok && r.result?.job) {
          const h = r.result.h as number;
          s.patchJob(r.result.job, { h, status: "queued", it: 0, history: [], spec: r.spec_hash });
          s.set({ rungJob: { ...useStore.getState().rungJob, [String(h)]: r.result.job } });
        }
        // a certified release on screen is shown as its marching-cubes surface, not as voxels
        if (r.type === "result" && r.result?.certified && r.result?.release_rung && !r.result?.preview &&
            Object.values(st().rungJob).includes(r.result.job)) void showSurface(r.result.job).catch(() => undefined);
        break;
      }
      case "chat": s.set({ chat: [...s.chat, { role: m.role, text: m.text, run: m.run }] }); break;
      case "agent_event": onAgent(m); break;
      case "clarify": s.set({ clarify: { qid: m.qid, question: m.question, options: m.options, previews: m.previews, previews_s: m.previews_s } }); break;
      case "clarify_done": if (s.clarify?.qid === m.qid) s.set({ clarify: null }); break;
      case "agent_accepted": {
        const r = s.agentRun;
        if (r && r.id === m.run && m.ok) {
          const decided = { ...r.decided };
          (m.cards as string[]).forEach((c) => { decided[c] = "accepted"; });
          s.set({ agentRun: { ...r, decided } });
        }
        break;
      }
      case "pong": if (typeof m.echo === "number") s.setConn("open", performance.now() - m.echo); break;
      case "worker": s.set({ worker: m.state }); break;
      case "started": s.patchJob(m.job, { status: "running", warm: m.warm }); s.set({ worker: "ready" }); break;
      case "progress": {
        const j = s.jobs[m.job];
        s.patchJob(m.job, { it: m.it, c_phys: m.c_phys, vol: m.vol, history: [...(j?.history ?? []), m.c_phys] });
        break;
      }
      case "frame": frameHeader = { job: m.job, shape: m.shape, final: m.final }; break;
      case "done":
        s.patchJob(m.job, { status: m.status, it: m.iterations, c_phys: m.compliance_phys, vol: m.volume,
          compute_s: m.compute_s, certified: !!m.certificate?.certified, message: m.error ?? undefined });
        break;
      case "skipped": s.patchJob(m.job, { status: "skipped" }); break;
      case "error": if (m.job) s.patchJob(m.job, { status: "error", message: m.message ?? m.kind }); break;
    }
  };

  const open = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/session`);
    ws.binaryType = "arraybuffer";
    st().setConn("connecting");
    ws.onopen = () => {
      retry = 500;
      st().setConn("open");
      send({ type: "worker_start" });
      timer = window.setInterval(() => send({ type: "ping", t: performance.now() }), 5000);
    };
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        if (frameHeader) {
          const cur = st().rungJob;
          const isCurrent = Object.values(cur).includes(frameHeader.job);
          const keep = st().surface?.job === frameHeader.job;     // a late frame of the job whose surface is shown
          if (isCurrent) st().set({ frame: { ...frameHeader, data: new Uint8Array(ev.data) }, ...(keep ? {} : { surface: null }) });
          frameHeader = null;
        }
        return;
      }
      onEvent(JSON.parse(ev.data));
    };
    ws.onclose = () => {
      window.clearInterval(timer);
      st().setConn("closed");
      if (!stopped) window.setTimeout(open, (retry = Math.min(retry * 2, 8000)));
    };
  };
  open();
  return () => { stopped = true; ws?.close(); };
}
