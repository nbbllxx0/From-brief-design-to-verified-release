import { useEffect, useState } from "react";
import { connectSession, getJSON, postJSON, redo, undo, validate } from "./api";
import { useStore, type ChatMsg, type LogRec, type Rung } from "./store";
import type { Spec } from "./types";
import { Viewport } from "./components/Viewport";
import { Assistant, Checks, Inspector, IntentTree, Ladder, ResultCard } from "./components/Panels";
import { FirstRunTips, Overlays } from "./components/Overlays";

const DEFAULT = "task:b06_motor_mount";

function TopBar({ sources }: { sources: string[] }) {
  const { theme, setTheme, conn, rttMs, report, source, spec } = useStore();
  const errs = (report?.findings?.filter((f) => f.severity === "error").length ?? 0) + (report?.schema_errors?.length ?? 0);
  const warns = report?.findings?.filter((f) => f.severity === "warning").length ?? 0;
  const status = !report ? ["", "checking…"] : errs ? ["err", `${errs} error${errs > 1 ? "s" : ""}`]
    : warns ? ["warn", `valid · ${warns} warning${warns > 1 ? "s" : ""}`] : ["ok", "valid spec"];
  return (
    <header className="topbar">
      <span className="brand">AgenticTO</span>
      <select className="picker" aria-label="Problem" value={source} onChange={(e) => loadSource(e.target.value)}>
        {source === "session" && <option value="session">live session</option>}
        {source === "new" && <option value="new">untitled</option>}
        <option value="new:">New design…</option>
        {sources.map((s) => <option key={s} value={s}>{s.replace(/^(lib|task):/, "")}</option>)}
      </select>
      <span className="spacer" />
      {spec && <span className="pill"><span className={`dot ${status[0]}`} />{status[1]}</span>}
      <span className="pill" title="backend connection">
        <span className={`dot ${conn === "open" ? "ok" : conn === "closed" ? "err" : ""}`} />
        {conn === "open" ? `backend${rttMs != null ? ` · ${rttMs.toFixed(0)} ms` : ""}` : conn}
      </span>
      <button className="iconbtn" aria-label="Toggle theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
        {theme === "dark" ? "Light" : "Dark"}
      </button>
    </header>
  );
}

async function loadSource(src: string) {
  const { load } = useStore.getState();
  if (src === "new:") {                       // a blank box: size, units and material are set in the inspector
    load("new", { schema_version: "agentto.spec/v0", name: "untitled", units: { length: "mm", force: "N" },
      domain: { kind: "box", size: [100, 50, 50] }, material: { name: "steel", E: 210e3, nu: 0.3 },
      volume_fraction: 0.3, supports: [], loads: [], passive: [], resolution: { cells_short_edge: 20 } });
    useStore.getState().select({ kind: "domain" });
  } else if (src.startsWith("task:")) {
    const t = await getJSON<{ spec: Spec; brief: string }>(`/api/tasks/${src.slice(5)}`);
    load(src, { ...t.spec, passive: t.spec.passive ?? [] }, t.brief);
  } else {
    load(src, await getJSON<Spec>(`/api/library/${src.slice(4)}`));
  }
}

/** Re-validate and refresh the resolution ladder 300 ms after the last spec edit. */
function useLiveChecks() {
  const spec = useStore((s) => s.spec);
  useEffect(() => {
    if (!spec) return;
    const t = window.setTimeout(async () => {
      const [report, rungs] = await Promise.all([validate(spec), postJSON<Rung[]>("/api/ladder", spec).catch(() => [])]);
      if (useStore.getState().spec === spec) useStore.getState().set({ report, rungs: Array.isArray(rungs) ? rungs : [] });
    }, 300);
    return () => window.clearTimeout(t);
  }, [spec]);
}

export function App() {
  const theme = useStore((s) => s.theme);
  const [sources, setSources] = useState<string[]>([]);
  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  useLiveChecks();
  useEffect(() => connectSession(), []);
  useEffect(() => {
    (async () => {
      const [lib, tasks, sess] = await Promise.all([getJSON<string[]>("/api/library"), getJSON<string[]>("/api/tasks"),
        getJSON<{ spec: Spec | null; log: LogRec[]; chat?: ChatMsg[] }>("/api/session")]);
      const all = [...tasks.map((t) => `task:${t}`), ...lib.map((l) => `lib:${l}`)];
      setSources(all);
      useStore.getState().set({ chat: sess.chat ?? [] });
      if (sess.spec) {                       // a live session (e.g. an agent's) already holds a spec: show it
        useStore.getState().serverSpec(sess.spec);
        useStore.getState().set({ source: "session", log: sess.log });
      } else {
        await loadSource(all.includes(DEFAULT) ? DEFAULT : all[0]);
      }
    })().catch(console.error);
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (/INPUT|SELECT|TEXTAREA/.test(t.tagName) || !(e.ctrlKey || e.metaKey)) return;
      const k = e.key.toLowerCase();
      if (k === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
      else if ((k === "z" && e.shiftKey) || k === "y") { e.preventDefault(); redo(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return (
    <div className="app">
      <TopBar sources={sources} />
      <aside className="left">
        <IntentTree />
        <Checks />
        <Inspector />
      </aside>
      <Viewport />
      <aside className="right"><FirstRunTips /><ResultCard /><Assistant /></aside>
      <Ladder />
      <Overlays />
    </div>
  );
}
