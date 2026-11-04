import { useEffect, useMemo, useRef, useState } from "react";
import { cancelSolve, exportDesign, redo, showSurface, solveRung, undo } from "../api";
import { useStore, type Tool } from "../store";

// ---- keyboard map (?) ------------------------------------------------------------------------------------

const KEYS: [string, string][] = [
  ["V", "select"], ["S", "support tool: click a face"], ["L", "load tool: click a face"],
  ["K", "keep-in tool: click a face"], ["O", "keep-out tool: click a face"], ["Esc", "cancel the tool, close this"],
  ["Ctrl+Z", "undo"], ["Ctrl+Shift+Z / Ctrl+Y", "redo"], ["Ctrl+K", "command palette"], ["?", "this keyboard map"],
];

export function KeyMap({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal" role="dialog" aria-label="Keyboard map" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <h2>Keyboard</h2>
        <table className="keys"><tbody>
          {KEYS.map(([k, v]) => <tr key={k}><td><kbd>{k}</kbd></td><td>{v}</td></tr>)}
        </tbody></table>
        <div className="note">Shortcuts are off while you type in a field.</div>
      </div>
    </div>
  );
}

// ---- command palette (Ctrl+K) ------------------------------------------------------------------------------

interface Cmd { id: string; label: string; hint?: string; run: () => void; enabled: boolean }

function useCommands(): Cmd[] {
  const rungs = useStore((s) => s.rungs);
  const jobs = useStore((s) => s.jobs);
  const rungJob = useStore((s) => s.rungJob);
  const report = useStore((s) => s.report);
  const worker = useStore((s) => s.worker);
  const setTool = useStore((s) => s.setTool);
  const theme = useStore((s) => s.theme);
  const setTheme = useStore((s) => s.setTheme);
  return useMemo(() => {
    const shown = rungs.slice(-4);
    const names = ["Sketch", "Preview", "Refine", "Release"].slice(4 - shown.length);
    const ok = !!report?.ok && worker !== "dead";
    const done = Object.values(rungJob).map((id) => jobs[id]).filter((j) => j && j.status === "converged");
    const last = done[done.length - 1];
    const running = Object.values(rungJob).some((id) => ["running", "queued"].includes(jobs[id]?.status ?? ""));
    const cmds: Cmd[] = shown.map((r, i) => ({ id: `rung${i}`, label: `Run ${names[i]}`, hint: `${r.n_elem.toLocaleString()} elements`,
      run: () => solveRung(r.h), enabled: ok }));
    const below = shown.length >= 2 ? jobs[rungJob[String(shown[shown.length - 2].h)]] : undefined;
    if (shown.length >= 2)                     // as the ladder's button: only once the rung below has converged
      cmds.push({ id: "opt", label: "Optimize release", hint: "re-opened warm start at the release rung",
        run: () => solveRung(shown[shown.length - 1].h, true), enabled: ok && !running && below?.status === "converged" });
    cmds.push(
      { id: "stop", label: "Stop the running solve", run: cancelSolve, enabled: running },
      { id: "surf", label: "Show the surface of the last design", run: () => last && void showSurface(last.id), enabled: !!last },
      { id: "stl", label: "Export STL and report", hint: "the last finished design", run: () => last && void exportDesign(last.id), enabled: !!last },
      { id: "undo", label: "Undo", hint: "Ctrl+Z", run: undo, enabled: true },
      { id: "redo", label: "Redo", hint: "Ctrl+Shift+Z", run: redo, enabled: true },
      ...(["select", "support", "load", "keepin", "keepout"] as Tool[]).map((t) => ({
        id: `tool-${t}`, label: `Tool: ${{ select: "select", support: "support", load: "load", keepin: "keep-in", keepout: "keep-out" }[t]}`,
        run: () => setTool(t), enabled: true })),
      { id: "theme", label: `Switch to ${theme === "dark" ? "light" : "dark"} theme`, run: () => setTheme(theme === "dark" ? "light" : "dark"), enabled: true },
    );
    return cmds;
  }, [rungs, jobs, rungJob, report, worker, setTool, theme, setTheme]);
}

export function Palette({ onClose }: { onClose: () => void }) {
  const cmds = useCommands();
  const [q, setQ] = useState("");
  const [i, setI] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => input.current?.focus(), []);
  const hits = cmds.filter((c) => c.enabled && c.label.toLowerCase().includes(q.toLowerCase()));
  const go = (c?: Cmd) => { if (c) { onClose(); c.run(); } };
  return (
    <div className="modal" role="dialog" aria-label="Command palette" onClick={onClose}>
      <div className="sheet palette" onClick={(e) => e.stopPropagation()}>
        <input ref={input} value={q} placeholder="Type a command…" aria-label="Command"
          onChange={(e) => { setQ(e.target.value); setI(0); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setI((i + 1) % Math.max(hits.length, 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setI((i - 1 + hits.length) % Math.max(hits.length, 1)); }
            else if (e.key === "Enter") go(hits[i]);
            else if (e.key === "Escape") onClose();
          }} />
        <ul role="listbox">
          {hits.map((c, k) => (
            <li key={c.id} role="option" aria-selected={k === i} className={k === i ? "on" : ""}
              onMouseEnter={() => setI(k)} onClick={() => go(c)}>
              {c.label}{c.hint && <span className="hint">{c.hint}</span>}
            </li>))}
          {!hits.length && <li className="note">No command matches.</li>}
        </ul>
      </div>
    </div>
  );
}

// ---- first-run tips ----------------------------------------------------------------------------------------

const TIPS_KEY = "agentto.tips.v1";

export function FirstRunTips() {
  const [open, setOpen] = useState(() => { try { return localStorage.getItem(TIPS_KEY) !== "done"; } catch { return true; } });
  if (!open) return null;
  const close = () => { try { localStorage.setItem(TIPS_KEY, "done"); } catch { /* storage unavailable */ } setOpen(false); };
  return (
    <div className="tips" role="note" aria-label="Getting started">
      <strong>Getting started</strong>
      <ol>
        <li>Describe the part to the assistant, or build it yourself: <kbd>S</kbd> support, <kbd>L</kbd> load, <kbd>K</kbd>/<kbd>O</kbd> keep-in/out, then click a face.</li>
        <li>Fix anything the checks list in red; the ladder unlocks when the spec is valid.</li>
        <li>Run Sketch for a design in seconds, then Preview and Refine. Release runs the finest grid; only a <em>certified</em> release is a finished design.</li>
      </ol>
      <div className="row"><span className="note"><kbd>?</kbd> keyboard · <kbd>Ctrl+K</kbd> commands</span>
        <button className="linkbtn" onClick={close}>Got it</button></div>
    </div>
  );
}

// ---- host: global keys for the overlays -------------------------------------------------------------------

export function Overlays() {
  const [which, setWhich] = useState<"" | "keys" | "palette">("");
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      const typing = /INPUT|SELECT|TEXTAREA/.test(t.tagName) || t.isContentEditable;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setWhich((w) => (w === "palette" ? "" : "palette")); }
      else if (e.key === "Escape") setWhich("");
      else if (!typing && e.key === "?") { e.preventDefault(); setWhich((w) => (w === "keys" ? "" : "keys")); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  if (which === "keys") return <KeyMap onClose={() => setWhich("")} />;
  if (which === "palette") return <Palette onClose={() => setWhich("")} />;
  return null;
}
