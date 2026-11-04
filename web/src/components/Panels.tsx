import { useEffect, useRef, useState } from "react";
import { acceptCards, answerClarify, cancelSolve, revertAgent, exportDesign, rejectCards, sendAgent, showSurface, solveRung, stopAgent } from "../api";
import { useStore, type JobState } from "../store";
import type { AgentRunState, AgentStep, ChatMsg, LogRec } from "../store";
import type { Axis, Face, Finding, Region, Spec, Vec3 } from "../types";

const fmt = (x: number) => (Math.abs(x) >= 1e4 || (x !== 0 && Math.abs(x) < 1e-2) ? x.toExponential(2) : String(+x.toFixed(4)));
const sizeLabel = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M` : `${(n / 1e3).toFixed(0)}k`);

function findingsFor(id: string, fs: Finding[]) {
  const mine = fs.filter((f) => f.ids.includes(id));
  if (mine.some((f) => f.severity === "error")) return <span className="badge err">error</span>;
  if (mine.some((f) => f.severity === "warning")) return <span className="badge warn">check</span>;
  return null;
}

export function IntentTree() {
  const spec = useStore((s) => s.spec);
  const report = useStore((s) => s.report);
  const sel = useStore((s) => s.selection);
  const select = useStore((s) => s.select);
  if (!spec) return <div className="tree" />;
  const fs = report?.findings ?? [];
  const item = (kind: "support" | "load" | "passive", id: string, color: string) => (
    <button key={id} className="tree-item" role="treeitem" aria-selected={sel?.id === id} onClick={() => select({ kind, id })}>
      <span className="swatch" style={{ background: color }} />
      {id}
      {findingsFor(id, fs)}
    </button>
  );
  const keepIn = spec.passive.filter((p) => p.kind === "solid");
  const keepOut = spec.passive.filter((p) => p.kind === "void");
  return (
    <div className="tree" role="tree" aria-label="Design intent">
      <div className="section-title">Design intent</div>
      <div className="tree-group">
        <button className="tree-item" role="treeitem" aria-selected={sel?.kind === "domain"} onClick={() => select({ kind: "domain" })}>
          Domain <span className="badge">{spec.domain.size.join("×")}</span>
        </button>
        <button className="tree-item" role="treeitem" aria-selected={sel?.kind === "material"} onClick={() => select({ kind: "material" })}>
          Material <span className="badge">{spec.material.name}</span>
        </button>
      </div>
      {([["Supports", spec.supports, "support", "var(--support)"], ["Loads", spec.loads, "load", "var(--load)"],
        ["Keep-in", keepIn, "passive", "var(--keepin)"], ["Keep-out", keepOut, "passive", "var(--keepout)"]] as const).map(
        ([title, items, kind, color]) => (
          <div className="tree-group" key={title}>
            <div className="tree-head">{title}<span className="count">{items.length}</span></div>
            {items.map((x) => item(kind, x.id, color))}
          </div>
        ))}
    </div>
  );
}

// ---- inspector: exact values, edited in place ------------------------------------------------------------

function Num({ label, value, onChange, unit, step }: { label: string; value: number; onChange: (v: number) => void; unit?: string; step?: number }) {
  const [text, setText] = useState<string | null>(null);
  return (
    <>
      <dt>{label}</dt>
      <dd>
        <input className="num" type="number" step={step ?? "any"} aria-label={label} value={text ?? String(value)}
          onChange={(e) => {
            setText(e.target.value);
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v)) onChange(v);
          }}
          onBlur={() => setText(null)} />
        {unit && <span className="unit">{unit}</span>}
      </dd>
    </>
  );
}

function VecEdit({ label, v, onChange, unit, names = ["x", "y", "z"] }: { label: string; v: number[]; onChange: (v: number[]) => void; unit?: string; names?: string[] }) {
  return (
    <>
      {v.map((x, i) => (
        <Num key={i} label={`${label} ${names[i]}`} value={x} unit={unit}
          onChange={(nv) => { const c = [...v]; c[i] = nv; onChange(c); }} />
      ))}
    </>
  );
}

const inPlane = (r: Region) => {
  if (r.kind !== "patch") return ["a", "b"];
  return ({ x: ["y", "z"], y: ["x", "z"], z: ["x", "y"] } as Record<string, string[]>)[r.face[0]];
};

const FACES: Face[] = ["x-", "x+", "y-", "y+", "z-", "z+"];
const AXES: Axis[] = ["x", "y", "z"];
const r4 = (v: number) => +v.toPrecision(4);
const IDX = { x: 0, y: 1, z: 2 } as const;

/** `r` as another region kind, keeping what carries over (the face, the axis) and sizing the rest from the domain. */
function convert(r: Region, kind: Region["kind"], L: number[]): Region {
  const face: Face = "face" in r ? r.face : "x+";
  const axis: Axis = "axis" in r ? r.axis : "z";
  const ia = IDX[axis], [ib, ic] = [0, 1, 2].filter((i) => i !== ia);
  const fa = IDX[face[0] as Axis], [fb, fc] = [0, 1, 2].filter((i) => i !== fa);
  switch (kind) {
    case "face": return { kind, face };
    case "patch": return { kind, face, center: [r4(L[fb] / 2), r4(L[fc] / 2)], size: [r4(L[fb] / 10), r4(L[fc] / 10)], shape: "rect" };
    case "edge": return { kind, axis, at: [0, 0], span: null };
    case "box": return { kind, min: L.map((v) => r4(v * 0.45)) as Vec3, max: L.map((v) => r4(v * 0.55)) as Vec3 };
    case "cylinder": return { kind, axis, center: [r4(L[ib] / 2), r4(L[ic] / 2)], radius: r4(Math.min(L[ib], L[ic]) / 10), span: [0, L[ia]] };
  }
}

function RegionEdit({ r, onChange, unit, L, kinds }: { r: Region; onChange: (r: Region) => void; unit: string; L: number[]; kinds: Region["kind"][] }) {
  return (<>
    <dt>region</dt><dd>
      <select className="picker" aria-label="region kind" value={r.kind} onChange={(e) => onChange(convert(r, e.target.value as Region["kind"], L))}>
        {kinds.map((k) => <option key={k}>{k}</option>)}
      </select>
    </dd>
    <RegionFields r={r} onChange={onChange} unit={unit} L={L} />
  </>);
}

function RegionFields({ r, onChange, unit, L }: { r: Region; onChange: (r: Region) => void; unit: string; L: number[] }) {
  if (r.kind === "face") {
    return (<><dt>face</dt><dd>
      <select className="picker" aria-label="face" value={r.face} onChange={(e) => onChange({ ...r, face: e.target.value as Face })}>
        {FACES.map((f) => <option key={f}>{f}</option>)}
      </select>
    </dd></>);
  }
  if (r.kind === "patch") {
    const n = inPlane(r);
    return (<>
      <dt>face</dt><dd>
        <select className="picker" aria-label="face" value={r.face} onChange={(e) => onChange({ ...r, face: e.target.value as Face })}>
          {FACES.map((f) => <option key={f}>{f}</option>)}
        </select>
        <select className="picker" aria-label="patch shape" value={r.shape ?? "rect"} onChange={(e) => onChange({ ...r, shape: e.target.value as "rect" | "ellipse" })}>
          <option>rect</option><option>ellipse</option>
        </select>
      </dd>
      <VecEdit label="centre" v={r.center} names={n} unit={unit} onChange={(c) => onChange({ ...r, center: [c[0], c[1]] })} />
      <VecEdit label="size" v={r.size} names={n} unit={unit} onChange={(c) => onChange({ ...r, size: [c[0], c[1]] })} />
    </>);
  }
  if (r.kind === "box") {
    return (<>
      <VecEdit label="min" v={r.min} unit={unit} onChange={(c) => onChange({ ...r, min: [c[0], c[1], c[2]] })} />
      <VecEdit label="max" v={r.max} unit={unit} onChange={(c) => onChange({ ...r, max: [c[0], c[1], c[2]] })} />
    </>);
  }
  if (r.kind === "cylinder") {
    return (<>
      <dt>axis</dt><dd><AxisPick v={r.axis} onChange={(axis) => onChange({ ...r, axis })} /></dd>
      <VecEdit label="centre" v={r.center} names={["a", "b"]} unit={unit} onChange={(c) => onChange({ ...r, center: [c[0], c[1]] })} />
      <Num label="radius" value={r.radius} unit={unit} onChange={(v) => onChange({ ...r, radius: v })} />
      <VecEdit label="span" v={r.span} names={["from", "to"]} unit={unit} onChange={(c) => onChange({ ...r, span: [c[0], c[1]] })} />
    </>);
  }
  return (<>
    <dt>edge along</dt><dd><AxisPick v={r.axis} onChange={(axis) => onChange({ ...r, axis })} /></dd>
    <VecEdit label="at" v={r.at} names={["a", "b"]} unit={unit} onChange={(c) => onChange({ ...r, at: [c[0], c[1]] })} />
    {r.span
      ? <><VecEdit label="span" v={r.span} names={["from", "to"]} unit={unit} onChange={(c) => onChange({ ...r, span: [c[0], c[1]] })} />
          <dt /><dd><button className="linkbtn" onClick={() => onChange({ ...r, span: null })}>whole edge</button></dd></>
      : <><dt>span</dt><dd>whole edge <button className="linkbtn" onClick={() => onChange({ ...r, span: [0, L[IDX[r.axis]]] })}>limit</button></dd></>}
  </>);
}

function AxisPick({ v, onChange }: { v: Axis; onChange: (a: Axis) => void }) {
  return (<select className="picker" aria-label="axis" value={v} onChange={(e) => onChange(e.target.value as Axis)}>
    {AXES.map((a) => <option key={a}>{a}</option>)}
  </select>);
}

export function Inspector() {
  const spec = useStore((s) => s.spec);
  const sel = useStore((s) => s.selection);
  const editSpec = useStore((s) => s.editSpec);
  const select = useStore((s) => s.select);
  if (!spec || !sel) {
    return (
      <div className="inspector" aria-label="Inspector">
        <div className="section-title">Inspector</div>
        <div className="kv" style={{ color: "var(--text-3)", display: "block" }}>Select an item in the tree or the viewport. Press S, L, K or O to add a support, a load, a keep-in or a keep-out.</div>
      </div>
    );
  }
  const u = spec.units;
  const body = (() => {
    if (sel.kind === "domain") {
      return (<>
        <dt>units</dt><dd title="Relabels the numbers; it does not convert them">
          <select className="picker" aria-label="length unit" value={u.length} onChange={(e) => editSpec((s) => ({ ...s, units: { ...s.units, length: e.target.value } }))}>
            {["mm", "m", "in"].map((x) => <option key={x}>{x}</option>)}
          </select>
          <select className="picker" aria-label="force unit" value={u.force} onChange={(e) => editSpec((s) => ({ ...s, units: { ...s.units, force: e.target.value } }))}>
            {["N", "kN", "lbf"].map((x) => <option key={x}>{x}</option>)}
          </select></dd>
        <VecEdit label="size" v={spec.domain.size} unit={u.length} onChange={(c) => editSpec((s) => ({ ...s, domain: { ...s.domain, size: [c[0], c[1], c[2]] } }))} />
        <Num label="volume %" value={+(spec.volume_fraction * 100).toFixed(2)} onChange={(v) => editSpec((s) => ({ ...s, volume_fraction: v / 100 }))} />
        <dt>mesh by</dt><dd>
          <select className="picker" aria-label="mesh by" value={spec.resolution.h != null ? "h" : "cells"}
            onChange={(e) => editSpec((s) => {
              const sh = Math.min(...s.domain.size);
              return { ...s, resolution: e.target.value === "h" ? { h: +(sh / (s.resolution.cells_short_edge ?? 20)).toPrecision(3) }
                : { cells_short_edge: Math.max(2, Math.round(sh / (s.resolution.h ?? sh / 20))) } };
            })}>
            <option value="h">h</option><option value="cells">cells</option>
          </select></dd>
        {spec.resolution.h != null
          ? <Num label="mesh h"value={spec.resolution.h} unit={u.length} onChange={(v) => editSpec((s) => ({ ...s, resolution: { h: v } }))} />
          : <Num label="cells short" value={spec.resolution.cells_short_edge ?? 0} step={1} onChange={(v) => editSpec((s) => ({ ...s, resolution: { cells_short_edge: Math.round(v) } }))} />}
        <Num label="min feature" value={spec.min_feature ?? 0} unit={u.length} onChange={(v) => editSpec((s) => ({ ...s, min_feature: v > 0 ? v : null }))} />
      </>);
    }
    if (sel.kind === "material") {
      return (<>
        <dt>name</dt><dd><input className="num" aria-label="material name" value={spec.material.name}
          onChange={(e) => editSpec((s) => ({ ...s, material: { ...s.material, name: e.target.value } }))} /></dd>
        <Num label="E" value={spec.material.E} unit={`${u.force}/${u.length}²`} onChange={(v) => editSpec((s) => ({ ...s, material: { ...s.material, E: v } }))} />
        <Num label="ν" value={spec.material.nu} onChange={(v) => editSpec((s) => ({ ...s, material: { ...s.material, nu: v } }))} />
      </>);
    }
    const listKey = sel.kind === "support" ? "supports" : sel.kind === "load" ? "loads" : "passive";
    const list = spec[listKey] as Array<{ id: string; region: Region }>;
    const x = list.find((o) => o.id === sel.id);
    if (!x) return null;
    const update = (fn: (o: any) => any) => editSpec((s) => ({ ...s, [listKey]: (s[listKey] as any[]).map((o) => (o.id === sel.id ? fn(o) : o)) } as Spec), `edit ${sel.id}`);
    return (<>
      <dt>id</dt><dd>{x.id}</dd>
      <RegionEdit r={x.region} unit={u.length} L={spec.domain.size}
        kinds={sel.kind === "passive" ? ["box", "cylinder"] : ["face", "patch", "edge", "box", "cylinder"]}
        onChange={(r) => update((o) => ({ ...o, region: r }))} />
      {sel.kind === "support" && (<><dt>fixed</dt><dd className="checks">
        {(["x", "y", "z"] as Axis[]).map((a) => {
          const fix = (x as unknown as { fix: Axis[] }).fix;
          return (<label key={a}><input type="checkbox" checked={fix.includes(a)}
            onChange={(e) => update((o) => ({ ...o, fix: e.target.checked ? [...new Set([...o.fix, a])].sort() : o.fix.filter((b: Axis) => b !== a) }))} /> {a}</label>);
        })}
      </dd></>)}
      {sel.kind === "load" && <VecEdit label="force" v={(x as unknown as { force: number[] }).force} unit={u.force}
        onChange={(c) => update((o) => ({ ...o, force: [c[0], c[1], c[2]] }))} />}
      {sel.kind === "passive" && <><dt>type</dt><dd>
        <select className="picker" aria-label="passive type" value={(x as unknown as { kind: string }).kind}
          onChange={(e) => update((o) => ({ ...o, kind: e.target.value }))}>
          <option value="solid">keep-in</option><option value="void">keep-out</option>
        </select></dd></>}
      <dt />
      <dd><button className="linkbtn danger" onClick={() => {
        editSpec((s) => ({ ...s, [listKey]: (s[listKey] as any[]).filter((o) => o.id !== sel.id) } as Spec), `delete ${sel.id}`);
        select(null);
      }}>Delete {x.id}</button></dd>
    </>);
  })();
  return (
    <div className="inspector" aria-label="Inspector">
      <div className="section-title">Inspector</div>
      <dl className="kv">{body}</dl>
    </div>
  );
}

const NEEDS: Record<string, string> = {
  supports: "No support yet: press S and click the face to fix.",
  loads: "No load yet: press L and click the face to load.",
};

/** A pydantic schema error in words, with what to do for the common ones. */
function schemaMessage(e: unknown): string {
  const x = e as { loc?: (string | number)[]; msg?: string; type?: string };
  const loc = x.loc ?? [];
  if (x.type === "too_short" && loc.length === 1 && NEEDS[String(loc[0])]) return NEEDS[String(loc[0])];
  return `${loc.join(" › ") || "spec"}: ${x.msg ?? JSON.stringify(e)}`;
}

export function Checks() {
  const report = useStore((s) => s.report);
  const fs = report?.findings ?? [];
  if (!report) return null;
  return (
    <>
      <div className="section-title">Checks</div>
      <div className="findings">
        {(report.schema_errors ?? []).map((e, i) => (
          <div key={`s${i}`} className="finding error"><div className="code">schema</div>{schemaMessage(e)}</div>))}
        {fs.length === 0 && !report.schema_errors && <div style={{ color: "var(--text-3)" }}>All rules pass.</div>}
        {fs.map((f, i) => (
          <div key={i} className={`finding ${f.severity}`}>
            <div className="code">{f.code} · {f.severity}</div>
            {f.message}
          </div>
        ))}
      </div>
    </>
  );
}

// ---- result card: the latest finished design, its certificate and its exports ----------------------------

const CHECK_NAMES: Record<string, string> = {
  converged: "converged", state_solve_accepted: "FP64 state solve accepted", no_vram_spill: "no VRAM spill",
  spill_monitored: "spill monitored", volume_ok: "volume within 0.1%", passive_exact: "keep-in/out exact",
};

export function ResultCard() {
  const log = useStore((s) => s.log);
  const surface = useStore((s) => s.surface);
  const exports = useStore((s) => s.exports);
  const u = useStore((s) => s.spec?.units);
  const set = useStore((s) => s.set);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState(true);
  const last = [...log].reverse().find((r) => r.type === "result" && !r.result?.preview &&
    (r.result?.status === "converged" || r.result?.status === "not_converged"));
  const current = [...log].reverse().find((x) => x.type === "call" && x.ok)?.spec_hash;
  if (!last?.result) return null;
  const r = last.result;
  const stale = !!r.spec_hash && !!current && r.spec_hash !== current;
  const ex = exports[r.job] ?? {};
  const shown = surface?.job === r.job;
  const checks = Object.entries((r.checks ?? {}) as Record<string, boolean>);
  return (
    <div className="result" aria-label="Latest result">
      <button className="section-title linkbtn result-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} Result</button>
      <div className="result-head">
        {r.certified ? <span className="seal">certified</span> : <span className="badge warn">not certified</span>}
        {stale && <span className="badge err" title="The design was edited after this solve">stale</span>}
        <span>{sizeLabel(r.n_elem)} · {r.iterations} it · {r.compute_s?.toFixed(1)} s{r.warm_used ? " · warm" : ""}</span>
      </div>
      {open && <>
      <dl className="kv">
        <dt>compliance</dt><dd>{fmt(r.compliance_phys)} {u ? `${u.force}·${u.length}` : ""}</dd>
        <dt>volume</dt><dd>{(100 * r.volume).toFixed(2)}%</dd>
        <dt>grid</dt><dd>{(r.grid ?? []).join("×")}</dd>
      </dl>
      <ul className="checklist">
        {checks.map(([k, v]) => <li key={k} className={v ? "pass" : "fail"}>{v ? "✓" : "✗"} {CHECK_NAMES[k] ?? k}</li>)}
      </ul>
      {!r.release_rung && <div className="note">Only a certified Release-rung design is a released design.</div>}
      <div className="result-actions">
        <button className="iconbtn" onClick={() => (shown ? set({ surface: null })
          : showSurface(r.job).then(() => setErr(null), (e) => setErr(String(e))))}>{shown ? "Voxels" : "Surface"}</button>
        <button className="iconbtn" disabled={!!ex.busy} onClick={() => void exportDesign(r.job)}>
          {ex.busy ? `Writing ${ex.busy}…` : "Export STL + report"}</button>
      </div>
      {shown && surface && <div className="note">surface: {surface.triangles.toLocaleString()} triangles (display pooled ×{surface.pool})</div>}
      {ex.stl && <div className="files"><a href={`/api/files/${ex.stl}`} download>{ex.stl}</a> · {ex.triangles?.toLocaleString()} tri · {ex.mb?.toFixed(1)} MB</div>}
      {ex.report && <div className="files"><a href={`/api/files/${ex.markdown}`} download>{ex.markdown}</a> · <a href={`/api/files/${ex.report}`} download>json</a></div>}
      {err && <div className="note err">{err}</div>}
      </>}
    </div>
  );
}

function Elapsed({ t0 }: { t0: number }) {
  const [, tick] = useState(0);
  useEffect(() => { const t = window.setInterval(() => tick((n) => n + 1), 500); return () => window.clearInterval(t); }, []);
  return <>{((Date.now() - t0) / 1000).toFixed(0)} s</>;
}

function Steps({ steps, open }: { steps: AgentStep[]; open: boolean }) {
  const [show, setShow] = useState(open);
  if (!steps.length) return null;
  const failed = steps.filter((x) => x.ok === false).length;
  return (
    <div className="steps">
      <button className="linkbtn" onClick={() => setShow(!show)} aria-expanded={show}>
        {show ? "▾" : "▸"} {steps.length} tool call{steps.length > 1 ? "s" : ""}{failed ? ` · ${failed} failed` : ""}
      </button>
      {show && <ol>{steps.map((x) => (
        <li key={x.id} className={x.ok === false ? "fail" : x.ok ? "pass" : "run"} title={x.summary ?? ""}>
          <code>{x.tool}</code> {x.args && Object.keys(x.args as object).length ? <span className="args">{JSON.stringify(x.args).slice(0, 90)}</span> : null}
          {x.ok === false && x.summary && <details><summary>error</summary><div className="err">{x.summary}</div></details>}
        </li>))}</ol>}
    </div>
  );
}

function Cards({ run }: { run: AgentRunState }) {
  const set = useStore((s) => s.set);
  const open = run.cards.filter((c) => !run.decided[c.id]);
  if (!run.cards.length) return run.running || run.mode === "act" ? null : <div className="note">No changes proposed.</div>;
  return (
    <div className="cards" aria-label="Proposed changes">
      {run.cards.map((c) => {
        const d = run.decided[c.id];
        return (
          <div key={c.id} className={`card ${c.op} ${d ?? ""}`} onMouseEnter={() => set({ ghostCard: c.id })}
            onMouseLeave={() => set({ ghostCard: null })}>
            <span className="op">{c.op}</span><span className="text">{c.text.replace(/^(add|change|remove|set) /, "")}</span>
            {d ? <span className="decided">{d}</span> : (<>
              <button className="iconbtn" onClick={() => acceptCards(run.id, [c.id])}>Accept</button>
              <button className="linkbtn" onClick={() => rejectCards([c.id])}>Reject</button>
            </>)}
          </div>
        );
      })}
      {open.length > 1 && <div className="card-actions">
        <button className="iconbtn" onClick={() => acceptCards(run.id, open.map((c) => c.id))}>Accept all {open.length}</button>
        <button className="linkbtn" onClick={() => rejectCards(open.map((c) => c.id))}>Reject all</button>
      </div>}
    </div>
  );
}

/** One clarification question: a chip per reading, each with the sketch-rung design it leads to. */
function ClarifyChips() {
  const q = useStore((s) => s.clarify);
  const u = useStore((s) => s.spec?.units);
  const [comment, setComment] = useState("");
  const box = useRef<HTMLDivElement>(null);
  // a new question can open below the fold of a crowded panel: bring it into view (heuristic review 2)
  useEffect(() => { if (q) box.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }); }, [q?.qid]);
  if (!q) return null;
  return (
    <div className="clarify" role="group" aria-label="Clarification question" ref={box}>
      <div className="q">{q.question}</div>
      <div className="chips">
        {q.options.map((o, i) => {
          const p = q.previews?.[i];
          return (
            <button key={i} className="clar-chip" onClick={() => answerClarify(q.qid, i, o.label)}>
              {p?.png_b64 ? <img alt="" src={`data:image/png;base64,${p.png_b64}`} /> : <span className="noimg">{p?.error ?? "no preview"}</span>}
              <span className="lab">{o.label}</span>
              {p?.compliance_phys != null && <span className="c">C {fmt(p.compliance_phys)} {u ? `${u.force}·${u.length}` : ""}</span>}
            </button>
          );
        })}
      </div>
      <div className="none">
        <input placeholder="None of these — say what you mean" value={comment} onChange={(e) => setComment(e.target.value)} aria-label="Answer in words" />
        <button className="linkbtn" onClick={() => answerClarify(q.qid, null, undefined, comment || "None of these.")}>Send</button>
      </div>
      {q.previews_s != null && <div className="note">previews: {q.options.length} sketches in {q.previews_s.toFixed(1)} s</div>}
    </div>
  );
}

export function Assistant() {
  const brief = useStore((s) => s.brief);
  const chat = useStore((s) => s.chat);
  const run = useStore((s) => s.agentRun);
  const mode = useStore((s) => s.agentMode);
  const sel = useStore((s) => s.selection);
  const set = useStore((s) => s.set);
  const [text, setText] = useState("");
  const chatRef = useRef<HTMLDivElement>(null);
  useEffect(() => { chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight }); }, [chat, run]);
  const busy = !!run?.running;
  const submit = () => {
    const t = text.trim();
    if (!t || busy) return;
    sendAgent(t);
    setText("");
  };
  const lastRunMsg = run ? chat.findIndex((m) => m.role === "assistant" && m.run === run.id) : -1;
  const before = lastRunMsg >= 0 ? chat.slice(0, lastRunMsg) : chat;
  const after = lastRunMsg >= 0 ? chat.slice(lastRunMsg) : [];
  const msg = (m: ChatMsg, i: number) => <div key={i} className={`msg ${m.role}`}>{m.text}</div>;
  return (
    <div className="assistant-panel" aria-label="Assistant">
      <div className="section-title assistant-head">Assistant
        <span className="dial" role="radiogroup" aria-label="Autonomy">
          {(["propose", "act"] as const).map((m) => (
            <button key={m} role="radio" aria-checked={mode === m} disabled={busy} onClick={() => set({ agentMode: m })}
              title={m === "propose" ? "The assistant proposes changes as cards; you accept them" :
                "The assistant edits the design directly (undoable) and may run sketch/preview solves"}>
              {m === "propose" ? "Propose" : "Act"}</button>))}
        </span>
      </div>
      <div className="chat" ref={chatRef}>
        {brief && <div className="brief">{brief.trim().replace(/([^\n])\n(?!\n)/g, "$1 ")}</div>}
        {chat.length === 0 && <div className="note">Describe a change, e.g. “add a 2 kN downward load on the top face”. In Propose mode
          the changes come back as cards to accept; in Act mode they apply at once (Ctrl+Z undoes).</div>}
        {before.map(msg)}
        {run && (
          <div className="run">
            {run.running && <div className="working"><span className="dot run" /> working · <Elapsed t0={run.t0} />
              <button className="linkbtn" onClick={stopAgent}>Stop</button></div>}
            {run.messages.length > 1 && run.running && <div className="msg assistant interim">{run.messages[run.messages.length - 1]}</div>}
            <Steps steps={run.steps} open={run.running} />
            <ClarifyChips />
            <Cards run={run} />
            {run.error && <div className="note err">{run.error.slice(0, 300)}</div>}
            {!run.running && run.mode === "act" && run.steps.some((x) => /^(set_|add_|update_|remove_|new_)/.test(x.tool)) &&
              <button className="linkbtn" onClick={() => revertAgent(run.id)}>Undo this run's edits</button>}
            {!run.running && run.timing && <div className="note">{run.timing.wall_s?.toFixed(1)} s
              {run.usage?.input_tokens ? ` · ${(run.usage.input_tokens / 1000).toFixed(0)}k in (${((run.usage.cached_input_tokens ?? 0) / 1000).toFixed(0)}k cached) · ${run.usage.output_tokens} out` : ""}</div>}
          </div>
        )}
        {!run && <ClarifyChips />}{/* a question from an attached or stand-alone session, outside a GUI agent run */}
        {after.map(msg)}
      </div>
      <div className="composer">
        {sel && <div className="composer-ctx"><span className="chip" title="The assistant sees this selection">{sel.kind}{"id" in sel ? ` ${sel.id}` : ""}
          <button className="linkbtn" aria-label="Clear selection" onClick={() => useStore.getState().select(null)}>×</button></span></div>}
        <textarea rows={2} placeholder={busy ? "Working…" : mode === "propose" ? "Ask for a change (proposed as cards)…" : "Ask for a change (applied directly)…"}
          aria-label="Message the assistant" value={text} disabled={busy}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} />
        <button className="iconbtn" disabled={busy || !text.trim()} onClick={submit}>Send</button>
      </div>
    </div>
  );
}

// ---- resolution ladder -------------------------------------------------------------------------------------

const RUNG_NAMES = ["Sketch", "Preview", "Refine", "Release"];

function Spark({ v }: { v: number[] }) {
  if (v.length < 2) return null;
  const lo = Math.min(...v), hi = Math.max(...v);
  const pts = v.map((y, i) => `${(i / (v.length - 1)) * 80},${18 - ((y - lo) / (hi - lo || 1)) * 16}`).join(" ");
  return <svg className="spark" width="80" height="20" aria-hidden="true"><polyline points={pts} fill="none" stroke="currentColor" strokeWidth="1.2" /></svg>;
}

function statusDot(j?: JobState) {
  if (!j) return "";
  if (j.status === "running" || j.status === "queued") return "run";
  if (j.status === "converged") return j.certified ? "ok" : "done";
  if (j.status === "cancelled" || j.status === "skipped") return "";
  return "err";
}

export function Ladder() {
  const rungs = useStore((s) => s.rungs);
  const jobs = useStore((s) => s.jobs);
  const rungJob = useStore((s) => s.rungJob);
  const report = useStore((s) => s.report);
  const worker = useStore((s) => s.worker);
  const u = useStore((s) => s.spec?.units);
  const shown = rungs.slice(-4);
  const names = RUNG_NAMES.slice(4 - shown.length);
  const running = Object.values(rungJob).map((id) => jobs[id]).find((j) => j && (j.status === "running" || j.status === "queued"));
  const active = running ?? Object.values(rungJob).map((id) => jobs[id]).filter(Boolean).slice(-1)[0];
  const blocked = report ? !report.ok : true;
  return (
    <div className="bottom">
      <div className="ladder" aria-label="Resolution ladder">
        {shown.map((r, i) => {
          const j = jobs[rungJob[String(r.h)]];
          return (
            <button key={r.h} className="rung" disabled={blocked || worker === "dead"} onClick={() => solveRung(r.h)}
              title={`${r.grid.join("×")} elements, h = ${r.h} ${u?.length ?? ""}${blocked ? " — fix the errors first" : ""}`}>
              <span className={`dot ${statusDot(j)}`} />{names[i]}<span className="n">{sizeLabel(r.n_elem)}</span>
              {j && j.status === "queued" && <span className="n">queued</span>}
              {j && j.status === "running" && <span className="n">{j.it ? `it ${j.it}` : `setting up ${sizeLabel(r.n_elem)}…`}</span>}
              {j && j.status === "converged" && <span className="n">{j.compute_s?.toFixed(1)}s{j.warm ? " ↑" : ""}</span>}
              {j?.certified && <span className="seal" title="certified">certified</span>}
            </button>
          );
        })}
        {shown.length >= 2 && (() => {
          const rel = shown[shown.length - 1], below = jobs[rungJob[String(shown[shown.length - 2].h)]];
          return below?.status === "converged" && (
            <button className="iconbtn" disabled={blocked || worker === "dead" || !!running} onClick={() => solveRung(rel.h, true)}
              title="Release with a re-opened warm start: about 60 iterations, compliance as good as a cold run (X1)">Optimize release</button>
          );
        })()}
        {running && <button className="iconbtn" onClick={cancelSolve}>Stop</button>}
        {shown.length > 0 && !Object.keys(rungJob).length &&
          <span className="note" title="Each rung starts from the design of the rung below; the finest is the release">
            Release = the finest rung; only a certified release is a finished design.</span>}
        {active && (
          <span className="pill">
            <Spark v={active.history} />
            {active.c_phys != null && <>C = {fmt(active.c_phys)} {u ? `${u.force}·${u.length}` : ""}</>}
            {active.vol != null && <> · V = {(100 * active.vol).toFixed(1)}%</>}
            {active.message && <> · {active.message}</>}
          </span>
        )}
        <span className="spacer" />
        <span className="pill" title="GPU worker"><span className={`dot ${worker === "ready" ? "ok" : worker === "dead" ? "err" : ""}`} />GPU {worker}</span>
      </div>
      <Timeline />
    </div>
  );
}

// ---- session timeline: the provenance log, newest right ---------------------------------------------------

function recLabel(r: LogRec): string {
  if (r.type === "result") return `result ${r.result?.status ?? ""}${r.result?.certified ? " · certified" : ""}`;
  if (r.type === "answer") return `answer: ${r.answer?.label ?? (r.answer?.comment || "none")}`;
  const lab = r.meta?.label && r.meta.label !== "edit" ? r.meta.label : "";
  if (r.tool === "solve" && r.result?.h) return `solve h=${r.result.h}`;
  const id = (r.args as any)?.id;
  return lab || `${r.tool}${id ? ` ${id}` : ""}`;
}

export function Timeline() {
  const log = useStore((s) => s.log);
  const shown = log.filter((r) => r.type !== "session").slice(-14);
  return (
    <div className="timeline" aria-label="Session timeline">
      <span className="tl-head">Timeline</span>
      {shown.length === 0 && <span className="tl-empty">every edit, agent step and solve is logged here</span>}
      <div className="tl-items">{/* right-aligned: when the row overflows, the oldest records are the ones cut */}
      {shown.map((r, i) => (
        <span key={`${r.seq}-${r.type}-${i}`} className={`tl-item ${r.actor ?? "system"}${r.ok === false ? " fail" : ""}`}
          title={`#${r.seq} ${r.actor ?? ""} ${r.tool ?? r.type} ${r.t ?? ""}${r.error ? `
${r.error}` : ""}`}>
          <i>{r.type === "result" ? "gpu" : r.type === "answer" ? (r.answer?.by ?? "user") : r.actor}</i>{recLabel(r)}
        </span>
      ))}
      </div>
    </div>
  );
}
