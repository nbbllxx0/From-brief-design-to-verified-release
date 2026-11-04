import { useEffect, useMemo } from "react";
import { Canvas, useThree, type ThreeEvent } from "@react-three/fiber";
import { Edges, GizmoHelper, GizmoViewcube, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { useStore, type Tool } from "../store";
import type { Face, Region, Spec, Vec3 } from "../types";

const AX = { x: 0, y: 1, z: 2 } as const;
const others = (a: number) => [0, 1, 2].filter((i) => i !== a) as [number, number];

function cssColor(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

/** Axis-aligned extent [min, max] of a region in spec coordinates (thin regions get zero thickness). */
function extent(r: Region, L: Vec3): { lo: Vec3; hi: Vec3 } {
  const lo: Vec3 = [0, 0, 0], hi: Vec3 = [L[0], L[1], L[2]];
  if (r.kind === "face" || r.kind === "patch") {
    const a = AX[r.face[0] as "x" | "y" | "z"];
    lo[a] = hi[a] = r.face[1] === "-" ? 0 : L[a];
    if (r.kind === "patch") {
      const [b, c] = others(a);
      [[b, 0], [c, 1]].forEach(([ax, k]) => {
        lo[ax] = Math.max(0, r.center[k] - r.size[k] / 2);
        hi[ax] = Math.min(L[ax], r.center[k] + r.size[k] / 2);
      });
    }
  } else if (r.kind === "edge") {
    const a = AX[r.axis];
    const [b, c] = others(a);
    lo[b] = hi[b] = r.at[0];
    lo[c] = hi[c] = r.at[1];
    if (r.span) { lo[a] = r.span[0]; hi[a] = r.span[1]; }
  } else if (r.kind === "box") {
    return { lo: [...r.min] as Vec3, hi: [...r.max] as Vec3 };
  } else {
    const a = AX[r.axis];
    const [b, c] = others(a);
    lo[a] = r.span[0]; hi[a] = r.span[1];
    lo[b] = r.center[0] - r.radius; hi[b] = r.center[0] + r.radius;
    lo[c] = r.center[1] - r.radius; hi[c] = r.center[1] + r.radius;
  }
  return { lo, hi };
}

function faceNormal(r: Region): Vec3 | null {
  if (r.kind !== "face" && r.kind !== "patch") return null;
  const n: Vec3 = [0, 0, 0];
  n[AX[r.face[0] as "x" | "y" | "z"]] = r.face[1] === "-" ? -1 : 1;
  return n;
}

interface Frame { s: number; L: Vec3; w: (p: Vec3) => THREE.Vector3 }

/** Spec -> scene transform. `also` (an open proposal's domain size) widens the framed box from the same min corner,
 *  so a proposed bigger box stays in view. */
function frameFor(L: Vec3, also?: Vec3 | null): Frame {
  const U = also ? (L.map((v, i) => Math.max(v, also[i])) as Vec3) : L;
  const s = 2.4 / Math.max(...U);
  return { s, L, w: (p: Vec3) => new THREE.Vector3((p[0] - U[0] / 2) * s, (p[1] - U[1] / 2) * s, (p[2] - U[2] / 2) * s) };
}

function useFrame(spec: Spec): Frame {
  const also = useStore((s) => {
    const r = s.agentRun;
    return r?.proposal?.domain?.size && r.cards.some((c) => !r.decided[c.id]) ? r.proposal.domain.size : null;
  });
  return useMemo(() => frameFor(spec.domain.size, also), [spec, also]);
}

function RegionSolid({ r, f, color, opacity, selected }: { r: Region; f: Frame; color: string; opacity: number; selected: boolean }) {
  const { lo, hi } = extent(r, f.L);
  const t = 0.004 / f.s;                       // minimum visible thickness, spec units
  const size = [0, 1, 2].map((i) => Math.max(hi[i] - lo[i], t) * f.s) as Vec3;
  const c = f.w([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]);
  const mat = <meshStandardMaterial color={color} transparent opacity={opacity} depthWrite={false} />;
  if (r.kind === "cylinder") {
    const rot: Vec3 = r.axis === "x" ? [0, 0, Math.PI / 2] : r.axis === "z" ? [Math.PI / 2, 0, 0] : [0, 0, 0];
    return (
      <mesh position={c} rotation={rot}>
        <cylinderGeometry args={[r.radius * f.s, r.radius * f.s, (r.span[1] - r.span[0]) * f.s, 40]} />
        {mat}
        {selected && <Edges color={color} />}
      </mesh>
    );
  }
  return (
    <mesh position={c}>
      <boxGeometry args={size} />
      {mat}
      {selected && <Edges color={color} />}
    </mesh>
  );
}

function SupportGlyphs({ r, f, color }: { r: Region; f: Frame; color: string }) {
  const { lo, hi } = extent(r, f.L);
  const n = faceNormal(r);
  const pts: Vec3[] = [];
  const steps = r.kind === "edge" ? [7, 1, 1] : [4, 4, 4];
  const lin = (a: number, b: number, k: number) => (k === 1 ? [(a + b) / 2] : Array.from({ length: k }, (_, i) => a + ((b - a) * i) / (k - 1)));
  for (const x of lo[0] === hi[0] ? [lo[0]] : lin(lo[0], hi[0], steps[0]))
    for (const y of lo[1] === hi[1] ? [lo[1]] : lin(lo[1], hi[1], r.kind === "edge" ? 1 : steps[1]))
      for (const z of lo[2] === hi[2] ? [lo[2]] : lin(lo[2], hi[2], r.kind === "edge" ? 1 : steps[2]))
        pts.push([x, y, z]);
  if (r.kind === "edge") {
    const a = AX[r.axis];
    pts.length = 0;
    lin(lo[a], hi[a], 7).forEach((v) => { const p: Vec3 = [lo[0], lo[1], lo[2]]; p[a] = v; pts.push(p); });
  }
  const dir = new THREE.Vector3(...(n ?? [0, -1, 0])).negate();     // cones point into the support surface
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir);
  const h = 0.07;
  return (
    <>
      {pts.map((p, i) => {
        const pos = f.w(p).add(dir.clone().multiplyScalar(-h / 2));
        return (
          <mesh key={i} position={pos} quaternion={q}>
            <coneGeometry args={[0.028, h, 16]} />
            <meshStandardMaterial color={color} />
          </mesh>
        );
      })}
    </>
  );
}

function LoadArrow({ r, force, f, color }: { r: Region; force: Vec3; f: Frame; color: string }) {
  const arrow = useMemo(() => {
    const { lo, hi } = extent(r, f.L);
    const tip = f.w([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]);
    const d = new THREE.Vector3(...force);
    if (d.lengthSq() === 0) return null;
    d.normalize();
    const len = 0.45;
    return new THREE.ArrowHelper(d, tip.sub(d.clone().multiplyScalar(len)), len, new THREE.Color(color), 0.1, 0.06);
  }, [r, f, force, color]);
  return arrow ? <primitive object={arrow} /> : null;
}

const FACE_OF = (n: THREE.Vector3): Face => {
  const a = Math.abs(n.x) > 0.5 ? "x" : Math.abs(n.y) > 0.5 ? "y" : "z";
  const v = a === "x" ? n.x : a === "y" ? n.y : n.z;
  return `${a}${v > 0 ? "+" : "-"}` as Face;
};

function nextId(spec: Spec, base: string) {
  const ids = new Set([...spec.supports, ...spec.loads, ...spec.passive].map((x) => x.id));
  let i = 1;
  while (ids.has(`${base}_${i}`)) i++;
  return `${base}_${i}`;
}

/** Released or streamed density as instanced voxels: only voxels above 0.5 with an exposed face are drawn. */
function Density({ f }: { f: Frame }) {
  const frame = useStore((s) => s.frame);
  const theme = useStore((s) => s.theme);
  const mesh = useMemo(() => {
    if (!frame) return null;
    const [nx, ny, nz] = frame.shape;
    const d = frame.data;
    const at = (i: number, j: number, k: number) =>
      i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz ? 0 : d[(i * ny + j) * nz + k];
    const pts: number[] = [];
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
      if (at(i, j, k) < 128) continue;
      if (at(i - 1, j, k) >= 128 && at(i + 1, j, k) >= 128 && at(i, j - 1, k) >= 128 && at(i, j + 1, k) >= 128 &&
          at(i, j, k - 1) >= 128 && at(i, j, k + 1) >= 128) continue;
      pts.push(i, j, k);
    }
    const n = pts.length / 3;
    const cell = [f.L[0] / nx, f.L[1] / ny, f.L[2] / nz];
    const geo = new THREE.BoxGeometry(cell[0] * f.s, cell[1] * f.s, cell[2] * f.s);
    const mat = new THREE.MeshStandardMaterial({ color: theme === "dark" ? "#cfc8bb" : "#b9b2a4", roughness: 0.6 });
    const im = new THREE.InstancedMesh(geo, mat, Math.max(n, 1));
    const m = new THREE.Matrix4();
    for (let q = 0; q < n; q++) {
      m.setPosition(f.w([(pts[3 * q] + 0.5) * cell[0], (pts[3 * q + 1] + 0.5) * cell[1], (pts[3 * q + 2] + 0.5) * cell[2]]));
      im.setMatrixAt(q, m);
    }
    im.count = n;
    im.instanceMatrix.needsUpdate = true;
    return im;
  }, [frame, f, theme]);
  return mesh ? <primitive object={mesh} /> : null;
}

/** Released surface (rho = 0.5 isosurface from the backend), drawn in spec units under the viewport transform. */
function SurfaceMesh({ f }: { f: Frame }) {
  const surface = useStore((s) => s.surface);
  const theme = useStore((s) => s.theme);
  const geo = useMemo(() => {
    if (!surface) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(surface.pos, 3));
    g.setIndex(new THREE.BufferAttribute(surface.idx, 1));
    g.computeVertexNormals();
    return g;
  }, [surface]);
  useEffect(() => () => geo?.dispose(), [geo]);
  if (!geo) return null;
  const o = f.w([0, 0, 0]);
  return (
    <mesh geometry={geo} position={o} scale={f.s}>
      <meshStandardMaterial color={theme === "dark" ? "#d6cfc2" : "#b3ab9c"} roughness={0.45} metalness={0.05} side={THREE.DoubleSide} />
    </mesh>
  );
}

/** Ghost of the assistant's proposal: the open cards' items in the accent colour (removed items in red);
 *  the card under the pointer is drawn stronger. */
function Ghost({ f: cur }: { f: Frame }) {
  const run = useStore((s) => s.agentRun);
  const hover = useStore((s) => s.ghostCard);
  if (!run?.cards.length) return null;
  const accent = cssColor("--accent"), err = cssColor("--error");
  // a proposal that resizes the box: its items sit on the proposed box, drawn (outlined) at the current scale from
  // the same min corner, so the change in size shows
  const L = run.proposal?.domain?.size ?? cur.L;
  const f: Frame = { ...cur, L };
  const resized = L.some((v, i) => v !== cur.L[i]);
  return (
    <>
      {resized && (
        <mesh position={cur.w([L[0] / 2, L[1] / 2, L[2] / 2])}>
          <boxGeometry args={[L[0] * cur.s, L[1] * cur.s, L[2] * cur.s]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
          <Edges color={accent} />
        </mesh>
      )}
      {run.cards.filter((c) => c.list && !run.decided[c.id] && (c.after?.region || c.before?.region)).map((c) => {
        const item = c.op === "remove" ? c.before : c.after;
        return (
          <RegionSolid key={c.id} r={item.region} f={f} color={c.op === "remove" ? err : accent}
            opacity={hover === c.id ? 0.6 : 0.25} selected={hover === c.id} />
        );
      })}
    </>
  );
}

function Scene({ spec }: { spec: Spec }) {
  const f = useFrame(spec);
  const theme = useStore((s) => s.theme);
  const sel = useStore((s) => s.selection);
  const tool = useStore((s) => s.tool);
  const hasSurface = useStore((s) => s.surface !== null);
  const showDensity = useStore((s) => s.frame !== null) || hasSurface;
  const { select, editSpec, setTool } = useStore.getState();
  const col = useMemo(() => ({
    support: cssColor("--support"), load: cssColor("--load"), keepin: cssColor("--keepin"),
    keepout: cssColor("--keepout"), line: cssColor("--line-strong"), text: cssColor("--text-2"),
  }), [theme]);
  const L = spec.domain.size;

  const onDomainClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    if (!e.face) return;
    const face = FACE_OF(e.face.normal);
    if (tool === "select") return select({ kind: "domain" });
    const p = e.point.clone().divideScalar(f.s);
    const pt: Vec3 = [p.x + L[0] / 2, p.y + L[1] / 2, p.z + L[2] / 2];
    const a = AX[face[0] as "x" | "y" | "z"];
    const [b, c] = others(a);
    const round = (v: number) => +v.toPrecision(4);
    if (tool === "keepin" || tool === "keepout") {
      // a slab on the clicked face, a tenth of the domain deep; its box is then edited in the inspector
      const id = nextId(spec, tool === "keepin" ? "keep_in" : "keep_out");
      const min: Vec3 = [0, 0, 0], max: Vec3 = [L[0], L[1], L[2]];
      if (face[1] === "+") min[a] = round(L[a] * 0.9); else max[a] = round(L[a] * 0.1);
      editSpec((s) => ({ ...s, passive: [...s.passive, { id, kind: tool === "keepin" ? "solid" : "void",
        region: { kind: "box", min, max } }] }), `add ${tool === "keepin" ? "keep-in" : "keep-out"} ${id} on ${face}`);
      select({ kind: "passive", id });
    } else if (tool === "support") {
      const id = nextId(spec, "support");
      editSpec((s) => ({ ...s, supports: [...s.supports, { id, region: { kind: "face", face }, fix: ["x", "y", "z"] }] }), `add support ${id} on ${face}`);
      select({ kind: "support", id });
    } else {
      const id = nextId(spec, "load");
      const size: [number, number] = [round(L[b] / 10), round(L[c] / 10)];
      const n: Vec3 = [0, 0, 0];
      n[a] = face[1] === "+" ? -1000 : 1000;
      editSpec((s) => ({ ...s, loads: [...s.loads, { id, force: n,
        region: { kind: "patch", face, center: [round(pt[b]), round(pt[c])], size, shape: "rect" } }] }), `add load ${id} on ${face}`);
      select({ kind: "load", id });
    }
    setTool("select");
  };
  const pick = (kind: "support" | "load" | "passive", id: string) => (e: ThreeEvent<MouseEvent>) => {
    if (tool !== "select") return;
    e.stopPropagation();
    select({ kind, id });
  };

  return (
    <>
      <ambientLight intensity={0.65} />
      <directionalLight position={[3, 5, 4]} intensity={1.1} />
      <directionalLight position={[-4, -2, -3]} intensity={0.35} />
      <mesh onClick={onDomainClick} position={f.w([L[0] / 2, L[1] / 2, L[2] / 2])}>
        <boxGeometry args={[L[0] * f.s, L[1] * f.s, L[2] * f.s]} />
        <meshBasicMaterial transparent opacity={tool === "select" ? 0 : 0.06} color={cssColor("--accent")} depthWrite={false} />
        <Edges color={sel?.kind === "domain" ? cssColor("--accent") : col.text} />
      </mesh>
      {hasSurface ? <SurfaceMesh f={f} /> : showDensity && <Density f={f} />}
      <Ghost f={f} />
      {spec.supports.map((s) => (
        <group key={s.id} onClick={pick("support", s.id)}>
          <RegionSolid r={s.region} f={f} color={col.support} opacity={0.35} selected={sel?.id === s.id} />
          <SupportGlyphs r={s.region} f={f} color={col.support} />
        </group>
      ))}
      {spec.loads.map((l) => (
        <group key={l.id} onClick={pick("load", l.id)}>
          <RegionSolid r={l.region} f={f} color={col.load} opacity={0.5} selected={sel?.id === l.id} />
          <LoadArrow r={l.region} force={l.force} f={f} color={col.load} />
        </group>
      ))}
      {spec.passive.map((p) => (
        <group key={p.id} onClick={pick("passive", p.id)}>
          <RegionSolid r={p.region} f={f} color={p.kind === "solid" ? col.keepin : col.keepout}
            opacity={showDensity ? 0.18 : p.kind === "solid" ? 0.45 : 0.28} selected={sel?.id === p.id} />
        </group>
      ))}
    </>
  );
}

function Toolbar() {
  const tool = useStore((s) => s.tool);
  const setTool = useStore((s) => s.setTool);
  const hasFrame = useStore((s) => s.frame !== null || s.surface !== null);
  const set = useStore((s) => s.set);
  const tools: [Tool, string, string][] = [["select", "Select", "V"], ["support", "Support", "S"], ["load", "Load", "L"],
    ["keepin", "Keep-in", "K"], ["keepout", "Keep-out", "O"]];
  return (
    <div className="toolbar" role="toolbar" aria-label="Viewport tools">
      {tools.map(([t, label, key]) => (
        <button key={t} aria-pressed={tool === t} title={t === "select" ? `Select (${key})` : `Add a ${label.toLowerCase()}: click a face (${key})`} onClick={() => setTool(t)}>{label}</button>
      ))}
      {hasFrame && <button onClick={() => set({ frame: null, surface: null })} title="Hide the design">Hide design</button>}
      {tool !== "select" && <span className="hint">Click a face of the domain</span>}
    </div>
  );
}

/** With ?test in the URL only: lets scripts/gui_test.py turn the camera and find a spec point on screen, so it can
 *  click the viewport like a user (window.__agentto). */
function TestHook() {
  const { camera, controls, gl } = useThree();
  useEffect(() => {
    if (!new URLSearchParams(window.location.search).has("test")) return;
    const ctl = controls as unknown as { target: THREE.Vector3; update: () => void } | null;
    (window as unknown as Record<string, unknown>).__agentto = {
      store: useStore,
      view: (dir: Vec3) => {
        camera.position.set(...dir).normalize().multiplyScalar(5.74);
        ctl?.target.set(0, 0, 0);
        camera.lookAt(0, 0, 0);
        ctl?.update();
      },
      project: (p: Vec3) => {
        const spec = useStore.getState().spec;
        if (!spec) return null;
        const r0 = useStore.getState().agentRun;
        const also = r0?.proposal?.domain?.size && r0.cards.some((c) => !r0.decided[c.id]) ? r0.proposal.domain.size : null;
        const v = frameFor(spec.domain.size, also).w(p).project(camera);
        const r = gl.domElement.getBoundingClientRect();
        return [r.left + ((v.x + 1) / 2) * r.width, r.top + ((1 - v.y) / 2) * r.height];
      },
    };
  }, [camera, controls, gl]);
  return null;
}

export function Viewport() {
  const spec = useStore((s) => s.spec);
  const setTool = useStore((s) => s.setTool);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (/INPUT|SELECT|TEXTAREA/.test((e.target as HTMLElement)?.tagName ?? "") || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "v" || k === "escape") setTool("select");
      else if (k === "s") setTool("support");
      else if (k === "l") setTool("load");
      else if (k === "k") setTool("keepin");
      else if (k === "o") setTool("keepout");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setTool]);
  return (
    <div className="view" aria-label="3D viewport">
      <Canvas camera={{ position: [3.3, 2.3, 4.1], fov: 38 }} dpr={[1, 2]}>
        {spec && <Scene spec={spec} />}
        <OrbitControls makeDefault enableDamping />
        <TestHook />
        <GizmoHelper alignment="top-right" margin={[64, 64]}>
          <GizmoViewcube />
        </GizmoHelper>
      </Canvas>
      <Toolbar />
      <div className="overlay" aria-hidden="true">
        <span className="legend-item"><span className="swatch" style={{ background: "var(--support)" }} />support</span>
        <span className="legend-item"><span className="swatch" style={{ background: "var(--load)" }} />load</span>
        <span className="legend-item"><span className="swatch" style={{ background: "var(--keepin)" }} />keep-in</span>
        <span className="legend-item"><span className="swatch" style={{ background: "var(--keepout)" }} />keep-out</span>
        {spec && <span>{spec.domain.size.join(" × ")} {spec.units.length}</span>}
      </div>
    </div>
  );
}
