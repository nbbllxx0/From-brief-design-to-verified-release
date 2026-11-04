// Mirrors agentto.spec (schema agentto.spec/v0). Only what the G0 workspace reads.
export type Vec3 = [number, number, number];
export type Vec2 = [number, number];
export type Face = "x-" | "x+" | "y-" | "y+" | "z-" | "z+";
export type Axis = "x" | "y" | "z";

export type Region =
  | { kind: "face"; face: Face }
  | { kind: "patch"; face: Face; center: Vec2; size: Vec2; shape?: "rect" | "ellipse" }
  | { kind: "edge"; axis: Axis; at: Vec2; span?: Vec2 | null }
  | { kind: "box"; min: Vec3; max: Vec3 }
  | { kind: "cylinder"; axis: Axis; center: Vec2; radius: number; span: Vec2 };

export interface Support { id: string; region: Region; fix: Axis[]; note?: string }
export interface Load { id: string; region: Region; force: Vec3; note?: string }
export interface Passive { id: string; kind: "solid" | "void"; region: Region; note?: string }

export interface Spec {
  schema_version: string;
  name: string;
  description?: string;
  units: { length: string; force: string };
  domain: { kind: "box"; size: Vec3 };
  material: { name: string; E: number; nu: number };
  volume_fraction: number;
  supports: Support[];
  loads: Load[];
  passive: Passive[];
  resolution: { h?: number | null; cells_short_edge?: number | null };
  min_feature?: number | null;
}

export interface Finding { code: string; severity: "error" | "warning" | "info"; message: string; ids: string[] }
export interface Report {
  ok: boolean;
  resolution?: { h: number; grid: Vec3; n_elem: number; n_design: number; mg_levels?: number; est_bytes?: number };
  findings?: Finding[];
  schema_errors?: unknown[];
}

export type Selection = { kind: "domain" | "material" | "support" | "load" | "passive"; id?: string } | null;
