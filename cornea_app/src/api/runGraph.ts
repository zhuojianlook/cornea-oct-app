/* ──────────────────────────────────────────────────────────
   Run-provenance graph client (Steps v2 "Run tree").
   The sidecar's run_provenance module READS what the last preprocessing run recorded (manifest.oct_params =
   every reviewer input, manifest.oct_iter = what the run did, border_cache/*.npz surfaces, raw + corrected
   volumes) and returns it as a graph of canonical nodes — nothing is re-run. Endpoints (api_server.py):
     POST /api/case/{id}/oct-run-graph      → RunGraph
     POST /api/case/{id}/export-run-report  → RunReportResult (figure.png/svg, diagram.svg, methods.md, …+ zip)
     GET  /api/case/{id}/run-report.zip     → newest zip (browser download; token-exempt GET)
     POST /api/case/{id}/run-report-save    → copy the newest zip to {dest} (Tauri-shell save path)
   Schema: run-graph/v1 — see the node schema in the Steps v2 spec. The frontend never assumes an image
   exists on a node (images: [] + image_error is a legal, expected shape).
   ────────────────────────────────────────────────────────── */

import { api } from "./client";

export type NodeKind = "input" | "user" | "auto" | "decision" | "output";
export type NodeStatus = "applied" | "declined" | "not_run" | "missing_data" | "error";
export type NodeRole = "spine" | "input";
export type RunMode = "corrections" | "automatic" | "none";
export type ImageOrientation = "sagittal" | "axial" | "plot" | "none";
export type EdgeKind = "spine" | "input" | "note";

/** A node number's value. The backend flattens dict values to strings (L12), but older records / new
    builders may still hand back an object or a list of objects — renderers must format those defensively
    (see fmtValue in RunNodeDetail) and never fall through to String(v) → "[object Object]". */
export type RunNumberValue =
  | string | number | boolean | null | undefined
  | RunNumberValue[]
  | { [key: string]: RunNumberValue };

export interface RunNumber {
  label: string;
  value: RunNumberValue;
  unit?: string;
}

export interface RunImage {
  role: string; // primary | central | most_edited | requested | plot
  slice_index: number | null;
  orientation: ImageOrientation;
  data_url: string;
  width: number;
  height: number;
  caption: string;
  file: string;
}

export interface RunNode {
  id: string;
  kind: NodeKind;
  role: NodeRole;
  status: NodeStatus;
  status_reason: string | null; // mandatory when status != applied
  title: string;
  summary: string; // ≤ 2 lines on the card
  description: string; // plain-language paragraph
  method: string; // past-tense Methods sentence with this run's numbers
  numbers: RunNumber[];
  params: Record<string, unknown>;
  images: RunImage[];
  image_error: string | null;
  parents: string[];
  children: string[];
  warnings: string[];
  source: { manifest: string[]; files: string[] };
}

export interface RunEdge {
  from: string;
  to: string;
  kind: EdgeKind;
  label: string | null;
}

export interface RunSlices {
  central: number;
  most_edited: number;
  rendered: number[];
  drawn_top: number[];
  drawn_bottom: number[];
}

export interface RunHeader {
  case_id: string;
  patient_id: string | null;
  eye: string | null;
  mode: RunMode;
  run_time: string | null;
  input_volume: string | null;
  raw_volume: string | null;
  dims_raw: number[] | null; // [laterals, depth, frames]
  dims_corrected: number[] | null;
  spacing_mm: number[] | null;
  slices: RunSlices;
  stages: string[]; // spine order
  warnings: string[];
  generated_at: string;
  generator: string;
}

export interface RunGraph {
  schema_version: number;
  run: RunHeader;
  nodes: RunNode[];
  edges: RunEdge[];
}

export interface RunReportResult {
  folder: string;
  zip: string;
  zip_name: string;
  download_url: string;
  files: string[];
  n_nodes: number;
  mode: string;
  bytes: number;
}

export interface RunGraphOptions {
  sliceIndex?: number | null; // render node images at this lateral only (default: central + most-edited)
  wantImages?: boolean; // false → images: [] on every node (fast path)
  thumbH?: number; // thumbnail height, backend clamps 120..600
}

/** Build the run-provenance graph for a case (a READ of the run record; never re-runs the worker). */
export function fetchRunGraph(caseId: string, opts: RunGraphOptions = {}): Promise<RunGraph> {
  const body = {
    slice_index: opts.sliceIndex ?? null,
    want_images: opts.wantImages ?? true,
    thumb_h: opts.thumbH ?? 300,
  };
  return api.json<RunGraph>(
    `/api/case/${encodeURIComponent(caseId)}/oct-run-graph`, "POST", JSON.stringify(body),
  );
}

/** Write the publication report folder + zip under cases/<id>/exports/ and return its paths. */
export function exportRunReport(
  caseId: string, opts: { sliceIndex?: number | null; dpi?: number; includeFilmstrip?: boolean } = {},
): Promise<RunReportResult> {
  const body = {
    slice_index: opts.sliceIndex ?? null,
    dpi: opts.dpi ?? 300,
    include_filmstrip: opts.includeFilmstrip ?? false,
  };
  return api.json<RunReportResult>(
    `/api/case/${encodeURIComponent(caseId)}/export-run-report`, "POST", JSON.stringify(body),
  );
}

/** Tauri-shell save path: the sidecar copies the newest report zip to `dest` (outside the case store). */
export function saveRunReportZip(caseId: string, dest: string): Promise<{ dest: string; bytes: number }> {
  return api.json<{ dest: string; bytes: number }>(
    `/api/case/${encodeURIComponent(caseId)}/run-report-save`, "POST", JSON.stringify({ dest }),
  );
}

/** Download path for the newest report zip (browser <a download>; GET is token-exempt). */
export function runReportZipPath(caseId: string): string {
  return `/api/case/${encodeURIComponent(caseId)}/run-report.zip`;
}
