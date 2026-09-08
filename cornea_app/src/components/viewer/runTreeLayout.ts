/* Pure layout for the run-provenance tree. Mirrors run_provenance.layout_graph on the sidecar (same
   constants — keep SPINE_X / INPUT_X / FAR_X / NODE_W / NODE_H / GAP in step with it) so the on-screen tree
   and the exported diagram.svg line up:
     • spine nodes stack top-down at x=SPINE_X in `run.stages` order (+ any spine node the header forgot);
     • reviewer-input nodes sit in a left column at x=INPUT_X, level with the first spine node they feed
       (shifted down a row when that level is already taken);
     • an input node that feeds ONLY other input nodes (n11_pane_edits → n02_top_lines / n05_bottom_lines in
       line mode, M6) goes one column further left at x=FAR_X, level with its first placed child, so a
       user→user edge can be drawn left-to-right like every other input edge and carry its label;
     • "unused" nodes — the other chain's stages that did not run and touch no edge — are left out of the
       layout by default (kept in the graph / nodes.json); pass { showUnused: true } to place them in the
       input column (M10).
   Note edges (kind "note", e.g. "drawn on the raw scan") are not drawn, as in diagram.svg. */

import type { RunGraph, RunNode } from "../../api/runGraph";

export const NODE_W = 340;   // fits "Automatic corneal-surface detection (DP)" at the 12 px card font
export const NODE_H = 84;    // chips row + a two-line title + a two-line summary
export const GAP = 28;
export const COL_GAP = 40;
export const FAR_X = 40;
export const INPUT_X = FAR_X + NODE_W + COL_GAP;      // 420
export const SPINE_X = INPUT_X + NODE_W + COL_GAP;    // 800
const TOP = 16;
const STEP = NODE_H + GAP;

export interface Box { x: number; y: number; w: number; h: number }
export interface LayoutEdge {
  from: string; to: string; d: string; kind: "spine" | "input" | "note"; label: string | null;
  /** where to put the label (mid-point of the curve); null for unlabeled edges */
  labelAt: { x: number; y: number } | null;
}
export interface RunLayout {
  boxes: Record<string, Box>; edges: LayoutEdge[]; width: number; height: number;
  /** ids of nodes left out of the layout (unused other-chain stages) */
  hidden: string[];
  /** true when the far-left column is in use (a user node feeds only other input nodes) */
  hasFarColumn: boolean;
}
export interface LayoutOptions { showUnused?: boolean }

/** An "unused stage": a non-spine node that did not run and is attached to no DRAWN edge (spine / input;
    note edges such as "drawn on the raw scan" don't count) — i.e. the other chain's canonical stage kept in
    nodes.json for completeness, or a reviewer-input slot with nothing on record and no stage to feed on
    this chain (M10). A not-run input that still has an edge to the stage it would feed stays visible. */
export function isUnusedStage(n: RunNode, g?: RunGraph): boolean {
  if (n.role === "spine" || n.status !== "not_run") return false;
  if (g && g.edges.length) return !g.edges.some((e) => e.kind !== "note" && (e.from === n.id || e.to === n.id));
  return (n.parents?.length ?? 0) === 0 && (n.children?.length ?? 0) === 0;
}

export function layoutRunGraph(g: RunGraph, opts: LayoutOptions = {}): RunLayout {
  const byId = new Map<string, RunNode>();
  for (const n of g.nodes) byId.set(n.id, n);
  const hidden = opts.showUnused ? [] : g.nodes.filter((n) => isUnusedStage(n, g)).map((n) => n.id);
  const hiddenSet = new Set(hidden);
  const visible = g.nodes.filter((n) => !hiddenSet.has(n.id));

  // Spine order: run.stages first (only ids that exist), then any spine node the header forgot to list.
  const spineIds: string[] = [];
  for (const id of g.run.stages ?? []) {
    const n = byId.get(id);
    if (n && n.role === "spine" && !spineIds.includes(id)) spineIds.push(id);
  }
  for (const n of visible) if (n.role === "spine" && !spineIds.includes(n.id)) spineIds.push(n.id);
  const spineSet = new Set(spineIds);

  // Input nodes. Those with at least one spine child (or no children at all) go in the input column;
  // those whose children are all non-spine nodes go one column further left (placed after their children).
  // Without a far column the tree keeps its two-column footprint (input at FAR_X, spine at INPUT_X).
  const inputs = visible.filter((n) => !spineSet.has(n.id));
  const feedsSpine = (n: RunNode) => (n.children ?? []).some((c) => spineSet.has(c));
  const feedsOnlyInputs = (n: RunNode) => (n.children ?? []).length > 0 && !feedsSpine(n);
  const near = inputs.filter((n) => !feedsOnlyInputs(n));
  const far = inputs.filter(feedsOnlyInputs);
  const hasFarColumn = far.length > 0;
  const inputX = hasFarColumn ? INPUT_X : FAR_X;
  const spineX = hasFarColumn ? SPINE_X : INPUT_X;

  const boxes: Record<string, Box> = {};
  spineIds.forEach((id, i) => { boxes[id] = { x: spineX, y: TOP + i * STEP, w: NODE_W, h: NODE_H }; });
  const placeColumn = (nodes: RunNode[], x: number) => {
    const usedRows = new Set<number>();
    for (const n of nodes) {
      const child = (n.children ?? []).find((c) => boxes[c] !== undefined);
      let y = child ? boxes[child].y : TOP;
      while (usedRows.has(y)) y += STEP;
      usedRows.add(y);
      boxes[n.id] = { x, y, w: NODE_W, h: NODE_H };
    }
  };
  placeColumn(near, inputX);
  placeColumn(far, FAR_X);

  // Edges: the graph's own list when present, else derived from parents. Note edges are not drawn.
  const src = g.edges.length
    ? g.edges
    : visible.flatMap((n) => (n.parents ?? []).map((p) => ({ from: p, to: n.id, kind: "spine" as const, label: null })));
  const edges: LayoutEdge[] = [];
  for (const e of src) {
    if (e.kind === "note") continue;
    const a = boxes[e.from];
    const b = boxes[e.to];
    if (!a || !b) continue;
    const fromSpine = spineSet.has(e.from);
    const toSpine = spineSet.has(e.to);
    let d: string;
    let labelAt: { x: number; y: number } | null = null;
    if (fromSpine && toSpine) {
      // straight drop from bottom-centre to the next node's top-centre
      d = `M ${a.x + a.w / 2} ${a.y + a.h} L ${b.x + b.w / 2} ${b.y}`;
      if (e.label) labelAt = { x: a.x + a.w / 2 + 6, y: (a.y + a.h + b.y) / 2 };
    } else if (a.x < b.x) {
      // left column → right column (input → spine, or far input → input): cubic from the right edge into
      // the child's left edge; the label sits at the curve's mid-point (t = 0.5 of this symmetric cubic).
      const x1 = a.x + a.w, y1 = a.y + a.h / 2;
      const x2 = b.x, y2 = b.y + b.h / 2;
      const cx = (x1 + x2) / 2;
      d = `M ${x1} ${y1} C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`;
      if (e.label) labelAt = { x: (x1 + x2) / 2, y: (y1 + y2) / 2 };
    } else {
      // same column (or backwards): a short hook out of the right edge and back into the child's right
      // edge so the two cards never overlap the line; rare (only when the graph lists an unexpected edge).
      const x1 = a.x + a.w, y1 = a.y + a.h / 2;
      const x2 = b.x + b.w, y2 = b.y + b.h / 2;
      const cx = Math.max(x1, x2) + COL_GAP / 2;
      d = `M ${x1} ${y1} C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`;
      if (e.label) labelAt = { x: cx, y: (y1 + y2) / 2 };
    }
    edges.push({ from: e.from, to: e.to, d, kind: e.kind, label: e.label, labelAt });
  }

  const maxY = Object.values(boxes).reduce((m, b) => Math.max(m, b.y), 0);
  const width = spineX + NODE_W + COL_GAP;
  return { boxes, edges, width, height: (spineIds.length ? maxY : TOP) + NODE_H + TOP, hidden, hasFarColumn };
}
