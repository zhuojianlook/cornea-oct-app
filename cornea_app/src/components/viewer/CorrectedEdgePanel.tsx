import { useEffect, useMemo, useRef, useState } from "react";
import { api, resourceUrl } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import { usePendingEditStore } from "../../store/pendingEditStore";
import { parseCropBands, interpBand } from "../../store/cropBands";
import type { CaseInfo } from "../../api/types";

// CORRECTED-RESULT sagittal fix-tool — the RIGHT ("corrected") pane of the before/after fix-columns view.
//
// Edge detection is CLEANER on the flattened/motion-corrected result, and this reaches residual INTER-FRAME
// drift (a trough on early frames, a bump on later ones) that editing the raw GT cannot: a +12px nudge to the
// raw surface moved the corrected surface 0.66px (gain 0.055) because the rigid inter-frame alignment
// re-smooths it. So the correction is applied where it lands — a POST-HOC per-frame RIGID depth shift on the
// finished corrected volume (apply_sagittal_surface_gt), the SAGITTAL sibling of the axial fix-tool.
//
// It shares the ONE "Correct & re-run" button with the raw editor: the reviewer picks which line to edit via
// the Original|Corrected toggle (pendingEditStore.editTarget); this panel only PUBLISHES its drawn anchors to
// the store (setCorrectedEdge) and TimelineBar commits them (commitCorrectedEdgeAnchors) on the shared re-run.
// The two corrections COMPOSE (raw = detection+base warp; corrected = post-hoc rigid drift fix on top).
//
// GEOMETRY. A sagittal slice = one LATERAL; the pane shows (depth × frames), horizontal = frame, vertical =
// depth (0 = TOP). The parent applies scaleX(-1) (frame 0 on the visual RIGHT) + zoom/pan, so the drag's
// screen-x fraction is inverted (1 - fx) to recover the array frame — exactly like the left editor. Backend
// (all ARRAY indices, corrected-output depth):
//   GET  oct-corrected-slice?slice_index=lat  → corrected B-scan PNG (depth rows × frame cols, depth 0 = TOP)
//   POST oct-corrected-curve  {slice_index:lat} → {edge[frame], fit, slices, depth_vox, n_frames}
type CeMap = Map<number, Map<number, number>>;   // lateral -> frame -> depth (corrected-output depth, 0 = TOP)

function cloneMap(m: CeMap): CeMap { const o: CeMap = new Map(); m.forEach((inner, l) => o.set(l, new Map(inner))); return o; }
function sig(m: CeMap): string {
  return [...m.keys()].sort((a, b) => a - b).map((l) =>
    `${l}:${[...(m.get(l) as Map<number, number>).entries()].sort((a, b) => a[0] - b[0])
      .map(([f, d]) => `${f}=${Math.round(d)}`).join(",")}`).join(";");
}
function ocParams(ci: CaseInfo | null): Record<string, unknown> {
  return ((ci?.manifest as Record<string, unknown> | undefined)?.oct_params as Record<string, unknown> | undefined) ?? {};
}
function anchorsToApi(m: CeMap): Record<string, Record<string, number>> {
  const o: Record<string, Record<string, number>> = {};
  m.forEach((inner, l) => {
    const io: Record<string, number> = {};
    inner.forEach((d, f) => { io[String(f)] = Math.round(d); });
    if (Object.keys(io).length) o[String(l)] = io;
  });
  return o;
}

export function CorrectedEdgePanel({ sliceIndex, bDispW, bDispH, bSized, bZoom, bPan, filterCss, readOnly = false, onZoomWheel }: {
  sliceIndex: number;
  bDispW: number; bDispH: number; bSized: boolean; bZoom: number; bPan: { x: number; y: number };
  filterCss?: string; readOnly?: boolean;
  /** scroll-to-zoom, shared with the left editor: (clientX, clientY, deltaY, this pane's host rect). */
  onZoomWheel?: (clientX: number, clientY: number, deltaY: number, rect: DOMRect | undefined) => void;
}) {
  const caseId = useCaseStore((s) => s.caseId);
  const caseInfo = useCaseStore((s) => s.caseInfo);
  const segVersion = useWorkflowStore((s) => s.segVersion);
  const editTarget = usePendingEditStore((s) => s.editTarget);
  const setCorrectedEdge = usePendingEditStore((s) => s.setCorrectedEdge);
  const edit = editTarget === "corrected" && !readOnly;

  // persisted corrected-edge anchors (keyed by ARRAY lateral) → seed the editable set
  const persistedSig = JSON.stringify(ocParams(caseInfo).corrected_edge_anchors ?? {});
  const persisted = useMemo(() => {
    const o: CeMap = new Map();
    try {
      for (const [l, frames] of Object.entries(JSON.parse(persistedSig) as Record<string, Record<string, number>>)) {
        const inner = new Map<number, number>();
        for (const [f, d] of Object.entries(frames)) inner.set(Number(f), Number(d));
        if (inner.size) o.set(Number(l), inner);
      }
    } catch { /* ignore malformed */ }
    return o;
  }, [persistedSig]);

  const [nFrames, setNFrames] = useState(0);
  const [depthVox, setDepthVox] = useState(0);
  const [edge, setEdge] = useState<number[] | null>(null);
  const [anchors, setAnchors] = useState<CeMap>(new Map());

  useEffect(() => { setAnchors(cloneMap(persisted)); }, [persistedSig]);   // re-seed on case load / after a re-run

  // detected corrected surface across FRAMES for the current lateral slice. Re-fetches after a re-run (segVersion)
  // so the line moves onto the target — the convergence is visible in the pane.
  useEffect(() => {
    if (!caseId) { setEdge(null); return; }
    let cancel = false;
    api.json<{ slices: number; depth_vox: number; n_frames: number; index: number; edge: number[]; fit: number[] }>(
      `/api/case/${caseId}/oct-corrected-curve`, "POST", JSON.stringify({ slice_index: sliceIndex }))
      .then((r) => { if (cancel) return; setEdge(r.edge); setNFrames(r.n_frames); setDepthVox(r.depth_vox); })
      .catch(() => { if (!cancel) { setEdge(null); } });
    return () => { cancel = true; };
  }, [caseId, sliceIndex, segVersion]);

  const curAnchors = anchors.get(sliceIndex) ?? null;
  const anchorCount = useMemo(() => { let n = 0; anchors.forEach((m) => { n += m.size; }); return n; }, [anchors]);
  const editedSlices = useMemo(() => [...anchors.keys()].filter((l) => (anchors.get(l)?.size ?? 0) > 0)
    .sort((a, b) => a - b), [anchors]);
  const dirty = sig(anchors) !== sig(persisted);

  // PUBLISH the drawn anchors to the shared store, so the single "Correct & re-run" (TimelineBar) commits them
  // alongside any raw edits. Keyed by case so a stale corrected-edge edit can never land on the next scan.
  useEffect(() => {
    if (!caseId) return;
    setCorrectedEdge({ caseId, anchors: anchorsToApi(anchors), nPoints: anchorCount, dirty });
  }, [caseId, anchors, anchorCount, dirty, setCorrectedEdge]);

  const edgeY = (f: number): number => {
    const a = curAnchors?.get(f);
    if (a != null) return a;                       // un-confirmed drag → WYSIWYG (the line you're moving is visible)
    return edge ? edge[f] : 0;
  };
  // #9 v3: the interpolated artifact band for THIS lateral — the corrected result has these frames ZEROED (black),
  // so the red surface line is BROKEN over them (drawing a line across the removed/black region is misleading).
  const cropBandsSig = JSON.stringify(ocParams(caseInfo).crop_bands ?? {});
  const curBand = useMemo(() => interpBand(parseCropBands(JSON.parse(cropBandsSig)), sliceIndex), [cropBandsSig, sliceIndex]);
  const inBand = (f: number) => curBand != null && f >= curBand[0] && f <= curBand[1];
  const segPts = (): string[] => {
    if (!edge || nFrames < 2) return [];
    const segs: string[][] = []; let cur: string[] = [];
    for (let f = 0; f < nFrames; f++) {
      const y = inBand(f) ? NaN : edgeY(f);
      if (Number.isFinite(y)) cur.push(`${f + 0.5},${y}`);
      else if (cur.length) { segs.push(cur); cur = []; }
    }
    if (cur.length) segs.push(cur);
    return segs.filter((s) => s.length > 1).map((s) => s.join(" "));
  };

  // drag: screen → (frame, depth). Parent is scaleX(-1)-flipped so frame 0 is on the VISUAL RIGHT; invert the
  // screen-x fraction (1 - fx) to recover the array frame. A drag within 1px of the detected edge clears that
  // anchor (snap-back), so dragging back onto the line undoes it.
  const hostRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef(false);
  const applyDrag = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    if (!edit || !edge || nFrames < 2 || depthVox < 2) return;
    const r = svg.getBoundingClientRect();
    const frame = Math.round((1 - (clientX - r.left) / r.width) * nFrames - 0.5);
    if (frame < 0 || frame >= nFrames || frame >= edge.length) return;
    const depth = Math.round(Math.max(0, Math.min(depthVox - 1, ((clientY - r.top) / r.height) * depthVox)));
    setAnchors((prev) => {
      const o = cloneMap(prev);
      let inner = o.get(sliceIndex); if (!inner) { inner = new Map(); o.set(sliceIndex, inner); }
      if (Math.abs(depth - (edge[frame] ?? depth)) <= 1) inner.delete(frame);   // back on the detected line → clear
      else inner.set(frame, depth);
      if (inner.size === 0) o.delete(sliceIndex);
      return o;
    });
  };
  const onDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!edit) return;
    dragRef.current = true; e.currentTarget.setPointerCapture(e.pointerId); applyDrag(e.clientX, e.clientY, e.currentTarget);
  };
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => { if (dragRef.current) applyDrag(e.clientX, e.clientY, e.currentTarget); };
  const onUp = () => { dragRef.current = false; };
  const clearSlice = () => setAnchors((prev) => { const o = cloneMap(prev); o.delete(sliceIndex); return o; });

  return (
    <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center", gap: 0, position: "relative" }}>
      <span className="text-[11px]" style={{ color: edit ? "#22d3ee" : "var(--c-green)", position: "absolute", top: 0, right: 0,
                                             zIndex: 4, pointerEvents: "none", background: "var(--c-bg)", padding: "0 4px" }}>
        corrected (result){edit ? " — drag to fix inter-frame drift" : ""}</span>
      {edit && (
        <div style={{ position: "absolute", top: 0, left: 0, zIndex: 5, display: "flex", gap: 4, alignItems: "center",
                      background: "var(--c-bg)", padding: "1px 3px", flexWrap: "wrap", maxWidth: "70%" }}>
          <button onClick={clearSlice} disabled={!(curAnchors?.size)}
                  style={{ padding: "2px 8px", borderRadius: 6, fontSize: 11, lineHeight: 1.4, cursor: "pointer",
                           border: "1px solid var(--c-border)", background: "var(--c-surface)", color: "var(--c-text)" }}>Clear slice</button>
          {editedSlices.length > 0 && <span style={{ fontSize: 10, opacity: 0.7 }}>edited: {editedSlices.slice(0, 10).join(", ")}{editedSlices.length > 10 ? "…" : ""}{dirty ? " (uncommitted)" : ""}</span>}
        </div>
      )}
      <div ref={hostRef}
           onWheel={onZoomWheel ? (e) => { e.preventDefault(); onZoomWheel(e.clientX, e.clientY, e.deltaY, hostRef.current?.getBoundingClientRect()); } : undefined}
           style={{ flex: 1, minHeight: 0, width: "100%", display: "flex", alignItems: "center", justifyContent: "center", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "relative",
                      ...(bSized ? { width: bDispW, height: bDispH } : { display: "inline-block", maxHeight: "100%", maxWidth: "100%" }),
                      transform: `translate(${bPan.x}px, ${bPan.y}px) scale(${bZoom}) scaleX(-1)`, transformOrigin: "center center" }}>
          {caseId && <img src={resourceUrl(`/api/case/${caseId}/oct-corrected-slice?slice_index=${sliceIndex}&t=${segVersion}`)}
            alt="corrected" draggable={false}
            style={bSized
              ? { display: "block", width: "100%", height: "100%", objectFit: "fill", imageRendering: "pixelated", filter: filterCss }
              : { display: "block", maxHeight: "100%", maxWidth: "100%", imageRendering: "pixelated", filter: filterCss }} />}
          {edge && nFrames > 1 && depthVox > 1 && (
            <svg viewBox={`0 0 ${nFrames} ${depthVox}`} preserveAspectRatio="none"
                 onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}
                 style={{ position: "absolute", inset: 0, width: "100%", height: "100%",
                          cursor: edit ? "row-resize" : "default", touchAction: "none",
                          pointerEvents: edit ? "auto" : "none" }}>
              {/* detected / dragged corrected surface (red) — broken over the cropped (zeroed) artifact band */}
              {segPts().map((sg, i) => (
                <polyline key={`ce${i}`} fill="none" stroke="#ff4d4d" vectorEffect="non-scaling-stroke"
                          strokeWidth={dirty ? 1.3 : 0.9} opacity={edit ? (dirty ? 0.95 : 0.8) : 0.5} points={sg} />
              ))}
              {/* anchored frames → pink vertical ticks (circles squash under the stretched viewBox) */}
              {edit && curAnchors && [...curAnchors.entries()].map(([f, d], i) => (
                <line key={`a${i}`} x1={f + 0.5} y1={d - depthVox / 50} x2={f + 0.5} y2={d + depthVox / 50}
                      stroke="#ff5db0" strokeWidth={1.6} vectorEffect="non-scaling-stroke" opacity={0.95} />
              ))}
            </svg>
          )}
        </div>
      </div>
    </div>
  );
}
