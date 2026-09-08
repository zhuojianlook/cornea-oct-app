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

export function CorrectedEdgePanel({ sliceIndex, bDispW, bDispH, bSized, bZoom, bPan, filterCss, readOnly = false, stairEdge = false, markMode = false, onPan, onZoomWheel, origDepthVox, bottomEditNonce }: {
  sliceIndex: number;
  bDispW: number; bDispH: number; bSized: boolean; bZoom: number; bPan: { x: number; y: number };
  filterCss?: string; readOnly?: boolean;
  /** Stairstep display, shared with the left editor's ⊐ toggle: draw each FRAME as a horizontal step at
   *  its exact depth instead of a sloped centre-to-centre polyline. The correction applied here is one
   *  RIGID depth per frame, so the staircase is literally the shape of what is being edited. */
  stairEdge?: boolean;
  /** ⚑ Mark on the CORRECTED result: drag across frames to flag a region that could still improve.
   *  Marks nothing in the pipeline — it is how the reviewer points at something on the picture they are
   *  actually judging, instead of describing it in words. */
  markMode?: boolean;
  /** depth (rows) of the ORIGINAL pane's image. A surface-cropped run extends the canvas, so the corrected volume is
   *  TALLER; the pane used to be forced into the original's box, squashing the extra rows away (reviewer 2026-09-04:
   *  "the image height of the corrected slice appears to be the same"). With this the box grows by depthVox/origDepthVox
   *  so one row here is one row there and the raised columns are visibly higher. */
  origDepthVox?: number;
  /** bumped by the gallery's bottom-line queue: switch this pane to editing its BOTTOM line (reviewer 2026-09-05) */
  bottomEditNonce?: number;
  /** Middle/right-drag pan, delegated to the parent (which owns bPan). The corrected pane had NO pan handler
   *  of its own — the original pane's lives on a different SVG — so a middle-drag here did nothing useful and,
   *  before the button guard, dragged the surface instead. */
  onPan?: (dx: number, dy: number) => void;
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
  // Frames the run treated as LIVE (dead / decorrelated frames are black in the corrected volume and carry
  // detector junk), plus the dome shape the run enforces — both from oct-corrected-curve — so the blue quadratic
  // below follows the same rule as the run and the original pane (reviewer 2026-09-08: "the blue guide on the
  // corrected slice still appears inverted").
  const [liveMask, setLiveMask] = useState<boolean[] | null>(null);
  const [shapeCurv, setShapeCurv] = useState<number | null>(null);
  const [shapeMinFrac, setShapeMinFrac] = useState<number>(0.5);
  // The reviewer's TARGET line: the served surface after generalize_corrected_surface + their drawn-anchor re-pin.
  // `edge` (red) is the constrained DETECTION, which is the array the corrections-path warp stages actually diff
  // against — so drawing both is the only honest way to show "where it reads" vs "where you said it should be".
  // Absent (null) whenever the two coincide, i.e. on any scan with no corrected_edge_anchors.
  const [target, setTarget] = useState<number[] | null>(null);
  const [anchors, setAnchors] = useState<CeMap>(new Map());
  // BOTTOM (posterior) surface on the corrected result — surface-cropped scans only (the run caches the served
  // posterior; the endpoint carries it by the measured move). In the cropped frames the bottom edge is the real one
  // and the top is an estimate, so the reviewer verifies / redraws the bottom there (2026-09-04). Drawn orange,
  // stairstepped with the same toggle, edited when `editBottom` is on, autosaved as corrected_post_anchors.
  const [bottom, setBottom] = useState<(number | null)[] | null>(null);
  const [postAnchors, setPostAnchors] = useState<CeMap>(new Map());
  const [editBottom, setEditBottom] = useState(false);
  useEffect(() => { if (bottomEditNonce) setEditBottom(true); }, [bottomEditNonce]);
  const persistedPostSig = JSON.stringify(ocParams(caseInfo).corrected_post_anchors ?? {});
  const persistedPost = useMemo(() => {
    const o: CeMap = new Map();
    try {
      for (const [l, frames] of Object.entries(JSON.parse(persistedPostSig) as Record<string, Record<string, number>>)) {
        const inner = new Map<number, number>();
        for (const [f, d] of Object.entries(frames)) inner.set(Number(f), Number(d));
        if (inner.size) o.set(Number(l), inner);
      }
    } catch { /* ignore malformed */ }
    return o;
  }, [persistedPostSig]);
  const savedPostSigRef = useRef<string | null>(null);
  useEffect(() => {
    if (persistedPostSig === savedPostSigRef.current) return;
    setPostAnchors(cloneMap(persistedPost));
  }, [persistedPostSig]);

  // AUTOSAVE state. savedSigRef holds the anchor signature this panel last successfully wrote, so the re-seed
  // below can tell "the manifest changed because WE just saved" from "the manifest changed because the case
  // reloaded / a Run finished". Without that distinction the autosave's own write bounces back through
  // persistedSig and re-seeds the editor mid-drawing, discarding anything drawn since the POST left.
  const savedSigRef = useRef<string | null>(null);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");

  useEffect(() => {                                                        // re-seed on case load / after a re-run
    if (persistedSig === savedSigRef.current) return;                      // ...but never on our own autosave echo
    setAnchors(cloneMap(persisted));
  }, [persistedSig]);

  // detected corrected surface across FRAMES for the current lateral slice. Re-fetches after a re-run (segVersion)
  // so the line moves onto the target — the convergence is visible in the pane.
  useEffect(() => {
    if (!caseId) { setEdge(null); setTarget(null); return; }
    let cancel = false;
    api.json<{ slices: number; depth_vox: number; n_frames: number; index: number; edge: number[]; fit: number[];
               target?: number[]; bottom?: (number | null)[]; live_frames?: boolean[]; shape_curv?: number | null;
               shape_min_fraction?: number }>(
      `/api/case/${caseId}/oct-corrected-curve`, "POST", JSON.stringify({ slice_index: sliceIndex }))
      .then((r) => { if (cancel) return; setEdge(r.edge); setTarget(r.target ?? null); setBottom(r.bottom ?? null);
                     setNFrames(r.n_frames); setDepthVox(r.depth_vox);
                     setLiveMask(Array.isArray(r.live_frames) ? r.live_frames : null);
                     setShapeCurv(typeof r.shape_curv === "number" && Number.isFinite(r.shape_curv) ? r.shape_curv : null);
                     setShapeMinFrac(typeof r.shape_min_fraction === "number" ? r.shape_min_fraction : 0.5); })
      .catch(() => { if (!cancel) { setEdge(null); setTarget(null); } });
    return () => { cancel = true; };
  }, [caseId, sliceIndex, segVersion]);

  const curAnchors = anchors.get(sliceIndex) ?? null;
  const anchorCount = useMemo(() => { let n = 0; anchors.forEach((m) => { n += m.size; }); return n; }, [anchors]);
  const editedSlices = useMemo(() => [...anchors.keys()].filter((l) => (anchors.get(l)?.size ?? 0) > 0)
    .sort((a, b) => a - b), [anchors]);
  const postDirty = sig(postAnchors) !== sig(persistedPost);
  const dirty = sig(anchors) !== sig(persisted) || postDirty;

  // PUBLISH the drawn anchors to the shared store, so the single "Correct & re-run" (TimelineBar) commits them
  // alongside any raw edits. Keyed by case so a stale corrected-edge edit can never land on the next scan.
  useEffect(() => {
    if (!caseId) return;
    setCorrectedEdge({ caseId, anchors: anchorsToApi(anchors), postAnchors: anchorsToApi(postAnchors), nPoints: anchorCount, dirty });
  }, [caseId, anchors, postAnchors, anchorCount, dirty, setCorrectedEdge]);

  // AUTOSAVE the drawn corrected-edge anchors. Previously a drawing lived ONLY in this component's state until
  // the reviewer pressed "Run with corrections" or a verdict button (TimelineBar takeCorrectedEdge →
  // commitCorrectedEdgeAnchors), so anything short of that — a reload, a case switch, closing the app — threw the
  // drawing away silently. That cost the reviewer the same hand-drawn slice twice, and hand-drawn GT is the most
  // expensive thing in this app to reproduce. Debounced ~900ms after the last change so a drag writes once at the
  // end rather than per pointer-move, and skipped while a drag is in flight.
  // NOTE this makes a drawing STICKY as soon as it is made: corrected_edge_anchors is consumed by the next Run.
  // That is the point (it is ground truth), and "Clear slice" / "Clear all corrections" still remove it.
  const commitCorrectedEdgeAnchors = useCaseStore((s) => s.commitCorrectedEdgeAnchors);
  const anchorsSig = useMemo(() => sig(anchors), [anchors]);
  const postSig = useMemo(() => sig(postAnchors), [postAnchors]);
  useEffect(() => {
    if (!caseId || readOnly || !dirty) return;
    const t = setTimeout(() => {
      if (dragRef.current) return;                       // still drawing — the next change reschedules this
      const payload = anchorsToApi(anchors);
      const postPayload = anchorsToApi(postAnchors);
      const myCase = caseId;
      setSaveState("saving");
      void (async () => {
        try {
          // Guard against a case switch landing between the debounce firing and the request going out: the store's
          // commit targets whatever case is CURRENT, so writing case A's anchors while B is open would corrupt B.
          if (useCaseStore.getState().caseId !== myCase) return;
          await commitCorrectedEdgeAnchors(payload, postPayload);
          savedSigRef.current = JSON.stringify(payload);   // matches how persistedSig is built
          savedPostSigRef.current = JSON.stringify(postPayload);
          setSaveState("saved");
        } catch {
          setSaveState("error");                           // local state is untouched → the next change retries
        }
      })();
    }, 900);
    return () => clearTimeout(t);
  }, [caseId, readOnly, dirty, anchorsSig, postSig, anchors, postAnchors, commitCorrectedEdgeAnchors]);

  const curPostAnchors = postAnchors.get(sliceIndex);
  const postY = (f: number): number => {
    const a = curPostAnchors?.get(f);
    if (a != null) return a;
    const b = bottom ? bottom[f] : null;
    return b == null ? NaN : b;
  };
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
  // (3) THE BLUE QUADRATIC — the deg-2 LEAST-SQUARES best fit of the corrected surface across frames, drawn
  // for reference. CALCULATED, never editable and never a target: the reviewer's quality criterion for a
  // corrected result is that the anterior surface is a smooth quadratic best fit, so this shows what that
  // fit IS and how far the surface sits from it. Fitted to the DISPLAYED edge (edgeY, so un-confirmed drags
  // are included) and to real frames only — the zeroed artifact band contributes nothing. Drawn smooth even
  // under the stairstep toggle: a quadratic is a curve, and stepping it would misrepresent the fit.
  const quadPts = (): string | null => {
    if (!edge || nFrames < 3) return null;
    let n = 0, Sx = 0, Sx2 = 0, Sx3 = 0, Sx4 = 0, Sy = 0, Sxy = 0, Sx2y = 0;
    const isLive = (f: number) => !liveMask || liveMask.length !== nFrames || liveMask[f];
    let fLo = -1, fHi = -1;
    for (let f = 0; f < nFrames; f++) {
      const y = inBand(f) || !isLive(f) ? NaN : edgeY(f);          // dead frames carry no surface: excluded
      if (!Number.isFinite(y)) continue;
      if (fLo < 0) fLo = f; fHi = f;
      const x = f, x2 = x * x;
      n += 1; Sx += x; Sx2 += x2; Sx3 += x2 * x; Sx4 += x2 * x2;
      Sy += y; Sxy += x * y; Sx2y += x2 * y;
    }
    if (n < 3) return null;
    // normal equations for y = a x^2 + b x + c, solved by Cramer (3x3 — no matrix library needed)
    const m = [[Sx4, Sx3, Sx2], [Sx3, Sx2, Sx], [Sx2, Sx, n]];
    const v = [Sx2y, Sxy, Sy];
    const det3 = (q: number[][]): number =>
      q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1])
      - q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0])
      + q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0]);
    const D = det3(m);
    if (!Number.isFinite(D) || Math.abs(D) < 1e-9) return null;   // degenerate (e.g. all points on one frame)
    const sub = (col: number) => det3(m.map((row, r) => row.map((val, c) => (c === col ? v[r] : val))));
    let a = sub(0) / D, b = sub(1) / D, c = sub(2) / D;
    // THE DOME RULE, same as the run and the original pane: a fit that is sign-wrong or flatter than
    // shapeMinFrac × the B-scan-plane dome is motion-dominated — impose the dome, apex at the middle of the
    // live frames, only the level fitted.
    if (shapeCurv != null && shapeCurv > 0 && a < shapeMinFrac * shapeCurv && fHi > fLo) {
      const fmid = 0.5 * (fLo + fHi);
      a = shapeCurv; b = -2 * a * fmid;
      let sum = 0, cnt = 0;
      for (let f = 0; f < nFrames; f++) {
        const y = inBand(f) || !isLive(f) ? NaN : edgeY(f);
        if (!Number.isFinite(y)) continue;
        sum += y - a * f * f - b * f; cnt += 1;
      }
      c = cnt ? sum / cnt : c;
    }
    const pts: string[] = [];
    for (let f = 0; f < nFrames; f++) {
      const y = a * f * f + b * f + c;
      if (Number.isFinite(y)) pts.push(`${f + 0.5},${y}`);
    }
    return pts.length > 1 ? pts.join(" ") : null;
  };

  const segPts = (): string[] => {
    if (!edge || nFrames < 2) return [];
    const segs: string[][] = []; let cur: string[] = [];
    for (let f = 0; f < nFrames; f++) {
      const y = inBand(f) ? NaN : edgeY(f);
      // stairEdge: a HORIZONTAL step spanning the whole frame [f, f+1] at its exact depth (the polyline
      // then rises vertically to the next frame → a staircase), so each frame's own depth is readable
      // and can be placed exactly. else: sloped centre-to-centre.
      if (Number.isFinite(y)) { if (stairEdge) cur.push(`${f},${y}`, `${f + 1},${y}`); else cur.push(`${f + 0.5},${y}`); }
      else if (cur.length) { segs.push(cur); cur = []; }
    }
    if (cur.length) segs.push(cur);
    return segs.filter((s) => s.length > 1).map((s) => s.join(" "));
  };

  // drag: screen → (frame, depth). Parent is scaleX(-1)-flipped so frame 0 is on the VISUAL RIGHT; invert the
  // screen-x fraction (1 - fx) to recover the array frame. A drag within 1px of the detected edge clears that
  // anchor (snap-back), so dragging back onto the line undoes it.
  // ── CORRECTED-SPACE DEFECT MARKS ─────────────────────────────────────────────────────────────────────
  // {lateral: [[f0,f1], …]} in CORRECTED frame coordinates, persisted to oct_params.corrected_defect_marks.
  // Kept apart from manifest.defect_marks (raw geometry, wiped by a re-preprocess) because these describe
  // the corrected picture and must survive it.
  const [marks, setMarks] = useState<Record<string, number[][]>>({});
  useEffect(() => {
    const stored = (ocParams(caseInfo).corrected_defect_marks ?? {}) as Record<string, number[][]>;
    setMarks(stored);
  }, [caseInfo]);
  const curMarks: number[][] = marks[String(sliceIndex)] ?? [];
  const pushMarks = (next: Record<string, number[][]>) => {
    setMarks(next);
    if (!caseId) return;
    void api.json(`/api/case/${caseId}/oct-corrected-marks`, "POST",
                  JSON.stringify({ params: { marks: next } })).catch(() => { /* pointing is cheap; retry by re-marking */ });
  };
  const markStart = useRef<number | null>(null);
  const panRef = useRef<{ x: number; y: number } | null>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef(false);
  const applyDrag = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    if (!edit || !edge || nFrames < 2 || depthVox < 2) return;
    const r = svg.getBoundingClientRect();
    const frame = Math.round((1 - (clientX - r.left) / r.width) * nFrames - 0.5);
    if (frame < 0 || frame >= nFrames || frame >= edge.length) return;
    const depth = Math.round(Math.max(0, Math.min(depthVox - 1, ((clientY - r.top) / r.height) * depthVox)));
    if (editBottom) {                                   // the BOTTOM line: same gesture, its own anchor set
      const ref = bottom ? bottom[frame] : null;
      setPostAnchors((prev) => {
        const o = cloneMap(prev);
        let inner = o.get(sliceIndex); if (!inner) { inner = new Map(); o.set(sliceIndex, inner); }
        if (ref != null && Math.abs(depth - ref) <= 1) inner.delete(frame);   // back on the carried line → clear
        else inner.set(frame, depth);
        if (inner.size === 0) o.delete(sliceIndex);
        return o;
      });
      return;
    }
    setAnchors((prev) => {
      const o = cloneMap(prev);
      let inner = o.get(sliceIndex); if (!inner) { inner = new Map(); o.set(sliceIndex, inner); }
      if (Math.abs(depth - (edge[frame] ?? depth)) <= 1) inner.delete(frame);   // back on the detected line → clear
      else inner.set(frame, depth);
      if (inner.size === 0) o.delete(sliceIndex);
      return o;
    });
  };
  /** screen-x → array FRAME. The parent is scaleX(-1)-flipped, so the fraction is inverted, exactly as in
   *  applyDrag — the mark and the line must agree about which frame is under the cursor. */
  const frameAt = (clientX: number, svg: SVGSVGElement): number => {
    const r = svg.getBoundingClientRect();
    return Math.max(0, Math.min(nFrames - 1, Math.round((1 - (clientX - r.left) / r.width) * nFrames - 0.5)));
  };
  const [markDrag, setMarkDrag] = useState<[number, number] | null>(null);
  const onDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!edit) return;
    // MIDDLE / RIGHT BUTTON PANS. Only the LEFT button edits. There is no pan handler underneath this pane —
    // the original pane owns the one that moves bPan — so the drag is captured here and forwarded to the
    // parent, otherwise a middle-drag simply did nothing (reviewer, 2026-09-02).
    if (e.button !== 0) {
      panRef.current = { x: e.clientX, y: e.clientY };
      e.currentTarget.setPointerCapture(e.pointerId);
      e.preventDefault();
      return;
    }
    if (markMode) {                       // ⚑ start a band (or click an existing one to remove it)
      const f = frameAt(e.clientX, e.currentTarget);
      markStart.current = f; setMarkDrag([f, f]);
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }
    dragRef.current = true; e.currentTarget.setPointerCapture(e.pointerId); applyDrag(e.clientX, e.clientY, e.currentTarget);
  };
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (panRef.current) {                            // middle/right drag → move the image, never the surface
      const d = panRef.current;
      onPan?.(e.clientX - d.x, e.clientY - d.y);
      panRef.current = { x: e.clientX, y: e.clientY };
      return;
    }
    if (e.buttons && !(e.buttons & 1)) return;
    if (markMode) {
      if (markStart.current != null) setMarkDrag([markStart.current, frameAt(e.clientX, e.currentTarget)]);
      return;
    }
    if (dragRef.current) applyDrag(e.clientX, e.clientY, e.currentTarget);
  };
  const onUp = (e?: React.PointerEvent<SVGSVGElement>) => {
    if (panRef.current) { panRef.current = null; return; }
    if (markMode && markStart.current != null) {
      const a = markStart.current;
      const b = e ? frameAt(e.clientX, e.currentTarget) : a;
      markStart.current = null; setMarkDrag(null);
      const lo = Math.min(a, b), hi = Math.max(a, b);
      const key = String(sliceIndex);
      const hit = curMarks.findIndex(([x, y]) => lo >= x && hi <= y);
      const next = { ...marks };
      if (hi - lo <= 1 && hit >= 0) {                 // a click INSIDE a band removes it
        const rest = curMarks.filter((_, i) => i !== hit);
        if (rest.length) next[key] = rest; else delete next[key];
      } else if (hi > lo) {                            // a drag ADDS one
        // De-duplicate: re-marking the same region (easy to do when checking a slice twice) was storing the
        // band twice, which then double-counted in every reader of corrected_defect_marks.
        const dedup = curMarks.filter(([u, v]) => !(u === lo && v === hi));
        next[key] = [...dedup, [lo, hi]].sort((u, v) => u[0] - v[0]);
      } else {
        return;                                        // a stray click on empty space marks nothing
      }
      pushMarks(next);
      return;
    }
    dragRef.current = false;
  };
  const clearSlice = () => {
    setAnchors((prev) => { const o = cloneMap(prev); o.delete(sliceIndex); return o; });
    setPostAnchors((prev) => { const o = cloneMap(prev); o.delete(sliceIndex); return o; });
  };

  return (
    <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center", gap: 0, position: "relative" }}>
      <span className="text-[11px]" style={{ color: edit ? "#22d3ee" : "var(--c-green)", position: "absolute", top: 0, right: 0,
                                             zIndex: 4, pointerEvents: "none", background: "var(--c-bg)", padding: "0 4px" }}>
        corrected (result){edit ? " — drag the line onto the true edge" : ""}</span>
      {edit && (
        <div style={{ position: "absolute", top: 0, left: 0, zIndex: 5, display: "flex", gap: 4, alignItems: "center",
                      background: "var(--c-bg)", padding: "1px 3px", flexWrap: "wrap", maxWidth: "70%" }}>
          {markMode && curMarks.length > 0 && (
            <button onClick={() => { const n = { ...marks }; delete n[String(sliceIndex)]; pushMarks(n); }}
                    title="Remove every ⚑ mark on this corrected slice"
                    style={{ padding: "2px 8px", borderRadius: 6, fontSize: 11, lineHeight: 1.4, cursor: "pointer",
                             border: "1px solid #ff5db0", background: "rgba(255,93,176,0.14)", color: "#ffc2e0" }}>
              Clear ⚑ ({curMarks.length})</button>
          )}
          {bottom && (
            <button onClick={() => setEditBottom((v) => !v)}
                    title={editBottom
                      ? "Editing the BOTTOM (posterior) line. Click to go back to the top edge."
                      : "Edit the BOTTOM (posterior) line — on a surface-cropped scan the bottom edge is the real one in the cropped frames (the top there is an estimate). Drag the orange line onto the true bottom edge; it saves as ground truth for the bottom edge (folded into the original scan's bottom anchors on Regenerate)."}
                    style={{ padding: "2px 8px", borderRadius: 6, fontSize: 11, lineHeight: 1.4, cursor: "pointer",
                             border: "1px solid #ffaa28", background: editBottom ? "#ffaa28" : "rgba(255,170,40,0.14)",
                             color: editBottom ? "#0b0f14" : "#ffd28a" }}>
              {editBottom ? "● bottom line" : "○ bottom line"}</button>
          )}
          <button onClick={clearSlice} disabled={!(curAnchors?.size) && !(curPostAnchors?.size)}
                  style={{ padding: "2px 8px", borderRadius: 6, fontSize: 11, lineHeight: 1.4, cursor: "pointer",
                           border: "1px solid var(--c-border)", background: "var(--c-surface)", color: "var(--c-text)" }}>Clear slice</button>
          {editedSlices.length > 0 && <span style={{ fontSize: 10, opacity: 0.7 }}>edited: {editedSlices.slice(0, 10).join(", ")}{editedSlices.length > 10 ? "…" : ""}</span>}
          {/* Autosave state, so the reviewer can see their drawing is safe instead of trusting that it is.
              "unsaved" only ever shows in the ~900ms debounce window or after a failed write. */}
          {editedSlices.length > 0 && (
            <span style={{ fontSize: 10, fontWeight: 600,
                           color: saveState === "error" ? "#f87171"
                                : saveState === "saving" ? "var(--c-text-dim)"
                                : dirty ? "#fbbf24" : "#4ade80" }}
                  title={saveState === "error"
                    ? "The drawing could NOT be saved — it is still here on screen; draw another point to retry."
                    : dirty ? "Not written yet — saving shortly."
                    : "Saved to this scan as ground truth (oct_params.corrected_edge_anchors)."}>
              {saveState === "error" ? "⚠ not saved" : saveState === "saving" ? "saving…" : dirty ? "unsaved" : "✓ saved"}
            </span>
          )}
        </div>
      )}
      <div ref={hostRef}
           onWheel={onZoomWheel ? (e) => { e.preventDefault(); onZoomWheel(e.clientX, e.clientY, e.deltaY, hostRef.current?.getBoundingClientRect()); } : undefined}
           style={{ flex: 1, minHeight: 0, width: "100%", display: "flex", alignItems: "center", justifyContent: "center", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "relative",
                      ...(bSized ? { width: bDispW,
                            // same rows-per-pixel as the original pane: a taller corrected canvas shows as a taller image
                            height: Math.round(bDispH * ((origDepthVox && origDepthVox > 1 && depthVox > 1) ? depthVox / origDepthVox : 1)) } : { display: "inline-block", maxHeight: "100%", maxWidth: "100%" }),
                      transform: `translate(${bPan.x}px, ${bPan.y}px) scale(${bZoom}) scaleX(-1)`, transformOrigin: "center center" }}>
          {caseId && <img src={resourceUrl(`/api/case/${caseId}/oct-corrected-slice?slice_index=${sliceIndex}&t=${segVersion}`)}
            alt="corrected" draggable={false}
            style={bSized
              ? { display: "block", width: "100%", height: "100%", objectFit: "fill", imageRendering: "pixelated", filter: filterCss }
              : { display: "block", maxHeight: "100%", maxWidth: "100%", imageRendering: "pixelated", filter: filterCss }} />}
          {edge && nFrames > 1 && depthVox > 1 && (
            <svg viewBox={`0 0 ${nFrames} ${depthVox}`} preserveAspectRatio="none"
                 onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}
                 onAuxClick={(e) => e.preventDefault()} onContextMenu={(e) => e.preventDefault()}
                 style={{ position: "absolute", inset: 0, width: "100%", height: "100%",
                          cursor: edit ? (markMode ? "col-resize" : "row-resize") : "default", touchAction: "none",
                          pointerEvents: edit ? "auto" : "none" }}>
              {/* YOUR TARGET (amber): the served surface after generalize + your drawn-anchor re-pin — where your
                  own corrected-edge drawing says the surface should be. Drawn UNDER the red line so the red
                  detection stays legible, and only when it actually differs (otherwise it would just double the
                  red line). Where amber and red separate, "what you approve" and "what Run uses" differ by that
                  much — the same quantity the endpoint reports as warp_gap. */}
              {/* ⚑ marks: pink bands over the frames the reviewer flagged on THIS corrected slice, plus the one
                  being dragged. Full depth, drawn first so the surface lines stay on top. */}
              {[...curMarks, ...(markDrag ? [[Math.min(...markDrag), Math.max(...markDrag)]] : [])].map((b, i) => (
                <rect key={`mk${i}`} x={b[0]} y={0} width={Math.max(1, b[1] - b[0] + 1)} height={depthVox}
                      fill="rgba(255,93,176,0.22)" stroke="#ff5db0" strokeWidth={0.4}
                      vectorEffect="non-scaling-stroke" />
              ))}
              {target && nFrames > 1 && (() => {
                const segs: string[][] = []; let cur: string[] = [];
                for (let f = 0; f < nFrames; f++) {
                  const y = inBand(f) ? NaN : target[f];
                  // steps with the red line (same toggle): amber vs red is read PER FRAME, so both must
                  // use the same geometry or the gap between them is an artifact of the drawing style.
                  if (Number.isFinite(y)) { if (stairEdge) cur.push(`${f},${y}`, `${f + 1},${y}`); else cur.push(`${f + 0.5},${y}`); }
                  else if (cur.length) { segs.push(cur); cur = []; }
                }
                if (cur.length) segs.push(cur);
                return segs.filter((s) => s.length > 1).map((s, i) => (
                  <polyline key={`tg${i}`} fill="none" stroke="#fbbf24" vectorEffect="non-scaling-stroke"
                            strokeWidth={1.1} opacity={0.85} strokeDasharray="4 3" points={s.join(" ")} />
                ));
              })()}
              {/* (3) the calculated deg-2 best fit (blue). Under the red line so the surface stays legible;
                  thin and unbroken, because it is a reference curve rather than an observation. */}
              {(() => { const q = quadPts(); return q ? (
                <polyline key="quad" fill="none" stroke="#22d3ee" vectorEffect="non-scaling-stroke"
                          strokeWidth={0.9} opacity={0.75} points={q} />) : null; })()}
              {/* BOTTOM (posterior) line, orange — the served posterior carried onto this result, or the reviewer's
                  own bottom drawing where they redrew it. Same stairstep geometry as the red edge. Only on
                  surface-cropped scans (no `bottom` otherwise). */}
              {bottom && (() => {
                const segs: string[][] = []; let cur: string[] = [];
                for (let f = 0; f < nFrames; f++) {
                  const y = inBand(f) ? NaN : postY(f);
                  if (Number.isFinite(y)) { if (stairEdge) cur.push(`${f},${y}`, `${f + 1},${y}`); else cur.push(`${f + 0.5},${y}`); }
                  else if (cur.length) { segs.push(cur); cur = []; }
                }
                if (cur.length) segs.push(cur);
                return segs.filter((s) => s.length > 1).map((s, i) => (
                  <polyline key={`pb${i}`} fill="none" stroke="#ffaa28" vectorEffect="non-scaling-stroke"
                            strokeWidth={editBottom ? 1.5 : 1.1} opacity={editBottom ? 0.95 : 0.7} points={s.join(" ")} />
                ));
              })()}
              {edit && editBottom && curPostAnchors && [...curPostAnchors.entries()].map(([f, d], i) => (
                <line key={`pa${i}`} x1={f + 0.5} y1={d - depthVox / 50} x2={f + 0.5} y2={d + depthVox / 50}
                      stroke="#ffaa28" strokeWidth={1.6} vectorEffect="non-scaling-stroke" opacity={0.95} />
              ))}
              {/* detected / dragged corrected surface (red) — broken over the cropped (zeroed) artifact band */}
              {segPts().map((sg, i) => (
                <polyline key={`ce${i}`} fill="none" stroke="#ff4d4d" vectorEffect="non-scaling-stroke"
                          // IDENTICAL to the left editor's red line (reviewer 2026-09-01: "make sure the appearance
                          // of the lines in the original and the corrected slice are the same"). Same widths, same
                          // opacities, same dirty state — SliceGallery.tsx `ed${i}`. The panel no longer dims itself
                          // when it is not the edit target: which pane you are editing is shown by the toolbar
                          // Original|Corrected toggle, not by making the same surface look different in each pane.
                          strokeWidth={dirty ? 1.4 : 1.0} opacity={dirty ? 0.7 : 0.6} points={sg} />
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
