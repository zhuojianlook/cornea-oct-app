/* No-WebGL 2D slice viewer.
   Shows the in-sidecar-rendered preview PNGs (grayscale slices or the
   segmentation overlay) as plain <img>, so the OCT is viewable in browsers
   without WebGL2 (e.g. the VS Code Simple Browser). */

import { useEffect, useMemo, useRef, useState } from "react";
import { ToggleButton, ToggleButtonGroup, Slider, CircularProgress, Select, MenuItem } from "@mui/material";
import { api, resourceUrl } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import { pxToIjk, brushVoxels } from "../../api/coords";
import { octProposals } from "../../api/lifecycle";
import { usePendingEditStore } from "../../store/pendingEditStore";
import { CorrectedEdgePanel } from "./CorrectedEdgePanel";
import type { PreviewImage } from "../../api/types";

// A preview either carries an inline base64 data_url (segmentation/consensus) or a lazy `src`
// URL loaded on demand (dense context scrub) — resolve `src` to an absolute sidecar URL.
const imgSrc = (im?: PreviewImage | null): string =>
  im ? (im.src ? resourceUrl(im.src) : im.data_url) : "";

type Group = "segmentation" | "context";
const GROUP_LABEL: Record<Group, string> = {
  segmentation: "Segmentation",
  context: "Slices",
};
const ORIENTS = ["axial", "coronal", "sagittal"] as const;

// When embedded as the de-nested "Fix columns" panel (driven by the single top toolbar in
// VolumeCanvas), `fixCols` auto-enters column-marking and hides this panel's own duplicate toggles
// (group/orient/before-after/contrast/blur/scar) — orientation + display filter come from props so the
// ONE top toolbar drives them. Called with NO props on the no-WebGL fallback path (unchanged behaviour).
export function SliceGallery({ fixCols = false, cropStart = false, orientProp, filterCss, showRaw = false, readOnly = false, onToggleRaw }: {
  fixCols?: boolean;
  cropStart?: boolean; // open fix-columns directly in surface-crop mode (auto-detect the apex-cropped frames)
  orientProp?: "axial" | "coronal" | "sagittal";
  filterCss?: string;
  showRaw?: boolean; // fix-cols: show the raw "before" beside the markable corrected "after"
  onToggleRaw?: () => void;  // hide/show that corrected panel from inside the editor toolbar
  readOnly?: boolean; // inspecting an earlier (completed) step → view only; no border edits until rollback
} = {}) {
  const caseId = useCaseStore((s) => s.caseId);
  const caseInfo = useCaseStore((s) => s.caseInfo);
  const openCase = useCaseStore((s) => s.openCase); // refetch caseInfo after a fix-cols re-run (fresh persisted nudges)
  // Iterative-refinement pass count (for the "fix at pass" selector) — from the manifest.
  const octIter = (caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_iter as { passes?: number } | undefined;
  const passCount = Math.max(1, Number(octIter?.passes ?? 1));
  // Which iteration pass the column fix is injected at (per-pass only). null = legacy single re-run.
  const [fixPass, setFixPass] = useState<number | null>(null);
  // Clamp the chosen pass if the case switched to one with fewer passes (else the selector shows an
  // out-of-range value AND the backend would silently skip a never-reached inject pass).
  useEffect(() => {
    if (fixPass != null && fixPass > passCount) setFixPass(passCount > 1 ? passCount : null);
  }, [passCount, fixPass]);
  // #2: the manual depth nudges already baked into the current corrected volume (persisted on the case).
  // A stable JSON signature drives a re-seed only when they actually change (case load / after a re-run),
  // never mid-drag.
  const persistedSig = JSON.stringify(
    ((caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_params as Record<string, unknown> | undefined)
      ?.manual_shifts ?? {});
  const persistedShifts = useMemo(() => {
    const m = new Map<number, number>();
    try {
      for (const [k, v] of Object.entries(JSON.parse(persistedSig) as Record<string, number>)) {
        const f = Number(k), px = Number(v);
        if (Number.isFinite(f) && Number.isFinite(px) && px) m.set(f, Math.round(px));
      }
    } catch { /* none */ }
    return m;
  }, [persistedSig]);
  // Fix-columns "Confirm" anchors: the user drags the red detected border onto the TRUE corneal surface →
  // an ABSOLUTE depth anchor per (slice, frame). They ACCUMULATE across slices. Confirm sends them to the
  // backend, which infers ONE GLOBAL detection band and re-detects the whole volume; scrubbing then shows
  // the new detected border. Persisted in oct_params.border_anchors so they survive reopen + drive the warp.
  // WHICH SCAN THE EDITOR IS CURRENTLY SHOWING. Every piece of edit state is reset against this, NOT against
  // the persisted-correction signature it is seeded from. Keying the resets on the signature meant two
  // consecutive scans that both have no saved corrections produced the identical signature ("{}" === "{}"),
  // the seeding effect never re-ran, and the previous scan's drawn edge / curve / posterior points / crop
  // columns stayed live in the editor — drawn over the new scan and offered for saving by Reject ("41 border
  // pt + 56 bottom pt" pending on a scan whose manifest held none). In a queue that advances by itself, one
  // reviewer's corrections silently landing on the next scan is the worst thing this component can do.
  // Read from the MANIFEST's own id, not the store's caseId, so it can never run ahead of the data it seeds.
  const openCaseKey = ((caseInfo?.case_id as string | undefined) ?? caseId ?? "");
  const persistedAnchorsSig = JSON.stringify(
    ((caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_params as Record<string, unknown> | undefined)
      ?.border_anchors ?? {});
  const persistedAnchors = useMemo(() => {
    const m = new Map<number, Map<number, number>>();
    try {
      for (const [s, frames] of Object.entries(JSON.parse(persistedAnchorsSig) as Record<string, Record<string, number>>)) {
        const si = Number(s); if (!Number.isFinite(si) || !frames) continue;
        const fm = new Map<number, number>();
        for (const [f, d] of Object.entries(frames)) { const fi = Number(f), di = Number(d); if (Number.isFinite(fi) && Number.isFinite(di)) fm.set(fi, Math.round(di)); }
        if (fm.size) m.set(si, fm);
      }
    } catch { /* none */ }
    return m;
  }, [persistedAnchorsSig]);
  // Whole-volume GENERALIZE mode: when set, the backend serves/warps the generalized surface (the learned
  // correction interpolated across ALL slices) instead of the local-band redetect. Read from the manifest so
  // it persists across reopen and the button state reflects it.
  // Re-fetch when the segmentation changes (SAM2/correct/scar re-render previews).
  const segSig = useWorkflowStore((s) => s.segVersion);
  const hintMode = useWorkflowStore((s) => s.hintMode);
  const hintPositive = useWorkflowStore((s) => s.hintPositive);
  const scarHints = useWorkflowStore((s) => s.scarHints);
  const addScarHint = useWorkflowStore((s) => s.addScarHint);

  // When a consensus tab is active the store pins the preview group (the voted
  // map, or a scan warped into the common frame); otherwise we auto-select below.
  const previewGroup = useWorkflowStore((s) => s.previewGroup);

  // manual 2D scar editing
  const scarEditMode = useWorkflowStore((s) => s.scarEditMode);
  const scarErase = useWorkflowStore((s) => s.scarErase);
  const scarBrush = useWorkflowStore((s) => s.scarBrush);
  const scarBusy = useWorkflowStore((s) => s.scarBusy);
  const runScarEdit = useWorkflowStore((s) => s.runScarEdit);
  const wfSet = useWorkflowStore((s) => s.set);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const paintingRef = useRef(false);
  const voxelsRef = useRef<Map<string, [number, number, number]>>(new Map());

  const [group, setGroup] = useState<Group>("context");
  const [orient, setOrient] = useState<(typeof ORIENTS)[number]>("axial");
  const [images, setImages] = useState<PreviewImage[]>([]);
  const [rawImages, setRawImages] = useState<PreviewImage[]>([]);
  const [beforeAfter, setBeforeAfter] = useState(false);
  const [idx, setIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  // "Fix columns" → mark BAD frame-columns with the mouse (click/drag; click again to unmark),
  // then re-run preprocessing on them. Every non-bad column is an anchor (good) by default.
  const [colSel, setColSel] = useState(false);
  const [badCols, setBadCols] = useState<Set<number>>(new Set());
  const [rerunBusy, setRerunBusy] = useState(false);
  const colPaintingRef = useRef(false);
  const colDragModeRef = useRef<"add" | "remove">("add"); // a drag adds or removes (set on press)
  const lastCaseRef = useRef<string | null>(null); // reset slice position only when the case changes
  // #2 fix-columns: mark BAD frames with the mouse, then nudge the marked columns UP/DOWN in depth with
  // the ARROW KEYS to their correct position — a manual ground-truth correction (depth VOXELS, applied
  // LAST in preprocessing). manualShifts holds the ABSOLUTE per-frame depth offset to send; it is seeded
  // from the persisted oct_params.manual_shifts so earlier nudges are never lost.
  const [manualShifts, setManualShifts] = useState<Map<number, number>>(new Map());
  // Re-seed the editable shifts from the persisted set whenever it changes (clears pending after a re-run).
  useEffect(() => { setManualShifts(new Map(persistedShifts)); }, [persistedShifts]);
  // Pending = differs from what's already baked into the displayed volume (drives chips + the re-run enable).
  // Must check BOTH directions: a frame newly nudged/changed (in manualShifts) AND a persisted frame the
  // user dragged back to zero (removed from manualShifts) — else "drag the last nudge to zero" wouldn't
  // register as dirty and the clear would never be sent.
  const pendingFrames = useMemo(() => {
    const s = new Set<number>();
    manualShifts.forEach((v, k) => { if ((persistedShifts.get(k) ?? 0) !== v) s.add(k); });
    persistedShifts.forEach((_v, k) => { if (!manualShifts.has(k)) s.add(k); });
    return s;
  }, [manualShifts, persistedShifts]);
  const shiftsDirty = pendingFrames.size > 0;
  // Display-only image enhancement (does NOT change the data) to aid seeing the corneal border.
  const [enhContrast, setEnhContrast] = useState(false);
  const [enhBlur, setEnhBlur] = useState(false);
  // Preprocessing-steps filmstrip: every intermediate output for the central sagittal slice,
  // shown in an overlay on demand (button or double-click on a slice).
  const [stepsOpen, setStepsOpen] = useState(false);
  const [stepsBusy, setStepsBusy] = useState(false);
  const [steps, setSteps] = useState<{ label: string; data_url?: string; kind?: string; branch?: string; group?: string }[]>([]);
  // Fix-columns border-drag: the DETECTED corneal surface (red) + RANSAC best-fit (blue) for the current
  // sagittal slice as COORDINATE arrays (depth per frame, on the working volume), drawn over the slice so
  // the user DRAGS a frame's surface to where it should be → a per-frame depth nudge (manual_shifts). Auto
  // on in fix-columns; no column selection. x=frame/n_frames, y=depth/depth_vox (depth 0 = top).
  // ALL per-slice border curves (FAST detector), fetched ONCE per pass → scrubbing is an instant client-side
  // lookup instead of a ~258ms per-slice round-trip (the user's "can't wait for the red line" complaint).
  const [allCurves, setAllCurves] = useState<{ edges: number[][]; fits: number[][] } | null>(null);
  // The slice the user SETTLES on is refined to the slower, more ACCURATE (robust) detector + cached here,
  // so the border you actually inspect/drag is the precise one while scrubbing stays smooth.
  const [accurate, setAccurate] = useState<Map<number, { edge: number[]; fit: number[] }>>(new Map());
  const accurateRef = useRef(accurate); accurateRef.current = accurate;
  const [borderBusy, setBorderBusy] = useState(false);
  const borderDragRef = useRef<{ x: number; y: number; moved: boolean; mode: "edit" | "pan" } | null>(null);
  // last (frame, depth) the edge drag painted, so the gap between two sampled pointer events can be filled in
  // (see applyBorderDrag). Null between gestures — a stale point would draw a line from wherever the last
  // drag ended to wherever the next one starts.
  const borderPaintRef = useRef<{ f: number; d: number } | null>(null);
  // Fix-columns ZOOM/PAN — magnify the slice so the border can be corrected precisely. A CSS transform on
  // the panel content; getBoundingClientRect stays transform-aware so the drag→(frame,depth) mapping is
  // unchanged at any zoom. Wheel = zoom-to-cursor; middle/shift-drag = pan; left-drag = edit (unchanged).
  const [bZoom, setBZoom] = useState(1);
  const [bPan, setBPan] = useState({ x: 0, y: 0 });
  const borderHostRef = useRef<HTMLDivElement | null>(null);
  const borderRoRef = useRef<ResizeObserver | null>(null);
  // Live size of the border-editor host → lets us render the B-scan at an INTEGER pixels-per-frame so every
  // frame column is the same width AND pixel-sharp (#1 "uniform AND crisp"); see borderPanel below. A CALLBACK
  // ref attaches the observer exactly when the host node mounts (a [fixCols] effect raced the conditional
  // mount and missed it) and seeds the size synchronously so the first paint is already correctly scaled.
  const [hostSize, setHostSize] = useState({ w: 0, h: 0 });
  const measureHost = (el: HTMLDivElement) => {
    const r = el.getBoundingClientRect();
    setHostSize((s) => (Math.abs(s.w - r.width) < 0.5 && Math.abs(s.h - r.height) < 0.5 ? s : { w: r.width, h: r.height }));
  };
  const setBorderHost = (el: HTMLDivElement | null) => {
    borderHostRef.current = el;
    borderRoRef.current?.disconnect();
    borderRoRef.current = null;
    if (el && typeof ResizeObserver !== "undefined") {
      const ro = new ResizeObserver(() => measureHost(el));
      ro.observe(el);
      borderRoRef.current = ro;
      measureHost(el);
    }
  };
  const resetBorderView = () => { setBZoom(1); setBPan({ x: 0, y: 0 }); centerBorderScroll(); };
  // The scroll container is always wider than the view (the lateral margins exist precisely so there is
  // somewhere to scroll to), and `justify-content: center` would make the left overflow unreachable. So the
  // image is centred by SCROLL POSITION instead: park the viewport in the middle of the extent, which puts
  // the image in the centre with equal margin reachable on either side.
  const centerBorderScroll = () => {
    const el = borderHostRef.current;
    if (!el) return;
    requestAnimationFrame(() => {
      const over = el.scrollWidth - el.clientWidth;
      if (over > 0) el.scrollLeft = over / 2;
    });
  };
  useEffect(() => { resetBorderView(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [caseId, fixCols]);
  // Drop the previous case's steps filmstrip. Unlike the preview lists (which each fetch REPLACES), these
  // are inline base64 PNGs that loadSteps only clears when it is next opened — so a filmstrip viewed on
  // one scan would sit in state, tens of MB, across every following case in a triage run.
  useEffect(() => { setSteps([]); setStepsOpen(false); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [caseId]);
  // Zoom around the cursor, keeping the point under it fixed. `host` is the container whose CENTRE is the
  // transform origin — the LEFT editor's for a left-pane scroll, the CORRECTED pane's for a right-pane scroll —
  // so scrolling over EITHER pane zooms and anchors to what you're pointing at (both panes share bZoom/bPan).
  const zoomBorderAtRect = (clientX: number, clientY: number, factor: number, host: DOMRect | undefined) => {
    setBZoom((z) => {
      const nz = Math.max(1, Math.min(10, z * factor));
      if (nz === z) return z;
      if (nz <= 1.0001) { setBPan({ x: 0, y: 0 }); return 1; }
      if (host) {
        const cx = host.left + host.width / 2, cy = host.top + host.height / 2;
        const ratio = nz / z;
        setBPan((p) => ({ x: p.x + (clientX - cx - p.x) * (1 - ratio), y: p.y + (clientY - cy - p.y) * (1 - ratio) }));
      }
      return nz;
    });
  };
  const zoomBorderAt = (clientX: number, clientY: number, factor: number) =>
    zoomBorderAtRect(clientX, clientY, factor, borderHostRef.current?.getBoundingClientRect());
  // Scroll-to-zoom for the CORRECTED (right) pane — same zoom state, but anchored to the RIGHT pane's own centre.
  const onCorrectedWheel = (clientX: number, clientY: number, deltaY: number, rect: DOMRect | undefined) =>
    zoomBorderAtRect(clientX, clientY, deltaY < 0 ? 1.2 : 1 / 1.2, rect);
  const zoomBorderCentered = (factor: number) => {
    const h = borderHostRef.current?.getBoundingClientRect();
    if (h) zoomBorderAt(h.left + h.width / 2, h.top + h.height / 2, factor);
  };
  // Editable anchor set (seeded from persisted; drag adds; Confirm persists). sliceIdx → frame → depth.
  const [borderAnchors, setBorderAnchors] = useState<Map<number, Map<number, number>>>(new Map());
  const [redetectBusy] = useState(false);   // retained: still gates spinners/disabled states
  const cloneAnchors = (m: Map<number, Map<number, number>>) => { const o = new Map<number, Map<number, number>>(); m.forEach((fm, s) => o.set(s, new Map(fm))); return o; };
  const anchorsToApi = (m: Map<number, Map<number, number>>) => {
    const o: Record<string, Record<string, number>> = {};
    m.forEach((fm, s) => { if (fm.size) { const inner: Record<string, number> = {}; fm.forEach((d, f) => { inner[String(f)] = Math.round(d); }); o[String(s)] = inner; } });
    return o;
  };
  const anchorsSig = (m: Map<number, Map<number, number>>) => [...m.keys()].sort((a, b) => a - b)
    .map((s) => { const fm = m.get(s)!; return fm.size ? s + ":" + [...fm.keys()].sort((a, b) => a - b).map((f) => f + "=" + Math.round(fm.get(f)!)).join(",") : ""; })
    .filter(Boolean).join(";");
  // Re-seed editable anchors from the persisted set whenever it changes (case load / after Confirm).
  useEffect(() => { setBorderAnchors(cloneAnchors(persistedAnchors)); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [persistedAnchorsSig, openCaseKey]);
  const anchorsDirty = anchorsSig(borderAnchors) !== anchorsSig(persistedAnchors);
  const anchorCount = useMemo(() => { let n = 0; borderAnchors.forEach((fm) => { n += fm.size; }); return n; }, [borderAnchors]);
  const setPendingEdit = usePendingEditStore((s) => s.setPending);
  // Before/after: which red line is being edited (original/raw left vs corrected-result right). Shared with the
  // CorrectedEdgePanel + TimelineBar so ONE "Correct & re-run" commits whichever line was drawn.
  const editTarget = usePendingEditStore((s) => s.editTarget);
  const setEditTarget = usePendingEditStore((s) => s.setEditTarget);
  // Border edit MODE (2c): drag the noisy per-frame EDGE (red) or the smooth PARABOLA (blue). In parabola mode
  // a drag adds a point the quadratic must pass through; the curve re-fits live and Confirm uses it EXACTLY.
  const [borderMode, setBorderMode] = useState<"edge" | "parabola">("edge");
  // Parabola points (2c): sliceIdx → frame → depth the quadratic must pass through. The displayed parabola
  // re-fits through (detected edge with these points overriding); Confirm sends it as the EXACT surface.
  const [paraAnchors, setParaAnchors] = useState<Map<number, Map<number, number>>>(new Map());
  const paraCount = useMemo(() => { let n = 0; paraAnchors.forEach((fm) => { n += fm.size; }); return n; }, [paraAnchors]);
  // CUT mode (request 1): drag a TOP (apex/axial) line + LEFT/RIGHT lines marking where the surface leaves the
  // frame; "Re-run with cuts" excludes those from the fit (which extrapolates) + leaves them unwarped → robust.
  const persistedCutSig = JSON.stringify(
    (((caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_params as Record<string, unknown> | undefined)
      ?.surface_cut) ?? {});
  const [cutMode, setCutMode] = useState(false);
  const [cut, setCut] = useState<{ top: number; left: number; right: number }>({ top: 0, left: 0, right: 0 });
  useEffect(() => {
    try { const c = JSON.parse(persistedCutSig) as { top?: number; left?: number; right?: number };
      setCut({ top: Math.round(c.top || 0), left: Math.round(c.left || 0), right: Math.round(c.right || 0) });
    } catch { setCut({ top: 0, left: 0, right: 0 }); }
  }, [persistedCutSig]);
  const cutDragRef = useRef<null | "top" | "left" | "right">(null);
  // SURFACE-CROP mode: mark the B-scan columns whose APEX is cropped (no top surface). "Detect" auto-suggests
  // them; the user verifies/edits; "Confirm & re-run" reconstructs those frames by POSTERIOR CONTINUITY
  // (their visible bottom edge, matched to the non-cropped frames' bottom edge). A STICKY oct_param.
  const persistedCropSig = JSON.stringify(
    (((caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_params as Record<string, unknown> | undefined)
      ?.surface_crop_frames) ?? []);
  const persistedCrop = useMemo(() => {
    try { return new Set((JSON.parse(persistedCropSig) as number[]).map(Number)); } catch { return new Set<number>(); }
  }, [persistedCropSig]);
  const [cropMode, setCropMode] = useState(false);
  const [cropCols, setCropCols] = useState<Set<number>>(new Set());
  const [cropBusy, setCropBusy] = useState(false);
  const [cropCounts, setCropCounts] = useState<Record<string, number>>({});
  // Per-slice surface-crop PREVIEW: the detected bottom (posterior) edge + the reconstructed anterior surface
  // (posterior continuity) for the slice being viewed — so the user sees the guidance the correction is based
  // on, not the failing top-edge detection. Fetched (debounced) as the slice / cropCols change.
  const [cropPreview, setCropPreview] = useState<{ top: number[]; bottom: number[]; recon: number[] } | null>(null);
  const cropPaintRef = useRef<null | "add" | "remove">(null);
  const cropColsSig = useMemo(() => [...cropCols].sort((a, b) => a - b).join(","), [cropCols]);
  // Did the PIPELINE take the surface-crop path on this scan? Shown as a ✓ on the mode button (see below).
  const scAuto = ((caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_iter as
    Record<string, unknown> | undefined)?.stopped === "surface_crop";
  useEffect(() => { setCropCols(new Set(persistedCrop)); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [persistedCrop, openCaseKey]);
  const cropDirty = useMemo(
    () => cropCols.size !== persistedCrop.size || [...cropCols].some((f) => !persistedCrop.has(f)),
    [cropCols, persistedCrop]);
  // #9 CROP REGION mode (distinct from surface-crop above): remove certain FRAME columns (the horizontal axis
  // of the sagittal display = the 101-frame slow axis) over a RANGE of LATERAL slices (the sagittal slice
  // index = the 513 fast axis) — a BOX in the lateral×frame en-face plane, zeroed across depth before SAM2 and
  // recorded so scar-alignment excludes it. SAGITTAL-ONLY. Persisted as oct_params.crop_region.
  const persistedCropRegionSig = JSON.stringify(
    (((caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_params as Record<string, unknown> | undefined)
      ?.crop_region) ?? null);
  const persistedCropRegion = useMemo(() => {
    try {
      const r = JSON.parse(persistedCropRegionSig) as { lateral: [number, number]; frames: number[] } | null;
      return r && Array.isArray(r.lateral) && r.lateral.length === 2 && Array.isArray(r.frames)
        ? { lo: Number(r.lateral[0]), hi: Number(r.lateral[1]), frames: new Set(r.frames.map(Number)) } : null;
    } catch { return null; }
  }, [persistedCropRegionSig]);
  // Crop-approval: the auto de-tilt / off-cornea crop / clipped-apex surface-crop DETECTED but not applied
  // (manifest.oct_proposals). Shown as a pink overlay here, glows the "⊟ Crop region" tab, and SEEDS the
  // editable frame set when the user enters the crop tool with no manual crop yet — so "clicking the crop
  // region lets them manipulate it".
  const proposals = useMemo(() => octProposals(caseInfo?.manifest ?? null), [caseInfo?.manifest]);
  const proposedFrames = useMemo(() => new Set(proposals.frames), [proposals]);
  const [latCropMode, setLatCropMode] = useState(false);
  // ⚑ DEFECT MARKS as a mode of the border editor. Ported from the niivue overlay, which could not work here:
  // that version painted on the niivue canvas and went inert whenever a 2-D overlay was open — and the border
  // editor IS a 2-D overlay, so the button would have been dead the moment it moved. Reuses the crop-column
  // painting instead. Unlike border anchors these carry no depth; they say "this column is wrong", which is
  // worth keeping alongside the optional note for problems the anchors cannot express.
  const [markCols, setMarkCols] = useState<Set<number>>(new Set());
  const [markMode, setMarkMode] = useState(false);
  // Surface-crop has TWO distinct gestures on the same picture: choosing WHICH frames are cropped (column
  // painting) and correcting WHERE the bottom edge is (line dragging). One pointer cannot serve both, so they
  // are separate sub-modes rather than a modifier key nobody would discover.
  const [cropSub, setCropSub] = useState<"cols" | "line">("cols");
  // Manual posterior points: sliceIdx -> frame -> depth. Same shape as border anchors, and they WIN over the
  // detector for those frames (see _crop_reconstruct_slice).
  const [postAnchors, setPostAnchors] = useState<Map<number, Map<number, number>>>(new Map());
  const postCount = useMemo(() => { let n = 0; postAnchors.forEach((fm) => { n += fm.size; }); return n; }, [postAnchors]);
  // Frames the reviewer corrected INDIVIDUALLY. These are evidence about corneal thickness; the bulk
  // shift-drag is not (it anchors every frame at once, so counting it would make the estimate chase the drag).
  const [postManual, setPostManual] = useState<Map<number, Set<number>>>(new Map());
  const [latCropFrames, setLatCropFrames] = useState<Set<number>>(new Set());  // marked frame COLUMNS
  const [latCropLo, setLatCropLo] = useState<number | null>(null);             // lateral range start (slice index)
  const [latCropHi, setLatCropHi] = useState<number | null>(null);             // lateral range end (slice index)
  const [latCropBusy] = useState(false);    // retained: still gates disabled states
  useEffect(() => {
    // Seed from the persisted crop, but NOT while the user is actively editing (latCropMode) — a concurrent
    // manifest change must not wipe their unsaved marks. On confirm, local already matches persisted.
    if (latCropMode) return;
    setLatCropFrames(new Set(persistedCropRegion?.frames ?? []));
    setLatCropLo(persistedCropRegion?.lo ?? null);
    setLatCropHi(persistedCropRegion?.hi ?? null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [persistedCropRegion, openCaseKey]);

  // The edits with NO persisted counterpart to re-seed from, cleared outright when the scan changes. Without
  // this they had no reset path at all: the quadratic's control points and the posterior line's points simply
  // carried from one scan to the next (see openCaseKey).
  useEffect(() => {
    setParaAnchors(new Map());
    setPostAnchors(new Map());
    setPostManual(new Map());
    setMarkCols(new Set());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openCaseKey]);
  // Seed the editable crop from the PROPOSED frames when entering the crop tool with no manual/persisted crop
  // yet, so the auto-detected region is pre-marked and the user can immediately manipulate (add/remove) it.
  const seedFromProposal = () => {
    if (proposedFrames.size === 0) return;
    if (latCropFrames.size > 0 || (persistedCropRegion?.frames.size ?? 0) > 0) return;  // don't clobber real edits
    setLatCropFrames(new Set(proposedFrames));
    if (proposals.cropLateral) { setLatCropLo(proposals.cropLateral[0]); setLatCropHi(proposals.cropLateral[1]); }
  };
  const latCropDirty = useMemo(() => {
    const pf = persistedCropRegion?.frames ?? new Set<number>();
    const framesDiff = latCropFrames.size !== pf.size || [...latCropFrames].some((f) => !pf.has(f));
    return framesDiff || latCropLo !== (persistedCropRegion?.lo ?? null) || latCropHi !== (persistedCropRegion?.hi ?? null);
  }, [latCropFrames, latCropLo, latCropHi, persistedCropRegion]);
  const latCropFrameRanges = useMemo(() => {
    const xs = [...latCropFrames].sort((a, b) => a - b);
    const runs: string[] = []; let s0: number | null = null, prev = -2;
    for (const c of xs) { if (c !== prev + 1) { if (s0 != null) runs.push(prev > s0 ? `${s0}–${prev}` : `${s0}`); s0 = c; } prev = c; }
    if (s0 != null) runs.push(prev > s0 ? `${s0}–${prev}` : `${s0}`);
    return runs;
  }, [latCropFrames]);
  // #9 — tell the viewer (VolumeCanvas) that Crop mode is active so it forces SAGITTAL and disables coronal.
  useEffect(() => { wfSet("cropRegionMode", fixCols && latCropMode); return () => wfSet("cropRegionMode", false); }, [fixCols, latCropMode]);
  // Bumped after we render context previews on demand, to force the fetch effect to
  // re-pull (can't reuse segSig — the auto-select effect depends on it and would loop).
  const [refetchTick, setRefetchTick] = useState(0);

  // Embedded fix-columns: auto-enter marking on the corrected slices (no inner ▥ click needed) and
  // mirror the top toolbar's orientation. The depth-fix workflow lives entirely in the colSel controls.
  useEffect(() => {
    if (!fixCols) return;
    setColSel(true);
    setGroup("context");
    setBeforeAfter(false);
    wfSet("scarEditMode", false);
    // Default to fixing pass 1 (edit the border on the RAW original — the most common + impactful fix).
    if (passCount > 1) setFixPass((p) => (p == null || p > passCount ? 1 : p));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fixCols, passCount]);
  useEffect(() => {
    // Fix-columns anchors are keyed by the SAGITTAL slice index (the backend re-detects on sagittal
    // slices arr[idx]); dragging in another orientation would write mis-indexed anchors. So the border
    // editor is sagittal-only — force it regardless of the incoming 2-D orientation.
    if (fixCols) { setOrient("sagittal"); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fixCols, orientProp]);

  const effectiveGroup = previewGroup ?? group;

  // Auto-select the richest available group (segmentation > slices) so the
  // viewer (and screenshots of it) show the latest result by default. Skipped
  // when a consensus tab pins the group OR in fix-columns mode (which must stay on
  // the CORRECTED "context" slices — otherwise, for a scan that already has SAM2,
  // this races the fixCols effect and flips the group to "segmentation", leaving the
  // corrected panel empty when before/after is combined with Fix-columns).
  useEffect(() => {
    if (!caseId || previewGroup || fixCols) return;
    let cancelled = false;
    (async () => {
      for (const g of ["segmentation", "context"] as Group[]) {
        try {
          const r = await api.json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/${g}`);
          if (cancelled) return;
          if ((r.images || []).length > 0) {
            setGroup(g);
            return;
          }
        } catch {
          /* try next */
        }
      }
      // Nothing rendered yet — generate grayscale context slices so the OCT shows.
      // A new DICOM is converted to NIfTI here (slow), so show the spinner meanwhile.
      try {
        if (!cancelled) setLoading(true);
        await api.json(`/api/case/${caseId}/context-previews`, "POST", JSON.stringify({}));
        // Now that the slices exist, force the fetch effect to re-pull and show them.
        if (!cancelled) {
          setGroup("context");
          setRefetchTick((t) => t + 1);
        }
      } catch {
        /* no volume yet — fine */
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [caseId, segSig, previewGroup, fixCols]);

  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;
    setLoading(true);
    api
      .json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/${effectiveGroup}`)
      .then((r) => {
        if (cancelled) return;
        const imgs = r.images || [];
        setImages(imgs);
        // Reset to a middle slice ONLY when the case changed (a new scan) AND real slices have
        // arrived. Claiming the case on the first (often EMPTY) response would skip centering once
        // the real slices render. For re-renders of the SAME case — re-run preprocessing, SAM2, scar
        // edit, group switch — keep the current frame (safeIdx clamps if the slice count shrank).
        if (lastCaseRef.current !== caseId && imgs.length) {
          lastCaseRef.current = caseId;
          const mid = imgs.filter((i) => i.orientation === orient);
          // Centre on the MIDDLE slice — unless the border editor is about to jump to the ranked
          // most-questionable one. This effect fires when the previews arrive, which is AFTER the auto-jump
          // has run and claimed its once-per-case guard, so it silently overwrote the chosen slice and the
          // scan opened on the middle every time (513 frames -> slice 257). The jump does not re-fire, so it
          // has to be yielded to here instead.
          const willAutoJump = fixCols && orient === "sagittal" && worstSlicesRef.current.length > 0;
          if (mid.length && !willAutoJump) setIdx(Math.floor(mid.length / 2));
        }
      })
      .catch(() => !cancelled && setImages([]))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [caseId, effectiveGroup, segSig, refetchTick]);

  // The pre-correction ("before") slices for the same scan. They exist only once a scan has
  // been preprocessed (the corrected slices then live in the normal context group = "after").
  useEffect(() => {
    setRawImages([]);   // clear first so a previous case's raw can't pair with the new corrected
    if (!caseId) return;
    let cancelled = false;
    api
      .json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/context_raw`)
      .then((r) => !cancelled && setRawImages(r.images || []))
      .catch(() => !cancelled && setRawImages([]));
    return () => { cancelled = true; };
  }, [caseId, segSig]);

  // The 3rd before/after panel overlays for this scan, rendered dense+rotated to match the
  // context slices: "seg" = this scan's own cornea+scar (after SAM2), "cons" = its subgroup
  // consensus (after a per-subgroup consensus build). Empty until those have run.
  const [segImages, setSegImages] = useState<PreviewImage[]>([]);
  const [consImages, setConsImages] = useState<PreviewImage[]>([]);
  const [thirdMode, setThirdMode] = useState<"seg" | "cons">("seg");
  useEffect(() => {
    setSegImages([]);
    setConsImages([]);
    if (!caseId) return;
    let cancelled = false;
    api.json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/context_seg`)
      .then((r) => !cancelled && setSegImages(r.images || [])).catch(() => undefined);
    api.json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/context_cons`)
      .then((r) => !cancelled && setConsImages(r.images || [])).catch(() => undefined);
    return () => { cancelled = true; };
  }, [caseId, segSig]);

  const orientImgs = useMemo(
    () =>
      images
        .filter((i) => i.orientation === orient)
        // DESCENDING by slice_index so scrubbing matches the normal niivue view's direction: niivue slice s ↔
        // array slice (n-1-s) (RAS-canonical flip of the all-negative OCT affine), so ascending array order ran
        // OPPOSITE to niivue ("slice N" showed a different B-scan). Descending → panel position p == niivue slice
        // p. Data is keyed by cur.slice_index (true array index), unchanged.
        .sort((a, b) => Number(b.slice_index ?? 0) - Number(a.slice_index ?? 0)),
    [images, orient],
  );
  const safeIdx = Math.min(idx, Math.max(0, orientImgs.length - 1));
  const cur = orientImgs[safeIdx];

  // "Skip by propagation range": a fix-columns border correction re-detects ±redetect_slice_band neighbouring
  // SLICES (oct_preprocess.py DEFAULT_PARAMS redetect_slice_band = 20), so drawing on one slice fills a band.
  // These ⏮/⏭ buttons jump the slice cursor by exactly that band (mapped to the nearest available preview
  // slice by slice_index, so it's correct even if previews are sub-sampled) → the next slice you draw on sits
  // at the edge of the current correction's reach, giving contiguous coverage with no gaps. Guaranteed to
  // advance ≥1 slice in the requested direction even if the nearest-by-index lands back on the current slice.
  // ── TOOL COLOURS ────────────────────────────────────────────────────────────────────────────────────────
// Each tool's button carries the colour of the thing it edits, so the toolbar reads as a legend for the
// overlay rather than four identical grey buttons over a picture with five coloured curves on it. These MUST
// stay equal to the stroke colours used in the border overlay below; they are the same constants.
//   edge     red    #ff4d4d  the raw detected border you drag
//   parabola green  #39d98a  the shaped quadratic (also the reconstructed surface it produces)
//   crop     orange #ffaa28  the posterior/bottom edge that drives the reconstruction + the marked columns
//   latcrop  blue   #5db0ff  the cropped frame-columns
// Parabola is CYAN, not green: the shaped quadratic and the cyan "surface the correction applies" are the
// same object — one auto-fitted, one hand-shaped — so they are drawn as ONE curve and the tool that edits it
// carries its colour. Green is left to mean only the surface-crop RECONSTRUCTION, which is a different thing.
const MODE_COLOR = { edge: "#ff4d4d", parabola: "#22d3ee", crop: "#ffaa28", latcrop: "#5db0ff",
                     mark: "#ff5db0" } as const;   // pink = the defect-mark bands, as on the niivue overlay
// Floor for an above-window curvature point, in depth rows. Mirrors oct_preprocess DEFAULT_PARAMS
// crop_max_pad — the cap on how far warp_surface_crop_extend will extend the canvas, so the UI cannot ask
// for an apex the reconstruction is unable to deliver.
const PARA_MIN_DEPTH = 160;
// px: how close a shift-drag must come to the estimated bottom position before it latches on.
// Wide enough to be easy to hit, narrow enough that a deliberate placement elsewhere still holds.
const POST_SNAP_PX = 10;
// px on screen: how near the dashed estimate a CLICK must land to read as "use it here".
const POST_CLICK_PX = 12;
// A posterior anchor stored at >= depthVox means ABSENT: the reviewer dragged it off the bottom of the image
// because the posterior edge genuinely leaves the frame there. Encoded as a sentinel rather than a separate
// set so it travels with the anchors through the preview, the commit and the manifest unchanged.
const postAbsent = (d: number, depthVox: number) => d >= depthVox;
const modeBtnSx = (c: string) => ({
  py: 0.3, px: 1.2, fontSize: 11.5, textTransform: "none" as const, letterSpacing: 0.1, whiteSpace: "nowrap" as const,
  color: c, borderColor: c, opacity: 0.75,
  "&.Mui-selected": {
    opacity: 1, color: "#0b0f14", backgroundColor: c, borderColor: c,
    "&:hover": { backgroundColor: c },
  },
});

const PROP_SLICE_BAND = 20;
  const skipBand = (dir: 1 | -1) => {
    if (!orientImgs.length || cur?.slice_index == null) return;
    // orientImgs is now DESCENDING (matches the niivue view), so ⏭ (dir=+1, "next") must go to a LOWER array
    // slice_index → subtract. The position-based fallback (safeIdx + dir) below stays correct as-is.
    const target = Number(cur.slice_index) - dir * PROP_SLICE_BAND;
    let best = safeIdx;
    let bestD = Infinity;
    orientImgs.forEach((im, i) => {
      const d = Math.abs(Number(im.slice_index ?? 0) - target);
      if (d < bestD) { bestD = d; best = i; }
    });
    if (best === safeIdx) best = Math.min(orientImgs.length - 1, Math.max(0, safeIdx + dir));
    setIdx(best);
  };

  // ── MOST PROBLEMATIC SLICE ───────────────────────────────────────────────────────────────────────────
  // An edit is only worth making where the detection is actually questionable, and hunting for that slice by
  // scrubbing 513 of them is the slowest part of the loop. Rank slices by how far the RAW detected edge (red)
  // departs from its own robust fit (cyan): that gap is precisely what the user would drag, and it is high
  // both where detection has gone wrong and where the cornea genuinely leaves the fitted curve — either way,
  // the slice where an edit carries the most information.
  //
  // Scored as the mean of the WORST 15% of per-frame deviations, not the overall RMS: a slice with one
  // spike and a slice with a sustained bad region score very differently under this, and the sustained one is
  // the one worth correcting. A plain max would rank single-frame detector noise top.
  //
  // Costs nothing extra — allCurves is already fetched for instant scrubbing.
  const worstSlices = useMemo(() => {
    if (!allCurves?.edges?.length || !allCurves?.fits?.length) return [] as number[];
    const scored: Array<{ s: number; score: number }> = [];
    for (let s = 0; s < allCurves.edges.length; s++) {
      const e = allCurves.edges[s], fi = allCurves.fits[s];
      if (!e || !fi || e.length !== fi.length || e.length < 8) continue;
      const dev: number[] = [];
      for (let f = 0; f < e.length; f++) {
        const d = Math.abs(e[f] - fi[f]);
        if (Number.isFinite(d)) dev.push(d);
      }
      if (dev.length < 8) continue;
      dev.sort((a, b) => b - a);
      const k = Math.max(1, Math.round(dev.length * 0.15));
      let sum = 0;
      for (let i = 0; i < k; i++) sum += dev[i];
      scored.push({ s, score: sum / k });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.map((x) => x.s);
  }, [allCurves]);
  useEffect(() => { worstSlicesRef.current = worstSlices; }, [worstSlices]);

  // Jump to a TRUE array slice index by nearest available preview (same mapping skipBand uses — previews may
  // be sub-sampled, so the array index is not a position).
  const jumpToSlice = (sliceIndex: number) => {
    if (!orientImgs.length) return;
    let best = safeIdx, bestD = Infinity;
    orientImgs.forEach((im, i) => {
      const d = Math.abs(Number(im.slice_index ?? 0) - sliceIndex);
      if (d < bestD) { bestD = d; best = i; }
    });
    setIdx(best);
  };

  // The number to SHOW the reviewer for a given array slice index. The scrubber counts panel positions, and
  // orientImgs is sorted DESCENDING (see above) so that position p is niivue slice p — meaning the array index
  // and the number on screen differ by a flip. Quoting the array index in a tooltip therefore named a
  // different slice from the one the scrubber read, on the very picture the jump had landed on ("worst is
  // slice 214" while the bar said 299) — which reads as the jump having failed.
  const dispSlice = (s: number): number => {
    if (!orientImgs.length) return s;
    let best = 0, bestD = Infinity;
    orientImgs.forEach((im, i) => {
      const d = Math.abs(Number(im.slice_index ?? 0) - s);
      if (d < bestD) { bestD = d; best = i; }
    });
    return best + 1;
  };

  const autoWorstRef = useRef<string | null>(null);
  // mirrors worstSlices for the preview-fetch callback above, which closes over stale state
  const worstSlicesRef = useRef<number[]>([]);

  // Before/after is only meaningful on the working "context" slices, once the pre-correction
  // ("before") snapshot exists. The current slice (cur) is the corrected "after"; match the
  // raw "before" to it by slice index (raw + corrected share the same geometry).
  const canBeforeAfter = effectiveGroup === "context" && rawImages.length > 0;
  const rawCur = canBeforeAfter && cur
    ? rawImages.find((i) => i.orientation === orient && i.slice_index === cur.slice_index)
    : undefined;
  const showBeforeAfter = beforeAfter && canBeforeAfter && !!cur;

  // 3rd panel (shown beside before/after): toggles between this scan's own segmentation and its
  // subgroup consensus, whichever are available. Matched to the corrected slice by index.
  const hasSeg = segImages.length > 0;
  const hasCons = consImages.length > 0;
  const canThird = effectiveGroup === "context" && (hasSeg || hasCons);
  const effThird: "seg" | "cons" = thirdMode === "cons" && hasCons ? "cons" : hasSeg ? "seg" : "cons";
  const thirdList = effThird === "cons" ? consImages : segImages;
  const thirdCur = showBeforeAfter && cur
    ? thirdList.find((i) => i.orientation === orient && i.slice_index === cur.slice_index)
    : undefined;

  // "Mark bad columns" → re-run preprocessing on those frames. A scan is eligible the moment it's
  // PREPROCESSED (context_raw exists) — NO SAM2 needed. Frame count comes from the raw snapshot's
  // sagittal preview, so the button is discoverable on ANY view (Slices OR Segmentation); clicking
  // it switches to the corrected sagittal view where the column band is shown.
  const nFrames = rawImages.find((i) => i.orientation === "sagittal")?.source_height ?? 0;
  // Mirror the SAGITTAL frame axis so this view agrees with every other view of the same volume (the
  // fix-columns editor, the before/after strip and the axial gallery all run reversed vs the array).
  // Applied to the panel CONTAINER; the three screen-x → source-fraction mappings invert to match.
  const mirrorSag = orient === "sagittal";
  // Depth voxel count = the sagittal preview's SOURCE width (rgb is frames×depth pre-rotation), used to
  // convert a vertical screen drag → a depth-voxel shift for #2 drag-to-correct.
  const depthVox = rawImages.find((i) => i.orientation === "sagittal")?.source_width ?? 0;
  const canMarkColumns = !previewGroup && rawImages.length > 0 && nFrames > 1 && !showBeforeAfter;

  // Which pass to fix → its INPUT is what we detect + draw the border on (pass 1 = the RAW original; pass k
  // = pass k-1's output). Editing the detection on the INPUT improves that pass's result — editing the
  // border on the downstream/corrected result is meaningless.
  // The anchor re-detect always operates on the RAW volume (pass 1): the marched surface is built on raw
  // and a single warp flattens raw to it. (The "fix at pass" selector is for the legacy column path only.)
  const borderPass = fixCols ? 1 : (passCount > 1 ? (fixPass ?? 1) : 1);
  // Open on the WORST slice, once per case+pass, and only in the sagittal border editor where the ranking
  // means anything. Declared here rather than beside worstSlices because it reads borderPass, which is
  // defined just above — referencing it earlier in the component body is a temporal-dead-zone throw, not a
  // hoisting convenience. Never re-fires while the user scrubs (that would yank the view out from under
  // them), and never overrides a case they have already started editing (anchorsDirty).
  useEffect(() => {
    if (!fixCols || orient !== "sagittal" || !caseId || !worstSlices.length || anchorsDirty) return;
    // The preview list must be loaded, or jumpToSlice no-ops. Claiming the ref BEFORE a successful jump made
    // the whole feature silently do nothing: the effect fired while orientImgs was still empty, the jump
    // returned early, and the guard then blocked every retry. Claim it only once the jump can actually land.
    if (!orientImgs.length) return;
    const key = `${caseId}#${borderPass}`;
    if (autoWorstRef.current === key) return;
    autoWorstRef.current = key;
    jumpToSlice(worstSlices[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fixCols, orient, caseId, borderPass, worstSlices, anchorsDirty, orientImgs.length]);
  const passInputLabel = borderPass <= 1 ? "original (raw)" : `pass ${borderPass - 1} output`;
  // The input IMAGE the border is drawn over: raw (pass 1) or the prior pass's preview (pass > 1).
  const [passInputImg, setPassInputImg] = useState<string | null>(null);
  useEffect(() => {
    if (!fixCols || borderPass <= 1 || !caseId || cur == null) { setPassInputImg(null); return; }
    let cancelled = false;
    api.json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/context_iter${borderPass - 1}`)
      .then((r) => {
        if (cancelled) return;
        const im = (r.images || []).find((i) => i.orientation === orient && i.slice_index === cur.slice_index);
        setPassInputImg(im ? imgSrc(im) : null);
      })
      .catch(() => !cancelled && setPassInputImg(null));
    return () => { cancelled = true; };
  }, [fixCols, borderPass, caseId, cur?.slice_index, orient, segSig]);
  // Fix-columns editor: use the NATIVE-resolution B-scan (no physical-aspect nearest-neighbour upscaling) so we
  // can render uniform, pixel-sharp frame columns (#1). Non-fixCols legacy path keeps the prior pass preview.
  const inputSrc = fixCols
    ? (caseId && cur ? resourceUrl(`/api/case/${caseId}/oct-border-slice?slice_index=${cur.slice_index}&border_pass=${borderPass}`) : null)
    : (borderPass > 1 ? passInputImg : (rawCur ? imgSrc(rawCur) : null));
  // Self-heal a transient B-scan load failure. On scan-advance (Approve → next) the sidecar can be momentarily
  // busy (committing the just-approved scan / warming the next scan's caches), so this one <img> can fail to
  // load while the JSON curve for the same frame succeeds — leaving the fix-columns pane showing the red/cyan
  // lines floating over a BLANK B-scan, with no recovery because <img> never retries itself. So on error we
  // re-request a few times with a cache-busting suffix (fix-columns http URL only — blob srcs can't take a
  // query param). The counter resets whenever the underlying src changes, so each B-scan gets a fresh budget.
  const [imgRetry, setImgRetry] = useState(0);
  useEffect(() => { setImgRetry(0); }, [inputSrc]);
  const inputSrcR = (fixCols && inputSrc && imgRetry > 0)
    ? `${inputSrc}${inputSrc.includes("?") ? "&" : "?"}_r=${imgRetry}`
    : inputSrc;
  // Tracks whether THIS src's B-scan has painted yet. First-open of a scan computes its surface / surface-crop
  // caches on the sidecar (25 s+ uncached), so the PNG can be in flight for a while — during which the panel
  // would otherwise show the red/cyan lines floating over black, which reads as "broken". Gate a "loading"
  // affordance on this instead. onLoad is the fast path; the ref-check covers a cached image that finished
  // loading before React bound the handler (onLoad would never fire, stranding the overlay over a good image).
  const bImgRef = useRef<HTMLImageElement | null>(null);
  const [bImgLoaded, setBImgLoaded] = useState(false);
  useEffect(() => {
    const el = bImgRef.current;
    setBImgLoaded(!!(el && el.complete && el.naturalWidth > 0));
  }, [inputSrcR]);

  const borderSliceIdx = cur?.slice_index ?? null;
  // Seed from the marks already stored for THIS slice, so an existing mark can be seen and amended rather
  // than silently replaced by whatever is drawn next.
  const storedMarks = useMemo(() => {
    const m = (caseInfo?.manifest as Record<string, unknown> | undefined)?.defect_marks;
    return Array.isArray(m) ? (m as Array<{ orient: string; slice: number; cols: number[]; tag?: string }>) : [];
  }, [caseInfo]);
  useEffect(() => {
    if (borderSliceIdx == null) { setMarkCols(new Set()); return; }
    const cols = storedMarks.filter((k) => k.orient === "sagittal" && k.slice === borderSliceIdx)
      .flatMap((k) => k.cols || []);
    setMarkCols(new Set(cols));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [borderSliceIdx, storedMarks]);
  const markDirty = useMemo(() => {
    if (borderSliceIdx == null) return false;
    const stored = new Set(storedMarks.filter((k) => k.orient === "sagittal" && k.slice === borderSliceIdx)
      .flatMap((k) => k.cols || []));
    return stored.size !== markCols.size || [...markCols].some((c) => !stored.has(c));
  }, [markCols, storedMarks, borderSliceIdx]);
  // 1) Fetch ALL slices' borders ONCE (fast detector) so scrubbing is instant. Re-fetched on pass change
  //    or after a re-detect/preprocess (segVersion). x=frame/n_frames, y=depth/depth_vox.
  useEffect(() => {
    if (!fixCols || !caseId) { setAllCurves(null); setAccurate(new Map()); return; }
    let cancelled = false;
    setBorderBusy(true); setAllCurves(null); setAccurate(new Map());
    api.json<{ edges: number[][]; fits: number[][] }>(
      `/api/case/${caseId}/oct-border-curves-all`, "POST", JSON.stringify({ border_pass: borderPass }))
      .then((r) => { if (!cancelled) setAllCurves({ edges: r.edges || [], fits: r.fits || [] }); })
      .catch(() => !cancelled && setAllCurves(null))
      .finally(() => !cancelled && setBorderBusy(false));
    return () => { cancelled = true; };
  }, [fixCols, caseId, borderPass, segSig]);
  // 2) When the user SETTLES on a slice (~250ms), refine it to the accurate per-slice detector + cache it,
  //    so the border you inspect/drag is precise while scrubbing stays smooth (fast curves).
  useEffect(() => {
    if (!fixCols || !caseId || borderSliceIdx == null || !allCurves) return;
    const idx = borderSliceIdx;
    if (accurateRef.current.has(idx)) return;
    const t = setTimeout(() => {
      api.json<{ edge: number[]; fit: number[] }>(
        `/api/case/${caseId}/oct-border-curve`, "POST", JSON.stringify({ slice_index: idx, border_pass: borderPass }))
        .then((r) => { if (r.edge) setAccurate((prev) => prev.has(idx) ? prev : new Map(prev).set(idx, { edge: r.edge, fit: r.fit })); })
        .catch(() => { /* keep the fast curve */ });
    }, 250);
    return () => clearTimeout(t);
  }, [fixCols, caseId, borderSliceIdx, borderPass, allCurves]);
  // Clear any stale preview the instant the SLICE or CASE changes, so the prior slice's orange/green curves
  // can't paint on the new slice during the 200ms debounce. NOT keyed on cropCols → editing columns keeps the
  // current preview visible (smooth) until the refetch lands.
  useEffect(() => { setCropPreview(null); }, [caseId, borderSliceIdx]);
  // SURFACE-CROP preview: fetch the current slice's bottom (posterior) edge + reconstructed anterior for the
  // current cropCols (debounced, so dragging columns doesn't spam). The bottom edge is the guidance; the recon
  // is what the re-run will apply (and can leave the top of the frame where the apex is cropped).
  useEffect(() => {
    // Fetched whenever this scan HAS a surface crop, not only while the crop tool is selected — the markings
    // have to stay visible from every mode (reviewer: "the markings need [to be] seen no matter what option
    // is currently selected"), and they cannot be drawn without their preview data. Scans with no crop still
    // make no request, so nothing extra is fetched for the 269 scans that never had one.
    // …and whenever the reviewer has DRAWN posterior points, even on a scan with no cropped columns: leaving
    // this out made the bottom line they had just placed disappear the moment they switched to Edge, because
    // the only thing holding it on screen was a preview that was no longer being fetched.
    if (!fixCols || !caseId || borderSliceIdx == null || !(cropMode || cropCols.size > 0 || postCount > 0)) {
      setCropPreview(null); return;
    }
    const idx = borderSliceIdx;
    let cancelled = false;
    const t = setTimeout(() => {
      api.json<{ top: number[]; bottom: number[]; recon: number[] }>(
        `/api/case/${caseId}/oct-surface-crop/preview`, "POST",
        JSON.stringify({ slice_index: idx,
                         surface_crop_frames: cropColsSig ? cropColsSig.split(",").map(Number) : [],
                         // live posterior points, so the preview shows the line being dragged rather than
                         // the detector's version of it (preview == what a re-run would use)
                         crop_post_anchors: anchorsToApi(postAnchors) }))
        .then((r) => { if (!cancelled) setCropPreview({ top: r.top || [], bottom: r.bottom || [], recon: r.recon || [] }); })
        .catch(() => { if (!cancelled) setCropPreview(null); });
    }, 200);
    return () => { cancelled = true; clearTimeout(t); };
  }, [fixCols, cropMode, caseId, borderSliceIdx, cropColsSig, cropCols.size, postCount, postAnchors]);
  // The border for the CURRENT slice: the accurate (settled) curve if we have it, else the instant fast one.
  const curEdge = (borderSliceIdx != null ? (accurate.get(borderSliceIdx)?.edge ?? allCurves?.edges[borderSliceIdx]) : null) ?? null;
  const curFit = (borderSliceIdx != null ? (accurate.get(borderSliceIdx)?.fit ?? allCurves?.fits[borderSliceIdx]) : null) ?? null;

  // ── "THERE IS NO ANTERIOR SURFACE ON THIS FRAME" ─────────────────────────────────────────────────────
  // The reviewer says so by dragging the red line to the image floor, which stores the absent sentinel
  // instead of a depth. Once that has been done across the whole frame there is no top edge left to
  // describe, and the quadratic fit must go with it: three points is the MINIMUM that defines a quadratic,
  // and fitting one to fewer (or to nothing) would draw — and, on Reject, commit — a smooth arc asserting a
  // surface the reviewer has just stated is not there. The flatten aligns frames to that surface, so the
  // assertion is not cosmetic.
  // Note this keys on the reviewer's EXPLICIT marks, never on a failed detection: a surface-cropped scan
  // whose apex sits above the window still has flank edges and still needs the curve, which is the whole
  // reason the quadratic tool has headroom.
  const sliceEdgeGone = (s: number, n: number): boolean => {
    const fm = borderAnchors.get(s);
    if (!fm || n <= 0) return false;
    let present = 0;
    for (let f = 0; f < n; f++) {
      const a = fm.get(f);
      if (!(a != null && postAbsent(a, depthVox))) present++;
    }
    return present < 3;
  };
  const surfaceGone = borderSliceIdx != null && curEdge != null && sliceEdgeGone(borderSliceIdx, nFrames);

  // Drag the detected border (red) onto where the TRUE surface is → an ABSOLUTE depth ANCHOR for that
  // (slice, frame). Red follows the cursor (WYSIWYG); anchored frames turn PINK. Anchors accumulate across
  // slices; Confirm infers ONE global detection band from them and re-detects the whole volume. Dragging a
  // frame back to its detected depth removes its anchor. (NOT a manual_shift — anchors steer DETECTION.)
  const applyBorderDrag = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    // sagittal-only: anchors are keyed by the sagittal slice index (see the fix-columns orient effect)
    if (orient !== "sagittal" || !curEdge || nFrames <= 1 || depthVox <= 1 || borderSliceIdx == null) return;
    const r = svg.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return;
    // frameAtX, not a local copy of the formula: it is the one conversion that reads the overlay's ACTUAL
    // horizontal span (see frameAtBorder). Identical to the old expression whenever there is no lateral
    // headroom, which is every mode but quadratic fit.
    const frame = Math.round(frameAtX(clientX, r));
    if (frame < 0 || frame >= nFrames || frame >= curEdge.length) return;
    // ABOVE-WINDOW allowed (on a surface-cropped scan the anterior IS outside the window), and BELOW-FLOOR
    // means ABSENT: sometimes there is no top surface on a frame at all, and dragging the line to the bottom
    // is how the reviewer says so. Clamping that to depth-1 would assert a real surface lying along the floor
    // — the same phantom-edge mistake the posterior had, and just as damaging, since the flatten aligns
    // frames to this surface.
    const rawY = depthAtY(clientY, r);
    const depth = rawY >= depthVox - 1 ? depthVox : Math.round(Math.max(vbTop, rawY));
    const s = borderSliceIdx;
    // CONTINUOUS along the drag, not one frame per pointer event. Pointer events are sampled (~60/s), so a
    // sweep across 101 frames used to land on a fraction of them and leave the rest untouched — the line
    // came out spiky where the gesture was smooth, and "drag the WHOLE edge to the floor" (= this frame has
    // no anterior surface at all) was not achievable by hand at any realistic speed. Fill every frame between
    // the previous sample and this one.
    const prevPt = borderPaintRef.current;
    borderPaintRef.current = { f: frame, d: depth };
    setBorderAnchors((prev) => {
      const mm = cloneAnchors(prev);
      const fm = mm.get(s) ?? new Map<number, number>();
      // dragging onto the detected edge (±0.5) clears the anchor; otherwise set the absolute true depth
      const put = (f: number, d: number) => {
        if (f < 0 || f >= nFrames || f >= curEdge.length) return;
        if (Math.abs(d - curEdge[f]) < 1) fm.delete(f); else fm.set(f, d);
      };
      if (prevPt && Math.abs(prevPt.f - frame) > 1) {
        const span = frame - prevPt.f, step = span > 0 ? 1 : -1;
        // The ABSENT sentinel is a flag, not a depth: interpolating between it and a real depth would invent
        // in-between surfaces halfway down the image. When either end is absent the swept frames take the
        // value under the cursor, so sweeping along the floor marks the whole run absent.
        const lerp = !postAbsent(prevPt.d, depthVox) && !postAbsent(depth, depthVox);
        for (let f = prevPt.f + step; f !== frame; f += step) {
          put(f, lerp ? Math.round(prevPt.d + (depth - prevPt.d) * ((f - prevPt.f) / span)) : depth);
        }
      }
      put(frame, depth);
      if (fm.size) mm.set(s, fm); else mm.delete(s);
      return mm;
    });
  };
  // 2c: least-squares degree-2 fit through the detected edge with the user's parabola points overriding their
  // frames → the "clean quadratic" the user shapes by dragging. Returns the curve sampled per frame.
  // Least-squares polynomial through (x,y) of the given degree. Normal equations + Gaussian elimination with
  // partial pivoting — general in the degree, because the fit's ORDER now depends on how many control points
  // the reviewer has placed (see fitQuadratic).
  const polyFit = (xs: number[], ys: number[], deg: number): { co: number[]; x0: number; sx: number } | null => {
    const m = deg + 1;
    // CENTRE AND SCALE x first. With raw frame indices (0..100) a cubic needs sums of x^6 ~ 1e12, and the
    // normal-equations matrix becomes badly enough conditioned that a cubic through four points came out
    // several px away from them — the fit looked wrong when the maths was right. Mapping x to roughly
    // [-1, 1] keeps every power near unity and makes the interpolation exact.
    const x0 = xs.reduce((a, b) => a + b, 0) / xs.length;
    const sx = Math.max(1e-6, Math.max(...xs.map((x) => Math.abs(x - x0))));
    const us = xs.map((x) => (x - x0) / sx);
    const pw = new Array(2 * deg + 1).fill(0);
    for (let i = 0; i < us.length; i++) {
      let xp = 1;
      for (let k = 0; k <= 2 * deg; k++) { pw[k] += xp; xp *= us[i]; }
    }
    const A: number[][] = Array.from({ length: m }, () => new Array(m + 1).fill(0));
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < m; c++) A[r][c] = pw[r + c];
      let acc = 0;
      for (let i = 0; i < us.length; i++) acc += ys[i] * Math.pow(us[i], r);
      A[r][m] = acc;
    }
    for (let col = 0; col < m; col++) {
      let piv = col;
      for (let r = col + 1; r < m; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
      if (Math.abs(A[piv][col]) < 1e-12) return null;
      const tmp = A[col]; A[col] = A[piv]; A[piv] = tmp;
      for (let r = 0; r < m; r++) {
        if (r === col) continue;
        const f = A[r][col] / A[col][col];
        for (let c = col; c <= m; c++) A[r][c] -= f * A[col][c];
      }
    }
    return { co: A.map((row, i) => row[m] / A[i][i]), x0, sx };   // coefficients in u = (x-x0)/sx
  };

  const fitQuadratic = (edge: number[], pts?: Map<number, number>): number[] => {
    const n = edge.length;
    if (n < 3) return edge.slice();
    // CONTROL POINTS WIN once there are three, and the curve PASSES THROUGH them: the degree is
    // (number of points - 1), capped at cubic. Three points give the quadratic it always was; four give a
    // cubic, which is the only way one smooth curve can honour four arbitrary points. Fitting a QUADRATIC to
    // four points instead (least squares) left gaps of up to 30 px between the curve and the handles that
    // were supposed to be driving it — which reads as the tool ignoring you.
    // Past four points it stays cubic and becomes least-squares again: the reviewer's own note is that these
    // corrections are approximate, so averaging many of them is right, whereas a high-order polynomial
    // threaded exactly through all of them would oscillate between them.
    const cps = pts && pts.size >= 3 ? [...pts.entries()].sort((a, b) => a[0] - b[0]) : null;
    if (cps) {
      const cxs = cps.map(([x]) => x), cys = cps.map(([, y]) => y);
      const fit = polyFit(cxs, cys, Math.min(cps.length - 1, 3));
      if (fit) {
        return Array.from({ length: n }, (_v, x) => {
          const u = (x - fit.x0) / fit.sx;
          let acc = 0, xp = 1;
          for (let k = 0; k < fit.co.length; k++) { acc += fit.co[k] * xp; xp *= u; }
          return acc;
        });
      }
    }
    // No control points: the plain quadratic through the detected edge, with any points overriding it.
    const xs: number[] = [], ys: number[] = [];
    for (let x = 0; x < n; x++) { xs.push(x); ys.push(pts?.get(x) ?? edge[x]); }
    const f2 = polyFit(xs, ys, 2);
    if (!f2) return edge.slice();
    return Array.from({ length: n }, (_v, x) => {
      const u = (x - f2.x0) / f2.sx;
      return f2.co[0] + f2.co[1] * u + f2.co[2] * u * u;
    });
  };
  // Parabola-mode drag: set the depth the quadratic must pass through at this frame (no auto-clear; dragging
  // a point shapes the smooth curve).
  // HANDLE-DRIVEN QUADRATIC. A quadratic is fixed by three points; a fourth makes the fit least-squares.
  // The editor behaves as four handles:
  // grabbing one moves THAT handle (in both frame and depth) and the arc re-solves through the three. Without
  // a grab identity, a drag wrote a new point at every frame it crossed — you ended up with dozens of points
  // and a least-squares blur instead of an arc you positioned.
  // THE WHOLE HANDLE SET AS IT STOOD WHEN THE PRESS LANDED, plus which one was grabbed and where it was held.
  // Identity used to be the frame KEY, which is also the handle's x position — so moving a handle meant
  // delete-then-insert, and every safeguard around that (a stale key, a collision, the count invariant) could
  // add or remove a DIFFERENT handle than the one under the cursor. Handles the reviewer had carefully placed
  // were destroyed and re-created along the drag path, which is how three of them ended up bunched together
  // off-image. Dragging now rebuilds the set from this snapshot with only the grabbed INDEX replaced: the
  // others are copied through untouched, and the count cannot change.
  const paraDragRef = useRef<{ pts: Array<[number, number]>; idx: number; df: number; dd: number } | null>(null);
  // SHIFT-drag in surface-crop mode acts on the WHOLE thing rather than one frame: the entire bottom line in
  // "bottom line" sub-mode, a contiguous column RANGE in "columns". Recorded on press so the gesture cannot
  // change meaning halfway through a drag.
  const cropShiftRef = useRef<{ y: number; base: Map<number, number>; snapDy: number | null } | null>(null);
  // A press in the bottom-line tool that lands ON the dashed estimate and does NOT turn into a drag = "accept
  // the estimate for this column". Recorded on press, applied on release, so it never fires mid-drag.
  const postClickRef = useRef<{ frame: number; onEstimate: boolean } | null>(null);
  // ── HEADROOM ABOVE THE IMAGE ─────────────────────────────────────────────────────────────────────────
  // A surface-cropped cornea has its apex ABOVE the captured window — that is the definition of the case —
  // so the curve the user needs to specify passes through depths the image does not contain. With the SVG
  // pinned to the image (inset:0, viewBox 0 0 nFrames depthVox) there was no coordinate space up there and
  // the drag clamped at row 0, making the apex impossible to place. Extend the overlay UPWARD by `headroom`
  // rows in PARABOLA and SURFACE-CROP modes, where a negative depth is meaningful; other modes keep the
  // exact old geometry so nothing else shifts.
  // Sized off crop_max_pad (120 rows, the cap on how far the pipeline will extend the canvas) so anything
  // the reconstruction can actually deliver is reachable, with a floor for shallow volumes.
  // Headroom applies to EDGE mode as well: on a clipped scan the anterior surface really is above the
  // captured window, so the reviewer must be able to say so with the edge tool, not only with the curve.
  const headroom = (borderMode === "parabola" || borderMode === "edge" || cropMode)
    ? Math.max(60, Math.min(160, Math.round(depthVox * 0.28))) : 0;
  const vbTop = -headroom;
  const vbH = depthVox + headroom;
  // LATERAL headroom, for the same reason as the vertical one: the curve's shape at the acquisition edges is
  // set by where it is HEADING outside the captured frames, and with control points confined to [0, nFrames)
  // the only way to steer an end was to place a point exactly on it — which pins the curve rather than aiming
  // it. Quadratic-fit mode only: every other tool marks things that exist IN the image, and widening their
  // coordinate space would just allow marks where there is no data.
  // Gated on the quadratic tool being the ACTIVE one, not merely the last border tool chosen. Picking Surface
  // crop / Mark / Crop artifact sets its own flag and leaves borderMode alone, so after any use of the
  // quadratic fit the overlay stayed three frame-spans wide underneath those tools — the drag then landed
  // about a third of the panel away from the cursor, which is the "bottom edge is not where my pointer is"
  // that only appeared once the quadratic had been touched.
  const latHead = (borderMode === "parabola" && !cropMode && !markMode && !latCropMode && !cutMode)
    ? Math.max(40, nFrames) : 0;   // a full frame-span each side
  const vbLeft = -latHead;
  const vbW = nFrames + 2 * latHead;
  // screen x -> frame in the overlay's own space. The panel is mirrored (scaleX(-1)), hence the 1 - fx.
  // May return an OUT-OF-RANGE frame; callers that mark real image content clamp it, the curve does not.
  const frameAtX = (clientX: number, r: DOMRect): number =>
    vbLeft + (1 - (clientX - r.left) / Math.max(1, r.width)) * vbW - 0.5;
  // screen y -> depth, in the overlay's own coordinate space. MUST be used by every pointer handler: the
  // overlay no longer spans exactly the image, so the naive (clientY-top)/height*depthVox is wrong whenever
  // headroom is non-zero and would silently offset every drag.
  const depthAtY = (clientY: number, r: DOMRect) => vbTop + ((clientY - r.top) / r.height) * vbH;
  // Seed handles from the CURRENT curve so the arc starts exactly where the existing fit is and the first
  // drag is a correction rather than a jump. Placed at 15/50/85% of the frame span: far enough apart that the
  // quadratic is well-conditioned, inset from the acquisition edges where detection is least trustworthy.
  const seedParaPts = (edge: number[], sample: (f: number) => number): Map<number, number> => {
    const n = edge.length;
    // Seeded by sampling THE FUNCTION THAT DRAWS THE LINE, not a parallel re-derivation of it. Passing the
    // fit array separately meant a length mismatch (or a missing fit) silently fell back to a locally
    // re-fitted quadratic, leaving the handles several px off the curve they were supposed to be sitting on
    // before anything had been touched. One source, no drift.
    const base = Array.from({ length: n }, (_v, f) => sample(f));
    const m = new Map<number, number>();
    // THREE handles — extreme left, centre, extreme right. Three points define a quadratic EXACTLY, so the
    // curve passes through every one of them; a fourth was tried and reverted, because a quadratic cannot
    // honour four arbitrary points and the least-squares compromise left visible gaps between the curve and
    // the handles driving it. The ends are included deliberately: insetting them would leave the one part
    // that cannot be grabbed exactly where the "edges don't follow the curvature" complaints live.
    // Handles may be dragged ABOVE the image (negative depth) — on a clipped scan the apex genuinely is
    // outside the captured window.
    for (const frac of [0, 0.5, 1]) {
      const f = Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
      const y = Number.isFinite(base[f]) ? base[f] : edge[f];
      if (Number.isFinite(y)) m.set(f, Math.round(y));
    }
    return m;
  };
  // A press grabs a handle ONLY if it actually lands on one, measured in SCREEN pixels so the target matches
  // what the cross looks like. Previously any press grabbed the nearest handle outright once the full set
  // existed, so clicking the image to look at something yanked a cross across the panel; and the grabbed
  // handle jumped to the pointer rather than moving with it, so catching one slightly off-centre snapped it.
  // Both are recorded here: the handle keeps its offset from the cursor (df/dd) and follows the drag.
  const PARA_GRAB_PX = 22;
  const grabParaPoint = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    paraDragRef.current = null;
    if (orient !== "sagittal" || !curEdge || borderSliceIdx == null || surfaceGone) return;
    const r = svg.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return;
    // The handles on screen right now — the reviewer's own if placed, otherwise the seeded ones they can see.
    const live = curParaHandles;
    if (!live || !live.size) return;
    const pts: Array<[number, number]> = [...live.entries()].sort((a, b) => a[0] - b[0]);
    const pf = frameAtX(clientX, r), pd = depthAtY(clientY, r);
    const pxPerFrame = r.width / Math.max(1, vbW), pxPerRow = r.height / Math.max(1, vbH);
    let idx = -1, bestPx = Infinity;
    pts.forEach(([f, d], i) => {
      const px = Math.hypot((f - pf) * pxPerFrame, (d - pd) * pxPerRow);
      if (px < bestPx) { bestPx = px; idx = i; }
    });
    if (idx < 0 || bestPx > PARA_GRAB_PX) return;   // not on a cross → the press leaves the curve alone
    paraDragRef.current = { pts, idx, df: pts[idx][0] - pf, dd: pts[idx][1] - pd };
  };
  const applyParaDrag = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    if (orient !== "sagittal" || !curEdge || nFrames <= 1 || depthVox <= 1 || borderSliceIdx == null) return;
    if (surfaceGone) return;   // no anterior surface on this frame → nothing for a curve to describe
    const r = svg.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return;
    // OUT-OF-RANGE frames are allowed, and are the point: a control point to the LEFT or RIGHT of the image
    // aims the curve's end instead of pinning it. Bounded by the lateral headroom so a point cannot be lost
    // off-panel.
    const grab = paraDragRef.current;
    if (!grab) return;                       // the press did not land on a cross → nothing to move
    // The handle keeps the offset it was caught at, so it tracks the pointer instead of snapping to it.
    const frame = Math.round(Math.max(vbLeft, Math.min(nFrames - 1 + latHead, frameAtX(clientX, r) + grab.df)));
    // NEGATIVE depths are allowed: on a surface-cropped scan the apex sits above the captured window, so
    // the curvature through it can only be specified outside the image. Clamped to the headroom, not to 0.
    const depth = Math.round(Math.max(vbTop, Math.min(depthVox - 1, depthAtY(clientY, r) + grab.dd)));
    setParaAnchors((prev) => {
      const mm = cloneAnchors(prev);
      // REBUILT FROM THE PRESS-TIME SNAPSHOT, every time. Only grab.idx is given a new position; the other
      // handles are copied through exactly as they were when the press landed, so an unedited cross cannot
      // move, cannot be dropped, and no fourth one can appear — none of which the old delete-and-reinsert
      // could promise, because a handle's identity was its own x position.
      const taken = new Set<number>();
      grab.pts.forEach(([f], i) => { if (i !== grab.idx) taken.add(f); });
      // A handle may pass another, but two cannot share a frame: this is a frame-keyed map, so equal frames
      // would silently merge two handles into one and leave a pair, which cannot define a quadratic.
      let put = frame;
      if (taken.has(put)) {
        const lo = vbLeft, hi = nFrames - 1 + latHead;
        for (let k = 1; k < vbW; k++) {
          if (frame + k <= hi && !taken.has(frame + k)) { put = frame + k; break; }
          if (frame - k >= lo && !taken.has(frame - k)) { put = frame - k; break; }
        }
      }
      const fm = new Map<number, number>();
      grab.pts.forEach(([f, d], i) => { if (i === grab.idx) fm.set(put, depth); else fm.set(f, d); });
      mm.set(borderSliceIdx, fm);
      return mm;
    });
  };
  // Anchor ONLY on a deliberate DRAG, never on the press — a click/tap (or sub-threshold jitter) must NOT
  // drop a stray anchor where you merely touched the line. We start anchoring once the pointer moves past a
  // small threshold from the press point; dragging then reshapes the border (and a stretch dragged back onto
  // the detected edge auto-clears, so it merges cleanly).
  // frame index under the pointer, from the border SVG's on-screen rect (x spans nFrames).
  const frameAtBorder = (clientX: number, svg: Element): number | null => {
    const r = svg.getBoundingClientRect();
    if (r.width <= 0) return null;
    // viewBox x spans [vbLeft, vbLeft+vbW); frame f occupies the band [f, f+1) → floor maps a screen x to its
    // column. It must read the ACTUAL span: in quadratic-fit mode the overlay is widened by a frame-span on
    // each side, and assuming a bare nFrames there put every pointer mapping out by that factor.
    return Math.max(0, Math.min(nFrames - 1,
      Math.floor(vbLeft + (1 - (clientX - r.left) / r.width) * vbW)));
  };
  const paintCrop = (clientX: number, svg: Element) => {
    const f = frameAtBorder(clientX, svg);
    if (f == null) return;
    setCropCols((prev) => {
      const next = new Set(prev);
      if (cropPaintRef.current === "remove") next.delete(f); else next.add(f);
      return next;
    });
  };
  // #9 crop region: add/remove a FRAME column to the box (drag-paint, reuses cropPaintRef since crop and
  // surface-crop modes are mutually exclusive).
  const applyPostDrag = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    if (orient !== "sagittal" || nFrames <= 1 || depthVox <= 1 || borderSliceIdx == null) return;
    const r = svg.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return;
    const frame = Math.round(frameAtX(clientX, r));   // reads the overlay's real span (see frameAtBorder)
    if (frame < 0 || frame >= nFrames) return;
    // Dragged BELOW the image floor = "no bottom edge on this frame". The posterior does not always span the
    // full width — it can exit through the bottom — and clamping to depthVox-1 turned that into a flat line
    // pinned along the floor, which the reconstruction then treated as a real edge. Anything at or past the
    // floor is stored as the absent sentinel instead. Above the image is still clamped: the posterior cannot
    // be above the window.
    const raw = depthAtY(clientY, r);
    const depth = raw >= depthVox - 1 ? depthVox : Math.round(Math.max(0, raw));
    // CONTINUOUS along the drag, exactly as the anterior edge is (see applyBorderDrag). Writing only the
    // frame under each sampled pointer event left holes THROUGH the swept run — frames the reviewer had
    // visibly dragged over kept their old depth, so the line came out zig-zagging between the new position
    // and the old one. That is what "the line doesn't go where my pointer is" looks like: the placement is
    // pixel-accurate (measured at 0.1 px), it is just not applied to every frame the drag crossed.
    const prevPt = borderPaintRef.current;
    borderPaintRef.current = { f: frame, d: depth };
    const filled: number[] = [];
    setPostAnchors((prev) => {
      const mm = cloneAnchors(prev);
      const fm = mm.get(borderSliceIdx) ?? new Map<number, number>();
      if (prevPt && Math.abs(prevPt.f - frame) > 1) {
        const span = frame - prevPt.f, step = span > 0 ? 1 : -1;
        // the ABSENT sentinel is a flag, not a depth — never interpolate through it
        const lerp = !postAbsent(prevPt.d, depthVox) && !postAbsent(depth, depthVox);
        for (let f = prevPt.f + step; f !== frame; f += step) {
          if (f < 0 || f >= nFrames) continue;
          fm.set(f, lerp ? Math.round(prevPt.d + (depth - prevPt.d) * ((f - prevPt.f) / span)) : depth);
          filled.push(f);
        }
      }
      fm.set(frame, depth); mm.set(borderSliceIdx, fm);
      return mm;
    });
    setPostManual((prev) => {
      const mm = new Map(prev);
      const set = new Set(mm.get(borderSliceIdx) ?? []);
      // the swept frames count as individually placed too — the reviewer dragged over them, so they are
      // evidence about corneal thickness in exactly the way the frame under the cursor is
      set.add(frame); filled.forEach((f) => set.add(f));
      mm.set(borderSliceIdx, set);
      return mm;
    });
  };
  const paintMark = (clientX: number, svg: Element) => {
    const f = frameAtBorder(clientX, svg);
    if (f == null) return;
    setMarkCols((prev) => {
      const next = new Set(prev);
      if (cropPaintRef.current === "remove") next.delete(f); else next.add(f);
      return next;
    });
  };
  const paintLatCrop = (clientX: number, svg: Element) => {
    const f = frameAtBorder(clientX, svg);
    if (f == null) return;
    setLatCropFrames((prev) => {
      const next = new Set(prev);
      if (cropPaintRef.current === "remove") next.delete(f); else next.add(f);
      return next;
    });
  };
  const onBorderDown = (e: React.PointerEvent<SVGSVGElement>) => {
    e.preventDefault(); (e.target as Element).setPointerCapture?.(e.pointerId);
    // A new gesture starts fresh — never bridge from where the last one ended. Must be here, ABOVE the
    // crop/cut branches: they return early, so clearing it further down missed every posterior-line drag.
    borderPaintRef.current = null;
    // Before/after: when editing the CORRECTED (right) line, the raw (left) line is view-only — pan/zoom still
    // work, but no edit — so the reviewer edits ONE line at a time and the single "Correct & re-run" is
    // unambiguous. (The two corrections still compose; this only gates which one you're drawing right now.)
    if (showRaw && editTarget !== "original") {
      borderDragRef.current = { x: e.clientX, y: e.clientY, moved: false, mode: "pan" }; return;
    }
    if (cropMode && cropSub === "line" && e.shiftKey && !readOnly && e.button !== 1) {
      // SHIFT moves the WHOLE bottom line. Only in the line sub-mode: column painting is already a plain
      // click-and-hold drag and needed no modifier — adding one there just made an ordinary action feel
      // conditional. Middle-button still pans, so panning is not lost.
      // Snapshot the WHOLE line as it stands at press. Basing each move on the live line instead made every
      // pointer event re-apply the offset to the already-moved curve, so the line ran away under the cursor.
      const snap = new Map<number, number>();
      for (let f = 0; f < nFrames; f++) {
        const v = postY(f);
        if (Number.isFinite(v)) snap.set(f, v);
      }
      // The snap target is computed ONCE, here. Recomputing it per move would let it drift toward wherever
      // the line currently is — the detent would follow the cursor and never actually catch anything.
      let snapDy: number | null = null;
      if (postThickness != null && snap.size) {
        const deltas: number[] = [];
        snap.forEach((b0, f) => {
          const want = edgeY(f) + postThickness;
          if (Number.isFinite(want) && Number.isFinite(b0)) deltas.push(want - b0);
        });
        if (deltas.length) { deltas.sort((a, b) => a - b); snapDy = deltas[Math.floor(deltas.length / 2)]; }
      }
      cropShiftRef.current = { y: e.clientY, base: snap, snapDy };
      borderDragRef.current = { x: e.clientX, y: e.clientY, moved: false, mode: "edit" };
      return;
    }
    if (cropMode && cropSub === "line") {   // drag the ORANGE posterior edge onto the true bottom
      if (readOnly || e.button === 1 || e.shiftKey) { borderDragRef.current = { x: e.clientX, y: e.clientY, moved: false, mode: "pan" }; return; }
      // did this press land ON the dashed estimate? (click = adopt it here; drag = place by hand, as before)
      const rr = e.currentTarget.getBoundingClientRect();
      const ff = frameAtBorder(e.clientX, e.currentTarget);
      if (ff != null && postThickness != null && rr.height > 0) {
        const want = edgeY(ff) + postThickness;
        const px = Math.abs(depthAtY(e.clientY, rr) - want) * (rr.height / Math.max(1, vbH));
        postClickRef.current = { frame: ff, onEstimate: px <= POST_CLICK_PX };
      } else {
        postClickRef.current = null;
      }
      borderDragRef.current = { x: e.clientX, y: e.clientY, moved: false, mode: "edit" };
      return;
    }
    if (cropMode) {   // crop mode: drag to add/remove cropped frame-columns (pan with shift/middle or readOnly)
      if (readOnly || e.button === 1 || e.shiftKey) { borderDragRef.current = { x: e.clientX, y: e.clientY, moved: false, mode: "pan" }; return; }
      const f = frameAtBorder(e.clientX, e.currentTarget);
      cropPaintRef.current = (f != null && cropCols.has(f)) ? "remove" : "add";
      paintCrop(e.clientX, e.currentTarget);
      return;
    }
    if (cutMode) return;   // cut-line elements own their pointerdown; an empty-area press does nothing here
    if (markMode) {        // ⚑ defect marks: drag to add/remove wrong frame-columns
      if (readOnly || e.button === 1 || e.shiftKey) { borderDragRef.current = { x: e.clientX, y: e.clientY, moved: false, mode: "pan" }; return; }
      const f = frameAtBorder(e.clientX, e.currentTarget);
      cropPaintRef.current = (f != null && markCols.has(f)) ? "remove" : "add";
      paintMark(e.clientX, e.currentTarget);
      return;
    }
    if (latCropMode) {     // #9 crop region: drag to add/remove FRAME columns (pan with shift/middle or readOnly)
      if (readOnly || e.button === 1 || e.shiftKey) { borderDragRef.current = { x: e.clientX, y: e.clientY, moved: false, mode: "pan" }; return; }
      const f = frameAtBorder(e.clientX, e.currentTarget);
      cropPaintRef.current = (f != null && latCropFrames.has(f)) ? "remove" : "add";
      paintLatCrop(e.clientX, e.currentTarget);
      return;
    }
    // middle-button OR shift+left = PAN (so you can move around while zoomed); plain left = edit the border.
    // readOnly (inspecting an earlier completed step) → PAN only; the border can't be edited until rollback.
    const mode: "edit" | "pan" = (readOnly || e.button === 1 || e.shiftKey) ? "pan" : "edit";
    borderDragRef.current = { x: e.clientX, y: e.clientY, moved: false, mode };
    // Quadratic-fit mode: decide WHICH of the three handles this press owns, before any movement. Selecting on
    // press (not on move) is what makes the handles feel grabbed rather than redrawn.
    if (mode === "edit" && borderMode === "parabola") grabParaPoint(e.clientX, e.clientY, e.currentTarget);
  };
  const onBorderMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (cropMode && cropPaintRef.current) { paintCrop(e.clientX, e.currentTarget); return; }
    if (markMode && cropPaintRef.current) { paintMark(e.clientX, e.currentTarget); return; }
    if (latCropMode && cropPaintRef.current) { paintLatCrop(e.clientX, e.currentTarget); return; }
    if (cutDragRef.current) {   // dragging a cut line
      const r = e.currentTarget.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0) return;
      if (cutDragRef.current === "top") {
        const d = Math.round(Math.max(0, Math.min(depthVox - 1, depthAtY(e.clientY, r))));
        setCut((c) => ({ ...c, top: d }));
      } else {
        const f = Math.round(Math.max(0, Math.min(nFrames - 1, (1 - (e.clientX - r.left) / r.width) * nFrames)));
        setCut((c) => {
          if (cutDragRef.current === "left") {        // keep left < right with ≥5 in-frame columns between
            const rightEff = c.right > 0 ? c.right : nFrames - 1;
            return { ...c, left: Math.min(f, Math.max(0, rightEff - 5)) };
          }
          return { ...c, right: Math.max(f, Math.min(nFrames - 1, c.left + 5)) };
        });
      }
      return;
    }
    const d = borderDragRef.current;
    if (!d) return;
    if (d.mode === "pan") {
      setBPan((p) => ({ x: p.x + (e.clientX - d.x), y: p.y + (e.clientY - d.y) }));
      d.x = e.clientX; d.y = e.clientY; return;
    }
    if (!d.moved) {
      if (Math.hypot(e.clientX - d.x, e.clientY - d.y) < 4) return;   // ignore click jitter
      d.moved = true;
    }
    if (cropShiftRef.current) {
      const sh = cropShiftRef.current;
      const r = e.currentTarget.getBoundingClientRect();
      if (r.height <= 0 || borderSliceIdx == null) return;
      if (cropSub === "line") {
        // translate the ENTIRE detected bottom line by the vertical drag — anchoring every frame at
        // (its current line depth + dy), so the whole curve moves rigidly instead of one point at a time.
        let dy = ((e.clientY - sh.y) / r.height) * vbH;
        // SNAP to the estimated position — the corrected top edge plus the measured thickness. dyTarget is the
        // translation that best puts the line there; within POST_SNAP_PX the drag latches onto it, so the
        // reviewer can place the line exactly where the geometry says it belongs rather than by eye. Outside
        // that window the drag is untouched, so a deliberate placement elsewhere is never overridden.
        if (sh.snapDy != null && Math.abs(dy - sh.snapDy) <= POST_SNAP_PX) dy = sh.snapDy;
        setPostAnchors((prev) => {
          const mm = cloneAnchors(prev);
          const fm = new Map<number, number>();
          sh.base.forEach((b0, f) => {
            if (Number.isFinite(b0)) fm.set(f, Math.round(Math.max(0, Math.min(depthVox - 1, b0 + dy))));
          });
          if (fm.size) mm.set(borderSliceIdx, fm);
          return mm;
        });
      }
      return;
    }
    if (cropMode && cropSub === "line") applyPostDrag(e.clientX, e.clientY, e.currentTarget);
    else if (borderMode === "parabola") applyParaDrag(e.clientX, e.clientY, e.currentTarget);
    else applyBorderDrag(e.clientX, e.clientY, e.currentTarget);
  };
  const onBorderUp = () => {
    // CLICK ON THE ESTIMATE → adopt it for that column. Checked BEFORE borderDragRef is cleared, since
    // "click or drag?" is exactly what that ref records. Deliberately NOT counted as manual evidence for the
    // thickness estimate: accepting the estimate and then treating it as a measurement of itself would let
    // the line drift a little further every time it was clicked.
    const pc = postClickRef.current;
    if (pc && pc.onEstimate && !borderDragRef.current?.moved && postThickness != null && borderSliceIdx != null) {
      const want = Math.round(Math.max(0, Math.min(depthVox - 1, edgeY(pc.frame) + postThickness)));
      setPostAnchors((prev) => {
        const mm = cloneAnchors(prev);
        const fm = mm.get(borderSliceIdx) ?? new Map<number, number>();
        fm.set(pc.frame, want); mm.set(borderSliceIdx, fm);
        return mm;
      });
    }
    postClickRef.current = null;
    borderDragRef.current = null; cutDragRef.current = null; cropPaintRef.current = null;
    borderPaintRef.current = null;
    paraDragRef.current = null;   // release the arc handle, so the next press picks its own
    cropShiftRef.current = null;  // end a shift gesture with the drag that started it
  };
  const onBorderWheel = (e: React.WheelEvent) => { e.preventDefault(); zoomBorderAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.2 : 1 / 1.2); };

  // #2 (merged): once columns are marked BAD, the ARROW KEYS nudge the whole marked set UP/DOWN in depth
  // to its correct position (↓ = deeper = +depth voxel, ↑ = -; Shift = ×5). Each marked frame's absolute
  // offset accumulates in manualShifts; on re-run a nudged frame is manually positioned (manual_shifts)
  // and a marked-but-un-nudged frame is auto-interpolated (force_columns).
  // CAPTURE phase + stopImmediatePropagation: the slice <img> isn't focusable, but the SLICE SLIDER (MUI)
  // and the niivue canvas underneath BOTH grab arrow keys (scrolling the slice) before a bubble-phase
  // window listener would see them — that was the "arrows scroll the slice instead of nudging" bug. A
  // capture listener on window runs FIRST and stops the event before either can scroll. We still defer to
  // genuine TEXT entry (so typing isn't hijacked); the range slider IS intercepted on purpose.
  useEffect(() => {
    if (!colSel) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
      if (badCols.size === 0 || depthVox < 2) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      const type = (el as HTMLInputElement | null)?.type;
      if (tag === "TEXTAREA" || (tag === "INPUT" && type !== "range")) return; // don't steal arrows from text fields
      e.preventDefault();
      e.stopImmediatePropagation(); // beat the MUI slice slider + niivue's own arrow-key slice nav
      // Step is in DEPTH VOXELS. The depth axis is ~640 voxels tall, so a 1-voxel nudge is sub-pixel on
      // screen (looks like nothing happened). Use a visible default (5 vox ≈ several px) with Shift for
      // bigger jumps — corrections are typically tens of voxels anyway.
      // Fine control: 1 image-pixel (1 depth voxel) per press, per the user's "1 pixel at a time"; Shift =
      // coarse (10) for big moves. The numeric "↕N vox" readout + ghost make even a 1-voxel move legible.
      const step = (e.shiftKey ? 10 : 1) * (e.key === "ArrowDown" ? 1 : -1); // ↓ deeper (+), ↑ shallower (−)
      setManualShifts((prev) => {
        const next = new Map(prev);
        badCols.forEach((f) => {
          const abs = (next.get(f) ?? 0) + step;
          if (abs) next.set(f, abs); else next.delete(f); // keep the map zero-free (matches persisted)
        });
        return next;
      });
    };
    window.addEventListener("keydown", onKey, true); // capture
    return () => window.removeEventListener("keydown", onKey, true);
  }, [colSel, badCols, depthVox]);

  // Map a pointer position to a FRAME index. The same frame is a vertical COLUMN in the sagittal
  // view (horizontal axis = frames) and a horizontal ROW in the coronal view (vertical axis =
  // frames, flipped by the display flipud) — so a bad frame can be marked from whichever view
  // shows it best (coronal often makes a misaligned frame's jagged left/right border obvious).
  const frameAt = (clientX: number, clientY: number): number | null => {
    const img = imgRef.current;
    if (!img || nFrames < 2) return null;
    const rect = img.getBoundingClientRect();
    // Edge basis: the frame under the cursor is floor(fraction × nFrames), matching the band span
    // [f/nFrames,(f+1)/nFrames] drawn above — so click and highlight land on the same texel. (#1)
    if (orient === "sagittal") {
      // getBoundingClientRect is transform-aware, so this fraction is in DISPLAY space; the panel is
      // scaleX(-1)-mirrored (see correctedPanel), so invert it to get the source frame fraction.
      const ffx0 = (clientX - rect.left) / rect.width;
      const ffx = mirrorSag ? 1 - ffx0 : ffx0;
      return ffx < 0 || ffx > 1 ? null : Math.min(nFrames - 1, Math.max(0, Math.floor(ffx * nFrames)));
    }
    if (orient === "coronal") {
      const ffy = (clientY - rect.top) / rect.height;
      return ffy < 0 || ffy > 1 ? null : Math.min(nFrames - 1, Math.max(0, Math.floor((1 - ffy) * nFrames))); // frames run bottom→top
    }
    return null;
  };
  const paintColAt = (clientX: number, clientY: number) => {
    const f = frameAt(clientX, clientY);
    if (f == null) return;
    const r = 1; // ±1 frame brush
    const touched: number[] = [];
    for (let k = Math.max(0, f - r); k <= Math.min(nFrames - 1, f + r); k++) touched.push(k);
    setBadCols((p) => {
      const next = new Set(p);
      if (colDragModeRef.current === "add") touched.forEach((x) => next.add(x));
      else touched.forEach((x) => next.delete(x)); // click-again-to-deselect
      return next;
    });
  };

  // Fix-columns "Confirm": send the accumulated anchors → the backend MARCHES a tilt-aware re-detection of
  // the whole raw volume, caches the surface + persists the anchors, then we refetch the case + bump
  // segVersion so the border-curve fetch re-pulls the RE-DETECTED surface for the current slice. Scrubbing
  // then shows the new detection everywhere; Run flattens to the SAME surface, so preview == result.
  // Confirm with no anchors clears it (revert to auto).
  // The exact payload "Confirm border" would POST, built from whatever is currently drawn. Extracted so the
  // review loop can commit a correction on Reject WITHOUT the user pressing Confirm — and so the two paths can
  // never diverge into "what you confirmed" vs "what got recorded".
  const buildBorderPayload = (): { anchors: Record<string, Record<string, number>>; parabola: boolean;
                                   parabolaSlices: string[] } => {
    // EVERYTHING DRAWN, not just whatever tool happens to be selected.
    //
    // This used to branch on borderMode: in quadratic-fit mode it sent ONLY the shaped curves, and in edge
    // mode ONLY the point drags. So a reviewer who corrected the edge and then left the toolbar on Quadratic
    // fit committed a curve and silently discarded their edge anchors — the pink ticks stayed on screen while
    // the payload contained none of them, and the button's "(101 border pt)" gave no hint that a different
    // set of their corrections was being dropped. Which tool is selected is a statement about what you are
    // editing NOW, never about which of your corrections are real.
    //
    // The two kinds mean different things and are kept apart: a shaped curve IS the surface for its slice
    // (exact, seed window 0), while point drags say roughly where the edge is (approximate, default window).
    // A slice carrying both is reported as NOT exact — it contains approximate points, and pinning those to
    // the pixel would read in more than the reviewer said.
    {
      const out: Record<string, Record<string, number>> = {};
      const exact: string[] = [];
      paraAnchors.forEach((pts, sIdx) => {
        const e = accurate.get(sIdx)?.edge ?? allCurves?.edges[sIdx];
        if (!e || !pts.size || sliceEdgeGone(sIdx, e.length)) return;
        const q = fitQuadratic(e, pts);
        const inner: Record<string, number> = {};
        for (let f = 0; f < q.length; f++) {
          inner[String(f)] = Math.round(Math.max(-PARA_MIN_DEPTH, Math.min(depthVox - 1, q[f])));
        }
        out[String(sIdx)] = inner;
        exact.push(String(sIdx));
      });
      // Point drags overlay the curve on their own frames: an explicit drag at a frame is a statement about
      // THAT frame and outranks a curve fitted through handles elsewhere.
      Object.entries(anchorsToApi(borderAnchors)).forEach(([sKey, fm]) => {
        if (out[sKey]) {
          Object.assign(out[sKey], fm);
          const i = exact.indexOf(sKey);
          if (i >= 0) exact.splice(i, 1);        // mixed slice → no longer exact
        } else {
          out[sKey] = { ...fm };
        }
      });
      return { anchors: out, parabola: exact.length > 0, parabolaSlices: exact };
    }
  };

  // Publish whatever is currently drawn so "✗ Reject → next" can commit it without a Confirm. Runs on every
  // change to either anchor set; cleared when nothing is drawn, so an emptied editor cannot commit a stale
  // correction. Keyed by case, and the store re-checks that key on take — the queue advances by itself, so a
  // pending edit MUST NOT be able to land on whatever scan happens to be open when the flush runs.
  useEffect(() => {
    if (!caseId || !fixCols || orient !== "sagittal") { setPendingEdit(null); return; }
    const { anchors, parabola, parabolaSlices } = buildBorderPayload();
    const nSlices = Object.keys(anchors).length;
    const nPoints = Object.values(anchors).reduce((a, m) => a + Object.keys(m).length, 0);
    // Crop marks ride along, so "Confirm & re-run" is not needed to keep them either. Only published when
    // they DIFFER from what is already persisted — re-committing an unchanged set on every rejection would
    // rewrite the manifest for nothing and make surface_crop_mode "manual" on scans the user never touched.
    const cropFrames = cropDirty ? [...cropCols].sort((a, b) => a - b) : null;
    const cropRegion = latCropDirty && latCropFrames.size > 0
      ? { lateral: [latCropLo ?? 0, latCropHi ?? latCropLo ?? 0] as [number, number],
          frames: [...latCropFrames].sort((a, b) => a - b) }
      : null;
    // ⚑ defect marks for the CURRENT slice ride along too, so a marked column is recorded by Reject exactly
    // like a border drag. Only when changed — otherwise every rejection would rewrite the mark list.
    const defectCols = (markDirty && borderSliceIdx != null)
      ? { slice: borderSliceIdx, cols: [...markCols].sort((a, b) => a - b) } : null;
    const postApi = postCount > 0 ? anchorsToApi(postAnchors) : null;
    const hasWork = nPoints > 0 || cropFrames !== null || cropRegion !== null || defectCols !== null
      || postApi !== null;
    setPendingEdit(hasWork
      ? { caseId, anchors, parabola, parabolaSlices, nSlices, nPoints, bordersDirty: anchorsDirty,
          cropFrames, cropRegion, defectCols, postAnchors: postApi }
      : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId, fixCols, orient, borderMode, anchorCount, paraCount, paraAnchors, borderAnchors, anchorsDirty,
      cropDirty, cropColsSig, latCropDirty, latCropFrames, latCropLo, latCropHi,
      markDirty, markCols, borderSliceIdx, postCount, postAnchors]);

  // SMOOTH the already-corrected volume: apply the guarded post-hoc smoothing passes (axial_consistency +
  // frame_boundary) to the corrected output, removing the residual slice-to-slice jitter the manual
  // provided_edges warp left (inter-slice smoothing is disabled on that path). Never-worse / preserves the
  // corrected depths. Drops the segmentation (geometry shifted → re-run SAM2).
  // Request 1: re-run the DEFAULT preprocessing with the clipped surfaces cut off (sent in params.surface_cut).
  const rerunWithCut = async () => {
    if (!caseId) return;
    setRerunBusy(true);
    try {
      await api.json(`/api/case/${caseId}/oct-preprocess`, "POST",
        JSON.stringify({ params: { surface_cut: { top: cut.top, left: cut.left, right: cut.right } } }));
      await openCase();
      wfSet("segVersion", segSig + 1);
    } catch {
      /* surfaced via the spinner stopping */
    } finally {
      setRerunBusy(false);
    }
  };
  // Surface-crop: AUTO-DETECT the cropped B-scan columns (apex above the window) for the user to verify/edit.
  const detectCrop = async () => {
    if (!caseId) return;
    setCropBusy(true);
    try {
      const r = await api.json<{ frames: number[]; counts: Record<string, number>; selected: number[] }>(
        `/api/case/${caseId}/oct-surface-crop/detect`, "POST", JSON.stringify({}));
      setCropCounts(r.counts || {});
      // union the auto-suggested set with anything already confirmed, so a re-detect never silently drops the
      // user's persisted/selected frames.
      setCropCols(new Set([...(r.frames || []).map(Number), ...persistedCrop]));
    } catch {
      /* surfaced via the spinner stopping */
    } finally {
      setCropBusy(false);
    }
  };
  // Surface-crop: re-run preprocessing reconstructing the marked frames by posterior continuity (bottom-edge
  // guidance). Sends the confirmed frame set; an empty set clears the crop (plain auto preprocess).
  // #9: re-run preprocessing with the marked BOX removed (the frame columns over the lateral-slice range,
  // zeroed across depth before SAM2). Empty frames / no range → clears the crop. Drops the segmentation; the
  // box is recorded crop-aware. The lateral range defaults to the WHOLE volume if the user marked columns but
  // never set a range (i.e. crop those columns on every slice).
  // Mark the current sagittal slice as the start/end of the lateral-slice RANGE the cropped frame-columns
  // apply to (each sagittal slice = one lateral index). "end" pairs with the last "start".
  // Open directly in surface-crop mode when launched from the toolbar's "Detect surface crop" (cropStart),
  // and auto-detect the cropped frames once. Switching cropStart back off returns to the normal border editor.
  useEffect(() => {
    if (!fixCols) return;
    if (cropStart) {
      setCropMode(true); setCutMode(false);
      if (cropCols.size === 0 && Object.keys(cropCounts).length === 0) void detectCrop();
    } else {
      setCropMode(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fixCols, cropStart]);
  const rerunColumns = async () => {
    // A marked frame the user has NUDGED (in manualShifts) is manually positioned (manual_shifts, applied
    // last) — it must NOT also be auto-interpolated, or the nudge would be relative to a re-interpolated
    // base instead of what the user saw. So force_columns = marked frames WITHOUT a manual nudge.
    const forced = [...badCols].filter((f) => !manualShifts.has(f));
    const hasForced = forced.length > 0;
    if (!caseId) return;
    // fix-columns: Run flattens the volume to the CONFIRMED re-detected surface (use_redetect) — the cached
    // marched surface the scrub preview drew, so preview == result. The button is only enabled when the case
    // has persisted anchors with no un-confirmed drags. Non-fix-columns keeps the legacy "needs a change" guard.
    if (!fixCols && !hasForced && !shiftsDirty) return;
    setRerunBusy(true);
    try {
      // fix-columns Run → apply the confirmed re-detected surface (single warp). Otherwise the legacy paths:
      // iterative scan injects the column fix at the chosen pass; single-pass scan does the targeted re-run.
      const body: Record<string, unknown> = fixCols
        ? { use_redetect: true }
        : (passCount > 1 && fixPass && hasForced)
          ? { inject_pass: fixPass, force_columns: forced, good_columns: [] }
          : hasForced
            ? { force_columns: forced, good_columns: [], max_iterations: 1 }
            // Shifts-only re-run (no columns marked this session): OMIT force_columns so the backend's
            // persisted set carries through unchanged. Sending [] would CLEAR a prior column fix (the
            // re-run would reprocess without it → a different, degraded volume + wrong labels).
            : {};
      // ONLY touch manual_shifts when the user actually changed a nudge this session (shiftsDirty).
      // Omitting it makes the backend KEEP the persisted set, so a plain mark-only re-run can never
      // erase prior nudges (the data-loss the review caught). When dirty we send the COMPLETE absolute
      // set (an empty {} then means "the user dragged every nudge back to zero" → clear). Non-finite
      // values are filtered so a bad drag can't persist garbage.
      if (shiftsDirty) {
        const manual_shifts: Record<string, number> = {};
        manualShifts.forEach((v, k) => { if (Number.isFinite(v) && v) manual_shifts[String(k)] = Math.round(v); });
        body.manual_shifts = manual_shifts;
      }
      await api.json(`/api/case/${caseId}/oct-preprocess`, "POST", JSON.stringify(body));
      // In fix-columns, STAY in the border editor after Run so the user can inspect the result and keep
      // correcting (Confirm/Run stay available) — exiting (colSel=false) dropped those buttons and surfaced
      // the unrelated "✎ Scar" instead. The legacy column-marking path still exits.
      if (!fixCols) setColSel(false);
      setBadCols(new Set());
      // Refetch the case so caseInfo.manifest.oct_params.manual_shifts is FRESH — without this,
      // persistedShifts stays stale, the "shifted" badge sticks, and a later re-run would resend a
      // stale/incomplete set and silently drop nudges (the root cause behind the data-loss cluster).
      await openCase();
      wfSet("segVersion", segSig + 1); // refetch corrected previews (re-rendered) + dropped seg
    } catch {
      /* surfaced via the spinner stopping; the volume is unchanged on failure */
    } finally {
      setRerunBusy(false);
    }
  };
  // Render the preprocessing filmstrip for the central slice. Reflects the CURRENT bad-column
  // selection (or the persisted one on a plain double-click) so step 8 matches a real re-run. NOTE: the
  // steps show the AUTOMATIC boundary-correction stages only; the #2 manual depth nudges are a final
  // post-correction applied to the whole volume (visible in the main viewer), not in this diagnostic.
  const loadSteps = async () => {
    if (!caseId) return;
    setStepsOpen(true);
    setStepsBusy(true);
    setSteps([]);
    try {
      const body = colSel ? { force_columns: [...badCols] } : {};
      const r = await api.json<{ steps: { label: string; data_url?: string; kind?: string; branch?: string; group?: string }[] }>(
        `/api/case/${caseId}/oct-preprocess-steps`, "POST", JSON.stringify(body),
      );
      setSteps(r.steps || []);
    } catch {
      setSteps([]);
    } finally {
      setStepsBusy(false);
    }
  };

  // Double-click a slice → open the steps filmstrip (the discoverable gesture the user expects).
  const onSliceDoubleClick = () => {
    if (caseId && (effectiveGroup === "context" || showBeforeAfter)) loadSteps();
  };

  // Contiguous runs of selected frames → bands to draw on the slice.
  const colRuns = (s: Set<number>): [number, number][] => {
    const a = [...s].sort((x, y) => x - y);
    const out: [number, number][] = [];
    let st: number | null = null, pr: number | null = null;
    for (const f of a) {
      if (st == null) { st = f; pr = f; }
      else if (f === (pr as number) + 1) { pr = f; }
      else { out.push([st, pr as number]); st = f; pr = f; }
    }
    if (st != null) out.push([st, pr as number]);
    return out;
  };
  // CSS filter for the display-only enhancement (applied to grayscale OCT images only).
  // In the de-nested fix-columns panel the display filter comes from the top toolbar's sliders (blur is
  // greyed there); otherwise it's driven by this panel's own ◐ Contrast / ◌ Blur toggles.
  const enhanceFilter = fixCols
    ? (filterCss || undefined)
    : ([enhContrast ? "contrast(2.2) brightness(1.12)" : "", enhBlur ? "blur(0.8px)" : ""]
        .filter(Boolean).join(" ") || undefined);

  const onImgClick = (e: React.MouseEvent<HTMLImageElement>) => {
    // Hints may land on ANY preview group: pxToIjk (coords.ts) undoes the display rot90+flipud via
    // previewToSource — the SAME rotation-aware mapping brushVoxels uses — so a click on a
    // display-rotated "context" slice maps to the correct voxel, consistent with how paintAt works.
    if (!hintMode || !cur) return;
    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();
    const fx = mirrorSag ? 1 - (e.clientX - rect.left) / rect.width      // undo the scaleX(-1) panel mirror
                         : (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    if (fx < 0 || fy < 0 || fx > 1 || fy > 1) return;
    const px = fx * (cur.image_width ?? img.naturalWidth);
    const py = fy * (cur.image_height ?? img.naturalHeight);
    const ijk = pxToIjk(cur, px, py);
    if (!ijk || cur.orientation == null || cur.slice_index == null) return;
    addScarHint({ ijk, orientation: cur.orientation, slice_index: cur.slice_index, positive: hintPositive, fx, fy });
  };

  // Hints painted on the slice currently shown.
  const hintsHere = cur
    ? (scarHints ?? []).filter((h) => h.orientation === cur.orientation && h.slice_index === cur.slice_index)
    : [];

  // ── 2D scar brush (paint cornea→scar / erase scar→cornea on the current slice) ──
  // readOnly (inspecting an earlier completed step) blocks the brush — roll back to edit.
  const editing = scarEditMode && !!cur && !scarBusy && !readOnly;

  useEffect(() => {
    const img = imgRef.current, cv = canvasRef.current;
    if (!img || !cv) return;
    const sync = () => {
      cv.width = img.clientWidth;
      cv.height = img.clientHeight;
    };
    if (img.complete) sync();
    img.addEventListener("load", sync);
    window.addEventListener("resize", sync);
    return () => {
      img.removeEventListener("load", sync);
      window.removeEventListener("resize", sync);
    };
  }, [cur, scarEditMode]);

  const paintAt = (e: React.PointerEvent) => {
    const img = imgRef.current, cv = canvasRef.current;
    if (!img || !cur) return;
    const rect = img.getBoundingClientRect();
    // SOURCE fraction (mirror undone). One inversion serves both uses: brushVoxels needs source coords, and
    // the cursor circle is drawn into a canvas that is itself inside the mirrored container — so drawing at
    // the source fraction lands it back under the pointer.
    const fx = mirrorSag ? 1 - (e.clientX - rect.left) / rect.width
                         : (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    if (fx < 0 || fy < 0 || fx > 1 || fy > 1) return;
    for (const v of brushVoxels(cur, fx, fy, scarBrush)) voxelsRef.current.set(v.join(","), v);
    const ctx = cv?.getContext("2d");
    if (cv && ctx) {
      // The displayed width spans source COLUMNS normally, but rot90(rotate_k) with odd k swaps the
      // axes so it spans source ROWS (source_height) — size the cursor against the displayed axis so
      // the circle matches the brushVoxels footprint on rotated views.
      const rk = (((cur.rotate_k ?? 0) % 4) + 4) % 4;
      const dispVox = (rk % 2 === 1 ? cur.source_height : cur.source_width) ?? 1;
      const rpx = Math.max(2, scarBrush * (rect.width / dispVox));
      ctx.fillStyle = scarErase ? "rgba(57,208,255,0.45)" : "rgba(255,46,85,0.45)";
      ctx.beginPath();
      ctx.arc(fx * cv.width, fy * cv.height, rpx, 0, Math.PI * 2);
      ctx.fill();
    }
  };
  const onPointerDown = (e: React.PointerEvent) => {
    // In fix-columns the border SVG handles dragging (no column marking) — skip the marking path.
    if (colSel && !fixCols) {   // legacy column-marking (no-WebGL path): toggle frame-columns
      e.preventDefault();
      (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
      colPaintingRef.current = true;
      const f = frameAt(e.clientX, e.clientY);
      // Press on an already-bad frame → this drag REMOVES (deselect); else it ADDS.
      colDragModeRef.current = f != null && badCols.has(f) ? "remove" : "add";
      paintColAt(e.clientX, e.clientY);
      return;
    }
    if (colSel) return;        // fix-columns border mode: the SVG owns the interaction
    if (!editing) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    paintingRef.current = true;
    voxelsRef.current.clear();
    paintAt(e);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (colSel) {
      if (!fixCols && colPaintingRef.current) paintColAt(e.clientX, e.clientY);
      return;
    }
    if (editing && paintingRef.current) paintAt(e);
  };
  const onPointerUp = async () => {
    if (colSel) { colPaintingRef.current = false; return; }
    if (!editing || !paintingRef.current) return;
    paintingRef.current = false;
    const voxels = Array.from(voxelsRef.current.values());
    voxelsRef.current.clear();
    const cv = canvasRef.current;
    cv?.getContext("2d")?.clearRect(0, 0, cv.width, cv.height);
    if (voxels.length) await runScarEdit(voxels, scarErase ? "erase" : "paint");
  };

  // The markable corrected panel (imgRef + overlay canvas + colSel bands + depth-nudge ghosts + hints).
  // Factored out so fix-columns can render it EITHER standalone OR as the "after" beside a raw "before"
  // (showRaw) without duplicating the panel. The relative-positioned <div> must stay the single
  // positioning context for the absolute overlays, so it's kept intact as one unit.
  const correctedPanel = cur ? (
    // MIRRORED on SAGITTAL so every view of this volume numbers the frame axis the same way. The fix-columns
    // editor (scaleX(-1), see its panel below), the before/after strip (BeforeAfterViewer) and the axial
    // gallery's frame labels all run REVERSED vs the array; this plain slice view was the only surface still
    // running in array order, so the same frame was "column 40" here and "frame 61" there. The flip lives on
    // the CONTAINER, which is the single positioning context for the canvas, the column bands and the hint
    // markers — so every overlay mirrors WITH the image and none of them needs its own handling. Only the
    // three screen-x → source-fraction mappings (frameAt, onImgClick, paintAt) invert, each marked below.
    <div style={{ position: "relative", display: "inline-block", maxHeight: "100%", maxWidth: "100%",
                  transform: mirrorSag ? "scaleX(-1)" : undefined }}>
      <img
        ref={imgRef}
        src={imgSrc(cur)}
        alt={cur.file_name}
        draggable={false}
        onClick={onImgClick}
        onDoubleClick={onSliceDoubleClick}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        style={{
          display: "block",
          maxHeight: "100%",
          maxWidth: "100%",
          imageRendering: "pixelated",
          touchAction: editing || colSel ? "none" : undefined,
          cursor: editing || hintMode || colSel ? "crosshair" : "zoom-in",
          filter: effectiveGroup === "context" ? enhanceFilter : undefined,
        }}
      />
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
      />
      {colSel && !fixCols && (orient === "sagittal" || orient === "coronal") && nFrames > 1 &&
        colRuns(badCols).map(([a, b], i) => {
          // Legacy no-WebGL column marking: frame f occupies the image span [f/nFrames, (f+1)/nFrames].
          const lo = a / nFrames, hiEnd = (b + 1) / nFrames;
          const pos = orient === "sagittal"
            ? { left: `${lo * 100}%`, width: `${(hiEnd - lo) * 100}%`, top: 0, bottom: 0 }
            : { top: `${Math.max(0, 1 - hiEnd) * 100}%`, height: `${(hiEnd - lo) * 100}%`, left: 0, right: 0 };
          return <div key={`b${i}`} style={{ position: "absolute", ...pos, background: "rgba(255,70,70,0.32)", pointerEvents: "none" }} />;
        })}
      {hintsHere.map((h, i) => (
        <span
          key={i}
          title={h.positive ? "scar hint" : "not-scar hint"}
          style={{
            position: "absolute",
            left: `${h.fx * 100}%`,
            top: `${h.fy * 100}%`,
            width: 12,
            height: 12,
            marginLeft: -6,
            marginTop: -6,
            borderRadius: "50%",
            border: "2px solid #fff",
            background: h.positive ? "#ff2e55" : "#39d0ff",
            boxShadow: "0 0 3px rgba(0,0,0,0.8)",
            pointerEvents: "none",
          }}
        />
      ))}
    </div>
  ) : null;

  // Fix-columns BORDER panel: the selected pass's INPUT image (raw for pass 1) with the DETECTED surface
  // (red, draggable) + RANSAC best-fit (blue) over it. Drag a frame's red point to the true surface → that
  // frame's manual_shifts; edited segments turn PINK. viewBox (n_frames × depth_vox) is stretched to the
  // image (depth 0 = top), so points map x=frame, y=depth.
  // The red border's y for frame f on the CURRENT slice: an UN-confirmed anchor follows the cursor (its
  // absolute depth); a confirmed/un-anchored frame shows the DETECTED edge (which, after Confirm, is the
  // band-re-detected surface). So before Confirm the user sees their drag; after Confirm they see the new
  // detection passing through it.
  const curAnchors = borderSliceIdx != null ? borderAnchors.get(borderSliceIdx) : undefined;
  const persistedCur = borderSliceIdx != null ? persistedAnchors.get(borderSliceIdx) : undefined;
  const edgeY = (f: number): number => {
    const a = curAnchors?.get(f);
    if (a != null && postAbsent(a, depthVox)) return NaN;               // marked "no surface on this frame"
    if (a != null && a !== (persistedCur?.get(f) ?? null)) return a;   // un-confirmed drag → WYSIWYG
    return curEdge ? curEdge[f] : 0;                                    // detected / band-re-detected
  };
  // Anchors split by KIND. A normal one says WHERE the surface is and gets a pink tick on the line; an
  // absent-sentinel one says there ISN'T one, so there is no line position to tick — edgeY is NaN there, and
  // running the pink marker through it emitted NaN coordinates, which SVG resolves to 0 and paints as a pink
  // streak along the top of the image. They are marked at the floor instead, where the drag was made.
  const anchoredFrames = useMemo(() => {
    const s = new Set<number>();
    curAnchors?.forEach((d, f) => { if (!postAbsent(d, depthVox)) s.add(f); });
    return s;
  }, [curAnchors, depthVox]);
  const absentFrames = useMemo(() => {
    const s = new Set<number>();
    curAnchors?.forEach((d, f) => { if (postAbsent(d, depthVox)) s.add(f); });
    return s;
  }, [curAnchors, depthVox]);
  // The posterior line as it should LOOK right now: the reviewer's own points win, the fetched preview fills
  // the rest. Local-first so dragging is immediate and never blanks while a request is in flight.
  const curPostPts = borderSliceIdx != null ? postAnchors.get(borderSliceIdx) : undefined;
  const postY = (f: number): number => {
    const a = curPostPts?.get(f);
    if (a != null) return postAbsent(a, depthVox) ? NaN : a;
    // NaN, not 0, when there is nothing to show for this frame: 0 is the top row of the image, so a missing
    // or short preview drew the posterior edge pinned along the very top — a confident line where there is
    // no data. The segment renderers skip non-finite frames.
    return cropPreview && cropPreview.bottom.length > f ? cropPreview.bottom[f] : NaN;
  };
  // ROUGHLY-CONSTANT THICKNESS. The reviewer fixes the TOP edge first, and the posterior then sits a
  // near-constant distance below it — so once the top is right, the bottom's expected position is known and
  // the drag can snap to it instead of being placed by eye. Measured as the MEDIAN gap between the detected
  // bottom and the CORRECTED anterior (edgeY, which already includes the reviewer's edge drags), over frames
  // where the gap is physically plausible. Median rather than mean so the mis-locked frames the reviewer is
  // about to fix cannot drag the estimate toward themselves.
  const postThickness = useMemo(() => {
    if (!curEdge) return null;
    const plaus = (g: number) => Number.isFinite(g) && g > 4 && g < depthVox * 0.6;
    // 1) THE REVIEWER'S OWN corrected points win once there are a few — they are ground truth about this
    //    cornea's thickness, so correcting a handful of frames re-aims the estimate for all the rest. Only
    //    individually-placed points count; the bulk shift would otherwise feed its own result back in.
    const manual = borderSliceIdx != null ? postManual.get(borderSliceIdx) : undefined;
    const cur = borderSliceIdx != null ? postAnchors.get(borderSliceIdx) : undefined;
    if (manual && cur && manual.size >= 3) {
      const g: number[] = [];
      manual.forEach((f) => { const d = cur.get(f); if (d != null && plaus(d - edgeY(f))) g.push(d - edgeY(f)); });
      if (g.length >= 3) { g.sort((a, b) => a - b); return g[Math.floor(g.length / 2)]; }
    }
    // 2) otherwise fall back to the DETECTED bottom, which is what there is to go on before any correction
    if (!cropPreview || cropPreview.bottom.length !== nFrames) return null;
    const gaps: number[] = [];
    for (let f = 0; f < nFrames; f++) {
      const g = cropPreview.bottom[f] - edgeY(f);
      if (plaus(g)) gaps.push(g);
    }
    if (gaps.length < Math.max(8, nFrames * 0.2)) return null;   // too few plausible frames to trust
    gaps.sort((a, b) => a - b);
    return gaps[Math.floor(gaps.length / 2)];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cropPreview, curEdge, nFrames, depthVox, anchorCount, borderSliceIdx, postManual, postAnchors]);
  // Parabola mode: the live editable quadratic = fit through the detected edge with the user's points overriding.
  const curParaPts = borderSliceIdx != null ? paraAnchors.get(borderSliceIdx) : undefined;
  // The shaped arc is drawn in EVERY mode once the user has actually placed points, so switching to Edge or
  // Surface crop no longer hides a curvature correction that is still pending Confirm. With no points there
  // is nothing to preserve, so it is only computed in parabola mode (where the seeded handles live).
  // ONLY once points exist. In quadratic-fit mode with nothing placed yet this used to compute a fresh
  // quadratic through the DETECTED EDGE, while the handles were seeded from curFit — two different curves, so
  // the handles sat several px off the line they were supposed to be driving and the tool looked broken
  // before it had been touched. With no points the line simply stays curFit, which is exactly where the
  // handles are; the first drag seeds all four from curFit, so the handover is seamless.
  // …and never on a frame whose anterior surface the reviewer has marked absent (see sliceEdgeGone).
  const curPara = curEdge && curParaPts && curParaPts.size > 0 && !surfaceGone
    ? fitQuadratic(curEdge, curParaPts) : null;
  // Show the handles from the moment quadratic-fit mode opens, seeded off the current curve, so there is
  // something to grab before any drag has happened. They sit ON the existing fit, so displaying them changes
  // nothing about the arc — until one is moved, at which point applyParaDrag commits the same seed.
  // Placed handles show everywhere (they are a marking); the SEEDED ones only in parabola mode, since they
  // are an editing affordance — three ghost crosses on a scan nobody is shaping would be noise, not
  // information.
  const curParaHandles = (curEdge && !surfaceGone)
    ? ((curParaPts && curParaPts.size) ? curParaPts
       : (borderMode === "parabola" ? seedParaPts(curEdge, (f) => (curPara ? curPara[f] : (curFit ? curFit[f] : curEdge[f]))) : undefined))
    : undefined;
  const paraHandlesAreSeed = !(curParaPts && curParaPts.size);
  // Which overlay element the CURRENT tool edits — the one that pulses. Null while read-only or mid-apply:
  // pulsing something you cannot currently drag would be misleading, which is worse than no cue at all.
  const editPulse: "edge" | "parabola" | "crop" | "latcrop" | "mark" | null =
    (readOnly || redetectBusy || borderBusy) ? null
    : markMode ? "mark"
    : latCropMode ? "latcrop"
    : cropMode ? "crop"
    : orient !== "sagittal" ? null            // border editing is sagittal-only
    : borderMode === "parabola" ? (surfaceGone ? null : "parabola")   // no curve on this frame to pulse
    : "edge";
  // Draw the curves spanning the FULL slice width: frame f is centred at x=f+0.5, so a plain map leaves a
  // half-column gap at each end (frame 0 / last frame's outer half un-drawn). Anchor the ends at x=0 and
  // x=nFrames (repeating the first/last value) so the edge reaches the very first/last pixel columns.
  const spanPts = (yAt: (f: number) => number): string => {
    const pts: string[] = [`0,${yAt(0)}`];
    for (let f = 0; f < nFrames; f++) pts.push(`${f + 0.5},${yAt(f)}`);
    pts.push(`${nFrames},${yAt(nFrames - 1)}`);
    return pts.join(" ");
  };
  // …and the same thing BROKEN at the frames where y is not a number, for any curve derived from the anterior
  // edge: a frame marked "no surface" has no y, and a single point of NaN in a polyline's `points` aborts the
  // parse at that coordinate — so one absent frame silently truncated the rest of the line.
  const segPts = (yAt: (f: number) => number): string[] => {
    const segs: string[][] = []; let cur: string[] = [];
    for (let f = 0; f < nFrames; f++) {
      const y = yAt(f);
      if (Number.isFinite(y)) cur.push(`${f + 0.5},${y}`);
      else if (cur.length) { segs.push(cur); cur = []; }
    }
    if (cur.length) segs.push(cur);
    return segs.filter((s) => s.length > 1).map((s) => s.join(" "));
  };
  // #1 "uniform AND crisp" + morphologically correct: render the native B-scan at an INTEGER pixels-per-frame
  // (kf) so every frame column is exactly kf px wide (no non-integer nearest-neighbour artefact), AND keep the
  // PHYSICAL aspect (frames are physically far wider-spaced than depth, so a frame "pixel" is a WIDE rectangle —
  // the cornea must not be vertically stretched). physAspect = the physically-scaled context preview's display
  // width/height; dispH = dispW / physAspect. kf is the largest integer fitting BOTH host width and that height.
  const sagPrev = rawImages.find((i) => i.orientation === "sagittal");
  const physAspect = (sagPrev?.image_width && sagPrev?.image_height)
    ? sagPrev.image_width / sagPrev.image_height
    : nFrames / Math.max(1, depthVox);
  const bSized = hostSize.w > 1 && hostSize.h > 1 && nFrames > 1 && physAspect > 0;
  // The image keeps its FULL size in every mode. In quadratic-fit mode the overlay is much wider than the
  // image and simply OVERFLOWS the host — drawing across the corrected panel beside it — rather than shrinking
  // the picture to make room (reviewer: "it is okay for the cross to be on the corrected image"). The host's
  // overflow is switched to visible in that mode so the margin is hit-testable, not merely drawn.
  // 100% MEANS THE SAME SIZE either way. Hiding the corrected panel hands this host the full row width, which
  // doubled the image and made "100%" mean two different things depending on a panel toggle — so a scan
  // looked twice as rough with the comparison off. The scale is therefore always solved against the
  // TWO-PANEL column width; hiding the corrected panel now buys blank space (and reach), and zoom is how you
  // get bigger. bZoom still scales freely on top.
  const kfHostW = showRaw ? (hostSize.w || 0) : (hostSize.w || 0) / 2;
  const bKf = Math.max(1, Math.floor(Math.min(
    kfHostW / Math.max(1, nFrames),
    ((hostSize.h || 0) * physAspect) / Math.max(1, nFrames),
  )));
  const bDispW = nFrames * bKf;
  const bDispH = Math.max(1, Math.round(bDispW / physAspect));   // depth height for the physical aspect (rectangular px)
  // Re-centre when the lateral margins appear or disappear (entering/leaving quadratic-fit mode) and when the
  // panel is re-sized. Declared here, after the sizes it depends on — referencing them earlier is a temporal
  // dead-zone throw, not a hoisting convenience.
  useEffect(() => { centerBorderScroll(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ },
            [latHead, bDispW, hostSize.w, borderSliceIdx]);
  const borderPanel = (inputSrc && curEdge && curFit && nFrames > 1 && depthVox > 1) ? (
    <div style={{ position: "relative",
                  // The overlay is absolutely positioned and so contributes NO layout width. These margins
                  // give the scroll container something to scroll to — without them the host has nothing
                  // wider than the image and the far margin stays unreachable.
                  ...(latHead > 0 ? { marginLeft: latHead * bKf, marginRight: latHead * bKf, flex: "0 0 auto" } : {}),
                  ...(bSized ? { width: bDispW, height: bDispH } : { display: "inline-block", maxHeight: "100%", maxWidth: "100%" }),
                  // scaleX(-1): flip the frame axis so the fix-columns editor matches the niivue sagittal view
                  // (frame0 on the RIGHT). Image + SVG overlay are children, so they flip together and stay
                  // aligned; the four screen-x→frame conversions invert the fraction (1 - fx) to compensate.
                  transform: `translate(${bPan.x}px, ${bPan.y}px) scale(${bZoom}) scaleX(-1)`, transformOrigin: "center center" }}>
      <img ref={bImgRef} src={inputSrcR ?? undefined} alt="pass input" draggable={false}
        onLoad={() => setBImgLoaded(true)}
        onError={() => { if (fixCols && imgRetry < 5) window.setTimeout(() => setImgRetry((n) => n + 1), 300); }}
        style={bSized
          ? { display: "block", width: "100%", height: "100%", objectFit: "fill", imageRendering: "pixelated", filter: enhanceFilter }
          : { display: "block", maxHeight: "100%", maxWidth: "100%", imageRendering: "pixelated", filter: enhanceFilter }} />
      {/* The overlay extends ABOVE the image by `headroom` rows in parabola / surface-crop modes so a curve
          can be specified through the apex when the apex is outside the captured window. `inset: 0` is
          replaced by an explicit top/height in percentages of the IMAGE height, which is what the parent is
          sized to — so the image still occupies viewBox rows 0..depthVox and the extra space sits above it.
          headroom === 0 reproduces the original geometry exactly. */}
      <svg viewBox={`${vbLeft} ${vbTop} ${vbW} ${vbH}`} preserveAspectRatio="none"
        onPointerDown={onBorderDown} onPointerMove={onBorderMove} onPointerUp={onBorderUp} onPointerLeave={onBorderUp}
        style={{ position: "absolute",
                 left: `${-100 * latHead / Math.max(1, nFrames)}%`,
                 width: `${100 * vbW / Math.max(1, nFrames)}%`,
                 top: `${-100 * headroom / Math.max(1, depthVox)}%`,
                 height: `${100 * vbH / Math.max(1, depthVox)}%`,
                 cursor: "row-resize", touchAction: "none", overflow: "visible",
                 zIndex: latHead > 0 ? 4 : undefined }}>
        {/* PULSE on the element the selected tool actually edits. With every marking now drawn in every mode,
            the overlay carries five curves at once and "which one am I dragging?" stops being obvious — the
            pulse answers it without hiding anything. Suppressed in readOnly (nothing is editable, so a pulse
            would be a lie) and while a correction is being applied. Uses a CSS keyframe rather than SMIL:
            <animate> is patchily supported across the WebKitGTK build the desktop app ships. */}
        <style>{`@keyframes bpulse{0%,100%{opacity:1}50%{opacity:.28}}`
          + `.bpulse{animation:bpulse 1.5s ease-in-out infinite}`}</style>
        {/* Marks where the captured image actually starts. Without it the headroom reads as black image
            rather than as "outside the scan", and a handle placed up there looks like a mistake. */}
        {headroom > 0 && (
          <line x1={vbLeft} y1={0} x2={nFrames + latHead} y2={0} stroke="#64748b" strokeWidth={1}
            strokeDasharray="4 3" vectorEffect="non-scaling-stroke" opacity={0.7} />
        )}
        {latHead > 0 && [0, nFrames].map((x) => (
          <line key={`ib${x}`} x1={x} y1={vbTop} x2={x} y2={depthVox} stroke="#64748b" strokeWidth={1}
            strokeDasharray="4 3" vectorEffect="non-scaling-stroke" opacity={0.7} />
        ))}
        {/* #2: the CYAN line is the surface the correction actually flattens to (the RANSAC fit the warp
            targets) — prominent so what you see == what's used. The RED is the raw detected edge AND the
            one you DRAG; while there are un-confirmed anchors (anchorsDirty) make the RED prominent so the
            line you're manipulating is the visible one (WYSIWYG), and demote the cyan to a reference. */}
        {/* In SURFACE-CROP mode the top-edge detection + its quadratic are meaningless where the apex is
            cropped (they fail / pin at the top), so suppress them and show the bottom-edge-based preview below. */}
        {/* RESTING VISIBILITY (reviewer request): at rest the red was 0.7px @ 0.3 opacity — a hairline the
            reviewer could not actually see, on the very line they are asked to inspect and drag. The detected
            edge has to be legible BEFORE you touch it, or you cannot judge whether it needs correcting. Raised
            to 1.0px @ 0.8; the anchorsDirty state stays stronger still, so the WYSIWYG hierarchy (red dominant
            while you are editing, cyan dominant when showing what the warp targets) is preserved. */}
        {/* ALWAYS DRAWN, in every mode. The selected tool decides what you EDIT, not what you can SEE —
            otherwise switching to the crop tool hid the border you were judging the crop against. In crop
            mode they are dimmed rather than hidden: where the apex is cropped the top-edge detection really
            is unreliable (it pins at the frame top), so it must not compete visually with the crop preview,
            but it still tells you WHERE it failed, which is the reason you are in crop mode at all. */}
        {/* Z-ORDER: the SELECTED tool's line paints LAST, i.e. in the foreground. SVG has no z-index — paint
            order is document order — so the layers are emitted from an array that puts the active one at the
            end. Where the two curves run within a pixel of each other (which is most of a good scan) whichever
            is on top is the only one you can actually see, so the one you are editing has to be it. */}
        {(() => {
          // THE ANTERIOR LINE — the DETECTED anterior, and nothing else. It must stay edgeY: that is the only
          // function that applies an un-confirmed drag (WYSIWYG), so sourcing this line from anywhere else
          // makes the red line ignore the reviewer's own corrections — pink anchor ticks appear and the line
          // does not move.
          // The reconstructed anterior is deliberately NOT drawn. On a surface-cropped frame the anterior is
          // not visible, and the correct response is the reviewer's: use the BOTTOM EDGE as the alignment
          // target for that frame, rather than synthesising an anterior from it and presenting the synthetic
          // curve as if it were observed. The warp already works this way — warp_surface_crop_extend flattens
          // to the per-slice POSTERIOR parabola, and build_surface_crop_edges calls the reconstructed anterior
          // a "GUIDANCE view", not the thing being matched. Drawing it coupled the orange line to a red one
          // for no benefit.
          const edgeLine = (
            <g key="edge">
              {(() => {
                const segs: string[][] = []; let cur: string[] = [];
                for (let f = 0; f < nFrames; f++) {
                  const y = edgeY(f);
                  if (Number.isFinite(y)) cur.push(`${f + 0.5},${y}`);
                  else if (cur.length) { segs.push(cur); cur = []; }
                }
                if (cur.length) segs.push(cur);
                return segs.filter((sg) => sg.length > 1).map((sg, i) => (
                  <polyline key={`ed${i}`} fill="none" stroke="#ff4d4d" vectorEffect="non-scaling-stroke"
                    className={editPulse === "edge" ? "bpulse" : undefined}
                    strokeWidth={anchorsDirty ? 1.4 : 1.0}
                    opacity={cropMode ? 0.55 : (anchorsDirty ? 0.95 : 0.8)}
                    points={sg.join(" ")} />
                ));
              })()}
            </g>
          );
          /* ONE smooth curve, not two. This is "the surface the correction applies": the auto RANSAC fit until
             the user shapes it, and their quadratic from then on — the same object either way, so drawing a
             separate green parabola alongside a cyan fit made one thing look like two. Parabola mode edits
             THIS line, which is why its button is cyan. */
          /* …and it disappears entirely on a frame the reviewer has marked as having no anterior surface.
             A fit needs something to fit: leaving the cyan arc on screen there would show a confident smooth
             cornea top drawn straight through the region that was just declared empty. */
          const smoothLine = surfaceGone ? null : (
            <polyline key="smooth" fill="none" stroke="#22d3ee" vectorEffect="non-scaling-stroke"
              className={editPulse === "parabola" ? "bpulse" : undefined}
              strokeWidth={curPara ? 1.5 : (anchorsDirty ? 0.8 : 1.3)}
              opacity={cropMode ? 0.35 : (anchorsDirty ? 0.5 : 0.95)}
              points={spanPts((f) => (curPara ? curPara[f] : curFit[f]))} />
          );
          /* SURFACE-CROP preview: the detected BOTTOM (posterior) edge (orange = the guidance the
             reconstruction follows) and the RECONSTRUCTED anterior surface (green = what the re-run applies;
             it ascends OFF the top of the frame where the apex is cropped rather than pinning at the top).
             The preview's own faint red top-detection is dropped — the real detected edge is drawn above in
             every mode, and two near-identical faint red curves read as a rendering fault. */
          /* Drawn when there is a preview to draw OR the reviewer has placed points of their own. Gating it on
             the preview alone meant their own bottom-line points were invisible on any scan the preview did
             not cover — their marking, hidden by the tool that made it. */
          const cropLines = ((cropPreview && cropPreview.top.length === nFrames
            && cropPreview.bottom.length === nFrames && cropPreview.recon.length === nFrames) || postCount > 0) ? (
            <g key="croplines" opacity={editPulse === "crop" ? 1 : 0.75}>
              {/* ORANGE = the detected POSTERIOR (bottom) edge — a genuinely different boundary from the
                  anterior, so it keeps its own line. The green "reconstructed surface" that used to sit here
                  is gone: it WAS the anterior, and the anterior is drawn once, in red, above. */}
              {/* Drawn from the LOCAL anchors where they exist, falling back to the fetched preview. It used
                  to read the preview only, so during a drag the line vanished until the debounced round-trip
                  came back — the edit was fine, the feedback was missing. */}
              {/* the ESTIMATED bottom (corrected top edge + measured thickness), dashed, shown only while the
                  bottom-line tool is active — so the reviewer can see what the drag will snap to rather than
                  discovering the detent by feel */}
              {cropSub === "line" && postThickness != null && segPts((f) => edgeY(f) + postThickness).map((p, i) => (
                <polyline key={`es${i}`} fill="none" stroke="#ffaa28" vectorEffect="non-scaling-stroke" strokeWidth={0.8}
                  strokeDasharray="5 4" opacity={0.5} points={p} />
              ))}
              {/* Drawn as SEGMENTS, split wherever the reviewer marked the bottom edge absent. A single
                  polyline would bridge the gap with a straight run — exactly the phantom flat edge that
                  dragging off the floor is meant to remove. */}
              {(() => {
                const segs: string[][] = []; let cur: string[] = [];
                for (let f = 0; f < nFrames; f++) {
                  const y = postY(f);
                  if (Number.isFinite(y)) cur.push(`${f + 0.5},${y}`);
                  else if (cur.length) { segs.push(cur); cur = []; }
                }
                if (cur.length) segs.push(cur);
                return segs.filter((sg) => sg.length > 1).map((sg, i) => (
                  <polyline key={`pb${i}`} fill="none" stroke="#ffaa28" vectorEffect="non-scaling-stroke"
                    // the ACTIVE tool's marking blinks (reviewer's rule); the amber COLUMNS stay steady,
                    // since dozens of wide bands blinking together is noise rather than a cue
                    className={editPulse === "crop" ? "bpulse" : undefined}
                    strokeWidth={1.3} opacity={0.95} points={sg.join(" ")} />
                ));
              })()}
            </g>
          ) : null;
          /* The three HANDLES, on the SAME cyan curve. Crosses (vertical tick + short horizontal bar) so
             they read as grabbable, never <circle> — that squashes to a dash under the stretched viewBox.
             Seeded handles are dimmer: they mark where the curve can be grabbed without claiming an edit.
             Z-ORDERED with everything else: in front while the curve is the active tool, BEHIND the lines
             otherwise, so they stop sitting on top of a red edge or an orange bottom the reviewer is working
             on. */
          const handleLayer = curParaHandles ? (
            <g key="handles" opacity={paraHandlesAreSeed ? 0.55 : 0.95}>
              {[...curParaHandles.entries()].map(([f, d], i) => (
                <g key={`pp${i}`}>
                  <line x1={f + 0.5} y1={d - depthVox / 40} x2={f + 0.5} y2={d + depthVox / 40}
                    stroke="#22d3ee" strokeWidth={2.4} vectorEffect="non-scaling-stroke" />
                  <line x1={f + 0.5 - nFrames / 90} y1={d} x2={f + 0.5 + nFrames / 90} y2={d}
                    stroke="#22d3ee" strokeWidth={2.4} vectorEffect="non-scaling-stroke" />
                </g>
              ))}
            </g>
          ) : null;
          const base = editPulse === "parabola"
            ? [edgeLine, smoothLine, cropLines, handleLayer].filter(Boolean)
            : [handleLayer, edgeLine, smoothLine, cropLines].filter(Boolean);
          const activeKey = editPulse === "edge" ? "edge" : editPulse === "parabola" ? "smooth"
            : editPulse === "crop" ? "croplines" : null;
          // stable order, then lift the active layer to the end (= painted last = on top)
          const ordered = activeKey
            ? [...base.filter((n) => (n as { key: string }).key !== activeKey),
               ...base.filter((n) => (n as { key: string }).key === activeKey)]
            : base;
          // ...and the handles ride ON TOP of the curve they belong to. This has to run AFTER the lift above,
          // which would otherwise leave them buried under the very curve they grab.
          if (activeKey === "smooth") {
            const hi = ordered.findIndex((n) => (n as { key: string }).key === "handles");
            if (hi >= 0) ordered.push(...ordered.splice(hi, 1));
          }
          return ordered;
        })()}
        {/* anchored frames on this slice → pink (over the red) — thin + translucent. A SINGLE anchor is drawn
            as a short VERTICAL tick, NOT a <circle>: the SVG viewBox (nFrames×depthVox) is stretched with
            preserveAspectRatio="none", so a circle squashes into a wide horizontal pink dash ("artifact line"). */}
        {/* Shown in every mode: an edge anchor is a MARKING the user made, and hiding it behind its own tool
            meant you could not see your border corrections while judging a crop. */}
        {colRuns(anchoredFrames).map(([a, b], i) => a === b
          ? <line key={`pk${i}`} x1={a + 0.5} y1={edgeY(a) - depthVox / 60} x2={a + 0.5} y2={edgeY(a) + depthVox / 60}
              stroke="#ff5db0" strokeWidth={1.1} vectorEffect="non-scaling-stroke" opacity={0.8} />
          : <polyline key={`pk${i}`} fill="none" stroke="#ff5db0" strokeWidth={1.0} vectorEffect="non-scaling-stroke" opacity={0.7}
              points={Array.from({ length: b - a + 1 }, (_x, k) => `${a + k + 0.5},${edgeY(a + k)}`).join(" ")} />)}
        {/* frames marked "NO anterior surface here" → a DASHED pink run along the image floor. Dashed and on
            the floor precisely so it cannot be read as a corneal boundary: it is the absence of one. Without
            it the only feedback for the gesture is a gap in the red line, which is easy to miss on one frame. */}
        {colRuns(absentFrames).map(([a, b], i) => (
          <line key={`ab${i}`} x1={a} y1={depthVox - 1} x2={b + 1} y2={depthVox - 1}
            stroke="#ff5db0" strokeWidth={1.6} strokeDasharray="3 3" vectorEffect="non-scaling-stroke" opacity={0.75} />
        ))}
        {/* CUT lines (request 1): drag the TOP/LEFT/RIGHT lines marking where the surface leaves the frame */}
        {cutMode && (() => {
          const topY = cut.top, leftX = cut.left, rightX = cut.right > 0 ? cut.right : nFrames - 1;
          const onCut = (which: "top" | "left" | "right") => (e: React.PointerEvent) => {
            if (readOnly) return;   // inspecting an earlier step → cut lines are view-only until rollback
            // capture on the stable SVG (not the line, which React recreates on setCut → would drop capture)
            e.stopPropagation(); (e.currentTarget as Element).closest("svg")?.setPointerCapture?.(e.pointerId);
            cutDragRef.current = which;
          };
          const C = "#ffd24d";
          const hit = { stroke: "transparent", strokeWidth: 12, vectorEffect: "non-scaling-stroke" as const };
          const ln = (on: boolean) => ({ stroke: C, strokeWidth: 1.3, strokeDasharray: "4 3", vectorEffect: "non-scaling-stroke" as const, opacity: on ? 0.95 : 0.4, pointerEvents: "none" as const });
          return (
            <>
              <line x1={0} y1={topY} x2={nFrames} y2={topY} {...hit} style={{ cursor: "row-resize" }} onPointerDown={onCut("top")} />
              <line x1={0} y1={topY} x2={nFrames} y2={topY} {...ln(cut.top > 0)} />
              <line x1={leftX} y1={0} x2={leftX} y2={depthVox} {...hit} style={{ cursor: "col-resize" }} onPointerDown={onCut("left")} />
              <line x1={leftX} y1={0} x2={leftX} y2={depthVox} {...ln(cut.left > 0)} />
              <line x1={rightX} y1={0} x2={rightX} y2={depthVox} {...hit} style={{ cursor: "col-resize" }} onPointerDown={onCut("right")} />
              <line x1={rightX} y1={0} x2={rightX} y2={depthVox} {...ln(cut.right > 0 && cut.right < nFrames - 1)} />
            </>
          );
        })()}
        {/* SURFACE-CROP: the marked cropped frame-columns (amber bands). Auto-detected (>= crop_min_slices) but
            not yet selected frames show as a faint outline so the user can see suggestions they removed. */}
        {/* Surface-crop column marks. The SUGGESTED (dashed) set is an editing aid and stays inside the crop
            tool; the user's own MARKED columns are drawn in every mode — they are the marking, and needing to
            switch tools to see which frames are cropped defeats reviewing the border against them. */}
        {cropMode && Object.keys(cropCounts).filter((k) => !cropCols.has(Number(k))).map((k) => (
          <rect key={`cs${k}`} x={Number(k)} y={0} width={1} height={depthVox}
            fill="rgba(255,170,40,0.10)" stroke="#ffaa28" strokeWidth={0.25} strokeDasharray="1 1"
            vectorEffect="non-scaling-stroke" pointerEvents="none" />
        ))}
        {/* ⚑ defect-mark bands for this slice — pink, drawn in every mode like the other markings. */}
        {[...markCols].map((f) => (
          <rect key={`dm${f}`} x={f} y={0} width={1} height={depthVox}
            className={editPulse === "mark" ? "bpulse" : undefined}
            fill={markMode ? "rgba(255,93,176,0.34)" : "rgba(255,93,176,0.16)"} stroke="none"
            pointerEvents="none" />
        ))}
        {[...cropCols].map((f) => (
          <rect key={`cc${f}`} x={f} y={0} width={1} height={depthVox}
            // NOT pulsed: dozens of wide amber bands blinking together is distracting rather than
            // informative, and the mode is already obvious from the toolbar. The orange line still pulses.
            fill={cropMode ? "rgba(255,170,40,0.34)" : "rgba(255,170,40,0.16)"} stroke="none"
            pointerEvents="none" />
        ))}
        {/* CROP-APPROVAL: the PROPOSED (auto-detected but unapplied) crop-region + surface-crop frames as a
            PINK overlay so the user can see + approve the auto crop. Drawn in the crop-region tool (latCropMode)
            behind the user's own blue marks; suppressed for frames the user has already marked (blue wins). */}
        {latCropMode && proposals.hasProposal && [...proposedFrames].filter((f) => !latCropFrames.has(f)).map((f) => (
          <rect key={`pp${f}`} x={f} y={0} width={1} height={depthVox}
            fill="rgba(255,93,176,0.28)" stroke="#ff5db0" strokeWidth={0.3} strokeDasharray="1 1"
            vectorEffect="non-scaling-stroke" pointerEvents="none" />
        ))}
        {/* #9 CROP REGION: the marked FRAME columns (blue bands), shown ONLY on slices INSIDE the marked
            lateral range. Until "Mark end" is clicked the range is just the start slice (Mark start sets
            lo=hi), so the bands appear on the start slice alone — not on every slice. Before any range is
            marked (lo==null) they show on the current slice so column-marking stays visible. */}
        {(latCropLo == null
              || (borderSliceIdx != null && borderSliceIdx >= latCropLo && borderSliceIdx <= (latCropHi ?? latCropLo)))
          && [...latCropFrames].map((f) => (
            <rect key={`lc${f}`} x={f} y={0} width={1} height={depthVox}
              className={editPulse === "latcrop" ? "bpulse" : undefined}
              fill={latCropMode ? "rgba(93,176,255,0.34)" : "rgba(93,176,255,0.16)"} stroke="none"
              pointerEvents="none" />
          ))}
      </svg>
      {/* First-open of a scan can leave this PNG in flight while the sidecar computes its caches; without this
          the pane shows the red/cyan lines over black, which reads as "broken". Cover it with a plain "loading"
          state until the image paints. transform: scaleX(-1) counters the parent flip so the text is readable. */}
      {fixCols && !bImgLoaded && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
                      background: "rgba(15,18,24,0.82)", pointerEvents: "none", zIndex: 6, transform: "scaleX(-1)" }}>
          <span style={{ color: "#cbd5e1", fontSize: 13, letterSpacing: 0.3 }}>loading B-scan…</span>
        </div>
      )}
    </div>
  ) : null;

  return (
    <div className="flex flex-col h-full min-h-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div
        className="flex items-center gap-2 px-3 border-b flex-wrap"
        style={{ minHeight: 40, borderColor: "var(--c-border)" }}
      >
        {!previewGroup && !fixCols && (
          <ToggleButtonGroup size="small" exclusive value={group} onChange={(_, v) => v && setGroup(v)}>
            <ToggleButton value="segmentation">Segmentation</ToggleButton>
            <ToggleButton value="context">Slices</ToggleButton>
          </ToggleButtonGroup>
        )}
        {!fixCols && (
        <ToggleButtonGroup
          size="small"
          exclusive
          value={orient}
          onChange={(_, v) => {
            if (v) {
              setOrient(v);
              const n = images.filter((i) => i.orientation === v).length;
              setIdx(n ? Math.floor(n / 2) : 0);
            }
          }}
        >
          {ORIENTS.map((o) => (
            <ToggleButton key={o} value={o} style={{ textTransform: "capitalize" }}>
              {o}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        )}

        {!fixCols && canBeforeAfter && !colSel && (
          <ToggleButton size="small" value="ba" selected={beforeAfter}
            onChange={() => setBeforeAfter((b) => !b)}
            sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
            title="View all: original (raw), preprocessed (corrected), and segmented side by side, scrubbed together">
            ⇆ View all
          </ToggleButton>
        )}

        {!fixCols && canMarkColumns && (
          <ToggleButton size="small" value="cols" selected={colSel}
            onChange={() => {
              const on = !colSel;
              setColSel(on);
              if (on) {
                // Switch to the corrected "Slices" view; default to sagittal but KEEP coronal if the
                // user is already there (both allow marking). Don't reset the slice position.
                setGroup("context");
                if (orient !== "sagittal" && orient !== "coronal") setOrient("sagittal");
                wfSet("scarEditMode", false); // only one editing mode owns the canvas at a time
                if (passCount > 1 && (fixPass == null || fixPass > passCount)) setFixPass(passCount); // default/clamp: fix the last pass
              }
            }}
            disabled={rerunBusy}
            sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
            title="Mark BAD columns (the rest are good anchors), then re-run preprocessing on just those columns (no SAM2 needed)">
            ▥ Fix columns
          </ToggleButton>
        )}
        {!fixCols && canMarkColumns && (
          <ToggleButton size="small" value="steps" selected={stepsOpen}
            onClick={loadSteps} disabled={stepsBusy}
            sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
            title="Show every preprocessing step for the central sagittal slice (image enhancement, edge, quadratic fit, 3D active, final warp). Tip: double-click a slice to open this.">
            ⚙ Steps
          </ToggleButton>
        )}
        {colSel && canMarkColumns && (
          <>
            {fixCols ? (
              <>
                <ToggleButtonGroup size="small" exclusive value={markMode ? "mark" : latCropMode ? "latcrop" : cropMode ? "crop" : cutMode ? "cut" : borderMode}
                  onChange={(_, v) => {
                    if (!v) return;
                    // EDITS SURVIVE A MODE SWITCH. This used to wipe the inactive mode's un-confirmed edits,
                    // so leaving Parabola discarded the shaped curve and leaving Edge reset the drags to the
                    // persisted set — which is why every other view still showed the ORIGINAL parabola. The
                    // reason for wiping was that a stray edit in one mode could block Confirm/Run in another;
                    // those buttons are gone, and "✗ Reject → next" now commits every kind of edit together,
                    // so discarding one because the reviewer looked at another is pure data loss.
                    if (v === "crop") {
                      setCropMode(true); setCutMode(false); setLatCropMode(false); setMarkMode(false);
                      if (cropCols.size === 0 && Object.keys(cropCounts).length === 0) void detectCrop();
                    } else if (v === "latcrop") { setLatCropMode(true); setCropMode(false); setCutMode(false); setMarkMode(false); seedFromProposal(); }
                    else if (v === "mark") { setMarkMode(true); setCropMode(false); setLatCropMode(false); setCutMode(false); }
                    else if (v === "cut") { setCutMode(true); setCropMode(false); setLatCropMode(false); setMarkMode(false); }
                    else { setCutMode(false); setCropMode(false); setLatCropMode(false); setMarkMode(false); setBorderMode(v); }
                  }}>
                  <ToggleButton value="edge" sx={modeBtnSx(MODE_COLOR.edge)}
                    title={"RED line — where the detector thinks the corneal surface is.\nDrag it onto the true surface. Only the frames you drag (and nearby slices) change; the rest is left alone.\nYou can drag above the image when the apex is outside the scan.\nSaved as ground truth when you Reject."}>Edge</ToggleButton>
                  <ToggleButton value="parabola" sx={modeBtnSx(MODE_COLOR.parabola)}
                    title={"CYAN line — the quadratic fit the correction flattens to.\nDrag the three handles and the quadratic re-solves through them exactly; they can be placed ABOVE the image when the apex is outside the scan.\nUse this when the whole surface shape is wrong rather than a few frames.\nSaved as ground truth when you Reject."}>Quadratic fit</ToggleButton>
                  {/* ✂ Cut RETIRED from the toolbar (reviewer: "I dont think the Cut is necessary because the
                      surface crop covers most of it"). Both address a clipped surface, but Cut only EXCLUDES
                      the clipped columns from the fit and leaves them unwarped, whereas Surface crop
                      RECONSTRUCTS those frames from the still-visible posterior edge and extends the canvas —
                      strictly more capable on the same problem. 0 of 308 scans in the store had a surface_cut
                      set, so nothing depended on it.
                      The BACKEND handling of oct_params.surface_cut is deliberately left intact: it is sticky
                      per-scan state, and removing the reader as well would silently change the output of any
                      case that still carries one. This hides the way to create new ones. */}
                  {/* The ✓ is an INDICATOR that the pipeline surface-cropped this scan, folded into the mode
                      button that owns the frames. It replaces the separate "⬚ Surface-crop (auto)" flag button,
                      which was a boolean living beside the real frame set — same words, different data. The
                      frames themselves are cleared in here, with Clear, where they are drawn. */}
                  <ToggleButton value="crop" sx={modeBtnSx(MODE_COLOR.crop)}
                    title={scAuto
                      ? `AMBER bands + ORANGE line — this scan WAS surface-cropped by the pipeline (${cropCols.size} frame(s)).\nThose frames have no visible apex, so they are aligned by their bottom edge instead.\nVerify the frames and the orange line, or clear them if the detector was wrong.`
                      : "AMBER bands + ORANGE line — frames whose apex sits above the captured window.\nThey have no visible surface, so they are aligned by their bottom (posterior) edge instead.\nMark the frames, then correct the orange line if the detector missed it."}>
                    ✛ Surface crop{scAuto ? "✓" : ""}</ToggleButton>
                  <ToggleButton value="mark" sx={modeBtnSx(MODE_COLOR.mark)}
                    title={"PINK bands — frames that are simply wrong.\nDrag across the bad region to mark it. No depths are recorded, so use this for problems a line correction cannot express.\nPairs with the note box. Saved when you Reject."}>⚑ Mark</ToggleButton>
                  <ToggleButton value="latcrop" sx={modeBtnSx(MODE_COLOR.latcrop)}
                    // GLOW pink when an off-cornea crop was auto-detected but not applied — the proposed frames
                    // are pre-seeded on entry so the user can manipulate/approve them.
                    className={proposals.hasProposal ? "crop-proposal-glow" : undefined}
                    title={proposals.hasProposal
                      ? "BLUE bands — frames to remove from the scan entirely.\nAn automatic crop was DETECTED but not applied: open to load the pink proposal and adjust it.\nSaved when you Reject."
                      : "BLUE bands — frames to remove from the scan entirely (blink, off-cornea, junk).\nDrag across them to add, drag again to remove. Applies to all slices.\nThese frames are zeroed before SAM2 and excluded from scar alignment. Saved when you Reject."}>⊟ Crop artifact</ToggleButton>
                </ToggleButtonGroup>
                {/* Surface-crop sub-mode. Painting columns and dragging the bottom edge are different
                    gestures on the same picture, so they get their own switch rather than a hidden modifier. */}
                {onToggleRaw && (
                  <button onClick={onToggleRaw}
                    title={showRaw
                      ? "Hide the corrected panel — the original then gets the full width, so the image is larger to judge and to draw on."
                      : "Show the corrected result beside the original."}
                    style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4,
                             color: showRaw ? "var(--c-green)" : "var(--c-text-dim)", cursor: "pointer",
                             fontSize: 11, padding: "2px 7px", whiteSpace: "nowrap" }}>
                    {showRaw ? "⇆ corrected: on" : "⇆ corrected: off"}
                  </button>
                )}
                {/* WHICH red line the before/after view edits. Lives in the toolbar (not floating over a pane) so
                    it never overlaps the zoom / Clear-slice controls. Only shown once the corrected panel is on.
                    The two corrections COMPOSE (original = detection + base warp; corrected = post-hoc rigid drift
                    fix); you edit ONE at a time so the single "Correct & re-run" is unambiguous. */}
                {onToggleRaw && showRaw && (
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 2, whiteSpace: "nowrap" }}>
                    <span style={{ fontSize: 10, opacity: 0.6, marginLeft: 2 }}>edit line:</span>
                    {(["original", "corrected"] as const).map((t) => (
                      <button key={t} onClick={() => setEditTarget(t)}
                        title={t === "original"
                          ? "Edit the ORIGINAL (left) red line — fixes detection + reshapes the base surface."
                          : "Edit the CORRECTED-result (right) red line — fixes residual inter-frame drift via a post-hoc rigid per-frame warp."}
                        style={{ background: editTarget === t ? (t === "corrected" ? "rgba(34,211,238,0.18)" : "var(--c-surface2)") : "none",
                                 border: "1px solid", borderColor: editTarget === t ? (t === "corrected" ? "#22d3ee" : "var(--c-accent)") : "var(--c-border)",
                                 borderRadius: 4, color: editTarget === t ? (t === "corrected" ? "#22d3ee" : "var(--c-text)") : "var(--c-text-dim)",
                                 cursor: "pointer", fontSize: 11, padding: "2px 7px", whiteSpace: "nowrap" }}>
                        {t === "original" ? "Original" : "Corrected"}</button>
                    ))}
                  </span>
                )}
                {cropMode && (
                  <ToggleButtonGroup size="small" exclusive value={cropSub}
                    onChange={(_, v) => { if (v) setCropSub(v); }}>
                    <ToggleButton value="cols" sx={modeBtnSx(MODE_COLOR.crop)}
                      title={"Pick which frames are cropped.\nDrag across their columns to add, drag again to remove."}>columns</ToggleButton>
                    <ToggleButton value="line" sx={modeBtnSx(MODE_COLOR.crop)}
                      title={"Correct where the bottom edge is.\nDrag the orange line onto the true edge. CLICK the dashed estimate to adopt it for that column; shift-drag moves the whole line and snaps to it.\nThe estimate is your corrected top edge plus the measured corneal thickness — so fix the top edge first.\nYour points override the detector for those frames."}>bottom line</ToggleButton>
                  </ToggleButtonGroup>
                )}
                <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                  {borderBusy || redetectBusy ? (redetectBusy ? "Applying correction…" : "Detecting border…") :
                    latCropMode ? (<>Drag to mark <b style={{ color: "#5db0ff" }}>frame-columns</b> to crop, set the lateral <b>slice range</b> (Mark start/end). current slice <b>{borderSliceIdx ?? "—"}</b>{latCropLo != null ? <> · range <b style={{ color: "#5db0ff" }}>{latCropLo}{latCropHi != null && latCropHi !== latCropLo ? `–${latCropHi}` : ""}</b></> : " · range not set (defaults to all slices)"} · {latCropFrames.size} col(s){latCropFrameRanges.length ? ` [${latCropFrameRanges.join(", ")}]` : ""}</>) :
                    cropMode ? (cropBusy ? "Detecting surface-cropped frames…" : (<>The <b style={{ color: "#ffaa28" }}>amber</b> columns are surface-cropped — aligned by the <b style={{ color: "#ffaa28" }}>orange bottom edge</b> → <b style={{ color: "#39d98a" }}>green reconstructed surface</b> (it leaves the top where the apex is cropped). Click/drag columns to add/remove, then <b>Confirm &amp; re-run</b>. · {cropCols.size} frame(s)</>)) :
                    cutMode ? (<>Drag the <b style={{ color: "#ffd24d" }}>yellow lines</b> to where the surface leaves the frame (top / left / right), then <b>Re-run with cuts</b>.</>) :
                    borderMode === "parabola" ? (surfaceGone
                      ? (<><b style={{ color: "#ff5db0" }}>No anterior surface on this frame</b> — the top edge is marked absent, so there is no curve to shape here. Scrub to another frame, or drag the <b style={{ color: "#ff4d4d" }}>red edge</b> back up off the floor if that was a mistake.</>)
                      : (<>Drag points to shape the <b style={{ color: "#22d3ee" }}>quadratic fit</b>, then <b>Confirm</b>; scrub, then <b>Run</b>.{paraCount ? ` · ${paraCount} pt(s)` : ""}</>)) :
                    (<>The <b style={{ color: "#22d3ee" }}>cyan line</b> is the surface the correction applies (the <b style={{ color: "#ff4d4d" }}>red</b> is the raw detection — its artifacts are smoothed out). Drag onto the true surface (local), then <b>Confirm</b>; scrub, then <b>Run preprocessing</b>.{anchorCount ? ` · ${anchorCount} anchor(s)` : ""}</>)}
                </span>
              </>
            ) : (
              <>
                <span className="text-[11px]" style={{ color: "#ff6b6b" }}>bad frames: {badCols.size}</span>
                {(() => {
                  const f = badCols.size ? Math.min(...badCols) : (pendingFrames.size ? Math.min(...pendingFrames) : null);
                  const off = f == null ? 0 : (manualShifts.get(f) ?? 0) - (persistedShifts.get(f) ?? 0);
                  return (
                    <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>
                      mark in sagittal (columns) or coronal (rows) · click again to unmark · then <b>↑/↓</b> to move
                      the marked columns to the right depth (Shift = bigger)
                      {off ? <b style={{ color: "#5db0ff", marginLeft: 4 }}>{off > 0 ? `↓${off}` : `↑${-off}`} vox{orient !== "sagittal" ? " — view in Sagittal to see it" : ""}</b> : null}
                    </span>
                  );
                })()}
              </>
            )}
            {passCount > 1 && !fixCols && (
              <span className="flex items-center gap-1" title="Apply this fix at ONLY this iteration pass, then re-converge the later passes from it. Earlier passes are unchanged.">
                <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>fix at pass</span>
                <Select size="small" variant="standard" value={fixPass ?? passCount}
                  onChange={(e) => setFixPass(Number(e.target.value))}
                  sx={{ fontSize: 11 }}>
                  {Array.from({ length: passCount }, (_, i) => i + 1).map((k) => (
                    <MenuItem key={k} value={k} sx={{ fontSize: 11 }}>{k}</MenuItem>
                  ))}
                </Select>
              </span>
            )}
            {fixCols ? (
              latCropMode ? (
                <>
                  {/* Mark start / Mark end retired: the marks are committed by the review loop and the range
                      defaults to ALL slices, which is what a reviewer marking a bad column actually means. */}
                  {(latCropFrames.size > 0 || latCropLo != null) && !readOnly && (
                    <button onClick={() => { setLatCropFrames(new Set()); setLatCropLo(null); setLatCropHi(null); }} disabled={latCropBusy}
                      style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "2px 6px" }}>
                      Clear
                    </button>
                  )}
                </>
              ) : cropMode ? (
                <>
                  <button onClick={detectCrop} disabled={cropBusy || rerunBusy || readOnly}
                    title="Auto-detect surface-cropped frames (apex above the window) — verify/edit the amber columns, then Confirm & re-run"
                    style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: (cropBusy || rerunBusy || readOnly) ? "default" : "pointer", fontSize: 11, padding: "2px 6px", opacity: (cropBusy || rerunBusy || readOnly) ? 0.6 : 1 }}>
                    {cropBusy ? "Detecting…" : "Detect"}
                  </button>
                  {/* The bottom-line clear lives HERE, in the crop branch — the generic one below is only
                      reachable in edge/parabola mode, so a dragged posterior had no discard path at all. */}
                  {cropSub === "line" && postCount > 0 && !readOnly && (
                    <button onClick={() => { setPostAnchors(new Map()); setPostManual(new Map()); }} disabled={cropBusy || rerunBusy}
                      title="Discard your manual bottom-edge points on this scan and go back to the detected line"
                      style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "2px 6px" }}>
                      Clear bottom line ({postCount})
                    </button>
                  )}
                  {cropCols.size > 0 && !readOnly && (
                    <button onClick={() => setCropCols(new Set())} disabled={cropBusy || rerunBusy}
                      title="Unmark every surface-cropped frame on this scan"
                      style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "2px 6px" }}>
                      Clear columns
                    </button>
                  )}
                </>
              ) : cutMode ? (
                <>
                  {(cut.top > 0 || cut.left > 0 || cut.right > 0) && (
                    <button onClick={() => setCut({ top: 0, left: 0, right: 0 })} disabled={rerunBusy}
                      style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "2px 6px" }}>
                      Reset cuts
                    </button>
                  )}
                  {(() => {
                    // an ACTIVE cut = at least one surface is actually cut in (a cut line off at its frame edge
                    // is not a cut). Disable Re-run otherwise, so it can't run a plain preprocess that would
                    // silently discard a previously confirmed edge/parabola correction.
                    const cutActive = cut.top > 0 || cut.left > 0 || (cut.right > 0 && cut.right < nFrames - 1);
                    return (
                  <button onClick={rerunWithCut} disabled={rerunBusy || redetectBusy || !cutActive}
                    title={cutActive ? "Re-run preprocessing excluding the cut surfaces from the fit (which extrapolates across them) — robust on clipped scans" : "Drag a cut line in first"}
                    style={{ background: cutActive ? "var(--c-accent)" : "var(--c-surface2)", color: "#fff", border: "none", borderRadius: 4, cursor: (rerunBusy || redetectBusy || !cutActive) ? "default" : "pointer", fontSize: 11, padding: "3px 8px", opacity: (rerunBusy || redetectBusy || !cutActive) ? 0.6 : 1 }}>
                    {rerunBusy ? "Running…" : "Re-run with cuts"}
                  </button>
                    );
                  })()}
                </>
              ) : (
              <>
                {/* Clear discards the CURRENT mode's un-committed edit. It used to know only about border and
                    parabola anchors, so a dragged bottom edge or a set of defect marks had no way back short of
                    reloading the scan — and they would then be committed by the next Reject. */}
                {(() => {
                  const dirty = markMode ? markDirty
                    : (cropMode && cropSub === "line") ? postCount > 0
                    : borderMode === "parabola" ? paraCount > 0
                    : (anchorCount > 0 || anchorsDirty);
                  return dirty && !readOnly ? (
                  <button onClick={() => {
                    if (markMode) setMarkCols(new Set());
                    else if (cropMode && cropSub === "line") { setPostAnchors(new Map()); setPostManual(new Map()); }
                    else if (borderMode === "parabola") setParaAnchors(new Map());
                    else setBorderAnchors(new Map());
                  }} disabled={redetectBusy || rerunBusy || readOnly}
                    style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "2px 6px" }}>
                    {markMode ? "Clear marks"
                      : (cropMode && cropSub === "line") ? "Clear bottom line"
                      : borderMode === "parabola" ? "Clear curve"
                      : "Clear edge"}
                  </button>
                ) : null; })()}
                {/* Confirm border / Run preprocessing / Smooth corrected volume RETIRED from the review path.
                    A correction is now committed by "✗ Reject → next" (which persists the anchors via the same
                    endpoint Confirm used, then records them as ground truth), and the corrected VOLUME is
                    regenerated in bulk by the guarded re-run rather than one scan at a time — a full reprocess
                    is ~2 min, which is the wrong thing to sit through mid-review. Clear (above) still discards
                    an in-progress edit. */}
              </>
              )
            ) : (
              <>
                {(badCols.size > 0 || shiftsDirty) && (
                  <button onClick={() => { setBadCols(new Set()); setManualShifts(new Map(persistedShifts)); }} disabled={rerunBusy}
                    style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "2px 6px" }}>
                    Clear
                  </button>
                )}
                {(() => {
                  const ready = badCols.size > 0 || shiftsDirty;
                  return (
                    <button onClick={rerunColumns} disabled={rerunBusy || !ready || readOnly}
                      style={{ background: ready ? "var(--c-accent)" : "var(--c-surface2)", color: "#fff", border: "none", borderRadius: 4, cursor: rerunBusy || !ready ? "default" : "pointer", fontSize: 11, padding: "3px 8px", opacity: rerunBusy || !ready ? 0.6 : 1 }}>
                      {rerunBusy ? "Re-running…" : (passCount > 1 && fixPass && badCols.size > 0 ? `Re-run (fix at pass ${fixPass})` : "Re-run preprocessing")}
                    </button>
                  );
                })()}
              </>
            )}
          </>
        )}

        {/* Display-only image enhancement (contrast / denoise blur) to make the corneal border
            easier to see when marking bad columns. Does NOT change the data. */}
        {!fixCols && cur && (effectiveGroup === "context" || showBeforeAfter) && (
          <>
            <ToggleButton size="small" value="contrast" selected={enhContrast}
              onChange={() => setEnhContrast((v) => !v)}
              sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
              title="Display-only contrast boost (does not change the data)">
              ◐ Contrast
            </ToggleButton>
            <ToggleButton size="small" value="blur" selected={enhBlur}
              onChange={() => setEnhBlur((v) => !v)}
              sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
              title="Display-only denoise blur — smooths speckle so the border is clearer (does not change the data)">
              ◌ Blur
            </ToggleButton>
          </>
        )}

        {cur && !showBeforeAfter && !colSel && (
          <ToggleButton
            size="small"
            value="edit"
            selected={scarEditMode}
            onChange={() => {
              const on = !scarEditMode;
              wfSet("scarEditMode", on);
              if (on && !previewGroup) setGroup("segmentation"); // show the scar to edit it
            }}
            sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
            title="Paint / erase scar on this slice"
          >
            ✎ Scar
          </ToggleButton>
        )}
        {scarEditMode && (
          <>
            <ToggleButtonGroup
              size="small"
              exclusive
              value={scarErase ? "erase" : "paint"}
              onChange={(_, v) => v && wfSet("scarErase", v === "erase")}
            >
              <ToggleButton value="paint" sx={{ py: 0.25, px: 1, fontSize: 11, textTransform: "none" }}>Paint</ToggleButton>
              <ToggleButton value="erase" sx={{ py: 0.25, px: 1, fontSize: 11, textTransform: "none" }}>Erase</ToggleButton>
            </ToggleButtonGroup>
            <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>brush</span>
            <Slider size="small" min={1} max={20} value={scarBrush} sx={{ width: 64 }}
              onChange={(_, v) => wfSet("scarBrush", v as number)} />
          </>
        )}
        <div className="flex-1" />
        {(loading || scarBusy) && <CircularProgress size={16} />}
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
          {scarEditMode ? "drag to edit scar" : fixCols ? "fix columns" : "2D view (no WebGL)"}
        </span>
      </div>

      <div className="flex-1 min-h-0 flex items-center justify-center p-3">
        {!cur ? (
          loading ? (
            <div className="text-center" style={{ color: "var(--c-text-dim)" }}>
              <div style={{ fontSize: 13 }}>Rendering slices…</div>
              <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>
                Converting the volume (DICOM → NIfTI can take a moment).
              </div>
            </div>
          ) : (
            <div className="text-center" style={{ color: "var(--c-text-dim)" }}>
              <div style={{ fontSize: 13 }}>
                No {(previewGroup ? "overlay" : GROUP_LABEL[group].toLowerCase())} {orient} slices yet.
              </div>
              <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>
                {previewGroup
                  ? "Build the consensus first, then pick a tab."
                  : group === "segmentation"
                    ? "Segment the cornea (SAM2) first."
                    : "Register a volume to render slices."}
              </div>
            </div>
          )
        ) : showBeforeAfter ? (
          // #6: each panel gets an EQUAL flex column whose image AREA is flex:1; the <img> fills it with
          // objectFit:contain so it scales UP to use the whole canvas, and equal boxes + shared slice
          // geometry render raw and corrected at the SAME size (no more "original larger").
          <div style={{ display: "flex", gap: 10, width: "100%", height: "100%", alignItems: "stretch", justifyContent: "center" }}>
            <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>original (raw)</span>
              <div style={{ flex: 1, minHeight: 0, width: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                {rawCur ? (
                  <img src={imgSrc(rawCur)} alt="raw" draggable={false}
                    style={{ width: "100%", height: "100%", objectFit: "contain", imageRendering: "pixelated", filter: enhanceFilter }} />
                ) : (
                  <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>no raw slice here</span>
                )}
              </div>
            </div>
            <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <span className="text-[11px]" style={{ color: "var(--c-green)" }}>preprocessed</span>
              <div style={{ flex: 1, minHeight: 0, width: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <img src={imgSrc(cur)} alt="corrected" draggable={false} onDoubleClick={onSliceDoubleClick}
                  title="Double-click for the preprocessing steps"
                  style={{ width: "100%", height: "100%", objectFit: "contain", imageRendering: "pixelated", filter: enhanceFilter, cursor: "zoom-in" }} />
              </div>
            </div>
            {canThird && (
              <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                {hasSeg && hasCons ? (
                  <ToggleButtonGroup size="small" exclusive value={effThird} onChange={(_, v) => v && setThirdMode(v)}>
                    <ToggleButton value="seg" sx={{ py: 0, px: 0.8, fontSize: 10, textTransform: "none" }}>This scan</ToggleButton>
                    <ToggleButton value="cons" sx={{ py: 0, px: 0.8, fontSize: 10, textTransform: "none" }}>Consensus</ToggleButton>
                  </ToggleButtonGroup>
                ) : (
                  <span className="text-[11px]" style={{ color: "var(--c-accent)" }}>
                    {effThird === "cons" ? "subgroup consensus" : "this scan (segmented)"}
                  </span>
                )}
                <div style={{ flex: 1, minHeight: 0, width: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  {thirdCur ? (
                    <img src={imgSrc(thirdCur)} alt={effThird} draggable={false}
                      style={{ width: "100%", height: "100%", objectFit: "contain", imageRendering: "pixelated" }} />
                  ) : (
                    <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>no slice here</span>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : fixCols ? (
          // Fix-columns: edit the border on the selected pass's INPUT (left, editable); when before/after
          // is on, show the corrected RESULT beside it (right, read-only) so the effect is visible after a
          // Re-run. Each panel is in a sized flex box so its inline-block img gets a definite height.
          // The panel is scaled at an INTEGER pixels-per-frame so every frame column is the same width and
          // pixel-sharp. That makes the gutter expensive in a way it does not look: at 101 frames the step
          // from 5 to 6 px/frame needs just 606 px, so a 10 px gutter that leaves each column at 604 costs a
          // FIFTH of the image size — the picture renders 505x251 instead of 606x301 to save 10 px of
          // whitespace. Trimmed to 2, and the captions are absolutely positioned so they cost no height
          // either (17 px + 4 gap each, which binds whenever the window is short).
          <div style={{ display: "flex", gap: 2, width: "100%", height: "100%", alignItems: "stretch", justifyContent: "center", position: "relative" }}>
            <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 0, position: "relative" }}>
              <span className="text-[11px]" style={{ color: editTarget === "original" || !showRaw ? "var(--c-accent)" : "var(--c-text-dim)", position: "absolute", top: 0, left: 0,
                                                     zIndex: 4, pointerEvents: "none", background: "var(--c-bg)", padding: "0 4px" }}>
                {passInputLabel}{editTarget === "original" || !showRaw ? " — drag the red border" : " (view)"}{bZoom > 1 ? " · shift/middle-drag to pan" : " · scroll to zoom"}
              </span>
              <div ref={setBorderHost} onWheel={borderPanel ? onBorderWheel : undefined}
                style={{ flex: 1, minHeight: 0, width: "100%", display: "flex", alignItems: "center", position: "relative",
                         // Normally hidden (it contains zoom/pan). In quadratic-fit mode the overlay is far
                         // wider than the panel, so the host SCROLLS horizontally rather than spilling: an
                         // overflowing overlay drew across the SIDEBAR, which is not ours to cover. overflow-y
                         // stays hidden — the vertical headroom already fits within the host height.
                         overflowX: latHead > 0 ? "auto" : "hidden",
                         overflowY: "hidden",
                         justifyContent: latHead > 0 ? "flex-start" : "center",
                         zIndex: latHead > 0 ? 3 : undefined }}>
                {borderPanel ?? (
                  <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>{borderBusy ? "Detecting border…" : "No border for this slice."}</span>
                )}

              </div>
                {borderPanel && (
                  // Anchored to the COLUMN, not the scrolling host: an absolutely positioned child of a
                  // scroll container scrolls with its content (the controls slid off screen), and a sticky
                  // flex item takes layout width (it shoved the image ~46 px off centre). The column does not
                  // scroll, so `absolute` there keeps them pinned AND out of the flow.
                  <div style={{ position: "absolute", top: 6, right: 6,
                                display: "flex", alignItems: "center", gap: 2, zIndex: 6,
                                background: "var(--c-surface)", border: "1px solid var(--c-border)", borderRadius: 6, padding: "1px 2px", opacity: 0.92 }}>
                    {([["−", () => zoomBorderCentered(1 / 1.4)],
                       [`${Math.round(bZoom * 100)}%`, resetBorderView],
                       ["+", () => zoomBorderCentered(1.4)]] as const).map(([lbl, fn], i) => (
                      <button key={i} onClick={fn} title={i === 1 ? "Reset zoom" : i === 0 ? "Zoom out" : "Zoom in"}
                        style={{ background: "none", border: "none", color: "var(--c-text)", cursor: "pointer", fontSize: 12,
                                 padding: "1px 6px", minWidth: i === 1 ? 42 : 18, textAlign: "center", fontVariantNumeric: "tabular-nums" }}>
                        {lbl}
                      </button>
                    ))}
                  </div>
                )}
            </div>
            {showRaw && cur && (
              // The CORRECTED (result) pane. Now EDITABLE: the reviewer can drag the anterior surface here (along
              // the frame axis) to correct residual inter-frame drift the raw edit can't reach — it applies as a
              // post-hoc per-frame RIGID warp (apply_sagittal_surface_gt) on Correct & re-run. Same physical-aspect
              // box (bDispW×bDispH), scaleX(-1) frame flip, and zoom/pan as the LEFT editor, so before/after render
              // same-size + same-orientation. Backdrop is the NATIVE corrected B-scan so the line lands on-grid.
              <CorrectedEdgePanel sliceIndex={cur.slice_index ?? 0} bDispW={bDispW} bDispH={bDispH} bSized={bSized}
                                  bZoom={bZoom} bPan={bPan} filterCss={enhanceFilter} readOnly={readOnly}
                                  onZoomWheel={onCorrectedWheel} />
            )}
          </div>
        ) : correctedPanel}
      </div>

      {orientImgs.length > 0 && (
        <div className="flex items-center gap-3 px-4 py-2 border-t" style={{ borderColor: "var(--c-border)" }}>
          <span className="text-xs whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>
            {orient} slice {safeIdx + 1} / {orientImgs.length}
          </span>
          <button
            type="button"
            onClick={() => skipBand(-1)}
            disabled={safeIdx <= 0}
            title={`Skip back ${PROP_SLICE_BAND} slices — one border-correction propagation band, so the next slice sits at the edge of the current correction's reach (no gap)`}
            className="text-xs px-1.5 py-0.5 rounded border whitespace-nowrap disabled:opacity-40"
            style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}
          >
            ⏮{PROP_SLICE_BAND}
          </button>
          <Slider
            size="small"
            min={0}
            max={Math.max(0, orientImgs.length - 1)}
            value={safeIdx}
            onChange={(_, v) => setIdx(v as number)}
            sx={{ flex: 1, minWidth: 80 }}
          />
          <button
            type="button"
            onClick={() => skipBand(1)}
            disabled={safeIdx >= orientImgs.length - 1}
            title={`Skip forward ${PROP_SLICE_BAND} slices — one border-correction propagation band, so the next slice sits at the edge of the current correction's reach (no gap)`}
            className="text-xs px-1.5 py-0.5 rounded border whitespace-nowrap disabled:opacity-40"
            style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}
          >
            {PROP_SLICE_BAND}⏭
          </button>
          {/* Two SPREAD-OUT slices (~1/3 and ~2/3 across the volume). Correcting the SAME defect on 2+ spread
              slices is what lets Generalize propagate it across the whole scan — a single slice is a no-op
              (gen_min_slices=2), so one edit never reaches its neighbours. These give the reviewer a fast way to
              seed that: correct here, correct the other, then Re-run and the whole volume follows. */}
          {fixCols && orient === "sagittal" && (allCurves?.edges?.length ?? 0) > 6 &&
            [Math.round((allCurves!.edges.length) / 3), Math.round((2 * allCurves!.edges.length) / 3)].map((s, i) => (
              <button
                key={`spread${i}`}
                type="button"
                onClick={() => jumpToSlice(s)}
                title={`Jump to spread-out slice ${dispSlice(s)} (one of two spread ~evenly across the volume). Correct the SAME defect on BOTH spread slices, then Re-run, so Generalize propagates it volume-wide — a single-slice correction cannot.`}
                className="text-xs px-1.5 py-0.5 rounded border whitespace-nowrap"
                style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}
              >
                ⤢ slice {dispSlice(s)}
              </button>
            ))}
          {/* Step through the slices ranked most-questionable first (raw detected edge vs its own fit), so an
              edit lands where it carries the most information instead of wherever scrubbing stopped. The
              editor already opens on rank 1; this walks to the next one. */}
          {fixCols && orient === "sagittal" && worstSlices.length > 0 && (
            <button
              type="button"
              onClick={() => {
                const here = cur?.slice_index;
                const at = here == null ? -1 : worstSlices.indexOf(Number(here));
                jumpToSlice(worstSlices[(at + 1) % worstSlices.length]);
              }}
              title={`Jump to the next most problematic slice — ranked by how far the raw detected edge (red) departs from its own fit (cyan), which is exactly what you would drag. Worst is slice ${dispSlice(worstSlices[0])}.`}
              className="text-xs px-1.5 py-0.5 rounded border whitespace-nowrap"
              style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}
            >
              ◎ worst
            </button>
          )}
          {/* Back to the slice the editor CHOSE on open — the most questionable one. After scrubbing around,
              getting back to it meant remembering its number; ◎ worst only steps to the NEXT one, so it could
              not return you either. */}
          {fixCols && orient === "sagittal" && worstSlices.length > 0 && (
            <button
              type="button"
              onClick={() => jumpToSlice(worstSlices[0])}
              disabled={cur?.slice_index === worstSlices[0]}
              title={`Return to the automatically chosen slice (${dispSlice(worstSlices[0])}) — the one this scan opened on, ranked most questionable.`}
              className="text-xs px-1.5 py-0.5 rounded border whitespace-nowrap disabled:opacity-40"
              style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}
            >
              ⌂ auto
            </button>
          )}
        </div>
      )}

      {stepsOpen && (
        <div
          onClick={() => setStepsOpen(false)}
          style={{ position: "fixed", inset: 0, zIndex: 1300, background: "rgba(0,0,0,0.72)", display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{ background: "var(--c-surface, #1b1b1f)", border: "1px solid var(--c-border)", borderRadius: 8, maxWidth: "94vw", maxHeight: "92vh", display: "flex", flexDirection: "column", overflow: "hidden" }}
          >
            <div className="flex items-center gap-3 px-4 py-2 border-b" style={{ borderColor: "var(--c-border)" }}>
              <span className="text-sm" style={{ color: "var(--c-text)" }}>
                Preprocessing steps — central sagittal slice
              </span>
              {stepsBusy && <CircularProgress size={16} />}
              <div className="flex-1" />
              <button onClick={() => setStepsOpen(false)}
                style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 13, padding: "2px 10px" }}>
                Close ✕
              </button>
            </div>
            <div style={{ overflow: "auto", padding: 14 }}>
              {stepsBusy && steps.length === 0 ? (
                <div className="text-center" style={{ color: "var(--c-text-dim)", padding: 40, fontSize: 13 }}>
                  Rendering every step (reads the .OCT + runs the pipeline)…
                </div>
              ) : steps.length === 0 ? (
                <div className="text-center" style={{ color: "var(--c-text-dim)", padding: 40, fontSize: 13 }}>No steps produced.</div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 14 }}>
                  {steps.map((s, i) => (
                    <div key={i} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <span className="text-[11px]" style={{ color: s.group === "volume" ? "var(--c-accent)" : s.kind === "decision" ? "#f59e0b" : "var(--c-text-dim)" }}>{s.label}</span>
                      {s.data_url ? (
                        <img src={s.data_url} alt={s.label} draggable={false}
                          style={{ width: "100%", border: "1px solid var(--c-border)", borderRadius: 4, imageRendering: "pixelated" }} />
                      ) : null}
                      {s.branch && (
                        <span className="text-[10px]" style={{ color: "var(--c-text-dim)", fontStyle: "italic" }}>↳ {s.branch}</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
