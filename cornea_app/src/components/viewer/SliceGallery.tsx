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
export function SliceGallery({ fixCols = false, cropStart = false, orientProp, filterCss, showRaw = false, readOnly = false }: {
  fixCols?: boolean;
  cropStart?: boolean; // open fix-columns directly in surface-crop mode (auto-detect the apex-cropped frames)
  orientProp?: "axial" | "coronal" | "sagittal";
  filterCss?: string;
  showRaw?: boolean; // fix-cols: show the raw "before" beside the markable corrected "after"
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
  const octPreprocessed = Boolean((caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_preprocessed);
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
  const resetBorderView = () => { setBZoom(1); setBPan({ x: 0, y: 0 }); };
  useEffect(() => { resetBorderView(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [caseId, fixCols]);
  // Drop the previous case's steps filmstrip. Unlike the preview lists (which each fetch REPLACES), these
  // are inline base64 PNGs that loadSteps only clears when it is next opened — so a filmstrip viewed on
  // one scan would sit in state, tens of MB, across every following case in a triage run.
  useEffect(() => { setSteps([]); setStepsOpen(false); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [caseId]);
  const zoomBorderAt = (clientX: number, clientY: number, factor: number) => {
    const host = borderHostRef.current?.getBoundingClientRect();
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
  const zoomBorderCentered = (factor: number) => {
    const h = borderHostRef.current?.getBoundingClientRect();
    if (h) zoomBorderAt(h.left + h.width / 2, h.top + h.height / 2, factor);
  };
  // Editable anchor set (seeded from persisted; drag adds; Confirm persists). sliceIdx → frame → depth.
  const [borderAnchors, setBorderAnchors] = useState<Map<number, Map<number, number>>>(new Map());
  const [redetectBusy, setRedetectBusy] = useState(false);
  const [smoothBusy, setSmoothBusy] = useState(false);
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
  useEffect(() => { setBorderAnchors(cloneAnchors(persistedAnchors)); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [persistedAnchorsSig]);
  const anchorsDirty = anchorsSig(borderAnchors) !== anchorsSig(persistedAnchors);
  const anchorCount = useMemo(() => { let n = 0; borderAnchors.forEach((fm) => { n += fm.size; }); return n; }, [borderAnchors]);
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
  useEffect(() => { setCropCols(new Set(persistedCrop)); }, [persistedCrop]);
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
  const [latCropFrames, setLatCropFrames] = useState<Set<number>>(new Set());  // marked frame COLUMNS
  const [latCropLo, setLatCropLo] = useState<number | null>(null);             // lateral range start (slice index)
  const [latCropHi, setLatCropHi] = useState<number | null>(null);             // lateral range end (slice index)
  const [latCropBusy, setLatCropBusy] = useState(false);
  useEffect(() => {
    // Seed from the persisted crop, but NOT while the user is actively editing (latCropMode) — a concurrent
    // manifest change must not wipe their unsaved marks. On confirm, local already matches persisted.
    if (latCropMode) return;
    setLatCropFrames(new Set(persistedCropRegion?.frames ?? []));
    setLatCropLo(persistedCropRegion?.lo ?? null);
    setLatCropHi(persistedCropRegion?.hi ?? null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [persistedCropRegion]);
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
          if (mid.length) setIdx(Math.floor(mid.length / 2));
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

  const borderSliceIdx = cur?.slice_index ?? null;
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
    if (!fixCols || !cropMode || !caseId || borderSliceIdx == null) { setCropPreview(null); return; }
    const idx = borderSliceIdx;
    let cancelled = false;
    const t = setTimeout(() => {
      api.json<{ top: number[]; bottom: number[]; recon: number[] }>(
        `/api/case/${caseId}/oct-surface-crop/preview`, "POST",
        JSON.stringify({ slice_index: idx, surface_crop_frames: cropColsSig ? cropColsSig.split(",").map(Number) : [] }))
        .then((r) => { if (!cancelled) setCropPreview({ top: r.top || [], bottom: r.bottom || [], recon: r.recon || [] }); })
        .catch(() => { if (!cancelled) setCropPreview(null); });
    }, 200);
    return () => { cancelled = true; clearTimeout(t); };
  }, [fixCols, cropMode, caseId, borderSliceIdx, cropColsSig]);
  // The border for the CURRENT slice: the accurate (settled) curve if we have it, else the instant fast one.
  const curEdge = (borderSliceIdx != null ? (accurate.get(borderSliceIdx)?.edge ?? allCurves?.edges[borderSliceIdx]) : null) ?? null;
  const curFit = (borderSliceIdx != null ? (accurate.get(borderSliceIdx)?.fit ?? allCurves?.fits[borderSliceIdx]) : null) ?? null;

  // Drag the detected border (red) onto where the TRUE surface is → an ABSOLUTE depth ANCHOR for that
  // (slice, frame). Red follows the cursor (WYSIWYG); anchored frames turn PINK. Anchors accumulate across
  // slices; Confirm infers ONE global detection band from them and re-detects the whole volume. Dragging a
  // frame back to its detected depth removes its anchor. (NOT a manual_shift — anchors steer DETECTION.)
  const applyBorderDrag = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    // sagittal-only: anchors are keyed by the sagittal slice index (see the fix-columns orient effect)
    if (orient !== "sagittal" || !curEdge || nFrames <= 1 || depthVox <= 1 || borderSliceIdx == null) return;
    const r = svg.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return;
    const frame = Math.round((1 - (clientX - r.left) / r.width) * nFrames - 0.5);   // mirrored panel: invert screen-x
    if (frame < 0 || frame >= nFrames || frame >= curEdge.length) return;
    const depth = Math.round(Math.max(0, Math.min(depthVox - 1, ((clientY - r.top) / r.height) * depthVox)));
    const s = borderSliceIdx;
    setBorderAnchors((prev) => {
      const mm = cloneAnchors(prev);
      const fm = mm.get(s) ?? new Map<number, number>();
      // dragging onto the detected edge (±0.5) clears the anchor; otherwise set the absolute true depth
      if (Math.abs(depth - curEdge[frame]) < 1) fm.delete(frame); else fm.set(frame, depth);
      if (fm.size) mm.set(s, fm); else mm.delete(s);
      return mm;
    });
  };
  // 2c: least-squares degree-2 fit through the detected edge with the user's parabola points overriding their
  // frames → the "clean quadratic" the user shapes by dragging. Returns the curve sampled per frame.
  const fitQuadratic = (edge: number[], pts?: Map<number, number>): number[] => {
    const n = edge.length;
    if (n < 3) return edge.slice();
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, ty = 0, txy = 0, tx2y = 0;
    for (let x = 0; x < n; x++) {
      const y = pts?.get(x) ?? edge[x];
      const x2 = x * x;
      s0 += 1; s1 += x; s2 += x2; s3 += x2 * x; s4 += x2 * x2;
      ty += y; txy += x * y; tx2y += x2 * y;
    }
    const M = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]];
    const det3 = (m: number[][]) =>
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const D = det3(M);
    if (Math.abs(D) < 1e-9) return edge.slice();
    const col = [tx2y, txy, ty];
    const repl = (j: number) => M.map((row, i) => row.map((v, k) => (k === j ? col[i] : v)));
    const a = det3(repl(0)) / D, b = det3(repl(1)) / D, c = det3(repl(2)) / D;
    return Array.from({ length: n }, (_v, x) => a * x * x + b * x + c);
  };
  // Parabola-mode drag: set the depth the quadratic must pass through at this frame (no auto-clear; dragging
  // a point shapes the smooth curve).
  const applyParaDrag = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    if (orient !== "sagittal" || !curEdge || nFrames <= 1 || depthVox <= 1 || borderSliceIdx == null) return;
    const r = svg.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return;
    const frame = Math.round((1 - (clientX - r.left) / r.width) * nFrames - 0.5);   // mirrored panel: invert screen-x
    if (frame < 0 || frame >= nFrames) return;
    const depth = Math.round(Math.max(0, Math.min(depthVox - 1, ((clientY - r.top) / r.height) * depthVox)));
    setParaAnchors((prev) => {
      const mm = cloneAnchors(prev);
      const fm = mm.get(borderSliceIdx) ?? new Map<number, number>();
      fm.set(frame, depth); mm.set(borderSliceIdx, fm);
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
    // viewBox x spans [0, nFrames); frame f occupies the band [f, f+1) → floor maps a screen x to its column.
    return Math.max(0, Math.min(nFrames - 1, Math.floor((1 - (clientX - r.left) / r.width) * nFrames)));
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
    if (cropMode) {   // crop mode: drag to add/remove cropped frame-columns (pan with shift/middle or readOnly)
      if (readOnly || e.button === 1 || e.shiftKey) { borderDragRef.current = { x: e.clientX, y: e.clientY, moved: false, mode: "pan" }; return; }
      const f = frameAtBorder(e.clientX, e.currentTarget);
      cropPaintRef.current = (f != null && cropCols.has(f)) ? "remove" : "add";
      paintCrop(e.clientX, e.currentTarget);
      return;
    }
    if (cutMode) return;   // cut-line elements own their pointerdown; an empty-area press does nothing here
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
  };
  const onBorderMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (cropMode && cropPaintRef.current) { paintCrop(e.clientX, e.currentTarget); return; }
    if (latCropMode && cropPaintRef.current) { paintLatCrop(e.clientX, e.currentTarget); return; }
    if (cutDragRef.current) {   // dragging a cut line
      const r = e.currentTarget.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0) return;
      if (cutDragRef.current === "top") {
        const d = Math.round(Math.max(0, Math.min(depthVox - 1, ((e.clientY - r.top) / r.height) * depthVox)));
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
    if (borderMode === "parabola") applyParaDrag(e.clientX, e.clientY, e.currentTarget);
    else applyBorderDrag(e.clientX, e.clientY, e.currentTarget);
  };
  const onBorderUp = () => { borderDragRef.current = null; cutDragRef.current = null; cropPaintRef.current = null; };
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
      const ffx = (clientX - rect.left) / rect.width;
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
  const confirmRedetect = async () => {
    if (!caseId) return;
    setRedetectBusy(true);
    try {
      let anchorsApi: Record<string, Record<string, number>>;
      let parabola = false;
      if (borderMode === "parabola") {
        // parabola mode: send each edited slice's fitted quadratic DENSELY → the backend uses it EXACTLY
        // (seed window 0) so the warp flattens to the clean curve the user shaped, not a re-snapped edge.
        parabola = true;
        anchorsApi = {};
        paraAnchors.forEach((pts, s) => {
          const e = accurate.get(s)?.edge ?? allCurves?.edges[s];
          if (!e || !pts.size) return;
          const q = fitQuadratic(e, pts);
          const inner: Record<string, number> = {};
          for (let f = 0; f < q.length; f++) inner[String(f)] = Math.round(Math.max(0, Math.min(depthVox - 1, q[f])));
          anchorsApi[String(s)] = inner;
        });
      } else {
        anchorsApi = anchorsToApi(borderAnchors);
      }
      // Parabola Confirm with NO buildable anchors (e.g. the edge source was briefly unavailable during a
      // curves re-fetch) would POST {} → the backend treats empty anchors as REVERT-to-auto and we'd silently
      // discard the shaped curve. Abort instead (keep the points so the user can re-Confirm).
      if (parabola && Object.keys(anchorsApi).length === 0) { setRedetectBusy(false); return; }
      await api.json(`/api/case/${caseId}/oct-border-redetect`, "POST",
        JSON.stringify({ border_pass: borderPass, border_anchors: anchorsApi, parabola }));
      if (borderMode === "parabola") setParaAnchors(new Map());   // baked into the cached surface now
      await openCase();                       // refresh oct_params (persisted anchors → enables Run)
      wfSet("segVersion", segSig + 1);        // re-pull the re-detected border (this slice + on scrub)
    } catch {
      /* surfaced via the spinner stopping; the volume is unchanged on failure */
    } finally {
      setRedetectBusy(false);
    }
  };
  // SMOOTH the already-corrected volume: apply the guarded post-hoc smoothing passes (axial_consistency +
  // frame_boundary) to the corrected output, removing the residual slice-to-slice jitter the manual
  // provided_edges warp left (inter-slice smoothing is disabled on that path). Never-worse / preserves the
  // corrected depths. Drops the segmentation (geometry shifted → re-run SAM2).
  const smoothCorrected = async () => {
    if (!caseId) return;
    setSmoothBusy(true);
    try {
      await api.json(`/api/case/${caseId}/oct-smooth-corrected`, "POST", JSON.stringify({}));
      await openCase();                       // refresh manifest (seg dropped; oct_preprocessed stays true)
      wfSet("segVersion", segSig + 1);        // re-pull slices/border curves for the smoothed volume
    } catch {
      /* spinner stops; the volume is unchanged on failure */
    } finally {
      setSmoothBusy(false);
    }
  };
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
  const rerunCrop = async () => {
    if (!caseId) return;
    setRerunBusy(true);
    try {
      await api.json(`/api/case/${caseId}/oct-preprocess`, "POST",
        JSON.stringify({ surface_crop_frames: [...cropCols].sort((a, b) => a - b) }));
      await openCase();
      wfSet("segVersion", segSig + 1);
    } catch {
      /* surfaced via the spinner stopping; the volume is unchanged on failure */
    } finally {
      setRerunBusy(false);
    }
  };
  // #9: re-run preprocessing with the marked BOX removed (the frame columns over the lateral-slice range,
  // zeroed across depth before SAM2). Empty frames / no range → clears the crop. Drops the segmentation; the
  // box is recorded crop-aware. The lateral range defaults to the WHOLE volume if the user marked columns but
  // never set a range (i.e. crop those columns on every slice).
  const rerunLatCrop = async () => {
    if (!caseId) return;
    const frames = [...latCropFrames].sort((a, b) => a - b);
    const nLat = orientImgs.length;
    const lo = latCropLo ?? 0, hi = latCropHi ?? (nLat > 0 ? nLat - 1 : 0);
    setLatCropBusy(true);
    try {
      await api.json(`/api/case/${caseId}/oct-preprocess`, "POST",
        JSON.stringify({ crop_region: frames.length ? { lateral: [lo, hi], frames } : {} }));
      await openCase();
      wfSet("segVersion", segSig + 1);
    } catch {
      /* surfaced via the spinner stopping; the volume is unchanged on failure */
    } finally {
      setLatCropBusy(false);
    }
  };
  // Mark the current sagittal slice as the start/end of the lateral-slice RANGE the cropped frame-columns
  // apply to (each sagittal slice = one lateral index). "end" pairs with the last "start".
  const latMarkStart = () => { if (borderSliceIdx != null) { setLatCropLo(borderSliceIdx); setLatCropHi(borderSliceIdx); } };
  const latMarkEnd = () => {
    if (borderSliceIdx == null) return;
    const a = latCropLo ?? borderSliceIdx;
    setLatCropLo(Math.min(a, borderSliceIdx)); setLatCropHi(Math.max(a, borderSliceIdx));
  };
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
    const fx = (e.clientX - rect.left) / rect.width;
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
    const fx = (e.clientX - rect.left) / rect.width;
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
    <div style={{ position: "relative", display: "inline-block", maxHeight: "100%", maxWidth: "100%" }}>
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
    if (a != null && a !== (persistedCur?.get(f) ?? null)) return a;   // un-confirmed drag → WYSIWYG
    return curEdge ? curEdge[f] : 0;                                    // detected / band-re-detected
  };
  const anchoredFrames = useMemo(() => new Set(curAnchors ? curAnchors.keys() : []), [curAnchors]);
  // Parabola mode: the live editable quadratic = fit through the detected edge with the user's points overriding.
  const curParaPts = borderSliceIdx != null ? paraAnchors.get(borderSliceIdx) : undefined;
  const curPara = (borderMode === "parabola" && curEdge) ? fitQuadratic(curEdge, curParaPts) : null;
  // Draw the curves spanning the FULL slice width: frame f is centred at x=f+0.5, so a plain map leaves a
  // half-column gap at each end (frame 0 / last frame's outer half un-drawn). Anchor the ends at x=0 and
  // x=nFrames (repeating the first/last value) so the edge reaches the very first/last pixel columns.
  const spanPts = (yAt: (f: number) => number): string => {
    const pts: string[] = [`0,${yAt(0)}`];
    for (let f = 0; f < nFrames; f++) pts.push(`${f + 0.5},${yAt(f)}`);
    pts.push(`${nFrames},${yAt(nFrames - 1)}`);
    return pts.join(" ");
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
  const bKf = Math.max(1, Math.floor(Math.min(
    (hostSize.w || 0) / Math.max(1, nFrames),
    ((hostSize.h || 0) * physAspect) / Math.max(1, nFrames),
  )));
  const bDispW = nFrames * bKf;
  const bDispH = Math.max(1, Math.round(bDispW / physAspect));   // depth height for the physical aspect (rectangular px)
  const borderPanel = (inputSrc && curEdge && curFit && nFrames > 1 && depthVox > 1) ? (
    <div style={{ position: "relative",
                  ...(bSized ? { width: bDispW, height: bDispH } : { display: "inline-block", maxHeight: "100%", maxWidth: "100%" }),
                  // scaleX(-1): flip the frame axis so the fix-columns editor matches the niivue sagittal view
                  // (frame0 on the RIGHT). Image + SVG overlay are children, so they flip together and stay
                  // aligned; the four screen-x→frame conversions invert the fraction (1 - fx) to compensate.
                  transform: `translate(${bPan.x}px, ${bPan.y}px) scale(${bZoom}) scaleX(-1)`, transformOrigin: "center center" }}>
      <img src={inputSrc} alt="pass input" draggable={false}
        style={bSized
          ? { display: "block", width: "100%", height: "100%", objectFit: "fill", imageRendering: "pixelated", filter: enhanceFilter }
          : { display: "block", maxHeight: "100%", maxWidth: "100%", imageRendering: "pixelated", filter: enhanceFilter }} />
      <svg viewBox={`0 0 ${nFrames} ${depthVox}`} preserveAspectRatio="none"
        onPointerDown={onBorderDown} onPointerMove={onBorderMove} onPointerUp={onBorderUp} onPointerLeave={onBorderUp}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", cursor: "row-resize", touchAction: "none" }}>
        {/* #2: the CYAN line is the surface the correction actually flattens to (the RANSAC fit the warp
            targets) — prominent so what you see == what's used. The RED is the raw detected edge AND the
            one you DRAG; while there are un-confirmed anchors (anchorsDirty) make the RED prominent so the
            line you're manipulating is the visible one (WYSIWYG), and demote the cyan to a reference. */}
        {/* In SURFACE-CROP mode the top-edge detection + its quadratic are meaningless where the apex is
            cropped (they fail / pin at the top), so suppress them and show the bottom-edge-based preview below. */}
        {!cropMode && <polyline fill="none" stroke="#ff4d4d" vectorEffect="non-scaling-stroke"
          strokeWidth={anchorsDirty ? 1.3 : 0.7} opacity={anchorsDirty ? 0.9 : 0.3}
          points={spanPts(edgeY)} />}
        {!cropMode && <polyline fill="none" stroke="#22d3ee" vectorEffect="non-scaling-stroke"
          strokeWidth={anchorsDirty ? 0.8 : 1.3} opacity={anchorsDirty ? 0.5 : 0.95}
          points={spanPts((f) => curFit[f])} />}
        {/* SURFACE-CROP preview: the detected BOTTOM (posterior) edge (orange = the guidance) and the
            RECONSTRUCTED anterior surface (green = what the re-run applies; it ascends OFF the top of the frame
            where the apex is cropped, instead of pinning at the top edge). The faint red is the raw top
            detection (which fails in the cropped band). */}
        {cropMode && cropPreview && cropPreview.top.length === nFrames
          && cropPreview.bottom.length === nFrames && cropPreview.recon.length === nFrames && (<>
          <polyline fill="none" stroke="#ff4d4d" vectorEffect="non-scaling-stroke" strokeWidth={0.7} opacity={0.3}
            points={spanPts((f) => cropPreview.top[f])} />
          <polyline fill="none" stroke="#ffaa28" vectorEffect="non-scaling-stroke" strokeWidth={1.3} opacity={0.95}
            points={spanPts((f) => cropPreview.bottom[f])} />
          <polyline fill="none" stroke="#39d98a" vectorEffect="non-scaling-stroke" strokeWidth={1.6} opacity={0.97}
            points={spanPts((f) => cropPreview.recon[f])} />
        </>)}
        {/* anchored frames on this slice → pink (over the red) — thin + translucent. A SINGLE anchor is drawn
            as a short VERTICAL tick, NOT a <circle>: the SVG viewBox (nFrames×depthVox) is stretched with
            preserveAspectRatio="none", so a circle squashes into a wide horizontal pink dash ("artifact line"). */}
        {!cropMode && borderMode === "edge" && colRuns(anchoredFrames).map(([a, b], i) => a === b
          ? <line key={`pk${i}`} x1={a + 0.5} y1={edgeY(a) - depthVox / 60} x2={a + 0.5} y2={edgeY(a) + depthVox / 60}
              stroke="#ff5db0" strokeWidth={1.1} vectorEffect="non-scaling-stroke" opacity={0.8} />
          : <polyline key={`pk${i}`} fill="none" stroke="#ff5db0" strokeWidth={1.0} vectorEffect="non-scaling-stroke" opacity={0.7}
              points={Array.from({ length: b - a + 1 }, (_x, k) => `${a + k + 0.5},${edgeY(a + k)}`).join(" ")} />)}
        {/* parabola mode: the live editable quadratic (green) + the points the user dragged it through.
            Suppressed in crop mode (its green clashes with the crop preview's reconstructed surface). */}
        {!cropMode && curPara && (
          <polyline fill="none" stroke="#39d98a" strokeWidth={1.5} vectorEffect="non-scaling-stroke" opacity={0.95}
            points={spanPts((f) => curPara[f])} />
        )}
        {!cropMode && curPara && [...(curParaPts?.entries() ?? [])].map(([f, d], i) => (
          // vertical tick, not <circle> — circles squash to horizontal dashes under the stretched viewBox
          <line key={`pp${i}`} x1={f + 0.5} y1={d - depthVox / 50} x2={f + 0.5} y2={d + depthVox / 50}
            stroke="#39d98a" strokeWidth={2} vectorEffect="non-scaling-stroke" opacity={0.95} />
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
        {cropMode && (<>
          {Object.keys(cropCounts).filter((k) => !cropCols.has(Number(k))).map((k) => (
            <rect key={`cs${k}`} x={Number(k)} y={0} width={1} height={depthVox}
              fill="rgba(255,170,40,0.10)" stroke="#ffaa28" strokeWidth={0.25} strokeDasharray="1 1"
              vectorEffect="non-scaling-stroke" pointerEvents="none" />
          ))}
          {[...cropCols].map((f) => (
            <rect key={`cc${f}`} x={f} y={0} width={1} height={depthVox}
              fill="rgba(255,170,40,0.34)" stroke="none" pointerEvents="none" />
          ))}
        </>)}
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
        {latCropMode
          && (latCropLo == null
              || (borderSliceIdx != null && borderSliceIdx >= latCropLo && borderSliceIdx <= (latCropHi ?? latCropLo)))
          && [...latCropFrames].map((f) => (
            <rect key={`lc${f}`} x={f} y={0} width={1} height={depthVox}
              fill="rgba(93,176,255,0.34)" stroke="none" pointerEvents="none" />
          ))}
      </svg>
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
                <ToggleButtonGroup size="small" exclusive value={latCropMode ? "latcrop" : cropMode ? "crop" : cutMode ? "cut" : borderMode}
                  onChange={(_, v) => {
                    if (!v) return;
                    // Clear the inactive modes' UNCONFIRMED edits on switch so a stray point/drag in one mode
                    // can't block Run/Confirm in another (paraCount/anchorsDirty are global). Persisted
                    // (confirmed) anchors are untouched — borderAnchors just re-seeds from them.
                    if (v !== "parabola") setParaAnchors(new Map());
                    if (v !== "edge") setBorderAnchors(cloneAnchors(persistedAnchors));
                    if (v === "crop") {
                      setCropMode(true); setCutMode(false); setLatCropMode(false);
                      if (cropCols.size === 0 && Object.keys(cropCounts).length === 0) void detectCrop();
                    } else if (v === "latcrop") { setLatCropMode(true); setCropMode(false); setCutMode(false); seedFromProposal(); }
                    else if (v === "cut") { setCutMode(true); setCropMode(false); setLatCropMode(false); }
                    else { setCutMode(false); setCropMode(false); setLatCropMode(false); setBorderMode(v); }
                  }}>
                  <ToggleButton value="edge" sx={{ py: 0.25, px: 1, fontSize: 11, textTransform: "none" }}
                    title="Drag the noisy detected edge onto the true surface — a LOCAL correction (only the dragged region + nearby slices change)">Edge</ToggleButton>
                  <ToggleButton value="parabola" sx={{ py: 0.25, px: 1, fontSize: 11, textTransform: "none" }}
                    title="Drag points to shape a clean quadratic; the warp flattens to it (no fighting the noisy per-frame edge)">Parabola</ToggleButton>
                  <ToggleButton value="cut" sx={{ py: 0.25, px: 1, fontSize: 11, textTransform: "none" }}
                    title="Mark a clipped surface (top/left/right) to exclude from the fit, then re-run — robust on clipped scans">✂ Cut</ToggleButton>
                  <ToggleButton value="crop" sx={{ py: 0.25, px: 1, fontSize: 11, textTransform: "none" }}
                    title="Detect surface-cropped frames (apex above the window) and re-run — those frames are aligned by their visible BOTTOM edge (posterior continuity), not the missing top">✛ Surface crop</ToggleButton>
                  <ToggleButton value="latcrop" sx={{ py: 0.25, px: 1, fontSize: 11, textTransform: "none" }}
                    // GLOW pink when an off-cornea crop was auto-detected but not applied — the proposed frames
                    // are pre-seeded on entry so the user can manipulate/approve them.
                    className={proposals.hasProposal ? "crop-proposal-glow" : undefined}
                    title={proposals.hasProposal
                      ? "An automatic crop was DETECTED but not applied — click to load the pink proposed region, adjust it, then Confirm & re-run (or Approve preprocessing)"
                      : "Crop away problematic COLUMNS within the slice (drag to mark frame-columns) over a RANGE of sagittal slices (Mark start/end), then Confirm & re-run. Sagittal-only. The box is removed before SAM2 and excluded from scar-alignment (crop-aware)."}>⊟ Crop region</ToggleButton>
                </ToggleButtonGroup>
                <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                  {borderBusy || redetectBusy ? (redetectBusy ? "Applying correction…" : "Detecting border…") :
                    latCropMode ? (<>Drag to mark <b style={{ color: "#5db0ff" }}>frame-columns</b> to crop, set the lateral <b>slice range</b> (Mark start/end). current slice <b>{borderSliceIdx ?? "—"}</b>{latCropLo != null ? <> · range <b style={{ color: "#5db0ff" }}>{latCropLo}{latCropHi != null && latCropHi !== latCropLo ? `–${latCropHi}` : ""}</b></> : " · range not set (defaults to all slices)"} · {latCropFrames.size} col(s){latCropFrameRanges.length ? ` [${latCropFrameRanges.join(", ")}]` : ""}</>) :
                    cropMode ? (cropBusy ? "Detecting surface-cropped frames…" : (<>The <b style={{ color: "#ffaa28" }}>amber</b> columns are surface-cropped — aligned by the <b style={{ color: "#ffaa28" }}>orange bottom edge</b> → <b style={{ color: "#39d98a" }}>green reconstructed surface</b> (it leaves the top where the apex is cropped). Click/drag columns to add/remove, then <b>Confirm &amp; re-run</b>. · {cropCols.size} frame(s)</>)) :
                    cutMode ? (<>Drag the <b style={{ color: "#ffd24d" }}>yellow lines</b> to where the surface leaves the frame (top / left / right), then <b>Re-run with cuts</b>.</>) :
                    borderMode === "parabola" ? (<>Drag points to shape the <b style={{ color: "#39d98a" }}>green parabola</b>, then <b>Confirm</b>; scrub, then <b>Run</b>.{paraCount ? ` · ${paraCount} pt(s)` : ""}</>) :
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
                  <button onClick={latMarkStart} disabled={latCropBusy || readOnly || orient !== "sagittal" || borderSliceIdx == null}
                    title="Set the START of the lateral SLICE range = the current sagittal slice"
                    style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: (latCropBusy || readOnly || orient !== "sagittal") ? "default" : "pointer", fontSize: 11, padding: "2px 6px", opacity: (latCropBusy || readOnly || orient !== "sagittal") ? 0.6 : 1 }}>
                    Mark start{latCropLo != null ? ` (${latCropLo})` : ""}
                  </button>
                  <button onClick={latMarkEnd} disabled={latCropBusy || readOnly || orient !== "sagittal" || borderSliceIdx == null}
                    title="Set the END of the lateral SLICE range = the current sagittal slice — the marked frame-columns are cropped over [start, end]"
                    style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: (latCropBusy || readOnly || orient !== "sagittal") ? "default" : "pointer", fontSize: 11, padding: "2px 6px", opacity: (latCropBusy || readOnly || orient !== "sagittal") ? 0.6 : 1 }}>
                    Mark end{latCropHi != null && latCropHi !== latCropLo ? ` (${latCropHi})` : ""}
                  </button>
                  {(latCropFrames.size > 0 || latCropLo != null) && !readOnly && (
                    <button onClick={() => { setLatCropFrames(new Set()); setLatCropLo(null); setLatCropHi(null); }} disabled={latCropBusy}
                      style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "2px 6px" }}>
                      Clear
                    </button>
                  )}
                  <button onClick={rerunLatCrop} disabled={latCropBusy || !latCropDirty || readOnly}
                    title={readOnly ? "Inspecting an earlier step — roll back to it to edit" : latCropDirty ? "Re-run preprocessing with the marked frame-columns removed over the lateral-slice range (zeroed before SAM2; excluded from scar-alignment)" : "Mark frame-columns + a slice range first (or Clear to remove the crop)"}
                    style={{ background: (latCropDirty && !readOnly) ? "var(--c-accent)" : "var(--c-surface2)", color: "#fff", border: "none", borderRadius: 4, cursor: (latCropBusy || !latCropDirty || readOnly) ? "default" : "pointer", fontSize: 11, padding: "3px 8px", opacity: (latCropBusy || !latCropDirty || readOnly) ? 0.6 : 1 }}>
                    {latCropBusy ? "Running…" : `Confirm & re-run${latCropFrames.size ? ` (${latCropFrames.size} col)` : ""}`}
                  </button>
                </>
              ) : cropMode ? (
                <>
                  <button onClick={detectCrop} disabled={cropBusy || rerunBusy || readOnly}
                    title="Auto-detect surface-cropped frames (apex above the window) — verify/edit the amber columns, then Confirm & re-run"
                    style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: (cropBusy || rerunBusy || readOnly) ? "default" : "pointer", fontSize: 11, padding: "2px 6px", opacity: (cropBusy || rerunBusy || readOnly) ? 0.6 : 1 }}>
                    {cropBusy ? "Detecting…" : "Detect"}
                  </button>
                  {cropCols.size > 0 && !readOnly && (
                    <button onClick={() => setCropCols(new Set())} disabled={cropBusy || rerunBusy}
                      style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "2px 6px" }}>
                      Clear
                    </button>
                  )}
                  <button onClick={rerunCrop} disabled={rerunBusy || cropBusy || !cropDirty || readOnly}
                    title={readOnly ? "Inspecting an earlier step — roll back to it to edit" : cropDirty ? "Re-run preprocessing — the marked frames are reconstructed by their bottom edge (posterior continuity)" : "Mark or detect cropped frames first"}
                    style={{ background: (cropDirty && !readOnly) ? "var(--c-accent)" : "var(--c-surface2)", color: "#fff", border: "none", borderRadius: 4, cursor: (rerunBusy || cropBusy || !cropDirty || readOnly) ? "default" : "pointer", fontSize: 11, padding: "3px 8px", opacity: (rerunBusy || cropBusy || !cropDirty || readOnly) ? 0.6 : 1 }}>
                    {rerunBusy ? "Running…" : `Confirm & re-run${cropCols.size ? ` (${cropCols.size})` : ""}`}
                  </button>
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
                {(() => { const dirty = borderMode === "parabola" ? paraCount > 0 : (anchorCount > 0 || anchorsDirty); return dirty && !readOnly ? (
                  <button onClick={() => { if (borderMode === "parabola") setParaAnchors(new Map()); else setBorderAnchors(new Map()); }} disabled={redetectBusy || rerunBusy || readOnly}
                    style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "2px 6px" }}>
                    Clear
                  </button>
                ) : null; })()}
                {(() => { const dirty = borderMode === "parabola" ? paraCount > 0 : anchorsDirty; return (
                  <button onClick={confirmRedetect} disabled={redetectBusy || rerunBusy || !dirty || readOnly}
                    title={readOnly ? "Inspecting an earlier step — roll back to it to edit" : borderMode === "parabola" ? "Apply the shaped parabola — then scrub to verify" : "Re-detect the corneal border locally around your correction — then scrub to verify"}
                    style={{ background: dirty ? "var(--c-accent)" : "var(--c-surface2)", color: "#fff", border: "none", borderRadius: 4, cursor: (redetectBusy || rerunBusy || !dirty) ? "default" : "pointer", fontSize: 11, padding: "3px 8px", opacity: (redetectBusy || rerunBusy || !dirty) ? 0.6 : 1 }}>
                    {redetectBusy ? "Applying…" : "Confirm border"}
                  </button>
                ); })()}
                {(() => {
                  // ready ONLY when the case has confirmed anchors PERSISTED (== what the backend has cached
                  // to apply) and there are no un-confirmed drags. This both fixes the reopen deadlock and
                  // prevents enabling Run after a Clear+Confirm revert-to-auto (empty anchors → backend 400).
                  const ready = !anchorsDirty && paraCount === 0 && persistedAnchors.size > 0;
                  return (
                    <button onClick={rerunColumns} disabled={rerunBusy || redetectBusy || !ready || readOnly}
                      title={readOnly ? "Inspecting an earlier step — roll back to it to edit" : (anchorsDirty || paraCount > 0) ? "Confirm your changes first, then scrub to verify" : "Run preprocessing with the corrected border — only when you're satisfied"}
                      style={{ background: ready ? "var(--c-accent)" : "var(--c-surface2)", color: "#fff", border: "none", borderRadius: 4, cursor: (rerunBusy || redetectBusy || !ready) ? "default" : "pointer", fontSize: 11, padding: "3px 8px", opacity: (rerunBusy || redetectBusy || !ready) ? 0.6 : 1 }}>
                      {rerunBusy ? "Running…" : "Run preprocessing"}
                    </button>
                  );
                })()}
                {/* SMOOTH the already-corrected volume: a guarded post-hoc smoothing round that removes the
                    residual slice-to-slice jitter left by the manual warp (which disabled inter-slice smoothing).
                    Never-worse — keeps the corrected depths. Enabled once the scan is preprocessed. */}
                <button onClick={smoothCorrected}
                  disabled={smoothBusy || rerunBusy || redetectBusy || anchorsDirty || paraCount > 0 || !octPreprocessed || persistedAnchors.size === 0 || readOnly}
                  title="Smooth the ALREADY-corrected volume — removes residual slice-to-slice jitter left by the manual border warp (which turns off inter-slice smoothing to honour your exact drag). Gated / never-worse: keeps your corrected depths, only cleans up rough spots. Drops the segmentation (re-run SAM2 after)."
                  style={{ background: "none", border: "1px solid var(--c-accent)", borderRadius: 4, color: "var(--c-accent)", cursor: (smoothBusy || !octPreprocessed || persistedAnchors.size === 0) ? "default" : "pointer", fontSize: 11, padding: "3px 8px", opacity: (smoothBusy || rerunBusy || redetectBusy || anchorsDirty || paraCount > 0 || !octPreprocessed || persistedAnchors.size === 0 || readOnly) ? 0.5 : 1 }}>
                  {smoothBusy ? "Smoothing…" : "∿ Smooth corrected volume"}
                </button>
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
          <div style={{ display: "flex", gap: 10, width: "100%", height: "100%", alignItems: "center", justifyContent: "center" }}>
            <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4 }}>
              <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                {passInputLabel} — drag the red border{bZoom > 1 ? " · shift/middle-drag to pan" : " · scroll to zoom"}
              </span>
              <div ref={setBorderHost} onWheel={borderPanel ? onBorderWheel : undefined}
                style={{ flex: 1, minHeight: 0, width: "100%", display: "flex", alignItems: "center", justifyContent: "center", position: "relative", overflow: "hidden" }}>
                {borderPanel ?? (
                  <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>{borderBusy ? "Detecting border…" : "No border for this slice."}</span>
                )}
                {borderPanel && (
                  <div style={{ position: "absolute", top: 6, right: 6, display: "flex", alignItems: "center", gap: 2, zIndex: 5,
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
            </div>
            {showRaw && cur && (
              <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4 }}>
                <span className="text-[11px]" style={{ color: "var(--c-green)" }}>corrected (result)</span>
                <div style={{ flex: 1, minHeight: 0, width: "100%", display: "flex", alignItems: "center", justifyContent: "center", position: "relative", overflow: "hidden" }}>
                  {/* Display-only mirror of the CORRECTED slice, matching the LEFT editor panel EXACTLY: same
                      physical-aspect pixel box (bDispW×bDispH), same scaleX(-1) frame flip, and same zoom/pan —
                      so before (left) and after (right) render same-size + same-orientation for a fair
                      comparison. Previously the right panel reused the un-flipped, maxWidth-sized interactive
                      correctedPanel, so it looked LR-mirrored AND a different size than the editor. */}
                  <div style={{ position: "relative",
                                ...(bSized ? { width: bDispW, height: bDispH } : { display: "inline-block", maxHeight: "100%", maxWidth: "100%" }),
                                transform: `translate(${bPan.x}px, ${bPan.y}px) scale(${bZoom}) scaleX(-1)`, transformOrigin: "center center" }}>
                    <img src={imgSrc(cur)} alt="corrected" draggable={false}
                      style={bSized
                        ? { display: "block", width: "100%", height: "100%", objectFit: "fill", imageRendering: "pixelated", filter: effectiveGroup === "context" ? enhanceFilter : undefined }
                        : { display: "block", maxHeight: "100%", maxWidth: "100%", imageRendering: "pixelated", filter: effectiveGroup === "context" ? enhanceFilter : undefined }} />
                  </div>
                </div>
              </div>
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
