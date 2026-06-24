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
export function SliceGallery({ fixCols = false, orientProp, filterCss, showRaw = false }: {
  fixCols?: boolean;
  orientProp?: "axial" | "coronal" | "sagittal";
  filterCss?: string;
  showRaw?: boolean; // fix-cols: show the raw "before" beside the markable corrected "after"
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
  const [borderEdge, setBorderEdge] = useState<number[] | null>(null);
  const [borderFit, setBorderFit] = useState<number[] | null>(null);
  const [borderBusy, setBorderBusy] = useState(false);
  const borderDragRef = useRef(false);
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
    if (passCount > 1) setFixPass((p) => (p == null || p > passCount ? passCount : p));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fixCols, passCount]);
  useEffect(() => {
    if (fixCols && orientProp) { setOrient(orientProp); }
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
        .sort((a, b) => Number(a.slice_index ?? 0) - Number(b.slice_index ?? 0)),
    [images, orient],
  );
  const safeIdx = Math.min(idx, Math.max(0, orientImgs.length - 1));
  const cur = orientImgs[safeIdx];

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

  // Fix-columns border-drag: fetch the detected surface + best-fit COORDS for the current sagittal slice
  // (auto in fix-columns; re-fetched on slice change / after a re-run). x=frame/n_frames, y=depth/depth_vox.
  const borderSliceIdx = cur?.slice_index ?? null;
  useEffect(() => {
    if (!fixCols || !caseId || borderSliceIdx == null) { setBorderEdge(null); setBorderFit(null); return; }
    let cancelled = false;
    setBorderBusy(true);
    api.json<{ n_frames: number; depth_vox: number; edge: number[]; fit: number[] }>(
      `/api/case/${caseId}/oct-border-curve`, "POST", JSON.stringify({ slice_index: borderSliceIdx }))
      .then((r) => { if (cancelled) return; setBorderEdge(r.edge || null); setBorderFit(r.fit || null); })
      .catch(() => { if (!cancelled) { setBorderEdge(null); setBorderFit(null); } })
      .finally(() => !cancelled && setBorderBusy(false));
    return () => { cancelled = true; };
  }, [fixCols, caseId, borderSliceIdx, segSig]);

  // Drag a frame's surface to where it should be → set that frame's depth nudge (manual_shifts) so the red
  // border follows the cursor (WYSIWYG); the existing ghost + Re-run apply it. Same sign as the arrow nudge.
  const applyBorderDrag = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    if (!borderEdge || nFrames <= 1 || depthVox <= 1) return;
    const r = svg.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return;
    const frame = Math.round(((clientX - r.left) / r.width) * nFrames - 0.5);
    if (frame < 0 || frame >= nFrames || frame >= borderEdge.length) return;
    const depth = Math.max(0, Math.min(depthVox - 1, ((clientY - r.top) / r.height) * depthVox));
    // red shows at edge[f] + (manual − persisted); to put it at the cursor depth:
    const shift = Math.round(depth - borderEdge[frame] + (persistedShifts.get(frame) ?? 0));
    setManualShifts((prev) => { const mm = new Map(prev); if (shift === 0) mm.delete(frame); else mm.set(frame, shift); return mm; });
  };
  const onBorderDown = (e: React.PointerEvent<SVGSVGElement>) => {
    e.preventDefault(); (e.target as Element).setPointerCapture?.(e.pointerId);
    borderDragRef.current = true; applyBorderDrag(e.clientX, e.clientY, e.currentTarget);
  };
  const onBorderMove = (e: React.PointerEvent<SVGSVGElement>) => { if (borderDragRef.current) applyBorderDrag(e.clientX, e.clientY, e.currentTarget); };
  const onBorderUp = () => { borderDragRef.current = false; };

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

  const rerunColumns = async () => {
    // A marked frame the user has NUDGED (in manualShifts) is manually positioned (manual_shifts, applied
    // last) — it must NOT also be auto-interpolated, or the nudge would be relative to a re-interpolated
    // base instead of what the user saw. So force_columns = marked frames WITHOUT a manual nudge.
    const forced = [...badCols].filter((f) => !manualShifts.has(f));
    const hasForced = forced.length > 0;
    if (!caseId || (!hasForced && !shiftsDirty)) return;
    setRerunBusy(true);
    try {
      // Iterative scan (passCount>1): inject the column fix at the chosen pass ONLY and let the iteration
      // re-converge from there. Single-pass scan: the legacy targeted re-run (one pass). No forced columns
      // (nudges only): keep the persisted pipeline so the nudge lands relative to the result the user saw,
      // and clear any stale forced columns.
      const body: Record<string, unknown> = (passCount > 1 && fixPass && hasForced)
        ? { inject_pass: fixPass, force_columns: forced, good_columns: [] }
        : hasForced
          ? { force_columns: forced, good_columns: [], max_iterations: 1 }
          : { force_columns: [], good_columns: [] };
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
      setColSel(false);
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
  // Pending depth-nudge runs as [start, end, delta]: maximal CONTIGUOUS frames that share the SAME pending
  // delta (vs persisted). Splitting on delta (not just contiguity) is required so the ghost + chip show the
  // right offset when a contiguous span has mixed nudges (e.g. a sub-range re-marked and arrowed again).
  const pendingRuns = (): [number, number, number][] => {
    const frames = [...pendingFrames].sort((x, y) => x - y);
    const dlt = (f: number) => (manualShifts.get(f) ?? 0) - (persistedShifts.get(f) ?? 0);
    const out: [number, number, number][] = [];
    let st: number | null = null, pr = 0, d = 0;
    for (const f of frames) {
      const fd = dlt(f);
      if (st === null) { st = f; pr = f; d = fd; }
      else if (f === pr + 1 && fd === d) { pr = f; }
      else { out.push([st, pr, d]); st = f; pr = f; d = fd; }
    }
    if (st !== null) out.push([st, pr, d]);
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
    // Scar hints must land on the UNROTATED previews (segmentation/consensus). The "context"
    // slices are display-rotated for review, so their pixel→voxel mapping (pxToIjk) wouldn't
    // match — ignore clicks there so a hint can't be placed at the wrong voxel.
    if (!hintMode || !cur || effectiveGroup === "context") return;
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
  const editing = scarEditMode && !!cur && !scarBusy;

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
      const rpx = Math.max(2, scarBrush * (rect.width / (cur.source_width ?? 1)));
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
      {/* Fix-columns border: DETECTED surface (red, draggable) + RANSAC best-fit (blue). Drag a frame's
          red point to where the surface should be → that frame's depth nudge (manual_shifts). viewBox is
          (n_frames × depth_vox) stretched to the image (depth 0 = top), so points map x=frame, y=depth. */}
      {fixCols && borderEdge && borderFit && orient === "sagittal" && nFrames > 1 && depthVox > 1 && (
        <svg viewBox={`0 0 ${nFrames} ${depthVox}`} preserveAspectRatio="none"
          onPointerDown={onBorderDown} onPointerMove={onBorderMove} onPointerUp={onBorderUp} onPointerLeave={onBorderUp}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", cursor: "row-resize", touchAction: "none" }}>
          <polyline fill="none" stroke="#5db0ff" strokeWidth={1.2} vectorEffect="non-scaling-stroke" opacity={0.85}
            points={borderFit.map((d, f) => `${f + 0.5},${d}`).join(" ")} />
          <polyline fill="none" stroke="#ff4d4d" strokeWidth={1.8} vectorEffect="non-scaling-stroke"
            points={borderEdge.map((d, f) => `${f + 0.5},${d + (manualShifts.get(f) ?? 0) - (persistedShifts.get(f) ?? 0)}`).join(" ")} />
        </svg>
      )}
      {colSel && !fixCols && (orient === "sagittal" || orient === "coronal") && nFrames > 1 &&
        colRuns(badCols).map(([a, b], i) => {
          // Voxel-EDGE fractions: frame f occupies the image span [f/nFrames, (f+1)/nFrames] of
          // the nFrames-wide PNG (NOT /(nFrames-1), which is the first→last CENTER span and drifts
          // the band ~1 frame off by the far edge). Must match frameAt's inverse below. (#1)
          const lo = a / nFrames, hiEnd = (b + 1) / nFrames;
          const pos = orient === "sagittal"
            ? { left: `${lo * 100}%`, width: `${(hiEnd - lo) * 100}%`, top: 0, bottom: 0 }
            : { top: `${Math.max(0, 1 - hiEnd) * 100}%`, height: `${(hiEnd - lo) * 100}%`, left: 0, right: 0 };
          return <div key={`b${i}`} style={{ position: "absolute", ...pos, background: "rgba(255,70,70,0.32)", pointerEvents: "none" }} />;
        })}
      {/* #2: pending (un-re-run) depth nudges from the arrow keys. Per same-delta run, GHOST the
          frame strip translated to its new depth (so the user SEES where the columns will land) + a
          chip with the voxel offset. translateY is a % of the strip's own height (= full displayed
          depth), so delta/depthVox maps directly to the on-screen move with no pixel math. Runs are
          split by delta (pendingRuns) so a mixed-nudge span renders each part at its own offset.
          (sagittal only — that's the only view whose vertical axis is depth.) */}
      {colSel && orient === "sagittal" && nFrames > 1 && depthVox > 1 && cur &&
        pendingRuns().map(([a, b, delta], i) => {
          const lo = a / nFrames, hiEnd = (b + 1) / nFrames, w = hiEnd - lo;
          const tyPct = (delta / depthVox) * 100; // +down, as a fraction of the full depth
          return (
            <div key={`s${i}`} style={{ position: "absolute", top: 0, bottom: 0, left: `${lo * 100}%`, width: `${w * 100}%`, overflow: "hidden", pointerEvents: "none", outline: "1px solid rgba(93,176,255,0.85)", outlineOffset: -1, display: "flex", justifyContent: "center" }}>
              <img src={imgSrc(cur)} alt="" draggable={false}
                style={{ position: "absolute", top: 0, left: `${-(lo / w) * 100}%`, width: `${100 / w}%`, height: "100%", transform: `translateY(${tyPct}%)`, imageRendering: "pixelated", opacity: 0.92, filter: effectiveGroup === "context" ? enhanceFilter : undefined }} />
              <span style={{ position: "relative", height: "fit-content", marginTop: 2, background: "rgba(30,80,150,0.92)", color: "#fff", fontSize: 10, lineHeight: 1.3, padding: "1px 4px", borderRadius: 3, whiteSpace: "nowrap" }}>
                {delta > 0 ? `↓${delta}` : `↑${-delta}`} vox
              </span>
            </div>
          );
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
              <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                {borderBusy ? "Detecting border…" : (
                  <>Drag the <b style={{ color: "#ff4d4d" }}>red border</b> onto the true surface (<b style={{ color: "#5db0ff" }}>blue</b> = best fit), then <b>Re-run</b>{pendingFrames.size ? ` · ${pendingFrames.size} frame(s) adjusted` : ""}.</>
                )}
              </span>
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
            {passCount > 1 && (
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
            {(badCols.size > 0 || shiftsDirty) && (
              <button onClick={() => { setBadCols(new Set()); setManualShifts(new Map(persistedShifts)); }} disabled={rerunBusy}
                style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "2px 6px" }}>
                Clear
              </button>
            )}
            {(() => {
              const ready = badCols.size > 0 || shiftsDirty;
              return (
                <button onClick={rerunColumns} disabled={rerunBusy || !ready}
                  style={{ background: ready ? "var(--c-accent)" : "var(--c-surface2)", color: "#fff", border: "none", borderRadius: 4, cursor: rerunBusy || !ready ? "default" : "pointer", fontSize: 11, padding: "3px 8px", opacity: rerunBusy || !ready ? 0.6 : 1 }}>
                  {rerunBusy ? "Re-running…" : (passCount > 1 && fixPass && badCols.size > 0 ? `Re-run (fix at pass ${fixPass})` : "Re-run preprocessing")}
                </button>
              );
            })()}
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
          <div style={{ display: "flex", gap: 10, width: "100%", height: "100%", alignItems: "center", justifyContent: "center" }}>
            <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4 }}>
              <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>original (raw)</span>
              {rawCur ? (
                <img src={imgSrc(rawCur)} alt="raw" draggable={false}
                  style={{ maxHeight: "calc(100% - 28px)", maxWidth: "100%", objectFit: "contain", imageRendering: "pixelated", filter: enhanceFilter }} />
              ) : (
                <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>no raw slice here</span>
              )}
            </div>
            <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4 }}>
              <span className="text-[11px]" style={{ color: "var(--c-green)" }}>preprocessed</span>
              <img src={imgSrc(cur)} alt="corrected" draggable={false} onDoubleClick={onSliceDoubleClick}
                title="Double-click for the preprocessing steps"
                style={{ maxHeight: "calc(100% - 28px)", maxWidth: "100%", objectFit: "contain", imageRendering: "pixelated", filter: enhanceFilter, cursor: "zoom-in" }} />
            </div>
            {canThird && (
              <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4 }}>
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
                {thirdCur ? (
                  <img src={imgSrc(thirdCur)} alt={effThird} draggable={false}
                    style={{ maxHeight: "calc(100% - 28px)", maxWidth: "100%", objectFit: "contain", imageRendering: "pixelated" }} />
                ) : (
                  <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>no slice here</span>
                )}
              </div>
            )}
          </div>
        ) : (fixCols && showRaw && rawCur) ? (
          // Fix-columns + before/after: raw "before" (left) beside the corrected "after" with the draggable
          // border (right). The corrected panel is wrapped in a sized flex box so its (inline-block) img
          // gets a definite height in this column (avoids the "empty corrected panel" collapse).
          <div style={{ display: "flex", gap: 10, width: "100%", height: "100%", alignItems: "center", justifyContent: "center" }}>
            <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4 }}>
              <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>original (raw)</span>
              <img src={imgSrc(rawCur)} alt="reference" draggable={false}
                style={{ maxHeight: "calc(100% - 28px)", maxWidth: "100%", objectFit: "contain", imageRendering: "pixelated", filter: enhanceFilter }} />
            </div>
            <div style={{ flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4 }}>
              <span className="text-[11px]" style={{ color: "var(--c-green)" }}>corrected — drag the red border</span>
              <div style={{ flex: 1, minHeight: 0, width: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                {correctedPanel}
              </div>
            </div>
          </div>
        ) : correctedPanel}
      </div>

      {orientImgs.length > 0 && (
        <div className="flex items-center gap-3 px-4 py-2 border-t" style={{ borderColor: "var(--c-border)" }}>
          <span className="text-xs whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>
            {orient} slice {cur?.slice_index ?? "—"} ({safeIdx + 1}/{orientImgs.length})
          </span>
          <Slider
            size="small"
            min={0}
            max={Math.max(0, orientImgs.length - 1)}
            value={safeIdx}
            onChange={(_, v) => setIdx(v as number)}
          />
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
