/* Niivue volume viewer — base grayscale OCT volume with view controls. */

import { useEffect, useRef, useState } from "react";
import { ToggleButton, ToggleButtonGroup, Slider } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import { attach, loadVolume, loadCropMask, setView, setSegmentationOpacity, setInterpolationMode, webglFailure, sliceCount, getSliceIndex, setSliceIndex, setOverlayCanvas, renderDrawOverlay, scheduleOverlay, brushScreenSize, onContextRestored, screenToColumn, setDefectBands, releaseAll, type ViewName } from "../../niivue/nvController";

/** True if this scan had a region cropped out (blink / off-cornea) — used to overlay the crop-mask in red. */
function hasCrop(m: Record<string, unknown> | null): boolean {
  if (!m) return false;
  const op = (m["oct_params"] as Record<string, unknown> | undefined) ?? {};
  const cr = (op["crop_region"] ?? m["auto_crop_region"]) as { frames?: unknown[] } | undefined;
  const cl = op["crop_lateral"] as unknown[] | undefined;
  return (Array.isArray(cr?.frames) && cr!.frames!.length > 0) || (Array.isArray(cl) && cl.length > 0);
}
import type { DefectMark } from "../../store/caseStore";
import { scanStep, hasSegmentation, octProposals, awaitingCorneaDetection } from "../../api/lifecycle";
import { PaintToolbar } from "./PaintToolbar";
import { SliceGallery } from "./SliceGallery";
import { AxialGallery } from "./AxialGallery";
import { SubgroupGrid } from "./SubgroupGrid";
import { GtCompareViewer } from "./GtCompareViewer";
import { BeforeAfterViewer } from "./BeforeAfterViewer";
import { StepsViewer } from "./StepsViewer";
import { MotionPanel } from "./MotionPanel";
import { AlignTabs } from "../panels/AlignTabs";
import { SubgroupAlignDialog } from "../panels/GroupAlignPanel";

// One size for every toolbar toggle, so the view group and the Slices/Segmentation group line up instead of
// sitting at two different heights.
const viewBtnSx = { py: 0.25, px: 1.1, fontSize: 12, textTransform: "none" as const };

export function VolumeCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);   // #paint — WebGL-independent 2-D drawing overlay
  const volumeUrl = useCaseStore((s) => s.volumeUrl);
  const editorResetNonce = useCaseStore((s) => s.editorResetNonce);
  const caseId = useCaseStore((s) => s.caseId);
  const caseInfo = useCaseStore((s) => s.caseInfo);
  const setCaseId = useCaseStore((s) => s.setCaseId);
  const openCase = useCaseStore((s) => s.openCase);
  const reviewConsensusId = useWorkflowStore((s) => s.reviewConsensusId);
  const gtViewerActive = useWorkflowStore((s) => s.gtViewerActive);
  const gtViewerName = useWorkflowStore((s) => s.gtViewerName);
  const gtViewerClass = useWorkflowStore((s) => s.gtViewerClass);
  const wfSet = useWorkflowStore((s) => s.set);
  // #2/#3: the timeline step being viewed drives which viewer tools are available. The preprocessing tools
  // (Before/after, Fix-columns, Steps) belong to the Auto→Vetted steps (2–3); Aligned (4) is view-only (a
  // group-aligned volume takes no border edits); from Cornea/SAM2 (5) on, the viewer is Slices/Segmentation.
  // Inspecting an earlier step is read-only (no border edits until rollback).
  const selectedStep = useWorkflowStore((s) => s.selectedStep);
  const manifest = (caseInfo?.manifest ?? null) as Record<string, unknown> | null;
  // Crop-approval: an auto de-tilt / off-cornea crop / clipped-apex surface-crop was DETECTED but left
  // UNAPPLIED (manifest.oct_proposals). When present we GLOW the Fix-columns button + show a pink banner over
  // the basic viewer so the user reviews it in Fix-columns and Approves (which bakes in the corrections).
  const proposals = octProposals(manifest);
  const manifestStep = scanStep(manifest);
  const effStep = selectedStep ?? manifestStep;
  const inspecting = selectedStep != null && selectedStep < manifestStep;
  const preprocStep = effStep >= 1 && effStep <= 3;   // Raw/Auto/Vetted → preprocessing tools belong here
  // Step 4 "Aligned" (reviewer spec 2026-09-11 #3): the main area shows the alignment of this scan's subgroup
  // (consensus + pairs / 3-D view / scrub) as an overlay over the still-mounted niivue canvas; "⧉ Aligned" in the
  // toolbar toggles it off to see the plain slices. The group id is <patient>_<eye>_s<subgroup> (caseStore).
  // It STAYS visible at 5. Cornea while cornea detection is pending (the approval moved the subgroup there; reviewer
  // 2026-09-12) — a read-only view of the approved alignment — until a segmentation exists.
  const alignedStep = effStep === 4 || (effStep === 5 && awaitingCorneaDetection(manifest));
  const [alignPane, setAlignPane] = useState(true);
  const [alignGid, setAlignGid] = useState<string | null>(null);
  // Reviewer 2026-09-12: "⧉ Align group" must serve up the subgroup specifier AGAIN even after the group was aligned —
  // so the pane's button opens SubgroupAlignDialog (with "re-run an existing result" pre-checked) instead of re-running
  // straight away. When its jobs finish, re-read the manifest and remount AlignTabs so the new result is shown.
  const [alignAskOpen, setAlignAskOpen] = useState(false);
  const [alignReloadNonce, setAlignReloadNonce] = useState(0);
  const resolveAlignGroupId = useCaseStore((s) => s.resolveAlignGroupId);
  const alignSub = manifest?.scar_subgroup;
  useEffect(() => {
    let cancelled = false;
    if (!alignedStep || !caseInfo?.case_id) { setAlignGid(null); return; }
    void resolveAlignGroupId().then((g) => { if (!cancelled) setAlignGid(g); });
    setAlignAskOpen(false);
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [alignedStep, caseInfo?.case_id, alignSub]);
  // The Segmentation toggle should enable as soon as the scan HAS a segmentation (per the manifest), not
  // only after the overlay finishes (re)loading — otherwise it greys on open then ungreys "after a while".
  const hasSeg = hasSegmentation(manifest);
  const correcting = useWorkflowStore((s) => s.correcting);
  const stage = useWorkflowStore((s) => s.stage);
  const segLoaded = useWorkflowStore((s) => s.segLoaded);
  const segOpacity = useWorkflowStore((s) => s.segOpacity);
  const paintMode = useWorkflowStore((s) => s.paintMode);
  const penLabel = useWorkflowStore((s) => s.penLabel);
  const penSize = useWorkflowStore((s) => s.penSize);
  // Defect-marking: when ON, the main single-plane viewer becomes interactive and the user drags to mark the
  // WRONG columns of the current sagittal/axial slice → manifest.defect_marks (read/written via caseStore).
  const markDefectMode = useWorkflowStore((s) => s.markDefectMode);
  const defectTag = useWorkflowStore((s) => s.defectTag);
  const setWf = useWorkflowStore((s) => s.set);
  const setDefectMarks = useCaseStore((s) => s.setDefectMarks);
  const [customTag, setCustomTag] = useState("");
  const [view, setViewState] = useState<ViewName>("multi");
  // #2 — visible slice scrollbar for the single-plane (axial/coronal/sagittal) niivue views.
  const [sliceIdx, setSliceIdx] = useState(0);
  const [sliceMax, setSliceMax] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [noWebgl, setNoWebgl] = useState<string | null>(null);
  // Bumped when a lost WebGL context is restored (nvController rebuilds niivue) → re-runs the volume
  // load effect below so the rebuilt viewer is repopulated instead of staying black.
  const [glTick, setGlTick] = useState(0);
  const [brush, setBrush] = useState<{ x: number; y: number; size: number } | null>(null);
  // Before/after comparison (raw vs preprocessed). The button + view only exist once the scan has
  // been preprocessed — i.e. its raw snapshot (context_raw previews) was captured.
  const [hasRaw, setHasRaw] = useState(false);
  const [compareView, setCompareView] = useState(false);
  // Fix-columns 3-state view toggle: show the original (raw, editable) pane only, the corrected result only, or both.
  // Decoupled from compareView (which drives the non-fix-cols before/after viewer); the fix-cols "⇆ view" button cycles it.
  const [fcViewMode, setFcViewMode] = useState<"original" | "both" | "corrected">("original");
  // AUTO-OPEN before/after when the pipeline surface-cropped this scan. The clip is only visible on the
  // ORIGINAL panel — the correction's whole purpose is that the output no longer shows it — so a scan that
  // was auto-cropped should land straight in the view where that decision can be seen and edited, rather
  // than requiring the user to know to go looking for it. Fires once per case, and never fights the user:
  // it only turns the view ON, so closing it stays closed.
  const autoCropOpenedRef = useRef<string | null>(null);
  useEffect(() => {
    if (!caseId || !hasRaw) return;
    if (autoCropOpenedRef.current === caseId) return;      // already handled this case
    const it = (caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_iter as
      Record<string, unknown> | undefined;
    if (it?.stopped === "surface_crop") {
      autoCropOpenedRef.current = caseId;
      setCompareView(true);
      // Surface-crop marks are SAGITTAL-only (a frame reads as a column there); the mark seed, the strip and
      // the CropMarkedImage overlay in BeforeAfterViewer are all gated on orient==="sagittal". `view` is NOT
      // reset on a case switch, so a carried-over axial/coronal view from a prior scan (e.g. after Fix-axial,
      // or reviewing a de-tilt proposal — which forces axial) would leave this auto-opened before/after
      // non-sagittal and the marks would never seed. Force sagittal here so the clip marks always appear.
      if (view !== "sagittal") onView(null, "sagittal");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId, hasRaw, caseInfo]);
  // Manual "Fix columns" correction (mark bad B-scan frames → re-run preprocessing) lives in the 2D
  // SliceGallery; on the WebGL/3D desktop path it was otherwise unreachable. This opens it. Surface-crop
  // detection is a MODE within this menu (the SliceGallery toolbar's "✛ Surface crop" tab), not a separate
  // top-level button — so all border-fixing tools live in the one Fix-columns menu.
  const [fixColsView, setFixColsView] = useState(false);
  // AXIAL fix-tool: correct the anterior surface in the AXIAL (B-scan) plane — drag the surface across laterals
  // on a FIXED FRAME — reaching apex/limbus notches on the low-signal first/last frames that the SAGITTAL
  // Fix-columns tool (which corrects along the frame axis) structurally cannot. Its own 2-D overlay (AxialGallery).
  const [fixAxialView, setFixAxialView] = useState(false);
  const autoFixOpenedRef = useRef<string | null>(null);   // once-per-case guard for the editor auto-open

  // The niivue view active BEFORE entering an overlay (before/after or fix-columns). Those modes force
  // SAGITTAL on entry (their 2-D panels are sagittal); we restore this on exit so the user lands back on the
  // view they were inspecting (e.g. axial) instead of being stranded on sagittal — the "the view changed and
  // persisted after exiting fix-columns" report. Saved on the normal→overlay transition, restored on overlay→normal.
  const preOverlayViewRef = useRef<ViewName | null>(null);
  // Preprocessing-steps filmstrip (per-stage diagnostic) — also otherwise unreachable on the 3D path.
  const [stepsView, setStepsView] = useState(false);
  // Display-only contrast / brightness / gaussian-blur (CSS filter on the viewer — covers BOTH the niivue
  // canvas and the 2-D overlays, in the ONE top toolbar so the user never enters a nested sub-UI for them).
  // Blur is reset + disabled while Fix-columns is active (you need the true pixels to mark a border).
  const [contrast, setContrast] = useState(100);   // %
  const [brightness, setBrightness] = useState(100); // %
  const [blur, setBlur] = useState(0);              // px
  // CRISP ⇄ SMOOTH display toggle. DEFAULT CRISP (reviewer request). The earlier default was smooth, on the
  // argument that the warp is sub-pixel so a linear render is the more faithful one and a nearest render adds
  // voxel steps that are not in the anatomy. True as far as it goes — but this viewer's job is JUDGING the
  // surface and marking it, and interpolation is exactly what hides the small-amplitude roughness being looked
  // for: a smoothed render makes a jagged border look acceptable. Showing the true voxels means what you accept
  // is what is in the training volume. Smooth stays one click away. Display-only either way — the data is
  // identical in both modes.
  const [crisp, setCrisp] = useState(true);
  const viewerFilter = `contrast(${contrast}%) brightness(${brightness}%)` + (blur > 0 && !fixColsView ? ` blur(${blur}px)` : "");
  // The 2-D overlays (before/after, fix-columns) are driven by the SAME top toolbar — no nested sub-UI.
  // They're 2-D, so Multi/3D don't apply; fix-columns marks along depth, so it's coronal/sagittal only.
  const alignPaneOn = alignedStep && alignPane && !!alignGid;
  const overlay2d = compareView || fixColsView || fixAxialView || alignPaneOn;
  // #9 — Fix-columns "Crop region" mode is SAGITTAL-ONLY (the crop is defined in sagittal terms: frame
  // columns over a lateral-slice range), so force sagittal + disable the coronal option while it's active.
  const cropRegionMode = useWorkflowStore((s) => s.cropRegionMode);
  // Fix-columns is SAGITTAL-ONLY: the coronal option did nothing there (the border edit + crop are defined in
  // sagittal terms), so it's removed from the toolbar and the orientation is forced to sagittal in fix-columns.
  const orient2d: "axial" | "coronal" | "sagittal" = fixColsView
    ? "sagittal"
    : fixAxialView
    ? "axial"
    : (view === "axial" || view === "coronal" || view === "sagittal") ? view : "sagittal";
  // Slices | Segmentation overlay toggle (Segmentation greyed until SAM2 has run). On the niivue path it
  // simply shows/hides the cornea/scar overlay by opacity; the 2-D gallery reads the same flag.
  // Slices | Segmentation toggle is now the STORE flag (showSegmentation) so runSam2 can auto-switch to
  // the overlay after a run (#6a) and OverlayControls stays in sync. Defaults to Slices per case (reset
  // in resetForCase + on case change below); Segmentation is greyed until SAM2 has run.
  const showSeg = useWorkflowStore((s) => s.showSegmentation);
  useEffect(() => { setSegmentationOpacity(showSeg && segLoaded ? segOpacity : 0); }, [showSeg, segLoaded, segOpacity]);
  // brush-cursor colour per pen (0 erase, 1 cornea, 2 background, 3 scar)
  const PEN_COLOR: Record<number, string> = { 0: "#c7c7cc", 1: "#1ab2ff", 2: "#8e8e93", 3: "#ff453a" };
  const painting = correcting && paintMode && view !== "render";
  const showBrush = painting && brush;

  // A per-subgroup consensus case → show the synchronized multi-scan grid instead of the
  // single-volume viewer (works with or without WebGL — it's a 2D comparison).
  const consensusScans = (caseInfo?.manifest as Record<string, unknown> | undefined)?.consensus_cases as string[] | undefined;
  const isSubgroup = !!(consensusScans && consensusScans.length > 1);

  const backToSubgroup = async () => {
    if (!reviewConsensusId) return;
    // Re-render the just-corrected scan's per-scan overlay so the grid reflects the edit
    // (done once on return, not per brush stroke — the dense render is ~seconds).
    const corrected = caseInfo?.case_id;
    if (corrected && corrected !== reviewConsensusId) {
      try { await api.json(`/api/case/${corrected}/refresh-panel`, "POST", JSON.stringify({})); } catch { /* best-effort */ }
    }
    setCaseId(reviewConsensusId);
    await openCase();
    wfSet("reviewConsensusId", null);
  };
  const backBanner = reviewConsensusId && !isSubgroup ? (
    <div className="flex items-center gap-2 px-3 py-1 border-b text-xs"
      style={{ borderColor: "var(--c-border)", background: "var(--c-surface)", color: "var(--c-text-dim)" }}>
      <button onClick={backToSubgroup}
        style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-accent)", cursor: "pointer", fontSize: 11, padding: "1px 8px" }}>
        ← Back to subgroup
      </button>
      <span>Correcting one scan — its scar updates the per-scan overlay.</span>
    </div>
  ) : null;

  useEffect(() => {
    if (canvasRef.current) {
      attach(canvasRef.current);
      setNoWebgl(webglFailure());
    }
    // Recover from a lost WebGL context (WebKitGTK drops it under pressure): nvController rebuilds niivue
    // on "restored"; we bump glTick so the load effect repopulates the rebuilt viewer (else it stays black).
    onContextRestored(() => setGlTick((t) => t + 1));
    // #paint — register the WebGL-independent 2-D drawing overlay (so brush strokes are visible on the
    // WebKitGTK stack where niivue's draw tile renders blank), and keep it sized/redrawn on resize.
    setOverlayCanvas(overlayRef.current);
    let ro: ResizeObserver | undefined;
    if (overlayRef.current?.parentElement && typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => requestAnimationFrame(() => renderDrawOverlay()));
      ro.observe(overlayRef.current.parentElement);
    }
    // Teardown: disconnect the observer, drop the overlay/context-restore callbacks (both hold this
    // component's closures), and RELEASE the niivue payload. releaseAll matters because niivue is a
    // module singleton that outlives this component — without it the last case's decoded volume
    // (63-126 MB) and drawing bitmap would sit in the instance until the window is closed.
    return () => { ro?.disconnect(); setOverlayCanvas(null); onContextRestored(null); releaseAll(); };
  }, []);

  useEffect(() => {
    // No volume (fresh start, or a case reset/unload that interrupts an in-flight load): make sure the
    // "Loading volume…" indicator can't stay stuck on — reset it instead of leaving the prior state.
    if (!volumeUrl || webglFailure()) { setLoading(false); setError(null); return; }
    let cancelled = false;
    setLoading(true);
    setError(null);
    loadVolume(volumeUrl)
      .then(() => {
        if (cancelled) return;
        setView(view);
        setInterpolationMode(crisp);   // apply the crisp/smooth choice to the freshly-loaded volume
        // Re-show an existing segmentation when reopening a finished case — but ONLY if the manifest says
        // one exists. Calling it for an unsegmented (step-2) case fetches segmentation-display.nii.gz that
        // isn't there yet and logs a 404 on every open; the overlay correctly stays hidden either way.
        const liveManifest = useCaseStore.getState().caseInfo?.manifest as Record<string, unknown> | null;
        if (hasSegmentation(liveManifest)) useWorkflowStore.getState().tryLoadExistingSegmentation();
        // CROP-SHADE: overlay the cropped (blink/off-cornea) region in RED so it's highlighted. Only in the
        // preprocessing view (no segmentation yet); the seg overlay replaces it later. Best-effort (404 → skip).
        else if (hasCrop(liveManifest)) loadCropMask(volumeUrl.replace("volume.nii.gz", "crop-mask.nii.gz")).catch(() => {});
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volumeUrl, glTick]);

  // Is a raw (pre-preprocessing) snapshot available for this case? Re-checked on case load AND on
  // every (re)open (volumeUrl is cache-busted per openCase, so it changes right after a preprocess
  // finishes → the Before/after button appears as soon as the raw snapshot exists).
  useEffect(() => {
    const id = caseInfo?.case_id;
    if (!id) { setHasRaw(false); return; }
    let cancelled = false;
    // Reset to false for the NEW case BEFORE the async check. Otherwise the surface-crop auto-open effect
    // (above) can fire on the PREVIOUS case's stale hasRaw=true during the case-change commit, get clobbered
    // by the case-change reset (below), and — because it fired once — never re-open. Forcing a false→true
    // transition when THIS case's raw snapshot is confirmed re-triggers the auto-open cleanly, after the reset.
    setHasRaw(false);
    api
      .json<{ images: unknown[] }>(`/api/case/${id}/previews/context_raw`)
      .then((r) => !cancelled && setHasRaw((r.images || []).length > 0))
      .catch(() => !cancelled && setHasRaw(false));
    return () => { cancelled = true; };
  }, [caseInfo?.case_id, volumeUrl]);

  // #2/#3: when the VIEWED step is no longer a preprocessing step (e.g. after the group alignment advances it to
  // 4 / SAM2 to 5, or
  // the user inspects Cornea+), close the preprocessing overlays so the niivue Slices/Segmentation
  // view shows. (Fixes "after SAM2 the user is still in Fix-columns and can't see the segmentation".)
  useEffect(() => {
    if (!preprocStep) { setCompareView(false); setFixColsView(false); setFixAxialView(false); setStepsView(false); }
  }, [preprocStep]);

  // Leave the comparison / fix-columns / steps overlays when switching to a different case. The Slices|
  // Segmentation toggle DEFAULTS to Segmentation for a scan that already HAS a segmentation (#16 — opening a
  // segmented scan should land on its segmentation, not raw Slices); otherwise Slices (greyed until SAM2).
  useEffect(() => {
    setCompareView(false); setFixColsView(false); setFixAxialView(false); setStepsView(false); setAlignPane(true);
    autoCropOpenedRef.current = null;   // clear the once-per-case guard so the surface-crop auto-open can fire
    autoFixOpenedRef.current = null;    // ...and the border-editor auto-open below
    wfSet("showSegmentation", hasSegmentation(manifest));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseInfo?.case_id]);

  // OPEN THE BORDER EDITOR BY DEFAULT at the preprocessing-review step. Declared AFTER the case-change reset
  // above ON PURPOSE: effects run in declaration order, so with this first the reset closed the editor a
  // moment after it opened, and — the once-per-case guard having already been claimed — it never re-opened.
  // That is why rejecting a scan landed on the next one with no editing tools showing.
  // hasRaw is deliberately NOT required: it gates the before/after COMPARISON, while the editor only needs
  // the pass input, so demanding it kept the tools hidden on any scan without a raw snapshot.
  // Keyed on caseInfo.case_id — the SAME id the case-change reset above uses. Keying this on caseStore's
  // `caseId` instead meant the two ran in different commits: the editor opened, then the reset (firing later,
  // on the other id) closed it, and the guard was already claimed so it never reopened. That is why advancing
  // to the next scan landed with no editing tools even though opening the same scan from the sidebar worked.
  useEffect(() => {
    const cid = caseInfo?.case_id;
    if (!cid || !preprocStep || inspecting) return;
    if (autoFixOpenedRef.current === cid) return;
    autoFixOpenedRef.current = cid;
    setFixColsView(true);
    if (view !== "sagittal" && view !== "coronal") onView(null, "sagittal");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseInfo?.case_id, preprocStep, inspecting]);

  // #2 — keep the slice scrollbar in sync with niivue while a single-plane view is active (also catches the
  // user scrubbing by mouse-wheel/click). Polls niivue's crosshair position + per-axis slice count.
  const singlePlane = (view === "axial" || view === "coronal" || view === "sagittal") && !overlay2d && !noWebgl && !!volumeUrl;
  const lastSliderRef = useRef<number>(0);   // #2 — suppress poll sync briefly after a user drag (no snap-back)
  useEffect(() => {
    if (!singlePlane) { setSliceMax(0); return; }
    const id = window.setInterval(() => {
      const n = sliceCount(view);
      setSliceMax(n > 1 ? n - 1 : 0);
      // Don't overwrite the slider while the user is actively dragging it (setSliceIndex already moved
      // niivue synchronously); only adopt niivue's position when the user scrubs by other means.
      if (n > 1 && Date.now() - lastSliderRef.current >= 250) {
        const live = getSliceIndex(view); setSliceIdx((cur) => (live !== cur ? live : cur));
      }
    }, 160);
    return () => window.clearInterval(id);
  }, [singlePlane, view, volumeUrl, loading]);

  // ── Defect-marking (mark WRONG columns of the current sagittal/axial slice) ────────────────────────────
  // Active ONLY in mark mode on a single-plane sagittal/axial view; otherwise the overlay is inert (today's
  // behaviour). Marks are the manifest.defect_marks list, keyed by (orient, slice); drag over the slice to
  // add a column range, click to add one column. Persisted via caseStore (optimistic manifest update + POST).
  const defectOrient: "sagittal" | "axial" | null =
    view === "sagittal" || view === "axial" ? view : null;
  const markActive = markDefectMode && singlePlane && defectOrient != null;
  const allMarks: DefectMark[] = Array.isArray(manifest?.defect_marks)
    ? (manifest!.defect_marks as DefectMark[]) : [];
  // Columns already marked on the CURRENT (orient, slice) — rendered as pink bands + extended by a new drag.
  // union of ALL tags' cols on this slice (display shows every marked column regardless of tag)
  const curMarkCols: number[] = defectOrient
    ? [...new Set(allMarks.filter((m) => m.orient === defectOrient && m.slice === sliceIdx).flatMap((m) => m.cols))].sort((a, b) => a - b)
    : [];
  // Workflow: DRAW a column selection (drag over the slice) → it stays PENDING (light pink). Commit it to just
  // this frame ("This frame") OR across a slice RANGE: click "Start" on one slice, scrub, click "End" on
  // another — the pending columns apply to every slice in between. Pending shows on EVERY slice while open, so
  // scrubbing previews where the range will land.
  const [pendingCols, setPendingCols] = useState<number[]>([]);
  const [rangeStart, setRangeStart] = useState<number | null>(null);
  const dragRef = useRef<{ startCol: number } | null>(null);

  useEffect(() => {
    if (!markActive || !defectOrient) { setDefectBands(null, null); return; }
    setDefectBands(view, curMarkCols, pendingCols);   // committed = solid, pending selection = light preview
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [markActive, defectOrient, view, sliceIdx, JSON.stringify(pendingCols), JSON.stringify(curMarkCols)]);
  useEffect(() => () => setDefectBands(null, null), []);
  // Leaving mark mode / switching orientation / changing case resets the transient selection + range anchor.
  useEffect(() => { setPendingCols([]); setRangeStart(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [markDefectMode, view, caseInfo?.case_id]);

  // Union `cols` onto EVERY slice in [s0,s1] for this orient (keeping any existing marks there), then persist.
  const commitRange = (s0: number, s1: number, cols: number[]) => {
    if (!defectOrient || !cols.length) return;
    const lo = Math.min(s0, s1), hi = Math.max(s0, s1);
    const tag = defectTag || "edge_detection";
    // Merge into the marks of THIS orient+tag only (keyed by slice); marks of a DIFFERENT tag (or orient) are kept
    // untouched so several defect types can coexist on the same slice, each tagged.
    const byKey = new Map<number, Set<number>>();
    for (const m of allMarks) if (m.orient === defectOrient && (m.tag ?? "") === tag) byKey.set(m.slice, new Set(m.cols));
    for (let s = lo; s <= hi; s++) {
      const set = byKey.get(s) ?? new Set<number>();
      for (const c of cols) if (c >= 0) set.add(c);
      byKey.set(s, set);
    }
    const next: DefectMark[] = allMarks.filter((m) => !(m.orient === defectOrient && (m.tag ?? "") === tag));
    byKey.forEach((set, s) => { if (set.size) next.push({ orient: defectOrient, slice: s, cols: [...set].sort((a, b) => a - b), tag }); });
    void setDefectMarks(next);
  };
  const commitThisFrame = () => { if (pendingCols.length) { commitRange(sliceIdx, sliceIdx, pendingCols); setPendingCols([]); setRangeStart(null); } };
  const commitEnd = () => { if (pendingCols.length && rangeStart != null) { commitRange(rangeStart, sliceIdx, pendingCols); setPendingCols([]); setRangeStart(null); } };

  const onMarkDown = (e: React.PointerEvent) => {
    if (!markActive) return;
    const r = e.currentTarget.getBoundingClientRect();
    const col = screenToColumn(e.clientX - r.left, e.clientY - r.top, view);
    if (col == null) return;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    dragRef.current = { startCol: col };
    setPendingCols([col]);
  };
  const onMarkMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!markActive || !d) return;
    const r = e.currentTarget.getBoundingClientRect();
    const col = screenToColumn(e.clientX - r.left, e.clientY - r.top, view);
    if (col == null) return;
    const lo = Math.min(d.startCol, col), hi = Math.max(d.startCol, col);
    const range: number[] = [];
    for (let c = lo; c <= hi; c++) range.push(c);
    setPendingCols(range);
  };
  const onMarkUp = () => { dragRef.current = null; };   // the drawn selection stays PENDING until you commit it
  // RIGHT-CLICK: clear the committed band-run under the cursor on this slice (or drop the pending selection).
  const onMarkContext = (e: React.MouseEvent) => {
    if (!markActive || !defectOrient) return;
    e.preventDefault();
    const r = e.currentTarget.getBoundingClientRect();
    const col = screenToColumn(e.clientX - r.left, e.clientY - r.top, view);
    if (col == null) return;
    const set = new Set(curMarkCols);
    if (!set.has(col)) { setPendingCols([]); setRangeStart(null); return; }   // not on a band → just drop pending
    let lo = col, hi = col;
    while (set.has(lo - 1)) lo--;
    while (set.has(hi + 1)) hi++;
    // remove the band [lo,hi] from EVERY tag's mark on this slice (keep each tag's remaining cols)
    const rm = (c: number) => c >= lo && c <= hi;
    const next: DefectMark[] = [];
    for (const m of allMarks) {
      if (m.orient === defectOrient && m.slice === sliceIdx) {
        const keptCols = m.cols.filter((c) => !rm(c));
        if (keptCols.length) next.push({ ...m, cols: keptCols });
      } else next.push(m);
    }
    void setDefectMarks(next);
  };

  const mb = (on: boolean): React.CSSProperties => ({ background: on ? "rgba(255,93,176,0.28)" : "none",
    border: "1px solid #ff5db0", borderRadius: 4, color: on ? "#fff" : "#ff9fd0", opacity: on ? 1 : 0.45,
    cursor: on ? "pointer" : "default", fontSize: 11, padding: "1px 7px", fontWeight: 600 });

  const onView = (_: unknown, v: ViewName | null) => {
    if (!v) return;
    // The border editor can only host SAGITTAL and CORONAL (it marks along depth). Those views used to be the
    // only ones OFFERED while it was open — fine when opening it was a deliberate choice, wrong now that it
    // opens by default, because the reviewer was left with a single view button and no way to look at the
    // scan any other way. Every view is offered again; choosing one the editor cannot host simply closes it.
    if (fixColsView && v !== "sagittal" && v !== "coronal") setFixColsView(false);
    // SYMMETRIC RE-OPEN: the line above closes the border editor when leaving to a plane it can't host (Axial/
    // Multi/3D), but nothing re-opened it on the way back — so sagittal→axial→sagittal left fixColsView stuck
    // false and the cyan/red + band overlay never returned (only the bare niivue B-scan showed): the reviewer's
    // "the edges/lines disappear after switching views". Returning to a hostable plane at a preprocessing step
    // restores it. Mirrors the once-per-case auto-open conditions (preprocStep && !inspecting) and yields to the
    // sibling 2-D overlays (before/after, steps, axial-fix) when the user deliberately has one of those open.
    else if (!fixColsView && (v === "sagittal" || v === "coronal") && preprocStep && !inspecting
             && !compareView && !stepsView && !fixAxialView) setFixColsView(true);
    if (compareView && (v === "multi" || v === "render")) setCompareView(false);
    setViewState(v);
    setView(v);
  };

  // #1 — correction (cornea/background vet AND full correct) is 2-D slice work: the editable cornea is
  // drawn over the grayscale slice, and you paint per B-scan. If we enter correction from a 3-D/multiplanar
  // view the user can't see a single slice + the slice scrollbar, so drop to the B-scan (axial) plane on the
  // transition into correcting (doesn't fight a later manual view change).
  useEffect(() => {
    if (correcting && (view === "multi" || view === "render")) { setViewState("axial"); setView("axial"); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [correcting]);

  // Manual-GT comparison: swap in the auto-vs-GT agreement overlay viewer (its own niivue).
  if (gtViewerActive && gtViewerName && caseInfo?.case_id) {
    return (
      <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
        <GtCompareViewer
          caseId={caseInfo.case_id}
          name={gtViewerName}
          klass={gtViewerClass}
          onClose={() => wfSet("gtViewerActive", false)}
          onClassChange={(k) => wfSet("gtViewerClass", k)}
        />
      </div>
    );
  }

  // A per-subgroup consensus: the synchronized multi-scan grid is the view (2D, no WebGL needed) — EXCEPT
  // while correcting the consensus itself, where the niivue paint layer + PaintToolbar must show (otherwise
  // the grid would hide the brush). The grid returns as soon as the correction is saved/cancelled.
  if (isSubgroup && !correcting) {
    return (
      <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
        <SubgroupGrid />
      </div>
    );
  }

  // No WebGL (e.g. VS Code Simple Browser): fall back to the 2D PNG slice viewer
  // so the OCT + overlays are still viewable without a 3D context.
  if (noWebgl) {
    return (
      <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
        {backBanner}
        <div
          className="px-3 py-1.5 border-b text-xs"
          style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)", backgroundColor: "var(--c-surface)" }}
          title={noWebgl}
        >
          3D viewer needs WebGL2 (unavailable here) — showing 2D slices instead.
        </div>
        <div className="flex-1 min-h-0">
          <SliceGallery readOnly={inspecting} />
        </div>
      </div>
    );
  }

  // NOTE: the before/after, fix-columns and steps views are rendered as ABSOLUTE OVERLAYS at the end
  // of the main return — NOT early returns — so the niivue <canvas> never unmounts. niivue is a
  // singleton bound to the first canvas it attaches to (attach() early-returns the existing instance),
  // so unmounting/remounting the canvas left it bound to a dead element → a blank 3D view after
  // toggling those views (the bug: "switching scans after before/after shows no image"). Keeping the
  // canvas mounted under the overlay keeps niivue live, so scan switches reload + render correctly.
  return (
    <div className="relative flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      {backBanner}
      <div
        className="flex items-center gap-2 px-3 border-b overflow-x-auto [&>*]:shrink-0"
        style={{ minHeight: 36, borderColor: "var(--c-border)" }}
      >
        {/* Reviewer 2026-09-12: at "4. Aligned" the pane covers the canvas, so the slice/plane/tool controls are
            hidden — only the "⧉ Aligned" toggle (which brings the plain slices back) stays on this row. */}
        {!alignPaneOn && <ToggleButtonGroup size="small" exclusive value={showSeg ? "seg" : "slices"}
          onChange={(_, v) => { if (!v) return; wfSet("showSegmentation", v === "seg");
            // If enabling Segmentation before the overlay has finished loading (toggle now enabled from the
            // manifest, not the load), kick off a load so it appears without waiting for the open effect.
            if (v === "seg" && !segLoaded) void useWorkflowStore.getState().tryLoadExistingSegmentation(); }}
          title={(segLoaded || hasSeg) ? "" : "Run SAM2 first to view the segmentation overlay"}>
          <ToggleButton value="slices" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Slices</ToggleButton>
          <ToggleButton value="seg" disabled={!segLoaded && !hasSeg} sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Segmentation</ToggleButton>
        </ToggleButtonGroup>}
        {!alignPaneOn && (<>
        <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />
        <ToggleButtonGroup size="small" exclusive value={overlay2d ? orient2d : view} onChange={onView}>
          {/* sized to match the Slices/Segmentation group beside them — they had no sx, so they rendered at
              MUI's default height and made the toolbar look like two different bars stitched together */}
          <ToggleButton value="multi" sx={viewBtnSx} title="All three planes at once">Multi</ToggleButton>
          <ToggleButton value="axial" sx={viewBtnSx} title="Axial — the B-scan plane, as acquired">Axial</ToggleButton>
          {!cropRegionMode && <ToggleButton value="coronal" sx={viewBtnSx} title="Coronal — en-face">Coronal</ToggleButton>}
          <ToggleButton value="sagittal" sx={viewBtnSx} title="Sagittal — across frames. The plane the border editor works in.">Sagittal</ToggleButton>
          <ToggleButton value="render" sx={viewBtnSx} title="3-D volume render">3D</ToggleButton>
        </ToggleButtonGroup>
        {/* Display-only contrast / brightness / blur sliders (CSS filter on the viewer; does not change the
            data). Blur is disabled while Fix-columns is active. */}
        <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />
        {([
          ["C", contrast, setContrast, 50, 250, false, "Contrast"],
          ["B", brightness, setBrightness, 50, 250, false, "Brightness"],
          ["✷", blur, setBlur, 0, 4, true, "Gaussian blur"],
        ] as const).map(([lbl, val, setter, lo, hi, isBlur, title]) => (
          <span key={title} className="flex items-center gap-1" title={`${title} (display only)`}>
            <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>{lbl}</span>
            <input type="range" min={lo} max={hi} step={isBlur ? 0.5 : 5} value={isBlur && fixColsView ? 0 : val}
              disabled={isBlur && fixColsView}
              onChange={(e) => setter(Number(e.target.value))} style={{ width: 56, opacity: isBlur && fixColsView ? 0.4 : 1 }} />
          </span>
        ))}
        <button onClick={() => { setContrast(100); setBrightness(100); setBlur(0); }}
          title="Reset display adjustments"
          style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 11, padding: "1px 6px" }}>
          reset
        </button>
        <button onClick={() => { const c = !crisp; setCrisp(c); setInterpolationMode(c); }}
          title={crisp
            ? "CRISP display: true voxels — matches the fix-columns/before-after previews and the training volume. Steep corneal edges show the voxel staircase. Click for Smooth."
            : "SMOOTH display: linear — anti-aliases the edge staircase but blurs the frame axis (sagittal/coronal) and differs from the crisp previews. Click for Crisp. (Data is identical either way.)"}
          style={{ background: crisp ? "none" : "rgba(56,189,248,0.15)", border: "1px solid var(--c-border)", borderRadius: 4,
            color: crisp ? "var(--c-text-dim)" : "#38bdf8", cursor: "pointer", fontSize: 11, padding: "1px 7px", whiteSpace: "nowrap" }}>
          {crisp ? "◾ Crisp" : "◇ Smooth"}
        </button>
        <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />
        {/* HIDDEN IN FIX-COLUMNS (reviewer, 2026-09-02). There it only drove fcViewMode original<->both, which
            is a strict subset of what "⇆ view:" already does (original -> both -> corrected) — two buttons
            writing the same state, one of them able to reach less. Everywhere ELSE it is not redundant: it
            opens BeforeAfterViewer with the iterative-pass stepper, which the view cycle has no equivalent
            for, so it stays outside fix-columns. */}
        {hasRaw && preprocStep && !fixColsView && (
          <ToggleButton
            size="small"
            value="ba"
            // In fix-columns mode the raw-beside-corrected split is driven by fcViewMode (the panel reads it,
            // NOT compareView), so reflect THAT here — otherwise the button looks unselected while the split is on.
            selected={fixColsView ? fcViewMode !== "original" : compareView}
            onChange={() => {
              // Before/after is COMBINABLE with Fix-columns (it doesn't clear it): turning it on while
              // fix-columns is active makes the fix-columns panel show raw beside the markable corrected.
              // The fix-cols panel is gated on fcViewMode (not compareView), so in that mode DRIVE fcViewMode
              // (original <-> both) instead — else this button is a dead no-op there ("before/after doesn't work").
              // "both" keeps the original pane shown+editable (showOriginal stays true), so no tool gesture breaks.
              if (fixColsView) {            // unreachable since the button is hidden there; kept as a
                setFcViewMode((v) => (v === "original" ? "both" : "original"));   // safe fallback
                return;
              }
              const on = !compareView;
              if (on && !compareView && !fixColsView) preOverlayViewRef.current = view; // entering from normal → remember view
              setCompareView(on); setStepsView(false); setFixAxialView(false);
              if (on && (view === "multi" || view === "render")) onView(null, "sagittal");
              else if (!on && !fixColsView) { // back to normal → restore the pre-overlay view
                const prev = preOverlayViewRef.current; preOverlayViewRef.current = null;
                if (prev && prev !== view) onView(null, prev);
              }
            }}
            sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
            title="Show the original (raw) scan beside the preprocessed result (combines with Fix columns)"
          >
            ⇆ Before/after
          </ToggleButton>
        )}
        {/* Fix columns and Fix axial were retired here: the border editor IS the review surface now and
            opens automatically at the preprocessing step (autoFixOpenedRef above), so a button to reach it
            just sat between the reviewer and the thing they came to look at. Fix axial went with it; the
            sagittal editor plus the rigid per-frame passes cover that plane. fixAxialView state is kept so
            AxialGallery still renders if something sets it, only the entry point is gone. */}
        {hasRaw && preprocStep && (
          <ToggleButton
            size="small"
            value="steps"
            selected={stepsView}
            onChange={() => { setStepsView((v) => !v); setCompareView(false); setFixColsView(false); setFixAxialView(false); }}
            sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
            title="Every step of the last preprocessing run, including your inputs, as a tree — plus the detector filmstrip"
          >
            ⚙ Steps
          </ToggleButton>
        )}
        </>)}
        {alignedStep && (
          <ToggleButton
            size="small"
            value="aligned"
            selected={alignPane}
            onChange={() => setAlignPane((v) => !v)}
            sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
            data-testid="aligned-pane-toggle"
            title="The group alignment of this scan's subgroup — consensus + pairs, 3-D view, scrub (toggle off to see the plain slices)"
          >
            ⧉ Aligned
          </ToggleButton>
        )}
        {/* ⚑ Mark columns MOVED into the border editor's mode group (SliceGallery: "⚑ Mark"). It could not
            simply be relocated — the marking was a niivue overlay (setDefectBands + handlers on the niivue
            container, gated on singlePlane), so it went inert exactly when the border editor was open, which
            is now always. The ported version paints on the editor's own SVG and commits with Reject alongside
            the border and crop marks. markDefectMode state is left in place for the niivue path. */}
        <div className="flex-1" />
        {loading && <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Loading volume…</span>}
        {error && <span className="text-xs" style={{ color: "var(--c-red)" }}>{error}</span>}
      </div>

      <PaintToolbar />

      <div
        className="relative flex-1 min-h-0 min-w-0"
        onMouseMove={(e) => {
          if (!painting) { if (brush) setBrush(null); return; }
          const r = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - r.left, y = e.clientY - r.top;
          // #1 — size the cursor to the ACTUAL painted footprint (penSize voxels → screen px), not a guess.
          const sz = brushScreenSize(x, y);
          setBrush({ x, y, size: sz ? Math.max(4, sz.w) : Math.max(6, penSize * 3) });
          // #paint — keep the 2-D drawing overlay live during a stroke even if niivue's location callback
          // lags (rAF-coalesced; cheap). The brush updates the drawBitmap; this reflects it on screen.
          scheduleOverlay();
        }}
        onMouseUp={() => scheduleOverlay()}
        onMouseLeave={() => { setBrush(null); scheduleOverlay(); }}
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full"
          style={{ cursor: painting ? "crosshair" : "default", filter: viewerFilter || undefined }} />
        {/* Crop-approval PINK banner (basic view): a per-frame niivue voxel overlay is impractical, so a
            clearly-visible pink pill over the viewer signals that an auto de-tilt/crop was proposed but not
            applied — pointing the user to Fix-columns to review the pink region + Approve to bake it in. Shown
            only on the plain niivue view (not while a 2-D overlay covers the canvas) at the preprocessing step. */}
        {proposals.hasProposal && preprocStep && !overlay2d && !stepsView && (
          <div className="absolute top-2 left-1/2 z-30 pointer-events-none"
            style={{
              transform: "translateX(-50%)", maxWidth: "92%",
              display: "flex", alignItems: "center", gap: 8,
              padding: "4px 12px", borderRadius: 999, fontSize: 12, lineHeight: 1.3,
              color: "#fff", background: "rgba(255,93,176,0.22)", border: "1px solid #ff5db0",
              boxShadow: "0 0 8px 1px rgba(255,93,176,0.6)", whiteSpace: "nowrap",
              overflow: "hidden", textOverflow: "ellipsis",
            }}
            title={proposals.reasons.join(" · ") || "An automatic correction was detected but not applied."}>
            <span style={{ color: "#ff5db0", fontWeight: 700 }}>⬤</span>
            Auto-correction proposed
            {(proposals.hasDetilt ? [" de-tilt"] : []).concat(proposals.frames.length ? [` crop ${proposals.frames.length} frame${proposals.frames.length === 1 ? "" : "s"}`] : []).join(" ·")}
            {" — review in ▥ Fix columns and Approve to apply"}
          </div>
        )}
        {/* #paint — WebGL-independent 2-D drawing overlay: renders brush strokes that niivue's WebGL draw
            tile doesn't show on the WebKitGTK stack. pointer-events:none so paint clicks still hit niivue —
            EXCEPT in defect-mark mode, where it becomes interactive so the user can drag to mark WRONG columns
            (the same canvas also renders the pink defect bands via setDefectBands). */}
        <canvas ref={overlayRef} className="absolute inset-0 h-full w-full"
          style={{ pointerEvents: markActive ? "auto" : "none", cursor: markActive ? "col-resize" : "default" }}
          onPointerDown={markActive ? onMarkDown : undefined}
          onPointerMove={markActive ? onMarkMove : undefined}
          onPointerUp={markActive ? onMarkUp : undefined}
          onPointerLeave={markActive ? onMarkUp : undefined}
          onContextMenu={markActive ? onMarkContext : undefined} />
        {/* Defect-mark toolbar (only in mark mode): drag to select columns → commit to this frame OR a slice
            range (start → scrub → end). Pending selection shows light-pink on every slice; committed = solid. */}
        {markActive && (
          <div className="absolute top-2 right-2 z-30 flex items-center gap-1.5 pointer-events-auto flex-wrap justify-end"
            style={{ maxWidth: "78%", padding: "4px 9px", borderRadius: 12, fontSize: 12, color: "#fff",
              background: "rgba(20,20,24,0.7)", border: "1px solid #ff5db0" }}>
            <span style={{ color: "#ff5db0", fontWeight: 700 }}
              title="Drag over the slice to select WRONG columns, then commit to this frame or a slice range. Right-click a band to remove it.">⚑ {view}</span>
            {/* Defect-TYPE tag: newly committed marks carry this type (persisted per-mark for the assistant). */}
            <span style={{ opacity: 0.8 }}>type:</span>
            {([["edge_detection", "Edge"], ["curvature", "Curvature"], ["surface_roughness", "Roughness"]] as const).map(([k, lbl]) => (
              <button key={k} onClick={() => setWf("defectTag", k)} style={mb(defectTag === k)}
                title={`Tag new marks as “${lbl}”. Existing marks of other types stay.`}>{lbl}</button>
            ))}
            <input value={customTag} placeholder="other…"
              onChange={(e) => { const t = e.target.value; setCustomTag(t); if (t.trim()) setWf("defectTag", t.trim()); }}
              style={{ width: 78, background: "rgba(255,255,255,0.08)", border: `1px solid ${defectTag === customTag.trim() && customTag.trim() ? "#fff" : "#ff5db0"}`,
                borderRadius: 4, color: "#fff", fontSize: 11, padding: "1px 6px" }}
              title="Type a custom defect type — new marks use it (overrides the preset until you click one)." />
            <span style={{ opacity: 0.9 }}>{pendingCols.length ? `${pendingCols.length} col${pendingCols.length === 1 ? "" : "s"} selected` : "drag to select"}</span>
            {rangeStart != null && <span style={{ color: "#ffd0e8" }}>· start@{rangeStart + 1}</span>}
            <button disabled={!pendingCols.length} onClick={commitThisFrame} style={mb(pendingCols.length > 0)}
              title="Save the selected columns on THIS slice only.">this frame</button>
            <button disabled={!pendingCols.length} onClick={() => setRangeStart(sliceIdx)} style={mb(pendingCols.length > 0)}
              title="Set the START slice of a range; scrub to another slice, then click End.">start</button>
            <button disabled={!pendingCols.length || rangeStart == null} onClick={commitEnd} style={mb(pendingCols.length > 0 && rangeStart != null)}
              title="Apply the selected columns to every slice from Start to the current slice.">end</button>
            {(pendingCols.length > 0 || rangeStart != null) && (
              <button onClick={() => { setPendingCols([]); setRangeStart(null); }} style={mb(true)} title="Discard the current selection/range.">×sel</button>
            )}
            <span style={{ opacity: 0.85 }}>· {allMarks.length} saved{allMarks.length > 0 && ` (${
              Object.entries(allMarks.reduce((a, m) => { const t = m.tag ?? "untagged"; a[t] = (a[t] || 0) + 1; return a; }, {} as Record<string, number>))
                .map(([t, n]) => `${({ edge_detection: "edge", curvature: "curv", surface_roughness: "rough" } as Record<string, string>)[t] || t}:${n}`).join(" ")})`}</span>
            {allMarks.length > 0 && <button onClick={() => void setDefectMarks([])} style={mb(true)} title="Clear ALL marks on this scan.">clear all</button>}
          </div>
        )}
        {showBrush && (
          // Brush-size cursor (approx — actual voxels depend on zoom): shows the active pen + size.
          <div
            className="absolute rounded-full pointer-events-none"
            style={{
              left: brush!.x, top: brush!.y, transform: "translate(-50%, -50%)",
              width: brush!.size, height: brush!.size,
              border: `1.5px solid ${PEN_COLOR[penLabel] ?? "#fff"}`,
              boxShadow: "0 0 0 1px rgba(0,0,0,0.5)",
              background: penLabel === 0 ? "transparent" : `${PEN_COLOR[penLabel]}22`,
            }}
          />
        )}
        {noWebgl && (
          <div
            className="absolute inset-0 flex items-center justify-center flex-col gap-2 p-6 text-center"
            style={{ color: "var(--c-text-dim)" }}
          >
            <span style={{ fontSize: 14, color: "var(--c-red)" }}>3D viewer unavailable</span>
            <span style={{ fontSize: 12, opacity: 0.85, maxWidth: 460 }}>{noWebgl}</span>
          </div>
        )}
        {/* The before/after, fix-columns and steps panels render as overlays over the (still-mounted)
            niivue canvas but ONLY cover the content area — the single top toolbar stays visible and
            drives their orientation + display filter, so the user is never pushed into a nested sub-UI.
            (See the note above the return for why the canvas must stay mounted.) */}
        {/* Before/after alone → the pass-stepper viewer. Fix-columns (with or without before/after) → the
            marking panel, which shows raw beside the corrected when before/after is also on (showRaw). When
            BOTH are on only ONE overlay mounts (the fix-cols panel) so they never stack. */}
        {compareView && !fixColsView && volumeUrl && (
          <div className={`absolute inset-0 z-20 flex flex-col${crisp ? "" : " oct-smooth-imgs"}`} style={{ backgroundColor: "var(--c-bg)" }}>
            <BeforeAfterViewer orient={orient2d} filter={viewerFilter} />
          </div>
        )}
        {fixColsView && volumeUrl && (
          <div className={`absolute inset-0 z-20 flex flex-col${crisp ? "" : " oct-smooth-imgs"}`} style={{ backgroundColor: "var(--c-bg)" }}>
            <SliceGallery key={`fixcols-${editorResetNonce}`} fixCols showRaw={fcViewMode !== "original"} showOriginal={fcViewMode !== "corrected"} viewMode={fcViewMode}
              orientProp={orient2d} filterCss={viewerFilter} readOnly={inspecting}
              onToggleRaw={() => setFcViewMode((v) => (v === "original" ? "both" : v === "both" ? "corrected" : "original"))}
              onNeedOriginalPane={() => setFcViewMode((v) => (v === "corrected" ? "both" : v))} />
          </div>
        )}
        {/* AXIAL fix-tool — a 2-D overlay over the (still-mounted) niivue canvas. Mutually exclusive with the
            other 2-D panels (its toolbar onChange clears them). Do NOT early-return: the niivue singleton must
            stay mounted (see the note above the return). */}
        {fixAxialView && volumeUrl && (
          <div className={`absolute inset-0 z-20 flex flex-col${crisp ? "" : " oct-smooth-imgs"}`} style={{ backgroundColor: "var(--c-bg)" }}>
            <AxialGallery filterCss={viewerFilter} readOnly={inspecting}
              initialSlice={getSliceIndex("axial")} sliceTotal={sliceCount("axial")} />
          </div>
        )}
        {stepsView && volumeUrl && (
          <div className="absolute inset-0 z-20 flex flex-col" style={{ backgroundColor: "var(--c-bg)" }}>
            <StepsViewer onClose={() => setStepsView(false)} />
          </div>
        )}
        {/* STEP 4 "Aligned" pane: the alignment content (AlignTabs — the same component the pairs dialog wraps) over the
            still-mounted niivue canvas; the group id is this scan's alignment subgroup. */}
        {alignPaneOn && (
          <div className="absolute inset-0 z-20 flex flex-col" style={{ backgroundColor: "var(--c-bg)" }} data-testid="aligned-pane">
            <AlignTabs key={`${alignGid!}#${alignReloadNonce}`} gid={alignGid}
              onReAlign={() => setAlignAskOpen(true)}
              title={<span style={{ fontSize: 13 }}><b style={{ color: "#ec4899" }}>4. Aligned</b>{effStep === 5 ? <span style={{ color: "#4ade80" }}> · axial changes approved</span> : null} — subgroup group <b>{alignGid}</b></span>} />
            {/* "⧉ Align group" on an already-aligned group: ask for the subgroups again, then re-run (reviewer 2026-09-12). */}
            <SubgroupAlignDialog open={alignAskOpen} gid={alignGid} defaultForce onClose={() => setAlignAskOpen(false)}
              onAligned={() => {
                void useCaseStore.getState().refreshCaseInfo();
                const wf = useWorkflowStore.getState(); wf.set("casesVersion", wf.casesVersion + 1);
                setAlignReloadNonce((n) => n + 1);
              }} />
          </div>
        )}
        {stage === 4 && (
          <div className="absolute inset-0 z-20 flex flex-col" style={{ backgroundColor: "var(--c-bg)" }}>
            <MotionPanel />
          </div>
        )}
      </div>

      {/* #2 — slice scrollbar for the single-plane (axial/coronal/sagittal) niivue views (works in BOTH
          normal and paint modes). The 2-D overlays (before/after, fix-columns) carry their own slider. */}
      {singlePlane && sliceMax > 0 && (
        <div className="flex items-center gap-3 px-4 py-2 border-t" style={{ borderColor: "var(--c-border)", backgroundColor: "var(--c-surface)" }}>
          <span className="text-xs whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>
            {view} slice {sliceIdx + 1} / {sliceMax + 1}
          </span>
          <Slider size="small" min={0} max={sliceMax} value={Math.min(sliceIdx, sliceMax)}
            onChange={(_, v) => { const n = v as number; lastSliderRef.current = Date.now(); setSliceIdx(n); setSliceIndex(view, n); }} />
        </div>
      )}
    </div>
  );
}
