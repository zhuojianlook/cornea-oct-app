/* Niivue volume viewer — base grayscale OCT volume with view controls. */

import { useEffect, useRef, useState } from "react";
import { ToggleButton, ToggleButtonGroup } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import { attach, loadVolume, setView, setSegmentationOpacity, webglFailure, type ViewName } from "../../niivue/nvController";
import { PaintToolbar } from "./PaintToolbar";
import { SliceGallery } from "./SliceGallery";
import { SubgroupGrid } from "./SubgroupGrid";
import { GtCompareViewer } from "./GtCompareViewer";
import { BeforeAfterViewer } from "./BeforeAfterViewer";
import { StepsViewer } from "./StepsViewer";

export function VolumeCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const volumeUrl = useCaseStore((s) => s.volumeUrl);
  const caseInfo = useCaseStore((s) => s.caseInfo);
  const setCaseId = useCaseStore((s) => s.setCaseId);
  const openCase = useCaseStore((s) => s.openCase);
  const reviewConsensusId = useWorkflowStore((s) => s.reviewConsensusId);
  const gtViewerActive = useWorkflowStore((s) => s.gtViewerActive);
  const gtViewerName = useWorkflowStore((s) => s.gtViewerName);
  const gtViewerClass = useWorkflowStore((s) => s.gtViewerClass);
  const wfSet = useWorkflowStore((s) => s.set);
  const correcting = useWorkflowStore((s) => s.correcting);
  const segLoaded = useWorkflowStore((s) => s.segLoaded);
  const segOpacity = useWorkflowStore((s) => s.segOpacity);
  const paintMode = useWorkflowStore((s) => s.paintMode);
  const penLabel = useWorkflowStore((s) => s.penLabel);
  const penSize = useWorkflowStore((s) => s.penSize);
  const [view, setViewState] = useState<ViewName>("multi");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [noWebgl, setNoWebgl] = useState<string | null>(null);
  const [brush, setBrush] = useState<{ x: number; y: number } | null>(null);
  // Before/after comparison (raw vs preprocessed). The button + view only exist once the scan has
  // been preprocessed — i.e. its raw snapshot (context_raw previews) was captured.
  const [hasRaw, setHasRaw] = useState(false);
  const [compareView, setCompareView] = useState(false);
  // Manual "Fix columns" correction (mark bad B-scan frames → re-run preprocessing) lives in the 2D
  // SliceGallery; on the WebGL/3D desktop path it was otherwise unreachable. This opens it.
  const [fixColsView, setFixColsView] = useState(false);
  // Preprocessing-steps filmstrip (per-stage diagnostic) — also otherwise unreachable on the 3D path.
  const [stepsView, setStepsView] = useState(false);
  // Display-only contrast / brightness / gaussian-blur (CSS filter on the viewer — covers BOTH the niivue
  // canvas and the 2-D overlays, in the ONE top toolbar so the user never enters a nested sub-UI for them).
  // Blur is reset + disabled while Fix-columns is active (you need the true pixels to mark a border).
  const [contrast, setContrast] = useState(100);   // %
  const [brightness, setBrightness] = useState(100); // %
  const [blur, setBlur] = useState(0);              // px
  const viewerFilter = `contrast(${contrast}%) brightness(${brightness}%)` + (blur > 0 && !fixColsView ? ` blur(${blur}px)` : "");
  // The 2-D overlays (before/after, fix-columns) are driven by the SAME top toolbar — no nested sub-UI.
  // They're 2-D, so Multi/3D don't apply; fix-columns marks along depth, so it's coronal/sagittal only.
  const overlay2d = compareView || fixColsView;
  const orient2d: "axial" | "coronal" | "sagittal" = fixColsView
    ? (view === "coronal" ? "coronal" : "sagittal")
    : (view === "axial" || view === "coronal" || view === "sagittal") ? view : "sagittal";
  // Slices | Segmentation overlay toggle (Segmentation greyed until SAM2 has run). On the niivue path it
  // simply shows/hides the cornea/scar overlay by opacity; the 2-D gallery reads the same flag.
  const [showSeg, setShowSeg] = useState(true);
  useEffect(() => { setSegmentationOpacity(showSeg && segLoaded ? segOpacity : 0); }, [showSeg, segLoaded, segOpacity]);
  // brush-cursor colour per pen (0 erase, 1 cornea, 2 background, 3 scar)
  const PEN_COLOR: Record<number, string> = { 0: "#c7c7cc", 1: "#1ab2ff", 2: "#ff8c1a", 3: "#ff453a" };
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
        // Re-show an existing segmentation when reopening a finished case.
        useWorkflowStore.getState().tryLoadExistingSegmentation();
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volumeUrl]);

  // Is a raw (pre-preprocessing) snapshot available for this case? Re-checked on case load AND on
  // every (re)open (volumeUrl is cache-busted per openCase, so it changes right after a preprocess
  // finishes → the Before/after button appears as soon as the raw snapshot exists).
  useEffect(() => {
    const id = caseInfo?.case_id;
    if (!id) { setHasRaw(false); return; }
    let cancelled = false;
    api
      .json<{ images: unknown[] }>(`/api/case/${id}/previews/context_raw`)
      .then((r) => !cancelled && setHasRaw((r.images || []).length > 0))
      .catch(() => !cancelled && setHasRaw(false));
    return () => { cancelled = true; };
  }, [caseInfo?.case_id, volumeUrl]);

  // Leave the comparison / fix-columns / steps overlays when switching to a different case.
  useEffect(() => { setCompareView(false); setFixColsView(false); setStepsView(false); }, [caseInfo?.case_id]);

  const onView = (_: unknown, v: ViewName | null) => {
    if (!v) return;
    setViewState(v);
    setView(v);
  };

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

  // A per-subgroup consensus: the synchronized multi-scan grid is the view (2D, no WebGL needed).
  if (isSubgroup) {
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
          <SliceGallery />
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
        className="flex items-center gap-2 px-3 border-b"
        style={{ height: 36, borderColor: "var(--c-border)" }}
      >
        <ToggleButtonGroup size="small" exclusive value={showSeg ? "seg" : "slices"}
          onChange={(_, v) => { if (v) setShowSeg(v === "seg"); }}
          title={segLoaded ? "" : "Run SAM2 first to view the segmentation overlay"}>
          <ToggleButton value="slices" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Slices</ToggleButton>
          <ToggleButton value="seg" disabled={!segLoaded} sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Segmentation</ToggleButton>
        </ToggleButtonGroup>
        <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />
        <ToggleButtonGroup size="small" exclusive value={overlay2d ? orient2d : view} onChange={onView}>
          {!overlay2d && <ToggleButton value="multi">Multi</ToggleButton>}
          {!fixColsView && <ToggleButton value="axial">Axial</ToggleButton>}
          <ToggleButton value="coronal">Coronal</ToggleButton>
          <ToggleButton value="sagittal">Sagittal</ToggleButton>
          {!overlay2d && <ToggleButton value="render">3D</ToggleButton>}
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
        <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />
        {hasRaw && (
          <ToggleButton
            size="small"
            value="ba"
            selected={compareView}
            onChange={() => {
              // Before/after is COMBINABLE with Fix-columns (it doesn't clear it): turning it on while
              // fix-columns is active makes the fix-columns panel show raw beside the markable corrected.
              const on = !compareView;
              setCompareView(on); setStepsView(false);
              if (on && (view === "multi" || view === "render")) onView(null, "sagittal");
            }}
            sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
            title="Show the original (raw) scan beside the preprocessed result (combines with Fix columns)"
          >
            ⇆ Before/after
          </ToggleButton>
        )}
        {hasRaw && (
          <ToggleButton
            size="small"
            value="fix"
            selected={fixColsView}
            onChange={() => {
              // Fix-columns is COMBINABLE with Before/after (it doesn't clear it).
              const on = !fixColsView;
              setFixColsView(on); setStepsView(false);
              if (on && view !== "coronal" && view !== "sagittal") onView(null, "sagittal");
              else if (!on) void openCase(); // leaving fix-cols: reload the 3D volume in case a re-run changed it
            }}
            sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
            title="Manually fix mis-aligned B-scan frames: mark bad frames on a slice, then re-run preprocessing on them"
          >
            ▥ Fix columns
          </ToggleButton>
        )}
        {hasRaw && (
          <ToggleButton
            size="small"
            value="steps"
            selected={stepsView}
            onChange={() => { setStepsView((v) => !v); setCompareView(false); setFixColsView(false); }}
            sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
            title="Preview every preprocessing step (hist-eq → edge → quadratic fit → 3D active → warp) for the central slice"
          >
            ⚙ Steps
          </ToggleButton>
        )}
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
          setBrush({ x: e.clientX - r.left, y: e.clientY - r.top });
        }}
        onMouseLeave={() => setBrush(null)}
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full"
          style={{ cursor: painting ? "crosshair" : "default", filter: viewerFilter || undefined }} />
        {showBrush && (
          // Brush-size cursor (approx — actual voxels depend on zoom): shows the active pen + size.
          <div
            className="absolute rounded-full pointer-events-none"
            style={{
              left: brush!.x, top: brush!.y, transform: "translate(-50%, -50%)",
              width: Math.max(6, penSize * 3), height: Math.max(6, penSize * 3),
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
          <div className="absolute inset-0 z-20 flex flex-col" style={{ backgroundColor: "var(--c-bg)" }}>
            <BeforeAfterViewer orient={orient2d} filter={viewerFilter} />
          </div>
        )}
        {fixColsView && volumeUrl && (
          <div className="absolute inset-0 z-20 flex flex-col" style={{ backgroundColor: "var(--c-bg)" }}>
            <SliceGallery fixCols showRaw={compareView} orientProp={orient2d} filterCss={viewerFilter} />
          </div>
        )}
        {stepsView && volumeUrl && (
          <div className="absolute inset-0 z-20 flex flex-col" style={{ backgroundColor: "var(--c-bg)" }}>
            <StepsViewer onClose={() => setStepsView(false)} />
          </div>
        )}
      </div>
    </div>
  );
}
