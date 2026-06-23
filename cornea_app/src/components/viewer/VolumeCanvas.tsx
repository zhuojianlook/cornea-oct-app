/* Niivue volume viewer — base grayscale OCT volume with view controls. */

import { useEffect, useRef, useState } from "react";
import { ToggleButton, ToggleButtonGroup } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import { attach, loadVolume, setView, webglFailure, type ViewName } from "../../niivue/nvController";
import { PaintToolbar } from "./PaintToolbar";
import { SliceGallery } from "./SliceGallery";
import { SubgroupGrid } from "./SubgroupGrid";
import { GtCompareViewer } from "./GtCompareViewer";

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
  const paintMode = useWorkflowStore((s) => s.paintMode);
  const penLabel = useWorkflowStore((s) => s.penLabel);
  const penSize = useWorkflowStore((s) => s.penSize);
  const [view, setViewState] = useState<ViewName>("multi");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [noWebgl, setNoWebgl] = useState<string | null>(null);
  const [brush, setBrush] = useState<{ x: number; y: number } | null>(null);
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

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      {backBanner}
      <div
        className="flex items-center gap-2 px-3 border-b"
        style={{ height: 36, borderColor: "var(--c-border)" }}
      >
        <ToggleButtonGroup size="small" exclusive value={view} onChange={onView}>
          <ToggleButton value="multi">Multi</ToggleButton>
          <ToggleButton value="axial">Axial</ToggleButton>
          <ToggleButton value="coronal">Coronal</ToggleButton>
          <ToggleButton value="sagittal">Sagittal</ToggleButton>
          <ToggleButton value="render">3D</ToggleButton>
        </ToggleButtonGroup>
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
          style={{ cursor: painting ? "crosshair" : "default" }} />
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
        {noWebgl ? (
          <div
            className="absolute inset-0 flex items-center justify-center flex-col gap-2 p-6 text-center"
            style={{ color: "var(--c-text-dim)" }}
          >
            <span style={{ fontSize: 14, color: "var(--c-red)" }}>3D viewer unavailable</span>
            <span style={{ fontSize: 12, opacity: 0.85, maxWidth: 460 }}>{noWebgl}</span>
          </div>
        ) : (
          !volumeUrl && (
            <div
              className="absolute inset-0 flex items-center justify-center flex-col gap-2 pointer-events-none"
              style={{ color: "var(--c-text-dim)" }}
            >
              <span style={{ fontSize: 14 }}>No volume loaded</span>
              <span style={{ fontSize: 12, opacity: 0.7 }}>Register an OCT volume from the sidebar.</span>
            </div>
          )
        )}
      </div>
    </div>
  );
}
