/* Niivue viewer + brush cursor for the annotator (mirrors the main app's hardened brush UX:
   paint/navigate, sized brush cursor coloured by pen, hidden in 3D / navigate). */

import { useEffect, useRef, useState } from "react";
import { CircularProgress, Slider, ToggleButton, ToggleButtonGroup } from "@mui/material";
import { attach, setView, webglFailure, brushScreenSize, lockCrosshair, restoreCrosshair, setStroke, type ViewName } from "../niivue/nvController";
import { useStore } from "../store/annotatorStore";
import { tr, type TKey } from "../i18n";

const PEN_COLOR: Record<number, string> = { 0: "#c7c7cc", 1: "#1ab2ff", 2: "#ff453a", 3: "#9aa0aa" };
const PEN_KEY = { 0: "pen.erase", 1: "pen.cornea", 2: "pen.scar", 3: "pen.background" } as const;
const VIEWS: ViewName[] = ["multi", "axial", "coronal", "sagittal", "render"];

// Through-plane voxel axis per plane (axial=z=2, coronal=y=1, sagittal=x=0) for the slice scrollbars.
const VIEW_AXIS: { plane: "axial" | "coronal" | "sagittal"; axis: 0 | 1 | 2; key: TKey }[] = [
  { plane: "axial", axis: 2, key: "view.axial" },
  { plane: "coronal", axis: 1, key: "view.coronal" },
  { plane: "sagittal", axis: 0, key: "view.sagittal" },
];

export function AnnotatorCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [view, setV] = useState<ViewName>("multi");
  const [noWebgl, setNoWebgl] = useState<string | null>(null);
  const [brush, setBrush] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const loaded = useStore((s) => s.loaded);
  const busy = useStore((s) => s.busy);
  const paintMode = useStore((s) => s.paintMode);
  const penLabel = useStore((s) => s.penLabel);
  const penSize = useStore((s) => s.penSize);
  const volName = useStore((s) => s.activeVolume?.name);
  const lang = useStore((s) => s.lang);
  const dims = useStore((s) => s.dims);
  const vox = useStore((s) => s.vox);
  const setSliceAxis = useStore((s) => s.setSliceAxis);
  const syncVox = useStore((s) => s.syncVox);

  useEffect(() => {
    if (canvasRef.current) { attach(canvasRef.current); setNoWebgl(webglFailure()); }
  }, []);

  const painting = loaded && paintMode && view !== "render";
  const showBrush = painting && brush;

  if (noWebgl) {
    return (
      <div className="flex flex-1 items-center justify-center p-6 text-center" style={{ color: "var(--c-red)" }}>
        <span style={{ maxWidth: 460 }}>{noWebgl}</span>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      {/* View bar */}
      <div className="flex items-center gap-3 px-4 border-b flex-none min-w-0" style={{ height: 44, borderColor: "var(--c-border)" }}>
        <ToggleButtonGroup size="small" exclusive value={view} onChange={(_, v) => { if (v) { setV(v); setView(v); } }}>
          {VIEWS.map((vw) => (
            <ToggleButton key={vw} value={vw} sx={{ py: 0.4, px: 1.25, fontSize: 12, textTransform: "none" }}>
              {tr(lang, `view.${vw}`)}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        <span className="flex-1" />
        {loaded && paintMode && view !== "render" && (
          <span className="flex items-center gap-1.5" style={{ fontSize: 11, color: "var(--c-text-dim)" }}>
            <span style={{ width: 9, height: 9, borderRadius: "50%", background: PEN_COLOR[penLabel],
                           display: "inline-block", boxShadow: "0 0 0 1px rgba(0,0,0,0.35)" }} />
            {tr(lang, PEN_KEY[penLabel])} · {penSize}px
          </span>
        )}
        <span className="truncate flex-none" style={{ fontSize: 12, maxWidth: 240, color: loaded ? "var(--c-text)" : "var(--c-text-dim)" }}
          title={volName ?? ""}>{volName ?? tr(lang, "canvas.noVol")}</span>
      </div>

      {/* Canvas + overlays */}
      <div
        className="relative flex-1 min-h-0 min-w-0"
        onMouseDownCapture={() => { if (painting) setStroke(true); }}
        onMouseDown={() => { if (painting) restoreCrosshair(); }}
        onMouseUp={() => { if (painting) { restoreCrosshair(); setStroke(false); syncVox(); } }}
        onMouseMove={(e) => {
          // #2 — painting must not drag the crosshair (other views stay on the chosen slice). While a
          // paint stroke is active (left button down) restore the crosshair niivue just moved; otherwise
          // (hover / navigate) remember the current crosshair so a stroke can snap back to it.
          if (painting && (e.buttons & 1)) restoreCrosshair(); else lockCrosshair();
          if (!painting) { if (brush) setBrush(null); return; }
          const r = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - r.left, y = e.clientY - r.top;
          const sz = brushScreenSize(x, y); // accurate voxel→screen size; fall back to a rough estimate
          setBrush({ x, y, w: sz ? Math.max(3, sz.w) : penSize * 3, h: sz ? Math.max(3, sz.h) : penSize * 3 });
        }}
        onMouseLeave={() => { if (painting) { restoreCrosshair(); setStroke(false); syncVox(); } setBrush(null); }}
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" style={{ cursor: painting ? "crosshair" : "default" }} />
        {showBrush && (
          <div className="absolute pointer-events-none" style={{
            left: brush!.x, top: brush!.y, transform: "translate(-50%, -50%)",
            width: brush!.w, height: brush!.h, borderRadius: 2,
            border: `1.5px solid ${PEN_COLOR[penLabel] ?? "#fff"}`, boxShadow: "0 0 0 1px rgba(0,0,0,0.5)",
            background: penLabel === 0 ? "transparent" : `${PEN_COLOR[penLabel]}22`,
          }} />
        )}
        {/* Opaque cover when nothing is shown yet (hides niivue's default canvas text). */}
        {!loaded && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center px-6"
            style={{ background: "var(--c-bg)", color: "var(--c-text-dim)" }}>
            {busy ? (
              <>
                <CircularProgress size={26} />
                <span style={{ fontSize: 13 }}>{tr(lang, "canvas.loading")}</span>
              </>
            ) : (
              <>
                <span style={{ fontSize: 40, opacity: 0.55 }}>👁</span>
                <span style={{ fontSize: 14, color: "var(--c-text)" }}>{tr(lang, "canvas.noVolume")}</span>
                <span style={{ fontSize: 12, maxWidth: 320 }}>{tr(lang, "canvas.noVolumeHint")}</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Per-view slice scrollbars (#1): scrub a specific slice in each plane (one per single view,
          all three in Multi). Stays in sync with scroll-wheel navigation via onLocationChange. */}
      {loaded && view !== "render" && dims && (
        <div className="flex items-center gap-5 px-4 border-t flex-none overflow-x-auto"
          style={{ minHeight: 40, borderColor: "var(--c-border)", backgroundColor: "var(--c-surface)" }}>
          {VIEW_AXIS.filter((a) => view === "multi" || view === a.plane).map((a) => {
            const n = dims[a.axis];
            const cur = Math.min(Math.max(0, vox[a.axis]), Math.max(0, n - 1));
            return (
              <div key={a.plane} className="flex items-center gap-2" style={{ flex: 1, minWidth: 150 }}>
                <span style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em",
                               color: "var(--c-text-dim)", whiteSpace: "nowrap" }}>{tr(lang, a.key)}</span>
                <Slider size="small" min={0} max={Math.max(0, n - 1)} step={1} value={cur} valueLabelDisplay="auto"
                  disabled={n <= 1} onChange={(_, v) => setSliceAxis(a.axis, v as number)} />
                <span style={{ fontSize: 11, width: 58, textAlign: "right", color: "var(--c-text-dim)",
                               whiteSpace: "nowrap" }}>{cur + 1}/{n}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
