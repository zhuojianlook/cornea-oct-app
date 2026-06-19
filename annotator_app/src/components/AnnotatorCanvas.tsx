/* Niivue viewer + brush cursor for the annotator (mirrors the main app's hardened brush UX:
   paint/navigate, sized brush cursor coloured by pen, hidden in 3D / navigate). */

import { useEffect, useRef, useState } from "react";
import { CircularProgress, ToggleButton, ToggleButtonGroup } from "@mui/material";
import { attach, setView, webglFailure, type ViewName } from "../niivue/nvController";
import { useStore } from "../store/annotatorStore";
import { tr } from "../i18n";

const PEN_COLOR: Record<number, string> = { 0: "#c7c7cc", 1: "#1ab2ff", 2: "#ff453a" };
const PEN_KEY = { 0: "pen.erase", 1: "pen.cornea", 2: "pen.scar" } as const;
const VIEWS: ViewName[] = ["multi", "axial", "coronal", "sagittal", "render"];

export function AnnotatorCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [view, setV] = useState<ViewName>("multi");
  const [noWebgl, setNoWebgl] = useState<string | null>(null);
  const [brush, setBrush] = useState<{ x: number; y: number } | null>(null);
  const loaded = useStore((s) => s.loaded);
  const busy = useStore((s) => s.busy);
  const paintMode = useStore((s) => s.paintMode);
  const penLabel = useStore((s) => s.penLabel);
  const penSize = useStore((s) => s.penSize);
  const volName = useStore((s) => s.activeVolume?.name);
  const lang = useStore((s) => s.lang);

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
        onMouseMove={(e) => {
          if (!painting) { if (brush) setBrush(null); return; }
          const r = e.currentTarget.getBoundingClientRect();
          setBrush({ x: e.clientX - r.left, y: e.clientY - r.top });
        }}
        onMouseLeave={() => setBrush(null)}
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" style={{ cursor: painting ? "crosshair" : "default" }} />
        {showBrush && (
          <div className="absolute rounded-full pointer-events-none" style={{
            left: brush!.x, top: brush!.y, transform: "translate(-50%, -50%)",
            width: Math.max(6, penSize * 3), height: Math.max(6, penSize * 3),
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
    </div>
  );
}
