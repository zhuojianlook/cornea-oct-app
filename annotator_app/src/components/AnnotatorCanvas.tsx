/* Niivue viewer + brush cursor for the annotator (mirrors the main app's hardened brush UX:
   paint/navigate, sized brush cursor coloured by pen, hidden in 3D / navigate). */

import { useEffect, useRef, useState } from "react";
import { ToggleButton, ToggleButtonGroup } from "@mui/material";
import { attach, setView, webglFailure, type ViewName } from "../niivue/nvController";
import { useStore } from "../store/annotatorStore";

const PEN_COLOR: Record<number, string> = { 0: "#c7c7cc", 1: "#1ab2ff", 2: "#ff453a" };
const VIEWS: ViewName[] = ["multi", "axial", "coronal", "sagittal", "render"];

export function AnnotatorCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [view, setV] = useState<ViewName>("multi");
  const [noWebgl, setNoWebgl] = useState<string | null>(null);
  const [brush, setBrush] = useState<{ x: number; y: number } | null>(null);
  const loaded = useStore((s) => s.loaded);
  const paintMode = useStore((s) => s.paintMode);
  const penLabel = useStore((s) => s.penLabel);
  const penSize = useStore((s) => s.penSize);
  const volName = useStore((s) => s.activeVolume?.name);

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
      <div className="flex items-center gap-2 px-3 border-b" style={{ height: 36, borderColor: "var(--c-border)" }}>
        <ToggleButtonGroup size="small" exclusive value={view} onChange={(_, v) => { if (v) { setV(v); setView(v); } }}>
          {VIEWS.map((vw) => (
            <ToggleButton key={vw} value={vw} style={{ textTransform: "capitalize" }}>{vw === "render" ? "3D" : vw}</ToggleButton>
          ))}
        </ToggleButtonGroup>
        <span className="flex-1" />
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>{volName ?? "no volume"}</span>
      </div>
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
        {!loaded && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ color: "var(--c-text-dim)" }}>
            <span>Select a volume from the left to annotate.</span>
          </div>
        )}
      </div>
    </div>
  );
}
