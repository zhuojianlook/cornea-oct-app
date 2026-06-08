/* Niivue volume viewer — base grayscale OCT volume with view controls. */

import { useEffect, useRef, useState } from "react";
import { ToggleButton, ToggleButtonGroup } from "@mui/material";
import { useCaseStore } from "../../store/caseStore";
import { usePaintStore } from "../../store/paintStore";
import { attach, loadVolume, setView, type ViewName } from "../../niivue/nvController";
import { PaintToolbar } from "./PaintToolbar";

export function VolumeCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const volumeUrl = useCaseStore((s) => s.volumeUrl);
  const [view, setViewState] = useState<ViewName>("multi");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (canvasRef.current) attach(canvasRef.current);
  }, []);

  useEffect(() => {
    if (!volumeUrl) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    loadVolume(volumeUrl)
      .then(() => {
        if (cancelled) return;
        setView(view);
        // Re-show an existing segmentation when reopening a finished case.
        usePaintStore.getState().tryLoadExistingSegmentation();
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

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
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

      <div className="relative flex-1 min-h-0 min-w-0">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
        {!volumeUrl && (
          <div
            className="absolute inset-0 flex items-center justify-center flex-col gap-2 pointer-events-none"
            style={{ color: "var(--c-text-dim)" }}
          >
            <span style={{ fontSize: 14 }}>No volume loaded</span>
            <span style={{ fontSize: 12, opacity: 0.7 }}>Register an OCT volume from the sidebar.</span>
          </div>
        )}
      </div>
    </div>
  );
}
