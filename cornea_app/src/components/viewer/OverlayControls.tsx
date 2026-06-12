/* Segmentation overlay controls + per-class voxel/volume summary. */

import { Slider, Switch, FormControlLabel } from "@mui/material";
import { useWorkflowStore } from "../../store/workflowStore";

interface SegStat {
  voxel_count?: number;
  volume_mm3?: number;
}

export function OverlayControls() {
  const { segLoaded, segOpacity, showSegmentation, segQa, setSegOpacity, toggleSegmentation } = useWorkflowStore();
  if (!segLoaded) return null;

  const segs = (segQa?.segments as Record<string, SegStat> | undefined) || undefined;

  return (
    <div className="rounded p-2 flex flex-col gap-2" style={{ backgroundColor: "var(--c-surface2)" }}>
      <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
        Segmentation overlay
      </div>
      <FormControlLabel
        control={<Switch size="small" checked={showSegmentation} onChange={(e) => toggleSegmentation(e.target.checked)} />}
        label={<span style={{ fontSize: 12 }}>Show overlay</span>}
      />
      <div className="flex items-center gap-2">
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
          Opacity
        </span>
        <Slider
          min={0}
          max={1}
          step={0.05}
          value={segOpacity}
          disabled={!showSegmentation}
          onChange={(_, v) => setSegOpacity(v as number)}
        />
      </div>
      {segs && (
        <div className="flex flex-col gap-1">
          {Object.entries(segs).map(([name, stat]) => (
            <div key={name} className="flex justify-between text-xs">
              <span style={{ color: "var(--c-text-dim)" }}>{name}</span>
              <span>
                {stat.voxel_count?.toLocaleString() ?? "—"} vox
                {typeof stat.volume_mm3 === "number" ? ` · ${stat.volume_mm3.toFixed(2)} mm³` : ""}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
