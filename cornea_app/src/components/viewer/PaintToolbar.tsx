/* Pen controls for the correction drawing layer (cornea / background / scar / erase). */

import { ToggleButton, ToggleButtonGroup, Slider, Tooltip } from "@mui/material";
import { useWorkflowStore, type PenLabel } from "../../store/workflowStore";
import { setDrawOpacity } from "../../niivue/nvController";

const PENS: { value: PenLabel; label: string; color: string }[] = [
  { value: 1, label: "Cornea", color: "#1ab2ff" },
  { value: 2, label: "Background", color: "#ff8c1a" },
  { value: 3, label: "Scar", color: "#ff453a" },
  { value: 0, label: "Erase", color: "#8e8e93" },
];

export function PaintToolbar() {
  const { penLabel, correcting, drawOpacity, setPenLabel, set } = useWorkflowStore();
  if (!correcting) return null;

  return (
    <div className="flex items-center gap-3 px-3 border-b" style={{ height: 36, borderColor: "var(--c-border)" }}>
      <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
        Pen
      </span>
      <ToggleButtonGroup size="small" exclusive value={penLabel} onChange={(_, v) => v !== null && setPenLabel(v)}>
        {PENS.map((p) => (
          <ToggleButton key={p.value} value={p.value}>
            <span
              style={{ width: 10, height: 10, borderRadius: "50%", background: p.color, marginRight: 6, display: "inline-block" }}
            />
            {p.label}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
      <Tooltip title="Drawing opacity">
        <div className="flex items-center gap-2" style={{ width: 140 }}>
          <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
            Opacity
          </span>
          <Slider
            min={0}
            max={1}
            step={0.05}
            value={drawOpacity}
            onChange={(_, v) => {
              const o = v as number;
              set("drawOpacity", o);
              setDrawOpacity(o);
            }}
          />
        </div>
      </Tooltip>
    </div>
  );
}
