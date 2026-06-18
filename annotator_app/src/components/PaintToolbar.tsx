/* Pen controls: paint/navigate, pens (cornea/scar/erase), brush size, filled-region pen, smart fill,
   undo, clear, and drawing opacity. Disabled until a volume is loaded. */

import { Button, Slider, ToggleButton, ToggleButtonGroup, Tooltip } from "@mui/material";
import { useStore, type Pen } from "../store/annotatorStore";

const PENS: { value: Pen; label: string; color: string }[] = [
  { value: 1, label: "Cornea", color: "#1ab2ff" },
  { value: 2, label: "Scar", color: "#ff453a" },
  { value: 0, label: "Erase", color: "#8e8e93" },
];

export function PaintToolbar() {
  const { loaded, penLabel, penSize, penFilled, paintMode, drawOpacity,
          setPenLabel, setPenSize, setPenFilled, setPaintMode, setDrawOpacity, smartFill, undo, clearDrawing } = useStore();
  const off = !loaded;

  return (
    <div className="flex items-center gap-3 px-3 border-b overflow-x-auto [&>*]:shrink-0"
      style={{ minHeight: 40, backgroundColor: "var(--c-surface)", borderColor: "var(--c-border)", opacity: off ? 0.5 : 1 }}>
      <ToggleButtonGroup size="small" exclusive disabled={off} value={paintMode ? "paint" : "nav"}
        onChange={(_, v) => v !== null && setPaintMode(v === "paint")}>
        <ToggleButton value="paint" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>✏ Paint</ToggleButton>
        <ToggleButton value="nav" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>✋ Navigate</ToggleButton>
      </ToggleButtonGroup>
      <div style={{ width: 1, height: 22, background: "var(--c-border)" }} />

      <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>Pen</span>
      <ToggleButtonGroup size="small" exclusive disabled={off || !paintMode} value={penLabel} onChange={(_, v) => v !== null && setPenLabel(v)}>
        {PENS.map((p) => (
          <ToggleButton key={p.value} value={p.value}>
            <span style={{ width: 10, height: 10, borderRadius: "50%", background: p.color, marginRight: 6, display: "inline-block" }} />
            {p.label}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>

      <Tooltip title="Brush size (voxels)">
        <div className="flex items-center gap-2" style={{ width: 120 }}>
          <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>Size</span>
          <Slider size="small" min={1} max={15} step={1} value={penSize} valueLabelDisplay="auto"
            disabled={off} onChange={(_, v) => setPenSize(v as number)} />
        </div>
      </Tooltip>
      <Tooltip title="Filled pen: draw a closed outline → fill the enclosed region (one stroke per patch).">
        <ToggleButton size="small" value="filled" selected={penFilled} disabled={off || !paintMode}
          onChange={() => setPenFilled(!penFilled)} sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>▣ Fill region</ToggleButton>
      </Tooltip>
      <Tooltip title="Smart fill (GrowCut): scribble a little Cornea AND Scar on a few slices, then propagate through the whole 3-D volume by intensity similarity — so you don't paint every slice.">
        <Button size="small" variant="contained" disabled={off} onClick={() => smartFill()}
          sx={{ py: 0.25, px: 1.2, fontSize: 12, textTransform: "none" }}>✨ Smart fill</Button>
      </Tooltip>
      <Button size="small" variant="outlined" disabled={off} onClick={() => undo()}
        sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>↶ Undo</Button>
      <Button size="small" variant="outlined" color="inherit" disabled={off} onClick={() => clearDrawing()}
        sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Clear</Button>

      <Tooltip title="Label overlay opacity">
        <div className="flex items-center gap-2" style={{ width: 120 }}>
          <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>Opacity</span>
          <Slider size="small" min={0.1} max={1} step={0.05} value={drawOpacity}
            disabled={off} onChange={(_, v) => setDrawOpacity(v as number)} />
        </div>
      </Tooltip>
    </div>
  );
}
