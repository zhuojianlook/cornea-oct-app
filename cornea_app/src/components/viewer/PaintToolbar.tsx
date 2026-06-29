/* Pen controls for the correction drawing layer (cornea / background / scar / erase) + brush size,
   filled-region pen, and the Smart fill (3-D GrowCut) that propagates scribbles across all slices. */

import { ToggleButton, ToggleButtonGroup, Slider, Tooltip, Button, CircularProgress } from "@mui/material";
import { useWorkflowStore, type PenLabel } from "../../store/workflowStore";
import { setDrawOpacity } from "../../niivue/nvController";

const PENS: { value: PenLabel; label: string; color: string }[] = [
  { value: 1, label: "Cornea", color: "#1ab2ff" },
  { value: 2, label: "Background", color: "#8e8e93" },   // grey: a real seed (Smart fill) that → canonical 0 on save
  { value: 3, label: "Scar", color: "#ff453a" },
  { value: 0, label: "Erase", color: "#8e8e93" },
];

export function PaintToolbar() {
  const { penLabel, penSize, penFilled, paintMode, correcting, drawOpacity, corneaOnlyPaint, smartFillBusy,
          setPenLabel, setPenSize, setPenFilled, setPaintMode, runSmartFill, undoCorrection, set } = useWorkflowStore();
  if (!correcting) return null;
  // #3/#11 cornea/background vet step exposes two clear pens: Cornea (label 1, blue) and Background (label 2,
  // GREY). Background is a real non-zero SEED (so Smart fill/GrowCut has a background seed — fixes the hang)
  // that maps to canonical 0 on save; painting it over cornea removes that cornea → background. (No separate
  // Erase needed here — Background grey IS the "remove cornea" tool.)
  const pens = corneaOnlyPaint ? [PENS[0], PENS[1]] : PENS;   // [Cornea(1,blue), Background(2,grey)]

  return (
    <div className="flex items-center gap-3 px-3 border-b overflow-x-auto [&>*]:shrink-0" style={{ minHeight: 36, borderColor: "var(--c-border)" }}>
      <ToggleButtonGroup size="small" exclusive value={paintMode ? "paint" : "nav"}
        onChange={(_, v) => v !== null && setPaintMode(v === "paint")}>
        <ToggleButton value="paint" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>✏ Paint</ToggleButton>
        <ToggleButton value="nav" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>✋ Navigate</ToggleButton>
      </ToggleButtonGroup>
      <div style={{ width: 1, height: 22, background: "var(--c-border)" }} />
      <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)", opacity: paintMode ? 1 : 0.4 }}>
        Pen
      </span>
      <ToggleButtonGroup size="small" exclusive disabled={!paintMode} value={penLabel} onChange={(_, v) => v !== null && setPenLabel(v)}>
        {pens.map((p) => (
          <ToggleButton key={p.value} value={p.value}>
            <span
              style={{ width: 10, height: 10, borderRadius: "50%", background: p.color, marginRight: 6, display: "inline-block" }}
            />
            {p.label}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
      <Tooltip title="Brush size (voxels)">
        <div className="flex items-center gap-2" style={{ width: 120 }}>
          <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>Size</span>
          <Slider size="small" min={1} max={15} step={1} value={penSize} valueLabelDisplay="auto"
            onChange={(_, v) => setPenSize(v as number)} />
        </div>
      </Tooltip>
      <Tooltip title="Filled pen: draw a closed outline around a region and the enclosed area is painted (one stroke per patch).">
        <ToggleButton size="small" value="filled" selected={penFilled}
          onChange={() => setPenFilled(!penFilled)} sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>
          ▣ Fill region
        </ToggleButton>
      </Tooltip>
      <Tooltip title="Undo the last brush stroke / smart fill (Ctrl+Z also works in niivue)">
        <Button size="small" variant="outlined" onClick={() => undoCorrection()}
          sx={{ py: 0.25, px: 1.2, fontSize: 12, textTransform: "none" }}>
          ↶ Undo
        </Button>
      </Tooltip>
      <Tooltip title="Smart fill (GrowCut): after scribbling a little Cornea, Background AND Scar on a few slices, this propagates those labels through the whole 3-D volume by intensity similarity — so you don't paint every slice/view. Runs on the full volume (a few seconds). Review and correct, then Apply.">
        <span>
          <Button size="small" variant="contained" onClick={() => runSmartFill()} disabled={smartFillBusy}
            startIcon={smartFillBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
            sx={{ py: 0.25, px: 1.2, fontSize: 12, textTransform: "none" }}>
            {smartFillBusy ? "Filling…" : "✨ Smart fill"}
          </Button>
        </span>
      </Tooltip>
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
