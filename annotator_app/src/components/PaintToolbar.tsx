/* Pen controls, grouped into sections: Mode · Pen (+ size, fill) · Actions · Opacity.
   Disabled until a volume is loaded. */

import type { ReactNode } from "react";
import { Button, Slider, ToggleButton, ToggleButtonGroup, Tooltip } from "@mui/material";
import { useStore, type Pen } from "../store/annotatorStore";
import { tr, type TKey } from "../i18n";

const PENS: { value: Pen; key: TKey; color: string }[] = [
  { value: 1, key: "pen.cornea", color: "#1ab2ff" },
  { value: 2, key: "pen.scar", color: "#ff453a" },
  { value: 3, key: "pen.background", color: "#9aa0aa" }, // seed "not cornea" for Smart fill
  { value: 0, key: "pen.erase", color: "#c7c7cc" },
];

const Label = ({ children }: { children: ReactNode }) => (
  <span style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--c-text-dim)" }}>{children}</span>
);
const Sep = () => <div style={{ width: 1, height: 24, background: "var(--c-border)", flex: "none" }} />;

const tbSx = { py: 0.4, px: 1.1, fontSize: 12, textTransform: "none" as const, lineHeight: 1.4 };

export function PaintToolbar() {
  const { loaded, penLabel, penSize, penFilled, paintMode, drawOpacity, lang,
          setPenLabel, setPenSize, setPenFilled, setPaintMode, setDrawOpacity, smartFill, undo, clearDrawing } = useStore();
  const off = !loaded;

  return (
    <div className="flex items-center gap-3 px-4 border-b overflow-x-auto [&>*]:flex-none"
      style={{ minHeight: 50, flex: "none", backgroundColor: "var(--c-surface)", borderColor: "var(--c-border)", opacity: off ? 0.55 : 1 }}>

      {/* Mode */}
      <ToggleButtonGroup size="small" exclusive disabled={off} value={paintMode ? "paint" : "nav"}
        onChange={(_, v) => v !== null && setPaintMode(v === "paint")}>
        <ToggleButton value="paint" sx={tbSx}>✏ {tr(lang, "tb.paint")}</ToggleButton>
        <ToggleButton value="nav" sx={tbSx}>✋ {tr(lang, "tb.navigate")}</ToggleButton>
      </ToggleButtonGroup>

      <Sep />

      {/* Pen */}
      <Label>{tr(lang, "tb.pen")}</Label>
      <ToggleButtonGroup size="small" exclusive disabled={off || !paintMode} value={penLabel} onChange={(_, v) => v !== null && setPenLabel(v)}>
        {PENS.map((p) => (
          <ToggleButton key={p.value} value={p.value} sx={tbSx}>
            <span style={{ width: 10, height: 10, borderRadius: "50%", background: p.color, marginRight: 7,
                           display: "inline-block", boxShadow: "0 0 0 1px rgba(0,0,0,0.35)" }} />
            {tr(lang, p.key)}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>

      <Tooltip title={tr(lang, "tb.sizeTip")} arrow>
        <div className="flex items-center gap-2" style={{ width: 132 }}>
          <Label>{tr(lang, "tb.size")}</Label>
          <Slider size="small" min={1} max={40} step={1} value={penSize} valueLabelDisplay="auto"
            disabled={off} onChange={(_, v) => setPenSize(v as number)} />
          <span style={{ fontSize: 11, width: 16, textAlign: "right", color: "var(--c-text-dim)" }}>{penSize}</span>
        </div>
      </Tooltip>

      <Tooltip title={tr(lang, "tb.fillTip")} arrow>
        <ToggleButton size="small" value="filled" selected={penFilled} disabled={off || !paintMode}
          onChange={() => setPenFilled(!penFilled)} sx={tbSx}>▣ {tr(lang, "tb.fill")}</ToggleButton>
      </Tooltip>

      <Sep />

      {/* Actions */}
      <Tooltip title={tr(lang, "tb.smartTip")} arrow>
        <span>
          <Button size="small" variant="contained" disableElevation disabled={off} onClick={() => smartFill()} sx={tbSx}>✨ {tr(lang, "tb.smartFill")}</Button>
        </span>
      </Tooltip>
      <Button size="small" variant="outlined" disabled={off} onClick={() => undo()} sx={tbSx}>↶ {tr(lang, "tb.undo")}</Button>
      <Button size="small" variant="outlined" color="inherit" disabled={off} onClick={() => clearDrawing()} sx={tbSx}>{tr(lang, "tb.clear")}</Button>

      <Sep />

      {/* Opacity */}
      <Tooltip title={tr(lang, "tb.opacityTip")} arrow>
        <div className="flex items-center gap-2" style={{ width: 150 }}>
          <Label>{tr(lang, "tb.opacity")}</Label>
          <Slider size="small" min={0.1} max={1} step={0.05} value={drawOpacity} valueLabelDisplay="auto"
            valueLabelFormat={(v) => `${Math.round(v * 100)}%`}
            disabled={off} onChange={(_, v) => setDrawOpacity(v as number)} />
        </div>
      </Tooltip>
    </div>
  );
}
