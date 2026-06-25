/* Toolbar grouped into clear sections: TOOL · (PEN | WAND, by active tool) · EDIT · DISPLAY.
   The Pen controls show only while painting; the Wand controls only while the wand is active — so each
   tool exposes just its own options (#3). Disabled until a volume is loaded. */

import type { ReactNode } from "react";
import { Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, LinearProgress, Slider, ToggleButton, ToggleButtonGroup, Tooltip } from "@mui/material";
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
const Sep = () => <div style={{ width: 1, height: 26, background: "var(--c-border)", flex: "none", margin: "0 2px" }} />;
const Dot = ({ color, sz = 10 }: { color: string; sz?: number }) => (
  <span style={{ width: sz, height: sz, borderRadius: "50%", background: color, marginRight: 6, display: "inline-block",
                 boxShadow: "0 0 0 1px rgba(0,0,0,0.35)" }} />
);

const tbSx = { py: 0.4, px: 1.1, fontSize: 12, textTransform: "none" as const, lineHeight: 1.4 };

export function PaintToolbar() {
  const { loaded, penLabel, penSize, penFilled, tool, drawOpacity, lang, busy, smartPct, canConfirm,
          canUndo, canRedo, brightness, contrast, locked, confirmClear,
          wandThreshold, wandTolerance, wandMode, wandScope, wandTarget,
          setPenLabel, setPenSize, setPenFilled, setTool, setWandThreshold, setWandTolerance, setWandMode, setWandScope,
          setWandTarget, setDrawOpacity, smartFill, confirmFill, showAnnotations, setShowAnnotations,
          undo, redo, requestClear, cancelClear, clearDrawing, setBrightness, setContrast, resetWindow, toggleLock } = useStore();
  const off = !loaded || busy;            // lock controls while smart fill computes
  const filling = busy && smartPct !== null;
  const paint = tool === "paint";
  const wand = tool === "wand";

  return (
    <div className="flex items-center gap-2.5 px-4 border-b overflow-x-auto [&>*]:flex-none"
      style={{ position: "relative", minHeight: 50, flex: "none", backgroundColor: "var(--c-surface)", borderColor: "var(--c-border)", opacity: !loaded ? 0.55 : 1 }}>
      {filling && (
        <LinearProgress variant="determinate" value={smartPct ?? 0}
          sx={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 3 }} />
      )}

      {/* ── TOOL ─────────────────────────────────────────── */}
      <Label>{tr(lang, "tb.tool")}</Label>
      <ToggleButtonGroup size="small" exclusive disabled={off} value={tool}
        onChange={(_, v) => v !== null && setTool(v)}>
        <ToggleButton value="paint" sx={tbSx}>✏ {tr(lang, "tb.paint")}</ToggleButton>
        <Tooltip title={tr(lang, "tb.wandTip")} arrow><ToggleButton value="wand" sx={tbSx}>✨ {tr(lang, "tb.wand")}</ToggleButton></Tooltip>
        <ToggleButton value="navigate" sx={tbSx}>✋ {tr(lang, "tb.navigate")}</ToggleButton>
      </ToggleButtonGroup>

      <Sep />

      {/* ── PEN (paint mode) ─────────────────────────────── */}
      {paint && <>
        <Label>{tr(lang, "tb.pen")}</Label>
        <ToggleButtonGroup size="small" exclusive disabled={off} value={penLabel} onChange={(_, v) => v !== null && setPenLabel(v)}>
          {PENS.map((p) => (
            <ToggleButton key={p.value} value={p.value} sx={tbSx}><Dot color={p.color} />{tr(lang, p.key)}</ToggleButton>
          ))}
        </ToggleButtonGroup>
        <Tooltip title={tr(lang, "tb.sizeTip")} arrow>
          <div className="flex items-center gap-2" style={{ width: 130 }}>
            <Label>{tr(lang, "tb.size")}</Label>
            <Slider size="small" min={1} max={60} step={1} value={penSize} valueLabelDisplay="auto"
              disabled={off} onChange={(_, v) => setPenSize(v as number)} />
            <span style={{ fontSize: 11, width: 16, textAlign: "right", color: "var(--c-text-dim)" }}>{penSize}</span>
          </div>
        </Tooltip>
        <Tooltip title={tr(lang, "tb.fillTip")} arrow>
          <ToggleButton size="small" value="filled" selected={penFilled} disabled={off}
            onChange={() => setPenFilled(!penFilled)} sx={tbSx}>▣ {tr(lang, "tb.fill")}</ToggleButton>
        </Tooltip>
        <Sep />
      </>}

      {/* ── WAND (wand mode) ─────────────────────────────── */}
      {wand && <>
        <Label>{tr(lang, "tb.target")}</Label>
        <Tooltip title={tr(lang, "tb.targetTip")} arrow>
          <ToggleButtonGroup size="small" exclusive disabled={off} value={wandTarget} onChange={(_, v) => v !== null && setWandTarget(v)}>
            <ToggleButton value={2} sx={tbSx}><Dot color="#ff453a" />{tr(lang, "pen.scar")}</ToggleButton>
            <ToggleButton value={1} sx={tbSx}><Dot color="#1ab2ff" />{tr(lang, "pen.cornea")}</ToggleButton>
          </ToggleButtonGroup>
        </Tooltip>
        <Tooltip title={tr(lang, "tb.modeTip")} arrow>
          <ToggleButtonGroup size="small" exclusive disabled={off} value={wandMode} onChange={(_, v) => v !== null && setWandMode(v)}>
            <ToggleButton value="threshold" sx={tbSx}>{tr(lang, "tb.modeThreshold")}</ToggleButton>
            <ToggleButton value="tolerance" sx={tbSx}>{tr(lang, "tb.modeTolerance")}</ToggleButton>
          </ToggleButtonGroup>
        </Tooltip>
        <Tooltip title={tr(lang, "tb.scopeTip")} arrow>
          <ToggleButtonGroup size="small" exclusive disabled={off} value={wandScope} onChange={(_, v) => v !== null && setWandScope(v)}>
            <ToggleButton value="3d" sx={tbSx}>3D</ToggleButton>
            <ToggleButton value="2d" sx={tbSx}>2D</ToggleButton>
          </ToggleButtonGroup>
        </Tooltip>
        {wandMode === "threshold" ? (
          <Tooltip title={tr(lang, "tb.wandThreshTip")} arrow>
            <div className="flex items-center gap-2" style={{ width: 160 }}>
              <Label>{tr(lang, "tb.threshold")}</Label>
              <Slider size="small" min={0} max={1} step={0.01} value={wandThreshold} valueLabelDisplay="auto"
                valueLabelFormat={(v) => `${Math.round(v * 100)}%`} disabled={off} onChange={(_, v) => setWandThreshold(v as number)} />
              <span style={{ fontSize: 11, width: 30, textAlign: "right", color: "var(--c-text-dim)" }}>{Math.round(wandThreshold * 100)}%</span>
            </div>
          </Tooltip>
        ) : (
          <Tooltip title={tr(lang, "tb.toleranceTip")} arrow>
            <div className="flex items-center gap-2" style={{ width: 160 }}>
              <Label>{tr(lang, "tb.tolerance")}</Label>
              <Slider size="small" min={0} max={0.5} step={0.005} value={wandTolerance} valueLabelDisplay="auto"
                valueLabelFormat={(v) => `±${Math.round(v * 100)}%`} disabled={off} onChange={(_, v) => setWandTolerance(v as number)} />
              <span style={{ fontSize: 11, width: 34, textAlign: "right", color: "var(--c-text-dim)" }}>±{Math.round(wandTolerance * 100)}%</span>
            </div>
          </Tooltip>
        )}
        <Sep />
      </>}

      {/* ── EDIT ─────────────────────────────────────────── */}
      <Label>{tr(lang, "tb.edit")}</Label>
      {paint && (
        <Tooltip title={tr(lang, "tb.smartTip")} arrow>
          <span>
            <Button size="small" variant="contained" disableElevation disabled={off} onClick={() => smartFill()} sx={tbSx}
              startIcon={filling ? <CircularProgress size={12} color="inherit" /> : undefined}>
              {filling ? `${tr(lang, "tb.smartFill")} ${smartPct}%` : `✨ ${tr(lang, "tb.smartFill")}`}
            </Button>
          </span>
        </Tooltip>
      )}
      <Tooltip title={tr(lang, "tb.confirmTip")} arrow>
        <span>
          <Button size="small" variant="contained" color="success" disableElevation disabled={off || !canConfirm}
            onClick={() => confirmFill()} sx={tbSx}>✓ {tr(lang, "tb.confirmFill")}</Button>
        </span>
      </Tooltip>
      <Button size="small" variant="outlined" disabled={off || !canUndo} onClick={() => undo()} sx={tbSx}>↶ {tr(lang, "tb.undo")}</Button>
      <Button size="small" variant="outlined" disabled={off || !canRedo} onClick={() => redo()} sx={tbSx}>↷ {tr(lang, "tb.redo")}</Button>
      <Button size="small" variant="outlined" color="inherit" disabled={off} onClick={() => requestClear()} sx={tbSx}>{tr(lang, "tb.clear")}</Button>

      <Sep />

      {/* ── DISPLAY ──────────────────────────────────────── */}
      <Label>{tr(lang, "tb.display")}</Label>
      <Tooltip title={tr(lang, "tb.annotationsTip")} arrow>
        <ToggleButton size="small" value="anno" selected={showAnnotations} disabled={off}
          onChange={() => setShowAnnotations(!showAnnotations)} sx={tbSx}>
          {showAnnotations ? "👁" : "🚫"} {tr(lang, "tb.annotations")}
        </ToggleButton>
      </Tooltip>
      <Tooltip title={tr(lang, "tb.opacityTip")} arrow>
        <div className="flex items-center gap-2" style={{ width: 140 }}>
          <Label>{tr(lang, "tb.opacity")}</Label>
          <Slider size="small" min={0.1} max={1} step={0.05} value={drawOpacity} valueLabelDisplay="auto"
            valueLabelFormat={(v) => `${Math.round(v * 100)}%`}
            disabled={off || !showAnnotations} onChange={(_, v) => setDrawOpacity(v as number)} />
        </div>
      </Tooltip>
      <Tooltip title={tr(lang, "tb.bcTip")} arrow>
        <div className="flex items-center gap-2" style={{ width: 116 }}>
          <Label>{tr(lang, "tb.brightness")}</Label>
          <Slider size="small" min={-1} max={1} step={0.02} value={brightness} disabled={off} onChange={(_, v) => setBrightness(v as number)} />
        </div>
      </Tooltip>
      <Tooltip title={tr(lang, "tb.bcTip")} arrow>
        <div className="flex items-center gap-2" style={{ width: 116 }}>
          <Label>{tr(lang, "tb.contrast")}</Label>
          <Slider size="small" min={-1} max={1} step={0.02} value={contrast} disabled={off} onChange={(_, v) => setContrast(v as number)} />
        </div>
      </Tooltip>
      <Button size="small" variant="outlined" color="inherit" disabled={off} onClick={() => resetWindow()} sx={tbSx}>{tr(lang, "tb.reset")}</Button>

      <Sep />

      {/* Label locks — protect a label from brush/erase/smart-fill/wand */}
      <Tooltip title={tr(lang, "tb.lockTip")} arrow>
        <span className="flex items-center gap-1.5">
          <Label>{tr(lang, "tb.lock")}</Label>
          {([[1, "pen.cornea", "#1ab2ff"], [2, "pen.scar", "#ff453a"], [0, "pen.background", "#9aa0aa"]] as const).map(([v, key, color]) => (
            <ToggleButton key={v} size="small" value={v} selected={locked.includes(v)} disabled={off}
              onChange={() => toggleLock(v)} sx={tbSx}>
              {locked.includes(v) ? "🔒" : "🔓"}
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, margin: "0 5px",
                             display: "inline-block", boxShadow: "0 0 0 1px rgba(0,0,0,0.35)" }} />
              {tr(lang, key as TKey)}
            </ToggleButton>
          ))}
        </span>
      </Tooltip>

      {/* Clear confirmation */}
      <Dialog open={confirmClear} onClose={() => cancelClear()}>
        <DialogTitle>{tr(lang, "tb.clearTitle")}</DialogTitle>
        <DialogContent><DialogContentText>{tr(lang, "tb.clearBody")}</DialogContentText></DialogContent>
        <DialogActions>
          <Button onClick={() => cancelClear()}>{tr(lang, "save.cancel")}</Button>
          <Button variant="contained" color="error" disableElevation onClick={() => clearDrawing()}>{tr(lang, "tb.clear")}</Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}
