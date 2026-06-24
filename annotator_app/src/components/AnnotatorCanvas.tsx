/* Niivue viewer + brush cursor for the annotator (mirrors the main app's hardened brush UX:
   paint/navigate, sized brush cursor coloured by pen, hidden in 3D / navigate). */

import { useEffect, useRef, useState } from "react";
import { Button, CircularProgress, Slider, ToggleButton, ToggleButtonGroup } from "@mui/material";
import { attach, setView, webglFailure, brushScreenSize, lockCrosshair, restoreCrosshair, setStroke, redraw, paintBrush, beginStroke, tileAtScreen, tileThroughAxis, voxAtScreen, voxAtScreenClamped, flushCompose, setCrosshairAtScreen, type ViewName } from "../niivue/nvController";
import { useStore } from "../store/annotatorStore";
import { tr, type TKey } from "../i18n";

/** Live voxel counts + cursor intensity — isolated so per-move intensity updates don't re-render the canvas. */
function Readout() {
  const lang = useStore((s) => s.lang);
  const cornea = useStore((s) => s.corneaVox);
  const scar = useStore((s) => s.scarVox);
  const tool = useStore((s) => s.tool);
  const intensity = useStore((s) => s.cursorIntensity);
  const intensity01 = useStore((s) => s.cursorIntensity01);
  return (
    <span className="flex items-center gap-3 flex-none" style={{ fontSize: 11, color: "var(--c-text-dim)" }}>
      <span><b style={{ color: "#1ab2ff" }}>{cornea.toLocaleString()}</b> {tr(lang, "pen.cornea")}</span>
      <span><b style={{ color: "#ff453a" }}>{scar.toLocaleString()}</b> {tr(lang, "pen.scar")}</span>
      {tool === "wand" && (
        <span className="flex items-center gap-1.5" title={tr(lang, "tb.intensityTip")}>
          {tr(lang, "tb.intensity")}:
          {intensity != null ? <>
            <b style={{ color: "var(--c-text)" }}>{Math.round(intensity)}</b>
            <span style={{ width: 56, height: 6, borderRadius: 3, background: "var(--c-border)", position: "relative", overflow: "hidden" }}>
              <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${Math.round((intensity01 ?? 0) * 100)}%`,
                             background: "linear-gradient(90deg,#3a3a3c,#e5e5ea)" }} />
            </span>
            <b style={{ color: "var(--c-text)", width: 30, textAlign: "right" }}>{Math.round((intensity01 ?? 0) * 100)}%</b>
          </> : <span style={{ opacity: 0.6 }}>— hover the image</span>}
        </span>
      )}
    </span>
  );
}

const PEN_COLOR: Record<number, string> = { 0: "#c7c7cc", 1: "#1ab2ff", 2: "#ff453a", 3: "#9aa0aa" };
const PEN_KEY = { 0: "pen.erase", 1: "pen.cornea", 2: "pen.scar", 3: "pen.background" } as const;
const VIEWS: ViewName[] = ["multi", "axial", "coronal", "sagittal", "render"];

// Per-plane through-plane RAS axis for the slice scrollbars. The AXIAL↔CORONAL axes are SWAPPED to match
// the user-facing label swap in nvController's SLICE map: the user's "Axial" view shows niivue's CORONAL
// plane (RAS through-axis 1 = the B-scan / frame axis) and the user's "Coronal" shows niivue's AXIAL plane
// (RAS through-axis 2 = the en-face/depth axis). So the "Axial" scrollbar scrolls B-scans, as expected.
const VIEW_AXIS: { plane: "axial" | "coronal" | "sagittal"; axis: 0 | 1 | 2; key: TKey }[] = [
  { plane: "axial", axis: 1, key: "view.axial" },     // user Axial = niivue coronal (B-scan), RAS axis 1
  { plane: "coronal", axis: 2, key: "view.coronal" }, // user Coronal = niivue axial (en-face), RAS axis 2
  { plane: "sagittal", axis: 0, key: "view.sagittal" },
];

export function AnnotatorCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastVox = useRef<[number, number, number] | null>(null); // previous painted voxel (stroke continuity)
  const strokeTile = useRef<number>(-1);                          // pane the current stroke started on
  const strokeAxis = useRef<number | null>(null);                 // that pane's through-plane axis (paint stays on one slice)
  const [view, setV] = useState<ViewName>("multi");
  const [noWebgl, setNoWebgl] = useState<string | null>(null);
  const [brush, setBrush] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const loaded = useStore((s) => s.loaded);
  const busy = useStore((s) => s.busy);
  const tool = useStore((s) => s.tool);
  const penLabel = useStore((s) => s.penLabel);
  const penSize = useStore((s) => s.penSize);
  const volName = useStore((s) => s.activeVolume?.name);
  const lang = useStore((s) => s.lang);
  const dims = useStore((s) => s.dims);
  const vox = useStore((s) => s.vox);
  const setSliceAxis = useStore((s) => s.setSliceAxis);
  const syncVox = useStore((s) => s.syncVox);
  const wandAt = useStore((s) => s.wandAt);
  const setCursorIntensity = useStore((s) => s.setCursorIntensity);
  const refreshStats = useStore((s) => s.refreshStats);
  const autosaveDraw = useStore((s) => s.autosaveDraw);
  const zoomIn = useStore((s) => s.zoomIn);
  const zoomOut = useStore((s) => s.zoomOut);
  const resetView = useStore((s) => s.resetView);

  useEffect(() => {
    if (canvasRef.current) {
      attach(canvasRef.current);
      const err = webglFailure();
      setNoWebgl(err);
      if (!err) useStore.getState().resumePending(); // #2 reopen last session's volume once attached
    }
  }, []);

  // #5: best-effort final autosave when the window/app is closing (the per-stroke debounced autosave
  // already covers the common case; this catches a close right after the last edit).
  useEffect(() => {
    const flush = () => { void useStore.getState().flushAutosave(); };
    window.addEventListener("beforeunload", flush);
    return () => window.removeEventListener("beforeunload", flush);
  }, []);

  // Defensive repaint once the layout has SETTLED after a volume loads / the view changes. WebKitGTK
  // (the desktop webview) leaves the GL canvas BLACK if its drawing buffer is reallocated by a resize
  // after the first draw — and a single requestAnimationFrame can fire BEFORE the layout has settled
  // (the extra on-load work in #5 made this worse: panes stayed black). So redraw on a few escalating
  // delays AND whenever the canvas container actually RESIZES (a ResizeObserver is the reliable signal
  // that WebKitGTK has reallocated the buffer — re-issuing drawScene then recovers it). Chromium is
  // unaffected; these extra drawScene() calls are cheap and harmless. (#2 / v0.1.21)
  useEffect(() => {
    if (!loaded) return;
    const raf = requestAnimationFrame(() => redraw());
    const timers = [60, 180, 400, 800].map((d) => setTimeout(() => redraw(), d));
    const host = canvasRef.current?.parentElement;
    let ro: ResizeObserver | undefined;
    if (host && typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => requestAnimationFrame(() => redraw()));
      ro.observe(host);
    }
    return () => { cancelAnimationFrame(raf); timers.forEach(clearTimeout); ro?.disconnect(); };
  }, [loaded, view]);

  const painting = loaded && tool === "paint" && view !== "render";
  const wandActive = loaded && tool === "wand" && view !== "render";
  const showBrush = painting && brush;
  const showStrip = loaded && view !== "render" && !!dims;

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
        {loaded && view !== "render" && (
          <span className="flex items-center gap-0.5 flex-none" title={tr(lang, "view.zoomTip")}>
            <Button size="small" onClick={() => zoomOut()} sx={{ minWidth: 24, px: 0, fontSize: 16, lineHeight: 1, color: "var(--c-text-dim)" }}>−</Button>
            <Button size="small" onClick={() => resetView()} sx={{ minWidth: 24, px: 0, fontSize: 13, lineHeight: 1, color: "var(--c-text-dim)" }}>⤢</Button>
            <Button size="small" onClick={() => zoomIn()} sx={{ minWidth: 24, px: 0, fontSize: 16, lineHeight: 1, color: "var(--c-text-dim)" }}>+</Button>
          </span>
        )}
        <span className="flex-1" />
        {loaded && <Readout />}
        {loaded && tool === "paint" && view !== "render" && (
          <span className="flex items-center gap-1.5 flex-none" style={{ fontSize: 11, color: "var(--c-text-dim)" }}>
            <span style={{ width: 9, height: 9, borderRadius: "50%", background: PEN_COLOR[penLabel],
                           display: "inline-block", boxShadow: "0 0 0 1px rgba(0,0,0,0.35)" }} />
            {tr(lang, PEN_KEY[penLabel])} · {penSize}px
          </span>
        )}
        <span className="truncate flex-none" style={{ fontSize: 12, maxWidth: 180, color: loaded ? "var(--c-text)" : "var(--c-text-dim)" }}
          title={volName ?? ""}>{volName ?? tr(lang, "canvas.noVol")}</span>
      </div>

      {/* Canvas + overlays */}
      <div
        className="relative flex-1 min-h-0 min-w-0"
        onMouseDownCapture={() => { if (painting) setStroke(true); }}
        onMouseDown={(e) => {
          if (e.shiftKey || e.ctrlKey) return;                     // shift/ctrl+drag = pan (niivue handles it)
          const r = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - r.left, y = e.clientY - r.top;
          if (painting) {
            strokeTile.current = tileAtScreen(x, y);
            if (strokeTile.current < 0) return;                    // started off any 2-D pane → ignore
            strokeAxis.current = tileThroughAxis(strokeTile.current); // confine paint to this pane's slice
            beginStroke();                                         // snapshot for undo (start of stroke)
            // Clamp to the stroke pane so a brush near (or past) the edge still paints the border
            // pixels instead of vanishing — the round brush's centre clamps to the edge voxel (#1).
            const v = voxAtScreenClamped(x, y, strokeTile.current);
            if (v) { paintBrush(v[0], v[1], v[2], v[0], v[1], v[2], penLabel, strokeAxis.current); lastVox.current = v; }
            restoreCrosshair();
          } else if (wandActive) {
            const t = tileAtScreen(x, y);
            if (t < 0) return;
            const v = voxAtScreen(x, y);
            if (v) wandAt(v[0], v[1], v[2], tileThroughAxis(t));   // seed the wand → live preview (Confirm bakes it)
          } else if (loaded && tool === "navigate") {
            setCrosshairAtScreen(x, y);                            // #2: ONLY Navigate moves the crosshair (shift/ctrl already returned = pan)
          }
        }}
        onMouseUp={() => { if (painting) { lastVox.current = null; restoreCrosshair(); setStroke(false); syncVox(); refreshStats(); flushCompose(); autosaveDraw(); } }}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - r.left, y = e.clientY - r.top;
          const modPan = e.shiftKey || e.ctrlKey;
          const tile = (painting || wandActive) ? tileAtScreen(x, y) : -1;
          // paint a 3-D sphere at the cursor voxel, confined to the stroke's pane; crosshair stays put.
          if (painting && !modPan && (e.buttons & 1)) {
            // Clamp to the STROKE pane: keep painting the edge even when the cursor leaves the image /
            // pane, so a round brush covers the border pixels instead of disappearing (#1). The clamp
            // also maps off-image cursors to the nearest edge voxel (never the crosshair centre).
            const v = strokeTile.current >= 0 ? voxAtScreenClamped(x, y, strokeTile.current) : null;
            if (v) { const p = lastVox.current ?? v; paintBrush(v[0], v[1], v[2], p[0], p[1], p[2], penLabel, strokeAxis.current); lastVox.current = v; }
            else lastVox.current = null;
            restoreCrosshair();
          } else if (painting && !modPan) lockCrosshair();
          else if (loaded && tool === "navigate" && !modPan && (e.buttons & 1)) setCrosshairAtScreen(x, y); // #2: drag scrubs crosshair in Navigate
          // wand: show the intensity under the cursor (helps choose the threshold)
          if (wandActive && tile >= 0) { const vx = voxAtScreen(x, y); if (vx) setCursorIntensity(vx[0], vx[1], vx[2]); }
          // brush cursor (paint only)
          if (painting && !modPan && tile >= 0) {
            const sz = brushScreenSize(x, y);
            setBrush({ x, y, w: sz ? Math.max(4, sz.w) : Math.max(4, penSize), h: sz ? Math.max(4, sz.h) : Math.max(4, penSize) });
          } else if (brush) setBrush(null);
        }}
        onMouseLeave={() => { if (painting) { lastVox.current = null; restoreCrosshair(); setStroke(false); syncVox(); refreshStats(); flushCompose(); autosaveDraw(); } setBrush(null); }}
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" style={{ cursor: painting || wandActive ? "crosshair" : "default" }} />
        {showBrush && (
          <div className="absolute pointer-events-none" style={{
            left: brush!.x, top: brush!.y, transform: "translate(-50%, -50%)",
            width: brush!.w, height: brush!.h, borderRadius: "50%",
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

      {/* Per-view slice scrollbars (#1): one per single view, all three in Multi; synced to scroll-wheel
          via onLocationChange. The bar's height is ALWAYS reserved (even with no volume / in 3D) so that
          loading a volume never RESIZES the niivue canvas — WebKitGTK can render a freshly-loaded volume
          black if its GL drawing buffer is reallocated by a resize right after the first draw. (#2) */}
      <div className="flex items-center gap-5 px-4 flex-none overflow-x-auto"
        style={{ height: 40, borderTop: showStrip ? "1px solid var(--c-border)" : "none",
                 backgroundColor: showStrip ? "var(--c-surface)" : "transparent" }}>
        {showStrip && VIEW_AXIS.filter((a) => view === "multi" || view === a.plane).map((a) => {
          const n = dims![a.axis];
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
    </div>
  );
}
