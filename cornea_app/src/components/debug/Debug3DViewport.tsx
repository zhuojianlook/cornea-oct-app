/* Debug tab — INTERACTIVE 3-D viewport (replaces the pre-baked turntable).

   One large niivue volume-render (SLICE_TYPE.RENDER) driven live on the GPU at the fine registration
   grid, plus a METHOD SWITCHER radio. Switching method (or the 3D Overlap / 3D Disagreement / 3D
   Consensus mode) only swaps the loaded volumes — the camera pose is kept, so hotspots compare by
   flipping methods at a fixed angle. Uses a SEPARATE niivue instance (debugNvController) created lazily
   and destroyed on unmount, so at most one extra WebGL context is alive on the fragile WebKitGTK stack.

   Two shapes of content share this host:
   • PAIRWISE (overlap / disagreement) — fixed vs moving, per-method results, magenta/green or hot.
   • CONSENSUS — ALL replicates of the eye at once, aligned by one method, composited by the backend into
     ONE min/excess RGBA volume (each replicate its own hue, white where they all agree). */

import { useEffect, useRef, useState } from "react";
import { CircularProgress, ToggleButton, ToggleButtonGroup, Tooltip } from "@mui/material";
import * as dbg from "../../niivue/debugNvController";
import {
  ALIGN_METHODS,
  viewUrl,
  type AlignResult,
  type ConsensusResult,
  type MethodId,
  type Mode3d,
} from "../../store/debugStore";

const num = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);
const fmt = (v: number | null | undefined, dp = 4): string =>
  num(v) ? v.toFixed(dp) : "—";
const fmtSigned = (v: number | null | undefined, dp = 1): string =>
  num(v) ? `${v >= 0 ? "+" : ""}${v.toFixed(dp)}` : "—";
const fmtResid = (r: AlignResult): string =>
  num(r.resid_um) ? `${r.resid_um.toFixed(1)} µm${num(r.resid_vox) ? ` (${r.resid_vox.toFixed(1)} vox)` : ""}`
    : num(r.resid_vox) ? `${r.resid_vox.toFixed(1)} vox`
    : "—";

const HOT_SWATCH = "linear-gradient(90deg, #400 0%, #b00 40%, #f30 65%, #fd0 85%, #fff 100%)";

const methodLabel = (m: string): string => ALIGN_METHODS.find((x) => x.id === m)?.label ?? m;
const rgb = (c: [number, number, number]): string => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

interface Props {
  mode: Mode3d;
  // ── pairwise (overlap / disagreement) ──
  results: AlignResult[];
  focus: MethodId;
  setFocus: (m: MethodId) => void;
  fixed3d: string | null;
  isoMm: number | null;
  /** Shared fixed intensity window [lo, hi] from the backend (geometry.window). */
  window: number[] | undefined;
  running: boolean;
  // ── consensus (all replicates of the eye) ──
  consensus?: ConsensusResult | null;
  consensusMethod?: MethodId;
  setConsensusMethod?: (m: MethodId) => void;
  consensusRunning?: boolean;
  consensusError?: string | null;
}

export function Debug3DViewport({
  mode,
  results,
  focus,
  setFocus,
  fixed3d,
  isoMm,
  window: win,
  running,
  consensus,
  consensusMethod,
  setConsensusMethod,
  consensusRunning,
  consensusError,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [glErr, setGlErr] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  const isConsensus = mode === "consensus";

  // Create the debug niivue instance for this viewport's lifetime; destroy (free the WebGL context) on
  // unmount. debugNvController defers the destroy one tick so React StrictMode's mount→unmount→mount is a
  // no-op rather than a teardown. A ResizeObserver keeps the render repainted as the container resizes.
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const inst = dbg.create(cv);
    if (!inst) {
      setGlErr(dbg.webglError() ?? "WebGL2 unavailable");
      setReady(false);
    } else {
      setGlErr(null);
      setReady(true);
    }
    let ro: ResizeObserver | undefined;
    if (wrapRef.current && typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => requestAnimationFrame(() => dbg.redraw()));
      ro.observe(wrapRef.current);
    }
    return () => {
      ro?.disconnect();
      dbg.destroy();
      setReady(false);
    };
  }, []);

  const focused = results.find((r) => r.method === focus) ?? results[0];
  const v3 = focused?.volumes3d ?? null;
  const haveVolumes = !!fixed3d && !!v3;
  const failed = focused ? focused.raised === true || focused.ok === false : false;

  // Shared window as primitives so the load effect doesn't re-fire on every poll (array identity churns).
  const winLo = Array.isArray(win) && win.length === 2 ? Number(win[0]) : null;
  const winHi = Array.isArray(win) && win.length === 2 ? Number(win[1]) : null;
  const movingUrl = v3?.moving ?? null;
  const disagreeUrl = v3?.disagree ?? null;
  const disagreeMax = typeof v3?.disagree_max === "number" ? v3.disagree_max : null;

  const consensusVolUrl = consensus?.volume ?? null;
  const haveConsensus = !!consensusVolUrl;

  // PAIRWISE (re)load whenever the mode / focused method / URLs change. The camera pose persists across the
  // swap (scene state, not per-volume), so flipping methods keeps the angle. Guarded off in consensus mode:
  // consensus has no moving/disagree volume, and firing show() with a pair payload would clear the render.
  useEffect(() => {
    if (isConsensus) return;
    if (!ready || glErr || !fixed3d || !movingUrl || !disagreeUrl) return;
    void dbg.show({
      mode,
      fixedUrl: viewUrl(fixed3d),
      movingUrl: viewUrl(movingUrl),
      disagreeUrl: viewUrl(disagreeUrl),
      disagreeMax,
      window: winLo != null && winHi != null ? [winLo, winHi] : null,
    });
  }, [isConsensus, ready, glErr, mode, fixed3d, movingUrl, disagreeUrl, disagreeMax, winLo, winHi]);

  // CONSENSUS load: the single min/excess RGBA volume (colours baked in). Keyed on the volume url alone, so
  // switching method (new url) reloads while the camera pose is kept — the whole point of the shared instance.
  useEffect(() => {
    if (!isConsensus) return;
    if (!ready || glErr || !consensusVolUrl) return;
    void dbg.loadConsensus(viewUrl(consensusVolUrl));
  }, [isConsensus, ready, glErr, consensusVolUrl]);

  return (
    <div className="flex flex-col min-h-0" style={{ height: "100%", gap: 8 }}>
      {/* ── method switcher ── */}
      <div className="flex items-center gap-3 flex-wrap">
        <span style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5, color: "var(--c-text-dim)" }}>
          {isConsensus ? "Align by" : "Method"}
        </span>
        {isConsensus ? (
          <ToggleButtonGroup
            size="small"
            exclusive
            value={consensusMethod ?? "fixed"}
            onChange={(_, v) => v && setConsensusMethod?.(v as MethodId)}
          >
            {ALIGN_METHODS.map((m) => (
              <ToggleButton key={m.id} value={m.id} sx={{ py: 0.1, px: 1, fontSize: 11, textTransform: "none" }}>
                {m.label}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        ) : results.length > 0 ? (
          <ToggleButtonGroup size="small" exclusive value={focus} onChange={(_, v) => v && setFocus(v as MethodId)}>
            {results.map((r) => (
              <ToggleButton key={r.method} value={r.method} sx={{ py: 0.1, px: 1, fontSize: 11, textTransform: "none" }}>
                {methodLabel(r.method)}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        ) : (
          <span style={{ fontSize: 11, color: "var(--c-text-dim)" }}>…</span>
        )}
        <span style={{ fontSize: 11, color: "var(--c-text-dim)" }}>
          {isConsensus
            ? "Drag to rotate · scroll to zoom · right-drag to pan"
            : "Drag to rotate · scroll to zoom · right-drag to pan · ←/→ flips methods"}
        </span>
        {num(isConsensus ? consensus?.iso_mm ?? null : isoMm) && (
          <Tooltip title="The interactive volume is resampled to this isotropic GPU grid — much finer than the old turntable's pre-baked frames." arrow>
            <span style={{ fontSize: 10, color: "var(--c-text-dim)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "0 5px", cursor: "help" }}>
              {(isConsensus ? consensus!.iso_mm : isoMm!).toFixed(3)} mm grid
            </span>
          </Tooltip>
        )}
      </div>

      {/* ── viewport + readout ── */}
      <div style={{ flex: 1, minHeight: 0, display: "flex", gap: 8 }}>
        <div
          ref={wrapRef}
          style={{ flex: 1, minWidth: 0, position: "relative", background: "#000", borderRadius: 8, overflow: "hidden", border: "1px solid var(--c-border)" }}
        >
          <canvas
            ref={canvasRef}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", display: "block", touchAction: "none" }}
          />
          {glErr ? (
            <div style={overlayStyle}>
              <span style={{ maxWidth: 360, textAlign: "center" }}>{glErr}</span>
            </div>
          ) : isConsensus ? (
            consensusError ? (
              <div style={overlayStyle}>
                <span style={{ maxWidth: 360, textAlign: "center", color: "var(--c-red)" }}>{consensusError}</span>
              </div>
            ) : !haveConsensus ? (
              <div style={overlayStyle}>
                <span className="flex items-center gap-2" style={{ color: "var(--c-text-dim)" }}>
                  <CircularProgress size={16} color="inherit" />
                  {consensusRunning ? "rendering consensus…" : "preparing consensus…"}
                </span>
              </div>
            ) : null
          ) : !haveVolumes ? (
            <div style={overlayStyle}>
              {failed ? (
                <span style={{ maxWidth: 360, textAlign: "center", color: "var(--c-text-dim)" }}>
                  {focused?.error ? focused.error : "This method fell back to identity — no 3-D volume."}
                </span>
              ) : (
                <span className="flex items-center gap-2" style={{ color: "var(--c-text-dim)" }}>
                  <CircularProgress size={16} color="inherit" />
                  {running ? "rendering 3-D…" : "preparing 3-D…"}
                </span>
              )}
            </div>
          ) : null}

          {/* Consensus keeps its (possibly stale) composite on screen while a new method re-renders, so mark
              the in-flight state without blanking the viewport. */}
          {isConsensus && haveConsensus && consensusRunning && (
            <div style={{ position: "absolute", top: 8, right: 8, display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--c-text-dim)", background: "rgba(0,0,0,0.45)", borderRadius: 6, padding: "2px 8px", pointerEvents: "none" }}>
              <CircularProgress size={11} color="inherit" /> re-rendering…
            </div>
          )}
        </div>

        {/* readout + legend, beside the viewport */}
        <div style={{ width: 208, flex: "none", display: "flex", flexDirection: "column", gap: 8, fontSize: 11, color: "var(--c-text-dim)" }}>
          {isConsensus ? (
            <>
              {consensus && (
                <div style={{ border: "1px solid var(--c-border)", borderRadius: 8, padding: "8px 10px", background: "var(--c-surface)" }}>
                  <div style={{ fontSize: 12, color: "var(--c-text)", marginBottom: 4 }}>
                    {methodLabel(consensusMethod ?? "fixed")}
                  </div>
                  <div style={{ fontSize: 8, letterSpacing: 0.5, opacity: 0.7 }}>CONSENSUS</div>
                  <div style={{ color: "var(--c-text)", fontSize: 13 }}>
                    <b>{consensus.replicates.length}</b> replicates
                  </div>
                  <div>aligned to {consensus.replicates.find((r) => r.is_ref)?.label ?? "the reference"}</div>
                </div>
              )}

              <div style={{ border: "1px solid var(--c-border)", borderRadius: 8, padding: "8px 10px", background: "var(--c-surface)", lineHeight: 1.7 }}>
                <div style={{ fontSize: 8, letterSpacing: 0.5, opacity: 0.7, marginBottom: 2 }}>LEGEND</div>
                {(consensus?.replicates ?? []).map((r) => (
                  <div key={r.case} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ width: 9, height: 9, background: rgb(r.color), borderRadius: 2, flex: "none" }} />
                    {r.label}
                    {r.is_ref && (
                      <span style={{ fontSize: 8, color: "var(--c-text-dim)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "0 4px" }}>
                        REF
                      </span>
                    )}
                  </div>
                ))}
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
                  <span style={{ width: 9, height: 9, background: rgb(consensus?.agree_color ?? [255, 255, 255]), borderRadius: 2, flex: "none", border: "1px solid var(--c-border)" }} />
                  agree (all replicates)
                </div>
                <div style={{ opacity: 0.85, marginTop: 4 }}>
                  White = every replicate agrees; a replicate's own colour = it diverges there (misalignment or a
                  unique feature).
                </div>
              </div>
            </>
          ) : (
            <>
              {focused && (
                <div style={{ border: "1px solid var(--c-border)", borderRadius: 8, padding: "8px 10px", background: "var(--c-surface)" }}>
                  <div style={{ fontSize: 12, color: "var(--c-text)", marginBottom: 4 }}>{methodLabel(focused.method)}</div>
                  <div style={{ fontSize: 8, letterSpacing: 0.5, opacity: 0.7 }}>GEOMETRY</div>
                  <div style={{ color: "var(--c-text)", fontSize: 13 }}>
                    residual <b>{fmtResid(focused)}</b>
                  </div>
                  <div>tilt {num(focused.tilt_vox) ? `${fmtSigned(focused.tilt_vox)} vox` : "—"}</div>
                  <div style={{ fontSize: 8, letterSpacing: 0.5, opacity: 0.7, marginTop: 6 }}>PROXY</div>
                  <div>primary {fmt(focused.primary)}</div>
                  <div>rot {fmt(focused.rot_deg, 3)}°</div>
                </div>
              )}

              <div style={{ border: "1px solid var(--c-border)", borderRadius: 8, padding: "8px 10px", background: "var(--c-surface)", lineHeight: 1.7 }}>
                {mode === "disagreement" ? (
                  <>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ width: 30, height: 9, background: HOT_SWATCH, borderRadius: 2, flex: "none" }} />
                      hot = replicates differ
                    </div>
                    <div style={{ opacity: 0.85, marginTop: 4 }}>
                      Transparent where they agree. A residual tilt reads as an edge band; a per-replicate scar
                      difference as a localized blob.
                    </div>
                  </>
                ) : (
                  <>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                      <span style={{ width: 9, height: 9, background: "#e879f9", borderRadius: 2 }} /> fixed
                      <span style={{ width: 9, height: 9, background: "#4ade80", borderRadius: 2 }} /> moving
                      <span style={{ width: 9, height: 9, background: "#d4d4d8", borderRadius: 2 }} /> agree
                    </div>
                    <div style={{ opacity: 0.85, marginTop: 4 }}>
                      White/grey where the two replicates agree; magenta/green where they don't.
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          <div style={{ opacity: 0.75, fontSize: 10, lineHeight: 1.6 }}>
            {isConsensus
              ? "Live GPU volume render — all replicates at once. The consensus is the white core; the coloured fringe is where a replicate parts from it."
              : "Live GPU volume render — no winner is declared here; judge the geometry on the residual and the pixels."}
          </div>
        </div>
      </div>
    </div>
  );
}

const overlayStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 24,
  fontSize: 11,
  pointerEvents: "none",
  background: "rgba(0,0,0,0.25)",
};
