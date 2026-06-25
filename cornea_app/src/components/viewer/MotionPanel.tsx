/* Eye-motion analysis tab (stage 4). The 3-D scan's slow (frame) axis is a TIME axis (~136 Hz on the
   Avanti), so the detected corneal surface — once the smooth corneal shape is removed — is the patient's
   eye/head motion during the ~0.7 s scan. Shows: the motion(t) trace, its power spectrum (with the
   unresolved/Nyquist caveats made visible), a labelled dominant-frequency table, candidate saccade spikes,
   and a dominant motion direction. Plots are hand-rolled SVG (no charting dependency). */
import { useEffect, useState } from "react";
import { Button, CircularProgress } from "@mui/material";
import { useWorkflowStore } from "../../store/workflowStore";
import { useCaseStore } from "../../store/caseStore";

const PAD = { l: 44, r: 10, t: 10, b: 28 };
// Nominal A-scans per B-scan for the canonical 3-D Cornea volume (513×640×101). Used ONLY for the
// pre-analysis advisory frame-rate estimate; once analysed, the panel uses the result's exact per-volume
// frame_rate_hz / nyquist_hz instead.
const NOMINAL_APF = 513;

function niceMax(v: number): number { return v <= 0 ? 1 : v; }

export function MotionPanel() {
  const r = useWorkflowStore((s) => s.motionResult);
  const busy = useWorkflowStore((s) => s.motionBusy);
  const ascanRate = useWorkflowStore((s) => s.ascanRateHz);
  const sinc = useWorkflowStore((s) => s.motionSinc);
  const set = useWorkflowStore((s) => s.set);
  const run = useWorkflowStore((s) => s.runMotionAnalysis);
  const hasVolume = useCaseStore((s) => Boolean(s.volumeUrl));

  // A-scan-rate input: edit as raw text, commit-clamp on blur/Enter (clamping per-keystroke mangles typed
  // values — e.g. the first digit of any number is <1000 and would snap to 1000). Re-sync if the store value
  // changes externally (e.g. seeded from a case's persisted calibration on case switch).
  const [rateText, setRateText] = useState(String(ascanRate));
  useEffect(() => { setRateText(String(ascanRate)); }, [ascanRate]);
  const commitRate = () => {
    const n = Number(rateText);
    if (Number.isFinite(n) && n > 0) {
      const clamped = Math.min(400000, Math.max(1000, n));
      set("ascanRateHz", clamped);
      setRateText(String(clamped));
    } else {
      setRateText(String(ascanRate));   // revert blank/invalid entry
    }
  };

  const W = 560, H = 200;
  const ax = { x0: PAD.l, x1: W - PAD.r, y0: PAD.t, y1: H - PAD.b };
  const axW = ax.x1 - ax.x0, axH = ax.y1 - ax.y0;

  // motion(t): µm vs ms, centred on zero
  const traceSvg = (() => {
    if (!r) return null;
    const A = niceMax(Math.max(...r.motion_um.map((v) => Math.abs(v))) * 1.1);
    const T = niceMax(r.time_ms[r.time_ms.length - 1] || 1);
    const X = (t: number) => ax.x0 + (t / T) * axW;
    const Y = (um: number) => ax.y0 + axH / 2 - (um / A) * (axH / 2);
    const pts = r.motion_um.map((um, i) => `${X(r.time_ms[i]).toFixed(1)},${Y(um).toFixed(1)}`).join(" ");
    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", maxHeight: 220 }}>
        <line x1={ax.x0} y1={Y(0)} x2={ax.x1} y2={Y(0)} stroke="var(--c-border)" strokeWidth={1} />
        <line x1={ax.x0} y1={ax.y0} x2={ax.x0} y2={ax.y1} stroke="var(--c-border)" strokeWidth={1} />
        {r.spikes.map((s, i) => (
          <line key={`sp${i}`} x1={X(s.t_ms)} y1={ax.y0} x2={X(s.t_ms)} y2={ax.y1} stroke="#f5a623" strokeWidth={1} strokeDasharray="2 2" opacity={0.8} />
        ))}
        <polyline fill="none" stroke="#c0392b" strokeWidth={1.6} vectorEffect="non-scaling-stroke" points={pts} />
        <text x={ax.x0 - 4} y={Y(A)} textAnchor="end" fontSize={9} fill="var(--c-text-dim)">{A.toFixed(0)}</text>
        <text x={ax.x0 - 4} y={Y(-A) + 6} textAnchor="end" fontSize={9} fill="var(--c-text-dim)">-{A.toFixed(0)}</text>
        <text x={ax.x0 - 4} y={Y(0) + 3} textAnchor="end" fontSize={9} fill="var(--c-text-dim)">0</text>
        <text x={(ax.x0 + ax.x1) / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="var(--c-text-dim)">time (ms) → {T.toFixed(0)}</text>
        <text x={6} y={ax.y0 + 8} fontSize={9} fill="var(--c-text-dim)">µm</text>
      </svg>
    );
  })();

  // power spectrum: norm power vs Hz, with unresolved (<1.5·df) shaded grey
  const specSvg = (() => {
    if (!r) return null;
    const fmax = niceMax(r.nyquist_hz);
    const X = (hz: number) => ax.x0 + (hz / fmax) * axW;
    const Y = (p: number) => ax.y1 - p * axH;
    const pts = r.freqs_hz.map((hz, i) => `${X(hz).toFixed(1)},${Y(r.power[i]).toFixed(1)}`).join(" ");
    const unresolvedX = X(1.5 * r.df_hz);
    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", maxHeight: 220 }}>
        <rect x={ax.x0} y={ax.y0} width={Math.max(0, unresolvedX - ax.x0)} height={axH} fill="var(--c-border)" opacity={0.35} />
        <line x1={ax.x0} y1={ax.y1} x2={ax.x1} y2={ax.y1} stroke="var(--c-border)" strokeWidth={1} />
        <line x1={ax.x0} y1={ax.y0} x2={ax.x0} y2={ax.y1} stroke="var(--c-border)" strokeWidth={1} />
        <polyline fill="none" stroke="#2c3e50" strokeWidth={1.6} vectorEffect="non-scaling-stroke" points={pts} />
        {r.peaks.filter((p) => p.resolved).slice(0, 4).map((p, i) => {
          let bi = 0, bd = Infinity;                              // nearest spectrum bin to the peak freq
          for (let j = 0; j < r.freqs_hz.length; j++) { const dd = Math.abs(r.freqs_hz[j] - p.hz); if (dd < bd) { bd = dd; bi = j; } }
          const py = Y(r.power[bi] ?? 0);
          return (
            <g key={`pk${i}`}>
              <circle cx={X(p.hz)} cy={py} r={3} fill="#c0392b" />
              <text x={X(p.hz)} y={ax.y0 + 9 + i * 11} textAnchor="middle" fontSize={9} fill="#c0392b">{p.hz}Hz</text>
            </g>
          );
        })}
        <text x={ax.x0 + 2} y={ax.y1 + 12} fontSize={8} fill="var(--c-text-dim)">&lt;{(1.5 * r.df_hz).toFixed(1)}Hz unresolved</text>
        <text x={ax.x1} y={ax.y1 + 12} textAnchor="end" fontSize={9} fill="var(--c-text-dim)">{fmax.toFixed(0)} Hz (Nyquist)</text>
        <text x={6} y={ax.y0 + 8} fontSize={9} fill="var(--c-text-dim)">power</text>
      </svg>
    );
  })();

  // direction indicator: arrow tilted from the surface normal (vertical) by tilt_from_normal_deg
  const dirSvg = (() => {
    if (!r) return null;
    const d = r.direction;
    const sgn = d.lateral_azimuth.startsWith("nasal") ? -1 : 1;
    const ang = (sgn * d.tilt_from_normal_deg) * Math.PI / 180;   // 0 = along normal (up)
    const cx = 60, cy = 70, len = 46;
    const tipX = cx + len * Math.sin(ang), tipY = cy - len * Math.cos(ang);
    return (
      <svg viewBox="0 0 120 120" style={{ width: 120, height: 120 }}>
        <circle cx={cx} cy={cy} r={48} fill="none" stroke="var(--c-border)" />
        <line x1={cx} y1={cy} x2={cx} y2={cy - 48} stroke="var(--c-border)" strokeDasharray="3 3" />
        <text x={cx} y={cy - 52} textAnchor="middle" fontSize={8} fill="var(--c-text-dim)">normal</text>
        <line x1={cx} y1={cy} x2={tipX} y2={tipY} stroke="#c0392b" strokeWidth={2.5} />
        <circle cx={tipX} cy={tipY} r={3.5} fill="#c0392b" />
        <text x={cx} y={cy + 60} textAnchor="middle" fontSize={9} fill="var(--c-text-dim)">{d.tilt_from_normal_deg}° from normal</text>
      </svg>
    );
  })();

  const td = { padding: "2px 6px", borderBottom: "1px solid var(--c-border)", fontSize: 11 } as const;

  return (
    <div className="flex flex-col h-full min-h-0" style={{ backgroundColor: "var(--c-bg)", color: "var(--c-text)" }}>
      {/* header / controls */}
      <div className="flex items-center gap-3 px-4 border-b flex-wrap" style={{ minHeight: 44, borderColor: "var(--c-border)" }}>
        <span style={{ fontWeight: 600 }}>Eye motion</span>
        <label className="flex items-center gap-1 text-xs" style={{ color: "var(--c-text-dim)" }}
          title="A-scan (line) rate of the device → frame rate → the Hz axis. Avanti spec ≈ 70000. The .OCT carries no timing, so all frequencies scale with this.">
          A-scan rate (Hz)
          <input type="number" min={1000} max={400000} step={1000} value={rateText}
            onChange={(e) => setRateText(e.target.value)}
            onBlur={commitRate}
            onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
            style={{ width: 80, fontSize: 12, padding: "2px 4px", background: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-border)", borderRadius: 4 }} />
        </label>
        <label className="flex items-center gap-1 text-xs" style={{ color: "var(--c-text-dim)" }}
          title="Divide out the intra-frame motion-blur boxcar (each B-scan integrates over ~7 ms as the fast axis sweeps).">
          <input type="checkbox" checked={sinc} onChange={(e) => set("motionSinc", e.target.checked)} /> blur-correct
        </label>
        <Button size="small" variant="contained" disableElevation disabled={busy || !hasVolume} onClick={() => run()}
          startIcon={busy ? <CircularProgress size={12} color="inherit" /> : undefined}
          sx={{ textTransform: "none" }}>{busy ? "Analysing…" : "Analyze motion"}</Button>
        {r && (
          <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
            {r.frame_rate_hz} Hz · {r.total_s}s · Nyquist {r.nyquist_hz} Hz · res {r.df_hz} Hz
            {r.snr != null && <> · <b style={{ color: r.snr >= 3 ? "var(--c-green)" : "var(--c-red)" }}>SNR {r.snr}×</b></>}
          </span>
        )}
      </div>

      {/* body */}
      <div className="flex-1 min-h-0 overflow-auto p-4">
        {!r ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center" style={{ color: "var(--c-text-dim)" }}>
            <span style={{ maxWidth: 540, fontSize: 13 }}>
              The 3-D scan acquires B-scans over time (the slow axis ≈ {Math.round((ascanRate || 70000) / NOMINAL_APF)} Hz nominal),
              so the detected corneal surface is a depth-vs-time trace. After removing the corneal shape, the residual is the
              patient's eye/head motion during the ~0.7 s scan — its spectrum, dominant frequencies, and direction.
            </span>
            <span style={{ fontSize: 12 }}>Resolvable band ≈ 1.5–{Math.round((ascanRate || 70000) / NOMINAL_APF / 2)} Hz · slower drift/pulse and tremor &gt;Nyquist are not recoverable in a 0.7 s scan.</span>
            <Button variant="contained" disableElevation disabled={busy || !hasVolume} onClick={() => run()} sx={{ textTransform: "none" }}>
              {busy ? "Analysing…" : "Analyze motion"}
            </Button>
            {!hasVolume && <span style={{ fontSize: 11 }}>Load + preprocess a scan first.</span>}
          </div>
        ) : (
          <div className="flex flex-wrap gap-4">
            <div style={{ flex: "1 1 480px", minWidth: 360 }}>
              <div className="text-xs mb-1" style={{ color: "var(--c-text-dim)" }}>Axial eye motion vs time (orange = candidate saccade/microsaccade)</div>
              {traceSvg}
              <div className="text-xs mb-1 mt-3" style={{ color: "var(--c-text-dim)" }}>Motion power spectrum</div>
              {specSvg}
            </div>
            <div style={{ flex: "1 1 280px", minWidth: 260 }}>
              <div className="text-xs mb-1" style={{ color: "var(--c-text-dim)" }}>Dominant frequencies</div>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead><tr style={{ textAlign: "left", color: "var(--c-text-dim)", fontSize: 10 }}>
                  <th style={td}>Hz</th><th style={td}>period</th><th style={td}>power</th><th style={td}>band</th><th style={td}>resolved</th>
                </tr></thead>
                <tbody>
                  {r.peaks.slice(0, 5).map((p, i) => (
                    <tr key={i}>
                      <td style={td}><b>{p.hz}</b></td>
                      <td style={td}>{p.period_ms != null ? `${p.period_ms} ms` : "—"}</td>
                      <td style={td}>{Math.round(p.power_frac * 100)}%</td>
                      <td style={td}>{p.label}</td>
                      <td style={{ ...td, color: p.resolved ? "var(--c-green)" : "var(--c-red)" }}>{p.resolved ? "yes" : "no"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="text-xs mb-1 mt-4" style={{ color: "var(--c-text-dim)" }}>Motion direction (relative to cornea)</div>
              <div className="flex items-center gap-2">
                {dirSvg}
                <div className="text-xs" style={{ color: "var(--c-text-dim)", lineHeight: 1.6 }}>
                  <div>axial: <b style={{ color: "var(--c-text)" }}>{r.direction.axial_um_rms} µm</b> RMS</div>
                  {r.direction.inplane_lateral_um_rms != null ? (
                    <>
                      <div>in-plane (lateral): <b style={{ color: "var(--c-text)" }}>{r.direction.inplane_lateral_um_rms} µm</b></div>
                      <div>tilt: <b style={{ color: "var(--c-text)" }}>{r.direction.tilt_from_normal_deg}°</b> toward {r.direction.lateral_azimuth}</div>
                    </>
                  ) : (
                    <div>in-plane (lateral): <b style={{ color: "var(--c-text)" }}>n/a</b> — surface too flat to resolve lateral slip</div>
                  )}
                  <div>direction coherence: <b style={{ color: r.direction.coherence >= 0.5 ? "var(--c-green)" : "var(--c-text)" }}>{r.direction.coherence}</b></div>
                </div>
              </div>
              <div className="text-[10px] mt-2" style={{ color: "var(--c-text-dim)" }}>
                Only the lateral in-plane axis is recoverable from one 3-D scan; amplitude (µm) is shape-model dependent (±~2×) — the
                trace shape, dominant frequencies and direction are the robust outputs.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
