/* Pairwise OVERLAP viewer (cornea-align + per-scan-vs-consensus "Both").
   Renders a single 3-label overlap map served by /api/case/{id}/overlap/{a}/{b}.nii.gz?label=…:
     1 = A only (blue) · 2 = B only (amber) · 3 = both (red) · 0 = neither (transparent).
   There is NO grayscale background — only the cornea (or scar) of the two registered scans, so where the
   red overlaps the corneas are aligned and a coloured ghost off the red is residual misalignment. Used for
   "Volume align" (label=cornea, A=reference) and "Both" (label=scar, A=voted consensus). Owns its Niivue. */

import { useEffect, useRef, useState } from "react";
import { Niivue, SLICE_TYPE } from "@niivue/niivue";
import { ToggleButton, ToggleButtonGroup, Slider } from "@mui/material";
import { resourceUrl } from "../../api/client";

const VIEWS = {
  render: SLICE_TYPE.RENDER,
  multi: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.AXIAL,
  sagittal: SLICE_TYPE.SAGITTAL,
} as const;
type ViewKey = keyof typeof VIEWS;
const short = (cid: string) => cid.split("_").pop() || cid;

// value 0 → transparent; 1 = A (blue); 2 = B (amber); 3 = both (red). With cal_min 0 / cal_max 3 the values
// map to LUT indices 0 / 85 / 170 / 255 = the four control points below.
const PAIR_CMAP = { R: [0, 60, 245, 235], G: [0, 165, 200, 40], B: [0, 255, 45, 40], A: [0, 215, 215, 255], I: [0, 85, 170, 255] };
const A_RGB = "rgb(60,165,255)", B_RGB = "rgb(245,200,45)", BOTH_RGB = "rgb(235,40,40)";

export function OverlapPairViewer({
  caseId, members, refCid, label, aSource, aLabel, bLabel,
}: {
  caseId: string; members: string[]; refCid?: string;
  label: "cornea" | "scar"; aSource: "ref" | "consensus"; aLabel: string; bLabel: string;
}) {
  const ref = refCid && members.includes(refCid) ? refCid : members[0];
  const a = aSource === "consensus" ? "consensus" : ref;
  const bOptions = aSource === "consensus" ? members : members.filter((m) => m !== ref);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [b, setB] = useState(bOptions[0] ?? "");
  const [view, setView] = useState<ViewKey>("render");
  const [opacity, setOpacity] = useState(0.9);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const ovUrl = (bb: string) => resourceUrl(`/api/case/${caseId}/overlap/${a}/${bb}.nii.gz?label=${label}&t=${Date.now()}`);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!canvas.getContext("webgl2")) { setError("Needs a WebGL2 context — open in Chrome/Firefox."); setLoading(false); return; }
    let cancelled = false;
    const nv = new Niivue({ backColor: [0.11, 0.11, 0.12, 1], show3Dcrosshair: true, isColorbar: false, dragAndDropEnabled: false });
    try {
      nv.attachToCanvas(canvas);
      try { nv.addColormap("pair3", PAIR_CMAP); } catch { /* older niivue → fall back to "warm" */ }
      nvRef.current = nv;
    } catch (e) {
      setError(`Niivue failed to initialise: ${e instanceof Error ? e.message : String(e)}`);
      setLoading(false);
      return;
    }
    (async () => {
      const cmap = (nv.colormaps?.() ?? []).includes("pair3") ? "pair3" : "warm";
      if (b) await nv.loadVolumes([{ url: ovUrl(b), colormap: cmap, opacity, cal_min: 0, cal_max: 3 }]);
      if (cancelled) return;
      nv.setSliceType(VIEWS.render);
      nv.updateGLVolume();
    })()
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; nvRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId]);

  // swap the compared scan. Skip the initial run (volumes still empty) — the mount effect does the first
  // load; otherwise both fire on mount and race to append two copies (breaking the single-volume opacity).
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !b || nv.volumes.length === 0) return;
    let cancelled = false;
    (async () => {
      const cmap = (nv.colormaps?.() ?? []).includes("pair3") ? "pair3" : "warm";
      await nv.loadVolumes([{ url: ovUrl(b), colormap: cmap, opacity, cal_min: 0, cal_max: 3 }]);
      if (!cancelled) { nv.setSliceType(VIEWS[view]); nv.updateGLVolume(); }
    })().catch((e) => !cancelled && setError(String(e)));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b]);

  useEffect(() => { nvRef.current?.setSliceType(VIEWS[view]); }, [view]);
  useEffect(() => { const nv = nvRef.current; if (nv && nv.volumes.length) { nv.setOpacity(0, opacity); nv.drawScene(); } }, [opacity]);

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div className="flex items-center gap-3 px-3 border-b flex-wrap" style={{ minHeight: 40, borderColor: "var(--c-border)" }}>
        <ToggleButtonGroup size="small" exclusive value={view} onChange={(_, v) => v && setView(v)}>
          <ToggleButton value="render">3D</ToggleButton>
          <ToggleButton value="multi">Multi</ToggleButton>
          <ToggleButton value="axial">Axial</ToggleButton>
          <ToggleButton value="sagittal">Sagittal</ToggleButton>
        </ToggleButtonGroup>
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>{aLabel} vs</span>
        <ToggleButtonGroup size="small" exclusive value={b} onChange={(_, v) => v && setB(v)}>
          {bOptions.map((m) => (
            <ToggleButton key={m} value={m} sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>{short(m)}</ToggleButton>
          ))}
        </ToggleButtonGroup>
        <div className="flex items-center gap-2" style={{ width: 130 }}>
          <span className="text-[11px] whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>opacity</span>
          <Slider size="small" min={0} max={1} step={0.05} value={opacity} onChange={(_, v) => setOpacity(v as number)} />
        </div>
        <div className="flex items-center gap-3" style={{ marginLeft: "auto" }}>
          <span className="flex items-center gap-1 text-[11px]" style={{ color: "var(--c-text-dim)" }}><span style={{ width: 10, height: 10, borderRadius: 2, background: A_RGB }} />{aLabel}</span>
          <span className="flex items-center gap-1 text-[11px]" style={{ color: "var(--c-text-dim)" }}><span style={{ width: 10, height: 10, borderRadius: 2, background: B_RGB }} />{bLabel}</span>
          <span className="flex items-center gap-1 text-[11px]" style={{ color: "var(--c-text-dim)" }}><span style={{ width: 10, height: 10, borderRadius: 2, background: BOTH_RGB }} />overlap</span>
        </div>
      </div>
      <div className="px-3 py-1 border-b text-[11px]" style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}>
        {label === "cornea"
          ? "Cornea of each registered scan in the reference frame (no background). Overlap (red) = aligned; a coloured ghost off the red = residual misalignment (rigid translation/rotation only — no warping)."
          : "Per-scan scar vs the voted consensus scar. Overlap (red) = agreement; coloured-only = where they differ."}
      </div>
      <div className="relative flex-1 min-h-0 min-w-0">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
        {(loading || error) && (
          <div className="absolute inset-0 flex items-center justify-center p-6 text-center pointer-events-none" style={{ color: "var(--c-text-dim)" }}>
            {error ? <span style={{ fontSize: 13, color: "var(--c-red)", maxWidth: 460 }}>{error}</span> : <span style={{ fontSize: 13 }}>Loading…</span>}
          </div>
        )}
      </div>
    </div>
  );
}
