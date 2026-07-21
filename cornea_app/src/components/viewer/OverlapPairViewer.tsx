/* Volume-align viewer — pairwise cornea+scar alignment of two replicates in the reference frame.
   The A/B overlap is split into THREE region maps served by /api/case/{id}/align-region/{a}/{b}/{region}:
   region "a" (in A only), "b" (in B only), "both" (the intersection). Each is loaded as its own niivue
   volume so it gets its OWN colour and its OWN opacity slider — and within each, the cornea is faint and
   the scar opaque (the bright scar "constellation" is what alignment is really about). So: red (both) = the
   two replicates align; a coloured ghost off the red = a residual rigid (xyz + rotation) offset. There is
   NO grayscale background. Owns its own niivue instance. */

import { useEffect, useRef, useState } from "react";
import { Niivue, SLICE_TYPE } from "@niivue/niivue";
import { ToggleButton, ToggleButtonGroup, Slider } from "@mui/material";
import { resourceUrl } from "../../api/client";
import { releaseVolumes, destroyNiivue, withVolumeNames } from "../../niivue/nvRelease";

const VIEWS = {
  render: SLICE_TYPE.RENDER,
  multi: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.AXIAL,
  sagittal: SLICE_TYPE.SAGITTAL,
} as const;
type ViewKey = keyof typeof VIEWS;

// Per-region colormaps over value 0/1/2 (cal_min 0 / cal_max 2 → indices 0/128/255). Value 1 = cornea
// (faint, A≈95), value 2 = scar (opaque, A≈235). A = replicate-A colour, B = replicate-B colour, both = red.
const RG_A = { R: [0, 70, 70], G: [0, 150, 180], B: [0, 255, 255], A: [0, 95, 235], I: [0, 128, 255] };
const RG_B = { R: [0, 245, 245], G: [0, 185, 205], B: [0, 40, 40], A: [0, 95, 235], I: [0, 128, 255] };
const RG_BOTH = { R: [0, 235, 235], G: [0, 70, 45], B: [0, 70, 45], A: [0, 95, 235], I: [0, 128, 255] };
const A_RGB = "rgb(70,170,255)", B_RGB = "rgb(245,195,45)", BOTH_RGB = "rgb(235,55,45)";

export function OverlapPairViewer({ caseId, members, refCid }: { caseId: string; members: string[]; refCid?: string }) {
  const aRef = refCid && members.includes(refCid) ? refCid : members[0];
  const bOptions = members.filter((m) => m !== aRef);
  const repName = (cid: string) => `replicate ${members.indexOf(cid) + 1}`;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [b, setB] = useState(bOptions[0] ?? "");
  const [view, setView] = useState<ViewKey>("render");
  const [opA, setOpA] = useState(0.85);
  const [opB, setOpB] = useState(0.85);
  const [opBoth, setOpBoth] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const regionUrl = (region: string, bb: string) =>
    resourceUrl(`/api/case/${caseId}/align-region/${aRef}/${bb}/${region}.nii.gz?t=${Date.now()}`);

  const loadingRef = useRef(false);   // serialize loads so a rapid replicate switch can't race two loadVolumes
  const loadPair = async (nv: Niivue, bb: string) => {
    loadingRef.current = true;
    try {
      // loadVolumes() replaces its list by assignment and so never runs removeVolume(), the only path that
      // drops the NVImage from niivue's strong mediaUrlMap — so each replicate switch stranded all THREE
      // decoded region volumes for the life of this viewer (see src/niivue/nvRelease.ts). withVolumeNames
      // also suppresses niivue's discarded full-size probe fetch, which the ?t= URLs would trigger per volume.
      releaseVolumes(nv);
      await nv.loadVolumes(withVolumeNames([
        { url: regionUrl("a", bb), colormap: "rgA", opacity: opA, cal_min: 0, cal_max: 2 },
        { url: regionUrl("b", bb), colormap: "rgB", opacity: opB, cal_min: 0, cal_max: 2 },
        { url: regionUrl("both", bb), colormap: "rgBoth", opacity: opBoth, cal_min: 0, cal_max: 2 },
      ]));
    } finally {
      loadingRef.current = false;
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!canvas.getContext("webgl2")) { setError("Needs a WebGL2 context — open in Chrome/Firefox."); setLoading(false); return; }
    let cancelled = false;
    // isNearestInterpolation: crisp voxels — matches the main grayscale viewer (nvController); also loads DISCRETE
    // region maps (0/1/2), so linear sampling would produce fractional labels / haloed borders anyway.
    const nv = new Niivue({ backColor: [0.11, 0.11, 0.12, 1], show3Dcrosshair: true, isColorbar: false, dragAndDropEnabled: false, isNearestInterpolation: true });
    try {
      nv.attachToCanvas(canvas);
      try { nv.addColormap("rgA", RG_A); nv.addColormap("rgB", RG_B); nv.addColormap("rgBoth", RG_BOTH); } catch { /* older niivue */ }
      nvRef.current = nv;
    } catch (e) {
      setError(`Niivue failed to initialise: ${e instanceof Error ? e.message : String(e)}`);
      setLoading(false);
      return;
    }
    (async () => {
      if (b) await loadPair(nv, b);
      if (cancelled) return;
      // No updateGLVolume(): loadPair's loadVolumes already rebuilt the layer stack (addVolume → setVolume
      // → updateGLVolume per volume) and setSliceType ends in drawScene(). An extra call would re-allocate
      // the layer-1 3-D texture and orphan the live one (see nvRelease).
      nv.setSliceType(VIEWS.render);
    })()
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
      // Free the decoded volumes, niivue's observers (without which this instance stays reachable for the
      // session) and the WebGL context, so repeated mode switches don't exhaust the browser's context budget.
      destroyNiivue(nv);
      nvRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId]);

  // swap the compared replicate
  useEffect(() => {
    const nv = nvRef.current;
    // skip the initial run (mount effect does the first load) and any time a load is already in flight.
    if (!nv || !b || nv.volumes.length === 0 || loadingRef.current) return;
    let cancelled = false;
    (async () => {
      await loadPair(nv, b);
      // setSliceType redraws; the extra updateGLVolume() this used to make orphaned a layer-1 texture on
      // every replicate swap (see nvRelease).
      if (!cancelled) nv.setSliceType(VIEWS[view]);
    })().catch((e) => !cancelled && setError(String(e)));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b]);

  useEffect(() => { nvRef.current?.setSliceType(VIEWS[view]); }, [view]);
  useEffect(() => { const nv = nvRef.current; if (nv && nv.volumes.length >= 1) { nv.setOpacity(0, opA); nv.drawScene(); } }, [opA]);
  useEffect(() => { const nv = nvRef.current; if (nv && nv.volumes.length >= 2) { nv.setOpacity(1, opB); nv.drawScene(); } }, [opB]);
  useEffect(() => { const nv = nvRef.current; if (nv && nv.volumes.length >= 3) { nv.setOpacity(2, opBoth); nv.drawScene(); } }, [opBoth]);

  const opSlider = (label: string, rgb: string, val: number, set: (v: number) => void) => (
    <div className="flex items-center gap-2" style={{ width: 168 }}>
      <span className="flex items-center gap-1 text-[11px] whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>
        <span style={{ width: 10, height: 10, borderRadius: 2, background: rgb, flex: "none" }} />{label}
      </span>
      <Slider size="small" min={0} max={1} step={0.05} value={val} onChange={(_, v) => set(v as number)} />
    </div>
  );

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div className="flex items-center gap-3 px-3 border-b flex-wrap" style={{ minHeight: 40, borderColor: "var(--c-border)" }}>
        <ToggleButtonGroup size="small" exclusive value={view} onChange={(_, v) => v && setView(v)}>
          <ToggleButton value="render">3D</ToggleButton>
          <ToggleButton value="multi">Multi</ToggleButton>
          <ToggleButton value="axial">Axial</ToggleButton>
          <ToggleButton value="sagittal">Sagittal</ToggleButton>
        </ToggleButtonGroup>
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>{repName(aRef)} vs</span>
        <ToggleButtonGroup size="small" exclusive value={b} onChange={(_, v) => v && setB(v)}>
          {bOptions.map((m) => (
            <ToggleButton key={m} value={m} sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>{repName(m)}</ToggleButton>
          ))}
        </ToggleButtonGroup>
      </div>
      <div className="flex items-center gap-4 px-3 py-1 border-b flex-wrap" style={{ minHeight: 36, borderColor: "var(--c-border)" }}>
        {opSlider(repName(aRef), A_RGB, opA, setOpA)}
        {opSlider(repName(b) || "replicate B", B_RGB, opB, setOpB)}
        {opSlider("overlap", BOTH_RGB, opBoth, setOpBoth)}
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>cornea faint · scar opaque</span>
      </div>
      <div className="px-3 py-1 border-b text-[11px]" style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}>
        Two replicates in the reference frame (cornea + scar, no background). <b style={{ color: BOTH_RGB }}>Red</b> = the
        scans align; a coloured ghost off the red = a residual rigid offset (translation/rotation only — no warping).
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
