/* Manual-GT comparison viewer.
   Renders the case's grayscale working volume + an AGREEMENT overlay comparing the app's auto
   segmentation against an imported MANUAL ground-truth labelmap, for one class (scar or cornea):
     green = agree (both)        red = auto-only (over-segmentation, FP)        blue = GT-only (missed, FN)
   So the boundary disagreement — where the semiautomated method differs from the human label — is
   visible in 3D / on every slice. Owns its own Niivue instance (separate from the shared controller). */

import { useEffect, useRef, useState } from "react";
import { Niivue, SLICE_TYPE } from "@niivue/niivue";
import { ToggleButton, ToggleButtonGroup, Slider } from "@mui/material";
import { resourceUrl } from "../../api/client";

const VIEWS = {
  multi: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.AXIAL,
  coronal: SLICE_TYPE.CORONAL,
  sagittal: SLICE_TYPE.SAGITTAL,
  render: SLICE_TYPE.RENDER,
} as const;
type ViewKey = keyof typeof VIEWS;

// Agreement values: 1=agree (TP), 2=auto-only (FP), 3=GT-only (FN). With cal_min 0 / cal_max 3 these
// land on LUT indices 0 / 85 / 170 / 255 → transparent / green / red / blue. (Background 0 → transparent.)
const GT_CMAP = {
  R: [0, 40, 235, 60],
  G: [0, 200, 60, 120],
  B: [0, 70, 50, 245],
  A: [0, 255, 255, 255],
  I: [0, 85, 170, 255],
};

type Klass = "scar" | "cornea";

export function GtCompareViewer({
  caseId, name, klass, onClose, onClassChange,
}: {
  caseId: string; name: string; klass: Klass;
  onClose: () => void; onClassChange: (k: Klass) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);
  const firstKlass = useRef(true); // the mount effect already loads the initial-klass overlay
  const [view, setView] = useState<ViewKey>("multi");
  const [opacity, setOpacity] = useState(0.6);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const agreementUrl = (k: Klass) =>
    resourceUrl(`/api/case/${caseId}/manual-gt/${encodeURIComponent(name)}/agreement.nii.gz?klass=${k}&t=${Date.now()}`);

  // mount: gray working volume + agreement overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!canvas.getContext("webgl2")) {
      setError("The comparison viewer needs a WebGL2 context — open the app in Chrome/Firefox.");
      setLoading(false);
      return;
    }
    let cancelled = false;
    const nv = new Niivue({
      backColor: [0.11, 0.11, 0.12, 1], show3Dcrosshair: true, isColorbar: false,
      dragAndDropEnabled: false, isNearestInterpolation: true,
    });
    try {
      nv.attachToCanvas(canvas);
      try { nv.addColormap("gtcompare", GT_CMAP); }
      catch (e) { console.warn("gtcompare colormap unavailable — overlay falls back to 'warm' (legend may not match):", e); }
      nvRef.current = nv;
    } catch (e) {
      setError(`Niivue failed to initialise: ${e instanceof Error ? e.message : String(e)}`);
      setLoading(false);
      return;
    }
    (async () => {
      const vol = resourceUrl(`/api/case/${caseId}/volume.nii.gz?t=${Date.now()}`);
      await nv.loadVolumes([{ url: vol, colormap: "gray" }]);
      const cmap = (nv.colormaps?.() ?? []).includes("gtcompare") ? "gtcompare" : "warm";
      if (cancelled) return;
      await nv.addVolumeFromUrl({ url: agreementUrl(klass), colormap: cmap, opacity, cal_min: 0, cal_max: 3 });
      if (cancelled) return;
      nv.setSliceType(VIEWS.multi);
      nv.updateGLVolume();
    })()
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
      // Free the WebGL2 context this viewer owns (matches OverlapViewer / OverlapPairViewer). Without this,
      // each open/close of the GT-compare panel leaks a context; after the browser's ~8-16 live-context cap
      // the comparison + main viewer render blank. VolumeCanvas mounts/unmounts this on every GT toggle.
      try { (nv as unknown as { gl?: WebGLRenderingContext }).gl?.getExtension("WEBGL_lose_context")?.loseContext(); } catch { /* best-effort */ }
      nvRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId, name]);

  // swap the overlay when the class changes (the mount effect already loaded the initial klass)
  useEffect(() => {
    if (firstKlass.current) { firstKlass.current = false; return; }
    const nv = nvRef.current;
    if (!nv || nv.volumes.length === 0) return;
    let cancelled = false;
    (async () => {
      const cmap = (nv.colormaps?.() ?? []).includes("gtcompare") ? "gtcompare" : "warm";
      while (nv.volumes.length > 1) nv.removeVolumeByIndex(nv.volumes.length - 1);
      await nv.addVolumeFromUrl({ url: agreementUrl(klass), colormap: cmap, opacity, cal_min: 0, cal_max: 3 });
      if (!cancelled) nv.updateGLVolume();
    })().catch((e) => !cancelled && setError(String(e)));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [klass]);

  useEffect(() => { nvRef.current?.setSliceType(VIEWS[view]); }, [view]);
  useEffect(() => {
    const nv = nvRef.current;
    if (nv && nv.volumes.length > 1) { nv.setOpacity(nv.volumes.length - 1, opacity); nv.drawScene(); }
  }, [opacity]);

  const swatch = (c: string, label: string) => (
    <span className="flex items-center gap-1 text-[11px]" style={{ color: "var(--c-text-dim)" }}>
      <span style={{ width: 10, height: 10, borderRadius: 2, background: c, flex: "none" }} /> {label}
    </span>
  );

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div className="flex items-center gap-2 px-3 py-1 border-b text-xs"
        style={{ borderColor: "var(--c-border)", background: "var(--c-surface)", color: "var(--c-text-dim)" }}>
        <button onClick={onClose}
          style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-accent)", cursor: "pointer", fontSize: 11, padding: "1px 8px" }}>
          ← Back to segmentation
        </button>
        <span>Auto segmentation vs manual GT <b style={{ color: "var(--c-text)" }}>{name}</b></span>
      </div>

      <div className="flex items-center gap-3 px-3 border-b flex-wrap" style={{ minHeight: 40, borderColor: "var(--c-border)" }}>
        <ToggleButtonGroup size="small" exclusive value={view} onChange={(_, v) => v && setView(v)}>
          <ToggleButton value="multi">Multi</ToggleButton>
          <ToggleButton value="axial">Axial</ToggleButton>
          <ToggleButton value="coronal">Coronal</ToggleButton>
          <ToggleButton value="sagittal">Sagittal</ToggleButton>
          <ToggleButton value="render">3D</ToggleButton>
        </ToggleButtonGroup>
        <ToggleButtonGroup size="small" exclusive value={klass} onChange={(_, v) => v && onClassChange(v)}>
          <ToggleButton value="scar" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Scar</ToggleButton>
          <ToggleButton value="cornea" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Cornea</ToggleButton>
        </ToggleButtonGroup>
        <div className="flex items-center gap-2" style={{ width: 140 }}>
          <span className="text-[11px] whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>overlay</span>
          <Slider size="small" min={0} max={1} step={0.05} value={opacity} onChange={(_, v) => setOpacity(v as number)} />
        </div>
        <div className="flex items-center gap-3" style={{ marginLeft: "auto" }}>
          {swatch("rgb(40,200,70)", "agree")}
          {swatch("rgb(235,60,50)", "auto only")}
          {swatch("rgb(60,120,245)", "GT only")}
        </div>
      </div>

      <div className="px-3 py-1 border-b text-[11px]" style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}>
        <span style={{ color: "rgb(40,200,70)" }}>Green</span> = both agree ·{" "}
        <span style={{ color: "rgb(235,60,50)" }}>red</span> = auto segmented it but the human didn't (over-segmentation) ·{" "}
        <span style={{ color: "rgb(60,120,245)" }}>blue</span> = the human labelled it but auto missed it.
      </div>

      <div className="relative flex-1 min-h-0 min-w-0">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
        {(loading || error) && (
          <div className="absolute inset-0 flex items-center justify-center flex-col gap-2 p-6 text-center pointer-events-none" style={{ color: "var(--c-text-dim)" }}>
            {error ? <span style={{ fontSize: 13, color: "var(--c-red)", maxWidth: 460 }}>{error}</span>
                   : <span style={{ fontSize: 13 }}>Loading comparison…</span>}
          </div>
        )}
      </div>
    </div>
  );
}
