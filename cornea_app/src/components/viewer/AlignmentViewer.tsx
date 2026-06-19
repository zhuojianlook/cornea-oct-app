/* Volume-alignment viewer.
   Overlays the REFERENCE warped volume (gray) with a selected replicate's warped volume (red, semi-
   transparent) — both already in the reference frame (registered by build_consensus). Where the scans
   align, the red corneal band sits on top of the gray one; where they don't, the red ghosts off to the
   side. Scrub the slices (Multi/Axial/Sagittal) or rotate (3D), adjust opacity, or Blink to flip the
   moving scan on/off. Use this to judge VOLUME alignment before trusting the scar overlap.
   Owns its own Niivue instance. */

import { useEffect, useRef, useState } from "react";
import { Niivue, SLICE_TYPE } from "@niivue/niivue";
import { ToggleButton, ToggleButtonGroup, Slider, Button } from "@mui/material";
import { resourceUrl } from "../../api/client";

const VIEWS = {
  multi: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.AXIAL,
  sagittal: SLICE_TYPE.SAGITTAL,
  render: SLICE_TYPE.RENDER,
} as const;
type ViewKey = keyof typeof VIEWS;

const short = (cid: string) => cid.split("_").pop() || cid;

export function AlignmentViewer({ caseId, members, refCid }: { caseId: string; members: string[]; refCid?: string }) {
  const ref = refCid && members.includes(refCid) ? refCid : members[0];
  const movings = members.filter((m) => m !== ref);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [moving, setMoving] = useState(movings[0] ?? "");
  const [view, setView] = useState<ViewKey>("multi");
  const [opacity, setOpacity] = useState(0.5);
  const [shown, setShown] = useState(true);   // blink toggle
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const volUrl = (cid: string) => resourceUrl(`/api/case/${caseId}/scan/${cid}/volume.nii.gz?t=${Date.now()}`);

  // mount: reference volume as the gray base
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!canvas.getContext("webgl2")) {
      setError("The alignment viewer needs a WebGL2 context — open the app in Chrome/Firefox.");
      setLoading(false);
      return;
    }
    let cancelled = false;
    const nv = new Niivue({ backColor: [0.11, 0.11, 0.12, 1], show3Dcrosshair: true, isColorbar: false, dragAndDropEnabled: false });
    try {
      nv.attachToCanvas(canvas);
      nvRef.current = nv;
    } catch (e) {
      setError(`Niivue failed to initialise: ${e instanceof Error ? e.message : String(e)}`);
      setLoading(false);
      return;
    }
    (async () => {
      await nv.loadVolumes([{ url: volUrl(ref), colormap: "gray" }]);
      if (!cancelled && moving) await nv.addVolumeFromUrl({ url: volUrl(moving), colormap: "red", opacity });
      if (cancelled) return;
      nv.setSliceType(VIEWS.multi);
      nv.updateGLVolume();
    })()
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; nvRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId]);

  // swap the moving (red) overlay when the selected scan changes
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !moving || nv.volumes.length === 0) return;
    let cancelled = false;
    (async () => {
      while (nv.volumes.length > 1) nv.removeVolumeByIndex(nv.volumes.length - 1);
      await nv.addVolumeFromUrl({ url: volUrl(moving), colormap: "red", opacity: shown ? opacity : 0 });
      if (!cancelled) nv.updateGLVolume();
    })().catch((e) => !cancelled && setError(String(e)));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [moving]);

  useEffect(() => { nvRef.current?.setSliceType(VIEWS[view]); }, [view]);
  useEffect(() => {
    const nv = nvRef.current;
    if (nv && nv.volumes.length > 1) { nv.setOpacity(nv.volumes.length - 1, shown ? opacity : 0); nv.drawScene(); }
  }, [opacity, shown]);

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div className="flex items-center gap-3 px-3 border-b flex-wrap" style={{ minHeight: 40, borderColor: "var(--c-border)" }}>
        <ToggleButtonGroup size="small" exclusive value={view} onChange={(_, v) => v && setView(v)}>
          <ToggleButton value="multi">Multi</ToggleButton>
          <ToggleButton value="axial">Axial</ToggleButton>
          <ToggleButton value="sagittal">Sagittal</ToggleButton>
          <ToggleButton value="render">3D</ToggleButton>
        </ToggleButtonGroup>
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
          ref <b style={{ color: "var(--c-text)" }}>{short(ref)}</b> (gray) vs
        </span>
        <ToggleButtonGroup size="small" exclusive value={moving} onChange={(_, v) => v && setMoving(v)}>
          {movings.map((m) => (
            <ToggleButton key={m} value={m} sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>{short(m)}</ToggleButton>
          ))}
        </ToggleButtonGroup>
        <span className="text-[11px]" style={{ color: "rgb(235,80,80)" }}>(red)</span>
        <div className="flex items-center gap-2" style={{ width: 150 }}>
          <span className="text-[11px] whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>red opacity</span>
          <Slider size="small" min={0} max={1} step={0.05} value={opacity} onChange={(_, v) => setOpacity(v as number)} />
        </div>
        <Button size="small" variant={shown ? "outlined" : "contained"} onClick={() => setShown((s) => !s)}
          sx={{ py: 0, px: 1, fontSize: 11, textTransform: "none" }}>
          {shown ? "Blink (hide red)" : "Show red"}
        </Button>
      </div>

      <div className="px-3 py-1 border-b text-[11px]" style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}>
        Registered volumes in the reference frame — where the <span style={{ color: "rgb(235,80,80)" }}>red</span> corneal
        band overlaps the gray one, the scans are aligned; a red ghost offset from the gray anatomy is residual misalignment.
      </div>

      <div className="relative flex-1 min-h-0 min-w-0">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
        {(loading || error) && (
          <div className="absolute inset-0 flex items-center justify-center flex-col gap-2 p-6 text-center pointer-events-none" style={{ color: "var(--c-text-dim)" }}>
            {error ? <span style={{ fontSize: 13, color: "var(--c-red)", maxWidth: 460 }}>{error}</span>
                   : <span style={{ fontSize: 13 }}>Loading volumes…</span>}
          </div>
        )}
      </div>
    </div>
  );
}
