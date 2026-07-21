/* 3D replicate-overlap viewer.
   Renders the consensus case's grayscale volume + the agreement map (per-voxel % of replicate
   scans whose scar covers it) as a tiered heat overlay, so the reproducible scar CORE (all scans
   agree) and the variable BOUNDARY fringe (only 1–2 agree) are visible together in 3D.
   A boundary-tolerance slider re-scores the agreement allowing a small residual shift (mm): the
   fringe collapses into the core as the slack absorbs sub-voxel / through-plane misregistration.
   Owns its own Niivue instance (separate from the shared single-volume controller). */

import { useEffect, useRef, useState } from "react";
import { Niivue, SLICE_TYPE } from "@niivue/niivue";
import { ToggleButton, ToggleButtonGroup, Slider, Tooltip } from "@mui/material";
import { resourceUrl } from "../../api/client";
import { releaseVolumes, destroyNiivue, withVolumeNames } from "../../niivue/nvRelease";

const VIEWS = {
  render: SLICE_TYPE.RENDER,
  multi: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.AXIAL,
  sagittal: SLICE_TYPE.SAGITTAL,
} as const;
type ViewKey = keyof typeof VIEWS;

// Agreement values are 0 / 33 / 66 / 100 for 3 scans. With cal_min 16 / cal_max 100 they map to LUT
// indices ~51 / ~152 / 255 → blue (1 scan, least reproducible) → yellow (2) → red (all, robust core).
const OVERLAP_CMAP = {
  R: [0, 40, 245, 235],
  G: [0, 120, 215, 40],
  B: [0, 235, 40, 40],
  A: [0, 150, 205, 255],
  I: [0, 50, 150, 255],
};

interface Stats {
  tol_mm: number; n: number; mean_pairwise_dice: number | null; strict_pairwise_dice: number | null;
  native_scar_mm3: number | null; native_scar_cv_percent: number | null;
  consensus_mm3: number; core_mm3: number;
}

export function OverlapViewer({ caseId, nScans }: { caseId: string; nScans: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [view, setView] = useState<ViewKey>("render");
  const [opacity, setOpacity] = useState(0.75);
  const [tolMm, setTolMm] = useState(0);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  const agreementUrl = (tol: number) =>
    resourceUrl(`/api/case/${caseId}/agreement.nii.gz?tol_mm=${tol}&t=${Date.now()}`);

  // Load the faint cornea base + the agreement overlay together. The cornea is best-effort context: if
  // /cornea.nii.gz is unavailable, fall back to the agreement alone so the core scar-overlap still renders.
  // The agreement is always the LAST volume.
  const loadOverlay = async (nv: Niivue, tol: number) => {
    const cmap = (nv.colormaps?.() ?? []).includes("overlap3") ? "overlap3" : "warm";
    const agr = { url: agreementUrl(tol), colormap: cmap, opacity, cal_min: 16, cal_max: 100 };
    // loadVolumes() REPLACES the volume list by assignment, which never runs removeVolume() — the only
    // path that drops the NVImage from niivue's strong mediaUrlMap. So without this drain every tolerance
    // change stranded both decoded volumes for the life of this viewer (see src/niivue/nvRelease.ts).
    // withVolumeNames additionally suppresses niivue's discarded full-size probe fetch on ?t= URLs.
    releaseVolumes(nv);
    try {
      const cornea = { url: resourceUrl(`/api/case/${caseId}/cornea.nii.gz?t=${Date.now()}`), colormap: "gray", opacity: 1, cal_min: 0, cal_max: 4 };
      await nv.loadVolumes(withVolumeNames([cornea, agr]));
    } catch {
      releaseVolumes(nv);   // a partial load may have landed one volume before failing
      await nv.loadVolumes(withVolumeNames([agr]));   // cornea context unavailable — show the agreement alone
    }
  };

  const fetchStats = async (tol: number) => {
    try {
      const r = await fetch(resourceUrl(`/api/case/${caseId}/agreement-stats?tol_mm=${tol}`));
      if (r.ok) setStats(await r.json());
    } catch { /* readout is best-effort */ }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!canvas.getContext("webgl2")) {
      setError("The 3D overlap needs a WebGL2 context — open the app in Chrome/Firefox (not the VS Code Simple Browser).");
      setLoading(false);
      return;
    }
    let cancelled = false;
    // isNearestInterpolation: crisp voxels — matches the main grayscale viewer (nvController), and this loads a
    // DISCRETE agreement map so linear sampling would halo/blend between its tiers into false values anyway.
    const nv = new Niivue({ backColor: [0.11, 0.11, 0.12, 1], show3Dcrosshair: true, isColorbar: false, dragAndDropEnabled: false, isNearestInterpolation: true });
    try {
      nv.attachToCanvas(canvas);
      try { nv.addColormap("overlap3", OVERLAP_CMAP); } catch { /* older niivue → fall back to "warm" */ }
      nvRef.current = nv;
    } catch (e) {
      setError(`Niivue failed to initialise: ${e instanceof Error ? e.message : String(e)}`);
      setLoading(false);
      return;
    }
    (async () => {
      await loadOverlay(nv, 0);   // faint consensus cornea (context) + scar agreement (overlap core = red)
      if (cancelled) return;
      // No updateGLVolume(): loadVolumes → addVolumesFromUrl → addVolume → setVolume already ran one per
      // volume, and setSliceType ends in drawScene(). An extra call re-allocates the layer-1 3-D texture
      // and orphans the live one (niivue never deletes the old handle — see nvRelease).
      nv.setSliceType(VIEWS.render);
      fetchStats(0);
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

  useEffect(() => { nvRef.current?.setSliceType(VIEWS[view]); }, [view]);
  useEffect(() => {
    const nv = nvRef.current;
    if (nv && nv.volumes.length > 1) { nv.setOpacity(nv.volumes.length - 1, opacity); nv.drawScene(); }
  }, [opacity]);

  // Reload the overlay at a new boundary tolerance (slider release) + refresh the readout.
  const applyTolerance = async (tol: number) => {
    const nv = nvRef.current;
    if (!nv) return;
    setBusy(true);
    try {
      await loadOverlay(nv, tol);   // replaces both layers (cornea base + agreement) — no accumulation
      // setSliceType redraws and loadOverlay's loadVolumes already rebuilt the layer stack; a further
      // updateGLVolume() would orphan a full-size layer-1 texture on EVERY tolerance-slider release.
      nv.setSliceType(VIEWS[view]);
      await fetchStats(tol);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const swatch = (c: string, label: string) => (
    <span className="flex items-center gap-1 text-[11px]" style={{ color: "var(--c-text-dim)" }}>
      <span style={{ width: 10, height: 10, borderRadius: 2, background: c, flex: "none" }} /> {label}
    </span>
  );

  const dice = stats?.mean_pairwise_dice;
  const strict = stats?.strict_pairwise_dice;

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div className="flex items-center gap-3 px-3 border-b flex-wrap" style={{ minHeight: 40, borderColor: "var(--c-border)" }}>
        <ToggleButtonGroup size="small" exclusive value={view} onChange={(_, v) => v && setView(v)}>
          <ToggleButton value="render">3D</ToggleButton>
          <ToggleButton value="multi">Multi</ToggleButton>
          <ToggleButton value="axial">Axial</ToggleButton>
          <ToggleButton value="sagittal">Sagittal</ToggleButton>
        </ToggleButtonGroup>
        <div className="flex items-center gap-2" style={{ width: 140 }}>
          <span className="text-[11px] whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>overlay</span>
          <Slider size="small" min={0} max={1} step={0.05} value={opacity} onChange={(_, v) => setOpacity(v as number)} />
        </div>
        <Tooltip arrow title="Allow the scar to match if it lies within this distance of another scan — absorbs small residual shift / through-plane sampling so a thin boundary offset isn't counted as disagreement.">
          <div className="flex items-center gap-2" style={{ width: 200 }}>
            <span className="text-[11px] whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>tolerance {tolMm.toFixed(2)}mm</span>
            <Slider size="small" min={0} max={0.15} step={0.01} value={tolMm} disabled={busy}
              onChange={(_, v) => setTolMm(v as number)}
              onChangeCommitted={(_, v) => applyTolerance(v as number)} />
          </div>
        </Tooltip>
        <div className="flex items-center gap-3" style={{ marginLeft: "auto" }}>
          {swatch("rgb(40,120,235)", `1/${nScans}`)}
          {swatch("rgb(245,215,40)", `2/${nScans}`)}
          {swatch("rgb(235,40,40)", `${nScans}/${nScans} core`)}
        </div>
      </div>

      {/* reproducibility readout */}
      <div className="flex items-center gap-4 px-3 py-1 border-b text-[11px] flex-wrap" style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}>
        {stats?.native_scar_mm3 != null && (
          <span>scar <b style={{ color: "var(--c-text)" }}>{stats.native_scar_mm3} mm³</b>
            {stats.native_scar_cv_percent != null && <> · CV {stats.native_scar_cv_percent}%</>}</span>
        )}
        {dice != null && (
          <span>pairwise scar Dice{" "}
            <b style={{ color: "var(--c-text)" }}>
              {tolMm > 0 && strict != null ? `${strict} → ${dice}` : dice}
            </b>
            {tolMm > 0 ? ` @ ±${tolMm.toFixed(2)}mm tolerance` : " (strict overlap)"}</span>
        )}
        {busy && <span style={{ color: "var(--c-accent)" }}>recomputing…</span>}
      </div>

      <div className="relative flex-1 min-h-0 min-w-0">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
        {(loading || error) && (
          <div className="absolute inset-0 flex items-center justify-center flex-col gap-2 p-6 text-center pointer-events-none" style={{ color: "var(--c-text-dim)" }}>
            {error ? <span style={{ fontSize: 13, color: "var(--c-red)", maxWidth: 460 }}>{error}</span>
                   : <span style={{ fontSize: 13 }}>Loading 3D overlap…</span>}
          </div>
        )}
      </div>
    </div>
  );
}
