/* Shared Niivue controller for the ground-truth annotator.
   Pure client-side: load a volume from bytes, paint on a BLANK drawing layer (cornea=1, scar=2,
   erase=0), smart-fill (GrowCut), and export the drawing as a 0/1/2 NIfTI labelmap co-registered to
   the volume. No backend. */

import { Niivue, SLICE_TYPE } from "@niivue/niivue";

export type ViewName = "multi" | "axial" | "coronal" | "sagittal" | "render";
const SLICE: Record<ViewName, number> = {
  multi: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.AXIAL,
  coronal: SLICE_TYPE.CORONAL,
  sagittal: SLICE_TYPE.SAGITTAL,
  render: SLICE_TYPE.RENDER,
};

// Pen labels → ground-truth labels (exported directly): 1 cornea (blue), 2 scar (red), 0 erase.
// 3 = BACKGROUND seed (gray) — used only to seed Smart fill (GrowCut) like Slicer "Grow from
// seeds"; it's mapped back to 0 (background) after growcut / on export, so it never ships.
// Cornea semi-transparent so the opaque scar is visible WITHIN the cornea in the 3D render.
const DRAW_CMAP = {
  R: [0, 70, 235, 150],
  G: [0, 160, 95, 150],
  B: [0, 235, 95, 165],
  A: [0, 130, 255, 90],
  I: [0, 1, 2, 3],
  labels: ["", "cornea", "scar", "background"],
};
const BG_SEED = 3;

let nv: Niivue | null = null;
let webglError: string | null = null;

export function getNv(): Niivue | null { return nv; }
export function webglFailure(): string | null { return webglError; }

export function attach(canvas: HTMLCanvasElement): Niivue | null {
  if (nv) return nv;
  if (!canvas.getContext("webgl2")) {
    webglError = "This window can't provide a WebGL2 context, so the 3D viewer/annotator is disabled.";
    return null;
  }
  try {
    nv = new Niivue({ backColor: [0.11, 0.11, 0.12, 1], show3Dcrosshair: true, isColorbar: false, dragAndDropEnabled: false });
    nv.attachToCanvas(canvas);
    nv.setSliceType(SLICE.multi);
    if (typeof window !== "undefined") (window as unknown as { nv: Niivue }).nv = nv;  // debug hook
    webglError = null;
    return nv;
  } catch (e) {
    webglError = `Niivue failed to initialise WebGL: ${e instanceof Error ? e.message : String(e)}`;
    nv = null;
    return null;
  }
}

/** Load a grayscale volume from raw NIfTI bytes and start a BLANK editable drawing. */
export async function loadVolumeBytes(bytes: Uint8Array, name: string, drawOpacity: number): Promise<void> {
  if (!nv) throw new Error("Niivue not attached");
  while (nv.volumes.length) nv.removeVolumeByIndex(nv.volumes.length - 1);
  // copy into a fresh ArrayBuffer (avoids any shared-buffer surprises from the fs read)
  const ab = bytes.slice().buffer;
  await nv.loadFromArrayBuffer(ab, name);
  nv.createEmptyDrawing();
  try { nv.setDrawColormap(DRAW_CMAP as unknown as string); } catch { /* default LUT */ }
  nv.setDrawOpacity(drawOpacity);
  nv.setDrawingEnabled(true);
  nv.drawScene();
}

export function hasVolume(): boolean { return !!nv && nv.volumes.length > 0; }
export function setView(v: ViewName): void { if (nv) nv.setSliceType(SLICE[v]); }

// ── Pen / brush ────────────────────────────────────────────────────────────
export function setPen(label: number, filled = false): void {
  if (!nv) return;
  nv.setDrawingEnabled(true);
  nv.setPenValue(label, filled);
}
export function setPenSize(size: number): void { if (nv) nv.opts.penSize = Math.max(1, Math.round(size)); }
export function setDrawingEnabled(on: boolean): void { if (nv) nv.setDrawingEnabled(on); }
export function setDrawOpacity(o: number): void { if (nv) { nv.drawOpacity = o; nv.drawScene(); } }

// ── Crosshair lock ───────────────────────────────────────────────────────────
// Painting in niivue moves the crosshair (and thus the other views) to the brush. The canvas captures
// the crosshair while NOT painting (navigate / hover) and restores it during a paint stroke, so the
// other views stay put on the slice the user chose. (#2 — paint is a separate gesture from navigation.)
let savedCrosshair: Float32Array | null = null;
export function lockCrosshair(): void {
  const p = nv?.scene?.crosshairPos as Float32Array | undefined;
  if (p && p.length >= 3) savedCrosshair = p.slice();
}
export function restoreCrosshair(): void {
  if (!nv || !savedCrosshair) return;
  (nv.scene as unknown as { crosshairPos: Float32Array }).crosshairPos = savedCrosshair.slice();
  nv.drawScene();
}

/** On-screen brush size (CSS px) for the current penSize at a canvas-relative point, or null if the
    point isn't over a 2-D slice tile. niivue paints a penSize×penSize voxel block in-plane, but its
    on-screen size depends on the per-tile zoom — so a fixed cursor over-/under-states the real paint.
    This maps penSize voxels through the hovered tile's pixels-per-voxel so the cursor matches the
    voxels actually painted (#3). */
export function brushScreenSize(xCss: number, yCss: number): { w: number; h: number } | null {
  if (!nv || !nv.volumes.length) return null;
  const slices = (nv as unknown as {
    screenSlices?: Array<{ leftTopWidthHeight: number[]; axCorSag: number; fovMM: number[] }>;
  }).screenSlices;
  if (!slices?.length) return null;
  const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
  const xd = xCss * dpr, yd = yCss * dpr; // niivue tile coords are in device px (canvas backing store)
  const pen = Math.max(1, nv.opts.penSize ?? 1);
  const isMM = !!nv.opts.isSliceMM;
  const pd = nv.volumes[0]?.hdr?.pixDims;
  const sp = (i: number) => (pd && pd.length > i && pd[i] ? Math.abs(pd[i]) : 1);
  for (const s of slices) {
    if (s.axCorSag > 2) continue; // 0/1/2 = axial/coronal/sagittal; skip the 3-D render tile
    const [lx, ly, lw, lh] = s.leftTopWidthHeight;
    if (lw <= 0 || lh <= 0) continue;
    if (xd < lx || yd < ly || xd > lx + lw || yd > ly + lh) continue;
    const fov = s.fovMM;
    if (!fov || fov.length < 2 || fov[0] <= 0 || fov[1] <= 0) return null;
    // In voxel mode fov is voxels (no spacing). In mm mode convert penSize voxels → mm via the two
    // in-plane physical axes: axial=x,y · coronal=x,z · sagittal=y,z (pixDims 1=x 2=y 3=z).
    let spH = 1, spV = 1;
    if (isMM) {
      if (s.axCorSag === 0) { spH = sp(1); spV = sp(2); }
      else if (s.axCorSag === 1) { spH = sp(1); spV = sp(3); }
      else { spH = sp(2); spV = sp(3); }
    }
    return { w: (pen * (lw / fov[0]) * spH) / dpr, h: (pen * (lh / fov[1]) * spV) / dpr };
  }
  return null;
}

/** Overwrite every `from`-labelled voxel with `to` in the drawing bitmap. */
function remapLabel(from: number, to: number): void {
  const b = nv?.drawBitmap;
  if (!b) return;
  for (let i = 0; i < b.length; i++) if (b[i] === from) b[i] = to;
}

/** Smart 3-D fill = Slicer "Grow from seeds": GrowCut partitions every unlabelled voxel among the
    nearest scribbled seed (cornea=1, scar=2, AND background=3), then background seeds map back to 0.
    Without background seeds GrowCut floods the whole volume into cornea/scar; the background pen lets
    the user mark "this is NOT cornea" so the grown region stops at a real boundary. */
export function smartFill(): void {
  if (!nv) return;
  nv.drawGrowCut();
  remapLabel(BG_SEED, 0);
  try { nv.refreshDrawing(true); } catch { /* texture refresh best-effort */ }
  nv.drawScene();
}
export function undoDrawing(): void { if (nv) { try { nv.drawUndo(); nv.drawScene(); } catch { /* nothing */ } } }

/** Reset to a blank drawing (discard all paint on the current volume). */
export function clearDrawing(): void {
  if (!nv) return;
  nv.createEmptyDrawing();
  nv.drawScene();
}

// ── Export ──────────────────────────────────────────────────────────────────
/** The drawing as NIfTI bytes (values 0/1/2, co-registered to the volume). */
export async function exportLabelmapBytes(): Promise<Uint8Array | null> {
  if (!nv) return null;
  remapLabel(BG_SEED, 0); // background seeds are a labelling aid only — never part of the saved GT
  try { nv.refreshDrawing(true); } catch { /* best-effort */ }
  const r = await nv.saveImage({ filename: "", isSaveDrawing: true, volumeByIndex: 0 });
  return r instanceof Uint8Array ? r : null;
}

/** Voxel counts per painted label + voxel volume (mm³) from the loaded volume header. */
export function drawStats(): { cornea: number; scar: number; mm3: number; spacing: [number, number, number] } {
  const out = { cornea: 0, scar: 0, mm3: 0, spacing: [0, 0, 0] as [number, number, number] };
  if (!nv || !nv.drawBitmap) return out;
  const b = nv.drawBitmap;
  for (let i = 0; i < b.length; i++) { if (b[i] === 1) out.cornea++; else if (b[i] === 2) out.scar++; }
  const pd = nv.volumes[0]?.hdr?.pixDims;
  if (pd && pd.length >= 4) {
    out.spacing = [pd[1], pd[2], pd[3]];
    out.mm3 = Math.abs(pd[1] * pd[2] * pd[3]);
  }
  return out;
}
