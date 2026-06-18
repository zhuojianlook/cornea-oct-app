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
// Cornea semi-transparent so the opaque scar is visible WITHIN the cornea in the 3D render.
const DRAW_CMAP = {
  R: [0, 70, 235],
  G: [0, 160, 95],
  B: [0, 235, 95],
  A: [0, 130, 255],
  I: [0, 1, 2],
  labels: ["", "cornea", "scar"],
};

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

/** Smart 3-D fill: GrowCut propagates the scribbled cornea/scar labels through the whole volume. */
export function smartFill(): void { if (nv) { nv.drawGrowCut(); nv.drawScene(); } }
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
