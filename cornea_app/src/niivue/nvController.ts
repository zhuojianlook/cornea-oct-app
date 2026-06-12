/* ──────────────────────────────────────────────────────────
   Shared Niivue instance controller.
   A single Niivue is attached to the viewer canvas; toolbars and panels
   reach it through this module so they all drive the same scene.
   ────────────────────────────────────────────────────────── */

import { Niivue, SLICE_TYPE } from "@niivue/niivue";

export type ViewName = "multi" | "axial" | "coronal" | "sagittal" | "render";

const SLICE: Record<ViewName, number> = {
  multi: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.AXIAL,
  coronal: SLICE_TYPE.CORONAL,
  sagittal: SLICE_TYPE.SAGITTAL,
  render: SLICE_TYPE.RENDER,
};

let nv: Niivue | null = null;

export function getNv(): Niivue | null {
  return nv;
}

let webglError: string | null = null;
export function webglFailure(): string | null {
  return webglError;
}

/** Attach Niivue to the canvas. Returns null (and records webglFailure) if the
 *  browser can't provide a WebGL2 context, so the rest of the app still works. */
export function attach(canvas: HTMLCanvasElement): Niivue | null {
  if (nv) return nv;
  // Probe for WebGL2 first — niivue throws hard without it, which would
  // otherwise blank the whole React tree.
  const probe = canvas.getContext("webgl2");
  if (!probe) {
    webglError =
      "This browser/window can't provide a WebGL2 context, so the 3D viewer is disabled. " +
      "Open the app in Chrome or Firefox (not the VS Code Simple Browser). " +
      "Seed/segmentation thumbnails on the right still work.";
    return null;
  }
  try {
    nv = new Niivue({
      backColor: [0.11, 0.11, 0.12, 1],
      show3Dcrosshair: true,
      isColorbar: false,
      dragAndDropEnabled: false,
    });
    nv.attachToCanvas(canvas);
    nv.setSliceType(SLICE.multi);
    webglError = null;
    return nv;
  } catch (e) {
    webglError = `Niivue failed to initialise WebGL: ${e instanceof Error ? e.message : String(e)}`;
    nv = null;
    return null;
  }
}

/** Load (or replace) the grayscale base volume. */
export async function loadVolume(url: string): Promise<void> {
  if (!nv) throw new Error("Niivue not attached");
  await nv.loadVolumes([{ url, colormap: "gray" }]);
}

export function setView(view: ViewName): void {
  if (!nv) return;
  nv.setSliceType(SLICE[view]);
}

export function hasVolume(): boolean {
  return !!nv && nv.volumes.length > 0;
}

// ── Drawing layer (interactive seed editing) ───────────────────────────────
/** Load a label NIfTI as the editable drawing bitmap (no binarize → keep 1/2/3). */
export async function loadDrawing(url: string): Promise<void> {
  if (!nv) throw new Error("Niivue not attached");
  await nv.loadDrawingFromUrl(url, false);
  nv.setDrawingEnabled(true);
}

/** Pen label: 0 erase, 1 cornea, 2 background, 3 scar. */
export function setPen(label: number): void {
  if (!nv) return;
  nv.setDrawingEnabled(true);
  nv.setPenValue(label, false);
}

export function setDrawingEnabled(on: boolean): void {
  if (!nv) return;
  nv.setDrawingEnabled(on);
}

export function setDrawOpacity(opacity: number): void {
  if (!nv) return;
  nv.drawOpacity = opacity;
  nv.drawScene();
}

/** Export the edited drawing bitmap as NIfTI bytes. */
export async function exportDrawing(): Promise<Uint8Array | null> {
  if (!nv) return null;
  const result = await nv.saveImage({ filename: "", isSaveDrawing: true, volumeByIndex: 0 });
  return result instanceof Uint8Array ? result : null;
}

export function hasDrawing(): boolean {
  return !!nv && !!nv.drawBitmap;
}

// ── Segmentation overlay (Stage 2 result) ──────────────────────────────────
let segUrl: string | null = null;

/** Add/replace the segmentation labelmap as a coloured overlay on the volume. */
export async function loadSegmentation(url: string, opacity: number): Promise<void> {
  if (!nv) throw new Error("Niivue not attached");
  // Remove a prior segmentation overlay (any volume past the base).
  while (nv.volumes.length > 1) nv.removeVolumeByIndex(nv.volumes.length - 1);
  // Labels are 0=bg (transparent, below cal_min), 1=cornea, 2=scar.
  await nv.addVolumeFromUrl({ url, colormap: "warm", opacity, cal_min: 0.9, cal_max: 2.1 });
  segUrl = url;
  nv.updateGLVolume();
}

export function setSegmentationOpacity(opacity: number): void {
  if (!nv || nv.volumes.length < 2) return;
  nv.setOpacity(nv.volumes.length - 1, opacity);
}

export function removeSegmentation(): void {
  if (!nv) return;
  while (nv.volumes.length > 1) nv.removeVolumeByIndex(nv.volumes.length - 1);
  segUrl = null;
}

export function hasSegmentation(): boolean {
  return segUrl !== null && !!nv && nv.volumes.length > 1;
}
