/* ──────────────────────────────────────────────────────────
   Shared Niivue instance controller.
   A single Niivue is attached to the viewer canvas; toolbars and panels
   reach it through this module so they all drive the same scene.
   ────────────────────────────────────────────────────────── */

import { Niivue, SLICE_TYPE } from "@niivue/niivue";

export type ViewName = "multi" | "axial" | "coronal" | "sagittal" | "render";

// These OCT volumes' direction (NIFTI_DIRECTION) makes niivue's AXIAL tile the EN-FACE/depth plane (the
// ring-shaped view) and niivue's CORONAL tile the B-scan stack — the opposite of the clinical naming the
// user expects. So the user-facing "Coronal" button maps to niivue AXIAL (en-face ring) and "Axial" to
// niivue CORONAL (B-scan). (Same swap the annotator app made; keep the labels clinically intuitive.)
const SLICE: Record<ViewName, number> = {
  multi: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.CORONAL,    // user "Axial" = niivue coronal = B-scan stack
  coronal: SLICE_TYPE.AXIAL,    // user "Coronal" = niivue axial = en-face ring
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
    if (typeof window !== "undefined") (window as unknown as { nv: Niivue }).nv = nv;  // debug/testing hook
    // Distinct overlay colors so cornea (label 1) and scar (label 2) are easy to tell apart in the
    // 3D/WebGL viewer (the default "warm" ramp makes both warm). With cal_min 0.9 / cal_max 2.1,
    // label 1 lands at LUT index ~21 (blue = cornea) and label 2 at ~234 (red = scar); index 0 (bg)
    // is transparent. Matches the 2D viewer's blue-cornea / red-scar convention.
    try {
      // Cornea (idx 1–120) is semi-transparent so the opaque scar (idx 132–255) shows THROUGH the
      // cornea shell in the 3D render (otherwise the blue cornea occludes the embedded red scar).
      nv.addColormap("corneaScar", {
        R: [0, 50, 50, 255, 255],
        G: [0, 140, 140, 58, 58],
        B: [0, 255, 255, 71, 71],
        A: [0, 90, 90, 255, 255],
        I: [0, 1, 120, 132, 255],
      });
    } catch { /* older niivue without addColormap → loadSegmentation falls back to "warm" */ }
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
// Drawing-layer colours per pen label, matching the segmentation convention so painting shows the
// RIGHT colour (1 cornea=blue, 2 background=orange, 3 scar=red). Slightly muted + drawn translucent
// (drawOpacity) so the editable layer reads as "in progress". Index 0 (unpainted) is transparent.
const DRAW_CMAP = {
  R: [0, 70, 230, 235],
  G: [0, 160, 150, 95],
  B: [0, 235, 70, 95],
  A: [0, 255, 255, 255],
  I: [0, 1, 2, 3],
  labels: ["", "cornea", "background", "scar"],
};

/** Load a label NIfTI as the editable drawing bitmap (no binarize → keep 1/2/3). */
export async function loadDrawing(url: string): Promise<void> {
  if (!nv) throw new Error("Niivue not attached");
  // loadDrawingFromUrl returns FALSE (doesn't throw) on a fetch failure or dimension mismatch; if we
  // ignored it, setDrawingEnabled would create a BLANK drawing and the user would paint on an empty
  // layer (silent loss of the segmentation). Surface it as an error instead.
  const ok = await nv.loadDrawingFromUrl(url, false);
  if (!ok) throw new Error("Correction layer failed to load (segmentation drawing missing or mismatched).");
  try { nv.setDrawColormap(DRAW_CMAP as unknown as string); } catch { /* older niivue → default LUT */ }
  nv.setDrawingEnabled(true);
}

/** End correction: stop the pen AND clear the drawing bitmap so it can't linger over the committed
 *  overlay or leak into other stages/cases (a stale drawBitmap renders even when drawing is disabled). */
export function endDrawing(): void {
  if (!nv) return;
  nv.setDrawingEnabled(false);
  try { nv.closeDrawing(); } catch { /* no-op if no drawing */ }
  nv.drawScene();
}

/** Undo the last brush stroke / smart fill (niivue keeps a drawing undo stack). */
export function undoDrawing(): void {
  if (!nv) return;
  try { nv.drawUndo(); nv.drawScene(); } catch { /* nothing to undo */ }
}

/** Pen label: 0 erase, 1 cornea, 2 background, 3 scar. `filled`=true auto-fills a closed outline
 *  (draw a loop around a region → the enclosed area is painted), so a whole patch is one stroke. */
export function setPen(label: number, filled = false): void {
  if (!nv) return;
  nv.setDrawingEnabled(true);
  nv.setPenValue(label, filled);
}

/** Brush thickness (voxels). */
export function setPenSize(size: number): void {
  if (!nv) return;
  nv.opts.penSize = Math.max(1, Math.round(size));
}

/** Smart 3-D fill: GrowCut propagates the scribbled labels (bg/cornea/scar) through the whole volume by
 *  intensity similarity, so the user seeds a few slices and the rest of every view is filled in. */
export function smartFill(): void {
  if (!nv) return;
  nv.drawGrowCut();
  nv.drawScene();
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
  // Labels are 0=bg (transparent, below cal_min), 1=cornea (blue), 2=scar (red) via "corneaScar"
  // (falls back to "warm" if the custom colormap wasn't registered).
  let cmap = "warm";
  try {
    if ((nv.colormaps?.() ?? []).includes("corneaScar")) cmap = "corneaScar";
  } catch { /* keep warm */ }
  await nv.addVolumeFromUrl({ url, colormap: cmap, opacity, cal_min: 0.9, cal_max: 2.1 });
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
