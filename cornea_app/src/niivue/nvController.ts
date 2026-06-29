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
      // Display labels: 1=cornea (blue, semi-transparent so the opaque scar shows THROUGH the shell in
      // 3D), 2/3/4 = scar DENSITY tiers diffuse→moderate→dense (warm-orange → orange-red → deep red),
      // matching the 2D SEGMENT_COLORS. With cal_min 0.5 / cal_max 4.5 (set in loadSegmentation), label
      // v lands at LUT index (v-0.5)/4·255 → 1→32, 2→96, 3→159, 4→223; control points sit at those.
      nv.addColormap("corneaScar", {
        R: [0,  50, 255, 255, 255, 255],
        G: [0, 140, 180, 110,  30,  30],
        B: [0, 255, 120,  80,  70,  70],
        A: [0,  90, 255, 255, 255, 255],
        I: [0,  32,  96, 159, 223, 255],
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

// ── Single-plane slice navigation (#2 — a visible slice scrollbar) ──────────────────────────────────
// niivue keeps the slice position as scene.crosshairPos (a vec3 of 0..1 fractions, one per niivue axis:
// 0=sagittal/L-R, 1=coronal/A-P, 2=axial/I-S) and the per-axis slice COUNT as volumes[0].dimsRAS
// ([n,x,y,z]). We read/set the SAME axis for both, so this is orientation-safe despite the OCT direction
// swap. The user-facing view maps to a niivue axis (matching the SLICE map's axial↔coronal swap):
//   user "sagittal" → niivue sagittal → axis 0 ; "axial" → niivue coronal → axis 1 ; "coronal" → niivue axial → axis 2
const VIEW_AXIS: Record<ViewName, number> = { sagittal: 0, axial: 1, coronal: 2, multi: -1, render: -1 };

/** Number of slices along the active single-plane view's axis (0 for multi/3D or no volume). */
export function sliceCount(view: ViewName): number {
  const ax = VIEW_AXIS[view];
  if (!nv || ax < 0 || nv.volumes.length === 0) return 0;
  const d = (nv.volumes[0] as unknown as { dimsRAS?: number[] }).dimsRAS;
  if (d && d.length >= 4) return Math.max(0, Math.round(d[ax + 1]));
  const dd = (nv.volumes[0] as unknown as { dims?: number[] }).dims;   // fallback: raw NIfTI dims
  return dd && dd.length >= 4 ? Math.max(0, Math.round(dd[ax + 1])) : 0;
}

/** Current slice index (0-based) for the active single-plane view. */
export function getSliceIndex(view: ViewName): number {
  const ax = VIEW_AXIS[view];
  const n = sliceCount(view);
  if (!nv || ax < 0 || n <= 1) return 0;
  const frac = nv.scene.crosshairPos[ax];
  return Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
}

/** Move the active single-plane view to slice `idx` (clamped) and redraw. */
export function setSliceIndex(view: ViewName, idx: number): void {
  const ax = VIEW_AXIS[view];
  const n = sliceCount(view);
  if (!nv || ax < 0 || n <= 1) return;
  nv.scene.crosshairPos[ax] = Math.max(0, Math.min(1, idx / (n - 1)));   // mutate in place (keeps the vec3 type)
  nv.drawScene();
}

export function hasVolume(): boolean {
  return !!nv && nv.volumes.length > 0;
}

// ── Drawing layer (interactive seed editing) ───────────────────────────────
// Drawing-layer colours per pen label, matching the segmentation convention so painting shows the
// RIGHT colour (1 cornea=blue, 2 background=GREY, 3 scar=red). Background is a real non-zero SEED label
// (needed for Smart-fill/GrowCut) that maps to canonical 0 on save — it's drawn GREY (not the old orange,
// which read like scar and vanished on save). Slightly muted + drawn translucent (drawOpacity). Index 0
// (unpainted) is transparent.
const DRAW_CMAP = {
  R: [0, 70, 142, 235],
  G: [0, 160, 142, 95],
  B: [0, 235, 147, 95],
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

/** Count of DISTINCT non-zero seed labels in the drawing bitmap (early-exits at 2). GrowCut/Smart-fill
 *  needs ≥2 (e.g. cornea AND background) to be well-posed — with a single seed it grows that label over
 *  the whole volume and stalls on per-slice readPixels. Used to guard Smart fill (#2). */
export function drawingSeedCount(): number {
  if (!nv || !nv.drawBitmap) return 0;
  const b = nv.drawBitmap as Uint8Array;
  const seen = new Set<number>();
  for (let i = 0; i < b.length; i++) {
    const v = b[i];
    if (v > 0) { seen.add(v); if (seen.size >= 2) return 2; }
  }
  return seen.size;
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
  // Display labels: 0=bg (transparent, below cal_min), 1=cornea (blue), 2/3/4=scar density tiers
  // (diffuse→dense) via "corneaScar" (falls back to "warm" if the custom colormap wasn't registered).
  let cmap = "warm";
  try {
    if ((nv.colormaps?.() ?? []).includes("corneaScar")) cmap = "corneaScar";
  } catch { /* keep warm */ }
  await nv.addVolumeFromUrl({ url, colormap: cmap, opacity, cal_min: 0.5, cal_max: 4.5 });
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
