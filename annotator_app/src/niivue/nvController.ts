/* Shared Niivue controller for the ground-truth annotator.
   Pure client-side: load a volume from bytes, paint on a BLANK drawing layer (cornea=1, scar=2,
   erase=0), smart-fill (GrowCut), and export the drawing as a 0/1/2 NIfTI labelmap co-registered to
   the volume. No backend. */

import { Niivue, NVImage, SLICE_TYPE, DRAG_MODE } from "@niivue/niivue";

export type ViewName = "multi" | "axial" | "coronal" | "sagittal" | "render";
// View-name → niivue SLICE_TYPE. NOTE the deliberate AXIAL↔CORONAL SWAP: these OCT volumes carry the
// main app's direction (1,0,0,0,0,1,0,-1,0), so the frame axis (B-scans) maps to physical A-P → niivue
// labels the B-scan plane "coronal" and the en-face/depth plane "axial". The user's convention (matching
// the main app, where a B-scan = "axial") is the opposite, so we expose niivue-CORONAL as the user's
// "Axial" (B-scan arcs) and niivue-AXIAL as the user's "Coronal" (en-face). Sagittal matches in both.
// Paint is UNAFFECTED — it uses niivue's real per-tile axCorSag (tileThroughAxis), not these labels.
const SLICE: Record<ViewName, number> = {
  multi: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.CORONAL,    // user "Axial" = B-scan plane = niivue coronal
  coronal: SLICE_TYPE.AXIAL,    // user "Coronal" = en-face/depth plane = niivue axial
  sagittal: SLICE_TYPE.SAGITTAL,
  render: SLICE_TYPE.RENDER,
};

// Pen labels → ground-truth labels (exported directly): 1 cornea (blue), 2 scar (red), 0 erase.
// 3 = BACKGROUND seed (gray) — used only to seed Smart fill (GrowCut) like Slicer "Grow from
// seeds"; it's mapped back to 0 (background) after growcut / on export, so it never ships.
// Cornea semi-transparent so the opaque scar is visible WITHIN the cornea in the 3D render.
// 4/5 = LIGHTER preview shades of cornea/scar — used only to display an unconfirmed smart-fill preview
// (mapped back to 1/2 on Confirm and on export); seeds/committed use the solid 1/2.
const DRAW_CMAP = {
  R: [0, 70, 235, 150, 120, 255],
  G: [0, 160, 95, 150, 195, 160],
  B: [0, 235, 95, 165, 255, 160],
  A: [0, 130, 255, 90, 80, 80],
  I: [0, 1, 2, 3, 4, 5],
  labels: ["", "cornea", "scar", "background", "cornea-preview", "scar-preview"],
};
const BG_SEED = 3;
const PREVIEW_CORNEA = 4, PREVIEW_SCAR = 5;

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
  // NEAREST-neighbour interpolation = crisp voxels. These OCT volumes are highly anisotropic (frames
  // ~0.04mm vs depth ~0.003mm), so any view spanning the coarse FRAME axis (sagittal = frames×depth,
  // coronal/en-face = lateral×frames) is upscaled ~13× and niivue's default LINEAR interp blurs it.
  // Nearest renders the true voxel grid (matches the main app's pixelated previews). (#blur)
  try { nv.setInterpolation(true); } catch { /* older niivue: leave default */ }
  // capture the auto display window for the brightness/contrast controls (#3)
  const v0 = nv.volumes[0] as unknown as { cal_min?: number; cal_max?: number; global_min?: number; global_max?: number };
  baseWinLo = v0.cal_min ?? v0.global_min ?? 0;
  baseWinHi = v0.cal_max ?? v0.global_max ?? (baseWinLo + 1);
  nv.createEmptyDrawing();
  try { nv.setDrawColormap(DRAW_CMAP as unknown as string); } catch { /* default LUT */ }
  nv.setDrawOpacity(drawOpacity);
  nv.setDrawingEnabled(false); // custom sphere brush owns painting; niivue navigation still sets crosshair
  nv.opts.penSize = 1;
  // Pan/zoom: left-drag stays crosshair (NOT contrast); shift/ctrl+left, middle & right drag = pan.
  (nv as unknown as { opts: { mouseEventConfig?: unknown } }).opts.mouseEventConfig = {
    // default tool is Paint → left-click must NOT move the crosshair (#2); setTool flips this to
    // DRAG_MODE.crosshair only in Navigate mode. Shift/Ctrl+drag stays pan everywhere.
    leftButton: { primary: DRAG_MODE.none, withShift: DRAG_MODE.pan, withCtrl: DRAG_MODE.pan },
    rightButton: DRAG_MODE.pan, centerButton: DRAG_MODE.pan,
  };
  try { nv.setPan2Dxyzmm([0, 0, 0, 1]); } catch { /* */ } // reset zoom/pan for the new volume
  rasIntensity = null; rasRange = null; // drop the previous volume's cached intensity + range
  wandVisited = null; // free the previous volume's wand scratch (re-allocated on next wand use)
  // Three-layer drawing model (#smartfill-preview): seeds = the user's brushstrokes (the ONLY input to
  // smart fill), committed = confirmed segmentation, preview = last smart-fill result (recomputed each
  // run). The displayed drawBitmap is composed = seeds ▸ preview ▸ committed. Confirm bakes preview→committed.
  if (nv.drawBitmap) { const n = nv.drawBitmap.length;
    seedBmp = new Uint8Array(n); committedBmp = new Uint8Array(n); previewBmp = new Uint8Array(n);
    previewing = false; undoStack = []; redoStack = []; }
  // Keep the per-view slice scrollbars (#1) in sync when the user scroll-wheels through slices.
  // Ignore the location changes niivue fires from its OWN paint mousedown/drag: its native canvas
  // listener runs before React's onMouseDown sets strokeActive, so also gate on uiData.mousedown
  // (set true at the very start of mouseDownListener) so the readout never follows the brush.
  (nv as unknown as { onLocationChange: (loc: unknown) => void }).onLocationChange = () => {
    if (strokeActive) return;
    const ui = (nv as unknown as { uiData?: { mousedown?: boolean } }).uiData;
    if (ui?.mousedown && nv!.opts.drawingEnabled) return; // a paint click/drag is in progress
    const v = currentVox();
    if (v && sliceListener) sliceListener(v);
  };
  nv.drawScene();
}

export function hasVolume(): boolean { return !!nv && nv.volumes.length > 0; }
export function setView(v: ViewName): void { if (nv) nv.setSliceType(SLICE[v]); }
/** Force a repaint. Defensive: WebKitGTK can leave a freshly-loaded volume black if the canvas was
    resized right after the first draw — re-issuing drawScene once the layout has settled recovers it. */
export function redraw(): void { if (nv) { try { nv.drawScene(); } catch { /* nothing */ } } }

// ── Pen / brush ────────────────────────────────────────────────────────────
export function setPen(label: number, filled = false): void {
  if (!nv) return;
  // niivue's own pen stays DISABLED — we paint a custom 3-D sphere brush (paintBrush). Keep its value
  // in sync harmlessly; do NOT enable niivue drawing (it would paint in-plane squares + own undo).
  nv.setPenValue(label, filled);
}
// The app paints a 3-D SPHERE brush (round in every view), not niivue's in-plane square. penSize is the
// app-level brush size; niivue's own pen is pinned to 1 voxel (it only sets the crosshair/centre voxel —
// our paintBrush draws the actual sphere). See paintBrush below.
let appPenSize = 3;
export function setPenSize(size: number): void { appPenSize = Math.max(1, Math.round(size)); if (nv) nv.opts.penSize = 1; }
export function getPenSize(): number { return appPenSize; }

// ── Brightness / contrast (display window, #3) ───────────────────────────────
let baseWinLo = 0, baseWinHi = 1; // the volume's auto window, captured at load
/** Apply brightness & contrast as a grayscale display window. Both ∈ [-1,1]; 0,0 = the auto window.
    +brightness lowers the window centre (image brighter); +contrast narrows the window (more contrast). */
export function setWindow(brightness: number, contrast: number): void {
  const v0 = nv?.volumes[0] as unknown as { cal_min: number; cal_max: number } | undefined;
  if (!nv || !v0) return;
  const center = (baseWinLo + baseWinHi) / 2, width = Math.max(1e-6, baseWinHi - baseWinLo);
  const w = width * Math.pow(2, -contrast * 2);
  const c = center - brightness * width;
  v0.cal_min = c - w / 2;
  v0.cal_max = c + w / 2;
  try { nv.updateGLVolume(); } catch { /* */ }
  nv.drawScene();
}

// ── Label locks (#4) ─────────────────────────────────────────────────────────
// Labels in this set are protected: brush, erase and smart-fill will not overwrite voxels that already
// carry a locked label (e.g. lock cornea → a scar/erase stroke can't change cornea voxels).
const lockedLabels = new Set<number>();
export function setLockedLabels(labels: number[]): void { lockedLabels.clear(); for (const l of labels) lockedLabels.add(l); }
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

/** Move the crosshair to the voxel under a canvas-relative point — used ONLY by Navigate mode (#2). The
    left button is inert at the niivue level (DRAG_MODE.none) in every mode, so a Paint/Wand click never
    jumps the crosshair; Navigate calls this explicitly so click/drag still scrubs the crosshair. */
export function setCrosshairAtScreen(xCss: number, yCss: number): void {
  if (!nv) return;
  const tile = tileAtScreen(xCss, yCss);
  if (tile < 0) return;
  const f2f = (nv as unknown as { screenXY2TextureFrac?: (x: number, y: number, t: number) => ArrayLike<number> }).screenXY2TextureFrac;
  if (typeof f2f !== "function") return;
  const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
  const f = f2f.call(nv, xCss * dpr, yCss * dpr, tile);
  if (!f || f[0] < 0) return;
  const cl = (v: number) => Math.max(0, Math.min(1, v));
  (nv.scene as unknown as { crosshairPos: Float32Array }).crosshairPos = new Float32Array([cl(f[0]), cl(f[1]), cl(f[2])]);
  try { nv.drawScene(); } catch { /* */ }
  const v = currentVox();
  if (v && sliceListener) sliceListener(v);
}

/** Reset the 2-D pan/zoom AND recentre the crosshair to the middle of every slice (#3, used by Clear). */
export function centerView(): void {
  if (!nv) return;
  try { nv.setPan2Dxyzmm([0, 0, 0, 1]); } catch { /* */ }
  try { (nv.scene as unknown as { crosshairPos: Float32Array }).crosshairPos = new Float32Array([0.5, 0.5, 0.5]); } catch { /* */ }
  savedCrosshair = null;
  forceDrawAll();
}

/** Force a COMPLETE redraw of every tile now + after the layout settles. WebKitGTK (desktop) can leave a
    pane (often the coronal one) showing a stale drawing after a paint/wand edit because the throttled
    single drawScene lands before the tile repaints — so re-issue refreshDrawing+drawScene on a couple of
    delays. Cheap; Chromium is unaffected. (#1) */
export function forceDrawAll(): void {
  if (!nv) return;
  const go = () => { if (!nv) return; try { nv.refreshDrawing(true); } catch { /* */ } try { nv.drawScene(); } catch { /* */ } };
  go();
  requestAnimationFrame(go);
  // Escalating retries: on some WebKitGTK builds a single late repaint still misses the B-scan
  // (niivue-coronal) tile, so re-issue a few more times as the layout settles. Cheap; Chromium no-ops.
  for (const ms of [60, 180, 400]) setTimeout(go, ms);
}

// ── Slice navigation (per-view scrollbars, #1) ───────────────────────────────
let strokeActive = false;
let sliceListener: ((vox: [number, number, number]) => void) | null = null;
/** Register a callback fired on scroll/navigation with the current [x,y,z] voxel slice indices. */
export function setSliceListener(fn: ((vox: [number, number, number]) => void) | null): void { sliceListener = fn; }
/** Mark a paint stroke active so the slice readout doesn't follow the brush mid-stroke. */
export function setStroke(active: boolean): void { strokeActive = active; }

/** The through-plane RAS axis (0=x/sagittal, 1=y/coronal, 2=z/axial) of a 2-D pane (tile index from
    tileAtScreen), or null for the 3-D render / invalid tile. niivue's axCorSag is 0=axial,1=coronal,
    2=sagittal, whose through-plane axis is z,y,x → 2-axCorSag. paintBrush uses this to confine a
    stroke to the slice under the cursor. */
export function tileThroughAxis(tile: number): number | null {
  const slices = (nv as unknown as { screenSlices?: Array<{ axCorSag: number }> }).screenSlices;
  if (!nv || !slices || tile < 0 || tile >= slices.length) return null;
  const acs = slices[tile].axCorSag;
  return acs >= 0 && acs <= 2 ? 2 - acs : null;
}

/** Per-axis voxel counts [nx, ny, nz] of the loaded volume (RAS/display order), or null. */
export function getDims(): [number, number, number] | null {
  const dr = nv?.volumes[0]?.dimsRAS as number[] | undefined;
  if (!dr || dr.length < 4) return null;
  return [dr[1], dr[2], dr[3]];
}
/** Current crosshair as integer voxel indices [x, y, z] (matches niivue's convertFrac2Vox). */
export function currentVox(): [number, number, number] | null {
  const dr = nv?.volumes[0]?.dimsRAS as number[] | undefined;
  const cp = nv?.scene?.crosshairPos as Float32Array | undefined;
  if (!dr || !cp || dr.length < 4) return null;
  const ix = (a: number) => Math.max(0, Math.min(dr[a + 1] - 1, Math.round(cp[a] * dr[a + 1] - 0.5)));
  return [ix(0), ix(1), ix(2)];
}
/** Jump the through-plane slice for a voxel axis (0=x/sagittal, 1=y/coronal, 2=z/axial) without
    disturbing the two in-plane coords. crosshairPos[axis] = (slice + 0.5)/n — niivue's exact
    voxel→frac, so the slider lands on voxel centres and round-trips with niivue's own scroll. */
export function setVoxAxis(axis: 0 | 1 | 2, slice: number): void {
  const dr = nv?.volumes[0]?.dimsRAS as number[] | undefined;
  const cp = nv?.scene?.crosshairPos as Float32Array | undefined;
  if (!nv || !dr || !cp || dr.length < 4) return;
  const n = dr[axis + 1];
  const s = Math.max(0, Math.min(n - 1, Math.round(slice)));
  cp[axis] = (s + 0.5) / n;
  lockCrosshair();          // a subsequent paint stroke restores to THIS slice, not the previous one
  nv.drawScene();
}

/** RAS-axis voxel spacing in mm, or null. */
function rasSpacing(): [number, number, number] | null {
  const pd = (nv?.volumes[0] as unknown as { pixDimsRAS?: number[] } | undefined)?.pixDimsRAS;
  if (pd && pd.length >= 4) return [Math.abs(pd[1]) || 1, Math.abs(pd[2]) || 1, Math.abs(pd[3]) || 1];
  return null;
}
/** Physical radius (mm) of the brush sphere. penSize is the diameter in median-spacing voxels, so it
    feels like the familiar in-plane voxel size while staying physically round in every view. */
function brushRadiusMm(): number {
  const sp = rasSpacing();
  if (!sp) return 0;
  const s = [sp[0], sp[1], sp[2]].sort((a, b) => a - b);
  return (appPenSize / 2) * s[1]; // median spacing
}

/** On-screen brush cursor as a CIRCLE diameter (CSS px), or null if the point isn't over a 2-D tile.
    The brush is a sphere of radius brushRadiusMm in physical space; niivue renders slices isotropically
    in mm (px-per-mm equal on both screen axes), so the sphere's cross-section is a circle of that radius.
    diameter = 2·R_mm·(tilePx/tileFovMM). (#2) */
export function brushScreenSize(xCss: number, yCss: number): { w: number; h: number } | null {
  if (!nv || !nv.volumes.length) return null;
  const slices = (nv as unknown as {
    screenSlices?: Array<{ leftTopWidthHeight: number[]; axCorSag: number; fovMM: number[] }>;
  }).screenSlices;
  if (!slices?.length) return null;
  const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
  const xd = xCss * dpr, yd = yCss * dpr;
  for (const s of slices) {
    if (s.axCorSag > 2) continue;
    const [lx, ly, lw, lh] = s.leftTopWidthHeight;
    if (lw <= 0 || lh <= 0 || xd < lx || yd < ly || xd > lx + lw || yd > ly + lh) continue;
    const fov = s.fovMM;
    if (!fov || fov.length < 2 || fov[0] <= 0) return null;
    const pxPerMM = lw / fov[0];                  // isotropic in mm
    const d = (2 * brushRadiusMm() * pxPerMM) / dpr;
    return d > 0 ? { w: d, h: d } : null;
  }
  return null;
}

// ── 3-D spherical brush ──────────────────────────────────────────────────────
// Paint a sphere (in physical/mm space) around a voxel, so the stroke is ROUND in axial/coronal/sagittal
// alike (niivue's native pen only paints an in-plane square). drawBitmap is RAS-ordered [nx,ny,nz].
let refreshScheduled = false;
function scheduleRefresh(): void {
  if (refreshScheduled || !nv) return;
  refreshScheduled = true;
  requestAnimationFrame(() => {
    refreshScheduled = false;
    if (!nv) return;
    try { nv.refreshDrawing(true); } catch { /* */ }
    nv.drawScene();
  });
}

/** Index of the 2-D slice tile (axial/coronal/sagittal) under a canvas-relative point, or -1 if the
    point is over the 3-D render pane / inter-pane margin / off-canvas. Used to confine a paint stroke to
    the pane it started on, so dragging outside that pane paints nothing (instead of streaking across). */
export function tileAtScreen(xCss: number, yCss: number): number {
  const slices = (nv as unknown as { screenSlices?: Array<{ leftTopWidthHeight: number[]; axCorSag: number }> }).screenSlices;
  if (!nv || !slices?.length) return -1;
  const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
  const xd = xCss * dpr, yd = yCss * dpr;
  for (let i = 0; i < slices.length; i++) {
    const s = slices[i];
    if (s.axCorSag > 2) continue; // skip the 3-D render tile — not paintable
    const [lx, ly, lw, lh] = s.leftTopWidthHeight;
    if (lw > 0 && lh > 0 && xd >= lx && yd >= ly && xd <= lx + lw && yd <= ly + lh) return i;
  }
  return -1;
}

/** The voxel [x,y,z] under a canvas-relative point (for the hover intensity readout), or null if not
    over a 2-D pane. Uses niivue's exact screen→texture transform. */
export function voxAtScreen(xCss: number, yCss: number): [number, number, number] | null {
  const dr = nv?.volumes[0]?.dimsRAS as number[] | undefined;
  const f2f = (nv as unknown as { screenXY2TextureFrac?: (x: number, y: number, t: number) => ArrayLike<number> }).screenXY2TextureFrac;
  if (!nv || !dr || typeof f2f !== "function") return null;
  const tile = tileAtScreen(xCss, yCss);
  if (tile < 0) return null;
  const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
  const f = f2f.call(nv, xCss * dpr, yCss * dpr, tile);
  if (!f || f[0] < 0) return null;
  const ix = (a: number) => Math.max(0, Math.min(dr[a + 1] - 1, Math.round(f[a] * dr[a + 1] - 0.5)));
  return [ix(0), ix(1), ix(2)];
}

/** Like voxAtScreen but CLAMPED to a specific pane (tile) — returns the NEAREST in-volume voxel even
    when the cursor is in the tile's letterbox padding or dragged off the pane entirely (#1). Lets a
    round brush paint right up to the image edge instead of vanishing (its centre clamps to the border
    voxel, so the disk still covers the edge pixels). The cursor is clamped to the pane rect and the
    texture frac to [0,1]. Used ONLY during an active stroke (the wand still needs the exact voxel). */
export function voxAtScreenClamped(xCss: number, yCss: number, tile: number): [number, number, number] | null {
  const dr = nv?.volumes[0]?.dimsRAS as number[] | undefined;
  const f2f = (nv as unknown as { screenXY2TextureFrac?: (x: number, y: number, t: number) => ArrayLike<number> }).screenXY2TextureFrac;
  const slices = (nv as unknown as { screenSlices?: Array<{ leftTopWidthHeight: number[]; axCorSag: number }> }).screenSlices;
  if (!nv || !dr || typeof f2f !== "function" || !slices || tile < 0 || tile >= slices.length) return null;
  const s = slices[tile];
  if (s.axCorSag > 2) return null; // not a paintable 2-D pane
  const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
  const [lx, ly, lw, lh] = s.leftTopWidthHeight;
  // Clamp the cursor into the pane, then the resulting texture frac into the image.
  const xd = Math.max(lx, Math.min(lx + lw, xCss * dpr));
  const yd = Math.max(ly, Math.min(ly + lh, yCss * dpr));
  const f = f2f.call(nv, xd, yd, tile);
  if (!f) return null;
  const cl = (v: number) => Math.max(0, Math.min(1, v));
  const ix = (a: number) => Math.max(0, Math.min(dr[a + 1] - 1, Math.round(cl(f[a]) * dr[a + 1] - 0.5)));
  return [ix(0), ix(1), ix(2)];
}

// ── Layered drawing: seeds (brushstrokes) ▸ preview (last smart fill) ▸ committed (confirmed) ────────
// Smart fill must grow ONLY from the user's brushstrokes (seeds), never from its own output — so the
// fill is a recomputable PREVIEW. "Confirm" bakes the preview into the committed segmentation. The
// displayed/exported drawBitmap is composed seeds ▸ preview ▸ committed (first non-zero wins).
let seedBmp: Uint8Array | null = null;
let committedBmp: Uint8Array | null = null;
let previewBmp: Uint8Array | null = null;
let previewing = false;
let rasIntensity: ArrayLike<number> | null = null; // RAS-ordered intensity (cached for wand + readout)
let rasRange: [number, number] | null = null;      // cached [min,max] (so the cursor % readout is cheap)
type Snap = { s: Uint8Array; c: Uint8Array };
let undoStack: Snap[] = [];
let redoStack: Snap[] = [];
const UNDO_CAP = 4;

export function isPreviewing(): boolean { return previewing; }
export function canUndo(): boolean { return undoStack.length > 0; }
export function canRedo(): boolean { return redoStack.length > 0; }

/** Recompose the displayed drawBitmap from the three layers (seeds ▸ preview ▸ committed). Seeds and
    committed show in solid colours; the unconfirmed preview shows in a LIGHTER shade (1→4, 2→5). */
function compose(): void {
  if (!nv?.drawBitmap || !seedBmp || !committedBmp || !previewBmp) return;
  const d = nv.drawBitmap, s = seedBmp, p = previewBmp, c = committedBmp;
  for (let i = 0; i < d.length; i++) {
    const sv = s[i];
    if (sv !== 0) d[i] = sv;
    else { const pv = p[i]; d[i] = pv !== 0 ? (pv === 1 ? PREVIEW_CORNEA : pv === 2 ? PREVIEW_SCAR : pv) : c[i]; }
  }
  try { nv.refreshDrawing(true); } catch { /* */ }
  nv.drawScene();
}

const snapshot = (): Snap => ({ s: seedBmp!.slice(), c: committedBmp!.slice() });
function restoreSnap(e: Snap): void {
  seedBmp!.set(e.s); committedBmp!.set(e.c); previewBmp!.fill(0); previewing = false; compose();
}
/** Snapshot seeds+committed for undo — call BEFORE a mutating action (stroke, smart fill, confirm, clear).
    A new action invalidates the redo stack. */
function pushUndo(): void {
  if (!seedBmp || !committedBmp) return;
  undoStack.push(snapshot());
  if (undoStack.length > UNDO_CAP) undoStack.shift();
  redoStack = [];
}
/** Begin a brush stroke: snapshot for undo. */
export function beginStroke(): void { pushUndo(); }

/** Stamp a sphere (mm) of the current brush at voxel (cx,cy,cz) into the SEEDS layer (erase clears all
    layers), interpolating from a previous voxel so fast drags stay continuous. label 0 erases. The
    display is updated in place; refresh is rAF-throttled to keep dragging responsive. */
export function paintBrush(cx: number, cy: number, cz: number, px: number, py: number, pz: number, label: number,
                           throughAxis: number | null = null): void {
  const d = nv?.drawBitmap;
  const dr = nv?.volumes[0]?.dimsRAS as number[] | undefined;
  const sp = rasSpacing();
  if (!d || !dr || !sp || !seedBmp || !committedBmp || !previewBmp) return;
  const nx = dr[1], ny = dr[2], nz = dr[3], nxny = nx * ny;
  const R = brushRadiusMm();
  if (R <= 0) return;
  const R2 = R * R;
  // Paint a 2-D DISK on the CURRENT slice of the pane being painted (throughAxis = that pane's
  // through-plane RAS axis), NOT a 3-D sphere: the cornea sits at a different depth on every slice,
  // so a stroke must annotate only the slice under the cursor. The user paints sparse slices across
  // panes; smart-fill then interpolates the 3-D label between them. Zeroing the through-axis radius
  // collapses the sphere to a disk on that one slice. (throughAxis null → 3-D sphere, legacy.)
  const rr = [Math.ceil(R / sp[0]), Math.ceil(R / sp[1]), Math.ceil(R / sp[2])];
  if (throughAxis !== null && throughAxis >= 0 && throughAxis <= 2) rr[throughAxis] = 0;
  const rx = rr[0], ry = rr[1], rz = rr[2];
  const hasLocks = lockedLabels.size > 0;
  const erasing = label === 0;
  const s = seedBmp, c = committedBmp, p = previewBmp;
  const stamp = (ox: number, oy: number, oz: number) => {
    for (let k = -rz; k <= rz; k++) { const zz = oz + k; if (zz < 0 || zz >= nz) continue; const kz = k * sp[2], kz2 = kz * kz;
      for (let j = -ry; j <= ry; j++) { const yy = oy + j; if (yy < 0 || yy >= ny) continue; const jy = j * sp[1], jyz2 = jy * jy + kz2;
        if (jyz2 > R2) continue;
        const base = zz * nxny + yy * nx;
        for (let i = -rx; i <= rx; i++) { const xx = ox + i; if (xx < 0 || xx >= nx) continue; const ix = i * sp[0];
          if (ix * ix + jyz2 <= R2) { const idx = base + xx;
            if (hasLocks && lockedLabels.has(d[idx])) continue;       // protect locked labels (as displayed)
            if (erasing) { s[idx] = 0; c[idx] = 0; p[idx] = 0; d[idx] = 0; } // eraser clears everything
            else { s[idx] = label; d[idx] = label; }                  // paint a seed; seed wins in the display
          } } } }
  };
  const dx = cx - px, dy = cy - py, dz = cz - pz;
  const dist = Math.hypot(dx, dy, dz);
  const stepVox = Math.max(1, Math.min(rx, ry, rz));
  const steps = Math.min(96, Math.max(1, Math.ceil(dist / stepVox)));
  for (let st = 0; st <= steps; st++) { const t = st / steps; stamp(Math.round(px + dx * t), Math.round(py + dy * t), Math.round(pz + dz * t)); }
  scheduleRefresh();
}

/** Overwrite every `from`-labelled voxel with `to` in the drawing bitmap. */
function remapLabel(from: number, to: number): void {
  const b = nv?.drawBitmap;
  if (!b) return;
  for (let i = 0; i < b.length; i++) if (b[i] === from) b[i] = to;
}

// ── Smart fill = CPU "Grow from seeds" ───────────────────────────────────────
// niivue's GPU drawGrowCut is unusable here: minutes-long and produces nothing (the render-to-3D-texture
// compute fails on this WebGL — and there's no GPU at all on some target machines). We run a CPU geodesic
// flood in a Web Worker instead (see growcut.worker.ts): every voxel is assigned to the geodesically
// nearest seed among cornea(1)/scar(2)/background(3); background then maps back to 0. Progress-reported.
let growWorker: Worker | null = null;
function getGrowWorker(): Worker {
  if (!growWorker) growWorker = new Worker(new URL("./growcut.worker.ts", import.meta.url), { type: "module" });
  return growWorker;
}

export async function smartFill(onProgress?: (pct: number) => void): Promise<{ ok: boolean; reason?: "no-volume" | "no-seeds" }> {
  if (!nv || !nv.volumes.length || !seedBmp || !previewBmp) return { ok: false, reason: "no-volume" };
  let hasFg = false;
  for (let i = 0; i < seedBmp.length; i++) { const s = seedBmp[i]; if (s === 1 || s === 2) { hasFg = true; break; } }
  if (!hasFg) return { ok: false, reason: "no-seeds" }; // grow only from the user's brushstrokes (seeds)

  // intensity in RAS order (matches drawBitmap), quantised to 8-bit over the display window
  const vol = nv.volumes[0] as unknown as {
    img2RAS: () => ArrayLike<number>; cal_min?: number; cal_max?: number; global_min?: number; global_max?: number;
  };
  const ras = vol.img2RAS();
  let lo = vol.cal_min ?? 0, hi = vol.cal_max ?? 0;
  if (!(hi > lo)) { lo = vol.global_min ?? 0; hi = vol.global_max ?? 1; }
  const scale = hi > lo ? 255 / (hi - lo) : 0;
  const q = new Uint8Array(ras.length);
  for (let i = 0; i < ras.length; i++) { const t = (ras[i] - lo) * scale; q[i] = t <= 0 ? 0 : t >= 255 ? 255 : t | 0; }
  const dr = nv.volumes[0].dimsRAS as number[];
  const seedsCopy = seedBmp.slice(); // ONLY the brushstrokes are sent to growcut (never the previous fill)

  const label = await new Promise<Uint8Array>((resolve, reject) => {
    const worker = getGrowWorker();
    const onMsg = (ev: MessageEvent) => {
      const d = ev.data as { type: string; pct?: number; label?: Uint8Array };
      if (d.type === "progress") onProgress?.(d.pct ?? 0);
      else if (d.type === "done") { worker.removeEventListener("message", onMsg); resolve(d.label!); }
    };
    worker.addEventListener("message", onMsg);
    worker.onerror = (e) => { worker.removeEventListener("message", onMsg); reject(new Error(e.message || "growcut worker failed")); };
    worker.postMessage({ intensity: q, seeds: seedsCopy, nx: dr[1], ny: dr[2], nz: dr[3], spatial: 1 },
                       [q.buffer, seedsCopy.buffer]);
  });

  pushUndo(); // so Undo can revert the fill (restores pre-fill state + clears the preview)
  for (let i = 0; i < previewBmp.length; i++) { const l = label[i]; previewBmp[i] = l === BG_SEED ? 0 : l; }
  previewing = true;
  compose(); // show committed + preview; seeds stay on top
  return { ok: true };
}

/** Confirm the current smart-fill preview → bake it into the committed segmentation; clear seeds+preview
    so the next smart fill starts fresh (respects label locks). Returns false if there's nothing to confirm. */
export function confirmFill(): boolean {
  if (!previewing || !nv || !seedBmp || !committedBmp || !previewBmp) return false;
  pushUndo();
  const hasLocks = lockedLabels.size > 0;
  for (let i = 0; i < committedBmp.length; i++) {
    let v = seedBmp[i] !== 0 ? seedBmp[i] : previewBmp[i]; // seeds (incl. post-fill strokes) then the fill
    if (v === BG_SEED) v = 0;                              // background seeds are never committed
    if (v !== 0 && !(hasLocks && lockedLabels.has(committedBmp[i]))) committedBmp[i] = v;
  }
  seedBmp.fill(0); previewBmp.fill(0); previewing = false;
  compose();
  return true;
}

export function undoDrawing(): void {
  if (!undoStack.length || !seedBmp || !committedBmp || !previewBmp) return;
  redoStack.push(snapshot());
  restoreSnap(undoStack.pop()!);
}
export function redoDrawing(): void {
  if (!redoStack.length || !seedBmp || !committedBmp || !previewBmp) return;
  undoStack.push(snapshot());
  restoreSnap(redoStack.pop()!);
}

// ── Intensity (RAS) cache — for the threshold wand + cursor readout ───────────
function getRasIntensity(): ArrayLike<number> | null {
  if (rasIntensity) return rasIntensity;
  const vol = nv?.volumes[0] as unknown as { img2RAS?: () => ArrayLike<number> } | undefined;
  if (!vol?.img2RAS) return null;
  rasIntensity = vol.img2RAS(); // read-only use; one allocation per volume
  return rasIntensity;
}
/** Raw image intensity at a RAS voxel, or null. */
export function intensityAt(x: number, y: number, z: number): number | null {
  const ras = getRasIntensity();
  const dr = nv?.volumes[0]?.dimsRAS as number[] | undefined;
  if (!ras || !dr) return null;
  const nx = dr[1], ny = dr[2], nz = dr[3];
  if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return null;
  return ras[z * nx * ny + y * nx + x];
}
/** Volume intensity range [min,max] (for mapping the wand's 0..1 threshold to absolute intensity).
    Cached per volume so the live cursor % readout doesn't rescan the volume on every mouse move. */
export function intensityRange(): [number, number] | null {
  if (rasRange) return rasRange;
  const vol = nv?.volumes[0] as unknown as { global_min?: number; global_max?: number } | undefined;
  if (vol && vol.global_min != null && vol.global_max != null && vol.global_max > vol.global_min) { rasRange = [vol.global_min, vol.global_max]; return rasRange; }
  const ras = getRasIntensity();
  if (!ras) return null;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < ras.length; i++) { const v = ras[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  rasRange = hi > lo ? [lo, hi] : null;
  return rasRange;
}
/** Cursor intensity as 0..1 of the volume range (for the wand intensity indicator), or null. */
export function intensityAtNorm(x: number, y: number, z: number): number | null {
  const v = intensityAt(x, y, z); const r = intensityRange();
  if (v == null || !r) return null;
  return Math.max(0, Math.min(1, (v - r[0]) / ((r[1] - r[0]) || 1)));
}

// ── Intensity wand (live preview) ────────────────────────────────────────────
export interface WandOpts {
  mode: "threshold" | "tolerance"; // absolute brightness cutoff, or ±band around the clicked voxel
  threshold01: number;             // threshold mode: 0..1 of the volume range
  tolerance01: number;             // tolerance mode: ± band as a fraction of the volume range
  scope: "2d" | "3d";             // flood within the clicked slice only, or the whole volume
  throughAxis: number | null;      // the clicked pane's through-plane RAS axis (for 2-D scope)
  target: 1 | 2;                   // paint cornea (1) or scar (2)
}
// reusable scratch buffers so the live preview can recompute on every slider tick without re-allocating
// ~33 MB per call (one per volume; reset per run).
let wandVisited: Uint8Array | null = null;
let wandStack: Int32Array | null = null;

/** Flood-fill from a clicked voxel into the PREVIEW layer (not committed) so the user can tune the
    threshold/tolerance and SEE the result live, then Confirm (reuses the smart-fill preview→confirm
    path). 6-connected in 3-D, or 4-connected within the clicked slice in 2-D. SCAR (target 2) is
    confined to committed cornea (it's a sub-region); CORNEA (target 1) is unconfined. Respects label
    locks (locked voxels are flooded through but not painted). Recomputes from scratch each call. */
export function wandPreview(x: number, y: number, z: number, opts: WandOpts): { ok: boolean; reason?: string; count?: number } {
  const ras = getRasIntensity();
  const dr = nv?.volumes[0]?.dimsRAS as number[] | undefined;
  if (!nv || !ras || !committedBmp || !previewBmp || !seedBmp || !dr) return { ok: false, reason: "no-volume" };
  const range = intensityRange();
  if (!range) return { ok: false, reason: "no-volume" };
  const nx = dr[1], ny = dr[2], nz = dr[3], nxny = nx * ny, N = nx * ny * nz;
  if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return { ok: false, reason: "out-of-bounds" };
  const span = (range[1] - range[0]) || 1;
  const seed = z * nxny + y * nx + x;
  const seedVal = ras[seed];
  const c = committedBmp, target = opts.target;
  // scar is a sub-region of cornea → confine target 2 to committed cornea/scar if any cornea exists
  let hasCornea = false;
  if (target === 2) { for (let i = 0; i < N; i++) { if (c[i] === 1) { hasCornea = true; break; } } }
  const confine = (i: number) => target === 2 ? (!hasCornea || c[i] === 1 || c[i] === 2) : true;
  const inMask = opts.mode === "threshold"
    ? (() => { const thr = range[0] + opts.threshold01 * span; return (i: number) => ras[i] >= thr && confine(i); })()
    : (() => { const tol = opts.tolerance01 * span; return (i: number) => Math.abs(ras[i] - seedVal) <= tol && confine(i); })();
  if (!inMask(seed)) return { ok: false, reason: (target === 2 && hasCornea) ? "outside-cornea" : "below-threshold" };
  const twoD = opts.scope === "2d" && opts.throughAxis !== null && opts.throughAxis >= 0 && opts.throughAxis <= 2;
  const ta = opts.throughAxis; // 0=x(step 1), 1=y(step nx), 2=z(step nxny); in 2-D skip steps along it
  if (!wandVisited || wandVisited.length !== N) wandVisited = new Uint8Array(N);
  const visited = wandVisited; visited.fill(0);
  if (!wandStack) wandStack = new Int32Array(1 << 16);
  let stack = wandStack, sp = 0;
  const push = (i: number) => { if (sp === stack.length) { const n = new Int32Array(stack.length * 2); n.set(stack); stack = n; wandStack = n; } stack[sp++] = i; };
  const hasLocks = lockedLabels.size > 0;
  previewBmp.fill(0); // a fresh preview each recompute (never accumulate across slider ticks)
  push(seed); visited[seed] = 1;
  let count = 0;
  while (sp > 0) {
    const i = stack[--sp];
    if (!(hasLocks && lockedLabels.has(c[i]))) { previewBmp[i] = target; count++; }
    const z2 = (i / nxny) | 0, r = i - z2 * nxny, y2 = (r / nx) | 0, x2 = r - y2 * nx;
    if (!twoD || ta !== 0) {
      if (x2 > 0 && !visited[i - 1] && inMask(i - 1)) { visited[i - 1] = 1; push(i - 1); }
      if (x2 < nx - 1 && !visited[i + 1] && inMask(i + 1)) { visited[i + 1] = 1; push(i + 1); }
    }
    if (!twoD || ta !== 1) {
      if (y2 > 0 && !visited[i - nx] && inMask(i - nx)) { visited[i - nx] = 1; push(i - nx); }
      if (y2 < ny - 1 && !visited[i + nx] && inMask(i + nx)) { visited[i + nx] = 1; push(i + nx); }
    }
    if (!twoD || ta !== 2) {
      if (z2 > 0 && !visited[i - nxny] && inMask(i - nxny)) { visited[i - nxny] = 1; push(i - nxny); }
      if (z2 < nz - 1 && !visited[i + nxny] && inMask(i + nxny)) { visited[i + nxny] = 1; push(i + nxny); }
    }
  }
  previewing = true;
  compose();
  return { ok: true, count };
}
/** Discard an un-confirmed wand/smart-fill preview (clears the preview layer, no commit). */
export function clearPreview(): void {
  if (!previewBmp) return;
  previewBmp.fill(0); previewing = false; compose();
}

// ── Load an existing segmentation (open a prior .nii.gz / session GT) into the committed layer ────────
export async function loadSegmentationBytes(bytes: Uint8Array, name: string): Promise<{ ok: boolean; reason?: string }> {
  if (!nv || !nv.drawBitmap || !committedBmp || !seedBmp || !previewBmp) return { ok: false, reason: "no-volume" };
  const url = URL.createObjectURL(new Blob([bytes as unknown as BlobPart]));
  try {
    const img = await NVImage.loadFromUrl({ url, name });
    const ok = nv.loadDrawing(img); // aligns the labelmap to the background; populates nv.drawBitmap
    if (!ok) return { ok: false, reason: "dims-mismatch" };
  } catch (e) {
    return { ok: false, reason: e instanceof Error ? e.message : "load-failed" };
  } finally {
    URL.revokeObjectURL(url);
  }
  pushUndo();
  // niivue put the loaded labelmap in drawBitmap → adopt it as the committed segmentation
  const d = nv.drawBitmap;
  for (let i = 0; i < committedBmp.length; i++) { const v = d[i]; committedBmp[i] = v === 1 || v === 2 ? v : 0; }
  seedBmp.fill(0); previewBmp.fill(0); previewing = false;
  compose();
  return { ok: true };
}

/** LOSSLESS autosave encoder: pack the EDITABLE state (seeds INCLUDING background + committed) into one
    labelmap so a RESTART restores seeds — which is what Smart fill grows from — not just a flattened
    committed result. Without this the disk autosave (exportLabelmapBytes) stripped background seeds
    (3→0) and flattened seeds into committed, so Smart fill after a restart found no seeds and did
    nothing. Codes: 1=committed cornea, 2=committed scar, 3=seed bg, 4=seed cornea, 5=seed scar (a seed
    wins where a voxel has both). */
export async function exportAutosaveBytes(): Promise<Uint8Array | null> {
  if (!nv?.drawBitmap || !seedBmp || !committedBmp) return null;
  const d = nv.drawBitmap, s = seedBmp, c = committedBmp;
  for (let i = 0; i < d.length; i++) {
    const sv = s[i];
    if (sv === BG_SEED) d[i] = 3;
    else if (sv === 1) d[i] = 4;
    else if (sv === 2) d[i] = 5;
    else { const cv = c[i]; d[i] = cv === 1 ? 1 : cv === 2 ? 2 : 0; }
  }
  try { nv.refreshDrawing(true); } catch { /* */ }
  const r = await nv.saveImage({ filename: "", isSaveDrawing: true, volumeByIndex: 0 });
  compose(); // restore the on-screen display (seeds/preview shading)
  return r instanceof Uint8Array ? r : null;
}

/** Decode an autosave written by exportAutosaveBytes back into the SEED + committed layers, so restored
    brushstrokes are SEEDS again (Smart fill works on them). Backward-compatible with the old 0/1/2
    autosave format (1/2 decode as committed cornea/scar; no seeds — same as before). */
export async function restoreAutosaveBytes(bytes: Uint8Array, name: string): Promise<{ ok: boolean; reason?: string }> {
  if (!nv || !nv.drawBitmap || !committedBmp || !seedBmp || !previewBmp) return { ok: false, reason: "no-volume" };
  const url = URL.createObjectURL(new Blob([bytes as unknown as BlobPart]));
  try {
    const img = await NVImage.loadFromUrl({ url, name });
    const ok = nv.loadDrawing(img);
    if (!ok) return { ok: false, reason: "dims-mismatch" };
  } catch (e) {
    return { ok: false, reason: e instanceof Error ? e.message : "load-failed" };
  } finally {
    URL.revokeObjectURL(url);
  }
  const d = nv.drawBitmap;
  for (let i = 0; i < d.length; i++) {
    const v = d[i];
    if (v === 1) { committedBmp[i] = 1; seedBmp[i] = 0; }
    else if (v === 2) { committedBmp[i] = 2; seedBmp[i] = 0; }
    else if (v === 3) { seedBmp[i] = BG_SEED; committedBmp[i] = 0; }
    else if (v === 4) { seedBmp[i] = 1; committedBmp[i] = 0; }
    else if (v === 5) { seedBmp[i] = 2; committedBmp[i] = 0; }
    else { seedBmp[i] = 0; committedBmp[i] = 0; }
  }
  previewBmp.fill(0); previewing = false;
  compose();
  return { ok: true };
}

// ── Annotation state get/set — for LOSSLESS volume swapping + autosave restore (#5) ──────────────
/** Snapshot the full editable state (seeds + committed + UNCONFIRMED preview) so swapping away from a
    volume and back restores EXACTLY what was on screen, including an un-confirmed smart fill. */
export function getAnnotationState(): { seed: Uint8Array; committed: Uint8Array; preview: Uint8Array; previewing: boolean } | null {
  if (!seedBmp || !committedBmp || !previewBmp) return null;
  return { seed: seedBmp.slice(), committed: committedBmp.slice(), preview: previewBmp.slice(), previewing };
}
/** Restore a snapshot onto the CURRENT volume's drawing (must be same dims). Resets undo history. */
export function setAnnotationState(st: { seed: Uint8Array; committed: Uint8Array; preview: Uint8Array; previewing: boolean }): boolean {
  if (!seedBmp || !committedBmp || !previewBmp) return false;
  if (st.seed.length !== seedBmp.length) return false; // dims mismatch — don't apply a wrong-sized state
  seedBmp.set(st.seed); committedBmp.set(st.committed); previewBmp.set(st.preview); previewing = !!st.previewing;
  undoStack = []; redoStack = [];
  compose();
  return true;
}
/** True if the current drawing has ANY paint (committed, seed, or preview) — gates autosave. */
export function hasPaint(): boolean {
  if (!seedBmp || !committedBmp || !previewBmp) return false;
  const s = seedBmp, c = committedBmp, p = previewBmp;
  for (let i = 0; i < c.length; i++) if (c[i] || s[i] || p[i]) return true;
  return false;
}

// ── 2-D zoom / pan ────────────────────────────────────────────────────────────
/** Zoom toward the RED CROSSHAIR, not the slice centre (#4). niivue zooms about the slice centre, so
    we pan the crosshair's mm position to the centre and zoom there — the region under the crosshair
    magnifies in place. pan2Dxyzmm = [panXmm, panYmm, panZmm, zoom]; pan moves the volume vs the centre,
    so pan = centre_mm − crosshair_mm. (One global zoom for all 2-D panes — niivue has no per-tile zoom.) */
export function zoomBy(factor: number): void {
  if (!nv) return;
  const p = nv.scene.pan2Dxyzmm as unknown as number[];
  const z = Math.max(1, Math.min(8, (p[3] || 1) * factor));
  let pan: [number, number, number] = [p[0], p[1], p[2]];
  try {
    const cp = nv.scene?.crosshairPos as Float32Array | undefined;
    const f2m = (nv as unknown as { frac2mm?: (f: number[]) => Float32Array }).frac2mm;
    if (cp && typeof f2m === "function") {
      const cm = f2m.call(nv, [cp[0], cp[1], cp[2]]);
      const ctr = f2m.call(nv, [0.5, 0.5, 0.5]);
      pan = [ctr[0] - cm[0], ctr[1] - cm[1], ctr[2] - cm[2]];
    }
  } catch { /* keep current pan if frac2mm is unavailable */ }
  try { nv.setPan2Dxyzmm([pan[0], pan[1], pan[2], z]); } catch { /* */ }
}
export function resetView(): void { if (nv) { try { nv.setPan2Dxyzmm([0, 0, 0, 1]); } catch { /* */ } } }

/** Reset to a blank drawing (discard all paint on the current volume). */
export function clearDrawing(): void {
  if (!nv || !seedBmp || !committedBmp || !previewBmp) return;
  pushUndo();
  seedBmp.fill(0); committedBmp.fill(0); previewBmp.fill(0); previewing = false;
  compose();
}

// ── Export ──────────────────────────────────────────────────────────────────
/** The drawing as NIfTI bytes (values 0/1/2, co-registered to the volume). Exports the full displayed
    segmentation (committed + preview + seeds); background seeds (3) are stripped. */
export async function exportLabelmapBytes(): Promise<Uint8Array | null> {
  if (!nv?.drawBitmap) return null;
  // the saved GT is strictly 0/1/2: background seeds → 0, preview shades → their real label
  remapLabel(BG_SEED, 0);
  remapLabel(PREVIEW_CORNEA, 1);
  remapLabel(PREVIEW_SCAR, 2);
  try { nv.refreshDrawing(true); } catch { /* best-effort */ }
  const r = await nv.saveImage({ filename: "", isSaveDrawing: true, volumeByIndex: 0 });
  compose(); // restore the on-screen display (re-adds bg seeds + preview shading)
  return r instanceof Uint8Array ? r : null;
}

/** Voxel counts per painted label + voxel volume (mm³) from the loaded volume header. */
export function drawStats(): { cornea: number; scar: number; mm3: number; spacing: [number, number, number] } {
  const out = { cornea: 0, scar: 0, mm3: 0, spacing: [0, 0, 0] as [number, number, number] };
  if (!nv || !nv.drawBitmap) return out;
  const b = nv.drawBitmap;
  for (let i = 0; i < b.length; i++) { const v = b[i]; if (v === 1 || v === PREVIEW_CORNEA) out.cornea++; else if (v === 2 || v === PREVIEW_SCAR) out.scar++; }
  const pd = nv.volumes[0]?.hdr?.pixDims;
  if (pd && pd.length >= 4) {
    out.spacing = [pd[1], pd[2], pd[3]];
    out.mm3 = Math.abs(pd[1] * pd[2] * pd[3]);
  }
  return out;
}
