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
  // capture the auto display window for the brightness/contrast controls (#3)
  const v0 = nv.volumes[0] as unknown as { cal_min?: number; cal_max?: number; global_min?: number; global_max?: number };
  baseWinLo = v0.cal_min ?? v0.global_min ?? 0;
  baseWinHi = v0.cal_max ?? v0.global_max ?? (baseWinLo + 1);
  nv.createEmptyDrawing();
  try { nv.setDrawColormap(DRAW_CMAP as unknown as string); } catch { /* default LUT */ }
  nv.setDrawOpacity(drawOpacity);
  nv.setDrawingEnabled(false); // custom sphere brush owns painting; niivue navigation still sets crosshair
  nv.opts.penSize = 1;
  // Three-layer drawing model (#smartfill-preview): seeds = the user's brushstrokes (the ONLY input to
  // smart fill), committed = confirmed segmentation, preview = last smart-fill result (recomputed each
  // run). The displayed drawBitmap is composed = seeds ▸ preview ▸ committed. Confirm bakes preview→committed.
  if (nv.drawBitmap) { const n = nv.drawBitmap.length;
    seedBmp = new Uint8Array(n); committedBmp = new Uint8Array(n); previewBmp = new Uint8Array(n);
    previewing = false; undoStack = []; }
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

// ── Slice navigation (per-view scrollbars, #1) ───────────────────────────────
let strokeActive = false;
let sliceListener: ((vox: [number, number, number]) => void) | null = null;
/** Register a callback fired on scroll/navigation with the current [x,y,z] voxel slice indices. */
export function setSliceListener(fn: ((vox: [number, number, number]) => void) | null): void { sliceListener = fn; }
/** Mark a paint stroke active so the slice readout doesn't follow the brush mid-stroke. */
export function setStroke(active: boolean): void { strokeActive = active; }

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

// ── Layered drawing: seeds (brushstrokes) ▸ preview (last smart fill) ▸ committed (confirmed) ────────
// Smart fill must grow ONLY from the user's brushstrokes (seeds), never from its own output — so the
// fill is a recomputable PREVIEW. "Confirm" bakes the preview into the committed segmentation. The
// displayed/exported drawBitmap is composed seeds ▸ preview ▸ committed (first non-zero wins).
let seedBmp: Uint8Array | null = null;
let committedBmp: Uint8Array | null = null;
let previewBmp: Uint8Array | null = null;
let previewing = false;
let undoStack: Array<{ s: Uint8Array; c: Uint8Array }> = [];
const UNDO_CAP = 4;

export function isPreviewing(): boolean { return previewing; }

/** Recompose the displayed/exported drawBitmap from the three layers and push it to the GPU. */
function compose(): void {
  if (!nv?.drawBitmap || !seedBmp || !committedBmp || !previewBmp) return;
  const d = nv.drawBitmap, s = seedBmp, p = previewBmp, c = committedBmp;
  for (let i = 0; i < d.length; i++) { const sv = s[i]; d[i] = sv !== 0 ? sv : (p[i] !== 0 ? p[i] : c[i]); }
  try { nv.refreshDrawing(true); } catch { /* */ }
  nv.drawScene();
}

/** Snapshot seeds+committed for undo — call BEFORE a mutating action (stroke, smart fill, confirm, clear). */
function pushUndo(): void {
  if (!seedBmp || !committedBmp) return;
  undoStack.push({ s: seedBmp.slice(), c: committedBmp.slice() });
  if (undoStack.length > UNDO_CAP) undoStack.shift();
}
/** Begin a brush stroke: snapshot for undo. */
export function beginStroke(): void { pushUndo(); }

/** Stamp a sphere (mm) of the current brush at voxel (cx,cy,cz) into the SEEDS layer (erase clears all
    layers), interpolating from a previous voxel so fast drags stay continuous. label 0 erases. The
    display is updated in place; refresh is rAF-throttled to keep dragging responsive. */
export function paintBrush(cx: number, cy: number, cz: number, px: number, py: number, pz: number, label: number): void {
  const d = nv?.drawBitmap;
  const dr = nv?.volumes[0]?.dimsRAS as number[] | undefined;
  const sp = rasSpacing();
  if (!d || !dr || !sp || !seedBmp || !committedBmp || !previewBmp) return;
  const nx = dr[1], ny = dr[2], nz = dr[3], nxny = nx * ny;
  const R = brushRadiusMm();
  if (R <= 0) return;
  const R2 = R * R;
  const rx = Math.ceil(R / sp[0]), ry = Math.ceil(R / sp[1]), rz = Math.ceil(R / sp[2]);
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
  const e = undoStack.pop();
  if (!e || !seedBmp || !committedBmp || !previewBmp) return;
  seedBmp.set(e.s); committedBmp.set(e.c); previewBmp.fill(0); previewing = false;
  compose();
}

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
  remapLabel(BG_SEED, 0); // strip background seeds from the saved GT (display is re-composed afterwards)
  try { nv.refreshDrawing(true); } catch { /* best-effort */ }
  const r = await nv.saveImage({ filename: "", isSaveDrawing: true, volumeByIndex: 0 });
  compose(); // restore the on-screen display (re-adds bg seeds + layer ordering)
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
