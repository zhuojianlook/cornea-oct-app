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

let _canvas: HTMLCanvasElement | null = null;
let _onRestored: (() => void) | null = null;
let _listenersBound = false;
let _contextLost = false;

/** True while the WebGL context is lost (between webglcontextlost and webglcontextrestored). */
export function contextLost(): boolean { return _contextLost; }

/** Register a callback fired AFTER a lost WebGL context is restored + niivue rebuilt, so the host can
 *  reload the current volume/overlay (niivue's GPU state is wiped by a context loss). null to clear. */
export function onContextRestored(cb: (() => void) | null): void { _onRestored = cb; }

/** Build (or rebuild) the Niivue instance on `canvas`. Shared by attach() and context-loss recovery. */
function _createNv(canvas: HTMLCanvasElement): Niivue | null {
  try {
    nv = new Niivue({
      backColor: [0.11, 0.11, 0.12, 1],
      show3Dcrosshair: true,
      isColorbar: false,
      dragAndDropEnabled: false,
    });
    nv.attachToCanvas(canvas);
    nv.setSliceType(SLICE.multi);
    // Live-update the 2-D drawing overlay whenever niivue changes the displayed location — which includes
    // every point of a native pen DRAG (niivue calls createOnLocationChange after each paint step — verified
    // in the dist PEN branch) and slice scroll/scrub — so brush strokes appear immediately even though
    // niivue's WebGL draw tile is blank here. rAF-coalesced (scheduleOverlay) so a fast drag fires at most
    // one overlay render per frame (the pixel loop is heavy).
    (nv as unknown as { onLocationChange: (loc: unknown) => void }).onLocationChange = () => scheduleOverlay();
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
  _canvas = canvas;
  const inst = _createNv(canvas);
  // WebGL CONTEXT-LOSS RECOVERY: WebKitGTK/NVIDIA can drop the WebGL2 context under pressure (many viewers
  // opened, a GPU reset, a resize storm). Without recovery the singleton niivue holds a DEAD context and the
  // canvas stays black forever — switching scans just reloads into the dead context. preventDefault() on
  // "lost" is REQUIRED for the browser to later fire "restored"; on "restored" we rebuild niivue and ask the
  // host (VolumeCanvas) to reload the current volume/overlay, since the GPU state was wiped.
  if (!_listenersBound) {
    _listenersBound = true;
    canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); _contextLost = true; }, false);
    canvas.addEventListener("webglcontextrestored", () => {
      nv = null;
      if (_canvas) _createNv(_canvas);
      _contextLost = false;
      try { _onRestored?.(); } catch { /* host reload is best-effort */ }
    }, false);
  }
  return inst;
}

/** Load (or replace) the grayscale base volume. */
export async function loadVolume(url: string): Promise<void> {
  if (!nv) throw new Error("Niivue not attached");
  await nv.loadVolumes([{ url, colormap: "gray" }]);
}

export function setView(view: ViewName): void {
  if (!nv) return;
  nv.setSliceType(SLICE[view]);
  // tile layout changed → re-render the 2-D overlay now + after the WebKitGTK layout settles.
  renderDrawOverlay(); requestAnimationFrame(renderDrawOverlay); setTimeout(renderDrawOverlay, 120);
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
  renderDrawOverlay();   // the displayed slice changed → re-render the overlay for the new slice
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
// Label 4 = AUTO-flooded background (fillBackgroundSeed) — rendered identically to user background (grey),
// but kept DISTINCT internally so smart-fill can re-grow untouched auto-bg yet NEVER re-grow background the
// user explicitly painted (label 2). Collapsed to 2 on export (see exportDrawing).
const AUTO_BG = 4;
const DRAW_CMAP = {
  R: [0, 70, 142, 235, 142],
  G: [0, 160, 142, 95, 142],
  B: [0, 235, 147, 95, 147],
  A: [0, 255, 255, 255, 255],
  I: [0, 1, 2, 3, 4],
  labels: ["", "cornea", "background", "scar", "background"],
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
  renderDrawOverlay();   // render the just-loaded labels on the 2-D overlay (WebGL draw tile is blank here)
}

/** End correction: stop the pen AND clear the drawing bitmap so it can't linger over the committed
 *  overlay or leak into other stages/cases (a stale drawBitmap renders even when drawing is disabled). */
export function endDrawing(): void {
  if (!nv) return;
  nv.setDrawingEnabled(false);
  try { nv.closeDrawing(); } catch { /* no-op if no drawing */ }
  nv.drawScene();
  renderDrawOverlay();   // drawBitmap is gone now → clears the 2-D overlay
}

/** Undo the last brush stroke / smart fill (niivue keeps a drawing undo stack). */
export function undoDrawing(): void {
  if (!nv) return;
  let undid = false;
  try { nv.drawUndo(); undid = true; } catch { /* nothing to undo */ }
  if (undid) { nv.drawScene(); renderDrawOverlay(); }   // reflect the undone stroke on the 2-D overlay
}

/** Pen label: 0 erase, 1 cornea, 2 background, 3 scar. `filled`=true auto-fills a closed outline
 *  (draw a loop around a region → the enclosed area is painted), so a whole patch is one stroke. */
export function setPen(label: number, filled = false): void {
  if (!nv) return;
  nv.setDrawingEnabled(true);
  nv.setPenValue(label, filled);
}

/** Brush thickness (voxels). */
let appPenSize = 3;
export function setPenSize(size: number): void {
  appPenSize = Math.max(1, Math.round(size));
  if (nv) nv.opts.penSize = appPenSize;
}

// #1 — the on-screen brush cursor must match the ACTUAL painted footprint (penSize VOXELS), not a fixed
// px guess. The brush is a sphere of radius brushRadiusMm in physical space; niivue renders slices
// isotropically in mm, so its on-screen diameter = 2·radiusMm · (tile px per mm). (Ported from the annotator.)
function rasSpacing(): [number, number, number] | null {
  const pd = (nv?.volumes[0] as unknown as { pixDimsRAS?: number[] } | undefined)?.pixDimsRAS;
  if (pd && pd.length >= 4) return [Math.abs(pd[1]) || 1, Math.abs(pd[2]) || 1, Math.abs(pd[3]) || 1];
  return null;
}
function brushRadiusMm(): number {
  const sp = rasSpacing();
  if (!sp) return 0;
  const s = [sp[0], sp[1], sp[2]].sort((a, b) => a - b);
  return (appPenSize / 2) * s[1];   // penSize = diameter in median-spacing voxels → physically round in every view
}
/** On-screen diameter (CSS px) of the brush at cursor (xCss,yCss), or null when not over a 2-D tile. */
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

// Smart fill = CPU "Grow from seeds" in a WEB WORKER (ported from the annotator). niivue's GPU drawGrowCut
// is unusably slow / hangs on the WebKitGTK + NVIDIA desktop stack (128 iterations × per-slice readPixels);
// the worker runs a multi-source geodesic flood (Dijkstra/Dial buckets, O(N), off the main thread, with
// progress) so the UI stays responsive. Seeds = the current drawBitmap labels (1 cornea, 2 USER background,
// 3 scar, 4 AUTO-flooded background); every unlabelled voxel is assigned the geodesically-nearest seed's
// label. The 4-vs-2 distinction (fillBackgroundSeed floods 4) lets smart fill re-grow untouched auto-bg into
// the cornea rim while treating user-painted background (2) as an absolute, never-re-grown barrier.

/** Fill every UNPAINTED (0) voxel in the drawing with the AUTO background seed (label 4, rendered grey) —
 *  used after SAM2 so the cornea-vet step shows a COMPLETE cornea/background partition to correct (and gives
 *  Smart fill a background seed). On save labels 2 and 4 → canonical background 0, so the labelmap is unchanged. */
export function fillBackgroundSeed(): void {
  if (!nv || !nv.drawBitmap) return;
  const b = nv.drawBitmap as Uint8Array;
  // Flood unlabelled → AUTO_BG (4), NOT plain background (2). Smart fill may re-grow auto-bg into the cornea
  // rim, but background the user later PAINTS (label 2) is an absolute barrier it can never re-grow.
  for (let i = 0; i < b.length; i++) if (b[i] === 0) b[i] = AUTO_BG;
  try { nv.refreshDrawing(true); } catch { /* */ }
  nv.drawScene();
  renderDrawOverlay();
}

/** Voxels within `r` (6-connected) of any cornea SEED (label 1) — the cornea RIM. Used to confine smart-fill
 *  re-growth to the corneal band (curvature) so it can't bloat into disconnected bright ARTIFACTS. */
function _dilatedCorneaMask(seeds: Uint8Array, nx: number, ny: number, nz: number, r: number): Uint8Array {
  const N = nx * ny * nz, nxny = nx * ny;
  let cur = new Uint8Array(N);
  for (let i = 0; i < N; i++) if (seeds[i] === 1) cur[i] = 1;
  for (let pass = 0; pass < r; pass++) {
    const next = cur.slice();
    for (let v = 0; v < N; v++) {
      if (cur[v]) continue;
      const z = (v / nxny) | 0, rem = v - z * nxny, y = (rem / nx) | 0, x = rem - y * nx;
      if ((x > 0 && cur[v - 1]) || (x < nx - 1 && cur[v + 1]) ||
          (y > 0 && cur[v - nx]) || (y < ny - 1 && cur[v + nx]) ||
          (z > 0 && cur[v - nxny]) || (z < nz - 1 && cur[v + nxny])) next[v] = 1;
    }
    cur = next;
  }
  return cur;
}

/** Smart 3-D fill via the CPU worker. Resolves {ok, reason}; reports 0..100 via onProgress. Pushes a niivue
 *  undo bitmap first so the fill is undoable. Returns reason:"no-seeds" if no cornea/scar seed exists. */
export async function smartFill(onProgress?: (pct: number) => void): Promise<{ ok: boolean; reason?: "no-volume" | "no-seeds" | "size-mismatch" }> {
  if (!nv || !nv.volumes.length || !nv.drawBitmap) return { ok: false, reason: "no-volume" };
  const dr = nv.volumes[0].dimsRAS as number[] | undefined;
  if (!dr || dr.length < 4) return { ok: false, reason: "no-volume" };
  const N = dr[1] * dr[2] * dr[3];
  const draw = nv.drawBitmap as Uint8Array;
  // intensity in RAS order (matches drawBitmap), quantised to 8-bit over the display window.
  const vol = nv.volumes[0] as unknown as {
    img2RAS: () => ArrayLike<number>; cal_min?: number; cal_max?: number; global_min?: number; global_max?: number;
  };
  const ras = vol.img2RAS();
  // GUARD: seeds (drawBitmap), intensity (img2RAS) and the worker's output must all be exactly N voxels —
  // a mismatch (e.g. a stale drawing from a differently-sized volume) would corrupt drawBitmap on set().
  if (draw.length !== N || ras.length !== N) return { ok: false, reason: "size-mismatch" };
  const seeds = draw.slice();
  // intensity quantised to 8-bit over the display window — needed BOTH for the flood and the cornea-vet dark cut.
  let lo = vol.cal_min ?? 0, hi = vol.cal_max ?? 0;
  if (!(hi > lo)) { lo = vol.global_min ?? 0; hi = vol.global_max ?? 1; }
  const scale = hi > lo ? 255 / (hi - lo) : 0;   // flat volume → 0 = spatial-only flood (no crash)
  const q = new Uint8Array(ras.length);
  for (let i = 0; i < ras.length; i++) { const t = (ras[i] - lo) * scale; q[i] = t <= 0 ? 0 : t >= 255 ? 255 : t | 0; }
  // ── Cornea-vet propagation ────────────────────────────────────────────────────────────────────────────
  // After SAM2 the WHOLE volume is labelled (SAM2 cornea = 1, every other voxel flooded to background = 2 by
  // fillBackgroundSeed). The geodesic flood only fills UNLABELLED (0) voxels, so with no 0-voxels Smart fill
  // would be a no-op: a cornea stroke the user painted over a region SAM2 wrongly called background would NOT
  // spread to neighbouring slices, forcing slice-by-slice correction. Fix: SOFTEN the BRIGHT untouched
  // auto-background (still 2 AND was 2 in the baseline) back to 0 so the flood can re-grow it, while keeping
  // every USER edit and the SAM2 cornea as HARD seeds. A background stroke the user painted over baseline-cornea
  // (seeds 2, baseline 1) is NOT softened → stays a hard background seed, so over-segmentation removals also
  // propagate. The "definitely background" level is data-driven = the SAM2 cornea's own 10th-percentile
  // intensity: auto-background DARKER than (almost) all cornea is air/shadow → kept as a HARD background seed
  // (gives the flood a well-posed background source and stops dark regions bloating to cornea); brighter
  // auto-background is the candidate the correction grows into. NO geometric-face anchors — the cornea can reach
  // the scan edge, so anchoring faces would CLIP it; the abundant dark air/shadow provides the seeds instead.
  // Net: painted cornea grows from BOTH SAM2's region and the user's stroke into intensity-similar tissue across
  // slices, bounded by intensity edges and the dark seeds; cornea only grows (SAM2 stays hard) unless removed.
  {
    // darkCut = the cornea's 10th-percentile intensity (data-driven "definitely background" level).
    const hist = new Int32Array(256);
    let cN = 0;
    for (let i = 0; i < seeds.length; i++) if (seeds[i] === 1) { hist[q[i]]++; cN++; }
    let darkCut = 0;
    if (cN > 0) { const target = cN * 0.10; let acc = 0; for (let b = 0; b < 256; b++) { acc += hist[b]; if (acc >= target) { darkCut = b; break; } } }
    // Re-grow ONLY AUTO-flooded background (AUTO_BG=4) that is bright AND on the cornea RIM (within a few voxels
    // of a cornea seed) → it becomes re-growable so the fill recovers the corneal band SAM2 under-segmented
    // (following the curvature). USER-painted background (label 2) is NEVER softened — an absolute barrier the
    // fill can't cross, so painting away an artifact STICKS even when it abuts the cornea. Far/dark auto-bg
    // stays a hard background seed (keeps the flood well-posed). dr=[n,nx,ny,nz].
    const near = _dilatedCorneaMask(seeds, dr[1], dr[2], dr[3], 3);
    for (let i = 0; i < seeds.length; i++) {
      if (seeds[i] === AUTO_BG && q[i] >= darkCut && near[i]) seeds[i] = 0;   // bright auto-bg on the rim → re-growable
    }
  }
  let hasFg = false;
  for (let i = 0; i < seeds.length; i++) { const s = seeds[i]; if (s === 1 || s === 3) { hasFg = true; break; } }
  if (!hasFg) return { ok: false, reason: "no-seeds" };   // need ≥1 cornea/scar seed to grow from
  // A FRESH worker per run, terminated in finally — no listener accumulation / no reuse of a hung worker.
  const worker = new Worker(new URL("./growcut.worker.ts", import.meta.url), { type: "module" });
  try {
    const label = await new Promise<Uint8Array>((resolve, reject) => {
      worker.onmessage = (ev: MessageEvent) => {
        const d = ev.data as { type: string; pct?: number; label?: Uint8Array };
        if (d.type === "progress") onProgress?.(d.pct ?? 0);
        else if (d.type === "done") {
          if (!d.label || d.label.length !== N) reject(new Error(`growcut returned ${d.label?.length ?? 0} voxels, expected ${N}`));
          else resolve(d.label);
        }
      };
      worker.onerror = (e) => reject(new Error(e.message || "growcut worker failed"));
      worker.postMessage({ intensity: q, seeds, nx: dr[1], ny: dr[2], nz: dr[3], spatial: 1 }, [q.buffer, seeds.buffer]);
    });
    try { nv.drawAddUndoBitmap(); } catch { /* undo push best-effort */ }
    draw.set(label);
    try { nv.refreshDrawing(true); } catch (e) { console.error("[smartFill] refreshDrawing failed", e); }
    nv.drawScene();
    renderDrawOverlay();
    return { ok: true };
  } finally {
    worker.terminate();
  }
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

// ── WebGL-independent 2-D drawing OVERLAY (the annotator's fix, ported) ──────────────────────────────
// On the desktop NVIDIA/WebKitGTK stack niivue's WebGL DRAW layer never reaches the 2-D slice tiles, so
// brush strokes are recorded in drawBitmap but INVISIBLE (the user sees only the crosshair move). We
// bypass that by rendering the drawing OURSELVES on a plain 2-D <canvas> over the niivue canvas
// (pointer-events:none): inverse-sample each device pixel → texture-frac via niivue's OWN
// screenXY2TextureFrac (exact per-tile affine from 3 inset points; no orientation maths) → drawBitmap
// label → colour. niivue's 3-D render + volume overlays still use WebGL. RULE (memory): when niivue's
// 2-D-tile draw rendering fails on a driver, bypass it with this overlay — don't keep poking refreshDrawing.
let overlayCanvas: HTMLCanvasElement | null = null;
export function setOverlayCanvas(c: HTMLCanvasElement | null): void { overlayCanvas = c; if (c) renderDrawOverlay(); }
// drawBitmap pen value → on-screen RGBA (translucent so the anatomy shows through; further scaled by the
// live drawOpacity). 1 cornea (blue), 2 background (grey), 3 scar (red) — matches the pen/segmentation colours.
const OVERLAY_RGBA: Record<number, [number, number, number, number]> = {
  1: [26, 178, 255, 150],
  2: [142, 142, 147, 110],
  3: [255, 69, 58, 160],
  4: [142, 142, 147, 110],   // auto-flooded background — same grey as user background (label 2)
};
let _overlayScheduled = false;
/** rAF-coalesced overlay re-render (one per frame) — used for the high-frequency paint/location stream. */
export function scheduleOverlay(): void {
  if (_overlayScheduled) return;
  _overlayScheduled = true;
  requestAnimationFrame(() => { _overlayScheduled = false; renderDrawOverlay(); });
}
export function renderDrawOverlay(): void {
  const cv = overlayCanvas;
  if (!nv || !cv) return;
  const d = nv.drawBitmap as Uint8Array | undefined;
  const dr = nv.volumes?.[0]?.dimsRAS as number[] | undefined;
  const gl = nv.gl as WebGL2RenderingContext | undefined;
  const slices = (nv as unknown as { screenSlices?: Array<{ axCorSag: number; leftTopWidthHeight: number[] }> }).screenSlices;
  const f2f = (nv as unknown as { screenXY2TextureFrac?: (x: number, y: number, t: number) => number[] | null }).screenXY2TextureFrac;
  const ctx = cv.getContext("2d");
  if (!gl || !ctx) return;
  const W = gl.drawingBufferWidth, H = gl.drawingBufferHeight;
  if (cv.width !== W) cv.width = W;
  if (cv.height !== H) cv.height = H;
  ctx.clearRect(0, 0, W, H);
  if (!d || !dr || !slices || typeof f2f !== "function") return;   // nothing being edited → overlay clears
  const nx = dr[1], ny = dr[2], nz = dr[3], nxny = nx * ny;
  const nvAny = nv as unknown as { drawOpacity?: number; opts?: { drawOpacity?: number } };
  const op = Math.max(0.15, Math.min(1, nvAny.drawOpacity ?? nvAny.opts?.drawOpacity ?? 0.5));
  const img = ctx.createImageData(W, H);
  const data = img.data;
  let any = false;
  for (let ti = 0; ti < slices.length; ti++) {
    const s = slices[ti];
    if (s.axCorSag > 2) continue;                       // skip the 3-D render tile
    const [lx, ly, lw, lh] = s.leftTopWidthHeight;
    if (lw <= 2 || lh <= 2) continue;
    // screen→frac is AFFINE per tile; derive it from 3 INSET points (inset dodges letterbox edge-clamp).
    const ox = Math.min(2, lw / 4), oy = Math.min(2, lh / 4);
    const p0 = f2f.call(nv, lx + ox, ly + oy, ti);
    const p1 = f2f.call(nv, lx + lw - ox, ly + oy, ti);
    const p2 = f2f.call(nv, lx + ox, ly + lh - oy, ti);
    if (!p0 || !p1 || !p2) continue;
    const dx = (lw - 2 * ox), dy = (lh - 2 * oy);
    const A = [0, 0, 0], B = [0, 0, 0], C = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      A[k] = (p1[k] - p0[k]) / dx;
      B[k] = (p2[k] - p0[k]) / dy;
      C[k] = p0[k] - A[k] * (lx + ox) - B[k] * (ly + oy);
    }
    const x0 = Math.max(0, Math.floor(lx)), x1 = Math.min(W, Math.ceil(lx + lw));
    const y0 = Math.max(0, Math.floor(ly)), y1 = Math.min(H, Math.ceil(ly + lh));
    for (let py = y0; py < y1; py++) {
      const fy0 = B[0] * py + C[0], fy1 = B[1] * py + C[1], fy2 = B[2] * py + C[2];
      const rowOff = py * W;
      for (let px = x0; px < x1; px++) {
        const fx = A[0] * px + fy0, fy = A[1] * px + fy1, fz = A[2] * px + fy2;
        const ix = Math.round(fx * nx - 0.5); if (ix < 0 || ix >= nx) continue;
        const iy = Math.round(fy * ny - 0.5); if (iy < 0 || iy >= ny) continue;
        const iz = Math.round(fz * nz - 0.5); if (iz < 0 || iz >= nz) continue;
        const lab = d[iz * nxny + iy * nx + ix];
        if (!lab) continue;
        const c = OVERLAY_RGBA[lab]; if (!c) continue;
        const o = (rowOff + px) * 4;
        data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2]; data[o + 3] = Math.round(c[3] * op); any = true;
      }
    }
  }
  if (any) ctx.putImageData(img, 0, 0);
}

export function setDrawOpacity(opacity: number): void {
  if (!nv) return;
  nv.drawOpacity = opacity;
  nv.drawScene();
  renderDrawOverlay();   // the overlay alpha tracks drawOpacity
}

/** Export the edited drawing bitmap as NIfTI bytes. */
export async function exportDrawing(): Promise<Uint8Array | null> {
  if (!nv) return null;
  // Collapse the auto-background sentinel (4) → plain background (2) so the saved labelmap is canonical
  // (the backend maps drawing label 2 → 0 = background).
  const d = nv.drawBitmap as Uint8Array | undefined;
  if (d) for (let i = 0; i < d.length; i++) if (d[i] === AUTO_BG) d[i] = 2;
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
