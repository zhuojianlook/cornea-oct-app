/* ──────────────────────────────────────────────────────────
   DEBUG 3-D viewport controller — a SEPARATE, INDEPENDENT Niivue instance.

   The Debug tab's interactive 3-D viewport (replicate-alignment overlap / disagreement) renders the
   cropped isotropic volumes LIVE on the GPU via niivue's 3-D volume-render path (SLICE_TYPE.RENDER) —
   free mouse rotate / zoom / pan, at the fine registration grid rather than the old pre-baked low-res
   turntable PNGs.

   This module owns its OWN niivue instance + module state. It NEVER touches nvController's singleton
   (the user-loaded case in VolumeCanvas) — do not import from nvController here. Lifecycle: create()
   lazily when a 3-D mode is entered, destroy() when leaving (frees the WebGL context) so at most ONE
   extra WebGL context is ever alive — which matters on the fragile WebKitGTK desktop stack. Only the
   3-D render path is used here (known to work on WebKitGTK, unlike the 2-D-tile draw path).
   ────────────────────────────────────────────────────────── */

import { Niivue, SLICE_TYPE, DRAG_MODE } from "@niivue/niivue";
import { releaseVolumes, destroyNiivue, withVolumeNames } from "./nvRelease";

// niivue does not export the ImageFromUrlOptions type by name — derive it from loadVolumes' signature.
type LoadVolumeOpts = Parameters<Niivue["loadVolumes"]>[0][number];

export type DebugMode3d = "overlap" | "disagreement" | "consensus";

export interface DebugContent {
  mode: DebugMode3d;
  fixedUrl: string; // pair modes only; "" for consensus (unused)
  movingUrl?: string | null; // overlap: the aligned-moving volume (green)
  disagreeUrl?: string | null; // disagreement: |fixed − moving| scalar (hot)
  disagreeMax?: number | null; // shared raw-scale cal_max for the hot volume (identity-derived 99th pct)
  consensusUrl?: string | null; // consensus: the single min/excess RGBA volume (colours baked in)
  window?: [number, number] | null; // shared fixed intensity window [lo, hi]
}

// FIXED = magenta, MOVING = green (matches the 2-D overlay + the panel legend #e879f9 / #4ade80). The
// green colormap ships with niivue; magenta does not, so we register it (mirror of niivue's green.json,
// same half-alpha ramp so agreement composites toward white). The disagreement "hot" ramp is registered
// with an alpha that RISES with intensity, so agreement is transparent and disagreement glows — matching
// the panel's HOT_SWATCH legend (dark-red → red → orange → yellow → white).
const CMAP_MAGENTA = "dbgMagenta";
const CMAP_HOT = "dbgHot";

// First-entry camera: a 3/4 view so the cornea dome reads immediately (the user then rotates freely).
const DEFAULT_CAM = { azimuth: 120, elevation: 15, scale: 1.0 };

interface Cam {
  azimuth: number;
  elevation: number;
  scale: number;
}

let nv: Niivue | null = null;
let _canvas: HTMLCanvasElement | null = null;
let _contextLost = false;
let _webglError: string | null = null;
// Camera pose is scene state (not per-volume), so it survives a volume swap automatically; we still
// snapshot it so it survives a DESTROY (re-entering 3-D restores the last pose) and a context-loss rebuild.
let _savedCam: Cam | null = null;
let _poseInitialized = false;
let _current: DebugContent | null = null; // last content — reloaded after a context-loss rebuild
let _loadToken = 0;
let _destroyTimer: number | null = null;

export function webglError(): string | null {
  return _webglError;
}
export function contextLost(): boolean {
  return _contextLost;
}

function _readCam(): Cam | null {
  if (!nv) return null;
  try {
    return {
      azimuth: nv.scene.renderAzimuth,
      elevation: nv.scene.renderElevation,
      scale: nv.volScaleMultiplier,
    };
  } catch {
    return null;
  }
}

function _applyCam(cam: Cam | null): void {
  if (!nv || !cam) return;
  try {
    nv.setRenderAzimuthElevation(cam.azimuth, cam.elevation);
    if (Number.isFinite(cam.scale) && cam.scale > 0) nv.volScaleMultiplier = cam.scale;
    nv.drawScene();
  } catch {
    /* best-effort */
  }
}

function _onLost(e: Event): void {
  e.preventDefault(); // REQUIRED for the browser to later fire "restored"
  _savedCam = _readCam() ?? _savedCam;
  _contextLost = true;
}

function _onRestored(): void {
  if (!_canvas) return;
  const canvas = _canvas;
  // Tear the doomed instance down first — niivue's observers on canvas.parentElement would otherwise keep
  // it (and its volumes) alive forever. No drain / no loseContext: GL state was just wiped and this very
  // context is about to be reused (see destroyNiivue).
  destroyNiivue(nv, { drainVolumes: false, loseContext: false });
  nv = null;
  _createNv(canvas);
  _contextLost = false;
  // GPU state was wiped — reload the last content (which restores the pose from _savedCam).
  if (_current) void show(_current);
}

function _createNv(canvas: HTMLCanvasElement): Niivue | null {
  try {
    nv = new Niivue({
      backColor: [0.05, 0.05, 0.06, 1],
      show3Dcrosshair: false, // keep the render clean — no crosshair over the cornea
      isColorbar: false,
      dragAndDropEnabled: false,
      // Smooth (linear) sampling for the render: the isotropic debug grid is fine and a smooth MIP reads
      // better than a voxel staircase here (unlike the main viewer, which needs crisp voxels to match the
      // 2-D previews / training data — a concern that does not apply to this comparison-only viewport).
      isNearestInterpolation: false,
      // Wheel over the render tile drives the 3-D zoom (volScaleMultiplier) when yoked.
      yoke3Dto2DZoom: true,
      // Left-drag = rotate the render (crosshair drag mode is niivue's native 3-D rotate in the render
      // tile); right/centre-drag = pan. Wheel zoom is native.
      mouseEventConfig: {
        leftButton: { primary: DRAG_MODE.crosshair },
        rightButton: DRAG_MODE.pan,
        centerButton: DRAG_MODE.pan,
      },
    });
    nv.attachToCanvas(canvas);
    nv.setSliceType(SLICE_TYPE.RENDER);
    try {
      // magenta = mirror of niivue's green.json (R/B ramp instead of G), same half-alpha for compositing.
      nv.addColormap(CMAP_MAGENTA, {
        R: [0, 128, 255],
        G: [0, 0, 0],
        B: [0, 128, 255],
        A: [0, 64, 128],
        I: [0, 128, 255],
      });
      // hot with an alpha that rises with intensity → agreement transparent, disagreement glows.
      nv.addColormap(CMAP_HOT, {
        R: [0, 68, 187, 255, 255, 255],
        G: [0, 0, 0, 51, 221, 255],
        B: [0, 0, 0, 0, 0, 255],
        A: [0, 40, 110, 190, 230, 255],
        I: [0, 32, 102, 166, 217, 255],
      });
    } catch {
      /* older niivue without addColormap → fall back to built-in "green"/"hot" in show() */
    }
    if (typeof window !== "undefined") (window as unknown as { debugNv: Niivue }).debugNv = nv; // test hook (NOT window.nv)
    _webglError = null;
    return nv;
  } catch (e) {
    _webglError = `Debug 3-D viewer failed to initialise WebGL: ${e instanceof Error ? e.message : String(e)}`;
    nv = null;
    return null;
  }
}

function _cancelPendingDestroy(): void {
  if (_destroyTimer != null) {
    clearTimeout(_destroyTimer);
    _destroyTimer = null;
  }
}

/** Build (or return) the debug niivue instance attached to `canvas`. Cancels any pending deferred
 *  destroy first, so React StrictMode's mount→unmount→mount double-invoke keeps a single live instance.
 *  Returns null (and records webglError) if the browser can't give a WebGL2 context. */
export function create(canvas: HTMLCanvasElement): Niivue | null {
  _cancelPendingDestroy();
  if (nv && _canvas === canvas && !_contextLost) return nv;
  if (nv || _canvas) destroyNow(); // stale instance on a previous canvas
  const probe = canvas.getContext("webgl2");
  if (!probe) {
    _webglError =
      "This window can't provide a WebGL2 context, so the interactive 3-D viewport is disabled. " +
      "The 2-D views still work.";
    return null;
  }
  _canvas = canvas;
  canvas.addEventListener("webglcontextlost", _onLost, false);
  canvas.addEventListener("webglcontextrestored", _onRestored, false);
  _poseInitialized = false; // re-entry → restore _savedCam (or the default) on the first show
  return _createNv(canvas);
}

/** Immediately tear down the instance and free the WebGL context (WebKitGTK is fragile — keep only one
 *  extra context alive). Snapshots the pose so re-entry restores it. */
function destroyNow(): void {
  _cancelPendingDestroy();
  if (nv) {
    _savedCam = _readCam() ?? _savedCam;
    // Drain the volumes as well as disconnecting the observers: the decoded NVImages are the bulk of what
    // this viewport holds, and dropping them here frees them at close rather than at the next GC.
    destroyNiivue(nv, { loseContext: false });   // the canvas below owns the context free
  }
  const cv = _canvas;
  if (cv) {
    cv.removeEventListener("webglcontextlost", _onLost, false);
    cv.removeEventListener("webglcontextrestored", _onRestored, false);
    try {
      const gl = cv.getContext("webgl2") as WebGL2RenderingContext | null;
      gl?.getExtension("WEBGL_lose_context")?.loseContext();
    } catch {
      /* best-effort context free */
    }
  }
  nv = null;
  _canvas = null;
  _contextLost = false;
  _current = null;
}

/** Deferred destroy: schedules a teardown on the next tick and cancels it if create() runs first — this
 *  makes React StrictMode's mount→unmount→mount cycle a no-op instead of tearing down the live instance.
 *  A real unmount lets the timer fire. Components should call this from their effect cleanup. */
export function destroy(): void {
  _cancelPendingDestroy();
  _destroyTimer = window.setTimeout(() => {
    _destroyTimer = null;
    destroyNow();
  }, 0);
}

/** Load the volumes for a content mode, preserving the camera pose. This is the single "swap volumes,
 *  keep the pose" operation the method/mode switcher rides on: switching method or mode only changes the
 *  loaded volumes; the scene camera persists (niivue keeps renderAzimuth/elevation/scale across a load,
 *  and we snapshot+restore it anyway to be safe). */
export async function show(content: DebugContent): Promise<void> {
  if (!nv || _contextLost) return;
  _current = content;
  const token = ++_loadToken;
  const camBefore = _readCam();
  const win = content.window && content.window.length === 2 ? content.window : null;

  const vols: LoadVolumeOpts[] = [];
  if (content.mode === "consensus") {
    // ONE RGBA volume: the min/excess decomposition (colours + per-voxel alpha) is BAKED IN by the backend.
    // niivue 0.68.2 renders DT_RGBA32 directly in 3-D — the RGB is displayed verbatim and A drives opacity,
    // so we pass NO colormap/cal_min/cal_max (they don't apply to RGBA); only the volume-level opacity scales it.
    if (content.consensusUrl) {
      vols.push({ url: content.consensusUrl, opacity: 1 });
    }
  } else if (content.mode === "overlap") {
    vols.push({
      url: content.fixedUrl,
      colormap: hasCmap(CMAP_MAGENTA) ? CMAP_MAGENTA : "red",
      opacity: 1,
      ...(win ? { cal_min: win[0], cal_max: win[1] } : {}),
    });
    if (content.movingUrl) {
      vols.push({
        url: content.movingUrl,
        colormap: "green",
        opacity: 1,
        ...(win ? { cal_min: win[0], cal_max: win[1] } : {}),
      });
    }
  } else {
    // faint grayscale fixed for context, hot |fixed − moving| glowing on top.
    vols.push({
      url: content.fixedUrl,
      colormap: "gray",
      opacity: 0.35,
      ...(win ? { cal_min: win[0], cal_max: win[1] } : {}),
    });
    if (content.disagreeUrl) {
      // The hot |fixed − moving| volume holds RAW intensity values (hundreds). Two wrong ways to scale it:
      //  (1) the ORIGINAL blocker — passing disagree_mean (a normalized [0,1] summary) as cal_max saturated
      //      everything to full-hot so every method looked identical; and
      //  (2) the over-correction — passing NO cal_max let niivue auto-window EACH method independently, which
      //      lost cross-method comparability (a good aligner no longer read visibly cooler than identity).
      // CORRECT: use disagree_max — the SHARED, identity-derived raw-scale 99th pct the backend computes for
      // exactly this — as a common cal_max across every method. Fall back to auto-window only if it's absent.
      const dmax = typeof content.disagreeMax === "number" && content.disagreeMax > 0 ? content.disagreeMax : null;
      vols.push({
        url: content.disagreeUrl,
        colormap: hasCmap(CMAP_HOT) ? CMAP_HOT : "hot",
        opacity: 1,
        alphaThreshold: true,
        ...(dmax ? { cal_min: dmax * 0.08, cal_max: dmax } : {}),
      });
    }
  }

  // Release the previously-shown volumes FIRST: nv.loadVolumes() replaces its list by assignment and so
  // never runs removeVolume(), the only path that drops the NVImage from niivue's strong mediaUrlMap —
  // so every method/mode switch would otherwise strand its 1-3 decoded volumes for the session (see
  // src/niivue/nvRelease.ts). The mode switcher is driven from a toolbar, so this fires often.
  releaseVolumes(nv);
  // withVolumeNames: these URLs are ?t= cache-busted, which makes niivue mis-sniff the type and issue a
  // discarded full-size probe fetch per volume before the real load.
  await nv.loadVolumes(withVolumeNames(vols));
  if (token !== _loadToken || !nv) return; // a newer show() superseded this one
  nv.setSliceType(SLICE_TYPE.RENDER);

  // Pose: within one instance restore the live pose (method/mode switch); on the first show after a
  // create/rebuild use the saved pose (re-entry) or the 3/4 default (first ever).
  _applyCam(_poseInitialized ? camBefore : _savedCam ?? DEFAULT_CAM);
  _poseInitialized = true;
  nv.drawScene();
}

/** Load the single min/excess RGBA consensus volume (all replicates of the eye, colours baked in) into the
 *  render, preserving the camera pose. Thin wrapper over show() so pose/token handling stays in one place —
 *  this is the "swap the consensus volume, keep the angle" op the method switcher rides on. */
export function loadConsensus(volumeUrl: string): Promise<void> {
  return show({ mode: "consensus", fixedUrl: "", consensusUrl: volumeUrl });
}

function hasCmap(name: string): boolean {
  try {
    return (nv?.colormaps?.() ?? []).includes(name);
  } catch {
    return false;
  }
}

/** Repaint (e.g. after a container resize). */
export function redraw(): void {
  if (nv && !_contextLost) {
    try {
      nv.drawScene();
    } catch {
      /* no-op */
    }
  }
}
