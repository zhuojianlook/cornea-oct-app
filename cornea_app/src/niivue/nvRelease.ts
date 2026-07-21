/* ──────────────────────────────────────────────────────────
   Niivue RESOURCE-RELEASE helpers — shared by EVERY Niivue instance in the app
   (nvController's singleton, debugNvController's, and the per-mount comparison viewers).

   WHY THIS MODULE EXISTS — the measured leak (app v0.0.209, WebKitGTK/Tauri):
   a 4h46m triage run of ~55 case opens grew WebKitWebProcess to 11.8 GB RSS (525 MB after a restart),
   ~215 MB per open, with 3x more kernel than user time (page-fault churn, not a JS loop).

   The proven owner is niivue's `Niivue.mediaUrlMap` — a STRONG `Map` keyed on the NVImage itself
   (build/niivue/index.js:35187, populated at :36817). `Niivue.loadVolumes()` empties its volume list by
   ASSIGNMENT (`this.volumes = []`, :38961) and therefore NEVER runs `removeVolume()`, which is the only
   code path that reaches `mediaUrlMap.delete(volume)` (:37887). So every base volume ever loaded — a
   63 MB Uint16Array, or 126 MB when the NIfTI header carries a non-identity scl_slope and niivue promotes
   the scalars to Float32 — stayed reachable for the life of the instance, i.e. the life of the app.
   Draining the list through removeVolumeByIndex() BEFORE each load is the fix; an instrumented harness
   measured 88.7 MB/open → ~0 and mediaUrlMap pinned at 1 with the drain alone.

   Note the leak is per-INSTANCE: an abandoned Niivue takes its whole mediaUrlMap with it *provided the
   instance itself becomes unreachable* — which is what destroyNiivue()'s cleanup() call ensures.

   Everything here is best-effort and idempotent: releasing memory must never be able to break a viewer,
   so every step is individually try/caught.
   ────────────────────────────────────────────────────────── */

import type { Niivue } from "@niivue/niivue";

/** Internals we must reach that aren't on niivue's public type surface. */
interface NvInternals {
  gl?: WebGL2RenderingContext;
  overlayTexture?: WebGLTexture | null;
  document?: {
    data?: { imageOptionsArray?: unknown[] };
    imageOptionsMap?: Map<unknown, unknown>;
  };
}

/**
 * Drop EVERY volume from `nv`, routing each removal through removeVolumeByIndex() → removeVolume(),
 * the only call that deletes the NVImage from the strong `mediaUrlMap` (see the header). Call this
 * before any loadVolumes(), which would otherwise strand the outgoing volume forever.
 * Safe to call with no volumes loaded.
 */
export function releaseVolumes(nv: Niivue | null | undefined): void {
  if (!nv) return;
  try {
    // Descending order: the final removal empties the list, so its internal updateGLVolume() has no
    // layer left to rebuild (removing index 0 first would make niivue re-upload an overlay as layer 0).
    while (nv.volumes.length) nv.removeVolumeByIndex(nv.volumes.length - 1);
  } catch (e) {
    console.warn("[nvRelease] releaseVolumes failed", e);
  }
  // niivue's NVDocument keeps a PARALLEL record of load options that removeVolume() does not prune
  // (Niivue.removeVolume never calls document.removeImage), so it grows one entry per load forever.
  // The volume list is empty now, so reset the record wholesale — splicing entries individually would
  // shift the array indices that imageOptionsMap stores and corrupt it. The app never uses
  // save/loadDocument, so nothing reads this back.
  try {
    const doc = (nv as unknown as NvInternals).document;
    if (doc && nv.volumes.length === 0) {
      if (doc.data?.imageOptionsArray) doc.data.imageOptionsArray.length = 0;
      doc.imageOptionsMap?.clear();
    }
  } catch { /* bookkeeping only — never fatal */ }
}

/**
 * A clean `name` for a volume load option, derived from the URL's basename.
 *
 * WHY: our volume URLs carry a `?t=<Date.now()>` cache-buster, which defeats niivue's extension
 * sniffing (getPrimaryExtension captures "gz?t=1784…", so NVIMAGE_TYPE.parse returns UNKNOWN). On
 * UNKNOWN, niivue issues a PROBE `fetch(url)` of the whole file whose body it never reads and never
 * cancels — a discarded second copy of every volume (measured: 2 network fetches per load, 1 with the
 * name passed). Passing the query-free filename makes the sniff succeed and the probe never fire.
 */
export function volumeName(url: string, fallback = "volume.nii.gz"): string {
  try {
    const path = url.split("?")[0].split("#")[0];
    const base = path.substring(path.lastIndexOf("/") + 1);
    return base.includes(".") ? base : fallback;
  } catch {
    return fallback;
  }
}

/** Add a query-free `name` to each volume load option (see volumeName) without disturbing the rest. */
export function withVolumeNames<T extends { url?: string; name?: string }>(vols: T[]): T[] {
  return vols.map((v) => (v.name || !v.url ? v : { ...v, name: volumeName(v.url) }));
}

/**
 * Handle of the layer-1 overlay texture, to be paired with releaseOrphanedOverlayTexture().
 *
 * WHY: niivue's refreshLayers() allocates the layer-1 texture with `rgbaTex(null, …)`
 * (allocateVolumeTextures, index.js:29509) and then OVERWRITES `this.overlayTexture` — unlike every
 * other texture path, which passes the existing handle so rgbaTex deletes it first. The previous
 * handle is simply orphaned: one full-size RGBA8 3-D texture (~126 MB at 513x640x101) leaked per
 * updateGLVolume()/setOpacity() while an overlay is loaded. GPU allocations are invisible to JS heap
 * snapshots and to performance.memory, so this never shows up in JS-level instrumentation.
 */
export function captureOverlayTexture(nv: Niivue | null | undefined): WebGLTexture | null {
  return (nv as unknown as NvInternals | null | undefined)?.overlayTexture ?? null;
}

/**
 * Delete the overlay texture captured before an overlay op, IF niivue replaced it (see
 * captureOverlayTexture). Only deletes when a replacement is actually installed, which guarantees the
 * old handle is unreferenced and that nothing is left bound to a deleted texture.
 */
export function releaseOrphanedOverlayTexture(nv: Niivue | null | undefined, before: WebGLTexture | null): void {
  if (!nv || !before) return;
  const internals = nv as unknown as NvInternals;
  const after = internals.overlayTexture ?? null;
  if (!after || after === before) return;
  try { internals.gl?.deleteTexture(before); } catch { /* best-effort GPU free */ }
}

/**
 * Tear down a Niivue instance that is being ABANDONED (component unmount, context-loss rebuild).
 *
 * cleanup() is the critical part: it is the ONLY thing that disconnects the ResizeObserver and
 * MutationObserver niivue installed on `canvas.parentElement` (index.js:35383-35393). Those closures
 * capture the instance, and the observed parent element outlives the viewer, so without cleanup() the
 * discarded Niivue — its GL context, its textures and every NVImage still in its mediaUrlMap — stays
 * reachable for the rest of the session and can never be collected.
 *
 * `drainVolumes:false` on the WebGL context-loss path: the whole instance is going away (so its
 * mediaUrlMap dies with it) and GL state has just been wiped, so the extra updateGLVolume() work each
 * removal triggers would run against textures that no longer exist.
 * `loseContext:false` likewise on that path — the context was just RESTORED and is about to be reused.
 */
export function destroyNiivue(
  nv: Niivue | null | undefined,
  opts: { drainVolumes?: boolean; loseContext?: boolean } = {},
): void {
  if (!nv) return;
  const { drainVolumes = true, loseContext = true } = opts;
  if (drainVolumes) releaseVolumes(nv);
  // A drawing bitmap + its undo stack are full-volume Uint8Arrays; closeDrawing also recycles drawTexture.
  try { nv.closeDrawing(); } catch { /* no drawing loaded */ }
  try { (nv as unknown as { cleanup?: () => void }).cleanup?.(); } catch (e) {
    console.warn("[nvRelease] niivue cleanup() failed", e);
  }
  if (loseContext) {
    try {
      (nv as unknown as NvInternals).gl?.getExtension("WEBGL_lose_context")?.loseContext();
    } catch { /* best-effort context free */ }
  }
}
