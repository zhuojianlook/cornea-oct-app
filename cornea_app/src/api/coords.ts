/* PNG-pixel → volume IJK for clicks on the 2D slice gallery.
   Mirrors the sidecar's preview rendering (per-orientation axis order + vertical
   flip), so a click on a preview maps to the correct voxel. */

import type { PreviewImage } from "./types";

export function pxToIjk(meta: PreviewImage, x: number, y: number): [number, number, number] | null {
  const sw = meta.source_width, sh = meta.source_height;
  const iw = meta.image_width, ih = meta.image_height;
  const slice = meta.slice_index;
  if (sw == null || sh == null || iw == null || ih == null || slice == null || !meta.orientation) return null;
  x = Math.min(Math.max(x, 0), iw - 1);
  y = Math.min(Math.max(y, 0), ih - 1);
  const srcCol = iw <= 1 ? 0 : (x * (sw - 1)) / (iw - 1);
  const unflippedRow = ih - 1 - y; // previews are saved vertically flipped
  const srcRow = ih <= 1 ? 0 : (unflippedRow * (sh - 1)) / (ih - 1);
  const col = Math.round(srcCol);
  const row = Math.round(srcRow);
  if (meta.orientation === "axial") return [col, row, slice];
  if (meta.orientation === "coronal") return [col, slice, row];
  if (meta.orientation === "sagittal") return [slice, col, row];
  return null;
}
