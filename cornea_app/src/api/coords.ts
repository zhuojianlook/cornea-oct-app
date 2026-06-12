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

/* Voxels under a circular brush centred at image-fraction (fx, fy) with `radius` in
   SOURCE voxels — enumerated densely in source space so a downscaled preview doesn't
   leave gaps in the painted scar. */
export function brushVoxels(meta: PreviewImage, fx: number, fy: number, radius: number): [number, number, number][] {
  const sw = meta.source_width, sh = meta.source_height, slice = meta.slice_index, o = meta.orientation;
  if (sw == null || sh == null || slice == null || !o) return [];
  const cCol = fx * (sw - 1);
  const cRow = (1 - fy) * (sh - 1); // previews are saved vertically flipped (match pxToIjk)
  const r = Math.max(0, Math.round(radius));
  const out: [number, number, number][] = [];
  for (let dr = -r; dr <= r; dr++) {
    for (let dc = -r; dc <= r; dc++) {
      if (dc * dc + dr * dr > r * r) continue;
      const col = Math.round(cCol + dc), row = Math.round(cRow + dr);
      if (col < 0 || col >= sw || row < 0 || row >= sh) continue;
      if (o === "axial") out.push([col, row, slice]);
      else if (o === "coronal") out.push([col, slice, row]);
      else if (o === "sagittal") out.push([slice, col, row]);
    }
  }
  return out;
}
