/* PNG-pixel → volume IJK for clicks on the 2D slice gallery.
   Mirrors the sidecar's preview rendering (per-orientation axis order, vertical flip, AND the
   90° rotation baked in for display) so a click on a preview maps to the correct voxel. */

import type { PreviewImage } from "./types";

// Map an image-fraction click (ffx across width, ffy down height) on the FINAL displayed PNG
// back to SOURCE voxel (col=column_axis, row=row_axis), undoing rot90(rotate_k) then the flipud.
function previewToSource(meta: PreviewImage, ffx: number, ffy: number): { col: number; row: number } | null {
  const sw = meta.source_width, sh = meta.source_height;
  if (sw == null || sh == null) return null;
  const k = (((meta.rotate_k ?? 0) % 4) + 4) % 4; // normalize (−1 → 3)
  // Undo the rotation: final = rot90(P, k), where P = flipud(scaled).
  let pfx = ffx, pfy = ffy;
  if (k === 2) { pfx = 1 - ffx; pfy = 1 - ffy; }
  else if (k === 1) { pfx = 1 - ffy; pfy = ffx; }   // rot90 CCW
  else if (k === 3) { pfx = ffy; pfy = 1 - ffx; }   // rot90 CW (k = −1)
  // Undo flipud (P row was flipped), then map linearly to source.
  const sfx = pfx, sfy = 1 - pfy;
  return { col: Math.round(sfx * (sw - 1)), row: Math.round(sfy * (sh - 1)) };
}

export function pxToIjk(meta: PreviewImage, x: number, y: number): [number, number, number] | null {
  const iw = meta.image_width, ih = meta.image_height, slice = meta.slice_index;
  if (iw == null || ih == null || slice == null || !meta.orientation) return null;
  x = Math.min(Math.max(x, 0), iw - 1);
  y = Math.min(Math.max(y, 0), ih - 1);
  const src = previewToSource(meta, iw <= 1 ? 0 : x / (iw - 1), ih <= 1 ? 0 : y / (ih - 1));
  if (!src) return null;
  const { col, row } = src;
  if (meta.orientation === "axial") return [col, row, slice];
  if (meta.orientation === "coronal") return [col, slice, row];
  if (meta.orientation === "sagittal") return [slice, col, row];
  return null;
}

/* Voxels under a circular brush centred at image-fraction (fx, fy) with `radius` in
   SOURCE voxels — enumerated densely in source space so a downscaled preview doesn't
   leave gaps in the painted scar. Rotation-aware via previewToSource. */
export function brushVoxels(meta: PreviewImage, fx: number, fy: number, radius: number): [number, number, number][] {
  const sw = meta.source_width, sh = meta.source_height, slice = meta.slice_index, o = meta.orientation;
  if (sw == null || sh == null || slice == null || !o) return [];
  const c = previewToSource(meta, fx, fy);
  if (!c) return [];
  const r = Math.max(0, Math.round(radius));
  const out: [number, number, number][] = [];
  for (let dr = -r; dr <= r; dr++) {
    for (let dc = -r; dc <= r; dc++) {
      if (dc * dc + dr * dr > r * r) continue;
      const col = c.col + dc, row = c.row + dr;
      if (col < 0 || col >= sw || row < 0 || row >= sh) continue;
      if (o === "axial") out.push([col, row, slice]);
      else if (o === "coronal") out.push([col, slice, row]);
      else if (o === "sagittal") out.push([slice, col, row]);
    }
  }
  return out;
}
