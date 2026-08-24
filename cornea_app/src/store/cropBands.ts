// Per-lateral ARTIFACT band (#9 v3, crop_bands) — the frontend twin of the backend's oct_preprocess._artifact_bands.
// A time-domain artifact (e.g. an eyelid) occupies a frame band [lo,hi] whose extent varies per sagittal slice; the
// reviewer marks it on a few laterals and it's INTERPOLATED across the laterals between them. These helpers give the
// interpolated band for any lateral so BOTH editor panes (original + corrected) can break their surface lines over it
// and paint the preview — kept in ONE place so the interpolation can never drift from the backend (or between panes).

export type CropBands = Record<number, [number, number]>;

/** Parse persisted oct_params.crop_bands ({str(lateral):[lo,hi]}) into a numeric, lo≤hi map. */
export function parseCropBands(raw: unknown): CropBands {
  const out: CropBands = {};
  if (raw && typeof raw === "object") {
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
      if (Array.isArray(v) && v.length === 2) {
        const lo = Number(v[0]), hi = Number(v[1]);
        if (Number.isFinite(lo) && Number.isFinite(hi)) out[Number(k)] = [Math.min(lo, hi), Math.max(lo, hi)];
      }
    }
  }
  return out;
}

/** The interpolated artifact band [lo,hi] for a lateral — MIRRORS the backend `_artifact_bands`: linear between the
 *  two nearest marked laterals, confined to the [min,max] marked span (null outside), single mark → that lateral.
 *  Rounds half-UP (Math.round) to match the Python `math.floor(x+0.5)`. Returns null when there is no band here. */
export function interpBand(bands: CropBands, lat: number): [number, number] | null {
  const marks = Object.entries(bands)
    .map(([k, v]) => ({ lat: Number(k), lo: Math.min(v[0], v[1]), hi: Math.max(v[0], v[1]) }))
    .sort((a, b) => a.lat - b.lat);
  const n = marks.length;
  if (!n || lat < marks[0].lat || lat > marks[n - 1].lat) return null;
  if (n === 1) return [Math.round(marks[0].lo), Math.round(marks[0].hi)];
  let i = 0; while (i < n - 1 && marks[i + 1].lat < lat) i++;
  const a = marks[i], b = marks[Math.min(i + 1, n - 1)];
  const span = b.lat - a.lat;
  const t = span > 0 ? (lat - a.lat) / span : 0;
  return [Math.round(a.lo + t * (b.lo - a.lo)), Math.round(a.hi + t * (b.hi - a.hi))];
}
