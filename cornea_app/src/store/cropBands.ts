// Per-lateral ARTIFACT bands (#9 v3, crop_bands) — the frontend twin of the backend's
// oct_preprocess.resolve_crop_bands. A time-domain artifact (an eyelid, a blink) occupies a frame band [lo,hi]
// whose extent varies per sagittal slice; the reviewer marks it on a few laterals and it is INTERPOLATED across
// the laterals between them. Several artifacts = several EXPLICIT bands, each with its own id and its own marks
// (reviewer 2026-09-09: "start band on one slice and end band on another … and also other bands with their own
// start and end"). Bands are independent: no linking, no grouping, no merging across bands — which band a mark
// belongs to is the reviewer's decision (the ACTIVE band in the ⊟ tool), never inferred from geometry.
//
// PERSISTED FORM (oct_params.crop_bands): {"bands": [{"id": <int>, "marks": {"<lateral>": [lo, hi], ...}}, ...]}
//   - ids are unique small positive ints; lo<=hi are inclusive frame indices; at most ONE mark per lateral per band.
//   - LEGACY form {"<lateral>": [lo, hi], ...} (the pre-contract single band) is read as one band (id 1) carrying
//     those marks. Writers always emit the new form. Empty {} / {"bands": []} = no band.
//   - The intermediate list-of-lists form {"<lateral>": [[lo,hi],...]} of the stopped attempt is NOT supported.
//
// RESOLUTION (bandsAt): a band's lateral extent is exactly [min marked lateral, max marked lateral] of ITS marks;
// lo/hi are linear between ITS two nearest marked laterals, rounded half-up (Math.round ≡ Python floor(x+0.5)
// for x ≥ 0); a single-mark band covers its slice only. The bands at a lateral are the list over ALL bands
// covering it (they may overlap in frames — consumers that need a mask take the union).
//
// Kept in ONE place so both editor panes (original + corrected) and the persistence path can never disagree
// about what is marked, and so the arithmetic stays bit-identical to the backend (shared fixture test).

export type Band = [number, number];
/** One explicit artifact band: its id and its marks, one [lo,hi] per marked lateral. */
export interface CropBand { id: number; marks: Record<number, Band> }
/** The whole set of bands on a scan. */
export interface CropBands { bands: CropBand[] }
/** A band resolved at one lateral. */
export interface BandAt { id: number; lo: number; hi: number }
/** The wire/persisted form (the new form only — the serialiser never writes legacy). */
export interface CropBandsWire { bands?: Array<{ id: number; marks: Record<string, Band> }> }

/** One [lo,hi] from an unknown value (two finite numbers, any order). Truncates like Python int(). */
function toBand(v: unknown): Band | null {
  if (!Array.isArray(v) || v.length !== 2) return null;
  const a = Number(v[0]), b = Number(v[1]);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  const lo = Math.trunc(a), hi = Math.trunc(b);
  return [Math.min(lo, hi), Math.max(lo, hi)];
}

/** Parse one marks object {"<lateral>": [lo,hi]} — junk keys/values dropped. */
function parseMarks(raw: unknown): Record<number, Band> {
  const out: Record<number, Band> = {};
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
      const lat = Number(k);
      if (!Number.isFinite(lat) || !Number.isInteger(lat)) continue;
      const b = toBand(v);
      if (b) out[lat] = b;
    }
  }
  return out;
}

export function emptyCropBands(): CropBands { return { bands: [] }; }

/** Parse persisted oct_params.crop_bands — the new {"bands": [...]} form OR the legacy {"<lateral>": [lo,hi]}
 *  form (→ one band, id 1). Bands are returned in ascending id order; a band whose id is missing, non-positive,
 *  non-integer or a duplicate is given the next free id so ids stay unique (nothing in the store is like that,
 *  this is belt-and-braces). Bands with NO valid marks are kept: an empty band is a legitimate UI state ("+ New
 *  band" before its first mark) and the serialiser drops it on write. */
export function parseCropBands(raw: unknown): CropBands {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return emptyCropBands();
  const obj = raw as Record<string, unknown>;
  if (Array.isArray(obj.bands)) {
    const used = new Set<number>();
    const bands: CropBand[] = [];
    for (const item of obj.bands) {
      if (!item || typeof item !== "object" || Array.isArray(item)) continue;
      const it = item as Record<string, unknown>;
      const marks = parseMarks(it.marks);
      let id = Number(it.id);
      if (!Number.isInteger(id) || id <= 0 || used.has(id)) { id = 1; while (used.has(id)) id++; }
      used.add(id);
      bands.push({ id, marks });
    }
    bands.sort((a, b) => a.id - b.id);
    return { bands };
  }
  // legacy single band {"<lateral>": [lo,hi]} — only when it actually carries a mark
  const marks = parseMarks(obj);
  return Object.keys(marks).length ? { bands: [{ id: 1, marks }] } : emptyCropBands();
}

/** Serialise for persistence in the NEW form. Bands with no marks are dropped (they have no effect and would
 *  only clutter the manifest); bands ascending by id; mark keys ascending laterals as strings. Returns {} when
 *  nothing is marked (= "no band", and what the sidecar pops from oct_params). */
export function serialiseCropBands(bands: CropBands | null | undefined): CropBandsWire {
  const out: NonNullable<CropBandsWire["bands"]> = [];
  for (const b of [...(bands?.bands ?? [])].sort((a, c) => a.id - c.id)) {
    const lats = Object.keys(b.marks).map(Number).filter(Number.isFinite).sort((x, y) => x - y);
    if (!lats.length) continue;
    const marks: Record<string, Band> = {};
    for (const lat of lats) { const m = b.marks[lat]; marks[String(lat)] = [Math.min(m[0], m[1]), Math.max(m[0], m[1])]; }
    out.push({ id: b.id, marks });
  }
  return out.length ? { bands: out } : {};
}

/** The marks the backend actually interpolates from, given the volume dims (both optional): a mark on a lateral
 *  outside [0, nLateral) is DROPPED, and each mark's lo/hi is CLAMPED into [0, nFrames−1] BEFORE interpolation (a
 *  mark that falls entirely outside is dropped). Ascending laterals. Mirrors resolve_crop_bands' per-mark
 *  validation, so a clamped end mark interpolates from its clamped value — not from the raw one. */
function effectiveMarks(band: CropBand, nFrames?: number, nLateral?: number): Array<{ lat: number; lo: number; hi: number }> {
  const hasF = nFrames != null && Number.isFinite(nFrames), hasL = nLateral != null && Number.isFinite(nLateral);
  const out: Array<{ lat: number; lo: number; hi: number }> = [];
  for (const k of Object.keys(band.marks)) {
    const lat = Number(k);
    if (!Number.isFinite(lat)) continue;
    if (hasL && (lat < 0 || lat >= (nLateral as number))) continue;
    const m = band.marks[lat];
    let lo = Math.min(m[0], m[1]), hi = Math.max(m[0], m[1]);
    if (hasF) { lo = Math.max(0, lo); hi = Math.min((nFrames as number) - 1, hi); if (lo > hi) continue; }
    out.push({ lat, lo, hi });
  }
  return out.sort((a, b) => a.lat - b.lat);
}

/** Resolve ONE band at a lateral (see the header): null outside its [min,max] marked span; single mark → its
 *  own lateral only; otherwise linear between the two nearest marked laterals with half-up rounding. Same
 *  arithmetic order as numpy.interp (slope·(x−x0)+y0) so the indices are bit-identical to the backend's.
 *  nFrames / nLateral (optional) validate + clamp the MARKS first, as the backend does (effectiveMarks). */
export function bandAt(band: CropBand, lat: number, nFrames?: number, nLateral?: number): Band | null {
  const ms = effectiveMarks(band, nFrames, nLateral);
  const n = ms.length;
  if (!n || lat < ms[0].lat || lat > ms[n - 1].lat) return null;
  if (n === 1) return [Math.round(ms[0].lo), Math.round(ms[0].hi)];
  let i = 0; while (i < n - 1 && ms[i + 1].lat < lat) i++;
  const a = ms[i], b = ms[Math.min(i + 1, n - 1)];
  const span = b.lat - a.lat;
  if (span <= 0 || lat === a.lat) return [Math.round(a.lo), Math.round(a.hi)];
  const dx = lat - a.lat;
  const lo = ((b.lo - a.lo) / span) * dx + a.lo;
  const hi = ((b.hi - a.hi) / span) * dx + a.hi;
  return [Math.round(lo), Math.round(hi)];
}

/** THE bands at a lateral: every band covering it, ascending by id, each as {id, lo, hi}. Pass nFrames (and
 *  nLateral) to validate + clamp the marks the way the backend does (see effectiveMarks) and to clamp the result
 *  to [0, nFrames−1] (a band that falls entirely outside is dropped). Bands may overlap in frames — this is a
 *  list, not a union. */
export function bandsAt(bands: CropBands, lat: number, nFrames?: number, nLateral?: number): BandAt[] {
  const out: BandAt[] = [];
  for (const b of [...bands.bands].sort((x, y) => x.id - y.id)) {
    const r = bandAt(b, lat, nFrames, nLateral);
    if (!r) continue;
    let [lo, hi] = r;
    if (nFrames != null && Number.isFinite(nFrames)) { lo = Math.max(0, lo); hi = Math.min(nFrames - 1, hi); }
    if (lo <= hi) out.push({ id: b.id, lo, hi });
  }
  return out;
}

/** Is frame f inside ANY of the resolved bands? (the union, for line breaks / masks) */
export function inAnyBand(resolved: BandAt[], f: number): boolean {
  return resolved.some((b) => f >= b.lo && f <= b.hi);
}

// ── editing helpers (pure; every one returns a NEW CropBands) ──────────────────────────────────────────────

/** The next free id: max existing + 1 (ids are never reused within a scan while a band still exists). */
export function nextBandId(bands: CropBands): number {
  return bands.bands.reduce((m, b) => Math.max(m, b.id), 0) + 1;
}

/** Create an EMPTY band and return it with the new set. */
export function newBand(bands: CropBands): { bands: CropBands; id: number } {
  const id = nextBandId(bands);
  return { bands: { bands: [...bands.bands, { id, marks: {} }] }, id };
}

/** Set band `id`'s mark on `lat` to [lo,hi] (REPLACING that band's mark there — one mark per lateral per band).
 *  A missing band id is created. */
export function setMark(bands: CropBands, id: number, lat: number, band: Band): CropBands {
  const mark: Band = [Math.min(band[0], band[1]), Math.max(band[0], band[1])];
  let found = false;
  const next = bands.bands.map((b) => {
    if (b.id !== id) return b;
    found = true;
    return { id: b.id, marks: { ...b.marks, [lat]: mark } };
  });
  if (!found) next.push({ id, marks: { [lat]: mark } });
  return { bands: next };
}

/** Remove band `id`'s mark on `lat` (the band itself stays, even if that was its last mark). */
export function clearMark(bands: CropBands, id: number, lat: number): CropBands {
  return { bands: bands.bands.map((b) => {
    if (b.id !== id || !(lat in b.marks)) return b;
    const marks = { ...b.marks }; delete marks[lat];
    return { id: b.id, marks };
  }) };
}

/** Remove band `id` and all its marks. */
export function removeBand(bands: CropBands, id: number): CropBands {
  return { bands: bands.bands.filter((b) => b.id !== id) };
}

export function findBand(bands: CropBands, id: number | null | undefined): CropBand | undefined {
  return id == null ? undefined : bands.bands.find((b) => b.id === id);
}

/** Ascending marked laterals of a band. */
export function markedLaterals(band: CropBand): number[] {
  return Object.keys(band.marks).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
}

/** Total number of marks over all bands. */
export function countMarks(bands: CropBands): number {
  return bands.bands.reduce((n, b) => n + Object.keys(b.marks).length, 0);
}

// ── colours: a small FIXED palette cycling by id, the same in both panes ─────────────────────────────────────
// Band 1 keeps the tool's own blue (#5db0ff) so a single-band scan looks exactly as before. The others are
// chosen away from the line colours already on the picture (red edge, cyan quadratic, orange posterior, amber
// target, pink ⚑ marks) so a band tint is never mistaken for a line.
export const BAND_PALETTE: ReadonlyArray<{ hex: string; rgb: string }> = [
  { hex: "#5db0ff", rgb: "93,176,255" },   // blue
  { hex: "#a78bfa", rgb: "167,139,250" },  // violet
  { hex: "#34d399", rgb: "52,211,153" },   // emerald
  { hex: "#e879f9", rgb: "232,121,249" },  // fuchsia
  { hex: "#a3e635", rgb: "163,230,53" },   // lime
  { hex: "#2dd4bf", rgb: "45,212,191" },   // teal
];

/** The colour for a band id (id 1 → palette[0], cycling). */
export function bandColor(id: number): { hex: string; rgb: string } {
  const i = Number.isInteger(id) && id > 0 ? (id - 1) % BAND_PALETTE.length : 0;
  return BAND_PALETTE[i];
}
