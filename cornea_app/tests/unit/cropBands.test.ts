// Unit tests for src/store/cropBands.ts — the frontend twin of the backend's oct_preprocess.resolve_crop_bands
// (EXPLICIT bands: {"bands": [{"id", "marks": {"<lateral>": [lo, hi]}}]}, each interpolated on its own marks).
//
// Run (no test runner is installed; Node ≥22.18 strips the types itself):
//   cd cornea_app && node --test tests/unit/cropBands.test.ts
//
// The FIXTURE test is the contract check between the two implementations: the backend implementer writes
// python-sidecar/tests/data/crop_band_fixtures.json — a list of cases {n_lateral, n_frames, crop_bands,
// probes: {"<lateral>": [[lo, hi, id], ...]}} generated from the Python resolve_crop_bands — and this test asserts
// `bandsAt` agrees on every probe (lo, hi, id). Path: $CROP_BAND_FIXTURES, else the shared file in python-sidecar.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  type CropBands,
  BAND_PALETTE, bandAt, bandColor, bandsAt, clearMark, countMarks, emptyCropBands, findBand, inAnyBand,
  markedLaterals, newBand, nextBandId, parseCropBands, removeBand, serialiseCropBands, setMark,
} from "../../src/store/cropBands.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_FIXTURE = path.resolve(HERE, "..", "..", "python-sidecar", "tests", "data", "crop_band_fixtures.json");

// ── parsing (both persisted forms) ────────────────────────────────────────────────────────────────────────────
test("parseCropBands: legacy {lateral: [lo,hi]} → ONE band (id 1) with those marks, lo≤hi", () => {
  assert.deepEqual(parseCropBands({ "3": [7, 2], "10": [4, 9] }),
    { bands: [{ id: 1, marks: { 3: [2, 7], 10: [4, 9] } }] });
});

test("parseCropBands: explicit {bands: [...]} form — ids kept, bands sorted by id, marks per band", () => {
  const got = parseCropBands({ bands: [
    { id: 2, marks: { "40": [10, 20], "60": [12, 22] } },
    { id: 1, marks: { "0": [30, 39] } },
  ] });
  assert.deepEqual(got, { bands: [
    { id: 1, marks: { 0: [30, 39] } },
    { id: 2, marks: { 40: [10, 20], 60: [12, 22] } },
  ] });
});

test("parseCropBands: empty {} / {bands: []} / null / non-object → no band", () => {
  assert.deepEqual(parseCropBands({}), { bands: [] });
  assert.deepEqual(parseCropBands({ bands: [] }), { bands: [] });
  assert.deepEqual(parseCropBands(null), { bands: [] });
  assert.deepEqual(parseCropBands([1, 2]), { bands: [] });
  assert.deepEqual(parseCropBands("x"), { bands: [] });
});

test("parseCropBands: junk is dropped — bad keys/values, missing/duplicate ids get the next free id, empty bands kept", () => {
  const got = parseCropBands({ bands: [
    { id: 3, marks: { "1": [1, 2], "x": [1, 2], "5": "nope", "6": [1], "7": [1, 2, 3], "8": [[1, 2]] } },
    { id: 3, marks: { "2": [3.9, 7.2] } },          // duplicate id → next free (1)
    { marks: { "9": [0, 0] } },                     // missing id → next free (2)
    { id: 4, marks: {} },                           // empty band survives the parse
    "junk", null, 7,
  ] });
  assert.deepEqual(got, { bands: [
    { id: 1, marks: { 2: [3, 7] } },                // truncates like Python int()
    { id: 2, marks: { 9: [0, 0] } },
    { id: 3, marks: { 1: [1, 2] } },
    { id: 4, marks: {} },
  ] });
});

test("parseCropBands: the intermediate list-of-lists form is NOT supported (its marks are dropped)", () => {
  assert.deepEqual(parseCropBands({ "3": [[1, 2], [5, 6]] }), { bands: [] });
});

// ── serialising (always the new form) ─────────────────────────────────────────────────────────────────────────
test("serialiseCropBands: new form only; bands ascending by id; empty bands dropped; nothing marked → {}", () => {
  const bands: CropBands = { bands: [
    { id: 2, marks: { 60: [12, 22], 40: [20, 10] } },
    { id: 1, marks: { 0: [30, 39] } },
    { id: 3, marks: {} },
  ] };
  assert.deepEqual(serialiseCropBands(bands), { bands: [
    { id: 1, marks: { "0": [30, 39] } },
    { id: 2, marks: { "40": [10, 20], "60": [12, 22] } },
  ] });
  assert.deepEqual(Object.keys(serialiseCropBands(bands).bands![1].marks), ["40", "60"]);
  assert.deepEqual(serialiseCropBands({ bands: [{ id: 1, marks: {} }] }), {});
  assert.deepEqual(serialiseCropBands(emptyCropBands()), {});
  assert.deepEqual(serialiseCropBands(null), {});
});

test("parse ∘ serialise is the identity on a set of marked bands; legacy round-trips into the new form", () => {
  const bands: CropBands = { bands: [{ id: 1, marks: { 0: [0, 3], 12: [10, 20] } }, { id: 5, marks: { 100: [7, 7] } }] };
  assert.deepEqual(parseCropBands(JSON.parse(JSON.stringify(serialiseCropBands(bands)))), bands);
  const legacy = parseCropBands({ "10": [20, 30], "20": [30, 40] });
  assert.deepEqual(serialiseCropBands(legacy), { bands: [{ id: 1, marks: { "10": [20, 30], "20": [30, 40] } }] });
});

// ── resolution: one band, on ITS marks only ───────────────────────────────────────────────────────────────────
test("bandAt: linear between the two nearest marks, confined to the marked span, single mark → its slice only", () => {
  const b = { id: 1, marks: { 10: [20, 30] as [number, number], 20: [30, 40] as [number, number] } };
  assert.deepEqual(bandAt(b, 15), [25, 35]);
  assert.deepEqual(bandAt(b, 10), [20, 30]);
  assert.deepEqual(bandAt(b, 20), [30, 40]);
  assert.equal(bandAt(b, 9), null);
  assert.equal(bandAt(b, 21), null);
  const one = { id: 2, marks: { 7: [3, 9] as [number, number] } };
  assert.deepEqual(bandAt(one, 7), [3, 9]);
  assert.equal(bandAt(one, 6), null);
  assert.equal(bandAt(one, 8), null);
  assert.equal(bandAt({ id: 3, marks: {} }, 0), null);
});

test("bandAt: a middle mark → piecewise interpolation (each segment between ITS neighbours)", () => {
  const b = { id: 1, marks: { 0: [0, 10] as [number, number], 10: [20, 30] as [number, number], 30: [20, 30] as [number, number] } };
  assert.deepEqual(bandAt(b, 5), [10, 20]);
  assert.deepEqual(bandAt(b, 10), [20, 30]);
  assert.deepEqual(bandAt(b, 20), [20, 30]);
});

test("rounding is half-up (Math.round ≡ Python floor(x+0.5))", () => {
  const b = parseCropBands({ "0": [0, 10], "10": [1, 15] });   // lo at lat 5 = 0.5 → 1; hi = 12.5 → 13
  assert.deepEqual(bandsAt(b, 5), [{ id: 1, lo: 1, hi: 13 }]);
});

// ── resolution: several EXPLICIT bands — independent, never linked or merged ──────────────────────────────────
test("bandsAt: two bands overlapping in laterals, disjoint in frames → both listed, each on its own marks", () => {
  const b = parseCropBands({ bands: [
    { id: 1, marks: { "10": [0, 10], "30": [2, 12] } },
    { id: 2, marks: { "10": [50, 60], "30": [52, 62] } },
  ] });
  assert.deepEqual(bandsAt(b, 20), [{ id: 1, lo: 1, hi: 11 }, { id: 2, lo: 51, hi: 61 }]);
  assert.deepEqual(bandsAt(b, 10), [{ id: 1, lo: 0, hi: 10 }, { id: 2, lo: 50, hi: 60 }]);
  assert.deepEqual(bandsAt(b, 31), []);
});

test("bandsAt: same frames on separate lateral spans stay separate bands — nothing is bridged between them", () => {
  const b = parseCropBands({ bands: [
    { id: 1, marks: { "0": [20, 30], "40": [20, 30] } },
    { id: 2, marks: { "200": [20, 30], "240": [20, 30] } },
  ] });
  assert.deepEqual(bandsAt(b, 20), [{ id: 1, lo: 20, hi: 30 }]);
  assert.deepEqual(bandsAt(b, 100), []);
  assert.deepEqual(bandsAt(b, 220), [{ id: 2, lo: 20, hi: 30 }]);
});

test("bandsAt: two bands overlapping in BOTH laterals and frames → both listed (a list, not a union); inAnyBand is the union", () => {
  const b = parseCropBands({ bands: [
    { id: 1, marks: { "0": [10, 20], "20": [10, 20] } },
    { id: 2, marks: { "10": [15, 30], "30": [15, 30] } },
  ] });
  const at15 = bandsAt(b, 15);
  assert.deepEqual(at15, [{ id: 1, lo: 10, hi: 20 }, { id: 2, lo: 15, hi: 30 }]);
  assert.equal(inAnyBand(at15, 12), true);
  assert.equal(inAnyBand(at15, 25), true);
  assert.equal(inAnyBand(at15, 31), false);
  assert.equal(inAnyBand(at15, 9), false);
});

test("bandsAt: a slice carrying marks of two bands; one band's marks never influence another's interpolation", () => {
  const b = parseCropBands({ bands: [
    { id: 1, marks: { "10": [0, 5], "20": [0, 5], "30": [0, 5] } },
    { id: 2, marks: { "20": [40, 50], "40": [60, 70] } },
  ] });
  assert.deepEqual(bandsAt(b, 20), [{ id: 1, lo: 0, hi: 5 }, { id: 2, lo: 40, hi: 50 }]);
  assert.deepEqual(bandsAt(b, 30), [{ id: 1, lo: 0, hi: 5 }, { id: 2, lo: 50, hi: 60 }]);
  assert.deepEqual(bandsAt(b, 35), [{ id: 2, lo: 55, hi: 65 }]);
});

test("bandsAt: empty band contributes nothing; ids come out ascending regardless of array order", () => {
  const b: CropBands = { bands: [{ id: 3, marks: { 5: [1, 2] } }, { id: 2, marks: {} }, { id: 1, marks: { 5: [7, 9] } }] };
  assert.deepEqual(bandsAt(b, 5), [{ id: 1, lo: 7, hi: 9 }, { id: 3, lo: 1, hi: 2 }]);
});

test("nFrames clamps like the backend; a band entirely outside is dropped", () => {
  assert.deepEqual(bandsAt(parseCropBands({ "10": [-5, 3] }), 10, 10), [{ id: 1, lo: 0, hi: 3 }]);
  assert.deepEqual(bandsAt(parseCropBands({ "10": [8, 20] }), 10, 10), [{ id: 1, lo: 8, hi: 9 }]);
  assert.deepEqual(bandsAt(parseCropBands({ "10": [12, 20] }), 10, 10), []);
});

test("marks are validated + clamped BEFORE interpolation (backend order): clamped end mark, out-of-range lateral dropped", () => {
  // n_frames 40: the mark [35,45] on lateral 5 is clamped to [35,39] first, so lateral 4 = midpoint of [2,9] and [35,39]
  assert.deepEqual(bandsAt(parseCropBands({ "5": [35, 45], "3": [9, 2] }), 4, 40, 8), [{ id: 1, lo: 19, hi: 24 }]);
  // without dims the raw marks interpolate (the UI's marks are always in range, so both agree there)
  assert.deepEqual(bandsAt(parseCropBands({ "5": [35, 45], "3": [9, 2] }), 4), [{ id: 1, lo: 19, hi: 27 }]);
  // n_lateral 8: the mark on lateral 9 is dropped, leaving a single-mark band on lateral 4
  const b = parseCropBands({ "9": [1, 2], "4": [5, 7] });
  assert.deepEqual(bandsAt(b, 4, 40, 8), [{ id: 1, lo: 5, hi: 7 }]);
  assert.deepEqual(bandsAt(b, 5, 40, 8), []);
  assert.deepEqual(bandsAt(b, 6, 40, 8), []);
  // a mark entirely outside the frames is dropped
  assert.deepEqual(bandsAt(parseCropBands({ bands: [{ id: 1, marks: { "2": [50, 60], "6": [1, 3] } }] }), 4, 20, 10), []);
  assert.deepEqual(bandsAt(parseCropBands({ bands: [{ id: 1, marks: { "2": [50, 60], "6": [1, 3] } }] }), 6, 20, 10), [{ id: 1, lo: 1, hi: 3 }]);
});

// ── editing helpers ───────────────────────────────────────────────────────────────────────────────────────────
test("newBand / nextBandId: ids are max+1, never reused while a higher one exists", () => {
  const a = newBand(emptyCropBands());
  assert.equal(a.id, 1);
  assert.deepEqual(a.bands, { bands: [{ id: 1, marks: {} }] });
  const b = newBand({ bands: [{ id: 1, marks: {} }, { id: 4, marks: {} }] });
  assert.equal(b.id, 5);
  assert.equal(nextBandId(removeBand(b.bands, 5)), 5);
});

test("setMark replaces THAT band's mark on the slice (one mark per lateral per band), other bands untouched; missing id created", () => {
  let s: CropBands = { bands: [{ id: 1, marks: { 10: [0, 5] } }, { id: 2, marks: { 10: [40, 50] } }] };
  s = setMark(s, 1, 10, [9, 3]);
  assert.deepEqual(s, { bands: [{ id: 1, marks: { 10: [3, 9] } }, { id: 2, marks: { 10: [40, 50] } }] });
  s = setMark(s, 2, 30, [45, 55]);
  assert.deepEqual(findBand(s, 2)?.marks, { 10: [40, 50], 30: [45, 55] });
  s = setMark(s, 7, 0, [1, 2]);
  assert.deepEqual(findBand(s, 7), { id: 7, marks: { 0: [1, 2] } });
  assert.equal(countMarks(s), 4);
});

test("clearMark drops one mark and keeps the band; removeBand drops the band with all its marks", () => {
  const s: CropBands = { bands: [{ id: 1, marks: { 10: [0, 5], 20: [0, 5] } }, { id: 2, marks: { 10: [40, 50] } }] };
  const c = clearMark(s, 1, 10);
  assert.deepEqual(c, { bands: [{ id: 1, marks: { 20: [0, 5] } }, { id: 2, marks: { 10: [40, 50] } }] });
  assert.deepEqual(clearMark(c, 1, 20), { bands: [{ id: 1, marks: {} }, { id: 2, marks: { 10: [40, 50] } }] });
  assert.deepEqual(clearMark(s, 9, 10), s);
  assert.deepEqual(removeBand(s, 1), { bands: [{ id: 2, marks: { 10: [40, 50] } }] });
  assert.deepEqual(markedLaterals({ id: 1, marks: { 30: [0, 1], 5: [0, 1], 12: [0, 1] } }), [5, 12, 30]);
});

test("bandColor cycles a fixed palette by id; band 1 is the tool's blue", () => {
  assert.equal(bandColor(1).hex, "#5db0ff");
  assert.equal(bandColor(2).hex, BAND_PALETTE[1].hex);
  assert.equal(bandColor(1 + BAND_PALETTE.length).hex, BAND_PALETTE[0].hex);
  assert.equal(bandColor(0).hex, BAND_PALETTE[0].hex);
  assert.equal(new Set(BAND_PALETTE.map((p) => p.hex)).size, BAND_PALETTE.length);
});

// ── the cross-implementation fixture ──────────────────────────────────────────────────────────────────────────
interface FixtureCase {
  name?: string; label?: string; id?: string;
  n_lateral: number; n_frames: number;
  crop_bands: unknown;
  probes?: Record<string, Array<[number, number, number]>>;
  expected?: unknown;
}

function loadFixture(): { path: string; cases: FixtureCase[] } | null {
  const candidates = [process.env.CROP_BAND_FIXTURES, REPO_FIXTURE].filter((p): p is string => !!p);
  const p = candidates.find((c) => existsSync(c));
  if (!p) return null;
  const j = JSON.parse(readFileSync(p, "utf8")) as unknown;
  const cases = (Array.isArray(j) ? j
    : (j as { cases?: unknown[]; fixtures?: unknown[] }).cases ?? (j as { fixtures?: unknown[] }).fixtures ?? []) as FixtureCase[];
  return { path: p, cases };
}

const fixture = loadFixture();

test("fixture file is present and in the EXPLICIT format (probes: {lateral: [[lo, hi, id], ...]})", () => {
  assert.ok(fixture, `no crop_band_fixtures.json at $CROP_BAND_FIXTURES or ${REPO_FIXTURE}`);
  assert.ok(fixture!.cases.length > 0, "fixture has no cases");
  const explicit = fixture!.cases.filter((c) => c.probes && typeof c.probes === "object");
  assert.equal(explicit.length, fixture!.cases.length,
    `${fixture!.cases.length - explicit.length}/${fixture!.cases.length} case(s) have no 'probes' — fixture still in the implicit (expected:) format?`);
});

test("bandsAt agrees with the backend resolve_crop_bands on every fixture probe (lo, hi, id)", { skip: !fixture }, () => {
  const mismatches: string[] = [];
  let probes = 0;
  const key = (t: [number, number, number]) => `${t[2]}:${t[0]}-${t[1]}`;
  const sortTriples = (ts: Array<[number, number, number]>) =>
    ts.slice().sort((a, b) => a[2] - b[2] || a[0] - b[0] || a[1] - b[1]);
  for (const [ci, c] of fixture!.cases.entries()) {
    const label = c.name ?? c.label ?? c.id ?? `case#${ci}`;
    const bands = parseCropBands(c.crop_bands);
    for (const [latKey, exp] of Object.entries(c.probes ?? {})) {
      const lat = Number(latKey);
      const got = sortTriples(bandsAt(bands, lat, c.n_frames, c.n_lateral).map((b): [number, number, number] => [b.lo, b.hi, b.id]));
      const want = sortTriples((exp ?? []).map((t): [number, number, number] => [Number(t[0]), Number(t[1]), Number(t[2])]));
      probes += 1;
      if (got.map(key).join(" ") !== want.map(key).join(" "))
        mismatches.push(`${label} lat ${lat}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`);
    }
  }
  assert.ok(probes > 0, "fixture has no probe laterals");
  assert.deepEqual(mismatches, [], `${mismatches.length}/${probes} probe(s) differ:\n` + mismatches.join("\n"));
});
