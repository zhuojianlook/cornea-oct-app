// Unit tests for src/api/lifecycle.ts — the per-scan lifecycle model (13 steps). Step 4 "Aligned" is the
// group-wise 3D alignment of a patient+eye group's replicate scans (regularising their sagittal curvature) and
// sits BETWEEN Vetted (3) and Cornea/SAM2 (5); the old scar-consensus "Aligned" is step 10 "Scar-aligned".
//
// Run (no test runner is installed; Node ≥22.18 strips the types itself):
//   cd cornea_app && node --test tests/unit/
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  LIFECYCLE_STEPS, hasSegmentation, isControl, lifecycleMeta, scanStep, stepApplicable, stepReached,
  type LifecycleStep,
} from "../../src/api/lifecycle.ts";

type M = Record<string, unknown>;
const base: M = { input_volume: "/cases/x/previews/volume.nii.gz", oct_preprocessed: true };
const vetted: M = { ...base, preproc_vetted: true };
const aligned: M = { ...vetted, group_aligned: { ts: "2026-09-10T00:00:00", group: "cs001_od", percent_match: 97.5 } };
const sam2: M = { ...aligned, sam2_meta: { vote: 2 } };
const control: M = { ...sam2, cornea_vetted: true, scar_classification: "control" };

const STEPS = LIFECYCLE_STEPS.map((_, i) => i as LifecycleStep);
// Steps the STRIP would colour (it renders 1..13; index 0 = no scan is never drawn).
const reached = (m: M | null) => STEPS.filter((i) => i > 0 && stepReached(m, i));
const range = (lo: number, hi: number) => STEPS.filter((i) => i >= lo && i <= hi);

test("table: 13 steps, Aligned inserted at 4, the scar consensus renamed Scar-aligned at 10", () => {
  assert.equal(LIFECYCLE_STEPS.length, 14);                       // index 0 = no scan
  assert.equal(LIFECYCLE_STEPS[3].short, "Vetted");
  assert.equal(LIFECYCLE_STEPS[4].short, "Aligned");
  assert.match(LIFECYCLE_STEPS[4].label, /curvature/i);
  assert.equal(LIFECYCLE_STEPS[5].short, "Cornea");
  assert.equal(LIFECYCLE_STEPS[6].short, "Cornea✓");
  assert.equal(LIFECYCLE_STEPS[7].short, "Classified");
  assert.equal(LIFECYCLE_STEPS[8].short, "Subgroup");
  assert.equal(LIFECYCLE_STEPS[9].short, "Scar");
  assert.equal(LIFECYCLE_STEPS[10].short, "Scar-aligned");
  assert.match(LIFECYCLE_STEPS[10].label, /^Scar replicates aligned/);
  assert.equal(LIFECYCLE_STEPS[11].short, "Normalized");
  assert.equal(LIFECYCLE_STEPS[12].short, "Corrected");
  assert.equal(LIFECYCLE_STEPS[13].short, "Scheduled");
  // Every step has its own short name + colour (the two "aligned" steps must be distinguishable).
  const shorts = LIFECYCLE_STEPS.map((s) => s.short);
  assert.equal(new Set(shorts).size, shorts.length);
  const colours = LIFECYCLE_STEPS.slice(1).map((s) => s.color);
  assert.equal(new Set(colours).size, colours.length);
  assert.match(LIFECYCLE_STEPS[4].color, /^#[0-9a-f]{6}$/i);
});

test("no manifest / raw / auto", () => {
  assert.equal(scanStep(null), 0);
  assert.equal(scanStep(undefined), 0);
  assert.equal(scanStep({}), 0);
  assert.equal(scanStep({ input_volume: "/x.nii.gz" }), 1);
  assert.equal(scanStep(base), 2);
  assert.deepEqual(reached(base), [1, 2]);
});

test("a vetted scan sits at 3 (Vetted) until group_aligned is set", () => {
  assert.equal(scanStep(vetted), 3);
  assert.equal(lifecycleMeta(vetted).short, "Vetted");
  assert.deepEqual(reached(vetted), [1, 2, 3]);
  assert.equal(stepReached(vetted, 4), false);
  // Empty / cleared values do not count as aligned.
  for (const v of [null, false, ""]) assert.equal(scanStep({ ...vetted, group_aligned: v }), 3, `group_aligned=${String(v)}`);
  assert.equal(hasSegmentation(vetted), false);
});

test("group_aligned advances a vetted scan to 4 (Aligned, awaiting SAM2)", () => {
  assert.equal(scanStep(aligned), 4);
  assert.equal(lifecycleMeta(aligned).short, "Aligned");
  assert.deepEqual(reached(aligned), [1, 2, 3, 4]);
  assert.equal(stepReached(aligned, 5), false);
  assert.equal(hasSegmentation(aligned), false);
  // `true` is enough (cases/list carries the flag as a boolean).
  assert.equal(scanStep({ ...vetted, group_aligned: true }), 4);
  // Each step requires the previous: a cleared preproc_vetted drops the scan back to Auto even though
  // group_aligned is still set (e.g. after a re-preprocess).
  assert.equal(scanStep({ ...base, group_aligned: true }), 2);
  assert.equal(stepReached({ ...base, group_aligned: true }, 4), true);   // its own flag IS set (honest strip)
});

test("SAM2 implies 5+ (Cornea) and marks Aligned as reached even without its own flag", () => {
  assert.equal(scanStep(sam2), 5);
  assert.deepEqual(reached(sam2), [1, 2, 3, 4, 5]);
  assert.equal(hasSegmentation(sam2), true);
  const sam2NoAlign: M = { ...vetted, sam2_meta: { vote: 2 } };
  assert.equal(scanStep(sam2NoAlign), 5);
  assert.equal(stepReached(sam2NoAlign, 4), true);
  // The scar branch, step by step.
  const cv: M = { ...sam2, cornea_vetted: true };
  assert.equal(scanStep(cv), 6);
  const cls: M = { ...cv, scar_classification: "scar" };
  assert.equal(scanStep(cls), 7);
  const sub: M = { ...cls, subgroup_confirmed: true };
  assert.equal(scanStep(sub), 8);
  const scar: M = { ...sub, scar_done: true };
  assert.equal(scanStep(scar), 9);
  assert.deepEqual(reached(scar), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const cons: M = { ...scar, consensus_case: "case_cs001_od_consensus" };
  assert.equal(scanStep(cons), 10);
  assert.equal(lifecycleMeta(cons).short, "Scar-aligned");
  assert.equal(scanStep({ ...cons, corrected_labelmap: "/lab.nii.gz" }), 12);
  assert.equal(scanStep({ ...cons, training_scheduled: true }), 13);
  // A scan scheduled straight from SAM2 must NOT show the skipped steps as reached (honest strip).
  const early: M = { ...sam2, training_scheduled: true };
  assert.equal(scanStep(early), 13);
  assert.deepEqual(reached(early), [1, 2, 3, 4, 5, 13]);
  for (const i of range(1, 13)) assert.equal(stepApplicable(scar, i), true);
});

test("control: Classified (7) → Scheduled (13); the scar steps 8-12 do not apply", () => {
  assert.equal(isControl(control), true);
  assert.equal(scanStep(control), 7);
  assert.equal(lifecycleMeta(control).short, "Classified");
  for (const i of range(8, 12)) assert.equal(stepApplicable(control, i), false, `step ${i}`);
  for (const i of [...range(1, 7), 13]) assert.equal(stepApplicable(control, i), true, `step ${i}`);
  // Stray scar-branch flags on a control never advance it, nor colour 8-12.
  const noisy: M = { ...control, subgroup_confirmed: true, scar_done: true, consensus_case: "c" };
  assert.equal(scanStep(noisy), 7);
  for (const i of range(8, 12)) assert.equal(stepReached(noisy, i), false, `step ${i}`);
  assert.equal(scanStep({ ...control, training_scheduled: true }), 13);
  assert.deepEqual(reached({ ...control, training_scheduled: true }), [1, 2, 3, 4, 5, 6, 7, 13]);
});

test("a built consensus case is the Scar-aligned artifact (10) → Normalized → Corrected → Scheduled", () => {
  const cons: M = { ...base, consensus_cases: ["a", "b"], consensus_report: { reference: "a" } };
  assert.equal(scanStep(cons), 10);
  assert.deepEqual(reached(cons), range(1, 10));
  assert.equal(scanStep({ ...cons, normalized: true }), 11);
  assert.equal(scanStep({ ...cons, corrected_labelmap: "/lab.nii.gz" }), 12);
  assert.equal(scanStep({ ...cons, training_scheduled: true }), 13);
});
