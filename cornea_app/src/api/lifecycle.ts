/* Per-scan lifecycle model — the single source of truth for the progress TIMELINE (TimelineBar) and the
   colour-coded scan entries (OctLoader). A scan advances linearly; each step requires the previous, so a
   later flag set while an earlier one is cleared (e.g. a re-preprocess resets preproc_vetted) correctly
   drops the scan back. Colours follow a smooth monotonic spectral ramp (see LIFECYCLE_STEPS): idle slate →
   red → pink → … → blue → … → green (done), so the strip reads as a natural progression. */

export type LifecycleStep = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;

export interface StepMeta { step: LifecycleStep; color: string; label: string; short: string; }

// Index = step number. step 0 = no scan loaded.
// Colours follow a SMOOTH MONOTONIC SPECTRAL RAMP so the strip reads as a natural progression (no jarring
// hue jumps): idle slate → red (needs work) → pink → fuchsia → purple → violet → indigo → blue → sky → cyan
// → teal → green (done). All bright 400-shades for good contrast with the dark pill text.
export const LIFECYCLE_STEPS: { color: string; label: string; short: string }[] = [
  { color: "transparent", label: "—", short: "—" },
  { color: "#94a3b8", label: "Raw image", short: "Raw" },                          // 1 slate (idle)
  { color: "#f87171", label: "Preprocessed · automatic", short: "Auto" },          // 2 red (needs vetting)
  { color: "#f472b6", label: "Preprocessed · manually vetted", short: "Vetted" },  // 3 pink
  { color: "#e879f9", label: "Cornea segmented (SAM2)", short: "Cornea" },          // 4 fuchsia
  { color: "#c084fc", label: "Cornea/background vetted (paint)", short: "Cornea✓" }, // 5 purple
  { color: "#a78bfa", label: "Scar / control classified", short: "Classified" },    // 6 violet
  { color: "#818cf8", label: "Subgroup assigned", short: "Subgroup" },            // 7 indigo (BEFORE scar)
  { color: "#60a5fa", label: "Scar segmented", short: "Scar" },                    // 8 blue
  { color: "#38bdf8", label: "Replicates aligned", short: "Aligned" },            // 9 sky
  { color: "#22d3ee", label: "Normalized against controls", short: "Normalized" }, // 10 cyan
  { color: "#2dd4bf", label: "Manually corrected", short: "Corrected" },           // 11 teal
  { color: "#4ade80", label: "Scheduled for training", short: "Scheduled" },       // 12 green (done)
];

type Manifest = Record<string, unknown> | null | undefined;
const set = (m: NonNullable<Manifest>, k: string) => m[k] != null && m[k] !== false && m[k] !== "";

/** The current (highest) lifecycle step a scan's manifest has reached (Raw→Auto→Vetted→Cornea→Cornea✓→
 *  Classified→Subgroup→Scar→Aligned→Normalized→Corrected→Scheduled). Classification (scar/control) comes AFTER
 *  cornea-vetting — it gates only the scar branch, not SAM2. Subgroup is assigned BEFORE scar so the
 *  per-subgroup strategy comparison is available at the Scar step. Cornea (SAM2) and Scar are separate steps. */
/** A no-scar (control) scan: it contributes a cornea-only training label + the normal baseline, so the scar
 *  steps (Subgroup 7, Scar 8, Aligned 9, Normalized 10, Corrected 11) do not apply — it goes Cornea✓ →
 *  Scheduled. */
export function isControl(m: Manifest): boolean {
  return !!m && m["scar_classification"] === "control";
}

export function scanStep(m: Manifest): LifecycleStep {
  if (!m) return 0;
  if (!set(m, "input_volume") && !set(m, "corrected_volume")) return 0;
  // A BUILT CONSENSUS case is the ALIGNED artifact (step 9); normalize/correct/schedule act on it.
  if (set(m, "consensus_cases") || set(m, "consensus_report")) {
    if (set(m, "training_scheduled")) return 12;
    if (set(m, "corrected_labelmap")) return 11;
    if (set(m, "normalized")) return 10;
    return 9;
  }
  if (!set(m, "oct_preprocessed")) return 1;                 // raw only
  // A SEGMENTED per-scan scan: Cornea(4, sam2_meta) → Cornea/bg vetted(5, cornea_vetted) → Classified(6,
  // scar_classification) → Subgroup(7) → Scar(8) → Aligned(9). Classification now comes AFTER cornea-vetting
  // (it gates only the scar branch, not SAM2): a control is READY to schedule once classified; a scar scan
  // proceeds to subgroup. Normalize(10) acts on the consensus case, so a member tops out at 9 (or 11/12 if its
  // own labelmap was corrected / scheduled).
  if (set(m, "sam2_meta") || set(m, "consensus_case") || set(m, "corrected_labelmap")) {
    if (set(m, "training_scheduled")) return 12;            // scheduled (green)
    // A CONTROL has no scar/subgroup/align/normalize/correct: once classified it is READY to schedule (steps
    // 7-11 do not apply). It never advances to 7-11 (those flags are ignored for it).
    if (isControl(m)) return 6;                             // classified control (violet) → ready to schedule
    if (set(m, "corrected_labelmap")) return 11;           // manually corrected (teal)
    if (set(m, "consensus_case")) return 9;                // aligned to the eye's consensus (sky)
    if (set(m, "scar_done")) return 8;                     // scar segmented (blue) — AFTER subgroup
    if (set(m, "subgroup_confirmed")) return 7;            // subgroup assigned (indigo) — BEFORE scar
    if (set(m, "scar_classification")) return 6;           // classified scar (violet) — next is subgroup
    if (set(m, "cornea_vetted")) return 5;                 // cornea/background paint-vetted (purple)
    return 4;                                               // cornea segmented, awaiting vet (fuchsia)
  }
  if (!set(m, "preproc_vetted")) return 2;                   // auto-preprocessed (red)
  return 3;                                                  // vetted, awaiting SAM2 cornea (pink)
}

export function lifecycleMeta(m: Manifest): StepMeta {
  const step = scanStep(m);
  return { step, ...LIFECYCLE_STEPS[step] };
}

/** Whether step `i` has GENUINELY been reached (its own flag is set) — used to colour the timeline
 *  strip honestly: a scan scheduled straight from SAM2 must NOT show Aligned/Corrected as done.
 *  A built consensus case is a finished artifact, so its earlier steps are treated as implicitly done. */
export function stepReached(m: Manifest, i: LifecycleStep): boolean {
  if (!m) return false;
  if (set(m, "consensus_cases") || set(m, "consensus_report")) return i <= scanStep(m);
  // A control skips the scar steps entirely — never colour 7-11 as reached for it (it goes Cornea✓ → Scheduled).
  if (isControl(m) && i >= 7 && i <= 11) return false;
  switch (i) {
    case 1: return set(m, "input_volume") || set(m, "corrected_volume");
    case 2: return set(m, "oct_preprocessed");
    case 3: return set(m, "preproc_vetted");
    case 4: return set(m, "sam2_meta") || set(m, "corrected_labelmap") || set(m, "consensus_case");   // cornea segmented
    // cornea/bg vetted — implied done once any LATER scar-branch step (subgroup/scar/aligned/corrected) is reached
    case 5: return set(m, "cornea_vetted") || set(m, "subgroup_confirmed") || set(m, "scar_done") || set(m, "consensus_case") || set(m, "corrected_labelmap");
    case 6: return set(m, "scar_classification");   // classified (scar/control) — now AFTER cornea✓
    // subgroup (7) — its OWN flag, or a consensus (built per-subgroup implies it). NOT scar_done: a CONTROL
    // skips subgroup and sets scar_done directly, so scar_done must not falsely colour subgroup as reached.
    case 7: return set(m, "subgroup_confirmed") || set(m, "consensus_case");
    // scar (8) — scar_done, a consensus (votes on scar), or a corrected labelmap (it carries scar labels)
    case 8: return set(m, "scar_done") || set(m, "consensus_case") || set(m, "corrected_labelmap");
    case 9: return set(m, "consensus_case");
    case 10: return set(m, "normalized");
    case 11: return set(m, "corrected_labelmap");
    case 12: return set(m, "training_scheduled");
    default: return false;
  }
}

/** Whether step `i` APPLIES to this scan. For a control the scar steps (7-11: Subgroup/Scar/Aligned/
 *  Normalized/Corrected) are not applicable — the timeline shows them greyed/"—" and a control advances
 *  Cornea✓ (6) → Scheduled (12). Everything applies to scar scans + consensus cases. */
export function stepApplicable(m: Manifest, i: LifecycleStep): boolean {
  return !(isControl(m) && i >= 7 && i <= 11);
}

/** Has SAM2 cornea segmentation been produced? (drives the Segmentation/Slices toggle greying.) */
export function hasSegmentation(m: Manifest): boolean {
  return !!m && (set(m, "sam2_meta") || set(m, "consensus_case") || set(m, "corrected_labelmap"));
}

/** Is the scan classified (scar/control set)? (gates the SCAR branch — Subgroup/Scar — not SAM2.) */
export function isClassified(m: Manifest): boolean {
  return !!m && set(m, "scar_classification");
}

// ── crop-approval proposals ────────────────────────────────────────────────
// Preprocessing may DETECT an auto de-tilt / off-cornea crop / clipped-apex surface-crop but leave the
// output volume UNCORRECTED, recording the finding in manifest.oct_proposals for the user to approve. The UI
// highlights the proposed crop region in pink + glows the Fix-columns / Crop-region controls, and the Approve
// button re-preprocesses with apply_proposals:true (baking the corrections) before vetting. See the backend
// contract in api_server.py (oct_proposals / apply_proposals).
export interface OctProposals {
  hasProposal: boolean;      // any of de-tilt / crop-region / surface-crop was detected but not applied
  hasDetilt: boolean;        // an automatic de-tilt was proposed
  hasCropRegion: boolean;    // an off-cornea lateral crop-region was proposed
  hasSurfaceCrop: boolean;   // clipped-apex surface-crop frames were proposed
  frames: number[];          // union of the proposed crop-region + surface-crop frame indices (slow axis)
  cropLateral: [number, number] | null;  // the proposed crop-region's lateral [lo, hi] slice range, if any
  reasons: string[];         // human-readable reason strings from the proposals (for the banner/tooltip)
}

/** Read manifest.oct_proposals into a flat, UI-friendly shape. Returns an all-empty proposal set when there
 *  is nothing to approve (no manifest / null oct_proposals / all sub-proposals null), so a scan with no
 *  proposal behaves exactly as before. */
export function octProposals(m: Manifest): OctProposals {
  const empty: OctProposals = { hasProposal: false, hasDetilt: false, hasCropRegion: false, hasSurfaceCrop: false, frames: [], cropLateral: null, reasons: [] };
  const raw = m ? (m as Record<string, unknown>).oct_proposals : null;
  if (!raw || typeof raw !== "object") return empty;
  const p = raw as Record<string, unknown>;
  const detilt = (p.detilt ?? null) as Record<string, unknown> | null;
  const cropRegion = (p.crop_region ?? null) as Record<string, unknown> | null;
  const surfaceCrop = (p.surface_crop ?? null) as Record<string, unknown> | null;
  const frameSet = new Set<number>();
  const readFrames = (o: Record<string, unknown> | null) => {
    const fs = o && Array.isArray(o.frames) ? (o.frames as unknown[]) : [];
    for (const f of fs) { const n = Number(f); if (Number.isFinite(n)) frameSet.add(Math.round(n)); }
  };
  readFrames(cropRegion);
  readFrames(surfaceCrop);
  let cropLateral: [number, number] | null = null;
  if (cropRegion && Array.isArray(cropRegion.lateral) && (cropRegion.lateral as unknown[]).length === 2) {
    const lo = Number((cropRegion.lateral as unknown[])[0]), hi = Number((cropRegion.lateral as unknown[])[1]);
    if (Number.isFinite(lo) && Number.isFinite(hi)) cropLateral = [Math.round(lo), Math.round(hi)];
  }
  const reasons: string[] = [];
  for (const o of [cropRegion, surfaceCrop]) {
    const r = o && typeof o.reason === "string" ? (o.reason as string) : "";
    if (r) reasons.push(r);
  }
  const hasDetilt = detilt != null;
  const hasCropRegion = cropRegion != null;
  const hasSurfaceCrop = surfaceCrop != null;
  return {
    hasProposal: hasDetilt || hasCropRegion || hasSurfaceCrop,
    hasDetilt, hasCropRegion, hasSurfaceCrop,
    frames: [...frameSet].sort((a, b) => a - b),
    cropLateral, reasons,
  };
}
