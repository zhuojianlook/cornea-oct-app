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
  { color: "#e879f9", label: "Scar / control classified", short: "Classified" },   // 4 fuchsia
  { color: "#c084fc", label: "Cornea segmented (SAM2)", short: "Cornea" },         // 5 purple
  { color: "#a78bfa", label: "Cornea/background vetted (paint)", short: "Cornea✓" }, // 6 violet
  { color: "#818cf8", label: "Subgroup assigned", short: "Subgroup" },            // 7 indigo (BEFORE scar)
  { color: "#60a5fa", label: "Scar segmented", short: "Scar" },                    // 8 blue
  { color: "#38bdf8", label: "Replicates aligned", short: "Aligned" },            // 9 sky
  { color: "#22d3ee", label: "Normalized against controls", short: "Normalized" }, // 10 cyan
  { color: "#2dd4bf", label: "Manually corrected", short: "Corrected" },           // 11 teal
  { color: "#4ade80", label: "Scheduled for training", short: "Scheduled" },       // 12 green (done)
];

type Manifest = Record<string, unknown> | null | undefined;
const set = (m: NonNullable<Manifest>, k: string) => m[k] != null && m[k] !== false && m[k] !== "";

/** The current (highest) lifecycle step a scan's manifest has reached (Raw→Auto→Vetted→Classified→Cornea→
 *  Cornea✓→Subgroup→Scar→Aligned→Normalized→Corrected→Scheduled). Subgroup is assigned BEFORE scar so the
 *  per-subgroup strategy comparison is available at the Scar step. Cornea (SAM2) and Scar are separate steps. */
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
  // A SEGMENTED per-scan scan: Cornea(5, sam2_meta) → Cornea/bg vetted(6, cornea_vetted) → Subgroup(7,
  // subgroup_confirmed) → Scar(8, scar_done) → Aligned(9, consensus_case link). Subgroup is assigned BEFORE
  // scar so the strategy comparison at the Scar step is per-subgroup. Normalize(10) acts on the consensus
  // case, not the member, so a member tops out at 9 (or 11/12 if its own labelmap was corrected / scheduled).
  if (set(m, "sam2_meta") || set(m, "consensus_case") || set(m, "corrected_labelmap")) {
    if (set(m, "training_scheduled")) return 12;            // scheduled (green)
    if (set(m, "corrected_labelmap")) return 11;           // manually corrected (dark blue)
    if (set(m, "consensus_case")) return 9;                // aligned to the eye's consensus (teal)
    if (set(m, "scar_done")) return 8;                     // scar segmented (rose) — AFTER subgroup
    if (set(m, "subgroup_confirmed")) return 7;            // subgroup assigned (purple) — BEFORE scar
    if (set(m, "cornea_vetted")) return 6;                 // cornea/background paint-vetted (indigo)
    return 5;                                               // cornea segmented, awaiting vet (light blue)
  }
  if (!set(m, "preproc_vetted")) return 2;                   // auto-preprocessed (red)
  if (!set(m, "scar_classification")) return 3;              // vetted (orange)
  return 4;                                                  // classified (yellow)
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
  switch (i) {
    case 1: return set(m, "input_volume") || set(m, "corrected_volume");
    case 2: return set(m, "oct_preprocessed");
    case 3: return set(m, "preproc_vetted");
    case 4: return set(m, "scar_classification");
    case 5: return set(m, "sam2_meta") || set(m, "corrected_labelmap") || set(m, "consensus_case");
    // cornea/bg vetted — implied done once any LATER step (subgroup/scar/aligned/corrected) is reached
    case 6: return set(m, "cornea_vetted") || set(m, "subgroup_confirmed") || set(m, "scar_done") || set(m, "consensus_case") || set(m, "corrected_labelmap");
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

/** Has SAM2 cornea segmentation been produced? (drives the Segmentation/Slices toggle greying.) */
export function hasSegmentation(m: Manifest): boolean {
  return !!m && (set(m, "sam2_meta") || set(m, "consensus_case") || set(m, "corrected_labelmap"));
}

/** Is the scan classified (scar/control set)? (gates running SAM2 — "wait to be labelled".) */
export function isClassified(m: Manifest): boolean {
  return !!m && set(m, "scar_classification");
}
