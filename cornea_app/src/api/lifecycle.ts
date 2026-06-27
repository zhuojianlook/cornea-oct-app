/* Per-scan lifecycle model — the single source of truth for the progress TIMELINE (TimelineBar) and the
   colour-coded scan entries (OctLoader). A scan advances linearly; each step requires the previous, so a
   later flag set while an earlier one is cleared (e.g. a re-preprocess resets preproc_vetted) correctly
   drops the scan back. Colours per the spec: raw=grey, auto=red, vetted=orange, classified=yellow,
   SAM2-auto=light blue, SAM2-corrected=dark blue, scheduled=green. */

export type LifecycleStep = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;

export interface StepMeta { step: LifecycleStep; color: string; label: string; short: string; }

// Index = step number. step 0 = no scan loaded.
export const LIFECYCLE_STEPS: { color: string; label: string; short: string }[] = [
  { color: "transparent", label: "—", short: "—" },
  { color: "#7d8794", label: "Raw image", short: "Raw" },                       // 1 grey
  { color: "#ef4444", label: "Preprocessed · automatic", short: "Auto" },       // 2 red
  { color: "#f59e0b", label: "Preprocessed · manually vetted", short: "Vetted" }, // 3 orange
  { color: "#eab308", label: "Scar / control + replicates set", short: "Classified" }, // 4 yellow
  { color: "#38bdf8", label: "SAM2 · cornea + scar", short: "SAM2" },           // 5 light blue
  { color: "#14b8a6", label: "Replicates aligned + normalized", short: "Aligned" }, // 6 teal
  { color: "#2563eb", label: "SAM2 · manually corrected", short: "Corrected" },  // 7 dark blue
  { color: "#22c55e", label: "Scheduled for training", short: "Scheduled" },     // 8 green
];

type Manifest = Record<string, unknown> | null | undefined;
const set = (m: NonNullable<Manifest>, k: string) => m[k] != null && m[k] !== false && m[k] !== "";

/** The current (highest) lifecycle step a scan's manifest has reached (8-step model: Raw→Auto→Vetted→
 *  Classified→SAM2→Aligned→Corrected→Scheduled). */
export function scanStep(m: Manifest): LifecycleStep {
  if (!m) return 0;
  if (!set(m, "input_volume") && !set(m, "corrected_volume")) return 0;
  // A BUILT CONSENSUS case (the aligned average of replicates) IS the "Aligned" artifact (step 6); it
  // never runs preprocess/vet/classify. It can then advance to corrected (7) / scheduled (8).
  if (set(m, "consensus_cases") || set(m, "consensus_report")) {
    if (set(m, "training_scheduled")) return 8;
    if (set(m, "corrected_labelmap")) return 7;
    return 6;
  }
  if (!set(m, "oct_preprocessed")) return 1;                 // raw only
  // A SEGMENTED scan is past preprocess/vet/classify regardless of a missing preproc_vetted flag (e.g. a
  // consensus member opened via "Correct"). consensus_case (the link to this eye's built consensus) means
  // the scan has been ALIGNED → step 6.
  if (set(m, "sam2_meta") || set(m, "consensus_case") || set(m, "corrected_labelmap")) {
    if (set(m, "training_scheduled")) return 8;              // scheduled (green)
    if (set(m, "corrected_labelmap")) return 7;             // manually corrected (dark blue)
    if (set(m, "consensus_case")) return 6;                 // aligned to the eye's consensus (teal)
    return 5;                                                // SAM2 cornea+scar (light blue)
  }
  if (!set(m, "preproc_vetted")) return 2;                   // auto-preprocessed (red)
  if (!set(m, "scar_classification")) return 3;              // vetted (orange)
  return 4;                                                  // classified (yellow)
}

export function lifecycleMeta(m: Manifest): StepMeta {
  const step = scanStep(m);
  return { step, ...LIFECYCLE_STEPS[step] };
}

/** Has SAM2 cornea segmentation been produced? (drives the Segmentation/Slices toggle greying.) */
export function hasSegmentation(m: Manifest): boolean {
  return !!m && (set(m, "sam2_meta") || set(m, "consensus_case") || set(m, "corrected_labelmap"));
}

/** Is the scan classified (scar/control set)? (gates running SAM2 — "wait to be labelled".) */
export function isClassified(m: Manifest): boolean {
  return !!m && set(m, "scar_classification");
}
