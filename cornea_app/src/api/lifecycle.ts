/* Per-scan lifecycle model — the single source of truth for the progress TIMELINE (TimelineBar) and the
   colour-coded scan entries (OctLoader). A scan advances linearly; each step requires the previous, so a
   later flag set while an earlier one is cleared (e.g. a re-preprocess resets preproc_vetted) correctly
   drops the scan back. Colours per the spec: raw=grey, auto=red, vetted=orange, classified=yellow,
   SAM2-auto=light blue, SAM2-corrected=dark blue, scheduled=green. */

export type LifecycleStep = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10;

export interface StepMeta { step: LifecycleStep; color: string; label: string; short: string; }

// Index = step number. step 0 = no scan loaded.
export const LIFECYCLE_STEPS: { color: string; label: string; short: string }[] = [
  { color: "transparent", label: "—", short: "—" },
  { color: "#7d8794", label: "Raw image", short: "Raw" },                       // 1 grey
  { color: "#ef4444", label: "Preprocessed · automatic", short: "Auto" },       // 2 red
  { color: "#f59e0b", label: "Preprocessed · manually vetted", short: "Vetted" }, // 3 orange
  { color: "#eab308", label: "Scar / control classified", short: "Classified" }, // 4 yellow
  { color: "#38bdf8", label: "Cornea + scar segmented", short: "Segmented" },    // 5 light blue
  { color: "#a855f7", label: "Subgroup assigned", short: "Subgroup" },          // 6 purple
  { color: "#14b8a6", label: "Replicates aligned", short: "Aligned" },          // 7 teal
  { color: "#06b6d4", label: "Normalized against controls", short: "Normalized" }, // 8 cyan
  { color: "#2563eb", label: "Manually corrected", short: "Corrected" },         // 9 dark blue
  { color: "#22c55e", label: "Scheduled for training", short: "Scheduled" },     // 10 green
];

type Manifest = Record<string, unknown> | null | undefined;
const set = (m: NonNullable<Manifest>, k: string) => m[k] != null && m[k] !== false && m[k] !== "";

/** The current (highest) lifecycle step a scan's manifest has reached (10-step model: Raw→Auto→Vetted→
 *  Classified→SAM2→Subgroup→Aligned→Normalized→Corrected→Scheduled). */
export function scanStep(m: Manifest): LifecycleStep {
  if (!m) return 0;
  if (!set(m, "input_volume") && !set(m, "corrected_volume")) return 0;
  // A BUILT CONSENSUS case is the ALIGNED artifact (step 7); normalize/correct/schedule act on it.
  if (set(m, "consensus_cases") || set(m, "consensus_report")) {
    if (set(m, "training_scheduled")) return 10;
    if (set(m, "corrected_labelmap")) return 9;
    if (set(m, "normalized")) return 8;
    return 7;
  }
  if (!set(m, "oct_preprocessed")) return 1;                 // raw only
  // A SEGMENTED per-scan scan: SAM2(5) → Subgroup confirmed(6) → linked to a consensus = Aligned(7).
  // Normalize(8) acts on the consensus case, not the member, so a member tops out at 7 (or 9/10 if its
  // own labelmap was corrected/scheduled directly).
  if (set(m, "sam2_meta") || set(m, "consensus_case") || set(m, "corrected_labelmap")) {
    if (set(m, "training_scheduled")) return 10;             // scheduled (green)
    if (set(m, "corrected_labelmap")) return 9;             // manually corrected (dark blue)
    if (set(m, "consensus_case")) return 7;                 // aligned to the eye's consensus (teal)
    if (set(m, "subgroup_confirmed")) return 6;             // subgroup assigned (purple)
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
    case 6: return set(m, "subgroup_confirmed") || set(m, "consensus_case");   // aligned implies subgroup done
    case 7: return set(m, "consensus_case");
    case 8: return set(m, "normalized");
    case 9: return set(m, "corrected_labelmap");
    case 10: return set(m, "training_scheduled");
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
