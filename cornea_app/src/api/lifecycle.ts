/* Per-scan lifecycle model — the single source of truth for the progress TIMELINE (TimelineBar) and the
   colour-coded scan entries (OctLoader). A scan advances linearly; each step requires the previous, so a
   later flag set while an earlier one is cleared (e.g. a re-preprocess resets preproc_vetted) correctly
   drops the scan back. Colours per the spec: raw=grey, auto=red, vetted=orange, classified=yellow,
   SAM2-auto=light blue, SAM2-corrected=dark blue, scheduled=green. */

export type LifecycleStep = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11;

export interface StepMeta { step: LifecycleStep; color: string; label: string; short: string; }

// Index = step number. step 0 = no scan loaded.
export const LIFECYCLE_STEPS: { color: string; label: string; short: string }[] = [
  { color: "transparent", label: "—", short: "—" },
  { color: "#7d8794", label: "Raw image", short: "Raw" },                       // 1 grey
  { color: "#ef4444", label: "Preprocessed · automatic", short: "Auto" },       // 2 red
  { color: "#f59e0b", label: "Preprocessed · manually vetted", short: "Vetted" }, // 3 orange
  { color: "#eab308", label: "Scar / control classified", short: "Classified" }, // 4 yellow
  { color: "#38bdf8", label: "Cornea segmented (SAM2)", short: "Cornea" },        // 5 light blue
  { color: "#f43f5e", label: "Scar segmented", short: "Scar" },                  // 6 rose
  { color: "#a855f7", label: "Subgroup assigned", short: "Subgroup" },          // 7 purple
  { color: "#14b8a6", label: "Replicates aligned", short: "Aligned" },          // 8 teal
  { color: "#06b6d4", label: "Normalized against controls", short: "Normalized" }, // 9 cyan
  { color: "#2563eb", label: "Manually corrected", short: "Corrected" },         // 10 dark blue
  { color: "#22c55e", label: "Scheduled for training", short: "Scheduled" },     // 11 green
];

type Manifest = Record<string, unknown> | null | undefined;
const set = (m: NonNullable<Manifest>, k: string) => m[k] != null && m[k] !== false && m[k] !== "";

/** The current (highest) lifecycle step a scan's manifest has reached (11-step model: Raw→Auto→Vetted→
 *  Classified→Cornea→Scar→Subgroup→Aligned→Normalized→Corrected→Scheduled). Cornea (SAM2) and Scar
 *  (strategy-compared) are now SEPARATE steps. */
export function scanStep(m: Manifest): LifecycleStep {
  if (!m) return 0;
  if (!set(m, "input_volume") && !set(m, "corrected_volume")) return 0;
  // A BUILT CONSENSUS case is the ALIGNED artifact (step 8); normalize/correct/schedule act on it.
  if (set(m, "consensus_cases") || set(m, "consensus_report")) {
    if (set(m, "training_scheduled")) return 11;
    if (set(m, "corrected_labelmap")) return 10;
    if (set(m, "normalized")) return 9;
    return 8;
  }
  if (!set(m, "oct_preprocessed")) return 1;                 // raw only
  // A SEGMENTED per-scan scan: Cornea(5, sam2_meta) → Scar(6, scar_done) → Subgroup(7) → Aligned(8,
  // consensus_case link). Normalize(9) acts on the consensus case, not the member, so a member tops out
  // at 8 (or 10/11 if its own labelmap was corrected / scheduled directly).
  if (set(m, "sam2_meta") || set(m, "consensus_case") || set(m, "corrected_labelmap")) {
    if (set(m, "training_scheduled")) return 11;            // scheduled (green)
    if (set(m, "corrected_labelmap")) return 10;           // manually corrected (dark blue)
    if (set(m, "consensus_case")) return 8;                // aligned to the eye's consensus (teal)
    if (set(m, "subgroup_confirmed")) return 7;            // subgroup assigned (purple)
    if (set(m, "scar_done")) return 6;                     // scar segmented (rose)
    return 5;                                               // cornea segmented (light blue)
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
    case 6: return set(m, "scar_done") || set(m, "consensus_case") || set(m, "corrected_labelmap");
    case 7: return set(m, "subgroup_confirmed") || set(m, "consensus_case");   // aligned implies subgroup done
    case 8: return set(m, "consensus_case");
    case 9: return set(m, "normalized");
    case 10: return set(m, "corrected_labelmap");
    case 11: return set(m, "training_scheduled");
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
