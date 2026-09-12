/* Per-scan lifecycle model — the single source of truth for the progress TIMELINE (TimelineBar) and the
   colour-coded scan entries (OctLoader). A scan advances linearly; each step requires the previous, so a
   later flag set while an earlier one is cleared (e.g. a re-preprocess resets preproc_vetted) correctly
   drops the scan back. Colours follow a smooth monotonic spectral ramp (see LIFECYCLE_STEPS): idle slate →
   red → pink → pink-red → … → blue → … → green (done), so the strip reads as a natural progression. */

export type LifecycleStep = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13;

export interface StepMeta { step: LifecycleStep; color: string; label: string; short: string; }

// Index = step number. step 0 = no scan loaded.
// Colours follow a SMOOTH MONOTONIC SPECTRAL RAMP so the strip reads as a natural progression (no jarring
// hue jumps): idle slate → red (needs work) → pink → pink-red → fuchsia → purple → violet → indigo → blue →
// sky → cyan → teal → green (done). All bright 400-shades for good contrast with the dark pill text.
export const LIFECYCLE_STEPS: { color: string; label: string; short: string }[] = [
  { color: "transparent", label: "—", short: "—" },
  { color: "#94a3b8", label: "Raw image", short: "Raw" },                          // 1 slate (idle)
  { color: "#f87171", label: "Preprocessed · automatic", short: "Auto" },          // 2 red (needs vetting)
  { color: "#f472b6", label: "Preprocessed · manually vetted", short: "Vetted" },  // 3 pink
  { color: "#ec4899", label: "Replicates aligned (curvature)", short: "Aligned" }, // 4 pink-red (group 3D alignment)
  { color: "#e879f9", label: "Cornea detection (SAM)", short: "Cornea" },           // 5 fuchsia (approved → detect → segmented)
  { color: "#c084fc", label: "Cornea/background vetted (paint)", short: "Cornea✓" }, // 6 purple
  { color: "#a78bfa", label: "Scar / control classified", short: "Classified" },    // 7 violet
  { color: "#818cf8", label: "Subgroup assigned", short: "Subgroup" },            // 8 indigo (BEFORE scar)
  { color: "#60a5fa", label: "Scar segmented", short: "Scar" },                    // 9 blue
  { color: "#38bdf8", label: "Scar replicates aligned", short: "Scar-aligned" },  // 10 sky (scar consensus)
  { color: "#22d3ee", label: "Normalized against controls", short: "Normalized" }, // 11 cyan
  { color: "#2dd4bf", label: "Manually corrected", short: "Corrected" },           // 12 teal
  { color: "#4ade80", label: "Scheduled for training", short: "Scheduled" },       // 13 green (done)
];

type Manifest = Record<string, unknown> | null | undefined;
const set = (m: NonNullable<Manifest>, k: string) => m[k] != null && m[k] !== false && m[k] !== "";

/** The current (highest) lifecycle step a scan's manifest has reached (Raw→Auto→Vetted→Aligned→Cornea→
 *  Cornea✓→Classified→Subgroup→Scar→Scar-aligned→Normalized→Corrected→Scheduled). Aligned (4, group_aligned)
 *  is the group-wise 3D alignment of the patient+eye's replicate scans (regularises their sagittal curvature)
 *  and comes BEFORE SAM2. Classification (scar/control) comes AFTER cornea-vetting — it gates only the scar
 *  branch, not SAM2. Subgroup is assigned BEFORE scar so the per-subgroup strategy comparison is available at
 *  the Scar step. Cornea (SAM2) and Scar are separate steps. */
/** A no-scar (control) scan: it contributes a cornea-only training label + the normal baseline, so the scar
 *  steps (Subgroup 8, Scar 9, Scar-aligned 10, Normalized 11, Corrected 12) do not apply — it goes Cornea✓ →
 *  Scheduled. */
export function isControl(m: Manifest): boolean {
  return !!m && m["scar_classification"] === "control";
}

/** Step 4 outcome of ONE scan (reviewer 2026-09-12: "all the scans should change status to aligned or (cannot align)"):
 *  "aligned"  — the group ran and this scan was PLACED on its canvas (reference / contributing / placed via another);
 *  "cannot"   — the group ran but this scan could not be placed (its pair was refused), or there was nothing to align
 *               it against (its subgroup has a single scan: manifest.align_not_possible);
 *  null       — the group has not been aligned yet. */
export function alignOutcome(m: Manifest): "aligned" | "cannot" | null {
  if (!m) return null;
  const ga = m.group_aligned as Record<string, unknown> | null | undefined;
  if (ga && typeof ga === "object") return (ga as { placed?: boolean }).placed === false ? "cannot" : "aligned";
  if (set(m, "group_aligned")) return "aligned";              // older stamp (no per-scan placement recorded)
  if (m.align_not_possible) return "cannot";
  return null;
}

export function scanStep(m: Manifest): LifecycleStep {
  if (!m) return 0;
  if (!set(m, "input_volume") && !set(m, "corrected_volume")) return 0;
  // A BUILT CONSENSUS case is the SCAR-ALIGNED artifact (step 10); normalize/correct/schedule act on it.
  if (set(m, "consensus_cases") || set(m, "consensus_report")) {
    if (set(m, "training_scheduled")) return 13;
    if (set(m, "corrected_labelmap")) return 12;
    if (set(m, "normalized")) return 11;
    return 10;
  }
  if (!set(m, "oct_preprocessed")) return 1;                 // raw only
  // A SEGMENTED per-scan scan: Cornea(5, sam2_meta) → Cornea/bg vetted(6, cornea_vetted) → Classified(7,
  // scar_classification) → Subgroup(8) → Scar(9) → Scar-aligned(10). Classification now comes AFTER
  // cornea-vetting (it gates only the scar branch, not SAM2): a control is READY to schedule once classified; a
  // scar scan proceeds to subgroup. Normalize(11) acts on the consensus case, so a member tops out at 10 (or
  // 12/13 if its own labelmap was corrected / scheduled). SAM2 implies the group alignment (4) is behind it.
  if (set(m, "sam2_meta") || set(m, "consensus_case") || set(m, "corrected_labelmap")) {
    if (set(m, "training_scheduled")) return 13;            // scheduled (green)
    // A CONTROL has no scar/subgroup/align/normalize/correct: once classified it is READY to schedule (steps
    // 8-12 do not apply). It never advances to 8-12 (those flags are ignored for it).
    if (isControl(m)) return 7;                             // classified control (violet) → ready to schedule
    if (set(m, "corrected_labelmap")) return 12;           // manually corrected (teal)
    if (set(m, "consensus_case")) return 10;               // aligned to the eye's scar consensus (sky)
    if (set(m, "scar_done")) return 9;                     // scar segmented (blue) — AFTER subgroup
    if (set(m, "subgroup_confirmed")) return 8;            // subgroup assigned (indigo) — BEFORE scar
    if (set(m, "scar_classification")) return 7;           // classified scar (violet) — next is subgroup
    if (set(m, "cornea_vetted")) return 6;                 // cornea/background paint-vetted (purple)
    return 5;                                               // cornea segmented, awaiting vet (fuchsia)
  }
  if (!set(m, "preproc_vetted")) return 2;                   // auto-preprocessed (red)
  // "cannot align" (single-scan subgroup) still LEAVES step 3: the group alignment was attempted and settled for
  // this scan, so it reads at step 4 with a "cannot align" label rather than waiting at Vetted for ever.
  if (!set(m, "group_aligned")) return m.align_not_possible ? 4 : 3;   // vetted, awaiting group alignment (pink)
  if (!set(m, "aligned_approved")) return 4;                 // group-aligned, awaiting the axial-changes approval (pink-red)
  // Approving the axial changes (reviewer 2026-09-12) moves the subgroup's scans to 5. Cornea: the scan is CURRENT
  // there with cornea detection (SAM) as its pending action; once sam2_meta lands the segmented branch above keeps it
  // at 5 (awaiting the cornea/background vet) — the steps ≥ 6 semantics are unchanged.
  return 5;                                                  // approved, awaiting cornea detection (fuchsia)
}

/** Is the scan at 5. Cornea WITHOUT a segmentation yet (axial changes approved, cornea detection pending)? Drives the
 *  step-5 action bar (detect vs vet) and keeps the alignment pane visible there. */
export function awaitingCorneaDetection(m: Manifest): boolean {
  return !!m && scanStep(m) === 5 && !hasSegmentation(m);
}

export function lifecycleMeta(m: Manifest): StepMeta {
  const step = scanStep(m);
  return { step, ...LIFECYCLE_STEPS[step] };
}

/** Whether step `i` has GENUINELY been reached (its own flag is set) — used to colour the timeline
 *  strip honestly: a scan scheduled straight from SAM2 must NOT show Scar-aligned/Corrected as done.
 *  A built consensus case is a finished artifact, so its earlier steps are treated as implicitly done. */
export function stepReached(m: Manifest, i: LifecycleStep): boolean {
  if (!m) return false;
  if (set(m, "consensus_cases") || set(m, "consensus_report")) return i <= scanStep(m);
  // A control skips the scar steps entirely — never colour 8-12 as reached for it (it goes Cornea✓ → Scheduled).
  if (isControl(m) && i >= 8 && i <= 12) return false;
  switch (i) {
    case 1: return set(m, "input_volume") || set(m, "corrected_volume");
    case 2: return set(m, "oct_preprocessed");
    case 3: return set(m, "preproc_vetted");
    // group aligned (4) — its OWN flag, or implied once any later (segmented) step is reached
    case 4: return set(m, "group_aligned") || set(m, "sam2_meta") || set(m, "corrected_labelmap") || set(m, "consensus_case");
    // cornea (5) — entered by the axial-changes approval (aligned_approved on a group-aligned scan), or implied by a
    // segmentation (auto-populated scans reach SAM2 without the group step).
    case 5: return (set(m, "aligned_approved") && set(m, "group_aligned")) || set(m, "sam2_meta") || set(m, "corrected_labelmap") || set(m, "consensus_case");
    // cornea/bg vetted — implied done once any LATER scar-branch step (subgroup/scar/aligned/corrected) is reached
    case 6: return set(m, "cornea_vetted") || set(m, "subgroup_confirmed") || set(m, "scar_done") || set(m, "consensus_case") || set(m, "corrected_labelmap");
    case 7: return set(m, "scar_classification");   // classified (scar/control) — now AFTER cornea✓
    // subgroup (8) — its OWN flag, or a consensus (built per-subgroup implies it). NOT scar_done: a CONTROL
    // skips subgroup and sets scar_done directly, so scar_done must not falsely colour subgroup as reached.
    case 8: return set(m, "subgroup_confirmed") || set(m, "consensus_case");
    // scar (9) — scar_done, a consensus (votes on scar), or a corrected labelmap (it carries scar labels)
    case 9: return set(m, "scar_done") || set(m, "consensus_case") || set(m, "corrected_labelmap");
    case 10: return set(m, "consensus_case");
    case 11: return set(m, "normalized");
    case 12: return set(m, "corrected_labelmap");
    case 13: return set(m, "training_scheduled");
    default: return false;
  }
}

/** Whether step `i` APPLIES to this scan. For a control the scar steps (8-12: Subgroup/Scar/Scar-aligned/
 *  Normalized/Corrected) are not applicable — the timeline shows them greyed/"—" and a control advances
 *  Cornea✓ (7 Classified) → Scheduled (13). Everything applies to scar scans + consensus cases. */
export function stepApplicable(m: Manifest, i: LifecycleStep): boolean {
  return !(isControl(m) && i >= 8 && i <= 12);
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
