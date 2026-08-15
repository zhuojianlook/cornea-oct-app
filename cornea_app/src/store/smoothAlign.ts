// Human-readable explanation of the corrected-mode "Smooth to trusted slices & re-run" outcome.
//
// That button runs align_corrected_to_smooth ("csa"): it propagates the reviewer's EDITED (drawn) + APPROVED
// curves across the volume as a per-frame RIGID depth-shift + tilt, and KEEPS the move only if it makes the
// across-frame surface more quadratic without dragging the drawn edges off. It very often DECLINES — correctly —
// because the base (step-2) correction already flattened the surface to a near-quadratic, so a rigid move can only
// make it worse OR would pull the pixel-exact drawn edges off their lines. When that happens the reviewer sees a
// good drawn edge but no volume change and, without this, no reason why. The backend writes the full result dict to
// manifest.oct_iter.corrected_smooth_align; this turns it into a headline + detail + tone used by BOTH the
// corrected-mode toolbar (persistent last-result line) and the re-run status toast, so the decline is never silent.

export interface SmoothAlignInfo {
  applied?: boolean;
  reason?: string;
  trusted_slices?: number;
  edited_slices?: number;
  approved_slices?: number;
  frames_adjusted?: number;
  max_shift?: number;
  max_tilt_swing?: number;
  off_quad_before?: number;   // px: across-frame RMS deviation from a quadratic BEFORE the move
  off_quad_after?: number;    // px: … AFTER the candidate rigid move (decline if this is not lower)
  anchor_dev?: number;        // px: how far a rigid move would displace the drawn anchors (anchor-fidelity guard)
}

export type SmoothAlignTone = "ok" | "info" | "muted";

export interface SmoothAlignDescription {
  headline: string;   // short, number-bearing, safe to show inline in the toolbar
  detail: string;     // full sentence for the tooltip / status toast
  tone: SmoothAlignTone;
}

const px = (v: number | undefined | null): string => (v == null ? "?" : `${v}px`);

/** Describe a csa outcome, or null when there is nothing to report (no result yet). */
export function describeSmoothAlign(csa: SmoothAlignInfo | null | undefined): SmoothAlignDescription | null {
  if (!csa || (csa.applied == null && !csa.reason)) return null;
  const edited = csa.edited_slices ?? 0;
  const approved = csa.approved_slices ?? 0;
  const trusted = csa.trusted_slices ?? (edited + approved);

  if (csa.applied) {
    const quad = (csa.off_quad_before != null && csa.off_quad_after != null)
      ? ` Across-frame surface ${px(csa.off_quad_before)}→${px(csa.off_quad_after)}.` : "";
    const frames = csa.frames_adjusted ?? 0;
    return {
      headline: `✓ Smoothed to your ${trusted} trusted curve${trusted === 1 ? "" : "s"} — ${frames} frame${frames === 1 ? "" : "s"} moved`,
      detail: `Propagated your ${edited} edited + ${approved} approved curve${trusted === 1 ? "" : "s"} across the volume — `
        + `${frames} B-scan${frames === 1 ? "" : "s"} rigidly moved (max ${px(csa.max_shift)}).${quad}`,
      tone: "ok",
    };
  }

  const reason = (csa.reason ?? "").toLowerCase();

  // Anchor-fidelity guard: a rigid move would drag the pixel-exact drawn edges off their lines. (No off_quad here —
  // this guard returns before the quadratic check.)
  if (csa.anchor_dev != null || reason.includes("drawn gt")) {
    return {
      headline: `Drawn edges already define the surface — no rigid move (would shift them ${px(csa.anchor_dev)})`,
      detail: `A rigid per-frame move would pull your drawn edges ${px(csa.anchor_dev)} off their lines, so nothing was shifted. `
        + `Your ${edited || trusted} edited curve${(edited || trusted) === 1 ? "" : "s"} are applied exactly — the reconstruction IS the surface — `
        + `and the un-drawn laterals can't be improved rigidly without dragging them.`,
      tone: "info",
    };
  }

  // Never-worse-quadratic guard: the move would make the across-frame surface LESS quadratic.
  if (reason.includes("not improved") || (csa.off_quad_before != null && csa.off_quad_after != null)) {
    return {
      headline: `Already near-optimal across frames (${px(csa.off_quad_before)}; a move → ${px(csa.off_quad_after)})`,
      detail: `The across-frame surface is already quadratic at ${px(csa.off_quad_before)}; a rigid move would worsen it to `
        + `${px(csa.off_quad_after)}, so nothing was changed. Your drawn edges are applied — this is already the best rigid fit.`,
      tone: "info",
    };
  }

  // Nothing trusted to smooth to yet.
  if (reason.includes("no trusted")) {
    return {
      headline: "Nothing to smooth to yet — draw or approve a slice first",
      detail: `Draw the good curve on a slice or approve one as trusted, then run "Smooth to trusted slices".`,
      tone: "muted",
    };
  }

  // Other no-ops (detect failed / too small / no residuals / no move needed).
  return {
    headline: `No change — ${csa.reason ?? "nothing to propagate"}`,
    detail: `Smooth-align made no change (${csa.reason ?? "nothing to propagate"}).`,
    tone: "muted",
  };
}
