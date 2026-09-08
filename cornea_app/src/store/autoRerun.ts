import type { CaseInfo } from "../api/types";

/* AUTO RE-RUN ON OPEN — the decision, kept pure so it can be driven without React or the store.
 *
 * Why: the pipeline moved to the tissue-measured per-frame move ("the new pipeline"), but a scan's stored result
 * is whatever pipeline last ran on it. The reviewer opened case_cs011_od_v3 from the approve queue, saw the OLD
 * flatten, and reported it as a regression. So when a scan opens with a result from a different pipeline version
 * AND nobody has approved that result, the app re-runs it through the current pipeline right away — the exact
 * action the "↻ Re-run with corrections" button takes — and shows the fresh result.
 *
 * The backend decides staleness (api_server.run_is_current: oct_iter.pipeline_version == PIPELINE_VERSION) and
 * reports it on the case-open response as `pipeline_current`. This module decides whether that verdict should
 * trigger a run.
 */

/** Exactly the body the "↻ Re-run with corrections" button POSTs to /api/case/{id}/oct-preprocess. */
export const RERUN_WITH_CORRECTIONS_BODY = Object.freeze({
  use_redetect: true,
  corrected_edit_feedback: false,
  corrected_smooth_align: false,
  corrected_trusted_laterals: null,
});

export type AutoRerunVerdict =
  | { run: true; why: string }
  | { run: false; why: string };

/**
 * Should opening `info` auto-dispatch a re-run through the current pipeline?
 *
 * Fires only when ALL hold:
 *  - the backend says the last run is NOT current (`pipeline_current === false`, strictly — an older sidecar
 *    that does not report the field never triggers a run);
 *  - the scan is not approved (`manifest.preproc_vetted` falsy) — approved results are never touched;
 *  - the scan has been preprocessed at all (`manifest.oct_preprocessed`) — a raw scan the reviewer is still
 *    scrubbing / tagging is preprocessed deliberately, not on open;
 *  - it is a single scan, not a consensus case;
 *  - it has not already been attempted this open (`attempted`) — a same-case reload after the run, or a failed
 *    run, must not fire again until the case is genuinely re-opened.
 */
export function shouldAutoRerun(info: CaseInfo | null | undefined, attempted: ReadonlySet<string>): AutoRerunVerdict {
  if (!info || !info.case_id) return { run: false, why: "no case" };
  if (info.pipeline_current !== false) {
    return { run: false, why: info.pipeline_current === true ? "run is current" : "backend does not report pipeline currency" };
  }
  const m = (info.manifest || {}) as Record<string, unknown>;
  if (m.preproc_vetted) return { run: false, why: "approved — never re-run automatically" };
  if (!m.oct_preprocessed) return { run: false, why: "never preprocessed" };
  if (m.consensus_cases) return { run: false, why: "consensus case" };
  if (attempted.has(info.case_id)) return { run: false, why: "already attempted this open" };
  return { run: true, why: info.pipeline_reason || "stale pipeline" };
}

/** The one-line notice shown where the running state shows. */
export function autoRerunNotice(info: CaseInfo): string {
  const was = info.pipeline_version_run ? `v${info.pipeline_version_run}` : "an unstamped run";
  const now = info.pipeline_version_now ? ` (${was} → v${info.pipeline_version_now})` : "";
  return `This scan was processed with an older pipeline${now} — re-running it now…`;
}
