/* Pending border corrections, held so the REVIEW LOOP can commit them without the user pressing Confirm.
 *
 * WHY THIS EXISTS. A correction the reviewer draws is ground truth about where the cornea is, and it should
 * cost one click to record: drag the line, hit Reject, next scan. Previously the anchors lived only in
 * SliceGallery's local state and reached the backend solely via "Confirm border" — so rejecting a scan you had
 * just corrected threw the correction away, silently, which is the opposite of what the gesture means.
 *
 * The editor publishes whatever is currently drawn here; TimelineBar's reject/approve handlers flush it before
 * writing the verdict. Keyed by case so a stale edit from a previous scan can never be committed against the
 * one now on screen — the single most dangerous failure mode for a queue that advances automatically. */
import { create } from "zustand";

export interface PendingBorderEdit {
  caseId: string;
  /** slice index -> frame -> depth. Depths may be NEGATIVE (an apex above the captured window). */
  anchors: Record<string, Record<string, number>>;
  /** true when ANY slice in `anchors` is a densely-sampled shaped quadratic rather than point drags. */
  parabola: boolean;
  /** WHICH slices those are. The payload now carries both kinds at once — a shaped curve is exact (seed
   *  window 0), point drags are approximate seeds — so "which are exact" can no longer be a single flag.
   *  A slice carrying both is deliberately absent here: it contains approximate points. */
  parabolaSlices: string[];
  /** counts, for the human-readable summary recorded with a rejection */
  nSlices: number;
  nPoints: number;
  /** surface-crop frame marks drawn but not yet committed (null = untouched, [] = explicitly cleared) */
  cropFrames: number[] | null;
  /** crop-region box drawn but not yet committed (null = untouched) */
  cropRegion: { lateral: [number, number]; frames: number[] } | null;
  /** defect-mark columns for the current slice (null = unchanged) */
  defectCols: { slice: number; cols: number[] } | null;
  /** manual POSTERIOR (bottom) edge points, {slice: {frame: depth}} (null = none drawn) */
  postAnchors: Record<string, Record<string, number>> | null;
}

interface PendingEditState {
  pending: PendingBorderEdit | null;
  setPending: (p: PendingBorderEdit | null) => void;
  /** Returns the edit only if it belongs to `caseId`; clears it either way, so a flush can never be
   *  double-committed and a stale edit cannot leak onto the next scan. */
  takePending: (caseId: string) => PendingBorderEdit | null;
}

export const usePendingEditStore = create<PendingEditState>((set, get) => ({
  pending: null,
  setPending: (p) => set({ pending: p }),
  takePending: (caseId) => {
    const p = get().pending;
    set({ pending: null });
    if (!p || p.caseId !== caseId) return null;
    // "Has anything to commit?" is not just anchors — a scan can carry ONLY crop marks, and dropping those
    // because no border point was drawn is the exact data-loss this store exists to prevent.
    const hasWork = p.nPoints > 0 || p.cropFrames !== null || p.cropRegion !== null
      || p.defectCols !== null || p.postAnchors !== null;
    return hasWork ? p : null;
  },
}));
