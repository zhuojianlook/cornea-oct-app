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
  /** true when the border anchors DIFFER from what is persisted (the reviewer actually changed the raw line
   *  this session). nPoints alone counts the re-seeded persisted set too, so it can't tell "changed" from
   *  "just loaded" — which a corrected-edge-only pin needs, to skip the full re-run. */
  bordersDirty: boolean;
  /** surface-crop frame marks drawn but not yet committed (null = untouched, [] = explicitly cleared) */
  cropFrames: number[] | null;
  /** crop-region box drawn but not yet committed (null = untouched) */
  cropRegion: { lateral: [number, number]; frames: number[] } | null;
  /** defect-mark columns for the current slice (null = unchanged) */
  defectCols: { slice: number; cols: number[] } | null;
  /** manual POSTERIOR (bottom) edge points, {slice: {frame: depth}} (null = none drawn) */
  postAnchors: Record<string, Record<string, number>> | null;
}

/** WHICH red line the before/after view is editing right now. The two corrections COMPOSE (raw = detection +
 *  base warp; corrected = a post-hoc rigid per-frame shift for residual drift), but the reviewer edits ONE at a
 *  time so a single "Correct & re-run" is unambiguous and no drawn edit is ever silently dropped. */
export type EditTarget = "original" | "corrected";

/** Corrected-result edge edits drawn on the RIGHT pane but not yet committed. Keyed by case like the border
 *  pending, so a stale edit can never land on the next scan. `dirty` = differs from what is persisted, so
 *  clearing (emptying the anchors) still commits (to remove the persisted set). */
export interface PendingCorrectedEdge {
  caseId: string;
  /** lateral -> frame -> depth, in CORRECTED-output depth space (0 = TOP). */
  anchors: Record<string, Record<string, number>>;
  nPoints: number;
  dirty: boolean;
}

interface PendingEditState {
  pending: PendingBorderEdit | null;
  setPending: (p: PendingBorderEdit | null) => void;
  /** Returns the edit only if it belongs to `caseId`; clears it either way, so a flush can never be
   *  double-committed and a stale edit cannot leak onto the next scan. */
  takePending: (caseId: string) => PendingBorderEdit | null;

  /** which line the before/after view is editing (default: the original/raw line). */
  editTarget: EditTarget;
  setEditTarget: (t: EditTarget) => void;

  /** corrected-pane edits, published by CorrectedEdgePanel, flushed by the same handlers that flush `pending`. */
  correctedEdge: PendingCorrectedEdge | null;
  setCorrectedEdge: (c: PendingCorrectedEdge | null) => void;
  /** Returns the corrected-edge anchors to commit (only when it belongs to `caseId` AND is dirty); clears it
   *  either way. Empty-but-dirty returns `{}` so a cleared correction is committed as a removal. */
  takeCorrectedEdge: (caseId: string) => Record<string, Record<string, number>> | null;

  /** TRUSTED SLICES for smooth-align: the sagittal slices (ARRAY laterals == slice_index) whose detected border
   *  the reviewer marked GOOD. smooth-align builds its per-frame curvature from ONLY these. Keyed by case so a
   *  stale set can't leak onto the next scan; persists across scrolling until re-run or cleared. */
  trustedSlices: { caseId: string; slices: number[] } | null;
  toggleTrustedSlice: (caseId: string, slice: number) => void;
  clearTrustedSlices: (caseId: string) => void;
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

  editTarget: "original",
  setEditTarget: (t) => set({ editTarget: t }),

  correctedEdge: null,
  setCorrectedEdge: (c) => set({ correctedEdge: c }),
  takeCorrectedEdge: (caseId) => {
    const c = get().correctedEdge;
    set({ correctedEdge: null });
    if (!c || c.caseId !== caseId || !c.dirty) return null;
    return c.anchors;   // may be {} → commit as a removal of the persisted corrected-edge set
  },

  trustedSlices: null,
  toggleTrustedSlice: (caseId, slice) => set((s) => {
    // a scan switch resets the set (belongs to a different case)
    const cur = s.trustedSlices && s.trustedSlices.caseId === caseId ? s.trustedSlices.slices : [];
    const has = cur.includes(slice);
    const slices = has ? cur.filter((x) => x !== slice) : [...cur, slice].sort((a, b) => a - b);
    return { trustedSlices: { caseId, slices } };
  }),
  clearTrustedSlices: (caseId) => set((s) => (
    s.trustedSlices && s.trustedSlices.caseId !== caseId ? {} : { trustedSlices: null })),
}));
