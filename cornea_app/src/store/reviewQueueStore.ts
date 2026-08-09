/* The approval BACKLOG as an ordered queue, so approving or rejecting a scan can advance to the next one
   without a trip back to the sidebar.

   WHY A STORE AND NOT A PROP. The queue is derived from state that lives inside OctLoader (the scan list, the
   grouping, the active filter and search), but the buttons that consume it live in TimelineBar, which is not a
   descendant. Threading a callback through the layout would couple two panels that otherwise share nothing;
   a two-field store keeps the coupling to "OctLoader publishes, TimelineBar consumes".

   THE ORDER IS THE SIDEBAR'S ORDER, deliberately: whatever the reviewer has filtered or searched to is the set
   they are working through, so "next" must mean the next row they can see, not the next row that happens to
   exist. OctLoader republishes on every change, so narrowing the filter mid-session re-aims the queue. */
import { create } from "zustand";

interface ReviewQueueState {
  /** Case ids still awaiting approval, in the sidebar's visible order. Excludes scans already vetted or
   *  flagged difficult — the same two states that retire a scan from the review loop. */
  queue: string[];
  /** Ids settled in THIS session. The queue is derived from the sidebar's `life` data, which only refreshes on
   *  a lifecycle re-hydrate, so a scan stays in it for a while after being approved or rejected — long enough
   *  to be handed back as "next", which reads as the verdict not having registered. Recording them here makes
   *  the queue correct at once rather than after the refresh. */
  settled: Set<string>;
  /** `wasReject` distinguishes the two verdicts. The tuning checkpoint counts REJECTIONS only: an approval
   *  says the detector already got this scan right, so it carries no correction and nothing to learn from.
   *  Counting both would fire the "improve the algorithm" offer on a run of clean approvals with no new
   *  geometry behind it. */
  markSettled: (caseId: string, wasReject?: boolean) => void;
  /** How many scans have been settled since the last checkpoint. The reviewer works in batches of 10 and is
   *  then offered "reprocess with this feedback" vs "carry on"; this is what counts toward that. Reset when
   *  they choose, NOT when they reprocess, so declining does not re-prompt on the very next scan. */
  sinceCheckpoint: number;
  /** Rejections — i.e. corrections drawn — since the last checkpoint. This is what the tuning button waits on. */
  rejectsSinceCheckpoint: number;
  clearCheckpoint: () => void;
  /** Open a case in the viewer — OctLoader's own `preview`, published so the timeline can call it. Null
   *  until the sidebar has loaded, which is also when `queue` is meaningfully populated. */
  open: ((caseId: string) => void | Promise<void>) | null;
  /** APPROVAL PROGRESS across the whole store, published with the queue: how many preprocessed scans carry an
   *  approval, out of how many could. Deliberately NOT filtered by the sidebar's current filter — the queue is
   *  "what am I working through now", this is "how far through the job am I", and the second must not move
   *  when the first is narrowed. Lives here for the same reason the queue does: derived in the sidebar,
   *  displayed on the timeline. */
  vetted: number;
  vettable: number;
  /** Scans the reviewer REJECTED (manifest.reviewer_rejected). Deliberately not manifest.difficult_scan:
   *  bulk preprocessing sets that too, so it reads as "everything is rejected". A rejected scan is NOT
   *  vetted — it stays outstanding until reprocessed — so these two counters do not sum to the total. */
  rejected: number;
  publish: (queue: string[], open: (caseId: string) => void | Promise<void>,
            vetted: number, vettable: number, rejected: number) => void;
}

export const useReviewQueueStore = create<ReviewQueueState>()((set) => ({
  queue: [],
  settled: new Set<string>(),
  sinceCheckpoint: 0,
  rejectsSinceCheckpoint: 0,
  clearCheckpoint: () => set({ sinceCheckpoint: 0, rejectsSinceCheckpoint: 0 }),
  markSettled: (caseId, wasReject) => set((s) => (s.settled.has(caseId)
    ? s                                        // already counted — re-settling must not advance the batch
    : { settled: new Set(s.settled).add(caseId),
        sinceCheckpoint: s.sinceCheckpoint + 1,
        rejectsSinceCheckpoint: s.rejectsSinceCheckpoint + (wasReject ? 1 : 0) })),
  open: null,
  vetted: 0,
  vettable: 0,
  rejected: 0,
  publish: (queue, open, vetted, vettable, rejected) =>
    set((s) => {
      // Skip the state write when nothing changed. `publish` is called from a render-driven effect on every
      // scan-list update (which includes each poll of the lifecycle data), and a new array identity each time
      // would re-render every subscriber for no reason. The counts are part of that comparison — omitting
      // them would freeze the ticker at whatever it read when the queue last changed shape.
      const same = s.queue.length === queue.length && s.queue.every((id, i) => id === queue[i]);
      return same && s.open === open && s.vetted === vetted && s.vettable === vettable
        && s.rejected === rejected
        ? s : { queue, open, vetted, vettable, rejected };
    }),
}));

/** The next case to review: the one AFTER `current` in the queue, skipping anything already settled in this
 *  session. Falls back to the first unsettled entry when `current` is not in the queue.
 *
 *  `queue` must be the FULL queue including `current` — an earlier version was passed a queue with `current`
 *  filtered out, so indexOf returned -1 and every advance jumped to queue[0], sending the reviewer back to the
 *  top of the list instead of forward one. Null when nothing is left. */
export function nextAfter(queue: string[], current: string | null, settled?: Set<string>): string | null {
  const live = queue.filter((id) => id !== current && !(settled && settled.has(id)));
  if (!live.length) return null;
  if (!current) return live[0];
  const i = queue.indexOf(current);
  if (i < 0) return live[0];
  // step forward from the current position, wrapping, and take the first still-live entry
  for (let k = 1; k <= queue.length; k++) {
    const cand = queue[(i + k) % queue.length];
    if (cand !== current && !(settled && settled.has(cand))) return cand;
  }
  return null;
}
