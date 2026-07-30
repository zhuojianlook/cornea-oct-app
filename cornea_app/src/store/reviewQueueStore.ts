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
  /** Open a case in the viewer — OctLoader's own `preview`, published so the timeline can call it. Null
   *  until the sidebar has loaded, which is also when `queue` is meaningfully populated. */
  open: ((caseId: string) => void | Promise<void>) | null;
  publish: (queue: string[], open: (caseId: string) => void | Promise<void>) => void;
}

export const useReviewQueueStore = create<ReviewQueueState>()((set) => ({
  queue: [],
  open: null,
  publish: (queue, open) =>
    set((s) => {
      // Skip the state write when nothing changed. `publish` is called from a render-driven effect on every
      // scan-list update (which includes each poll of the lifecycle data), and a new array identity each time
      // would re-render every subscriber for no reason.
      const same = s.queue.length === queue.length && s.queue.every((id, i) => id === queue[i]);
      return same && s.open === open ? s : { queue, open };
    }),
}));

/** The case after `current` in the queue, or the first one if `current` is not in it (which is the normal
 *  case straight after approving: the scan has just left the queue). Null when the queue is empty — i.e.
 *  the backlog is cleared, and the caller should say so rather than open something arbitrary. */
export function nextAfter(queue: string[], current: string | null): string | null {
  if (!queue.length) return null;
  if (!current) return queue[0];
  const i = queue.indexOf(current);
  if (i < 0) return queue[0];
  return queue[(i + 1) % queue.length];
}
