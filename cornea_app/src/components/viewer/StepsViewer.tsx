/* Preprocessing-steps filmstrip: every intermediate stage of the corneal-flattening algorithm for the
   central sagittal slice (original → hist-eq → bilateral → surface edge → side-corrected → quadratic
   fit → 3D active correction → final column warp). A diagnostic so the user can SEE what each step
   does; reflects the scan's persisted params + any marked bad columns. Opened from the 3D viewer. */

import { useEffect, useState } from "react";
import { CircularProgress } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";

export function StepsViewer({ onClose }: { onClose: () => void }) {
  const caseId = useCaseStore((s) => s.caseId);
  const segSig = useWorkflowStore((s) => s.segVersion); // re-render after a re-preprocess
  const [steps, setSteps] = useState<{ label: string; data_url: string }[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;
    setBusy(true);
    setErr(null);
    api
      .json<{ steps: { label: string; data_url: string }[] }>(
        `/api/case/${caseId}/oct-preprocess-steps`, "POST", JSON.stringify({}),
      )
      .then((r) => !cancelled && setSteps(r.steps || []))
      .catch((e) => !cancelled && setErr(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancelled && setBusy(false));
    return () => { cancelled = true; };
  }, [caseId, segSig]);

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div className="flex items-center gap-2 px-3 py-1 border-b flex-wrap"
        style={{ borderColor: "var(--c-border)", background: "var(--c-surface)" }}>
        <button onClick={onClose}
          style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-accent)", cursor: "pointer", fontSize: 12, padding: "2px 8px" }}>
          ← 3D view
        </button>
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
          Preprocessing steps for the central sagittal slice (uses this scan's params + marked columns).
        </span>
        {busy && <CircularProgress size={14} />}
      </div>
      <div className="flex-1 min-h-0 overflow-auto p-3">
        {err ? (
          <div className="text-center" style={{ color: "var(--c-red)", fontSize: 12 }}>Couldn't render steps: {err}</div>
        ) : busy && steps.length === 0 ? (
          <div className="text-center" style={{ color: "var(--c-text-dim)", fontSize: 13 }}>Rendering steps…</div>
        ) : steps.length === 0 ? (
          <div className="text-center" style={{ color: "var(--c-text-dim)", fontSize: 13 }}>No steps — preprocess the scan first.</div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 12 }}>
            {steps.map((s, i) => (
              <figure key={i} style={{ margin: 0, border: "1px solid var(--c-border)", borderRadius: 6, overflow: "hidden", background: "#000" }}>
                <img src={s.data_url} alt={s.label} draggable={false}
                  style={{ width: "100%", display: "block", imageRendering: "pixelated" }} />
                <figcaption className="px-2 py-1" style={{ fontSize: 11, color: "var(--c-text-dim)", background: "var(--c-surface)" }}>
                  {s.label}
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
