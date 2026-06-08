/* Right-hand panel: AI/heuristic paint status, marking summary, seed previews. */

import { useEffect } from "react";
import { LinearProgress } from "@mui/material";
import { usePaintStore } from "../../store/paintStore";
import { useCaseStore } from "../../store/caseStore";
import { OverlayControls } from "../viewer/OverlayControls";
import { FeedbackPanel } from "./FeedbackPanel";
import { ScarPanel } from "./ScarPanel";

const STATUS_COLOR: Record<string, string> = {
  idle: "var(--c-text-dim)",
  working: "var(--c-accent)",
  done: "var(--c-green)",
  error: "var(--c-red)",
};

export function AiPaintPanel() {
  const { status, result, seedImages, aiBusy, refreshSeedPreviews } = usePaintStore();
  const caseId = useCaseStore((s) => s.caseId);

  useEffect(() => {
    refreshSeedPreviews();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId]);

  const marking = result?.agent_marking || {};
  const confidence = typeof result?.confidence === "number" ? `${Math.round(result.confidence * 100)}%` : "—";
  const issues = result?.issues || [];

  return (
    <div className="flex flex-col h-full" style={{ backgroundColor: "var(--c-surface)" }}>
      <div className="px-3 py-2 border-b" style={{ borderColor: "var(--c-border)" }}>
        <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
          Paint Agent
        </div>
      </div>

      <div className="px-3 py-3 flex flex-col gap-3 overflow-y-auto">
        <div
          className="rounded p-2"
          style={{ backgroundColor: "var(--c-surface2)", borderLeft: `3px solid ${STATUS_COLOR[status.kind]}` }}
        >
          <div className="text-sm font-medium">{status.title}</div>
          <div className="text-xs mt-1" style={{ color: "var(--c-text-dim)" }}>
            {status.detail}
          </div>
          {aiBusy && <LinearProgress sx={{ mt: 1 }} />}
        </div>

        <OverlayControls />
        <ScarPanel />
        <FeedbackPanel />

        <div className="grid grid-cols-3 gap-2 text-center">
          {[
            ["Cornea", marking.cornea_stroke_count ?? "—"],
            ["Background", marking.background_stroke_count ?? "—"],
            ["Confidence", confidence],
          ].map(([k, v]) => (
            <div key={k} className="rounded p-2" style={{ backgroundColor: "var(--c-surface2)" }}>
              <div className="text-[10px] uppercase" style={{ color: "var(--c-text-dim)" }}>
                {k}
              </div>
              <div className="text-sm font-semibold">{v}</div>
            </div>
          ))}
        </div>

        {issues.length > 0 && (
          <ul className="text-xs list-disc pl-4" style={{ color: "var(--c-text-dim)" }}>
            {issues.map((it, i) => (
              <li key={i}>{it}</li>
            ))}
          </ul>
        )}

        <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
          Seed previews {seedImages.length ? `(${seedImages.length})` : ""}
        </div>
        {seedImages.length === 0 ? (
          <div className="text-xs" style={{ color: "var(--c-text-dim)" }}>
            No seed paint yet.
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            {seedImages.slice(0, 12).map((img) => (
              <figure key={img.file_name} className="m-0">
                <img
                  src={img.data_url}
                  alt={img.file_name}
                  className="w-full rounded"
                  style={{ border: "1px solid var(--c-border)" }}
                />
                <figcaption className="text-[10px] mt-0.5" style={{ color: "var(--c-text-dim)" }}>
                  {img.orientation} {img.slice_index ?? ""}
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
