/* Right-hand inspector: workflow status, overlay controls, scar quantification. */

import { LinearProgress } from "@mui/material";
import { useWorkflowStore } from "../../store/workflowStore";
import { OverlayControls } from "../viewer/OverlayControls";
import { ScarPanel } from "./ScarPanel";

const STATUS_COLOR: Record<string, string> = {
  idle: "var(--c-text-dim)",
  working: "var(--c-accent)",
  done: "var(--c-green)",
  error: "var(--c-red)",
};

export function InspectorPanel() {
  const status = useWorkflowStore((s) => s.status);
  const busy = useWorkflowStore((s) => s.segBusy || s.scarBusy);

  return (
    <div className="flex flex-col h-full" style={{ backgroundColor: "var(--c-surface)" }}>
      <div className="px-3 py-2 border-b" style={{ borderColor: "var(--c-border)" }}>
        <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
          Inspector
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
          {busy && <LinearProgress sx={{ mt: 1 }} />}
        </div>

        <OverlayControls />
        <ScarPanel />
      </div>
    </div>
  );
}
