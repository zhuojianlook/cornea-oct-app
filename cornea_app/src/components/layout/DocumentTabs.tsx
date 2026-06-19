/* Document tabs. For a single case this is just the case-id chip. For a
   multi-scan consensus case (manifest.consensus_cases) it becomes real tabs —
   [ Consensus | scan-1 | scan-2 | … ] — and, on a scan tab, a toggle that swaps
   the overlay between that scan's own scar and the voted consensus, both drawn
   on the scan's own image warped into the common frame. */

import { useEffect } from "react";
import { ToggleButton, ToggleButtonGroup, Tooltip } from "@mui/material";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import { useUpdater } from "../../store/updaterStore";
import type { ConsensusReport, ConsensusScan } from "../../api/types";

// case_cs001_od_v3 → "v3"; falls back to the last underscore segment.
const shortLabel = (cid: string): string => cid.split("_").pop() || cid;

export function DocumentTabs() {
  const caseId = useCaseStore((s) => s.caseId);
  const caseInfo = useCaseStore((s) => s.caseInfo);
  const activeTab = useWorkflowStore((s) => s.activeTab);
  const overlayMode = useWorkflowStore((s) => s.overlayMode);
  const initTabs = useWorkflowStore((s) => s.initTabs);
  const selectTab = useWorkflowStore((s) => s.selectTab);
  const setOverlayMode = useWorkflowStore((s) => s.setOverlayMode);
  const updBusy = useUpdater((s) => s.busy);
  const updMsg = useUpdater((s) => s.msg);
  const checkUpdates = useUpdater((s) => s.check);

  const m = caseInfo?.manifest as Record<string, unknown> | undefined;
  const scans = (m?.consensus_cases as string[] | undefined) ?? null;
  const refCid = m?.reference as string | undefined;
  const report = m?.consensus_report as ConsensusReport | undefined;
  const isConsensus = !!scans && scans.length > 1;
  const perScan: Record<string, ConsensusScan> = {};
  for (const p of report?.per_scan ?? []) perScan[p.case] = p;

  // Reset tab routing whenever the open case changes (single ↔ consensus).
  useEffect(() => {
    initTabs(isConsensus);
  }, [caseId, isConsensus, initTabs]);

  return (
    <div
      className="flex items-center px-3 gap-2 border-b"
      style={{ height: 38, backgroundColor: "var(--c-surface2)", borderColor: "var(--c-border)" }}
    >
      <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
        Cornea OCT
      </span>

      {!isConsensus ? (
        <div
          className="px-3 py-1 rounded text-xs"
          style={{ backgroundColor: "var(--c-surface)", color: "var(--c-text)" }}
        >
          {caseId ?? "—"}
        </div>
      ) : (
        <>
          <div className="flex items-center gap-1 overflow-x-auto">
            {tab("consensus", "Consensus", activeTab === "consensus", () => selectTab("consensus"))}
            {scans!.map((cid) => {
              const ps = perScan[cid];
              const warn = ps?.low_correspondence;
              const isRef = cid === refCid;
              const title = ps
                ? `${cid} · scar ${ps.scar_volume_mm3} mm³ · Dice-to-ref ${ps.scar_dice_to_ref} · ${Math.round(
                    ps.matched_fraction * 100,
                  )}% in consensus${warn ? " · low correspondence (likely a different FOV)" : ""}`
                : cid;
              return (
                <Tooltip key={cid} title={title} arrow>
                  <span>
                    {tab(
                      cid,
                      <span className="flex items-center gap-1">
                        {warn && (
                          <span
                            style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--c-red, #ff6b6b)" }}
                          />
                        )}
                        {shortLabel(cid)}
                        {isRef && <span style={{ color: "var(--c-text-dim)" }}>★</span>}
                      </span>,
                      activeTab === cid,
                      () => selectTab(cid),
                    )}
                  </span>
                </Tooltip>
              );
            })}
          </div>

          <div className="flex-1" />

          {activeTab !== "consensus" && (
            <div className="flex items-center gap-2">
              <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                Overlay
              </span>
              <ToggleButtonGroup
                size="small"
                exclusive
                value={overlayMode}
                onChange={(_, v) => v && setOverlayMode(v)}
              >
                <ToggleButton value="self" sx={{ py: 0, fontSize: 11, textTransform: "none" }}>
                  This scan
                </ToggleButton>
                <ToggleButton value="consensus" sx={{ py: 0, fontSize: 11, textTransform: "none" }}>
                  Consensus
                </ToggleButton>
              </ToggleButtonGroup>
            </div>
          )}
        </>
      )}

      {/* Manual update check (the app also checks silently on launch). */}
      <div className="flex items-center gap-2" style={{ marginLeft: "auto" }}>
        {updMsg && <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>{updMsg}</span>}
        <Tooltip title="Check for a newer version and install it in-app" arrow>
          <button onClick={() => checkUpdates(true)} disabled={updBusy}
            className="text-[11px] whitespace-nowrap"
            style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4,
                     color: "var(--c-text-dim)", cursor: updBusy ? "default" : "pointer", padding: "2px 8px" }}>
            {updBusy ? "Checking…" : "⟳ Updates"}
          </button>
        </Tooltip>
      </div>
    </div>
  );
}

function tab(key: string, label: React.ReactNode, active: boolean, onClick: () => void) {
  return (
    <button
      key={key}
      onClick={onClick}
      className="px-3 py-1 rounded text-xs whitespace-nowrap"
      style={{
        backgroundColor: active ? "var(--c-accent)" : "var(--c-surface)",
        color: active ? "#fff" : "var(--c-text)",
        border: "1px solid var(--c-border)",
        cursor: "pointer",
      }}
    >
      {label}
    </button>
  );
}
