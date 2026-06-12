/* Scar metrics (Stage 4). */

import type { ReactNode } from "react";
import { useWorkflowStore } from "../../store/workflowStore";

export function ScarPanel() {
  const scar = useWorkflowStore((s) => s.scarMetrics);
  const summaryInfo = useWorkflowStore((s) => s.scarSummaryInfo);
  if (!scar && !summaryInfo) return null;

  const row = (label: string, value: ReactNode) => (
    <div className="flex justify-between">
      <span style={{ color: "var(--c-text-dim)" }}>{label}</span>
      <span>{value}</span>
    </div>
  );

  return (
    <div className="rounded p-2 flex flex-col gap-1" style={{ backgroundColor: "var(--c-surface2)", borderLeft: "3px solid #ff453a" }}>
      <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
        Scar quantification
      </div>
      {scar && (!scar.scar_present ? (
        <div className="text-xs" style={{ color: "var(--c-text-dim)" }}>
          {scar.note || "No scar above the presence gate — paint manually if present."}
        </div>
      ) : (
        <div className="flex flex-col gap-1 text-xs">
          {row("Volume", <b>{scar.scar_volume_mm3?.toLocaleString()} mm³</b>)}
          {row("En-face area", <b>{scar.scar_area_mm2?.toLocaleString()} mm²</b>)}
          {row("% of cornea", `${Math.round((scar.scar_fraction_of_cornea ?? 0) * 100)}%`)}
          {scar.cornea_volume_mm3 != null && row("Cornea volume", `${scar.cornea_volume_mm3.toLocaleString()} mm³`)}
          {scar.scar_density?.mean != null &&
            row("Mean density", `${scar.scar_density.mean} (p10–90 ${scar.scar_density.p10}–${scar.scar_density.p90})`)}
          {scar.scar_density?.tier_volume_mm3 && (
            <div className="flex justify-between">
              <span style={{ color: "var(--c-text-dim)" }}>Density tiers</span>
              <span title="diffuse / moderate / dense (mm³)">
                {scar.scar_density.tier_volume_mm3.map((v) => v.toFixed(2)).join(" / ")}
              </span>
            </div>
          )}
          {scar.largest_component_fraction != null &&
            row("Continuity", `${Math.round(scar.largest_component_fraction * 100)}% in 1 region`)}
        </div>
      ))}
      {summaryInfo && (
        <div className="text-[11px] mt-1 break-all" style={{ color: "var(--c-text-dim)" }}>
          {summaryInfo}
        </div>
      )}
    </div>
  );
}
