/* Scar metrics (Stage 4). */

import { usePaintStore } from "../../store/paintStore";

export function ScarPanel() {
  const scar = usePaintStore((s) => s.scarMetrics);
  if (!scar) return null;

  return (
    <div className="rounded p-2 flex flex-col gap-1" style={{ backgroundColor: "var(--c-surface2)", borderLeft: "3px solid #ff453a" }}>
      <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
        Scar
      </div>
      {!scar.scar_present ? (
        <div className="text-xs" style={{ color: "var(--c-text-dim)" }}>
          {scar.note || "No scar detected in this sample."}
        </div>
      ) : (
        <div className="flex flex-col gap-1 text-xs">
          <div className="flex justify-between">
            <span style={{ color: "var(--c-text-dim)" }}>Scar voxels</span>
            <span>{scar.scar_voxels?.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span style={{ color: "var(--c-text-dim)" }}>% of cornea</span>
            <span>{Math.round((scar.scar_fraction_of_cornea ?? 0) * 100)}%</span>
          </div>
          {scar.scar_bounds_ijk && (
            <div className="flex justify-between">
              <span style={{ color: "var(--c-text-dim)" }}>Bounds IJK</span>
              <span>
                {scar.scar_bounds_ijk.min.join(",")} → {scar.scar_bounds_ijk.max.join(",")}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
