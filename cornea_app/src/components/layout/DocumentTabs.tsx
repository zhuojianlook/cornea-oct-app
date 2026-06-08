import { useCaseStore } from "../../store/caseStore";

export function DocumentTabs() {
  const caseId = useCaseStore((s) => s.caseId);

  return (
    <div
      className="flex items-center px-3 gap-2 border-b"
      style={{
        height: 38,
        backgroundColor: "var(--c-surface2)",
        borderColor: "var(--c-border)",
      }}
    >
      <span
        className="text-[11px] uppercase tracking-wide"
        style={{ color: "var(--c-text-dim)" }}
      >
        Cornea OCT
      </span>
      <div
        className="px-3 py-1 rounded text-xs"
        style={{ backgroundColor: "var(--c-surface)", color: "var(--c-text)" }}
      >
        {caseId ?? "—"}
      </div>
    </div>
  );
}
