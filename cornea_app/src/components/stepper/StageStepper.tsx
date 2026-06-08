import { usePaintStore } from "../../store/paintStore";

const STAGES = [
  { id: 1, label: "Seed Paint", enabled: true },
  { id: 2, label: "Grow from Seeds", enabled: true },
  { id: 3, label: "AI Vet & Correct", enabled: false },
  { id: 4, label: "Scar Detection", enabled: true },
] as const;

export function StageStepper() {
  const stage = usePaintStore((s) => s.stage);
  const setStage = usePaintStore((s) => s.setStage);

  return (
    <div
      className="flex items-center gap-1 px-3 border-b"
      style={{ height: 40, backgroundColor: "var(--c-surface2)", borderColor: "var(--c-border)" }}
    >
      {STAGES.map((st, i) => {
        const active = st.id === stage;
        return (
          <div key={st.id} className="flex items-center">
            <button
              disabled={!st.enabled}
              onClick={() => st.enabled && setStage(st.id as 1 | 2 | 4)}
              className="flex items-center gap-2 px-3 py-1 rounded text-xs"
              style={{
                cursor: st.enabled ? "pointer" : "not-allowed",
                backgroundColor: active ? "var(--c-accent)" : "transparent",
                color: active ? "#fff" : st.enabled ? "var(--c-text)" : "var(--c-text-dim)",
                opacity: st.enabled ? 1 : 0.5,
              }}
            >
              <span
                className="flex items-center justify-center rounded-full"
                style={{
                  width: 18,
                  height: 18,
                  fontSize: 11,
                  border: `1px solid ${active ? "#fff" : "var(--c-border)"}`,
                }}
              >
                {st.id}
              </span>
              {st.label}
              {!st.enabled && <span style={{ fontSize: 9, opacity: 0.7 }}>(soon)</span>}
            </button>
            {i < STAGES.length - 1 && (
              <span style={{ color: "var(--c-text-dim)", margin: "0 2px" }}>›</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
