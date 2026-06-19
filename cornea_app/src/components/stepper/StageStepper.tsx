import { useWorkflowStore, type Stage } from "../../store/workflowStore";

const STAGES = [
  { id: 1, label: "Segment" },
  { id: 2, label: "Correct" },
  { id: 3, label: "Scar" },
] as const;

export function StageStepper() {
  const stage = useWorkflowStore((s) => s.stage);
  const setStage = useWorkflowStore((s) => s.setStage);
  const correcting = useWorkflowStore((s) => s.correcting);

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
              onClick={() => { if (!correcting) setStage(st.id as Stage); }}
              disabled={correcting}
              title={correcting ? "Save or Cancel the correction first" : undefined}
              className="flex items-center gap-2 px-3 py-1 rounded text-xs"
              style={{
                cursor: correcting ? "not-allowed" : "pointer",
                opacity: correcting && !active ? 0.4 : 1,
                backgroundColor: active ? "var(--c-accent)" : "transparent",
                color: active ? "#fff" : "var(--c-text)",
              }}
            >
              <span
                className="flex items-center justify-center rounded-full"
                style={{ width: 18, height: 18, fontSize: 11, border: `1px solid ${active ? "#fff" : "var(--c-border)"}` }}
              >
                {st.id}
              </span>
              {st.label}
            </button>
            {i < STAGES.length - 1 && <span style={{ color: "var(--c-text-dim)", margin: "0 2px" }}>›</span>}
          </div>
        );
      })}
    </div>
  );
}
