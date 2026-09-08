/* Shared chips + colours for the run-provenance tree (RunTree cards and RunNodeDetail header). */

import type { NodeKind, NodeStatus } from "../../api/runGraph";

/** Border accent by node kind (input / user / auto / decision / output). */
export function kindAccent(kind: NodeKind): string {
  switch (kind) {
    case "input": return "var(--c-accent)";
    case "user": return "#5bc0ff";
    case "decision": return "#f59e0b";
    case "output": return "var(--c-green)";
    default: return "var(--c-border)";
  }
}

const chipBase: React.CSSProperties = {
  fontSize: 9, fontWeight: 700, borderRadius: 4, padding: "0 4px", lineHeight: "14px", whiteSpace: "nowrap",
};

/** "USER" / "DECISION" / "OUTPUT" kind label; nothing for plain auto/input stages. */
export function kindChip(kind: NodeKind) {
  const label = kind === "user" ? "USER" : kind === "decision" ? "DECISION" : kind === "output" ? "OUTPUT" : null;
  if (!label) return null;
  const c = kindAccent(kind);
  return <span style={{ ...chipBase, color: c, border: `1px solid ${c}` }}>{label}</span>;
}

/** applied ✓ / declined ✗ / not run / no data / error. */
export function statusChip(status: NodeStatus) {
  const map: Record<NodeStatus, { text: string; color: string }> = {
    applied: { text: "✓ applied", color: "var(--c-green)" },
    declined: { text: "✗ declined", color: "#f59e0b" },
    not_run: { text: "not run", color: "var(--c-text-dim)" },
    missing_data: { text: "no data", color: "var(--c-red)" },
    error: { text: "error", color: "var(--c-red)" },
  };
  const m = map[status] ?? map.error;
  return <span style={{ ...chipBase, color: m.color, border: `1px solid ${m.color}` }}>{m.text}</span>;
}
