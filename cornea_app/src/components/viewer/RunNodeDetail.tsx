/* Detail pane for ONE node of the run-provenance tree: status banner (with the recorded reason when the
   step was declined / not run / missing data), warnings, the annotated images with their captions, the
   plain-language description, the paper-ready Methods sentence (with a Copy button), the key numbers as a
   table, and the raw params / sources for the curious. Presentational only — RunTree owns the selection. */

import { useState } from "react";
import type { RunHeader, RunNode, RunNumberValue } from "../../api/runGraph";
import { kindChip, statusChip } from "./runTreeChips";

const td: React.CSSProperties = { padding: "2px 6px", borderBottom: "1px solid var(--c-border)", fontSize: 11 };

/** Format a node number for the table. Defensive on purpose (L12): dict values are rendered as
    "key: value; key: value" (one level, nested dicts/lists as compact JSON), lists element-wise, and
    nothing ever falls through to String(object) → "[object Object]". Exported for tests / other renderers. */
export function fmtValue(v: RunNumberValue, depth = 0): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : (Number.isNaN(v) ? "NaN" : (v > 0 ? "∞" : "−∞"));
  if (typeof v === "string") return v;
  if (Array.isArray(v)) {
    if (v.length === 0) return "—";
    if (depth >= 1) return safeJson(v);
    return v.map((x) => fmtValue(x, depth + 1)).join(", ");
  }
  if (typeof v === "object") {
    const entries = Object.entries(v);
    if (entries.length === 0) return "—";
    if (depth >= 1) return safeJson(v);
    // a {"text": ..., ...} / {"value": ..., ...} record leads with its payload, the rest in parentheses
    // (e.g. manifest.difficult_reason = {text, ts} → "curvature seems not smooth (ts: 1785718940.1)")
    const lead = entries.find(([k]) => k === "text" || k === "value");
    const rest = entries.filter((e) => e !== lead).map(([k, x]) => `${k}: ${fmtValue(x, depth + 1)}`).join("; ");
    if (lead) return fmtValue(lead[1], depth + 1) + (rest ? ` (${rest})` : "");
    return rest;
  }
  return safeJson(v);
}

function safeJson(v: unknown): string {
  try {
    const s = JSON.stringify(v);
    return s === undefined ? String(v) : s;
  } catch {
    return "(unrenderable value)";
  }
}

function bannerBg(status: RunNode["status"]): string {
  if (status === "declined") return "rgba(245,158,11,0.12)";
  if (status === "missing_data" || status === "error") return "rgba(255,69,58,0.12)";
  return "var(--c-surface2)";
}

export function RunNodeDetail({ node, run, onClose }: { node: RunNode; run: RunHeader; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const copyMethod = async () => {
    try {
      await navigator.clipboard.writeText(node.method);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard unavailable (insecure context) — nothing to do */ }
  };
  const hasParams = Object.keys(node.params ?? {}).length > 0;
  const src = node.source ?? { manifest: [], files: [] };

  return (
    <div className="flex flex-col min-h-0" style={{ color: "var(--c-text)" }}>
      <div className="flex items-center gap-2 px-3 py-2 border-b" style={{ borderColor: "var(--c-border)", background: "var(--c-surface)" }}>
        {kindChip(node.kind)}
        {statusChip(node.status)}
        <span style={{ fontSize: 13, fontWeight: 600, flex: 1, minWidth: 0 }}>{node.title}</span>
        <button onClick={onClose} title="Close detail" aria-label="Close detail"
          style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: "pointer", fontSize: 12, padding: "0 6px" }}>
          ✕
        </button>
      </div>
      <div className="flex-1 min-h-0 overflow-auto p-3 flex flex-col gap-3">
        {node.status !== "applied" && (
          <div style={{ background: bannerBg(node.status), border: "1px solid var(--c-border)", borderRadius: 6, padding: "6px 8px", fontSize: 12 }}>
            <span style={{ fontWeight: 600, marginRight: 6 }}>
              {node.status === "declined" ? "Declined" : node.status === "not_run" ? "Not run" : node.status === "missing_data" ? "No data" : "Error"}
            </span>
            <span style={{ color: "var(--c-text-dim)" }}>{node.status_reason ?? "no reason recorded"}</span>
          </div>
        )}
        {node.warnings.length > 0 && (
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: "#f59e0b" }}>
            {node.warnings.map((w, i) => <li key={i}>⚠ {w}</li>)}
          </ul>
        )}
        {node.images.length > 0 ? (
          node.images.map((im, i) => (
            <figure key={i} style={{ margin: 0 }}>
              <img src={im.data_url} alt={im.caption} draggable={false}
                style={{ width: "100%", maxWidth: 640, display: "block", background: "#000", imageRendering: "auto", borderRadius: 4 }} />
              <figcaption style={{ fontSize: 11, color: "var(--c-text-dim)", marginTop: 2 }}>
                {im.caption}
                {im.slice_index != null && im.orientation !== "plot" && <span> (slice {im.slice_index})</span>}
              </figcaption>
            </figure>
          ))
        ) : (
          <div style={{ fontSize: 11, color: "var(--c-text-dim)", fontStyle: "italic" }}>
            {node.image_error ? `No image: ${node.image_error}` : "No image for this step."}
          </div>
        )}
        {node.description && <p style={{ margin: 0, fontSize: 13, lineHeight: 1.45 }}>{node.description}</p>}
        {node.method && (
          <div>
            <div className="flex items-center gap-2" style={{ fontSize: 11, color: "var(--c-text-dim)", marginBottom: 3 }}>
              <span>Method</span>
              <button onClick={copyMethod} title="Copy the Methods sentence"
                style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-accent)", cursor: "pointer", fontSize: 11, padding: "0 6px" }}>
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <blockquote style={{ margin: 0, borderLeft: "2px solid var(--c-accent)", padding: "6px 8px", fontStyle: "italic", fontSize: 12, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", background: "var(--c-surface)" }}>
              {node.method}
            </blockquote>
          </div>
        )}
        {node.numbers.length > 0 && (
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <tbody>
              {node.numbers.map((n, i) => (
                <tr key={i}>
                  <td style={{ ...td, color: "var(--c-text-dim)", whiteSpace: "nowrap" }}>{n.label}</td>
                  <td style={td}>{fmtValue(n.value)}{n.unit ? <span style={{ color: "var(--c-text-dim)" }}> {n.unit}</span> : null}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {hasParams && (
          <details style={{ fontSize: 10, color: "var(--c-text-dim)" }}>
            <summary style={{ cursor: "pointer" }}>params</summary>
            <pre style={{ margin: "4px 0 0", whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: 10 }}>{JSON.stringify(node.params, null, 2)}</pre>
          </details>
        )}
        {(src.manifest.length > 0 || src.files.length > 0) && (
          <div style={{ fontSize: 10, color: "var(--c-text-dim)" }}>
            {src.manifest.length > 0 && <div>manifest: {src.manifest.join(", ")}</div>}
            {src.files.length > 0 && <div>files: {src.files.join(", ")}</div>}
          </div>
        )}
        <div style={{ fontSize: 10, color: "var(--c-text-dim)" }}>
          {run.case_id}{run.run_time ? ` · run ${run.run_time}` : ""} · node {node.id}
        </div>
      </div>
    </div>
  );
}
