/* Preprocessing DECISION TREE: the corneal-flattening algorithm shown as a top-down flow. The two surface
   detectors run IN PARALLEL — the native DP path and the legacy (old-method) RANSAC path — shown as two
   side-by-side branches that CONVERGE at the DP scar-guard cross-check (DP is kept in the vicinity of the
   legacy surface so it can't lock onto a bright internal scar). Below the merge: the shared per-slice stages
   (quadratic fit → 3D active → column warp) and then the whole-volume decisions (keep-best iteration,
   inter-slice smoothing, axial ping-pong refine, manual depth nudges). Per-slice stages carry an image of
   ONE sagittal slice (pick which slice with the selector) so you can SEE the detected border + fit on any
   slice; images are now drawn at the morphologically-correct ~2:1 aspect (no longer horizontally squashed). */

import { useEffect, useMemo, useState } from "react";
import { CircularProgress, Slider } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";

interface Step { label: string; data_url?: string; kind?: string; branch?: string; group?: string; lane?: string; }

export function StepsViewer({ onClose }: { onClose: () => void }) {
  const caseId = useCaseStore((s) => s.caseId);
  const segSig = useWorkflowStore((s) => s.segVersion); // re-render after a re-preprocess
  const [steps, setSteps] = useState<Step[]>([]);
  const [slices, setSlices] = useState(0);
  const [index, setIndex] = useState(0);
  const [sliceReq, setSliceReq] = useState<number | null>(null); // user-chosen sagittal slice (null = central)
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;
    setBusy(true);
    setErr(null);
    const body = sliceReq == null ? {} : { slice_index: sliceReq };
    api
      .json<{ steps: Step[]; slices: number; index: number }>(
        `/api/case/${caseId}/oct-preprocess-steps`, "POST", JSON.stringify(body),
      )
      .then((r) => { if (cancelled) return; setSteps(r.steps || []); setSlices(r.slices || 0); setIndex(r.index || 0); })
      .catch((e) => !cancelled && setErr(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancelled && setBusy(false));
    return () => { cancelled = true; };
  }, [caseId, segSig, sliceReq]);

  // Group the flat step list into blocks: a "full" step is one full-width node; a contiguous run of
  // non-full steps is one PARALLEL block (dp items left, legacy items right) that converges at the next
  // full node below. (lane defaults to "full" so older backends / volume nodes still render.)
  const blocks = useMemo(() => {
    const out: ({ type: "full"; step: Step } | { type: "branch"; dp: Step[]; legacy: Step[] })[] = [];
    let i = 0;
    while (i < steps.length) {
      const lane = steps[i].lane ?? "full";
      if (lane === "full") { out.push({ type: "full", step: steps[i] }); i++; continue; }
      const dp: Step[] = []; const legacy: Step[] = [];
      while (i < steps.length && (steps[i].lane ?? "full") !== "full") {
        if ((steps[i].lane ?? "") === "legacy") legacy.push(steps[i]); else dp.push(steps[i]);
        i++;
      }
      out.push({ type: "branch", dp, legacy });
    }
    return out;
  }, [steps]);

  const connector = (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", height: 18 }}>
      <div style={{ width: 2, flex: 1, background: "var(--c-border)" }} />
      <span style={{ color: "var(--c-border)", fontSize: 11, lineHeight: 1 }}>▼</span>
    </div>
  );

  const node = (s: Step, key: React.Key) => {
    const isDecision = s.kind === "decision";
    const isVolume = s.group === "volume";
    const accent = isVolume ? "var(--c-accent)" : isDecision ? "#f59e0b" : "var(--c-border)";
    return (
      <div key={key} style={{ width: "100%", border: `1px solid ${accent}`, borderRadius: 8, overflow: "hidden", background: "var(--c-surface)" }}>
        <div className="px-2 py-1 flex items-center gap-2" style={{ background: "var(--c-surface2)", borderBottom: s.data_url ? "1px solid var(--c-border)" : "none" }}>
          {isDecision && <span style={{ fontSize: 10, fontWeight: 700, color: accent, border: `1px solid ${accent}`, borderRadius: 4, padding: "0 4px" }}>{isVolume ? "VOLUME" : "DECISION"}</span>}
          <span style={{ fontSize: 12, color: "var(--c-text)" }}>{s.label}</span>
        </div>
        {s.data_url && (
          <img src={s.data_url} alt={s.label} draggable={false}
            style={{ width: "100%", display: "block", imageRendering: "pixelated", background: "#000" }} />
        )}
        {s.branch && (
          <div className="px-2 py-1" style={{ fontSize: 11, color: "var(--c-text-dim)", fontStyle: "italic" }}>
            ↳ {s.branch}
          </div>
        )}
      </div>
    );
  };

  // One column of a parallel branch (a vertical stack of nodes with connectors + a coloured header).
  const laneColumn = (title: string, color: string, items: Step[]) => (
    <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", alignItems: "stretch" }}>
      <div style={{ textAlign: "center", fontSize: 11, fontWeight: 700, color, border: `1px solid ${color}`,
        borderRadius: 6, padding: "1px 4px", marginBottom: 6, background: "var(--c-surface)" }}>{title}</div>
      {items.map((s, j) => (
        <div key={j} style={{ width: "100%", display: "flex", flexDirection: "column", alignItems: "center" }}>
          {j > 0 && connector}
          {node(s, j)}
        </div>
      ))}
    </div>
  );

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div className="flex items-center gap-2 px-3 py-1 border-b flex-wrap"
        style={{ borderColor: "var(--c-border)", background: "var(--c-surface)" }}>
        <button onClick={onClose}
          style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-accent)", cursor: "pointer", fontSize: 12, padding: "2px 8px" }}>
          ← 3D view
        </button>
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
          Preprocessing decision tree — both detectors run in parallel (red = detected border, green = legacy, blue = quadratic fit) and converge at the scar-guard.
        </span>
        {slices > 1 && (
          <span className="flex items-center gap-2 ml-2" style={{ minWidth: 220 }} title="Which sagittal slice to show the detected border + fit on">
            <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>slice {index}/{slices - 1}</span>
            <Slider size="small" min={0} max={slices - 1} value={sliceReq ?? index}
              valueLabelDisplay="auto" disabled={busy} sx={{ width: 150 }}
              onChange={(_, v) => setIndex(v as number)}
              onChangeCommitted={(_, v) => setSliceReq(v as number)} />
          </span>
        )}
        {busy && <CircularProgress size={14} />}
      </div>
      <div className="flex-1 min-h-0 overflow-auto p-4">
        {err ? (
          <div className="text-center" style={{ color: "var(--c-red)", fontSize: 12 }}>Couldn't render steps: {err}</div>
        ) : busy && steps.length === 0 ? (
          <div className="text-center" style={{ color: "var(--c-text-dim)", fontSize: 13 }}>Rendering steps…</div>
        ) : steps.length === 0 ? (
          <div className="text-center" style={{ color: "var(--c-text-dim)", fontSize: 13 }}>No steps — preprocess the scan first.</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", maxWidth: 760, margin: "0 auto" }}>
            {blocks.map((b, i) => (
              <div key={i} style={{ width: "100%", display: "flex", flexDirection: "column", alignItems: "center" }}>
                {i > 0 && connector}
                {b.type === "full" ? (
                  node(b.step, i)
                ) : (
                  <div style={{ width: "100%" }}>
                    <div style={{ textAlign: "center", fontSize: 10, color: "var(--c-text-dim)", marginBottom: 4 }}>
                      ⑂ two surface detectors run in parallel — they converge at the scar-guard below
                    </div>
                    <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                      {laneColumn("DP detector (native)", "#ff6b6b", b.dp)}
                      {laneColumn("Legacy (cross-check)", "#39d98a", b.legacy)}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
