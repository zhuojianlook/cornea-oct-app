/* "Run tree" (Steps v2 tab 1): every step the LAST preprocessing run actually took — including the
   reviewer's own inputs — as a vertical workflow tree. Spine = pipeline stages top-down; user-input nodes
   branch in from the left column; decision nodes carry an applied/declined badge with the recorded reason.
   Data = POST /api/case/{id}/oct-run-graph, a READ of manifest.oct_params + oct_iter + border_cache (never a
   re-run). Click a node → RunNodeDetail on the right. "Export for publication" → POST export-run-report,
   then a native Save dialog (Tauri) or a download link (browser) for the zip. The export result (folder,
   file list, download link) stays on screen even when the Tauri save step fails or is cancelled — the save
   error is shown as a separate warning and the save can be retried (L18). The other chain's canonical
   stages that did not run are kept in the graph but hidden from the tree behind "show unused stages" (M10).
   Dev aid: on the Vite dev server append ?fixture=1 to the URL to render the hand-written cs008 fixture
   instead of calling the sidecar (dev builds only). */

import { useEffect, useMemo, useState } from "react";
import { Button, CircularProgress, Slider } from "@mui/material";
import { resourceUrl } from "../../api/client";
import {
  exportRunReport, fetchRunGraph, runReportZipPath, saveRunReportZip,
  type RunGraph, type RunNode, type RunReportResult,
} from "../../api/runGraph";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import { layoutRunGraph, isUnusedStage, type Box } from "./runTreeLayout";
import { kindAccent, kindChip, statusChip } from "./runTreeChips";
import { RunNodeDetail } from "./RunNodeDetail";

function inTauri(): boolean {
  return typeof window !== "undefined" &&
    ("__TAURI_INTERNALS__" in window || "__TAURI__" in window || "__TAURI_IPC__" in window);
}

/** Dev-only fixture switch: Vite dev server + ?fixture in the query string. `import.meta.env.DEV` is a
    compile-time constant, so the fixture import below is dead code (not bundled) in a production build. */
function wantFixture(): boolean {
  try {
    return import.meta.env.DEV && typeof window !== "undefined" &&
      new URLSearchParams(window.location.search).has("fixture");
  } catch { return false; }
}

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

function NodeCard({ node, box, selected, onClick }: { node: RunNode; box: Box; selected: boolean; onClick: () => void }) {
  const accent = kindAccent(node.kind);
  const thumb = node.images[0];
  const dimmed = node.status === "not_run";
  const clamp = (lines: number, lh: number): React.CSSProperties => ({
    display: "-webkit-box", WebkitLineClamp: lines, WebkitBoxOrient: "vertical", overflow: "hidden", lineHeight: `${lh}px`,
    wordBreak: "break-word",
  });
  return (
    <div role="button" tabIndex={0} onClick={onClick}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } }}
      title={node.status_reason ? `${node.title} — ${node.status_reason}` : node.title}
      data-node-id={node.id}
      style={{
        position: "absolute", left: box.x, top: box.y, width: box.w, height: box.h, boxSizing: "border-box",
        border: `1px solid ${accent}`, borderRadius: 8, background: "var(--c-surface)", cursor: "pointer",
        boxShadow: selected ? "0 0 0 2px var(--c-accent)" : undefined, opacity: dimmed ? 0.6 : 1,
        display: "flex", gap: 6, padding: 4, overflow: "hidden",
      }}>
      <div style={{ width: 64, height: 40, flex: "none", background: "#000", borderRadius: 3, overflow: "hidden",
        display: "flex", alignItems: "center", justifyContent: "center", color: "var(--c-text-dim)", fontSize: 12, alignSelf: "center" }}>
        {thumb ? (
          <img src={thumb.data_url} alt="" draggable={false} style={{ width: "100%", height: "100%", objectFit: "cover", imageRendering: "auto", display: "block" }} />
        ) : "—"}
      </div>
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 2 }}>
        <div className="flex items-center gap-1" style={{ minWidth: 0, height: 14 }}>
          {kindChip(node.kind)}
          <span style={{ flex: 1 }} />
          {statusChip(node.status)}
        </div>
        {/* title wraps to two lines (M10) instead of being cut with an ellipsis */}
        <div style={{ fontSize: 12, color: "var(--c-text)", ...clamp(2, 15) }}>{node.title}</div>
        <div style={{ fontSize: 11, color: "var(--c-text-dim)", ...clamp(2, 14) }}>
          {node.summary || node.status_reason || ""}
        </div>
      </div>
    </div>
  );
}

export function RunTree() {
  const caseId = useCaseStore((s) => s.caseId);
  const segVersion = useWorkflowStore((s) => s.segVersion);
  const [graph, setGraph] = useState<RunGraph | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [sliceReq, setSliceReq] = useState<number | null>(null);
  const [sliceShown, setSliceShown] = useState<number | null>(null); // slider thumb while dragging
  const [report, setReport] = useState<RunReportResult | null>(null);
  const [reportBusy, setReportBusy] = useState(false);
  const [reportErr, setReportErr] = useState<string | null>(null);   // the export itself failed
  const [saveErr, setSaveErr] = useState<string | null>(null);       // export ok, native save step failed
  const [saveBusy, setSaveBusy] = useState(false);
  const [reportInfo, setReportInfo] = useState<string | null>(null);
  const [showUnused, setShowUnused] = useState(false);

  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;
    setBusy(true);
    setErr(null);
    const load = wantFixture()
      ? import("../../api/__fixtures__/runGraph.cs008.json").then((m) => m.default as unknown as RunGraph)
      : fetchRunGraph(caseId, { sliceIndex: sliceReq });
    load
      .then((g) => { if (!cancelled) setGraph(g); })
      .catch((e) => !cancelled && setErr(msg(e)))
      .finally(() => !cancelled && setBusy(false));
    return () => { cancelled = true; };
  }, [caseId, segVersion, sliceReq]);

  // A new case → drop the selection, slice request and the last export.
  useEffect(() => {
    setSelectedId(null); setSliceReq(null); setSliceShown(null);
    setReport(null); setReportErr(null); setSaveErr(null); setReportInfo(null);
  }, [caseId]);

  const layout = useMemo(() => (graph ? layoutRunGraph(graph, { showUnused }) : null), [graph, showUnused]);
  const nUnused = useMemo(() => (graph ? graph.nodes.filter((n) => isUnusedStage(n, graph)).length : 0), [graph]);
  const nodesById = useMemo(() => {
    const m = new Map<string, RunNode>();
    for (const n of graph?.nodes ?? []) m.set(n.id, n);
    return m;
  }, [graph]);
  const selected = selectedId ? nodesById.get(selectedId) ?? null : null;
  useEffect(() => { if (selectedId && layout && !layout.boxes[selectedId]) setSelectedId(null); }, [selectedId, layout]);

  const run = graph?.run ?? null;
  const nLat = run?.dims_raw?.[0] ?? 0;
  const modeColor = run?.mode === "corrections" ? "var(--c-green)" : run?.mode === "automatic" ? "var(--c-accent)" : "var(--c-text-dim)";
  const dimsText = (d: number[] | null | undefined) => (d && d.length ? d.join(" × ") : null);

  /** Native save step (Tauri shell only). Failures here never hide the export result: the report was
      written; the reviewer still gets the folder path + download link and can retry the save. */
  const saveViaDialog = async (r: RunReportResult) => {
    if (!caseId) return;
    setSaveBusy(true);
    setSaveErr(null);
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const dest = await save({ defaultPath: r.zip_name, filters: [{ name: "zip", extensions: ["zip"] }] });
      if (dest) {
        await saveRunReportZip(caseId, dest);
        setReportInfo(`Saved → ${dest}`);
      } else {
        setReportInfo("Save cancelled — the report is still in the exports folder");
      }
    } catch (e) {
      setSaveErr(msg(e));
    } finally {
      setSaveBusy(false);
    }
  };

  const onExport = async () => {
    if (!caseId) return;
    setReportBusy(true);
    setReportErr(null);
    setSaveErr(null);
    setReportInfo(null);
    let r: RunReportResult | null = null;
    try {
      r = await exportRunReport(caseId, { sliceIndex: sliceReq });
      setReport(r);
    } catch (e) {
      setReportErr(msg(e));
    } finally {
      setReportBusy(false);
    }
    if (r && inTauri()) await saveViaDialog(r);
  };

  const exportDisabled = reportBusy || !graph || graph.run.mode === "none";

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div className="flex items-center gap-2 px-3 py-1 border-b flex-wrap"
        style={{ borderColor: "var(--c-border)", background: "var(--c-surface)", minHeight: 32 }}>
        {run && (
          <>
            <span title="How the last run was produced"
              style={{ fontSize: 10, fontWeight: 700, color: modeColor, border: `1px solid ${modeColor}`, borderRadius: 4, padding: "0 5px", textTransform: "uppercase" }}>
              {run.mode}
            </span>
            {run.run_time && <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }} title="From the corrected volume's file time">run {run.run_time.replace("T", " ")}</span>}
            {dimsText(run.dims_raw) && (
              <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }} title="laterals × depth × frames (raw → corrected)">
                {dimsText(run.dims_raw)}{dimsText(run.dims_corrected) ? ` → ${dimsText(run.dims_corrected)}` : ""}
              </span>
            )}
            {run.warnings.map((w, i) => (
              <span key={i} title={w} style={{ color: "#f59e0b", fontSize: 11, cursor: "help" }}>⚠ {w.length > 48 ? w.slice(0, 48) + "…" : w}</span>
            ))}
          </>
        )}
        {run && nLat > 1 && (
          <span className="flex items-center gap-2 ml-2" style={{ minWidth: 220 }} title="Which sagittal slice the node images are rendered at (default: the most-edited slice)">
            <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>image slice {sliceShown ?? sliceReq ?? run.slices.most_edited}</span>
            <Slider size="small" min={0} max={nLat - 1} value={sliceShown ?? sliceReq ?? run.slices.most_edited}
              valueLabelDisplay="auto" disabled={busy} sx={{ width: 150 }}
              onChange={(_, v) => setSliceShown(v as number)}
              onChangeCommitted={(_, v) => { setSliceShown(null); setSliceReq(v as number); }} />
          </span>
        )}
        {busy && <CircularProgress size={14} />}
        <span style={{ flex: 1 }} />
        <span className="flex items-center gap-2">
          <Button size="small" variant="outlined" disabled={exportDisabled} onClick={onExport}
            startIcon={reportBusy ? <CircularProgress size={13} /> : undefined}
            title="Write figure.png/svg, diagram.svg, methods.md, nodes.json, report.html + zip into the case's exports folder"
            sx={{ textTransform: "none", fontSize: 12, py: 0.25 }}>
            📄 Export for publication
          </Button>
          {report && caseId && (
            <a href={resourceUrl(`${report.download_url || runReportZipPath(caseId)}?t=${Date.now()}`)} download={report.zip_name}
              style={{ color: "var(--c-accent)", fontSize: 12 }} title={report.zip}>⤓ Download zip</a>
          )}
          {report && inTauri() && (
            <Button size="small" variant="text" disabled={saveBusy || reportBusy} onClick={() => saveViaDialog(report)}
              title="Choose where to save the zip (the report is already written to the case's exports folder)"
              sx={{ textTransform: "none", fontSize: 12, py: 0.25, minWidth: 0 }}>
              {saveBusy ? "Saving…" : "Save zip…"}
            </Button>
          )}
        </span>
      </div>
      {reportErr && (
        <div className="px-3 py-1 border-b" style={{ borderColor: "var(--c-border)", fontSize: 11, color: "var(--c-red)" }}>
          Export failed: {reportErr}
        </div>
      )}
      {report && (
        <div className="px-3 py-1 border-b" style={{ borderColor: "var(--c-border)", fontSize: 11, color: "var(--c-text-dim)" }}
          title={report.files.join("\n")}>
          Written to <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>{report.folder}</span> ({report.files.length} files
          {reportInfo ? `; ${reportInfo}` : ""})
          {saveErr && (
            <span style={{ color: "#f59e0b", marginLeft: 8 }} title={saveErr}>
              ⚠ save failed: {saveErr} — the report is still in the folder above; use "Save zip…" to retry or "⤓ Download zip"
            </span>
          )}
        </div>
      )}
      <div className="flex flex-1 min-h-0 min-w-0">
        <div className="flex-1 min-w-0 min-h-0" style={{ overflow: "auto", position: "relative" }}>
          {err ? (
            <div className="text-center p-4" style={{ color: "var(--c-red)", fontSize: 12 }}>Couldn't build the run tree: {err}</div>
          ) : !graph || !layout ? (
            <div className="text-center p-4" style={{ color: "var(--c-text-dim)", fontSize: 13 }}>{busy ? "Building the run tree…" : "No run tree."}</div>
          ) : (
            <>
              {graph.run.mode === "none" && (
                <div className="text-center px-4 py-2" style={{ color: "var(--c-text-dim)", fontSize: 13, position: "sticky", top: 0, background: "var(--c-bg)", zIndex: 1 }}>
                  Preprocess the scan first — nothing has run on this scan yet.
                </div>
              )}
              {nUnused > 0 && (
                <label className="flex items-center gap-1 px-3 py-1" style={{ fontSize: 11, color: "var(--c-text-dim)", cursor: "pointer", userSelect: "none" }}
                  title="The other chain's canonical stages that did not run on this scan (kept in nodes.json for completeness)">
                  <input type="checkbox" checked={showUnused} onChange={(e) => setShowUnused(e.target.checked)} />
                  show {nUnused} unused stage{nUnused === 1 ? "" : "s"}
                </label>
              )}
              <div style={{ position: "relative", width: layout.width, height: layout.height, margin: "0 auto" }}>
                <svg width={layout.width} height={layout.height} style={{ position: "absolute", inset: 0, pointerEvents: "none" }} aria-hidden>
                  <defs>
                    <marker id="runtree-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                      <path d="M 0 0 L 8 4 L 0 8 z" fill="var(--c-border)" />
                    </marker>
                  </defs>
                  {layout.edges.map((e, i) => (
                    <path key={i} d={e.d} fill="none" stroke="var(--c-border)" strokeWidth={1.5}
                      strokeDasharray={e.kind === "input" ? "4 3" : undefined} markerEnd="url(#runtree-arrow)" />
                  ))}
                  {/* edge labels (e.g. "pinned", "band", "folded (line mode)") at the curve's mid-point */}
                  {layout.edges.map((e, i) => (e.label && e.labelAt) ? (
                    <text key={`l${i}`} x={e.labelAt.x} y={e.labelAt.y - 3} fontSize={10} textAnchor="middle"
                      fill="var(--c-text-dim)" stroke="var(--c-bg)" strokeWidth={3} paintOrder="stroke"
                      style={{ fontFamily: "inherit" }}>{e.label}</text>
                  ) : null)}
                </svg>
                {graph.nodes.map((n) => {
                  const box = layout.boxes[n.id];
                  if (!box) return null;
                  return <NodeCard key={n.id} node={n} box={box} selected={n.id === selectedId} onClick={() => setSelectedId(n.id)} />;
                })}
              </div>
            </>
          )}
        </div>
        <div className="flex flex-col min-h-0" style={{ width: 420, flex: "none", borderLeft: "1px solid var(--c-border)", overflow: "auto" }}>
          {selected && run ? (
            <RunNodeDetail node={selected} run={run} onClose={() => setSelectedId(null)} />
          ) : (
            <div className="p-4 text-center" style={{ color: "var(--c-text-dim)", fontSize: 12 }}>
              Click a node for its image, description, Methods sentence and numbers.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
