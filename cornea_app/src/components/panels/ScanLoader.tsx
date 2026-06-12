/* Unified loader: upload ONE scan or SEVERAL repeat scans of the same eye.
   Two steps so you can SEE the raw OCT before committing to SAM2:
     1. Load   — upload only; lands on the first scan so you can scrub it in the
                 viewer (no segmentation). Click any scan row to preview it. A scan
                 that was already segmented previews its existing overlay instead.
     2. Run SAM2 — segment each scan (SAM2 + scar); with >1 successful scan a
                 consensus is built. Scans that fail to segment are excluded.
   The result (single case or consensus) opens at the Correct stage. */

import { useRef, useState } from "react";
import { Button, Typography, LinearProgress } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import type { ConsensusReport } from "../../api/types";

type ScanStatus = "queued" | "uploading" | "ready" | "segmenting" | "done" | "error";
interface Scan {
  filename: string;
  caseId?: string;
  status: ScanStatus;
  scarVol?: number;
  error?: string;
}

const DOT: Record<ScanStatus, string> = {
  queued: "var(--c-text-dim)",
  uploading: "var(--c-accent)",
  ready: "var(--c-accent)",
  segmenting: "var(--c-accent)",
  done: "var(--c-green)",
  error: "var(--c-red)",
};
const LABEL: Record<ScanStatus, string> = {
  queued: "queued",
  uploading: "uploading…",
  ready: "ready",
  segmenting: "segmenting…",
  done: "done",
  error: "failed",
};

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function ScanLoader() {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [scans, setScans] = useState<Scan[]>([]);
  const [loaded, setLoaded] = useState(false); // uploaded + previewable, not yet segmented
  const [completed, setCompleted] = useState(false); // segmentation/consensus finished
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<string | null>(null);
  const [report, setReport] = useState<ConsensusReport | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const setCaseId = useCaseStore((s) => s.setCaseId);
  const openCase = useCaseStore((s) => s.openCase);
  const setStage = useWorkflowStore((s) => s.setStage);
  const wfSet = useWorkflowStore((s) => s.set);

  const patch = (i: number, p: Partial<Scan>) =>
    setScans((cur) => cur.map((s, k) => (k === i ? { ...s, ...p } : s)));

  // Make a scan the active case so the viewer shows it.
  const openScan = async (caseId: string) => {
    setActiveId(caseId);
    setCaseId(caseId);
    await openCase(); // openCase swallows its own errors → never throws here
  };

  // Row click → preview a scan (busy-gated so clicks serialise and don't race).
  const preview = async (caseId?: string) => {
    if (!caseId || busy) return;
    setBusy(true);
    try {
      await openScan(caseId);
      setStep("Scrub slices in the viewer. Run SAM2 when ready.");
    } finally {
      setBusy(false);
    }
  };

  const resetSelection = (fs: File[]) => {
    setFiles(fs);
    setScans(fs.map((f) => ({ filename: f.name, status: "queued" })));
    setLoaded(false);
    setCompleted(false);
    setReport(null);
    setActiveId(null);
    setStep(null);
  };

  // Step 1 — upload only (no segmentation). Land on the first scan to preview it.
  const load = async () => {
    if (files.length < 1) return;
    setBusy(true);
    setReport(null);
    setLoaded(false);
    setCompleted(false);
    // Clear any segmentation routing/flags left from a prior case so the fresh
    // (unsegmented) scan can't show a stale overlay or enable Correct/Scar.
    wfSet("segLoaded", false);
    wfSet("previewGroup", null);
    setScans(files.map((f) => ({ filename: f.name, status: "uploading" })));
    try {
      setStep(`Uploading ${files.length} scan(s)…`);
      const up = await api.upload<{ cases: { case_id: string; filename: string; segmented: boolean }[] }>(
        "/api/consensus/upload",
        files,
      );
      setScans(up.cases.map((c) => ({ filename: c.filename, caseId: c.case_id, status: c.segmented ? "done" : "ready" })));
      setLoaded(true);
      setStage(1); // pre-segmentation: stay on the Segment stage
      await openScan(up.cases[0]?.case_id ?? "");
      setStep("Loaded. Click a scan to preview it, scrub slices in the viewer, then Run SAM2.");
    } catch (e) {
      setStep(`Load failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Step 2 — segment each scan (skip already-segmented), then consensus over the
  // scans that SUCCEEDED. Failed scans are excluded and surfaced, never silently
  // advanced past.
  const runSam2 = async () => {
    if (scans.every((s) => !s.caseId)) return;
    setBusy(true);
    setReport(null);
    const ok: string[] = [];
    const failed: string[] = [];
    try {
      for (let i = 0; i < scans.length; i++) {
        const s = scans[i];
        if (!s.caseId) continue;
        if (s.status === "done") {
          ok.push(s.caseId);
          continue;
        }
        setStep(`Segmenting ${i + 1}/${scans.length} — ${s.filename} (SAM2 + scar, ~2–3 min)`);
        patch(i, { status: "segmenting" });
        try {
          const r = await api.json<{ scar_volume_mm3: number }>(
            `/api/case/${s.caseId}/consensus-segment`,
            "POST",
            JSON.stringify({}),
          );
          patch(i, { status: "done", scarVol: r.scar_volume_mm3 });
          ok.push(s.caseId);
        } catch (e) {
          patch(i, { status: "error", error: msg(e) });
          failed.push(s.filename);
        }
      }

      if (ok.length < 1) {
        // Nothing segmented — stay on Segment with a persistent, actionable message.
        setStep(`Segmentation failed${failed.length ? `: ${failed.join(", ")}` : ""}. Fix and Run SAM2 again.`);
        return;
      }

      let active = ok[0];
      if (ok.length > 1) {
        setStep("Registering repeat scans and voting the scar…");
        const res = await api.json<{ consensus_case: string; report: ConsensusReport }>(
          "/api/consensus/build",
          "POST",
          JSON.stringify({ cases: ok }),
        );
        active = res.consensus_case;
        setReport(res.report);
      }

      await openScan(active);
      setStage(2); // land in Correct on the (single or consensus) result
      setCompleted(true);
      setStep(
        failed.length
          ? `Opened ${ok.length}-scan result. Excluded ${failed.length} failed: ${failed.join(", ")}.`
          : null,
      );
    } catch (e) {
      setStep(`Failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const row = (label: string, value: React.ReactNode) => (
    <div className="flex justify-between text-xs">
      <span style={{ color: "var(--c-text-dim)" }}>{label}</span>
      <span>{value}</span>
    </div>
  );

  const scanCount = scans.filter((s) => s.caseId).length;

  return (
    <div className="flex flex-col gap-2">
      <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
        Upload one scan, or several repeat scans of the same eye (auto-consensus).
      </Typography>
      <input
        ref={fileRef}
        type="file"
        accept=".nrrd,.nii,.gz,.mha,.mhd,.dcm,.dicom"
        multiple
        hidden
        onChange={(e) => resetSelection(Array.from(e.target.files ?? []))}
      />
      <Button variant="outlined" size="small" onClick={() => fileRef.current?.click()} disabled={busy}>
        Select scan(s)…
      </Button>

      {scans.length > 0 && (
        <div className="flex flex-col gap-1">
          {loaded && (
            <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
              Loaded scans — click to preview
            </div>
          )}
          {scans.map((s, i) => {
            const clickable = loaded && !busy && !!s.caseId;
            const active = !!s.caseId && s.caseId === activeId;
            return (
              <div
                key={i}
                role={clickable ? "button" : undefined}
                tabIndex={clickable ? 0 : -1}
                aria-pressed={clickable ? active : undefined}
                onClick={() => clickable && preview(s.caseId)}
                onKeyDown={(e) => {
                  if (clickable && (e.key === "Enter" || e.key === " ")) {
                    e.preventDefault();
                    preview(s.caseId);
                  }
                }}
                title={s.error || (clickable ? `Preview ${s.filename}` : s.filename)}
                className="flex items-center gap-2 text-xs rounded px-1.5 py-2"
                style={{
                  cursor: clickable ? "pointer" : "default",
                  background: active ? "var(--c-surface2)" : "transparent",
                  borderLeft: active ? "2px solid var(--c-accent)" : "2px solid transparent",
                }}
              >
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: DOT[s.status], flex: "none" }} />
                <span className="truncate" style={{ flex: 1, color: active ? "var(--c-text)" : undefined }}>
                  {s.filename.replace(/^preprocessed_/, "")}
                </span>
                <span style={{ color: s.status === "error" ? "var(--c-red)" : "var(--c-text-dim)" }}>
                  {s.status === "done" && s.scarVol != null ? `${s.scarVol} mm³` : LABEL[s.status]}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {!loaded ? (
        <Button variant="contained" size="small" onClick={load} disabled={busy || files.length < 1}>
          {files.length > 1 ? `Load ${files.length} scans` : "Load scan"}
        </Button>
      ) : (
        <Button variant="contained" size="small" onClick={runSam2} disabled={busy || completed}>
          {completed ? "Done" : scanCount > 1 ? "Run SAM2 + build consensus" : "Run SAM2"}
        </Button>
      )}
      {busy && <LinearProgress />}
      {step && (
        <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-word" }}>
          {step}
        </Typography>
      )}

      {report && (
        <div className="rounded p-2 flex flex-col gap-1" style={{ backgroundColor: "var(--c-surface2)", borderLeft: "3px solid var(--c-green)" }}>
          <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
            Consensus reproducibility ({report.n_scans} scans)
          </div>
          {row("Scar volume", <b>{report.scar_volume_mm3.mean} ± {report.scar_volume_mm3.std} mm³</b>)}
          {row("Volume CV", <b>{report.scar_volume_mm3.cv_percent}%</b>)}
          {row("Consensus scar", `${report.consensus_scar_mm3} mm³`)}
          {row("All-scan core", `${report.core_full_agreement_mm3} mm³`)}
          {report.mean_pairwise_scar_dice != null && row("Scar Dice (test-retest)", report.mean_pairwise_scar_dice)}
          <Typography variant="caption" sx={{ color: "var(--c-text-dim)", mt: 0.5 }}>
            Volume reproduces well; shape only partly (repeats image slightly different patches).
            Open the scan tabs to compare each scan with the consensus.
          </Typography>
          {report.per_scan.some((p) => p.low_correspondence) && (
            <Typography variant="caption" sx={{ color: "var(--c-amber, #e0a800)" }}>
              Low correspondence:{" "}
              {report.per_scan
                .filter((p) => p.low_correspondence)
                .map((p) => p.case.split("_").pop())
                .join(", ")}{" "}
              (likely a different FOV)
            </Typography>
          )}
        </div>
      )}
    </div>
  );
}
