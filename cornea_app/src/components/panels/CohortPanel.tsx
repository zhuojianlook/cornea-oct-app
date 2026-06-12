/* Cohort batch processing: point at a directory of .OCT scans → group repeat scans by
   (patient, eye) → preprocess + SAM2 + scar per scan → consensus label per eye, to
   mass-produce the labeled training set. Runs server-side; this polls live progress. */

import { useEffect, useRef, useState } from "react";
import { Button, Typography, TextField, LinearProgress } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";

type ScanState = { filename: string; status: string; case_id?: string; scar_mm3?: number; error?: string };
type GroupState = {
  patient: string; eye: string; status: string; scans: ScanState[];
  consensus_case?: string; scar_volume_mm3?: number; cv_percent?: number; single_case?: string; error?: string;
};
type Status = { running: boolean; done: boolean; error: string | null; groups: GroupState[] };
type Plan = { n_groups: number; n_scans: number; groups: { patient: string; eye: string; scans: string[] }[] };

const DOT: Record<string, string> = {
  queued: "var(--c-text-dim)", running: "var(--c-accent)", preprocessing: "var(--c-accent)",
  segmenting: "var(--c-accent)", consensus: "var(--c-accent)", done: "var(--c-green)", error: "var(--c-red)",
};
const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function CohortPanel() {
  const [dir, setDir] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const pollFails = useRef(0);
  const setCaseId = useCaseStore((s) => s.setCaseId);
  const openCase = useCaseStore((s) => s.openCase);
  const exportNnunet = useCaseStore((s) => s.exportNnunet);
  const exportInfo = useCaseStore((s) => s.exportInfo);
  const setStage = useWorkflowStore((s) => s.setStage);

  useEffect(() => () => { if (pollRef.current) window.clearInterval(pollRef.current); }, []);

  const scan = async () => {
    if (!dir.trim()) return;
    setBusy(true); setPlan(null); setStatus(null); setStep(null);
    try {
      const p = await api.json<Plan>("/api/cohort/scan", "POST", JSON.stringify({ directory: dir.trim() }));
      setPlan(p);
      setStep(p.n_groups ? `${p.n_groups} eye-group(s), ${p.n_scans} scan(s) found.` : "No 3D Cornea .OCT scans found.");
    } catch (e) { setStep(`Scan failed: ${msg(e)}`); } finally { setBusy(false); }
  };

  const stopPolling = () => {
    if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
  };

  const poll = async () => {
    try {
      const s = await api.json<Status>("/api/cohort/status");
      pollFails.current = 0;
      setStatus(s);
      if (!s.running) { stopPolling(); setBusy(false); }
    } catch (e) {
      // Tolerate transient blips, but don't poll a dead backend forever with the UI
      // stuck busy — give up after several consecutive failures and surface it.
      pollFails.current += 1;
      if (pollFails.current >= 5) {
        stopPolling();
        setBusy(false);
        setStep(`Lost contact with the backend while polling cohort status: ${msg(e)}`);
      }
    }
  };

  const run = async () => {
    if (!dir.trim()) return;
    setBusy(true); setStep("Starting batch…");
    pollFails.current = 0;
    try {
      await api.json("/api/cohort/run", "POST", JSON.stringify({ directory: dir.trim(), params: {} }));
      setStep("Running — preprocess → SAM2 → scar → consensus per eye. This takes a while.");
      await poll();
      stopPolling();
      pollRef.current = window.setInterval(poll, 3000);
    } catch (e) { setStep(`Run failed: ${msg(e)}`); setBusy(false); }
  };

  const openCons = async (cid: string) => { setCaseId(cid); await openCase(); setStage(2); };

  const groups = status?.groups ?? [];
  const totalScans = groups.reduce((n, g) => n + g.scans.length, 0);
  const doneScans = groups.reduce((n, g) => n + g.scans.filter((s) => s.status === "done").length, 0);

  return (
    <div className="flex flex-col gap-2">
      <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
        Batch a folder of .OCT scans → one consensus label per (patient, eye). GPU must be free for SAM2.
      </Typography>
      <TextField size="small" label="Scans directory" value={dir} onChange={(e) => setDir(e.target.value)}
        placeholder="/path/to/OCT/scans" disabled={busy} fullWidth />
      <div className="flex gap-2">
        <Button variant="outlined" size="small" fullWidth onClick={scan} disabled={busy || !dir.trim()}>Scan plan</Button>
        <Button variant="contained" size="small" fullWidth onClick={run} disabled={busy || !plan?.n_groups}>
          Run batch
        </Button>
      </div>
      {busy && <LinearProgress />}
      {step && <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-word" }}>{step}</Typography>}

      {/* Plan preview (before running) */}
      {plan && !status && plan.groups.map((g, i) => (
        <div key={i} className="text-xs flex justify-between" style={{ color: "var(--c-text-dim)" }}>
          <span>{g.patient} {g.eye}</span><span>{g.scans.length} scans</span>
        </div>
      ))}

      {/* Live progress */}
      {status && (
        <div className="flex flex-col gap-1">
          <div className="text-[11px] uppercase tracking-wide flex justify-between" style={{ color: "var(--c-text-dim)" }}>
            <span>{status.running ? "Processing…" : status.done ? "Complete" : "Stopped"}</span>
            <span>{doneScans}/{totalScans} scans</span>
          </div>
          {status.error && <Typography variant="caption" sx={{ color: "var(--c-red)" }}>{status.error}</Typography>}
          {groups.map((g, i) => (
            <div key={i} className="rounded p-1.5" style={{ backgroundColor: "var(--c-surface2)" }}>
              <div className="flex items-center gap-2 text-xs">
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: DOT[g.status] ?? "var(--c-text-dim)", flex: "none" }} />
                <span style={{ flex: 1 }}><b>{g.patient} {g.eye}</b></span>
                {g.scar_volume_mm3 != null && <span style={{ color: "var(--c-green)" }}>{g.scar_volume_mm3} mm³ · CV {g.cv_percent}%</span>}
              </div>
              {g.scans.map((s, j) => (
                <div key={j} className="flex items-center gap-2 text-[11px] ml-3" title={s.error || s.filename}>
                  <span style={{ width: 6, height: 6, borderRadius: "50%", background: DOT[s.status] ?? "var(--c-text-dim)", flex: "none" }} />
                  <span className="truncate" style={{ flex: 1 }}>{s.filename.replace(/\.OCT$/i, "")}</span>
                  <span style={{ color: s.status === "error" ? "var(--c-red)" : "var(--c-text-dim)" }}>
                    {s.status === "done" && s.scar_mm3 != null ? `${s.scar_mm3} mm³` : s.status}
                  </span>
                </div>
              ))}
              {g.consensus_case && (
                <Button size="small" variant="text" sx={{ fontSize: 10, py: 0, mt: 0.25 }} onClick={() => openCons(g.consensus_case!)}>
                  open consensus →
                </Button>
              )}
            </div>
          ))}
          {status.done && (
            <Button variant="contained" color="success" size="small" onClick={exportNnunet} sx={{ mt: 0.5 }}>
              Export all → nnU-Net
            </Button>
          )}
          {exportInfo && <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-all" }}>{exportInfo}</Typography>}
        </div>
      )}
    </div>
  );
}
