/* nnU-Net training proof-of-concept panel.
   Trains a segmentation model on the PER-SCAN segmentations (each scan's own labelmap, NOT the
   consensus) across all subgroups, via the standard nnU-Net workflow in an isolated venv.
   Two modeling modes + config/length dropdowns; live status + log tail. */

import { useEffect, useRef, useState } from "react";
import { MenuItem, Select, Button, CircularProgress } from "@mui/material";
import { api } from "../../api/client";

type Mode = "single3" | "cascade";
type Config = "2d" | "3d_fullres";
type Length = "short" | "full";

interface TrainStatus {
  running: boolean;
  done: boolean;
  error: string | null;
  venv_ready: boolean;
  stage: string | null;
  steps: string[];
  datasets: { id: number; name: string; n: number; scar_present?: boolean }[];
  candidate_cases: string[];
  mode: string | null;
  config: string | null;
  trainer?: string | null;
  scar_present: boolean | null;
  log_tail: string;
  started_at: string | null;
  finished_at: string | null;
  first_run_dir?: string | null;
  n_trainval?: number;
  n_test?: number;
  run_version?: number | null;
}

const sel = { fontSize: 12, ".MuiSelect-select": { py: 0.5 } } as const;

export function NnunetTrainPanel() {
  const [mode, setMode] = useState<Mode>("single3");
  const [config, setConfig] = useState<Config>("2d");
  const [length, setLength] = useState<Length>("short");
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [setupPending, setSetupPending] = useState(false);
  const logRef = useRef<HTMLPreElement | null>(null);

  const fetchStatus = async () => {
    try {
      setStatus(await api.json<TrainStatus>("/api/train/nnunet/status"));
    } catch {
      /* sidecar busy — keep last */
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  // Poll while a job is running or the venv is being set up.
  const polling = (status?.running ?? false) || setupPending;
  useEffect(() => {
    if (!polling) return;
    const t = setInterval(fetchStatus, 2500);
    return () => clearInterval(t);
  }, [polling]);

  // Stop the setup spinner once the venv is ready.
  useEffect(() => {
    if (setupPending && status?.venv_ready) setSetupPending(false);
  }, [status?.venv_ready, setupPending]);

  // Keep the log scrolled to the bottom.
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [status?.log_tail]);

  const setup = async () => {
    setSetupPending(true);
    try {
      await api.json("/api/train/nnunet/setup", "POST", JSON.stringify({}));
    } catch {
      setSetupPending(false);
    }
    fetchStatus();
  };

  const start = async () => {
    setBusy(true);
    try {
      await api.json("/api/train/nnunet/start", "POST", JSON.stringify({ mode, config, length }));
      await fetchStatus();
    } catch (e) {
      setStatus((s) => (s ? { ...s, error: e instanceof Error ? e.message : String(e) } : s));
    } finally {
      setBusy(false);
    }
  };

  const venvReady = status?.venv_ready ?? false;
  const nCases = status?.candidate_cases?.length ?? 0;
  const running = status?.running ?? false;
  const dim = "var(--c-text-dim)";

  return (
    <div className="flex flex-col gap-2">
      <span className="text-[11px]" style={{ color: dim }}>
        Trains on each scan's own segmentation (not the consensus), all subgroups. Per-scan scans
        with a labelmap: <b style={{ color: "var(--c-text)" }}>{nCases}</b>
      </span>

      {/* Model + config dropdowns */}
      <label className="text-[10px] uppercase tracking-wide" style={{ color: dim }}>Model</label>
      <Select size="small" value={mode} onChange={(e) => setMode(e.target.value as Mode)} sx={sel} disabled={running}>
        <MenuItem value="single3" sx={{ fontSize: 12 }}>Single 3-class (bg / cornea / scar)</MenuItem>
        <MenuItem value="cascade" sx={{ fontSize: 12 }}>Two-stage cascade (cornea → scar in cornea)</MenuItem>
      </Select>

      <div className="flex gap-2">
        <div className="flex flex-col gap-1" style={{ flex: 1 }}>
          <label className="text-[10px] uppercase tracking-wide" style={{ color: dim }}>Config</label>
          <Select size="small" value={config} onChange={(e) => setConfig(e.target.value as Config)} sx={sel} disabled={running}>
            <MenuItem value="2d" sx={{ fontSize: 12 }}>2D</MenuItem>
            <MenuItem value="3d_fullres" sx={{ fontSize: 12 }}>3D full-res</MenuItem>
          </Select>
        </div>
        <div className="flex flex-col gap-1" style={{ flex: 1 }}>
          <label className="text-[10px] uppercase tracking-wide" style={{ color: dim }}>Length</label>
          <Select size="small" value={length} onChange={(e) => setLength(e.target.value as Length)} sx={sel} disabled={running}>
            <MenuItem value="short" sx={{ fontSize: 12 }}>Short (~10 epochs)</MenuItem>
            <MenuItem value="full" sx={{ fontSize: 12 }}>Full (1000 epochs)</MenuItem>
          </Select>
        </div>
      </div>

      {!venvReady ? (
        <>
          <Button variant="outlined" size="small" onClick={setup} disabled={setupPending}>
            {setupPending ? "Setting up nnU-Net…" : "Set up nnU-Net (one-time)"}
          </Button>
          <span className="text-[10px]" style={{ color: dim }}>
            Creates an isolated venv that reuses your PyTorch/CUDA; installs nnU-Net there.
          </span>
        </>
      ) : (
        <Button variant="contained" size="small" onClick={start} disabled={busy || running || nCases === 0}>
          {running ? "Training…" : "Start training"}
        </Button>
      )}

      {(running || setupPending) && <CircularProgress size={16} sx={{ alignSelf: "center" }} />}

      {/* Status */}
      {status && (status.stage || status.error || status.done) && (
        <div className="text-[11px]" style={{ color: dim }}>
          {status.error ? (
            <span style={{ color: "#ff6b6b" }}>Error: {status.error}</span>
          ) : status.done ? (
            <span style={{ color: "var(--c-green)" }}>✓ Finished ({status.mode}, {status.config}) — {status.finished_at}</span>
          ) : (
            <span>{status.stage}{status.trainer ? ` · ${status.trainer}` : ""}</span>
          )}
          {(status.n_trainval != null || status.n_test != null) && (
            <div style={{ marginTop: 2 }}>split: {status.n_trainval ?? "?"} train/val · {status.n_test ?? "?"} test (patient-grouped)</div>
          )}
          {status.done && status.first_run_dir && (
            <div style={{ marginTop: 2, color: "var(--c-text)", wordBreak: "break-all" }}>
              First-Run Folder{status.run_version != null ? ` v${status.run_version}` : ""}:{" "}
              <code style={{ fontSize: 10 }}>{status.first_run_dir}</code>
            </div>
          )}
          {status.datasets?.length > 0 && (
            <div style={{ marginTop: 2 }}>
              {status.datasets.map((d) => (
                <div key={d.id}>· {d.name} — {d.n} case(s){d.scar_present === false ? " · no scar" : ""}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {status?.log_tail && (running || status.error || status.done) && (
        <pre ref={logRef} style={{
          fontSize: 10, lineHeight: 1.35, color: dim, background: "var(--c-bg)", border: "1px solid var(--c-border)",
          borderRadius: 4, padding: 6, margin: 0, maxHeight: 160, overflow: "auto", whiteSpace: "pre-wrap", wordBreak: "break-all",
        }}>{status.log_tail}</pre>
      )}
    </div>
  );
}
