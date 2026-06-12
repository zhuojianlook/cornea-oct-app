/* OCT preprocessing loader (Optovue Avanti .OCT → corrected volume).
   Flow: load .OCT files OR a whole directory → click a scan to scrub the raw B-scans →
   mark Scar / Control (and a scar frame range) → tune correction params → select scans →
   Run preprocessing (OCT→correct, correct Avanti geometry) → Run SAM2 + consensus. */

import { useEffect, useRef, useState } from "react";
import { Button, Typography, LinearProgress, Slider, Checkbox, ToggleButton, ToggleButtonGroup, Collapse } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import type { ConsensusReport } from "../../api/types";

type Status = "queued" | "uploading" | "ready" | "preprocessing" | "done" | "error";
type Cls = "scar" | "control";
interface OctScan {
  filename: string;
  caseId?: string;
  patient?: string;
  eye?: string;
  nVolumes?: number;
  nFrames: number;
  status: Status;
  classification?: Cls;
  scarRange: [number, number];
  selected: boolean;
  error?: string;
}

const DOT: Record<Status, string> = {
  queued: "var(--c-text-dim)", uploading: "var(--c-accent)", ready: "var(--c-accent)",
  preprocessing: "var(--c-accent)", done: "var(--c-green)", error: "var(--c-red)",
};

// Smoother params (oct_preprocess.DEFAULT_PARAMS) exposed as sliders.
interface Param { key: string; label: string; min: number; max: number; step: number; def: number; }
const PARAMS: Param[] = [
  { key: "sigma", label: "Gaussian σ", min: 0.5, max: 5, step: 0.1, def: 2.0 },
  { key: "max_jump", label: "Max jump", min: 1, max: 50, step: 1, def: 10 },
  { key: "median_filter_size", label: "Median size", min: 3, max: 15, step: 2, def: 5 },
  { key: "d", label: "Bilateral d", min: 3, max: 15, step: 1, def: 9 },
  { key: "sigmaColor", label: "σ color", min: 10, max: 150, step: 5, def: 75 },
  { key: "sigmaSpace", label: "σ space", min: 10, max: 150, step: 5, def: 75 },
  { key: "side_window", label: "Side window", min: 5, max: 30, step: 1, def: 10 },
  { key: "side_threshold_factor", label: "Side thresh", min: 1, max: 5, step: 0.1, def: 2.0 },
  { key: "residual_threshold", label: "RANSAC resid", min: 1, max: 10, step: 0.5, def: 5.0 },
  { key: "active_threshold", label: "3D active thresh", min: 1, max: 20, step: 1, def: 5 },
  { key: "corr_factor", label: "Correction ×", min: 0, max: 1, step: 0.05, def: 1.0 },
];
const defaultParams = (): Record<string, number> => Object.fromEntries(PARAMS.map((p) => [p.key, p.def]));

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function OctLoader() {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const dirRef = useRef<HTMLInputElement | null>(null);
  const filesRef = useRef<File[]>([]);
  const [scans, setScans] = useState<OctScan[]>([]);

  // webkitdirectory isn't in the React input typings — set it imperatively.
  useEffect(() => {
    if (dirRef.current) {
      dirRef.current.setAttribute("webkitdirectory", "");
      dirRef.current.setAttribute("directory", "");
    }
  }, []);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [params, setParams] = useState<Record<string, number>>(defaultParams());
  const [paramsOpen, setParamsOpen] = useState(false);
  const [report, setReport] = useState<ConsensusReport | null>(null);
  const setCaseId = useCaseStore((s) => s.setCaseId);
  const openCase = useCaseStore((s) => s.openCase);
  const setStage = useWorkflowStore((s) => s.setStage);
  const initTabs = useWorkflowStore((s) => s.initTabs);

  const patch = (i: number, p: Partial<OctScan>) =>
    setScans((cur) => cur.map((s, k) => (k === i ? { ...s, ...p } : s)));

  // Keep only .OCT (+ companion .txt) from a file/dir selection.
  const pickFiles = (fs: File[]) => fs.filter((f) => /\.(oct|txt)$/i.test(f.name));

  const onPicked = (fs: File[]) => {
    const keep = pickFiles(fs);
    const octs = keep.filter((f) => /\.oct$/i.test(f.name));
    // An .OCT can't be read without its companion .txt (POCT filespec); warn up front.
    const txtStems = new Set(keep.filter((f) => /\.txt$/i.test(f.name)).map((f) => f.name.replace(/\.txt$/i, "").toLowerCase()));
    const missing = octs.filter((o) => !txtStems.has(o.name.replace(/\.oct$/i, "").toLowerCase()));
    setScans(octs.map((f) => ({ filename: f.name, nFrames: 101, status: "queued", scarRange: [1, 101], selected: true })));
    setLoaded(false);
    setReport(null);
    setActiveId(null);
    filesRef.current = keep;
    setStep(missing.length
      ? `⚠ ${missing.length}/${octs.length} .OCT are missing their .txt companion — also pick the .txt files, or use Folder. An .OCT can't be read without it.`
      : `${octs.length} .OCT + companion .txt selected.`);
  };

  const load = async () => {
    const files = filesRef.current;
    if (!files.length) return;
    setBusy(true);
    setReport(null);
    setScans((cur) => cur.map((s) => ({ ...s, status: "uploading" })));
    try {
      setStep(`Uploading ${files.length} file(s)…`);
      const up = await api.upload<{ cases: { case_id: string; filename: string; patient: string; eye: string; n_volumes: number; error?: string }[] }>(
        "/api/oct/upload",
        files,
      );
      setScans(up.cases.map((c) => ({
        filename: c.filename, caseId: c.case_id, patient: c.patient, eye: c.eye, nVolumes: c.n_volumes,
        nFrames: 101, status: c.error ? "error" : "ready", error: c.error, scarRange: [1, 101], selected: !c.error,
      })));
      setLoaded(true);
      setStage(1);
      setStep("Loaded. Click a scan to scrub it, mark Scar/Control, then preprocess the selected.");
      const first = up.cases.find((c) => !c.error);
      if (first) await preview(first.case_id);
    } catch (e) {
      setStep(`Load failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Scrub a scan: materialise its raw z-stack + show grayscale in the viewer.
  const preview = async (caseId?: string) => {
    if (!caseId || busy) return;
    setBusy(true);
    setActiveId(caseId);
    try {
      setCaseId(caseId);
      const r = await api.json<{ n_frames?: number; preprocessed?: boolean }>(
        `/api/case/${caseId}/oct-volume`, "POST", JSON.stringify({}),
      );
      // Use the scan's REAL frame count for the scar-range slider (not a hardcoded 101).
      const nf = r.n_frames && r.n_frames > 1 ? r.n_frames : 101;
      setScans((cur) => cur.map((s) => (s.caseId === caseId
        ? { ...s, nFrames: nf, scarRange: [Math.min(s.scarRange[0], nf), Math.min(s.scarRange[1], nf)] }
        : s)));
      await openCase();
      initTabs(false); // grayscale routing + refetch
      setStep(r.preprocessed
        ? "Showing the corrected volume. Run SAM2 + consensus when ready."
        : "Scrub the B-scans in the viewer. Mark Scar/Control + range, then preprocess.");
    } catch (e) {
      setStep(`Preview failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Changing params/marks invalidates any already-corrected scans (result is now stale).
  const markStale = (pred: (s: OctScan) => boolean) =>
    setScans((cur) => cur.map((s) => (pred(s) && s.status === "done" ? { ...s, status: "ready" } : s)));
  const setParam = (key: string, v: number) => {
    setParams((cur) => ({ ...cur, [key]: v }));
    markStale(() => true);
  };

  const runPreprocess = async () => {
    // Skip already-corrected (non-stale) scans — re-clicking is a no-op for them.
    const sel = scans.map((s, i) => ({ s, i })).filter(({ s }) =>
      s.selected && s.caseId && s.status !== "error" && s.status !== "done");
    if (!sel.length) {
      setStep("Nothing to preprocess (selected scans are already corrected — change params or marks to re-run).");
      return;
    }
    setBusy(true);
    setReport(null);
    try {
      for (let k = 0; k < sel.length; k++) {
        const { s, i } = sel[k];
        setStep(`Preprocessing ${k + 1}/${sel.length} — ${s.filename} (OCT→correct, ~25s)`);
        patch(i, { status: "preprocessing" });
        try {
          await api.json(`/api/case/${s.caseId}/oct-preprocess`, "POST", JSON.stringify({
            params,
            classification: s.classification ?? null,
            scar_range: s.classification === "scar" ? s.scarRange : null,
          }));
          patch(i, { status: "done" });
          // If the corrected scan is the one on screen, refresh the viewer to show it.
          if (s.caseId === activeId) initTabs(false);
        } catch (e) {
          patch(i, { status: "error", error: msg(e) });
        }
      }
      setStep("Preprocessing complete. Now Run SAM2 + consensus on the corrected scans.");
    } catch (e) {
      setStep(`Preprocessing failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Segment each preprocessed scan + (if >1) build the consensus, then open the result.
  const runSamConsensus = async () => {
    const ok = scans.filter((s) => s.selected && s.caseId && s.status === "done").map((s) => s.caseId!) as string[];
    if (ok.length < 1) {
      setStep("Preprocess at least one scan first.");
      return;
    }
    setBusy(true);
    setReport(null);
    try {
      for (const cid of ok) {
        setStep(`Segmenting ${cid} (SAM2 + scar, ~2–3 min)…`);
        await api.json(`/api/case/${cid}/consensus-segment`, "POST", JSON.stringify({}));
      }
      let active = ok[0];
      if (ok.length > 1) {
        setStep("Registering repeat scans and voting the scar…");
        const res = await api.json<{ consensus_case: string; report: ConsensusReport }>(
          "/api/consensus/build", "POST", JSON.stringify({ cases: ok }),
        );
        active = res.consensus_case;
        setReport(res.report);
      }
      setStep("Opening result…");
      setCaseId(active);
      await openCase();
      setStage(2);
      setStep(null);
    } catch (e) {
      setStep(`SAM2 / consensus failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const nDone = scans.filter((s) => s.status === "done" && s.selected).length;
  const nToRun = scans.filter((s) => s.selected && s.caseId && s.status !== "error" && s.status !== "done").length;

  return (
    <div className="flex flex-col gap-2">
      <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
        Load Optovue .OCT scans — each needs its <b>.txt</b> next to it (Folder grabs both),
        then scrub & mark scar/control and preprocess.
      </Typography>
      <input ref={fileRef} type="file" accept=".oct,.txt" multiple hidden
        onChange={(e) => onPicked(Array.from(e.target.files ?? []))} />
      {/* webkitdirectory (set imperatively above): load a whole folder at once */}
      <input ref={dirRef} type="file" multiple hidden
        onChange={(e) => onPicked(Array.from(e.target.files ?? []))} />
      <div className="flex gap-2">
        <Button variant="outlined" size="small" fullWidth onClick={() => fileRef.current?.click()} disabled={busy}>
          Files…
        </Button>
        <Button variant="outlined" size="small" fullWidth onClick={() => dirRef.current?.click()} disabled={busy}>
          Folder…
        </Button>
      </div>

      {scans.length > 0 && !loaded && (
        <Button variant="contained" size="small" onClick={load} disabled={busy || scans.length < 1}>
          Load {scans.length} scan{scans.length === 1 ? "" : "s"}
        </Button>
      )}

      {loaded && scans.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
            Scans — click to scrub
          </div>
          {scans.map((s, i) => {
            const active = !!s.caseId && s.caseId === activeId;
            const clickable = !busy && !!s.caseId && s.status !== "error";
            return (
              <div key={i} className="rounded px-1.5 py-1" style={{ background: active ? "var(--c-surface2)" : "transparent", borderLeft: active ? "2px solid var(--c-accent)" : "2px solid transparent" }}>
                <div className="flex items-center gap-1.5 text-xs">
                  <Checkbox size="small" checked={s.selected} disabled={busy || s.status === "error"} sx={{ p: 0.25 }}
                    onChange={(e) => patch(i, { selected: e.target.checked })} />
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: DOT[s.status], flex: "none" }} />
                  <span className="truncate" style={{ flex: 1, cursor: clickable ? "pointer" : "default" }}
                    title={s.error || s.filename} onClick={() => clickable && preview(s.caseId)}>
                    {s.filename.replace(/\.OCT$/i, "")}
                  </span>
                  <span style={{ color: s.status === "error" ? "var(--c-red)" : "var(--c-text-dim)" }}>
                    {s.status === "error" ? "failed" : s.status === "done" ? "✓" : s.status}
                  </span>
                </div>
                {s.status !== "error" && (
                  <div className="flex items-center gap-2 mt-1 ml-6">
                    <ToggleButtonGroup size="small" exclusive value={s.classification ?? null}
                      onChange={(_, v) => patch(i, { classification: v || undefined, status: s.status === "done" ? "ready" : s.status })}>
                      <ToggleButton value="scar" sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Scar</ToggleButton>
                      <ToggleButton value="control" sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Control</ToggleButton>
                    </ToggleButtonGroup>
                  </div>
                )}
                {s.classification === "scar" && (
                  <div className="ml-6 mt-1 pr-2">
                    <div className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>scar frames {s.scarRange[0]}–{s.scarRange[1]}</div>
                    <Slider size="small" min={1} max={s.nFrames} value={s.scarRange} disabled={busy}
                      onChange={(_, v) => patch(i, { scarRange: v as [number, number], status: s.status === "done" ? "ready" : s.status })} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {loaded && (
        <>
          <button className="text-[11px] uppercase tracking-wide text-left" style={{ color: "var(--c-text-dim)", cursor: "pointer", background: "none", border: "none", padding: 0 }}
            onClick={() => setParamsOpen((o) => !o)}>
            {paramsOpen ? "▾" : "▸"} Correction parameters
            <span style={{ marginLeft: 6, textTransform: "none" }} onClick={(e) => { e.stopPropagation(); setParams(defaultParams()); }}>· reset</span>
          </button>
          <Collapse in={paramsOpen}>
            <div className="flex flex-col gap-1 px-1">
              {PARAMS.map((p) => (
                <div key={p.key} className="flex items-center gap-2">
                  <span className="text-[10px]" style={{ width: 88, color: "var(--c-text-dim)" }}>{p.label}</span>
                  <Slider size="small" min={p.min} max={p.max} step={p.step} value={params[p.key]}
                    disabled={busy} onChange={(_, v) => setParam(p.key, v as number)} />
                  <span className="text-[10px]" style={{ width: 28, textAlign: "right" }}>{params[p.key]}</span>
                </div>
              ))}
            </div>
          </Collapse>

          <Button variant="contained" size="small" onClick={runPreprocess} disabled={busy || nToRun < 1}>
            Preprocess selected ({nToRun})
          </Button>
          {nDone > 0 && (
            <Button variant="contained" color="secondary" size="small" onClick={runSamConsensus} disabled={busy}>
              {nDone > 1 ? `Run SAM2 + consensus (${nDone})` : "Run SAM2"}
            </Button>
          )}
        </>
      )}

      {busy && <LinearProgress />}
      {step && (
        <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-word" }}>{step}</Typography>
      )}
      {report && (
        <div className="rounded p-2 flex flex-col gap-1" style={{ backgroundColor: "var(--c-surface2)", borderLeft: "3px solid var(--c-green)" }}>
          <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
            Consensus ({report.n_scans} scans)
          </div>
          <div className="flex justify-between text-xs"><span style={{ color: "var(--c-text-dim)" }}>Scar volume</span><b>{report.scar_volume_mm3.mean} ± {report.scar_volume_mm3.std} mm³</b></div>
          <div className="flex justify-between text-xs"><span style={{ color: "var(--c-text-dim)" }}>Volume CV</span><b>{report.scar_volume_mm3.cv_percent}%</b></div>
          {report.mean_pairwise_scar_dice != null && (
            <div className="flex justify-between text-xs"><span style={{ color: "var(--c-text-dim)" }}>Scar Dice</span><span>{report.mean_pairwise_scar_dice}</span></div>
          )}
        </div>
      )}
    </div>
  );
}
