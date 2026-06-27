import { useState } from "react";
import { Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, MenuItem, Select } from "@mui/material";
import { useWorkflowStore } from "../../store/workflowStore";
import { useCaseStore } from "../../store/caseStore";
import { LIFECYCLE_STEPS, scanStep, type LifecycleStep } from "../../api/lifecycle";

/* Per-scan lifecycle TIMELINE — the active scan's progress through 8 colour-coded steps, surfacing ONLY
   the next action(s). Order: Raw → Preprocessed[auto] → Vetted → Classified(scar/control) → SAM2(cornea
   +scar, one-go) → Aligned(replicates) → Corrected → Scheduled. SAM2 is gated until classified, and for
   a scar-labelled scan "Run SAM2" also runs scar in one go. Click any REACHED earlier step to roll back
   to it (clears the later steps). */
export function TimelineBar() {
  const segBusy = useWorkflowStore((s) => s.segBusy);
  const scarBusy = useWorkflowStore((s) => s.scarBusy);
  const correcting = useWorkflowStore((s) => s.correcting);
  const segLoaded = useWorkflowStore((s) => s.segLoaded);
  const sensitivity = useWorkflowStore((s) => s.scarSensitivity);
  const scarMethod = useWorkflowStore((s) => s.scarMethod);
  const runSam2 = useWorkflowStore((s) => s.runSam2);
  const buildEyeConsensus = useWorkflowStore((s) => s.buildEyeConsensus);
  const loadCorrectionLayer = useWorkflowStore((s) => s.loadCorrectionLayer);
  const saveCorrection = useWorkflowStore((s) => s.saveCorrection);
  const cancelCorrection = useWorkflowStore((s) => s.cancelCorrection);
  const undoCorrection = useWorkflowStore((s) => s.undoCorrection);
  const runScarAuto = useWorkflowStore((s) => s.runScarAuto);
  const runScarAutoSam2 = useWorkflowStore((s) => s.runScarAutoSam2);
  const exportScarSummary = useWorkflowStore((s) => s.exportScarSummary);
  const set = useWorkflowStore((s) => s.set);
  const hintMode = useWorkflowStore((s) => s.hintMode);
  const hintPositive = useWorkflowStore((s) => s.hintPositive);
  const hintCount = useWorkflowStore((s) => s.scarHints?.length ?? 0);
  const applyScarHints = useWorkflowStore((s) => s.applyScarHints);
  const clearScarHints = useWorkflowStore((s) => s.clearScarHints);

  const caseInfo = useCaseStore((s) => s.caseInfo);
  const manifest = (caseInfo?.manifest ?? null) as Record<string, unknown> | null;
  const classification = (manifest?.scar_classification as "scar" | "control" | null | undefined) ?? null;
  const setClassification = useCaseStore((s) => s.setClassification);
  const vetPreprocessing = useCaseStore((s) => s.vetPreprocessing);
  const approveRaw = useCaseStore((s) => s.approveRaw);
  const scheduleTraining = useCaseStore((s) => s.scheduleTraining);
  const resetStep = useCaseStore((s) => s.resetStep);
  const scheduled = Boolean(manifest?.training_scheduled);
  const isConsensus = Boolean(manifest?.consensus_cases);

  const busy = segBusy || scarBusy;
  const step: LifecycleStep = scanStep(manifest);
  const maxStep = LIFECYCLE_STEPS.length - 1;   // 8

  // #9 step regression: which step the user clicked to roll back to (confirm modal).
  const [resetTo, setResetTo] = useState<number | null>(null);
  const downstream = resetTo != null ? LIFECYCLE_STEPS.slice(resetTo + 1, step + 1).map((x) => x.short) : [];

  // ── the step strip (click a reached EARLIER step to roll back to it) ──────────
  const strip = (
    <div className="flex items-center gap-1">
      {([1, 2, 3, 4, 5, 6, 7, 8] as LifecycleStep[]).map((i) => {
        const reached = step >= i;
        const current = step === i;
        // A built consensus case can't be step-reset (its consensus_cases define it — rebuild instead).
        const canReset = reached && i < step && !busy && !correcting && !!caseInfo && !isConsensus;
        const meta = LIFECYCLE_STEPS[i];
        return (
          <div key={i} className="flex items-center gap-1"
            title={canReset ? `Roll back to “${meta.short}” — clears the later steps` : meta.label}>
            <span onClick={() => canReset && setResetTo(i)} style={{
              display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, lineHeight: 1,
              padding: "3px 7px", borderRadius: 11, whiteSpace: "nowrap",
              background: reached ? meta.color : "var(--c-surface2)",
              color: reached ? "#08121f" : "var(--c-text-dim)",
              fontWeight: current ? 700 : 500,
              outline: current ? "2px solid #fff" : "none", outlineOffset: -1,
              opacity: reached ? 1 : 0.7,
              cursor: canReset ? "pointer" : "default",
            }}>
              <b style={{ opacity: 0.7 }}>{i}</b>{meta.short}
            </span>
            {i < maxStep && <span style={{ color: "var(--c-text-dim)", fontSize: 10 }}>›</span>}
          </div>
        );
      })}
    </div>
  );

  // ── reusable action sub-controls ──
  const Correct = !correcting ? (
    <Button size="small" variant="outlined" disabled={busy || !segLoaded} onClick={() => loadCorrectionLayer()}
      title="Edit the labelmap with the pen (cornea/scar/erase), then Save">Correct ✎</Button>
  ) : (
    <>
      <Button size="small" variant="contained" color="secondary" disabled={busy} onClick={() => saveCorrection()}>Save correction</Button>
      <Button size="small" variant="outlined" disabled={busy} onClick={() => undoCorrection()} title="Undo last edit">↶</Button>
      <Button size="small" variant="outlined" color="inherit" disabled={busy} onClick={() => cancelCorrection()}>Cancel</Button>
    </>
  );

  const ScheduleBtn = (
    <Button size="small" variant={scheduled ? "outlined" : "contained"} color="success" disabled={busy || correcting}
      onClick={() => scheduleTraining(!scheduled)} title="Mark this scan ready for nnU-Net training (turns it green).">
      {scheduled ? "Scheduled ✓ (unschedule)" : "Schedule for training"}
    </Button>
  );
  const ExportBtn = (
    <Button size="small" variant="outlined" color="success" disabled={busy} onClick={() => exportScarSummary()}
      title="Recompute scar volume/area/density for every case → scar_summary.csv">Export metrics</Button>
  );
  const AlignBtn = (
    <Button size="small" variant="contained" color="info" disabled={busy || correcting} onClick={() => buildEyeConsensus()}
      title="Register + average this eye's repeat scans into one consensus, control-normalised by the tagged control (no-scar) scans. Run SAM2 on the eye's repeats first.">
      ⌖ Align replicates + normalize
    </Button>
  );

  // Scar method + sensitivity (sets what the one-go Run-SAM2 uses, and any re-run). Only meaningful for
  // a scar-labelled scan (a control runs cornea only).
  const ScarMethod = (
    <>
      <Select size="small" value={scarMethod} onChange={(e) => set("scarMethod", e.target.value)}
        disabled={busy} sx={{ fontSize: 12, maxWidth: 190, color: "var(--c-text)", ".MuiSelect-select": { py: 0.4 }, "& fieldset": { borderColor: "var(--c-border)" } }}
        title="Scar detection strategy (used by Run SAM2 and any re-run)">
        <MenuItem value="hysteresis" sx={{ fontSize: 12 }}>Hysteresis (best reproducibility)</MenuItem>
        <MenuItem value="depthnorm" sx={{ fontSize: 12 }}>Depth-normalised (uses controls)</MenuItem>
        <MenuItem value="normal_anchor" sx={{ fontSize: 12 }}>Normal-stroma anchor</MenuItem>
        <MenuItem value="robust_mad" sx={{ fontSize: 12 }}>Robust MAD</MenuItem>
        <MenuItem value="morph_lcc" sx={{ fontSize: 12 }}>Morph + largest component</MenuItem>
        <MenuItem value="brightness" sx={{ fontSize: 12 }}>Brightness percentile</MenuItem>
      </Select>
      <label className="flex items-center gap-1 text-xs" style={{ color: "var(--c-text-dim)" }} title="How much hyper-reflectivity to flag">
        sens<input type="range" min={1} max={40} value={sensitivity} style={{ width: 64 }} onChange={(e) => set("scarSensitivity", Number(e.target.value))} />
      </label>
    </>
  );

  // Re-run scar after SAM2 (iteration): detect/SAM2-scar + click-hints. Non-control only.
  const ScarReRun = classification !== "control" ? (
    <>
      {ScarMethod}
      <Button size="small" variant="outlined" color="error" disabled={busy || !segLoaded} onClick={() => runScarAuto()}
        title="Re-detect scar inside the cornea with the selected method.">Re-run scar</Button>
      <Button size="small" variant="outlined" color="error" disabled={busy || !segLoaded} onClick={() => runScarAutoSam2()}
        title="Re-run scar via SAM2 3-view consensus.">Scar (SAM2)</Button>
      <Button size="small" variant={hintMode ? "contained" : "outlined"} color="warning" disabled={busy || !segLoaded}
        onClick={() => set("hintMode", !hintMode)} title="Click scar areas on the slices to guide SAM2">
        {hintMode ? "Hinting…" : "Hint"}
      </Button>
      {hintMode && (
        <>
          <Button size="small" variant={hintPositive ? "contained" : "outlined"} color="error" onClick={() => set("hintPositive", true)}>scar</Button>
          <Button size="small" variant={!hintPositive ? "contained" : "outlined"} onClick={() => set("hintPositive", false)}>not</Button>
          <Button size="small" variant="contained" color="warning" disabled={busy || hintCount === 0} onClick={() => applyScarHints()}>Apply ({hintCount})</Button>
          <Button size="small" variant="outlined" disabled={busy || hintCount === 0} onClick={() => clearScarHints()}>Clear</Button>
        </>
      )}
    </>
  ) : null;

  const sep = <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />;

  // ── actions for the CURRENT step ──
  let actions: React.ReactNode = null;
  if (step <= 1) {
    actions = <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Preprocess this scan in the sidebar ← to begin.</span>;
  } else if (step === 2) {
    actions = (
      <>
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Review the preprocessing (Before/after · Fix-columns), then:</span>
        <Button size="small" variant="contained" color="warning" disabled={busy} onClick={() => vetPreprocessing()}
          title="Mark the preprocessing as manually vetted (turns the scan orange) — unlocks classification.">
          ✓ Approve preprocessing
        </Button>
        <Button size="small" variant="outlined" color="warning" disabled={busy} onClick={() => approveRaw()}
          title="Use the ORIGINAL (raw) scan as the working volume instead of the correction. Drops any segmentation; also marks it vetted.">
          ↩ Use original (raw)
        </Button>
      </>
    );
  } else if (step === 3) {
    actions = (
      <div className="flex items-center gap-1 text-xs" style={{ color: "var(--c-text-dim)" }}
        title="Does this corrected volume have a scar? 'No scar' marks it a control (normal baseline). Replicates/controls are grouped in the sidebar.">
        Classify:
        <Button size="small" variant={classification === "scar" ? "contained" : "outlined"} color="error"
          disabled={busy} onClick={() => setClassification(classification === "scar" ? null : "scar")}>Scar</Button>
        <Button size="small" variant={classification === "control" ? "contained" : "outlined"} color="inherit"
          disabled={busy} onClick={() => setClassification(classification === "control" ? null : "control")}>No scar (control)</Button>
      </div>
    );
  } else if (step === 4) {
    // classified (yellow) → Run SAM2 = one-go (cornea, + scar for a scar-labelled scan with the chosen method)
    actions = (
      <>
        {classification === "scar" && ScarMethod}
        <Button size="small" variant="contained" color="primary" disabled={busy} onClick={() => runSam2()}
          title={classification === "scar"
            ? "Run SAM2 cornea segmentation, then detect scar (cornea vs scar) with the chosen method — in one go."
            : "Run SAM2 cornea segmentation (control: no scar)."}>
          ▶ Run SAM2{classification === "scar" ? " + scar" : ""}
        </Button>
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>(classified as {classification})</span>
      </>
    );
  } else if (step === 5) {
    // SAM2 (cornea+scar) done → align replicates (primary next step), or correct / schedule; re-run scar to iterate
    actions = (
      <>
        {AlignBtn}{sep}{Correct}
        {ScarReRun && <>{sep}{ScarReRun}</>}
        {sep}{ScheduleBtn}{ExportBtn}
      </>
    );
  } else if (step === 6) {
    // aligned (teal) → correct the consensus / schedule
    actions = <>{Correct}{sep}{ScheduleBtn}{ExportBtn}</>;
  } else if (step === 7) {
    // manually corrected (dark blue)
    actions = <>{ScheduleBtn}{sep}{Correct}{ExportBtn}</>;
  } else {
    // step 8 — scheduled (green)
    actions = (
      <>
        <span className="text-xs" style={{ color: "#22c55e" }}>✓ Scheduled for training.</span>
        <Button size="small" variant="outlined" disabled={busy} onClick={() => scheduleTraining(false)}>Unschedule</Button>
        {Correct}{ExportBtn}
      </>
    );
  }

  return (
    <div className="flex items-center gap-3 px-3 border-b overflow-x-auto [&>*]:shrink-0"
      style={{ minHeight: 46, backgroundColor: "var(--c-surface)", borderColor: "var(--c-border)" }}>
      {strip}
      <span style={{ width: 1, height: 26, background: "var(--c-border)" }} />
      <div className="flex items-center gap-2 [&>*]:shrink-0">{caseInfo ? actions : <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Open or preprocess a scan to begin.</span>}</div>
      <div className="flex-1" />
      {busy && <CircularProgress size={16} />}

      <Dialog open={resetTo != null} onClose={() => setResetTo(null)}>
        <DialogTitle sx={{ fontSize: 16 }}>
          Roll back to “{resetTo != null ? LIFECYCLE_STEPS[resetTo].short : ""}”?
        </DialogTitle>
        <DialogContent sx={{ fontSize: 13 }}>
          This resets the later steps so you can redo them: <b>{downstream.join(" · ") || "(none)"}</b>.
          <br />
          The scan's files are kept on disk — re-running a step overwrites its result. You can re-advance afterwards.
        </DialogContent>
        <DialogActions>
          <Button size="small" onClick={() => setResetTo(null)}>Cancel</Button>
          <Button size="small" variant="contained" color="warning"
            onClick={() => { const s = resetTo; setResetTo(null); if (s != null) resetStep(s); }}>
            Reset to step {resetTo}
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}
