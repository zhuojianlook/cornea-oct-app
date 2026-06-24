import { Button, CircularProgress, MenuItem, Select } from "@mui/material";
import { useWorkflowStore } from "../../store/workflowStore";
import { useCaseStore } from "../../store/caseStore";
import { LIFECYCLE_STEPS, scanStep, type LifecycleStep } from "../../api/lifecycle";

/* Per-scan lifecycle TIMELINE — replaces the old stage Toolbar + StageStepper. It shows the active
   scan's progress through 7 colour-coded steps and surfaces ONLY the next action(s) for the current
   step. Workflow order (reordered): Raw → Preprocessed[auto] → vetted → classified (scar/control) →
   SAM2[auto] → SAM2[corrected] → scheduled for training. SAM2 is gated until the scan is classified. */
export function TimelineBar() {
  const segBusy = useWorkflowStore((s) => s.segBusy);
  const scarBusy = useWorkflowStore((s) => s.scarBusy);
  const correcting = useWorkflowStore((s) => s.correcting);
  const segLoaded = useWorkflowStore((s) => s.segLoaded);
  const sensitivity = useWorkflowStore((s) => s.scarSensitivity);
  const scarMethod = useWorkflowStore((s) => s.scarMethod);
  const runSam2 = useWorkflowStore((s) => s.runSam2);
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
  const scheduleTraining = useCaseStore((s) => s.scheduleTraining);
  const scheduled = Boolean(manifest?.training_scheduled);

  const busy = segBusy || scarBusy;
  const step: LifecycleStep = scanStep(manifest);

  // ── the step strip ──────────────────────────────────────────────────────────
  const strip = (
    <div className="flex items-center gap-1">
      {([1, 2, 3, 4, 5, 6, 7] as LifecycleStep[]).map((i) => {
        const reached = step >= i;
        const current = step === i;
        const meta = LIFECYCLE_STEPS[i];
        return (
          <div key={i} className="flex items-center gap-1" title={meta.label}>
            <span style={{
              display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, lineHeight: 1,
              padding: "3px 7px", borderRadius: 11, whiteSpace: "nowrap",
              background: reached ? meta.color : "var(--c-surface2)",
              color: reached ? "#08121f" : "var(--c-text-dim)",
              fontWeight: current ? 700 : 500,
              outline: current ? "2px solid #fff" : "none", outlineOffset: -1,
              opacity: reached ? 1 : 0.7,
            }}>
              <b style={{ opacity: 0.7 }}>{i}</b>{meta.short}
            </span>
            {i < 7 && <span style={{ color: "var(--c-text-dim)", fontSize: 10 }}>›</span>}
          </div>
        );
      })}
    </div>
  );

  // ── correction sub-controls (shared by steps 5/6) ──
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

  // ── actions for the CURRENT step ──
  let actions: React.ReactNode = null;
  if (step <= 1) {
    actions = <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Preprocess this scan in the sidebar ← to begin.</span>;
  } else if (step === 2) {
    // auto-preprocessed (red) → review + Approve
    actions = (
      <>
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Review the preprocessing (Before/after · Fix-columns), then:</span>
        <Button size="small" variant="contained" color="warning" disabled={busy} onClick={() => vetPreprocessing()}
          title="Mark the preprocessing as manually vetted (turns the scan orange) — unlocks classification.">
          ✓ Approve preprocessing
        </Button>
      </>
    );
  } else if (step === 3) {
    // vetted (orange) → classify scar/control (replicates/controls grouped in the sidebar)
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
    // classified (yellow) → run SAM2 (now enabled)
    actions = (
      <>
        <Button size="small" variant="contained" color="primary" disabled={busy} onClick={() => runSam2()}
          title="Run SAM2 cornea segmentation (axial+coronal+sagittal → 2-of-3 consensus).">
          ▶ Run SAM2
        </Button>
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>(classified as {classification})</span>
      </>
    );
  } else if (step === 5 || step === 6) {
    // SAM2 done → scar detection (auto), manual correction, then schedule
    actions = (
      <>
        {classification !== "control" && (
          <>
            <Select size="small" value={scarMethod} onChange={(e) => set("scarMethod", e.target.value)}
              disabled={busy} sx={{ fontSize: 12, maxWidth: 180, color: "var(--c-text)", ".MuiSelect-select": { py: 0.4 }, "& fieldset": { borderColor: "var(--c-border)" } }}
              title="Scar detection strategy">
              <MenuItem value="hysteresis" sx={{ fontSize: 12 }}>Hysteresis (best reproducibility)</MenuItem>
              <MenuItem value="depthnorm" sx={{ fontSize: 12 }}>Depth-normalised (uses controls)</MenuItem>
              <MenuItem value="normal_anchor" sx={{ fontSize: 12 }}>Normal-stroma anchor</MenuItem>
              <MenuItem value="robust_mad" sx={{ fontSize: 12 }}>Robust MAD</MenuItem>
              <MenuItem value="morph_lcc" sx={{ fontSize: 12 }}>Morph + largest component</MenuItem>
              <MenuItem value="brightness" sx={{ fontSize: 12 }}>Brightness percentile</MenuItem>
            </Select>
            <Button size="small" variant="contained" color="error" disabled={busy || !segLoaded} onClick={() => runScarAuto()}
              title="Detect scar inside the cornea with the selected method.">Detect scar</Button>
            <Button size="small" variant="outlined" color="error" disabled={busy || !segLoaded} onClick={() => runScarAutoSam2()}
              title="Auto scar via SAM2 3-view consensus.">Scar (SAM2)</Button>
            <label className="flex items-center gap-1 text-xs" style={{ color: "var(--c-text-dim)" }} title="How much hyper-reflectivity to flag">
              sens<input type="range" min={1} max={40} value={sensitivity} style={{ width: 64 }} onChange={(e) => set("scarSensitivity", Number(e.target.value))} />
            </label>
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
            <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />
          </>
        )}
        {Correct}
        <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />
        <Button size="small" variant={scheduled ? "outlined" : "contained"} color="success" disabled={busy || correcting}
          onClick={() => scheduleTraining(!scheduled)}
          title="Mark this scan ready for nnU-Net training (turns it green).">
          {scheduled ? "Scheduled ✓ (unschedule)" : "Schedule for training"}
        </Button>
        <Button size="small" variant="outlined" color="success" disabled={busy} onClick={() => exportScarSummary()}
          title="Recompute scar volume/area/density for every case → scar_summary.csv">Export metrics</Button>
      </>
    );
  } else {
    // step 7 — scheduled (green)
    actions = (
      <>
        <span className="text-xs" style={{ color: "#22c55e" }}>✓ Scheduled for training.</span>
        <Button size="small" variant="outlined" disabled={busy} onClick={() => scheduleTraining(false)}>Unschedule</Button>
        {Correct}
        <Button size="small" variant="outlined" color="success" disabled={busy} onClick={() => exportScarSummary()}>Export metrics</Button>
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
    </div>
  );
}
