import { useEffect, useState } from "react";
import { Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, MenuItem, Select } from "@mui/material";
import { useWorkflowStore } from "../../store/workflowStore";
import { useCaseStore } from "../../store/caseStore";
import { LIFECYCLE_STEPS, scanStep, stepReached, type LifecycleStep } from "../../api/lifecycle";

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
  const sam2RunningCaseId = useWorkflowStore((s) => s.sam2RunningCaseId);
  const selectedStep = useWorkflowStore((s) => s.selectedStep);
  const selectStep = useWorkflowStore((s) => s.selectStep);
  const alignReplicates = useWorkflowStore((s) => s.alignReplicates);
  const normalizeConsensus = useWorkflowStore((s) => s.normalizeConsensus);
  const compareStrategies = useWorkflowStore((s) => s.compareStrategies);
  const strategyComparison = useWorkflowStore((s) => s.strategyComparison);
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
  const status = useWorkflowStore((s) => s.status);

  const caseInfo = useCaseStore((s) => s.caseInfo);
  const manifest = (caseInfo?.manifest ?? null) as Record<string, unknown> | null;
  const classification = (manifest?.scar_classification as "scar" | "control" | null | undefined) ?? null;
  const setClassification = useCaseStore((s) => s.setClassification);
  const vetPreprocessing = useCaseStore((s) => s.vetPreprocessing);
  const approveRaw = useCaseStore((s) => s.approveRaw);
  const scheduleTraining = useCaseStore((s) => s.scheduleTraining);
  const resetStep = useCaseStore((s) => s.resetStep);
  const confirmSubgroup = useCaseStore((s) => s.confirmSubgroup);
  const skipScar = useCaseStore((s) => s.skipScar);
  const scheduled = Boolean(manifest?.training_scheduled);
  const isConsensus = Boolean(manifest?.consensus_cases);
  const subgroup = String(manifest?.scar_subgroup ?? "1") || "1";

  const busy = segBusy || scarBusy;
  const step: LifecycleStep = scanStep(manifest);
  const maxStep = LIFECYCLE_STEPS.length - 1;   // 10
  // Which step is being VIEWED, and whether that's an inspect (earlier, read-only) vs the live step.
  const viewStep = (selectedStep ?? step) as LifecycleStep;
  const inspecting = selectedStep != null && selectedStep < step;

  // #9 step regression: which step the user chose to roll back to (confirm modal); set by the inspect-mode
  // "Roll back to this step" button (NOT by merely clicking a step — clicking just inspects it).
  const [resetTo, setResetTo] = useState<number | null>(null);
  // Step-6 subgroup input, re-seeded from the manifest on case/subgroup change.
  const [subInput, setSubInput] = useState(subgroup);
  useEffect(() => { setSubInput(subgroup); }, [caseInfo?.case_id, subgroup]);
  // Strategy-comparison results dialog (publication).
  const [showCompare, setShowCompare] = useState(false);
  const CMP_COLS: { key: keyof NonNullable<typeof strategyComparison>["rows"][number]; label: string }[] = [
    { key: "strategy", label: "Strategy" },
    { key: "mean_pairwise_dice", label: "Pairwise Dice ↑" },
    { key: "mean_pairwise_hd95_mm", label: "HD95 mm ↓" },
    { key: "cv_percent", label: "Volume CV% ↓" },
    { key: "rc_mm3", label: "RC mm³ ↓" },
    { key: "mean_volume_mm3", label: "Mean vol mm³" },
  ];
  const downloadComparisonCsv = () => {
    if (!strategyComparison) return;
    const head = CMP_COLS.map((c) => c.label).join(",");
    const body = strategyComparison.rows.map((r) =>
      CMP_COLS.map((c) => { const v = (r as unknown as Record<string, unknown>)[c.key as string]; return v == null ? "" : String(v); }).join(",")).join("\n");
    const csv = `# scar strategy reproducibility · n=${strategyComparison.n} replicates · phi=${strategyComparison.phi_percentile} · subgroup=${strategyComparison.subgroup ?? ""}\n${head}\n${body}\n`;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = `scar_strategy_reproducibility_${strategyComparison.reference ?? "eye"}.csv`;
    a.click(); URL.revokeObjectURL(a.href);
  };
  const downstream = resetTo != null ? LIFECYCLE_STEPS.slice(resetTo + 1, step + 1).map((x) => x.short) : [];

  // ── the step strip: click a REACHED step to VIEW it (earlier = inspect read-only; current = back to live) ──
  const strip = (
    <div className="flex items-center gap-1">
      {([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] as LifecycleStep[]).map((i) => {
        const reached = stepReached(manifest, i);   // per-flag, so a SKIPPED step doesn't falsely colour
        const current = step === i;
        const viewing = viewStep === i;
        // Any reached step is clickable to inspect; NOT while a correction is in progress (switching the
        // action bar to inspect would strip Save/Undo/Cancel and leave the niivue pen live); a consensus
        // case isn't step-navigable.
        const canView = reached && !!caseInfo && !isConsensus && !correcting;
        const meta = LIFECYCLE_STEPS[i];
        return (
          <div key={i} className="flex items-center gap-1"
            title={canView ? (current ? `“${meta.short}” (current step)` : `Inspect “${meta.short}” (read-only; roll back to edit)`) : meta.label}>
            <span onClick={() => canView && selectStep(i === step ? null : i)} style={{
              display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, lineHeight: 1,
              padding: "3px 7px", borderRadius: 11, whiteSpace: "nowrap",
              background: reached ? meta.color : "var(--c-surface2)",
              color: reached ? "#08121f" : "var(--c-text-dim)",
              fontWeight: current ? 700 : 500,
              // solid white outline = the LIVE current step; dashed = the step you're inspecting.
              outline: current ? "2px solid #fff" : (viewing ? "2px dashed #fff" : "none"), outlineOffset: -1,
              opacity: reached ? 1 : 0.7,
              cursor: canView ? "pointer" : "default",
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
    <Button size="small" variant="contained" color="info" disabled={busy || correcting} onClick={() => alignReplicates()}
      title="Register + vote this eye's repeat scans (same subgroup) into one consensus, using the scar as-is. Normalization against controls is the next step.">
      ⌖ Align replicates
    </Button>
  );
  const NormalizeBtn = (
    <Button size="small" variant="contained" color="info" disabled={busy || correcting} onClick={() => normalizeConsensus()}
      title="Re-derive scar as excess over the control (no-scar) baseline and rebuild the consensus. Needs tagged + segmented control scans.">
      ◎ Normalize against controls
    </Button>
  );
  // STEP 6: confirm this scan's subgroup (which lesion set it belongs to → which repeats align together).
  const SubgroupConfirm = (
    <span className="flex items-center gap-1 text-xs" style={{ color: "var(--c-text-dim)" }}>
      subgroup
      <input value={subInput} disabled={busy} placeholder="1" onChange={(e) => setSubInput(e.target.value)}
        style={{ fontSize: 11, width: 90, color: "var(--c-text)", background: "var(--c-surface2)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "1px 5px" }} />
      <Button size="small" variant="contained" color="secondary" disabled={busy} onClick={() => confirmSubgroup(subInput)}
        title="Confirm which scar subgroup (lesion set) this scan belongs to, so the right repeats align together.">
        ✓ Confirm subgroup
      </Button>
    </span>
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
        title="CLASSICAL detector: threshold the scar inside the cornea using the selected method (e.g. hysteresis) + sensitivity. Fast (seconds). Re-run after changing the method/sensitivity.">Detect scar (threshold)</Button>
      <Button size="small" variant="outlined" color="error" disabled={busy || !segLoaded} onClick={() => runScarAutoSam2()}
        title="SAM2 (deep-learning) scar: run SAM2 on cornea-vs-scar across axial/coronal/sagittal and take the 2-of-3 vote. Slower (~1–2 min); an alternative to the threshold detector when it struggles.">Scar via SAM2</Button>
      <Button size="small" variant={hintMode ? "contained" : "outlined"} color="warning" disabled={busy || !segLoaded}
        onClick={() => set("hintMode", !hintMode)}
        title="Optional touch-up: click ON a scar region (then 'scar') or on non-scar tissue (then 'not') in the slices to give SAM2 point prompts, then Apply to re-segment the scar from your clicks.">
        {hintMode ? "Guiding… (click slices)" : "Guide scar (click)"}
      </Button>
      {hintMode && (
        <>
          <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>click marks:</span>
          <Button size="small" variant={hintPositive ? "contained" : "outlined"} color="error" onClick={() => set("hintPositive", true)} title="Clicks add scar">scar</Button>
          <Button size="small" variant={!hintPositive ? "contained" : "outlined"} onClick={() => set("hintPositive", false)} title="Clicks remove scar">not scar</Button>
          <Button size="small" variant="contained" color="warning" disabled={busy || hintCount === 0} onClick={() => applyScarHints()}>Apply ({hintCount})</Button>
          <Button size="small" variant="outlined" disabled={busy || hintCount === 0} onClick={() => clearScarHints()}>Clear</Button>
        </>
      )}
    </>
  ) : null;

  const sep = <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />;

  // ── actions ──
  let actions: React.ReactNode = null;
  if (inspecting) {
    // Viewing an earlier completed step: its own tools show in the viewer (read-only). Consequential
    // edits are disabled until the user explicitly rolls back to it (which clears the later steps).
    const vm = LIFECYCLE_STEPS[viewStep];
    actions = (
      <>
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
          Inspecting <b style={{ color: vm.color }}>{vm.short}</b> (read-only) ·
        </span>
        <Button size="small" variant="contained" color="warning" disabled={busy || correcting} onClick={() => setResetTo(viewStep)}
          title={`Roll back to “${vm.short}” to edit it — this clears the later steps.`}>
          ↩ Roll back to this step to edit
        </Button>
        <Button size="small" variant="text" disabled={busy} onClick={() => selectStep(null)} title="Return to the current step">
          ✕ Back to current ({LIFECYCLE_STEPS[step].short})
        </Button>
      </>
    );
  } else if (step <= 1) {
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
    // classified (yellow) → segment the CORNEA only (SAM2). Scar is the next, separate step.
    actions = (
      <>
        <Button size="small" variant="contained" color="primary" disabled={busy || !!sam2RunningCaseId} onClick={() => runSam2()}
          title="Run SAM2 cornea-vs-background segmentation. Scar is segmented in the next step.">
          ▶ Run SAM2 (cornea)
        </Button>
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>(classified as {classification})</span>
      </>
    );
  } else if (step === 5) {
    // cornea segmented → SCAR step. Scar scan: detect / compare strategies. Control: no scar → continue.
    actions = classification === "control" ? (
      <>
        <Button size="small" variant="contained" color="secondary" disabled={busy} onClick={() => skipScar()}
          title="This scan is a control (no scar) — mark the scar step done and continue to subgroup assignment.">
          ✓ No scar (control) — continue
        </Button>
        {sep}{Correct}
      </>
    ) : (
      <>
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Segment scar:</span>
        {ScarReRun}{sep}{Correct}
      </>
    );
  } else if (step === 6) {
    // scar segmented (rose) → assign this scan's SUBGROUP (gates align); re-run scar / correct to iterate.
    actions = (
      <>
        {SubgroupConfirm}{sep}{Correct}
        {ScarReRun && <>{sep}{ScarReRun}</>}
      </>
    );
  } else if (step === 7) {
    // subgroup assigned (purple) → align this subgroup's replicates
    actions = (
      <>
        {AlignBtn}
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>(subgroup “{subgroup}”)</span>
        {sep}{SubgroupConfirm}{sep}{Correct}
      </>
    );
  } else if (step === 8) {
    // aligned (teal) → normalize against controls (next), or correct / schedule
    actions = <>{NormalizeBtn}{sep}{Correct}{sep}{ScheduleBtn}{ExportBtn}</>;
  } else if (step === 9) {
    // normalized (cyan) → correct / schedule
    actions = <>{Correct}{sep}{ScheduleBtn}{ExportBtn}</>;
  } else if (step === 10) {
    // manually corrected (dark blue)
    actions = <>{ScheduleBtn}{sep}{Correct}{ExportBtn}</>;
  } else {
    // step 11 — scheduled (green)
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
      {/* PUBLICATION: compare scar-detection strategies' reproducibility on this eye's replicates. Available
          once a scan/consensus is segmented (needs ≥2 segmented replicates of the eye+subgroup). */}
      {caseInfo && (step >= 5 || isConsensus) && (
        <Button size="small" variant="outlined" color="info" disabled={busy || correcting}
          onClick={() => { setShowCompare(true); compareStrategies(); }}
          title="Run every scar strategy on this eye's replicates and tabulate test–retest reproducibility (pairwise Dice, HD95, volume CV%) — for strategy comparison in the paper. Read-only; doesn't change the scan.">
          ⚖ Compare strategies
        </Button>
      )}
      {/* Live progress text (SAM2 per-plane %, scar phase, …) next to the spinner — not just an icon. */}
      {(busy || !!sam2RunningCaseId) && status.kind === "working" && (
        <span className="text-xs" style={{ color: "var(--c-text-dim)", maxWidth: 360, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
          title={status.detail}>{status.detail}</span>
      )}
      {(busy || !!sam2RunningCaseId) && <CircularProgress size={16} />}

      {/* PUBLICATION: scar-strategy reproducibility table. */}
      <Dialog open={showCompare} onClose={() => setShowCompare(false)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ fontSize: 16 }}>Scar strategy reproducibility (test–retest)</DialogTitle>
        <DialogContent sx={{ fontSize: 13 }}>
          {!strategyComparison && scarBusy && (
            <div className="flex items-center gap-2 py-4"><CircularProgress size={18} /> Running each strategy on the eye's replicates…</div>
          )}
          {strategyComparison && (
            <>
              <div className="text-xs mb-2" style={{ color: "var(--c-text-dim)" }}>
                {strategyComparison.n} replicates{strategyComparison.subgroup ? ` · subgroup “${strategyComparison.subgroup}”` : ""} · φ={strategyComparison.phi_percentile} ·
                reproducibility only (no manual GT). Higher Dice / lower HD95·CV·RC = more reproducible; read Dice alongside volume (Dice rises with mask size).
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr>{CMP_COLS.map((c) => (
                    <th key={c.key as string} style={{ textAlign: c.key === "strategy" ? "left" : "right", padding: "4px 8px", borderBottom: "1px solid var(--c-border)", color: "var(--c-text-dim)" }}>{c.label}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {strategyComparison.rows.map((r) => (
                    <tr key={r.strategy}>
                      {CMP_COLS.map((c) => {
                        const v = (r as unknown as Record<string, unknown>)[c.key as string];
                        return <td key={c.key as string} style={{ textAlign: c.key === "strategy" ? "left" : "right", padding: "4px 8px", borderBottom: "1px solid var(--c-border)", fontWeight: c.key === "strategy" ? 600 : 400 }}>
                          {r.error && c.key === "strategy" ? `${r.strategy} (error)` : (v == null ? "—" : String(v))}
                        </td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button size="small" disabled={!strategyComparison} onClick={downloadComparisonCsv}>⤓ Download CSV</Button>
          <Button size="small" variant="contained" onClick={() => setShowCompare(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      <Dialog open={resetTo != null} onClose={() => setResetTo(null)}>
        <DialogTitle sx={{ fontSize: 16 }}>
          Roll back to “{resetTo != null ? LIFECYCLE_STEPS[resetTo].short : ""}”?
        </DialogTitle>
        <DialogContent sx={{ fontSize: 13 }}>
          This resets the later steps so you can redo them: <b>{downstream.join(" · ") || "(none)"}</b>.
          <br />
          The segmentation for those steps is dropped (re-running re-creates it); the preprocessed volume is kept.
        </DialogContent>
        <DialogActions>
          <Button size="small" onClick={() => setResetTo(null)}>Cancel</Button>
          <Button size="small" variant="contained" color="warning"
            onClick={() => { const s = resetTo; setResetTo(null); selectStep(null); if (s != null) resetStep(s); }}>
            Reset to step {resetTo}
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}
