import { useEffect, useState } from "react";
import { Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, MenuItem, Select } from "@mui/material";
import { useWorkflowStore } from "../../store/workflowStore";
import { useCaseStore } from "../../store/caseStore";
import { LIFECYCLE_STEPS, scanStep, stepReached, stepApplicable, octProposals, type LifecycleStep } from "../../api/lifecycle";

/* Per-scan lifecycle TIMELINE — the active scan's progress through the colour-coded steps, surfacing ONLY
   the next action(s). Order: Raw → Preprocessed[auto] → Vetted → SAM2(cornea) → Cornea✓ → Classified(scar/
   control) → Subgroup → Scar → Aligned → Normalized → Corrected → Scheduled. Classification comes AFTER
   cornea-vetting (it gates only the scar branch, not SAM2), so a control can schedule straight after Cornea✓.
   Click any REACHED earlier step to roll back to it (clears the later steps). */
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
  const skipNormalization = useWorkflowStore((s) => s.skipNormalization);
  const applyConsensusScar = useWorkflowStore((s) => s.applyConsensusScar);
  const correctBusy = useWorkflowStore((s) => s.correctBusy);
  const consensusScarMode = useWorkflowStore((s) => s.consensusScarMode);
  const compareStrategies = useWorkflowStore((s) => s.compareStrategies);
  const cancelCompareStrategies = useWorkflowStore((s) => s.cancelCompareStrategies);
  const strategyComparison = useWorkflowStore((s) => s.strategyComparison);
  const autoSubgroups = useWorkflowStore((s) => s.autoSubgroups);
  const applySubgroups = useWorkflowStore((s) => s.applySubgroups);
  const subgroupProposal = useWorkflowStore((s) => s.subgroupProposal);
  const subgroupBusy = useWorkflowStore((s) => s.subgroupBusy);
  const loadCorrectionLayer = useWorkflowStore((s) => s.loadCorrectionLayer);
  const saveCorrection = useWorkflowStore((s) => s.saveCorrection);
  const startCorneaVetPaint = useWorkflowStore((s) => s.startCorneaVetPaint);
  const confirmCorneaVet = useWorkflowStore((s) => s.confirmCorneaVet);
  const corneaVetBusy = useWorkflowStore((s) => s.corneaVetBusy);
  const cancelCorrection = useWorkflowStore((s) => s.cancelCorrection);
  const runScarAuto = useWorkflowStore((s) => s.runScarAuto);
  const runScarAutoSam2 = useWorkflowStore((s) => s.runScarAutoSam2);
  const exportScarSummary = useWorkflowStore((s) => s.exportScarSummary);
  const scarSummaryInfo = useWorkflowStore((s) => s.scarSummaryInfo);
  const exportCorrectionMp4 = useWorkflowStore((s) => s.exportCorrectionMp4);
  const mp4Busy = useWorkflowStore((s) => s.mp4Busy);
  const correctionMp4Url = useWorkflowStore((s) => s.correctionMp4Url);
  const correctionMp4Info = useWorkflowStore((s) => s.correctionMp4Info);
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
  const setDifficult = useCaseStore((s) => s.setDifficult);
  const markDefectMode = useWorkflowStore((s) => s.markDefectMode);
  const approvePreprocessing = useCaseStore((s) => s.approvePreprocessing);
  const applyCorrections = useCaseStore((s) => s.applyCorrections);
  const approveRaw = useCaseStore((s) => s.approveRaw);
  const rerunPreprocess = useCaseStore((s) => s.rerunPreprocess);
  const caseBusy = useCaseStore((s) => s.busy);
  const scheduleTraining = useCaseStore((s) => s.scheduleTraining);
  const resetStep = useCaseStore((s) => s.resetStep);
  const confirmSubgroup = useCaseStore((s) => s.confirmSubgroup);
  const skipScar = useCaseStore((s) => s.skipScar);
  const scheduled = Boolean(manifest?.training_scheduled);
  const isConsensus = Boolean(manifest?.consensus_cases);
  // Crop-approval: an auto de-tilt/crop/surface-crop was DETECTED but not applied — the Approve action bakes
  // it in first (via approvePreprocessing), and the button relabels to make that clear.
  const proposals = octProposals(manifest);
  const subgroup = String(manifest?.scar_subgroup ?? "1") || "1";

  const busy = segBusy || scarBusy || caseBusy;
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
  // Auto subgroup-assignment dialog (bright-spot alignment + overlay → editable grouping → apply).
  const [showSubgroup, setShowSubgroup] = useState(false);
  // #14a: which scar op is running (so its button shows a spinner + live progress, not just the global one).
  const [scarKind, setScarKind] = useState<"threshold" | "sam2" | "hints" | null>(null);
  useEffect(() => { if (!scarBusy) setScarKind(null); }, [scarBusy]);
  // #spinner: which slow shared-busy action is running (align/normalize/skip-norm/use-raw/export), so its
  // OWN button shows a spinner — these all flip the shared segBusy/scarBusy/caseBusy, so we name the
  // specific one here and clear it when the work settles.
  const [busyAction, setBusyAction] = useState<string | null>(null);
  useEffect(() => { if (!busy) setBusyAction(null); }, [busy]);
  // Short live-progress label for the running scar button (e.g. SAM2 per-plane %). Falls back to a verb.
  const scarProgress = status.kind === "working" ? status.detail : "";
  const [editAssign, setEditAssign] = useState<Record<string, string>>({});
  useEffect(() => {
    if (subgroupProposal) {
      setEditAssign(Object.fromEntries(Object.entries(subgroupProposal.subgroups).map(([k, v]) => [k, String(v)])));
    }
  }, [subgroupProposal]);
  // Subgroup swatch colours — must mirror subgroup._SUBGROUP_RGB so the table legend matches the overlay.
  const SUBGROUP_RGB = ["#ff5050", "#5ac86e", "#5a96ff", "#ebc846", "#d26eeb", "#5adcdc"];
  const subColor = (label: string) => SUBGROUP_RGB[(Math.max(1, parseInt(label || "1", 10) || 1) - 1) % SUBGROUP_RGB.length];
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
      {([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as LifecycleStep[]).map((i) => {
        const applicable = stepApplicable(manifest, i);   // control: scar steps 7-11 are N/A
        const reached = applicable && stepReached(manifest, i);   // per-flag, so a SKIPPED step doesn't falsely colour
        const current = step === i;
        const viewing = viewStep === i;
        // Any reached step is clickable to inspect; NOT while a correction is in progress (switching the
        // action bar to inspect would strip Save/Undo/Cancel and leave the niivue pen live); a consensus
        // case isn't step-navigable.
        const canView = reached && !!caseInfo && !isConsensus && !correcting;
        const meta = LIFECYCLE_STEPS[i];
        return (
          <div key={i} className="flex items-center gap-1"
            title={!applicable ? `“${meta.short}” — not applicable to a control (no-scar) scan` : canView ? (current ? `“${meta.short}” (current step)` : `Inspect “${meta.short}” (read-only; roll back to edit)`) : meta.label}>
            <span onClick={() => canView && selectStep(i === step ? null : i)} style={{
              display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, lineHeight: 1,
              padding: "3px 7px", borderRadius: 11, whiteSpace: "nowrap",
              background: reached ? meta.color : "var(--c-surface2)",
              color: reached ? "#08121f" : "var(--c-text-dim)",
              fontWeight: current ? 700 : 500,
              // solid white outline = the LIVE current step; dashed = the step you're inspecting.
              outline: current ? "2px solid #fff" : (viewing ? "2px dashed #fff" : "none"), outlineOffset: -1,
              opacity: !applicable ? 0.3 : reached ? 1 : 0.7,
              textDecoration: !applicable ? "line-through" : "none",
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
    <Button size="small" variant="outlined" disabled={busy || !segLoaded || correctBusy} onClick={() => { set("corneaOnlyPaint", false); loadCorrectionLayer(); }}
      startIcon={correctBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
      title="Edit the labelmap with the pen (cornea/scar/erase), then Save">{correctBusy ? "Loading…" : "Correct ✎"}</Button>
  ) : (
    <>
      <Button size="small" variant="contained" color="secondary" disabled={busy || correctBusy} onClick={() => saveCorrection()}
        startIcon={segBusy ? <CircularProgress size={13} color="inherit" /> : undefined}>{segBusy ? "Saving…" : "Save correction"}</Button>
      {/* Undo lives in the pen bar (PaintToolbar) — no duplicate here (#4). */}
      <Button size="small" variant="outlined" color="inherit" disabled={busy || correctBusy} onClick={() => cancelCorrection()}
        startIcon={correctBusy ? <CircularProgress size={13} color="inherit" /> : undefined}>{correctBusy ? "Cancelling…" : "Cancel"}</Button>
    </>
  );

  // #11 — STEP 5 cornea/background vet: paint cornea/background (scar pen hidden), then confirm → unlocks Scar.
  const CorneaVet = !correcting ? (
    <>
      <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Vet the cornea/background segmentation, then:</span>
      <Button size="small" variant="outlined" disabled={busy || !segLoaded || corneaVetBusy} onClick={() => startCorneaVetPaint()}
        startIcon={corneaVetBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
        title="Paint to correct the SAM2 cornea/background: Cornea (blue) to add, Background (grey) to remove. No scar yet.">{corneaVetBusy ? "Loading…" : "✎ Paint cornea/background"}</Button>
      <Button size="small" variant="contained" color="secondary" disabled={busy || !segLoaded} onClick={() => confirmCorneaVet()}
        title="Confirm the cornea/background is correct — unlocks scar detection/editing.">✓ Confirm cornea/background</Button>
    </>
  ) : (
    <>
      <Button size="small" variant="contained" color="secondary" disabled={busy} onClick={() => confirmCorneaVet()}
        title="Save the cornea/background edits and unlock scar detection.">✓ Confirm cornea/background</Button>
      {/* Undo lives in the pen bar (PaintToolbar) — no duplicate here (#4). */}
      <Button size="small" variant="outlined" color="inherit" disabled={busy} onClick={() => cancelCorrection()}>Cancel</Button>
    </>
  );

  const ScheduleBtn = (
    <Button size="small" variant={scheduled ? "outlined" : "contained"} color="success" disabled={busy || correcting}
      onClick={() => scheduleTraining(!scheduled)} title="Mark this scan ready for nnU-Net training (turns it green).">
      {scheduled ? "Scheduled ✓ (unschedule)" : "Schedule for training"}
    </Button>
  );
  // #6 — Auto subgroup assignment lives WITH the subgroup controls (steps 7/8), not in the global top-right.
  const AutoSubgroupBtn = !isConsensus ? (
    <Button size="small" variant="outlined" color="secondary" disabled={busy || subgroupBusy || correcting}
      onClick={() => { setShowSubgroup(true); autoSubgroups(); }}
      title="Automatically group this eye's repeat scans into subgroups by aligning their hysteresis bright spots (each lesion's replicates cluster together; a displaced lesion splits off), with an overlay to verify before applying.">
      ⊞ Auto subgroups
    </Button>
  ) : null;
  const ExportBtn = (
    <>
      <Button size="small" variant="outlined" color="success" disabled={busy} onClick={() => { setBusyAction("export"); exportScarSummary(); }}
        startIcon={busyAction === "export" ? <CircularProgress size={13} color="inherit" /> : undefined}
        title="Recompute scar volume/area/density for every case → scar_summary.csv">{busyAction === "export" ? "Exporting…" : "Export metrics"}</Button>
      {scarSummaryInfo && !scarBusy && (
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)", maxWidth: 320, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
          title={scarSummaryInfo}>{scarSummaryInfo}</span>
      )}
    </>
  );
  const AlignBtn = (
    <Button size="small" variant="contained" color="info" disabled={busy || correcting} onClick={() => { setBusyAction("align"); alignReplicates(); }}
      startIcon={busyAction === "align" ? <CircularProgress size={13} color="inherit" /> : undefined}
      title="Register + vote this eye's repeat scans (same subgroup) into one consensus, using the scar as-is. Normalization against controls is the next step.">
      {busyAction === "align" ? "Aligning…" : "⌖ Align replicates"}
    </Button>
  );
  const NormalizeBtn = (
    <Button size="small" variant="contained" color="info" disabled={busy || correcting} onClick={() => { setBusyAction("normalize"); normalizeConsensus(); }}
      startIcon={busyAction === "normalize" ? <CircularProgress size={13} color="inherit" /> : undefined}
      title="Re-derive scar as excess over the control (no-scar) baseline and rebuild the consensus. Needs tagged + segmented control scans.">
      {busyAction === "normalize" ? "Normalizing…" : "◎ Normalize against controls"}
    </Button>
  );
  const SkipNormBtn = (
    <Button size="small" variant="outlined" color="inherit" disabled={busy || correcting} onClick={() => { setBusyAction("skipnorm"); skipNormalization(); }}
      startIcon={busyAction === "skipnorm" ? <CircularProgress size={13} color="inherit" /> : undefined}
      title="Skip control-normalisation — keep the aligned consensus as-is, then correct / schedule it.">
      {busyAction === "skipnorm" ? "Skipping…" : "⏭ Skip normalization"}
    </Button>
  );
  // STEP 9 scar-source decision: which scar boundary becomes each replicate's TRAINING label.
  const scarSource = (manifest?.consensus_scar_source as string | undefined) ?? null;
  const ScarSource = (
    <span className="flex items-center gap-1.5">
      <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Training scar:</span>
      <Button size="small" variant={scarSource === "consensus" ? "contained" : "outlined"} color="info" disabled={busy || correcting}
        startIcon={consensusScarMode === "consensus" ? <CircularProgress size={13} color="inherit" /> : undefined}
        onClick={() => applyConsensusScar("consensus")}
        title="Use the voted CONSENSUS scar for EVERY replicate (each truncated to its own data FOV, so a partial scan only gets the part within its data). Most reproducible training label.">
        {consensusScarMode === "consensus" ? "Applying…" : "Use consensus (all)"}
      </Button>
      <Button size="small" variant={scarSource === "own" ? "contained" : "outlined"} color="inherit" disabled={busy || correcting}
        startIcon={consensusScarMode === "own" ? <CircularProgress size={13} color="inherit" /> : undefined}
        onClick={() => applyConsensusScar("own")}
        title="Keep each replicate's OWN scar boundary as its training label.">
        {consensusScarMode === "own" ? "Saving…" : "Keep each replicate's"}
      </Button>
    </span>
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
        <MenuItem value="brightness" sx={{ fontSize: 12 }}>Brightness percentile</MenuItem>
      </Select>
      <label className="flex items-center gap-1 text-xs" style={{ color: "var(--c-text-dim)" }} title="How much hyper-reflectivity to flag">
        sens<input type="range" min={1} max={40} value={sensitivity} style={{ width: 64 }} onChange={(e) => set("scarSensitivity", Number(e.target.value))} />
      </label>
    </>
  );

  // INITIAL scar DETECTION (step 6 "Scar"): pick a strategy + run a detector → produces the scar. The scar
  // CORRECTION tools (guide-hints, Correct ✎) live in step 7 once an initial scar exists. Non-control only.
  const ScarDetect = classification !== "control" ? (
    <>
      {ScarMethod}
      <Button size="small" variant="outlined" color="error" disabled={busy || !segLoaded}
        onClick={() => { setScarKind("threshold"); runScarAuto(); }}
        startIcon={scarKind === "threshold" ? <CircularProgress size={13} color="inherit" /> : undefined}
        title="CLASSICAL detector: threshold the scar inside the cornea using the selected method (e.g. hysteresis) + sensitivity. Fast (seconds). Re-run after changing the method/sensitivity.">
        {scarKind === "threshold" ? "Detecting…" : "Detect scar (threshold)"}
      </Button>
      <Button size="small" variant="outlined" color="error" disabled={busy || !segLoaded}
        onClick={() => { setScarKind("sam2"); runScarAutoSam2(); }}
        startIcon={scarKind === "sam2" ? <CircularProgress size={13} color="inherit" /> : undefined}
        title="SAM2 (deep-learning) scar: run SAM2 on cornea-vs-scar across axial/coronal/sagittal and take the 2-of-3 vote. Slower (~1–2 min); an alternative to the threshold detector when it struggles.">
        {scarKind === "sam2" ? "Running SAM2…" : "Scar via SAM2"}
      </Button>
      {scarKind && scarProgress && (
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)", maxWidth: 300, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={scarProgress}>{scarProgress}</span>
      )}
    </>
  ) : null;

  // Scar REFINEMENT (correction): click-hint touch-up of an EXISTING scar. Lives in step 7. Non-control only.
  const ScarRefine = classification !== "control" ? (
    <>
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
          <Button size="small" variant="contained" color="warning" disabled={busy || hintCount === 0}
            startIcon={scarKind === "hints" ? <CircularProgress size={13} color="inherit" /> : undefined}
            onClick={() => { setScarKind("hints"); applyScarHints(); }}>{scarKind === "hints" ? "Applying…" : `Apply (${hintCount})`}</Button>
          <Button size="small" variant="outlined" disabled={busy || hintCount === 0} onClick={() => clearScarHints()}>Clear</Button>
        </>
      )}
    </>
  ) : null;

  // Full scar controls = detect + refine, shown together in the Scar-correction step (7) so you can iterate.
  const ScarReRun = classification !== "control" ? <>{ScarDetect}{ScarRefine}</> : null;

  const sep = <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />;

  // Auto-populated scans reach Cornea (SAM2) WITHOUT a human approving the preprocessing (preproc_vetted unset,
  // e.g. the batch populate). Offer a NON-destructive approve at the segmentation steps so the Vetted step can be
  // filled without a rollback (which would clear SAM2). Null once vetted, so it never shows for the normal flow.
  const ApprovePreproc = !manifest?.preproc_vetted ? (
    <Button size="small" variant="outlined" color="warning" disabled={busy}
      onClick={() => { setBusyAction("approve"); approvePreprocessing(); }}
      startIcon={busyAction === "approve" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
      title="Mark the preprocessing as manually vetted (fills the Vetted step). Non-destructive — keeps the SAM2 segmentation. Does NOT apply any auto-detected corrections. Shown because this scan was segmented without an explicit preprocessing approval.">
      {busyAction === "approve" && caseBusy ? "Approving…" : "✓ Approve preprocessing"}
    </Button>
  ) : null;

  // DEFECT-MARKING (replaces the old #A/#B/#C issue flags): a "⚑ Mark defect" TOGGLE that puts the main
  // viewer into column-marking mode (drag over a sagittal/axial slice to mark the WRONG columns → persisted to
  // manifest.defect_marks so the assistant reads exactly which frames/columns are wrong), plus a "⚠ Difficult"
  // TOGGLE bound to manifest.difficult_scan (this scan needs manual help). Shown at the cornea-review steps (4/5).
  const defectMarks: unknown[] = Array.isArray(manifest?.defect_marks) ? (manifest!.defect_marks as unknown[]) : [];
  const markCount = defectMarks.length;
  const difficult = Boolean(manifest?.difficult_scan);
  const FlagButtons = (
    <span className="flex items-center gap-1 text-xs" style={{ color: "var(--c-text-dim)" }}>
      <Button size="small" variant={markDefectMode ? "contained" : "outlined"} color="warning" disabled={busy}
        onClick={() => set("markDefectMode", !markDefectMode)}
        title="Mark WRONG columns: toggle on, then drag over the current sagittal/axial slice in the main viewer to mark the bad columns. Marks accumulate across slices and are saved to manifest.defect_marks (the assistant reads them).">
        ⚑ Mark defect{markCount > 0 ? ` (${markCount})` : ""}
      </Button>
      <Button size="small" variant={difficult ? "contained" : "outlined"} color="error" disabled={busy}
        onClick={() => void setDifficult(!difficult)}
        title="Mark this scan as a DIFFICULT SCAN needing manual help — persisted to manifest.difficult_scan so the assistant knows to hand-correct it.">
        ⚠ Difficult
      </Button>
    </span>
  );

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
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
          {proposals.hasProposal
            ? "An auto-correction was detected (shown in pink). Approve the output as-is, or apply the correction first:"
            : "Review the preprocessing (Before/after · Fix-columns), then:"}
        </span>
        <Button size="small" variant="contained" color="warning" disabled={busy}
          onClick={() => { setBusyAction("approve"); approvePreprocessing(); }}
          startIcon={busyAction === "approve" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
          title="Mark the preprocessing as manually vetted (turns the scan orange) — unlocks classification. Accepts the CURRENT output as-is; does NOT apply any auto-detected correction.">
          {busyAction === "approve" && caseBusy ? "Approving…" : "✓ Approve preprocessing"}
        </Button>
        {proposals.hasProposal && (
          <Button size="small" variant="outlined" color="secondary" disabled={busy}
            onClick={() => { setBusyAction("apply"); applyCorrections(); }}
            startIcon={busyAction === "apply" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
            title="Bake in the auto-detected de-tilt / crop / surface-crop (shown in pink) and re-warp from the raw .OCT. Produces a fresh output to re-inspect (resets to Preprocessed · Auto); drops any segmentation. Approve it after reviewing.">
            {busyAction === "apply" && caseBusy ? "Applying…" : "⟳ Apply corrections"}
          </Button>
        )}
        <Button size="small" variant="outlined" color="primary" disabled={busy} onClick={() => rerunPreprocess()}
          startIcon={caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
          title="Re-run the full auto preprocessing on the raw .OCT again (fresh surface detect + warp). Keeps this scan's params/classification; drops the current correction's segmentation.">
          {caseBusy ? "Re-running…" : "↻ Re-run preprocessing"}
        </Button>
        <Button size="small" variant="outlined" color="warning" disabled={busy} onClick={() => { setBusyAction("useraw"); approveRaw(); }}
          startIcon={busyAction === "useraw" ? <CircularProgress size={13} color="inherit" /> : undefined}
          title="Use the ORIGINAL (raw) scan as the working volume instead of the correction. Drops any segmentation; also marks it vetted.">
          {busyAction === "useraw" ? "Loading…" : "↩ Use original (raw)"}
        </Button>
        {sep}{FlagButtons}
      </>
    );
  } else if (step === 3) {
    // vetted (pink) → segment the CORNEA (SAM2). Classification (scar/control) is a LATER step now — it comes
    // after cornea-vetting and gates only the scar branch, so SAM2 runs without it.
    actions = (
      <div className="flex items-center gap-2 text-xs" style={{ color: "var(--c-text-dim)" }}>
        <Button size="small" variant="contained" color="primary" disabled={busy || !!sam2RunningCaseId} onClick={() => runSam2()}
          title="Run SAM2 cornea-vs-background segmentation. Scar/control classification and scar segmentation come in later steps.">
          ▶ Run SAM2 (cornea)
        </Button>
        {sep}
        {/* #10 — save the preprocessing correction as an MP4 grid (planes × passes, before↔after). */}
        <Button size="small" variant="outlined" disabled={busy || mp4Busy} onClick={() => exportCorrectionMp4()}
          startIcon={mp4Busy ? <CircularProgress size={13} color="inherit" /> : undefined}
          title="Render this scan's correction as an MP4: rows = axial/coronal/sagittal, columns = after (final) → passes → before (raw), scrubbing every slice.">
          {mp4Busy ? "Rendering MP4…" : "🎞 Save correction MP4"}
        </Button>
        {correctionMp4Url && !mp4Busy && (
          <a href={correctionMp4Url} download style={{ color: "var(--c-accent)", fontSize: 12 }} title={correctionMp4Info}>⤓ Download MP4</a>
        )}
        {sep}{FlagButtons}
      </div>
    );
  } else if (step === 4) {
    // cornea segmented (fuchsia) → VET the cornea/background (paint, scar pen hidden), then confirm → unlocks
    // classification. Scar detection is NOT shown here until cornea/background is confirmed AND the scan is classified.
    // Auto-populated scans also get a non-destructive "Approve preprocessing" here (their Vetted step was skipped).
    actions = <>{ApprovePreproc && <>{ApprovePreproc}{sep}</>}{CorneaVet}{sep}{FlagButtons}</>;
  } else if (step === 5) {
    // cornea/background vetted (purple) → CLASSIFY scar/control. Moved here from before SAM2 (it only gates the
    // scar branch): a control schedules next; a scar scan proceeds to subgroup.
    actions = (
      <div className="flex items-center gap-2 text-xs" style={{ color: "var(--c-text-dim)" }}>
        {ApprovePreproc && <>{ApprovePreproc}{sep}</>}
        <span className="flex items-center gap-1"
          title="Does this corrected volume have a scar? 'No scar' marks it a control (normal baseline). Replicates/controls are grouped in the sidebar.">
          Classify:
          <Button size="small" variant={classification === "scar" ? "contained" : "outlined"} color="error"
            disabled={busy} onClick={() => setClassification(classification === "scar" ? null : "scar")}>Scar</Button>
          <Button size="small" variant={classification === "control" ? "contained" : "outlined"} color="inherit"
            disabled={busy} onClick={() => setClassification(classification === "control" ? null : "control")}>No scar (control)</Button>
        </span>
        {sep}{FlagButtons}
      </div>
    );
  } else if (step === 6) {
    // classified (violet) → SUBGROUP step (assigned BEFORE scar so the strategy comparison at the Scar step is
    // per-subgroup). Scar scan: assign which lesion set it belongs to. Control: no lesion subgroup is needed
    // (the control baseline is eye-wide, control_cases() ignores subgroup) → skip straight to "no scar".
    actions = classification === "control" ? (
      <>
        {/* A control (no scar) is READY once its cornea is vetted — the scar/subgroup/align/normalize/correct
            steps (7-11) don't apply (greyed in the strip). Its cornea-only label is the training label + the
            normal baseline, so the next action is Schedule. Correct stays available to touch up the cornea. */}
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Control (no scar) — cornea vetted; no scar/align/normalize needed.</span>
        {ScheduleBtn}{sep}{Correct}
      </>
    ) : (
      <>
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Assign this scan's scar subgroup (which lesion set — groups the repeats that align together), then detect scar:</span>
        {SubgroupConfirm}{sep}{AutoSubgroupBtn}
      </>
    );
  } else if (step === 7) {
    // subgroup assigned (purple) → SCAR DETECTION. Subgroup is confirmed, so "⚖ Compare strategies" (right
    // bar) is now PER-SUBGROUP. A control normally skips this step; a fallback skip is shown just in case.
    actions = classification === "control" ? (
      <>
        <Button size="small" variant="contained" color="secondary" disabled={busy} onClick={() => skipScar()}
          title="Control (no scar) — mark the scar step done and continue.">✓ No scar (control) — continue</Button>
        {sep}{Correct}
      </>
    ) : (
      <>
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Detect the scar (pick a strategy + run; refine/correct in the next step):</span>
        {ScarDetect}
      </>
    );
  } else if (step === 8) {
    // scar segmented (rose) → refine/correct the scar, then align this subgroup's replicates.
    actions = (
      <>
        {ScarReRun && <>{ScarReRun}{sep}</>}{Correct}{sep}{AlignBtn}
        {classification !== "control" && (
          <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>(subgroup “{subgroup}”)</span>
        )}
      </>
    );
  } else if (step === 9) {
    // aligned (teal) → choose the TRAINING scar (each replicate's own vs the voted consensus), then normalize
    // against controls or SKIP normalization (use the consensus as-is); Correct to touch up. Schedule/Export
    // are NOT here — they live at the later (corrected/scheduled) steps.
    actions = <>{ScarSource}{sep}{NormalizeBtn}{sep}{SkipNormBtn}{sep}{Correct}</>;
  } else if (step === 10) {
    // normalized (cyan) → correct only; scheduling/export live at the corrected/scheduled steps (11/12).
    actions = <>{Correct}</>;
  } else if (step === 11) {
    // manually corrected (dark blue)
    actions = <>{ScheduleBtn}{sep}{Correct}{ExportBtn}</>;
  } else {
    // step 12 — scheduled (green)
    actions = (
      <>
        <span className="text-xs" style={{ color: "#4ade80" }}>✓ Scheduled for training.</span>
        <Button size="small" variant="outlined" disabled={busy} onClick={() => scheduleTraining(false)}>Unschedule</Button>
        {Correct}{ExportBtn}
      </>
    );
  }

  return (
    <div className="flex flex-col border-b" style={{ backgroundColor: "var(--c-surface)", borderColor: "var(--c-border)" }}>
      {/* Row 1 — the per-scan lifecycle STEPS. */}
      <div className="flex items-center gap-3 px-3 overflow-x-auto [&>*]:shrink-0" style={{ minHeight: 28 }}>
        {strip}
      </div>
      {/* Row 2 — the action BUTTONS for the current step (+ compare strategies + live progress). Wraps onto
          extra rows when a step has many controls, rather than scrolling off-screen. */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-1 border-t" style={{ minHeight: 40, borderColor: "var(--c-border)" }}>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">{caseInfo ? actions : <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>Open or preprocess a scan to begin.</span>}</div>
        <div className="flex-1" style={{ minWidth: 12 }} />
        {/* PUBLICATION: compare scar-detection strategies' reproducibility across the eye's replicates. Lives
            ONLY in the Scar-detection step (7) — where you pick a detector, AFTER subgroup is assigned, so the
            comparison is PER-SUBGROUP. Needs ≥2 cornea-segmented replicates of the eye+subgroup; a control has
            no scar so it's hidden. (Not shown on the aligned consensus / step 9.) */}
        {caseInfo && step === 7 && classification !== "control" && (
          <Button size="small" variant="outlined" color="info" disabled={busy || correcting}
            onClick={() => { setShowCompare(true); compareStrategies(); }}
            title="Run every scar strategy on this eye's replicates and tabulate test–retest reproducibility (pairwise Dice, HD95, volume CV%) — for strategy comparison in the paper. Read-only; doesn't change the scan.">
            ⚖ Compare strategies
          </Button>
        )}
        {/* #6 — "⊞ Auto subgroups" lives in the Subgroup steps (7/8) actions, not here. */}
        {/* Live progress text (SAM2 per-plane %, scar phase, …) next to the spinner — not just an icon. */}
        {(busy || !!sam2RunningCaseId) && status.kind === "working" && (
          <span className="text-xs" style={{ color: "var(--c-text-dim)", maxWidth: 360, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
            title={status.detail}>{status.detail}</span>
        )}
        {(busy || !!sam2RunningCaseId) && <CircularProgress size={16} />}
      </div>

      {/* PUBLICATION: scar-strategy reproducibility table. */}
      <Dialog open={showCompare} onClose={() => setShowCompare(false)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ fontSize: 16 }}>Scar strategy reproducibility (test–retest)</DialogTitle>
        <DialogContent sx={{ fontSize: 13 }}>
          {!strategyComparison && scarBusy && (
            <div className="flex items-center gap-2 py-4"><CircularProgress size={18} />
              <span style={{ maxWidth: 520, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={status.detail}>
                {status.kind === "working" ? status.detail : "Running each strategy on the eye's replicates…"}
              </span>
            </div>
          )}
          {strategyComparison?.cancelled && (
            <div className="text-xs mb-2" style={{ color: "var(--c-amber, #ffaa28)" }}>⚠ Stopped early — partial results below.</div>
          )}
          {strategyComparison && (
            <>
              <div className="text-xs mb-2" style={{ color: "var(--c-text-dim)" }}>
                {strategyComparison.n} replicates{strategyComparison.subgroup ? ` · subgroup “${strategyComparison.subgroup}”` : ""} · φ={strategyComparison.phi_percentile} ·
                reproducibility only (no manual GT). Higher Dice / lower HD95·CV·RC = more reproducible; read Dice alongside volume (Dice rises with mask size).
                {strategyComparison.crop_aware && <><br /><b style={{ color: "var(--c-amber, #ffaa28)" }}>⊟ Crop-aware:</b> a replicate has a cropped lateral band — metrics use only the region valid in every replicate.</>}
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
          {scarBusy && (
            <Button size="small" color="error" variant="outlined" onClick={() => cancelCompareStrategies()}
              title="Stop the run — the current step finishes, then no further strategies/replicates are processed.">
              ✕ Cancel run
            </Button>
          )}
          <Button size="small" disabled={!strategyComparison} onClick={downloadComparisonCsv}>⤓ Download CSV</Button>
          {/* Closing while running also cancels (otherwise the slow run would keep grinding in the background). */}
          <Button size="small" variant="contained" onClick={() => { if (scarBusy) cancelCompareStrategies(); setShowCompare(false); }}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* AUTO SUBGROUP assignment: overlay + editable proposed grouping. */}
      <Dialog open={showSubgroup} onClose={() => setShowSubgroup(false)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ fontSize: 16 }}>Auto subgroup assignment (bright-spot alignment)</DialogTitle>
        <DialogContent sx={{ fontSize: 13 }}>
          {!subgroupProposal && subgroupBusy && (
            <div className="flex items-center gap-2 py-4"><CircularProgress size={18} /> Aligning hysteresis bright spots across the eye's scans…</div>
          )}
          {!subgroupProposal && !subgroupBusy && (
            <div className="py-4 text-sm" style={{ color: status.kind === "error" ? "var(--c-red)" : "var(--c-text-dim)" }}>
              {status.kind === "error" ? status.detail : "Need ≥2 cornea-segmented scar scans of this eye to auto-assign subgroups."}
            </div>
          )}
          {subgroupProposal && (
            <>
              <div className="text-xs mb-2" style={{ color: "var(--c-text-dim)" }}>
                {subgroupProposal.members.length} scans → <b>{subgroupProposal.n_subgroups}</b> proposed subgroup(s)
                {subgroupProposal.patient ? ` · ${String(subgroupProposal.patient).toUpperCase()} ${String(subgroupProposal.eye).toUpperCase()}` : ""}.
                Overlay = each scan's scar footprint in a common cornea frame, coloured by proposed subgroup: same
                lesion piles up (white where replicates agree); a displaced lesion shows its colour apart. Edit a
                label if the grouping is wrong, then Apply.
              </div>
              {subgroupProposal.overlay && (
                <img src={subgroupProposal.overlay} alt="subgroup overlay"
                  style={{ display: "block", width: "100%", maxHeight: 220, objectFit: "contain", imageRendering: "pixelated",
                           border: "1px solid var(--c-border)", borderRadius: 4, marginBottom: 8, background: "#000" }} />
              )}
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr>{["scan", "scar mm³", "blobs", "subgroup"].map((h, i) => (
                    <th key={h} style={{ textAlign: i === 0 ? "left" : i === 3 ? "left" : "right", padding: "4px 8px",
                      borderBottom: "1px solid var(--c-border)", color: "var(--c-text-dim)" }}>{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {subgroupProposal.members.map((cid) => {
                    const b = subgroupProposal.blobs[cid] || { scar_mm3: 0, n_blobs: 0 };
                    const lab = editAssign[cid] ?? String(subgroupProposal.subgroups[cid] ?? 1);
                    return (
                      <tr key={cid}>
                        <td style={{ padding: "4px 8px", borderBottom: "1px solid var(--c-border)", fontWeight: 600 }}>{cid.split("_").pop()}</td>
                        <td style={{ textAlign: "right", padding: "4px 8px", borderBottom: "1px solid var(--c-border)" }}>{b.scar_mm3?.toFixed?.(3) ?? "—"}</td>
                        <td style={{ textAlign: "right", padding: "4px 8px", borderBottom: "1px solid var(--c-border)" }}>{b.n_blobs}</td>
                        <td style={{ padding: "4px 8px", borderBottom: "1px solid var(--c-border)" }}>
                          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, marginRight: 6, background: subColor(lab), verticalAlign: "middle" }} />
                          <input value={lab} onChange={(e) => setEditAssign((m) => ({ ...m, [cid]: e.target.value }))}
                            style={{ width: 54, fontSize: 12, padding: "2px 4px", background: "var(--c-surface2)", color: "var(--c-text)", border: "1px solid var(--c-border)", borderRadius: 3 }} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div className="text-[11px] mt-2" style={{ color: "var(--c-text-dim)" }}>
                pairwise: {subgroupProposal.pairs.map((p) => `${p.a.split("_").pop()}~${p.b.split("_").pop()} sim=${p.sim} (Δ=${p.centroid_dist_mm ?? "∞"}mm)`).join("  ·  ")}
              </div>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button size="small" disabled={subgroupBusy} onClick={() => autoSubgroups()}>↻ Re-run</Button>
          <Button size="small" variant="contained" disabled={!subgroupProposal || subgroupBusy}
            onClick={async () => { await applySubgroups(editAssign); setShowSubgroup(false); }}>
            Apply &amp; confirm
          </Button>
          <Button size="small" onClick={() => setShowSubgroup(false)}>Close</Button>
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
