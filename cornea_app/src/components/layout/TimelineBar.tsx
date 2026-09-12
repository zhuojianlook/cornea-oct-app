import { useEffect, useState } from "react";
import { Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, MenuItem, Select } from "@mui/material";
import { useWorkflowStore } from "../../store/workflowStore";
import { useCaseStore, type DefectMark } from "../../store/caseStore";
import { api } from "../../api/client";
import { LIFECYCLE_STEPS, scanStep, stepReached, stepApplicable, octProposals, awaitingCorneaDetection, type LifecycleStep } from "../../api/lifecycle";
import { useReviewQueueStore, nextAfter } from "../../store/reviewQueueStore";
import { usePendingEditStore } from "../../store/pendingEditStore";
import { SubgroupAlignDialog } from "../panels/GroupAlignPanel";

/* Per-scan lifecycle TIMELINE — the active scan's progress through the colour-coded steps, surfacing ONLY
   the next action(s). Order: Raw → Preprocessed[auto] → Vetted → Aligned(group curvature) → SAM2(cornea) →
   Cornea✓ → Classified(scar/control) → Subgroup → Scar → Scar-aligned → Normalized → Corrected → Scheduled.
   Classification comes AFTER cornea-vetting (it gates only the scar branch, not SAM2), so a control can schedule
   straight after Cornea✓.
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
  const activeCaseId = useCaseStore((s) => s.caseId);
  const manifest = (caseInfo?.manifest ?? null) as Record<string, unknown> | null;
  const classification = (manifest?.scar_classification as "scar" | "control" | null | undefined) ?? null;
  const setClassification = useCaseStore((s) => s.setClassification);
  const setDifficult = useCaseStore((s) => s.setDifficult);
  const setRejected = useCaseStore((s) => s.setRejected);
  // Border corrections drawn but not Confirmed — committed by the reject handler so a correction costs one
  // click, not three. See pendingEditStore.
  const pendingEdit = usePendingEditStore((s) => s.pending);
  const takePendingEdit = usePendingEditStore((s) => s.takePending);
  // CORRECTED-result edge edits (before/after right pane), committed alongside raw edits by the SAME re-run.
  const correctedEdge = usePendingEditStore((s) => s.correctedEdge);
  const takeCorrectedEdge = usePendingEditStore((s) => s.takeCorrectedEdge);
  const editTarget = usePendingEditStore((s) => s.editTarget);
  const trustedSlices = usePendingEditStore((s) => s.trustedSlices);
  const commitCorrectedEdgeAnchors = useCaseStore((s) => s.commitCorrectedEdgeAnchors);
  const correctedPts = correctedEdge?.dirty ? correctedEdge.nPoints : 0;
  // Did the reviewer actually CHANGE the raw line/marks (vs just re-loading a scan that already carries them)?
  // Only a real change reshapes the volume → full re-run; a corrected-edge-only pin stays instant.
  const rawDirty = !!pendingEdit && (pendingEdit.bordersDirty || pendingEdit.cropFrames !== null
    || pendingEdit.cropRegion !== null || pendingEdit.defectCols !== null || pendingEdit.postAnchors !== null);
  // In Corrected mode, "Smooth to trusted slices" propagates the reviewer's EDITED (drawn) + APPROVED slices
  // across the whole volume. It works WITH edits (the drawn line IS a trusted good curve), so — unlike the old
  // "use the detected edge" — there is no correctedPts===0 requirement; only a raw change diverts to a raw re-run.
  const smoothAlignReady = editTarget === "corrected" && !rawDirty;
  // Trusted slices = APPROVED laterals (their detection is a good curve, sent as trustedLaterals) + slices EDITED
  // this session (their drawn curve is a trusted curve, carried by the persisted corrected_edge_anchors).
  const trustedForCase = (trustedSlices && trustedSlices.caseId === activeCaseId) ? trustedSlices.slices : [];
  const editedLaterals = (correctedEdge?.dirty && correctedEdge.anchors)
    ? Object.keys(correctedEdge.anchors).filter((k) => Object.keys(correctedEdge.anchors[k] ?? {}).length > 0).length : 0;
  const trustedCount = trustedForCase.length + editedLaterals;
  // FOLD-BACK readiness. Counts the PERSISTED corrected-edge laterals (they autosave ~900 ms after drawing,
  // so by the time the reviewer reaches a button they are already on disk) unioned with anything still dirty
  // in the panel. Persisted-only is the normal case after a reload, and it must still be foldable.
  const dirtyCorrLats = (correctedEdge?.dirty && correctedEdge.anchors)
    ? Object.keys(correctedEdge.anchors).filter((k) => Object.keys(correctedEdge.anchors[k] ?? {}).length > 0) : [];
  // (The smooth-align verdict readout lived here until 2026-09-01. Its button is gone, so nothing renders it;
  //  store/smoothAlign.ts and the backend record are left in place for when the guard is fixed.)
  // What the reject button is about to save. Counting only border POINTS read "(0)" whenever the pending work
  // was crop or defect marks — telling the reviewer their marks would be discarded, which was the opposite of
  // the truth.
  const pendingSummary = pendingEdit ? [
    pendingEdit.nPoints > 0 ? `${pendingEdit.nPoints} border pt` : null,
    pendingEdit.cropFrames?.length ? `${pendingEdit.cropFrames.length} crop frame` : null,
    pendingEdit.cropRegion?.frames.length ? `${pendingEdit.cropRegion.frames.length} region col` : null,
    pendingEdit.defectCols?.cols.length ? `${pendingEdit.defectCols.cols.length} marked col` : null,
    pendingEdit.postAnchors
      ? `${Object.values(pendingEdit.postAnchors).reduce((a, m) => a + Object.keys(m).length, 0)} bottom pt` : null,
    correctedPts > 0 ? `${correctedPts} corrected-edge pt` : null,
  ].filter(Boolean).join(" + ") || "cleared marks"
    : (correctedPts > 0 ? `${correctedPts} corrected-edge pt` : "");
  const commitBorderAnchors = useCaseStore((s) => s.commitBorderAnchors);
  const commitOctMarks = useCaseStore((s) => s.commitOctMarks);
  const startReprocessBatch = useCaseStore((s) => s.startReprocessBatch);
  const startDetectorTune = useCaseStore((s) => s.startDetectorTune);
  const rerunWithCorrections = useCaseStore((s) => s.rerunWithCorrections);
  const setDefectMarks = useCaseStore((s) => s.setDefectMarks);
  const approvePreprocessing = useCaseStore((s) => s.approvePreprocessing);
  const rerunPreprocess = useCaseStore((s) => s.rerunPreprocess);
  const clearAllCorrections = useCaseStore((s) => s.clearAllCorrections);
  // Ground-truth capture: this scan carries a manual border correction (Fix-columns anchors) → Approving records
  // it as CONFIRMED ground truth for the auto-detector training corpus. The toggle lets the user EXCLUDE an
  // idealised case (e.g. a motion-corrupted scan whose hand-drawn border is not real geometry) from the corpus.
  const octParams = (manifest?.oct_params ?? null) as Record<string, unknown> | null;
  const anchorObj = (octParams?.border_anchors ?? null) as Record<string, unknown> | null;
  const hasBorderCorrection = Boolean(anchorObj && Object.keys(anchorObj).length);
  // VERIFIED slices on the corrected scan. The reviewer's loop (2026-09-02) treats a DRAWN edge and a slice
  // MARKED accurate as the same thing — "in either case those slices are verified as accurate" — so both feed
  // the regenerate. Sources: persisted drawings, persisted marks, and anything still dirty in the panel.
  const verifiedDrawn = [...Object.keys((octParams?.corrected_edge_anchors ?? {}) as Record<string, unknown>),
                         ...Object.keys((octParams?.corrected_post_anchors ?? {}) as Record<string, unknown>)];
  const verifiedMarked = Object.keys((octParams?.corrected_accurate ?? {}) as Record<string, unknown>);
  // Marks the reviewer has just made live in pendingEditStore until the POST round-trips, so include them too:
  // otherwise the button flickers out between the click and the manifest mirror, and a mark made while the
  // write is in flight looks like it did not count.
  const foldableLats = new Set([...verifiedDrawn, ...verifiedMarked, ...dirtyCorrLats,
                                ...trustedForCase.map(String)]).size;
  const [corpusEligible, setCorpusEligible] = useState(true);
  // ARM-THEN-ACT, because window.confirm is inert here. It returns FALSE without ever showing a dialog in the
  // WebKitGTK webview AND in the review browser pane (probed 2026-09-02), so every button guarded by it did
  // nothing at all when clicked: "↻ Re-preprocess" on any scan carrying border corrections, and "⟲ Clear all
  // corrections" always. OctLoader hit this before and swapped in a Dialog (OctLoader.tsx:749); this is the
  // same fix in one line of state — first click arms and relabels, second acts, 4 s to change your mind.
  const [armed, setArmed] = useState<null | "reprocess" | "clearall">(null);
  const arm = (k: "reprocess" | "clearall") => { setArmed(k); window.setTimeout(() => setArmed((c) => (c === k ? null : c)), 4000); };
  // FIXED HEIGHT for the review-row buttons (reviewer, 2026-09-02). Their labels swap between states —
  // "✓ Approve → next (1)" / "Approving…" / "Opening next…", "↻ Re-preprocess" / "↻ Discards border
  // corrections — click again" — and a spinner startIcon appears mid-action, so the row changed height and
  // nudged everything below it. Height pinned, labels never wrap.
  const ACT_SX = { height: 26, minHeight: 26, py: 0, px: 1, fontSize: 12,
                   textTransform: "none" as const, whiteSpace: "nowrap" as const, lineHeight: 1.2 };
  const applyCorrections = useCaseStore((s) => s.applyCorrections);
  const approveRaw = useCaseStore((s) => s.approveRaw);
  const caseBusy = useCaseStore((s) => s.busy);
  // Stale-pipeline auto re-run notice (caseStore.openCase): why this scan is re-running on open, then the outcome.
  const autoRerunNote = useCaseStore((s) => s.autoRerunNote);
  const scheduleTraining = useCaseStore((s) => s.scheduleTraining);
  const resetStep = useCaseStore((s) => s.resetStep);
  const confirmSubgroup = useCaseStore((s) => s.confirmSubgroup);
  // Step 3 → 4: group-wise 3D alignment of this patient+eye's replicate scans, per SUBGROUP (reviewer spec 2026-09-11):
  // "⧉ Align group" first asks about subgroups (SubgroupAlignDialog), the job stamps group_aligned when it completes.
  const resolveGroupId = useCaseStore((s) => s.resolveGroupId);
  const refreshCaseInfo = useCaseStore((s) => s.refreshCaseInfo);
  const approveAligned = useCaseStore((s) => s.approveAligned);
  const skipScar = useCaseStore((s) => s.skipScar);
  const scheduled = Boolean(manifest?.training_scheduled);
  const isConsensus = Boolean(manifest?.consensus_cases);
  // Crop-approval: an auto de-tilt/crop/surface-crop was DETECTED but not applied — the Approve action bakes
  // it in first (via approvePreprocessing), and the button relabels to make that clear.
  const proposals = octProposals(manifest);
  const subgroup = String(manifest?.scar_subgroup ?? "1") || "1";

  const busy = segBusy || scarBusy || caseBusy;
  const step: LifecycleStep = scanStep(manifest);
  const maxStep = LIFECYCLE_STEPS.length - 1;   // 13
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
  // "⧉ Align group" outcome (the stub's reason + the group's members) shown beside the button; per scan.
  const [alignNote, setAlignNote] = useState<string | null>(null);
  useEffect(() => { setAlignNote(null); }, [caseInfo?.case_id]);
  // The "⧉ Alignment (pairs)…" dialog was RETIRED (reviewer 2026-09-12 #1): the step-4 pane in the main area (AlignTabs)
  // carries the same content, so the button was redundant. GroupAlignPanel.tsx keeps the dialog component (unused).
  // "Subgroups for <group>" — the question "⧉ Align group" asks first (reviewer spec #1); per eye (base group id).
  const [subOpen, setSubOpen] = useState(false);
  const [subGid, setSubGid] = useState<string | null>(null);
  useEffect(() => { setSubOpen(false); setSubGid(null); }, [caseInfo?.case_id]);
  const openSubgroups = () => {
    setAlignNote(null);
    void resolveGroupId().then((gid) => {
      if (!gid) { setAlignNote("This scan's patient/eye is unknown — set the group's eye in the sidebar first."); return; }
      setSubGid(gid); setSubOpen(true);
    });
  };
  // REVIEW LOOP — approve/reject then jump to the next scan awaiting approval, so a pass through the backlog
  // is one click per scan instead of click-approve, hunt for the next row, click it.
  const reviewQueue = useReviewQueueStore((s) => s.queue);
  const openCaseInSidebar = useReviewQueueStore((s) => s.open);
  const queueSettled = useReviewQueueStore((s) => s.settled);
  const markSettled = useReviewQueueStore((s) => s.markSettled);
  const rejectsSinceCheckpoint = useReviewQueueStore((s) => s.rejectsSinceCheckpoint);
  const clearCheckpoint = useReviewQueueStore((s) => s.clearCheckpoint);
  const nVetted = useReviewQueueStore((s) => s.vetted);
  const nVettable = useReviewQueueStore((s) => s.vettable);
  const nRejected = useReviewQueueStore((s) => s.rejected);
  // How many corrections to bank before offering to learn from them. Persisted so it survives a reload.
  const [checkpointEvery, setCheckpointEvery] = useState<number>(() => {
    const v = Number(localStorage.getItem("cornea.checkpointEvery"));
    return Number.isFinite(v) && v >= 1 ? Math.min(200, Math.round(v)) : 10;
  });
  useEffect(() => { localStorage.setItem("cornea.checkpointEvery", String(checkpointEvery)); }, [checkpointEvery]);
  const [rejecting, setRejecting] = useState(false);      // legacy: kept so existing guards read false
  const [rejectReason, setRejectReason] = useState("");   // OPTIONAL note; empty is fine
  const [queueNote, setQueueNote] = useState<string | null>(null);   // "backlog is empty" / advance errors
  // Detector-tuning progress. Polled while a run is live so an hours-long background search is VISIBLE —
  // a button that appears to do nothing for an hour is indistinguishable from a broken one. Kept after it
  // finishes so the verdict ("adopted X" / "nothing beat the current detector") is readable.
  const [tuneBusy, setTuneBusy] = useState(false);
  const [tune, setTune] = useState<null | { running?: boolean; phase?: string; done?: number; total?: number;
    note?: string; adopted?: Record<string, number> | null; baseline?: { within?: number } | null;
    best?: { within?: number } | null }>(null);
  useEffect(() => {
    let stop = false;
    const poll = async () => {
      try {
        const r = await api.json<{ running?: boolean; phase?: string }>("/api/review/tune-status");
        if (stop) return;
        setTune(r as never);
        if (!r?.running) setTuneBusy(false);
      } catch { /* transient — the next tick retries */ }
    };
    void poll();
    // Only while something is live: idle polling of a machine that is otherwise busy preprocessing is waste.
    if (!tuneBusy && !tune?.running) return;
    const t = setInterval(() => void poll(), 4000);
    return () => { stop = true; clearInterval(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tuneBusy, tune?.running]);
  // Opening the NEXT scan is tracked separately from the approve/reject WRITE. They were one flag, so the
  // button sat on "Rejecting…" while the next volume loaded — and a first-time surface-crop detection can take
  // 25 s or more, which reads as a hang even though the rejection had already reached disk.
  const [navigating, setNavigating] = useState(false);
  // What is genuinely left: the published queue minus anything settled in this session and minus the open scan.
  const nLeft = reviewQueue.filter((id) => id !== activeCaseId && !queueSettled.has(id)).length;
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
      {([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] as LifecycleStep[]).map((i) => {
        const applicable = stepApplicable(manifest, i);   // control: scar steps 8-12 are N/A
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

  // #11 — STEP 6 cornea/background vet: paint cornea/background (scar pen hidden), then confirm → unlocks Scar.
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
  // #6 — Auto subgroup assignment lives WITH the subgroup controls (steps 8/9), not in the global top-right.
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
      {busyAction === "align" ? "Aligning…" : "⌖ Align scar replicates"}
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
  // STEP 10 scar-source decision: which scar boundary becomes each replicate's TRAINING label.
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
  // STEP 7: confirm this scan's subgroup (which lesion set it belongs to → which repeats align together).
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

  // INITIAL scar DETECTION (offered at step 8 "Subgroup"): pick a strategy + run a detector → produces the scar.
  // The scar CORRECTION tools (guide-hints, Correct ✎) live in step 9 once an initial scar exists. Non-control only.
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

  // Scar REFINEMENT (correction): click-hint touch-up of an EXISTING scar. Lives in step 9. Non-control only.
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

  // Full scar controls = detect + refine, shown together in the Scar-correction step (9) so you can iterate.
  const ScarReRun = classification !== "control" ? <>{ScarDetect}{ScarRefine}</> : null;

  // Move to the next scan awaiting approval. Called AFTER the approve/reject write has landed, so the scan
  // just handled has already left the queue and `nextAfter` falls through to the first remaining entry.
  const advance = async () => {
    // The FULL queue, so nextAfter can find the current scan's position and step forward from it. Passing a
    // pre-filtered queue made indexOf return -1 and every advance jumped back to the top of the list.
    let next = nextAfter(reviewQueue, activeCaseId, queueSettled);
    // FALLBACK: ask the SERVER which scans await approval. The published queue is derived from OctLoader's
    // in-memory list, and when that is empty or stale — a reload, a filter change, a lifecycle re-hydrate that
    // has not landed — the loop reported "no scans left" while 138 genuinely awaited review, stranding the
    // reviewer on the scan they had just judged. The queue is an optimisation; the backlog is a fact, so fall
    // back to the fact rather than stopping.
    if (!next) {
      try {
        const r = await api.json<{ cases: Array<{ case_id: string; life?: Record<string, unknown> }> }>(
          "/api/cases/list");
        const awaiting = (r.cases || []).filter((c) => {
          const l = (c.life ?? {}) as Record<string, unknown>;
          // NOT filtered on difficult_scan: a rejected scan is still awaiting approval — the reviewer's own
          // backlog is 138 scans of which all 138 are flagged difficult, so excluding them matched nothing and
          // the loop reported an empty queue. Only an APPROVED scan leaves the backlog.
          return Boolean(l.oct_preprocessed) && !l.preproc_vetted;
        }).map((c) => c.case_id);
        next = awaiting.find((id) => id !== activeCaseId && !queueSettled.has(id)) ?? null;
      } catch {
        /* fall through to the note below */
      }
    }
    if (!next || !openCaseInSidebar) {
      setQueueNote(next ? "Cannot open the next scan — the scan list is still loading." : "No scans left awaiting approval.");
      return;
    }
    setQueueNote(null);
    setNavigating(true);
    try {
      // BOUNDED. Opening a scan can be slow the first time (the surface-crop detection is computed on open and
      // is not cached yet), and without a bound a slow open leaves the review loop looking wedged with no way
      // out. On timeout the verdict is already saved — only the navigation failed — so say exactly that.
      await Promise.race([
        Promise.resolve(openCaseInSidebar(next)),
        new Promise((_r, rej) => setTimeout(() => rej(new Error("timeout")), 60000)),
      ]);
    } catch {
      setQueueNote("Your verdict was saved, but the next scan is slow to open — click it in the list.");
    } finally { setNavigating(false); }
  };

  // Wait for the write, but not forever. The store applies every one of these optimistically, so the UI is
  // already correct when the click lands; the await is only to catch an outright failure. `fetch` here has no
  // timeout, and a POST can sit queued behind a long-running request on the same sidecar (opening a
  // surface-crop scan computes its crop detection, which takes 25 s+ uncached) — which left the button reading
  // "Rejecting…" long after the rejection had reached disk. The request is never aborted: that could drop a
  // verdict. It simply finishes in the background.
  const settleWrite = async (write: Promise<unknown>, what: string) => {
    let slow = false;
    const timer = setTimeout(() => {
      slow = true;
      setQueueNote(`${what} saved — the sidecar is busy, so it is still confirming in the background.`);
    }, 6000);
    try {
      await Promise.race([write, new Promise((r) => setTimeout(r, 6000))]);
    } finally {
      clearTimeout(timer);
      if (!slow) setQueueNote(null);
    }
    void write.catch(() => setQueueNote(`${what} may NOT have saved — check the scan before moving on.`));
  };

  const approveAndNext = async () => {
    setBusyAction("approve");
    try {
      if (activeCaseId) markSettled(activeCaseId, false);
      await settleWrite(approvePreprocessing(corpusEligible), "Approval");
    } catch (e) {
      setQueueNote(`Approval may not have saved (${e instanceof Error ? e.message : String(e)}) — moving on anyway; re-check this scan.`);
    } finally { setBusyAction(null); }
    await advance();
  };

  // FOLD THE CORRECTED-PANE EDITS INTO THE ORIGINAL EDGE, then re-run from the improved GT.
  // Reviewer directive 2026-09-01: "corrections on the corrected image are always to be treated as more
  // accurate and therefore should inform an appropriate change in the edge of the original such that an
  // improved corrected image is resultant." Where the corrected-pane line disagrees with an earlier raw
  // anchor at the same (lateral, frame), the corrected one WINS and replaces it.
  // This is the only action that rewrites the reviewer's own raw ground truth, so it is deliberately a
  // SEPARATE button (never the default re-run) and the backend snapshots the pre-fold anchors first.
  // STEP 3 of the reviewer's loop: take every verified slice on the corrected scan (drawn or marked) and
  // regenerate the corrected scan from them, then hand back the new one with the verifications cleared.
  const foldToOriginalAndRerun = async () => {
    setBusyAction("rerun");
    try {
      const ce = activeCaseId ? takeCorrectedEdge(activeCaseId) : null;
      if (ce) await commitCorrectedEdgeAnchors(ce);       // persist anything still dirty, then fold ALL of it
      const ok = await rerunWithCorrections({ foldToOriginal: true,
                                              trustedLaterals: trustedForCase.length ? trustedForCase : undefined });
      if (!ok) setQueueNote("Fold + re-run failed — your corrected-edge drawing is saved; try again or Skip.");
      else setQueueNote(null);
    } catch (e) {
      setQueueNote(`Fold problem (${e instanceof Error ? e.message : String(e)}) — the drawing is saved.`);
    } finally { setBusyAction(null); }
  };

  // ITERATE ON THIS SCAN. Commit whatever is drawn, re-run the scan against it, and STAY — so the reviewer
  // can look at the result and correct again. This is the loop that actually converges per scan: unlike the
  // detector-parameter search, an anchor is not a hint the detector may decline, it defines the surface.
  const correctAndRerun = async () => {
    setBusyAction("rerun");
    try {
      const edit = activeCaseId ? takePendingEdit(activeCaseId) : null;
      // Did the reviewer actually CHANGE the raw line/marks this session? nPoints alone counts the re-seeded
      // persisted set, so it can't tell "changed" from "just loaded". Only a real raw change reshapes the
      // volume and needs the full re-run.
      const rawChanged = !!edit && (edit.bordersDirty || edit.cropFrames !== null || edit.cropRegion !== null
        || edit.defectCols !== null || edit.postAnchors !== null);
      if (rawChanged) {
        // Same commit path the verdict buttons use, so a correction recorded here is byte-identical to one
        // recorded by a rejection — there is no second, weaker kind of correction.
        if (edit!.bordersDirty && edit!.nPoints > 0) await commitBorderAnchors(edit!.anchors, edit!.parabola, edit!.parabolaSlices);
        if (edit!.cropFrames !== null || edit!.cropRegion !== null || edit!.postAnchors) {
          await commitOctMarks(edit!.cropFrames, edit!.cropRegion, edit!.postAnchors);
        }
      }
      // CORRECTED-result edge edits: persist the reviewer's drawn curves as corrected_edge_anchors. In the
      // smooth-align workflow the pipeline CONSUMES them as EDITED trusted curves (propagated across the volume),
      // so committing them here is what feeds the smooth-align its ground truth.
      const ce = activeCaseId ? takeCorrectedEdge(activeCaseId) : null;
      if (ce) await commitCorrectedEdgeAnchors(ce);
      // In Corrected mode, the re-run is "Smooth to trusted slices": propagate the edited (drawn, just committed)
      // + approved slices across the volume. Only a raw change diverts to a plain raw re-run.
      // Always a plain corrections re-run now. The smooth-align branch that used to live here is gone with
      // its button (2026-09-01); corrected-pane edits go through the fold instead, which is the reviewer's
      // stated model — the corrected line is the more accurate observation, so it belongs in the ORIGINAL GT.
      const ok = await rerunWithCorrections(undefined);
      if (!ok) setQueueNote("Re-run failed — your correction is saved; try again or Skip.");
      else setQueueNote(null);
    } catch (e) {
      setQueueNote(`Re-run problem (${e instanceof Error ? e.message : String(e)}) — the correction is saved.`);
    } finally { setBusyAction(null); }
  };

  // MOVE ON WITHOUT A VERDICT. Not every scan can be fixed, and forcing a reviewer to either approve
  // something they do not believe or reject something they may come back to is a false choice. Skipping
  // settles it for THIS session only — nothing is written, so the scan is still awaiting approval tomorrow.
  const skipToNext = async () => {
    setBusyAction("skip");
    try {
      if (activeCaseId) markSettled(activeCaseId, false);
      setQueueNote(null);
    } finally { setBusyAction(null); }
    await advance();
  };

  // SAVE WHAT WAS DRAWN before a verdict that leaves the scan (Difficult or Reject): a border the reviewer drew
  // is ground truth about where the cornea is, whatever the verdict. Returns the derived summary note.
  const savePendingForVerdict = async (): Promise<string> => {
    let note = "";
    {
      const edit = activeCaseId ? takePendingEdit(activeCaseId) : null;
      const ce = activeCaseId ? takeCorrectedEdge(activeCaseId) : null;
      if (edit || ce) {
        const parts: string[] = [];
        try {
          if (ce) {
            await commitCorrectedEdgeAnchors(ce);
            const n = Object.values(ce).reduce((a, m) => a + Object.keys(m).length, 0);
            if (n) parts.push(`corrected-edge: ${n} point(s)`);
          }
          if (edit && edit.nPoints > 0) {
            await commitBorderAnchors(edit.anchors, edit.parabola, edit.parabolaSlices);
            parts.push(`border corrected: ${edit.nPoints} point(s) on ${edit.nSlices} slice(s)`);
          }
          if (edit && (edit.cropFrames !== null || edit.cropRegion !== null || edit.postAnchors)) {
            await commitOctMarks(edit.cropFrames, edit.cropRegion, edit.postAnchors);
            if (edit.cropFrames !== null) parts.push(`${edit.cropFrames.length} surface-crop frame(s)`);
            if (edit.cropRegion !== null) parts.push(`crop region ${edit.cropRegion.frames.length} col(s)`);
            if (edit.postAnchors) {
              const n = Object.values(edit.postAnchors).reduce((a, m) => a + Object.keys(m).length, 0);
              parts.push(`${n} bottom-edge point(s)`);
            }
          }
          if (edit && edit.defectCols) {
            // Replace this slice's marks with what is currently drawn, leaving every OTHER slice alone —
            // the store takes the whole list, so a naive write would wipe marks made on other slices.
            const existing = (((caseInfo?.manifest as Record<string, unknown> | undefined)?.defect_marks) ?? []) as
              DefectMark[];
            const others = existing.filter((k) => !(k.orient === "sagittal" && k.slice === edit.defectCols!.slice));
            const next = edit.defectCols.cols.length
              ? [...others, { orient: "sagittal" as const, slice: edit.defectCols.slice,
                              cols: edit.defectCols.cols, tag: "reviewer" }]
              : others;
            await setDefectMarks(next);
            parts.push(`${edit.defectCols.cols.length} marked column(s)`);
          }
          note = parts.join("; ");
        } catch {
          note = "correction FAILED to save — re-open the scan and retry";
        }
      }
    }
    return note;
  };

  const rejectAndNext = async () => {
    setBusyAction("reject");
    try {
      if (activeCaseId) markSettled(activeCaseId, true);
      // COMMIT THE CORRECTION FIRST. A border the reviewer drew is ground truth about where the cornea is,
      // and rejecting is how they say "this scan is wrong — here is what it should have been". Requiring a
      // separate "Confirm border" click before that counted meant a correction drawn and then rejected was
      // silently discarded, which is the reverse of what the gesture means. The write must land BEFORE the
      // difficult flag, because the backend harvests border_gt from the persisted anchors at that moment.
      const note = await savePendingForVerdict();
      // The reason is DERIVED, not typed: with a correction on screen the correction is the signal, and a
      // sentence of prose adds nothing the anchors do not already say. A free-text note is still accepted by
      // the endpoint for callers that want one.
      // A TYPED note always wins; the derived summary is only the fallback so a rejection is never recorded
      // blank. Typing stays optional — the box does not gate the button.
      await settleWrite(setDifficult(true, rejectReason.trim() || note), "Rejection");
      setRejecting(false);
      setRejectReason("");
    } catch (e) {
      // A failed write must not strand the reviewer on the scan they just judged. settleWrite re-throws if
      // the request rejects inside its 6 s race, and that propagated past the advance below — so one slow or
      // failed save silently stopped the loop dead, which is indistinguishable from the button not working.
      setQueueNote(`Rejection may not have saved (${e instanceof Error ? e.message : String(e)}) — moving on anyway; re-check this scan.`);
    } finally { setBusyAction(null); }
    await advance();
  };

  // ✗ REJECT → NEXT (reviewer, 2026-09-10): "serve up scans that have been fixed and if I determine unfixable I will
  // reject". Rejecting is final for the queue: the scan leaves "awaiting approval" and is never re-served after a
  // fix (manifest.rejected_unfixable). Difficult stays the softer verdict for scans that may come back.
  const rejectUnfixableAndNext = async () => {
    setBusyAction("rejectfinal");
    try {
      if (activeCaseId) markSettled(activeCaseId, true);
      const note = await savePendingForVerdict();
      await settleWrite(setRejected(true, rejectReason.trim() || note), "Rejection");
      setRejectReason("");
    } catch (e) {
      setQueueNote(`Rejection may not have saved (${e instanceof Error ? e.message : String(e)}) — moving on anyway; re-check this scan.`);
    } finally { setBusyAction(null); }
    await advance();
  };

  // CHECKPOINT after N REJECTIONS — N chosen by the reviewer, not baked in. The whole point of the loop is
  // that corrections accumulate until there are enough of them to be worth learning from, and only the
  // reviewer knows when that is: a handful of corrections on one bad eye says less than the same number
  // spread across five. Counted on rejections alone (see markSettled) — an approval carries no correction.
  const atCheckpoint = rejectsSinceCheckpoint >= checkpointEvery;
  const Checkpoint = atCheckpoint ? (
    <span className="flex items-center gap-2 text-xs" style={{ color: "var(--c-text)" }}>
      <span style={{ color: "var(--c-amber, #d9a441)" }}>
        {rejectsSinceCheckpoint} correction{rejectsSinceCheckpoint === 1 ? "" : "s"} banked
      </span>
      <Button size="small" variant="contained" color="warning" disabled={busy}
        onClick={() => { clearCheckpoint(); void startReprocessBatch(); }}
        title={"Re-run preprocessing on the scans you corrected, using those corrections.\n"
          + "Runs in the background (~2 min per scan). Approved scans are protected: a scan is only re-written\n"
          + "if it measures BETTER on both roughness and per-frame undulation, otherwise it keeps what it has.\n"
          + "Whatever changes comes back flagged for a second look.\n\n"
          + "This applies each scan's OWN corrections to that scan. It does not change the detector."}>
        ⟳ Re-run corrected scans
      </Button>
      {/* THE CONVERGENCE STEP. The other button fixes the scans you corrected; this one changes the detector
          so the scans nobody has looked at get better too — which is the only way the queue can empty. */}
      <Button size="small" variant="contained" color="primary" disabled={busy || tuneBusy}
        onClick={() => { clearCheckpoint(); void startDetectorTune(); setTuneBusy(true); }}
        title={"Use every correction you have drawn to improve the DETECTOR ITSELF, globally.\n\n"
          + "Searches the detector's parameters for a setting that reproduces your corrections better, scored\n"
          + "with a tolerance band (your corrections are approximate by design, so it is not fitted to the pixel).\n"
          + "Adopts it ONLY if it also leaves the scans you already approved undisturbed — otherwise nothing\n"
          + "changes and it says so.\n\n"
          + "Runs in the background for a while; approved scans keep the exact settings they were approved under."}>
        ⟲ Improve detector
      </Button>
      <Button size="small" variant="outlined" disabled={busy}
        onClick={() => clearCheckpoint()}
        title={`Keep reviewing; you will be asked again after another ${checkpointEvery} corrections.`}>
        Keep reviewing
      </Button>
    </span>
  ) : null;

  // HOW FAR THROUGH THE JOB. The queue count on the Approve button answers "what is left in front of me",
  // which is not the same question — it moves with the sidebar filter and says nothing about the 300-odd
  // scans behind. This is the standing total, so a session's work is visible as it accumulates.
  // A REJECTION IS NOT PROGRESS HERE, and the tooltip says so: rejecting leaves a scan unvetted by design
  // (it needs reprocessing before it can be approved), so it stays outstanding and will come back. Counting
  // it as done would make the ticker overstate how much of the corpus is actually settled.
  const vettedPct = nVettable > 0 ? Math.round((nVetted / nVettable) * 100) : 0;
  const nThisSession = queueSettled.size;
  const VettedTicker = nVettable > 0 ? (
    <span className="flex items-center gap-1.5 text-[11px]" style={{ color: "var(--c-text-dim)" }}
      title={`${nVetted} of ${nVettable} preprocessed scans approved (${vettedPct}%). ${nRejected} rejected.`
        + (nThisSession ? ` ${nThisSession} settled in this session.` : "")
        + " Rejecting does not vet a scan — it stays outstanding until it has been reprocessed, so the two"
        + " counts do not add up to the total."}>
      {/* The bar shows APPROVED only. Rejections are not progress along it — a rejected scan comes back. */}
      <span style={{ width: 54, height: 5, borderRadius: 3, background: "var(--c-surface2)",
                     border: "1px solid var(--c-border)", overflow: "hidden", display: "inline-block" }}>
        <span style={{ display: "block", height: "100%", width: `${vettedPct}%`,
                       background: "var(--c-green)", transition: "width .3s ease" }} />
      </span>
      <span style={{ whiteSpace: "nowrap" }}>
        <b style={{ color: "var(--c-green)" }}>{nVetted}</b>
        <span style={{ opacity: 0.75 }}>/{nVettable} vetted</span>
        {nRejected > 0 && <b style={{ color: "var(--c-red, #e5534b)", marginLeft: 5 }}>✗{nRejected}</b>}
        {nThisSession > 0 && <b style={{ color: "var(--c-accent)", marginLeft: 5 }}>+{nThisSession} now</b>}
      </span>
      {/* Corrections banked toward the next tuning offer, and the threshold that triggers it. Editable here
          rather than buried in settings: it is the one number that paces the whole loop. */}
      <span style={{ whiteSpace: "nowrap", opacity: 0.85 }} title="Corrections banked / how many to bank before offering to learn from them.">
        <b style={{ color: rejectsSinceCheckpoint >= checkpointEvery ? "var(--c-amber, #d9a441)" : "var(--c-text)" }}>
          {rejectsSinceCheckpoint}
        </b>
        <span style={{ opacity: 0.6 }}>/</span>
        <input type="number" min={1} max={200} value={checkpointEvery}
          onChange={(e) => { const v = Math.round(Number(e.target.value)); if (Number.isFinite(v) && v >= 1) setCheckpointEvery(Math.min(200, v)); }}
          style={{ width: 34, fontSize: 11, marginLeft: 1, padding: "0 2px", color: "var(--c-text)",
                   background: "var(--c-surface2)", border: "1px solid var(--c-border)", borderRadius: 3 }} />
      </span>
    </span>
  ) : null;

  // The two review-loop buttons + the reason box, shared by every step that offers approval.
  const ReviewLoop = (
    <span className="flex items-center gap-1">
      {VettedTicker}
      <Button size="small" variant="contained" color="success" disabled={busy || rejecting || navigating} sx={ACT_SX}
        onClick={() => void approveAndNext()}
        startIcon={busyAction === "approve" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
        title={`Approve this scan's preprocessing and open the next one awaiting approval (${nLeft} in the queue).`}>
        {busyAction === "approve" ? "Approving…"
          : navigating ? "Opening next…"
          : `✓ Approve → next${nLeft ? ` (${nLeft})` : ""}`}
      </Button>
      {/* ONE CLICK, with an OPTIONAL note. The box is always visible rather than a mode you have to enter:
          typing is never required (a drawn correction already says more than a sentence could), but when the
          problem is something the anchors cannot express — "wrong curvature", "eyelash across the apex" — it
          is right there without an extra click. Enter submits, so a typed rejection is still one gesture. */}
      <input
        value={rejectReason}
        onChange={(e) => setRejectReason(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void rejectAndNext(); } }}
        placeholder="optional note…"
        disabled={busy || navigating}
        style={{ fontSize: 11, width: 210, color: "var(--c-text)", background: "var(--c-surface2)",
                 border: "1px solid var(--c-border)", borderRadius: 4, padding: "3px 6px" }}
      />
      {/* ITERATE — the scan stays on screen. Primary action whenever something is drawn: a correction is
          worth more applied to this scan than filed against it.
          NOT rendered in corrected mode with no raw change (`smoothAlignReady`), because there this button WAS
          "Smooth to N trusted slices & re-run" — the post-hoc align_corrected_to_smooth warp. Removed on the
          reviewer's instruction 2026-09-01: it declined on every recorded real-scan run, and its decline test
          compares a CONSTRAINED-line off-quadratic against a FREE-detection one (1.19 vs 16.41 px on the same
          unmoved volume), so its verdicts carry no information. The engine and its request flag are untouched —
          only the way to invoke it from here is gone. Restoring it should wait until that guard measures both
          sides with the same line and csa_tilt_min_laterals goes 2 → 8.
          Corrected mode keeps: ⤴ fold-to-original (which also pins approved/trusted laterals), Approve, Skip,
          Difficult, Re-preprocess. A pending RAW change still shows this button as "Correct & re-run". */}
      {!smoothAlignReady && (
      <Button size="small" variant="contained" color="warning" disabled={busy || navigating} sx={ACT_SX}
        onClick={() => {
          // GUARD THE FREE PATH. On a scan with NO border_anchors the smooth-align stage is called without a
          // constrained surface, so an APPROVED lateral's target becomes a completely unconstrained detection —
          // measured a median 5.4 px and up to 255 px from the reviewer's own drawn line. Across 30 held-out
          // trials on that path it fired 28 times and was worse on 23-25 of 26 slices, pushed the surface
          // 16-55 px RMS off the drawn line, took zero-voxels 6.30% -> 9.72% and cost 3.2% of signal energy.
          // 300 of the 308 scans in the store are in exactly that state, so this is the common case, not an
          // edge case. Confirm rather than disable: the reviewer may still want it, but not by accident.
          if (smoothAlignReady && !hasBorderCorrection && !window.confirm(
            "This scan has no drawn border corrections, so \"trusted\" slices are judged against the automatic "
            + "detector rather than your own lines.\n\nMeasured on scans in this state, the move it makes is "
            + "wrong far more often than it is right — it typically pushes the surface further from a hand-drawn "
            + "edge and empties voxels at the volume ends.\n\nDraw a border correction first, or use the "
            + "corrected-edge fix-tool on the specific slice.\n\nRun it anyway?")) return;
          void correctAndRerun();
        }}
        startIcon={busyAction === "rerun" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
        title={smoothAlignReady && !hasBorderCorrection
          ? "NOT RECOMMENDED on this scan: it has no drawn border corrections, so an approved slice's target is\n"
            + "the automatic detector's own smoothed curve, not your line. Measured on scans in this state the\n"
            + "move is worse on 23-25 of 26 slices and lands 16-55 px from a hand-drawn edge. Draw a border\n"
            + "correction first, or use the corrected-edge fix-tool on the slice you care about."
          : smoothAlignReady
          ? "Smooth to trusted slices (~2 min). Takes the corrected-surface curves you EDITED (drew) plus the\n"
            + "slices you APPROVED as trusted ground truth, and propagates them across the whole volume — each\n"
            + "B-scan rigidly shifts/rotates so the result follows the curvature YOU defined. Rigid only: the\n"
            + "within-frame arc is untouched. Mark a few good slices / draw a few, then run."
          : "Apply what you drew to THIS scan and re-run it (~2 min), then stay here so you can look at the\n"
          + "result and correct again. Repeat until you are happy, then Approve.\n\n"
          + "The correction defines the surface — it is not a hint the detector can decline — so this converges\n"
          + "on the scan in front of you, with no dependence on the detector being well-tuned."}>
        {busyAction === "rerun" ? "Re-running…"
          : rawDirty ? `↻ Correct & re-run (${pendingSummary})`
          : editTarget === "corrected"
              ? `↻ Smooth to ${trustedCount} trusted slice${trustedCount === 1 ? "" : "s"} & re-run`
          : "↻ Re-run with corrections"}
      </Button>
      )}
      {/* FOLD INTO THE ORIGINAL EDGE. Separate from the primary re-run because it changes the reviewer's raw GT
          rather than the delivered volume: the corrected-pane line is taken as the more accurate observation and
          REPLACES the raw anchor at the same (lateral, frame). Corrected mode only, and only when there is
          something to fold. Green — it is a "promote my better observation" action, not a destructive one from
          the reviewer's point of view, though it does rewrite border_anchors (backed up server-side first). */}
      {editTarget === "corrected" && foldableLats > 0 && (
        <Button size="small" variant="outlined" color="success" disabled={busy || navigating}
          onClick={() => void foldToOriginalAndRerun()}
          startIcon={busyAction === "rerun" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
          title={(rawDirty ? "NOTE: you have unconfirmed edits on the ORIGINAL pane — they are not part of this regenerate. Confirm them first if you want them in.\n\n" : "")
            + "REGENERATE: write the lines you confirmed on this corrected scan into the ground truth, then re-run (~2 min).\n\n"
            + "What goes in:\n"
            + "  \u2022 slices you DREW on — those depths become the original scan's edge line, replacing the earlier\n"
            + "    raw anchor wherever they disagree;\n"
            + "  \u2022 slices you MARKED accurate — their line is pinned frame-by-frame as it is.\n\n"
            + "What this changes: the EDGE LINE drawn on both panes and the exported ground truth.\n"
            + "What it does NOT change: the tissue. The per-frame move is measured from the tissue itself\n"
            + "(adjacent-frame correlation of the whole B-scan), not from your lines — a move fitted from hand-drawn\n"
            + "lines measured worse than none (cs002, 2026-09-05). So drawing or marking here cannot make the\n"
            + "corrected scan smoother; it makes the line right where you have checked it.\n\n"
            + "Your verifications are CLEARED afterwards: they described the previous corrected scan, and the next\n"
            + "round starts on the one just produced. Their readings are kept in oct_iter.corrected_fold, and the\n"
            + "pre-run anchors are snapshotted to cases/<id>/fold_backup/prefold_<ts>.json."}
          sx={ACT_SX}>
          {busyAction === "rerun" ? "Regenerating…"
            : `\u2934 Confirm ${foldableLats} slice${foldableLats === 1 ? "" : "s"} into the ground truth & re-run`}
        </Button>
      )}
      {/* The "why the smooth-align did nothing" line went with its button (2026-09-01). It explained declines of
          an action that can no longer be invoked here; the outcome is still recorded in
          manifest.oct_iter.corrected_smooth_align for anyone reading the case. */}
      {/* MOVE ON without judging. Session-only: nothing is written, so it returns to the queue next time. */}
      <Button size="small" variant="outlined" disabled={busy || navigating} sx={ACT_SX}
        onClick={() => void skipToNext()}
        title={"Leave this scan unjudged and open the next one. Nothing is written — it stays in the queue\n"
          + "and will come back. Use it when a scan needs thought, or cannot be fixed right now."}>
        {busyAction === "skip" ? "Skipping…" : "⤼ Skip → next"}
      </Button>
      {/* Flag as difficult + advance. Now the LAST resort rather than the way to record a correction — the
          correction path is the re-run above, which acts on the scan instead of filing a complaint. */}
      <Button size="small" variant="outlined" color="error" disabled={busy || navigating} sx={ACT_SX}
        onClick={() => void rejectAndNext()}
        startIcon={busyAction === "reject" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
        title={pendingEdit
          ? `Save what you drew, flag this scan as difficult, and open the next one: ${pendingSummary}.`
          : "Flag this scan as difficult (excluded from training) and open the next one. The note is optional."}>
        {busyAction === "reject" ? "Rejecting…" : "✗ Difficult → next"}
      </Button>
      {/* REJECT = UNFIXABLE (reviewer, 2026-09-10). Leaves the approval queue for good; corrections drawn are still
          saved as ground truth. Difficult (left) is the softer verdict for scans that may come back. */}
      <Button size="small" variant="contained" color="error" disabled={busy || navigating} sx={ACT_SX}
        onClick={() => void rejectUnfixableAndNext()}
        startIcon={busyAction === "rejectfinal" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
        title={pendingEdit
          ? `Save what you drew, REJECT this scan as unfixable (it leaves the queue for good), and open the next one: ${pendingSummary}.`
          : "REJECT this scan as unfixable — it leaves the approval queue for good — and open the next one. The note is optional."}>
        {busyAction === "rejectfinal" ? "Rejecting…" : "✗ Reject → next"}
      </Button>
      {Checkpoint}
      {tune && (tune.running || tune.note) && (
        <span className="text-[11px] flex items-center gap-1"
              style={{ color: tune.running ? "var(--c-accent)" : (tune.adopted ? "var(--c-green)" : "var(--c-text-dim)") }}
              title={tune.note || ""}>
          {tune.running && <CircularProgress size={11} color="inherit" />}
          {tune.running
            ? `detector: ${tune.phase ?? "working"}${tune.total ? ` ${tune.done ?? 0}/${tune.total}` : ""}`
            : (tune.adopted
                ? `detector improved — ${Object.keys(tune.adopted).length} setting(s) adopted`
                : `detector: ${tune.note}`)}
        </span>
      )}
      {queueNote && <span className="text-[11px]" style={{ color: "var(--c-amber, #d9a441)" }}>{queueNote}</span>}
      {/* Stale-pipeline auto re-run (openCase): the scan opened with a result from an older pipeline version and
          is being re-run through the current one — the same call as ↻ Re-run with corrections. While it runs the
          global status line (right) carries the same text; this one stays after it finishes with the outcome. */}
      {autoRerunNote && (
        <span className="text-[11px]" data-testid="auto-rerun-note" title={autoRerunNote}
              style={{ color: "var(--c-amber, #d9a441)", maxWidth: 420, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {autoRerunNote}
        </span>
      )}
    </span>
  );

  const sep = <span style={{ width: 1, height: 22, background: "var(--c-border)" }} />;

  // Auto-populated scans reach Cornea (SAM2) WITHOUT a human approving the preprocessing (preproc_vetted unset,
  // e.g. the batch populate). Offer a NON-destructive approve at the segmentation steps so the Vetted step can be
  // filled without a rollback (which would clear SAM2). Null once vetted, so it never shows for the normal flow.
  const ApprovePreproc = !manifest?.preproc_vetted ? (
    <Button size="small" variant="outlined" color="warning" disabled={busy}
      onClick={() => { setBusyAction("approve"); approvePreprocessing(corpusEligible); }}
      startIcon={busyAction === "approve" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
      title="Mark the preprocessing as manually vetted (fills the Vetted step). Non-destructive — keeps the SAM2 segmentation. Does NOT apply any auto-detected corrections. Shown because this scan was segmented without an explicit preprocessing approval.">
      {busyAction === "approve" && caseBusy ? "Approving…" : "✓ Approve preprocessing"}
    </Button>
  ) : null;

  // Marking is now CONSOLIDATED into Fix-columns: the user corrects the border there (the correction IS the
  // "mark", and on Approve it becomes ground truth), so the separate ⚑ Mark-defect toggle is retired. Only the
  // "⚠ Difficult" flag (this scan needs manual help / can't be fixed) remains.
  // Surface-crop (clipped cornea): AUTO = the pipeline took the surface-crop path; MANUAL = human review override
  // (true/false). The effective state = manual if the human set it, else the auto detection.
  // FlagButtons (⬚ Surface-crop / ⚠ Difficult) REMOVED.
  //   ⚠ Difficult set exactly the flag "✗ Reject → next" now sets — but without saving the corrections,
  //     recording a note, or advancing the queue. Two buttons for one verdict, one of them worse.
  //   ⬚ Surface-crop was a BOOLEAN classification living next to the editor's ✛ Surface crop mode, which owns
  //     the actual frame set. Same words, different data, two sources of truth. It is now an INDICATOR on
  //     that mode button (a ✓ when the pipeline surface-cropped this scan), and the frames are cleared in the
  //     editor where they are drawn.
  // GT toggle shown next to Approve when the scan has a manual border correction (Fix-columns anchors).
  const CorpusToggle = hasBorderCorrection ? (
    <label className="flex items-center gap-1 text-[11px]" style={{ color: "var(--c-text-dim)", cursor: "pointer" }}
      title="This scan has a manual border correction. Approving records it as GROUND TRUTH that trains/validates the auto-detector. Uncheck to keep the correction applied but EXCLUDE an idealised case (e.g. a motion-corrupted scan whose hand-drawn border is not real geometry) from the training corpus.">
      <input type="checkbox" checked={corpusEligible} onChange={(e) => setCorpusEligible(e.target.checked)} disabled={busy} />
      🎯 use correction as training ground truth
    </label>
  ) : null;

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
            : "Review + correct in Fix-columns (the correction becomes ground truth on Approve), then:"}
        </span>
        {CorpusToggle}
        {/* The review LOOP lives here: approve (or reject with a reason) and the next scan awaiting approval
            opens straight away. Working through the backlog is the dominant activity at this step, and the
            old single "✓ Approve preprocessing" left the reviewer to find the next row themselves. */}
        {ReviewLoop}
        {/* FULL reset to automatic preprocessing from the raw .OCT (reviewer 2026-09-04): discards EVERY manual
            correction and mark and re-runs fresh surface detect + surface-crop detect + warp. It used to keep the
            sticky crop / surface-crop marks and drop only border corrections, so a scan could never be brought back
            to a pure auto state from here. Always confirms (destructive of all edits). */}
        <Button size="small" variant="outlined" color="warning" disabled={busy} sx={ACT_SX}
          onClick={() => {
            if (armed !== "reprocess") { arm("reprocess"); return; }   // always confirm: it discards everything
            setArmed(null);
            setBusyAction("rerun"); void rerunPreprocess();
          }}
          startIcon={busyAction === "rerun" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
          title="Reset this scan to AUTOMATIC preprocessing from the raw .OCT: discards EVERY manual correction and mark (border / corrected-edge / axial anchors, bottom lines, artifact + surface crops, marks, force/good columns, manual patch/shifts, edit transform) and re-runs fresh surface detection, surface-crop detection and warp. Same as ⟲ Clear all corrections. Confirms on a second click.">
          {busyAction === "rerun" && caseBusy ? "Re-preprocessing…"
            : armed === "reprocess" ? "↻ Resets EVERYTHING to auto — click again" : "↻ Re-preprocess"}
        </Button>
        {/* FULL reset — the superset of Re-preprocess. Discards EVERY manual correction (border + corrected-edge +
            axial anchors, artifact/surface crops, marks, force/good columns, manual patch/shifts) and re-runs pure
            AUTO. For starting a scan over from scratch. Always confirms (it is destructive of all edits). */}
        <Button size="small" variant="outlined" color="error" disabled={busy} sx={ACT_SX}
          onClick={() => {
            if (armed !== "clearall") { arm("clearall"); return; }
            setArmed(null);
            setBusyAction("clearall"); void clearAllCorrections();
          }}
          startIcon={busyAction === "clearall" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
          title="Discard ALL manual corrections on this scan (border / corrected-edge / axial anchors, artifact + surface crops, marks, approved slices, force/good columns, manual patch + shifts) and re-run pure AUTO detection from the raw .OCT — a clean slate. KEEPS detection settings, classification and training GT. Resets to Preprocessed — re-inspect, then Approve.">
          {busyAction === "clearall" && caseBusy ? "Clearing…"
            : armed === "clearall" ? "⟲ Discards EVERYTHING — click again" : "⟲ Clear all corrections"}
        </Button>
        {proposals.hasProposal && (
          <Button size="small" variant="outlined" color="secondary" disabled={busy}
            onClick={() => { setBusyAction("apply"); applyCorrections(); }}
            startIcon={busyAction === "apply" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
            title="Bake in the auto-detected de-tilt / crop / surface-crop (shown in pink) and re-warp from the raw .OCT. Produces a fresh output to re-inspect (resets to Preprocessed · Auto); drops any segmentation. Approve it after reviewing.">
            {busyAction === "apply" && caseBusy ? "Applying…" : "⟳ Apply corrections"}
          </Button>
        )}
        {/* "↻ Re-run preprocessing" (full auto re-run) removed from the correction workflow: it DISCARDS the
            user's fix-column border corrections (the auto re-run pops border_anchors), so it was a footgun
            here. Re-run via Fix-columns → Run (which keeps corrections), or "Use original (raw)" below. */}
        <Button size="small" variant="outlined" color="warning" disabled={busy} onClick={() => { setBusyAction("useraw"); approveRaw(); }}
          startIcon={busyAction === "useraw" ? <CircularProgress size={13} color="inherit" /> : undefined}
          title="Use the ORIGINAL (raw) scan as the working volume instead of the correction — for when the correction is worse than doing nothing. Drops any segmentation. Does NOT approve the scan: it returns to Preprocessed and still needs your Approve.">
          {busyAction === "useraw" ? "Loading…" : "↩ Use original (raw)"}
        </Button>
              </>
    );
  } else if (step === 3) {
    // vetted (pink) → ALIGN the patient+eye GROUP: register this eye's replicate scans in 3D and regularise their
    // sagittal curvature (the engine is a separate workflow — until it lands the backend stub only reports the
    // group's members, and nothing is written). SAM2 comes NEXT, at Aligned (4).
    actions = (
      <div className="flex items-center gap-2 text-xs" style={{ color: "var(--c-text-dim)" }}>
        <Button size="small" variant="contained" color="primary" disabled={busy} onClick={openSubgroups} data-testid="align-group-btn"
          title="First confirm whether all scans of this eye are one subgroup (or specify subgroups per scan), then each subgroup is aligned as its own group (every scan vs the reference; a few minutes per pair). When the job finishes its scans advance to 4. Aligned.">
          ⧉ Align group
        </Button>
        {alignNote && (
          <span className="text-[11px]" style={{ color: "var(--c-text-dim)", maxWidth: 420, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
            title={alignNote}>{alignNote}</span>
        )}
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
      </div>
    );
  } else if (step === 4) {
    // group-aligned (pink-red). Reviewer spec 2026-09-11 #4 / 2026-09-12 #2: "✓ Approve axial changes" = the reviewer
    // approves that the axial changes applied to each replicate meet the consensus (recorded on every scan of the
    // subgroup). Approving MOVES the subgroup's scans to 5. Cornea, where the cornea-detection button lives (its own
    // action bar below). "Rectify" is a placeholder (function not yet specified). The pane in the main area shows the
    // alignment (consensus + pairs / 3-D view / scrub); the separate pairs dialog was retired (redundant with the pane).
    const RectifyBtn = (
      <span title="Rectify — not yet specified (placeholder for correcting an alignment that does not meet the consensus)." data-testid="aligned-rectify-wrap">
        <Button size="small" variant="outlined" color="warning" disabled sx={ACT_SX} data-testid="aligned-rectify">Rectify</Button>
      </span>
    );
    actions = (
      <div className="flex items-center gap-2 text-xs" style={{ color: "var(--c-text-dim)" }}>
        <span>Review the consensus + pairs, 3-D view and scrub (pane →), then:</span>
        <Button size="small" variant="contained" color="success" disabled={busy || !manifest?.group_aligned} sx={ACT_SX} data-testid="aligned-approve"
          startIcon={busyAction === "alignapprove" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
          onClick={() => { setBusyAction("alignapprove"); void approveAligned(true).then(() => setBusyAction((b) => (b === "alignapprove" ? null : b))); }}
          title="Approve that the axial changes applied to each replicate of this subgroup meet the consensus. Recorded on EVERY scan of the subgroup (manifest aligned_approved); moves them to 5. Cornea (cornea detection).">
          {busyAction === "alignapprove" ? "Approving…" : "✓ Approve axial changes"}
        </Button>
        {RectifyBtn}
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
      </div>
    );
  } else if (step === 5 && awaitingCorneaDetection(manifest)) {
    // 5. Cornea, NOT yet segmented (fuchsia): the axial-changes approval brought the subgroup here (reviewer 2026-09-12
    // #2) — the pending action is cornea detection (SAM2 today; SAM3 is a later model upgrade — the segmentation itself is
    // unchanged). "undo approval" returns the whole subgroup to 4. Aligned. The alignment pane stays visible (read-only).
    const approved = (manifest?.aligned_approved ?? null) as Record<string, unknown> | null;
    actions = (
      <div className="flex items-center gap-2 text-xs" style={{ color: "var(--c-text-dim)" }}>
        <Button size="small" variant="contained" color="primary" disabled={busy || !!sam2RunningCaseId} sx={ACT_SX} onClick={() => runSam2()} data-testid="cornea-detect-sam"
          title="Run the cornea-vs-background segmentation (SAM2 today; SAM3 is a later model upgrade). Scar/control classification and scar segmentation come in later steps.">
          ▶ Cornea detection (SAM)
        </Button>
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)", whiteSpace: "nowrap" }} data-testid="cornea-detect-caption">runs on the consensus volume: pending</span>
        {sep}
        <span style={{ color: "#4ade80", whiteSpace: "nowrap" }} data-testid="aligned-approved"
          title={`Axial changes approved${approved?.ts ? ` at ${approved.ts}` : ""}${approved?.group ? ` (group ${approved.group})` : ""}${approved?.note ? ` — ${approved.note}` : ""}`}>
          ✓ axial changes approved
        </span>
        <Button size="small" variant="text" color="inherit" disabled={busy} sx={ACT_SX} data-testid="aligned-unapprove"
          startIcon={busyAction === "alignapprove" && caseBusy ? <CircularProgress size={13} color="inherit" /> : undefined}
          onClick={() => { setBusyAction("alignapprove"); void approveAligned(false).then(() => setBusyAction((b) => (b === "alignapprove" ? null : b))); }}
          title="Withdraw the approval for every scan of the subgroup — returns them to 4. Aligned.">
          {busyAction === "alignapprove" ? "Undoing…" : "undo approval"}
        </Button>
      </div>
    );
  } else if (step === 5) {
    // cornea segmented (fuchsia) → VET the cornea/background (paint, scar pen hidden), then confirm → unlocks
    // classification. Scar detection is NOT shown here until cornea/background is confirmed AND the scan is classified.
    // Auto-populated scans also get a non-destructive "Approve preprocessing" here (their Vetted step was skipped).
    actions = <>{ApprovePreproc && <>{ApprovePreproc}{sep}</>}{CorneaVet}</>;
  } else if (step === 6) {
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
              </div>
    );
  } else if (step === 7) {
    // classified (violet) → SUBGROUP step (assigned BEFORE scar so the strategy comparison at the Scar step is
    // per-subgroup). Scar scan: assign which lesion set it belongs to. Control: no lesion subgroup is needed
    // (the control baseline is eye-wide, control_cases() ignores subgroup) → skip straight to "no scar".
    actions = classification === "control" ? (
      <>
        {/* A control (no scar) is READY once its cornea is vetted — the scar/subgroup/align/normalize/correct
            steps (8-12) don't apply (greyed in the strip). Its cornea-only label is the training label + the
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
  } else if (step === 8) {
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
  } else if (step === 9) {
    // scar segmented (rose) → refine/correct the scar, then align this subgroup's replicates.
    actions = (
      <>
        {ScarReRun && <>{ScarReRun}{sep}</>}{Correct}{sep}{AlignBtn}
        {classification !== "control" && (
          <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>(subgroup “{subgroup}”)</span>
        )}
      </>
    );
  } else if (step === 10) {
    // aligned (teal) → choose the TRAINING scar (each replicate's own vs the voted consensus), then normalize
    // against controls or SKIP normalization (use the consensus as-is); Correct to touch up. Schedule/Export
    // are NOT here — they live at the later (corrected/scheduled) steps.
    actions = <>{ScarSource}{sep}{NormalizeBtn}{sep}{SkipNormBtn}{sep}{Correct}</>;
  } else if (step === 11) {
    // normalized (cyan) → correct only; scheduling/export live at the corrected/scheduled steps (12/13).
    actions = <>{Correct}</>;
  } else if (step === 12) {
    // manually corrected (dark blue)
    actions = <>{ScheduleBtn}{sep}{Correct}{ExportBtn}</>;
  } else {
    // step 13 — scheduled (green)
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
            ONLY in the Scar-detection step (8) — where you pick a detector, AFTER subgroup is assigned, so the
            comparison is PER-SUBGROUP. Needs ≥2 cornea-segmented replicates of the eye+subgroup; a control has
            no scar so it's hidden. (Not shown on the scar-aligned consensus / step 10.) */}
        {caseInfo && step === 8 && classification !== "control" && (
          <Button size="small" variant="outlined" color="info" disabled={busy || correcting}
            onClick={() => { setShowCompare(true); compareStrategies(); }}
            title="Run every scar strategy on this eye's replicates and tabulate test–retest reproducibility (pairwise Dice, HD95, volume CV%) — for strategy comparison in the paper. Read-only; doesn't change the scan.">
            ⚖ Compare strategies
          </Button>
        )}
        {/* #6 — "⊞ Auto subgroups" lives in the Subgroup steps (8/9) actions, not here. */}
        {/* Live progress text (SAM2 per-plane %, scar phase, …) next to the spinner — not just an icon. */}
        {(busy || !!sam2RunningCaseId) && status.kind === "working" && (
          <span className="text-xs" style={{ color: "var(--c-text-dim)", maxWidth: 360, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
            title={status.detail}>{status.detail}</span>
        )}
        {(busy || !!sam2RunningCaseId) && <CircularProgress size={16} />}
      </div>

      {/* "⧉ Align group" → subgroups first; when the per-subgroup jobs finish their scans carry group_aligned → re-read
          the open scan's manifest so the timeline advances to 4. Aligned without a reload. */}
      <SubgroupAlignDialog open={subOpen} gid={subGid} onClose={() => setSubOpen(false)}
        onAligned={() => { void refreshCaseInfo(); const wf = useWorkflowStore.getState(); wf.set("casesVersion", wf.casesVersion + 1); }} />

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
