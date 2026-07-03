import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { api, resourceUrl } from "../api/client";
import type { ScarMetrics } from "../api/types";
import { useCaseStore } from "./caseStore";
import * as nv from "../niivue/nvController";

// The read-only overlay loads the DISPLAY labelmap (cornea=1, scar density tiers 2/3/4) so the viewer
// shows scar reflectivity tiers instead of one flat red. The canonical 0/1/2 training label + the
// correction drawing are untouched (they go through best_labelmap_nnunet / the drawing round-trip).
const overlayUrl = (caseId: string) => resourceUrl(`/api/case/${caseId}/segmentation-display.nii.gz?t=${Date.now()}`);

// Pen labels for the correction drawing: 0 erase, 1 cornea, 2 background, 3 scar.
export type PenLabel = 0 | 1 | 2 | 3;
// Workflow stages: 1 Segment (SAM2) → 2 Correct → 3 Scar (detect + quantify) → 4 Motion (eye-motion spectrum).
export type Stage = 1 | 2 | 3 | 4;

// Eye-motion analysis result (POST /api/case/{id}/oct-motion) — the slow/frame axis is time, so the
// detected corneal surface (shape removed) is the patient's motion during the ~0.7s scan.
export interface MotionResult {
  n_frames: number; frame_rate_hz: number; total_s: number; nyquist_hz: number; df_hz: number;
  ascans_per_frame: number; ascan_rate_hz: number; um_per_px: number;
  time_ms: number[]; motion_um: number[]; freqs_hz: number[]; power: number[];
  peaks: { hz: number; period_ms: number | null; power_frac: number; label: string; resolved: boolean }[];
  spikes: { frame: number; t_ms: number; velocity_um_per_s: number }[];
  direction: {
    axial_um_rms: number; inplane_lateral_um_rms: number | null; inplane_reliable?: boolean;
    axial_frac: number; lateral_frac: number;
    tilt_from_normal_deg: number; lateral_azimuth: string; coherence: number; variance_explained: number;
  };
  snr: number | null;
}
// Consensus scan-tab overlay: the scan's own scar mask vs the voted consensus mask.
export type OverlayMode = "self" | "consensus";

// Preview group for a consensus tab: "consensus" → the voted map on the reference
// image; a scan tab → that scan warped into the common frame with the chosen overlay.
const groupFor = (tab: string, mode: OverlayMode): string =>
  tab === "consensus" ? "segmentation" : `scan_${tab}_${mode === "consensus" ? "cons" : "self"}`;

export interface ScarHint {
  ijk: [number, number, number];
  orientation: string;
  slice_index: number;
  positive: boolean;
  fx: number; // fraction across the preview image (for the marker)
  fy: number;
}

// Per-strategy test–retest reproducibility comparison (publication): one row per scar strategy.
export interface StrategyRow {
  strategy: string; mean_volume_mm3?: number; cv_percent?: number; rc_mm3?: number;
  mean_pairwise_dice?: number | null; mean_pairwise_hd95_mm?: number | null; n?: number; error?: string;
}
export interface StrategyComparison {
  rows: StrategyRow[]; members: string[]; n: number; phi_percentile: number; reference?: string; subgroup?: string;
  cancelled?: boolean;   // #15 — true when the run was stopped early via Cancel (rows are partial)
  crop_aware?: boolean;  // #9 — true when a replicate was cropped, so metrics use the common valid region
}

export interface SubgroupPair { a: string; b: string; dice: number; centroid_dist_mm: number | null; sim: number; }
export interface SubgroupProposal {
  members: string[];
  subgroups: Record<string, number>;          // case_id → proposed subgroup label (1..k)
  n_subgroups: number;
  similarity: number[][];
  pairs: SubgroupPair[];
  blobs: Record<string, { n_blobs: number; scar_mm3: number; empty: boolean }>;
  overlay?: string;                            // base64 en-face overlay PNG (coloured by subgroup)
  patient?: string; eye?: string;
}

export type StatusKind = "idle" | "working" | "done" | "error";
export interface WorkflowStatus {
  kind: StatusKind;
  title: string;
  detail: string;
}

interface WorkflowState {
  stage: Stage;

  // segmentation overlay
  segLoaded: boolean;
  segVersion: number; // bumps whenever previews re-render → gallery re-fetches
  segOpacity: number;
  showSegmentation: boolean;
  segQa: Record<string, unknown> | null;
  // The case a SAM2 run is currently in flight for (null = none). Lets a case switch NOT look like an
  // abort: the run keeps going + saves; the timeline shows a background banner instead of clearing.
  sam2RunningCaseId: string | null;
  // Timeline navigation: which step the user is VIEWING (null = follow the live/current step). Clicking an
  // earlier reached step inspects it READ-ONLY (its own viewer tools show, but consequential edits are
  // disabled until "Roll back to this step"). Drives the viewer mode (fix-columns/steps at Auto→Vetted,
  // segmentation at SAM2+) so e.g. after SAM2 the view follows to the segmentation automatically.
  selectedStep: number | null;

  // correction drawing
  penLabel: PenLabel;
  penSize: number;        // brush thickness (voxels)
  penFilled: boolean;     // filled pen: draw a closed outline → fill the enclosed region
  paintMode: boolean;     // true = brush paints; false = navigate (click moves crosshair, no paint)
  cropRegionMode: boolean; // #9 — Fix-columns "Crop" box mode is active → viewer forces sagittal + disables coronal
  drawOpacity: number;
  correcting: boolean;

  // scar
  scarMetrics: ScarMetrics | null;
  scarSensitivity: number; // 1–40; higher highlights more (percentile = 100 − this)
  scarMethod: string; // scar strategy: hysteresis | normal_anchor | robust_mad | morph_lcc | brightness
  scarSummaryInfo: string | null;
  strategyComparison: StrategyComparison | null;   // last per-strategy reproducibility comparison
  subgroupProposal: SubgroupProposal | null;       // last auto subgroup-assignment proposal (bright-spot)
  subgroupBusy: boolean;

  // Defect-marking: when ON, the main single-plane viewer becomes interactive so the user drags to mark the
  // WRONG columns of the current sagittal/axial slice. Marks live in manifest.defect_marks (caseStore); this
  // is just the transient viewer toggle. Off by default → viewer behaves exactly as today.
  markDefectMode: boolean;

  // SAM2 scar hints (click to guide)
  hintMode: boolean;
  hintPositive: boolean; // current click polarity (scar vs not-scar)
  scarHints: ScarHint[];

  // manual 2D scar editing (brush on the slice gallery)
  scarEditMode: boolean;
  scarErase: boolean; // brush erases (scar→cornea) instead of painting (cornea→scar)
  scarBrush: number; // brush radius in source voxels
  runScarEdit: (voxels: [number, number, number][], mode: "paint" | "erase") => Promise<void>;

  // consensus tabs (multi-scan): each repeat scan + the consensus are tabs.
  previewGroup: string | null; // when set, the gallery shows this preview group
  activeTab: string; // "consensus" or a scan caseId
  overlayMode: OverlayMode; // for a scan tab: its own scar vs the consensus

  // Subgroup review: the consensus case to return to after focusing one of its scans
  // to correct it (survives the case switch — deliberately NOT cleared by resetForCase).
  reviewConsensusId: string | null;

  // Manual ground-truth comparison: the Sidebar ManualGtPanel sets these to swap the central
  // viewer to the GtCompareViewer (auto vs imported GT agreement overlay). Reset on case change.
  gtViewerActive: boolean;
  gtViewerName: string | null;          // which imported GT is being compared
  gtViewerClass: "scar" | "cornea";     // which class the agreement overlay shows (a preference)

  // busy + status
  segBusy: boolean;
  scarBusy: boolean;
  // eye-motion tab (stage 4)
  motionBusy: boolean;
  motionResult: MotionResult | null;
  ascanRateHz: number;   // A-scan (line) rate → frame rate → Hz axis; Avanti spec ~70000, editable
  motionSinc: boolean;   // divide out the intra-frame motion-blur boxcar
  status: WorkflowStatus;

  setStage: (s: Stage) => void;
  set: <K extends keyof WorkflowState>(key: K, value: WorkflowState[K]) => void;
  resetForCase: () => void;
  initTabs: (isConsensus: boolean) => void;
  selectTab: (tab: string) => void;
  setOverlayMode: (mode: OverlayMode) => void;
  selectStep: (step: number | null) => void;

  runSam2: () => Promise<void>;
  alignReplicates: () => Promise<void>;
  normalizeConsensus: () => Promise<void>;
  skipNormalization: () => Promise<void>;
  applyConsensusScar: (mode: "consensus" | "own") => Promise<void>;
  compareStrategies: () => Promise<void>;
  cancelCompareStrategies: () => Promise<void>;
  autoSubgroups: () => Promise<void>;
  applySubgroups: (assignments: Record<string, string>) => Promise<void>;
  loadCorrectionLayer: () => Promise<void>;
  saveCorrection: () => Promise<void>;
  // #11 cornea/background vet step: paint cornea/background only (scar pen hidden), then confirm → sets
  // cornea_vetted (gates the Scar step). corneaOnlyPaint hides the Scar pen in the PaintToolbar.
  corneaOnlyPaint: boolean;
  corneaVetBusy: boolean;   // #1 — loading the cornea/background paint layer (spinner on the Paint button)
  correctBusy: boolean;     // loading the Correct paint layer / restoring on Cancel (spinner on Correct & Cancel)
  consensusScarMode: "consensus" | "own" | null;   // which step-9 scar-source choice is being applied (spinner)
  smartFillBusy: boolean;   // #4 — GrowCut smart-fill running (spinner + disable, avoids the "hung" look)
  smartFillPct: number;     // smart-fill progress 0..100 (drives the progress bar)
  startCorneaVetPaint: () => Promise<void>;
  confirmCorneaVet: () => Promise<void>;
  cancelCorrection: () => Promise<void>;
  undoCorrection: () => void;
  setPenLabel: (label: PenLabel) => void;
  setPenSize: (n: number) => void;
  setPenFilled: (f: boolean) => void;
  setPaintMode: (on: boolean) => void;
  runSmartFill: () => Promise<void>;
  runScarAuto: () => Promise<void>;
  runScarAutoSam2: () => Promise<void>;
  runMotionAnalysis: () => Promise<void>;
  addScarHint: (hint: ScarHint) => void;
  clearScarHints: () => void;
  applyScarHints: () => Promise<void>;
  exportScarSummary: () => Promise<void>;
  // #10 — render this scan's preprocessing correction as an MP4 grid (planes × passes). Sets mp4Busy and,
  // on success, correctionMp4Url (a download link).
  exportCorrectionMp4: () => Promise<void>;
  mp4Busy: boolean;
  correctionMp4Url: string | null;
  correctionMp4Info: string;
  tryLoadExistingSegmentation: () => Promise<void>;
  setSegOpacity: (o: number) => void;
  toggleSegmentation: (show: boolean) => void;
}

export const useWorkflowStore = create<WorkflowState>()(
  immer((set, get) => ({
    stage: 1,

    segLoaded: false,
    segVersion: 0,
    segOpacity: 0.5,
    showSegmentation: true,
    segQa: null,
    sam2RunningCaseId: null,
    selectedStep: null,

    corneaOnlyPaint: false,
    corneaVetBusy: false,
    correctBusy: false,
    consensusScarMode: null,
    smartFillBusy: false,
    smartFillPct: 0,
    penLabel: 1,
    penSize: 3,
    penFilled: false,
    paintMode: true,
    cropRegionMode: false,
    drawOpacity: 0.5,
    correcting: false,

    scarMetrics: null,
    scarSensitivity: 8,
    scarMethod: "hysteresis",
    scarSummaryInfo: null,
    strategyComparison: null,
    subgroupProposal: null,
    subgroupBusy: false,

    markDefectMode: false,

    hintMode: false,
    hintPositive: true,
    scarHints: [],

    scarEditMode: false,
    scarErase: false,
    scarBrush: 6,

    previewGroup: null,
    activeTab: "consensus",
    overlayMode: "self",
    reviewConsensusId: null,

    gtViewerActive: false,
    gtViewerName: null,
    gtViewerClass: "scar",

    segBusy: false,
    scarBusy: false,
    mp4Busy: false,
    correctionMp4Url: null,
    correctionMp4Info: "",
    motionBusy: false,
    motionResult: null,
    ascanRateHz: 70000,
    motionSinc: false,
    status: { kind: "idle", title: "Waiting", detail: "Register a volume, then segment the cornea." },

    setStage: (s) =>
      set((state) => {
        state.stage = s;
      }),

    set: (key, value) =>
      set((state) => {
        (state as Record<string, unknown>)[key as string] = value;
      }),

    // Clear per-case workflow state when switching cases so the previous scan's scar
    // metrics / QA / stage / hints / tab routing can't bleed into the new one. Keeps
    // user PREFERENCES (opacity, sensitivity, brush, pen) — only the case-specific
    // results are reset.
    resetForCase: () => {
      nv.endDrawing();   // a case switch must clear any live drawing bitmap (else it leaks onto the new case)
      set((s) => {
        s.stage = 1;
        s.segLoaded = false;
        s.selectedStep = null;        // a new case follows its live step (no stale inspect selection)
        s.showSegmentation = false;   // default each newly-opened scan to Slices; runSam2 flips it on (#6a)
        s.segQa = null;
        s.scarMetrics = null;
        s.scarSummaryInfo = null;
        s.strategyComparison = null;
        s.subgroupProposal = null;
        s.correcting = false;
        s.corneaOnlyPaint = false;
        s.cropRegionMode = false;
        s.markDefectMode = false;
        s.hintMode = false;
        s.scarHints = [];
        s.scarEditMode = false;
        s.scarErase = false;
        s.previewGroup = null;
        s.activeTab = "consensus";
        s.overlayMode = "self";
        s.gtViewerActive = false;
        s.gtViewerName = null;
        s.gtViewerClass = "scar";
        s.segBusy = false;
        s.scarBusy = false;
        s.motionBusy = false;
        s.motionResult = null;   // motion is per-scan; don't bleed across cases (ascanRateHz/sinc are prefs, kept)
        // A SAM2 run in flight for another scan is NOT aborted by switching — keep a background banner
        // instead of clearing to idle (which made it look stopped). The run finishes + saves; reopen to review.
        s.status = s.sam2RunningCaseId
          ? { kind: "working", title: "Segmenting in background",
              detail: `"${s.sam2RunningCaseId}" is still running SAM2 — it will finish and save. Reopen it to review.` }
          : { kind: "idle", title: "Waiting", detail: "Segment the cornea to begin." };
        s.segVersion += 1;
      });
    },

    // Multi-scan consensus: when a consensus case opens, start on the Consensus tab
    // (voted map); a single case clears tab routing so the gallery auto-selects.
    initTabs: (isConsensus) =>
      set((s) => {
        s.activeTab = "consensus";
        s.overlayMode = "self";
        s.previewGroup = isConsensus ? "segmentation" : null;
        s.segVersion += 1; // make the gallery re-fetch for the new routing
      }),

    selectTab: (tab) =>
      set((s) => {
        s.activeTab = tab;
        s.previewGroup = groupFor(tab, s.overlayMode);
        s.segVersion += 1;
      }),

    setOverlayMode: (mode) =>
      set((s) => {
        s.overlayMode = mode;
        if (s.activeTab !== "consensus") s.previewGroup = groupFor(s.activeTab, mode);
        s.segVersion += 1;
      }),

    selectStep: (stepN) => set((s) => { s.selectedStep = stepN; }),

    runSam2: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      // Classification is NOT required before SAM2 — it moved to AFTER cornea-vetting and gates only the scar
      // branch (Subgroup/Scar), so SAM2 runs on any preprocessed scan (no "classify first" gate).
      set((s) => {
        s.segBusy = true;
        s.sam2RunningCaseId = caseId;
        s.status = { kind: "working", title: "Running SAM2 (cornea)", detail: "Tracking the cornea through axial, coronal and sagittal movies, then fusing in 3D. This takes a few minutes." };
      });
      const stillHere = () => useCaseStore.getState().caseId === caseId;
      // Coarse overall % across the one-go pipeline so the user sees a number, not just a spinner.
      // Cornea SAM2 = 3 plane passes + a 3D fuse (~0→80%); the chained scar phase is ~85%; done = 100%.
      const PHASE_PCT: Record<string, number> = { start: 3, axial: 12, coronal: 35, sagittal: 58, fuse: 78 };
      const poll = setInterval(() => {
        api.json<{ phase: string; message: string }>(`/api/case/${caseId}/segment/sam2/status`)
          .then((p) => {
            if (!stillHere() || !p?.message) return;
            if (p.phase === "idle" || p.phase === "done" || p.phase === "error") return;
            const pct = PHASE_PCT[p.phase];
            const detail = pct != null ? `${p.message} · ${pct}%` : p.message;
            // Keyed on sam2RunningCaseId (not segBusy) so reopening the still-running case resumes the live
            // progress text even though the case switch cleared the global segBusy (#8).
            set((s) => { if (s.sam2RunningCaseId === caseId) s.status = { kind: "working", title: "Running SAM2", detail }; });
          })
          .catch(() => undefined);
      }, 1200);
      try {
        const res = await api.json<{ qa: Record<string, unknown> }>(
          `/api/case/${caseId}/segment/sam2`,
          "POST",
          JSON.stringify({ vote: 2 }),
        );
        // CORNEA ONLY now — scar is a SEPARATE step (the user runs / compares scar strategies next).
        // If the user switched cases mid-run, the cornea is saved on disk; don't paint it onto the new case.
        const bgDone = () => set((s) => {
          if (s.sam2RunningCaseId === caseId)
            s.status = { kind: "done", title: "Background scan ready", detail: `"${caseId}" finished cornea segmentation — reopen it to review.` };
        });
        if (!stillHere()) { bgDone(); return; }
        await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0);
        if (!stillHere()) { bgDone(); return; }
        // Advance the timeline NOW: TimelineBar reads scanStep(caseInfo.manifest), only refreshed by openCase.
        // Optimistically set sam2_meta (mirrors caseStore.vetPreprocessing) so the step moves 4→5 (Cornea).
        useCaseStore.setState((cs) => { if (cs.caseInfo) (cs.caseInfo.manifest as Record<string, unknown>).sam2_meta = true; });
        set((s) => {
          s.segQa = res.qa;
          s.segLoaded = true;
          s.segVersion += 1;
          s.stage = 2;
          s.showSegmentation = true;       // auto-switch the viewer to the segmentation overlay
          s.status = { kind: "done", title: "Cornea segmented",
            detail: "Cornea done. Next: segment scar (and compare strategies), then assign subgroup." };
        });
      } catch (e) {
        if (stillHere()) set((s) => { s.status = { kind: "error", title: "SAM2 failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        clearInterval(poll);
        // Only THIS run's case clears the shared busy/running flags — a second run started on another
        // case (after a switch cleared segBusy) must not be broken by this one's completion.
        set((s) => { if (s.sam2RunningCaseId === caseId) { s.segBusy = false; s.sam2RunningCaseId = null; } });
      }
    },

    // POST-SAM2 NEXT STEP: align this eye's replicate scans into one consensus, control-normalised by the
    // tagged control (no-scar) scans. The backend builds the control baseline, control-normalises each
    // replicate's scar, registers + votes them, then we open the consensus case to show the result.
    // STEP 7: align this eye+subgroup's segmented replicates into one consensus using the scar AS-IS
    // (no normalization here). Opens the consensus case to show the result.
    alignReplicates: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.segBusy = true;
        s.status = { kind: "working", title: "Aligning replicates",
          detail: "Registering + voting this eye's repeat scans (same subgroup) into one consensus. This takes a few minutes." };
      });
      try {
        const r = await api.json<{ consensus_case: string; n_replicates: number }>(
          `/api/case/${caseId}/align-replicates`, "POST", JSON.stringify({}));
        useCaseStore.getState().setCaseId(r.consensus_case);
        await useCaseStore.getState().openCase();
        set((s) => {
          s.segVersion += 1; s.selectedStep = null;
          s.status = { kind: "done", title: "Replicates aligned",
            detail: `Consensus of ${r.n_replicates} replicate${r.n_replicates === 1 ? "" : "s"} (raw scar). Normalize against controls next, or correct.` };
        });
      } catch (e) {
        set((s) => { s.status = { kind: "error", title: "Align replicates failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        set((s) => { s.segBusy = false; });
      }
    },

    // STEP 8: re-derive each member's scar as excess over the tagged control (no-scar) baseline
    // (control-normalised, reproducible) and rebuild the consensus. Acts on the open consensus case.
    normalizeConsensus: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.segBusy = true;
        s.status = { kind: "working", title: "Normalizing against controls",
          detail: "Re-deriving scar as excess over the normal-cornea baseline from the control scans, then rebuilding the consensus." };
      });
      try {
        const r = await api.json<{ consensus_case: string; n_controls: number; n_replicates: number }>(
          `/api/case/${caseId}/normalize-consensus`, "POST", JSON.stringify({}));
        useCaseStore.getState().setCaseId(r.consensus_case);
        await useCaseStore.getState().openCase();
        set((s) => {
          s.segVersion += 1; s.selectedStep = null;
          s.status = { kind: "done", title: "Normalized",
            detail: `Consensus control-normalised by ${r.n_controls} control scan${r.n_controls === 1 ? "" : "s"}. Correct it, then schedule.` };
        });
      } catch (e) {
        set((s) => { s.status = { kind: "error", title: "Normalize failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        set((s) => { s.segBusy = false; });
      }
    },

    // STEP 9 "Skip normalization": keep the aligned consensus AS-IS (no control-baseline re-derivation) and
    // advance the timeline so it can be corrected / scheduled. Records normalization_skipped for the export.
    skipNormalization: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => { s.segBusy = true; s.status = { kind: "working", title: "Skipping normalization", detail: "Keeping the aligned consensus as-is." }; });
      try {
        await api.json(`/api/case/${caseId}/skip-normalization`, "POST", JSON.stringify({}));
        useCaseStore.setState((cs) => { if (cs.caseInfo) { const mm = cs.caseInfo.manifest as Record<string, unknown>; mm.normalized = true; mm.normalization_skipped = true; } });
        set((s) => {
          s.segVersion += 1; s.selectedStep = null;
          s.status = { kind: "done", title: "Normalization skipped", detail: "Using the aligned consensus as-is — correct it, then schedule for training." };
        });
      } catch (e) {
        set((s) => { s.status = { kind: "error", title: "Skip failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        set((s) => { s.segBusy = false; });
      }
    },

    // STEP 9 scar-source decision. "consensus" → push the voted consensus scar to EVERY replicate's labelmap
    // (truncated to each replicate's own cornea + FOV by the backend); "own" → keep each replicate's own scar.
    applyConsensusScar: async (mode) => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.segBusy = true;
        s.consensusScarMode = mode;
        s.status = { kind: "working", title: mode === "consensus" ? "Applying consensus scar" : "Keeping per-replicate scars",
          detail: mode === "consensus" ? "Writing the voted consensus scar into every replicate (truncated to each scan's own data)." : "Each replicate keeps its own scar boundary." };
      });
      try {
        const r = await api.json<{ applied?: string[]; skipped?: string[] }>(`/api/case/${caseId}/consensus-scar`, "POST", JSON.stringify({ mode }));
        useCaseStore.setState((cs) => { if (cs.caseInfo) (cs.caseInfo.manifest as Record<string, unknown>).consensus_scar_source = mode; });
        const nSkip = r.skipped?.length ?? 0;
        set((s) => {
          s.segVersion += 1;
          s.status = { kind: "done", title: "Scar source set",
            detail: mode === "consensus"
              ? `Consensus scar applied to ${r.applied?.length ?? 0} replicate(s)` + (nSkip ? `; ${nSkip} kept their own (no consensus scar in their data).` : ".")
              : "Each replicate keeps its own scar." };
        });
      } catch (e) {
        set((s) => { s.status = { kind: "error", title: "Scar-source choice failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        set((s) => { s.segBusy = false; s.consensusScarMode = null; });
      }
    },

    // PUBLICATION: run every scar strategy on this eye's replicates + tabulate test–retest reproducibility
    // (pairwise Dice, HD95, volume CV%, RC). READ-ONLY — does not change the scan's scar.
    compareStrategies: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.scarBusy = true;
        s.status = { kind: "working", title: "Comparing scar strategies",
          detail: "Running each detector (incl. SAM2) on this eye's replicates and computing reproducibility (Dice · HD95 · CV%). SAM2 runs per replicate, so this can take several minutes." };
      });
      try {
        const r = await api.json<StrategyComparison>(`/api/case/${caseId}/compare-strategies`, "POST", JSON.stringify({}));
        if (useCaseStore.getState().caseId !== caseId) return;
        set((s) => {
          s.strategyComparison = r;
          s.status = r.cancelled
            ? { kind: "done", title: "Strategy comparison stopped", detail: `Cancelled after ${r.rows.length} strategy(ies) — partial results shown.` }
            : { kind: "done", title: "Strategy comparison ready",
                detail: `${r.rows.length} strategies × ${r.n} replicates — see the table (download CSV for the paper).` };
        });
      } catch (e) {
        set((s) => { s.status = { kind: "error", title: "Comparison failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        set((s) => { s.scarBusy = false; });
      }
    },
    cancelCompareStrategies: async () => {
      // #15 — actually stop the in-flight (slow, SAM2) compare run: the backend polls this flag between
      // strategies/replicates and returns the partial table. Best-effort; scarBusy clears when it returns.
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => { s.status = { kind: "working", title: "Stopping comparison", detail: "Finishing the current step, then stopping…" }; });
      try { await api.json(`/api/case/${caseId}/compare-strategies/cancel`, "POST", JSON.stringify({})); }
      catch { /* best-effort — if the run already finished there's nothing to cancel */ }
    },
    autoSubgroups: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.subgroupBusy = true; s.subgroupProposal = null;
        s.status = { kind: "working", title: "Auto-assigning subgroups",
          detail: "Aligning each scan's hysteresis bright spots across the eye's replicates (pure bright-spot fit) and clustering by lesion — robust to partial scar cutoff." };
      });
      try {
        const r = await api.json<SubgroupProposal>(`/api/case/${caseId}/subgroup/auto`, "POST", JSON.stringify({}));
        if (useCaseStore.getState().caseId !== caseId) return;
        set((s) => {
          s.subgroupProposal = r;
          s.status = { kind: "done", title: "Subgroup proposal ready",
            detail: `${r.members.length} scans → ${r.n_subgroups} subgroup(s). Verify the overlay, then Apply.` };
        });
      } catch (e) {
        set((s) => { s.status = { kind: "error", title: "Auto-subgroup failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        set((s) => { s.subgroupBusy = false; });
      }
    },
    applySubgroups: async (assignments) => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => { s.subgroupBusy = true; });
      try {
        await api.json(`/api/case/${caseId}/subgroup/auto/apply`, "POST",
          JSON.stringify({ assignments, confirm: true }));
        set((s) => {
          s.subgroupProposal = null;
          s.status = { kind: "done", title: "Subgroups applied", detail: "Each scan's subgroup was set; members that have finished the scar step were confirmed." };
        });
        await useCaseStore.getState().openCase();   // refresh manifest (scar_subgroup / subgroup_confirmed)
      } catch (e) {
        set((s) => { s.status = { kind: "error", title: "Apply failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        set((s) => { s.subgroupBusy = false; });
      }
    },

    loadCorrectionLayer: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => { s.correctBusy = true; });   // spinner on the Correct button while the paint layer loads
      try {
        // Hide the committed colour overlay so it can't blend UNDER the editable drawing layer
        // (two translucent label layers → muddy colours / erase looks like a no-op).
        nv.removeSegmentation();
        await nv.loadDrawing(resourceUrl(`/api/case/${caseId}/segmentation-drawing.nii.gz?t=${Date.now()}`));
        nv.setDrawOpacity(get().drawOpacity);
        nv.setPenSize(get().penSize);
        const lbl = get().penLabel == null ? 1 : get().penLabel;   // keep Erase (0); only default null→1
        nv.setPen(lbl, get().penFilled);
        set((s) => {
          s.correcting = true;
          s.paintMode = true;   // start in paint mode (loadDrawing enabled the pen)
          if (s.penLabel == null) s.penLabel = 1;
          s.status = { kind: "working", title: "Correcting segmentation", detail: "Paint/Navigate toggle in the pen bar. Pen: cornea=blue, scar=red, background=grey (remove over-seg; also a Smart-fill seed), Erase=grey. Brush size, Fill region, Smart fill, Undo; then Save or Cancel." };
        });
      } catch (e) {
        nv.endDrawing();
        set((s) => {
          s.correcting = false;   // defensive: never strand the UI in a half-entered correcting state on load failure
          s.corneaOnlyPaint = false;
          s.status = { kind: "error", title: "Could not load correction layer", detail: e instanceof Error ? e.message : String(e) };
        });
      } finally {
        set((s) => { s.correctBusy = false; });
      }
    },

    saveCorrection: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.segBusy = true;
        s.status = { kind: "working", title: "Saving correction", detail: "Writing the corrected labelmap." };
      });
      try {
        const bytes = await nv.exportDrawing();
        if (!bytes) throw new Error("Could not export the correction drawing.");
        const file = new File([bytes as unknown as BlobPart], "seg-drawing.nii.gz");
        const res = await api.upload<{ qa: Record<string, unknown> }>(`/api/case/${caseId}/segmentation/from-drawing`, [file]);
        if (useCaseStore.getState().caseId !== caseId) return;   // case switched mid-save — don't touch the new case
        nv.endDrawing();   // clear the drawing bitmap BEFORE reloading the overlay (else it double-renders)
        await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0);
        if (useCaseStore.getState().caseId !== caseId) return;
        // Advance the timeline to "Corrected" (mirrors the runSam2 optimistic manifest refresh).
        useCaseStore.setState((cs) => { if (cs.caseInfo) (cs.caseInfo.manifest as Record<string, unknown>).corrected_labelmap = true; });
        set((s) => {
          s.segQa = res.qa;
          s.segLoaded = true;
          s.segVersion += 1;
          s.correcting = false;
          s.status = { kind: "done", title: "Correction saved", detail: "The corrected labelmap is now the source for overlay, metrics and nnU-Net export." };
        });
      } catch (e) {
        set((s) => {
          s.status = { kind: "error", title: "Save correction failed", detail: e instanceof Error ? e.message : String(e) };
        });
      } finally {
        set((s) => {
          s.segBusy = false;
        });
      }
    },

    cancelCorrection: async () => {
      // Discard the drawing edits and restore the committed overlay, without writing anything.
      const caseId = useCaseStore.getState().caseId;
      set((s) => { s.correctBusy = true; });   // spinner on Cancel while the committed overlay reloads
      nv.endDrawing();
      if (caseId) {
        try { await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0); } catch { /* nothing to restore */ }
      }
      set((s) => {
        s.correcting = false;
        s.corneaOnlyPaint = false;
        s.correctBusy = false;
        s.status = { kind: "idle", title: "Correction cancelled", detail: "Edits discarded; the labelmap is unchanged." };
      });
    },

    undoCorrection: () => {
      nv.undoDrawing();
    },

    // #11 — enter cornea/background paint mode: same drawing layer as Correct, but the PaintToolbar hides
    // the Scar pen (corneaOnlyPaint) and the pen defaults to Cornea. Confirm goes through confirmCorneaVet.
    startCorneaVetPaint: async () => {
      // cornea-vet exposes Cornea(1) + Background(2, grey); force the pen into that set (a carried-over
      // Scar(3) or Erase(0) pen would leave the toolbar with nothing selected).
      set((s) => { s.corneaVetBusy = true; s.corneaOnlyPaint = true; if (s.penLabel !== 1 && s.penLabel !== 2) s.penLabel = 1; });
      try {
        await get().loadCorrectionLayer();
        // #2 — after SAM2 the cornea is complete but the BACKGROUND is unpainted/invisible. Fill every
        // non-cornea voxel as the background seed (pen 2, grey) so the user sees + edits a COMPLETE
        // cornea/background partition (on save pen 2 → canonical background 0, so the labelmap is unchanged).
        nv.fillBackgroundSeed();
        set((s) => {
          if (s.penLabel !== 1 && s.penLabel !== 2) s.penLabel = 1;
          s.status = { kind: "working", title: "Vetting cornea/background",
            detail: "Cornea (blue) + background (grey) shown. Paint CORNEA where missing; paint BACKGROUND over wrong cornea to remove it. Then Confirm." };
        });
        nv.setPen(get().penLabel, get().penFilled);
      } finally {
        set((s) => { s.corneaVetBusy = false; });
      }
    },

    // #11 — confirm cornea/background: if the user painted (correcting), save the drawing with cornea_vet=true
    // (sets cornea_vetted, NOT corrected_labelmap); otherwise just mark cornea_vetted. Advances Cornea → vetted.
    confirmCorneaVet: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => { s.segBusy = true; s.status = { kind: "working", title: "Confirming cornea/background", detail: "Saving the vetted cornea segmentation." }; });
      try {
        if (get().correcting) {
          const bytes = await nv.exportDrawing();
          if (!bytes) throw new Error("Could not export the cornea/background drawing.");
          const file = new File([bytes as unknown as BlobPart], "seg-drawing.nii.gz");
          // dedicated endpoint (no ?query) — the upload proxy was dropping ?cornea_vet=true, so the backend
          // set corrected_labelmap instead of cornea_vetted and Confirm appeared to do nothing.
          await api.upload(`/api/case/${caseId}/segmentation/from-drawing-cornea-vet`, [file]);
          if (useCaseStore.getState().caseId !== caseId) return;
          nv.endDrawing();
          await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0);
          if (useCaseStore.getState().caseId !== caseId) return;
        } else {
          await api.json(`/api/case/${caseId}/vet-cornea`, "POST", JSON.stringify({}));
          if (useCaseStore.getState().caseId !== caseId) return;
        }
        // Advance the timeline Cornea(5) → Cornea/bg vetted(6) optimistically.
        useCaseStore.setState((cs) => { if (cs.caseInfo) (cs.caseInfo.manifest as Record<string, unknown>).cornea_vetted = true; });
        set((s) => {
          s.correcting = false;
          s.corneaOnlyPaint = false;
          s.segLoaded = true;
          s.segVersion += 1;
          s.status = { kind: "done", title: "Cornea/background vetted", detail: "Scar detection is now unlocked." };
        });
      } catch (e) {
        set((s) => { s.status = { kind: "error", title: "Confirm failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        set((s) => { s.segBusy = false; });
      }
    },

    setPenLabel: (label) => {
      nv.setPen(label, get().penFilled);
      set((s) => {
        s.penLabel = label;
      });
    },

    setPenSize: (n) => {
      nv.setPenSize(n);
      set((s) => { s.penSize = n; });
    },

    setPenFilled: (f) => {
      nv.setPen(get().penLabel, f);   // re-arm the current pen as filled/unfilled
      set((s) => { s.penFilled = f; });
    },

    setPaintMode: (on) => {
      // Paint mode: brush draws. Navigate mode: drawing off → left-click moves the crosshair / scrubs
      // slices without painting (resolves the crosshair-vs-paint input overlap).
      if (on) nv.setPen(get().penLabel, get().penFilled);
      nv.setDrawingEnabled(on);
      set((s) => { s.paintMode = on; });
    },

    runSmartFill: async () => {
      // CPU geodesic flood in a Web Worker (ported from the annotator) — niivue's GPU drawGrowCut hangs on
      // the WebKitGTK/NVIDIA stack. The worker runs off the main thread (UI stays responsive) and reports
      // progress. Needs ≥1 cornea/scar seed; it assigns every unlabelled voxel the nearest seed's label.
      if (get().smartFillBusy) return;
      set((s) => {
        s.smartFillBusy = true;
        s.smartFillPct = 0;
        s.status = { kind: "working", title: "Smart fill", detail: "Growing labels from your scribbles (CPU, off the main thread)…" };
      });
      try {
        const res = await nv.smartFill((pct) => set((s) => {
          s.smartFillPct = pct;
          if (s.smartFillBusy) s.status = { kind: "working", title: "Smart fill", detail: `Growing labels from your scribbles… ${pct}%` };
        }));
        if (!res.ok) {
          set((s) => { s.status = { kind: "error", title: "Smart fill",
            detail: res.reason === "no-seeds"
              ? "Paint a little Cornea (and Background) on a few slices first — Smart fill grows outward from your scribbles."
              : res.reason === "size-mismatch"
                ? "The drawing and volume sizes don't match — reopen the scan and try again."
                : "No volume loaded." }; });
        } else {
          set((s) => {
            s.segVersion = s.segVersion + 1;
            s.status = { kind: "done", title: "Smart fill complete", detail: "Labels propagated — review/correct, then Confirm or Save." };
          });
        }
      } catch (e) {
        set((s) => { s.status = { kind: "error", title: "Smart fill failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        set((s) => { s.smartFillBusy = false; s.smartFillPct = 0; });
      }
    },

    runScarAuto: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.scarBusy = true;
        s.status = { kind: "working", title: "Detecting scar", detail: "Flagging hyper-reflective regions inside the cornea (a starting mask to correct)." };
      });
      try {
        const percentile = Math.min(99, Math.max(60, 100 - get().scarSensitivity));
        const res = await api.json<{ metrics: ScarMetrics }>(
          `/api/case/${caseId}/scar/auto`,
          "POST",
          JSON.stringify({ percentile, method: get().scarMethod }),
        );
        if (useCaseStore.getState().caseId !== caseId) return;   // case switched mid-run — don't write onto the new case
        await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0);
        if (useCaseStore.getState().caseId !== caseId) return;
        // Scar is its own step now — advance the timeline 5(Cornea)→6(Scar) optimistically (backend set scar_done).
        useCaseStore.setState((cs) => { if (cs.caseInfo) (cs.caseInfo.manifest as Record<string, unknown>).scar_done = true; });
        set((s) => {
          s.scarMetrics = res.metrics;
          s.segLoaded = true;
          s.segVersion += 1;
          s.status = res.metrics.scar_present
            ? { kind: "done", title: "Scar pre-annotated", detail: `${res.metrics.scar_volume_mm3 ?? 0} mm³ · ${res.metrics.scar_area_mm2 ?? 0} mm² en-face (${Math.round((res.metrics.scar_fraction_of_cornea ?? 0) * 100)}% of cornea). Correct it, then export.` }
            : { kind: "done", title: "No scar found", detail: "No hyper-reflective region flagged — raise sensitivity or paint scar manually." };
        });
      } catch (e) {
        set((s) => {
          s.status = { kind: "error", title: "Scar detection failed", detail: e instanceof Error ? e.message : String(e) };
        });
      } finally {
        set((s) => {
          s.scarBusy = false;
        });
      }
    },

    runMotionAnalysis: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.motionBusy = true;
        s.status = { kind: "working", title: "Analysing eye motion", detail: "Tracking the corneal surface over the scan (the slow axis is time) + its spectrum." };
      });
      try {
        const r = await api.json<MotionResult>(
          `/api/case/${caseId}/oct-motion`, "POST",
          JSON.stringify({ ascan_rate_hz: get().ascanRateHz, sinc_correct: get().motionSinc }),
        );
        if (useCaseStore.getState().caseId !== caseId) return;   // case switched mid-run — don't write onto the new case
        set((s) => {
          s.motionResult = r;
          const top = r.peaks && r.peaks[0];
          s.status = { kind: "done", title: "Eye motion analysed",
            detail: `${r.frame_rate_hz} Hz frames · ${top ? `${top.hz} Hz dominant` : "no clear peak"} · SNR ${r.snr ?? "—"}` };
        });
      } catch (e) {
        set((s) => {
          s.motionResult = null;
          s.status = { kind: "error", title: "Motion analysis failed", detail: e instanceof Error ? e.message : String(e) };
        });
      } finally {
        set((s) => { s.motionBusy = false; });
      }
    },

    runScarAutoSam2: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.scarBusy = true;
        s.status = { kind: "working", title: "Auto scar (SAM2 · 3 views)", detail: "Seeding from the brightest in-cornea tissue, running SAM2 on axial/coronal/sagittal as videos, then taking the ≥2-of-3 consensus. This takes ~1–2 min." };
      });
      try {
        const percentile = Math.min(99, Math.max(60, 100 - get().scarSensitivity));
        const res = await api.json<{ metrics: ScarMetrics }>(
          `/api/case/${caseId}/scar/auto-sam2`,
          "POST",
          JSON.stringify({ percentile }),
        );
        if (useCaseStore.getState().caseId !== caseId) return;   // case switched mid-run — don't write onto the new case
        await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0);
        if (useCaseStore.getState().caseId !== caseId) return;
        // Scar is its own step now — advance the timeline 5(Cornea)→6(Scar) optimistically (backend set scar_done).
        useCaseStore.setState((cs) => { if (cs.caseInfo) (cs.caseInfo.manifest as Record<string, unknown>).scar_done = true; });
        set((s) => {
          s.scarMetrics = res.metrics;
          s.segLoaded = true;
          s.segVersion += 1;
          s.status = res.metrics.scar_present
            ? { kind: "done", title: "Scar (SAM2 3-view consensus)", detail: `${res.metrics.scar_volume_mm3 ?? 0} mm³ · ${res.metrics.scar_area_mm2 ?? 0} mm² en-face (${Math.round((res.metrics.scar_fraction_of_cornea ?? 0) * 100)}% of cornea). Correct it, then export.` }
            : { kind: "done", title: "No scar found", detail: "3-view SAM2 consensus flagged nothing — raise sensitivity, or use clicks." };
        });
      } catch (e) {
        set((s) => {
          s.status = { kind: "error", title: "Auto scar (SAM2) failed", detail: e instanceof Error ? e.message : String(e) };
        });
      } finally {
        set((s) => {
          s.scarBusy = false;
        });
      }
    },

    addScarHint: (hint) =>
      set((s) => {
        s.scarHints.push(hint);
      }),

    clearScarHints: () =>
      set((s) => {
        s.scarHints = [];
      }),

    applyScarHints: async () => {
      const caseId = useCaseStore.getState().caseId;
      const hints = get().scarHints;
      if (!caseId || hints.length === 0) return;
      set((s) => {
        s.scarBusy = true;
        s.status = { kind: "working", title: "SAM2 scar hint", detail: "Guiding SAM2 with your clicks, then keeping the hyper-reflective tissue. This takes ~20s per plane." };
      });
      try {
        const points = hints.map((h) => ({ ijk: h.ijk, orientation: h.orientation, positive: h.positive }));
        const res = await api.json<{ metrics: ScarMetrics }>(
          `/api/case/${caseId}/scar/sam2-hint`,
          "POST",
          JSON.stringify({ points, replace: false }),
        );
        // case switched mid-run — don't write metrics or wipe the NEW case's freshly-placed hints
        if (useCaseStore.getState().caseId !== caseId) return;
        await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0);
        if (useCaseStore.getState().caseId !== caseId) return;
        // Scar is its own step now — advance the timeline 5(Cornea)→6(Scar) optimistically (backend set scar_done).
        useCaseStore.setState((cs) => { if (cs.caseInfo) (cs.caseInfo.manifest as Record<string, unknown>).scar_done = true; });
        set((s) => {
          s.scarMetrics = res.metrics;
          s.segLoaded = true;
          s.segVersion += 1;
          s.scarHints = [];
          s.status = { kind: "done", title: "Scar updated", detail: `${res.metrics.scar_volume_mm3 ?? 0} mm³ · ${res.metrics.scar_area_mm2 ?? 0} mm² en-face after your hints.` };
        });
      } catch (e) {
        set((s) => {
          s.status = { kind: "error", title: "Scar hint failed", detail: e instanceof Error ? e.message : String(e) };
        });
      } finally {
        set((s) => {
          s.scarBusy = false;
        });
      }
    },

    runScarEdit: async (voxels, mode) => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId || voxels.length === 0) return;
      set((s) => {
        s.scarBusy = true;
      });
      try {
        const res = await api.json<{ metrics: ScarMetrics }>(
          `/api/case/${caseId}/scar/edit`,
          "POST",
          JSON.stringify({ voxels, mode }),
        );
        if (useCaseStore.getState().caseId !== caseId) return;   // case switched mid-edit — don't write onto the new case
        // niivue refresh is best-effort (no-op without WebGL); the 2D gallery refetches via segVersion.
        try {
          await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0);
        } catch {
          /* no WebGL — gallery updates from segVersion below */
        }
        if (useCaseStore.getState().caseId !== caseId) return;
        // Scar is its own step now — advance the timeline 5(Cornea)→6(Scar) optimistically (backend set scar_done).
        useCaseStore.setState((cs) => { if (cs.caseInfo) (cs.caseInfo.manifest as Record<string, unknown>).scar_done = true; });
        set((s) => {
          s.scarMetrics = res.metrics;
          s.segLoaded = true;
          s.segVersion += 1;
          s.status = {
            kind: "done",
            title: mode === "erase" ? "Scar erased" : "Scar painted",
            detail: `${res.metrics.scar_volume_mm3 ?? 0} mm³ · ${res.metrics.scar_area_mm2 ?? 0} mm² en-face after edit.`,
          };
        });
      } catch (e) {
        set((s) => {
          s.status = { kind: "error", title: "Scar edit failed", detail: e instanceof Error ? e.message : String(e) };
        });
      } finally {
        set((s) => {
          s.scarBusy = false;
        });
      }
    },

    exportScarSummary: async () => {
      set((s) => {
        s.scarBusy = true;
        s.scarSummaryInfo = "Computing scar metrics across all cases…";
      });
      try {
        const res = await api.json<{ csv: string; n_cases: number }>(`/api/metrics/summary`, "POST", JSON.stringify({}));
        set((s) => {
          s.scarSummaryInfo = `Wrote scar_summary.csv (${res.n_cases} case${res.n_cases === 1 ? "" : "s"}) → ${res.csv}`;
        });
      } catch (e) {
        set((s) => {
          s.scarSummaryInfo = `Export failed: ${e instanceof Error ? e.message : String(e)}`;
        });
      } finally {
        set((s) => {
          s.scarBusy = false;
        });
      }
    },

    exportCorrectionMp4: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => { s.mp4Busy = true; s.correctionMp4Url = null; s.correctionMp4Info = "Rendering correction MP4 (planes × passes)…"; });
      try {
        const res = await api.json<{ out: string; frames: number; columns: string[]; download_url: string }>(
          `/api/case/${caseId}/export-correction-mp4`, "POST", JSON.stringify({}));
        if (useCaseStore.getState().caseId !== caseId) return;
        set((s) => {
          s.correctionMp4Url = resourceUrl(`${res.download_url}?t=${Date.now()}`);
          s.correctionMp4Info = `Saved ${res.frames}-frame MP4 (${res.columns.join(" · ")}) → ${res.out}`;
        });
      } catch (e) {
        set((s) => { s.correctionMp4Info = `Export failed: ${e instanceof Error ? e.message : String(e)}`; });
      } finally {
        set((s) => { s.mp4Busy = false; });
      }
    },

    tryLoadExistingSegmentation: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      try {
        await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0);
        set((s) => {
          s.segLoaded = true;
          s.segVersion += 1;
        });
      } catch {
        /* no segmentation yet — fine */
      }
    },

    setSegOpacity: (o) => {
      nv.setSegmentationOpacity(o);
      set((s) => {
        s.segOpacity = o;
      });
    },

    toggleSegmentation: (show) => {
      nv.setSegmentationOpacity(show ? get().segOpacity : 0);
      set((s) => {
        s.showSegmentation = show;
      });
    },
  })),
);
