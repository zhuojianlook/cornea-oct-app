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

  // correction drawing
  penLabel: PenLabel;
  penSize: number;        // brush thickness (voxels)
  penFilled: boolean;     // filled pen: draw a closed outline → fill the enclosed region
  paintMode: boolean;     // true = brush paints; false = navigate (click moves crosshair, no paint)
  drawOpacity: number;
  correcting: boolean;

  // scar
  scarMetrics: ScarMetrics | null;
  scarSensitivity: number; // 1–40; higher highlights more (percentile = 100 − this)
  scarMethod: string; // scar strategy: hysteresis | normal_anchor | robust_mad | morph_lcc | brightness
  scarSummaryInfo: string | null;

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

  runSam2: () => Promise<void>;
  buildEyeConsensus: () => Promise<void>;
  loadCorrectionLayer: () => Promise<void>;
  saveCorrection: () => Promise<void>;
  cancelCorrection: () => Promise<void>;
  undoCorrection: () => void;
  setPenLabel: (label: PenLabel) => void;
  setPenSize: (n: number) => void;
  setPenFilled: (f: boolean) => void;
  setPaintMode: (on: boolean) => void;
  runSmartFill: () => void;
  runScarAuto: () => Promise<void>;
  runScarAutoSam2: () => Promise<void>;
  runMotionAnalysis: () => Promise<void>;
  addScarHint: (hint: ScarHint) => void;
  clearScarHints: () => void;
  applyScarHints: () => Promise<void>;
  exportScarSummary: () => Promise<void>;
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

    penLabel: 1,
    penSize: 3,
    penFilled: false,
    paintMode: true,
    drawOpacity: 0.5,
    correcting: false,

    scarMetrics: null,
    scarSensitivity: 8,
    scarMethod: "hysteresis",
    scarSummaryInfo: null,

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
        s.showSegmentation = false;   // default each newly-opened scan to Slices; runSam2 flips it on (#6a)
        s.segQa = null;
        s.scarMetrics = null;
        s.scarSummaryInfo = null;
        s.correcting = false;
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

    runSam2: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      // Reordered lifecycle: SAM2 is gated until the scan is classified (scar/control set) — the timeline's
      // "wait for selected to be labelled" between step 4 (classified) and step 5 (SAM2).
      const m = (useCaseStore.getState().caseInfo?.manifest ?? {}) as Record<string, unknown>;
      if (!m.scar_classification) {
        set((s) => { s.status = { kind: "error", title: "Classify first",
          detail: "Mark this scan as Scar or Control (and set replicates/controls) before running SAM2." }; });
        return;
      }
      const isScar = m.scar_classification === "scar";
      set((s) => {
        s.segBusy = true;
        s.sam2RunningCaseId = caseId;
        s.status = { kind: "working", title: "Running SAM2", detail: "Tracking the cornea through axial, coronal and sagittal movies, then fusing in 3D. This takes a few minutes." };
      });
      const stillHere = () => useCaseStore.getState().caseId === caseId;
      // Poll live progress (per-plane / fuse) so the user sees phases, not just a spinner; only writes
      // while THIS case is still open + the run is busy (never overwrites another case's status).
      const poll = setInterval(() => {
        api.json<{ phase: string; message: string }>(`/api/case/${caseId}/segment/sam2/status`)
          .then((p) => {
            if (!stillHere() || !p?.message) return;
            if (p.phase === "idle" || p.phase === "done" || p.phase === "error") return;
            // Keyed on sam2RunningCaseId (not segBusy) so reopening the still-running case resumes the live
            // progress text even though the case switch cleared the global segBusy (#8).
            set((s) => { if (s.sam2RunningCaseId === caseId) s.status = { kind: "working", title: "Running SAM2", detail: p.message }; });
          })
          .catch(() => undefined);
      }, 1200);
      try {
        const res = await api.json<{ qa: Record<string, unknown> }>(
          `/api/case/${caseId}/segment/sam2`,
          "POST",
          JSON.stringify({ vote: 2 }),
        );
        // ONE-GO pipeline (#6): a scar-labelled scan then runs scar detection with the chosen method
        // (cornea-vs-scar); a control runs cornea only. A scar failure must NOT blank the saved cornea.
        let scarRan = false;
        if (isScar) {
          if (stillHere()) set((s) => { s.status = { kind: "working", title: "Detecting scar", detail: "Flagging hyper-reflective tissue inside the cornea…" }; });
          try {
            const pct = Math.min(99, Math.max(60, 100 - get().scarSensitivity));
            await api.json(`/api/case/${caseId}/scar/auto`, "POST", JSON.stringify({ percentile: pct, method: get().scarMethod }));
            scarRan = true;
          } catch { /* cornea is already saved server-side; the user can re-run scar from the timeline */ }
        }
        // The user may switch cases during the multi-minute run (the consensus "focus → correct → back"
        // flow does this). If so, the result is saved on disk — don't paint it onto whatever case is now
        // open; instead replace the "still running in background" banner with a "done — reopen" notice.
        const bgDone = () => set((s) => {
          if (s.sam2RunningCaseId === caseId)
            s.status = { kind: "done", title: "Background scan ready",
              detail: `"${caseId}" finished segmenting${scarRan ? " (cornea + scar)" : ""} — reopen it to review.` };
        });
        if (!stillHere()) { bgDone(); return; }
        await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0);
        if (!stillHere()) { bgDone(); return; }
        // Advance the timeline NOW: TimelineBar reads scanStep(caseInfo.manifest), which is only refreshed
        // by openCase. Optimistically set the segmentation flag (mirrors caseStore.vetPreprocessing) so the
        // step moves 4→5 without a full reload (the backend already persisted sam2_meta).
        useCaseStore.setState((cs) => { if (cs.caseInfo) (cs.caseInfo.manifest as Record<string, unknown>).sam2_meta = true; });
        set((s) => {
          s.segQa = res.qa;
          s.segLoaded = true;
          s.segVersion += 1;
          s.stage = 2;
          s.showSegmentation = true;       // #6a: auto-switch the viewer to the segmentation overlay
          s.status = { kind: "done", title: scarRan ? "Cornea + scar segmented" : "Cornea segmented",
            detail: isScar
              ? (scarRan ? "Cornea + scar done. Align this eye's replicates, or correct, then schedule."
                         : "Cornea done; scar detection failed — re-run scar from the timeline.")
              : "Cornea done (control — no scar). Align this eye's replicates, or correct, then schedule." };
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
    buildEyeConsensus: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.segBusy = true;
        s.status = { kind: "working", title: "Aligning replicates",
          detail: "Building the control baseline, control-normalising scar, and registering + voting this eye's repeat scans into one consensus. This takes a few minutes." };
      });
      try {
        const r = await api.json<{ consensus_case: string; n_replicates: number; n_controls: number; control_normalized: boolean }>(
          `/api/case/${caseId}/build-eye-consensus`, "POST", JSON.stringify({}));
        const norm = r.control_normalized
          ? `control-normalised by ${r.n_controls} control scan${r.n_controls === 1 ? "" : "s"}`
          : "no control baseline yet (tag control scans to normalise)";
        useCaseStore.getState().setCaseId(r.consensus_case);
        await useCaseStore.getState().openCase();
        set((s) => {
          s.segVersion += 1;
          s.status = { kind: "done", title: "Replicates aligned",
            detail: `Consensus of ${r.n_replicates} replicate${r.n_replicates === 1 ? "" : "s"} — ${norm}. Showing the consensus.` };
        });
      } catch (e) {
        set((s) => { s.status = { kind: "error", title: "Align replicates failed", detail: e instanceof Error ? e.message : String(e) }; });
      } finally {
        set((s) => { s.segBusy = false; });
      }
    },

    loadCorrectionLayer: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
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
          s.status = { kind: "working", title: "Correcting segmentation", detail: "Paint/Navigate toggle in the pen bar. Pen: cornea=blue, scar=red, background=orange (erase). Brush size, Fill region, Smart fill, Undo; then Save or Cancel." };
        });
      } catch (e) {
        nv.endDrawing();
        set((s) => {
          s.status = { kind: "error", title: "Could not load correction layer", detail: e instanceof Error ? e.message : String(e) };
        });
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
      nv.endDrawing();
      if (caseId) {
        try { await nv.loadSegmentation(overlayUrl(caseId), get().showSegmentation ? get().segOpacity : 0); } catch { /* nothing to restore */ }
      }
      set((s) => {
        s.correcting = false;
        s.status = { kind: "idle", title: "Correction cancelled", detail: "Edits discarded; the labelmap is unchanged." };
      });
    },

    undoCorrection: () => {
      nv.undoDrawing();
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

    runSmartFill: () => {
      // GrowCut propagates the scribbled bg/cornea/scar labels through the whole 3-D volume.
      nv.smartFill();
      set((s) => { s.segVersion = s.segVersion + 1; });
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
