import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { api, resourceUrl } from "../api/client";
import type { ScarMetrics } from "../api/types";
import { useCaseStore } from "./caseStore";
import * as nv from "../niivue/nvController";

// Pen labels for the correction drawing: 0 erase, 1 cornea, 2 background, 3 scar.
export type PenLabel = 0 | 1 | 2 | 3;
// Workflow stages: 1 Segment (SAM2) → 2 Correct → 3 Scar (detect + quantify).
export type Stage = 1 | 2 | 3;
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
  status: WorkflowStatus;

  setStage: (s: Stage) => void;
  set: <K extends keyof WorkflowState>(key: K, value: WorkflowState[K]) => void;
  resetForCase: () => void;
  initTabs: (isConsensus: boolean) => void;
  selectTab: (tab: string) => void;
  setOverlayMode: (mode: OverlayMode) => void;

  runSam2: () => Promise<void>;
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

    penLabel: 1,
    penSize: 3,
    penFilled: false,
    paintMode: true,
    drawOpacity: 0.5,
    correcting: false,

    scarMetrics: null,
    scarSensitivity: 10,
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
        s.status = { kind: "idle", title: "Waiting", detail: "Segment the cornea to begin." };
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
      set((s) => {
        s.segBusy = true;
        s.status = { kind: "working", title: "SAM2 segmenting", detail: "Tracking the cornea through axial, coronal and sagittal movies, then fusing in 3D. This takes a few minutes." };
      });
      try {
        const res = await api.json<{ qa: Record<string, unknown> }>(
          `/api/case/${caseId}/segment/sam2`,
          "POST",
          JSON.stringify({ vote: 2 }),
        );
        await nv.loadSegmentation(resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`), get().segOpacity);
        set((s) => {
          s.segQa = res.qa;
          s.segLoaded = true;
          s.segVersion += 1;
          s.stage = 2;
          s.status = { kind: "done", title: "Cornea segmented", detail: "Cornea fused from three orthogonal SAM2 passes. Review/correct it, then move to scar." };
        });
      } catch (e) {
        set((s) => {
          s.status = { kind: "error", title: "SAM2 failed", detail: e instanceof Error ? e.message : String(e) };
        });
      } finally {
        set((s) => {
          s.segBusy = false;
        });
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
        nv.endDrawing();   // clear the drawing bitmap BEFORE reloading the overlay (else it double-renders)
        await nv.loadSegmentation(resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`), get().segOpacity);
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
        try { await nv.loadSegmentation(resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`), get().segOpacity); } catch { /* nothing to restore */ }
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
        await nv.loadSegmentation(resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`), get().segOpacity);
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
        await nv.loadSegmentation(resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`), get().segOpacity);
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
        await nv.loadSegmentation(resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`), get().segOpacity);
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
        // niivue refresh is best-effort (no-op without WebGL); the 2D gallery refetches via segVersion.
        try {
          await nv.loadSegmentation(resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`), get().segOpacity);
        } catch {
          /* no WebGL — gallery updates from segVersion below */
        }
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
        await nv.loadSegmentation(resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`), get().segOpacity);
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
