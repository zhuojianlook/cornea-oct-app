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
  drawOpacity: number;
  correcting: boolean;

  // scar
  scarMetrics: ScarMetrics | null;
  scarSensitivity: number; // 1–40; higher highlights more (percentile = 100 − this)
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

  // busy + status
  segBusy: boolean;
  scarBusy: boolean;
  status: WorkflowStatus;

  setStage: (s: Stage) => void;
  set: <K extends keyof WorkflowState>(key: K, value: WorkflowState[K]) => void;
  initTabs: (isConsensus: boolean) => void;
  selectTab: (tab: string) => void;
  setOverlayMode: (mode: OverlayMode) => void;

  runSam2: () => Promise<void>;
  loadCorrectionLayer: () => Promise<void>;
  saveCorrection: () => Promise<void>;
  setPenLabel: (label: PenLabel) => void;
  runScarAuto: () => Promise<void>;
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
    drawOpacity: 0.5,
    correcting: false,

    scarMetrics: null,
    scarSensitivity: 10,
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
        await nv.loadDrawing(resourceUrl(`/api/case/${caseId}/segmentation-drawing.nii.gz?t=${Date.now()}`));
        nv.setDrawOpacity(get().drawOpacity);
        nv.setPen(get().penLabel || 1);
        set((s) => {
          s.correcting = true;
          if (!s.penLabel) s.penLabel = 1;
          s.status = { kind: "working", title: "Correcting segmentation", detail: "Pen: cornea=1, scar=3, background=2 (erase). Then Save correction." };
        });
      } catch (e) {
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

    setPenLabel: (label) => {
      nv.setPen(label);
      set((s) => {
        s.penLabel = label;
      });
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
          JSON.stringify({ percentile }),
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
