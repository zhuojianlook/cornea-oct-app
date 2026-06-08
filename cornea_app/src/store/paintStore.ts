import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { api, resourceUrl } from "../api/client";
import type { AiPaintResult, PreviewImage, ScarMetrics, Stage } from "../api/types";
import { useCaseStore } from "./caseStore";
import * as nv from "../niivue/nvController";

export type Provider = "local" | "openai" | "medgemma";
export type PenLabel = 0 | 1 | 2; // 0 erase, 1 cornea, 2 background

export type StatusKind = "idle" | "working" | "done" | "error";
export interface PaintStatus {
  kind: StatusKind;
  title: string;
  detail: string;
}

interface PaintState {
  stage: Stage;
  provider: Provider;
  model: string;
  apiKey: string;
  baseUrl: string;
  reviewerPrompt: string;

  penLabel: PenLabel;
  brushSize: number;
  drawOpacity: number;
  showSeeds: boolean;
  showSegmentation: boolean;
  segOpacity: number;

  aiBusy: boolean;
  editing: boolean;
  drawingLoaded: boolean;
  status: PaintStatus;
  result: AiPaintResult["qa"] | null;
  seedImages: PreviewImage[];

  growBusy: boolean;
  growQa: Record<string, unknown> | null;
  segLoaded: boolean;

  scarBusy: boolean;
  scarMetrics: ScarMetrics | null;

  setStage: (s: Stage) => void;
  set: <K extends keyof PaintState>(key: K, value: PaintState[K]) => void;
  aiPaint: (useHeuristic: boolean) => Promise<void>;
  refreshSeedPreviews: () => Promise<void>;
  loadDrawingLayer: () => Promise<void>;
  setPenLabel: (label: PenLabel | 3) => void;
  applyEdits: () => Promise<void>;
  runGrow: () => Promise<void>;
  setSegOpacity: (o: number) => void;
  toggleSegmentation: (show: boolean) => void;
  tryLoadExistingSegmentation: () => Promise<void>;
  submitFeedback: (decision: "accept" | "reject", notes: string) => Promise<void>;
  detectScar: (useHeuristic: boolean) => Promise<void>;
}

export const usePaintStore = create<PaintState>()(
  immer((set, get) => ({
    stage: 1,
    provider: "local",
    model: "local-vision-model",
    apiKey: "",
    baseUrl: "http://127.0.0.1:1234/v1",
    reviewerPrompt: "",

    penLabel: 1,
    brushSize: 4,
    drawOpacity: 0.5,
    showSeeds: true,
    showSegmentation: true,
    segOpacity: 0.5,

    aiBusy: false,
    editing: false,
    drawingLoaded: false,
    status: { kind: "idle", title: "Waiting", detail: "Register a volume, then generate seed paint." },
    result: null,
    seedImages: [],

    growBusy: false,
    growQa: null,
    segLoaded: false,

    scarBusy: false,
    scarMetrics: null,

    setStage: (s) =>
      set((state) => {
        state.stage = s;
      }),

    set: (key, value) =>
      set((state) => {
        (state as Record<string, unknown>)[key as string] = value;
      }),

    refreshSeedPreviews: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      try {
        const res = await api.json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/seeds`);
        set((s) => {
          s.seedImages = res.images;
        });
      } catch {
        /* previews are best-effort */
      }
    },

    aiPaint: async (useHeuristic: boolean) => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) {
        set((s) => {
          s.status = { kind: "error", title: "No case", detail: "Open a case first." };
        });
        return;
      }
      const p = get();
      set((s) => {
        s.aiBusy = true;
        s.result = null;
        s.status = {
          kind: "working",
          title: useHeuristic ? "Heuristic paint running" : "AI paint running",
          detail: useHeuristic
            ? "Slicer is generating deterministic seed paint."
            : "Rendering context slices and asking the vision model for seed strokes.",
        };
      });
      try {
        let res: AiPaintResult;
        if (useHeuristic) {
          res = await api.json<AiPaintResult>(`/api/case/${caseId}/ai-paint/heuristic`, "POST");
        } else {
          res = await api.json<AiPaintResult>(
            `/api/case/${caseId}/ai-paint`,
            "POST",
            JSON.stringify({
              provider: p.provider,
              model: p.model,
              api_key: p.apiKey || null,
              local_base_url: p.baseUrl,
              reviewer_prompt: p.reviewerPrompt,
            }),
          );
        }
        const marking = res.qa?.agent_marking || {};
        set((s) => {
          s.result = res.qa;
        });
        await get().refreshSeedPreviews();
        await get().loadDrawingLayer();
        set((s) => {
          s.status = {
            kind: "done",
            title: useHeuristic ? "Heuristic paint ready" : "AI paint ready",
            detail: `${marking.cornea_stroke_count ?? 0} cornea + ${
              marking.background_stroke_count ?? 0
            } background stroke(s). Review and edit, then Grow from Seeds.`,
          };
        });
      } catch (e) {
        set((s) => {
          s.status = {
            kind: "error",
            title: "Paint failed",
            detail: e instanceof Error ? e.message : String(e),
          };
        });
      } finally {
        set((s) => {
          s.aiBusy = false;
        });
      }
    },

    loadDrawingLayer: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      try {
        await nv.loadDrawing(resourceUrl(`/api/case/${caseId}/seed-drawing.nii.gz?t=${Date.now()}`));
        nv.setDrawOpacity(get().drawOpacity);
        nv.setPen(get().penLabel);
        set((s) => {
          s.drawingLoaded = true;
          s.editing = true;
        });
      } catch (e) {
        set((s) => {
          s.status = {
            kind: "error",
            title: "Drawing layer failed",
            detail: e instanceof Error ? e.message : String(e),
          };
        });
      }
    },

    setPenLabel: (label) => {
      nv.setPen(label);
      set((s) => {
        s.penLabel = label as PenLabel;
      });
    },

    runGrow: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.growBusy = true;
        s.status = { kind: "working", title: "Grow from Seeds running", detail: "Slicer is growing the full segmentation from your seeds." };
      });
      try {
        const res = await api.json<{ qa: Record<string, unknown> }>(
          `/api/case/${caseId}/grow`,
          "POST",
          JSON.stringify({ seed_locality_factor: 0 }),
        );
        await nv.loadSegmentation(
          resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`),
          get().segOpacity,
        );
        set((s) => {
          s.growQa = res.qa;
          s.segLoaded = true;
          s.stage = 2;
          s.status = { kind: "done", title: "Segmentation ready", detail: "Grow from Seeds finished. The labelmap overlays the volume." };
        });
      } catch (e) {
        set((s) => {
          s.status = { kind: "error", title: "Grow failed", detail: e instanceof Error ? e.message : String(e) };
        });
      } finally {
        set((s) => {
          s.growBusy = false;
        });
      }
    },

    submitFeedback: async (decision, notes) => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      try {
        await api.json(`/api/case/${caseId}/feedback`, "POST", JSON.stringify({ decision, notes }));
      } catch (e) {
        set((s) => {
          s.status = { kind: "error", title: "Feedback failed", detail: e instanceof Error ? e.message : String(e) };
        });
        return;
      }
      if (decision === "reject") {
        // Active-learning loop: the next AI paint reads this feedback from the prompt.
        set((s) => {
          s.status = { kind: "working", title: "Repainting with feedback", detail: "Your notes are included in the model prompt." };
        });
        await get().aiPaint(false);
      } else {
        set((s) => {
          s.status = { kind: "done", title: "Accepted", detail: "Saved your verdict for this case." };
        });
      }
    },

    detectScar: async (useHeuristic: boolean) => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      const p = get();
      set((s) => {
        s.scarBusy = true;
        s.status = { kind: "working", title: "Detecting scar", detail: "Finding scar within the cornea and re-growing." };
      });
      try {
        const res = await api.json<{ metrics: ScarMetrics; qa: Record<string, unknown> }>(
          useHeuristic ? `/api/case/${caseId}/scar/heuristic` : `/api/case/${caseId}/scar`,
          "POST",
          useHeuristic
            ? undefined
            : JSON.stringify({ provider: p.provider, model: p.model, api_key: p.apiKey || null, local_base_url: p.baseUrl, reviewer_prompt: p.reviewerPrompt }),
        );
        await nv.loadSegmentation(
          resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`),
          get().segOpacity,
        );
        set((s) => {
          s.scarMetrics = res.metrics;
          s.growQa = res.qa;
          s.segLoaded = true;
          s.status = res.metrics.scar_present
            ? { kind: "done", title: "Scar detected", detail: `${res.metrics.scar_voxels?.toLocaleString()} scar voxels (${Math.round((res.metrics.scar_fraction_of_cornea ?? 0) * 100)}% of cornea).` }
            : { kind: "done", title: "No scar found", detail: res.metrics.note || "No scar region in this sample." };
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

    tryLoadExistingSegmentation: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      try {
        const qa = await api.json<Record<string, unknown>>(`/api/case/${caseId}/segmentation/qa`);
        await nv.loadSegmentation(
          resourceUrl(`/api/case/${caseId}/segmentation.nii.gz?t=${Date.now()}`),
          get().segOpacity,
        );
        set((s) => {
          s.growQa = qa;
          s.segLoaded = true;
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

    applyEdits: async () => {
      const caseId = useCaseStore.getState().caseId;
      if (!caseId) return;
      set((s) => {
        s.aiBusy = true;
        s.status = { kind: "working", title: "Applying edits", detail: "Converting your paint to seeds." };
      });
      try {
        const bytes = await nv.exportDrawing();
        if (!bytes) throw new Error("Could not export the drawing layer.");
        const file = new File([bytes as unknown as BlobPart], "drawing.nii.gz");
        const res = await api.upload<{ counts: Record<string, number> }>(
          `/api/case/${caseId}/seeds/from-drawing`,
          [file],
        );
        await get().refreshSeedPreviews();
        const c = res.counts || {};
        set((s) => {
          s.status = {
            kind: "done",
            title: "Edits applied",
            detail: `Seeds updated — cornea ${c.cornea ?? 0}, background ${c.background ?? 0}, scar ${
              c.scar ?? 0
            } voxels.`,
          };
        });
      } catch (e) {
        set((s) => {
          s.status = { kind: "error", title: "Apply failed", detail: e instanceof Error ? e.message : String(e) };
        });
      } finally {
        set((s) => {
          s.aiBusy = false;
        });
      }
    },
  })),
);
