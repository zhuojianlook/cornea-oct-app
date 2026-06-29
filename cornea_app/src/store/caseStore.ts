import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { api, checkHealth, resourceUrl } from "../api/client";
import type { AppConfig, CaseInfo } from "../api/types";
import { useWorkflowStore } from "./workflowStore";

// The last case openCase() actually switched to — so we only reset the per-case
// workflow state on a genuine case CHANGE, not on a same-case reopen/refresh.
let _lastOpenedCase: string | null = null;

interface CaseState {
  config: AppConfig | null;
  healthy: boolean;
  apiError: string | null;
  caseId: string | null;
  caseInfo: CaseInfo | null;
  volumeUrl: string | null;
  busy: boolean;

  fetchConfig: () => Promise<void>;
  setApiError: (msg: string | null) => void;
  setCaseId: (id: string) => void;
  clearCase: () => void;   // empty the viewer immediately (e.g. wipe-all): drop caseId/caseInfo/volumeUrl
  openCase: () => Promise<void>;
  registerVolume: (path: string) => Promise<void>;
  uploadVolume: (file: File) => Promise<void>;
  exportNnunet: () => Promise<void>;
  exportInfo: string | null;
  preprocessed: boolean;
  setPreprocess: (enabled: boolean) => Promise<void>;
  // #4: scar / not-scar (control) decision, made AFTER preprocessing. Persists to the case manifest
  // without re-running the correction; null = undecided.
  setClassification: (cls: "scar" | "control" | null) => Promise<void>;
  // Timeline step 3 (orange): mark preprocessing manually vetted. Step 7 (green): schedule for training.
  vetPreprocessing: () => Promise<void>;
  scheduleTraining: (scheduled: boolean) => Promise<void>;
  // Before/after "Use original (raw)": discard the correction, make the raw .OCT the working volume +
  // mark it vetted (drops any segmentation; reloads the volume).
  approveRaw: () => Promise<void>;
  // #3 Auto step: re-run the full auto preprocessing on the raw .OCT again (fresh auto detect/warp,
  // keeping the scan's persisted params + classification). Drops any segmentation; reloads the volume.
  rerunPreprocess: () => Promise<void>;
  // Step regression: roll the scan back to `step`, clearing every later step's manifest flag so the
  // user can redo from there (flag-only on the backend; files remain and are overwritten on re-run).
  resetStep: (step: number) => Promise<void>;
  // Step 6: set this scan's scar-subgroup AND confirm it (gates align so the right repeats group together).
  confirmSubgroup: (sub: string) => Promise<void>;
  // Step 6 for a control (no-scar) scan: mark the scar step done without running a detector.
  skipScar: () => Promise<void>;
}

function volumeUrlFor(caseId: string): string {
  // Cache-bust so a re-registered volume reloads in niivue.
  return resourceUrl(`/api/case/${caseId}/volume.nii.gz?t=${Date.now()}`);
}

function hasVolume(info: CaseInfo): boolean {
  const m = info.manifest || {};
  return Boolean(m["corrected_volume"] || m["input_volume"]);
}

export const useCaseStore = create<CaseState>()(
  immer((set, get) => ({
    config: null,
    healthy: false,
    apiError: null,
    caseId: null,
    caseInfo: null,
    volumeUrl: null,
    busy: false,
    exportInfo: null,
    preprocessed: false,

    setPreprocess: async (enabled: boolean) => {
      const id = get().caseId;
      if (!id) return;
      set((s) => {
        s.busy = true;
        s.apiError = null;
      });
      try {
        await api.json(`/api/case/${id}/preprocess`, "POST", JSON.stringify({ enabled }));
        set((s) => {
          s.preprocessed = enabled;
          if (s.caseInfo) s.volumeUrl = volumeUrlFor(id); // cache-bust so the viewer reloads
        });
      } catch (e) {
        set((s) => {
          s.apiError = e instanceof Error ? e.message : String(e);
        });
      } finally {
        set((s) => {
          s.busy = false;
        });
      }
    },

    setClassification: async (cls) => {
      const id = get().caseId;
      if (!id) return;
      // optimistic: reflect the choice immediately, persist to the manifest in the background
      const prev = (get().caseInfo?.manifest as Record<string, unknown> | undefined)?.scar_classification;
      set((s) => { if (s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).scar_classification = cls; });
      try {
        await api.json(`/api/case/${id}/classification`, "POST", JSON.stringify({ classification: cls }));
      } catch (e) {
        // revert the optimistic write so the SAM2 gate (reads manifest.scar_classification) reflects persisted truth
        set((s) => {
          if (s.caseId === id && s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).scar_classification = prev;
          s.apiError = e instanceof Error ? e.message : String(e);
        });
      }
    },

    vetPreprocessing: async () => {
      const id = get().caseId;
      if (!id) return;
      set((s) => { if (s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).preproc_vetted = true; });
      try {
        await api.json(`/api/case/${id}/vet-preprocessing`, "POST", "{}");
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
      }
    },

    scheduleTraining: async (scheduled) => {
      const id = get().caseId;
      if (!id) return;
      set((s) => { if (s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).training_scheduled = scheduled; });
      try {
        await api.json(`/api/case/${id}/training/schedule`, "POST", JSON.stringify({ scheduled }));
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
      }
    },

    confirmSubgroup: async (sub) => {
      const id = get().caseId;
      if (!id) return;
      const v = (sub || "1").trim() || "1";
      set((s) => { if (s.caseInfo) { const mm = s.caseInfo.manifest as Record<string, unknown>; mm.scar_subgroup = v; mm.subgroup_confirmed = true; } });
      try {
        await api.json(`/api/case/${id}/subgroup`, "POST", JSON.stringify({ subgroup: v }));
        await api.json(`/api/case/${id}/subgroup/confirm`, "POST", "{}");
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
      }
    },

    skipScar: async () => {
      const id = get().caseId;
      if (!id) return;
      set((s) => { if (s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).scar_done = true; });
      try {
        await api.json(`/api/case/${id}/scar/skip`, "POST", "{}");
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
      }
    },

    resetStep: async (step) => {
      const id = get().caseId;
      if (!id) return;
      set((s) => { s.busy = true; s.apiError = null; });
      try {
        await api.json(`/api/case/${id}/reset-step`, "POST", JSON.stringify({ step }));
        await get().openCase();                  // refresh the manifest so the timeline drops back
        const wf = useWorkflowStore.getState();  // re-render previews + reflect any dropped segmentation
        wf.set("segVersion", wf.segVersion + 1);
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
      } finally {
        set((s) => { s.busy = false; });
      }
    },

    approveRaw: async () => {
      const id = get().caseId;
      if (!id) return;
      set((s) => { s.busy = true; s.apiError = null; });
      try {
        await api.json(`/api/case/${id}/keep-raw`, "POST", "{}");
        await get().openCase();                 // reload the now-raw working volume (cache-busted URL)
        const wf = useWorkflowStore.getState();  // refresh previews + reflect the dropped segmentation
        wf.set("segVersion", wf.segVersion + 1);
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
      } finally {
        set((s) => { s.busy = false; });
      }
    },

    rerunPreprocess: async () => {
      const id = get().caseId;
      if (!id) return;
      set((s) => { s.busy = true; s.apiError = null; });
      useWorkflowStore.getState().set("status", { kind: "working", title: "Re-running preprocessing",
        detail: "Re-detecting the corneal surface and re-warping from the raw .OCT — this can take a minute." });
      try {
        // Empty params → a NORMAL auto preprocess from the raw .OCT (drops stale border anchors/cache;
        // keeps the scan's persisted params + classification + any sticky manual corrections). The endpoint
        // also drops the segmentation + resets preproc_vetted, so the timeline falls back to Auto (red).
        await api.json(`/api/case/${id}/oct-preprocess`, "POST", JSON.stringify({ params: {} }));
        await get().openCase();                 // reload the re-corrected working volume (cache-busted URL)
        const wf = useWorkflowStore.getState();  // refresh previews + reflect the dropped segmentation
        wf.set("segVersion", wf.segVersion + 1);
        wf.set("status", { kind: "done", title: "Preprocessing re-run", detail: "Fresh auto correction applied — review (Before/after · Fix-columns), then Approve." });
      } catch (e) {
        const m = e instanceof Error ? e.message : String(e);
        set((s) => { s.apiError = m; });
        useWorkflowStore.getState().set("status", { kind: "error", title: "Re-run failed", detail: m });
      } finally {
        set((s) => { s.busy = false; });
      }
    },

    fetchConfig: async () => {
      // The desktop shell spawns its OWN sidecar on launch; importing torch/SAM2 can take ~10–15s, so
      // POLL the health check instead of failing on the first miss (a fresh start would otherwise look
      // broken). Shows a transient "starting…" status; only errors out if it never comes up.
      let ok = await checkHealth();
      for (let i = 0; i < 60 && !ok; i++) {
        set((s) => { s.apiError = "Starting the Python sidecar… (first launch can take ~15s)"; });
        await new Promise((r) => setTimeout(r, 750));
        ok = await checkHealth();
      }
      set((s) => {
        s.healthy = ok;
      });
      if (!ok) {
        set((s) => {
          s.apiError =
            "Couldn't reach the Python sidecar. It may have failed to start — check sidecar.log in the " +
            "app's data folder, and that python3 has the required packages (fastapi, torch, SAM2, SimpleITK).";
        });
        return;
      }
      try {
        const config = await api.getConfig();
        set((s) => {
          s.config = config;
          // Start blank on (re)load: do NOT adopt the persisted last case, so a refresh
          // shows no volume/segmentation until the user loads or opens one.
          s.apiError = null;
        });
      } catch (e) {
        set((s) => {
          s.apiError = e instanceof Error ? e.message : String(e);
        });
      }
    },

    setApiError: (msg) =>
      set((s) => {
        s.apiError = msg;
      }),

    setCaseId: (id) =>
      set((s) => {
        s.caseId = id;
      }),

    clearCase: () => {
      // Empty the viewer right away (wipe-all): no open case → VolumeCanvas drops the volume + overlays.
      useWorkflowStore.getState().resetForCase();
      // Forget the last-opened case so the next openCase of ANY id (including the same one) is treated as a
      // genuine switch and runs the full reset + ascanRateHz re-seed (otherwise a re-opened case inherits stale state).
      _lastOpenedCase = null;
      set((s) => { s.caseId = null; s.caseInfo = null; s.volumeUrl = null; });
    },

    openCase: async () => {
      const id = get().caseId;
      if (!id) return;
      set((s) => {
        s.busy = true;
        s.apiError = null;
      });
      try {
        const info = await api.json<CaseInfo>("/api/case", "POST", JSON.stringify({ case_id: id }));
        if (info.case_id !== _lastOpenedCase) {
          // Switching to a different case: clear the prior case's stale workflow state.
          useWorkflowStore.getState().resetForCase();
          // Re-seed the A-scan rate from THIS case's persisted calibration (manifest.oct_params.ascan_rate_hz)
          // so the Motion tab reflects what the user calibrated for this scan instead of silently defaulting
          // to 70000 and overwriting the stored value on the next Analyze.
          const rate = (info.manifest?.oct_params as Record<string, unknown> | undefined)?.ascan_rate_hz;
          if (typeof rate === "number" && Number.isFinite(rate)) {
            useWorkflowStore.getState().set("ascanRateHz", rate);
          }
          _lastOpenedCase = info.case_id;
        }
        set((s) => {
          s.caseInfo = info;
          s.caseId = info.case_id;
          s.volumeUrl = hasVolume(info) ? volumeUrlFor(info.case_id) : null;
        });
        // Remember this case so the app reopens to it across restarts.
        api.putConfig({ default_case_id: info.case_id }).catch(() => {});
      } catch (e) {
        set((s) => {
          s.apiError = e instanceof Error ? e.message : String(e);
        });
      } finally {
        set((s) => {
          s.busy = false;
        });
      }
    },

    registerVolume: async (path: string) => {
      const id = get().caseId;
      if (!id) return;
      set((s) => {
        s.busy = true;
        s.apiError = null;
      });
      try {
        const info = await api.json<CaseInfo>(
          `/api/case/${id}/volume/register`,
          "POST",
          JSON.stringify({ volume_path: path }),
        );
        set((s) => {
          s.caseInfo = info;
          s.volumeUrl = volumeUrlFor(info.case_id);
        });
      } catch (e) {
        set((s) => {
          s.apiError = e instanceof Error ? e.message : String(e);
        });
      } finally {
        set((s) => {
          s.busy = false;
        });
      }
    },

    exportNnunet: async () => {
      set((s) => {
        s.busy = true;
        s.exportInfo = "Exporting…";
      });
      try {
        const res = await api.json<{ dataset_dir: string; num_training: number }>(
          "/api/export/nnunet",
          "POST",
          JSON.stringify({}),
        );
        set((s) => {
          s.exportInfo = `Exported ${res.num_training} case(s) → ${res.dataset_dir}`;
        });
      } catch (e) {
        set((s) => {
          s.exportInfo = `Export failed: ${e instanceof Error ? e.message : String(e)}`;
        });
      } finally {
        set((s) => {
          s.busy = false;
        });
      }
    },

    uploadVolume: async (file: File) => {
      const id = get().caseId;
      if (!id) return;
      set((s) => {
        s.busy = true;
        s.apiError = null;
      });
      try {
        const info = await api.upload<CaseInfo>(`/api/case/${id}/volume/upload`, [file]);
        set((s) => {
          s.caseInfo = info;
          s.volumeUrl = volumeUrlFor(info.case_id);
        });
      } catch (e) {
        set((s) => {
          s.apiError = e instanceof Error ? e.message : String(e);
        });
      } finally {
        set((s) => {
          s.busy = false;
        });
      }
    },
  })),
);
