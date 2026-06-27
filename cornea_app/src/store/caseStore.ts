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
      set((s) => { if (s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).scar_classification = cls; });
      try {
        await api.json(`/api/case/${id}/classification`, "POST", JSON.stringify({ classification: cls }));
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
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
