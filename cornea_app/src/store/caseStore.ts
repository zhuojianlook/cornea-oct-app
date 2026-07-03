import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { api, checkHealth, resourceUrl } from "../api/client";
import type { AppConfig, CaseInfo } from "../api/types";
import { octProposals } from "../api/lifecycle";
import { useWorkflowStore } from "./workflowStore";

// The last case openCase() actually switched to — so we only reset the per-case
// workflow state on a genuine case CHANGE, not on a same-case reopen/refresh.
let _lastOpenedCase: string | null = null;

// Per-case serialization of review-flag writes: each toggle POSTs the FULL flag set, so rapid toggles must
// land on disk IN CLICK ORDER (out-of-order arrival on the sidecar threadpool would let an older/smaller set
// win and silently drop a flag on reload). Mirrors OctLoader's persistClassification chain.
const _reviewFlagChain = new Map<string, Promise<unknown>>();
// Same per-case serialization for defect-mark writes (each write POSTs the FULL mark list, so rapid marks
// across slices must land on disk in order — an out-of-order clobber would drop marks on reload).
const _defectMarkChain = new Map<string, Promise<unknown>>();

// A single defect mark: the columns of one sagittal/axial slice the user flagged as WRONG (columns = the
// non-depth in-plane axis: frame indices for sagittal, lateral indices for axial). Persisted to
// manifest.defect_marks so the assistant reads exactly which frames/columns to fix.
export interface DefectMark {
  orient: "sagittal" | "axial";
  slice: number;
  cols: number[];
}

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
  // Approve the preprocessing AS-IS: mark it vetted WITHOUT applying any auto-detected proposals (the user is
  // accepting the current output and declining the de-tilt/crop/surface-crop). Non-destructive — identical to
  // vetPreprocessing(); kept as a distinct name so the button intent is explicit. (Applying corrections is the
  // SEPARATE applyCorrections() action, so an unwanted/false proposal never blocks a plain approve — e.g. a
  // spurious de-tilt on an off-centre dome.)
  approvePreprocessing: () => Promise<void>;
  // Apply the auto-detected proposals (manifest.oct_proposals: de-tilt / crop / surface-crop): re-preprocess
  // with apply_proposals:true to BAKE the corrections into a fresh warped output. Does NOT auto-vet — the new
  // output resets to "Preprocessed [Auto]" (red) so the user re-inspects it, then approves as-is. Separate from
  // approvePreprocessing so approve and apply are independent actions (the OS(4) fix).
  applyCorrections: () => Promise<void>;
  // Reviewer issue flags (#A/#B/#C) set during the cybernetic loop; persisted to manifest.review_flags
  // so the assistant can find flagged scans. Metadata only — does not affect the volume/segmentation.
  setReviewFlags: (flags: string[]) => Promise<void>;
  // Defect-marking: persist the full list of per-slice wrong-column marks to manifest.defect_marks so the
  // assistant reads exactly which frames/columns are wrong. Optimistic + serialized per case (mirrors flags).
  setDefectMarks: (marks: DefectMark[]) => Promise<void>;
  // "Difficult scan" toggle → manifest.difficult_scan (needs manual help). Optimistic, mirrors setReviewFlags.
  setDifficult: (difficult: boolean) => Promise<void>;
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

    // Approve AS-IS: vet the current output WITHOUT applying any proposals (declining the auto de-tilt/crop/
    // surface-crop). Non-destructive; keeps any segmentation. Applying corrections is the separate action below.
    approvePreprocessing: async () => {
      await get().vetPreprocessing();
    },

    // Apply the auto-detected corrections (bake in de-tilt/crop/surface-crop) as a fresh warp. Does NOT auto-vet
    // — the re-preprocessed output resets to "Preprocessed [Auto]" so the user re-inspects it, then approves.
    applyCorrections: async () => {
      const id = get().caseId;
      if (!id) return;
      const hasProposal = octProposals(get().caseInfo?.manifest ?? null).hasProposal;
      if (!hasProposal) return;                 // nothing to apply
      set((s) => { s.busy = true; s.apiError = null; });
      useWorkflowStore.getState().set("status", { kind: "working", title: "Applying auto-corrections",
        detail: "Baking in the detected de-tilt / crop / surface-crop and re-warping from the raw .OCT — this can take a minute." });
      try {
        // apply_proposals is merged into this ONE run's params server-side (popped before persist), so it
        // bakes the corrections without becoming a sticky param. No vet — the fresh output is re-reviewed.
        await api.json(`/api/case/${id}/oct-preprocess`, "POST", JSON.stringify({ params: { apply_proposals: true } }));
        await get().openCase();                 // reload the now-corrected working volume (cache-busted URL)
        const wf = useWorkflowStore.getState();  // refresh previews + reflect the dropped segmentation
        wf.set("segVersion", wf.segVersion + 1);
        wf.set("status", { kind: "done", title: "Corrections applied", detail: "Corrections baked in — re-inspect the fresh output, then Approve preprocessing." });
      } catch (e) {
        const m = e instanceof Error ? e.message : String(e);
        set((s) => { s.apiError = m; });
        useWorkflowStore.getState().set("status", { kind: "error", title: "Apply corrections failed", detail: m });
      } finally {
        set((s) => { s.busy = false; });
      }
    },

    setReviewFlags: async (flags) => {
      const id = get().caseId;
      if (!id) return;
      set((s) => { if (s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).review_flags = flags; });
      // Chain after this case's previous flag write so rapid toggles land on disk in the clicked order (the
      // newest full set is always the last to reach the manifest); avoids an out-of-order clobber dropping a flag.
      const prev = _reviewFlagChain.get(id) ?? Promise.resolve();
      const next = prev
        .catch(() => undefined)
        .then(() => api.json(`/api/case/${id}/review-flag`, "POST", JSON.stringify({ flags }))
          .catch((e) => { set((s) => { s.apiError = e instanceof Error ? e.message : String(e); }); }));
      _reviewFlagChain.set(id, next);
      await next;
    },

    setDefectMarks: async (marks) => {
      const id = get().caseId;
      if (!id) return;
      // optimistic: reflect the marks in the manifest immediately (the viewer + sidebar read from there)
      set((s) => { if (s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).defect_marks = marks; });
      // Serialize per case so rapid marks across slices land on disk in order (newest full list wins last).
      const prev = _defectMarkChain.get(id) ?? Promise.resolve();
      const next = prev
        .catch(() => undefined)
        .then(() => api.json(`/api/case/${id}/defect-marks`, "POST", JSON.stringify({ marks }))
          .catch((e) => { set((s) => { s.apiError = e instanceof Error ? e.message : String(e); }); }));
      _defectMarkChain.set(id, next);
      await next;
    },

    setDifficult: async (difficult) => {
      const id = get().caseId;
      if (!id) return;
      set((s) => { if (s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).difficult_scan = difficult; });
      try {
        await api.json(`/api/case/${id}/difficult`, "POST", JSON.stringify({ difficult }));
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
        const wf = useWorkflowStore.getState();
        // #5 — a rollback is a SAME-case openCase, so resetForCase() never ran; explicitly restore the
        // per-step editing state so the rolled-back step behaves EXACTLY like reaching it fresh: leave any
        // paint/correction mode and clear the inspect selection. segLoaded is reset to false here and then
        // RESTORED by VolumeCanvas's reload (openCase cache-busts volumeUrl → it reloads the volume and
        // calls tryLoadExistingSegmentation, which sets segLoaded true again IF the step still has a
        // labelmap). Rolling back below Cornea deletes the labelmap, so it correctly stays false.
        wf.set("correcting", false);
        wf.set("corneaOnlyPaint", false);
        wf.set("selectedStep", null);
        wf.set("segLoaded", false);
        // Clear the same per-step state resetForCase() would on a fresh load, so a rolled-back step doesn't
        // carry stale overlay/scar/hint state from the step we rolled back FROM.
        wf.set("showSegmentation", false);
        wf.set("scarMetrics", null);
        wf.set("hintMode", false);
        wf.set("scarHints", []);
        wf.set("scarEditMode", false);
        wf.set("scarErase", false);
        wf.set("segVersion", wf.segVersion + 1);   // re-render previews + reflect any dropped segmentation
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
