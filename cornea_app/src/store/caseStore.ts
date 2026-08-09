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
  // Defect TYPE the user tagged this region with (so the assistant knows which KIND of problem it is):
  // "edge_detection" | "curvature" | "surface_roughness" | any free-text. Absent on older (untagged) marks.
  tag?: string;
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
  vetPreprocessing: (corpusEligible?: boolean) => Promise<void>;
  // Approve the preprocessing AS-IS: mark it vetted WITHOUT applying any auto-detected proposals (the user is
  // accepting the current output and declining the de-tilt/crop/surface-crop). Non-destructive — identical to
  // vetPreprocessing(); kept as a distinct name so the button intent is explicit. (Applying corrections is the
  // SEPARATE applyCorrections() action, so an unwanted/false proposal never blocks a plain approve — e.g. a
  // spurious de-tilt on an off-centre dome.)
  // corpusEligible (default true): when this scan carries a manual border correction, Approve records it as
  // CONFIRMED ground truth for the algorithm-training corpus; pass false to keep the correction applied but
  // EXCLUDE an idealised case (e.g. a motion-corrupted scan whose hand-drawn border is not real geometry).
  approvePreprocessing: (corpusEligible?: boolean) => Promise<void>;
  // Apply the auto-detected proposals (manifest.oct_proposals: de-tilt / crop / surface-crop): re-preprocess
  // with apply_proposals:true to BAKE the corrections into a fresh warped output. Does NOT auto-vet — the new
  // output resets to "Preprocessed [Auto]" (red) so the user re-inspects it, then approves as-is. Separate from
  // approvePreprocessing so approve and apply are independent actions (the OS(4) fix).
  applyCorrections: () => Promise<void>;
  // Reviewer issue flags naming the defect ("zeroed-frames", "still-clipped", …); persisted to
  // manifest.review_flags so the assistant can find flagged scans. The backend accepts any slug matching
  // ^[a-z][a-z0-9-]{1,31}$ and DROPS the rest, so anything else set here shows optimistically but will not
  // survive a reload; see api/reviewFlags.ts for the known vocabulary. Metadata only — no volume/seg change.
  setReviewFlags: (flags: string[]) => Promise<void>;
  // Defect-marking: persist the full list of per-slice wrong-column marks to manifest.defect_marks so the
  // assistant reads exactly which frames/columns are wrong. Optimistic + serialized per case (mirrors flags).
  setDefectMarks: (marks: DefectMark[]) => Promise<void>;
  // "Difficult scan" toggle → manifest.difficult_scan (needs manual help). Optimistic, mirrors setReviewFlags.
  // `reason` is the reviewer's own words for WHY, persisted as manifest.difficult_reason and cleared with the
  // flag; omitting it leaves any existing reason alone so a plain toggle-on does not wipe one.
  setDifficult: (difficult: boolean, reason?: string) => Promise<void>;
  /** Persist drawn border anchors WITHOUT the user pressing "Confirm border". Used by the review loop so
   *  a correction + Reject is one click; the backend then harvests them as border_gt when the difficult
   *  flag is written, which is why this must complete BEFORE setDifficult. */
  commitBorderAnchors: (anchors: Record<string, Record<string, number>>, parabola: boolean,
                        parabolaSlices?: string[]) => Promise<void>;
  /** Persist crop marks (surface-crop frames / crop region) WITHOUT re-running the pipeline — the cheap
   *  path the review loop needs, since "Confirm & re-run" costs a full ~2 min reprocess. */
  commitOctMarks: (cropFrames: number[] | null, cropRegion: { lateral: [number, number]; frames: number[] } | null,
                   postAnchors?: Record<string, Record<string, number>> | null) => Promise<void>;
  /** Kick off the guarded re-run over the scans corrected so far. Returns immediately; the pass runs
   *  in the background on the sidecar and only re-writes a scan that measures better. */
  startReprocessBatch: () => Promise<{ started: number } | null>;
  /** Search + adopt better DETECTOR parameters from every correction drawn so far (guarded). */
  startDetectorTune: () => Promise<{ started: number; n_points?: number; note?: string } | null>;
  /** Re-run THIS scan against the corrections drawn on it, and stay on it. The per-scan iteration step. */
  rerunWithCorrections: () => Promise<boolean>;
  // "Surface-crop" manual mark → manifest.surface_crop_manual (human review of the auto-detected clipped-cornea set).
  setSurfaceCrop: (surfaceCrop: boolean) => Promise<void>;
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

    vetPreprocessing: async (corpusEligible = true) => {
      const id = get().caseId;
      if (!id) return;
      set((s) => { if (s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).preproc_vetted = true; });
      try {
        // corpus_eligible: if the scan has a manual border correction, approving confirms it as GROUND TRUTH
        // for the training corpus (unless excluded here, e.g. an idealised motion scan).
        await api.json(`/api/case/${id}/vet-preprocessing`, "POST", JSON.stringify({ corpus_eligible: corpusEligible }));
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
      }
    },

    // Approve AS-IS: vet the current output WITHOUT applying any proposals (declining the auto de-tilt/crop/
    // surface-crop). Non-destructive; keeps any segmentation. Applying corrections is the separate action below.
    approvePreprocessing: async (corpusEligible = true) => {
      await get().vetPreprocessing(corpusEligible);
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
      // Evict once this case's writes have drained, so the map can't retain one settled promise per case
      // touched for the whole session. Only the LAST write clears the slot — evicting mid-chain would drop
      // the ordering guarantee for a write already queued behind this one.
      void next.finally(() => { if (_reviewFlagChain.get(id) === next) _reviewFlagChain.delete(id); });
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
      void next.finally(() => { if (_defectMarkChain.get(id) === next) _defectMarkChain.delete(id); });   // see setReviewFlags
      await next;
    },

    startReprocessBatch: async () => {
      try {
        const r = await api.json<{ started: number }>("/api/review/reprocess-batch", "POST", "{}");
        useWorkflowStore.getState().set("status", { kind: "working", title: "Reprocessing with your corrections",
          detail: `${r.started} scan(s) queued. Only scans that measure better are re-written; the rest keep what they have.` });
        return r;
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
        return null;
      }
    },

    // PER-SCAN ITERATION — correct a slice, re-run THIS scan against it, look again, repeat.
    //
    // WHY THIS AND NOT PARAMETER TUNING. Measured across six eyes, the detector's whole parameter space moves
    // the surface by 1.5 px on a typical slice and 2.6 px on the most responsive one — while the detector sat
    // ~23 px from the reviewer's corrections on the scan they had actually corrected. Tuning cannot close a
    // gap an order of magnitude wider than its own authority. Anchors can: redetect_surface is seeded by the
    // drag and, for a shaped curve, follows it exactly (seed window 0), so the correction IS the surface
    // rather than a hint the detector may decline. There is no leverage ceiling.
    //
    // Stays on the scan deliberately. The verdict buttons advance; this one is the iteration, and a loop you
    // cannot go round twice without losing your place is not a loop.
    rerunWithCorrections: async () => {
      const id = get().caseId;
      if (!id) return false;
      set((s) => { s.busy = true; s.apiError = null; });
      useWorkflowStore.getState().set("status", { kind: "working", title: "Re-running with your corrections",
        detail: "Flattening this scan to the surface you drew — about two minutes." });
      try {
        // GENERALIZE FIRST, then warp. This is the difference between a correction reaching the output and
        // being outvoted by it.
        //
        // The flatten shifts each frame by the MEDIAN of the surface across all 513 laterals — one rigid
        // shift per B-scan, because a B-scan is captured instantaneously and may only be translated. A
        // correction confined to a few laterals is therefore 1-in-513 per frame and the median ignores it:
        // measured on a real scan, corrections of 14-24 px moved the applied shift by 0.89 px.
        // generalize_surface learns the residual (correction − auto) and interpolates it across laterals, so
        // the correction reaches enough of them to move the median. Measured on the same scan: anchors
        // spanning the volume reached 84-100% of laterals and shifted by 3.4 px, against 16% and 0.9 px for
        // the local ±20-slice band that use_redetect alone applies.
        //
        // It has to run AFTER committing the anchors: oct-border-redetect deliberately clears the generalize
        // flag (a fresh local Confirm exits generalize mode), so setting it first would simply be undone.
        // Non-fatal — if generalizing fails the warp still runs off the local band, which is the old behaviour
        // rather than no behaviour.
        try {
          await api.json(`/api/case/${id}/oct-border-generalize`, "POST", "{}");
        } catch { /* fall through to the local-band redetect */ }
        // GUIDED RE-DETECTION, GUARDED. The generalized surface above is only a PRIOR — it says where to
        // look. This re-detects every slice from the image inside a window around it, so the 512 slices you
        // did not draw on are DETECTED rather than interpolated. Measured against the reviewer's own
        // corrections it cut the median error on a held-out slice from 11.1 px to 3.0 px.
        // It is kept only if it measures better than auto (by the anchors where they exist, otherwise by
        // gradient + delivered shift-roughness) — unguarded it improved two approved scans and regressed two.
        // If it loses, the endpoint reverts the scan and the warp below uses the surface it would have used.
        let guided: { accepted?: boolean; why?: string } | null = null;
        try {
          guided = await api.json(`/api/case/${id}/oct-border-guided`, "POST", "{}");
        } catch { /* guard unavailable → warp to the generalized/local surface, i.e. the old behaviour */ }
        // use_redetect: flatten to the CONFIRMED surface — which _redetect_surface_cached now serves from
        // generalize.npz because the flag above is set. The editor previews the same surface, so what you
        // judged is what you get.
        await api.json(`/api/case/${id}/oct-preprocess`, "POST", JSON.stringify({ use_redetect: true }));
        await get().openCase();                  // reload the re-corrected volume (cache-busted URL)
        const wf = useWorkflowStore.getState();
        wf.set("segVersion", wf.segVersion + 1);  // re-render previews
        wf.set("status", { kind: "done", title: "Re-run complete",
          detail: (guided?.accepted
                    ? `Detection improved and kept — ${guided.why}. `
                    : (guided ? `Guided detection did NOT beat auto (${guided.why}), so the scan keeps the better surface. ` : ""))
                 + "Inspect it — correct again, Approve, or Skip." });
        return true;
      } catch (e) {
        const m = e instanceof Error ? e.message : String(e);
        set((s) => { s.apiError = m; });
        useWorkflowStore.getState().set("status", { kind: "error", title: "Re-run failed", detail: m });
        return false;
      } finally {
        set((s) => { s.busy = false; });
      }
    },

    // GLOBAL DETECTOR TUNING — the step that makes the review loop converge. Unlike startReprocessBatch,
    // which re-applies each scan's own corrections to that scan, this searches the DETECTOR's parameters for
    // a setting that reproduces the accumulated corrections better and adopts it for every scan, guarded so
    // approved scans are not disturbed. Long-running; poll /api/review/tune-status.
    startDetectorTune: async () => {
      try {
        const r = await api.json<{ started: number; n_points?: number; note?: string }>(
          "/api/review/tune-detector", "POST", "{}");
        useWorkflowStore.getState().set("status", r.started
          ? { kind: "working", title: "Improving the detector from your corrections",
              detail: `Searching detector settings against ${r.n_points ?? 0} corrected point(s). `
                + "Nothing changes unless it beats the current detector AND leaves approved scans alone." }
          : { kind: "done", title: "Nothing to learn from yet", detail: r.note || "No corrections recorded." });
        return r;
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
        return null;
      }
    },

    commitOctMarks: async (cropFrames, cropRegion, postAnchors) => {
      const id = get().caseId;
      if (!id || (cropFrames === null && cropRegion === null && !postAnchors)) return;
      const body: Record<string, unknown> = {};
      if (cropFrames !== null) body.surface_crop_frames = cropFrames;
      if (cropRegion !== null) body.crop_region = cropRegion;
      if (postAnchors) body.crop_post_anchors = postAnchors;
      await api.json(`/api/case/${id}/oct-marks`, "POST", JSON.stringify(body));
    },

    commitBorderAnchors: async (anchors, parabola, parabolaSlices) => {
      const id = get().caseId;
      if (!id || !anchors || Object.keys(anchors).length === 0) return;
      // Same endpoint "Confirm border" uses, so a correction committed this way is byte-identical to a
      // confirmed one — there is no second, weaker kind of ground truth.
      await api.json(`/api/case/${id}/oct-border-redetect`, "POST",
        JSON.stringify({ border_pass: 1, border_anchors: anchors, parabola,
                         parabola_slices: parabolaSlices ?? null }));
    },

    setDifficult: async (difficult, reason) => {
      const id = get().caseId;
      if (!id) return;
      const text = (reason ?? "").trim();
      set((s) => {
        if (!s.caseInfo) return;
        const m = s.caseInfo.manifest as Record<string, unknown>;
        m.difficult_scan = difficult;
        // Mirror the server's rule locally so the optimistic view matches what lands on disk: clearing the
        // flag clears the reason, and a reason is only recorded when one was actually given.
        if (!difficult) m.difficult_reason = null;
        else if (text) m.difficult_reason = { text, ts: Date.now() / 1000 };
      });
      try {
        const body: Record<string, unknown> = { difficult };
        if (text) body.reason = text;
        await api.json(`/api/case/${id}/difficult`, "POST", JSON.stringify(body));
      } catch (e) {
        set((s) => { s.apiError = e instanceof Error ? e.message : String(e); });
      }
    },

    setSurfaceCrop: async (surfaceCrop) => {
      const id = get().caseId;
      if (!id) return;
      set((s) => { if (s.caseInfo) (s.caseInfo.manifest as Record<string, unknown>).surface_crop_manual = surfaceCrop; });
      try {
        await api.json(`/api/case/${id}/surface-crop`, "POST", JSON.stringify({ surface_crop: surfaceCrop }));
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
