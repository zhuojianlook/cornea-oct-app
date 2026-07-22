/* Before/after comparison + iterative-pass stepper. The left panel is the original (raw) scan; the
   right panel steps through every preprocessing pass — pass 1 → 2 → … → final — so the user can watch
   the corneal boundary refine. Pure 2D PNG previews (works with/without WebGL); raw and every pass
   share geometry + slice indices, so they pair 1:1 by slice_index.

   Pass groups: pass 0 = context_raw (left, fixed); pass k = context_iter{k} (intermediate); the final
   pass = context (the working volume's slices). The pass count comes from manifest.oct_iter (written
   by the iterative preprocess); a non-iterative scan has one corrected pass = a plain raw|final view. */

import { useEffect, useMemo, useRef, useState } from "react";
import { ToggleButton, ToggleButtonGroup, Slider, CircularProgress } from "@mui/material";
import { api, resourceUrl } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import type { PreviewImage } from "../../api/types";

const imgSrc = (im?: PreviewImage | null): string =>
  im ? (im.src ? resourceUrl(im.src) : im.data_url) : "";
const ORIENTS = ["axial", "coronal", "sagittal"] as const;

interface OctIter {
  passes?: number;
  metrics?: number[]; // boundary deviation (px) of each chain volume: [raw, pass1, …, passM]
  best_pass?: number; // index of the KEPT volume (the least-deviant one); 0 = raw
  stopped?: string;
  surface_crop?: { n_frames?: number; frames?: number[]; auto?: boolean; n_frames_total?: number };
}

// ── SURFACE-CROP MARKING ───────────────────────────────────────────────────────────────────────────
// A surface-cropped frame is a B-scan whose corneal APEX sits above the acquisition window, so it has no
// anterior surface to detect. The marks belong HERE, on the ORIGINAL (raw) panel, because that is the only
// place the clip is still visible: the whole point of the correction is that the output no longer shows it.
// In the SAGITTAL preview the frame axis runs left→right, so a marked frame is a COLUMN — which is what the
// user is looking for. (Axial shows one frame per image, so a per-column overlay is meaningless there; the
// per-frame toggle in the Fix-axial tool covers that view instead.)
interface CropDetect {
  frames: number[];        // auto-suggested set (detector, run on the RAW volume)
  selected: number[];      // the persisted confirmed set, if the user has edited before
  counts: Record<string, number>;  // per-frame count of sagittal slices flagged clipped → confidence
  n_frames: number;
}

// Orientation + display filter are driven by the single top toolbar in VolumeCanvas (this panel is
// embedded, not a separate sub-UI) — so this component owns ONLY the refinement-pass selector + the
// slice scrubber. The top "⇆ Before/after" toggle is what closes it.
export function BeforeAfterViewer({ orient, filter }: {
  orient: (typeof ORIENTS)[number];
  filter?: string;
}) {
  const caseId = useCaseStore((s) => s.caseId);
  const caseInfo = useCaseStore((s) => s.caseInfo);
  // Re-fetch when previews re-render (e.g. a re-preprocess bumps segVersion).
  const segSig = useWorkflowStore((s) => s.segVersion);
  const wfSet = useWorkflowStore((s) => s.set);

  const octIter = (caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_iter as OctIter | undefined;
  const passCount = Math.max(1, Number(octIter?.passes ?? 1));
  const metrics = Array.isArray(octIter?.metrics) ? (octIter!.metrics as number[]) : [];
  const bestPass = Number(octIter?.best_pass ?? passCount); // 0 = raw, k = pass k

  // The right-panel steps. "Original (raw)" is ALWAYS first — sometimes the raw is good enough to keep
  // directly (Approve original). Then the CORRECTED passes: iterative (passCount>1) lists each
  // context_iter{k} with the KEPT/best pass marked (a worse pass shows a HIGHER deviation, so the user
  // sees why it was dropped); single pass is a plain "preprocessed". best_pass===0 ⇒ raw was auto-kept.
  const steps = useMemo(() => {
    const out: { group: string; label: string; metric: number | null; best: boolean }[] = [];
    out.push({ group: "context_raw", label: "Original (raw)", metric: metrics[0] ?? null, best: bestPass === 0 });
    if (passCount <= 1) {
      out.push({ group: "context", label: "preprocessed", metric: null, best: bestPass !== 0 });
    } else {
      for (let k = 1; k <= passCount; k++) {
        out.push({
          group: `context_iter${k}`,
          label: `pass ${k}${k === bestPass ? " · best" : ""}`,
          metric: metrics[k] ?? null, // boundary deviation (px) of pass k's result
          best: k === bestPass,
        });
      }
    }
    return out;
  }, [passCount, bestPass, metrics]);
  const bestStepIdx = Math.max(0, steps.findIndex((s) => s.best));

  const [passIdx, setPassIdx] = useState(bestStepIdx); // default: the kept (best) pass
  const [raw, setRaw] = useState<PreviewImage[]>([]); // context_raw (left)
  const [after, setAfter] = useState<PreviewImage[]>([]); // selected pass (right)
  const [idx, setIdx] = useState(0);
  const [loading, setLoading] = useState(false);

  // Default to the kept (best) pass when the case (and thus pass set) changes.
  useEffect(() => {
    setPassIdx(bestStepIdx);
  }, [caseId, steps.length, bestStepIdx]);

  // Left panel: the raw snapshot (fetched once per case).
  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;
    api
      .json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/context_raw`)
      .then((r) => !cancelled && setRaw(r.images || []))
      .catch(() => !cancelled && setRaw([]));
    return () => { cancelled = true; };
  }, [caseId, segSig]);

  // Right panel: the currently selected pass's slices (re-fetched when the pass changes).
  const curStep = steps[Math.min(passIdx, steps.length - 1)];
  useEffect(() => {
    if (!caseId || !curStep) return;
    let cancelled = false;
    setLoading(true);
    api
      .json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/${curStep.group}`)
      .then((r) => !cancelled && setAfter(r.images || []))
      .catch(() => !cancelled && setAfter([]))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [caseId, curStep?.group, segSig]);

  const afterImgs = useMemo(
    () =>
      after
        .filter((i) => i.orientation === orient)
        // DESCENDING by slice_index so the scrubber navigates in the SAME direction as the normal niivue view:
        // niivue reorients to RAS-canonical and the OCT affine has all three voxel axes NEGATIVE, so niivue slice
        // s ↔ array slice (n-1-s). The 2-D previews are array-indexed, so scrubbing them ascending ran OPPOSITE to
        // the niivue view (its "slice N" showed a different B-scan). Descending makes panel position p == niivue
        // slice p == the normal view's slice N. Data pairs by slice_index, so the true array index is unchanged.
        .sort((a, b) => Number(b.slice_index ?? 0) - Number(a.slice_index ?? 0)),
    [after, orient],
  );
  const safeIdx = Math.min(idx, Math.max(0, afterImgs.length - 1));
  const cur = afterImgs[safeIdx];
  const rawCur = cur
    ? raw.find((i) => i.orientation === orient && i.slice_index === cur.slice_index)
    : undefined;

  // Center on a middle slice when orientation changes or slices first arrive.
  useEffect(() => {
    if (afterImgs.length) setIdx(Math.floor(afterImgs.length / 2));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orient, after.length]);

  // ── surface-crop marks on the ORIGINAL panel ────────────────────────────────────────────────────
  const ocParams = (caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_params as
    Record<string, unknown> | undefined;
  const scPersisted = useMemo(() => {
    const a = ocParams?.surface_crop_frames;
    return Array.isArray(a) ? (a as number[]).map(Number) : null;
  }, [ocParams]);
  // The pipeline records which frames IT cropped (v0.0.211). Older runs stored only a count, so fall back
  // to the detector for those — same detector, same raw volume, so the set is what it would have chosen.
  const scFromRun = useMemo(() => {
    const f = octIter?.surface_crop?.frames;
    return Array.isArray(f) ? f.map(Number) : null;
  }, [octIter]);
  const wasAutoCropped = octIter?.stopped === "surface_crop";

  const [scDetect, setScDetect] = useState<CropDetect | null>(null);
  const [scMarks, setScMarks] = useState<Set<number> | null>(null);   // null until seeded
  const [scEdit, setScEdit] = useState(false);
  const [scBusy, setScBusy] = useState(false);
  const [scMsg, setScMsg] = useState("");

  // Seed the marks: persisted user set → the set this run actually cropped → the detector's suggestion.
  useEffect(() => {
    if (!caseId || orient !== "sagittal") return;
    let cancel = false;
    const seed = (d: CropDetect | null) => {
      if (cancel) return;
      const chosen = (scPersisted && scPersisted.length ? scPersisted
        : scFromRun && scFromRun.length ? scFromRun
        : d?.selected?.length ? d.selected
        : d?.frames) ?? [];
      setScMarks(new Set(chosen.map(Number)));
    };
    api.json<CropDetect>(`/api/case/${caseId}/oct-surface-crop/detect`, "POST", "{}")
      .then((d) => { if (!cancel) { setScDetect(d); seed(d); } })
      .catch(() => seed(null));   // detector unavailable → still show whatever is persisted
    return () => { cancel = true; };
  }, [caseId, orient, segSig, JSON.stringify(scPersisted), JSON.stringify(scFromRun)]);

  const nFrames = scDetect?.n_frames ?? octIter?.surface_crop?.n_frames_total ?? 0;
  const autoSuggested = useMemo(() => new Set((scDetect?.frames ?? []).map(Number)), [scDetect]);
  const scDirty = useMemo(() => {
    if (!scMarks) return false;
    const base = (scPersisted ?? scFromRun ?? scDetect?.frames ?? []).map(Number).sort((a, b) => a - b).join(",");
    return [...scMarks].sort((a, b) => a - b).join(",") !== base;
  }, [scMarks, scPersisted, scFromRun, scDetect]);

  const applyMarks = async (mode: "manual" | "off" | "auto") => {
    if (!caseId || scBusy) return;
    setScBusy(true); setScMsg(mode === "off" ? "Turning surface crop off…" : "Applying…");
    try {
      const frames = [...(scMarks ?? [])].sort((a, b) => a - b);
      await api.json(`/api/case/${caseId}/oct-preprocess`, "POST", JSON.stringify({
        surface_crop_mode: mode,
        ...(mode === "manual" ? { surface_crop_frames: frames } : {}),
      }));
      await useCaseStore.getState().openCase();
      wfSet("segVersion", segSig + 1);
      setScMsg(mode === "off" ? "Surface crop OFF for this scan."
        : mode === "auto" ? "Reset to AUTO."
        : `Applied ${frames.length} frame(s).`);
    } catch { setScMsg("Failed."); } finally { setScBusy(false); }
  };

  // #6: the <img> fills its (equal) panel area and scales UP to maximize the canvas; objectFit:contain +
  // equal boxes + shared slice geometry render raw and corrected at the SAME size (no "original larger").
  const imgStyle: React.CSSProperties = {
    width: "100%",
    height: "100%",
    objectFit: "contain",
    imageRendering: "pixelated",
    filter: filter || undefined,
    // The 2-D sagittal PNGs lay frames out frame0→left, but niivue's sagittal render (the reference the user
    // trusts) draws frame0 on the RIGHT (affine NIFTI_DIRECTION), so before/after looked LR-mirrored vs the
    // niivue view. Flip the frame axis for sagittal so the 2-D apex sits on the same side as niivue.
    transform: orient === "sagittal" ? "scaleX(-1)" : undefined,
  };
  const panelCol: React.CSSProperties = {
    flex: 1, minWidth: 0, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
  };
  const imgArea: React.CSSProperties = {
    flex: 1, minHeight: 0, width: "100%", display: "flex", alignItems: "center", justifyContent: "center",
  };

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      {/* Pass stepper + slice counter. Orientation / contrast live in the single top toolbar; this strip
          only carries what's unique to before/after — the refinement-pass selector. */}
      <div
        className="flex items-center gap-2 px-3 py-1 border-b flex-wrap"
        style={{ minHeight: 32, borderColor: "var(--c-border)", background: "var(--c-surface)" }}
      >
        {steps.length > 1 ? (
          <>
            <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
              refinement pass:
            </span>
            <ToggleButtonGroup
              size="small"
              exclusive
              value={Math.min(passIdx, steps.length - 1)}
              onChange={(_, v) => v !== null && setPassIdx(v as number)}
            >
              {steps.map((s, i) => {
                const sel = i === Math.min(passIdx, steps.length - 1);
                return (
                  <ToggleButton key={s.group} value={i} sx={{ py: 0.1, px: 1, fontSize: 11, textTransform: "none", gap: 0.5 }}>
                    {/* #7: spinner on the SELECTED pass while its previews load, so the click has feedback. */}
                    {sel && loading && <CircularProgress size={11} color="inherit" />}
                    {s.label}
                  </ToggleButton>
                );
              })}
            </ToggleButtonGroup>
            {curStep?.metric != null && (
              <span className="text-[11px]" style={{ color: curStep.best ? "var(--c-green)" : "var(--c-text-dim)" }}>
                boundary deviation {curStep.metric.toFixed(2)} px{curStep.best ? " · kept (best)" : ""}
              </span>
            )}
            {octIter?.stopped && (
              <span className="text-[10px]" style={{ color: "var(--c-text-dim)", opacity: 0.8 }}>
                · lower = flatter · stopped: {octIter.stopped}
              </span>
            )}
          </>
        ) : (
          <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
            original (raw) vs preprocessed — scrub the slice below
          </span>
        )}
        <div className="flex-1" />
        {cur && (
          <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
            slice {safeIdx + 1} / {afterImgs.length}
          </span>
        )}
      </div>

      {/* Surface-crop strip. Sagittal only — that is the view where a frame reads as a column. Shown
          whenever this scan has (or could have) a clip, so an auto-applied crop is visible without hunting. */}
      {orient === "sagittal" && nFrames > 1 && (scMarks?.size || autoSuggested.size || wasAutoCropped) ? (
        <div className="flex items-center gap-2 px-3 py-1 border-b flex-wrap"
             style={{ minHeight: 30, borderColor: "var(--c-border)", background: "rgba(91,192,255,0.07)" }}>
          <span className="text-[11px]" style={{ fontWeight: 600, color: "#5bc0ff" }}>surface crop</span>
          <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
            {wasAutoCropped ? "auto-applied on this scan · " : ""}
            {scMarks?.size ?? 0} marked{autoSuggested.size ? ` · ${autoSuggested.size} suggested` : ""}
          </span>
          <ToggleButton size="small" value="edit" selected={scEdit} disabled={scBusy}
                        onChange={() => setScEdit((v) => !v)}
                        style={{ padding: "1px 8px", fontSize: 11, textTransform: "none" }}>
            {scEdit ? "editing — click a column" : "edit marks"}
          </ToggleButton>
          {scDetect?.frames?.length ? (
            <button className="text-[11px]" disabled={scBusy}
                    onClick={() => setScMarks(new Set(scDetect.frames.map(Number)))}
                    style={{ padding: "2px 8px", borderRadius: 6, border: "1px solid var(--c-border)",
                             background: "var(--c-surface)", color: "var(--c-text)", cursor: "pointer" }}>
              use detector&apos;s {scDetect.frames.length}
            </button>
          ) : null}
          <button className="text-[11px]" disabled={scBusy || !scMarks?.size}
                  onClick={() => setScMarks(new Set())}
                  style={{ padding: "2px 8px", borderRadius: 6, border: "1px solid var(--c-border)",
                           background: "var(--c-surface)", color: "var(--c-text)", cursor: "pointer" }}>
            clear
          </button>
          <span style={{ flex: 1 }} />
          {scMsg && <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>{scMsg}</span>}
          {scBusy && <CircularProgress size={13} />}
          <button className="text-[11px]" disabled={scBusy}
                  onClick={() => applyMarks("off")}
                  title="Never surface-crop this scan, even if the detector flags it (sticks across re-runs)"
                  style={{ padding: "2px 8px", borderRadius: 6, border: "1px solid var(--c-border)",
                           background: "var(--c-surface)", color: "var(--c-text)", cursor: "pointer" }}>
            turn off
          </button>
          <button className="text-[11px]" disabled={scBusy || !scDirty}
                  onClick={() => applyMarks("manual")}
                  title="Re-run preprocessing using exactly the marked frames"
                  style={{ padding: "2px 10px", borderRadius: 6,
                           border: `1px solid ${scDirty ? "#5bc0ff" : "var(--c-border)"}`,
                           background: scDirty ? "#5bc0ff" : "var(--c-surface)",
                           color: scDirty ? "#00263a" : "var(--c-text)", cursor: "pointer" }}>
            apply &amp; re-run
          </button>
        </div>
      ) : null}

      <div className="flex-1 min-h-0 flex items-center justify-center p-3">
        {!cur ? (
          <div className="text-center" style={{ color: "var(--c-text-dim)" }}>
            <div style={{ fontSize: 13 }}>{loading ? "Loading…" : "No preprocessed slices yet."}</div>
            {!loading && (
              <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>Preprocess the scan first.</div>
            )}
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              gap: 10,
              width: "100%",
              height: "100%",
              alignItems: "stretch",
              justifyContent: "center",
            }}
          >
            <div style={panelCol}>
              <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                original (raw){orient === "sagittal" && scMarks && scMarks.size > 0
                  ? ` — ${scMarks.size} surface-cropped frame${scMarks.size === 1 ? "" : "s"}` : ""}
              </span>
              <div style={imgArea}>
                {rawCur ? (
                  <CropMarkedImage
                    src={imgSrc(rawCur)} style={imgStyle}
                    // Marks only make sense on sagittal, where the frame axis runs across the image.
                    active={orient === "sagittal" && nFrames > 1}
                    flipped={orient === "sagittal"}
                    nFrames={nFrames} marks={scMarks} auto={autoSuggested} counts={scDetect?.counts}
                    editable={scEdit}
                    onToggle={(f) => setScMarks((prev) => {
                      const o = new Set(prev ?? []);
                      if (o.has(f)) o.delete(f); else o.add(f);
                      return o;
                    })}
                  />
                ) : (
                  <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                    no raw slice here
                  </span>
                )}
              </div>
            </div>
            <div style={panelCol}>
              <span className="text-[11px]" style={{ color: "var(--c-green)" }}>
                {curStep?.label ?? "preprocessed"}
              </span>
              <div style={imgArea}>
                <img src={imgSrc(cur)} alt="preprocessed" draggable={false} style={imgStyle} />
              </div>
            </div>
          </div>
        )}
      </div>

      {afterImgs.length > 1 && (
        <div className="px-4 py-2 border-t flex items-center gap-3" style={{ borderColor: "var(--c-border)" }}>
          <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
            slice
          </span>
          <Slider
            size="small"
            min={0}
            max={afterImgs.length - 1}
            value={safeIdx}
            onChange={(_, v) => setIdx(v as number)}
          />
        </div>
      )}
    </div>
  );
}

// Raw preview with the surface-crop frame marks drawn ON the image.
// The <img> uses objectFit:contain, so the rendered picture is usually SMALLER than its box — the overlay
// must therefore track the CONTAINED rect (derived from naturalWidth/Height), not the box, or the columns
// would drift out of register with the anatomy. The sagittal preview is also scaleX(-1)-flipped to match
// niivue's frame direction, so the overlay lives inside the same flipped element and needs no separate
// mirroring: frame f is simply at x = f / nFrames.
function CropMarkedImage({ src, style, active, flipped, nFrames, marks, auto, counts, editable, onToggle }: {
  src: string; style: React.CSSProperties; active: boolean; flipped: boolean; nFrames: number;
  marks: Set<number> | null; auto: Set<number>; counts?: Record<string, number>;
  editable: boolean; onToggle: (f: number) => void;
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [nat, setNat] = useState({ w: 0, h: 0 });
  const [box, setBox] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = boxRef.current; if (!el) return;
    const upd = () => setBox({ w: el.clientWidth, h: el.clientHeight });
    const ro = new ResizeObserver(upd); ro.observe(el); upd();
    return () => ro.disconnect();
  }, []);
  // The overlay needs the image's INTRINSIC size to place the objectFit:contain rect. Reading it from
  // onLoad ALONE is a bug: the raw preview is normally already in the cache, so the image is `complete`
  // before React attaches the handler and onLoad never fires — nat stays 0, the contain-rect is 0-wide,
  // and the marks are silently not rendered. (They then appeared on the next re-render, which is why
  // toggling "edit marks" seemed to summon them.) Read it directly whenever src changes, and keep
  // onLoad for the genuinely-uncached case.
  const readNat = () => {
    const im = imgRef.current;
    if (im && im.naturalWidth > 0 && im.naturalHeight > 0) {
      setNat((p) => (p.w === im.naturalWidth && p.h === im.naturalHeight
        ? p : { w: im.naturalWidth, h: im.naturalHeight }));
    }
  };
  useEffect(readNat, [src]);
  // the objectFit:contain rect of the image inside the box
  let iw = 0, ih = 0, ox = 0, oy = 0;
  if (nat.w > 0 && nat.h > 0 && box.w > 0 && box.h > 0) {
    const s = Math.min(box.w / nat.w, box.h / nat.h);
    iw = nat.w * s; ih = nat.h * s; ox = (box.w - iw) / 2; oy = (box.h - ih) / 2;
  }
  const maxCount = useMemo(() => {
    const v = Object.values(counts ?? {}); return v.length ? Math.max(...v) : 0;
  }, [counts]);
  const pick = (clientX: number) => {
    const el = boxRef.current; if (!el || iw <= 0 || nFrames < 1) return;
    const r = el.getBoundingClientRect();
    let fx = (clientX - r.left - ox) / iw;              // 0..1 across the IMAGE
    if (fx < 0 || fx > 1) return;
    if (flipped) fx = 1 - fx;                            // undo the scaleX(-1) presentation
    onToggle(Math.max(0, Math.min(nFrames - 1, Math.floor(fx * nFrames))));
  };
  return (
    <div ref={boxRef} style={{ position: "relative", width: "100%", height: "100%" }}>
      <img ref={imgRef} src={src} alt="raw" draggable={false} style={style} onLoad={readNat} />
      {active && iw > 0 && (
        <div style={{ position: "absolute", left: ox, top: oy, width: iw, height: ih,
                      transform: flipped ? "scaleX(-1)" : undefined,
                      pointerEvents: editable ? "auto" : "none",
                      cursor: editable ? "pointer" : "default" }}
             onPointerDown={(e) => { if (editable) pick(e.clientX); }}>
          {Array.from({ length: nFrames }, (_, f) => {
            const on = marks?.has(f) ?? false;
            const sug = auto.has(f);
            if (!on && !sug) return null;
            // confidence = how many sagittal slices flagged this frame; drives opacity so a marginal
            // frame reads differently from one the detector is sure about
            const c = Number(counts?.[String(f)] ?? 0);
            const conf = maxCount > 0 ? Math.min(1, c / maxCount) : 1;
            return (
              <div key={f} title={`frame ${f}${c ? ` — clipped in ${c} slices` : ""}${on ? " (marked)" : " (suggested)"}`}
                   style={{ position: "absolute", top: 0, bottom: 0,
                            left: `${(f / nFrames) * 100}%`, width: `${(1 / nFrames) * 100}%`,
                            background: on ? `rgba(91,192,255,${0.25 + 0.35 * conf})` : "rgba(255,193,7,0.16)",
                            borderLeft: on ? "1px solid rgba(91,192,255,0.9)" : "none",
                            borderRight: on ? "1px solid rgba(91,192,255,0.9)" : "none" }} />
            );
          })}
        </div>
      )}
    </div>
  );
}
