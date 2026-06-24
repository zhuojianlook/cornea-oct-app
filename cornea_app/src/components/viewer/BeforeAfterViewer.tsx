/* Before/after comparison + iterative-pass stepper. The left panel is the original (raw) scan; the
   right panel steps through every preprocessing pass — pass 1 → 2 → … → final — so the user can watch
   the corneal boundary refine. Pure 2D PNG previews (works with/without WebGL); raw and every pass
   share geometry + slice indices, so they pair 1:1 by slice_index.

   Pass groups: pass 0 = context_raw (left, fixed); pass k = context_iter{k} (intermediate); the final
   pass = context (the working volume's slices). The pass count comes from manifest.oct_iter (written
   by the iterative preprocess); a non-iterative scan has one corrected pass = a plain raw|final view. */

import { useEffect, useMemo, useState } from "react";
import { ToggleButton, ToggleButtonGroup, Slider } from "@mui/material";
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

  const octIter = (caseInfo?.manifest as Record<string, unknown> | undefined)?.oct_iter as OctIter | undefined;
  const passCount = Math.max(1, Number(octIter?.passes ?? 1));
  const metrics = Array.isArray(octIter?.metrics) ? (octIter!.metrics as number[]) : [];
  const bestPass = Number(octIter?.best_pass ?? passCount); // 0 = raw, k = pass k

  // The right-panel steps. Iterative (passCount>1): one per CORRECTED pass (context_iter{k}), with
  // the KEPT/best pass marked — a worse pass shows a HIGHER deviation, so the user sees why it was
  // dropped. Single pass: a plain raw|preprocessed (the working "context" group; no per-pass groups).
  const steps = useMemo(() => {
    const out: { group: string; label: string; metric: number | null; best: boolean }[] = [];
    if (passCount <= 1) {
      out.push({ group: "context", label: "preprocessed", metric: null, best: true });
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
        .sort((a, b) => Number(a.slice_index ?? 0) - Number(b.slice_index ?? 0)),
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

  const imgStyle: React.CSSProperties = {
    maxHeight: "calc(100% - 28px)",
    maxWidth: "100%",
    objectFit: "contain",
    imageRendering: "pixelated",
    filter: filter || undefined,
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
              {steps.map((s, i) => (
                <ToggleButton key={s.group} value={i} sx={{ py: 0.1, px: 1, fontSize: 11, textTransform: "none" }}>
                  {s.label}
                </ToggleButton>
              ))}
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
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <div
              style={{
                flex: 1,
                minWidth: 0,
                height: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: 4,
              }}
            >
              <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                original (raw)
              </span>
              {rawCur ? (
                <img src={imgSrc(rawCur)} alt="raw" draggable={false} style={imgStyle} />
              ) : (
                <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                  no raw slice here
                </span>
              )}
            </div>
            <div
              style={{
                flex: 1,
                minWidth: 0,
                height: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: 4,
              }}
            >
              <span className="text-[11px]" style={{ color: "var(--c-green)" }}>
                {curStep?.label ?? "preprocessed"}
              </span>
              <img src={imgSrc(cur)} alt="preprocessed" draggable={false} style={imgStyle} />
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
