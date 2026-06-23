/* Before/after comparison — the original (raw) scan vs the preprocessed result, side by side and
   scrubbed together. Opened from the 3D viewer once a scan has been preprocessed (its raw snapshot,
   the `context_raw` preview group, exists). Pure 2D PNG previews, so it works with or without WebGL.
   Raw and corrected slices share geometry and slice indices, so they pair 1:1 by slice_index. */

import { useEffect, useMemo, useState } from "react";
import { ToggleButton, ToggleButtonGroup, Slider } from "@mui/material";
import { api, resourceUrl } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import type { PreviewImage } from "../../api/types";

const imgSrc = (im?: PreviewImage | null): string =>
  im ? (im.src ? resourceUrl(im.src) : im.data_url) : "";
const ORIENTS = ["axial", "coronal", "sagittal"] as const;

export function BeforeAfterViewer({ onClose }: { onClose: () => void }) {
  const caseId = useCaseStore((s) => s.caseId);
  // Re-fetch when previews re-render (e.g. a "Fix columns" re-preprocess bumps segVersion).
  const segSig = useWorkflowStore((s) => s.segVersion);

  const [orient, setOrient] = useState<(typeof ORIENTS)[number]>("axial");
  const [after, setAfter] = useState<PreviewImage[]>([]); // preprocessed (context)
  const [before, setBefore] = useState<PreviewImage[]>([]); // raw (context_raw)
  const [idx, setIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [enhContrast, setEnhContrast] = useState(false);

  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;
    setLoading(true);
    Promise.all([
      api.json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/context`).catch(() => ({ images: [] as PreviewImage[] })),
      api.json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/context_raw`).catch(() => ({ images: [] as PreviewImage[] })),
    ])
      .then(([a, b]) => {
        if (cancelled) return;
        setAfter(a.images || []);
        setBefore(b.images || []);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [caseId, segSig]);

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
    ? before.find((i) => i.orientation === orient && i.slice_index === cur.slice_index)
    : undefined;

  // Center on a middle slice when the orientation changes or slices first arrive.
  useEffect(() => {
    if (afterImgs.length) setIdx(Math.floor(afterImgs.length / 2));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orient, after.length]);

  const enhanceFilter = enhContrast ? "contrast(1.6) brightness(1.05)" : undefined;
  const imgStyle: React.CSSProperties = {
    maxHeight: "calc(100% - 28px)",
    maxWidth: "100%",
    objectFit: "contain",
    imageRendering: "pixelated",
    filter: enhanceFilter,
  };

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div
        className="flex items-center gap-2 px-3 border-b flex-wrap"
        style={{ minHeight: 40, borderColor: "var(--c-border)" }}
      >
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "1px solid var(--c-border)",
            borderRadius: 4,
            color: "var(--c-accent)",
            cursor: "pointer",
            fontSize: 12,
            padding: "2px 8px",
          }}
        >
          ← 3D view
        </button>
        <ToggleButtonGroup size="small" exclusive value={orient} onChange={(_, v) => v && setOrient(v)}>
          {ORIENTS.map((o) => (
            <ToggleButton key={o} value={o} style={{ textTransform: "capitalize" }}>
              {o}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        <ToggleButton
          size="small"
          value="contrast"
          selected={enhContrast}
          onChange={() => setEnhContrast((v) => !v)}
          sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}
          title="Display-only contrast boost (does not change the data)"
        >
          ◐ Contrast
        </ToggleButton>
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
                preprocessed
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
