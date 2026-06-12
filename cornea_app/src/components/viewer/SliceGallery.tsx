/* No-WebGL 2D slice viewer.
   Shows the in-sidecar-rendered preview PNGs (grayscale slices or the
   segmentation overlay) as plain <img>, so the OCT is viewable in browsers
   without WebGL2 (e.g. the VS Code Simple Browser). */

import { useEffect, useMemo, useState } from "react";
import { ToggleButton, ToggleButtonGroup, Slider, CircularProgress } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import { pxToIjk } from "../../api/coords";
import type { PreviewImage } from "../../api/types";

type Group = "segmentation" | "context";
const GROUP_LABEL: Record<Group, string> = {
  segmentation: "Segmentation",
  context: "Slices",
};
const ORIENTS = ["axial", "coronal", "sagittal"] as const;

export function SliceGallery() {
  const caseId = useCaseStore((s) => s.caseId);
  // Re-fetch when the segmentation changes (SAM2/correct/scar re-render previews).
  const segSig = useWorkflowStore((s) => s.segVersion);
  const hintMode = useWorkflowStore((s) => s.hintMode);
  const hintPositive = useWorkflowStore((s) => s.hintPositive);
  const scarHints = useWorkflowStore((s) => s.scarHints);
  const addScarHint = useWorkflowStore((s) => s.addScarHint);

  // When a consensus tab is active the store pins the preview group (the voted
  // map, or a scan warped into the common frame); otherwise we auto-select below.
  const previewGroup = useWorkflowStore((s) => s.previewGroup);

  const [group, setGroup] = useState<Group>("context");
  const [orient, setOrient] = useState<(typeof ORIENTS)[number]>("axial");
  const [images, setImages] = useState<PreviewImage[]>([]);
  const [idx, setIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  // Bumped after we render context previews on demand, to force the fetch effect to
  // re-pull (can't reuse segSig — the auto-select effect depends on it and would loop).
  const [refetchTick, setRefetchTick] = useState(0);

  const effectiveGroup = previewGroup ?? group;

  // Auto-select the richest available group (segmentation > slices) so the
  // viewer (and screenshots of it) show the latest result by default. Skipped
  // when a consensus tab pins the group.
  useEffect(() => {
    if (!caseId || previewGroup) return;
    let cancelled = false;
    (async () => {
      for (const g of ["segmentation", "context"] as Group[]) {
        try {
          const r = await api.json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/${g}`);
          if (cancelled) return;
          if ((r.images || []).length > 0) {
            setGroup(g);
            return;
          }
        } catch {
          /* try next */
        }
      }
      // Nothing rendered yet — generate grayscale context slices so the OCT shows.
      // A new DICOM is converted to NIfTI here (slow), so show the spinner meanwhile.
      try {
        if (!cancelled) setLoading(true);
        await api.json(`/api/case/${caseId}/context-previews`, "POST", JSON.stringify({}));
        // Now that the slices exist, force the fetch effect to re-pull and show them.
        if (!cancelled) {
          setGroup("context");
          setRefetchTick((t) => t + 1);
        }
      } catch {
        /* no volume yet — fine */
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [caseId, segSig, previewGroup]);

  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;
    setLoading(true);
    api
      .json<{ images: PreviewImage[] }>(`/api/case/${caseId}/previews/${effectiveGroup}`)
      .then((r) => {
        if (cancelled) return;
        const imgs = r.images || [];
        setImages(imgs);
        // Default to a middle slice of the current orientation (edge slices often
        // miss the cornea), so the viewer/screenshots show the structure.
        const mid = imgs.filter((i) => i.orientation === orient);
        if (mid.length) setIdx(Math.floor(mid.length / 2));
      })
      .catch(() => !cancelled && setImages([]))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [caseId, effectiveGroup, segSig, refetchTick]);

  const orientImgs = useMemo(
    () =>
      images
        .filter((i) => i.orientation === orient)
        .sort((a, b) => Number(a.slice_index ?? 0) - Number(b.slice_index ?? 0)),
    [images, orient],
  );
  const safeIdx = Math.min(idx, Math.max(0, orientImgs.length - 1));
  const cur = orientImgs[safeIdx];

  const onImgClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!hintMode || !cur) return;
    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    if (fx < 0 || fy < 0 || fx > 1 || fy > 1) return;
    const px = fx * (cur.image_width ?? img.naturalWidth);
    const py = fy * (cur.image_height ?? img.naturalHeight);
    const ijk = pxToIjk(cur, px, py);
    if (!ijk || cur.orientation == null || cur.slice_index == null) return;
    addScarHint({ ijk, orientation: cur.orientation, slice_index: cur.slice_index, positive: hintPositive, fx, fy });
  };

  // Hints painted on the slice currently shown.
  const hintsHere = cur
    ? (scarHints ?? []).filter((h) => h.orientation === cur.orientation && h.slice_index === cur.slice_index)
    : [];

  return (
    <div className="flex flex-col h-full min-h-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div
        className="flex items-center gap-2 px-3 border-b flex-wrap"
        style={{ minHeight: 40, borderColor: "var(--c-border)" }}
      >
        {!previewGroup && (
          <ToggleButtonGroup size="small" exclusive value={group} onChange={(_, v) => v && setGroup(v)}>
            <ToggleButton value="segmentation">Segmentation</ToggleButton>
            <ToggleButton value="context">Slices</ToggleButton>
          </ToggleButtonGroup>
        )}
        <ToggleButtonGroup
          size="small"
          exclusive
          value={orient}
          onChange={(_, v) => {
            if (v) {
              setOrient(v);
              const n = images.filter((i) => i.orientation === v).length;
              setIdx(n ? Math.floor(n / 2) : 0);
            }
          }}
        >
          {ORIENTS.map((o) => (
            <ToggleButton key={o} value={o} style={{ textTransform: "capitalize" }}>
              {o}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        <div className="flex-1" />
        {loading && <CircularProgress size={16} />}
        <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
          2D view (no WebGL)
        </span>
      </div>

      <div className="flex-1 min-h-0 flex items-center justify-center p-3">
        {!cur ? (
          loading ? (
            <div className="text-center" style={{ color: "var(--c-text-dim)" }}>
              <div style={{ fontSize: 13 }}>Rendering slices…</div>
              <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>
                Converting the volume (DICOM → NIfTI can take a moment).
              </div>
            </div>
          ) : (
            <div className="text-center" style={{ color: "var(--c-text-dim)" }}>
              <div style={{ fontSize: 13 }}>
                No {(previewGroup ? "overlay" : GROUP_LABEL[group].toLowerCase())} {orient} slices yet.
              </div>
              <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>
                {previewGroup
                  ? "Build the consensus first, then pick a tab."
                  : group === "segmentation"
                    ? "Segment the cornea (SAM2) first."
                    : "Register a volume to render slices."}
              </div>
            </div>
          )
        ) : (
          <div style={{ position: "relative", display: "inline-block", maxHeight: "100%", maxWidth: "100%" }}>
            <img
              src={cur.data_url}
              alt={cur.file_name}
              onClick={onImgClick}
              style={{
                display: "block",
                maxHeight: "100%",
                maxWidth: "100%",
                imageRendering: "pixelated",
                cursor: hintMode ? "crosshair" : "default",
              }}
            />
            {hintsHere.map((h, i) => (
              <span
                key={i}
                title={h.positive ? "scar hint" : "not-scar hint"}
                style={{
                  position: "absolute",
                  left: `${h.fx * 100}%`,
                  top: `${h.fy * 100}%`,
                  width: 12,
                  height: 12,
                  marginLeft: -6,
                  marginTop: -6,
                  borderRadius: "50%",
                  border: "2px solid #fff",
                  background: h.positive ? "#ff2e55" : "#39d0ff",
                  boxShadow: "0 0 3px rgba(0,0,0,0.8)",
                  pointerEvents: "none",
                }}
              />
            ))}
          </div>
        )}
      </div>

      {orientImgs.length > 0 && (
        <div className="flex items-center gap-3 px-4 py-2 border-t" style={{ borderColor: "var(--c-border)" }}>
          <span className="text-xs whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>
            {orient} slice {cur?.slice_index ?? "—"} ({safeIdx + 1}/{orientImgs.length})
          </span>
          <Slider
            size="small"
            min={0}
            max={Math.max(0, orientImgs.length - 1)}
            value={safeIdx}
            onChange={(_, v) => setIdx(v as number)}
          />
        </div>
      )}
    </div>
  );
}
