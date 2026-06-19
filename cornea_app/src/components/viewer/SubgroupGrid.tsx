/* Subgroup review grid (no-WebGL friendly).
   When a per-subgroup CONSENSUS case is open, show ALL of its member scans at once —
   each row = before (raw) | after (corrected) | scar — with ONE shared orientation and
   ONE shared scrubbing timeline, and a toggle that swaps the scar column between each
   scan's own segmentation and the subgroup consensus. This is the unified view used to
   review/correct the subgroup's scans side by side. */

import { useEffect, useMemo, useState } from "react";
import { ToggleButton, ToggleButtonGroup, Slider, CircularProgress, Button, Tooltip } from "@mui/material";
import { api, resourceUrl } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import type { PreviewImage, ConsensusReport, ConsensusScan } from "../../api/types";
import { OverlapViewer } from "./OverlapViewer";
import { AlignmentViewer } from "./AlignmentViewer";

const ORIENTS = ["axial", "coronal", "sagittal"] as const;
type Orient = (typeof ORIENTS)[number];
type Overlay = "cons" | "seg";

const imgSrc = (im?: PreviewImage | null): string => (im ? (im.src ? resourceUrl(im.src) : im.data_url) : "");
const shortLabel = (cid: string): string => cid.split("_").pop() || cid;

interface ScanPreviews {
  context: PreviewImage[];
  raw: PreviewImage[];
  seg: PreviewImage[];
  cons: PreviewImage[];
}

export function SubgroupGrid() {
  const caseInfo = useCaseStore((s) => s.caseInfo);
  const setCaseId = useCaseStore((s) => s.setCaseId);
  const openCase = useCaseStore((s) => s.openCase);
  const wfSet = useWorkflowStore((s) => s.set);
  const segSig = useWorkflowStore((s) => s.segVersion);

  const m = caseInfo?.manifest as Record<string, unknown> | undefined;
  const consensusId = caseInfo?.case_id ?? null;
  const scans = useMemo(() => (m?.consensus_cases as string[] | undefined) ?? [], [m]);
  const refCid = m?.reference as string | undefined;
  const report = m?.consensus_report as ConsensusReport | undefined;
  const subgroupLabel = (m?.scar_subgroup as string | undefined) || report?.subgroup;
  const perScan: Record<string, ConsensusScan> = {};
  for (const p of report?.per_scan ?? []) perScan[p.case] = p;

  const [orient, setOrient] = useState<Orient>("sagittal");
  const [overlay, setOverlay] = useState<Overlay>("cons");
  const [mode, setMode] = useState<"grid" | "align" | "overlap">("grid");
  const [masterIdx, setMasterIdx] = useState(0);
  const [data, setData] = useState<Record<string, ScanPreviews>>({});
  const [loading, setLoading] = useState(false);

  // Fetch the four preview groups for every member scan (lazy URL listings → cheap).
  const scanKey = scans.join(",");
  useEffect(() => {
    if (!scans.length) return;
    let cancelled = false;
    setLoading(true);
    (async () => {
      const get = async (cid: string, grp: string) => {
        try {
          return (await api.json<{ images: PreviewImage[] }>(`/api/case/${cid}/previews/${grp}`)).images || [];
        } catch {
          return [];
        }
      };
      const out: Record<string, ScanPreviews> = {};
      await Promise.all(
        scans.map(async (cid) => {
          const [context, raw, seg, cons] = await Promise.all([
            get(cid, "context"), get(cid, "context_raw"), get(cid, "context_seg"), get(cid, "context_cons"),
          ]);
          out[cid] = { context, raw, seg, cons };
        }),
      );
      if (!cancelled) {
        setData(out);
        setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [scanKey, segSig, scans]);

  const byOrient = (list: PreviewImage[]) =>
    list.filter((i) => i.orientation === orient).sort((a, b) => Number(a.slice_index ?? 0) - Number(b.slice_index ?? 0));

  // Master timeline = the longest member's corrected-slice count for this orientation; each
  // scan maps the master position proportionally to its own slices (scans differ in length).
  const maxLen = useMemo(() => {
    let n = 0;
    for (const cid of scans) n = Math.max(n, byOrient(data[cid]?.context ?? []).length);
    return n;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, orient, scanKey]);

  // Clamp / reset the master index to the middle when orientation or data changes.
  useEffect(() => {
    if (maxLen > 0) setMasterIdx((i) => Math.min(Math.max(i, 0), maxLen - 1) || Math.floor(maxLen / 2));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [maxLen, orient]);

  const focusCorrect = async (cid: string) => {
    wfSet("reviewConsensusId", consensusId);
    setCaseId(cid);
    await openCase();
  };

  const overlayList = (p: ScanPreviews) => (overlay === "cons" ? p.cons : p.seg);
  const overlayMissing = (p: ScanPreviews) => byOrient(overlayList(p)).length === 0;

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      {/* Controls: orientation · scar source · shared scrub timeline */}
      <div className="flex items-center gap-3 px-3 border-b flex-wrap" style={{ minHeight: 44, borderColor: "var(--c-border)" }}>
        <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
          Subgroup{subgroupLabel && subgroupLabel !== "1" ? ` · ${subgroupLabel}` : ""} — {scans.length} scans
        </span>
        <ToggleButtonGroup size="small" exclusive value={mode} onChange={(_, v) => v && setMode(v)}>
          <ToggleButton value="grid" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Scans grid</ToggleButton>
          <ToggleButton value="align" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Volume align</ToggleButton>
          <ToggleButton value="overlap" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Scar overlap</ToggleButton>
        </ToggleButtonGroup>
        {mode === "grid" && (
          <>
            <ToggleButtonGroup size="small" exclusive value={orient} onChange={(_, v) => v && setOrient(v)}>
              {ORIENTS.map((o) => (
                <ToggleButton key={o} value={o} style={{ textTransform: "capitalize" }}>{o}</ToggleButton>
              ))}
            </ToggleButtonGroup>
            <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>Scar</span>
            <ToggleButtonGroup size="small" exclusive value={overlay} onChange={(_, v) => v && setOverlay(v)}>
              <ToggleButton value="seg" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Per scan</ToggleButton>
              <ToggleButton value="cons" sx={{ py: 0.25, px: 1, fontSize: 12, textTransform: "none" }}>Consensus</ToggleButton>
            </ToggleButtonGroup>
            <div className="flex items-center gap-2" style={{ flex: 1, minWidth: 180 }}>
              <span className="text-xs whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>
                slice {maxLen ? masterIdx + 1 : 0}/{maxLen}
              </span>
              <Slider size="small" min={0} max={Math.max(0, maxLen - 1)} value={Math.min(masterIdx, Math.max(0, maxLen - 1))}
                onChange={(_, v) => setMasterIdx(v as number)} />
            </div>
            {loading && <CircularProgress size={16} />}
          </>
        )}
      </div>

      {mode === "align" && consensusId && (
        <AlignmentViewer caseId={consensusId} members={scans} refCid={refCid} />
      )}
      {mode === "overlap" && consensusId && (
        <OverlapViewer caseId={consensusId} nScans={scans.length} />
      )}
      {mode === "grid" && (<>
        {/* grid headers + rows */}

      {/* Column headers */}
      <div className="flex items-center px-3 py-1 border-b text-[11px]" style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}>
        <div style={{ width: 132, flex: "none" }}>scan</div>
        <div style={{ flex: 1, textAlign: "center" }}>before (raw)</div>
        <div style={{ flex: 1, textAlign: "center", color: "var(--c-green)" }}>after (corrected)</div>
        <div style={{ flex: 1, textAlign: "center", color: "var(--c-accent)" }}>
          {overlay === "cons" ? "consensus scar" : "this scan's scar"}
        </div>
      </div>

      {/* One row per member scan, scrubbed together */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {scans.length === 0 && (
          <div className="p-6 text-center text-sm" style={{ color: "var(--c-text-dim)" }}>No subgroup scans.</div>
        )}
        {scans.map((cid) => {
          const p = data[cid];
          const master = p ? byOrient(p.context) : [];
          const scanIdx = master.length ? Math.round((masterIdx / Math.max(1, maxLen - 1)) * (master.length - 1)) : 0;
          const cur = master[scanIdx];
          const find = (list: PreviewImage[]) =>
            cur ? list.find((i) => i.orientation === orient && i.slice_index === cur.slice_index) : undefined;
          const rawCur = p ? find(p.raw) : undefined;
          const ovCur = p ? find(overlayList(p)) : undefined;
          const ps = perScan[cid];
          const isRef = cid === refCid;
          return (
            <div key={cid} className="flex items-stretch border-b" style={{ borderColor: "var(--c-border)", minHeight: 168 }}>
              {/* Row header */}
              <div style={{ width: 132, flex: "none" }} className="flex flex-col gap-1 px-2 py-2">
                <span className="text-xs flex items-center gap-1" style={{ color: "var(--c-text)" }}>
                  {ps?.low_correspondence && (
                    <span title="low correspondence (likely a different FOV)"
                      style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--c-red, #ff6b6b)", flex: "none" }} />
                  )}
                  {shortLabel(cid)} {isRef && <span style={{ color: "var(--c-text-dim)" }} title="reference">★</span>}
                </span>
                {ps && (
                  <span className="text-[10px]" style={{ color: "var(--c-text-dim)", lineHeight: 1.3 }}>
                    {ps.scar_volume_mm3} mm³
                    {ps.role !== "reference" && ps.scar_dice_to_ref_fov != null && (
                      <><br />Dice {ps.scar_dice_to_ref}{" "}
                        <span title="Dice on the shared field-of-view (partial-cut-aware)">(FOV {ps.scar_dice_to_ref_fov})</span></>
                    )}
                    {ps.matched_fraction != null && <><br />{Math.round(ps.matched_fraction * 100)}% matched</>}
                  </span>
                )}
                <Tooltip title="Open this scan to brush/erase its scar, then return to the subgroup" arrow>
                  <Button size="small" variant="outlined" onClick={() => focusCorrect(cid)}
                    sx={{ py: 0, px: 0.8, fontSize: 10, textTransform: "none", mt: "auto", alignSelf: "flex-start" }}>
                    Correct ✎
                  </Button>
                </Tooltip>
              </div>
              {/* Cells */}
              {[
                { im: rawCur, alt: "raw", note: "no raw" },
                { im: cur, alt: "corrected", note: "—" },
                { im: ovCur, alt: "scar", note: overlayMissing(p ?? { context: [], raw: [], seg: [], cons: [] }) ? "run SAM2/consensus" : "no scar here" },
              ].map((c, k) => (
                <div key={k} className="flex items-center justify-center" style={{ flex: 1, minWidth: 0, padding: 6, borderLeft: "1px solid var(--c-border)" }}>
                  {c.im ? (
                    <img src={imgSrc(c.im)} alt={c.alt} draggable={false}
                      style={{ maxHeight: 150, maxWidth: "100%", objectFit: "contain", imageRendering: "pixelated" }} />
                  ) : (
                    <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>{p ? c.note : "…"}</span>
                  )}
                </div>
              ))}
            </div>
          );
        })}
      </div>
      </>)}
    </div>
  );
}
