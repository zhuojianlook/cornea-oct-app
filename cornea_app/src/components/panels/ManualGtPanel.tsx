/* Manual ground-truth import + comparison panel (Sidebar).
   Imports a labelmap produced by the companion annotator app (0/1/2, painted on this case's exported
   working volume), validates it aligns to the open case, lists imported GTs, and — per GT — computes
   Dice / HD95 / volume agreement of the human label vs the app's auto segmentation, and opens an
   agreement overlay in the central viewer. */

import { useEffect, useRef, useState } from "react";
import { Button, Typography } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import type { GtCompareResult, ManualGtImportResult, ManualGtInfo, ManualGtList } from "../../api/types";

const fmt = (n: number | null | undefined, d = 3) =>
  n === null || n === undefined || Number.isNaN(n) ? "—" : Number(n).toFixed(d);

function MetricsTable({ r }: { r: GtCompareResult }) {
  const Row = ({ label, k }: { label: string; k: "cornea" | "scar" }) => {
    const c = r.classes[k];
    return (
      <div className="grid grid-cols-[64px_1fr] gap-x-2 text-[11px] py-1 border-t" style={{ borderColor: "var(--c-border)" }}>
        <div style={{ color: "var(--c-text)", fontWeight: 600 }}>{label}</div>
        <div className="flex flex-col gap-0.5" style={{ color: "var(--c-text-dim)" }}>
          <div>Dice <b style={{ color: "var(--c-text)" }}>{fmt(c.dice)}</b> · Jaccard {fmt(c.jaccard)} · HD95 {fmt(c.hd95_mm)}mm · ASSD {fmt(c.assd_mm)}mm</div>
          <div>vol manual {fmt(c.gt_volume_mm3, 4)} → auto {fmt(c.auto_volume_mm3, 4)} mm³ (Δ {c.volume_rel_diff_pct === null ? "—" : `${c.volume_rel_diff_pct > 0 ? "+" : ""}${c.volume_rel_diff_pct}%`})</div>
          <div>overlap TP {c.tp.toLocaleString()} · auto-only {c.fp.toLocaleString()} · missed {c.fn.toLocaleString()}</div>
          {k === "scar" && (c.gt_area_mm2 != null || c.auto_area_mm2 != null) && (
            <div>en-face area manual {fmt(c.gt_area_mm2, 3)} → auto {fmt(c.auto_area_mm2, 3)} mm²</div>
          )}
        </div>
      </div>
    );
  };
  return (
    <div className="mt-1 rounded p-2" style={{ background: "var(--c-surface)" }}>
      <div className="text-[11px] mb-1" style={{ color: "var(--c-text-dim)" }}>
        Manual GT <b style={{ color: "var(--c-text)" }}>{r.name}</b> vs auto ({r.auto_source || "segmentation"})
      </div>
      <Row label="Cornea" k="cornea" />
      <Row label="Scar" k="scar" />
    </div>
  );
}

export function ManualGtPanel() {
  const caseInfo = useCaseStore((s) => s.caseInfo);
  const caseId = caseInfo?.case_id ?? null;
  const wfSet = useWorkflowStore((s) => s.set);
  const gtViewerActive = useWorkflowStore((s) => s.gtViewerActive);
  const gtViewerName = useWorkflowStore((s) => s.gtViewerName);

  const fileRef = useRef<HTMLInputElement | null>(null);
  const compareReq = useRef<string | null>(null); // latest requested GT — drop out-of-order responses
  const [gts, setGts] = useState<ManualGtInfo[]>([]);
  const [hasSeg, setHasSeg] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [compareName, setCompareName] = useState<string | null>(null);
  const [compare, setCompare] = useState<GtCompareResult | null>(null);

  const refresh = async (cid: string) => {
    try {
      const r = await api.json<ManualGtList>(`/api/case/${cid}/manual-gt`);
      setGts(r.gts);
      setHasSeg(r.has_segmentation);
    } catch {
      setGts([]); setHasSeg(false);
    }
  };

  useEffect(() => {
    setCompare(null); setCompareName(null); setMsg(null); setError(null);
    if (caseId) refresh(caseId);
    else setGts([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId]);

  const onPick = () => fileRef.current?.click();

  const onFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = ""; // allow re-importing the same file
    if (!caseId || files.length === 0) return;
    setBusy(true); setError(null); setMsg(null);
    try {
      const r = await api.upload<ManualGtImportResult>(`/api/case/${caseId}/manual-gt`, files);
      setGts(r.gts);
      const okN = r.imported.length;
      const errs = r.errors.map((x) => `${x.file}: ${x.error}`);
      setMsg(`Imported ${okN} labelmap${okN === 1 ? "" : "s"}.`);
      if (errs.length) setError(errs.join(" · "));
      await refresh(caseId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const onCompare = async (name: string) => {
    if (!caseId) return;
    compareReq.current = name;
    setBusy(true); setError(null); setCompareName(name); setCompare(null);
    try {
      const r = await api.json<GtCompareResult>(`/api/case/${caseId}/manual-gt/${encodeURIComponent(name)}/compare`);
      if (compareReq.current === name) setCompare(r); // ignore a stale response if a newer compare started
    } catch (err) {
      if (compareReq.current === name) setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (compareReq.current === name) setBusy(false);
    }
  };

  const onOverlay = (name: string) => {
    wfSet("gtViewerName", name);
    wfSet("gtViewerClass", "scar");
    wfSet("gtViewerActive", true);
  };

  const onDelete = async (name: string) => {
    if (!caseId) return;
    setBusy(true); setError(null);
    try {
      const r = await api.json<{ gts: ManualGtInfo[] }>(`/api/case/${caseId}/manual-gt/${encodeURIComponent(name)}`, "DELETE");
      setGts(r.gts);
      if (compareName === name) { setCompare(null); setCompareName(null); }
      if (gtViewerName === name && gtViewerActive) { wfSet("gtViewerActive", false); wfSet("gtViewerName", null); }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (!caseId) {
    return (
      <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
        Open a case first, then import a manual ground-truth labelmap to compare against its segmentation.
      </Typography>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
        Import a manual labelmap (0/1/2) made in the annotator app on this scan's exported volume, then
        compare the human label against this app's semiautomated segmentation.
      </Typography>

      <input ref={fileRef} type="file" accept=".nii,.nii.gz,application/gzip" multiple
        onChange={onFiles} style={{ display: "none" }} />
      <Button variant="outlined" size="small" disabled={busy} onClick={onPick}>
        Import ground-truth .nii.gz…
      </Button>

      {!hasSeg && gts.length > 0 && (
        <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
          No auto segmentation yet — run SAM2 / scar detection to enable comparison.
        </Typography>
      )}

      {gts.length === 0 && (
        <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>No imported ground truth yet.</Typography>
      )}

      {gts.map((g) => (
        <div key={g.name} className="rounded p-2 flex flex-col gap-1" style={{ background: "var(--c-surface)" }}>
          <div className="text-xs" style={{ color: "var(--c-text)", wordBreak: "break-all" }}>{g.name}</div>
          <div className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
            {g.error ? g.error : `cornea ${g.cornea_voxels.toLocaleString()} · scar ${g.scar_voxels.toLocaleString()} vox`}
          </div>
          {!g.error && (
            <div className="flex gap-1 flex-wrap">
              <Button size="small" variant="outlined" disabled={busy || !hasSeg} onClick={() => onCompare(g.name)}
                sx={{ py: 0, px: 1, fontSize: 11, textTransform: "none" }}>Compare</Button>
              <Button size="small" variant={gtViewerActive && gtViewerName === g.name ? "contained" : "outlined"}
                disabled={busy || !hasSeg} onClick={() => onOverlay(g.name)}
                sx={{ py: 0, px: 1, fontSize: 11, textTransform: "none" }}>Overlay</Button>
              <Button size="small" color="error" variant="outlined" disabled={busy} onClick={() => onDelete(g.name)}
                sx={{ py: 0, px: 1, fontSize: 11, textTransform: "none" }}>Delete</Button>
            </div>
          )}
        </div>
      ))}

      {busy && <Typography variant="caption" sx={{ color: "var(--c-accent)" }}>working…</Typography>}
      {msg && <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-word" }}>{msg}</Typography>}
      {error && <Typography variant="caption" sx={{ color: "var(--c-red)", wordBreak: "break-word" }}>{error}</Typography>}

      {compare && compareName && <MetricsTable r={compare} />}
    </div>
  );
}
