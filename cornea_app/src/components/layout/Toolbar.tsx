import { Button, CircularProgress, MenuItem, Select } from "@mui/material";
import { useWorkflowStore } from "../../store/workflowStore";
import { useCaseStore } from "../../store/caseStore";

export function Toolbar() {
  const stage = useWorkflowStore((s) => s.stage);
  const segBusy = useWorkflowStore((s) => s.segBusy);
  const scarBusy = useWorkflowStore((s) => s.scarBusy);
  const correcting = useWorkflowStore((s) => s.correcting);
  const segLoaded = useWorkflowStore((s) => s.segLoaded);
  const sensitivity = useWorkflowStore((s) => s.scarSensitivity);
  const scarMethod = useWorkflowStore((s) => s.scarMethod);
  const runSam2 = useWorkflowStore((s) => s.runSam2);
  const loadCorrectionLayer = useWorkflowStore((s) => s.loadCorrectionLayer);
  const saveCorrection = useWorkflowStore((s) => s.saveCorrection);
  const cancelCorrection = useWorkflowStore((s) => s.cancelCorrection);
  const undoCorrection = useWorkflowStore((s) => s.undoCorrection);
  const runScarAuto = useWorkflowStore((s) => s.runScarAuto);
  const runScarAutoSam2 = useWorkflowStore((s) => s.runScarAutoSam2);
  const exportScarSummary = useWorkflowStore((s) => s.exportScarSummary);
  const setStage = useWorkflowStore((s) => s.setStage);
  const set = useWorkflowStore((s) => s.set);
  const hintMode = useWorkflowStore((s) => s.hintMode);
  const hintPositive = useWorkflowStore((s) => s.hintPositive);
  const hintCount = useWorkflowStore((s) => s.scarHints?.length ?? 0);
  const applyScarHints = useWorkflowStore((s) => s.applyScarHints);
  const clearScarHints = useWorkflowStore((s) => s.clearScarHints);
  const hasVolume = useCaseStore((s) => Boolean(s.volumeUrl));
  // #4: scar / not-scar (control) decision — made HERE, after preprocessing (not in the loader up front).
  const classification = useCaseStore((s) => (s.caseInfo?.manifest as Record<string, unknown> | undefined)?.scar_classification as ("scar" | "control" | null | undefined)) ?? null;
  const setClassification = useCaseStore((s) => s.setClassification);
  const busy = segBusy || scarBusy;

  const Correct = (
    <>
      {!correcting ? (
        <Button variant="outlined" disabled={busy || !segLoaded} onClick={() => loadCorrectionLayer()}
          title="Edit the labelmap with the pen (cornea=1, scar=3, background=2 erases)">
          Correct ✎
        </Button>
      ) : (
        <>
          <Button variant="contained" color="secondary" disabled={busy} onClick={() => saveCorrection()}>
            Save correction
          </Button>
          <Button variant="outlined" disabled={busy} onClick={() => undoCorrection()}
            title="Undo the last brush stroke / smart fill">↶ Undo</Button>
          <Button variant="outlined" color="inherit" disabled={busy} onClick={() => cancelCorrection()}
            title="Discard all edits and exit correction (labelmap unchanged)">Cancel</Button>
        </>
      )}
    </>
  );

  return (
    <div
      className="flex items-center gap-2 px-3 border-b overflow-x-auto [&>*]:shrink-0"
      style={{ minHeight: 44, backgroundColor: "var(--c-surface)", borderColor: "var(--c-border)" }}
    >
      {stage === 1 ? (
        <>
          <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
            Load &amp; preview scans in the sidebar ←, then Run SAM2 there. Or re-run cornea on just this scan:
          </span>
          <Button variant="outlined" disabled={busy || !hasVolume} onClick={() => runSam2()}
            title="Re-run SAM2 cornea segmentation on this one scan (cornea only, no scar/consensus — use the sidebar for the full pipeline)">
            Re-run SAM2 (this scan)
          </Button>
          <Button variant="contained" color="secondary" disabled={busy || correcting || !segLoaded} onClick={() => setStage(2)}>
            Correct →
          </Button>
        </>
      ) : stage === 2 ? (
        <>
          {Correct}
          <Button variant="outlined" disabled={busy || correcting} onClick={() => setStage(1)}>
            ← Segment
          </Button>
          <div style={{ width: 1, height: 24, background: "var(--c-border)" }} />
          <Button variant="contained" color="secondary" disabled={busy || correcting || !segLoaded} onClick={() => setStage(3)}>
            Scar →
          </Button>
        </>
      ) : stage === 3 ? (
        <>
          {/* #4: decide scar vs control HERE (post-preprocess), not in the loader up front. */}
          <div className="flex items-center gap-1 text-xs" style={{ color: "var(--c-text-dim)" }}
            title="Does this corrected volume have a scar? Decide now (after preprocessing). Persists to the case; 'No scar' marks it a control (used as the normal baseline) and skips scar detection.">
            scar?
            <Button size="small" variant={classification === "scar" ? "contained" : "outlined"} color="error"
              disabled={busy} onClick={() => setClassification(classification === "scar" ? null : "scar")}>Scar</Button>
            <Button size="small" variant={classification === "control" ? "contained" : "outlined"} color="inherit"
              disabled={busy} onClick={() => setClassification(classification === "control" ? null : "control")}>No scar</Button>
          </div>
          <div style={{ width: 1, height: 24, background: "var(--c-border)" }} />
          <Select size="small" value={scarMethod} onChange={(e) => set("scarMethod", e.target.value)}
            disabled={busy} sx={{ fontSize: 12, maxWidth: 200, color: "var(--c-text)", ".MuiSelect-select": { py: 0.4, textOverflow: "ellipsis" }, "& fieldset": { borderColor: "var(--c-border)" } }}
            title="Scar detection strategy (compare them on the same scan)">
            <MenuItem value="hysteresis" sx={{ fontSize: 12 }}>Hysteresis (best reproducibility)</MenuItem>
            <MenuItem value="depthnorm" sx={{ fontSize: 12 }}>Depth-normalised (subtract normal Bowman's; uses controls)</MenuItem>
            <MenuItem value="normal_anchor" sx={{ fontSize: 12 }}>Normal-stroma anchor (highest overlap, larger vol)</MenuItem>
            <MenuItem value="robust_mad" sx={{ fontSize: 12 }}>Robust MAD</MenuItem>
            <MenuItem value="morph_lcc" sx={{ fontSize: 12 }}>Morph + largest component</MenuItem>
            <MenuItem value="brightness" sx={{ fontSize: 12 }}>Brightness percentile (baseline)</MenuItem>
          </Select>
          <Button variant="contained" color="error" disabled={busy || !segLoaded || classification === "control"} onClick={() => runScarAuto()}
            title={classification === "control" ? "Marked 'No scar' (control) — toggle to Scar to detect." : "Run the selected scar strategy inside the cornea (overwrites/merges the scar candidate; correct as needed)."}>
            Detect scar (auto)
          </Button>
          <Button variant="outlined" color="error" disabled={busy || !segLoaded || classification === "control"} onClick={() => runScarAutoSam2()}
            title={classification === "control" ? "Marked 'No scar' (control) — toggle to Scar to detect." : "Auto scar via SAM2 on all 3 views as videos → ≥2-of-3 consensus, seeded from the brightest in-cornea tissue (and confined to your scar frame-range). Slower (~1–2 min) but more coherent."}>
            Scar (SAM2 3-view)
          </Button>
          <label className="flex items-center gap-1 text-xs" style={{ color: "var(--c-text-dim)" }}
            title="How much hyper-reflectivity to flag as scar candidates (higher = more)">
            sens
            <input type="range" min={1} max={40} value={sensitivity} style={{ width: 80 }}
              onChange={(e) => set("scarSensitivity", Number(e.target.value))} />
            <span style={{ width: 18 }}>{sensitivity}</span>
          </label>
          <div style={{ width: 1, height: 24, background: "var(--c-border)" }} />
          <Button variant={hintMode ? "contained" : "outlined"} color="warning" disabled={busy || !segLoaded}
            onClick={() => set("hintMode", !hintMode)}
            title="Click scar areas on the slices to guide SAM2">
            {hintMode ? "Hinting…" : "Hint (SAM2)"}
          </Button>
          {hintMode && (
            <>
              <Button size="small" variant={hintPositive ? "contained" : "outlined"} color="error"
                onClick={() => set("hintPositive", true)} title="Clicks mark scar">scar</Button>
              <Button size="small" variant={!hintPositive ? "contained" : "outlined"}
                onClick={() => set("hintPositive", false)} title="Clicks mark not-scar (negative)">not</Button>
              <Button variant="contained" color="warning" disabled={busy || hintCount === 0} onClick={() => applyScarHints()}>
                Apply ({hintCount})
              </Button>
              <Button size="small" variant="outlined" disabled={busy || hintCount === 0} onClick={() => clearScarHints()}>
                Clear
              </Button>
            </>
          )}
          {Correct}
          <div style={{ width: 1, height: 24, background: "var(--c-border)" }} />
          <Button variant="contained" color="success" disabled={busy} onClick={() => exportScarSummary()}
            title="Recompute scar volume (mm³) + en-face area (mm²) + density for every case → scar_summary.csv">
            Export scar metrics
          </Button>
          <Button variant="outlined" disabled={busy || correcting} onClick={() => setStage(2)}>
            ← Correct
          </Button>
          <div style={{ width: 1, height: 24, background: "var(--c-border)" }} />
          <Button variant="contained" color="secondary" disabled={busy || correcting} onClick={() => setStage(4)}
            title="Eye-motion spectrum derived from the corneal surface (the slow scan axis is a time axis)">
            Motion →
          </Button>
        </>
      ) : (
        <>
          <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
            Eye-motion spectrum from the detected corneal surface — set the A-scan rate &amp; Analyze in the panel.
          </span>
          <Button variant="outlined" disabled={busy} onClick={() => setStage(3)}>
            ← Scar
          </Button>
        </>
      )}
      <div className="flex-1" />
      {busy && <CircularProgress size={16} />}
      <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
        {stage === 1 ? "Segment" : stage === 2 ? "Correct" : stage === 3 ? "Scar" : "Motion"}
      </span>
    </div>
  );
}
