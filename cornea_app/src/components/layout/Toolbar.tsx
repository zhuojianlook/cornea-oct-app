import { Button, CircularProgress } from "@mui/material";
import { useWorkflowStore } from "../../store/workflowStore";
import { useCaseStore } from "../../store/caseStore";

export function Toolbar() {
  const stage = useWorkflowStore((s) => s.stage);
  const segBusy = useWorkflowStore((s) => s.segBusy);
  const scarBusy = useWorkflowStore((s) => s.scarBusy);
  const correcting = useWorkflowStore((s) => s.correcting);
  const segLoaded = useWorkflowStore((s) => s.segLoaded);
  const sensitivity = useWorkflowStore((s) => s.scarSensitivity);
  const runSam2 = useWorkflowStore((s) => s.runSam2);
  const loadCorrectionLayer = useWorkflowStore((s) => s.loadCorrectionLayer);
  const saveCorrection = useWorkflowStore((s) => s.saveCorrection);
  const runScarAuto = useWorkflowStore((s) => s.runScarAuto);
  const exportScarSummary = useWorkflowStore((s) => s.exportScarSummary);
  const setStage = useWorkflowStore((s) => s.setStage);
  const set = useWorkflowStore((s) => s.set);
  const hintMode = useWorkflowStore((s) => s.hintMode);
  const hintPositive = useWorkflowStore((s) => s.hintPositive);
  const hintCount = useWorkflowStore((s) => s.scarHints?.length ?? 0);
  const applyScarHints = useWorkflowStore((s) => s.applyScarHints);
  const clearScarHints = useWorkflowStore((s) => s.clearScarHints);
  const hasVolume = useCaseStore((s) => Boolean(s.volumeUrl));
  const busy = segBusy || scarBusy;

  const Correct = (
    <>
      {!correcting ? (
        <Button variant="outlined" disabled={busy || !segLoaded} onClick={() => loadCorrectionLayer()}
          title="Edit the labelmap with the pen (cornea=1, scar=3, background=2 erases)">
          Correct ✎
        </Button>
      ) : (
        <Button variant="contained" color="secondary" disabled={busy} onClick={() => saveCorrection()}>
          Save correction
        </Button>
      )}
    </>
  );

  return (
    <div
      className="flex items-center gap-2 px-3 border-b"
      style={{ height: 44, backgroundColor: "var(--c-surface)", borderColor: "var(--c-border)" }}
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
          <Button variant="contained" color="secondary" disabled={busy || !segLoaded} onClick={() => setStage(2)}>
            Correct →
          </Button>
        </>
      ) : stage === 2 ? (
        <>
          {Correct}
          <Button variant="outlined" disabled={busy} onClick={() => setStage(1)}>
            ← Segment
          </Button>
          <div style={{ width: 1, height: 24, background: "var(--c-border)" }} />
          <Button variant="contained" color="secondary" disabled={busy || !segLoaded} onClick={() => setStage(3)}>
            Scar →
          </Button>
        </>
      ) : (
        <>
          <Button variant="contained" color="error" disabled={busy || !segLoaded} onClick={() => runScarAuto()}
            title="Flag hyper-reflective regions inside the cornea as scar candidates to correct">
            Detect scar (auto)
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
          <Button variant="outlined" disabled={busy} onClick={() => setStage(2)}>
            ← Correct
          </Button>
        </>
      )}
      <div className="flex-1" />
      {busy && <CircularProgress size={16} />}
      <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
        {stage === 1 ? "Segment" : stage === 2 ? "Correct" : "Scar"}
      </span>
    </div>
  );
}
