import { Button, CircularProgress } from "@mui/material";
import { usePaintStore } from "../../store/paintStore";
import { useCaseStore } from "../../store/caseStore";

export function Toolbar() {
  const stage = usePaintStore((s) => s.stage);
  const aiBusy = usePaintStore((s) => s.aiBusy);
  const growBusy = usePaintStore((s) => s.growBusy);
  const aiPaint = usePaintStore((s) => s.aiPaint);
  const applyEdits = usePaintStore((s) => s.applyEdits);
  const loadDrawingLayer = usePaintStore((s) => s.loadDrawingLayer);
  const drawingLoaded = usePaintStore((s) => s.drawingLoaded);
  const runGrow = usePaintStore((s) => s.runGrow);
  const detectScar = usePaintStore((s) => s.detectScar);
  const scarBusy = usePaintStore((s) => s.scarBusy);
  const segLoaded = usePaintStore((s) => s.segLoaded);
  const seedCount = usePaintStore((s) => s.seedImages.length);
  const hasVolume = useCaseStore((s) => Boolean(s.volumeUrl));
  const busy = aiBusy || growBusy || scarBusy;

  return (
    <div
      className="flex items-center gap-2 px-3 border-b"
      style={{ height: 44, backgroundColor: "var(--c-surface)", borderColor: "var(--c-border)" }}
    >
      {stage === 1 ? (
        <>
          <Button variant="contained" disabled={busy || !hasVolume} onClick={() => aiPaint(false)}>
            AI Paint
          </Button>
          <Button variant="outlined" disabled={busy || !hasVolume} onClick={() => aiPaint(true)}>
            Heuristic
          </Button>
          <Button
            variant="outlined"
            disabled={busy || !hasVolume}
            onClick={() => loadDrawingLayer()}
            title="Load current seeds as an editable paint layer"
          >
            Edit seeds
          </Button>
          <Button
            variant="outlined"
            disabled={busy || !drawingLoaded}
            onClick={() => applyEdits()}
            title="Convert your paint edits back to seeds"
          >
            Apply edits
          </Button>
          <div style={{ width: 1, height: 24, background: "var(--c-border)" }} />
          <Button variant="contained" color="secondary" disabled={busy || seedCount === 0} onClick={() => runGrow()}>
            Grow from Seeds →
          </Button>
        </>
      ) : stage === 2 ? (
        <>
          <Button variant="contained" disabled={busy || seedCount === 0} onClick={() => runGrow()}>
            Re-grow from Seeds
          </Button>
          <Button variant="outlined" disabled={busy} onClick={() => usePaintStore.getState().setStage(1)}>
            ← Back to paint
          </Button>
          <Button variant="contained" color="secondary" disabled={busy || !segLoaded} onClick={() => usePaintStore.getState().setStage(4)}>
            Scar detection →
          </Button>
        </>
      ) : (
        <>
          <Button variant="contained" color="error" disabled={busy || !segLoaded} onClick={() => detectScar(false)}>
            Detect Scar (AI)
          </Button>
          <Button variant="outlined" disabled={busy || !segLoaded} onClick={() => detectScar(true)}>
            Detect Scar (heuristic)
          </Button>
          <Button variant="outlined" disabled={busy} onClick={() => usePaintStore.getState().setStage(2)}>
            ← Back to grow
          </Button>
        </>
      )}
      <div className="flex-1" />
      {busy && <CircularProgress size={16} />}
      <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
        {stage === 1 ? "Seed Paint" : stage === 2 ? "Grow" : "Scar Detection"}
      </span>
    </div>
  );
}
