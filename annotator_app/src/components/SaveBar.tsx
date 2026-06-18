/* Top bar: who is annotating (+ switch user), the output folder, Save ground truth, session + status. */

import { Button, CircularProgress } from "@mui/material";
import { useStore } from "../store/annotatorStore";

export function SaveBar() {
  const { activeUser, sessionId, outputDir, loaded, busy, status, save, chooseOutputDir } = useStore();

  return (
    <div className="flex items-center gap-3 px-3 border-b" style={{ minHeight: 44, backgroundColor: "var(--c-surface2)", borderColor: "var(--c-border)" }}>
      <span className="text-sm font-medium">Ground-Truth Annotator</span>
      <span className="text-xs px-2 py-0.5 rounded" style={{ background: "var(--c-surface)", border: "1px solid var(--c-border)" }}>
        👤 {activeUser}
      </span>
      <Button size="small" variant="text" sx={{ fontSize: 11, textTransform: "none", minWidth: 0 }}
        onClick={() => useStore.setState({ activeUser: null, loaded: false, activeVolume: null })}>switch</Button>

      <div style={{ width: 1, height: 22, background: "var(--c-border)" }} />
      <Button size="small" variant="outlined" onClick={() => chooseOutputDir()}
        sx={{ fontSize: 11, textTransform: "none" }}>{outputDir ? "Output…" : "Set output folder…"}</Button>
      {outputDir && <span className="text-[10px] truncate" style={{ color: "var(--c-text-dim)", maxWidth: 280 }} title={outputDir}>{outputDir}</span>}

      <div className="flex-1" />
      {busy && <CircularProgress size={16} />}
      <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>{status}</span>
      <Button size="small" variant="contained" color="success" disabled={!loaded || busy} onClick={() => save()}
        title="Write the painted labelmap (0/1/2) + a manifest row tagged with your username and this session">
        Save ground truth
      </Button>
      <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }} title="session id (this annotation occasion)">⧗ {sessionId}</span>
    </div>
  );
}
