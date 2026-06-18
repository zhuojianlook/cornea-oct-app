/* Slim footer: live status message (left), active volume + session id (right). */

import { CircularProgress } from "@mui/material";
import { useStore } from "../store/annotatorStore";

export function StatusBar() {
  const { status, busy, sessionId, activeVolume } = useStore();
  return (
    <footer className="flex items-center gap-3 px-4 border-t min-w-0"
      style={{ height: 26, flex: "none", fontSize: 11, backgroundColor: "var(--c-surface2)",
               borderColor: "var(--c-border)", color: "var(--c-text-dim)" }}>
      {busy && <CircularProgress size={11} thickness={6} />}
      <span className="truncate min-w-0" title={status}>{status}</span>
      <span className="flex-1" />
      {activeVolume && <span className="truncate flex-none" style={{ maxWidth: 320 }} title={activeVolume.name}>{activeVolume.name}</span>}
      <span title="Session id — this annotation occasion (one per app launch)">⧗ {sessionId}</span>
    </footer>
  );
}
