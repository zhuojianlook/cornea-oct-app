/* Shows when a newer signed release is available (from the launch check or the manual "Check for
   updates" button). Offers a one-click "Install & restart". Update state lives in the store so the
   header button and this banner stay in sync. */

import { Button, LinearProgress } from "@mui/material";
import { useStore } from "../store/annotatorStore";

export function UpdateBanner() {
  const update = useStore((s) => s.update);
  const updateBusy = useStore((s) => s.updateBusy);
  const updatePct = useStore((s) => s.updatePct);
  const installUpdate = useStore((s) => s.installUpdate);
  const dismissUpdate = useStore((s) => s.dismissUpdate);

  if (!update) return null;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "6px 16px",
                  background: "var(--c-accent)", color: "#fff", fontSize: 13, flex: "none" }}>
      <span>
        ⬆ Update available — <b>v{update.version}</b>
        {update.currentVersion ? ` (you have v${update.currentVersion})` : ""}.
      </span>
      <div style={{ flex: 1 }} />
      {updateBusy ? (
        <span style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 180 }}>
          <LinearProgress variant={updatePct == null ? "indeterminate" : "determinate"} value={updatePct ?? 0}
            sx={{ flex: 1, minWidth: 120 }} />
          <span style={{ width: 64, textAlign: "right" }}>{updatePct == null ? "installing…" : `${updatePct}%`}</span>
        </span>
      ) : (
        <>
          <Button size="small" variant="contained" onClick={() => installUpdate()}
            sx={{ bgcolor: "#fff", color: "var(--c-accent)", "&:hover": { bgcolor: "#f0f0f0" } }}>
            Install &amp; restart
          </Button>
          <Button size="small" variant="text" onClick={() => dismissUpdate()} sx={{ color: "#fff" }}>Later</Button>
        </>
      )}
    </div>
  );
}
