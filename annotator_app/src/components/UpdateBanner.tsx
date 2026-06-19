/* Shows when a newer signed release is available (launch check or the manual "Check for updates"
   button). Offers one-click "Install & restart"; if the install fails, shows the error inline with a
   manual-download fallback (so a failed auto-update is never silent). */

import { Button, LinearProgress } from "@mui/material";
import { useStore } from "../store/annotatorStore";

const RELEASES_URL = "https://github.com/zhuojianlook/cornea-oct-app/releases/latest";

export function UpdateBanner() {
  const update = useStore((s) => s.update);
  const updateBusy = useStore((s) => s.updateBusy);
  const updatePct = useStore((s) => s.updatePct);
  const updateMsg = useStore((s) => s.updateMsg);
  const installUpdate = useStore((s) => s.installUpdate);
  const dismissUpdate = useStore((s) => s.dismissUpdate);

  if (!update) return null;
  const failed = !updateBusy && updateMsg.startsWith("Update failed");

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "6px 16px",
                  background: "var(--c-accent)", color: "#fff", fontSize: 13, flex: "none", flexWrap: "wrap" }}>
      <span>
        ⬆ Update available — <b>v{update.version}</b>
        {update.currentVersion ? ` (you have v${update.currentVersion})` : ""}.
      </span>
      {failed && (
        <span style={{ fontSize: 12, color: "#ffe2de" }}>
          {updateMsg} — download it manually at {RELEASES_URL}
        </span>
      )}
      <div style={{ flex: 1, minWidth: 12 }} />
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
            {failed ? "Retry" : "Install & restart"}
          </Button>
          <Button size="small" variant="text" onClick={() => dismissUpdate()} sx={{ color: "#fff" }}>Later</Button>
        </>
      )}
    </div>
  );
}
