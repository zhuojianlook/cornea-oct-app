/* Slim banner shown when a newer signed cornea release is available (from the launch check or the
   header "Check for updates" button). Offers one-click Install & restart. Renders nothing otherwise. */

import { Button, LinearProgress } from "@mui/material";
import { useUpdater } from "../store/updaterStore";

export function UpdateBanner() {
  const { update, busy, pct, install, dismiss } = useUpdater();
  if (!update) return null;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "6px 16px",
                  background: "var(--c-accent)", color: "#fff", fontSize: 13, flex: "none" }}>
      <span>
        ⬆ Update available — <b>v{update.version}</b>
        {update.currentVersion ? ` (you have v${update.currentVersion})` : ""}.
      </span>
      <div style={{ flex: 1 }} />
      {busy ? (
        <span style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 180 }}>
          <LinearProgress variant={pct == null ? "indeterminate" : "determinate"} value={pct ?? 0}
            sx={{ flex: 1, minWidth: 120 }} />
          <span style={{ width: 64, textAlign: "right" }}>{pct == null ? "installing…" : `${pct}%`}</span>
        </span>
      ) : (
        <>
          <Button size="small" variant="contained" onClick={() => install()}
            sx={{ bgcolor: "#fff", color: "var(--c-accent)", "&:hover": { bgcolor: "#f0f0f0" } }}>
            Install &amp; restart
          </Button>
          <Button size="small" variant="text" onClick={() => dismiss()} sx={{ color: "#fff" }}>Later</Button>
        </>
      )}
    </div>
  );
}
