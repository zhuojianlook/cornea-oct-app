/* On launch, checks for a newer signed release and, if one exists, shows a slim banner offering a
   one-click "Install & restart". Renders nothing when up to date or outside the desktop shell. */

import { useEffect, useState } from "react";
import { Button, LinearProgress } from "@mui/material";
import type { Update } from "@tauri-apps/plugin-updater";
import { checkForUpdate, installAndRelaunch } from "../tauri/updater";

export function UpdateBanner() {
  const [update, setUpdate] = useState<Update | null>(null);
  const [busy, setBusy] = useState(false);
  const [pct, setPct] = useState<number | null>(null);
  const [err, setErr] = useState("");
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let alive = true;
    checkForUpdate().then((u) => { if (alive) setUpdate(u); }).catch(() => {});
    return () => { alive = false; };
  }, []);

  if (!update || dismissed) return null;

  const install = async () => {
    setBusy(true);
    setErr("");
    try {
      await installAndRelaunch(update, setPct);
      // relaunch() replaces the process; nothing runs after this on success.
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "6px 12px",
                  background: "var(--c-accent)", color: "#fff", fontSize: 13 }}>
      <span>
        ⬆ Update available — <b>v{update.version}</b>
        {update.currentVersion ? ` (you have v${update.currentVersion})` : ""}.
      </span>
      {err && <span style={{ color: "#ffd7d3" }}>Update failed: {err}</span>}
      <div style={{ flex: 1 }} />
      {busy ? (
        <span style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 180 }}>
          <LinearProgress variant={pct == null ? "indeterminate" : "determinate"} value={pct ?? 0}
            sx={{ flex: 1, minWidth: 120 }} />
          <span style={{ width: 64, textAlign: "right" }}>{pct == null ? "installing…" : `${pct}%`}</span>
        </span>
      ) : (
        <>
          <Button size="small" variant="contained" onClick={install}
            sx={{ bgcolor: "#fff", color: "var(--c-accent)", "&:hover": { bgcolor: "#f0f0f0" } }}>
            Install &amp; restart
          </Button>
          <Button size="small" variant="text" onClick={() => setDismissed(true)} sx={{ color: "#fff" }}>
            Later
          </Button>
        </>
      )}
    </div>
  );
}
