/* About dialog: brand, short description, author, and version. */

import { Button, Dialog, DialogContent } from "@mui/material";
import { useStore, APP_VERSION } from "../store/annotatorStore";
import { tr } from "../i18n";

export function AboutDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const lang = useStore((s) => s.lang);
  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogContent sx={{ textAlign: "center", py: 4 }}>
        <div style={{ width: 52, height: 52, margin: "0 auto 14px", borderRadius: 14, background: "var(--c-accent)",
                      color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 28, fontWeight: 700 }}>
          ◎
        </div>
        <div style={{ fontSize: 16, fontWeight: 600 }}>{tr(lang, "login.title")}</div>
        <div style={{ fontSize: 12.5, color: "var(--c-text-dim)", marginTop: 8, lineHeight: 1.5 }}>{tr(lang, "about.desc")}</div>
        <div style={{ height: 1, background: "var(--c-border)", margin: "18px 0" }} />
        <div style={{ fontSize: 13, fontWeight: 500 }}>{tr(lang, "about.madeBy")}</div>
        <div style={{ fontSize: 12, color: "var(--c-text-dim)", marginTop: 4 }}>{tr(lang, "about.version")} {APP_VERSION}</div>
        <Button onClick={onClose} variant="outlined" size="small" sx={{ mt: 3 }}>{tr(lang, "about.close")}</Button>
      </DialogContent>
    </Dialog>
  );
}
