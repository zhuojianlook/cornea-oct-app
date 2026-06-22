/* App header: brand, who is annotating (+ switch user), output folder, Save ground truth, and the
   meta controls (check for updates, About, language toggle). Transient status + session id + version
   live in the bottom StatusBar. */

import { useState } from "react";
import { Button, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, ToggleButton, ToggleButtonGroup, Tooltip } from "@mui/material";
import { useStore } from "../store/annotatorStore";
import { tr } from "../i18n";
import { AboutDialog } from "./AboutDialog";

export function SaveBar() {
  const { activeUser, outputDir, loaded, busy, save, chooseOutputDir, lang, setLang, loadSegmentation } = useStore();
  const confirmOverwrite = useStore((s) => s.confirmOverwrite);
  const cancelOverwrite = useStore((s) => s.cancelOverwrite);
  const activeVolume = useStore((s) => s.activeVolume);
  const checkUpdates = useStore((s) => s.checkUpdates);
  const updateBusy = useStore((s) => s.updateBusy);
  const [aboutOpen, setAboutOpen] = useState(false);
  const outName = outputDir ? outputDir.split(/[/\\]/).filter(Boolean).pop() : null;

  return (
    <header className="flex items-center gap-3 px-4 border-b min-w-0"
      style={{ height: 54, flex: "none", backgroundColor: "var(--c-surface2)", borderColor: "var(--c-border)" }}>
      {/* Brand */}
      <div className="flex items-center gap-2.5 flex-none">
        <span className="flex items-center justify-center rounded-md"
          style={{ width: 28, height: 28, background: "var(--c-accent)", color: "#fff", fontSize: 16, fontWeight: 700 }}>◎</span>
        <div className="leading-tight">
          <div style={{ fontSize: 13, fontWeight: 600 }}>{tr(lang, "app.title")}</div>
          <div style={{ fontSize: 10, color: "var(--c-text-dim)" }}>{tr(lang, "app.subtitle")}</div>
        </div>
      </div>

      <div style={{ width: 1, height: 26, background: "var(--c-border)" }} className="flex-none mx-1" />

      {/* Identity */}
      <Tooltip title={tr(lang, "save.userTip")} arrow>
        <span className="flex items-center gap-1.5 rounded-full flex-none"
          style={{ fontSize: 12, padding: "4px 10px", background: "var(--c-surface)", border: "1px solid var(--c-border)" }}>
          <span style={{ fontSize: 12 }}>👤</span><b>{activeUser}</b>
        </span>
      </Tooltip>
      <Button variant="text" title={tr(lang, "save.switchTip")}
        onClick={() => useStore.setState({ activeUser: null, loaded: false, activeVolume: null })}
        sx={{ fontSize: 11, minWidth: 0, px: 0.75, color: "var(--c-text-dim)" }}>{tr(lang, "save.switch")}</Button>

      <div className="flex-1" />

      {/* Output folder */}
      <Tooltip title={outputDir ? `${tr(lang, "save.outputTip")}\n${outputDir}` : tr(lang, "save.outputTip")} arrow>
        <Button variant="outlined" onClick={() => chooseOutputDir()} sx={{ fontSize: 11, maxWidth: 220 }}>
          <span className="truncate">{outName ? `📁 ${outName}` : tr(lang, "save.setOutput")}</span>
        </Button>
      </Tooltip>

      {/* Load an existing segmentation to correct */}
      <Tooltip title={tr(lang, "save.loadTip")} arrow>
        <span>
          <Button variant="outlined" disabled={!loaded || busy} onClick={() => loadSegmentation()} sx={{ fontSize: 11 }}>
            {tr(lang, "save.load")}
          </Button>
        </span>
      </Tooltip>

      {/* Save */}
      <Button variant="contained" color="success" disableElevation disabled={!loaded || busy} onClick={() => save()}
        sx={{ fontWeight: 600, px: 1.75 }} title={tr(lang, "save.saveTip")}>
        {tr(lang, "save.save")}
      </Button>

      <div style={{ width: 1, height: 26, background: "var(--c-border)" }} className="flex-none mx-1" />

      {/* Updates */}
      <Tooltip title={tr(lang, "updates.tip")} arrow>
        <span className="flex-none">
          <Button variant="text" disabled={updateBusy} onClick={() => checkUpdates(true)}
            sx={{ fontSize: 11, minWidth: 0, px: 0.75, color: "var(--c-text-dim)" }}>
            {updateBusy ? tr(lang, "updates.checking") : `⟳ ${tr(lang, "updates.label")}`}
          </Button>
        </span>
      </Tooltip>

      {/* About */}
      <Button variant="text" onClick={() => setAboutOpen(true)}
        sx={{ fontSize: 11, minWidth: 0, px: 0.75, color: "var(--c-text-dim)" }}>{tr(lang, "about.label")}</Button>

      {/* Language toggle */}
      <ToggleButtonGroup size="small" exclusive value={lang} onChange={(_, v) => v && setLang(v)} className="flex-none">
        <ToggleButton value="en" sx={{ py: 0, px: 1, fontSize: 11, textTransform: "none" }}>EN</ToggleButton>
        <ToggleButton value="zh" sx={{ py: 0, px: 1, fontSize: 11, textTransform: "none" }}>中文</ToggleButton>
      </ToggleButtonGroup>

      <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} />

      {/* Overwrite confirmation (#1) */}
      <Dialog open={confirmOverwrite} onClose={() => cancelOverwrite()}>
        <DialogTitle>{tr(lang, "save.overwriteTitle")}</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {tr(lang, "save.overwriteBody")}{activeVolume ? `\n\n${activeVolume.name}` : ""}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => cancelOverwrite()}>{tr(lang, "save.cancel")}</Button>
          <Button variant="contained" color="success" disableElevation onClick={() => save(true)}>{tr(lang, "save.overwrite")}</Button>
        </DialogActions>
      </Dialog>
    </header>
  );
}
