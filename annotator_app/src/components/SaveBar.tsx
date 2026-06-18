/* App header: brand, who is annotating (+ switch user), the output folder, and Save ground truth.
   Transient status + session id live in the bottom StatusBar, keeping this header uncluttered. */

import { Button, Tooltip } from "@mui/material";
import { useStore } from "../store/annotatorStore";

export function SaveBar() {
  const { activeUser, outputDir, loaded, busy, save, chooseOutputDir } = useStore();
  const outName = outputDir ? outputDir.split(/[/\\]/).filter(Boolean).pop() : null;

  return (
    <header className="flex items-center gap-3 px-4 border-b min-w-0"
      style={{ height: 54, flex: "none", backgroundColor: "var(--c-surface2)", borderColor: "var(--c-border)" }}>
      {/* Brand */}
      <div className="flex items-center gap-2.5 flex-none">
        <span className="flex items-center justify-center rounded-md"
          style={{ width: 28, height: 28, background: "var(--c-accent)", color: "#fff", fontSize: 16, fontWeight: 700 }}>◎</span>
        <div className="leading-tight">
          <div style={{ fontSize: 13, fontWeight: 600 }}>Ground-Truth Annotator</div>
          <div style={{ fontSize: 10, color: "var(--c-text-dim)" }}>Cornea scar segmentation</div>
        </div>
      </div>

      <div style={{ width: 1, height: 26, background: "var(--c-border)" }} className="flex-none mx-1" />

      {/* Identity */}
      <Tooltip title="The active annotator — recorded with every saved label for inter-/intra-observer analysis" arrow>
        <span className="flex items-center gap-1.5 rounded-full flex-none"
          style={{ fontSize: 12, padding: "4px 10px", background: "var(--c-surface)", border: "1px solid var(--c-border)" }}>
          <span style={{ fontSize: 12 }}>👤</span><b>{activeUser}</b>
        </span>
      </Tooltip>
      <Button variant="text" title="Switch to a different annotator"
        onClick={() => useStore.setState({ activeUser: null, loaded: false, activeVolume: null })}
        sx={{ fontSize: 11, minWidth: 0, px: 0.75, color: "var(--c-text-dim)" }}>switch</Button>

      <div className="flex-1" />

      {/* Output folder */}
      <Tooltip title={outputDir ? `Ground-truth output folder:\n${outputDir}` : "Choose where annotations are saved"} arrow>
        <Button variant="outlined" onClick={() => chooseOutputDir()} sx={{ fontSize: 11, maxWidth: 220 }}>
          <span className="truncate">{outName ? `📁 ${outName}` : "Set output folder…"}</span>
        </Button>
      </Tooltip>

      {/* Save */}
      <Button variant="contained" color="success" disableElevation disabled={!loaded || busy} onClick={() => save()}
        sx={{ fontWeight: 600, px: 1.75 }}
        title="Write the painted labelmap (0/1/2) + a manifest row tagged with your username and this session">
        Save ground truth
      </Button>
    </header>
  );
}
