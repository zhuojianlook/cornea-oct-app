/* Sidebar: pick a folder of NIfTI volumes and select one to annotate. A ✓ marks volumes this user
   has already saved (in the current output dir). */

import { Button } from "@mui/material";
import { useStore } from "../store/annotatorStore";
import { tr } from "../i18n";

export function VolumeBrowser() {
  const { folder, volumes, activeVolume, annotated, busy, pickFolder, openVolume, lang } = useStore();
  const folderName = folder ? folder.split(/[/\\]/).filter(Boolean).pop() : null;
  const nDone = volumes.filter((v) => annotated.has(v.name.replace(/\.nii(\.gz)?$/i, ""))).length;

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-3 pb-2 flex-none">
        <span style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--c-text-dim)" }}>{tr(lang, "vol.volumes")}</span>
        {volumes.length > 0 && (
          <span style={{ fontSize: 10, color: "var(--c-text-dim)" }}>
            {nDone > 0 && <b style={{ color: "var(--c-green)" }}>{nDone} {tr(lang, "vol.done")}</b>}{nDone > 0 ? " / " : ""}{volumes.length}
          </span>
        )}
      </div>

      {/* Folder picker + path pill */}
      <div className="px-4 flex flex-col gap-1.5 flex-none">
        <Button variant="outlined" size="small" fullWidth disabled={busy} onClick={() => pickFolder()}>
          {folder ? tr(lang, "vol.change") : tr(lang, "vol.pick")}
        </Button>
        {folderName && (
          <div className="flex items-center gap-1.5 rounded px-2 py-1 truncate" title={folder ?? ""}
            style={{ fontSize: 10, color: "var(--c-text-dim)", background: "var(--c-surface2)", border: "1px solid var(--c-border)" }}>
            <span style={{ flex: "none" }}>📁</span><span className="truncate">{folderName}</span>
          </div>
        )}
      </div>

      {/* List / empty states */}
      <div className="flex-1 min-h-0 overflow-y-auto px-2 py-2 mt-1 flex flex-col gap-0.5" style={{ scrollbarGutter: "stable" }}>
        {!folder && (
          <div className="flex flex-col items-center text-center gap-1 px-3" style={{ marginTop: 28, color: "var(--c-text-dim)" }}>
            <span style={{ fontSize: 26, opacity: 0.6 }}>📂</span>
            <span style={{ fontSize: 12 }}>{tr(lang, "vol.noFolder")}</span>
            <span style={{ fontSize: 11 }}>{tr(lang, "vol.noFolderHint")}</span>
          </div>
        )}
        {folder && volumes.length === 0 && (
          <div className="text-center px-3" style={{ marginTop: 24, fontSize: 11, color: "var(--c-text-dim)" }}>
            {tr(lang, "vol.noFiles")}
          </div>
        )}
        {volumes.map((v) => {
          const stem = v.name.replace(/\.nii(\.gz)?$/i, "");
          const done = annotated.has(stem);
          const active = activeVolume?.path === v.path;
          return (
            <button key={v.path} onClick={() => !busy && openVolume(v)} disabled={busy}
              className="flex items-center gap-2 rounded text-left transition-colors min-w-0"
              style={{
                cursor: busy ? "default" : "pointer", padding: "7px 9px", fontSize: 12,
                color: "var(--c-text)", opacity: busy && !active ? 0.5 : 1,
                background: active ? "rgba(122,166,214,0.20)" : "transparent",
                borderLeft: `2px solid ${active ? "var(--c-accent)" : "transparent"}`,
              }}
              onMouseEnter={(e) => { if (!busy && !active) e.currentTarget.style.background = "var(--c-surface2)"; }}
              onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}>
              <span style={{ width: 14, flex: "none", textAlign: "center", color: done ? "var(--c-green)" : "var(--c-text-dim)", fontSize: 12 }}>
                {done ? "✓" : "○"}
              </span>
              <span className="truncate" style={{ flex: 1, fontWeight: active ? 600 : 400 }} title={v.name}>{v.name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
