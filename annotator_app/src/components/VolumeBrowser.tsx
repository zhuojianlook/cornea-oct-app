/* Sidebar: pick a folder of NIfTI volumes and select one to annotate. A ✓ marks volumes this user
   has already saved (in the current output dir). */

import { Button, Typography } from "@mui/material";
import { useStore } from "../store/annotatorStore";

export function VolumeBrowser() {
  const { folder, volumes, activeVolume, annotated, busy, pickFolder, openVolume } = useStore();

  return (
    <div className="flex flex-col gap-2 p-3">
      <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>Volumes</Typography>
      <Button variant="outlined" size="small" disabled={busy} onClick={() => pickFolder()}>
        {folder ? "Change folder…" : "Pick folder of NIfTI…"}
      </Button>
      {folder && (
        <div className="text-[10px] truncate" style={{ color: "var(--c-text-dim)" }} title={folder}>{folder}</div>
      )}
      <div className="flex flex-col gap-0.5 mt-1">
        {volumes.length === 0 && folder && (
          <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>No .nii/.nii.gz files here.</span>
        )}
        {volumes.map((v) => {
          const stem = v.name.replace(/\.nii(\.gz)?$/i, "");
          const done = annotated.has(stem);
          const active = activeVolume?.path === v.path;
          return (
            <button key={v.path} onClick={() => !busy && openVolume(v)}
              className="flex items-center gap-2 text-xs rounded px-2 py-1.5 text-left"
              style={{ cursor: busy ? "default" : "pointer", background: active ? "var(--c-surface2)" : "transparent",
                       borderLeft: active ? "2px solid var(--c-accent)" : "2px solid transparent" }}>
              <span style={{ width: 12, color: done ? "var(--c-green)" : "var(--c-text-dim)" }}>{done ? "✓" : ""}</span>
              <span className="truncate" style={{ flex: 1 }}>{v.name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
