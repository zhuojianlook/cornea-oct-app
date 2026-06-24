/* Sidebar: pick a folder of NIfTI volumes; annotate a BLINDED, replicate-expanded queue (#4). Real scan
   names are HIDDEN ("Scan A · rep 1") so inter-/intra-observer data is unbiased; an admin (password
   OCTAPP) can reveal real names and set how many replicates each scan gets. A ✓ marks entries this user
   has already saved (per replicate). */

import { useState } from "react";
import { Button, TextField, MenuItem, Select } from "@mui/material";
import { useStore, type BlindEntry } from "../store/annotatorStore";
import { tr } from "../i18n";

export function VolumeBrowser() {
  const { folder, blindEntries, activeVolume, annotated, busy, pickFolder, openVolume, nextUnannotated, lang,
          adminUnlocked, unlockAdmin, lockAdmin, replicates, setReplicates,
          deleteGt, downloadGt, exportAllGt } = useStore();
  const [pwOpen, setPwOpen] = useState(false);
  const [pw, setPw] = useState("");
  const [pwErr, setPwErr] = useState(false);
  // The saved entry whose ✓ was clicked → show the manage modal (modify / download / delete).
  const [manage, setManage] = useState<BlindEntry | null>(null);
  const [confirmDel, setConfirmDel] = useState(false);
  const folderName = folder ? folder.split(/[/\\]/).filter(Boolean).pop() : null;
  const keyOf = (e: { stem: string; replicate: number }) => `${e.stem}__rep${e.replicate}`;
  const nDone = blindEntries.filter((e) => annotated.has(keyOf(e))).length;

  const tryUnlock = () => { if (unlockAdmin(pw.trim())) { setPwOpen(false); setPw(""); setPwErr(false); } else setPwErr(true); };

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center justify-between px-4 pt-3 pb-2 flex-none">
        <span style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--c-text-dim)" }}>{tr(lang, "vol.volumes")}</span>
        {blindEntries.length > 0 && (
          <span style={{ fontSize: 10, color: "var(--c-text-dim)" }}>
            {nDone > 0 && <b style={{ color: "var(--c-green)" }}>{nDone} {tr(lang, "vol.done")}</b>}{nDone > 0 ? " / " : ""}{blindEntries.length}
          </span>
        )}
      </div>

      <div className="px-4 flex flex-col gap-1.5 flex-none">
        <Button variant="outlined" size="small" fullWidth disabled={busy} onClick={() => pickFolder()}>
          {folder ? tr(lang, "vol.change") : tr(lang, "vol.pick")}
        </Button>
        {blindEntries.length > 0 && (
          <Button variant="contained" size="small" fullWidth disableElevation disabled={busy}
            onClick={() => nextUnannotated()} title={tr(lang, "vol.nextTip")}>
            {tr(lang, "vol.next")}
          </Button>
        )}
        {nDone > 0 && (
          <Button variant="text" size="small" fullWidth disabled={busy} onClick={() => exportAllGt()}
            sx={{ fontSize: 11, textTransform: "none" }}
            title="Copy every saved labelmap (+ the manifest) to a folder you choose">
            ⬇ Export all saved labelmaps ({nDone})
          </Button>
        )}
        {folderName && (
          <div className="flex items-center gap-1.5 rounded px-2 py-1 truncate" title={adminUnlocked ? (folder ?? "") : "folder hidden (blinded)"}
            style={{ fontSize: 10, color: "var(--c-text-dim)", background: "var(--c-surface2)", border: "1px solid var(--c-border)" }}>
            <span style={{ flex: "none" }}>📁</span><span className="truncate">{adminUnlocked ? folderName : "(blinded)"}</span>
          </div>
        )}
        {/* #4: blinding notice + admin unlock */}
        <div className="flex items-center justify-between" style={{ fontSize: 10, color: "var(--c-text-dim)" }}>
          <span title="Real scan names are hidden so your reads are unbiased. Each scan is repeated for intra-observer analysis.">
            {adminUnlocked ? "🔓 admin — names revealed" : "🔒 blinded"}
          </span>
          {adminUnlocked
            ? <button onClick={() => lockAdmin()} style={{ background: "none", border: "none", color: "var(--c-accent)", cursor: "pointer", fontSize: 10, padding: 0 }}>lock</button>
            : <button onClick={() => { setPwOpen((v) => !v); setPwErr(false); }} style={{ background: "none", border: "none", color: "var(--c-accent)", cursor: "pointer", fontSize: 10, padding: 0 }}>admin…</button>}
        </div>
        {pwOpen && !adminUnlocked && (
          <div className="flex items-center gap-1">
            <TextField type="password" size="small" placeholder="admin password" value={pw} error={pwErr}
              onChange={(e) => { setPw(e.target.value); setPwErr(false); }}
              onKeyDown={(e) => { if (e.key === "Enter") tryUnlock(); }}
              InputProps={{ sx: { fontSize: 11 } }} sx={{ flex: 1 }} />
            <Button size="small" variant="contained" disableElevation onClick={tryUnlock} sx={{ fontSize: 10, minWidth: 0, px: 1 }}>OK</Button>
          </div>
        )}
        {adminUnlocked && (
          <div className="flex items-center gap-2" style={{ fontSize: 10, color: "var(--c-text-dim)" }} title="Repeats per scan (each user annotates every scan this many times) — for intra-observer reproducibility.">
            replicates
            <Select size="small" value={replicates} onChange={(e) => setReplicates(Number(e.target.value))} disabled={busy}
              sx={{ fontSize: 11, ".MuiSelect-select": { py: 0.2 } }}>
              {[2, 3, 4].map((n) => <MenuItem key={n} value={n} sx={{ fontSize: 11 }}>{n}×</MenuItem>)}
            </Select>
          </div>
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-2 py-2 mt-1 flex flex-col gap-0.5" style={{ scrollbarGutter: "stable" }}>
        {!folder && (
          <div className="flex flex-col items-center text-center gap-1 px-3" style={{ marginTop: 28, color: "var(--c-text-dim)" }}>
            <span style={{ fontSize: 26, opacity: 0.6 }}>📂</span>
            <span style={{ fontSize: 12 }}>{tr(lang, "vol.noFolder")}</span>
            <span style={{ fontSize: 11 }}>{tr(lang, "vol.noFolderHint")}</span>
          </div>
        )}
        {folder && blindEntries.length === 0 && (
          <div className="text-center px-3" style={{ marginTop: 24, fontSize: 11, color: "var(--c-text-dim)" }}>
            {tr(lang, "vol.noFiles")}
          </div>
        )}
        {blindEntries.map((e) => {
          const done = annotated.has(keyOf(e));
          const active = activeVolume != null && activeVolume.path === e.path && activeVolume.replicate === e.replicate;
          const label = adminUnlocked ? `${e.stem} · rep ${e.replicate}` : e.name;
          return (
            <div key={keyOf(e)}
              className="flex items-center gap-2 rounded transition-colors min-w-0"
              style={{
                padding: "0 9px", fontSize: 12, color: "var(--c-text)", opacity: busy && !active ? 0.5 : 1,
                background: active ? "rgba(122,166,214,0.20)" : "transparent",
                borderLeft: `2px solid ${active ? "var(--c-accent)" : "transparent"}`,
              }}
              onMouseEnter={(ev) => { if (!busy && !active) ev.currentTarget.style.background = "var(--c-surface2)"; }}
              onMouseLeave={(ev) => { if (!active) ev.currentTarget.style.background = "transparent"; }}>
              {done ? (
                <button onClick={(ev) => { ev.stopPropagation(); setManage(e); setConfirmDel(false); }} disabled={busy}
                  title="Saved ground truth — modify, download, or delete"
                  style={{ width: 16, flex: "none", textAlign: "center", color: "var(--c-green)", fontSize: 13,
                           background: "none", border: "none", cursor: busy ? "default" : "pointer", padding: "7px 0" }}>✓</button>
              ) : (
                <span style={{ width: 16, flex: "none", textAlign: "center", color: "var(--c-text-dim)", fontSize: 12 }}>○</span>
              )}
              <button onClick={() => !busy && openVolume(e)} disabled={busy}
                className="truncate text-left" title={label}
                style={{ flex: 1, minWidth: 0, background: "none", border: "none", color: "inherit",
                         cursor: busy ? "default" : "pointer", fontWeight: active ? 600 : 400, padding: "7px 0" }}>
                {label}
              </button>
            </div>
          );
        })}
      </div>

      {/* Manage a saved ground truth (opened from its ✓ badge): modify / download / delete. */}
      {manage && (
        <div onClick={() => { setManage(null); setConfirmDel(false); }}
          style={{ position: "fixed", inset: 0, zIndex: 50, background: "rgba(0,0,0,0.5)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div onClick={(ev) => ev.stopPropagation()}
            style={{ width: 320, maxWidth: "90vw", background: "var(--c-surface)", border: "1px solid var(--c-border)", borderRadius: 8, padding: 16, display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--c-text)" }}>Saved ground truth</div>
            <div style={{ fontSize: 12, color: "var(--c-text-dim)" }}>
              {(adminUnlocked ? manage.stem : manage.blindLabel)} · rep {manage.replicate}
            </div>
            {!confirmDel ? (
              <>
                <Button variant="outlined" size="small" disabled={busy}
                  onClick={() => { const e = manage; setManage(null); openVolume(e); }}>
                  ✎ Modify (re-open to edit)
                </Button>
                <Button variant="outlined" size="small" disabled={busy} onClick={() => { void downloadGt(manage); }}>
                  ⬇ Download labelmap
                </Button>
                <Button variant="outlined" size="small" color="error" disabled={busy} onClick={() => setConfirmDel(true)}>
                  🗑 Delete ground truth
                </Button>
                <Button variant="text" size="small" onClick={() => { setManage(null); setConfirmDel(false); }}>Cancel</Button>
              </>
            ) : (
              <>
                <div style={{ fontSize: 12, color: "var(--c-text)" }}>
                  Delete this saved ground truth? Removes the labelmap file(s) + manifest entry — cannot be undone.
                </div>
                <div className="flex gap-2">
                  <Button variant="contained" size="small" color="error" fullWidth disabled={busy}
                    onClick={async () => { const e = manage; setManage(null); setConfirmDel(false); await deleteGt(e); }}>
                    Delete
                  </Button>
                  <Button variant="outlined" size="small" fullWidth onClick={() => setConfirmDel(false)}>Cancel</Button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
