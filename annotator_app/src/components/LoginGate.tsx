/* Username gate shown on launch. Users are defined here and selectable via dropdown; a username is
   REQUIRED before entering. The active username + session is later saved with every annotation so
   inter-observer (different users) and intra-observer (same user, different sessions) can be computed. */

import { useState } from "react";
import { Button, MenuItem, Select, TextField, Typography } from "@mui/material";
import { useStore } from "../store/annotatorStore";

export function LoginGate() {
  const users = useStore((s) => s.users);
  const addUser = useStore((s) => s.addUser);
  const selectUser = useStore((s) => s.selectUser);
  const [pick, setPick] = useState("");
  const [name, setName] = useState("");

  return (
    <div className="flex h-full w-full items-center justify-center" style={{ backgroundColor: "var(--c-bg)", color: "var(--c-text)" }}>
      <div className="flex flex-col gap-4 p-8 rounded-lg" style={{ width: 420, backgroundColor: "var(--c-surface)", border: "1px solid var(--c-border)" }}>
        <div>
          <Typography variant="h6">Cornea Ground-Truth Annotator</Typography>
          <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
            Choose who is annotating. Your username is recorded with every saved label for inter-/intra-observer analysis.
          </Typography>
        </div>

        {users.length > 0 && (
          <div className="flex gap-2 items-center">
            <Select size="small" fullWidth displayEmpty value={pick} onChange={(e) => setPick(e.target.value)}
              sx={{ fontSize: 14, color: "var(--c-text)", "& fieldset": { borderColor: "var(--c-border)" } }}>
              <MenuItem value="" disabled><em>Select existing user…</em></MenuItem>
              {users.map((u) => <MenuItem key={u} value={u} sx={{ fontSize: 14 }}>{u}</MenuItem>)}
            </Select>
            <Button variant="contained" disabled={!pick} onClick={() => selectUser(pick)}>Enter</Button>
          </div>
        )}

        <div className="flex items-center gap-2">
          <div style={{ flex: 1, height: 1, background: "var(--c-border)" }} />
          <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>or add a new user</span>
          <div style={{ flex: 1, height: 1, background: "var(--c-border)" }} />
        </div>

        <div className="flex gap-2">
          <TextField size="small" fullWidth placeholder="new username" value={name}
            onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && name.trim()) addUser(name); }}
            InputProps={{ sx: { fontSize: 14 } }} />
          <Button variant="outlined" disabled={!name.trim()} onClick={() => addUser(name)}>Add &amp; enter</Button>
        </div>
      </div>
    </div>
  );
}
