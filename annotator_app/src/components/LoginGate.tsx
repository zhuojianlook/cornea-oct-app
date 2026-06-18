/* Username gate shown on launch. Users are defined here and selectable via dropdown; a username is
   REQUIRED before entering. The active username + session is later saved with every annotation so
   inter-observer (different users) and intra-observer (same user, different sessions) can be computed. */

import { useState } from "react";
import { Button, MenuItem, Select, TextField, Typography } from "@mui/material";
import { useStore } from "../store/annotatorStore";

const label = { fontSize: 10, textTransform: "uppercase" as const, letterSpacing: "0.07em", color: "var(--c-text-dim)" };

export function LoginGate() {
  const users = useStore((s) => s.users);
  const addUser = useStore((s) => s.addUser);
  const selectUser = useStore((s) => s.selectUser);
  const [pick, setPick] = useState("");
  const [name, setName] = useState("");

  return (
    <div className="flex h-full w-full items-center justify-center p-6" style={{ backgroundColor: "var(--c-bg)", color: "var(--c-text)" }}>
      <div className="flex flex-col gap-5 rounded-xl"
        style={{ width: "min(440px, calc(100% - 32px))", padding: 32, backgroundColor: "var(--c-surface)", border: "1px solid var(--c-border)", boxShadow: "0 12px 40px rgba(0,0,0,0.35)" }}>

        {/* Brand */}
        <div className="flex items-center gap-3">
          <span className="flex items-center justify-center rounded-lg"
            style={{ width: 40, height: 40, background: "var(--c-accent)", color: "#fff", fontSize: 22, fontWeight: 700 }}>◎</span>
          <div className="leading-tight">
            <Typography sx={{ fontSize: 17, fontWeight: 600 }}>Cornea Ground-Truth Annotator</Typography>
            <Typography sx={{ fontSize: 12, color: "var(--c-text-dim)" }}>Manual scar segmentation</Typography>
          </div>
        </div>

        <Typography sx={{ fontSize: 12.5, color: "var(--c-text-dim)", lineHeight: 1.55 }}>
          Choose who is annotating. Your username is recorded with every saved label — enabling
          inter-observer (different people) and intra-observer (same person, different sessions) analysis.
        </Typography>

        {users.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <span style={label}>Existing user</span>
            <div className="flex gap-2 items-center">
              <Select size="small" fullWidth displayEmpty value={pick} onChange={(e) => setPick(e.target.value)} sx={{ fontSize: 14 }}>
                <MenuItem value="" disabled><em>Select a user…</em></MenuItem>
                {users.map((u) => <MenuItem key={u} value={u} sx={{ fontSize: 14 }}>{u}</MenuItem>)}
              </Select>
              <Button variant="contained" disableElevation disabled={!pick} onClick={() => selectUser(pick)} sx={{ flex: "none", px: 2 }}>Enter</Button>
            </div>
          </div>
        )}

        <div className="flex items-center gap-3">
          <div style={{ flex: 1, height: 1, background: "var(--c-border)" }} />
          <span style={{ fontSize: 11, color: "var(--c-text-dim)" }}>{users.length > 0 ? "or add a new user" : "add a user to begin"}</span>
          <div style={{ flex: 1, height: 1, background: "var(--c-border)" }} />
        </div>

        <div className="flex flex-col gap-1.5">
          <span style={label}>New user</span>
          <div className="flex gap-2">
            <TextField size="small" fullWidth placeholder="username" value={name}
              onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && name.trim()) addUser(name); }}
              InputProps={{ sx: { fontSize: 14 } }} />
            <Button variant="outlined" disabled={!name.trim()} onClick={() => addUser(name)} sx={{ flex: "none", px: 2 }}>Add &amp; enter</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
