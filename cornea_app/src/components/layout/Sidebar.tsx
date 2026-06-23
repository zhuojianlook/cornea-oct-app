import { useEffect, useState } from "react";
import { Button, Divider, TextField, Typography } from "@mui/material";
import { api } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { OctLoader } from "../panels/OctLoader";
import { NnunetTrainPanel } from "../panels/NnunetTrainPanel";
import { ManualGtPanel } from "../panels/ManualGtPanel";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="px-3 py-3">
      <div className="text-[11px] uppercase tracking-wide mb-2" style={{ color: "var(--c-text-dim)" }}>
        {title}
      </div>
      <div className="flex flex-col gap-2">{children}</div>
    </div>
  );
}

// Reopen any already-built case (or per-eye consensus) by id, without re-running the pipeline —
// e.g. a "<patient>_<eye>_consensus" case to inspect the replicate overlap in the 3D viewer.
function OpenCaseById() {
  const setCaseId = useCaseStore((s) => s.setCaseId);
  const openCase = useCaseStore((s) => s.openCase);
  const busy = useCaseStore((s) => s.busy);
  const [id, setId] = useState("");
  const open = async () => {
    const cid = id.trim();
    if (!cid) return;
    setCaseId(cid);
    await openCase();
  };
  return (
    <div className="flex gap-2">
      <TextField size="small" fullWidth placeholder="case id, e.g. case_cs001_os_consensus"
        value={id} onChange={(e) => setId(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") open(); }}
        InputProps={{ sx: { fontSize: 12 } }} />
      <Button variant="outlined" size="small" onClick={open} disabled={busy || !id.trim()}>Open</Button>
    </div>
  );
}

// Build the NORMAL reflectivity baseline from control scans, so depth-normalised scar detection
// flags only EXCESS over normal (ignores normal Bowman's/anterior hyper-reflectivity).
function NormalBaselinePanel() {
  const [info, setInfo] = useState<{ exists: boolean; controls: string[]; available_controls: string[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const load = async () => { try { setInfo(await api.json("/api/normal-profile")); } catch { /* sidecar may be down */ } };
  useEffect(() => { load(); }, []);
  const build = async () => {
    setBusy(true); setMsg(null);
    try {
      const r = await api.json<{ n_controls: number }>("/api/normal-profile/build", "POST", JSON.stringify({}));
      setMsg(`Built from ${r.n_controls} control scan(s).`);
      await load();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  };
  const avail = info?.available_controls?.length ?? 0;
  return (
    <div className="flex flex-col gap-2">
      <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
        Learn normal corneal brightness from control scans so the “Depth-normalised” scar method ignores
        normal Bowman's hyper-reflectivity. Tag scans “control”, segment the cornea, then build.
      </Typography>
      <div className="text-xs" style={{ color: "var(--c-text-dim)" }}>
        {info?.exists ? `Baseline: ${info.controls.length} control(s)` : "No baseline yet"} · {avail} labelled control(s) available
      </div>
      <Button variant="outlined" size="small" disabled={busy || avail === 0} onClick={build}>
        {info?.exists ? "Rebuild normal baseline" : "Build normal baseline"}
      </Button>
      {msg && <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-word" }}>{msg}</Typography>}
    </div>
  );
}

export function Sidebar() {
  return (
    <div className="flex flex-col">
      <Section title="Open existing case">
        <OpenCaseById />
      </Section>

      <Divider sx={{ borderColor: "var(--c-border)" }} />

      <Section title="OCT preprocessing">
        <OctLoader />
      </Section>

      <Divider sx={{ borderColor: "var(--c-border)" }} />

      <Section title="Normal baseline (controls)">
        <NormalBaselinePanel />
      </Section>

      <Divider sx={{ borderColor: "var(--c-border)" }} />

      <Section title="Manual ground truth (compare)">
        <ManualGtPanel />
      </Section>

      <Divider sx={{ borderColor: "var(--c-border)" }} />

      <Section title="nnU-Net training (proof of concept)">
        <NnunetTrainPanel />
      </Section>
    </div>
  );
}
