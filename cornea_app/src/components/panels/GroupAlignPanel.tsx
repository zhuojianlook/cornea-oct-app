/* ──────────────────────────────────────────────────────────
   Step 4 "Aligned" dialogs (reviewer spec 2026-09-11):
   • GroupAlignPanel — the "⧉ Alignment (pairs)…" dialog: a Dialog around the shared AlignTabs content (Consensus + pairs /
     3-D view / Scrub). The same content is embedded in the main area at step 4 (VolumeCanvas).
   • SubgroupAlignDialog — what "⧉ Align group" opens FIRST (#1): asks whether all scans of the eye (e.g. CS001_OS) belong
     to the same subgroup or whether different subgroups must be specified; "Confirm and align" writes the subgroups
     (POST /api/group/{gid}/subgroups → manifest scar_subgroup + align_subgroup_confirmed) and starts one align job per
     subgroup (each subgroup aligns as its own group, id <patient>_<eye>_s<k>); the job's completion stamps group_aligned
     on its scans, which advances them to "4. Aligned" (#2). The dialog polls the started / queued jobs and reports.
   ────────────────────────────────────────────────────────── */
import { useEffect, useRef, useState } from "react";
import { Button, Checkbox, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, Radio, RadioGroup } from "@mui/material";
import { api, resourceUrl } from "../../api/client";
import { AlignTabs, type AlignStatus } from "./AlignTabs";

export function GroupAlignPanel({ open, gid, onClose, autoStart }:
  { open: boolean; gid: string | null; onClose: () => void; autoStart?: boolean }) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth={false} fullWidth PaperProps={{ sx: { width: "96vw", maxWidth: 1700, height: "92vh" } }}>
      <DialogTitle sx={{ fontSize: 15, py: 1 }}>⧉ Alignment (pairs) — group <b>{gid ?? "?"}</b></DialogTitle>
      <DialogContent dividers sx={{ p: 0.5, display: "flex", flexDirection: "column" }}>
        {open && <AlignTabs gid={gid} autoStart={autoStart} title={<span style={{ color: "var(--c-text-dim)", fontSize: 12 }}>group <b>{gid ?? "?"}</b></span>} />}
      </DialogContent>
      <DialogActions sx={{ py: 0.5 }}>
        <Button size="small" onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

// GET /api/group/{gid}/subgroups
export interface SubgroupScan {
  case_id: string; subgroup: string; align_group: string; align_subgroup_confirmed: boolean; subgroup_confirmed: boolean;
  group_aligned: Record<string, unknown> | null; aligned_approved: Record<string, unknown> | null; preproc_vetted: boolean; sam2_meta: boolean;
}
export interface SubgroupsInfo {
  ok: boolean; group: string; patient: string | null; eye: string | null; scans: SubgroupScan[];
  subgroups: Record<string, { group: string; members: string[]; status: AlignStatus | null; alignable: boolean; error?: string }>;
  queued: string[];
}
// POST /api/group/{gid}/subgroups
export interface SubgroupsJob extends AlignStatus { subgroup: string; queued?: boolean; queued_behind?: string | null }
export interface SubgroupsPost {
  ok: boolean; group: string; patient: string | null; eye: string | null; written: Record<string, string>; rejected: string[];
  subgroups: Record<string, string>; jobs: SubgroupsJob[]; skipped: { subgroup: string; group: string; members: string[]; reason: string }[];
}

const short = (cid: string): string => cid.startsWith("case_") ? cid.slice(5) : cid;
const SUB_RGB = ["#ff5050", "#5ac86e", "#5a96ff", "#ebc846", "#d26eeb", "#5adcdc"];
const subColor = (label: string) => SUB_RGB[(Math.max(1, parseInt(label || "1", 10) || 1) - 1) % SUB_RGB.length];

/** A job is finished for the dialog's purpose when its result landed AND its scans were stamped (group_aligned), or it failed. */
const jobSettled = (s: AlignStatus | null | undefined): boolean =>
  !!s && !s.running && (!!s.error || (s.done && !!s.stamp?.stamped_timestamp));

export function SubgroupAlignDialog({ open, gid, onClose, onAligned, defaultForce }:
  { open: boolean; gid: string | null; onClose: () => void; onAligned?: (r: { groups: string[]; ok: boolean }) => void;
    /** Pre-check "re-run a subgroup that already has an alignment result" — set when the dialog is opened from an
     *  already-aligned group's "⧉ Align group" button (reviewer 2026-09-12: re-specify the subgroups, then re-run). */
    defaultForce?: boolean }) {
  const [info, setInfo] = useState<SubgroupsInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [mode, setMode] = useState<"one" | "many">("one");
  const [assign, setAssign] = useState<Record<string, string>>({});
  const [force, setForce] = useState(false);
  const [posting, setPosting] = useState(false);
  const [post, setPost] = useState<SubgroupsPost | null>(null);
  const [jobs, setJobs] = useState<Record<string, AlignStatus>>({});   // live status per watched subgroup group id
  const [phase, setPhase] = useState<"ask" | "running" | "done">("ask");
  // Middle-sagittal thumbnail per scan (reviewer 2026-09-12: "show the middle sagittal slice for each scan — this will aid the
  // user in determining if the scans belong to the same subgroup"): GET /api/case/{id}/sagittal-thumb?lateral=mid, cached
  // server-side under previews/. Click ⇒ full-size zoom dialog. `thumbNonce` busts the browser cache per dialog open.
  const [zoom, setZoom] = useState<string | null>(null);             // case id shown full-size, or null
  const [thumbErr, setThumbErr] = useState<Record<string, boolean>>({});
  const [thumbNonce] = useState(() => Date.now());
  // Reviewer 2026-09-12: SCRUB each thumbnail through that scan's sagittal slices, INDEPENDENTLY per scan (each slider
  // moves only its own scan). `lat` is the shown lateral per scan (default L/2 from GET …/sagittal-thumb/meta).
  // Neighbouring slices are prefetched; the sidecar serves them from a per-case uint8 stack (~30 ms) after one build.
  const [thumbMeta, setThumbMeta] = useState<Record<string, { laterals: number; mid: number }>>({});
  const [lat, setLat] = useState<Record<string, number>>({});
  const prefetch = useRef<Map<string, HTMLImageElement>>(new Map());
  const thumbUrl = (cid: string, l?: number) =>
    resourceUrl(`/api/case/${encodeURIComponent(cid)}/sagittal-thumb?lateral=${l === undefined ? "mid" : Math.round(l)}&t=${thumbNonce}`);
  const nLat = (cid: string) => thumbMeta[cid]?.laterals ?? 0;
  const latOf = (cid: string) => lat[cid] ?? thumbMeta[cid]?.mid ?? 0;
  /** Move THIS scan's thumbnail to `next`, clamped to its own lateral count (the sliders are independent per scan). */
  const setLateral = (cid: string, next: number) =>
    setLat((m) => ({ ...m, [cid]: Math.max(0, Math.min(Math.max(0, nLat(cid) - 1), Math.round(next))) }));
  const pollRef = useRef<number | null>(null);
  const stopPoll = () => { if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; } };

  useEffect(() => {
    if (!open || !gid) { stopPoll(); return; }
    let cancelled = false;
    setInfo(null); setErr(null); setPost(null); setJobs({}); setPhase("ask"); setPosting(false); setForce(!!defaultForce); setZoom(null); setThumbErr({});
    api.json<SubgroupsInfo>(`/api/group/${encodeURIComponent(gid)}/subgroups`).then((r) => {
      if (cancelled) return;
      setInfo(r);
      const a: Record<string, string> = {};
      for (const s of r.scans) a[s.case_id] = s.subgroup || "1";
      setAssign(a);
      setMode(new Set(Object.values(a)).size > 1 ? "many" : "one");
      // lateral count per scan, so each thumbnail can be scrubbed (reviewer 2026-09-12)
      for (const sc of r.scans) {
        void api.json<{ laterals: number; mid: number }>(`/api/case/${encodeURIComponent(sc.case_id)}/sagittal-thumb/meta`)
          .then((m) => { if (!cancelled && m?.laterals) setThumbMeta((t) => ({ ...t, [sc.case_id]: { laterals: m.laterals, mid: m.mid } })); })
          .catch(() => undefined);
      }
    }).catch((e) => !cancelled && setErr(e instanceof Error ? e.message : String(e)));
    return () => { cancelled = true; stopPoll(); };
  }, [open, gid, defaultForce]);

  // reset the scrub state on every open; prefetch the neighbours of each shown slice
  useEffect(() => { if (open) { setThumbMeta({}); setLat({}); prefetch.current.clear(); } }, [open, gid]);
  useEffect(() => {
    for (const cid of Object.keys(thumbMeta)) {
      const here = latOf(cid), n = nLat(cid);
      for (const d of [-1, 1, -2, 2]) {
        const l = here + d;
        if (l < 0 || l >= n) continue;
        const u = thumbUrl(cid, l);
        if (prefetch.current.has(u)) continue;
        const im = new Image(); im.src = u; prefetch.current.set(u, im);
        if (prefetch.current.size > 96) { const k = prefetch.current.keys().next().value; if (k !== undefined) prefetch.current.delete(k); }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lat, thumbMeta]);

  /** What the user typed for this scan (may be EMPTY while editing — the input shows this, so backspace works). */
  const typed = (cid: string) => assign[cid] ?? "1";
  /** The subgroup actually USED (grouping, colour, confirm): the typed value, or "1" when the field is blank. */
  const effective = (cid: string) => (mode === "one" ? "1" : typed(cid).trim() || "1");
  const groups: Record<string, string[]> = {};
  for (const s of info?.scans ?? []) (groups[effective(s.case_id)] ??= []).push(s.case_id);
  const singletons = Object.entries(groups).filter(([, m]) => m.length < 2);

  const finish = (watched: string[], statuses: Record<string, AlignStatus>) => {
    stopPoll();
    setPhase("done");
    const ok = watched.every((g) => statuses[g] && !statuses[g].error);
    onAligned?.({ groups: watched, ok });
  };

  const confirm = async () => {
    if (!gid || !info) return;
    setPosting(true); setErr(null);
    try {
      const assignments: Record<string, string> = {};
      for (const s of info.scans) assignments[s.case_id] = effective(s.case_id);
      const r = await api.json<SubgroupsPost>(`/api/group/${encodeURIComponent(gid)}/subgroups`, "POST",
        JSON.stringify({ assignments, confirm: true, force }));
      setPost(r);
      const watched = r.jobs.map((j) => j.group);
      const init: Record<string, AlignStatus> = {};
      for (const j of r.jobs) init[j.group] = j;
      setJobs(init);
      if (watched.length === 0 || watched.every((g) => jobSettled(init[g]))) { setPhase("done"); finish(watched, init); return; }
      setPhase("running");
      stopPoll();
      pollRef.current = window.setInterval(() => {
        void (async () => {
          const next: Record<string, AlignStatus> = { ...init };
          for (const g of watched) {
            try { next[g] = await api.json<AlignStatus>(`/api/group/${encodeURIComponent(g)}/align/status`); }
            catch (e) { next[g] = { ...(next[g] ?? init[g]), error: e instanceof Error ? e.message : String(e) } as AlignStatus; }
          }
          setJobs(next);
          if (watched.every((g) => jobSettled(next[g]))) finish(watched, next);
        })();
      }, 3000);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setPosting(false);
    }
  };

  const label = info ? `${(info.patient ?? "").toUpperCase()}_${(info.eye ?? "").toUpperCase()}`.replace(/^_|_$/g, "") || info.group : (gid ?? "?");
  const fmtJob = (g: string, s: AlignStatus | undefined) => {
    if (!s) return `${g}: …`;
    const p = s.progress;
    if (s.error) return `${g}: ⚠ ${s.error}`;
    if ((s as SubgroupsJob).queued && !s.running && !s.done) return `${g}: queued${(s as SubgroupsJob).queued_behind ? ` behind ${(s as SubgroupsJob).queued_behind}` : ""} (one alignment at a time)`;
    if (s.running) return `${g}: ${p?.phase ?? "running"} · members ${p?.members_done ?? 0}/${p?.members_total ?? s.members.length} · pairs ${p?.pairs_done ?? 0}/${p?.pairs_total ?? Math.max(0, s.members.length - 1)}${s.seconds != null ? ` · ${Math.round(s.seconds)} s` : ""}`;
    if (s.done && s.stamp?.stamped_timestamp) return `${g}: done — ${s.members.length} scans advanced to 4. Aligned${s.cached ? " (cached result)" : ""}`;
    if (s.done) return `${g}: result landed — stamping the scans…`;
    return `${g}: starting…`;
  };

  return (
    <Dialog open={open} onClose={phase === "running" ? undefined : onClose} maxWidth={false} fullWidth
      PaperProps={{ sx: { width: "94vw", maxWidth: 1440 } }} data-testid="subgroup-dialog">
      <DialogTitle sx={{ fontSize: 16 }}>Subgroups for {label}</DialogTitle>
      <DialogContent sx={{ fontSize: 13 }}>
        {err && <div className="text-xs py-1" style={{ color: "var(--c-danger, #ff5252)" }}>⚠ {err}</div>}
        {!info && !err && <div className="flex items-center gap-2 py-3 text-xs"><CircularProgress size={14} /> loading the group's scans…</div>}
        {info && phase === "ask" && (
          <>
            <div className="text-xs mb-2" style={{ color: "var(--c-text-dim)" }}>
              Do all {info.scans.length} scans of <b>{label}</b> belong to the same subgroup (replicates of the same eye that align together), or must
              different subgroups be specified? Each subgroup is aligned as its own group and its scans advance to <b>4. Aligned</b>.
            </div>
            <RadioGroup value={mode} onChange={(e) => setMode(e.target.value as "one" | "many")} data-testid="subgroup-mode">
              <FormControlLabel value="one" control={<Radio size="small" />} label={<span className="text-sm">All scans are one subgroup</span>} data-testid="subgroup-mode-one" />
              <FormControlLabel value="many" control={<Radio size="small" />} label={<span className="text-sm">Different subgroups — specify a subgroup number per scan</span>} data-testid="subgroup-mode-many" />
            </RadioGroup>
            <div className="text-[11px] mt-2 mb-1" style={{ color: "var(--c-text-dim)" }} data-testid="subgroup-thumb-caption">
              sagittal slice per scan (starts at the middle, lateral L/2 — the index is printed on each image) — compare the region imaged to decide subgroups ·
              drag the slider under a slice (or ◀ ▶; shift = 10) to scrub through THAT scan — each slider is independent · click a slice to enlarge
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 8 }} data-testid="subgroup-table">
              {info.scans.map((s) => {
                const sub = effective(s.case_id);
                const st = s.aligned_approved ? "aligned · approved" : s.group_aligned ? "aligned" : s.preproc_vetted ? "vetted" : "not vetted";
                return (
                  <div key={s.case_id} data-testid={`subgroup-row-${s.case_id}`}
                    style={{ border: "1px solid var(--c-border)", borderLeft: `4px solid ${subColor(sub)}`, borderRadius: 4, padding: 6, background: "var(--c-surface2)", display: "flex", flexDirection: "column", gap: 4 }}>
                    <div className="flex items-center gap-2" style={{ fontSize: 12 }}>
                      <span style={{ fontWeight: 600 }}>{short(s.case_id)}</span>
                      <span style={{ color: "var(--c-text-dim)", flex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{st}{s.align_subgroup_confirmed ? ` · subgroup ${s.subgroup} confirmed` : ""}</span>
                      <span title="subgroup" style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: subColor(sub) }} />
                      <input value={mode === "one" ? "1" : typed(s.case_id)} disabled={mode === "one"}
                        data-testid={`subgroup-input-${s.case_id}`} inputMode="numeric" maxLength={3}
                        placeholder="1" title="subgroup number (blank counts as 1)"
                        onChange={(e) => setAssign((m) => ({ ...m, [s.case_id]: e.target.value.replace(/[^0-9]/g, "").slice(0, 3) }))}
                        onBlur={() => setAssign((m) => ({ ...m, [s.case_id]: (m[s.case_id] ?? "").trim() || "1" }))}
                        style={{ width: 44, fontSize: 12, padding: "2px 4px", background: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-border)", borderRadius: 3, opacity: mode === "one" ? 0.6 : 1 }} />
                    </div>
                    {thumbErr[s.case_id] ? (
                      <div style={{ height: 140, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "var(--c-text-dim)", background: "#111", borderRadius: 3 }}>
                        no sagittal slice (volume missing?)
                      </div>
                    ) : (
                      <img src={thumbUrl(s.case_id, nLat(s.case_id) ? latOf(s.case_id) : undefined)}
                        alt={`sagittal slice of ${short(s.case_id)}`} data-testid={`subgroup-thumb-${s.case_id}`}
                        title="click to enlarge" onClick={() => setZoom(s.case_id)}
                        onError={() => setThumbErr((m) => ({ ...m, [s.case_id]: true }))}
                        style={{ height: 140, width: "auto", maxWidth: "100%", objectFit: "contain", alignSelf: "flex-start", background: "#000", borderRadius: 3, cursor: "zoom-in", imageRendering: "auto" }} />
                    )}
                    {/* scrub this scan's sagittal slices (reviewer 2026-09-12) */}
                    {!thumbErr[s.case_id] && nLat(s.case_id) > 1 && (
                      <div className="flex items-center gap-1" data-testid={`subgroup-scrub-${s.case_id}`}>
                        <button type="button" title="previous sagittal slice (hold shift for 10)" data-testid={`subgroup-scrub-prev-${s.case_id}`}
                          onClick={(e) => setLateral(s.case_id, latOf(s.case_id) - (e.shiftKey ? 10 : 1))}
                          style={{ fontSize: 11, lineHeight: 1, padding: "2px 5px", background: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-border)", borderRadius: 3, cursor: "pointer" }}>◀</button>
                        <input type="range" min={0} max={Math.max(0, nLat(s.case_id) - 1)} step={1} value={latOf(s.case_id)}
                          data-testid={`subgroup-scrub-slider-${s.case_id}`} aria-label={`sagittal slice of ${short(s.case_id)}`}
                          onChange={(e) => setLateral(s.case_id, Number(e.target.value))}
                          style={{ flex: 1, minWidth: 80, accentColor: subColor(sub) }} />
                        <button type="button" title="next sagittal slice (hold shift for 10)" data-testid={`subgroup-scrub-next-${s.case_id}`}
                          onClick={(e) => setLateral(s.case_id, latOf(s.case_id) + (e.shiftKey ? 10 : 1))}
                          style={{ fontSize: 11, lineHeight: 1, padding: "2px 5px", background: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-border)", borderRadius: 3, cursor: "pointer" }}>▶</button>
                        <span style={{ fontSize: 11, color: "var(--c-text-dim)", minWidth: 62, textAlign: "right" }}>
                          {latOf(s.case_id)} / {nLat(s.case_id) - 1}
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            <div className="text-[11px] mt-2" style={{ color: "var(--c-text-dim)" }} data-testid="subgroup-summary">
              → {Object.keys(groups).length} subgroup{Object.keys(groups).length === 1 ? "" : "s"}:{" "}
              {Object.entries(groups).sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true })).map(([k, m]) => (
                <span key={k} style={{ marginRight: 10 }}>
                  <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, marginRight: 3, background: subColor(k), verticalAlign: "middle" }} />
                  <b>{info.group}_s{k}</b> ({m.map(short).join(", ")})
                </span>
              ))}
              {singletons.length > 0 && (
                <span style={{ color: "var(--c-amber, #ffaa28)" }}>· a subgroup with a single scan cannot be aligned — it stays at Vetted: {singletons.map(([k]) => `s${k}`).join(", ")}</span>
              )}
            </div>
            <FormControlLabel sx={{ mt: 1 }} label={<span className="text-xs">re-run a subgroup that already has an alignment result</span>}
              control={<Checkbox size="small" checked={force} onChange={(e) => setForce(e.target.checked)} />} />
          </>
        )}
        {(phase === "running" || phase === "done") && post && (
          <div className="flex flex-col gap-1 text-xs" data-testid="subgroup-progress">
            {phase === "running" && <div className="flex items-center gap-2"><CircularProgress size={14} /> aligning each subgroup as its own group (a few minutes per pair)…</div>}
            {post.jobs.map((j) => (
              <div key={j.group} data-testid={`subgroup-job-${j.group}`} style={{ color: jobs[j.group]?.error ? "var(--c-danger, #ff5252)" : (jobSettled(jobs[j.group]) ? "var(--c-green, #4ade80)" : "inherit") }}>
                {fmtJob(j.group, jobs[j.group] ?? j)}
              </div>
            ))}
            {post.skipped.map((k) => (
              <div key={k.group} style={{ color: "var(--c-amber, #ffaa28)" }}>{k.group}: skipped — {k.reason}</div>
            ))}
            {post.rejected.length > 0 && <div style={{ color: "var(--c-amber, #ffaa28)" }}>not written (not this eye's scans): {post.rejected.join(", ")}</div>}
            {phase === "done" && (
              <div style={{ marginTop: 4 }} data-testid="subgroup-done">
                {post.jobs.length === 0 ? "Nothing to align." : post.jobs.every((j) => jobSettled(jobs[j.group]) && !jobs[j.group]?.error)
                  ? "Done — the aligned scans are now at 4. Aligned; review the consensus, 3-D view and scrub, then approve the axial changes."
                  : "Finished with errors — see above."}
              </div>
            )}
          </div>
        )}
      </DialogContent>
      <DialogActions>
        {phase === "ask" && (
          <>
            <Button size="small" onClick={onClose} disabled={posting}>Cancel</Button>
            <Button size="small" variant="contained" color="primary" disabled={!info || posting} onClick={() => void confirm()} data-testid="subgroup-confirm"
              startIcon={posting ? <CircularProgress size={13} color="inherit" /> : undefined}
              title="Write the subgroups, then align each subgroup with ≥ 2 scans as its own group.">
              {posting ? "Starting…" : "Confirm and align"}
            </Button>
          </>
        )}
        {phase === "running" && <Button size="small" onClick={() => { stopPoll(); onClose(); }} title="The alignment keeps running; the scans advance when it finishes.">Hide (keeps running)</Button>}
        {phase === "done" && <Button size="small" variant="contained" onClick={onClose} data-testid="subgroup-close">Close</Button>}
      </DialogActions>
      <Dialog open={zoom != null} onClose={() => setZoom(null)} maxWidth={false} PaperProps={{ sx: { background: "#000", maxWidth: "96vw" } }} data-testid="subgroup-zoom">
        {zoom && (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4, padding: 6 }}>
            <img src={thumbUrl(zoom, nLat(zoom) ? latOf(zoom) : undefined)} alt={`sagittal slice of ${short(zoom)} (enlarged)`} data-testid="subgroup-zoom-img"
              onClick={() => setZoom(null)} style={{ width: "min(92vw, 1400px)", height: "auto", imageRendering: "auto", cursor: "zoom-out" }} />
            <div className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>{short(zoom)} — middle sagittal slice · click to close</div>
          </div>
        )}
      </Dialog>
    </Dialog>
  );
}
