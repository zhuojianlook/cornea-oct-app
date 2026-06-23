/* OCT preprocessing loader (Optovue Avanti .OCT → corrected volume).
   Flow: upload .OCT files OR load a folder → scans auto-group by (patient, eye) →
   for each group tag Scar / Control (the whole group), scrub each replicate scan to set
   its scar frame-range → tune correction params → Preprocess the selected scans
   (OCT→correct, correct Avanti geometry). Grouping is editable: rename a group, or move a
   scan to another / a new group. (SAM2 + consensus runs per group, downstream of this.) */

import { useEffect, useRef, useState } from "react";
import { Button, Typography, TextField, LinearProgress, Slider, Checkbox, ToggleButton, ToggleButtonGroup, Collapse } from "@mui/material";
import { api, resourceUrl } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import type { ConsensusReport } from "../../api/types";

type Status = "queued" | "uploading" | "ready" | "preprocessing" | "done" | "error";
type Cls = "scar" | "control";

// A group is one (patient, eye) — a set of replicate scans that share a Scar/Control tag.
// origPatient/origEye record the auto-parsed identity so we can tell when the user has
// edited the header (and should persist the correction to the backend).
interface OctGroup {
  id: string;
  patient: string;
  eye: string;
  condition?: Cls;
  origPatient: string;
  origEye: string;
}
interface OctScan {
  id: string;
  groupId: string;
  filename: string;
  caseId?: string;
  nVolumes?: number;
  nFrames: number;
  status: Status;
  scarRange: [number, number]; // per-scan (only used when the scan's group is tagged "scar")
  // Replicate set WITHIN the eye group. One eye can hold scans of DIFFERENT scars (e.g. 3
  // posterior + 2 inferior) — those are NOT replicates of each other, so SAM2 + consensus run
  // PER subgroup, not per eye. Assigned after preprocessing; "1" = the default single subgroup.
  subgroup: string;
  selected: boolean;
  error?: string;
}

// Sanitise a subgroup label for the consensus case-id segment (lowercase alnum, "-"/"_" kept).
const subSlug = (s: string): string => (s || "1").trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "1";

// Loaded-case shape shared by /api/oct/upload and /api/oct/load-dir.
interface LoadedCase {
  case_id?: string;
  filename: string;
  patient?: string;
  eye?: string;
  n_volumes?: number;
  preprocessed?: boolean;
  error?: string;
}

const DOT: Record<Status, string> = {
  queued: "var(--c-text-dim)", uploading: "var(--c-accent)", ready: "var(--c-accent)",
  preprocessing: "var(--c-accent)", done: "var(--c-green)", error: "var(--c-red)",
};

// Smoother params (oct_preprocess.DEFAULT_PARAMS) exposed as sliders.
interface Param { key: string; label: string; min: number; max: number; step: number; def: number; }
const PARAMS: Param[] = [
  { key: "sigma", label: "Gaussian σ", min: 0.5, max: 5, step: 0.1, def: 2.0 },
  { key: "max_jump", label: "Max jump", min: 1, max: 50, step: 1, def: 10 },
  { key: "median_filter_size", label: "Median size", min: 3, max: 15, step: 2, def: 5 },
  { key: "d", label: "Bilateral d", min: 3, max: 15, step: 1, def: 9 },
  { key: "sigmaColor", label: "σ color", min: 10, max: 150, step: 5, def: 75 },
  { key: "sigmaSpace", label: "σ space", min: 10, max: 150, step: 5, def: 75 },
  { key: "side_window", label: "Side window", min: 5, max: 30, step: 1, def: 10 },
  { key: "side_threshold_factor", label: "Side thresh", min: 1, max: 5, step: 0.1, def: 2.0 },
  { key: "residual_threshold", label: "RANSAC resid", min: 1, max: 10, step: 0.5, def: 5.0 },
  { key: "active_threshold", label: "3D active thresh", min: 1, max: 20, step: 1, def: 5 },
  { key: "corr_factor", label: "Correction ×", min: 0, max: 1, step: 0.05, def: 1.0 },
];
const defaultParams = (): Record<string, number> => Object.fromEntries(PARAMS.map((p) => [p.key, p.def]));

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

// Monotonic client-side id for groups/scans (stable React keys + move targets).
let _seq = 0;
const uid = (p: string) => `${p}${++_seq}`;

// Download preprocessed scans straight from the sidecar for manual ground-truth segmentation.
// The endpoint sets Content-Disposition: attachment, so the browser saves the file (the anchor's
// `download` is just a same-origin hint). A downloaded set unzips to a folder of <case_id>.nii.gz
// that the companion annotator app opens directly.
const triggerDownload = (href: string, name: string) => {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
};
// Fetch the bytes then save via a blob URL — this GUARANTEES the saved filename matches the source
// scan (a cross-origin <a download> would otherwise be ignored). Falls back to a direct link (the
// endpoint also sets Content-Disposition to the same name) if the fetch fails.
async function downloadPreprocessed(cid: string, octName: string) {
  const name = `${octName.replace(/\.oct$/i, "")}.nii.gz`;
  const path = `/api/case/${encodeURIComponent(cid)}/preprocessed.nii.gz`;
  try {
    const res = await fetch(resourceUrl(path));
    if (!res.ok) throw new Error(String(res.status));
    const url = URL.createObjectURL(await res.blob());
    triggerDownload(url, name);
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch {
    triggerDownload(resourceUrl(path), name);
  }
}
const downloadPreprocessedZip = (cids: string[]) => {
  if (!cids.length) return;
  triggerDownload(
    resourceUrl(`/api/preprocessed-zip?cases=${cids.map(encodeURIComponent).join(",")}`),
    "preprocessed_scans.zip",
  );
};

export function OctLoader() {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const filesRef = useRef<File[]>([]);
  const [scans, setScans] = useState<OctScan[]>([]);
  const [groups, setGroups] = useState<OctGroup[]>([]);
  const [dirPath, setDirPath] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [params, setParams] = useState<Record<string, number>>(defaultParams());
  const [paramsOpen, setParamsOpen] = useState(false);
  const [report, setReport] = useState<ConsensusReport | null>(null);
  const [reportLabel, setReportLabel] = useState("");
  const [casesCount, setCasesCount] = useState<number | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const toggleCollapse = (id: string) =>
    setCollapsed((cur) => { const n = new Set(cur); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const setCaseId = useCaseStore((s) => s.setCaseId);
  const openCase = useCaseStore((s) => s.openCase);
  const setStage = useWorkflowStore((s) => s.setStage);
  const initTabs = useWorkflowStore((s) => s.initTabs);

  // Background pre-warm bookkeeping (so clicking a scan to scrub is instant).
  const busyRef = useRef(false);
  busyRef.current = busy;
  const scansRef = useRef<OctScan[]>([]);
  scansRef.current = scans;
  const warmedRef = useRef<Set<string>>(new Set());

  // Pre-warm each scan's raw scrub previews in the BACKGROUND while idle, so clicking a scan to
  // scrub is instantaneous (no per-click .OCT decode + render wait). Pauses during any
  // foreground action (preview / preprocess) so it never competes with what the user is doing.
  useEffect(() => {
    if (!loaded) return;
    let stop = false;
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
    (async () => {
      for (const s of scansRef.current) {
        if (stop) return;
        const cid = s.caseId;
        if (!cid || s.status === "error" || warmedRef.current.has(cid)) continue;
        while (busyRef.current && !stop) await sleep(250); // yield to the user
        if (stop) return;
        warmedRef.current.add(cid);
        try { await api.json(`/api/case/${cid}/oct-volume`, "POST", JSON.stringify({})); } catch { /* best-effort */ }
      }
    })();
    return () => { stop = true; };
  }, [loaded]);

  // Drop groups that no longer hold any scan (e.g. after moving the last scan out).
  useEffect(() => {
    setGroups((cur) => {
      const nonEmpty = cur.filter((g) => scans.some((s) => s.groupId === g.id));
      return nonEmpty.length === cur.length ? cur : nonEmpty;
    });
  }, [scans]);

  // How many cases are already persisted on disk (re-uploads reuse these folders + their old
  // segmentation; the Wipe button clears them for a clean slate).
  const refreshCount = async () => {
    try { setCasesCount((await api.json<{ count: number }>("/api/cases/stat")).count); }
    catch { /* leave the previous count */ }
  };
  useEffect(() => { refreshCount(); }, []);

  const patchScan = (id: string, p: Partial<OctScan>) =>
    setScans((cur) => cur.map((s) => (s.id === id ? { ...s, ...p } : s)));
  const updateGroup = (id: string, p: Partial<OctGroup>) =>
    setGroups((cur) => cur.map((g) => (g.id === id ? { ...g, ...p } : g)));

  // Tagging a group Scar/Control invalidates any already-corrected scans in it (the scar
  // range is baked into preprocessing, so the corrected result is now stale).
  const setGroupCondition = (id: string, condition?: Cls) => {
    updateGroup(id, { condition });
    setScans((cur) => cur.map((s) => (s.groupId === id && s.status === "done" ? { ...s, status: "ready" } : s)));
  };
  // A scan that changes group may now have a different Scar/Control tag than the one baked
  // into its corrected volume — re-mark "done" scans "ready" so they re-run under the new group.
  const reready = (s: OctScan): OctScan => (s.status === "done" ? { ...s, status: "ready" } : s);
  const moveScan = (scanId: string, targetGroupId: string) =>
    setScans((cur) => cur.map((s) => (s.id === scanId ? reready({ ...s, groupId: targetGroupId }) : s)));
  const moveScanToNewGroup = (scanId: string) => {
    // Inherit the source group's patient + tag (same patient; user re-specifies the eye).
    const src = groups.find((g) => g.id === scans.find((s) => s.id === scanId)?.groupId);
    const g: OctGroup = { id: uid("g"), patient: src?.patient ?? "New group", eye: "?", condition: src?.condition, origPatient: src?.patient ?? "New group", origEye: "?" };
    setGroups((cur) => [...cur, g]);
    setScans((cur) => cur.map((s) => (s.id === scanId ? reready({ ...s, groupId: g.id }) : s)));
  };
  // Merge a whole group into another (one click vs moving every scan); the emptied source
  // group is then auto-pruned by the effect above.
  const mergeGroup = (srcId: string, dstId: string) =>
    setScans((cur) => cur.map((s) => (s.groupId === srcId ? reready({ ...s, groupId: dstId }) : s)));
  // True once the user has edited the auto-parsed patient/eye (so we persist the correction).
  const isEdited = (g: OctGroup) =>
    g.patient.trim() !== g.origPatient.trim() || g.eye.trim().toUpperCase() !== g.origEye.trim().toUpperCase();

  // Build groups + scans from a set of loaded cases: auto-group by (patient, eye).
  const ingest = (cases: LoadedCase[]) => {
    warmedRef.current.clear(); // a fresh load → re-warm previews (old on-disk renders may be gone)
    const byKey = new Map<string, OctGroup>();
    const order: OctGroup[] = [];
    const newScans: OctScan[] = [];
    for (const c of cases) {
      const patient = (c.patient || "").trim() || c.filename.replace(/\.oct$/i, "");
      const eye = ((c.eye || "").trim() || "?").toUpperCase();
      // Group replicates by (patient, eye) — but NEVER collapse unknown ("?") eyes together:
      // two different eyes with unparsed laterality must not be voted as one. Each "?" scan
      // gets its own group; the user merges true replicates manually if needed.
      const key = eye === "?" ? `?|||${c.case_id ?? c.filename}` : `${patient.toLowerCase()}|||${eye}`;
      let g = byKey.get(key);
      if (!g) { g = { id: uid("g"), patient, eye, origPatient: patient, origEye: eye }; byKey.set(key, g); order.push(g); }
      newScans.push({
        id: uid("s"), groupId: g.id, filename: c.filename, caseId: c.case_id, nVolumes: c.n_volumes,
        // A scan already corrected in a prior session loads as "done" so it's coloured + skipped on re-run.
        nFrames: 101, status: c.error ? "error" : c.preprocessed ? "done" : "ready",
        error: c.error, scarRange: [1, 101], subgroup: "1", selected: !c.error,
      });
    }
    setGroups(order);
    setScans(newScans);
    return { nGroups: order.length, firstCase: cases.find((c) => !c.error)?.case_id };
  };

  // Keep only .OCT (+ companion .txt) from a file/dir selection.
  const pickFiles = (fs: File[]) => fs.filter((f) => /\.(oct|txt)$/i.test(f.name));

  const onPicked = (fs: File[]) => {
    const keep = pickFiles(fs);
    const octs = keep.filter((f) => /\.oct$/i.test(f.name));
    // An .OCT can't be read without its companion .txt (POCT filespec); warn up front.
    const txtStems = new Set(keep.filter((f) => /\.txt$/i.test(f.name)).map((f) => f.name.replace(/\.txt$/i, "").toLowerCase()));
    const missing = octs.filter((o) => !txtStems.has(o.name.replace(/\.oct$/i, "").toLowerCase()));
    // Placeholders (not rendered until uploaded) — just to drive the "Upload N files" button.
    setScans(octs.map((f) => ({ id: uid("s"), groupId: "_pending", filename: f.name, nFrames: 101, status: "queued", scarRange: [1, 101], subgroup: "1", selected: true })));
    setGroups([]);
    setLoaded(false);
    setReport(null);
    setActiveId(null);
    filesRef.current = keep;
    setStep(missing.length
      ? `⚠ ${missing.length}/${octs.length} .OCT are missing their .txt companion — also pick the .txt files, or use a folder. An .OCT can't be read without it.`
      : `${octs.length} .OCT + companion .txt selected.`);
  };

  const load = async () => {
    const files = filesRef.current;
    if (!files.length) return;
    setBusy(true);
    setReport(null);
    setScans((cur) => cur.map((s) => ({ ...s, status: "uploading" })));
    try {
      setStep(`Uploading ${files.length} file(s)…`);
      const up = await api.upload<{ cases: LoadedCase[] }>("/api/oct/upload", files);
      const { nGroups, firstCase } = ingest(up.cases);
      setLoaded(true);
      setStage(1);
      void refreshCount();
      setStep(`Uploaded ${up.cases.length} scan(s) into ${nGroups} group(s). Tag each group Scar/Control, scrub the scans, then preprocess.`);
      if (firstCase) await preview(firstCase);
    } catch (e) {
      setStep(`Load failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Open a native folder picker on this machine and load whatever the user selects.
  // One click, no typing, no upload — the picked folder is loaded in place below.
  const browseDir = async () => {
    setBusy(true);
    try {
      setStep("Opening folder picker…");
      const r = await api.json<{ directory: string | null }>("/api/oct/pick-dir", "POST", JSON.stringify({}));
      if (!r.directory) {
        setStep("No folder selected.");
        setBusy(false);
        return;
      }
      setDirPath(r.directory);
      setBusy(false);
      await loadDir(r.directory);
    } catch (e) {
      setStep(`Folder picker failed: ${msg(e)}`);
      setBusy(false);
    }
  };

  // Load every .OCT from a SERVER-SIDE folder (no browser upload, companions auto-paired).
  // Best for local data — the .OCT stay in place on disk. The folder comes from the native
  // picker (browseDir) or, as a fallback, a path typed into the field.
  const loadDir = async (dir?: string) => {
    const directory = (dir ?? dirPath).trim();
    if (!directory) return;
    setBusy(true);
    setReport(null);
    try {
      setStep("Scanning folder…");
      const up = await api.json<{ cases: (LoadedCase & { has_companion: boolean })[] }>(
        "/api/oct/load-dir", "POST", JSON.stringify({ directory }),
      );
      const { nGroups, firstCase } = ingest(up.cases);
      setLoaded(true);
      setStage(1);
      void refreshCount();
      const noTxt = up.cases.filter((c) => !c.has_companion).length;
      setStep(`Loaded ${up.cases.length} scan(s) into ${nGroups} group(s)${noTxt ? ` (⚠ ${noTxt} missing a .txt companion)` : ""}. Tag each group, scrub, then preprocess.`);
      if (firstCase) await preview(firstCase);
    } catch (e) {
      setStep(`Load folder failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // DESTRUCTIVE: delete every persisted case on disk so re-uploads start fresh instead of
  // reusing the deterministic case folder + its old segmentation. Guarded by a confirm.
  const wipeAll = async () => {
    const n = casesCount ?? 0;
    if (!window.confirm(
      `Delete ALL ${n || ""} saved case(s)?\n\nThis removes every corrected volume, segmentation, ` +
      `label and preview on disk — it cannot be undone. Re-uploads will then start fresh.`,
    )) return;
    setBusy(true);
    try {
      setStep("Wiping all saved cases…");
      const r = await api.json<{ removed: number; freed_bytes: number }>("/api/cases/wipe", "POST", JSON.stringify({}));
      // The just-loaded scans' case folders are gone — reset the loader to a clean state.
      setScans([]); setGroups([]); setLoaded(false); setReport(null); setReportLabel(""); setActiveId(null);
      setCasesCount(0);
      setStep(`Wiped ${r.removed} case(s), freed ${(r.freed_bytes / 1e9).toFixed(2)} GB. Upload to start fresh.`);
    } catch (e) {
      setStep(`Wipe failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Scrub a scan: materialise its raw z-stack + show grayscale in the viewer.
  const preview = async (caseId?: string) => {
    if (!caseId || busy) return;
    setBusy(true);
    setActiveId(caseId);
    try {
      setCaseId(caseId);
      const r = await api.json<{ n_frames?: number; preprocessed?: boolean }>(
        `/api/case/${caseId}/oct-volume`, "POST", JSON.stringify({}),
      );
      // Use the scan's REAL frame count for the scar-range slider (not a hardcoded 101).
      const nf = r.n_frames && r.n_frames > 1 ? r.n_frames : 101;
      setScans((cur) => cur.map((s) => (s.caseId === caseId
        ? { ...s, nFrames: nf,
            // Reflect the backend's corrected state so a previously-preprocessed scan colours as done.
            status: r.preprocessed && s.status !== "error" ? "done" : s.status,
            scarRange: [Math.min(s.scarRange[0], nf), Math.min(s.scarRange[1], nf)] }
        : s)));
      await openCase();
      initTabs(false); // grayscale routing + refetch
      setStep(r.preprocessed
        ? "Showing the corrected volume. Run SAM2 + consensus when ready."
        : "Scrub the B-scans in the viewer. Tag the group Scar/Control (+ scar range), then preprocess.");
    } catch (e) {
      setStep(`Preview failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Changing params invalidates any already-corrected scans (result is now stale).
  const setParam = (key: string, v: number) => {
    setParams((cur) => ({ ...cur, [key]: v }));
    setScans((cur) => cur.map((s) => (s.status === "done" ? { ...s, status: "ready" } : s)));
  };

  const runPreprocess = async () => {
    // Skip already-corrected (non-stale) scans — re-clicking is a no-op for them.
    const sel = scans.filter((s) => s.selected && s.caseId && s.status !== "error" && s.status !== "done");
    if (!sel.length) {
      setStep("Nothing to preprocess (selected scans are already corrected — change params or tags to re-run).");
      return;
    }
    setBusy(true);
    setReport(null);
    let failed = 0;
    try {
      for (let k = 0; k < sel.length; k++) {
        const s = sel[k];
        const g = groups.find((gg) => gg.id === s.groupId);
        const cls = g?.condition ?? null;
        const edited = g && isEdited(g);
        setStep(`Preprocessing ${k + 1}/${sel.length} — ${s.filename} (OCT→correct, up to ~20 min if the backend is busy)`);
        patchScan(s.id, { status: "preprocessing" });
        try {
          await api.json(`/api/case/${s.caseId}/oct-preprocess`, "POST", JSON.stringify({
            params,
            classification: cls,
            scar_range: cls === "scar" ? s.scarRange : null,
            // Persist a user-corrected identity so the rename reaches consensus/export.
            ...(edited ? { patient: g!.patient.trim(), eye: g!.eye.trim() } : {}),
          }));
          patchScan(s.id, { status: "done" });
          // If the corrected scan is the one on screen, refresh the viewer to show it.
          if (s.caseId === activeId) initTabs(false);
        } catch (e) {
          patchScan(s.id, { status: "error", error: msg(e) });
          failed++;
        }
      }
      // Don't claim success when some scans failed — surface the partial result.
      setStep(failed
        ? `Preprocessing finished: ${sel.length - failed}/${sel.length} OK, ${failed} failed (see the red rows).`
        : "Preprocessing complete. Now Run SAM2 + consensus on the corrected scans.");
    } catch (e) {
      setStep(`Preprocessing failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Segment each preprocessed scan, then build one consensus PER (eye × subgroup). Scans of the
  // same patient/eye but DIFFERENT scars (subgroups) are NOT replicates, so they're voted
  // separately — never across patients, eyes, OR subgroups.
  const runSamConsensus = async () => {
    const work = groups
      .flatMap((g) => {
        const done = scans.filter((s) => s.groupId === g.id && s.selected && s.caseId && s.status === "done");
        const bySub = new Map<string, string[]>();
        for (const s of done) {
          const key = subSlug(s.subgroup);
          const arr = bySub.get(key) ?? [];
          arr.push(s.caseId!);
          bySub.set(key, arr);
        }
        const multiSub = new Set(scans.filter((s) => s.groupId === g.id).map((s) => subSlug(s.subgroup))).size > 1;
        return [...bySub.entries()].map(([sub, cids]) => ({ g, sub, cids, multiSub }));
      })
      .filter((x) => x.cids.length > 0);
    if (!work.length) {
      setStep("Preprocess at least one scan first.");
      return;
    }
    setBusy(true);
    setReport(null);
    try {
      let lastResult: string | null = null;
      const summary: string[] = [];
      for (const { g, sub, cids, multiSub } of work) {
        const label = `${g.patient} ${g.eye}`.trim() + (multiSub ? ` · ${sub}` : "");
        const segmented: string[] = [];
        for (const cid of cids) {
          setStep(`Segmenting ${label} — ${cid} (SAM2 + scar, ~2–3 min)…`);
          try {
            await api.json(`/api/case/${cid}/consensus-segment`, "POST", JSON.stringify({}));
            segmented.push(cid);
          } catch (e) {
            setScans((cur) => cur.map((s) => (s.caseId === cid ? { ...s, status: "error", error: msg(e) } : s)));
          }
        }
        if (!segmented.length) { summary.push(`${label}: all failed`); continue; }
        if (segmented.length > 1) {
          setStep(`Building consensus for ${label} (${segmented.length} scans)…`);
          const res = await api.json<{ consensus_case: string; report: ConsensusReport }>(
            "/api/consensus/build", "POST", JSON.stringify({ cases: segmented, subgroup: sub }),
          );
          lastResult = res.consensus_case;
          setReport(res.report);
          setReportLabel(label);
          summary.push(`${label}: consensus/${segmented.length}`);
        } else {
          lastResult = segmented[0];
          summary.push(`${label}: 1 scan`);
        }
      }
      if (lastResult) {
        setStep("Opening result…");
        setCaseId(lastResult);
        await openCase();
        setStage(2);
      }
      setStep(`Done — ${summary.join(" · ")}`);
    } catch (e) {
      setStep(`SAM2 / consensus failed: ${msg(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const nDone = scans.filter((s) => s.status === "done" && s.selected).length;
  const nToRun = scans.filter((s) => s.selected && s.caseId && s.status !== "error" && s.status !== "done").length;
  // Tagging is the point of this panel: don't let a scan in an untagged group preprocess
  // (it would be committed with no Scar/Control label, silently producing unlabeled data).
  const runnableInGroup = (gid: string) =>
    scans.some((s) => s.groupId === gid && s.selected && s.caseId && s.status !== "error" && s.status !== "done");
  const untaggedGroups = groups.filter((g) => !g.condition && runnableInGroup(g.id));

  return (
    <div className="flex flex-col gap-2">
      <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
        Load Optovue .OCT scans — each needs its <b>.txt</b> next to it (a folder grabs both).
        Scans auto-group by patient/eye; tag each group Scar/Control, scrub the replicate scans, then preprocess.
      </Typography>
      <input ref={fileRef} type="file" accept=".oct,.txt" multiple hidden
        onChange={(e) => onPicked(Array.from(e.target.files ?? []))} />

      {/* Primary: native folder picker on this machine — one click, no typing, no upload. */}
      <Button variant="contained" size="small" fullWidth onClick={browseDir} disabled={busy}>
        Select folder…
      </Button>

      {/* Fallbacks: type a path (e.g. headless/remote host), or pick individual files. */}
      <div className="flex gap-2 items-end">
        <TextField size="small" label="…or type a folder path" value={dirPath}
          onChange={(e) => setDirPath(e.target.value)} placeholder="/home/…/OCT scans" disabled={busy} fullWidth />
        <Button variant="outlined" size="small" onClick={() => loadDir()} disabled={busy || !dirPath.trim()} sx={{ flex: "none" }}>
          Load
        </Button>
      </div>
      <Button variant="text" size="small" onClick={() => fileRef.current?.click()} disabled={busy} sx={{ alignSelf: "flex-start", textTransform: "none", minWidth: 0, p: 0.25 }}>
        or pick individual files…
      </Button>

      {/* Destructive reset: clear the on-disk case store so re-uploads don't reuse old output. */}
      {!!casesCount && (
        <Button variant="text" size="small" color="error" onClick={wipeAll} disabled={busy}
          sx={{ alignSelf: "flex-start", textTransform: "none", minWidth: 0, p: 0.25, fontSize: 11 }}>
          🗑 Wipe all saved cases ({casesCount})
        </Button>
      )}

      {scans.length > 0 && !loaded && (
        <Button variant="contained" size="small" onClick={load} disabled={busy || scans.length < 1}>
          Upload {scans.length} file{scans.length === 1 ? "" : "s"}
        </Button>
      )}

      {loaded && groups.length > 0 && (
        <div className="flex flex-col gap-2">
          <div className="text-[11px] uppercase tracking-wide flex items-center justify-between" style={{ color: "var(--c-text-dim)" }}>
            <span>Groups — one per patient/eye · tag · scrub · preprocess</span>
            <button style={{ background: "none", border: "none", color: "var(--c-text-dim)", cursor: "pointer", textTransform: "none", fontSize: 10, padding: 0 }}
              onClick={() => setCollapsed((cur) => (cur.size >= groups.length ? new Set() : new Set(groups.map((g) => g.id))))}>
              {collapsed.size >= groups.length ? "expand all" : "collapse all"}
            </button>
          </div>
          {groups.map((g) => {
            const groupScans = scans.filter((s) => s.groupId === g.id);
            const needsTag = !g.condition;
            const isCollapsed = collapsed.has(g.id);
            const nDoneInGroup = groupScans.filter((s) => s.status === "done").length;
            return (
              <div key={g.id} className="rounded" style={{ border: needsTag ? "1px solid var(--c-amber, #d9a441)" : "1px solid var(--c-border)" }}>
                {/* Group header: editable patient/eye + Scar/Control tag for the whole group. */}
                <div className="flex flex-col gap-1 px-1.5 py-1.5" style={{ background: "var(--c-surface2)" }}>
                  <div className="flex items-center gap-1">
                    <button onClick={() => toggleCollapse(g.id)} title={isCollapsed ? "expand" : "collapse"}
                      style={{ background: "none", border: "none", color: "var(--c-text-dim)", cursor: "pointer", padding: 0, width: 16, flex: "none", fontSize: 11 }}>
                      {isCollapsed ? "▸" : "▾"}
                    </button>
                    <TextField variant="standard" value={g.patient} disabled={busy} placeholder="patient"
                      onChange={(e) => updateGroup(g.id, { patient: e.target.value })}
                      sx={{ flex: 1 }} InputProps={{ sx: { fontSize: 12, fontWeight: 600 } }} />
                    <TextField variant="standard" value={g.eye} disabled={busy} placeholder="eye"
                      onChange={(e) => updateGroup(g.id, { eye: e.target.value })}
                      sx={{ width: 48 }} InputProps={{ sx: { fontSize: 12 } }} />
                  </div>
                  {g.eye === "?" && (
                    <span className="text-[10px]" style={{ color: "var(--c-amber, #d9a441)" }}>
                      eye unknown — set OD/OS (and merge replicates of the same eye)
                    </span>
                  )}
                  <div className="flex items-center gap-2">
                    <ToggleButtonGroup size="small" exclusive value={g.condition ?? null}
                      onChange={(_, v) => setGroupCondition(g.id, (v as Cls) || undefined)}>
                      <ToggleButton value="scar" sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Scar</ToggleButton>
                      <ToggleButton value="control" sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Control (no scar)</ToggleButton>
                    </ToggleButtonGroup>
                    <span className="text-[10px]" style={{ marginLeft: "auto", color: nDoneInGroup ? "var(--c-green)" : "var(--c-text-dim)" }}>
                      {groupScans.length} scan{groupScans.length === 1 ? "" : "s"}{nDoneInGroup ? ` · ${nDoneInGroup} done ✓` : ""}
                    </span>
                    {nDoneInGroup > 0 && (
                      <button title={`Download all ${nDoneInGroup} preprocessed scan(s) in this group as a .zip (a folder for manual segmentation)`}
                        onClick={() => downloadPreprocessedZip(groupScans.filter((s) => s.status === "done" && s.caseId).map((s) => s.caseId!))}
                        style={{ background: "none", border: "none", color: "var(--c-accent)", cursor: "pointer", padding: 0, fontSize: 10, flex: "none" }}>
                        ⬇ zip
                      </button>
                    )}
                  </div>
                  {(() => {
                    const subs = groupScans.reduce<Record<string, number>>((m, s) => { const k = subSlug(s.subgroup); m[k] = (m[k] || 0) + 1; return m; }, {});
                    const keys = Object.keys(subs);
                    return keys.length > 1 ? (
                      <span className="text-[10px]" style={{ color: "var(--c-accent)" }}>
                        {keys.length} subgroups: {keys.map((k) => `${k}×${subs[k]}`).join(", ")} — consensus runs per subgroup
                      </span>
                    ) : null;
                  })()}
                  {needsTag && (
                    <span className="text-[10px]" style={{ color: "var(--c-amber, #d9a441)" }}>⚠ tag this group Scar or Control</span>
                  )}
                  {groups.length > 1 && (
                    <select value="" disabled={busy}
                      onChange={(e) => { if (e.target.value) mergeGroup(g.id, e.target.value); }}
                      style={{ fontSize: 10, color: "var(--c-text-dim)", background: "var(--c-surface)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "1px 4px", alignSelf: "flex-start" }}>
                      <option value="">Merge into…</option>
                      {groups.filter((o) => o.id !== g.id).map((o) => (
                        <option key={o.id} value={o.id}>{o.patient} {o.eye}</option>
                      ))}
                    </select>
                  )}
                </div>

                {/* Replicate scans in the group (hidden when the group is collapsed). */}
                {!isCollapsed && (
                <div className="flex flex-col gap-1 px-1.5 py-1">
                  {groupScans.map((s) => {
                    const active = !!s.caseId && s.caseId === activeId;
                    const clickable = !busy && !!s.caseId && s.status !== "error";
                    const done = s.status === "done";
                    // The actively-viewed scan gets a clear ACCENT tint (not grey) so the selected
                    // scan stands out; a corrected (preprocessed) scan is green for before/after review.
                    const rowBg = active ? "rgba(90,127,168,0.32)" : done ? "rgba(63,185,80,0.12)" : "transparent";
                    const rowBorder = active ? "var(--c-accent)" : done ? "var(--c-green)" : "transparent";
                    return (
                      <div key={s.id} className="rounded px-1 py-0.5" style={{ background: rowBg, borderLeft: `2px solid ${rowBorder}` }}>
                        <div className="flex items-start gap-1.5 text-xs">
                          <Checkbox size="small" checked={s.selected} disabled={busy || s.status === "error"} sx={{ p: 0.25 }}
                            onChange={(e) => patchScan(s.id, { selected: e.target.checked })} />
                          <span style={{ width: 8, height: 8, borderRadius: "50%", background: DOT[s.status], flex: "none", marginTop: 5 }} />
                          {/* Full name (wrap, don't truncate) — Optovue .OCT names are long & spaceless. */}
                          <span style={{ flex: 1, minWidth: 0, overflowWrap: "anywhere", lineHeight: 1.35, color: done ? "var(--c-green)" : undefined, cursor: clickable ? "pointer" : "default" }}
                            title={s.error || s.filename} onClick={() => clickable && preview(s.caseId)}>
                            {s.filename.replace(/\.OCT$/i, "")}
                          </span>
                          <span style={{ color: s.status === "error" ? "var(--c-red)" : done ? "var(--c-green)" : "var(--c-text-dim)" }}>
                            {s.status === "error" ? "failed" : done ? "corrected ✓" : s.status}
                          </span>
                          {done && s.caseId && (
                            <button title="Download this preprocessed scan (.nii.gz) for manual segmentation"
                              onClick={(e) => { e.stopPropagation(); void downloadPreprocessed(s.caseId!, s.filename); }}
                              style={{ background: "none", border: "none", color: "var(--c-accent)", cursor: "pointer", padding: 0, fontSize: 13, lineHeight: 1, flex: "none", marginTop: 2 }}>
                              ⬇
                            </button>
                          )}
                        </div>

                        {/* Move this scan to another / a new group. */}
                        {s.status !== "error" && (
                          <div className="ml-6 mt-0.5">
                            <select value="" disabled={busy}
                              onChange={(e) => {
                                const v = e.target.value;
                                if (v === "__new") moveScanToNewGroup(s.id);
                                else if (v) moveScan(s.id, v);
                              }}
                              style={{ fontSize: 10, color: "var(--c-text-dim)", background: "var(--c-surface2)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "1px 4px" }}>
                              <option value="">Move to…</option>
                              {groups.filter((og) => og.id !== g.id).map((og) => (
                                <option key={og.id} value={og.id}>{og.patient} {og.eye}</option>
                              ))}
                              <option value="__new">＋ New group</option>
                            </select>
                          </div>
                        )}

                        {/* Subgroup: a replicate SET within this eye (e.g. posterior vs inferior).
                            Same eye, different scar → not replicates, so consensus runs per subgroup.
                            Type a name (autocompletes from this group's existing subgroups); "1" = default. */}
                        {s.status !== "error" && (
                          <div className="ml-6 mt-0.5 flex items-center gap-1">
                            <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>subgroup</span>
                            <input list={`subs-${g.id}`} value={s.subgroup} disabled={busy} placeholder="1"
                              onChange={(e) => patchScan(s.id, { subgroup: e.target.value })}
                              style={{ fontSize: 10, width: 92, color: "var(--c-text)", background: "var(--c-surface2)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "1px 4px" }} />
                            <datalist id={`subs-${g.id}`}>
                              {[...new Set(scans.filter((x) => x.groupId === g.id).map((x) => x.subgroup).filter(Boolean))].map((sv) => (
                                <option key={sv} value={sv} />
                              ))}
                            </datalist>
                          </div>
                        )}

                        {/* Scar frame-range (per scan) — only when the group is tagged Scar. */}
                        {g.condition === "scar" && s.status !== "error" && (
                          <div className="ml-6 mt-1 pr-2">
                            <div className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>scar frames {s.scarRange[0]}–{s.scarRange[1]}</div>
                            <Slider size="small" min={1} max={s.nFrames} value={s.scarRange} disabled={busy}
                              onChange={(_, v) => patchScan(s.id, { scarRange: v as [number, number], status: s.status === "done" ? "ready" : s.status })} />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {loaded && (
        <>
          <button className="text-[11px] uppercase tracking-wide text-left" style={{ color: "var(--c-text-dim)", cursor: "pointer", background: "none", border: "none", padding: 0 }}
            onClick={() => setParamsOpen((o) => !o)}>
            {paramsOpen ? "▾" : "▸"} Correction parameters
            <span style={{ marginLeft: 6, textTransform: "none" }} onClick={(e) => { e.stopPropagation(); setParams(defaultParams()); }}>· reset</span>
          </button>
          <Collapse in={paramsOpen}>
            <div className="flex flex-col gap-1 px-1">
              {PARAMS.map((p) => (
                <div key={p.key} className="flex items-center gap-2">
                  <span className="text-[10px]" style={{ width: 88, color: "var(--c-text-dim)" }}>{p.label}</span>
                  <Slider size="small" min={p.min} max={p.max} step={p.step} value={params[p.key]}
                    disabled={busy} onChange={(_, v) => setParam(p.key, v as number)} />
                  <span className="text-[10px]" style={{ width: 28, textAlign: "right" }}>{params[p.key]}</span>
                </div>
              ))}
            </div>
          </Collapse>

          <Button variant="contained" size="small" onClick={runPreprocess} disabled={busy || nToRun < 1 || untaggedGroups.length > 0}>
            Preprocess selected ({nToRun})
          </Button>
          {untaggedGroups.length > 0 && (
            <Typography variant="caption" sx={{ color: "var(--c-amber, #d9a441)", wordBreak: "break-word" }}>
              Tag {untaggedGroups.length} group{untaggedGroups.length === 1 ? "" : "s"} Scar/Control first: {untaggedGroups.map((g) => `${g.patient} ${g.eye}`.trim()).join(", ")}
            </Typography>
          )}
          {nDone > 0 && (
            <Button variant="outlined" size="small" disabled={busy}
              onClick={() => downloadPreprocessedZip(scans.filter((s) => s.status === "done" && s.selected && s.caseId).map((s) => s.caseId!))}
              title="Download the selected corrected scans as one .zip (a folder of <case_id>.nii.gz) — open it in the annotator app for manual ground-truth segmentation">
              ⬇ Download preprocessed (.zip) ({nDone})
            </Button>
          )}
          {nDone > 0 && (
            <Button variant="contained" color="secondary" size="small" onClick={runSamConsensus} disabled={busy}>
              {nDone > 1 ? `Run SAM2 + consensus (${nDone})` : "Run SAM2"}
            </Button>
          )}
        </>
      )}

      {busy && <LinearProgress />}
      {step && (
        <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-word" }}>{step}</Typography>
      )}
      {report && (
        <div className="rounded p-2 flex flex-col gap-1" style={{ backgroundColor: "var(--c-surface2)", borderLeft: "3px solid var(--c-green)" }}>
          <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
            Consensus{reportLabel ? ` — ${reportLabel}` : ""} ({report.n_scans} scans)
          </div>
          <div className="flex justify-between text-xs"><span style={{ color: "var(--c-text-dim)" }}>Scar volume</span><b>{report.scar_volume_mm3.mean} ± {report.scar_volume_mm3.std} mm³</b></div>
          <div className="flex justify-between text-xs"><span style={{ color: "var(--c-text-dim)" }}>Volume CV</span><b>{report.scar_volume_mm3.cv_percent}%</b></div>
          {report.mean_pairwise_scar_dice != null && (
            <div className="flex justify-between text-xs"><span style={{ color: "var(--c-text-dim)" }}>Scar Dice</span><span>{report.mean_pairwise_scar_dice}</span></div>
          )}
          {report.mean_pairwise_scar_dice_fov != null && (
            <div className="flex justify-between text-xs" title="Dice measured only where both scans have data — isolates true disagreement from partial field-of-view / partial cuts">
              <span style={{ color: "var(--c-text-dim)" }}>Scar Dice (shared FOV)</span><b>{report.mean_pairwise_scar_dice_fov}</b></div>
          )}
          <div className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>
            Volume CV is the reproducibility biomarker. A lower full Dice with a higher shared-FOV Dice = partial overlap (partial cuts), not mis-segmentation.
          </div>
        </div>
      )}
    </div>
  );
}
