/* OCT preprocessing loader (Optovue Avanti .OCT → corrected volume).
   Flow: upload .OCT files OR load a folder → scans auto-group by (patient, eye) →
   for each group tag Scar / Control (the whole group), scrub each replicate scan to set
   its scar frame-range → tune correction params → Preprocess the selected scans
   (OCT→correct, correct Avanti geometry). Grouping is editable: rename a group, or move a
   scan to another / a new group. (SAM2 + consensus runs per group, downstream of this.) */

import { useEffect, useRef, useState } from "react";
import { Button, Typography, TextField, LinearProgress, Slider, Checkbox, ToggleButton, ToggleButtonGroup, Collapse, Select, MenuItem } from "@mui/material";
import { api, resourceUrl } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import type { ConsensusReport } from "../../api/types";
import { scanStep, LIFECYCLE_STEPS } from "../../api/lifecycle";

type Status = "queued" | "uploading" | "ready" | "preprocessing" | "done" | "error";
type Cls = "scar" | "control";

// A group is one (patient, eye) — a set of replicate scans that share a Scar/Control tag.
// origPatient/origEye record the auto-parsed identity so we can tell when the user has
// edited the header (and should persist the correction to the backend).
interface OctGroup {
  id: string;
  patient: string;
  eye: string;
  condition?: Cls;  // DEPRECATED — scar/control is now per-SCAN (OctScan.condition). Kept only so the
                    // group header can offer a "set all in group" bulk shortcut; never read as the source of truth.
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
  condition?: Cls;             // per-scan scar / control (no scar) tag — individually specified
  scarRange: [number, number]; // per-scan (only used when this scan is tagged "scar")
  // Replicate set WITHIN the eye group. One eye can hold scans of DIFFERENT scars (e.g. 3
  // posterior + 2 inferior) — those are NOT replicates of each other, so SAM2 + consensus run
  // PER subgroup, not per eye. Assigned after preprocessing; "1" = the default single subgroup.
  subgroup: string;
  selected: boolean;
  passes?: number;   // iterative-refinement passes produced (drives the download pass selector)
  error?: string;
  life?: Record<string, unknown>;  // per-scan lifecycle flags from cases/list → entry colour (lifecycle.ts)
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
  passes?: number;   // iterative-refinement passes produced (for the "download which pass" selector)
  error?: string;
  life?: Record<string, unknown>;  // per-scan lifecycle flags (cases/list) → entry colour
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
const defaultParams = (): Record<string, number> => ({
  ...Object.fromEntries(PARAMS.map((p) => [p.key, p.def])),
  auto_tune: 1,                 // app auto-tunes the DP detector per scan (1 = on, 0 = off)
  autotune_smooth_weight: 18,   // auto-tune bias: lower = sharper/tighter surface, higher = smoother
});

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

// Monotonic client-side id for groups/scans (stable React keys + move targets).
let _seq = 0;
const uid = (p: string) => `${p}${++_seq}`;

// Download preprocessed scans for manual ground-truth segmentation. In the Tauri desktop shell we
// pop a NATIVE "Save As" dialog (the webview won't prompt) and have the sidecar write the file to the
// chosen path; in a plain browser we fall back to a normal download (saves to ~/Downloads).
function inTauri(): boolean {
  return typeof window !== "undefined" &&
    ("__TAURI_INTERNALS__" in window || "__TAURI__" in window || "__TAURI_IPC__" in window);
}
const triggerDownload = (href: string, name: string) => {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
};
async function downloadPreprocessed(cid: string, octName: string, passNum?: number | null) {
  const sfx = passNum ? `_pass${passNum}` : "";
  const name = `${octName.replace(/\.oct$/i, "")}${sfx}.nii.gz`;
  const q = passNum ? `?pass_num=${passNum}` : "";
  if (inTauri()) {
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const dest = await save({ defaultPath: name });
      if (!dest) return; // user cancelled
      await api.json(`/api/case/${encodeURIComponent(cid)}/save-preprocessed`, "POST",
        JSON.stringify({ dest, pass_num: passNum ?? null }));
    } catch (e) {
      alert(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    }
    return;
  }
  // Browser: fetch bytes → blob URL so the saved filename matches the source scan.
  const path = `/api/case/${encodeURIComponent(cid)}/preprocessed.nii.gz${q}`;
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
async function downloadPreprocessedZip(cids: string[], passNum?: number | null) {
  if (!cids.length) return;
  const sfx = passNum ? `_pass${passNum}` : "";
  if (inTauri()) {
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const dest = await save({ defaultPath: `preprocessed_scans${sfx}.zip` });
      if (!dest) return; // user cancelled
      await api.json(`/api/preprocessed-zip-save`, "POST",
        JSON.stringify({ cases: cids, dest, pass_num: passNum ?? null }));
    } catch (e) {
      alert(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    }
    return;
  }
  const q = passNum ? `&pass_num=${passNum}` : "";
  triggerDownload(
    resourceUrl(`/api/preprocessed-zip?cases=${cids.map(encodeURIComponent).join(",")}${q}`),
    `preprocessed_scans${sfx}.zip`,
  );
}

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
  // Surface-detector strategy: "dp" = the native dynamic-programming detector (default) | "legacy" = the old
  // gradient-argmax + RANSAC path. Exposed so the legacy strategy stays available (compare / fall back) and
  // can be retired once DP is proven. Sent in the preprocess params; auto-tune only applies to DP.
  const [detector, setDetectorState] = useState<"dp" | "legacy">("dp");
  // Iterative refinement: re-apply the correction until the boundary stops improving (auto-stops
  // before it worsens). Default 5 (auto-converge); 1 = the single faithful pass.
  const [maxPasses, setMaxPasses] = useState(5);
  // Which refinement pass to EXPORT when downloading (null = the working/best volume).
  const [downloadPass, setDownloadPass] = useState<number | null>(null);
  const [report, setReport] = useState<ConsensusReport | null>(null);
  const [reportLabel, setReportLabel] = useState("");
  const [casesCount, setCasesCount] = useState<number | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const toggleCollapse = (id: string) =>
    setCollapsed((cur) => { const n = new Set(cur); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const setCaseId = useCaseStore((s) => s.setCaseId);
  const clearCase = useCaseStore((s) => s.clearCase);
  const openCase = useCaseStore((s) => s.openCase);
  const setStage = useWorkflowStore((s) => s.setStage);
  const initTabs = useWorkflowStore((s) => s.initTabs);
  const segSig = useWorkflowStore((s) => s.segVersion); // bumps on a re-preprocess (incl. Fix-columns)

  // Background pre-warm bookkeeping (so clicking a scan to scrub is instant).
  const busyRef = useRef(false);
  busyRef.current = busy;
  const scansRef = useRef<OctScan[]>([]);
  scansRef.current = scans;
  const warmedRef = useRef<Set<string>>(new Set());
  // Per-caseId chain of classification writes → last user action is the last write to land (no out-of-order clobber).
  const classifyChainRef = useRef<Map<string, Promise<unknown>>>(new Map());

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

  // Persist a scan's scar/control decision to its manifest RIGHT AWAY (the /classification endpoint is
  // metadata-only — the geometric OCT correction never used it, so this is NOT a re-preprocess and a
  // "done" scan stays done). Best-effort: local state is the source of truth for the UI; a failed write
  // just won't survive a reload. Fire-and-forget so the toggle stays snappy.
  const persistClassification = (caseId: string | undefined, condition: Cls | undefined, scarRange: [number, number]) => {
    if (!caseId) return;
    const body = JSON.stringify({
      classification: condition ?? null,
      scar_range: condition === "scar" ? scarRange : null,
    });
    // Chain after this case's previous write so rapid toggles (e.g. scar → control) land in the order
    // the user clicked them; the newest call is always the last to reach the manifest.
    const prev = classifyChainRef.current.get(caseId) ?? Promise.resolve();
    const next = prev
      .catch(() => undefined)
      .then(() => api.json(`/api/case/${caseId}/classification`, "POST", body).catch(() => undefined));
    classifyChainRef.current.set(caseId, next);
  };
  // Tag ONE scan Scar / Control (no scar). Updates local state + the per-scan lifecycle flag (so the
  // entry colour / timeline reflect it) and persists. Does NOT invalidate the corrected volume.
  const withCondition = (s: OctScan, condition?: Cls): OctScan =>
    ({ ...s, condition, life: s.life ? { ...s.life, scar_classification: condition ?? null } : s.life });
  const setScanCondition = (scanId: string, condition?: Cls) => {
    const scan = scans.find((s) => s.id === scanId);
    setScans((cur) => cur.map((s) => (s.id === scanId ? withCondition(s, condition) : s)));
    if (scan) persistClassification(scan.caseId, condition, scan.scarRange);
  };
  // Group header shortcut: set EVERY (non-errored) scan in the group to the same tag at once.
  const setGroupCondition = (gid: string, condition?: Cls) => {
    const groupScans = scans.filter((s) => s.groupId === gid && s.status !== "error");
    setScans((cur) => cur.map((s) => (s.groupId === gid && s.status !== "error" ? withCondition(s, condition) : s)));
    groupScans.forEach((s) => persistClassification(s.caseId, condition, s.scarRange));
  };
  // The group's displayed tag = the common value when all its scans agree, else null (indeterminate).
  const groupCondition = (gid: string): Cls | null => {
    const cs = scans.filter((s) => s.groupId === gid && s.status !== "error");
    if (!cs.length) return null;
    const first = cs[0].condition ?? null;
    return cs.every((s) => (s.condition ?? null) === first) ? first : null;
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
        error: c.error, scarRange: [1, 101], subgroup: "1", selected: !c.error, passes: c.passes, life: c.life,
        condition: ((c.life?.scar_classification as Cls | null | undefined) ?? undefined) || undefined,
      });
    }
    setGroups(order);
    setScans(newScans);
    return { nGroups: order.length, firstCase: cases.find((c) => !c.error)?.case_id };
  };

  // On startup, re-hydrate the loader from cases already processed in prior sessions so they're
  // viewable without re-loading the folder. Retries while the sidecar is still starting; never
  // clobbers a folder the user loads manually in the meantime.
  useEffect(() => {
    let stop = false;
    (async () => {
      for (let i = 0; i < 20 && !stop; i++) {
        try {
          const r = await api.json<{ cases: LoadedCase[] }>("/api/cases/list");
          if (!stop && scansRef.current.length === 0 && r.cases?.length) { ingest(r.cases); setLoaded(true); }
          return; // got a response (even empty) → stop retrying
        } catch { await new Promise((res) => setTimeout(res, 600)); }
      }
    })();
    return () => { stop = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // A re-preprocess elsewhere (notably the viewer's Fix-columns "Re-run", which runs a SINGLE pass)
  // changes a scan's pass count. Refresh passes from cases/list when segVersion bumps, so the
  // "Download pass" selector never offers passes that no longer exist. Merge by caseId only.
  useEffect(() => {
    if (segSig === 0) return;
    let stop = false;
    (async () => {
      try {
        const r = await api.json<{ cases: LoadedCase[] }>("/api/cases/list");
        if (stop || !r.cases) return;
        const byId = new Map(r.cases.map((c) => [c.case_id, c]));
        setScans((cur) => cur.map((s) => { const c = s.caseId ? byId.get(s.caseId) : undefined;
          return c ? { ...s, passes: c.passes ?? s.passes, life: c.life ?? s.life } : s; }));
      } catch { /* best-effort */ }
    })();
    return () => { stop = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segSig]);

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
    // Clear the sidebar list + empty the viewer IMMEDIATELY (optimistic) so the UI responds at once — the
    // on-disk delete of many case folders can take a while; the busy LinearProgress shows it's working.
    setScans([]); setGroups([]); setLoaded(false); setReport(null); setReportLabel(""); setActiveId(null);
    setCasesCount(0); clearCase();
    setStep("Wiping all saved cases…");
    try {
      const r = await api.json<{ removed: number; freed_bytes: number }>("/api/cases/wipe", "POST", JSON.stringify({}));
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
  // Switching detector strategy also invalidates corrected scans (the boundary differs).
  const setDetector = (d: "dp" | "legacy") => {
    setDetectorState(d);
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
    let failed = 0, done = 0;
    let lastIter: { passes?: number; metrics?: number[]; best_pass?: number; stopped?: string } | null = null;
    let lastOkCase: string | null = null;
    // Process MULTIPLE scans CONCURRENTLY (was one-by-one, leaving cores idle during each scan's serial
    // phases). Ask the SIDECAR how many to run at once — it knows the REAL machine (CPU cores + RAM), so a
    // big box (e.g. 24-thread / 64 GB) runs many in parallel and a small one stays safe. Each scan then uses
    // cpu_budget//conc workers (K × that ≈ all cores), and one scan's serial phases overlap another's parallel
    // phases → fuller CPU/RAM use. Falls back to a CPU-only heuristic if the capabilities call fails.
    let conc = 1;
    if (sel.length > 1) {
      let maxConc = Math.max(1, Math.min(4, Math.floor((navigator.hardwareConcurrency || 8) / 4)));
      try {
        const caps = await api.json<{ max_concurrency?: number }>("/api/system/capabilities");
        if (caps?.max_concurrency && caps.max_concurrency > 0) maxConc = caps.max_concurrency;
      } catch { /* keep the heuristic fallback */ }
      conc = Math.max(1, Math.min(maxConc, sel.length));
    }
    try {
      const runOne = async (s: typeof sel[number]) => {
        const g = groups.find((gg) => gg.id === s.groupId);
        const cls = s.condition ?? null;   // per-scan scar/control tag (was group-level)
        const edited = g && isEdited(g);
        patchScan(s.id, { status: "preprocessing" });
        try {
          const r = await api.json<{ oct_iter?: { passes?: number; metrics?: number[]; best_pass?: number; stopped?: string } }>(
            `/api/case/${s.caseId}/oct-preprocess`, "POST", JSON.stringify({
              params: { ...params, detector },   // detector strategy (dp | legacy) for THIS preprocess
              classification: cls,
              scar_range: cls === "scar" ? s.scarRange : null,
              max_iterations: maxPasses,
              concurrency: conc,
              // Persist a user-corrected identity so the rename reaches consensus/export.
              ...(edited ? { patient: g!.patient.trim(), eye: g!.eye.trim() } : {}),
            }));
          lastIter = r.oct_iter ?? null;
          patchScan(s.id, { status: "done", passes: r.oct_iter?.passes ?? 1 });
          lastOkCase = s.caseId!;
        } catch (e) {
          patchScan(s.id, { status: "error", error: msg(e) });
          failed++;
        }
        done++;
        setStep(`Preprocessing ${done}/${sel.length}${conc > 1 ? ` (${conc} at a time)` : ""}${maxPasses > 1 ? ` · iterative ≤${maxPasses} passes` : ""} — up to ~20 min for a busy batch.`);
      };
      // concurrency-limited pool: `conc` worker loops pull from a shared index
      let next = 0;
      const pump = async () => { while (next < sel.length) { const i = next++; await runOne(sel[i]); } };
      await Promise.all(Array.from({ length: Math.min(conc, sel.length) }, () => pump()));
      // Show the last successfully-corrected scan in the viewer (one switch at the end avoids races between
      // concurrent openCase calls). openCase loads its corrected working volume; initTabs refetches previews.
      if (lastOkCase) { setActiveId(lastOkCase); setCaseId(lastOkCase); await openCase(); initTabs(false); }
      // Convergence summary (single scan, iterative): show every pass's boundary deviation (raw +
      // each pass) and WHICH pass was kept (the least-deviant) — so a pass that worsened the boundary
      // is visible and was NOT kept. Step through them in ⇆ Before/after.
      let conv = "";
      const li = lastIter as { passes?: number; metrics?: number[]; best_pass?: number; stopped?: string } | null;
      if (sel.length === 1 && li && (li.passes ?? 1) > 1) {
        const ms = (li.metrics ?? []).map((x: number) => x.toFixed(2)).join(" → ");
        const best = li.best_pass ?? li.passes;
        conv = ` Refined ${li.passes} passes — kept pass ${best} (boundary deviation raw→passes: ${ms} px; lower=flatter${li.stopped ? `, stopped: ${li.stopped}` : ""}). Step through them in ⇆ Before/after.`;
      }
      // Don't claim success when some scans failed — surface the partial result.
      setStep(failed
        ? `Preprocessing finished: ${sel.length - failed}/${sel.length} OK, ${failed} failed (see the red rows).`
        : `Preprocessing complete.${conv} Now Run SAM2 + consensus on the corrected scans.`);
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
  // Most refinement passes among the selected, done scans — bounds the "download which pass" selector.
  const maxScanPasses = Math.max(1, ...scans.filter((s) => s.status === "done" && s.selected).map((s) => s.passes ?? 1));
  // Tagging is the point of this panel: don't let a scan in an untagged group preprocess
  // (it would be committed with no Scar/Control label, silently producing unlabeled data).
  // Any selected scan still missing a scar/control tag → show the (optional) hint.
  const hasUntaggedRunnable = scans.some((s) => !s.condition && s.selected && s.caseId && s.status !== "error");

  return (
    <div className="flex flex-col gap-2">
      <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
        Load Optovue .OCT scans — each needs its <b>.txt</b> next to it (a folder grabs both).
        Scans auto-group by patient/eye; scrub the replicate scans, then preprocess. Decide Scar/Control
        later, in the Scar stage (it doesn't affect the correction).
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
            const isCollapsed = collapsed.has(g.id);
            const nDoneInGroup = groupScans.filter((s) => s.status === "done").length;
            return (
              <div key={g.id} className="rounded" style={{ border: "1px solid var(--c-border)" }}>
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
                    {/* "Set all" shortcut — tags EVERY scan in the group at once (each scan also has its own
                        toggle below). Shows the common tag, or blank when the group's scans disagree. */}
                    <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>all:</span>
                    <ToggleButtonGroup size="small" exclusive value={groupCondition(g.id)}
                      onChange={(_, v) => setGroupCondition(g.id, (v as Cls) || undefined)}>
                      <ToggleButton value="scar" sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Scar</ToggleButton>
                      <ToggleButton value="control" sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Control (no scar)</ToggleButton>
                    </ToggleButtonGroup>
                    <span className="text-[10px]" style={{ marginLeft: "auto", color: nDoneInGroup ? "var(--c-green)" : "var(--c-text-dim)" }}>
                      {groupScans.length} scan{groupScans.length === 1 ? "" : "s"}{nDoneInGroup ? ` · ${nDoneInGroup} done ✓` : ""}
                    </span>
                    {nDoneInGroup > 0 && (
                      <button title={`Download all ${nDoneInGroup} preprocessed scan(s) in this group as a .zip (a folder for manual segmentation)`}
                        onClick={() => downloadPreprocessedZip(groupScans.filter((s) => s.status === "done" && s.caseId).map((s) => s.caseId!), downloadPass)}
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
                    // Per-scan lifecycle colour (red→orange→yellow→light blue→dark blue→green) from the
                    // timeline step; only from step 2 (preprocessed-auto) onward. The actively-viewed scan
                    // keeps ITS lifecycle colour but a DARKER/stronger shade of it (not a blue accent) so the
                    // open scan stands out without losing its step colour; a colourless (raw) scan falls back
                    // to the accent tint so the selection is still visible.
                    const lifeStep = s.life ? scanStep(s.life) : 0;
                    const lifeColor = lifeStep >= 2 ? LIFECYCLE_STEPS[lifeStep].color : null;
                    const rowBg = active
                      ? (lifeColor ? `${lifeColor}66` : "rgba(90,127,168,0.32)")
                      : lifeColor ? `${lifeColor}22` : "transparent";
                    const rowBorder = active ? (lifeColor ?? "var(--c-accent)") : lifeColor ?? "transparent";
                    return (
                      <div key={s.id} className="rounded px-1 py-0.5" style={{ background: rowBg, borderLeft: `2px solid ${rowBorder}` }}>
                        <div className="flex items-start gap-1.5 text-xs">
                          <Checkbox size="small" checked={s.selected} disabled={busy || s.status === "error"} sx={{ p: 0.25 }}
                            onChange={(e) => patchScan(s.id, { selected: e.target.checked })} />
                          <span style={{ width: 8, height: 8, borderRadius: "50%", background: lifeColor ?? DOT[s.status], flex: "none", marginTop: 5 }} />
                          {/* Full name (wrap, don't truncate) — Optovue .OCT names are long & spaceless. */}
                          <span style={{ flex: 1, minWidth: 0, overflowWrap: "anywhere", lineHeight: 1.35, color: done ? "var(--c-green)" : undefined, cursor: clickable ? "pointer" : "default" }}
                            title={s.error || s.filename} onClick={() => clickable && preview(s.caseId)}>
                            {s.filename.replace(/\.OCT$/i, "")}
                          </span>
                          <span style={{ color: s.status === "error" ? "var(--c-red)" : lifeColor ?? (done ? "var(--c-green)" : "var(--c-text-dim)") }}>
                            {s.status === "error" ? "failed" : lifeColor ? LIFECYCLE_STEPS[lifeStep].short : s.status}
                          </span>
                          {done && s.caseId && (
                            <button title="Download this preprocessed scan (.nii.gz) for manual segmentation"
                              onClick={(e) => { e.stopPropagation(); void downloadPreprocessed(s.caseId!, s.filename, downloadPass && downloadPass <= (s.passes ?? 1) ? downloadPass : null); }}
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

                        {/* Per-scan Scar / Control (no scar) tag — individually specified for THIS scan
                            (not the whole group). Persists immediately; never re-preprocesses. */}
                        {s.status !== "error" && (
                          <div className="ml-6 mt-0.5 flex items-center gap-1">
                            <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>scar?</span>
                            <ToggleButtonGroup size="small" exclusive value={s.condition ?? null}
                              onChange={(_, v) => setScanCondition(s.id, (v as Cls) || undefined)}>
                              <ToggleButton value="scar" disabled={busy} sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Scar</ToggleButton>
                              <ToggleButton value="control" disabled={busy} sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>No scar</ToggleButton>
                            </ToggleButtonGroup>
                          </div>
                        )}

                        {/* Scar frame-range (per scan) — only when THIS scan is tagged Scar. The range is
                            scar-detection metadata (not geometry), so changing it never invalidates the
                            corrected volume; persisted on release. */}
                        {s.condition === "scar" && s.status !== "error" && (
                          <div className="ml-6 mt-1 pr-2">
                            <div className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>scar frames {s.scarRange[0]}–{s.scarRange[1]}</div>
                            <Slider size="small" min={1} max={s.nFrames} value={s.scarRange} disabled={busy}
                              onChange={(_, v) => patchScan(s.id, { scarRange: v as [number, number] })}
                              onChangeCommitted={(_, v) => persistClassification(s.caseId, "scar", v as [number, number])} />
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

          {/* DETECTOR strategy: DP (native dynamic-programming, default) vs Legacy (old gradient-argmax + RANSAC).
              Kept so legacy stays available to compare/fall back; remove the toggle once DP is proven. */}
          <div className="flex items-center gap-2 px-1" title="Corneal-surface detector. DP = the native dynamic-programming detector (recommended). Legacy = the old gradient-argmax + RANSAC method.">
            <span className="text-[10px]" style={{ width: 88, color: "var(--c-text-dim)" }}>Detector</span>
            <ToggleButtonGroup size="small" exclusive value={detector} disabled={busy}
              onChange={(_, v) => v && setDetector(v as "dp" | "legacy")}>
              <ToggleButton value="dp" sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>DP</ToggleButton>
              <ToggleButton value="legacy" sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Legacy</ToggleButton>
            </ToggleButtonGroup>
          </div>

          {/* Native AUTO-TUNE control (DP only): the app tunes the DP detector to each scan itself. The toggle
              turns it on/off; the bias slider shifts what it optimises for (sharper vs smoother surface). */}
          {detector === "dp" && (
            <div className="flex items-center gap-2 px-1" title="When on, the app auto-tunes the corneal-surface detector to EACH scan (no manual params) at preprocess time.">
              <span className="text-[10px]" style={{ width: 88, color: "var(--c-text-dim)" }}>Auto-tune detect</span>
              <ToggleButton size="small" value="at" selected={(params.auto_tune ?? 1) > 0} disabled={busy}
                onChange={() => setParam("auto_tune", (params.auto_tune ?? 1) > 0 ? 0 : 1)}
                sx={{ py: 0, px: 1.2, fontSize: 10, textTransform: "none" }}>
                {(params.auto_tune ?? 1) > 0 ? "On" : "Off"}
              </ToggleButton>
            </div>
          )}
          {detector === "dp" && (params.auto_tune ?? 1) > 0 && (
            <div className="flex items-center gap-2 px-1" title="Bias the per-scan auto-tune: lower = sharper / tighter to the boundary; higher = smoother surface.">
              <span className="text-[10px]" style={{ width: 88, color: "var(--c-text-dim)" }}>· sharper↔smoother</span>
              <Slider size="small" min={6} max={40} step={1} value={params.autotune_smooth_weight ?? 18}
                disabled={busy} onChange={(_, v) => setParam("autotune_smooth_weight", v as number)} />
              <span className="text-[10px]" style={{ width: 28, textAlign: "right" }}>{params.autotune_smooth_weight ?? 18}</span>
            </div>
          )}

          <div className="flex items-center gap-2 px-1" title="Re-applies the correction until the corneal boundary stops improving toward its curve fit (auto-stops just before it worsens). 1 = a single pass (faithful method).">
            <span className="text-[10px]" style={{ width: 88, color: "var(--c-text-dim)" }}>Refine passes</span>
            <Slider size="small" min={1} max={8} step={1} value={maxPasses} disabled={busy}
              onChange={(_, v) => setMaxPasses(v as number)} />
            <span className="text-[10px]" style={{ width: 28, textAlign: "right" }}>
              {maxPasses === 1 ? "1" : `≤${maxPasses}`}
            </span>
          </div>
          <span className="text-[10px] px-1" style={{ color: "var(--c-text-dim)", opacity: 0.8 }}>
            {maxPasses === 1 ? "Single pass (faithful method)." : "Iterative — auto-stops at convergence; step through passes in ⇆ Before/after."}
          </span>

          {/* #4: tagging Scar/Control is NO LONGER required before preprocessing — decide it after, in
              the Scar stage (the geometric correction never used it). Preprocess freely while untagged. */}
          <Button variant="contained" size="small" onClick={runPreprocess} disabled={busy || nToRun < 1}>
            Preprocess selected ({nToRun})
          </Button>
          {hasUntaggedRunnable && (
            <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-word" }}>
              Scar/Control is optional here — tag each scan now or after preprocessing in the Scar stage.
            </Typography>
          )}
          {nDone > 0 && maxScanPasses > 1 && (
            <div className="flex items-center gap-2 px-1" title="Choose which refinement pass the ⬇ downloads export. 'Best' = the pass the app kept. A scan with fewer passes falls back to its best.">
              <span className="text-[10px]" style={{ width: 88, color: "var(--c-text-dim)" }}>Download pass</span>
              <Select size="small" variant="standard" value={downloadPass ?? 0}
                onChange={(e) => { const v = Number(e.target.value); setDownloadPass(v === 0 ? null : v); }}
                sx={{ fontSize: 11, flex: 1 }}>
                <MenuItem value={0} sx={{ fontSize: 11 }}>Best (recommended)</MenuItem>
                {Array.from({ length: maxScanPasses }, (_, i) => i + 1).map((k) => (
                  <MenuItem key={k} value={k} sx={{ fontSize: 11 }}>Pass {k}</MenuItem>
                ))}
              </Select>
            </div>
          )}
          {nDone > 0 && (
            <Button variant="outlined" size="small" disabled={busy}
              onClick={() => downloadPreprocessedZip(scans.filter((s) => s.status === "done" && s.selected && s.caseId).map((s) => s.caseId!), downloadPass)}
              title="Download the selected corrected scans as one .zip (a folder of <case_id>.nii.gz) — open it in the annotator app for manual ground-truth segmentation">
              ⬇ Download preprocessed (.zip){downloadPass ? ` — pass ${downloadPass}` : ""} ({nDone})
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
