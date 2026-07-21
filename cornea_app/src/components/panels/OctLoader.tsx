/* OCT preprocessing loader (Optovue Avanti .OCT → corrected volume).
   Flow: upload .OCT files OR load a folder → scans auto-group by (patient, eye) →
   for each group tag Scar / Control (the whole group), scrub each replicate scan to set
   its scar frame-range → tune correction params → Preprocess the selected scans
   (OCT→correct, correct Avanti geometry). Grouping is editable: rename a group, or move a
   scan to another / a new group. (SAM2 + consensus runs per group, downstream of this.) */

import { useEffect, useRef, useState } from "react";
import { Button, Typography, TextField, LinearProgress, Slider, Checkbox, ToggleButton, ToggleButtonGroup, Collapse, Select, MenuItem, Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions } from "@mui/material";
import { api, resourceUrl } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import type { ConsensusReport } from "../../api/types";
import { scanStep, LIFECYCLE_STEPS } from "../../api/lifecycle";
import { REVIEW_FLAGS, reviewFlagMeta, reviewFlagsOf } from "../../api/reviewFlags";

type Status = "queued" | "uploading" | "ready" | "preprocessing" | "done" | "error";
type Cls = "scar" | "control";

// Case-type filter for the scan list below. View-only — see FILTER_OPTIONS / visibleScanIds.
// `flag:<slug>` is one option per known review flag, generated from REVIEW_FLAGS — open-ended by design,
// so adding a flag to that table needs no edit here.
type ScanFilter =
  | "all"
  | "surfacecrop" | "surfacecrop_new" | "surfacecrop_no" | "flagged" | "difficult" | "defects" | "failed"
  | `flag:${string}`
  | "raw" | "unvetted" | "nodata"
  | "scar" | "control" | "untagged";

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

// ── Case-type filter ────────────────────────────────────────────────────────────────────────
// Surface-crop resolution, hoisted out of the row badge so the badge and the filter read the SAME
// rule and can never drift apart. `auto` must check BOTH forms: most rows carry the server-derived
// surface_crop_auto from cases/list, but the OPEN scan's `life` is the full mirrored manifest
// (spread in by the live-mirror effect) and carries only oct_iter.stopped.
const scAutoOf = (life?: Record<string, unknown>): boolean =>
  Boolean(life?.surface_crop_auto)
  || ((life?.oct_iter as Record<string, unknown> | undefined)?.stopped === "surface_crop");
// The human verdict (surface_crop_manual) WINS over auto. It is TRI-STATE — true / false / null —
// so "unreviewed" is always `== null`, NEVER `!x`: false means "reviewed, and NOT a crop", and a
// loose test would fold rejected scans back into the unreviewed queue.
const scOf = (life?: Record<string, unknown>): boolean => {
  const m = life?.surface_crop_manual;
  return m != null ? Boolean(m) : scAutoOf(life);
};

// One filter option. `test` is a PURE read of the scan's own state — no network, no manifest write
// (filtering is view-only). Every predicate must be truthiness-based and tolerate life === undefined:
// a row's `life` is absent until cases/list hydrates, and over its lifetime it alternates between the
// list payload (bools + counts) and the full mirrored manifest (objects + arrays).
interface FilterOpt { key: ScanFilter; group: string; label: string; test: (s: OctScan) => boolean; }
const FILTER_GROUPS = ["Needs attention", "Lifecycle", "Tag"];
const FILTER_OPTIONS: FilterOpt[] = [
  { key: "surfacecrop", group: "Needs attention", label: "⬚ Surface crop",
    test: (s) => scOf(s.life) },
  { key: "surfacecrop_new", group: "Needs attention", label: "⬚ Surface crop · unreviewed",
    test: (s) => scAutoOf(s.life) && s.life?.surface_crop_manual == null },
  { key: "surfacecrop_no", group: "Needs attention", label: "⬚ Surface crop · rejected",
    test: (s) => s.life?.surface_crop_manual === false },
  // Catch-all: stays ahead of the per-flag options below because it is the only one that also finds
  // UNKNOWN slugs — a flag the backend gained before the table did, or a legacy A/B/C value.
  { key: "flagged", group: "Needs attention", label: "⚑ Flagged for review",
    test: (s) => reviewFlagsOf(s.life).length > 0 },
  // One option per KNOWN flag, generated from the shared table (api/reviewFlags.ts) so the option label
  // and the row badge can never name the same slug differently.
  ...REVIEW_FLAGS.map((f): FilterOpt => ({
    key: `flag:${f.slug}`, group: "Needs attention", label: `⚑ ${f.label}`,
    test: (s) => reviewFlagsOf(s.life).includes(f.slug),
  })),
  { key: "difficult", group: "Needs attention", label: "⚠ Difficult",
    test: (s) => Boolean(s.life?.difficult_scan) },
  // defect_marks is DUAL-SHAPE: a COUNT in the cases/list payload, the raw ARRAY on the open scan.
  { key: "defects", group: "Needs attention", label: "Has defect marks",
    test: (s) => { const d = s.life?.defect_marks; return Array.isArray(d) ? d.length > 0 : Number(d ?? 0) > 0; } },
  { key: "failed", group: "Needs attention", label: "Failed",
    test: (s) => s.status === "error" },
  // Local status is checked alongside the manifest flag: a scan preprocessed in THIS session is
  // status "done" long before its `life` refreshes (that only happens on a segVersion bump).
  { key: "raw", group: "Lifecycle", label: "Not preprocessed",
    test: (s) => s.status !== "error" && s.status !== "done" && !s.life?.oct_preprocessed },
  // NOT scanStep(life) === 2 — scanStep returns 4+ as soon as sam2_meta is set, so a
  // segmented-but-never-vetted scan would silently drop out of the vetting backlog.
  { key: "unvetted", group: "Lifecycle", label: "Preprocessed · not vetted",
    test: (s) => s.status !== "error" && (Boolean(s.life?.oct_preprocessed) || s.status === "done") && !s.life?.preproc_vetted },
  // /api/oct/load-dir returns rows with NO `life` until a cases/list hydration lands, so every other
  // predicate hides them. Give them a bucket rather than letting a fresh folder load look broken.
  { key: "nodata", group: "Lifecycle", label: "No lifecycle data yet",
    test: (s) => s.life == null },
  // Tag reads s.condition, not life.scar_classification: the toggles in THIS panel write s.condition
  // optimistically, so `life` lags a just-tagged scan until a segVersion refresh.
  { key: "scar", group: "Tag", label: "Scar",
    test: (s) => s.condition === "scar" },
  { key: "control", group: "Tag", label: "Control (no scar)",
    test: (s) => s.condition === "control" },
  { key: "untagged", group: "Tag", label: "Untagged",
    test: (s) => !s.condition && s.status !== "error" },
];

// Which scans the list is currently SHOWING. The batch handlers resolve their work set through this
// SAME function, so what you see is exactly what runs (selection safety — see the filter row below).
// Pure, with every dependency passed in, so it is safe to call from render AND from a handler closure.
// The currently-OPEN scan is always kept visible: filtering it away would make the row being actively
// corrected vanish under the cursor and drop it from the batch scope mid-edit. It is never auto-closed
// (openCase writes default_case_id via putConfig — a network write a view control must not trigger).
function visibleScanIds(scans: OctScan[], groups: OctGroup[], filter: ScanFilter, query: string, activeId: string | null): Set<string> {
  const q = query.trim().toLowerCase();
  if (filter === "all" && !q) return new Set(scans.map((s) => s.id));
  const opt = FILTER_OPTIONS.find((o) => o.key === filter);   // undefined for "all" = no constraint
  const gById = new Map(groups.map((g) => [g.id, g]));
  const out = new Set<string>();
  for (const s of scans) {
    if (s.caseId && s.caseId === activeId) { out.add(s.id); continue; }   // the open scan is pinned
    if (opt && !opt.test(s)) continue;
    if (q) {
      // Search the GROUP identity too, so typing "CS021" or "OD" surfaces whole eyes, not only the
      // scans whose filename happens to contain it.
      const g = gById.get(s.groupId);
      const hay = `${s.filename} ${s.caseId ?? ""} ${s.subgroup} ${g?.patient ?? ""} ${g?.eye ?? ""}`.toLowerCase();
      if (!hay.includes(q)) continue;
    }
    out.add(s.id);
  }
  return out;
}

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
  // Surface detector: ALWAYS the native dynamic-programming path WITH the legacy cross-check (scar-guard).
  // Preprocessing runs BOTH and keeps the DP result within the vicinity of the legacy edge, so the old
  // DP|Legacy selection toggle was removed. Kept as a const ("dp") so the auto-tune gating below still reads.
  const detector = "dp" as const;
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
  // View-only case-type filter. It narrows which scans the list SHOWS and — deliberately — which
  // ones the batch buttons act on: batch = VISIBLE ∧ selected, so nothing you cannot see can be
  // preprocessed or exported. It NEVER writes `selected`, `collapsed`, a manifest, or the network.
  // Session-only, never persisted (there is no client-side persistence anywhere in src/, and the
  // only durable channel is api.putConfig — a network write); also reset in ingest(), because an
  // app that boots showing 4 of 308 scans with the reason forgotten is the worst failure mode here.
  const [filter, setFilter] = useState<ScanFilter>("all");
  const [query, setQuery] = useState("");
  const filtering = filter !== "all" || query.trim() !== "";
  const clearFilter = () => { setFilter("all"); setQuery(""); };
  // Resolved ONCE per render. A batch handler closes over THIS render's set, so clicking a button
  // snapshots the visible scope at click time — a mid-run filter change cannot alter a running batch.
  const visibleIds = visibleScanIds(scans, groups, filter, query, activeId);
  // True while the post-load lifecycle hydration (hydrateLife) is in flight. The filter's per-option
  // counts are meaningless until it lands — /api/oct/load-dir and /api/oct/upload return NO `life`,
  // so every life-derived predicate reads 0 — and a disabled "⬚ Surface crop (0)" would assert the
  // corpus contains none when it may hold dozens. Drives the "loading…" note + the disabled gating.
  const [lifeLoading, setLifeLoading] = useState(false);
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
  // #1 mid-batch interaction: re-entrancy guard for preview() (so opening a finished scan during a batch
  // doesn't clobber the batch-wide `busy`), and a flag that the user opened a scan DURING a batch (so the
  // end-of-batch auto-switch doesn't yank them away from what they're correcting).
  const previewInFlightRef = useRef(false);
  const manualPreviewRef = useRef(false);
  // Bumped by every ingest(). A hydrateLife() response that comes back after a NEWER load started is
  // dropped rather than merged onto a list it no longer describes (load a folder, then immediately
  // load another — the first cases/list would otherwise stamp stale `life` onto the second's rows).
  const loadGenRef = useRef(0);

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

  // #4: keep the sidebar entry's step colour/label in sync with the OPEN scan INSTANTLY. Timeline actions
  // (approve / classify / SAM2 / schedule …) mutate caseStore.caseInfo.manifest (immer → fresh ref), but
  // each sidebar row colours from its own `life` snapshot (fetched from cases/list). Mirror the live manifest
  // onto the matching row here so it recolours the moment a step completes — not only after reselecting it.
  const liveCaseInfo = useCaseStore((s) => s.caseInfo);
  useEffect(() => {
    const cid = liveCaseInfo?.case_id;
    const man = liveCaseInfo?.manifest as Record<string, unknown> | undefined;
    if (!cid || !man) return;
    setScans((cur) => {
      let changed = false;
      const next = cur.map((s) => {
        if (s.caseId !== cid) return s;
        changed = true;
        return { ...s, life: { ...(s.life ?? {}), ...man } };
      });
      return changed ? next : cur;
    });
  }, [liveCaseInfo]);

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
  // Persist a scan's subgroup label to its manifest (metadata-only, like classification) so the
  // replicate SET it belongs to survives a reload — otherwise distinct lesions of one eye silently
  // collapse into a single consensus. Chained per-case after any pending classification write.
  const persistSubgroup = (caseId: string | undefined, subgroup: string) => {
    if (!caseId) return;
    const body = JSON.stringify({ subgroup: subgroup.trim() || "1" });
    const prev = classifyChainRef.current.get(caseId) ?? Promise.resolve();
    const next = prev
      .catch(() => undefined)
      .then(() => api.json(`/api/case/${caseId}/subgroup`, "POST", body).catch(() => undefined));
    classifyChainRef.current.set(caseId, next);
  };
  // Group header shortcut: set EVERY (non-errored) scan in the group to the same tag at once.
  const setGroupCondition = (gid: string, condition?: Cls) => {
    const groupScans = scans.filter((s) => s.groupId === gid && s.status !== "error");
    // Belt-and-braces for the disabled toggle in the header: this is the only action in the panel that
    // PERSISTS to disk for rows that are off screen (a classification POST each, which also nulls
    // scar_range on anything not tagged "scar" — fire-and-forget, no undo). Refuse rather than write a
    // partial set, so the guarantee survives a future edit that drops the `disabled` prop.
    if (groupScans.some((s) => !visibleIds.has(s.id))) {
      setStep("Clear the filter to tag a whole group — some of its scans are hidden. (Per-scan tags still work.)");
      return;
    }
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
    loadGenRef.current += 1;   // invalidates any hydrateLife() response still in flight for the old list
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
        error: c.error,
        // Restore the persisted per-scan scar frame-range (cases/list life) so a reload + re-preprocess
        // doesn't silently overwrite it with the full [1,nFrames]; clamped to the real frame count in preview.
        scarRange: (Array.isArray(c.life?.scar_range) && (c.life!.scar_range as unknown[]).length === 2
          ? [Number((c.life!.scar_range as unknown[])[0]), Number((c.life!.scar_range as unknown[])[1])] as [number, number]
          : [1, 101]),
        subgroup: ((c.life?.scar_subgroup as string | null | undefined)?.trim() || "1"),
        selected: !c.error, passes: c.passes, life: c.life,
        condition: ((c.life?.scar_classification as Cls | null | undefined) ?? undefined) || undefined,
      });
    }
    setGroups(order);
    setScans(newScans);
    clearFilter();   // a stale filter must never make a freshly loaded folder look empty
    return { nGroups: order.length, firstCase: cases.find((c) => !c.error)?.case_id };
  };

  // Fill in the per-scan lifecycle flags after a fresh load. REQUIRED, not an optimisation:
  // /api/oct/load-dir and /api/oct/upload return only {case_id, filename, patient, eye, preprocessed},
  // with NO `life` — and nothing else backfills it (the startup cases/list effect is gated on an EMPTY
  // list so it won't re-run, and the segSig effect only fires on a re-preprocess). Without this, every
  // life-derived filter option reads (0) and sits disabled immediately after "Select folder…", which
  // both hides the surface-crop scans the filter exists to find AND asserts there are none.
  // Read-only: a GET, merged by caseId. It never writes a manifest, `selected`, or `collapsed`.
  const hydrateLife = async () => {
    const gen = loadGenRef.current;
    setLifeLoading(true);
    try {
      const r = await api.json<{ cases: LoadedCase[] }>("/api/cases/list");
      if (loadGenRef.current !== gen || !r.cases) return;   // a newer load started → drop this result
      const byId = new Map(r.cases.map((c) => [c.case_id, c]));
      // The OPEN scan's live manifest outranks cases/list (a step that just completed may not be on disk
      // yet), and loadDir opens one right after calling us — so re-apply it on top, as the segSig refresh
      // below already does for review_flags. Both shapes satisfy the predicates (see scAutoOf).
      const live = useCaseStore.getState().caseInfo;
      const liveMan = live?.manifest as Record<string, unknown> | undefined;
      setScans((cur) => cur.map((s) => {
        const c = s.caseId ? byId.get(s.caseId) : undefined;
        if (!c?.life) return s;
        const life = (liveMan && s.caseId === live?.case_id) ? { ...c.life, ...liveMan } : c.life;
        // `life`/`passes` are server truth and always merge. The tag / scar-range / subgroup fields are
        // only FILLED IN where the row still holds the ingest default: the user can tag or scrub while
        // this request is in flight, and a hydration must never clobber an edit they just made.
        const persistedRange = (Array.isArray(c.life.scar_range) && (c.life.scar_range as unknown[]).length === 2
          ? [Number((c.life.scar_range as unknown[])[0]), Number((c.life.scar_range as unknown[])[1])] as [number, number]
          : null);
        const persistedSub = (c.life.scar_subgroup as string | null | undefined)?.trim();
        const untouchedRange = s.scarRange[0] === 1 && s.scarRange[1] === 101;
        return {
          ...s,
          life,
          passes: c.passes ?? s.passes,
          condition: s.condition ?? (((c.life.scar_classification as Cls | null | undefined) ?? undefined) || undefined),
          scarRange: (persistedRange && untouchedRange) ? persistedRange : s.scarRange,
          subgroup: (persistedSub && s.subgroup === "1") ? persistedSub : s.subgroup,
        };
      }));
    } catch { /* best-effort — the list still works; the filter shows the "no lifecycle data yet" bucket */ }
    finally { if (loadGenRef.current === gen) setLifeLoading(false); }
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
        // The OPEN scan's review_flags are authoritative in the live manifest (a just-toggled /review-flag POST
        // may not be on disk yet, so cases/list can return stale []). Preserve them so the ⚑ chip doesn't flicker.
        const live = useCaseStore.getState().caseInfo;
        const liveFlags = live?.manifest ? (live.manifest as Record<string, unknown>).review_flags : undefined;
        setScans((cur) => cur.map((s) => {
          const c = s.caseId ? byId.get(s.caseId) : undefined;
          if (!c) return s;
          let life = c.life ?? s.life;
          if (life && s.caseId === live?.case_id && Array.isArray(liveFlags)) life = { ...life, review_flags: liveFlags };
          return { ...s, passes: c.passes ?? s.passes, life };
        }));
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
      void hydrateLife();   // /api/oct/upload carries no `life` — backfill it so the filter/badges work
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
      void hydrateLife();   // /api/oct/load-dir carries no `life` — backfill it so the filter/badges work
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
  // reusing the deterministic case folder + its old segmentation. Guarded by a confirm MODAL
  // (window.confirm is unreliable inside the Tauri WebKitGTK webview) — see the Dialog at render.
  const [wipeConfirmOpen, setWipeConfirmOpen] = useState(false);
  const doWipe = async () => {
    setWipeConfirmOpen(false);
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

  // Scrub a scan: materialise its raw z-stack + show grayscale in the viewer. Re-entrancy-guarded (NOT
  // gated on the batch `busy`) so a FINISHED scan can be opened for manual correction WHILE a batch runs.
  const preview = async (caseId?: string) => {
    if (!caseId || previewInFlightRef.current) return;
    previewInFlightRef.current = true;
    const duringBatch = busyRef.current;
    setActiveId(caseId);
    try {
      setCaseId(caseId);
      const r = await api.json<{ n_frames?: number; preprocessed?: boolean }>(
        `/api/case/${caseId}/oct-volume`, "POST", JSON.stringify({}),
      );
      // Only latch "don't auto-switch at batch end" once the open SUCCEEDS — a failed/transient preview
      // must not permanently suppress the end-of-batch view switch.
      if (duringBatch) manualPreviewRef.current = true;
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
      previewInFlightRef.current = false;
    }
  };

  // Changing params invalidates any already-corrected scans (result is now stale).
  const setParam = (key: string, v: number) => {
    setParams((cur) => ({ ...cur, [key]: v }));
    setScans((cur) => cur.map((s) => (s.status === "done" ? { ...s, status: "ready" } : s)));
  };
  const runPreprocess = async () => {
    // Skip already-corrected (non-stale) scans — re-clicking is a no-op for them.
    // VISIBLE ∧ selected: the filter is the batch SCOPE, not just a lens — a scan the user cannot
    // see must never be rewritten. `visibleIds` is this render's set, i.e. snapshotted at click time,
    // so changing the filter during the ~20-min run below cannot alter the work set.
    const sel = scans.filter((s) => s.selected && s.caseId && s.status !== "error" && s.status !== "done" && visibleIds.has(s.id));
    if (!sel.length) {
      setStep(`Nothing to preprocess (selected scans are already corrected — change params or tags to re-run).${filtering ? ` A filter is active — ${nHiddenSelected} selected scan(s) are hidden and excluded.` : ""}`);
      return;
    }
    setBusy(true);
    manualPreviewRef.current = false;   // track whether the user opens a finished scan mid-batch (#1)
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
        // Persist the scan's RESOLVED group identity whenever the group has a determinate eye —
        // not only when the header was hand-edited. A scan MOVED/merged into another group keeps
        // its own filename-parsed identity otherwise, so its manifest (and thus the consensus id)
        // would silently disagree with the grouping the user sees. The backend re-normalizes/
        // ignores a "?"/blank eye, so sending an unedited determinate identity is an idempotent
        // no-op for scans that were never moved.
        const sendIdentity = !!g && (isEdited(g) || (g.eye.trim() !== "" && g.eye.trim() !== "?"));
        patchScan(s.id, { status: "preprocessing" });
        try {
          const r = await api.json<{ oct_iter?: { passes?: number; metrics?: number[]; best_pass?: number; stopped?: string } }>(
            `/api/case/${s.caseId}/oct-preprocess`, "POST", JSON.stringify({
              params: { ...params, detector },   // detector strategy (dp | legacy) for THIS preprocess
              classification: cls,
              scar_range: cls === "scar" ? s.scarRange : null,
              max_iterations: maxPasses,
              concurrency: conc,
              // Persist the resolved group identity so a rename OR a cross-group move reaches consensus/export.
              ...(sendIdentity ? { patient: g!.patient.trim(), eye: g!.eye.trim() } : {}),
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
      // BUT if the user opened a finished scan mid-batch to correct it (#1), don't yank them away from it.
      if (lastOkCase && !manualPreviewRef.current) { setActiveId(lastOkCase); setCaseId(lastOkCase); await openCase(); initTabs(false); }
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
    // REFUSED, not narrowed. Preprocess and zip are per-scan idempotent, so a smaller set is just a
    // smaller job — but consensus is a scientific AGGREGATE over a subgroup's replicates, and one
    // built from 2 of 4 silently produces a wrong reproducibility number that persists into the
    // artifact with no trace of the omission. The button is disabled while filtering; this is the guard.
    if (filtering) {
      setStep("Clear the filter first — consensus must see every replicate in a subgroup.");
      return;
    }
    const work = groups
      .flatMap((g) => {
        const done = scans.filter((s) => s.groupId === g.id && s.selected && s.caseId && s.status === "done" && visibleIds.has(s.id));
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
    manualPreviewRef.current = false;   // #20: don't yank the view if the user opens a scan mid-run
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
      if (lastResult && !manualPreviewRef.current) {
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

  // Every batch counter below is VISIBLE ∧ selected, so the number printed on each button equals the
  // number of ticked rows the user can literally count on screen. That equality IS the safety story.
  const nDone = scans.filter((s) => s.status === "done" && s.selected && visibleIds.has(s.id)).length;
  const nToRun = scans.filter((s) => s.selected && s.caseId && s.status !== "error" && s.status !== "done" && visibleIds.has(s.id)).length;
  // Most refinement passes among the selected, done scans — bounds the "download which pass" selector.
  const maxScanPasses = Math.max(1, ...scans.filter((s) => s.status === "done" && s.selected && visibleIds.has(s.id)).map((s) => s.passes ?? 1));
  // Tagging is the point of this panel: don't let a scan in an untagged group preprocess
  // (it would be committed with no Scar/Control label, silently producing unlabeled data).
  // Any selected scan still missing a scar/control tag → show the (optional) hint.
  const hasUntaggedRunnable = scans.some((s) => !s.condition && s.selected && s.caseId && s.status !== "error" && visibleIds.has(s.id));
  // Filter bookkeeping. Per-option counts are over the WHOLE corpus and ignore both the search box
  // and each other — they describe the corpus, not the result; nVisible is the only true total.
  const nVisible = visibleIds.size;
  const nHiddenSelected = scans.filter((s) => s.selected && !visibleIds.has(s.id)).length;
  // Computed UNCONDITIONALLY (not gated on `filtering`): it gates the option-disabling below, which is
  // evaluated the moment the user opens the dropdown — i.e. before any filter is active.
  const nNoLife = scans.filter((s) => !s.life).length;
  const filterCounts = Object.fromEntries(FILTER_OPTIONS.map((o) => [o.key, scans.filter(o.test).length])) as Record<string, number>;
  // A (0) is only a FINDING once every scan's lifecycle data has actually landed. While hydration is in
  // flight — or if it failed and rows still have no `life` — a life-derived count of 0 means "unknown",
  // so those options must stay selectable and the counts must be marked provisional. Tag/status options
  // (scar/control/untagged/failed/raw) read local state and are correct immediately, but one honest
  // rule for the whole list beats a per-option split the user would have to know about.
  const countsProvisional = lifeLoading || nNoLife > 0;
  // collapsed never evicts stale group ids (the prune effect and merge/move leave them behind), so
  // the old `collapsed.size >= groups.length` test could lie. Membership-based, over ALL groups.
  const allCollapsed = groups.length > 0 && groups.every((g) => collapsed.has(g.id));

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
        <Button variant="text" size="small" color="error" onClick={() => setWipeConfirmOpen(true)} disabled={busy}
          sx={{ alignSelf: "flex-start", textTransform: "none", minWidth: 0, p: 0.25, fontSize: 11 }}>
          🗑 Wipe all saved cases ({casesCount})
        </Button>
      )}

      {/* #12: confirmatory modal for the destructive wipe (replaces window.confirm, which the WebKitGTK
          webview can silently swallow — a stray click would otherwise delete everything with no prompt). */}
      <Dialog open={wipeConfirmOpen} onClose={() => setWipeConfirmOpen(false)} maxWidth="xs">
        <DialogTitle sx={{ color: "var(--c-red)", fontSize: 16 }}>Delete ALL saved cases?</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ fontSize: 13 }}>
            This permanently removes every corrected volume, segmentation, label and preview on disk for
            all <b>{casesCount ?? 0}</b> case(s). It <b>cannot be undone</b>. Re-uploads will start fresh.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setWipeConfirmOpen(false)} size="small" sx={{ textTransform: "none" }}>
            Cancel
          </Button>
          <Button onClick={doWipe} size="small" color="error" variant="contained" sx={{ textTransform: "none" }}>
            Delete all {casesCount ?? 0} cases
          </Button>
        </DialogActions>
      </Dialog>

      {scans.length > 0 && !loaded && (
        <Button variant="contained" size="small" onClick={load} disabled={busy || scans.length < 1}>
          Upload {scans.length} file{scans.length === 1 ? "" : "s"}
        </Button>
      )}

      {loaded && groups.length > 0 && (
        <div className="flex flex-col gap-2">
          {/* Case-type filter. VIEW-ONLY: it never writes a manifest, never fetches, and never touches
              a scan's checkbox — but it IS the batch SCOPE (batch = visible ∧ selected), so nothing
              hidden can be preprocessed or exported. Deliberately NOT disabled while `busy`, unlike
              every other control in this panel: filtering is pure view state, and a ~20-min preprocess
              run is exactly when browsing the list is wanted (runPreprocess snapshots its work set at
              click time). Deliberately NOT inside a <Collapse> either — a filter that can hide itself
              while still narrowing the buttons is the same footgun as a persisted one. */}
          <div className="text-[11px] uppercase tracking-wide flex items-center gap-1"
            title="Groups — one per patient/eye · tag · scrub · preprocess"
            style={{ color: "var(--c-text-dim)" }}>
            <select value={filter} onChange={(e) => setFilter(e.target.value as ScanFilter)}
              style={{ fontSize: 10, flex: 1, minWidth: 0, color: "var(--c-text-dim)", background: "var(--c-surface)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "1px 4px" }}>
              <option value="all">Show: all scans ({scans.length})</option>
              {FILTER_GROUPS.map((grp) => (
                <optgroup key={grp} label={grp}>
                  {/* A zero-count option stays VISIBLE but disabled — "⚠ Difficult (0)" is a finding
                      (the sweep found none), and an option list whose membership shifts under the
                      cursor is unlearnable. The active option is never disabled. NOR is anything
                      disabled while the counts are provisional: straight after a folder load the
                      rows have no `life` yet, so a 0 there means "not loaded", not "none exist" —
                      greying out ⬚ Surface crop at that moment would hide the very scans it finds. */}
                  {FILTER_OPTIONS.filter((o) => o.group === grp).map((o) => (
                    <option key={o.key} value={o.key} disabled={filterCounts[o.key] === 0 && filter !== o.key && !countsProvisional}>
                      {o.label} ({filterCounts[o.key]}{countsProvisional ? "?" : ""})
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
            {/* While filtering, collapse-all is inert (groups render expanded below), so the slot
                offers "clear" instead — never show a control that silently does nothing. */}
            <button
              onClick={() => (filtering ? clearFilter() : setCollapsed(allCollapsed ? new Set() : new Set(groups.map((g) => g.id))))}
              style={{ background: "none", border: "none", color: filtering ? "var(--c-accent)" : "var(--c-text-dim)", cursor: "pointer", textTransform: "none", fontSize: 10, padding: 0, flex: "none" }}>
              {filtering ? "clear" : allCollapsed ? "expand all" : "collapse all"}
            </button>
          </div>
          {/* Search ANDs with the category above. */}
          <div className="flex items-center gap-1">
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="find scan / patient / subgroup…"
              style={{ fontSize: 10, flex: 1, minWidth: 0, color: "var(--c-text)", background: "var(--c-surface2)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "1px 4px" }} />
            {filtering && (
              <span className="text-[10px]" style={{ flex: "none", color: nVisible ? "var(--c-text-dim)" : "var(--c-amber, #d9a441)" }}>
                {nVisible}/{scans.length}
              </span>
            )}
          </div>
          {filtering && nHiddenSelected > 0 && (
            <span className="text-[10px]" style={{ color: "var(--c-amber, #d9a441)" }}>
              {nHiddenSelected} selected scan{nHiddenSelected === 1 ? "" : "s"} hidden — batch actions skip them
            </span>
          )}
          {/* NOT gated on `filtering`: this is the note that explains a provisional "(0?)" in the dropdown,
              so it has to be on screen while the user is looking AT the dropdown — i.e. before they have
              managed to pick a filter. Straight after a folder load the counts are all 0 until hydrateLife
              lands; if it failed, the counts stay wrong, so offer a retry rather than a dead end. */}
          {countsProvisional && (
            <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>
              {lifeLoading
                ? "Loading lifecycle data — case-type counts are provisional…"
                : `${nNoLife} scan${nNoLife === 1 ? "" : "s"} have no lifecycle data yet — counts marked “?” may be low. `}
              {!lifeLoading && (
                <button onClick={() => void hydrateLife()}
                  style={{ background: "none", border: "none", color: "var(--c-accent)", cursor: "pointer", padding: 0, fontSize: 10, textTransform: "none" }}>
                  refresh
                </button>
              )}
            </span>
          )}
          {filtering && nVisible === 0 ? (
            <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>
              No scans match this filter —{" "}
              <button onClick={clearFilter}
                style={{ background: "none", border: "none", color: "var(--c-accent)", cursor: "pointer", padding: 0, fontSize: 10, textTransform: "none" }}>
                clear
              </button>
            </span>
          ) : groups.map((g) => {
            const allGroupScans = scans.filter((s) => s.groupId === g.id);
            const groupScans = filtering ? allGroupScans.filter((s) => visibleIds.has(s.id)) : allGroupScans;
            // Never render an empty group shell: the header alone is a full editable card (patient/eye
            // fields + merge dropdown), so a group with no matching rows must not appear at all.
            if (!groupScans.length) return null;
            const nHiddenInGroup = allGroupScans.length - groupScans.length;
            // While filtering, every matching group renders EXPANDED — a collapsed group reading
            // "2 of 7 scans" with no rows beneath it is indistinguishable from a bug, and requiring a
            // click per group to reveal a filter's own matches defeats the point. `collapsed` is never
            // mutated here, so the user's manual collapse state returns intact when the filter clears.
            const isCollapsed = !filtering && collapsed.has(g.id);
            const nDoneInGroup = groupScans.filter((s) => s.status === "done").length;
            return (
              <div key={g.id} className="rounded" style={{ border: "1px solid var(--c-border)" }}>
                {/* Group header: editable patient/eye + Scar/Control tag for the whole group. */}
                <div className="flex flex-col gap-1 px-1.5 py-1.5" style={{ background: "var(--c-surface2)" }}>
                  <div className="flex items-center gap-1">
                    <button onClick={() => toggleCollapse(g.id)} disabled={filtering}
                      title={filtering ? "collapse is off while a filter is active" : isCollapsed ? "expand" : "collapse"}
                      style={{ background: "none", border: "none", color: "var(--c-text-dim)", cursor: filtering ? "default" : "pointer", padding: 0, width: 16, flex: "none", fontSize: 11, opacity: filtering ? 0.4 : 1 }}>
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
                        toggle below). Shows the common tag, or blank when the group's scans disagree.

                        REFUSED (not narrowed) while the filter hides part of this group — the same rule the
                        consensus button uses below, and for the same reason. This is the ONE control in the
                        panel that PERSISTS to disk for scans that are off screen: it fires a classification
                        POST per scan and nulls scar_range on anything not tagged "scar", fire-and-forget,
                        with no undo and no on-screen trace. Narrowing it to the visible rows was the other
                        option, but then a button labelled "all" means "all 5" or "just this 1" depending on
                        view state — for a destructive, non-idempotent write that is worse than refusing.
                        Nothing is lost: the per-scan toggles on every visible row stay fully available. */}
                    <span className="text-[10px]" style={{ color: "var(--c-text-dim)", opacity: nHiddenInGroup > 0 ? 0.5 : 1 }}
                      title={nHiddenInGroup > 0
                        ? `Disabled: ${nHiddenInGroup} of this group's ${allGroupScans.length} scans are hidden by the filter, and "all" would overwrite their tags off screen. Clear the filter to tag the whole group, or use the per-scan toggles below.`
                        : "Tags every scan in this group at once"}>
                      all{nHiddenInGroup > 0 ? ` ${allGroupScans.length}` : ""}:
                    </span>
                    <ToggleButtonGroup size="small" exclusive value={groupCondition(g.id)} disabled={nHiddenInGroup > 0}
                      onChange={(_, v) => setGroupCondition(g.id, (v as Cls) || undefined)}>
                      <ToggleButton value="scar" sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Scar</ToggleButton>
                      <ToggleButton value="control" sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Control (no scar)</ToggleButton>
                    </ToggleButtonGroup>
                    {/* Visible reason, not just a hover title — a greyed-out control with no on-screen
                        explanation reads as a bug. Only rendered for the groups actually affected. */}
                    {nHiddenInGroup > 0 && (
                      <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>
                        group tag off while filtered
                      </span>
                    )}
                    <span className="text-[10px]" style={{ marginLeft: "auto", color: nDoneInGroup ? "var(--c-green)" : "var(--c-text-dim)" }}>
                      {groupScans.length}{nHiddenInGroup > 0 ? ` of ${allGroupScans.length}` : ""} scan{(nHiddenInGroup > 0 ? allGroupScans.length : groupScans.length) === 1 ? "" : "s"}{nDoneInGroup ? ` · ${nDoneInGroup} done ✓` : ""}
                    </span>
                    {nDoneInGroup > 0 && (
                      <button title={nHiddenInGroup > 0
                        ? `Download the ${nDoneInGroup} preprocessed scan(s) VISIBLE under the current filter as a .zip — ${nHiddenInGroup} scan(s) in this group are hidden and excluded`
                        : `Download all ${nDoneInGroup} preprocessed scan(s) in this group as a .zip (a folder for manual segmentation)`}
                        onClick={() => downloadPreprocessedZip(groupScans.filter((s) => s.status === "done" && s.caseId).map((s) => s.caseId!), downloadPass)}
                        style={{ background: "none", border: "none", color: "var(--c-accent)", cursor: "pointer", padding: 0, fontSize: 10, flex: "none" }}>
                        ⬇ zip
                      </button>
                    )}
                  </div>
                  {(() => {
                    const subs = allGroupScans.reduce<Record<string, number>>((m, s) => { const k = subSlug(s.subgroup); m[k] = (m[k] || 0) + 1; return m; }, {});
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
                    // #1: a FINISHED (done) scan stays clickable WHILE a batch runs, so the user can open it
                    // for manual correction; still-preprocessing/queued scans stay locked (no half-written volume).
                    const clickable = !!s.caseId && s.status !== "error" && (!busy || s.status === "done");
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
                          {/* Surface-crop badge: AUTO-detected (blue ⬚) or human-confirmed (⬚✓). A manual "not
                              surface-crop" override (surface_crop_manual === false) hides it. Lets the user find +
                              verify every auto-detected clipped-cornea scan. */}
                          {(() => {
                            const scM = s.life?.surface_crop_manual;   // tri-state: true / false / null
                            return scOf(s.life) ? (
                              <span title={`Surface-crop / clipped cornea${scM != null ? " · reviewed" : " · auto-detected (unreviewed)"}`}
                                style={{ color: "#38bdf8", fontWeight: 700, flex: "none" }}>⬚{scM != null ? "✓" : ""}</span>
                            ) : null;
                          })()}
                          {Boolean(s.life?.difficult_scan) && (
                            <span title="Marked DIFFICULT — excluded from training" style={{ color: "#ef4444", fontWeight: 700, flex: "none" }}>⚠</span>
                          )}
                          {done && s.caseId && (
                            <button title="Download this preprocessed scan (.nii.gz) for manual segmentation"
                              onClick={(e) => { e.stopPropagation(); void downloadPreprocessed(s.caseId!, s.filename, downloadPass && downloadPass <= (s.passes ?? 1) ? downloadPass : null); }}
                              style={{ background: "none", border: "none", color: "var(--c-accent)", cursor: "pointer", padding: 0, fontSize: 13, lineHeight: 1, flex: "none", marginTop: 2 }}>
                              ⬇
                            </button>
                          )}
                        </div>

                        {/* Review flags: ONE chip per flag (they used to be concatenated into one unreadable
                            run of letters), named + coloured from api/reviewFlags.ts, with the full meaning in
                            the tooltip. They get their OWN line (indented under the name, like the Move-to
                            select) rather than a box inside the name row: the row's only shrinkable item is the
                            filename, so ANY fixed-width box there is paid for by the name — 2-3 chips collapsed
                            it to 0px, which with overflowWrap:"anywhere" wrapped it one character per line
                            (row 49px → 518px tall) and still overflowed, pushing the ⬇ button past the
                            sidebar's overflow-x-hidden edge where it could not be clicked. On its own line the
                            chips wrap freely and cost ~16px only when a scan is actually flagged. */}
                        {(() => {
                          const flags = reviewFlagsOf(s.life);
                          return flags.length === 0 ? null : (
                            <div className="ml-6 mt-0.5 flex flex-wrap items-center" style={{ gap: 2 }}>
                              {flags.map((f) => {
                                const meta = reviewFlagMeta(f);
                                return (
                                  <span key={f} title={`⚑ ${meta.label} — ${meta.description}`}
                                    style={{ color: meta.color, background: `${meta.color}1f`, border: `1px solid ${meta.color}66`,
                                      borderRadius: 3, padding: "0 3px", fontSize: 9, lineHeight: "14px", fontWeight: 700,
                                      whiteSpace: "nowrap", maxWidth: "100%", overflow: "hidden", textOverflow: "ellipsis" }}>
                                    {meta.short}
                                  </span>
                                );
                              })}
                            </div>
                          );
                        })()}

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
                            <input list={`subs-${g.id}`} value={s.subgroup} disabled={busy && s.status !== "done"} placeholder="1"
                              onChange={(e) => patchScan(s.id, { subgroup: e.target.value })}
                              onBlur={(e) => persistSubgroup(s.caseId, e.target.value)}
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
                              <ToggleButton value="scar" disabled={busy && s.status !== "done"} sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>Scar</ToggleButton>
                              <ToggleButton value="control" disabled={busy && s.status !== "done"} sx={{ py: 0, px: 1, fontSize: 10, textTransform: "none" }}>No scar</ToggleButton>
                            </ToggleButtonGroup>
                          </div>
                        )}

                        {/* Scar frame-range (per scan) — only when THIS scan is tagged Scar. The range is
                            scar-detection metadata (not geometry), so changing it never invalidates the
                            corrected volume; persisted on release. */}
                        {s.condition === "scar" && s.status !== "error" && (
                          <div className="ml-6 mt-1 pr-2">
                            <div className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>scar frames {s.scarRange[0]}–{s.scarRange[1]}</div>
                            <Slider size="small" min={1} max={s.nFrames} value={s.scarRange} disabled={busy && s.status !== "done"}
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

          {/* Detector is always the native DP path WITH the legacy cross-check (scar-guard) — preprocessing uses
              BOTH, so the old DP|Legacy selection toggle was removed (detector stays "dp", which runs the guard). */}

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
          {filtering && nHiddenSelected > 0 && (
            <span className="text-[10px] px-1" style={{ color: "var(--c-amber, #d9a441)" }}>
              Filter active — {nHiddenSelected} selected scan{nHiddenSelected === 1 ? "" : "s"} hidden and excluded.{" "}
              <button onClick={clearFilter}
                style={{ background: "none", border: "none", color: "var(--c-accent)", cursor: "pointer", padding: 0, fontSize: 10, textTransform: "none" }}>
                clear filter
              </button>
            </span>
          )}
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
              onClick={() => downloadPreprocessedZip(scans.filter((s) => s.status === "done" && s.selected && s.caseId && visibleIds.has(s.id)).map((s) => s.caseId!), downloadPass)}
              title="Download the selected corrected scans as one .zip (a folder of <case_id>.nii.gz) — open it in the annotator app for manual ground-truth segmentation">
              ⬇ Download preprocessed (.zip){downloadPass ? ` — pass ${downloadPass}` : ""} ({nDone})
            </Button>
          )}
          {nDone > 0 && (
            <>
              {/* Consensus is REFUSED under a filter, not narrowed: it is a scientific aggregate over a
                  subgroup's replicates, so a partial set silently yields a wrong reproducibility number
                  baked into the persisted artifact. Preprocess/zip are per-scan idempotent — consensus is not. */}
              <Button variant="contained" color="secondary" size="small" onClick={runSamConsensus} disabled={busy || filtering}>
                {nDone > 1 ? `Run SAM2 + consensus (${nDone})` : "Run SAM2"}
              </Button>
              {filtering && (
                <span className="text-[10px] px-1" style={{ color: "var(--c-text-dim)" }}>
                  Clear the filter to run consensus — it must see every replicate in a subgroup.
                </span>
              )}
            </>
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
