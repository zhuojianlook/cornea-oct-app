import { create } from "zustand";
import type { Update } from "@tauri-apps/plugin-updater";
import type { Lang } from "../i18n";
import * as nv from "../niivue/nvController";
import * as io from "../tauri/io";
import { checkForUpdate, installAndRelaunch } from "../tauri/updater";

export type Pen = 0 | 1 | 2 | 3; // 0 erase, 1 cornea, 2 scar, 3 background seed (Smart fill only)
export const APP_VERSION = "0.1.39";

// #4: a BLINDED queue entry. The annotator sees only `name` ("Scan B · rep 1"); the real file is hidden
// (`stem`/`path`) unless an admin unlocks. Each real scan yields `replicates` entries so the same user
// annotates it multiple times (intra-observer); different users annotate the same scans (inter-observer).
export interface BlindEntry { name: string; path: string; stem: string; blindLabel: string; replicate: number; }
const ADMIN_PASSWORD = "OCTAPP";
// A→Z, AA→… blind label for the i-th scan (real name stays hidden).
function blindLetter(i: number): string { let s = ""; i++; while (i > 0) { s = String.fromCharCode(65 + (i - 1) % 26) + s; i = Math.floor((i - 1) / 26); } return s; }
// Build the blinded, replicate-expanded queue. Interleaved by replicate ROUND (all scans' rep 1, then
// all scans' rep 2, …) so a user doesn't annotate the same scan back-to-back — better intra-observer.
function buildBlindQueue(vols: io.VolumeEntry[], replicates: number): BlindEntry[] {
  const reps = Math.max(1, Math.min(4, Math.round(replicates || 2)));
  const out: BlindEntry[] = [];
  for (let r = 1; r <= reps; r++)
    for (let i = 0; i < vols.length; i++) {
      const v = vols[i]; const stem = v.name.replace(/\.nii(\.gz)?$/i, "");
      out.push({ name: `Scan ${blindLetter(i)} · rep ${r}`, path: v.path, stem, blindLabel: `Scan ${blindLetter(i)}`, replicate: r });
    }
  return out;
}
const entryKey = (e: { stem: string; replicate: number }): string => `${e.stem}__rep${e.replicate}`;
// Autosave/in-session-cache key MUST include the replicate — two replicates of one scan share a path but
// are independent annotations, so they need separate autosaves/caches (#4 + #5).
const aKey = (e: BlindEntry | null): string | null => (e ? `${e.path}#rep${e.replicate}` : null);
const sessionId = new Date().toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, 19);

// ── Annotation persistence (#5) — never lose work on volume swap OR app close/restart ─────────────
type AnnotSnap = { seed: Uint8Array; committed: Uint8Array; preview: Uint8Array; previewing: boolean };
// In-memory, keyed by volume path → LOSSLESS restore (incl. an unconfirmed smart fill) when swapping
// volumes within a session. The disk autosave (io.writeAutosave) backs this across app restarts.
const annotCache = new Map<string, AnnotSnap>();
// Key by user|path (mirrors the disk autosave key) so switching users mid-session never restores the
// previous user's paint onto the new user's volume.
const ckey = (user: string | null, p: string): string => `${user ?? ""}|${p}`;
let autosaveTimer: ReturnType<typeof setTimeout> | null = null;
let wandTimer: ReturnType<typeof setTimeout> | null = null; // debounce live wand-preview recompute
// Serialize snapshots: exportLabelmapBytes() temporarily mutates the shared drawBitmap, so two exports
// (or an export racing the next volume's load) must never overlap. openVolume awaits this before loading.
let snapInFlight: Promise<void> = Promise.resolve();

/** Capture the current volume's drawing into the in-memory cache + to disk, so it survives a swap and a
    restart. Serialized via snapInFlight; awaiting the returned promise drains any prior in-flight export. */
function snapshotVolume(user: string | null, volPath: string | null): Promise<void> {
  const run = snapInFlight.then(async () => {
    if (!volPath) return;
    const st = nv.getAnnotationState();
    if (!st) return;
    if (nv.hasPaint()) {
      annotCache.set(ckey(user, volPath), st);
      if (user) {
        try { const bytes = await nv.exportAutosaveBytes(); if (bytes) await io.writeAutosave(user, volPath, bytes); } catch { /* best-effort */ }
      }
    } else {
      annotCache.delete(ckey(user, volPath)); // cleared the drawing → don't restore stale paint
      if (user) { try { await io.removeAutosave(user, volPath); } catch { /* */ } }
    }
  });
  snapInFlight = run.catch(() => {}); // keep the chain alive even if one snapshot throws
  return run;
}

interface State {
  // identity / session
  users: string[];
  activeUser: string | null;
  sessionId: string;
  outputDir: string | null;
  // volumes
  folder: string | null;
  volumes: io.VolumeEntry[];       // the REAL files in the folder (names hidden from the UI)
  blindEntries: BlindEntry[];      // #4: the BLINDED queue actually shown — each scan ×N replicates
  activeVolume: BlindEntry | null; // the blinded entry being annotated (carries real stem + replicate)
  replicates: number;              // #4: how many repeats per scan each user does (default 2; admin 2–4)
  adminUnlocked: boolean;          // #4: OCTAPP entered → real scan names + replicate count revealed/editable
  loaded: boolean;
  annotated: Set<string>;          // `${stem}__rep${n}` keys this user already saved (this output dir)
  volumeStartMs: number;
  dims: [number, number, number] | null;   // [nx, ny, nz] of the loaded volume
  vox: [number, number, number];           // current crosshair slice indices [x, y, z]
  // pen / tool
  penLabel: Pen;
  penSize: number;
  penFilled: boolean;
  tool: "paint" | "navigate" | "wand";   // active interaction tool
  drawOpacity: number;
  showAnnotations: boolean;  // toggle the user's annotations (2-D overlay + niivue draw/3-D) on/off
  // intensity wand (live preview → Confirm)
  wandThreshold: number;     // threshold mode: brightness cutoff, 0..1 of the intensity range
  wandTolerance: number;     // tolerance mode: ± band around the clicked voxel, 0..1 of the range
  wandMode: "threshold" | "tolerance";
  wandScope: "2d" | "3d";    // flood the clicked slice only, or the whole volume
  wandTarget: 1 | 2;         // paint cornea (1) or scar (2)
  wandSeed: [number, number, number] | null; // current wand seed voxel (drives the live preview)
  wandSeedAxis: number | null;               // the seed pane's through-axis (for 2-D recompute)
  cursorIntensity01: number | null;          // cursor intensity as 0..1 of the range (indicator)
  // ui
  busy: boolean;
  status: string;
  lang: Lang;
  smartPct: number | null;   // smart-fill progress 0–100 while computing, else null
  canConfirm: boolean;       // a smart-fill preview is active and can be confirmed
  canUndo: boolean;
  canRedo: boolean;
  corneaVox: number;         // live voxel counts
  scarVox: number;
  cursorIntensity: number | null; // raw intensity under the cursor (for the wand)
  brightness: number;        // display window brightness, −1..1 (#3)
  contrast: number;          // display window contrast, −1..1 (#3)
  locked: number[];          // labels protected from brush/erase/smart-fill (#4)
  confirmOverwrite: boolean; // overwrite-confirmation dialog visible (#1)
  confirmClear: boolean;     // clear-confirmation dialog visible
  pendingVolume: string | null; // last volume to auto-reopen after login (#2)
  // self-update
  update: Update | null;
  updateBusy: boolean;
  updatePct: number | null;
  updateMsg: string;

  init: () => Promise<void>;
  setLang: (l: Lang) => void;
  checkUpdates: (manual: boolean) => Promise<void>;
  installUpdate: () => Promise<void>;
  dismissUpdate: () => void;
  addUser: (name: string) => Promise<void>;
  deleteUser: (name: string) => Promise<void>;
  selectUser: (name: string) => Promise<void>;
  pickFolder: () => Promise<void>;
  openVolume: (v: BlindEntry) => Promise<void>;
  nextUnannotated: () => void;
  unlockAdmin: (pw: string) => boolean;     // #4: reveal real names + replicate control (password OCTAPP)
  lockAdmin: () => void;
  setReplicates: (n: number) => void;       // #4: admin-only — repeats per scan (2–4); rebuilds the queue
  loadSegmentation: () => Promise<void>;
  chooseOutputDir: () => Promise<void>;
  setPenLabel: (p: Pen) => void;
  setPenSize: (n: number) => void;
  setPenFilled: (f: boolean) => void;
  setTool: (t: "paint" | "navigate" | "wand") => void;
  setWandThreshold: (t: number) => void;
  setWandTolerance: (t: number) => void;
  setWandMode: (m: "threshold" | "tolerance") => void;
  setWandScope: (s: "2d" | "3d") => void;
  setWandTarget: (t: 1 | 2) => void;
  wandRecompute: () => void;
  wandAt: (x: number, y: number, z: number, throughAxis: number | null) => void;
  setCursorIntensity: (x: number, y: number, z: number) => void;
  setDrawOpacity: (o: number) => void;
  setShowAnnotations: (v: boolean) => void;
  setSliceAxis: (axis: 0 | 1 | 2, s: number) => void;
  syncVox: () => void;
  refreshStats: () => void;
  smartFill: () => Promise<void>;
  confirmFill: () => void;
  undo: () => void;
  redo: () => void;
  requestClear: () => void;
  cancelClear: () => void;
  clearDrawing: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  resetView: () => void;
  save: (force?: boolean) => Promise<void>;
  // Manage a saved ground truth from its ✓ badge: re-open to edit is openVolume; these delete it / download
  // its labelmap; exportAllGt copies every saved labelmap to a chosen folder.
  deleteGt: (entry: BlindEntry) => Promise<void>;
  downloadGt: (entry: BlindEntry) => Promise<void>;
  exportAllGt: () => Promise<void>;
  cancelOverwrite: () => void;
  setBrightness: (b: number) => void;
  setContrast: (c: number) => void;
  resetWindow: () => void;
  toggleLock: (label: number) => void;
  persistConfig: () => void;
  resumePending: () => void;
  autosaveDraw: () => void;       // #5: debounced snapshot of the current drawing (cache + disk)
  flushAutosave: () => Promise<void>; // #5: synchronous flush (app close / before unload)
}

export const useStore = create<State>((set, get) => ({
  users: [],
  activeUser: null,
  sessionId,
  outputDir: null,
  folder: null,
  volumes: [],
  blindEntries: [],
  activeVolume: null,
  replicates: 2,
  adminUnlocked: false,
  loaded: false,
  annotated: new Set(),
  volumeStartMs: 0,
  dims: null,
  vox: [0, 0, 0],
  penLabel: 2,
  penSize: 8,
  penFilled: false,
  tool: "paint",
  showAnnotations: true,
  drawOpacity: 0.85,  // higher default so the SEMI-TRANSPARENT cornea (colormap alpha 130) is clearly
  wandThreshold: 0.55,
  wandTolerance: 0.08,
  wandMode: "threshold",
  wandScope: "3d",
  wandTarget: 2,
  wandSeed: null,
  wandSeedAxis: null,
  cursorIntensity01: null,
  busy: false,
  status: "Select or add a user to begin.",
  lang: "en",
  smartPct: null,
  canConfirm: false,
  canUndo: false,
  canRedo: false,
  corneaVox: 0,
  scarVox: 0,
  cursorIntensity: null,
  brightness: 0,
  contrast: 0,
  locked: [],
  confirmOverwrite: false,
  confirmClear: false,
  pendingVolume: null,
  update: null,
  updateBusy: false,
  updatePct: null,
  updateMsg: "",

  init: async () => {
    const cfg = await io.loadConfig();
    const reps = Math.max(1, Math.min(4, Math.round(cfg.replicates ?? 2)));
    set({ users: cfg.users, outputDir: cfg.outputDir, folder: cfg.lastFolder, lang: cfg.lang, replicates: reps, pendingVolume: cfg.lastVolume ?? null });
    // A2: auto-restore the last folder's volume list so the user need not re-pick the folder. The last
    // volume is reopened (resumePending) once a user logs in and the canvas has attached.
    if (cfg.lastFolder) {
      try {
        const volumes = await io.listNifti(cfg.lastFolder);
        set({ volumes, blindEntries: buildBlindQueue(volumes, reps) });   // #4 blinded queue
      } catch { set({ folder: null, pendingVolume: null }); } // folder moved/deleted
    }
  },

  // Persist the full config (users, output dir, last folder + volume, language, replicate count) — best-effort.
  persistConfig: () => {
    const s = get();
    void io.saveConfig({ users: s.users, outputDir: s.outputDir, lastFolder: s.folder, lang: s.lang, replicates: s.replicates, lastVolume: aKey(s.activeVolume) });
  },

  // #5: debounced autosave — coalesce rapid edits, then snapshot the current drawing (cache + disk).
  autosaveDraw: () => {
    if (autosaveTimer) clearTimeout(autosaveTimer);
    autosaveTimer = setTimeout(() => {
      autosaveTimer = null;
      void snapshotVolume(get().activeUser, aKey(get().activeVolume));
    }, 1200);
  },
  // #5: flush immediately (app close / beforeunload / explicit save).
  flushAutosave: async () => {
    if (autosaveTimer) { clearTimeout(autosaveTimer); autosaveTimer = null; }
    await snapshotVolume(get().activeUser, aKey(get().activeVolume));
  },

  // After login + canvas attach, reopen the blinded entry from the previous session (#2/#4).
  resumePending: () => {
    const s = get();
    if (!s.activeUser || s.activeVolume || s.loaded || !s.pendingVolume) return;
    const e = s.blindEntries.find((x) => aKey(x) === s.pendingVolume);
    set({ pendingVolume: null });
    if (e) void get().openVolume(e);
  },

  setLang: (l) => { set({ lang: l }); get().persistConfig(); },

  // Check the GitHub release feed. `manual` surfaces "up to date"/error feedback; the silent
  // launch check only speaks up when an update actually exists (shows the banner).
  checkUpdates: async (manual) => {
    if (get().updateBusy) return;
    set({ updateBusy: true, updateMsg: manual ? "Checking for updates…" : "" });
    try {
      const u = await checkForUpdate();
      if (u) set({ update: u, updateMsg: "" });
      else set({ update: null, updateMsg: manual ? `You're on the latest version (v${APP_VERSION}).` : "" });
    } catch (e) {
      set({ updateMsg: manual ? `Update check failed: ${e instanceof Error ? e.message : String(e)}` : "" });
    } finally {
      set({ updateBusy: false });
      if (manual) setTimeout(() => { if (get().updateMsg.startsWith("You're on")) set({ updateMsg: "" }); }, 6000);
    }
  },
  installUpdate: async () => {
    const u = get().update;
    if (!u || get().updateBusy) return;
    set({ updateBusy: true, updatePct: null, updateMsg: "Downloading update…" });
    try {
      await installAndRelaunch(u, (p) => set({ updatePct: p }));
      // relaunch() replaces the process on success; nothing runs after.
    } catch (e) {
      set({ updateBusy: false, updatePct: null, updateMsg: `Update failed: ${e instanceof Error ? e.message : String(e)}` });
    }
  },
  dismissUpdate: () => set({ update: null, updateMsg: "" }),

  addUser: async (name) => {
    const u = name.trim();
    if (!u) return;
    const users = Array.from(new Set([...get().users, u])).sort();
    set({ users });
    get().persistConfig();
    await get().selectUser(u);
  },

  deleteUser: async (name) => {
    const users = get().users.filter((u) => u !== name);
    const activeUser = get().activeUser === name ? null : get().activeUser;
    set({ users, activeUser });
    get().persistConfig();
  },

  selectUser: async (name) => {
    set({ activeUser: name, status: `Annotating as “${name}”. Pick a folder of NIfTI volumes.` });
    const done = await io.annotatedStems(get().outputDir, name);
    set({ annotated: done });
  },

  pickFolder: async () => {
    const folder = await io.pickFolder();
    if (!folder) return;
    set({ busy: true });
    try {
      const volumes = await io.listNifti(folder);
      const blindEntries = buildBlindQueue(volumes, get().replicates);   // #4: blind + replicate-expand
      set({ folder, volumes, blindEntries, status: `${blindEntries.length} blinded entries (${volumes.length} scan(s) × ${get().replicates}). Select one to annotate.` });
      get().persistConfig();
    } catch (e) {
      set({ status: `Could not list folder: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      set({ busy: false });
    }
  },

  openVolume: async (v) => {
    set({ busy: true, status: `Loading ${v.name}…`, loaded: false });
    try {
      // #5: capture the OUTGOING volume's drawing first so swapping never loses it (incl. an
      // unconfirmed smart fill). Cancel any pending debounced autosave — we flush synchronously here.
      if (autosaveTimer) { clearTimeout(autosaveTimer); autosaveTimer = null; }
      await snapshotVolume(get().activeUser, aKey(get().activeVolume));
      const bytes = await io.readVolume(v.path);                       // v.path = the REAL file
      // niivue picks its decoder from the NAME's extension (regex on the last '.'), so it MUST end in
      // .nii/.nii.gz. v.name is the BLINDED label ("Scan A · rep 1") with no '.' → niivue crashed
      // (undefined.toUpperCase). Feed it a neutral name carrying the real file's extension instead.
      const ext = v.path.match(/\.nii(\.gz)?$/i)?.[0] ?? ".nii.gz";
      const niiName = `volume${ext}`;
      await nv.loadVolumeBytes(bytes, niiName, get().drawOpacity);
      nv.setPenSize(get().penSize);
      nv.setPen(get().penLabel, get().penFilled);
      nv.setSliceListener((vx) => set({ vox: vx })); // keep slice scrollbars synced to scroll-wheel nav
      nv.setLockedLabels(get().locked);              // re-apply label locks for the new volume
      // #5: restore this ENTRY's drawing (keyed by path#rep so the two replicates stay independent) —
      // lossless from the in-memory cache (same session) or the disk autosave (restored as committed).
      let restoredPreviewing = false;
      const key = aKey(v)!;
      const cached = annotCache.get(ckey(get().activeUser, key));
      if (cached && nv.setAnnotationState(cached)) {
        restoredPreviewing = cached.previewing;
      } else {
        try {
          const ab = await io.readAutosave(get().activeUser ?? "", key);
          // restore as the LOSSLESS seed+committed encoding (so Smart fill works on restored strokes);
          // backward-compatible with old 0/1/2 autosaves (decoded as committed).
          if (ab) await nv.restoreAutosaveBytes(ab, `autosave${ext}`); // .nii.gz name (niivue needs the ext)
        } catch { /* no autosave / unreadable — start blank */ }
      }
      set({ activeVolume: v, loaded: true, tool: "paint", volumeStartMs: Date.now(), canConfirm: restoredPreviewing,
            wandSeed: null, wandSeedAxis: null,
            dims: nv.getDims(), vox: nv.currentVox() ?? [0, 0, 0], brightness: 0, contrast: 0,
            status: `Annotating ${v.name}. Paint cornea/scar; Smart fill helps; then Save.` });
      nv.setWindow(0, 0);     // new volume → reset display window + counts
      get().refreshStats();
      get().persistConfig();  // remember this as the last volume (#2)
    } catch (e) {
      set({ status: `Failed to load volume: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      set({ busy: false });
    }
  },

  nextUnannotated: () => {
    const { blindEntries, annotated, activeVolume } = get();
    if (!blindEntries.length) return;
    const startIdx = activeVolume ? blindEntries.findIndex((e) => aKey(e) === aKey(activeVolume)) : -1;
    for (let k = 1; k <= blindEntries.length; k++) {
      const e = blindEntries[(startIdx + k + blindEntries.length) % blindEntries.length];
      if (!annotated.has(entryKey(e))) { void get().openVolume(e); return; }
    }
    set({ status: "All blinded entries in this folder are annotated." });
  },

  // #4: admin controls — OCTAPP reveals real scan names + lets you set the replicate count.
  unlockAdmin: (pw) => { const ok = pw === ADMIN_PASSWORD; if (ok) set({ adminUnlocked: true }); return ok; },
  lockAdmin: () => set({ adminUnlocked: false }),
  setReplicates: (n) => {
    const reps = Math.max(1, Math.min(4, Math.round(n || 2)));
    set({ replicates: reps, blindEntries: buildBlindQueue(get().volumes, reps) });
    get().persistConfig();
  },

  loadSegmentation: async () => {
    if (!get().loaded) { set({ status: "Open a volume first, then load a segmentation to correct." }); return; }
    const path = await io.pickFile("Choose a ground-truth labelmap (.nii.gz) to load");
    if (!path) return;
    set({ busy: true, status: "Loading segmentation…" });
    try {
      const name = await io.stem(path);
      const bytes = await io.readVolume(path);
      const r = await nv.loadSegmentationBytes(bytes, name.endsWith(".nii.gz") ? name : `${name}.nii.gz`);
      if (!r.ok) set({ status: r.reason === "dims-mismatch" ? "That labelmap doesn't match this volume's dimensions." : `Could not load segmentation: ${r.reason}` });
      else { get().refreshStats(); set({ canConfirm: false, status: "Segmentation loaded — edit/correct, then Save." }); }
    } catch (e) {
      set({ status: `Could not load segmentation: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      set({ busy: false });
    }
  },

  chooseOutputDir: async () => {
    const dir = await io.pickFolder("Choose where to save ground-truth output");
    if (!dir) return;
    set({ outputDir: dir });
    get().persistConfig();
    if (get().activeUser) set({ annotated: await io.annotatedStems(dir, get().activeUser!) });
  },

  setPenLabel: (p) => { nv.setPen(p, get().penFilled); set({ penLabel: p }); },
  setPenSize: (n) => { nv.setPenSize(n); set({ penSize: n }); },
  setPenFilled: (f) => { nv.setPen(get().penLabel, f); set({ penFilled: f }); },
  setTool: (t) => { if (t === "paint") nv.lockCrosshair(); nv.setDrawingEnabled(false);
    // #2: left-click never moves the crosshair (niivue leftButton=none); Navigate sets it explicitly
    // (AnnotatorCanvas → setCrosshairAtScreen). So Paint/Wand clicks can't jump the crosshair.
    set(t === "wand" ? { tool: t } : { tool: t, wandSeed: null, wandSeedAxis: null }); },
  // Recompute the live wand preview from the current seed + params (after a param change). Debounced so
  // dragging a slider stays smooth. Clears the seed if a recompute no longer floods anything.
  wandRecompute: () => {
    const s = get();
    if (!s.wandSeed) return;
    if (wandTimer) clearTimeout(wandTimer);
    wandTimer = setTimeout(() => {
      wandTimer = null;
      const st = get();
      if (!st.wandSeed) return;
      const [x, y, z] = st.wandSeed;
      const r = nv.wandPreview(x, y, z, { mode: st.wandMode, threshold01: st.wandThreshold, tolerance01: st.wandTolerance,
        scope: st.wandScope, throughAxis: st.wandSeedAxis, target: st.wandTarget });
      if (r.ok) { get().refreshStats(); nv.forceDrawAll(); set({ canConfirm: true, status: `Wand preview: ${r.count} voxels — adjust, then Confirm (or click a new spot).` }); }
      else { nv.clearPreview(); nv.forceDrawAll(); get().refreshStats(); set({ canConfirm: false }); }
    }, 110);
  },
  setWandThreshold: (t) => { set({ wandThreshold: Math.max(0, Math.min(1, t)) }); if (get().wandMode === "threshold") get().wandRecompute(); },
  setWandTolerance: (t) => { set({ wandTolerance: Math.max(0, Math.min(1, t)) }); if (get().wandMode === "tolerance") get().wandRecompute(); },
  setWandMode: (m) => { set({ wandMode: m }); get().wandRecompute(); },
  setWandScope: (s) => { set({ wandScope: s }); get().wandRecompute(); },
  setWandTarget: (t) => { set({ wandTarget: t }); get().wandRecompute(); },
  // Click → seed the wand and show the live preview immediately (no commit yet — Confirm bakes it).
  wandAt: (x, y, z, throughAxis) => {
    const s = get();
    const r = nv.wandPreview(x, y, z, { mode: s.wandMode, threshold01: s.wandThreshold, tolerance01: s.wandTolerance,
      scope: s.wandScope, throughAxis, target: s.wandTarget });
    if (!r.ok) {
      nv.clearPreview();
      set({ wandSeed: null, wandSeedAxis: null, canConfirm: false,
        status: r.reason === "below-threshold" ? "Wand: nothing here at this threshold — lower it or click a brighter spot."
          : r.reason === "outside-cornea" ? "Wand: click inside the cornea (scar grows within it)." : "Wand: nothing to fill here." });
    } else {
      get().refreshStats();
      nv.forceDrawAll();   // #1/#2: ensure every tile (incl. coronal) shows the preview on WebKitGTK
      set({ wandSeed: [x, y, z], wandSeedAxis: throughAxis, canConfirm: true,
        status: `Wand preview: ${r.count} voxels — adjust ${s.wandMode === "threshold" ? "threshold" : "tolerance"}, then Confirm (or click a new spot).` });
    }
  },
  setCursorIntensity: (x, y, z) => { set({ cursorIntensity: nv.intensityAt(x, y, z), cursorIntensity01: nv.intensityAtNorm(x, y, z) }); },
  setDrawOpacity: (o) => { set({ drawOpacity: o }); if (get().showAnnotations) nv.setDrawOpacity(o); },
  setShowAnnotations: (v) => { set({ showAnnotations: v }); nv.setAnnotationsVisible(v, get().drawOpacity); },
  refreshStats: () => { const st = nv.drawStats(); set({ corneaVox: st.cornea, scarVox: st.scar, canUndo: nv.canUndo(), canRedo: nv.canRedo() }); },
  zoomIn: () => nv.zoomBy(1.25),
  zoomOut: () => nv.zoomBy(1 / 1.25),
  resetView: () => nv.resetView(),
  cancelOverwrite: () => set({ confirmOverwrite: false }),
  setBrightness: (b) => { nv.setWindow(b, get().contrast); set({ brightness: b }); },
  setContrast: (c) => { nv.setWindow(get().brightness, c); set({ contrast: c }); },
  resetWindow: () => { nv.setWindow(0, 0); set({ brightness: 0, contrast: 0 }); },
  toggleLock: (label) => {
    const locked = get().locked.includes(label) ? get().locked.filter((l) => l !== label) : [...get().locked, label];
    nv.setLockedLabels(locked);
    set({ locked });
  },
  setSliceAxis: (axis, s) => {
    nv.setVoxAxis(axis, s);
    const vox = [...get().vox] as [number, number, number];
    vox[axis] = Math.round(s);
    set({ vox });
  },
  syncVox: () => { const v = nv.currentVox(); if (v) set({ vox: v }); },
  smartFill: async () => {
    if (get().busy) return;
    set({ busy: true, smartPct: 0, status: "Smart fill: growing from seeds…" });
    try {
      const r = await nv.smartFill((pct) => set({ smartPct: pct }));
      if (!r.ok) {
        set({ status: r.reason === "no-seeds"
          ? "Smart fill needs seeds — scribble a little Cornea and Background (and Scar), then Smart fill."
          : "Load a volume first." });
      } else {
        get().refreshStats();
        get().autosaveDraw();
        set({ canConfirm: true,
          status: "Smart-fill PREVIEW — refine your brushstrokes and Smart fill again, or Confirm to keep it." });
      }
    } catch (e) {
      set({ status: `Smart fill failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      set({ busy: false, smartPct: null });
    }
  },
  confirmFill: () => {
    if (nv.confirmFill()) { get().refreshStats(); get().autosaveDraw(); set({ canConfirm: false, wandSeed: null, wandSeedAxis: null, status: "Confirmed — added to the segmentation." }); }
  },
  undo: () => { nv.undoDrawing(); get().refreshStats(); get().autosaveDraw(); set({ canConfirm: nv.isPreviewing(), wandSeed: null, wandSeedAxis: null }); },
  redo: () => { nv.redoDrawing(); get().refreshStats(); get().autosaveDraw(); set({ canConfirm: nv.isPreviewing(), wandSeed: null, wandSeedAxis: null }); },
  requestClear: () => set({ confirmClear: true }),
  cancelClear: () => set({ confirmClear: false }),
  clearDrawing: () => { nv.clearDrawing(); nv.centerView(); get().refreshStats(); void get().flushAutosave(); set({ canConfirm: false, confirmClear: false, wandSeed: null, wandSeedAxis: null, status: "Cleared — blank drawing; view recentred." }); },

  save: async (force = false) => {
    const { activeUser, activeVolume, outputDir, sessionId: sid } = get();
    if (!activeUser || !activeVolume) return;
    const vstem = activeVolume.stem;              // #4: the REAL stem (blinded in the UI)
    const rep = activeVolume.replicate;
    const key = entryKey(activeVolume);          // `${stem}__rep${rep}`
    // A1: if THIS replicate already has a saved ground truth, confirm before overwriting.
    if (!force && get().annotated.has(key)) { set({ confirmOverwrite: true }); return; }
    set({ confirmOverwrite: false });
    if (!outputDir) { await get().chooseOutputDir(); if (!get().outputDir) return; }
    set({ busy: true, status: "Saving ground truth…" });
    try {
      const bytes = await nv.exportLabelmapBytes();
      if (!bytes) throw new Error("Nothing to export.");
      nv.forceDrawAll();   // NEW: exportLabelmapBytes' saveImage can blank the GL canvas on WebKitGTK — repaint
      const st = nv.drawStats();
      const out = get().outputDir!;
      const file = await io.writeLabelmap(out, vstem, activeUser, sid, bytes, rep);
      // #1: persist the SAVED state as this entry's autosave so closing+reopening restores the segmentation
      // (the debounced autosave may not have fired before Save; this is the authoritative copy).
      try { const ak = aKey(activeVolume); if (ak) await io.writeAutosave(activeUser, ak, bytes); } catch { /* best-effort */ }
      await io.appendManifest(out, {
        username: activeUser, volume_stem: vstem, volume_path: activeVolume.path,
        session_id: sid, replicate: rep, blind_label: activeVolume.blindLabel, saved_at: new Date().toISOString(),
        cornea_voxels: st.cornea, scar_voxels: st.scar,
        scar_mm3: Math.round(st.scar * st.mm3 * 1e4) / 1e4,
        spacing: st.spacing.map(s => s.toFixed(4)).join("×"),
        duration_s: Math.round((Date.now() - get().volumeStartMs) / 1000),
        app_version: APP_VERSION,
      });
      const done = new Set(get().annotated); done.add(key);
      nv.forceDrawAll();   // NEW: ensure the image is visible again after the save round-trip (WebKitGTK)
      set({ annotated: done, status: `Saved ${file.split(/[/\\]/).pop()} (${activeVolume.blindLabel} rep ${rep}, scar ${st.scar} vox).` });
    } catch (e) {
      set({ status: `Save failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      set({ busy: false });
    }
  },

  deleteGt: async (entry) => {
    const { activeUser, outputDir } = get();
    if (!activeUser || !outputDir) return;
    set({ busy: true, status: "Deleting ground truth…" });
    try {
      const removed = await io.deleteLabelmaps(outputDir, entry.stem, activeUser, entry.replicate);
      try { const ak = aKey(entry); if (ak) await io.removeAutosave(activeUser, ak); } catch { /* best-effort */ }
      const done = new Set(get().annotated); done.delete(entryKey(entry));
      set({ annotated: done, status: `Deleted ground truth for ${entry.blindLabel} · rep ${entry.replicate} (${removed} file(s)).` });
    } catch (e) {
      set({ status: `Delete failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      set({ busy: false });
    }
  },

  downloadGt: async (entry) => {
    const { activeUser, outputDir } = get();
    if (!activeUser || !outputDir) return;
    try {
      const files = await io.listLabelmapFiles(outputDir, entry.stem, activeUser, entry.replicate);
      const latest = files[files.length - 1];
      if (!latest) { set({ status: "No saved labelmap to download." }); return; }
      const dest = await io.downloadLabelmap(latest, `${entry.stem}__${activeUser}__rep${entry.replicate}.nii.gz`);
      if (dest) set({ status: `Downloaded ${dest.split(/[/\\]/).pop()}.` });
    } catch (e) {
      set({ status: `Download failed: ${e instanceof Error ? e.message : String(e)}` });
    }
  },

  exportAllGt: async () => {
    const { outputDir, adminUnlocked } = get();
    if (!outputDir) { set({ status: "No saved ground truth yet (nothing to export)." }); return; }
    const dest = await io.pickFolder("Choose where to export all saved labelmaps");
    if (!dest) return;
    set({ busy: true, status: "Exporting all saved labelmaps…" });
    try {
      // blinded folder names always; the de-blind mapping (to re-pair with the main app) only for an admin
      const n = await io.exportAllLabelmaps(outputDir, dest, adminUnlocked);
      const where = dest.split(/[/\\]/).pop();
      set({ status: adminUnlocked
        ? `Exported ${n} labelmap file(s) (blinded names + _deblind_mapping.csv) to ${where}.`
        : `Exported ${n} labelmap file(s) under blinded names to ${where}. (Unlock admin to also get the de-blind mapping.)` });
    } catch (e) {
      set({ status: `Export failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      set({ busy: false });
    }
  },
}));
