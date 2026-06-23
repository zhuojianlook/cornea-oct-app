import { create } from "zustand";
import type { Update } from "@tauri-apps/plugin-updater";
import type { Lang } from "../i18n";
import * as nv from "../niivue/nvController";
import * as io from "../tauri/io";
import { checkForUpdate, installAndRelaunch } from "../tauri/updater";

export type Pen = 0 | 1 | 2 | 3; // 0 erase, 1 cornea, 2 scar, 3 background seed (Smart fill only)
export const APP_VERSION = "0.1.16";
const sessionId = new Date().toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, 19);

interface State {
  // identity / session
  users: string[];
  activeUser: string | null;
  sessionId: string;
  outputDir: string | null;
  // volumes
  folder: string | null;
  volumes: io.VolumeEntry[];
  activeVolume: io.VolumeEntry | null;
  loaded: boolean;
  annotated: Set<string>;          // volume stems this user already saved (this output dir)
  volumeStartMs: number;
  dims: [number, number, number] | null;   // [nx, ny, nz] of the loaded volume
  vox: [number, number, number];           // current crosshair slice indices [x, y, z]
  // pen / tool
  penLabel: Pen;
  penSize: number;
  penFilled: boolean;
  tool: "paint" | "navigate" | "wand";   // active interaction tool
  drawOpacity: number;
  wandThreshold: number;     // threshold scar wand, 0..1 of the intensity range
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
  openVolume: (v: io.VolumeEntry) => Promise<void>;
  nextUnannotated: () => void;
  loadSegmentation: () => Promise<void>;
  chooseOutputDir: () => Promise<void>;
  setPenLabel: (p: Pen) => void;
  setPenSize: (n: number) => void;
  setPenFilled: (f: boolean) => void;
  setTool: (t: "paint" | "navigate" | "wand") => void;
  setWandThreshold: (t: number) => void;
  wandAt: (x: number, y: number, z: number) => void;
  setCursorIntensity: (x: number, y: number, z: number) => void;
  setDrawOpacity: (o: number) => void;
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
  cancelOverwrite: () => void;
  setBrightness: (b: number) => void;
  setContrast: (c: number) => void;
  resetWindow: () => void;
  toggleLock: (label: number) => void;
  persistConfig: () => void;
  resumePending: () => void;
}

export const useStore = create<State>((set, get) => ({
  users: [],
  activeUser: null,
  sessionId,
  outputDir: null,
  folder: null,
  volumes: [],
  activeVolume: null,
  loaded: false,
  annotated: new Set(),
  volumeStartMs: 0,
  dims: null,
  vox: [0, 0, 0],
  penLabel: 2,
  penSize: 8,
  penFilled: false,
  tool: "paint",
  drawOpacity: 0.6,
  wandThreshold: 0.55,
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
    set({ users: cfg.users, outputDir: cfg.outputDir, folder: cfg.lastFolder, lang: cfg.lang, pendingVolume: cfg.lastVolume ?? null });
    // A2: auto-restore the last folder's volume list so the user need not re-pick the folder. The last
    // volume is reopened (resumePending) once a user logs in and the canvas has attached.
    if (cfg.lastFolder) {
      try {
        const volumes = await io.listNifti(cfg.lastFolder);
        set({ volumes });
      } catch { set({ folder: null, pendingVolume: null }); } // folder moved/deleted
    }
  },

  // Persist the full config (users, output dir, last folder + volume, language) — best-effort.
  persistConfig: () => {
    const s = get();
    void io.saveConfig({ users: s.users, outputDir: s.outputDir, lastFolder: s.folder, lang: s.lang, lastVolume: s.activeVolume?.path ?? null });
  },

  // After login + canvas attach, reopen the volume from the previous session (#2).
  resumePending: () => {
    const s = get();
    if (!s.activeUser || s.activeVolume || s.loaded || !s.pendingVolume) return;
    const v = s.volumes.find((x) => x.path === s.pendingVolume);
    set({ pendingVolume: null });
    if (v) void get().openVolume(v);
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
      set({ folder, volumes, status: `${volumes.length} volume(s) found. Select one to annotate.` });
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
      const bytes = await io.readVolume(v.path);
      await nv.loadVolumeBytes(bytes, v.name, get().drawOpacity);
      nv.setPenSize(get().penSize);
      nv.setPen(get().penLabel, get().penFilled);
      nv.setSliceListener((vx) => set({ vox: vx })); // keep slice scrollbars synced to scroll-wheel nav
      nv.setLockedLabels(get().locked);              // re-apply label locks for the new volume (#4)
      set({ activeVolume: v, loaded: true, tool: "paint", volumeStartMs: Date.now(), canConfirm: false,
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
    const { volumes, annotated, activeVolume } = get();
    if (!volumes.length) return;
    const startIdx = activeVolume ? volumes.findIndex((v) => v.path === activeVolume.path) : -1;
    for (let k = 1; k <= volumes.length; k++) {
      const v = volumes[(startIdx + k + volumes.length) % volumes.length];
      if (!annotated.has(v.name.replace(/\.nii(\.gz)?$/i, ""))) { void get().openVolume(v); return; }
    }
    set({ status: "All volumes in this folder are annotated." });
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
  setTool: (t) => { if (t === "paint") nv.lockCrosshair(); nv.setDrawingEnabled(false); set({ tool: t }); },
  setWandThreshold: (t) => set({ wandThreshold: Math.max(0, Math.min(1, t)) }),
  wandAt: (x, y, z) => {
    const r = nv.wandFill(x, y, z, get().wandThreshold);
    if (!r.ok) {
      set({ status: r.reason === "below-threshold" ? "Wand: that spot is below the threshold — lower it or click a brighter area."
        : r.reason === "outside-cornea" ? "Wand: click inside the cornea (scar grows within it)." : "Wand: nothing to fill here." });
    } else { get().refreshStats(); set({ status: `Wand: filled ${r.count} scar voxels. Adjust the threshold or Undo if needed.` }); }
  },
  setCursorIntensity: (x, y, z) => { set({ cursorIntensity: nv.intensityAt(x, y, z) }); },
  setDrawOpacity: (o) => { nv.setDrawOpacity(o); set({ drawOpacity: o }); },
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
    if (nv.confirmFill()) { get().refreshStats(); set({ canConfirm: false, status: "Smart fill confirmed — added to the segmentation." }); }
  },
  undo: () => { nv.undoDrawing(); get().refreshStats(); set({ canConfirm: nv.isPreviewing() }); },
  redo: () => { nv.redoDrawing(); get().refreshStats(); set({ canConfirm: nv.isPreviewing() }); },
  requestClear: () => set({ confirmClear: true }),
  cancelClear: () => set({ confirmClear: false }),
  clearDrawing: () => { nv.clearDrawing(); get().refreshStats(); set({ canConfirm: false, confirmClear: false, status: "Cleared — blank drawing." }); },

  save: async (force = false) => {
    const { activeUser, activeVolume, outputDir, sessionId: sid } = get();
    if (!activeUser || !activeVolume) return;
    const vstem = await io.stem(activeVolume.path);
    // A1: if this scan already has a saved ground truth, confirm before overwriting.
    if (!force && get().annotated.has(vstem)) { set({ confirmOverwrite: true }); return; }
    set({ confirmOverwrite: false });
    if (!outputDir) { await get().chooseOutputDir(); if (!get().outputDir) return; }
    set({ busy: true, status: "Saving ground truth…" });
    try {
      const bytes = await nv.exportLabelmapBytes();
      if (!bytes) throw new Error("Nothing to export.");
      const st = nv.drawStats();
      const out = get().outputDir!;
      const file = await io.writeLabelmap(out, vstem, activeUser, sid, bytes);
      await io.appendManifest(out, {
        username: activeUser, volume_stem: vstem, volume_path: activeVolume.path,
        session_id: sid, saved_at: new Date().toISOString(),
        cornea_voxels: st.cornea, scar_voxels: st.scar,
        scar_mm3: Math.round(st.scar * st.mm3 * 1e4) / 1e4,
        spacing: st.spacing.map(s => s.toFixed(4)).join("×"),
        duration_s: Math.round((Date.now() - get().volumeStartMs) / 1000),
        app_version: APP_VERSION,
      });
      const done = new Set(get().annotated); done.add(vstem);
      set({ annotated: done, status: `Saved ${file.split(/[/\\]/).pop()} (scar ${st.scar} vox).` });
    } catch (e) {
      set({ status: `Save failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      set({ busy: false });
    }
  },
}));
