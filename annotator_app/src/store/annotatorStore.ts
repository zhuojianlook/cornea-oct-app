import { create } from "zustand";
import type { Update } from "@tauri-apps/plugin-updater";
import type { Lang } from "../i18n";
import * as nv from "../niivue/nvController";
import * as io from "../tauri/io";
import { checkForUpdate, installAndRelaunch } from "../tauri/updater";

export type Pen = 0 | 1 | 2 | 3; // 0 erase, 1 cornea, 2 scar, 3 background seed (Smart fill only)
export const APP_VERSION = "0.1.12";
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
  // pen
  penLabel: Pen;
  penSize: number;
  penFilled: boolean;
  paintMode: boolean;
  drawOpacity: number;
  // ui
  busy: boolean;
  status: string;
  lang: Lang;
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
  selectUser: (name: string) => Promise<void>;
  pickFolder: () => Promise<void>;
  openVolume: (v: io.VolumeEntry) => Promise<void>;
  chooseOutputDir: () => Promise<void>;
  setPenLabel: (p: Pen) => void;
  setPenSize: (n: number) => void;
  setPenFilled: (f: boolean) => void;
  setPaintMode: (on: boolean) => void;
  setDrawOpacity: (o: number) => void;
  setSliceAxis: (axis: 0 | 1 | 2, s: number) => void;
  syncVox: () => void;
  smartFill: () => void;
  undo: () => void;
  clearDrawing: () => void;
  save: () => Promise<void>;
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
  penSize: 3,
  penFilled: false,
  paintMode: true,
  drawOpacity: 0.6,
  busy: false,
  status: "Select or add a user to begin.",
  lang: "en",
  update: null,
  updateBusy: false,
  updatePct: null,
  updateMsg: "",

  init: async () => {
    const cfg = await io.loadConfig();
    set({ users: cfg.users, outputDir: cfg.outputDir, folder: cfg.lastFolder, lang: cfg.lang });
  },

  setLang: (l) => {
    set({ lang: l });
    void io.saveConfig({ users: get().users, outputDir: get().outputDir, lastFolder: get().folder, lang: l });
  },

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
    await io.saveConfig({ users, outputDir: get().outputDir, lastFolder: get().folder, lang: get().lang });
    await get().selectUser(u);
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
      await io.saveConfig({ users: get().users, outputDir: get().outputDir, lastFolder: folder, lang: get().lang });
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
      set({ activeVolume: v, loaded: true, paintMode: true, volumeStartMs: Date.now(),
            dims: nv.getDims(), vox: nv.currentVox() ?? [0, 0, 0],
            status: `Annotating ${v.name}. Paint cornea/scar; Smart fill helps; then Save.` });
    } catch (e) {
      set({ status: `Failed to load volume: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      set({ busy: false });
    }
  },

  chooseOutputDir: async () => {
    const dir = await io.pickFolder("Choose where to save ground-truth output");
    if (!dir) return;
    set({ outputDir: dir });
    await io.saveConfig({ users: get().users, outputDir: dir, lastFolder: get().folder, lang: get().lang });
    if (get().activeUser) set({ annotated: await io.annotatedStems(dir, get().activeUser!) });
  },

  setPenLabel: (p) => { nv.setPen(p, get().penFilled); set({ penLabel: p }); },
  setPenSize: (n) => { nv.setPenSize(n); set({ penSize: n }); },
  setPenFilled: (f) => { nv.setPen(get().penLabel, f); set({ penFilled: f }); },
  setPaintMode: (on) => { if (on) { nv.setPen(get().penLabel, get().penFilled); nv.lockCrosshair(); } nv.setDrawingEnabled(on); set({ paintMode: on }); },
  setDrawOpacity: (o) => { nv.setDrawOpacity(o); set({ drawOpacity: o }); },
  setSliceAxis: (axis, s) => {
    nv.setVoxAxis(axis, s);
    const vox = [...get().vox] as [number, number, number];
    vox[axis] = Math.round(s);
    set({ vox });
  },
  syncVox: () => { const v = nv.currentVox(); if (v) set({ vox: v }); },
  smartFill: () => {
    const r = nv.smartFill();
    if (!r.ok) {
      set({ status: r.reason === "no-seeds"
        ? "Smart fill needs seeds — scribble a little Cornea and Background (and Scar), then Smart fill."
        : "Load a volume first." });
      return;
    }
    set({ status: "Smart fill done. If the cornea over-grew, add Background seeds around it and Smart fill again." });
  },
  undo: () => nv.undoDrawing(),
  clearDrawing: () => { nv.clearDrawing(); set({ status: "Cleared — blank drawing." }); },

  save: async () => {
    const { activeUser, activeVolume, outputDir, sessionId: sid } = get();
    if (!activeUser || !activeVolume) return;
    if (!outputDir) { await get().chooseOutputDir(); if (!get().outputDir) return; }
    set({ busy: true, status: "Saving ground truth…" });
    try {
      const bytes = await nv.exportLabelmapBytes();
      if (!bytes) throw new Error("Nothing to export.");
      const st = nv.drawStats();
      const out = get().outputDir!;
      const vstem = await io.stem(activeVolume.path);
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
