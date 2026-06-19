/* Dedicated store for the in-app updater (kept separate from caseStore/workflowStore). Both the
   header "Check for updates" button and the UpdateBanner read from here so they stay in sync. */

import { create } from "zustand";
import type { Update } from "@tauri-apps/plugin-updater";
import { checkForUpdate, installAndRelaunch } from "../tauri/updater";

export const APP_VERSION = "0.0.1";

interface UpdaterState {
  update: Update | null;
  busy: boolean;
  pct: number | null;
  msg: string;
  check: (manual: boolean) => Promise<void>;
  install: () => Promise<void>;
  dismiss: () => void;
}

export const useUpdater = create<UpdaterState>((set, get) => ({
  update: null,
  busy: false,
  pct: null,
  msg: "",
  check: async (manual) => {
    if (get().busy) return;
    set({ busy: true, msg: manual ? "Checking for updates…" : "" });
    try {
      const u = await checkForUpdate();
      if (u) set({ update: u, msg: "" });
      else set({ update: null, msg: manual ? `You're on the latest version (v${APP_VERSION}).` : "" });
    } catch (e) {
      set({ msg: manual ? `Update check failed: ${e instanceof Error ? e.message : String(e)}` : "" });
    } finally {
      set({ busy: false });
      if (manual) setTimeout(() => { if (get().msg.startsWith("You're on")) set({ msg: "" }); }, 6000);
    }
  },
  install: async () => {
    const u = get().update;
    if (!u || get().busy) return;
    set({ busy: true, pct: null, msg: "Downloading update…" });
    try {
      await installAndRelaunch(u, (p) => set({ pct: p }));
      // relaunch() replaces the process on success.
    } catch (e) {
      set({ busy: false, pct: null, msg: `Update failed: ${e instanceof Error ? e.message : String(e)}` });
    }
  },
  dismiss: () => set({ update: null, msg: "" }),
}));
