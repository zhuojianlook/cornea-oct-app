/* Self-update via the Tauri updater plugin (the native shell, src-tauri). Checks the cornea release
   manifest (signed; verified against the pubkey in tauri.conf.json). No-op outside the desktop shell
   (e.g. when the app runs in a plain browser served by the sidecar). */

import type { Update } from "@tauri-apps/plugin-updater";

function inTauri(): boolean {
  return (
    typeof window !== "undefined" &&
    ("__TAURI_INTERNALS__" in window || "__TAURI__" in window || "__TAURI_IPC__" in window)
  );
}

export async function checkForUpdate(): Promise<Update | null> {
  if (!inTauri()) return null;
  try {
    const { check } = await import("@tauri-apps/plugin-updater");
    return await check();
  } catch {
    return null;
  }
}

export async function installAndRelaunch(update: Update, onProgress?: (pct: number | null) => void): Promise<void> {
  let total = 0;
  let received = 0;
  await update.downloadAndInstall((event) => {
    if (event.event === "Started") {
      total = event.data.contentLength ?? 0;
      onProgress?.(total ? 0 : null);
    } else if (event.event === "Progress") {
      received += event.data.chunkLength;
      onProgress?.(total ? Math.min(100, Math.round((received / total) * 100)) : null);
    } else if (event.event === "Finished") {
      onProgress?.(100);
    }
  });
  const { relaunch } = await import("@tauri-apps/plugin-process");
  await relaunch();
}
