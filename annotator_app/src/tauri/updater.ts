/* Self-update via the Tauri updater plugin. Checks the GitHub release `latest.json` (signed with our
   private key; verified against the public key baked into tauri.conf.json). No-op outside the Tauri
   desktop shell (e.g. browser/dev), so the rest of the app is unaffected there. */

import type { Update } from "@tauri-apps/plugin-updater";

function inTauri(): boolean {
  return (
    typeof window !== "undefined" &&
    ("__TAURI_INTERNALS__" in window || "__TAURI__" in window || "__TAURI_IPC__" in window)
  );
}

/** Returns the available Update (if any), or null when up to date / not in the desktop shell. */
export async function checkForUpdate(): Promise<Update | null> {
  if (!inTauri()) return null;
  try {
    const { check } = await import("@tauri-apps/plugin-updater");
    return await check();
  } catch {
    // Offline, endpoint unreachable, or plugin missing — fail silent (no nag).
    return null;
  }
}

/** Download + install the update (reporting 0–100% progress), then relaunch into the new version. */
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
