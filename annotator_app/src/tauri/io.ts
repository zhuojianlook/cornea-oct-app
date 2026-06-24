/* Native IO via Tauri v2 plugins (dialog + fs). All file access for the annotator goes through here:
   pick a folder of NIfTI volumes, read a volume's bytes, write the exported labelmap, and persist the
   user list + the ground-truth manifest (for inter/intra-observer analysis). */

import { open, save } from "@tauri-apps/plugin-dialog";
import { readFile, writeFile, readTextFile, writeTextFile, readDir, mkdir, exists, remove } from "@tauri-apps/plugin-fs";
import { appConfigDir, join, basename } from "@tauri-apps/api/path";

export interface VolumeEntry { name: string; path: string; }
export interface AppConfig { users: string[]; outputDir: string | null; lastFolder: string | null; lang: "en" | "zh"; lastVolume?: string | null; replicates?: number; }
export interface ManifestRow {
  username: string; volume_stem: string; volume_path: string;
  session_id: string; saved_at: string; cornea_voxels: number; scar_voxels: number;
  scar_mm3: number; spacing: string; duration_s: number; app_version: string;
  replicate: number;   // #4: which repeat (1..N) this annotation is — drives intra-observer analysis
  blind_label: string; // the neutral label the annotator saw (e.g. "Scan B") — real name stays hidden in-app
}

const isNifti = (n: string) => /\.nii(\.gz)?$/i.test(n);

/** True only inside the Tauri desktop shell. In a plain browser (dev/preview) there is no native
    filesystem, so the file-backed helpers below degrade gracefully instead of throwing. */
function inTauri(): boolean {
  return (
    typeof window !== "undefined" &&
    ("__TAURI_INTERNALS__" in window || "__TAURI__" in window || "__TAURI_IPC__" in window)
  );
}

export async function pickFolder(title = "Choose a folder of NIfTI volumes"): Promise<string | null> {
  if (!inTauri()) return null;
  const r = await open({ directory: true, multiple: false, title });
  return typeof r === "string" ? r : null;
}

/** Pick a single NIfTI file (e.g. a prior ground-truth labelmap to load & correct). */
export async function pickFile(title = "Choose a NIfTI labelmap"): Promise<string | null> {
  if (!inTauri()) return null;
  const r = await open({ directory: false, multiple: false, title, filters: [{ name: "NIfTI", extensions: ["nii", "gz"] }] });
  return typeof r === "string" ? r : null;
}

export async function listNifti(folder: string): Promise<VolumeEntry[]> {
  const entries = await readDir(folder);
  const out: VolumeEntry[] = [];
  for (const e of entries) {
    if (e.isFile && isNifti(e.name)) out.push({ name: e.name, path: await join(folder, e.name) });
  }
  out.sort((a, b) => a.name.localeCompare(b.name));
  return out;
}

export async function readVolume(path: string): Promise<Uint8Array> {
  return await readFile(path);
}

export async function stem(path: string): Promise<string> {
  const b = await basename(path);
  return b.replace(/\.nii(\.gz)?$/i, "");
}

// ── app config (users + output dir), in the OS app-config dir ────────────────
async function configPath(): Promise<string> {
  const dir = await appConfigDir();
  if (!(await exists(dir))) await mkdir(dir, { recursive: true });
  return await join(dir, "annotator_config.json");
}

export async function loadConfig(): Promise<AppConfig> {
  try {
    const p = await configPath();
    if (await exists(p)) {
      const d = JSON.parse(await readTextFile(p));
      return { users: Array.isArray(d.users) ? d.users : [], outputDir: d.outputDir ?? null,
               lastFolder: d.lastFolder ?? null, lang: d.lang === "zh" ? "zh" : "en", lastVolume: d.lastVolume ?? null,
               replicates: typeof d.replicates === "number" ? d.replicates : 2 };
    }
  } catch { /* fall through to defaults */ }
  return { users: [], outputDir: null, lastFolder: null, lang: "en", lastVolume: null };
}

export async function saveConfig(cfg: AppConfig): Promise<void> {
  if (!inTauri()) return; // no native FS in the browser — skip persistence (don't block the user)
  try {
    await writeTextFile(await configPath(), JSON.stringify(cfg, null, 2));
  } catch { /* config persistence is best-effort — must never reject and abort an annotation Save */ }
}

// ── ground-truth output: labelmap file + manifest (json + csv) ───────────────
export async function writeLabelmap(outputDir: string, volumeStem: string, username: string,
                                    sessionId: string, bytes: Uint8Array, replicate = 1): Promise<string> {
  const dir = await join(outputDir, volumeStem);
  if (!(await exists(dir))) await mkdir(dir, { recursive: true });
  // include the replicate so the SAME user's two repeats of a scan never collide (#4 intra-observer)
  const fname = `${username}__rep${replicate}__${sessionId}.nii.gz`;
  const full = await join(dir, fname);
  await writeFile(full, bytes);
  return full;
}

// ── autosave: in-progress annotations survive app close/restart (#5) ──────────
// A per-(user,volume) cache of the CURRENT (unsaved) drawing in the app-config dir, so reopening a
// volume — or restarting the app — restores work that was never "Saved" as final ground truth. Keyed
// by a short hash of user|path so filenames stay short + collision-free across folders.
function autosaveName(user: string, volPath: string): string {
  const s = `${user}|${volPath}`;
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return `as_${h.toString(16)}.nii.gz`;
}
async function autosaveDir(): Promise<string> {
  const dir = await join(await appConfigDir(), "autosave");
  if (!(await exists(dir))) await mkdir(dir, { recursive: true });
  return dir;
}
export async function writeAutosave(user: string, volPath: string, bytes: Uint8Array): Promise<void> {
  await writeFile(await join(await autosaveDir(), autosaveName(user, volPath)), bytes);
}
export async function readAutosave(user: string, volPath: string): Promise<Uint8Array | null> {
  try {
    const p = await join(await autosaveDir(), autosaveName(user, volPath));
    if (await exists(p)) return await readFile(p);
  } catch { /* no autosave / unreadable */ }
  return null;
}
export async function removeAutosave(user: string, volPath: string): Promise<void> {
  try {
    const p = await join(await autosaveDir(), autosaveName(user, volPath));
    if (await exists(p)) await remove(p);
  } catch { /* already gone */ }
}

const CSV_COLS: (keyof ManifestRow)[] = ["username", "volume_stem", "volume_path", "session_id",
  "replicate", "blind_label", "saved_at", "cornea_voxels", "scar_voxels", "scar_mm3", "spacing", "duration_s", "app_version"];

export async function appendManifest(outputDir: string, row: ManifestRow): Promise<void> {
  if (!(await exists(outputDir))) await mkdir(outputDir, { recursive: true });
  const jsonP = await join(outputDir, "manifest.json");
  let rows: ManifestRow[] = [];
  try { if (await exists(jsonP)) rows = JSON.parse(await readTextFile(jsonP)); } catch { rows = []; }
  rows.push(row);
  await writeTextFile(jsonP, JSON.stringify(rows, null, 2));
  const csv = [CSV_COLS.join(",")]
    .concat(rows.map(r => CSV_COLS.map(c => {
      const v = String(r[c] ?? "");
      return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
    }).join(",")))
    .join("\n");
  await writeTextFile(await join(outputDir, "manifest.csv"), csv);
}

/** The saved ground-truth labelmap files for one (stem, replicate) by this user (across sessions),
    chronological (session id is ISO-ish, so lexical sort = oldest→newest; the LAST is the current GT). */
export async function listLabelmapFiles(outputDir: string, stem: string, user: string, replicate: number): Promise<string[]> {
  const dir = await join(outputDir, stem);
  if (!(await exists(dir))) return [];
  const prefix = `${user}__rep${replicate}__`;
  const out: string[] = [];
  for (const e of await readDir(dir)) {
    if (e.isFile && e.name.startsWith(prefix) && isNifti(e.name)) out.push(await join(dir, e.name));
  }
  out.sort();
  return out;
}

/** Delete one entry's saved ground truth: remove this user's labelmap file(s) for (stem, replicate) and
    drop the matching rows from manifest.json + .csv (other users' / other entries' GT is untouched). */
export async function deleteLabelmaps(outputDir: string, stem: string, user: string, replicate: number): Promise<number> {
  let removed = 0;
  for (const f of await listLabelmapFiles(outputDir, stem, user, replicate)) {
    try { await remove(f); removed++; } catch { /* already gone */ }
  }
  try {
    const jsonP = await join(outputDir, "manifest.json");
    if (await exists(jsonP)) {
      let rows: ManifestRow[] = JSON.parse(await readTextFile(jsonP));
      rows = rows.filter((r) => !(r.username === user && r.volume_stem === stem && (r.replicate ?? 1) === replicate));
      await writeTextFile(jsonP, JSON.stringify(rows, null, 2));
      const csv = [CSV_COLS.join(",")]
        .concat(rows.map(r => CSV_COLS.map(c => { const v = String(r[c] ?? ""); return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v; }).join(",")))
        .join("\n");
      await writeTextFile(await join(outputDir, "manifest.csv"), csv);
    }
  } catch { /* manifest update best-effort */ }
  return removed;
}

/** Copy one labelmap file to a user-chosen location (a Save-As dialog). Returns the dest path or null. */
export async function downloadLabelmap(srcPath: string, suggestedName: string): Promise<string | null> {
  if (!inTauri()) return null;
  const dest = await save({ defaultPath: suggestedName, filters: [{ name: "NIfTI", extensions: ["nii.gz", "nii", "gz"] }] });
  if (!dest) return null;
  await writeFile(dest, await readFile(srcPath));
  return dest;
}

/** Export EVERY saved labelmap (+ the manifest) to a chosen folder — for collecting the GT dataset. */
export async function exportAllLabelmaps(outputDir: string, destDir: string): Promise<number> {
  let n = 0;
  for (const e of await readDir(outputDir)) {
    if (e.isDirectory) {
      const sub = await join(outputDir, e.name);
      let made = false;
      for (const f of await readDir(sub)) {
        if (f.isFile && isNifti(f.name)) {
          const dDir = await join(destDir, e.name);
          if (!made) { if (!(await exists(dDir))) await mkdir(dDir, { recursive: true }); made = true; }
          await writeFile(await join(dDir, f.name), await readFile(await join(sub, f.name)));
          n++;
        }
      }
    } else if (e.isFile && /^manifest\.(json|csv)$/.test(e.name)) {
      if (!(await exists(destDir))) await mkdir(destDir, { recursive: true });
      await writeFile(await join(destDir, e.name), await readFile(await join(outputDir, e.name)));
    }
  }
  return n;
}

/** Which (stem, replicate) pairs this user has already saved — keys `${stem}__rep${replicate}` — for the
    per-entry ✓ badges (#4: each replicate is tracked separately so both repeats must be done). */
export async function annotatedStems(outputDir: string | null, username: string): Promise<Set<string>> {
  const done = new Set<string>();
  if (!outputDir) return done;
  try {
    const jsonP = await join(outputDir, "manifest.json");
    if (await exists(jsonP)) {
      const rows: ManifestRow[] = JSON.parse(await readTextFile(jsonP));
      for (const r of rows) if (r.username === username) done.add(`${r.volume_stem}__rep${r.replicate ?? 1}`);
    }
  } catch { /* ignore */ }
  return done;
}
