/* Native IO via Tauri v2 plugins (dialog + fs). All file access for the annotator goes through here:
   pick a folder of NIfTI volumes, read a volume's bytes, write the exported labelmap, and persist the
   user list + the ground-truth manifest (for inter/intra-observer analysis). */

import { open } from "@tauri-apps/plugin-dialog";
import { readFile, writeFile, readTextFile, writeTextFile, readDir, mkdir, exists } from "@tauri-apps/plugin-fs";
import { appConfigDir, join, basename } from "@tauri-apps/api/path";

export interface VolumeEntry { name: string; path: string; }
export interface AppConfig { users: string[]; outputDir: string | null; lastFolder: string | null; lang: "en" | "zh"; lastVolume?: string | null; }
export interface ManifestRow {
  username: string; volume_stem: string; volume_path: string;
  session_id: string; saved_at: string; cornea_voxels: number; scar_voxels: number;
  scar_mm3: number; spacing: string; duration_s: number; app_version: string;
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
               lastFolder: d.lastFolder ?? null, lang: d.lang === "zh" ? "zh" : "en", lastVolume: d.lastVolume ?? null };
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
                                    sessionId: string, bytes: Uint8Array): Promise<string> {
  const dir = await join(outputDir, volumeStem);
  if (!(await exists(dir))) await mkdir(dir, { recursive: true });
  const fname = `${username}__${sessionId}.nii.gz`;
  const full = await join(dir, fname);
  await writeFile(full, bytes);
  return full;
}

const CSV_COLS: (keyof ManifestRow)[] = ["username", "volume_stem", "volume_path", "session_id",
  "saved_at", "cornea_voxels", "scar_voxels", "scar_mm3", "spacing", "duration_s", "app_version"];

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

/** Which volume stems this user has already annotated (from the manifest) — for the volume-list badges. */
export async function annotatedStems(outputDir: string | null, username: string): Promise<Set<string>> {
  const done = new Set<string>();
  if (!outputDir) return done;
  try {
    const jsonP = await join(outputDir, "manifest.json");
    if (await exists(jsonP)) {
      const rows: ManifestRow[] = JSON.parse(await readTextFile(jsonP));
      for (const r of rows) if (r.username === username) done.add(r.volume_stem);
    }
  } catch { /* ignore */ }
  return done;
}
