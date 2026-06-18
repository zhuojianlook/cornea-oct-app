# Cornea Ground-Truth Annotator

A standalone **cross-platform Tauri desktop app** for humans to create **ground-truth scar
segmentations** on preprocessed OCT volumes, for benchmarking the main pipeline. It is fully
**client-side** (niivue in the webview) — no Python/sidecar — so it bundles cleanly on mac/Windows/Linux.

## What it does
- **Login on launch**: pick an existing user from a dropdown or add a new one (required to enter). The
  username is recorded with every saved label.
- **Pick a folder of NIfTI** (`.nii` / `.nii.gz`) volumes → select one → it loads **unlabelled**.
- **Paint** Cornea / Scar (Erase) with a sized brush (paint/navigate toggle, filled-region pen, brush
  cursor), and **Smart fill (GrowCut)** to propagate a few scribbles through the whole 3-D volume.
- **Save ground truth**: writes a 0/1/2 NIfTI labelmap (0=bg, 1=cornea, 2=scar) co-registered to the
  volume, plus a manifest row tagged with username + session.

## Output layout (enables inter-/intra-observer)
Chosen output folder:
```
<output>/
  <volume_stem>/<username>__<sessionId>.nii.gz   # labelmap (0/1/2)
  manifest.csv  manifest.json                     # one row per save
```
Manifest columns: `username, volume_stem, volume_path, session_id, saved_at, cornea_voxels,
scar_voxels, scar_mm3, spacing, duration_s, app_version`.
- **Inter-observer**: group by `volume_stem` across distinct `username`.
- **Intra-observer**: group by `username` + `volume_stem` across distinct `session_id` (a session id is
  the app-launch timestamp = one annotation occasion).

The labelmaps are 0/1/2 and co-registered, so they drop straight into the main project's benchmark
harnesses (compare GT scar vs detector scar with Dice / volume).

## Prerequisites
- **Node 18+** and **Rust** (`rustup`), per the Tauri v2 guide.
- **System libraries** (one-time):
  - **Linux**: `sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file libssl-dev libayatana-appindicator3-dev librsvg2-dev`
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`).
  - **Windows**: WebView2 runtime (preinstalled on Win11) + MSVC build tools.

## Develop
```bash
cd annotator_app
npm install
npm run tauri dev      # launches the desktop app with live reload
```

## Build installers
```bash
npm run tauri build
```
Outputs go to `src-tauri/target/release/bundle/`:
- **Linux**: `.AppImage` and `.deb`
- **macOS**: `.dmg` / `.app` (build on macOS)
- **Windows**: `.msi` / `.exe` (build on Windows)

Tauri does **not** cross-compile — build each OS on that OS, or use the included GitHub Actions workflow
(`.github/workflows/build-annotator.yml`, at the repo root) to produce all three from CI.

## Updating
From **v0.1.1** on, the app has a built-in updater: on launch it checks the latest GitHub release and,
if a newer **signed** build exists, shows an "Install & restart" banner (one click — download, verify,
relaunch). Updates are verified against the public key in `src-tauri/tauri.conf.json`; CI signs each
release with the matching private key (stored as the `TAURI_SIGNING_PRIVATE_KEY` Actions secret) and
publishes `latest.json`. The updater only applies versions released *after* the one introducing it, so
**v0.1.0 → v0.1.1 must be installed manually once**; subsequent updates are automatic.

## Notes
- Reuses the main app's hardened brush UX (paint/navigate toggle, brush-size cursor, filled pen, GrowCut
  smart fill, undo, overwrite-on-repaint, translucent cornea so scar shows in 3-D).
- Independent of the main app's `cases/` and Python sidecar — works on any folder of preprocessed NIfTI.
