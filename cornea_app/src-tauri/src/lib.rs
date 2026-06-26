// Cornea OCT — native Tauri shell.
//
// Route A ("thin shell"): this shell does NOT bundle Python/torch. It spawns the user's existing
// Python sidecar (api_server.py, bundled as a resource) on 127.0.0.1:8765, proxies the frontend's
// JSON/upload calls to it (proxy_request / proxy_upload — the names client.ts already invokes), and
// ships a cheap auto-updater for the shell + UI. The sidecar's Python deps (fastapi, torch, SAM2,
// SimpleITK, …) must be installed on the user's machine; missing deps surface as a failed health
// check in the UI.

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::Manager;

/// Holds the spawned sidecar process so it can be killed when the app exits.
struct Sidecar(Mutex<Option<Child>>);

/// Base URL of THIS app's own sidecar, e.g. "http://127.0.0.1:8765". Chosen at startup: 8765 when
/// free, otherwise an OS-assigned free port — so the app always owns its OWN sidecar instead of
/// silently talking to a stale/foreign one already holding 8765. proxy_request/proxy_upload read it,
/// and the frontend fetches it via get_sidecar_base for direct niivue resource loads.
struct SidecarBase(Mutex<String>);

/// Pick the sidecar port: prefer the conventional 8765; if it's already taken (a stale/foreign
/// sidecar), grab any free OS-assigned port so this instance gets its own.
fn pick_port() -> u16 {
    use std::net::TcpListener;
    if let Ok(l) = TcpListener::bind(("127.0.0.1", 8765u16)) {
        drop(l);
        return 8765;
    }
    if let Ok(l) = TcpListener::bind(("127.0.0.1", 0u16)) {
        if let Ok(addr) = l.local_addr() {
            let p = addr.port();
            drop(l);
            return p;
        }
    }
    8765
}

#[derive(serde::Deserialize)]
struct FilePayload {
    name: String,
    data: String, // base64-encoded file bytes
}

/// Forward a JSON/text request to the sidecar (frontend: invoke("proxy_request", { method, path, body })).
#[tauri::command]
async fn proxy_request(method: String, path: String, body: Option<String>, base: tauri::State<'_, SidecarBase>) -> Result<String, String> {
    let b = base.0.lock().unwrap().clone();
    let client = reqwest::Client::new();
    let m = reqwest::Method::from_bytes(method.to_uppercase().as_bytes()).map_err(|e| e.to_string())?;
    let mut req = client.request(m, format!("{b}{path}"));
    if let Some(b) = body {
        req = req.header("Content-Type", "application/json").body(b);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    resp.text().await.map_err(|e| e.to_string())
}

/// Forward a multipart upload to the sidecar (frontend: invoke("proxy_upload", { path, files, fieldName })).
#[tauri::command]
async fn proxy_upload(path: String, files: Vec<FilePayload>, field_name: String, base: tauri::State<'_, SidecarBase>) -> Result<String, String> {
    use base64::{engine::general_purpose::STANDARD, Engine};
    let b = base.0.lock().unwrap().clone();
    let mut form = reqwest::multipart::Form::new();
    for f in files {
        let bytes = STANDARD.decode(f.data.as_bytes()).map_err(|e| e.to_string())?;
        let part = reqwest::multipart::Part::bytes(bytes).file_name(f.name);
        form = form.part(field_name.clone(), part);
    }
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{b}{path}"))
        .multipart(form)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    resp.text().await.map_err(|e| e.to_string())
}

/// The base URL the frontend should use for DIRECT resource fetches (niivue volumes/previews/img),
/// which bypass the IPC proxy. Returns this app's own sidecar base (the port chosen at startup).
#[tauri::command]
fn get_sidecar_base(base: tauri::State<'_, SidecarBase>) -> String {
    base.0.lock().unwrap().clone()
}

/// Spawn the bundled Python sidecar on `port`. Returns None (and logs) if Python or the script is missing.
fn spawn_sidecar(app: &tauri::AppHandle, port: u16) -> Option<Child> {
    let res = app.path().resource_dir().ok()?;
    let sidecar_dir = res.join("python-sidecar");
    let script = sidecar_dir.join("api_server.py");
    let python = std::env::var("CORNEA_PYTHON").unwrap_or_else(|_| "python3".to_string());

    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .arg("--port")
        .arg(port.to_string())
        .current_dir(&sidecar_dir);
    // The AppImage runtime exports PYTHONHOME/PYTHONPATH pointing into its OWN mount; inherited by the
    // child these make the system python3 fail to load its stdlib ("No module named 'encodings'") and
    // crash on startup. Strip them so the sidecar interpreter uses its own stdlib + the user's
    // site-packages, exactly as in a normal shell. (Same reason the bundled sidecar never started.)
    cmd.env_remove("PYTHONHOME");
    cmd.env_remove("PYTHONPATH");
    // Stamp the sidecar with this shell's version so /api/health can confirm the right one is up.
    cmd.env("CORNEA_SHELL_VERSION", app.package_info().version.to_string());
    // Installed app: write cases/state to the OS app-data dir (not the read-only bundle), and tee the
    // sidecar's stdout/stderr to sidecar.log there so a failed start (missing Python deps, etc.) is
    // diagnosable instead of vanishing into /dev/null.
    let mut logged = false;
    if let Ok(data_dir) = app.path().app_data_dir() {
        let _ = std::fs::create_dir_all(&data_dir);
        cmd.env("CORNEA_DATA_DIR", &data_dir);
        if let Ok(log) = std::fs::File::create(data_dir.join("sidecar.log")) {
            if let Ok(errlog) = log.try_clone() {
                cmd.stdout(Stdio::from(log)).stderr(Stdio::from(errlog));
                logged = true;
            }
        }
    }
    if !logged {
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
    }
    match cmd.spawn() {
        Ok(child) => {
            log::info!("spawned sidecar: {python} {script:?} --port {port}");
            Some(child)
        }
        Err(e) => {
            log::error!("failed to spawn sidecar ({python}): {e}");
            None
        }
    }
}

fn kill_sidecar(app: &tauri::AppHandle) {
    if let Some(state) = app.try_state::<Sidecar>() {
        if let Some(mut child) = state.0.lock().unwrap().take() {
            let _ = child.kill();
        }
    }
}

/// Append a timestamped line to <app-data>/updater.log. Works in RELEASE builds (tauri_plugin_log only
/// inits under debug_assertions, so log::* is invisible in the shipped AppImage). This is how we make
/// the self-update path diagnosable: the startup line records the running version + the $APPIMAGE file
/// size, so after an update you can SEE whether the file was actually replaced.
fn ulog(app: &tauri::AppHandle, msg: &str) {
    use std::io::Write;
    if let Ok(dir) = app.path().app_data_dir() {
        let _ = std::fs::create_dir_all(&dir);
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("updater.log")) {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(f, "[{t}] {msg}");
        }
    }
}

/// Restart after a self-update. Kill the sidecar first so its port frees and the new instance can
/// respawn the *updated* sidecar (otherwise the stale one keeps serving and the update is wasted).
///
/// On Linux AppImages, `app.restart()` re-execs `current_exe()` — which is the binary INSIDE the
/// (old) AppImage mount, NOT the .AppImage file the updater just replaced — so it would relaunch the
/// OLD version, leaving the update banner up (the loop the user hit). Instead we run the (replaced)
/// $APPIMAGE file itself: exec first (replaces this process); if exec fails, spawn it detached and
/// exit. We NEVER fall back to app.restart() on an AppImage. Both paths + sizes are logged so a
/// failure is diagnosable. Non-AppImage installs (deb/rpm/dmg/msi) use Tauri's restart.
#[tauri::command]
fn restart_app(app: tauri::AppHandle) {
    kill_sidecar(&app);
    // Persist the current window size/position NOW: the AppImage update path below exec()s a new process,
    // which never fires Tauri's exit event, so the window-state plugin's save-on-exit would be skipped and an
    // update would forget a resize made this session.
    {
        use tauri_plugin_window_state::{AppHandleExt, StateFlags};
        let _ = app.save_window_state(StateFlags::all());
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(appimage) = std::env::var("APPIMAGE") {
            let size = std::fs::metadata(&appimage).map(|m| m.len()).unwrap_or(0);
            ulog(&app, &format!("restart_app: exec APPIMAGE={appimage} size={size}"));
            use std::os::unix::process::CommandExt;
            let err = Command::new(&appimage).exec(); // only returns on failure
            ulog(&app, &format!("restart_app: exec failed ({err}) — spawning detached + exit"));
            // exec failed → run the new AppImage as a detached process and exit, rather than
            // app.restart() (which would re-launch the OLD in-mount binary → the update is lost).
            let _ = Command::new(&appimage).spawn();
            std::process::exit(0);
        }
        ulog(&app, "restart_app: APPIMAGE unset — using app.restart() (update may not take effect)");
    }
    app.restart();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Linux AppImage self-update: Tauri downloads the new AppImage to a temp dir and renames it over
    // the running one, which fails if the temp dir is on a different mount (e.g. /tmp = tmpfs). Point
    // the temp dir at the AppImage's own directory (always same mount + writable).
    #[cfg(target_os = "linux")]
    if let Ok(appimage) = std::env::var("APPIMAGE") {
        if let Some(dir) = std::path::Path::new(&appimage).parent() {
            std::env::set_var("TMPDIR", dir);
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        // Remember the window's size + position across restarts AND updates (state file lives in the
        // OS app-config dir, keyed by the stable identifier, so a new version restores the old window).
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .manage(Sidecar(Mutex::new(None)))
        .manage(SidecarBase(Mutex::new("http://127.0.0.1:8765".to_string())))
        .invoke_handler(tauri::generate_handler![proxy_request, proxy_upload, restart_app, get_sidecar_base])
        .setup(|app| {
            #[cfg(desktop)]
            {
                app.handle().plugin(tauri_plugin_process::init())?;
                app.handle().plugin(tauri_plugin_updater::Builder::new().build())?;
            }
            // Window size/position persistence. The window-state plugin saves on close, but it does NOT
            // reliably RESTORE a window declared in tauri.conf.json (it gets created at the config size before
            // the plugin's auto-restore applies), so the app kept reopening at the default 1400x900 even
            // though .window-state.json held the user's last size. Restore it EXPLICITLY here, and enforce a
            // sane MINIMUM so a restored tiny/off-screen size can't make the app unusable.
            {
                use tauri_plugin_window_state::{StateFlags, WindowExt};
                if let Some(win) = app.get_webview_window("main") {
                    let _ = win.set_min_size(Some(tauri::LogicalSize::new(900.0, 600.0)));
                    // restore size + position (+maximized/fullscreen); the OS clamps to the min set above.
                    let _ = win.restore_state(StateFlags::all());
                }
            }
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            // Claim our OWN sidecar on a port we own (8765 if free, else a free one) so a stale or
            // foreign sidecar already holding 8765 can never be silently reused.
            let port = pick_port();
            *app.state::<SidecarBase>().0.lock().unwrap() = format!("http://127.0.0.1:{port}");
            let child = spawn_sidecar(&app.handle(), port);
            *app.state::<Sidecar>().0.lock().unwrap() = child;
            // Record this launch's version + the $APPIMAGE file size. After a self-update the new
            // launch logs the NEW version/size IFF the AppImage was actually replaced — the single
            // clearest signal for whether an updater loop is an install problem or a restart problem.
            #[cfg(target_os = "linux")]
            {
                let ai = std::env::var("APPIMAGE").unwrap_or_default();
                let size = if ai.is_empty() { 0 } else { std::fs::metadata(&ai).map(|m| m.len()).unwrap_or(0) };
                ulog(&app.handle(), &format!("startup v{} APPIMAGE='{}' size={}", app.package_info().version, ai, size));
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::Destroyed) {
                kill_sidecar(window.app_handle());
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                kill_sidecar(app);
            }
        });
}
