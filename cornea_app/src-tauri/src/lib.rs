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

/// Restart after a self-update. Kill the sidecar first so port 8765 frees and the new instance can
/// respawn the *updated* sidecar (otherwise the stale one keeps serving and the update is wasted).
/// On Linux AppImages Tauri's relaunch() can silently no-op, so exec the (already-updated) $APPIMAGE
/// directly — exec replaces this process, guaranteeing a restart into the new version. Falls back to
/// Tauri's restart elsewhere / if exec fails.
#[tauri::command]
fn restart_app(app: tauri::AppHandle) {
    kill_sidecar(&app);
    #[cfg(target_os = "linux")]
    {
        if let Ok(appimage) = std::env::var("APPIMAGE") {
            use std::os::unix::process::CommandExt;
            let err = Command::new(&appimage).exec(); // only returns on failure
            log::error!("exec restart failed: {err}");
        }
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
