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

const SIDECAR_PORT: &str = "8765";
const BASE: &str = "http://127.0.0.1:8765";

/// Holds the spawned sidecar process so it can be killed when the app exits.
struct Sidecar(Mutex<Option<Child>>);

#[derive(serde::Deserialize)]
struct FilePayload {
    name: String,
    data: String, // base64-encoded file bytes
}

/// Forward a JSON/text request to the sidecar (frontend: invoke("proxy_request", { method, path, body })).
#[tauri::command]
async fn proxy_request(method: String, path: String, body: Option<String>) -> Result<String, String> {
    let client = reqwest::Client::new();
    let m = reqwest::Method::from_bytes(method.to_uppercase().as_bytes()).map_err(|e| e.to_string())?;
    let mut req = client.request(m, format!("{BASE}{path}"));
    if let Some(b) = body {
        req = req.header("Content-Type", "application/json").body(b);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    resp.text().await.map_err(|e| e.to_string())
}

/// Forward a multipart upload to the sidecar (frontend: invoke("proxy_upload", { path, files, fieldName })).
#[tauri::command]
async fn proxy_upload(path: String, files: Vec<FilePayload>, field_name: String) -> Result<String, String> {
    use base64::{engine::general_purpose::STANDARD, Engine};
    let mut form = reqwest::multipart::Form::new();
    for f in files {
        let bytes = STANDARD.decode(f.data.as_bytes()).map_err(|e| e.to_string())?;
        let part = reqwest::multipart::Part::bytes(bytes).file_name(f.name);
        form = form.part(field_name.clone(), part);
    }
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{BASE}{path}"))
        .multipart(form)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    resp.text().await.map_err(|e| e.to_string())
}

/// Spawn the bundled Python sidecar. Returns None (and logs) if Python or the script is missing.
fn spawn_sidecar(app: &tauri::AppHandle) -> Option<Child> {
    let res = app.path().resource_dir().ok()?;
    let sidecar_dir = res.join("python-sidecar");
    let script = sidecar_dir.join("api_server.py");
    let python = std::env::var("CORNEA_PYTHON").unwrap_or_else(|_| "python3".to_string());
    match Command::new(&python)
        .arg(&script)
        .arg("--port")
        .arg(SIDECAR_PORT)
        .current_dir(&sidecar_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => {
            log::info!("spawned sidecar: {python} {script:?} --port {SIDECAR_PORT}");
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Sidecar(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![proxy_request, proxy_upload])
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
            let child = spawn_sidecar(&app.handle());
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
