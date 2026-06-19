use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  // Linux AppImage self-update: Tauri downloads the new AppImage to a temp dir and renames it over
  // the running one, which fails if the temp dir is on a different mount (e.g. /tmp = tmpfs) than the
  // AppImage. Point the temp dir at the AppImage's own directory (always same mount + writable).
  #[cfg(target_os = "linux")]
  if let Ok(appimage) = std::env::var("APPIMAGE") {
    if let Some(dir) = std::path::Path::new(&appimage).parent() {
      std::env::set_var("TMPDIR", dir);
    }
  }

  tauri::Builder::default()
    .plugin(tauri_plugin_fs::init())
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_process::init())
    .setup(|app| {
      // Self-update (desktop only; the updater crate isn't built for mobile targets).
      #[cfg(desktop)]
      app.handle().plugin(tauri_plugin_updater::Builder::new().build())?;

      // Force the live window/taskbar icon to the app logo. Linux/GTK doesn't reliably apply the
      // bundled icon to the running window, so it can fall back to a default — set it explicitly.
      if let Some(win) = app.get_webview_window("main") {
        if let Ok(img) = tauri::image::Image::from_bytes(include_bytes!("../icons/128x128.png")) {
          let _ = win.set_icon(img);
        }
      }
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
