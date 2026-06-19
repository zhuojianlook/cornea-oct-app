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
