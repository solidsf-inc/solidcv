//! Native desktop entry for the `solidcv` + `SuperMimic` Live Webcam-to-3D Studio.
//! The `wasm32` browser entry (`WebHandle`) for `app.solidsf.com` lives in `lib.rs`.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1480.0, 900.0])
            .with_min_inner_size([1024.0, 680.0])
            .with_title("solidSF · solidcv × SuperMimic Live 3D Webcam & Humanoid World-Model Studio"),
        ..Default::default()
    };
    eframe::run_native(
        "solidcv-supermimic-studio",
        native_options,
        Box::new(|cc| Ok(Box::new(solidcv_app::SolidCvMimicApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {}
