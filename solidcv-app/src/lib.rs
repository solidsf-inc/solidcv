//! `solidcv` + `SuperMimic` Live 3D Webcam & Humanoid Studio — `egui` front-end.
//! Native entry lives in `main.rs`; the `wasm32` entry ([`WebHandle`]) mounts
//! [`SolidCvMimicApp`] onto a `<canvas>` in the `solidcv` tab of `app.solidsf.com`.

mod app;
pub use app::SolidCvMimicApp;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WebHandle {
    runner: eframe::WebRunner,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WebHandle {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            runner: eframe::WebRunner::new(),
        }
    }

    /// Mount the `solidcv` + `SuperMimic` studio onto the given `<canvas>`.
    #[wasm_bindgen]
    pub async fn start(
        &self,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(), wasm_bindgen::JsValue> {
        self.runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| Ok(Box::new(SolidCvMimicApp::new(cc)))),
            )
            .await
    }
}
