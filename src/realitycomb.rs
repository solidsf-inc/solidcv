//! `realitycomb`: Sparse-Gated Pyramidal Optical Flow & Static-vs-Moving Scene Separator.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/solidcv/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SolidCvWsClient`] (`WsEngine::RealityComber`).
//! - **Sealed Binary Vault (`assets/solidcv_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_realitycomb.rs"));

/// Dispatch a `RealityComber` optical-flow & static-vs-moving scene separation request over
/// the active binary WebSocket connection (`wss://api.solidsf.com/v1/solidcv/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SolidCvWsClient,
    rows: i32,
    cols: i32,
    diff_tol: u8,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::RealityComber,
        operation: "comb_dense_velocity_field".to_string(),
        payload_json: format!("{{\"rows\":{rows},\"cols\":{cols},\"diff_tol\":{diff_tol}}}"),
    };
    client.call(&req)
}
