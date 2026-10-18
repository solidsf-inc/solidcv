//! `worldtube`: $(3+1)\text{D}$ Minkowskian Causal Worldtubes & $O(\log_2 k)$ Dyadic Ray Marching.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/solidcv/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SolidCvWsClient`] (`WsEngine::Worldtube4D`).
//! - **Sealed Binary Vault (`assets/solidcv_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_worldtube.rs"));

/// Dispatch a $(3+1)\text{D}$ `Worldtube` causal light-cone & dyadic ray-march request over the
/// active binary WebSocket connection (`wss://api.solidsf.com/v1/solidcv/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SolidCvWsClient,
    horizon_steps: usize,
    max_speed_vox_per_frame: f64,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::Worldtube4D,
        operation: "worldtube_dyadic_ray_march".to_string(),
        payload_json: format!(
            "{{\"horizon_steps\":{horizon_steps},\"c_max\":{max_speed_vox_per_frame}}}"
        ),
    };
    client.call(&req)
}
