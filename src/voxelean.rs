//! `voxelean`: 64-Bit Zobrist Octree $O(\log N)$ Volumetric Diff & $1+7=8$ Schur-Feshbach Wavelets.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/solidcv/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SolidCvWsClient`] (`WsEngine::VoxeleanOctree`).
//! - **Sealed Binary Vault (`assets/solidcv_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_voxelean.rs"));

/// Dispatch a 3D `Voxelean` Zobrist octree volumetric diff request over the active binary
/// WebSocket connection (`wss://api.solidsf.com/v1/solidcv/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SolidCvWsClient,
    grid_dim: usize,
    root_hash_a: u64,
    root_hash_b: u64,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::VoxeleanOctree,
        operation: "voxelean_zobrist_octree_diff".to_string(),
        payload_json: format!(
            "{{\"grid_dim\":{grid_dim},\"root_hash_a\":{root_hash_a},\"root_hash_b\":{root_hash_b}}}"
        ),
    };
    client.call(&req)
}
