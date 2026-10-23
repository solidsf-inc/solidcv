//! `cloud_ws`: Binary RFC-6455 WebSocket Client for `solidSF` Proprietary Cloud Engines.
//!
//! While `solidcv` provides 100% local, zero-FFI execution for all 2D/3D OpenCV primitives,
//! Apple Silicon NEON SIMD kernels, VLM ViT-16 patch-token pruning, 1-bit SAE circuit matching,
//! and non-hallucinating $SE(2)$ zoom/uplift, the following four proprietary `solidSF` engines
//! execute over a low-latency binary WebSocket stream (`wss://api.solidsf.com/v1/solidcv/ws`
//! or `SOLIDCV_WS_URL`):
//!
//! 1. **`WsEngine::Gb300TensorCorePtx` (`0x01`)**: Remote NVIDIA Grace-Blackwell `GB300` (`sm_100a`) /
//!    `GB10` (`sm_120a`) NVLink-C2C coherent slab & PTX 8.5+ `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc`
//!    Tensor-Core batch execution.
//! 2. **`WsEngine::RealityComber` (`0x02`)**: Remote sparse-gated pyramidal optical flow and
//!    static-vs-moving scene separation.
//! 3. **`WsEngine::VoxeleanOctree` (`0x03`)**: Remote 64-bit Zobrist octree construction, $O(\log N)$
//!    volumetric diff, and $1+7=8$ Schur-Feshbach 3D wavelet transforms.
//! 4. **`WsEngine::Worldtube4D` (`0x04`)**: Remote $(3+1)\text{D}$ Minkowskian causal worldtube
//!    evolution and $O(\log_2 k)$ dyadic ray marching.

use crate::core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

const STS_ERROR: i32 = -2;

/// Default production WebSocket endpoint for `solidSF` proprietary CV/4D engines.
pub const DEFAULT_SOLIDCV_WS_URL: &str = "wss://api.solidsf.com/v1/solidcv/ws";

/// Public self-serve `solidSF` Compute Credits storefront (`GET /api/fah/store`).
pub const CREDITS_STORE_URL: &str = "https://solidsf.com/compute-credits";

/// Stripe product contract identifier for `solidSF Credits` (`1 USD = 1,000,000 micro-USD`).
pub const CREDITS_PRODUCT_CONTRACT: &str = "solidsf.platform.billing.v2.product.credits";

/// Default platform markup percentage over raw provider/cluster cost (`30%`).
pub const DEFAULT_MARKUP_PERCENT: i64 = 30;

/// Welcome grant minted to new accounts (`10,000,000 micro-USD = $10.00`).
pub const WELCOME_CREDITS_MICROUSD: i64 = 10_000_000;

/// Referral award minted per verified referral (`5,000,000 micro-USD = $5.00`).
pub const REFERRER_CREDITS_MICROUSD: i64 = 5_000_000;

/// Minimum self-serve FAH store credit purchase (`$10.00`).
pub const FAH_STORE_MIN_PURCHASE_CENTS: i32 = 1_000;

/// Maximum self-serve FAH store credit purchase (`$1,000.00`; org packs scale to `$100,000.00`).
pub const FAH_STORE_MAX_PURCHASE_CENTS: i32 = 100_000;

/// Magic 6-byte binary frame preamble (`"SSFCV\x01"`).
pub const WS_FRAME_MAGIC: [u8; 6] = *b"SSFCV\x01";

/// Zobrist-Feistel encrypted vault seal (`#[used]` so the ciphertext remains in `.rodata`
/// across stripped binaries and static archives for reverse-engineering inspection).
#[used]
pub static SOLIDSF_ENCRYPTED_BOUNTY_BLOB: [u8; 737] = [
    0x1D, 0x0F, 0xDB, 0x6D, 0x15, 0x93, 0xA3, 0xD4, 0x3A, 0x0B, 0x23, 0xAE, 0xB9, 0x47, 0x4F, 0xB4,
    0x04, 0x0A, 0xD1, 0xAF, 0xA9, 0x55, 0x10, 0xED, 0x2E, 0x0F, 0x48, 0x43, 0x0E, 0x52, 0xF9, 0x42,
    0x8C, 0x28, 0x7B, 0xC3, 0xC4, 0xD9, 0x69, 0x03, 0xB9, 0x65, 0xDF, 0x78, 0x37, 0x20, 0x62, 0x55,
    0x14, 0xBD, 0xA9, 0x76, 0xD2, 0x6F, 0xAD, 0xF3, 0xF6, 0xD5, 0x38, 0xAE, 0x1D, 0xDA, 0x86, 0x9A,
    0x7B, 0xF6, 0x5E, 0xFA, 0xF2, 0xA7, 0x19, 0xC3, 0x26, 0xE0, 0x80, 0x63, 0x9D, 0x7A, 0x98, 0x33,
    0x72, 0x66, 0x54, 0x93, 0x95, 0xED, 0x58, 0xCF, 0x68, 0xE5, 0x2B, 0x10, 0x19, 0x8B, 0x65, 0x09,
    0xB1, 0x30, 0xAD, 0x25, 0x20, 0xBE, 0x4E, 0xFF, 0x35, 0x84, 0xE6, 0x99, 0x24, 0x25, 0x2F, 0x48,
    0xF1, 0x3A, 0xA6, 0x52, 0xDC, 0xA9, 0xE2, 0xB3, 0xD0, 0x5A, 0xD0, 0xAC, 0xBB, 0xD8, 0xF7, 0x8F,
    0x6D, 0x4D, 0xA9, 0x46, 0x34, 0x3E, 0x60, 0xC1, 0xFA, 0x53, 0xEE, 0x59, 0xC5, 0xC7, 0x50, 0x07,
    0xC1, 0xAD, 0x4C, 0x27, 0xA1, 0x8E, 0x13, 0x93, 0x57, 0x22, 0xBD, 0x75, 0xF8, 0x50, 0x75, 0x63,
    0x40, 0x3B, 0xD8, 0xA9, 0x65, 0x0B, 0x27, 0x8E, 0xD2, 0x62, 0x8B, 0x9C, 0x34, 0x8B, 0x15, 0x71,
    0x6A, 0x9C, 0x38, 0x41, 0x95, 0x1C, 0xB0, 0x94, 0xCA, 0x50, 0xED, 0xA5, 0x52, 0xC6, 0xFC, 0x47,
    0xD9, 0x50, 0xE6, 0x92, 0xF4, 0xEA, 0x35, 0xBA, 0x55, 0x56, 0xF1, 0x48, 0x5B, 0xEB, 0x9B, 0x8A,
    0x87, 0xA1, 0x2A, 0xEC, 0xC8, 0x80, 0x82, 0x34, 0x0D, 0xFB, 0xF4, 0x6A, 0x89, 0x81, 0x8E, 0x62,
    0x89, 0x14, 0x16, 0x6F, 0x70, 0xCB, 0x43, 0x94, 0x1F, 0x6F, 0x8C, 0xE9, 0x82, 0x2B, 0x5E, 0x86,
    0x01, 0x28, 0x30, 0x30, 0xE8, 0x35, 0xA1, 0x9B, 0xD8, 0xAB, 0xEA, 0xD1, 0x47, 0x0E, 0xB8, 0xBD,
    0x6E, 0x48, 0x33, 0x0A, 0x46, 0xC8, 0x5E, 0xC8, 0x2C, 0x3F, 0xAE, 0xD0, 0x2F, 0x5C, 0x70, 0x71,
    0xC6, 0xA2, 0xDB, 0x2C, 0x74, 0x4B, 0x66, 0xA2, 0x95, 0x36, 0xB0, 0xF9, 0x5A, 0xC6, 0x72, 0xF1,
    0xB4, 0xDD, 0x47, 0x64, 0x40, 0x53, 0x27, 0x23, 0x38, 0x0D, 0x0E, 0x8F, 0xCC, 0x47, 0x9A, 0xBB,
    0x9A, 0x10, 0x94, 0xFF, 0x0C, 0x88, 0x5C, 0xFE, 0x85, 0xB7, 0x21, 0xCF, 0x93, 0x3A, 0xE2, 0x9F,
    0x12, 0x5C, 0xFC, 0xA5, 0xD5, 0xA0, 0x0B, 0x16, 0x7C, 0x48, 0xD2, 0x48, 0x96, 0x57, 0x5D, 0xBF,
    0xDD, 0x80, 0xC7, 0x70, 0x7A, 0xCA, 0xB1, 0x67, 0xF7, 0x0F, 0xE1, 0x88, 0xD7, 0xB5, 0x93, 0x08,
    0xD9, 0xAF, 0x5A, 0x32, 0x6E, 0x8A, 0x31, 0xCB, 0xC4, 0xC5, 0x2D, 0xFB, 0xDA, 0xB0, 0x4F, 0xAA,
    0x0E, 0x96, 0x13, 0x82, 0x4D, 0xBA, 0xE6, 0xBC, 0x96, 0xA4, 0xF9, 0x3F, 0x5F, 0x78, 0xDF, 0x36,
    0x60, 0x73, 0x39, 0xAE, 0x99, 0xFD, 0xD3, 0x52, 0x30, 0x69, 0x09, 0x7B, 0x02, 0x1F, 0x6D, 0x74,
    0x30, 0x8A, 0x43, 0x79, 0xEC, 0x54, 0x7B, 0xB9, 0x8A, 0x39, 0xC4, 0x7A, 0x5C, 0x4F, 0xD8, 0xE6,
    0xA3, 0xBC, 0x95, 0x29, 0x02, 0x2A, 0x21, 0x2C, 0x0F, 0xA8, 0xA7, 0xDF, 0xB8, 0xA0, 0x1F, 0xBF,
    0x52, 0x95, 0x86, 0x9A, 0x4D, 0x52, 0x49, 0x1B, 0x81, 0xD3, 0x93, 0x7E, 0x71, 0x75, 0x4E, 0x7D,
    0x73, 0x95, 0x2A, 0xEB, 0xA3, 0x2A, 0x35, 0xAC, 0x1C, 0xC7, 0x19, 0x11, 0xBC, 0xCA, 0x41, 0x33,
    0xB7, 0x6E, 0x90, 0xF3, 0x1C, 0x45, 0xD2, 0x99, 0x4E, 0x56, 0xD5, 0x2F, 0x4C, 0x2B, 0x91, 0xF2,
    0x6E, 0x5D, 0xBF, 0xA1, 0xC5, 0xA5, 0x56, 0xBD, 0x59, 0x64, 0x4E, 0xF8, 0xBF, 0x0A, 0xE2, 0xF1,
    0xD3, 0xC2, 0x30, 0x1B, 0x42, 0xFC, 0xE8, 0xE4, 0x2A, 0xA1, 0x03, 0xA2, 0x96, 0x64, 0x89, 0xE3,
    0x67, 0xD2, 0xCC, 0x2E, 0x25, 0x99, 0xCE, 0x1C, 0x9E, 0x3C, 0x82, 0xBE, 0x85, 0xAE, 0x95, 0xE2,
    0x05, 0xB7, 0x22, 0x36, 0xA4, 0xA3, 0x3F, 0x19, 0x76, 0xEE, 0xB9, 0x3C, 0xB5, 0x30, 0x6D, 0x85,
    0x2A, 0x77, 0xEA, 0x7F, 0x3E, 0xA1, 0x91, 0x79, 0x2E, 0x68, 0x42, 0x08, 0x46, 0x47, 0xB6, 0x25,
    0xD2, 0xF0, 0xCC, 0x7D, 0x33, 0x05, 0x8C, 0xAC, 0xDC, 0xFF, 0xF2, 0x82, 0x3F, 0xA1, 0xA8, 0x09,
    0x5D, 0x5A, 0xE8, 0x4E, 0x56, 0x48, 0xAC, 0xEF, 0x4D, 0xD5, 0x3D, 0x6F, 0x28, 0x21, 0xA3, 0x6B,
    0x82, 0xFB, 0x9E, 0x28, 0x8D, 0x03, 0x9D, 0x13, 0x15, 0x8B, 0xA0, 0x67, 0x85, 0x18, 0xBC, 0x1A,
    0xA3, 0x54, 0xA5, 0x09, 0xFB, 0x3F, 0xCE, 0x20, 0xAE, 0xCA, 0xB3, 0x38, 0x54, 0x61, 0xBA, 0xC3,
    0xBD, 0x83, 0x6B, 0x05, 0xBA, 0x5F, 0x20, 0xF8, 0xBB, 0x7A, 0xFE, 0x01, 0xFE, 0xD6, 0x6B, 0xF2,
    0x6A, 0xD8, 0x3A, 0x86, 0x9B, 0xA7, 0x3F, 0x0A, 0xCC, 0x8C, 0xAC, 0xD3, 0x94, 0xF2, 0xBC, 0x03,
    0x62, 0x06, 0xE0, 0x83, 0x03, 0x7B, 0xA1, 0x00, 0x4F, 0x99, 0x95, 0xB2, 0x1C, 0x69, 0xAA, 0x74,
    0x74, 0x78, 0x25, 0x44, 0x2A, 0x28, 0x26, 0x06, 0xAB, 0x55, 0xB9, 0x99, 0x3B, 0x0C, 0x64, 0xEE,
    0x71, 0x31, 0x6D, 0x5F, 0x4E, 0xAD, 0x2C, 0xDD, 0xD4, 0x0E, 0xA7, 0x95, 0xFA, 0x46, 0x86, 0xDA,
    0x5F, 0x36, 0x64, 0x0F, 0x7E, 0xB1, 0xF8, 0x3F, 0x89, 0x60, 0xA3, 0x89, 0xFC, 0x67, 0x37, 0x71,
    0x3A, 0xE7, 0x18, 0x80, 0x8B, 0x26, 0xAD, 0x98, 0x80, 0x46, 0x0F, 0x72, 0xF5, 0x57, 0x71, 0x5F,
    0x9E,
];

/// Proprietary engine opcode multiplexed over the binary WebSocket connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum WsEngine {
    /// NVIDIA Grace-Blackwell GB300/GB10 PTX 8.5+ `b1.xor.popc` & `f16` Tensor-Core batch execution.
    Gb300TensorCorePtx = 0x01,
    /// `RealityComber` sparse-gated pyramidal optical flow & static-vs-moving scene separator.
    RealityComber = 0x02,
    /// `Voxelean` 64-bit Zobrist octree $O(\log N)$ volumetric diff & $1+7=8$ Schur-Feshbach wavelets.
    VoxeleanOctree = 0x03,
    /// `(3+1)D Worldtube` Minkowskian causal light-cone evolution & $O(\log_2 k)$ dyadic ray march.
    Worldtube4D = 0x04,
    /// `H3Av4k120` MiniMax H3 parity & superiority 4K60/4K120 Worldtube Geodesic + Spatial Audio + J1.3 Swarm engine.
    H3Av4k120 = 0x05,
    /// `BabyVisionSolver` exact pixel homology ($\beta_0, \beta_1, \chi$) + $SE(2)$ mass-conserving crop-and-zoom.
    BabyVisionSolver = 0x06,
    /// `CharXivPlotDecompiler` scientific plot RGB curve demixer, plateau detector, intersection solver, and integrator.
    CharXivPlotDecompiler = 0x07,
    /// `OSWorldGuiGrounder` deterministic desktop/web GUI action grounder + occlusion gate.
    OSWorldGuiGrounder = 0x08,
    /// `TrackEverything3D` long-horizon de-duplicated 3D scene tracker (`Uniform`/`SignedLog` Zobrist voxelization + 3D WAFT + `Sim(3)` alignment).
    TrackEverything3D = 0x09,
}

impl WsEngine {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Gb300TensorCorePtx),
            0x02 => Some(Self::RealityComber),
            0x03 => Some(Self::VoxeleanOctree),
            0x04 => Some(Self::Worldtube4D),
            0x05 => Some(Self::H3Av4k120),
            0x06 => Some(Self::BabyVisionSolver),
            0x07 => Some(Self::CharXivPlotDecompiler),
            0x08 => Some(Self::OSWorldGuiGrounder),
            0x09 => Some(Self::TrackEverything3D),
            _ => None,
        }
    }

    /// Retail base rate in `micro-USD` (`1 USD = 1,000,000 micro-USD`) per WebSocket RPC invocation,
    /// matching `solidSF` gateway `agentic_credit_ledger` (`credits_delta_microusd`).
    pub const fn base_rate_microusd(&self) -> i64 {
        match self {
            // 25 micro-USD / batch ($0.025 per 1,000 GB300 Tensor-Core batches; 400k batches per $10 pack)
            Self::Gb300TensorCorePtx => 25,
            // 15 micro-USD / frame ($0.015 per 1,000 RealityComber frames; 666.6k frames per $10 pack)
            Self::RealityComber => 15,
            // 5 micro-USD / diff ($0.005 per 1,000 Voxelean 3D octree diffs; 2.0M diffs per $10 pack)
            Self::VoxeleanOctree => 5,
            // 10 micro-USD / step ($0.010 per 1,000 (3+1)D Worldtube steps; 1.0M steps per $10 pack)
            Self::Worldtube4D => 10,
            // 20 micro-USD / packet ($0.020 per 1,000 H3 4K60/4K120 AV packets; 500k packets per $10 pack)
            Self::H3Av4k120 => 20,
            // 8 micro-USD / solve ($0.008 per 1,000 BabyVision topology solves; 1.25M solves per $10 pack)
            Self::BabyVisionSolver => 8,
            // 12 micro-USD / chart ($0.012 per 1,000 CharXiv plot decompiles; 833.3k charts per $10 pack)
            Self::CharXivPlotDecompiler => 12,
            // 6 micro-USD / ground ($0.006 per 1,000 OSWorld GUI groundings; 1.66M groundings per $10 pack)
            Self::OSWorldGuiGrounder => 6,
            // 14 micro-USD / window ($0.014 per 1,000 TrackEverything3D windows; 714.2k windows per $10 pack)
            Self::TrackEverything3D => 14,
        }
    }
}

/// Retail debit in `micro-USD`: raw provider/cluster cost marked up by `markup_percent` (default 30%),
/// rounded up to the next integer `micro-USD` so fractional dust never accrues against the platform.
/// Matches `backend/gateway/src/agentic_credits.rs::retail_debit_microusd`.
pub fn retail_debit_microusd(provider_cost_usd: f64, markup_percent: i64) -> Option<i64> {
    if !provider_cost_usd.is_finite() || provider_cost_usd <= 0.0 {
        return None;
    }
    let scaled = provider_cost_usd * 1_000_000.0 * (100 + markup_percent.clamp(0, 400)) as f64 / 100.0;
    if !scaled.is_finite() || scaled > i64::MAX as f64 {
        return None;
    }
    Some((scaled.ceil() as i64).max(1))
}

/// Convert whole-dollar purchase cents (`$10..$1,000` self-serve or `$25..$100,000` org) to `micro-USD`.
pub fn purchase_cents_to_microusd(amount_cents: i32) -> Result<i64> {
    if !(FAH_STORE_MIN_PURCHASE_CENTS..=10_000_000).contains(&amount_cents) || amount_cents % 100 != 0 {
        return Err(Error::new(
            STS_ERROR,
            format!(
                "solidSF Credits purchase must be whole dollars between ${} and ${}",
                FAH_STORE_MIN_PURCHASE_CENTS / 100,
                10_000_000 / 100
            ),
        ));
    }
    Ok(i64::from(amount_cents) * 10_000)
}

/// Keystream generator used to authenticate the binary WebSocket vault seal.
#[inline]
pub fn zobrist_feistel_keystream_byte(master_seed: u64, idx: usize) -> u8 {
    let mut z = master_seed ^ ((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z & 0xFF) as u8
}

/// Decrypt the embedded vault seal with a 64-bit key.
pub fn unlock_vault_seal(master_seed: u64) -> String {
    let bytes: Vec<u8> = SOLIDSF_ENCRYPTED_BOUNTY_BLOB
        .iter()
        .enumerate()
        .map(|(i, &b)| b ^ zobrist_feistel_keystream_byte(master_seed, i))
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Binary WebSocket request envelope sent to `wss://api.solidsf.com/v1/solidcv/ws`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsRpcRequest {
    pub engine: WsEngine,
    pub operation: String,
    pub payload_json: String,
}

/// Binary WebSocket response envelope returned by `wss://api.solidsf.com/v1/solidcv/ws`,
/// carrying real-time `solidSF Credits` ledger debit and remaining balance in `micro-USD`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsRpcResponse {
    pub ok: bool,
    pub engine: WsEngine,
    pub server_exec_us: f64,
    /// Credits debited for this call in `micro-USD` (`1 USD = 1,000,000 micro-USD`; `0` for `compute_owned`).
    pub credits_charged_microusd: i64,
    /// Caller's remaining `agentic_credit_ledger` balance in `micro-USD` after this call.
    pub credits_remaining_microusd: i64,
    /// `"debited"`, `"compute_owned"` (internal `solidsf` org), or `"payment_required_402"`.
    pub billing_status: String,
    pub result_json: String,
}

/// Encode a client-to-server RFC-6455 binary WebSocket frame (`FIN=1`, `Opcode=0x02`, `MASK=1`).
pub fn encode_rfc6455_binary_frame(payload: &[u8], mask_key: [u8; 4]) -> Vec<u8> {
    let len = payload.len();
    let mut frame = Vec::with_capacity(14 + len);
    frame.push(0x82); // FIN=1, Opcode=2 (Binary)
    if len <= 125 {
        frame.push(0x80 | (len as u8));
    } else if len <= 65535 {
        frame.push(0x80 | 126);
        frame.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        frame.push(0x80 | 127);
        frame.extend_from_slice(&(len as u64).to_be_bytes());
    }
    frame.extend_from_slice(&mask_key);
    for (i, &b) in payload.iter().enumerate() {
        frame.push(b ^ mask_key[i & 3]);
    }
    frame
}

/// Decode an RFC-6455 binary WebSocket frame from a stream.
pub fn decode_rfc6455_binary_frame<R: Read>(reader: &mut R) -> Result<Vec<u8>> {
    let mut hdr = [0u8; 2];
    reader.read_exact(&mut hdr).map_err(|e| {
        Error::new(
            STS_ERROR,
            format!("solidcv WebSocket frame header read error: {e}"),
        )
    })?;
    let masked = (hdr[1] & 0x80) != 0;
    let len_code = (hdr[1] & 0x7F) as usize;
    let payload_len = if len_code <= 125 {
        len_code
    } else if len_code == 126 {
        let mut b = [0u8; 2];
        reader
            .read_exact(&mut b)
            .map_err(|e| Error::new(STS_ERROR, format!("WS 16-bit length read error: {e}")))?;
        u16::from_be_bytes(b) as usize
    } else {
        let mut b = [0u8; 8];
        reader
            .read_exact(&mut b)
            .map_err(|e| Error::new(STS_ERROR, format!("WS 64-bit length read error: {e}")))?;
        u64::from_be_bytes(b) as usize
    };

    let mut mask_key = [0u8; 4];
    if masked {
        reader
            .read_exact(&mut mask_key)
            .map_err(|e| Error::new(STS_ERROR, format!("WS mask key read error: {e}")))?;
    }

    let mut payload = vec![0u8; payload_len];
    reader
        .read_exact(&mut payload)
        .map_err(|e| Error::new(STS_ERROR, format!("WS payload read error: {e}")))?;
    if masked {
        for (i, b) in payload.iter_mut().enumerate() {
            *b ^= mask_key[i & 3];
        }
    }
    Ok(payload)
}

/// Synchronous RFC-6455 Binary WebSocket Client for `solidSF` Cloud IP Engines.
pub struct SolidCvWsClient {
    endpoint: String,
    stream: TcpStream,
}

impl SolidCvWsClient {
    /// Connect to a `ws://host:port/path` endpoint (or `SOLIDCV_WS_URL`) and complete the
    /// RFC-6455 HTTP/1.1 `Upgrade: websocket` handshake with the `X-SolidSF-Vault-Seal` header.
    pub fn connect(url: &str, api_key: Option<&str>) -> Result<Self> {
        let stripped = url
            .strip_prefix("ws://")
            .or_else(|| url.strip_prefix("wss://"))
            .unwrap_or(url);
        let (host_port, path) = match stripped.find('/') {
            Some(idx) => (&stripped[..idx], &stripped[idx..]),
            None => (stripped, "/v1/solidcv/ws"),
        };

        let mut stream = TcpStream::connect(host_port).map_err(|e| {
            Error::new(
                STS_ERROR,
                format!("Failed to connect to solidSF WebSocket endpoint {host_port}: {e}"),
            )
        })?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));

        let auth_line = match api_key {
            Some(k) => format!("Authorization: Bearer {k}\r\n"),
            None => String::new(),
        };
        let req = format!(
            "GET {path} HTTP/1.1\r\n\
             Host: {host_port}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
             Sec-WebSocket-Version: 13\r\n\
             X-SolidSF-Vault-Seal: 534F4C4944534621\r\n\
             {auth_line}\r\n"
        );
        stream
            .write_all(req.as_bytes())
            .map_err(|e| Error::new(STS_ERROR, format!("WS handshake write failed: {e}")))?;

        let mut resp_buf = Vec::with_capacity(256);
        let mut byte = [0u8; 1];
        while !resp_buf.ends_with(b"\r\n\r\n") {
            stream
                .read_exact(&mut byte)
                .map_err(|e| Error::new(STS_ERROR, format!("WS handshake read failed: {e}")))?;
            resp_buf.push(byte[0]);
            if resp_buf.len() > 4096 {
                return Err(Error::new(STS_ERROR, "WS handshake header exceeded 4096B"));
            }
        }
        let header_str = String::from_utf8_lossy(&resp_buf);
        if !header_str.starts_with("HTTP/1.1 101") {
            return Err(Error::new(
                STS_ERROR,
                format!("WebSocket upgrade rejected: {header_str}"),
            ));
        }

        Ok(Self {
            endpoint: url.to_string(),
            stream,
        })
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Execute a binary RPC call over the active WebSocket stream.
    pub fn call(&mut self, request: &WsRpcRequest) -> Result<WsRpcResponse> {
        let mut wire = Vec::with_capacity(64 + request.payload_json.len());
        wire.extend_from_slice(&WS_FRAME_MAGIC);
        wire.push(request.engine as u8);
        let json_bytes = serde_json::to_vec(request)
            .map_err(|e| Error::new(STS_ERROR, format!("WS serialize error: {e}")))?;
        wire.extend_from_slice(&json_bytes);

        let frame = encode_rfc6455_binary_frame(&wire, [0x53, 0x53, 0x46, 0x21]);
        self.stream
            .write_all(&frame)
            .map_err(|e| Error::new(STS_ERROR, format!("WS frame write error: {e}")))?;

        let resp_wire = decode_rfc6455_binary_frame(&mut self.stream)?;
        if resp_wire.len() < 7 || resp_wire[0..6] != WS_FRAME_MAGIC {
            return Err(Error::new(
                STS_ERROR,
                "Invalid SSFCV\\x01 binary preamble in WebSocket response",
            ));
        }
        let resp: WsRpcResponse = serde_json::from_slice::<WsRpcResponse>(&resp_wire[7..])
            .map_err(|e| Error::new(STS_ERROR, format!("WS response deserialize error: {e}")))?;
        if !resp.ok && resp.billing_status == "payment_required_402" {
            return Err(Error::new(
                402,
                format!(
                    "402 Payment Required: solidSF Credits balance ({} micro-USD) insufficient for {:?} ({} micro-USD). Refill at {}",
                    resp.credits_remaining_microusd,
                    resp.engine,
                    resp.engine.base_rate_microusd(),
                    CREDITS_STORE_URL
                ),
            ));
        }
        Ok(resp)
    }
}

/// Ephemeral loopback RFC-6455 WebSocket server for local protocol verification and unit tests.
pub struct SolidCvLoopbackWsServer {
    pub ws_url: String,
    handle: Option<thread::JoinHandle<()>>,
}

impl SolidCvLoopbackWsServer {
    /// Spawn a single-session loopback WebSocket server on `127.0.0.1:0` seeded with the
    /// default `$10.00` welcome credit balance (`WELCOME_CREDITS_MICROUSD = 10_000_000`).
    pub fn spawn_ephemeral(n_requests: usize) -> Result<Self> {
        Self::spawn_with_balance(n_requests, WELCOME_CREDITS_MICROUSD)
    }

    /// Spawn a single-session loopback WebSocket server on `127.0.0.1:0` with an explicit
    /// starting `solidSF Credits` balance in `micro-USD`.
    pub fn spawn_with_balance(n_requests: usize, initial_balance_microusd: i64) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| Error::new(STS_ERROR, format!("Loopback WS bind failed: {e}")))?;
        let addr = listener
            .local_addr()
            .map_err(|e| Error::new(STS_ERROR, format!("Loopback WS local_addr failed: {e}")))?;
        let ws_url = format!("ws://{addr}/v1/solidcv/ws");

        let handle = thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let _ = stream.set_nodelay(true);
                let mut hdr = Vec::with_capacity(256);
                let mut b = [0u8; 1];
                while !hdr.ends_with(b"\r\n\r\n") {
                    if stream.read_exact(&mut b).is_err() {
                        return;
                    }
                    hdr.push(b[0]);
                }
                let hdr_str = String::from_utf8_lossy(&hdr);
                let is_internal_unmetered = hdr_str.contains("Bearer ssf_internal_");

                let upgrade_resp = "HTTP/1.1 101 Switching Protocols\r\n\
                                    Upgrade: websocket\r\n\
                                    Connection: Upgrade\r\n\
                                    Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n";
                if stream.write_all(upgrade_resp.as_bytes()).is_err() {
                    return;
                }

                let mut balance_microusd = initial_balance_microusd;

                for _ in 0..n_requests {
                    let Ok(wire) = decode_rfc6455_binary_frame(&mut stream) else {
                        break;
                    };
                    if wire.len() < 7 || wire[0..6] != WS_FRAME_MAGIC {
                        break;
                    }
                    let engine = WsEngine::from_u8(wire[6]).unwrap_or(WsEngine::RealityComber);
                    let req: WsRpcRequest = match serde_json::from_slice(&wire[7..]) {
                        Ok(r) => r,
                        Err(_) => break,
                    };
                    let rate = engine.base_rate_microusd();
                    let resp = if is_internal_unmetered {
                        WsRpcResponse {
                            ok: true,
                            engine,
                            server_exec_us: 1.95,
                            credits_charged_microusd: 0,
                            credits_remaining_microusd: balance_microusd,
                            billing_status: "compute_owned".to_string(),
                            result_json: format!(
                                "{{\"status\":\"executed_on_solidsf_cloud\",\"operation\":{:?}}}",
                                req.operation
                            ),
                        }
                    } else if balance_microusd >= rate && balance_microusd > 0 {
                        balance_microusd -= rate;
                        WsRpcResponse {
                            ok: true,
                            engine,
                            server_exec_us: 1.95,
                            credits_charged_microusd: rate,
                            credits_remaining_microusd: balance_microusd,
                            billing_status: "debited".to_string(),
                            result_json: format!(
                                "{{\"status\":\"executed_on_solidsf_cloud\",\"operation\":{:?}}}",
                                req.operation
                            ),
                        }
                    } else {
                        WsRpcResponse {
                            ok: false,
                            engine,
                            server_exec_us: 0.0,
                            credits_charged_microusd: 0,
                            credits_remaining_microusd: balance_microusd,
                            billing_status: "payment_required_402".to_string(),
                            result_json: format!(
                                "{{\"error\":\"payment_required\",\"store_url\":{:?}}}",
                                CREDITS_STORE_URL
                            ),
                        }
                    };
                    let Ok(resp_json) = serde_json::to_vec(&resp) else {
                        break;
                    };
                    let mut out_wire = Vec::with_capacity(7 + resp_json.len());
                    out_wire.extend_from_slice(&WS_FRAME_MAGIC);
                    out_wire.push(engine as u8);
                    out_wire.extend_from_slice(&resp_json);

                    // Server-to-client RFC-6455 binary frame (MASK=0)
                    let mut frame = Vec::with_capacity(10 + out_wire.len());
                    frame.push(0x82);
                    if out_wire.len() <= 125 {
                        frame.push(out_wire.len() as u8);
                    } else {
                        frame.push(126);
                        frame.extend_from_slice(&(out_wire.len() as u16).to_be_bytes());
                    }
                    frame.extend_from_slice(&out_wire);
                    let _ = stream.write_all(&frame);
                }
            }
        });

        Ok(Self {
            ws_url,
            handle: Some(handle),
        })
    }

    pub fn join(mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
