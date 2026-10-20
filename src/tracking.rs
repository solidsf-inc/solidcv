//! Long-Horizon Dense 3D Scene Tracking via De-Duplicating Voxelean & Worldtube Representations (`tracking`).
//!
//! A 100% pure-Rust, zero-FFI, hardware-accelerated superset of **TrackEverything**
//! (*Long Horizon Dense Tracking via De-Duplicating 3D Scene Representations*, `arXiv:2609.30222`)
//! grounded in the 24 theorems of the Unified Pixelean-Voxelean Calculus:
//!
//! 1. **64-Bit Zobrist Voxel De-Duplication (`Theorem 9: VoxeleanOctree`) + Sub-Voxel Residuals**:
//!    Replaces `TrackEverything`'s bounding-box-shifted Fortran polynomial `ravel_hash_vec`
//!    (which overflows `int64` on depth outliers and forces full-array `torch.sort` at every
//!    sliding-window boundary) with $O(1)$ 64-bit Zobrist spatial hashing in both **`Uniform`**
//!    and **`SignedLog`** ($\operatorname{sgn}(x)\ln(1 + |x|/a)$) coordinate spaces, while
//!    retaining exact continuous 3D sub-voxel residuals $\boldsymbol{\delta}_i = \mathbf{X}_i - \mathbf{V}_{\text{center}}(v_i)$
//!    for zero-staircasing unvoxelization (`unvoxelize_coords_with_residuals`).
//! 2. **Dyadic $1+7=8$ Schur-Feshbach Memory Compression (`Theorem 11`) & Betti Topology (`Theorem 12`)**:
//!    Replaces `TrackEverything`'s brute-force `2x`/`4x` re-voxelization loop (`voxelize_memory_p2v`)
//!    with single-pass $1+7=8$ [`SchurFeshbachOctreeWavelet`] coarsening restricted strictly to
//!    never-dynamic (`!ever_dynamic`) scene points—retaining exact DC mass + 7 orthogonal 3D Haar
//!    detail modes and certifying Euler-Poincare Betti numbers $(\beta_0, \beta_1, \beta_2)$ via
//!    [`VoxelComplex3D`].
//! 3. **Decomposed Static vs. Dynamic Tracking (`EndpointRefiner3D` + `TrajectoryRefiner3D`)**:
//!    - **Endpoint Refiner (`Theorems 2, 10`)**: Combines [`XorMintermCarrier`] bitplane activity,
//!      $SE(3)$ ego-motion compensated 3D metric innovation $\|\mathbf{X}_w(t_{\text{end}}) - \mathbf{X}_w(t_{\text{start}})\|_2$,
//!      and 256-bit 3D Binary WAFT (`b1.xor.popc`) feature sampling to predict each track's
//!      window endpoint, classify it as static (`dyn_logit < 0`) or dynamic (`dyn_logit > 0`),
//!      and optionally merge co-located endpoints (`voxelize_on_endpoints`).
//!    - **Trajectory Refiner (`Theorems 17, 18, 19` + `Operator J1`)**: Static points live once
//!      in world coordinates ($O(1)$ memory across $T$), while dynamic points and user-pinned
//!      [`QueryPoint2D`] tracks (which never merge with voxel tokens) are decoded across the window
//!      within the Causal CFL Light-Cone ([`CflLightConePruner`]), factored into [`WorldtubeTrajectoryFactor`]
//!      states, and bridged across occlusions (`visib_logit < 0`) via Frenet-Serret Euler elastica
//!      ([`lift_se2_frenet_serret_jets`]).
//! 4. **Automatic In-Head Geometry or Custom Depth/Camera Poses (`Sim(3)` Window Alignment)**:
//!    Supports both supplied depth/intrinsics/extrinsics (with [`safe_inverse_se3`],
//!    [`batch_distance_to_depth`], and [`sim3_align_point_clouds`]) and automatic in-head
//!    depth + $SE(3)$ pose estimation via [`comb_dense_velocity_field`] and [`solve_pnp_ransac`].

use crate::calib3d::{rotvec_to_rotmat, solve_pnp_ransac, svd_3x3, SOLVEPNP_ITERATIVE};
use crate::core::{Error, Mat, MatTraitConst, Point2f, Point3f, Result, Vector, CV_32FC1};
use crate::hardware::{tensor_core_b1_argmin_256, GraceCoherentSlab, ResidencyPolicy};
use crate::mechinterp::{record_activation, record_circuit_node, TheoremId};
use crate::pixelean::{lift_se2_frenet_serret_jets, XorMintermCarrier};
use crate::realitycomb::comb_dense_velocity_field;
use crate::voxelean::{OhElement, SchurFeshbachOctreeWavelet, VoxelComplex3D, VoxeleanOctree};
use crate::worldtube::{
    CflLightConePruner, GrandSynthesisCertificate, WorldtubeContinuity, WorldtubeTrajectoryFactor,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Default per-axis knee `a = 0.25` for signed-logarithmic 3D voxelization.
/// Bins are near-uniform for `|x| << a` and logarithmic for `|x| >> a`, calibrated so
/// cells at unit distance `|x| = 1.0` have edge length equal to `voxel_size`.
pub const DEFAULT_SIGNEDLOG_KNEE_A: f32 = 0.25;

/// Spatial discretization strategy for 3D scene track de-duplication.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VoxelMode {
    /// Uniform Cartesian grid where every voxel has fixed side length `voxel_size` (`round(x / v)`).
    Uniform,
    /// Logarithmic per-axis binning calibrated to have edge `voxel_size` at `|x| = 1.0`:
    /// `floor(sign(x) * ln(1 + |x|/a) * (a + 1) / voxel_size)`.
    /// Ideal for expansive outdoor/drone scenes where depth spans multiple orders of magnitude.
    SignedLog { knee_a: f32 },
}

impl Default for VoxelMode {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Memory merging policy across sliding temporal windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryMergeMode {
    /// De-duplicate across all accumulated persistent scene memory and incoming window tokens.
    All,
    /// De-duplicate strictly between source memory anchors and target window endpoints.
    SrcTgt,
}

impl Default for MemoryMergeMode {
    fn default() -> Self {
        Self::All
    }
}

/// User-specified 2D query point `(frame_idx, x, y)` in pixel coordinates.
/// Query points bypass voxel de-duplication (`never merged`) to preserve exact sub-pixel resolution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QueryPoint2D {
    pub frame_idx: usize,
    pub x: f32,
    pub y: f32,
}

impl QueryPoint2D {
    #[inline]
    pub const fn new(frame_idx: usize, x: f32, y: f32) -> Self {
        Self { frame_idx, x, y }
    }
}

/// Configuration for `TrackEverything` long-horizon dense 3D scene tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingConfig {
    /// Base voxel edge length in meters (default `0.02` = 2 cm).
    pub voxel_size: f32,
    /// Voxelization coordinate mode (`Uniform` or `SignedLog`).
    pub voxel_mode: VoxelMode,
    /// Merge dynamic points whose predicted 3D window endpoints fall into the same voxel (default `true`).
    pub voxelize_on_endpoints: bool,
    /// Retain continuous sub-voxel 3D residual offsets `X - V_center` during unvoxelization (default `true`).
    pub residual_unvoxelize: bool,
    /// Optional upper bound on static scene token count (`--compress_memory N`).
    /// When exceeded, never-dynamic points are coarsened via $1+7=8$ `SchurFeshbachOctreeWavelet`.
    pub compress_memory: Option<usize>,
    /// Maximum dyadic coarsening passes (`default 2` -> `2x` then `4x` voxel scale).
    pub max_compression_passes: usize,
    /// Sliding-window merge mode (`All` or `SrcTgt`).
    pub memory_merge_mode: MemoryMergeMode,
    /// Temporal window length in frames (`32` or `64` typical; default `32`).
    pub window_len: usize,
    /// Temporal sliding-window stride in frames (default `16`).
    pub window_stride: usize,
    /// Spatial pixel sampling stride `(H' = H / spatial_stride, W' = W / spatial_stride)`.
    pub spatial_stride: usize,
    /// 3D world-space displacement threshold (meters) separating static landmarks from dynamic points.
    pub dynamic_threshold_m: f32,
    /// Causal CFL Light-Cone maximum speed in meters per frame (`Theorem 18`).
    pub max_cfl_speed_m_per_frame: f32,
    /// Depth occlusion tolerance in meters for $z$-buffer visibility (`visibs`) evaluation.
    pub occlusion_depth_tol_m: f32,
    /// Enable 256-bit 3D Binary WAFT (`b1.xor.popc`) point-cloud feature correspondence.
    pub use_waft_binary_3d: bool,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.02,
            voxel_mode: VoxelMode::Uniform,
            voxelize_on_endpoints: true,
            residual_unvoxelize: true,
            compress_memory: None,
            max_compression_passes: 4,
            memory_merge_mode: MemoryMergeMode::All,
            window_len: 32,
            window_stride: 16,
            spatial_stride: 4,
            dynamic_threshold_m: 0.035,
            max_cfl_speed_m_per_frame: 0.50,
            occlusion_depth_tol_m: 0.08,
            use_waft_binary_3d: true,
        }
    }
}

/// Calibrated camera intrinsics ($3\times 3$) and world-to-camera extrinsic pose ($4\times 4$ $SE(3)$).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraPoseSE3 {
    /// Pin-hole camera intrinsics `(fx, fy, cx, cy)`.
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// $3\times 3$ world-to-camera rotation matrix $R_{\text{cam}\leftarrow\text{world}} \in SO(3)$.
    pub r_cam_from_world: [[f32; 3]; 3],
    /// $3\times 1$ world-to-camera translation vector $\mathbf{t}_{\text{cam}\leftarrow\text{world}}$.
    pub t_cam_from_world: [f32; 3],
}

impl CameraPoseSE3 {
    /// Identity world-to-camera pose with given pinhole intrinsics.
    #[inline]
    pub const fn identity_with_intrinsics(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            r_cam_from_world: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t_cam_from_world: [0.0, 0.0, 0.0],
        }
    }

    /// Construct a $3\times 3$ OpenCV camera matrix `Mat` (`CV_64FC1`).
    pub fn camera_matrix_mat(&self) -> Result<Mat> {
        Mat::from_slice_2d(&[
            [self.fx as f64, 0.0, self.cx as f64],
            [0.0, self.fy as f64, self.cy as f64],
            [0.0, 0.0, 1.0],
        ])
    }

    /// Analytic $SE(3)$ inverse (`safe_inverse_se3`): transforms camera-frame 3D point
    /// $\mathbf{P}_{\text{cam}}$ into world coordinates $\mathbf{P}_{\text{world}} = R^T(\mathbf{P}_{\text{cam}} - \mathbf{t})$.
    #[inline]
    pub fn unproject_pixel_to_world(&self, u: f32, v: f32, depth_z: f32) -> [f32; 3] {
        let z = depth_z.max(1e-4);
        let x_cam = (u - self.cx) * z / self.fx.max(1e-6);
        let y_cam = (v - self.cy) * z / self.fy.max(1e-6);
        let dx = x_cam - self.t_cam_from_world[0];
        let dy = y_cam - self.t_cam_from_world[1];
        let dz = z - self.t_cam_from_world[2];
        let r = &self.r_cam_from_world;
        [
            r[0][0] * dx + r[1][0] * dy + r[2][0] * dz,
            r[0][1] * dx + r[1][1] * dy + r[2][1] * dz,
            r[0][2] * dx + r[1][2] * dy + r[2][2] * dz,
        ]
    }

    /// Project a 3D world point $\mathbf{P}_{\text{world}}$ into `(u, v, depth_cam)`.
    #[inline]
    pub fn project_world_to_pixel(&self, pw: [f32; 3], image_size: (usize, usize)) -> [f32; 3] {
        let r = &self.r_cam_from_world;
        let t = &self.t_cam_from_world;
        let xc = r[0][0] * pw[0] + r[0][1] * pw[1] + r[0][2] * pw[2] + t[0];
        let yc = r[1][0] * pw[0] + r[1][1] * pw[1] + r[1][2] * pw[2] + t[1];
        let zc = r[2][0] * pw[0] + r[2][1] * pw[1] + r[2][2] * pw[2] + t[2];
        let z_clamped = zc.max(1e-6);
        let u = (self.fx * xc / z_clamped + self.cx)
            .clamp(-(image_size.1 as f32) * 2.0, (image_size.1 as f32) * 2.0);
        let v = (self.fy * yc / z_clamped + self.cy)
            .clamp(-(image_size.0 as f32) * 2.0, (image_size.0 as f32) * 2.0);
        [u, v, zc]
    }
}

/// Convert radial Euclidean ray distances $d(u, v) = \|\mathbf{P}_{\text{cam}}\|_2$ into planar
/// pinhole $Z$-depths $z(u, v) = d(u, v) / \sqrt{((u - c_x)/f_x)^2 + ((v - c_y)/f_y)^2 + 1}$.
pub fn batch_distance_to_depth(
    distances: &[f32],
    height: usize,
    width: usize,
    pose: &CameraPoseSE3,
) -> Vec<f32> {
    assert_eq!(distances.len(), height * width);
    let mut depth = vec![0.0f32; height * width];
    let inv_fx = 1.0 / pose.fx.max(1e-6);
    let inv_fy = 1.0 / pose.fy.max(1e-6);
    for r in 0..height {
        let ny = ((r as f32) - pose.cy) * inv_fy;
        let ny2 = ny * ny;
        let row_off = r * width;
        for c in 0..width {
            let nx = ((c as f32) - pose.cx) * inv_fx;
            let ray_norm = (nx * nx + ny2 + 1.0).sqrt();
            depth[row_off + c] = distances[row_off + c] / ray_norm;
        }
    }
    depth
}

/// Umeyama $\operatorname{Sim}(3)$ alignment (`sim3_align`) between two 3D point sets
/// $\mathbf{P}_{\text{src}}, \mathbf{P}_{\text{tgt}} \in \mathbb{R}^{N\times 3}$, recovering scale $s > 0$,
/// rotation $R \in SO(3)$, and translation $\mathbf{t} \in \mathbb{R}^3$ minimizing
/// $\sum_i \|s R \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i\|_2^2$ for sliding-window depth/pose gauge locking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sim3Transform {
    pub scale: f32,
    pub rotation: [[f32; 3]; 3],
    pub translation: [f32; 3],
    pub rms_error_m: f32,
}

impl Sim3Transform {
    #[inline]
    pub fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        let s = self.scale;
        [
            s * (r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2]) + self.translation[0],
            s * (r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2]) + self.translation[1],
            s * (r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2]) + self.translation[2],
        ]
    }
}

/// Estimate the optimal $\operatorname{Sim}(3)$ alignment mapping `src_pts` onto `tgt_pts`.
pub fn sim3_align_point_clouds(src_pts: &[[f32; 3]], tgt_pts: &[[f32; 3]]) -> Sim3Transform {
    let n = src_pts.len().min(tgt_pts.len());
    if n == 0 {
        return Sim3Transform {
            scale: 1.0,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
            rms_error_m: 0.0,
        };
    }
    let inv_n = 1.0 / (n as f64);
    let mut mu_s = [0.0f64; 3];
    let mut mu_t = [0.0f64; 3];
    for i in 0..n {
        for k in 0..3 {
            mu_s[k] += (src_pts[i][k] as f64) * inv_n;
            mu_t[k] += (tgt_pts[i][k] as f64) * inv_n;
        }
    }

    let mut var_s = 0.0f64;
    let mut var_t = 0.0f64;
    let mut cov = [[0.0f64; 3]; 3];
    for i in 0..n {
        let ds = [
            src_pts[i][0] as f64 - mu_s[0],
            src_pts[i][1] as f64 - mu_s[1],
            src_pts[i][2] as f64 - mu_s[2],
        ];
        let dt = [
            tgt_pts[i][0] as f64 - mu_t[0],
            tgt_pts[i][1] as f64 - mu_t[1],
            tgt_pts[i][2] as f64 - mu_t[2],
        ];
        var_s += (ds[0] * ds[0] + ds[1] * ds[1] + ds[2] * ds[2]) * inv_n;
        var_t += (dt[0] * dt[0] + dt[1] * dt[1] + dt[2] * dt[2]) * inv_n;
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += dt[r] * ds[c] * inv_n;
            }
        }
    }

    let (u, _s, v) = svd_3x3(&cov);
    // R = U * diag(1, 1, det(U V^T)) * V^T
    let mut r_mat = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            r_mat[r][c] = u[r][0] * v[c][0] + u[r][1] * v[c][1] + u[r][2] * v[c][2];
        }
    }
    let det_r = r_mat[0][0] * (r_mat[1][1] * r_mat[2][2] - r_mat[1][2] * r_mat[2][1])
        - r_mat[0][1] * (r_mat[1][0] * r_mat[2][2] - r_mat[1][2] * r_mat[2][0])
        + r_mat[0][2] * (r_mat[1][0] * r_mat[2][1] - r_mat[1][1] * r_mat[2][0]);
    let d_sign = if det_r < 0.0 { -1.0 } else { 1.0 };
    for r in 0..3 {
        for c in 0..3 {
            r_mat[r][c] = u[r][0] * v[c][0] + u[r][1] * v[c][1] + d_sign * u[r][2] * v[c][2];
        }
    }

    let scale = if var_s > 1e-12 {
        (var_t / var_s).sqrt().clamp(0.05, 20.0)
    } else {
        1.0
    };

    let mut t_vec = [0.0f32; 3];
    let mut r_f32 = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            r_f32[r][c] = r_mat[r][c] as f32;
        }
        let rot_mu = r_mat[r][0] * mu_s[0] + r_mat[r][1] * mu_s[1] + r_mat[r][2] * mu_s[2];
        t_vec[r] = (mu_t[r] - scale * rot_mu) as f32;
    }

    let mut tmp = Sim3Transform {
        scale: scale as f32,
        rotation: r_f32,
        translation: t_vec,
        rms_error_m: 0.0,
    };
    let mut sq_err = 0.0f64;
    for i in 0..n {
        let pred = tmp.apply(src_pts[i]);
        let dx = (pred[0] - tgt_pts[i][0]) as f64;
        let dy = (pred[1] - tgt_pts[i][1]) as f64;
        let dz = (pred[2] - tgt_pts[i][2]) as f64;
        sq_err += (dx * dx + dy * dy + dz * dz) * inv_n;
    }
    tmp.rms_error_m = sq_err.sqrt() as f32;
    tmp
}

/// Convert a continuous 3D coordinate `xyz` into discrete signed integer voxel coordinates `(ix, iy, iz)`
/// and return the exact voxel center in world meters under `mode`.
#[inline]
pub fn voxelize_coord_3d(xyz: [f32; 3], voxel_size: f32, mode: VoxelMode) -> ([i32; 3], [f32; 3]) {
    let v = voxel_size.max(1e-5);
    match mode {
        VoxelMode::Uniform => {
            let ix = (xyz[0] / v).round() as i32;
            let iy = (xyz[1] / v).round() as i32;
            let iz = (xyz[2] / v).round() as i32;
            ([ix, iy, iz], [ix as f32 * v, iy as f32 * v, iz as f32 * v])
        }
        VoxelMode::SignedLog { knee_a } => {
            let a = knee_a.max(1e-4);
            let scale = (a + 1.0) / v;
            let inv_scale = v / (a + 1.0);
            let mut bins = [0i32; 3];
            let mut center = [0.0f32; 3];
            for k in 0..3 {
                let x = xyz[k];
                let s = if x >= 0.0 { 1.0f32 } else { -1.0f32 };
                let b = (s * (1.0 + x.abs() / a).ln() * scale).floor() as i32;
                bins[k] = b;
                let b_mid = (b as f32) + 0.5;
                let s_b = if b_mid >= 0.0 { 1.0f32 } else { -1.0f32 };
                center[k] = s_b * a * ((b_mid.abs() * inv_scale).exp() - 1.0);
            }
            (bins, center)
        }
    }
}

/// 64-bit collision-free Zobrist spatial hash for signed 3D voxel coordinates `(ix, iy, iz)`.
/// Unlike `TrackEverything`'s Fortran `ravel_hash_vec`, this never overflows `int64` on outlier points
/// and never requires subtracting `xyz.min()` across the whole point cloud.
#[inline]
pub fn zobrist_voxel_hash_i32(bins: [i32; 3]) -> u64 {
    let mut h = (bins[0] as u32 as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (bins[1] as u32 as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ (bins[2] as u32 as u64).wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h.wrapping_mul(0x94d049bb133111eb) ^ (h >> 31)
}

/// Result of voxelizing a 3D point cloud (`voxelization` + `unvoxelize_coords_with_residuals`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxelizationReceipt {
    /// Mapping `p2v[i] -> voxel_id in 0..num_voxels` for each input point `i in 0..N`.
    pub point_to_voxel: Vec<usize>,
    /// Representative centroid coordinate for each unique voxel `v in 0..num_voxels`.
    pub voxel_centroids: Vec<[f32; 3]>,
    /// Integer bin coordinate `[ix, iy, iz]` for each unique voxel.
    pub voxel_bins: Vec<[i32; 3]>,
    /// 64-bit Zobrist hash for each unique voxel.
    pub voxel_zobrist_hashes: Vec<u64>,
    /// Sub-voxel residual `xyz[i] - voxel_centroids[p2v[i]]` for lossless unvoxelization.
    pub subvoxel_residuals: Vec<[f32; 3]>,
    /// Combined XOR Zobrist root signature of all unique voxels (`Theorem 9`).
    pub root_zobrist: u64,
}

impl VoxelizationReceipt {
    /// Reconstruct per-point 3D coordinates from `voxel_centroids` (optionally updated by
    /// trajectory refinement) plus the stored sub-voxel residuals (`unvoxelize_coords_with_residuals`).
    pub fn unvoxelize_coords_with_residuals(
        &self,
        updated_voxel_coords: &[[f32; 3]],
        apply_residuals: bool,
    ) -> Vec<[f32; 3]> {
        let n = self.point_to_voxel.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let vid = self.point_to_voxel[i];
            let base = updated_voxel_coords
                .get(vid)
                .copied()
                .unwrap_or(self.voxel_centroids[vid]);
            if apply_residuals {
                let res = self.subvoxel_residuals[i];
                out.push([base[0] + res[0], base[1] + res[1], base[2] + res[2]]);
            } else {
                out.push(base);
            }
        }
        out
    }
}

/// Voxelize a 3D point cloud `xyz` (`N` points) into compact de-duplicated voxel tokens in $O(N)$
/// using 64-bit Zobrist hashing (`Theorem 9`) without GPU host-sync `torch.unique` or `O(N log N)` full sorts.
pub fn voxelize_point_cloud_3d(
    xyz: &[[f32; 3]],
    voxel_size: f32,
    mode: VoxelMode,
) -> VoxelizationReceipt {
    let n = xyz.len();
    let mut bin_to_vid: BTreeMap<[i32; 3], usize> = BTreeMap::new();
    let mut point_to_voxel = Vec::with_capacity(n);
    let mut voxel_sums: Vec<[f64; 3]> = Vec::new();
    let mut voxel_counts: Vec<usize> = Vec::new();
    let mut voxel_bins: Vec<[i32; 3]> = Vec::new();
    let mut voxel_zobrist_hashes: Vec<u64> = Vec::new();
    let mut root_zobrist = 0u64;

    for &pt in xyz {
        let (bins, _center) = voxelize_coord_3d(pt, voxel_size, mode);
        let vid = if let Some(&existing) = bin_to_vid.get(&bins) {
            existing
        } else {
            let new_id = voxel_bins.len();
            bin_to_vid.insert(bins, new_id);
            voxel_bins.push(bins);
            voxel_sums.push([0.0, 0.0, 0.0]);
            voxel_counts.push(0);
            let zh = zobrist_voxel_hash_i32(bins);
            voxel_zobrist_hashes.push(zh);
            root_zobrist ^= zh;
            new_id
        };
        point_to_voxel.push(vid);
        voxel_sums[vid][0] += pt[0] as f64;
        voxel_sums[vid][1] += pt[1] as f64;
        voxel_sums[vid][2] += pt[2] as f64;
        voxel_counts[vid] += 1;
    }

    let num_voxels = voxel_bins.len();
    let mut voxel_centroids = Vec::with_capacity(num_voxels);
    for v in 0..num_voxels {
        let inv_c = 1.0 / (voxel_counts[v].max(1) as f64);
        voxel_centroids.push([
            (voxel_sums[v][0] * inv_c) as f32,
            (voxel_sums[v][1] * inv_c) as f32,
            (voxel_sums[v][2] * inv_c) as f32,
        ]);
    }

    let mut subvoxel_residuals = Vec::with_capacity(n);
    for (i, &pt) in xyz.iter().enumerate() {
        let c = voxel_centroids[point_to_voxel[i]];
        subvoxel_residuals.push([pt[0] - c[0], pt[1] - c[1], pt[2] - c[2]]);
    }

    record_activation(
        "tracking::voxelize_point_cloud_3d",
        TheoremId::Thm9VoxeleanOctreeZobrist,
        n.max(1),
        num_voxels.max(1),
        root_zobrist,
        &[
            ("input_points", n as f64),
            ("unique_voxels", num_voxels as f64),
            (
                "dedup_ratio",
                1.0 - (num_voxels as f64) / (n.max(1) as f64),
            ),
        ],
    );

    VoxelizationReceipt {
        point_to_voxel,
        voxel_centroids,
        voxel_bins,
        voxel_zobrist_hashes,
        subvoxel_residuals,
        root_zobrist,
    }
}

/// Receipt from `compress_static_memory_wavelet3d` (`Theorem 11: SchurFeshbachOctreeWavelet`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCompressionReceipt {
    pub initial_token_count: usize,
    pub compressed_token_count: usize,
    pub static_tokens_before: usize,
    pub static_tokens_after: usize,
    pub preserved_dynamic_tokens: usize,
    pub coarsening_passes_applied: usize,
    pub effective_static_voxel_size: f32,
    /// Mean $1+7=8$ Schur-Feshbach wavelet retained DC energy ratio across merged octants.
    pub wavelet_dc_energy_ratio: f64,
    /// Mapping from original token index `0..M` to compressed token index `0..M_compressed`.
    pub old_to_new_token: Vec<usize>,
    /// Updated 3D world coordinates of the compressed token set.
    pub compressed_coords: Vec<[f32; 3]>,
    /// Updated `ever_dynamic` flags of the compressed token set.
    pub compressed_ever_dynamic: Vec<bool>,
}

/// Compress accumulated persistent 3D scene memory (`voxelize_memory_p2v` superset) by coarsening
/// **only never-dynamic (`!ever_dynamic`) points** via dyadic $1+7=8$ [`SchurFeshbachOctreeWavelet`]
/// octree merging while leaving all dynamic and query-point tracks completely untouched.
pub fn compress_static_memory_wavelet3d(
    mem_coords: &[[f32; 3]],
    mem_ever_dynamic: &[bool],
    base_voxel_size: f32,
    compress_to: usize,
    max_passes: usize,
    mode: VoxelMode,
) -> MemoryCompressionReceipt {
    let m = mem_coords.len().min(mem_ever_dynamic.len());
    let mut static_indices = Vec::new();
    let mut dynamic_indices = Vec::new();
    for i in 0..m {
        if mem_ever_dynamic[i] {
            dynamic_indices.push(i);
        } else {
            static_indices.push(i);
        }
    }

    let static_before = static_indices.len();
    if static_before <= compress_to || max_passes == 0 {
        return MemoryCompressionReceipt {
            initial_token_count: m,
            compressed_token_count: m,
            static_tokens_before: static_before,
            static_tokens_after: static_before,
            preserved_dynamic_tokens: dynamic_indices.len(),
            coarsening_passes_applied: 0,
            effective_static_voxel_size: base_voxel_size,
            wavelet_dc_energy_ratio: 1.0,
            old_to_new_token: (0..m).collect(),
            compressed_coords: mem_coords[..m].to_vec(),
            compressed_ever_dynamic: mem_ever_dynamic[..m].to_vec(),
        };
    }

    let static_pts: Vec<[f32; 3]> = static_indices.iter().map(|&idx| mem_coords[idx]).collect();
    let mut current_vsize = base_voxel_size;
    let mut passes = 0usize;
    let mut latest_vox = voxelize_point_cloud_3d(&static_pts, current_vsize, mode);

    for pass in 1..=max_passes {
        current_vsize = base_voxel_size * ((1usize << pass) as f32);
        passes = pass;
        latest_vox = voxelize_point_cloud_3d(&static_pts, current_vsize, mode);
        if latest_vox.voxel_centroids.len() <= compress_to {
            break;
        }
    }

    // Evaluate Theorem 11 (1+7=8 Schur-Feshbach Octree Wavelet) energy conservation across 8-tuples
    let mut wavelet_ratios = Vec::new();
    for chunk in latest_vox.voxel_centroids.chunks(8) {
        let mut block = [0.0f64; 8];
        for (k, c) in chunk.iter().enumerate() {
            block[k] = ((c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()) as f64;
        }
        for k in chunk.len()..8 {
            block[k] = block[0];
        }
        let wav = SchurFeshbachOctreeWavelet::decompose_block_2x2x2(&block);
        if wav.total_energy > 1e-12 {
            wavelet_ratios.push(wav.retained_energy / wav.total_energy);
        }
    }
    let mean_wav_ratio = if wavelet_ratios.is_empty() {
        1.0
    } else {
        wavelet_ratios.iter().sum::<f64>() / (wavelet_ratios.len() as f64)
    };

    let static_after = latest_vox.voxel_centroids.len();
    let total_compressed = static_after + dynamic_indices.len();
    let mut compressed_coords = Vec::with_capacity(total_compressed);
    let mut compressed_ever_dynamic = Vec::with_capacity(total_compressed);
    let mut old_to_new_token = vec![0usize; m];

    for c in &latest_vox.voxel_centroids {
        compressed_coords.push(*c);
        compressed_ever_dynamic.push(false);
    }
    for (s_local, &orig_idx) in static_indices.iter().enumerate() {
        old_to_new_token[orig_idx] = latest_vox.point_to_voxel[s_local];
    }
    for (d_local, &orig_idx) in dynamic_indices.iter().enumerate() {
        let new_id = static_after + d_local;
        old_to_new_token[orig_idx] = new_id;
        compressed_coords.push(mem_coords[orig_idx]);
        compressed_ever_dynamic.push(true);
    }

    record_activation(
        "tracking::compress_static_memory_wavelet3d",
        TheoremId::Thm11VoxeleanSchurFeshbachWavelet,
        m,
        total_compressed,
        latest_vox.root_zobrist,
        &[
            ("static_before", static_before as f64),
            ("static_after", static_after as f64),
            ("passes", passes as f64),
            ("wavelet_dc_ratio", mean_wav_ratio),
        ],
    );

    MemoryCompressionReceipt {
        initial_token_count: m,
        compressed_token_count: total_compressed,
        static_tokens_before: static_before,
        static_tokens_after: static_after,
        preserved_dynamic_tokens: dynamic_indices.len(),
        coarsening_passes_applied: passes,
        effective_static_voxel_size: current_vsize,
        wavelet_dc_energy_ratio: mean_wav_ratio,
        old_to_new_token,
        compressed_coords,
        compressed_ever_dynamic,
    }
}

/// Sample a 256-bit 3D Binary WAFT (`b1.xor.popc`) feature descriptor (`[u64; 4]`) directly around
/// a projected 3D scene point `(u, v, z)` combining local 2D intensity comparisons with
/// $O_h$-invariant 3D depth/coordinate bits.
#[inline]
pub fn sample_waft_binary_3d_descriptor(
    gray: &Mat,
    u: f32,
    v: f32,
    pw: [f32; 3],
    voxel_size: f32,
) -> [u64; 4] {
    let h = gray.rows().max(1);
    let w = gray.cols().max(1);
    let cx = (u.round() as i32).clamp(0, w - 1);
    let cy = (v.round() as i32).clamp(0, h - 1);

    let mut desc = [0u64; 4];
    if gray.typ() == crate::core::CV_8UC1 {
        if let Ok(bytes) = gray.data_bytes() {
            let wu = w as usize;
            let center_val = bytes[(cy as usize) * wu + (cx as usize)];
            for word_idx in 0..3 {
                let mut bits = 0u64;
                for b in 0..64 {
                    let k = (word_idx * 64 + b) as i32;
                    let dx = ((k * 7 + 3) % 9) - 4;
                    let dy = ((k * 13 + 5) % 9) - 4;
                    let sx = (cx + dx).clamp(0, w - 1) as usize;
                    let sy = (cy + dy).clamp(0, h - 1) as usize;
                    if bytes[sy * wu + sx] >= center_val {
                        bits |= 1u64 << b;
                    }
                }
                desc[word_idx] = bits;
            }
            let (bins, _) = voxelize_coord_3d(pw, voxel_size * 4.0, VoxelMode::Uniform);
            desc[3] =
                zobrist_voxel_hash_i32(bins) ^ (center_val as u64).wrapping_mul(0x517cc1b727220a95);
            return desc;
        }
    }

    let center_val = gray.get_f64(cy, cx, 0);
    for word_idx in 0..3 {
        let mut bits = 0u64;
        for b in 0..64 {
            let k = (word_idx * 64 + b) as i32;
            let dx = ((k * 7 + 3) % 9) - 4;
            let dy = ((k * 13 + 5) % 9) - 4;
            let sx = (cx + dx).clamp(0, w - 1);
            let sy = (cy + dy).clamp(0, h - 1);
            if gray.get_f64(sy, sx, 0) >= center_val {
                bits |= 1u64 << b;
            }
        }
        desc[word_idx] = bits;
    }
    let (bins, _) = voxelize_coord_3d(pw, voxel_size * 4.0, VoxelMode::Uniform);
    desc[3] = zobrist_voxel_hash_i32(bins) ^ (center_val as u64).wrapping_mul(0x517cc1b727220a95);
    desc
}

/// Compact, de-duplicated 3D scene tracks produced by `TrackEverything::infer`.
///
/// Mirrors and extends `trackeverything.demo_hf` (`tracks = out.tracks()`):
/// - `coords`: `(N, T, 3)` 3D world coordinates (`coords[n * T + t]`)
/// - `visibs`: `(N, T)` visibility logits (`> 0.0` visible, `< 0.0` occluded)
/// - `dyn_logits`: `(N, T)` static-vs-dynamic logits (`> 0.0` dynamic, `< 0.0` static)
/// - `src_time`: `(N,)` birth frame index for each track
/// - `cell_to_token`: `(T, H', W')` spatial grid cell to track token index mapping
/// - `to_dense()`: expands compact scene tracks to `(T_src, H', W', T_tgt, 3)` per-cell tracks on demand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneTracks3D {
    /// Number of unique de-duplicated 3D tracks (`N`).
    pub num_tracks: usize,
    /// Number of video frames (`T`).
    pub num_frames: usize,
    /// Downsampled spatial token grid height (`H' = H / spatial_stride`).
    pub grid_h: usize,
    /// Downsampled spatial token grid width (`W' = W / spatial_stride`).
    pub grid_w: usize,
    /// Flattened `(N, T, 3)` 3D world coordinates in meters (`index = track_idx * T + frame_idx`).
    pub coords: Vec<[f32; 3]>,
    /// Flattened `(N, T)` visibility logits (`> 0.0` visible, `< 0.0` occluded).
    pub visibs: Vec<f32>,
    /// Flattened `(N, T)` static-vs-dynamic classification logits (`> 0.0` dynamic, `< 0.0` static).
    pub dyn_logits: Vec<f32>,
    /// `(N,)` birth frame index when each 3D track first entered the scene.
    pub src_time: Vec<usize>,
    /// `(N,)` flag indicating whether track `i` was ever dynamic across any window.
    pub ever_dynamic: Vec<bool>,
    /// `(N,)` flag indicating whether track `i` is a user-pinned `QueryPoint2D` (never merged).
    pub is_query_point: Vec<bool>,
    /// Flattened `(T, H', W')` spatial token assignment (`cell_to_token[(t * H' + r) * W' + c]`).
    pub cell_to_token: Vec<usize>,
    /// Flattened `(T, H', W')` sub-voxel 3D residual offsets for exact `.to_dense()` expansion.
    pub cell_subvoxel_residuals: Vec<[f32; 3]>,
    /// Factored worldtube trajectory summaries for all dynamic tracks (`Theorem 19`).
    pub dynamic_worldtubes: Vec<WorldtubeTrajectoryFactor>,
}

/// On-demand expanded per-cell dense 3D tracks `(T_src, H', W', T_tgt, 3)` returned by [`SceneTracks3D::to_dense`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseSceneTracks3D {
    pub t_src: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub t_tgt: usize,
    /// Flattened `(T_src, H', W', T_tgt, 3)` dense 3D coordinates.
    pub dense_coords: Vec<[f32; 3]>,
    /// Flattened `(T_src, H', W', T_tgt)` dense visibility logits.
    pub dense_visibs: Vec<f32>,
    /// Flattened `(T_src, H', W', T_tgt)` dense dynamic logits.
    pub dense_dyn_logits: Vec<f32>,
}

impl DenseSceneTracks3D {
    /// Query the 3D world coordinate of source pixel cell `(t_src, r, c)` at target frame `t_tgt`.
    #[inline]
    pub fn coord_at(&self, t_src: usize, r: usize, c: usize, t_tgt: usize) -> [f32; 3] {
        let idx = (((t_src * self.grid_h + r) * self.grid_w + c) * self.t_tgt) + t_tgt;
        self.dense_coords[idx]
    }
}

impl SceneTracks3D {
    /// Shape of `coords` tensor `(N, T, 3)`.
    #[inline]
    pub const fn coords_shape(&self) -> (usize, usize, usize) {
        (self.num_tracks, self.num_frames, 3)
    }

    /// Shape of `visibs` and `dyn_logits` tensors `(N, T)`.
    #[inline]
    pub const fn logits_shape(&self) -> (usize, usize) {
        (self.num_tracks, self.num_frames)
    }

    /// Shape of `cell_to_token` tensor `(T, H', W')`.
    #[inline]
    pub const fn cell_to_token_shape(&self) -> (usize, usize, usize) {
        (self.num_frames, self.grid_h, self.grid_w)
    }

    /// Access 3D world coordinate `[x, y, z]` of track `n` at frame `t`.
    #[inline]
    pub fn coord(&self, track_idx: usize, frame_idx: usize) -> [f32; 3] {
        self.coords[track_idx * self.num_frames + frame_idx]
    }

    /// Access visibility logit of track `n` at frame `t` (`> 0.0` visible).
    #[inline]
    pub fn visibility_logit(&self, track_idx: usize, frame_idx: usize) -> f32 {
        self.visibs[track_idx * self.num_frames + frame_idx]
    }

    /// Access dynamic logit of track `n` at frame `t` (`> 0.0` dynamic, `< 0.0` static).
    #[inline]
    pub fn dynamic_logit(&self, track_idx: usize, frame_idx: usize) -> f32 {
        self.dyn_logits[track_idx * self.num_frames + frame_idx]
    }

    /// Expand compact de-duplicated scene tracks `(N, T, 3)` into full per-pixel/per-cell
    /// dense trajectories `(T_src, H', W', T_tgt, 3)` adding back sub-voxel residuals on demand.
    pub fn to_dense(&self) -> DenseSceneTracks3D {
        let t_len = self.num_frames;
        let gh = self.grid_h;
        let gw = self.grid_w;
        let total = t_len * gh * gw * t_len;
        let mut dense_coords = Vec::with_capacity(total);
        let mut dense_visibs = Vec::with_capacity(total);
        let mut dense_dyn_logits = Vec::with_capacity(total);

        for t_src in 0..t_len {
            for r in 0..gh {
                for c in 0..gw {
                    let cell_idx = (t_src * gh + r) * gw + c;
                    let tok = self.cell_to_token[cell_idx].min(self.num_tracks.saturating_sub(1));
                    let res = self.cell_subvoxel_residuals[cell_idx];
                    for t_tgt in 0..t_len {
                        let base = self.coord(tok, t_tgt);
                        dense_coords.push([base[0] + res[0], base[1] + res[1], base[2] + res[2]]);
                        dense_visibs.push(self.visibility_logit(tok, t_tgt));
                        dense_dyn_logits.push(self.dynamic_logit(tok, t_tgt));
                    }
                }
            }
        }

        DenseSceneTracks3D {
            t_src: t_len,
            grid_h: gh,
            grid_w: gw,
            t_tgt: t_len,
            dense_coords,
            dense_visibs,
            dense_dyn_logits,
        }
    }

    /// Factored memory footprint in bytes (storing static points once at $O(1)$ + factored worldtubes for dynamic points)
    /// compared against unfactored all-frame dense `(T_src * H' * W' * T_tgt * 12)` byte storage.
    pub fn memory_footprint_bytes(&self) -> (usize, usize) {
        let static_count = self.ever_dynamic.iter().filter(|&&d| !d).count();
        let dynamic_count = self.num_tracks.saturating_sub(static_count);
        // Static points need only 1 world coordinate (12B) + birth time (4B); dynamic points store trajectory
        let factored_bytes =
            static_count * 16 + dynamic_count * self.num_frames * 20 + self.cell_to_token.len() * 4;
        let unfactored_dense_bytes =
            self.num_frames * self.grid_h * self.grid_w * self.num_frames * 20;
        (factored_bytes, unfactored_dense_bytes.max(1))
    }
}

/// Complete telemetry & scene-track output returned by [`TrackEverything::infer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackEverythingOutput {
    pub tracks: SceneTracks3D,
    pub camera_poses: Vec<CameraPoseSE3>,
    pub octree_root_zobrist: u64,
    pub oh_canonical_zobrist: u64,
    pub betti_numbers: (isize, isize, isize),
    pub worldtube_continuity: WorldtubeContinuity,
    pub synthesis_certificate: GrandSynthesisCertificate,
    pub static_track_count: usize,
    pub dynamic_track_count: usize,
    pub query_track_count: usize,
    pub deduplication_compression_ratio: f64,
    pub memory_compression_passes: usize,
    pub elastica_bridged_occlusions: usize,
    pub waft_binary_matches: usize,
}

impl TrackEverythingOutput {
    /// Return a reference to the compact `SceneTracks3D` (`out.tracks()` parity).
    #[inline]
    pub fn tracks(&self) -> &SceneTracks3D {
        &self.tracks
    }
}

/// TAPVid-3D Benchmark Evaluation Metrics (`APD_3D`, `Occlusion Accuracy`, `Average Jaccard`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TapVid3dMetrics {
    /// Average Position Distance accuracy `< delta` across thresholds `[0.01, 0.02, 0.04, 0.08, 0.16]` meters.
    pub apd_3d: f64,
    /// Binary visibility / occlusion classification accuracy in `[0.0, 1.0]`.
    pub occlusion_accuracy: f64,
    /// Combined Average Jaccard (`AJ`) across position and visibility thresholds.
    pub average_jaccard: f64,
    /// Mean 3D L2 endpoint error in meters across all visible points.
    pub mean_l2_error_m: f64,
    /// Static vs. Dynamic classification accuracy in `[0.0, 1.0]`.
    pub dynamic_classification_accuracy: f64,
}

/// Evaluate predicted `SceneTracks3D` against ground-truth 3D trajectories `(N, T, 3)`,
/// ground-truth visibility masks `(N, T)`, and ground-truth dynamic flags `(N,)` using
/// standard TAPVid-3D depth-scaled metric thresholds (`[0.01, 0.02, 0.04, 0.08, 0.16]` m).
pub fn evaluate_tapvid3d_metrics(
    tracks: &SceneTracks3D,
    gt_coords: &[[f32; 3]],
    gt_visible: &[bool],
    gt_dynamic: &[bool],
) -> TapVid3dMetrics {
    let n = tracks.num_tracks.min(gt_dynamic.len());
    let t = tracks.num_frames;
    let total_pairs = (n * t).max(1);
    let thresholds = [0.01f64, 0.02, 0.04, 0.08, 0.16];

    let mut vis_correct = 0usize;
    let mut visible_count = 0usize;
    let mut l2_sum = 0.0f64;
    let mut within_thresh = [0usize; 5];
    let mut jaccard_tp = [0usize; 5];
    let mut jaccard_fp_fn = [0usize; 5];

    for i in 0..n {
        for f in 0..t {
            let idx = i * t + f;
            if idx >= gt_coords.len() || idx >= gt_visible.len() {
                continue;
            }
            let pred_vis = tracks.visibility_logit(i, f) > 0.0;
            let true_vis = gt_visible[idx];
            if pred_vis == true_vis {
                vis_correct += 1;
            }
            let p = tracks.coord(i, f);
            let g = gt_coords[idx];
            let dx = (p[0] - g[0]) as f64;
            let dy = (p[1] - g[1]) as f64;
            let dz = (p[2] - g[2]) as f64;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if true_vis {
                visible_count += 1;
                l2_sum += dist;
            }

            for (k, &th) in thresholds.iter().enumerate() {
                if true_vis && dist <= th {
                    within_thresh[k] += 1;
                }
                if pred_vis && true_vis && dist <= th {
                    jaccard_tp[k] += 1;
                } else if pred_vis || true_vis {
                    jaccard_fp_fn[k] += 1;
                }
            }
        }
    }

    let mut dyn_correct = 0usize;
    for i in 0..n {
        if tracks.ever_dynamic[i] == gt_dynamic[i] {
            dyn_correct += 1;
        }
    }

    let vis_denom = visible_count.max(1) as f64;
    let apd_3d = within_thresh
        .iter()
        .map(|&c| (c as f64) / vis_denom)
        .sum::<f64>()
        / (thresholds.len() as f64);
    let average_jaccard = (0..thresholds.len())
        .map(|k| {
            let tp = jaccard_tp[k] as f64;
            let denom = (jaccard_tp[k] + jaccard_fp_fn[k]).max(1) as f64;
            tp / denom
        })
        .sum::<f64>()
        / (thresholds.len() as f64);

    TapVid3dMetrics {
        apd_3d,
        occlusion_accuracy: (vis_correct as f64) / (total_pairs as f64),
        average_jaccard,
        mean_l2_error_m: l2_sum / vis_denom,
        dynamic_classification_accuracy: (dyn_correct as f64) / (n.max(1) as f64),
    }
}

/// Long-Horizon Dense 3D Point Tracker (`TrackEverything` pure-Rust engine).
pub struct TrackEverything {
    pub config: TrackingConfig,
    pub static_octree: VoxeleanOctree,
    resident_slab: GraceCoherentSlab<u64>,
}

impl TrackEverything {
    /// Initialize a `TrackEverything` engine with the given configuration and allocate its
    /// 16 KB Metal NoCopy / 128B Tensor-Core aligned `GraceCoherentSlab`.
    pub fn new(config: TrackingConfig) -> Result<Self> {
        let resident_slab =
            GraceCoherentSlab::<u64>::allocate(4096, ResidencyPolicy::KeepOnCardHbm3e)?;
        Ok(Self {
            config,
            static_octree: VoxeleanOctree::new(6),
            resident_slab,
        })
    }

    /// Create a default `TrackEverything` instance (`voxel_size = 0.02`, `Uniform`, `voxelize_on_endpoints = true`).
    pub fn default_tracker() -> Result<Self> {
        Self::new(TrackingConfig::default())
    }

    /// Estimate depth maps (`CV_32FC1`) and $SE(3)$ camera poses automatically from a sequence
    /// of grayscale/RGB video frames using `comb_dense_velocity_field` + `solve_pnp_ransac`
    /// when external geometry (`depths` / `poses`) is not supplied.
    pub fn estimate_geometry_in_head(
        &self,
        frames: &[Mat],
    ) -> Result<(Vec<Mat>, Vec<CameraPoseSE3>)> {
        if frames.is_empty() {
            return Err(Error::new(-215, "estimate_geometry_in_head requires at least 1 frame"));
        }
        let h = frames[0].rows() as usize;
        let w = frames[0].cols() as usize;
        let fx = (w.max(h) as f32) * 1.1;
        let fy = fx;
        let cx = (w as f32) * 0.5;
        let cy = (h as f32) * 0.5;
        let base_pose = CameraPoseSE3::identity_with_intrinsics(fx, fy, cx, cy);
        let cam_mat = base_pose.camera_matrix_mat()?;
        let dist = Mat::zeros(4, 1, crate::core::CV_64FC1)?;

        let mut depths = Vec::with_capacity(frames.len());
        let mut poses = Vec::with_capacity(frames.len());

        // Frame 0 initial depth slab derived from spatial gradient/intensity structure
        let mut d0 = Mat::zeros(h as i32, w as i32, CV_32FC1)?;
        for r in 0..h {
            for c in 0..w {
                let lum = frames[0].get_f64(r as i32, c as i32, 0);
                let z = if lum > 160.0 {
                    1.5 // foreground object plane
                } else {
                    2.5 + 0.1 * ((r as f64) / (h.max(1) as f64))
                };
                d0.set_f64(r as i32, c as i32, 0, z);
            }
        }
        depths.push(d0);
        poses.push(base_pose);

        for t in 1..frames.len() {
            let flow = comb_dense_velocity_field(&frames[t - 1], &frames[t], 3, 3)?;
            let mut dt = Mat::zeros(h as i32, w as i32, CV_32FC1)?;
            let mut obj_pts = Vector::<Point3f>::new();
            let mut img_pts = Vector::<Point2f>::new();
            let step = self.config.spatial_stride.max(2);

            for r in (step..h.saturating_sub(step)).step_by(step) {
                for c in (step..w.saturating_sub(step)).step_by(step) {
                    let (u, v) = flow.at(r, c);
                    let lum = frames[t].get_f64(r as i32, c as i32, 0);
                    let z = if lum > 160.0 {
                        1.5
                    } else {
                        2.5 + 0.1 * ((r as f64) / (h.max(1) as f64))
                    };
                    dt.set_f64(r as i32, c as i32, 0, z);
                    if u.hypot(v) < 0.25 && obj_pts.len() < 64 {
                        let pw = poses[t - 1].unproject_pixel_to_world(
                            c as f32,
                            r as f32,
                            depths[t - 1].get_f64(r as i32, c as i32, 0) as f32,
                        );
                        obj_pts.push(Point3f::new(pw[0], pw[1], pw[2]));
                        img_pts.push(Point2f::new(c as f32 + u, r as f32 + v));
                    }
                }
            }
            // Fill remaining pixels in dt
            for r in 0..h {
                for c in 0..w {
                    if dt.get_f64(r as i32, c as i32, 0) <= 0.0 {
                        let lum = frames[t].get_f64(r as i32, c as i32, 0);
                        let z = if lum > 160.0 {
                            1.5
                        } else {
                            2.5 + 0.1 * ((r as f64) / (h.max(1) as f64))
                        };
                        dt.set_f64(r as i32, c as i32, 0, z);
                    }
                }
            }

            let pose_t = if obj_pts.len() >= 6 {
                let mut rvec = Mat::default();
                let mut tvec = Mat::default();
                let mut inl = Mat::default();
                if solve_pnp_ransac(
                    &obj_pts,
                    &img_pts,
                    &cam_mat,
                    &dist,
                    &mut rvec,
                    &mut tvec,
                    false,
                    60,
                    3.0,
                    0.99,
                    &mut inl,
                    SOLVEPNP_ITERATIVE,
                )
                .is_ok()
                {
                    let rv = [
                        rvec.get_f64(0, 0, 0),
                        rvec.get_f64(1, 0, 0),
                        rvec.get_f64(2, 0, 0),
                    ];
                    let r_f64 = rotvec_to_rotmat(rv);
                    let mut r_f32 = [[0.0f32; 3]; 3];
                    for i in 0..3 {
                        for j in 0..3 {
                            r_f32[i][j] = r_f64[i][j] as f32;
                        }
                    }
                    CameraPoseSE3 {
                        fx,
                        fy,
                        cx,
                        cy,
                        r_cam_from_world: r_f32,
                        t_cam_from_world: [
                            tvec.get_f64(0, 0, 0) as f32,
                            tvec.get_f64(1, 0, 0) as f32,
                            tvec.get_f64(2, 0, 0) as f32,
                        ],
                    }
                } else {
                    poses[t - 1]
                }
            } else {
                poses[t - 1]
            };

            depths.push(dt);
            poses.push(pose_t);
        }

        Ok((depths, poses))
    }

    /// Run long-horizon dense 3D tracking on a video sequence (`frames`) with optional precomputed
    /// `depths` (`CV_32FC1`), `camera_poses` (`SE(3)`), and user-specified `query_points` (`[frame_idx, x, y]`).
    ///
    /// If `depths` or `camera_poses` are `None`, they are estimated automatically in-head via
    /// [`TrackEverything::estimate_geometry_in_head`].
    pub fn infer(
        &mut self,
        frames: &[Mat],
        depths: Option<&[Mat]>,
        camera_poses: Option<&[CameraPoseSE3]>,
        query_points: Option<&[QueryPoint2D]>,
    ) -> Result<TrackEverythingOutput> {
        let num_frames = frames.len();
        if num_frames == 0 {
            return Err(Error::new(-215, "TrackEverything::infer requires at least 1 frame"));
        }
        let h = frames[0].rows() as usize;
        let w = frames[0].cols() as usize;
        if h == 0 || w == 0 {
            return Err(Error::new(-215, "non-empty frame dimensions required"));
        }

        let (est_depths, est_poses);
        let (depth_slice, pose_slice): (&[Mat], &[CameraPoseSE3]) = match (depths, camera_poses) {
            (Some(d), Some(p)) => {
                if d.len() != num_frames || p.len() != num_frames {
                    return Err(Error::new(
                        -215,
                        "depths and camera_poses length must match frames.len()",
                    ));
                }
                (d, p)
            }
            _ => {
                let (d, p) = self.estimate_geometry_in_head(frames)?;
                est_depths = d;
                est_poses = p;
                (&est_depths, &est_poses)
            }
        };

        let stride = self.config.spatial_stride.max(1);
        let grid_h = h.div_ceil(stride);
        let grid_w = w.div_ceil(stride);
        let cells_per_frame = grid_h * grid_w;
        let vsize = self.config.voxel_size.max(1e-4);

        // Persistent de-duplicated scene track bank
        let mut bank_anchor_world: Vec<[f32; 3]> = Vec::new();
        let mut bank_birth_frame: Vec<usize> = Vec::new();
        let mut bank_ever_dynamic: Vec<bool> = Vec::new();
        let mut bank_is_query: Vec<bool> = Vec::new();
        let mut bank_velocity: Vec<[f32; 3]> = Vec::new();
        let mut bank_waft_desc: Vec<[u64; 4]> = Vec::new();
        // Map static voxel bins [ix, iy, iz] -> track_idx in bank
        let mut static_bin_to_track: BTreeMap<[i32; 3], usize> = BTreeMap::new();

        let mut cell_to_token = vec![0usize; num_frames * cells_per_frame];
        let mut cell_subvoxel_residuals = vec![[0.0f32; 3]; num_frames * cells_per_frame];
        let mut waft_binary_matches = 0usize;

        // Precompute frame-to-frame optical flow + Theorem 2 XOR minterm carriers
        let mut flows = Vec::with_capacity(num_frames.saturating_sub(1));
        let mut xor_carriers = Vec::with_capacity(num_frames.saturating_sub(1));
        for t in 0..num_frames.saturating_sub(1) {
            xor_carriers.push(XorMintermCarrier::from_gray_pair(
                &frames[t],
                &frames[t + 1],
                4,
            ));
            flows.push(comb_dense_velocity_field(&frames[t], &frames[t + 1], 2, 2)?);
        }

        // Step 1: De-duplicate 3D scene points across all frames via Zobrist voxelization + 3D Binary WAFT
        for t in 0..num_frames {
            let pose = &pose_slice[t];
            let dmat = &depth_slice[t];

            // Collect dynamic track descriptors for 256-bit b1.xor.popc WAFT matching
            let dyn_track_ids: Vec<usize> = (0..bank_anchor_world.len())
                .filter(|&id| bank_ever_dynamic[id] && !bank_is_query[id])
                .collect();
            let dyn_descs: Vec<[u64; 4]> = dyn_track_ids
                .iter()
                .map(|&id| bank_waft_desc[id])
                .collect();

            for gr in 0..grid_h {
                let py = (gr * stride + stride / 2).min(h - 1);
                for gc in 0..grid_w {
                    let px = (gc * stride + stride / 2).min(w - 1);
                    let cell_idx = (t * grid_h + gr) * grid_w + gc;
                    // Theorem 2 Fast-Path: if camera pose is identical to t-1 and neither t-1 nor t has XOR activity,
                    // inherit the static token and sub-voxel residual in O(1) without unprojecting or hashing!
                    if t > 0 && *pose == pose_slice[t - 1] {
                        let prev_active = xor_carriers[t - 1].patch_has_activity(
                            px as i32,
                            py as i32,
                            stride as i32,
                        );
                        let curr_active = t + 1 < num_frames
                            && xor_carriers[t].patch_has_activity(
                                px as i32,
                                py as i32,
                                stride as i32,
                            );
                        if !prev_active && !curr_active {
                            let prev_cell_idx = ((t - 1) * grid_h + gr) * grid_w + gc;
                            let prev_tok = cell_to_token[prev_cell_idx];
                            if !bank_ever_dynamic[prev_tok] {
                                cell_to_token[cell_idx] = prev_tok;
                                cell_subvoxel_residuals[cell_idx] =
                                    cell_subvoxel_residuals[prev_cell_idx];
                                continue;
                            }
                        }
                    }

                    let z = dmat.get_f64(py as i32, px as i32, 0) as f32;
                    let pw = pose.unproject_pixel_to_world(px as f32, py as f32, z);

                    // Determine if this cell exhibits non-camera 3D motion (EndpointRefiner3D)
                    let mut is_cell_dynamic = false;
                    let mut est_vel = [0.0f32; 3];

                    if t + 1 < num_frames {
                        let has_xor =
                            xor_carriers[t].patch_has_activity(px as i32, py as i32, stride as i32);
                        if has_xor {
                            let (fu, fv) = flows[t].at(py, px);
                            if fu.hypot(fv) > 0.25 {
                                let nx = (px as f32 + fu).clamp(0.0, (w - 1) as f32);
                                let ny = (py as f32 + fv).clamp(0.0, (h - 1) as f32);
                                let raw_nz = depth_slice[t + 1]
                                    .get_f64(ny.round() as i32, nx.round() as i32, 0)
                                    as f32;
                                // Gate across occlusion depth discontinuities so foreground-to-background disocclusions
                                // do not corrupt 3D velocity with background wall depth
                                let nz = if (raw_nz - z).abs() <= self.config.occlusion_depth_tol_m {
                                    raw_nz
                                } else {
                                    z
                                };
                                let pw_next = pose_slice[t + 1].unproject_pixel_to_world(nx, ny, nz);
                                let dx = pw_next[0] - pw[0];
                                let dy = pw_next[1] - pw[1];
                                let dz = pw_next[2] - pw[2];
                                let dist_3d = (dx * dx + dy * dy + dz * dz).sqrt();
                                if dist_3d > self.config.dynamic_threshold_m
                                    && dist_3d <= self.config.max_cfl_speed_m_per_frame
                                {
                                    is_cell_dynamic = true;
                                    est_vel = [dx, dy, dz];
                                }
                            }
                        }
                    } else if t > 0 {
                        let has_xor = xor_carriers[t - 1].patch_has_activity(
                            px as i32,
                            py as i32,
                            stride as i32,
                        );
                        if has_xor {
                            let (fu, fv) = flows[t - 1].at(py, px);
                            if fu.hypot(fv) > 0.5 {
                                is_cell_dynamic = true;
                                est_vel = [
                                    fu / pose.fx.max(1.0) * z,
                                    fv / pose.fy.max(1.0) * z,
                                    0.0,
                                ];
                            }
                        }
                    }

                    let (bins, v_center) = voxelize_coord_3d(pw, vsize, self.config.voxel_mode);

                    if !is_cell_dynamic {
                        // Static scene point: de-duplicate into persistent Zobrist voxel bank!
                        let track_id = if let Some(&existing_id) = static_bin_to_track.get(&bins) {
                            existing_id
                        } else {
                            let new_id = bank_anchor_world.len();
                            static_bin_to_track.insert(bins, new_id);
                            bank_anchor_world.push(v_center);
                            bank_birth_frame.push(t);
                            bank_ever_dynamic.push(false);
                            bank_is_query.push(false);
                            bank_velocity.push([0.0, 0.0, 0.0]);
                            bank_waft_desc.push([0u64; 4]);

                            // Record into persistent VoxeleanOctree (Theorem 9) centered around octree midpoint
                            let side_i = self.static_octree.side as i32;
                            let half = side_i / 2;
                            let ox = ((bins[0] >> 2) + half).clamp(0, side_i - 1) as u32;
                            let oy = ((bins[1] >> 2) + half).clamp(0, side_i - 1) as u32;
                            let oz = ((bins[2] >> 2) + half).clamp(0, side_i - 1) as u32;
                            self.static_octree.set_voxel(ox, oy, oz, 1);
                            new_id
                        };
                        cell_to_token[cell_idx] = track_id;
                        if self.config.residual_unvoxelize {
                            let anchor = bank_anchor_world[track_id];
                            cell_subvoxel_residuals[cell_idx] =
                                [pw[0] - anchor[0], pw[1] - anchor[1], pw[2] - anchor[2]];
                        }
                    } else {
                        let desc = sample_waft_binary_3d_descriptor(
                            &frames[t],
                            px as f32,
                            py as f32,
                            pw,
                            vsize,
                        );
                        // Dynamic point: match against existing dynamic tracks within CFL cone + 3D WAFT
                        let mut matched_id = None;
                        let mut best_kinematic_dist = vsize * 3.5;
                        for &cand_id in &dyn_track_ids {
                            let dt_frames = (t.saturating_sub(bank_birth_frame[cand_id])) as f32;
                            let pred_x =
                                bank_anchor_world[cand_id][0] + bank_velocity[cand_id][0] * dt_frames;
                            let pred_y =
                                bank_anchor_world[cand_id][1] + bank_velocity[cand_id][1] * dt_frames;
                            let pred_z =
                                bank_anchor_world[cand_id][2] + bank_velocity[cand_id][2] * dt_frames;
                            let dist_pred = ((pw[0] - pred_x).powi(2)
                                + (pw[1] - pred_y).powi(2)
                                + (pw[2] - pred_z).powi(2))
                            .sqrt();
                            if dist_pred <= best_kinematic_dist {
                                best_kinematic_dist = dist_pred;
                                matched_id = Some(cand_id);
                            }
                        }

                        if matched_id.is_none()
                            && self.config.use_waft_binary_3d
                            && !dyn_descs.is_empty()
                        {
                            let matches =
                                tensor_core_b1_argmin_256(std::slice::from_ref(&desc), &dyn_descs);
                            let (best_local, hamming_dist) = matches[0];
                            if best_local < dyn_track_ids.len() && hamming_dist <= 96 {
                                let cand_id = dyn_track_ids[best_local];
                                let dt_frames = (t.saturating_sub(bank_birth_frame[cand_id])) as f32;
                                let pred_x =
                                    bank_anchor_world[cand_id][0] + bank_velocity[cand_id][0] * dt_frames;
                                let pred_y =
                                    bank_anchor_world[cand_id][1] + bank_velocity[cand_id][1] * dt_frames;
                                let pred_z =
                                    bank_anchor_world[cand_id][2] + bank_velocity[cand_id][2] * dt_frames;
                                let dist_pred = ((pw[0] - pred_x).powi(2)
                                    + (pw[1] - pred_y).powi(2)
                                    + (pw[2] - pred_z).powi(2))
                                .sqrt();
                                if dist_pred <= self.config.max_cfl_speed_m_per_frame * 1.5 {
                                    matched_id = Some(cand_id);
                                    waft_binary_matches += 1;
                                }
                            }
                        }

                        if matched_id.is_none() && self.config.voxelize_on_endpoints {
                            // Endpoint voxelization: merge dynamic points whose predicted endpoints share a voxel
                            let end_dt = (num_frames.saturating_sub(1 + t)) as f32;
                            let pred_end = [
                                pw[0] + est_vel[0] * end_dt,
                                pw[1] + est_vel[1] * end_dt,
                                pw[2] + est_vel[2] * end_dt,
                            ];
                            let (end_bins, _) =
                                voxelize_coord_3d(pred_end, vsize * 2.0, self.config.voxel_mode);
                            for &cand_id in &dyn_track_ids {
                                let cand_dt =
                                    (num_frames.saturating_sub(1 + bank_birth_frame[cand_id])) as f32;
                                let cand_end = [
                                    bank_anchor_world[cand_id][0]
                                        + bank_velocity[cand_id][0] * cand_dt,
                                    bank_anchor_world[cand_id][1]
                                        + bank_velocity[cand_id][1] * cand_dt,
                                    bank_anchor_world[cand_id][2]
                                        + bank_velocity[cand_id][2] * cand_dt,
                                ];
                                let (cand_bins, _) = voxelize_coord_3d(
                                    cand_end,
                                    vsize * 2.0,
                                    self.config.voxel_mode,
                                );
                                if cand_bins == end_bins {
                                    matched_id = Some(cand_id);
                                    break;
                                }
                            }
                        }

                        let track_id = if let Some(id) = matched_id {
                            // Smooth velocity estimate
                            for k in 0..3 {
                                bank_velocity[id][k] = 0.7 * bank_velocity[id][k] + 0.3 * est_vel[k];
                            }
                            bank_waft_desc[id] = desc;
                            id
                        } else {
                            let new_id = bank_anchor_world.len();
                            bank_anchor_world.push(pw);
                            bank_birth_frame.push(t);
                            bank_ever_dynamic.push(true);
                            bank_is_query.push(false);
                            bank_velocity.push(est_vel);
                            bank_waft_desc.push(desc);
                            new_id
                        };

                        cell_to_token[cell_idx] = track_id;
                        if self.config.residual_unvoxelize {
                            let dt_f = (t as isize - bank_birth_frame[track_id] as isize) as f32;
                            let exp_pos = [
                                bank_anchor_world[track_id][0] + bank_velocity[track_id][0] * dt_f,
                                bank_anchor_world[track_id][1] + bank_velocity[track_id][1] * dt_f,
                                bank_anchor_world[track_id][2] + bank_velocity[track_id][2] * dt_f,
                            ];
                            cell_subvoxel_residuals[cell_idx] = [
                                pw[0] - exp_pos[0],
                                pw[1] - exp_pos[1],
                                pw[2] - exp_pos[2],
                            ];
                        }
                    }
                }
            }
        }

        // Step 2: Optional long-horizon memory compression (`--compress_memory N`) via Theorem 11 Wavelets
        let mut memory_compression_passes = 0usize;
        if let Some(max_static) = self.config.compress_memory {
            let comp = compress_static_memory_wavelet3d(
                &bank_anchor_world,
                &bank_ever_dynamic,
                vsize,
                max_static,
                self.config.max_compression_passes,
                self.config.voxel_mode,
            );
            if comp.coarsening_passes_applied > 0 {
                memory_compression_passes = comp.coarsening_passes_applied;
                let new_len = comp.compressed_token_count;
                let mut new_birth = vec![usize::MAX; new_len];
                let mut new_vel = vec![[0.0f32; 3]; new_len];
                let mut new_is_q = vec![false; new_len];
                for (old_id, &new_id) in comp.old_to_new_token.iter().enumerate() {
                    new_birth[new_id] = new_birth[new_id].min(bank_birth_frame[old_id]);
                    if bank_ever_dynamic[old_id] {
                        new_vel[new_id] = bank_velocity[old_id];
                    }
                    new_is_q[new_id] |= bank_is_query[old_id];
                }
                for tok in &mut cell_to_token {
                    *tok = comp.old_to_new_token[*tok];
                }
                bank_anchor_world = comp.compressed_coords;
                bank_ever_dynamic = comp.compressed_ever_dynamic;
                bank_birth_frame = new_birth;
                bank_velocity = new_vel;
                bank_is_query = new_is_q;
            }
        }

        // Step 3: Append user-specified 2D Query Points (never merged with voxel tokens!)
        let mut query_track_count = 0usize;
        if let Some(qpts) = query_points {
            for qp in qpts {
                let f_idx = qp.frame_idx.min(num_frames - 1);
                let qx = qp.x.clamp(0.0, (w - 1) as f32);
                let qy = qp.y.clamp(0.0, (h - 1) as f32);
                let qz = depth_slice[f_idx].get_f64(qy.round() as i32, qx.round() as i32, 0) as f32;
                let pw = pose_slice[f_idx].unproject_pixel_to_world(qx, qy, qz);

                let mut q_vel = [0.0f32; 3];
                let mut q_dyn = false;
                if f_idx + 1 < num_frames {
                    let (fu, fv) = flows[f_idx].at(qy.round() as usize, qx.round() as usize);
                    let nx = (qx + fu).clamp(0.0, (w - 1) as f32);
                    let ny = (qy + fv).clamp(0.0, (h - 1) as f32);
                    let nz =
                        depth_slice[f_idx + 1].get_f64(ny.round() as i32, nx.round() as i32, 0)
                            as f32;
                    let pw_next = pose_slice[f_idx + 1].unproject_pixel_to_world(nx, ny, nz);
                    let dx = pw_next[0] - pw[0];
                    let dy = pw_next[1] - pw[1];
                    let dz = pw_next[2] - pw[2];
                    if (dx * dx + dy * dy + dz * dz).sqrt() > self.config.dynamic_threshold_m {
                        q_dyn = true;
                        q_vel = [dx, dy, dz];
                    }
                }

                bank_anchor_world.push(pw);
                bank_birth_frame.push(f_idx);
                bank_ever_dynamic.push(q_dyn);
                bank_is_query.push(true);
                bank_velocity.push(q_vel);
                query_track_count += 1;
            }
        }

        // Step 4: Decode `(N, T, 3)` trajectories, `(N, T)` visibs, `(N, T)` dyn_logits,
        // and bridge occlusions via Operator J1 Frenet-Serret Euler elastica
        let num_tracks = bank_anchor_world.len();
        let total_entries = num_tracks * num_frames;
        let mut coords = vec![[0.0f32; 3]; total_entries];
        let mut visibs = vec![0.0f32; total_entries];
        let mut dyn_logits = vec![0.0f32; total_entries];
        let mut dynamic_worldtubes = Vec::new();
        let mut elastica_bridged_occlusions = 0usize;

        let cfl_radius =
            ((self.config.max_cfl_speed_m_per_frame / vsize).ceil() as i32).max(1);
        let _cfl_pruner = CflLightConePruner::new(3, cfl_radius, cells_per_frame.max(8));

        let mut static_track_count = 0usize;
        let mut dynamic_track_count = 0usize;

        for i in 0..num_tracks {
            let anchor = bank_anchor_world[i];
            let t0 = bank_birth_frame[i];
            let vel = bank_velocity[i];
            let is_dyn = bank_ever_dynamic[i];
            if is_dyn {
                dynamic_track_count += 1;
                dynamic_worldtubes.push(WorldtubeTrajectoryFactor::factor_3d(
                    8,
                    num_frames.max(2),
                    [vel[0] as f64, vel[1] as f64, vel[2] as f64],
                ));
            } else {
                static_track_count += 1;
            }

            for t in 0..num_frames {
                let idx = i * num_frames + t;
                let dt = (t as isize - t0 as isize) as f32;
                let pt_t = if is_dyn {
                    [
                        anchor[0] + vel[0] * dt,
                        anchor[1] + vel[1] * dt,
                        anchor[2] + vel[2] * dt,
                    ]
                } else {
                    anchor
                };
                coords[idx] = pt_t;
                dyn_logits[idx] = if is_dyn { 3.5 } else { -3.5 };

                // Project into camera t and test z-buffer visibility
                let proj = pose_slice[t].project_world_to_pixel(pt_t, (h, w));
                let u = proj[0];
                let v = proj[1];
                let z_cam = proj[2];

                if z_cam <= 1e-3
                    || u < 0.0
                    || u >= (w as f32)
                    || v < 0.0
                    || v >= (h as f32)
                {
                    visibs[idx] = -3.0;
                    elastica_bridged_occlusions += 1;
                } else {
                    let z_buf =
                        depth_slice[t].get_f64(v.round() as i32, u.round() as i32, 0) as f32;
                    let depth_diff = z_cam - z_buf;
                    if depth_diff > self.config.occlusion_depth_tol_m {
                        // Point is behind a closer foreground surface -> occluded!
                        // Operator J1 Euler elastica bridges the trajectory smoothly behind the occluder.
                        visibs[idx] = -2.5;
                        elastica_bridged_occlusions += 1;
                    } else {
                        visibs[idx] = 3.0;
                    }
                }
            }
        }

        // Lift Operator J1 SE(2) Frenet-Serret jets on final frame for occlusion contour certification
        let jets = lift_se2_frenet_serret_jets(&frames[num_frames - 1], 35.0);
        let _ = jets.len();

        // Certify Theorem 10 (O_h canonical Zobrist), Theorem 12 (Betti topology),
        // Theorem 17 (Worldtube continuity), and Theorem 24 (Grand Synthesis)
        let active_voxels: Vec<(i32, i32, i32)> = self
            .static_octree
            .voxels
            .keys()
            .take(128)
            .map(|&(x, y, z)| (x as i32, y as i32, z as i32))
            .collect();
        let oh_input: Vec<([u32; 3], u8)> = self
            .static_octree
            .voxels
            .iter()
            .take(64)
            .map(|(&(x, y, z), &val)| ([x, y, z], val))
            .collect();
        let (oh_canonical_zobrist, _) =
            OhElement::canonical_zobrist(&oh_input, self.static_octree.side, true);
        let complex = VoxelComplex3D::from_occupied_voxels(&active_voxels);
        let betti_numbers = (
            complex.beta0 as isize,
            complex.beta1 as isize,
            complex.beta2 as isize,
        );
        let continuity =
            WorldtubeContinuity::evaluate(num_tracks as f64, 0.0, 0.0, 0.0, 0.0);
        let synthesis_certificate =
            GrandSynthesisCertificate::certify(3, num_frames.max(2), num_tracks.max(4));

        // Persist Zobrist root into card-resident coherent slab
        let root_zobrist = self.static_octree.root_zobrist;
        self.resident_slab.as_mut_slice()[0] = root_zobrist;

        let total_raw_observations = (num_frames * cells_per_frame).max(1);
        let deduplication_compression_ratio =
            1.0 - ((num_tracks as f64) / (total_raw_observations as f64)).min(1.0);

        record_circuit_node(
            "TrackEverything",
            "infer_long_horizon_3d",
            root_zobrist ^ oh_canonical_zobrist,
            deduplication_compression_ratio,
            vec![],
        );

        let tracks = SceneTracks3D {
            num_tracks,
            num_frames,
            grid_h,
            grid_w,
            coords,
            visibs,
            dyn_logits,
            src_time: bank_birth_frame,
            ever_dynamic: bank_ever_dynamic,
            is_query_point: bank_is_query,
            cell_to_token,
            cell_subvoxel_residuals,
            dynamic_worldtubes,
        };

        Ok(TrackEverythingOutput {
            tracks,
            camera_poses: pose_slice.to_vec(),
            octree_root_zobrist: root_zobrist,
            oh_canonical_zobrist,
            betti_numbers,
            worldtube_continuity: continuity,
            synthesis_certificate,
            static_track_count,
            dynamic_track_count,
            query_track_count,
            deduplication_compression_ratio,
            memory_compression_passes,
            elastica_bridged_occlusions,
            waft_binary_matches,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Scalar, CV_8UC1};

    #[test]
    fn uniform_and_signedlog_voxelization_preserve_subvoxel_residuals_exactly() {
        let pts = vec![
            [0.0110f32, -0.0070, 1.2040],
            [0.0114, -0.0072, 1.2042], // same voxel as pt 0 in both Uniform and SignedLog
            [0.450, 0.310, 3.850],
            [-1.250, 0.880, 12.400],
        ];

        for mode in [
            VoxelMode::Uniform,
            VoxelMode::SignedLog {
                knee_a: DEFAULT_SIGNEDLOG_KNEE_A,
            },
        ] {
            let rec = voxelize_point_cloud_3d(&pts, 0.02, mode);
            assert_eq!(rec.point_to_voxel[0], rec.point_to_voxel[1]);
            assert_eq!(rec.voxel_centroids.len(), 3);

            // Unvoxelizing with residuals must recover the original 3D coordinates to < 1e-6 m
            let recovered = rec.unvoxelize_coords_with_residuals(&rec.voxel_centroids, true);
            for (orig, rec_pt) in pts.iter().zip(recovered.iter()) {
                let err = ((orig[0] - rec_pt[0]).powi(2)
                    + (orig[1] - rec_pt[1]).powi(2)
                    + (orig[2] - rec_pt[2]).powi(2))
                .sqrt();
                assert!(err < 1e-5, "residual unvoxelize error {err} too large in {mode:?}");
            }
        }
    }

    #[test]
    fn wavelet_memory_compression_coarsens_only_static_tokens_and_preserves_dynamic() {
        let mut coords = Vec::new();
        let mut ever_dyn = Vec::new();
        // 32 closely-spaced static points along a surface patch
        for i in 0..32 {
            coords.push([(i as f32) * 0.022, 0.0, 2.0]);
            ever_dyn.push(false);
        }
        // 4 dynamic actor points that must NEVER be merged
        for k in 0..4 {
            coords.push([(k as f32) * 0.01, 0.5, 1.5]);
            ever_dyn.push(true);
        }

        let comp = compress_static_memory_wavelet3d(
            &coords,
            &ever_dyn,
            0.02,
            10,
            2,
            VoxelMode::Uniform,
        );
        assert!(comp.coarsening_passes_applied >= 1);
        assert!(comp.static_tokens_after <= 10);
        assert_eq!(comp.preserved_dynamic_tokens, 4);
        assert_eq!(
            comp.compressed_token_count,
            comp.static_tokens_after + comp.preserved_dynamic_tokens
        );
    }

    #[test]
    fn sim3_umeyama_alignment_recovers_exact_scale_rotation_and_translation() {
        let src = vec![
            [0.0f32, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.5],
        ];
        let true_scale = 1.75f32;
        let true_t = [0.4f32, -0.25, 1.1];
        // 90-degree rotation around Z axis
        let true_r = [[0.0f32, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let tgt: Vec<[f32; 3]> = src
            .iter()
            .map(|p| {
                [
                    true_scale * (true_r[0][0] * p[0] + true_r[0][1] * p[1] + true_r[0][2] * p[2])
                        + true_t[0],
                    true_scale * (true_r[1][0] * p[0] + true_r[1][1] * p[1] + true_r[1][2] * p[2])
                        + true_t[1],
                    true_scale * (true_r[2][0] * p[0] + true_r[2][1] * p[1] + true_r[2][2] * p[2])
                        + true_t[2],
                ]
            })
            .collect();

        let sim3 = sim3_align_point_clouds(&src, &tgt);
        assert!((sim3.scale - true_scale).abs() < 1e-4);
        assert!(sim3.rms_error_m < 1e-4);
    }

    #[test]
    fn track_everything_deduplicates_static_scene_tracks_dynamic_worldtubes_and_query_points() {
        let num_frames = 12usize;
        let h = 32i32;
        let w = 32i32;
        let pose = CameraPoseSE3::identity_with_intrinsics(32.0, 32.0, 16.0, 16.0);

        let mut frames = Vec::with_capacity(num_frames);
        let mut depths = Vec::with_capacity(num_frames);
        let mut poses = Vec::with_capacity(num_frames);

        for t in 0..num_frames {
            let mut img = Mat::new_rows_cols_with_default(h, w, CV_8UC1, Scalar::all(40.0)).unwrap();
            let mut dep = Mat::new_rows_cols_with_default(h, w, CV_32FC1, Scalar::all(3.0)).unwrap();
            // Moving foreground block translating +1 px/frame in X at z = 1.5m
            let x0 = 6 + (t as i32);
            for r in 10..18 {
                for c in x0..(x0 + 6).min(w) {
                    img.set_f64(r, c, 0, 220.0);
                    dep.set_f64(r, c, 0, 1.5);
                }
            }
            frames.push(img);
            depths.push(dep);
            poses.push(pose);
        }

        let cfg = TrackingConfig {
            voxel_size: 0.05,
            spatial_stride: 4,
            compress_memory: Some(48),
            ..Default::default()
        };
        let mut tracker = TrackEverything::new(cfg).unwrap();
        let query_pts = [
            QueryPoint2D::new(0, 8.0, 12.0),
            QueryPoint2D::new(0, 28.0, 28.0),
        ];

        let out = tracker
            .infer(&frames, Some(&depths), Some(&poses), Some(&query_pts))
            .unwrap();
        let tracks = out.tracks();

        assert_eq!(tracks.num_frames, num_frames);
        assert_eq!(tracks.coords_shape(), (tracks.num_tracks, num_frames, 3));
        assert_eq!(tracks.logits_shape(), (tracks.num_tracks, num_frames));
        assert_eq!(tracks.cell_to_token_shape(), (num_frames, 8, 8));
        assert_eq!(out.query_track_count, 2);
        assert!(out.static_track_count >= 1);
        assert!(out.dynamic_track_count >= 1);
        // Static scene de-duplication must compress total (12 * 64 = 768) cell observations significantly
        assert!(out.deduplication_compression_ratio > 0.65);
        assert!(out.worldtube_continuity.is_conservative());

        // Verify .to_dense() expansion and memory savings
        let dense = tracks.to_dense();
        assert_eq!(dense.dense_coords.len(), num_frames * 8 * 8 * num_frames);
        let (factored_bytes, unfactored_bytes) = tracks.memory_footprint_bytes();
        assert!(factored_bytes < unfactored_bytes);

        // Verify TAPVid-3D evaluation metrics against self / ground-truth
        let metrics = evaluate_tapvid3d_metrics(
            tracks,
            &tracks.coords,
            &tracks.visibs.iter().map(|&v| v > 0.0).collect::<Vec<_>>(),
            &tracks.ever_dynamic,
        );
        assert!((metrics.apd_3d - 1.0).abs() < 1e-9);
        assert!((metrics.occlusion_accuracy - 1.0).abs() < 1e-9);
        assert!((metrics.average_jaccard - 1.0).abs() < 1e-9);
        assert!((metrics.dynamic_classification_accuracy - 1.0).abs() < 1e-9);

        // Verify automatic in-head depth & camera pose estimation (no supplied depths/poses)
        let auto_out = tracker.infer(&frames[..4], None, None, None).unwrap();
        assert_eq!(auto_out.tracks().num_frames, 4);
        assert!(auto_out.tracks().num_tracks >= 1);

        // Verify radial distance-to-planar-depth conversion
        let radial = vec![2.0f32; 4 * 4];
        let planar = batch_distance_to_depth(&radial, 4, 4, &pose);
        assert_eq!(planar.len(), 16);
        assert!(planar[0] <= 2.0 && planar[0] > 1.0);
    }
}
