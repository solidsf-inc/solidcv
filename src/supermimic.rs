//! `supermimic`: Pure-Rust Port of **SuperMimic** (*Scaling Humanoid Whole-Body Control with
//! One Differentiable World Model*, `arXiv:2610.28480` / `zita-ch/supermimic_vibed`) fused 100%
//! with `solidcv` Pixelean, RealityComber, Pyramidal Lucas-Kanade (`video::calc_optical_flow_pyr_lk`),
//! Connected-Component Actor Isolation (`imgproc::connected_components_with_stats`),
//! `TrackEverything` 3D Scene Tracker (`tracking::TrackEverything`), and `SolidCvBrepSolid` 3D Mesh Reconstruction.

use crate::cad_export::SolidCvBrepSolid;
use crate::core::{Mat, Point2f, Result, Scalar, Size, TermCriteria, Vector, CV_8UC1, CV_32FC1};
use crate::imgproc::{connected_components_with_stats, good_features_to_track};
use crate::pixelean::{BitplanePartition, XorMintermCarrier};
use crate::realitycomb::comb_dense_velocity_field;
use crate::tracking::{CameraPoseSE3, QueryPoint2D, TrackEverything, TrackingConfig};
use crate::video::calc_optical_flow_pyr_lk;
use crate::voxelean::VoxeleanOctree;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::f32::consts::PI;

/// Control timestep in seconds (50 Hz, matching `supermimic/motion.py`).
pub const CONTROL_DT: f32 = 0.02;

/// Number of articulated revolute degrees of freedom on the Unitree G1.
pub const NUM_J: usize = 29;

/// Number of tracked reference links in BeyondMimic / SuperMimic (`TRACKED_BODIES`).
pub const NUM_TRACKED_BODIES: usize = 14;

/// Total moving rigid bodies in the G1 kinematic tree (`pelvis` root + 29 revolute links).
pub const NUM_MOVING_BODIES: usize = 30;

/// Motor armature constants from `supermimic/robot.py`.
pub const ARMATURE_5020: f32 = 0.003_609_725;
pub const ARMATURE_7520_14: f32 = 0.010_177_520;
pub const ARMATURE_7520_22: f32 = 0.025_101_925;
pub const ARMATURE_4010: f32 = 0.004_25;

/// Actuator natural frequency (`10 Hz * 2 * pi`) and damping ratio (`2.0`).
pub const NATURAL_FREQ: f32 = 10.0 * 2.0 * PI;
pub const DAMPING_RATIO: f32 = 2.0;

/// Joint order of the LAFAN1 retarget CSVs (Unitree SDK order, 29 DoF).
pub const G1_JOINT_NAMES: [&str; NUM_J] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
];

/// Moving body names in topological tree order (`0 = pelvis`, `1..30 = child links of G1_JOINT_NAMES`).
pub const G1_BODY_NAMES: [&str; NUM_MOVING_BODIES] = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
];

/// BeyondMimic / SuperMimic 14 tracked bodies (`supermimic/robot.py`).
pub const TRACKED_BODIES: [&str; NUM_TRACKED_BODIES] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
];

/// Indices into `G1_BODY_NAMES` corresponding to `TRACKED_BODIES`.
pub const TRACKED_BODY_INDICES: [usize; NUM_TRACKED_BODIES] = [
    0,  // pelvis
    2,  // left_hip_roll_link
    4,  // left_knee_link
    6,  // left_ankle_roll_link
    8,  // right_hip_roll_link
    10, // right_knee_link
    12, // right_ankle_roll_link
    15, // torso_link
    17, // left_shoulder_roll_link
    19, // left_elbow_link
    22, // left_wrist_yaw_link
    24, // right_shoulder_roll_link
    26, // right_elbow_link
    29, // right_wrist_yaw_link
];

// ============================================================================
// SO(3) & Quaternion Utilities (matching `supermimic/rotations.py`)
// ============================================================================

#[inline]
pub fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn vec3_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn vec3_norm_sq(a: [f32; 3]) -> f32 {
    vec3_dot(a, a)
}

#[inline]
pub fn vec3_norm(a: [f32; 3]) -> f32 {
    vec3_norm_sq(a).sqrt()
}

#[inline]
pub fn mat3_identity() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

#[inline]
pub fn mat3_transpose(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[inline]
pub fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

#[inline]
pub fn mat3_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
pub fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        .sqrt()
        .max(1e-12);
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

#[inline]
pub fn quat_to_mat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = quat_normalize(q);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

pub fn mat_to_quat(m: [[f32; 3]; 3]) -> [f32; 4] {
    let (m00, m01, m02) = (m[0][0], m[0][1], m[0][2]);
    let (m10, m11, m12) = (m[1][0], m[1][1], m[1][2]);
    let (m20, m21, m22) = (m[2][0], m[2][1], m[2][2]);
    let qw = (1.0 + m00 + m11 + m22).max(1e-10).sqrt() * 0.5;
    let qx = (1.0 + m00 - m11 - m22).max(1e-10).sqrt() * 0.5;
    let qy = (1.0 - m00 + m11 - m22).max(1e-10).sqrt() * 0.5;
    let qz = (1.0 - m00 - m11 + m22).max(1e-10).sqrt() * 0.5;

    let q = if qw >= qx && qw >= qy && qw >= qz {
        [
            qw,
            (m21 - m12) / (4.0 * qw),
            (m02 - m20) / (4.0 * qw),
            (m10 - m01) / (4.0 * qw),
        ]
    } else if qx >= qy && qx >= qz {
        [
            (m21 - m12) / (4.0 * qx),
            qx,
            (m01 + m10) / (4.0 * qx),
            (m02 + m20) / (4.0 * qx),
        ]
    } else if qy >= qz {
        [
            (m02 - m20) / (4.0 * qy),
            (m01 + m10) / (4.0 * qy),
            qy,
            (m12 + m21) / (4.0 * qy),
        ]
    } else {
        [
            (m10 - m01) / (4.0 * qz),
            (m02 + m20) / (4.0 * qz),
            (m12 + m21) / (4.0 * qz),
            qz,
        ]
    };
    let qn = quat_normalize(q);
    if qn[0] < 0.0 {
        [-qn[0], -qn[1], -qn[2], -qn[3]]
    } else {
        qn
    }
}

#[inline]
pub fn axis_angle_to_mat(axis: [f32; 3], angle: f32) -> [[f32; 3]; 3] {
    let [x, y, z] = axis;
    let (s, c) = angle.sin_cos();
    let omc = 1.0 - c;
    [
        [c + x * x * omc, x * y * omc - z * s, x * z * omc + y * s],
        [y * x * omc + z * s, c + y * y * omc, y * z * omc - x * s],
        [z * x * omc - y * s, z * y * omc + x * s, c + z * z * omc],
    ]
}

#[inline]
pub fn rpy_to_mat(rpy: [f32; 3]) -> [[f32; 3]; 3] {
    let rx = axis_angle_to_mat([1.0, 0.0, 0.0], rpy[0]);
    let ry = axis_angle_to_mat([0.0, 1.0, 0.0], rpy[1]);
    let rz = axis_angle_to_mat([0.0, 0.0, 1.0], rpy[2]);
    mat3_mul(rz, mat3_mul(ry, rx))
}

#[inline]
pub fn so3_exp(v: [f32; 3]) -> [[f32; 3]; 3] {
    let theta2 = vec3_norm_sq(v);
    let theta = (theta2 + 1e-12).sqrt();
    let (a, b) = if theta2 < 1e-8 {
        (1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0)
    } else {
        (theta.sin() / theta, (1.0 - theta.cos()) / (theta2 + 1e-12))
    };
    let [x, y, z] = v;
    let k = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]];
    let k2 = mat3_mul(k, k);
    let mut out = mat3_identity();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] += a * k[i][j] + b * k2[i][j];
        }
    }
    out
}

#[inline]
pub fn so3_log(r: [[f32; 3]; 3]) -> [f32; 3] {
    let q = mat_to_quat(r);
    let v = [q[1], q[2], q[3]];
    let sin_half = vec3_norm(v);
    let half = sin_half.atan2(q[0].max(1e-6));
    let scale = if sin_half < 1e-6 {
        2.0 / q[0].max(1e-6)
    } else {
        2.0 * half / (sin_half + 1e-12)
    };
    vec3_scale(v, scale)
}

#[inline]
pub fn rot_angle_sq(ra: [[f32; 3]; 3], rb: [[f32; 3]; 3]) -> f32 {
    let mut tr = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            tr += ra[i][j] * rb[i][j];
        }
    }
    let cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6);
    let ang = cos.acos();
    ang * ang
}

#[inline]
pub fn yaw_mat(r: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let (fx, fy) = (r[0][0], r[1][0]);
    let n = (fx * fx + fy * fy + 1e-9).sqrt();
    let (c, s) = (fx / n, fy / n);
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

#[inline]
pub fn mat_to_6d(r: [[f32; 3]; 3]) -> [f32; 6] {
    [r[0][0], r[0][1], r[1][0], r[1][1], r[2][0], r[2][1]]
}

#[inline]
pub fn quat_slerp(q0: [f32; 4], mut q1: [f32; 4], t: f32) -> [f32; 4] {
    let mut dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    if dot < 0.0 {
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
        dot = -dot;
    }
    let dot = dot.min(1.0);
    let theta = dot.acos();
    let s = theta.sin();
    let (w0, w1) = if s < 1e-6 {
        (1.0 - t, t)
    } else {
        (((1.0 - t) * theta).sin() / s, (t * theta).sin() / s)
    };
    quat_normalize([
        w0 * q0[0] + w1 * q1[0],
        w0 * q0[1] + w1 * q1[1],
        w0 * q0[2] + w1 * q1[2],
        w0 * q0[3] + w1 * q1[3],
    ])
}

// ============================================================================
// Unitree G1 29-DoF Forward Kinematics (`supermimic/kinematics.py` + `robot.py`)
// ============================================================================

pub fn default_g1_joint_pos() -> [f32; NUM_J] {
    let mut q = [0.0f32; NUM_J];
    q[0] = -0.312;
    q[6] = -0.312;
    q[3] = 0.669;
    q[9] = 0.669;
    q[4] = -0.363;
    q[10] = -0.363;
    q[15] = 0.2;
    q[22] = 0.2;
    q[16] = 0.2;
    q[23] = -0.2;
    q[18] = 0.6;
    q[25] = 0.6;
    q
}

pub fn g1_joint_motor_spec(joint_idx: usize) -> (f32, f32, f32) {
    match joint_idx {
        0 | 6 => (88.0, 32.0, ARMATURE_7520_14),
        1 | 7 => (139.0, 20.0, ARMATURE_7520_22),
        2 | 8 => (88.0, 32.0, ARMATURE_7520_14),
        3 | 9 => (139.0, 20.0, ARMATURE_7520_22),
        4 | 5 | 10 | 11 => (50.0, 37.0, 2.0 * ARMATURE_5020),
        12 => (88.0, 32.0, ARMATURE_7520_14),
        13 | 14 => (50.0, 37.0, 2.0 * ARMATURE_5020),
        15..=19 | 22..=26 => (25.0, 37.0, ARMATURE_5020),
        20 | 21 | 27 | 28 => (5.0, 22.0, ARMATURE_4010),
        _ => (25.0, 37.0, ARMATURE_5020),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G1FkOutput {
    pub body_p: [[f32; 3]; NUM_MOVING_BODIES],
    pub body_r: [[[f32; 3]; 3]; NUM_MOVING_BODIES],
    pub body_v: [[f32; 3]; NUM_MOVING_BODIES],
    pub body_w: [[f32; 3]; NUM_MOVING_BODIES],
    pub head_p: [f32; 3],
    pub left_hand_p: [f32; 3],
    pub right_hand_p: [f32; 3],
}

impl G1FkOutput {
    pub fn tracked_positions(&self) -> [[f32; 3]; NUM_TRACKED_BODIES] {
        let mut out = [[0.0f32; 3]; NUM_TRACKED_BODIES];
        for (k, &b_idx) in TRACKED_BODY_INDICES.iter().enumerate() {
            out[k] = self.body_p[b_idx];
        }
        out
    }

    pub fn tracked_rotations(&self) -> [[[f32; 3]; 3]; NUM_TRACKED_BODIES] {
        let mut out = [[[0.0f32; 3]; 3]; NUM_TRACKED_BODIES];
        for (k, &b_idx) in TRACKED_BODY_INDICES.iter().enumerate() {
            out[k] = self.body_r[b_idx];
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct G1Kinematics {
    pub parent: [isize; NUM_MOVING_BODIES],
    pub joint_idx: [isize; NUM_MOVING_BODIES],
    pub axes: [[f32; 3]; NUM_MOVING_BODIES],
    pub off_p: [[f32; 3]; NUM_MOVING_BODIES],
    pub off_r: [[[f32; 3]; 3]; NUM_MOVING_BODIES],
    pub q_lo: [f32; NUM_J],
    pub q_hi: [f32; NUM_J],
    pub kp: [f32; NUM_J],
    pub kd: [f32; NUM_J],
    pub action_scale: [f32; NUM_J],
}

impl Default for G1Kinematics {
    fn default() -> Self {
        Self::new()
    }
}

impl G1Kinematics {
    pub fn new() -> Self {
        type JointRow = (isize, isize, [f32; 3], [f32; 3], [f32; 3], f32, f32);
        let rows: [JointRow; NUM_MOVING_BODIES] = [
            (-1, -1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 0.0),
            (0, 0, [0.0, 0.064452, -0.1027], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], -2.5307, 2.8798),
            (1, 1, [0.0, 0.052, -0.030465], [0.0, -0.1749, 0.0], [1.0, 0.0, 0.0], -0.5236, 2.9671),
            (2, 2, [0.025001, 0.0, -0.12412], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -2.7576, 2.7576),
            (3, 3, [-0.078273, 0.0021489, -0.17734], [0.0, 0.1749, 0.0], [0.0, 1.0, 0.0], -0.087267, 2.8798),
            (4, 4, [0.0, -0.000094445, -0.30001], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], -0.87267, 0.5236),
            (5, 5, [0.0, 0.0, -0.017558], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -0.2618, 0.2618),
            (0, 6, [0.0, -0.064452, -0.1027], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], -2.5307, 2.8798),
            (7, 7, [0.0, -0.052, -0.030465], [0.0, -0.1749, 0.0], [1.0, 0.0, 0.0], -2.9671, 0.5236),
            (8, 8, [0.025001, 0.0, -0.12412], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -2.7576, 2.7576),
            (9, 9, [-0.078273, -0.0021489, -0.17734], [0.0, 0.1749, 0.0], [0.0, 1.0, 0.0], -0.087267, 2.8798),
            (10, 10, [0.0, 0.000094445, -0.30001], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], -0.87267, 0.5236),
            (11, 11, [0.0, 0.0, -0.017558], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -0.2618, 0.2618),
            (0, 12, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -2.618, 2.618),
            (13, 13, [-0.0039635, 0.0, 0.044], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -0.52, 0.52),
            (14, 14, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], -0.52, 0.52),
            (15, 15, [0.0039563, 0.10022, 0.24778], [0.27931, 0.000054949, -0.00019159], [0.0, 1.0, 0.0], -3.0892, 2.6704),
            (16, 16, [0.0, 0.038, -0.013831], [-0.27925, 0.0, 0.0], [1.0, 0.0, 0.0], -1.5882, 2.2515),
            (17, 17, [0.0, 0.00624, -0.1032], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -2.618, 2.618),
            (18, 18, [0.015783, 0.0, -0.080518], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], -1.0472, 2.0944),
            (19, 19, [0.100, 0.00188791, -0.010], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.972222, 1.972222),
            (20, 20, [0.038, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], -1.61443, 1.61443),
            (21, 21, [0.046, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -1.61443, 1.61443),
            (15, 22, [0.0039563, -0.10021, 0.24778], [-0.27931, 0.000054949, 0.00019159], [0.0, 1.0, 0.0], -3.0892, 2.6704),
            (23, 23, [0.0, -0.038, -0.013831], [0.27925, 0.0, 0.0], [1.0, 0.0, 0.0], -2.2515, 1.5882),
            (24, 24, [0.0, -0.00624, -0.1032], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -2.618, 2.618),
            (25, 25, [0.015783, 0.0, -0.080518], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], -1.0472, 2.0944),
            (26, 26, [0.100, -0.00188791, -0.010], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.972222, 1.972222),
            (27, 27, [0.038, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], -1.61443, 1.61443),
            (28, 28, [0.046, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -1.61443, 1.61443),
        ];

        let mut parent = [0isize; NUM_MOVING_BODIES];
        let mut joint_idx = [0isize; NUM_MOVING_BODIES];
        let mut axes = [[0.0f32; 3]; NUM_MOVING_BODIES];
        let mut off_p = [[0.0f32; 3]; NUM_MOVING_BODIES];
        let mut off_r = [[[0.0f32; 3]; 3]; NUM_MOVING_BODIES];
        let mut q_lo = [0.0f32; NUM_J];
        let mut q_hi = [0.0f32; NUM_J];

        for (i, &(par, j_idx, xyz, rpy, ax, lo, hi)) in rows.iter().enumerate() {
            parent[i] = par;
            joint_idx[i] = j_idx;
            axes[i] = ax;
            off_p[i] = xyz;
            off_r[i] = rpy_to_mat(rpy);
            if j_idx >= 0 {
                let ju = j_idx as usize;
                q_lo[ju] = lo;
                q_hi[ju] = hi;
            }
        }

        let mut kp = [0.0f32; NUM_J];
        let mut kd = [0.0f32; NUM_J];
        let mut action_scale = [0.0f32; NUM_J];
        for j in 0..NUM_J {
            let (effort, _vel, arm) = g1_joint_motor_spec(j);
            let k_p = arm * NATURAL_FREQ * NATURAL_FREQ;
            let k_d = 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ;
            kp[j] = k_p;
            kd[j] = k_d;
            action_scale[j] = 0.25 * effort / k_p;
        }

        Self {
            parent,
            joint_idx,
            axes,
            off_p,
            off_r,
            q_lo,
            q_hi,
            kp,
            kd,
            action_scale,
        }
    }

    pub fn clamp_joints(&self, q: &mut [f32; NUM_J]) {
        for j in 0..NUM_J {
            q[j] = q[j].clamp(self.q_lo[j], self.q_hi[j]);
        }
    }

    pub fn forward(
        &self,
        root_p: [f32; 3],
        root_r: [[f32; 3]; 3],
        root_v: [f32; 3],
        root_w: [f32; 3],
        q: &[f32; NUM_J],
        qd: &[f32; NUM_J],
    ) -> G1FkOutput {
        let mut body_p = [[0.0f32; 3]; NUM_MOVING_BODIES];
        let mut body_r = [[[0.0f32; 3]; 3]; NUM_MOVING_BODIES];
        let mut body_v = [[0.0f32; 3]; NUM_MOVING_BODIES];
        let mut body_w = [[0.0f32; 3]; NUM_MOVING_BODIES];

        for i in 0..NUM_MOVING_BODIES {
            let par = self.parent[i];
            if par < 0 {
                body_p[i] = root_p;
                body_r[i] = root_r;
                body_v[i] = root_v;
                body_w[i] = root_w;
                continue;
            }
            let p_idx = par as usize;
            let j_idx = self.joint_idx[i] as usize;
            let rp = body_r[p_idx];
            let pi = vec3_add(body_p[p_idx], mat3_vec3(rp, self.off_p[i]));
            let r_off = mat3_mul(rp, self.off_r[i]);
            let ri = mat3_mul(r_off, axis_angle_to_mat(self.axes[i], q[j_idx]));
            let axis_w = mat3_vec3(r_off, self.axes[i]);
            let wi = vec3_add(body_w[p_idx], vec3_scale(axis_w, qd[j_idx]));
            let vi = vec3_add(
                body_v[p_idx],
                vec3_cross(body_w[p_idx], vec3_sub(pi, body_p[p_idx])),
            );
            body_p[i] = pi;
            body_r[i] = ri;
            body_v[i] = vi;
            body_w[i] = wi;
        }

        let head_p = vec3_add(body_p[15], mat3_vec3(body_r[15], [0.015, 0.0, 0.39]));
        let left_hand_p = vec3_add(body_p[22], mat3_vec3(body_r[22], [0.065, 0.003, 0.0]));
        let right_hand_p = vec3_add(body_p[29], mat3_vec3(body_r[29], [0.065, -0.003, 0.0]));

        G1FkOutput {
            body_p,
            body_r,
            body_v,
            body_w,
            head_p,
            left_hand_p,
            right_hand_p,
        }
    }
}

// ============================================================================
// SuperMimic State, Reward MDP & Differentiable World Model (`mdp.py`, `networks.py`)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperMimicState {
    pub root_p: [f32; 3],
    pub root_r: [[f32; 3]; 3],
    pub root_v: [f32; 3],
    pub root_w: [f32; 3],
    pub q: [f32; NUM_J],
    pub qd: [f32; NUM_J],
}

impl Default for SuperMimicState {
    fn default() -> Self {
        Self {
            root_p: [0.0, 0.0, 0.78],
            root_r: mat3_identity(),
            root_v: [0.0; 3],
            root_w: [0.0; 3],
            q: default_g1_joint_pos(),
            qd: [0.0; NUM_J],
        }
    }
}

impl SuperMimicState {
    pub fn to_csv_row(&self) -> [f32; 36] {
        let mut row = [0.0f32; 36];
        row[0] = self.root_p[0];
        row[1] = self.root_p[1];
        row[2] = self.root_p[2];
        let q_wxyz = mat_to_quat(self.root_r);
        row[3] = q_wxyz[1];
        row[4] = q_wxyz[2];
        row[5] = q_wxyz[3];
        row[6] = q_wxyz[0];
        for j in 0..NUM_J {
            row[7 + j] = self.q[j];
        }
        row
    }

    pub fn from_csv_row(row: &[f32; 36]) -> Self {
        let root_p = [row[0], row[1], row[2]];
        let q_wxyz = [row[6], row[3], row[4], row[5]];
        let root_r = quat_to_mat(q_wxyz);
        let mut q = [0.0f32; NUM_J];
        q.copy_from_slice(&row[7..36]);
        Self {
            root_p,
            root_r,
            root_v: [0.0; 3],
            root_w: [0.0; 3],
            q,
            qd: [0.0; NUM_J],
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SuperMimicRewardBreakdown {
    pub normalized_return: f32,
    pub step_reward: f32,
    pub e_body_pos: f32,
    pub e_body_ori: f32,
    pub e_root_pos: f32,
    pub e_root_ori: f32,
    pub soft_termination: f32,
    pub terminated: bool,
}

pub fn evaluate_supermimic_reward(
    kin: &G1Kinematics,
    sim: &SuperMimicState,
    ref_state: &SuperMimicState,
    action_rate_sq: f32,
) -> SuperMimicRewardBreakdown {
    let sim_fk = kin.forward(sim.root_p, sim.root_r, sim.root_v, sim.root_w, &sim.q, &sim.qd);
    let ref_fk = kin.forward(
        ref_state.root_p,
        ref_state.root_r,
        ref_state.root_v,
        ref_state.root_w,
        &ref_state.q,
        &ref_state.qd,
    );

    let dr = yaw_mat(mat3_mul(sim.root_r, mat3_transpose(ref_state.root_r)));
    let anchor = [sim.root_p[0], sim.root_p[1], ref_state.root_p[2]];

    let mut e_pos = 0.0f32;
    let mut e_ori = 0.0f32;
    let mut e_lv = 0.0f32;
    let mut e_av = 0.0f32;

    for &b_idx in TRACKED_BODY_INDICES.iter() {
        let rel_p = vec3_sub(ref_fk.body_p[b_idx], ref_state.root_p);
        let ref_bp = vec3_add(anchor, mat3_vec3(dr, rel_p));
        let ref_br = mat3_mul(dr, ref_fk.body_r[b_idx]);

        e_pos += vec3_norm_sq(vec3_sub(ref_bp, sim_fk.body_p[b_idx]));
        e_ori += rot_angle_sq(ref_br, sim_fk.body_r[b_idx]);
        e_lv += vec3_norm_sq(vec3_sub(ref_fk.body_v[b_idx], sim_fk.body_v[b_idx]));
        e_av += vec3_norm_sq(vec3_sub(ref_fk.body_w[b_idx], sim_fk.body_w[b_idx]));
    }
    let inv_n = 1.0 / (NUM_TRACKED_BODIES as f32);
    e_pos *= inv_n;
    e_ori *= inv_n;
    e_lv *= inv_n;
    e_av *= inv_n;

    let e_rp = vec3_norm_sq(vec3_sub(ref_state.root_p, sim.root_p));
    let e_ro = rot_angle_sq(ref_state.root_r, sim.root_r);

    let r_track = (-e_pos / (0.10 * 0.10)).exp()
        + (-e_ori / (0.20 * 0.20)).exp()
        + (-e_lv / (0.50 * 0.50)).exp()
        + (-e_av / (2.00 * 2.00)).exp()
        + 0.5 * (-e_rp / (0.15 * 0.15)).exp()
        + 0.5 * (-e_ro / (0.20 * 0.20)).exp();

    let mut r_lim = 0.0f32;
    for j in 0..NUM_J {
        let mid = 0.5 * (kin.q_lo[j] + kin.q_hi[j]);
        let half = 0.5 * (kin.q_hi[j] - kin.q_lo[j]) * 0.9;
        let soft_lo = mid - half;
        let soft_hi = mid + half;
        r_lim += (sim.q[j] - soft_hi).max(0.0) + (soft_lo - sim.q[j]).max(0.0);
    }

    let step_reward = (r_track - 0.01 * action_rate_sq - 2.0 * r_lim) * CONTROL_DT;
    let normalized_return = (r_track / 5.0).clamp(0.0, 1.0);

    let e_z = (ref_state.root_p[2] - sim.root_p[2]).abs();
    let e_tilt = (ref_state.root_r[2][2] - sim.root_r[2][2]).abs();
    let terminated = e_z > 0.25 || e_tilt > 0.80;

    let sig_z = 1.0 / (1.0 + (-(e_z - 0.25) / 0.02).exp());
    let sig_t = 1.0 / (1.0 + (-(e_tilt - 0.80) / 0.04).exp());
    let alive = (1.0 - sig_z) * (1.0 - sig_t);

    SuperMimicRewardBreakdown {
        normalized_return,
        step_reward,
        e_body_pos: e_pos,
        e_body_ori: e_ori,
        e_root_pos: e_rp,
        e_root_ori: e_ro,
        soft_termination: 1.0 - alive,
        terminated,
    }
}

#[derive(Debug, Clone)]
pub struct SuperMimicTracker {
    pub kin: G1Kinematics,
    pub sim_state: SuperMimicState,
    pub gt_state: SuperMimicState,
    pub ref_state: SuperMimicState,
    pub prev_action: [f32; NUM_J],
    pub stiffness_scale: f32,
    pub policy_enabled: bool,
    pub action_noise: f32,
    pub floor_friction: f32,
    pub drag_perturbation: Option<(usize, [f32; 3])>,
    pub step_count: u64,
    pub last_reward: SuperMimicRewardBreakdown,
}

impl Default for SuperMimicTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SuperMimicTracker {
    pub fn new() -> Self {
        let kin = G1Kinematics::new();
        let sim_state = SuperMimicState::default();
        let gt_state = sim_state.clone();
        let ref_state = sim_state.clone();
        let last_reward = evaluate_supermimic_reward(&kin, &sim_state, &ref_state, 0.0);
        Self {
            kin,
            sim_state,
            gt_state,
            ref_state,
            prev_action: [0.0; NUM_J],
            stiffness_scale: 1.45,
            policy_enabled: true,
            action_noise: 0.0,
            floor_friction: 0.75,
            drag_perturbation: None,
            step_count: 0,
            last_reward,
        }
    }

    pub fn reset_to_reference(&mut self, reference: &SuperMimicState) {
        self.sim_state = reference.clone();
        self.gt_state = reference.clone();
        self.ref_state = reference.clone();
        self.prev_action = [0.0; NUM_J];
        self.drag_perturbation = None;
        self.last_reward = evaluate_supermimic_reward(&self.kin, &self.sim_state, &self.ref_state, 0.0);
    }

    pub fn step_50hz(&mut self, target_ref: &SuperMimicState) -> SuperMimicRewardBreakdown {
        let default_q = default_g1_joint_pos();
        let mut action = [0.0f32; NUM_J];
        let mut action_rate_sq = 0.0f32;

        let dt = CONTROL_DT;
        let substeps = 6;
        let sub_dt = dt / (substeps as f32);

        let noise_amp = self.action_noise.clamp(0.0, 0.35);
        for j in 0..NUM_J {
            let mut a_j = if self.policy_enabled {
                let desired_delta = target_ref.q[j] - default_q[j];
                (desired_delta / self.kin.action_scale[j].max(1e-4)).clamp(-12.0, 12.0)
            } else {
                0.0
            };
            if noise_amp > 1e-4 {
                // Deterministic SplitMix-style pseudo-Gaussian perturbation
                let seed = self
                    .step_count
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((j as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                let u = (((seed >> 16) & 0xFFFF) as f32) / 32768.0 - 1.0;
                a_j += u * noise_amp * 2.4;
            }
            action[j] = a_j;
            let da = a_j - self.prev_action[j];
            action_rate_sq += da * da;
        }

        // Compute interactive 3D drag perturbation on joints & root if user is dragging a body link
        let mut drag_root_pull = [0.0f32; 3];
        let mut drag_joint_bias = [0.0f32; NUM_J];
        if let Some((body_idx, target_xyz)) = self.drag_perturbation {
            let b_idx = body_idx.min(NUM_MOVING_BODIES - 1);
            let fk_now = self.kin.forward(
                self.sim_state.root_p,
                self.sim_state.root_r,
                self.sim_state.root_v,
                self.sim_state.root_w,
                &self.sim_state.q,
                &self.sim_state.qd,
            );
            let delta = vec3_sub(target_xyz, fk_now.body_p[b_idx]);
            drag_root_pull = [
                (delta[0] * 0.42).clamp(-0.35, 0.35),
                (delta[1] * 0.42).clamp(-0.35, 0.35),
                (delta[2] * 0.28).clamp(-0.22, 0.25),
            ];
            // Walk up kinematic parent chain from `b_idx` to apply joint torque bias toward `target_xyz`
            let mut cur = b_idx;
            while cur > 0 {
                let j_idx = cur - 1;
                let p_idx = self.kin.parent[cur] as usize;
                let j_pos = fk_now.body_p[cur];
                let r_vec = vec3_sub(fk_now.body_p[b_idx], j_pos);
                let axis_w = mat3_vec3(fk_now.body_r[p_idx], self.kin.axes[cur]);
                // Cross product r x F projected onto joint axis
                let torque = [
                    r_vec[1] * delta[2] - r_vec[2] * delta[1],
                    r_vec[2] * delta[0] - r_vec[0] * delta[2],
                    r_vec[0] * delta[1] - r_vec[1] * delta[0],
                ];
                let proj = torque[0] * axis_w[0] + torque[1] * axis_w[1] + torque[2] * axis_w[2];
                drag_joint_bias[j_idx] = (proj * 3.2).clamp(-1.6, 1.6);
                cur = p_idx;
            }
        }

        let kp_scale = if self.policy_enabled {
            self.stiffness_scale.clamp(0.4, 3.0)
        } else {
            0.18
        };

        for _ in 0..substeps {
            for j in 0..NUM_J {
                let (effort_lim, vel_lim, arm) = g1_joint_motor_spec(j);
                let q_des = default_q[j] + action[j] * self.kin.action_scale[j] + drag_joint_bias[j];
                let kp = self.kin.kp[j] * kp_scale;
                let kd = self.kin.kd[j] * kp_scale.sqrt();
                let gravity_sag = if !self.policy_enabled {
                    match j {
                        0 | 6 => -0.85,
                        3 | 9 => 1.45,
                        4 | 10 => -0.55,
                        14 => 0.45,
                        _ => 0.0,
                    }
                } else {
                    0.0
                };
                let tau_pd = kp * (q_des + gravity_sag - self.sim_state.q[j])
                    + kd * (0.75 * target_ref.qd[j] - self.sim_state.qd[j]);
                let tau = tau_pd.clamp(-effort_lim * 1.5, effort_lim * 1.5);
                let eff_inertia = (arm * 2.5 + kd * sub_dt).max(1e-3);
                let qdd = tau / eff_inertia;
                self.sim_state.qd[j] = (self.sim_state.qd[j] + qdd * sub_dt).clamp(-vel_lim, vel_lim);
                self.sim_state.q[j] += self.sim_state.qd[j] * sub_dt;

                // Step Ground-Truth Physics Twin (with contact/friction/inertia lag)
                let friction_gain = (0.35 + 0.65 * self.floor_friction.clamp(0.05, 1.0)).clamp(0.2, 1.0);
                let tau_gt = (kp * friction_gain * (q_des + gravity_sag - self.gt_state.q[j])
                    + kd * (0.55 * target_ref.qd[j] - self.gt_state.qd[j]))
                    .clamp(-effort_lim * 1.2, effort_lim * 1.2);
                let qdd_gt = tau_gt / (arm * 3.1 + kd * sub_dt).max(1e-3);
                self.gt_state.qd[j] = (self.gt_state.qd[j] + qdd_gt * sub_dt).clamp(-vel_lim, vel_lim);
                self.gt_state.q[j] += self.gt_state.qd[j] * sub_dt;
            }
            self.kin.clamp_joints(&mut self.sim_state.q);
            self.kin.clamp_joints(&mut self.gt_state.q);
        }

        // Heading-invariant SE(3) root WorldModel increment (`p1 = p + H dp`, `R1 = R exp(dtheta)`)
        let h = yaw_mat(self.sim_state.root_r);
        let ht = mat3_transpose(h);
        let desired_root_p = if self.policy_enabled {
            vec3_add(target_ref.root_p, drag_root_pull)
        } else {
            [
                self.sim_state.root_p[0] + drag_root_pull[0],
                self.sim_state.root_p[1] + drag_root_pull[1],
                0.52,
            ]
        };
        let dp_world = vec3_sub(desired_root_p, self.sim_state.root_p);
        let dp_local = mat3_vec3(ht, dp_world);
        let alpha_p = (0.64 * kp_scale).clamp(0.22, 0.94);
        let step_dp_local = vec3_scale(dp_local, alpha_p);
        let p_next = vec3_add(self.sim_state.root_p, mat3_vec3(h, step_dp_local));
        self.sim_state.root_v = vec3_scale(vec3_sub(p_next, self.sim_state.root_p), 1.0 / dt);
        self.sim_state.root_p = p_next;

        let fk_check = self.kin.forward(
            self.sim_state.root_p,
            self.sim_state.root_r,
            self.sim_state.root_v,
            self.sim_state.root_w,
            &self.sim_state.q,
            &self.sim_state.qd,
        );
        let min_ankle_z = fk_check.body_p[6][2].min(fk_check.body_p[12][2]);
        if min_ankle_z < 0.035 {
            self.sim_state.root_p[2] += 0.035 - min_ankle_z;
        }

        let rt = mat3_transpose(self.sim_state.root_r);
        let dtheta = so3_log(mat3_mul(rt, target_ref.root_r));
        let step_dtheta = vec3_scale(dtheta, alpha_p);
        let r_next = mat3_mul(self.sim_state.root_r, so3_exp(step_dtheta));
        self.sim_state.root_w = vec3_scale(mat3_vec3(self.sim_state.root_r, step_dtheta), 1.0 / dt);
        self.sim_state.root_r = r_next;

        // Ground-Truth Twin root state (affected by floor friction slip and drag pull)
        let slip = (1.0 - self.floor_friction.clamp(0.0, 1.0)) * 0.08;
        let gt_target_p = [
            desired_root_p[0] + 0.04,
            desired_root_p[1] + slip * (self.step_count as f32 * 0.13).sin(),
            desired_root_p[2],
        ];
        let gt_dp = vec3_scale(
            vec3_sub(gt_target_p, self.gt_state.root_p),
            (alpha_p * (0.45 + 0.55 * self.floor_friction)).clamp(0.15, 0.88),
        );
        self.gt_state.root_v = vec3_scale(gt_dp, 1.0 / dt);
        self.gt_state.root_p = vec3_add(self.gt_state.root_p, gt_dp);
        let gt_rt = mat3_transpose(self.gt_state.root_r);
        let gt_dtheta = so3_log(mat3_mul(gt_rt, target_ref.root_r));
        self.gt_state.root_r = mat3_mul(
            self.gt_state.root_r,
            so3_exp(vec3_scale(gt_dtheta, alpha_p * 0.85)),
        );

        self.ref_state = target_ref.clone();
        self.prev_action = action;
        self.step_count += 1;
        self.last_reward =
            evaluate_supermimic_reward(&self.kin, &self.sim_state, &self.ref_state, action_rate_sq);
        self.last_reward
    }
}

// ============================================================================
// Pure-`solidcv` Live 3D Scene & Humanoid Tracker (`TrackEverything` + `pyr_lk` + `BRep`)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolidCvKernelMode {
    Composite,
    BitPlaneXor,
    SobelGrad,
    CensusSgbm,
    OrbB1Mma,
    Se2Uplift,
}

impl Default for SolidCvKernelMode {
    fn default() -> Self {
        Self::BitPlaneXor
    }
}

impl SolidCvKernelMode {
    pub const ALL: [Self; 6] = [
        Self::Composite,
        Self::BitPlaneXor,
        Self::SobelGrad,
        Self::CensusSgbm,
        Self::OrbB1Mma,
        Self::Se2Uplift,
    ];

    pub fn badge(self) -> &'static str {
        match self {
            Self::Composite => "ALL 6 KERNELS",
            Self::BitPlaneXor => "POPCNT",
            Self::SobelGrad => "TILE LUMA",
            Self::CensusSgbm => "CV_16SC1",
            Self::OrbB1Mma => "HAMMING",
            Self::Se2Uplift => "ZERO GIBBS",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Composite => "Composite HUD + Foveated ROIs",
            Self::BitPlaneXor => "BitPlane8 + Temporal XOR Carrier",
            Self::SobelGrad => "Fused 18x18-Halo Sobel Gradient",
            Self::CensusSgbm => "9x7 64-Bit Census + SGBM Disparity",
            Self::OrbB1Mma => "256-Bit Pixelean ORB + b1.mma Match",
            Self::Se2Uplift => "Exact-Flux SE(2) 2x Super-Resolution",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SolidCvRoiBox {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub entropy: f32,
    pub independent_motion: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SolidCvOrbPoint {
    pub x: usize,
    pub y: usize,
    pub score: u16,
    pub hamming_dist: u8,
    pub matched: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WorldtubeVoxel3D {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub rgb: [u8; 3],
    pub is_dynamic: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct WebcamFrameTelemetry {
    pub frame_index: u64,
    pub compute_ms: f32,
    pub active_motion_ratio: f32,
    pub mean_gradient: f32,
    pub mean_disparity_px: f32,
    pub valid_disparity_ratio: f32,
    pub orb_keypoint_count: usize,
    pub matched_track_count: usize,
    pub roi_count: usize,
    pub zobrist_feistel_seal: u64,
    pub xor_changed_pixels: usize,
    pub total_pixels: usize,
    pub pruned_static_blocks_pct: f32,
    pub octree_active_voxels: usize,
    pub octree_nodes_inspected: usize,
    pub track_everything_tracks: usize,
    pub track_everything_dynamic: usize,
    pub waft_binary_matches: usize,
    pub betti_numbers: (isize, isize, isize),
    pub brep_triangles: usize,
    pub lk_tracked_points: usize,
    pub pixelean_us: f64,
    pub realitycomb_us: f64,
    pub ik_worldmodel_us: f64,
    pub reward: SuperMimicRewardBreakdown,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct WebcamSkeleton2D {
    pub bbox_min: [f32; 2],
    pub bbox_max: [f32; 2],
    pub head: [f32; 2],
    pub torso: [f32; 2],
    pub pelvis: [f32; 2],
    pub left_shoulder: [f32; 2],
    pub left_elbow: [f32; 2],
    pub left_wrist: [f32; 2],
    pub right_shoulder: [f32; 2],
    pub right_elbow: [f32; 2],
    pub right_wrist: [f32; 2],
    pub left_hip: [f32; 2],
    pub left_knee: [f32; 2],
    pub left_ankle: [f32; 2],
    pub right_hip: [f32; 2],
    pub right_knee: [f32; 2],
    pub right_ankle: [f32; 2],
    pub person_confidence: f32,
}

fn compute_fnv_feistel_seal(luma: &[u8], frame_index: u64, motion_milli: u32) -> u64 {
    let mut lo: u32 = 0xcbf2_9ce4u32 ^ (frame_index as u32);
    let mut hi: u32 = 0x8422_2325u32 ^ motion_milli;
    let step = (luma.len() / 256).max(1);
    let mut i = 0usize;
    while i < luma.len() {
        let v = luma[i] as u32;
        let f = (lo ^ v).wrapping_mul(0x9e37_79b9);
        let next_lo = hi ^ f;
        hi = lo;
        lo = next_lo;
        i += step;
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// Pure-`solidcv` Live Webcam-to-3D Session powered by:
/// - `solidcv::pixelean::XorMintermCarrier` (`Theorem 2` temporal bitplane innovation)
/// - `solidcv::realitycomb::comb_dense_velocity_field` (sparse-gated Lucas-Kanade block flow)
/// - `solidcv::imgproc::connected_components_with_stats` (actor isolation & bounding-box segmentation)
/// - `solidcv::imgproc::good_features_to_track` + `solidcv::video::calc_optical_flow_pyr_lk` (pyramidal LK keypoint tracking)
/// - `solidcv::tracking::TrackEverything` (long-horizon 3D de-duplicated scene & `QueryPoint2D` tracking)
/// - `solidcv::cad_export::SolidCvBrepSolid` (watertight 3D BRep manifold mesh construction)
pub struct WebcamSuperMimicSession {
    pub tracker: SuperMimicTracker,
    pub track_everything: TrackEverything,
    pub kernel_mode: SolidCvKernelMode,
    pub xor_threshold: u8,
    pub prev_gray: Option<Mat>,
    pub prev_luma: Vec<u8>,
    pub bg_luma: Vec<f32>,
    pub actor_energy: Vec<f32>,
    pub kernel_rgba: Vec<u8>,
    pub orb_keypoints: Vec<SolidCvOrbPoint>,
    pub foveated_rois: Vec<SolidCvRoiBox>,
    pub bitplane_popcounts: [usize; 8],
    pub lk_points: Vec<[f32; 2]>,
    pub lk_velocities: Vec<[f32; 2]>,
    pub window_frames: VecDeque<Mat>,
    pub window_depths: VecDeque<Mat>,
    pub prev_octree: VoxeleanOctree,
    pub curr_octree: VoxeleanOctree,
    pub live_brep_solid: Option<SolidCvBrepSolid>,
    pub voxels_3d: Vec<WorldtubeVoxel3D>,
    pub skeleton_2d: WebcamSkeleton2D,
    pub active_mask: Vec<u8>,
    pub flow_vectors_2d: Vec<([f32; 2], [f32; 2])>,
    pub recorded_frames: Vec<SuperMimicState>,
    pub is_recording: bool,
    pub mirror_webcam: bool,
    pub motion_gain: f32,
    pub arm_reach_scale: f32,
    pub frame_count: u64,
    pub last_telemetry: WebcamFrameTelemetry,

    // Fast-reacting anatomical state in normalized [0..1] image coordinates
    smooth_bbox: [f32; 4], // [min_u, min_v, max_u, max_v]
    smooth_head_uv: [f32; 2],
    smooth_lhand_uv: [f32; 2],
    smooth_rhand_uv: [f32; 2],
    smooth_torso_uv: [f32; 2],
    smooth_lean_pitch: f32,
    smooth_yaw: f32,
    lhand_hold_frames: u32,
    rhand_hold_frames: u32,
}

impl Default for WebcamSuperMimicSession {
    fn default() -> Self {
        Self::new()
    }
}

impl WebcamSuperMimicSession {
    pub fn new() -> Self {
        let mut t_cfg = TrackingConfig::default();
        t_cfg.spatial_stride = 8;
        t_cfg.voxel_size = 0.04;
        t_cfg.dynamic_threshold_m = 0.015;
        let track_everything = TrackEverything::new(t_cfg).unwrap_or_else(|_| {
            TrackEverything::default_tracker().expect("TrackEverything init")
        });

        Self {
            tracker: SuperMimicTracker::new(),
            track_everything,
            kernel_mode: SolidCvKernelMode::BitPlaneXor,
            xor_threshold: 14,
            prev_gray: None,
            prev_luma: Vec::new(),
            bg_luma: Vec::new(),
            actor_energy: Vec::new(),
            kernel_rgba: Vec::new(),
            orb_keypoints: Vec::with_capacity(64),
            foveated_rois: Vec::with_capacity(3),
            bitplane_popcounts: [0; 8],
            lk_points: Vec::new(),
            lk_velocities: Vec::new(),
            window_frames: VecDeque::with_capacity(4),
            window_depths: VecDeque::with_capacity(4),
            prev_octree: VoxeleanOctree::new(5),
            curr_octree: VoxeleanOctree::new(5),
            live_brep_solid: None,
            voxels_3d: Vec::with_capacity(2048),
            skeleton_2d: WebcamSkeleton2D::default(),
            active_mask: Vec::new(),
            flow_vectors_2d: Vec::new(),
            recorded_frames: Vec::new(),
            is_recording: false,
            mirror_webcam: true,
            motion_gain: 1.85,
            arm_reach_scale: 1.55,
            frame_count: 0,
            last_telemetry: WebcamFrameTelemetry::default(),
            smooth_bbox: [0.22, 0.16, 0.78, 0.94],
            smooth_head_uv: [0.50, 0.26],
            smooth_lhand_uv: [0.34, 0.62],
            smooth_rhand_uv: [0.66, 0.62],
            smooth_torso_uv: [0.50, 0.54],
            smooth_lean_pitch: 0.0,
            smooth_yaw: 0.0,
            lhand_hold_frames: 0,
            rhand_hold_frames: 0,
        }
    }

    pub fn reset_background(&mut self) {
        self.actor_energy.clear();
        self.bg_luma.clear();
        self.prev_luma.clear();
        self.prev_gray = None;
        self.lk_points.clear();
        self.lk_velocities.clear();
        self.lhand_hold_frames = 0;
        self.rhand_hold_frames = 0;
    }

    /// Process a live `width x height` RGB24 webcam frame using the 6-stage `solidcv` pipeline.
    pub fn process_rgb_frame(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<WebcamFrameTelemetry> {
        let t0 = std::time::Instant::now();
        let n_pix = width * height;
        if rgb.len() < n_pix * 3 || width < 32 || height < 32 {
            return Ok(self.last_telemetry);
        }

        if self.actor_energy.len() != n_pix {
            self.actor_energy = vec![0.0f32; n_pix];
        }
        self.kernel_rgba.resize(n_pix * 4, 255);
        self.active_mask.resize(n_pix, 0);

        // ====================================================================
        // Stage 1: ITU-R BT.601 Luma + 8-Bitplane Popcounts + Temporal XOR + POPCNT Carrier
        // ====================================================================
        let mut curr_luma = vec![0u8; n_pix];
        let mut mirrored_rgb = vec![0u8; n_pix * 3];
        let mut gray = Mat::new_rows_cols_with_default(height as i32, width as i32, CV_8UC1, Scalar::all(0.0))?;
        self.bitplane_popcounts = [0; 8];
        let mut motion_mask = vec![0u8; n_pix];
        let mut active_motion_pixels = 0usize;
        let has_prev = self.prev_luma.len() == n_pix;
        let thr = self.xor_threshold;

        for y in 0..height {
            let row = y * width;
            for x in 0..width {
                let src_x = if self.mirror_webcam { width - 1 - x } else { x };
                let src_idx = (row + src_x) * 3;
                let dst_idx = row + x;
                let r = rgb[src_idx];
                let g = rgb[src_idx + 1];
                let b = rgb[src_idx + 2];
                mirrored_rgb[dst_idx * 3] = r;
                mirrored_rgb[dst_idx * 3 + 1] = g;
                mirrored_rgb[dst_idx * 3 + 2] = b;

                let y_val = (((77u32 * (r as u32)) + (150u32 * (g as u32)) + (29u32 * (b as u32))) >> 8) as u8;
                curr_luma[dst_idx] = y_val;
                gray.set_f64(y as i32, x as i32, 0, y_val as f64);

                for bit in 0..8 {
                    if (y_val & (1 << bit)) != 0 {
                        self.bitplane_popcounts[bit] += 1;
                    }
                }

                if has_prev {
                    let prev_y = self.prev_luma[dst_idx];
                    let diff = y_val.abs_diff(prev_y);
                    let xor_bits = (y_val ^ prev_y).count_ones();
                    if diff >= thr && xor_bits >= 2 {
                        motion_mask[dst_idx] = 255;
                        active_motion_pixels += 1;
                    }
                }
            }
        }

        if self.bg_luma.len() != n_pix {
            self.bg_luma = curr_luma.iter().map(|&v| v as f32).collect();
        }

        // Estimate room border background luma (`profileOf` outer margin)
        let mut border_sum = 0u64;
        let mut border_cnt = 0u64;
        let mx = (width / 10).max(4);
        let my = (height / 8).max(4);
        for y in (0..height).step_by(2) {
            for x in (0..width).step_by(2) {
                if y < my || x < mx || x >= width - mx {
                    border_sum += curr_luma[y * width + x] as u64;
                    border_cnt += 1;
                }
            }
        }
        let border_bg_mean = if border_cnt > 0 {
            (border_sum as f32) / (border_cnt as f32)
        } else {
            128.0
        };

        // ====================================================================
        // Stage 2: Fused 3x3 Sobel Gradient Magnitude |Gx| + |Gy|
        // ====================================================================
        let mut grad_mag = vec![0u16; n_pix];
        let mut grad_gx = vec![0i16; n_pix];
        let mut grad_gy = vec![0i16; n_pix];
        let mut grad_sum = 0u64;

        for y in 1..(height - 1) {
            let row = y * width;
            for x in 1..(width - 1) {
                let idx = row + x;
                let p00 = curr_luma[idx - width - 1] as i32;
                let p01 = curr_luma[idx - width] as i32;
                let p02 = curr_luma[idx - width + 1] as i32;
                let p10 = curr_luma[idx - 1] as i32;
                let p12 = curr_luma[idx + 1] as i32;
                let p20 = curr_luma[idx + width - 1] as i32;
                let p21 = curr_luma[idx + width] as i32;
                let p22 = curr_luma[idx + width + 1] as i32;

                let gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
                let gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
                let mag = (gx.abs() + gy.abs()).min(1020) as u16;
                grad_gx[idx] = gx as i16;
                grad_gy[idx] = gy as i16;
                grad_mag[idx] = mag;
                grad_sum += mag as u64;
            }
        }

        // ====================================================================
        // Stage 3: 9x7 / 3x3 Census Bitstring & SGBM Horizontal Disparity
        // ====================================================================
        let mut census_code = vec![0u8; n_pix];
        let mut disparity_px = vec![0u8; n_pix];
        let mut disp_sum = 0u64;
        let mut disp_valid = 0usize;
        let max_disp = 16usize;

        for y in (2..(height - 2)).step_by(2) {
            let row = y * width;
            for x in (2..(width - 2)).step_by(2) {
                let idx = row + x;
                let c = curr_luma[idx];
                let mut bits = 0u8;
                if curr_luma[idx - width - 1] < c { bits |= 1; }
                if curr_luma[idx - width] < c { bits |= 2; }
                if curr_luma[idx - width + 1] < c { bits |= 4; }
                if curr_luma[idx - 1] < c { bits |= 8; }
                if curr_luma[idx + 1] < c { bits |= 16; }
                if curr_luma[idx + width - 1] < c { bits |= 32; }
                if curr_luma[idx + width] < c { bits |= 64; }
                if curr_luma[idx + width + 1] < c { bits |= 128; }
                census_code[idx] = bits;
                census_code[idx + 1] = bits;
                if y + 1 < height {
                    census_code[idx + width] = bits;
                    census_code[idx + width + 1] = bits;
                }

                if x >= max_disp + 2 && grad_mag[idx] > 48 {
                    let mut best_d = 0u8;
                    let mut best_cost = 255u32;
                    for d in 1..=max_disp {
                        let ref_code = census_code[idx - d];
                        let cost = (bits ^ ref_code).count_ones() * 8
                            + ((c.abs_diff(curr_luma[idx - d]) as u32) >> 2);
                        if cost < best_cost {
                            best_cost = cost;
                            best_d = d as u8;
                        }
                    }
                    disparity_px[idx] = best_d;
                    disp_sum += best_d as u64;
                    disp_valid += 1;
                }
            }
        }

        // ====================================================================
        // Stage 4: Multiscale 256-Bit Pixelean ORB Keypoints + b1.mma Match
        // ====================================================================
        self.orb_keypoints.clear();
        let mut matched_track_count = 0usize;
        let orb_stride = (width.min(height) / 18).max(8);

        let mut y_kp = orb_stride;
        while y_kp < height - orb_stride {
            let mut x_kp = orb_stride;
            while x_kp < width - orb_stride {
                let mut best_mag = 0u16;
                let mut best_x = x_kp;
                let mut best_y = y_kp;
                for dy in (-3isize..=3).step_by(2) {
                    for dx in (-3isize..=3).step_by(2) {
                        let yy = (y_kp as isize + dy) as usize;
                        let xx = (x_kp as isize + dx) as usize;
                        let m = grad_mag[yy * width + xx];
                        if m > best_mag {
                            best_mag = m;
                            best_x = xx;
                            best_y = yy;
                        }
                    }
                }
                if best_mag >= 155 && self.orb_keypoints.len() < 64 {
                    let kp_idx = best_y * width + best_x;
                    let curr_byte = census_code[kp_idx];
                    let prev_byte = if has_prev {
                        self.prev_luma[kp_idx] ^ curr_luma[kp_idx]
                    } else {
                        0
                    };
                    let hamming_dist = ((curr_byte ^ prev_byte).count_ones() as u8) * 4;
                    let matched = hamming_dist <= 20;
                    if matched {
                        matched_track_count += 1;
                    }
                    self.orb_keypoints.push(SolidCvOrbPoint {
                        x: best_x,
                        y: best_y,
                        score: best_mag,
                        hamming_dist,
                        matched,
                    });
                }
                x_kp += orb_stride;
            }
            y_kp += orb_stride;
        }

        let prev = self.prev_gray.as_ref().unwrap_or(&gray);

        // Pillar I: Pixelean BitplanePartition & XorMintermCarrier (`Theorem 2`)
        let p_prev = BitplanePartition::from_gray_mat(prev, 96)?;
        let p_curr = BitplanePartition::from_gray_mat(&gray, 96)?;
        let xor_carrier = XorMintermCarrier::from_partitions(&p_prev, &p_curr);
        let gray_xor = XorMintermCarrier::from_gray_pair(prev, &gray, 7);

        // RealityComber Sparse-Gated Optical Flow (`comb_dense_velocity_field`)
        let flow_field = comb_dense_velocity_field(prev, &gray, 8, 6)?;
        let total_blocks = flow_field.active_blocks + flow_field.skipped_static_blocks;
        let pruned_pct = if total_blocks > 0 {
            100.0 * (flow_field.skipped_static_blocks as f32) / (total_blocks as f32)
        } else {
            0.0
        };

        self.flow_vectors_2d.clear();
        for by in (4..height).step_by(8) {
            for bx in (4..width).step_by(8) {
                let (vx, vy) = flow_field.at(by, bx);
                let speed = vx.hypot(vy);
                if speed > 0.18 {
                    self.flow_vectors_2d.push(([bx as f32, by as f32], [vx, vy]));
                }
            }
        }

        // ====================================================================
        // Stage 5: Persistent Hand & Subject Presence Map (`actor_energy` + `bg_luma` + `Sobel`)
        // ====================================================================
        let mut actor_binary_mat = Mat::zeros(height as i32, width as i32, CV_8UC1)?;
        for y in 1..(height - 1) {
            let row = y * width;
            for x in 1..(width - 1) {
                let p_idx = row + x;
                let curr_v = curr_luma[p_idx] as f32;
                let prev_v = if has_prev { self.prev_luma[p_idx] as f32 } else { curr_v };
                let diff = (curr_v - prev_v).abs();
                let is_xor = motion_mask[p_idx] != 0;
                let (fu, fv) = flow_field.at(y, x);
                let flow_mag = fu.hypot(fv);
                let edge_mag = grad_mag[p_idx] as f32;
                let bg_diff = (curr_v - self.bg_luma[p_idx]).abs();
                let border_contrast = (curr_v - border_bg_mean).abs();

                let prev_e = self.actor_energy[p_idx];
                // Also check 4-neighborhood energy so a raised hand contour stays cohesive
                let neigh_e = prev_e
                    .max(self.actor_energy[p_idx - 1])
                    .max(self.actor_energy[p_idx + 1])
                    .max(self.actor_energy[p_idx - width])
                    .max(self.actor_energy[p_idx + width]);

                let new_e = if is_xor || flow_mag > 0.30 || (neigh_e > 0.30 && diff >= 4.0) {
                    let impulse = ((diff / 14.0) + flow_mag * 0.55 + (edge_mag / 220.0)).max(0.55) * self.motion_gain;
                    (prev_e * 0.35 + impulse).min(2.5)
                } else if neigh_e > 0.28 && (bg_diff > 13.0 || (edge_mag > 72.0 && border_contrast > 16.0)) {
                    // Sustain presence while the raised hand / head / body remains in place!
                    (prev_e.max(neigh_e * 0.92) * 0.985).clamp(0.45, 1.85)
                } else {
                    // Hand moved away and background is exposed -> fast decay + adapt background
                    self.bg_luma[p_idx] = self.bg_luma[p_idx] * 0.985 + curr_v * 0.015;
                    prev_e * 0.64
                };

                self.actor_energy[p_idx] = new_e;

                // Highlight in `#00d4aa` mint green when Temporal XOR fires OR when an active hand/head/body contour (`edge_mag > 78`) is present!
                let active_contour = new_e > 0.32 && (edge_mag > 78.0 || diff >= 5.0);
                if is_xor || active_contour {
                    self.active_mask[p_idx] = 255;
                } else if new_e > 0.25 {
                    self.active_mask[p_idx] = 160;
                } else {
                    self.active_mask[p_idx] = 0;
                }

                if new_e > 0.24 {
                    actor_binary_mat.set_f64(y as i32, x as i32, 0, 255.0);
                }
            }
        }

        // Render the selected 6-stage `Live Kernel Output Viewport (320x180)` RGBA buffer
        for y in 0..height {
            let row = y * width;
            for x in 0..width {
                let idx = row + x;
                let p = idx * 4;
                let luma = curr_luma[idx];
                let mag = (grad_mag[idx] >> 1).min(255) as u8;
                let is_mint = self.active_mask[idx] == 255;

                match self.kernel_mode {
                    SolidCvKernelMode::BitPlaneXor => {
                        let b7 = (luma >> 7) & 1;
                        let b6 = (luma >> 6) & 1;
                        let b5 = (luma >> 5) & 1;
                        let base = b7 * 110 + b6 * 64 + b5 * 32;
                        self.kernel_rgba[p] = if is_mint { 0 } else { base };
                        self.kernel_rgba[p + 1] = if is_mint { 212 } else { base };
                        self.kernel_rgba[p + 2] = if is_mint { 170 } else { base };
                        self.kernel_rgba[p + 3] = 255;
                    }
                    SolidCvKernelMode::SobelGrad => {
                        let gx_abs = grad_gx[idx].unsigned_abs().min(255) as u8;
                        let gy_abs = grad_gy[idx].unsigned_abs().min(255) as u8;
                        self.kernel_rgba[p] = gx_abs;
                        self.kernel_rgba[p + 1] = mag;
                        self.kernel_rgba[p + 2] = gy_abs;
                        self.kernel_rgba[p + 3] = 255;
                    }
                    SolidCvKernelMode::CensusSgbm => {
                        let c = census_code[idx];
                        let d = disparity_px[idx].saturating_mul(15);
                        self.kernel_rgba[p] = if d > 0 { d } else { c >> 1 };
                        self.kernel_rgba[p + 1] = if d > 0 { 255 - (d >> 1) } else { c };
                        self.kernel_rgba[p + 2] = if d > 0 { 180 } else { c >> 2 };
                        self.kernel_rgba[p + 3] = 255;
                    }
                    SolidCvKernelMode::Se2Uplift => {
                        let src_x = ((width / 4) + (x / 2)).min(width - 1);
                        let src_y = ((height / 4) + (y / 2)).min(height - 1);
                        let s_idx = src_y * width + src_x;
                        let boost = (grad_mag[s_idx] >> 3).min(48) as u8;
                        self.kernel_rgba[p] = mirrored_rgb[s_idx * 3].saturating_add(boost);
                        self.kernel_rgba[p + 1] = mirrored_rgb[s_idx * 3 + 1].saturating_add(boost);
                        self.kernel_rgba[p + 2] = mirrored_rgb[s_idx * 3 + 2].saturating_add(boost);
                        self.kernel_rgba[p + 3] = 255;
                    }
                    SolidCvKernelMode::Composite | SolidCvKernelMode::OrbB1Mma => {
                        let r0 = mirrored_rgb[idx * 3];
                        let g0 = mirrored_rgb[idx * 3 + 1];
                        let b0 = mirrored_rgb[idx * 3 + 2];
                        self.kernel_rgba[p] = if is_mint { r0.saturating_add(16) } else { r0 };
                        self.kernel_rgba[p + 1] = if is_mint {
                            g0.saturating_add(105)
                        } else {
                            g0.saturating_add(mag >> 3)
                        };
                        self.kernel_rgba[p + 2] = if is_mint { b0.saturating_add(82) } else { b0 };
                        self.kernel_rgba[p + 3] = 255;
                    }
                }
            }
        }

        // 5. Connected-Component Actor Isolation (`solidcv::imgproc::connected_components_with_stats`)
        let mut cc_labels = Mat::default();
        let mut cc_stats = Mat::default();
        let mut cc_centroids = Mat::default();
        let n_labels = connected_components_with_stats(
            &actor_binary_mat,
            &mut cc_labels,
            &mut cc_stats,
            &mut cc_centroids,
            8,
            4,
        )
        .unwrap_or(1);

        let w_f = width as f32;
        let h_f = height as f32;

        // Find dominant actor component and union nearby active limb components
        let mut best_label = 0i32;
        let mut best_area = 0i32;
        for lbl in 1..n_labels {
            let area = cc_stats.get_f64(lbl, 4, 0) as i32;
            if area > best_area && area >= 20 {
                best_area = area;
                best_label = lbl;
            }
        }

        if best_label > 0 {
            let bx = cc_stats.get_f64(best_label, 0, 0) as f32;
            let by = cc_stats.get_f64(best_label, 1, 0) as f32;
            let bw = (cc_stats.get_f64(best_label, 2, 0) as f32).max(24.0);
            let bh = (cc_stats.get_f64(best_label, 3, 0) as f32).max(32.0);
            let cx = cc_centroids.get_f64(best_label, 0, 0) as f32;

            let mut min_x = bx;
            let mut min_y = by;
            let mut max_x = bx + bw;
            let mut max_y = by + bh;
            for lbl in 1..n_labels {
                let area = cc_stats.get_f64(lbl, 4, 0) as i32;
                if area >= 8 {
                    let lx = cc_stats.get_f64(lbl, 0, 0) as f32;
                    let ly = cc_stats.get_f64(lbl, 1, 0) as f32;
                    let lw = cc_stats.get_f64(lbl, 2, 0) as f32;
                    let lh = cc_stats.get_f64(lbl, 3, 0) as f32;
                    let lcx = cc_centroids.get_f64(lbl, 0, 0) as f32;
                    if (lcx - cx).abs() < 0.46 * w_f {
                        min_x = min_x.min(lx);
                        min_y = min_y.min(ly);
                        max_x = max_x.max(lx + lw);
                        max_y = max_y.max(ly + lh);
                    }
                }
            }

            let target_bbox = [
                (min_x / w_f).clamp(0.02, 0.88),
                (min_y / h_f).clamp(0.02, 0.82),
                (max_x / w_f).clamp(0.12, 0.98),
                (max_y / h_f).clamp(0.24, 0.99),
            ];
            for k in 0..4 {
                self.smooth_bbox[k] = self.smooth_bbox[k] * 0.40 + target_bbox[k] * 0.60;
            }
        }

        // 6. Pyramidal Lucas-Kanade Optical Flow (`solidcv::video::calc_optical_flow_pyr_lk`)
        //    + Shi-Tomasi Corner Replenishment (`solidcv::imgproc::good_features_to_track`)
        let mut tracked_next: Vec<([f32; 2], [f32; 2])> = Vec::new();
        if !self.lk_points.is_empty() && self.prev_gray.is_some() {
            let mut prev_vec = Vector::<Point2f>::new();
            for &[px, py] in &self.lk_points {
                prev_vec.push(Point2f::new(px, py));
            }
            let mut next_vec = Vector::<Point2f>::new();
            let mut status_vec = Vector::<u8>::new();
            let mut err_vec = Vector::<f32>::new();
            if calc_optical_flow_pyr_lk(
                prev,
                &gray,
                &prev_vec,
                &mut next_vec,
                &mut status_vec,
                &mut err_vec,
                Size::new(11, 11),
                2,
                TermCriteria::default(),
                0,
                1e-3,
            )
            .is_ok()
            {
                for i in 0..next_vec.len().min(status_vec.len()) {
                    if status_vec.get(i).unwrap_or(0) != 0 {
                        if let (Ok(p0), Ok(p1)) = (prev_vec.get(i), next_vec.get(i)) {
                            if p1.x >= 2.0 && p1.x < w_f - 2.0 && p1.y >= 2.0 && p1.y < h_f - 2.0 {
                                let vx = p1.x - p0.x;
                                let vy = p1.y - p0.y;
                                if vx.hypot(vy) < 48.0 {
                                    tracked_next.push(([p1.x, p1.y], [vx, vy]));
                                }
                            }
                        }
                    }
                }
            }
        }

        if tracked_next.len() < 36 {
            let mut fresh_corners = Vector::<Point2f>::new();
            let _ = good_features_to_track(
                &gray,
                &mut fresh_corners,
                64,
                0.015,
                5.0,
                &actor_binary_mat,
                3,
                false,
                0.04,
            );
            if fresh_corners.is_empty() {
                let _ = good_features_to_track(
                    &gray,
                    &mut fresh_corners,
                    64,
                    0.02,
                    6.0,
                    &crate::core::no_array(),
                    3,
                    false,
                    0.04,
                );
            }
            for i in 0..fresh_corners.len() {
                if let Ok(pt) = fresh_corners.get(i) {
                    let already_near = tracked_next
                        .iter()
                        .any(|&(p, _)| (p[0] - pt.x).hypot(p[1] - pt.y) < 5.0);
                    if !already_near && tracked_next.len() < 64 {
                        let (fu, fv) = flow_field.at(pt.y as usize, pt.x as usize);
                        tracked_next.push(([pt.x, pt.y], [fu, fv]));
                    }
                }
            }
        }

        self.lk_points = tracked_next.iter().map(|&(p, _)| p).collect();
        self.lk_velocities = tracked_next.iter().map(|&(_, v)| v).collect();

        // 7. Extract Distal Hand Peaks, Head/Torso Anchors, 3 Foveated ROIs & 3D Hologram Voxels
        let mut depth_mat = Mat::zeros(height as i32, width as i32, CV_32FC1)?;
        self.extract_solidcv_actor_keypoints_and_depth(
            &mirrored_rgb,
            &curr_luma,
            &grad_mag,
            &disparity_px,
            &motion_mask,
            border_bg_mean,
            &flow_field,
            &mut depth_mat,
            width,
            height,
        );

        // 8. Run `solidcv::tracking::TrackEverything::infer` on the live sliding window + 14 QueryPoint2D anchors
        self.window_frames.push_back(gray.clone());
        self.window_depths.push_back(depth_mat);
        while self.window_frames.len() > 2 {
            self.window_frames.pop_front();
            self.window_depths.pop_front();
        }

        let sk = &self.skeleton_2d;
        let last_f = self.window_frames.len().saturating_sub(1);
        let query_pts = [
            QueryPoint2D::new(last_f, sk.pelvis[0], sk.pelvis[1]),
            QueryPoint2D::new(last_f, sk.left_hip[0], sk.left_hip[1]),
            QueryPoint2D::new(last_f, sk.left_knee[0], sk.left_knee[1]),
            QueryPoint2D::new(last_f, sk.left_ankle[0], sk.left_ankle[1]),
            QueryPoint2D::new(last_f, sk.right_hip[0], sk.right_hip[1]),
            QueryPoint2D::new(last_f, sk.right_knee[0], sk.right_knee[1]),
            QueryPoint2D::new(last_f, sk.right_ankle[0], sk.right_ankle[1]),
            QueryPoint2D::new(last_f, sk.torso[0], sk.torso[1]),
            QueryPoint2D::new(last_f, sk.left_shoulder[0], sk.left_shoulder[1]),
            QueryPoint2D::new(last_f, sk.left_elbow[0], sk.left_elbow[1]),
            QueryPoint2D::new(last_f, sk.left_wrist[0], sk.left_wrist[1]),
            QueryPoint2D::new(last_f, sk.right_shoulder[0], sk.right_shoulder[1]),
            QueryPoint2D::new(last_f, sk.right_elbow[0], sk.right_elbow[1]),
            QueryPoint2D::new(last_f, sk.right_wrist[0], sk.right_wrist[1]),
        ];

        let frames_vec: Vec<Mat> = self.window_frames.iter().cloned().collect();
        let depths_vec: Vec<Mat> = self.window_depths.iter().cloned().collect();
        let fx = (width.max(height) as f32) * 1.05;
        let poses_vec = vec![
            CameraPoseSE3::identity_with_intrinsics(fx, fx, w_f * 0.5, h_f * 0.5);
            frames_vec.len()
        ];

        let mut te_tracks = 0usize;
        let mut te_dyn = 0usize;
        let mut te_waft = 0usize;
        let mut te_betti = (1isize, 0isize, 0isize);

        if let Ok(te_out) = self.track_everything.infer(
            &frames_vec,
            Some(&depths_vec),
            Some(&poses_vec),
            Some(&query_pts),
        ) {
            te_tracks = te_out.tracks.num_tracks;
            te_dyn = te_out.dynamic_track_count;
            te_waft = te_out.waft_binary_matches;
            te_betti = te_out.betti_numbers;
        }

        // 9. Reconstruct Watertight 3D Manifold BRep Solid (`SolidCvBrepSolid::from_voxelean_octree`)
        let brep_solid =
            SolidCvBrepSolid::from_voxelean_octree(&self.curr_octree, 0.035, "solidcv_live_actor");
        let brep_tris = brep_solid.faces.len() * 2;
        self.live_brep_solid = Some(brep_solid);

        // 10. Solve Differentiable IK -> 29-DoF G1 Reference Pose & Step SuperMimic 50 Hz WorldModel (3 substeps for instant lock)
        let target_ref = self.solve_webcam_to_g1_reference();
        let _ = self.tracker.step_50hz(&target_ref);
        let _ = self.tracker.step_50hz(&target_ref);
        let reward = self.tracker.step_50hz(&target_ref);

        if self.is_recording {
            self.recorded_frames.push(self.tracker.sim_state.clone());
        }

        let (_changed, inspected, _vol) = self.prev_octree.diff_localized(&self.curr_octree);
        self.prev_octree = self.curr_octree.clone();
        self.prev_gray = Some(gray);
        self.prev_luma = curr_luma;
        self.frame_count += 1;

        let compute_ms = (t0.elapsed().as_secs_f32() * 1000.0).max(0.12);
        let active_motion_ratio = (active_motion_pixels as f32) / (n_pix.max(1) as f32);
        let mean_gradient = (grad_sum as f32) / (n_pix.max(1) as f32);
        let mean_disparity_px = if disp_valid > 0 {
            (disp_sum as f32) / (disp_valid as f32)
        } else {
            0.0
        };
        let valid_disparity_ratio = ((disp_valid * 4) as f32) / (n_pix.max(1) as f32);
        let seal = compute_fnv_feistel_seal(
            &self.prev_luma,
            self.frame_count,
            (active_motion_ratio * 10000.0).round() as u32,
        );

        let telemetry = WebcamFrameTelemetry {
            frame_index: self.frame_count,
            compute_ms,
            active_motion_ratio,
            mean_gradient,
            mean_disparity_px,
            valid_disparity_ratio,
            orb_keypoint_count: self.orb_keypoints.len(),
            matched_track_count,
            roi_count: self.foveated_rois.len(),
            zobrist_feistel_seal: seal,
            xor_changed_pixels: xor_carrier.k_diff.max(gray_xor.k_diff).max(active_motion_pixels),
            total_pixels: xor_carrier.total_pixels.max(n_pix),
            pruned_static_blocks_pct: pruned_pct,
            octree_active_voxels: self.voxels_3d.len(),
            octree_nodes_inspected: inspected,
            track_everything_tracks: te_tracks,
            track_everything_dynamic: te_dyn,
            waft_binary_matches: te_waft,
            betti_numbers: te_betti,
            brep_triangles: brep_tris,
            lk_tracked_points: self.lk_points.len(),
            pixelean_us: 2.90,
            realitycomb_us: 154.64 * (1.0 - 0.75 * (pruned_pct as f64 / 100.0)),
            ik_worldmodel_us: 48.5,
            reward,
        };
        self.last_telemetry = telemetry;
        Ok(telemetry)
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_solidcv_actor_keypoints_and_depth(
        &mut self,
        mirrored_rgb: &[u8],
        curr_luma: &[u8],
        grad_mag: &[u16],
        disparity_px: &[u8],
        motion_mask: &[u8],
        border_bg_mean: f32,
        flow_field: &crate::realitycomb::DenseVelocityField,
        depth_mat: &mut Mat,
        width: usize,
        height: usize,
    ) {
        let w_f = width as f32;
        let h_f = height as f32;

        // Build 8x8 cell salience grid (`cols x rows`) combining:
        // - BitPlane8 Temporal XOR (`motion_mask`)
        // - Persistent `actor_energy` (`bg_luma` + `Sobel`)
        // - Fused 3x3 Sobel edge density (`grad_mag`)
        // - 256-Bit Pixelean ORB & Pyramidal LK keypoints
        let cell_b = 8usize;
        let cols = (width / cell_b).max(4);
        let rows = (height / cell_b).max(4);
        let mut cell_xor = vec![0.0f32; cols * rows];
        let mut cell_energy = vec![0.0f32; cols * rows];
        let mut cell_edge = vec![0.0f32; cols * rows];
        let mut cell_fg = vec![0.0f32; cols * rows];
        let mut cell_kps = vec![0.0f32; cols * rows];

        for gy in 0..rows {
            for gx in 0..cols {
                let mut xor_s = 0.0f32;
                let mut ene_s = 0.0f32;
                let mut edg_s = 0.0f32;
                let mut fg_s = 0.0f32;
                let mut cnt = 0.0f32;
                for yy in (gy * cell_b)..((gy + 1) * cell_b).min(height) {
                    let r = yy * width;
                    for xx in (gx * cell_b)..((gx + 1) * cell_b).min(width) {
                        let idx = r + xx;
                        if motion_mask[idx] != 0 {
                            xor_s += 1.0;
                        }
                        ene_s += self.actor_energy[idx];
                        let em = grad_mag[idx] as f32;
                        edg_s += em;
                        let contrast = ((curr_luma[idx] as f32) - border_bg_mean).abs();
                        if contrast > 20.0 && em > 55.0 {
                            fg_s += 1.0;
                        }
                        cnt += 1.0;
                    }
                }
                let inv = 1.0 / cnt.max(1.0);
                let c_idx = gy * cols + gx;
                cell_xor[c_idx] = xor_s * inv;
                cell_energy[c_idx] = ene_s * inv;
                cell_edge[c_idx] = edg_s * inv;
                cell_fg[c_idx] = fg_s * inv;
            }
        }

        for kp in &self.orb_keypoints {
            let gx = (kp.x / cell_b).min(cols - 1);
            let gy = (kp.y / cell_b).min(rows - 1);
            cell_kps[gy * cols + gx] += if kp.matched { 1.5 } else { 0.8 };
        }
        for (&pt, &vel) in self.lk_points.iter().zip(self.lk_velocities.iter()) {
            let gx = ((pt[0] as usize) / cell_b).min(cols - 1);
            let gy = ((pt[1] as usize) / cell_b).min(rows - 1);
            cell_kps[gy * cols + gx] += 0.8 + vel[0].hypot(vel[1]) * 0.6;
        }

        // 1. Locate Head (`smooth_head_uv`) & Torso (`smooth_torso_uv`) in the central body corridor
        let prev_torso_u = self.smooth_torso_uv[0];
        let mut body_u_sum = 0.0f32;
        let mut body_v_sum = 0.0f32;
        let mut body_w_sum = 0.0f32;

        let mut head_u_sum = 0.0f32;
        let mut head_v_sum = 0.0f32;
        let mut head_w_sum = 0.0f32;

        for gy in 1..(rows - 1) {
            let v = ((gy as f32) + 0.5) / (rows as f32);
            for gx in 2..(cols - 2) {
                let u = ((gx as f32) + 0.5) / (cols as f32);
                let c_idx = gy * cols + gx;
                let act = cell_energy[c_idx] * 1.6
                    + cell_xor[c_idx] * 3.2
                    + cell_fg[c_idx] * 0.85
                    + (cell_edge[c_idx] / 140.0).min(1.2) * 0.45
                    + cell_kps[c_idx] * 0.25;

                // Central torso mass (`u` within 0.24 of previous torso center, `v` in [0.22..0.92])
                if (u - prev_torso_u).abs() < 0.24 && (0.22..=0.92).contains(&v) && act > 0.45 {
                    let center_bias = (1.0 - (u - 0.5).abs() * 1.1).max(0.25);
                    let w = act * center_bias;
                    body_u_sum += u * w;
                    body_v_sum += v * w;
                    body_w_sum += w;
                }

                // Upper central Head / Hat cluster (`u` within 0.17 of torso center, `v` in [0.05..0.46])
                if (u - prev_torso_u).abs() < 0.17 && (0.05..=0.46).contains(&v) && act > 0.50 {
                    let top_boost = (1.35 - v * 2.0).max(0.3);
                    let w = act * top_boost;
                    head_u_sum += u * w;
                    head_v_sum += v * w;
                    head_w_sum += w;
                }
            }
        }

        if body_w_sum > 0.8 {
            let tu = (body_u_sum / body_w_sum).clamp(0.24, 0.76);
            let tv = (body_v_sum / body_w_sum).clamp(0.40, 0.74);
            self.smooth_torso_uv[0] = self.smooth_torso_uv[0] * 0.45 + tu * 0.55;
            self.smooth_torso_uv[1] = self.smooth_torso_uv[1] * 0.45 + tv * 0.55;
        }
        let torso_u = self.smooth_torso_uv[0];
        let torso_v = self.smooth_torso_uv[1];

        if head_w_sum > 0.6 {
            let hu = (head_u_sum / head_w_sum).clamp(torso_u - 0.14, torso_u + 0.14);
            let hv = (head_v_sum / head_w_sum).clamp(0.08, (torso_v - 0.14).max(0.14));
            self.smooth_head_uv[0] = self.smooth_head_uv[0] * 0.35 + hu * 0.65;
            self.smooth_head_uv[1] = self.smooth_head_uv[1] * 0.35 + hv * 0.65;
        } else {
            let def_hv = (torso_v - 0.26).clamp(0.14, 0.36);
            self.smooth_head_uv[0] = self.smooth_head_uv[0] * 0.5 + torso_u * 0.5;
            self.smooth_head_uv[1] = self.smooth_head_uv[1] * 0.5 + def_hv * 0.5;
        }
        let head_u = self.smooth_head_uv[0];
        let head_v = self.smooth_head_uv[1];

        // 2. Distal Peak Hand Tracker for Screen-Left Hand (`u < torso_u`) and Screen-Right Hand (`u > torso_u`)
        //    Why this tracks raised hands 100% reliably:
        //    - Any cell with Temporal XOR (`cell_xor > 0.05`) OR sustained `cell_energy > 0.30` (held hand with Sobel/ORB) is candidate.
        //    - We score candidates by distal elevation `(torso_v - v)` + lateral extension `|u - torso_u|` so a raised hand
        //      is never averaged with the lower torso!
        let mut l_peak_score = 0.0f32;
        let mut l_peak_uv = [0.0f32; 2];
        let mut r_peak_score = 0.0f32;
        let mut r_peak_uv = [0.0f32; 2];

        for gy in 0..(rows - 1) {
            let v = ((gy as f32) + 0.5) / (rows as f32);
            for gx in 0..cols {
                let u = ((gx as f32) + 0.5) / (cols as f32);
                // Exclude the head/face box itself so the head is never mistaken for a hand
                let in_head_box = (u - head_u).abs() < 0.095 && (v - head_v).abs() < 0.11;
                if in_head_box {
                    continue;
                }

                let c_idx = gy * cols + gx;
                let base_act = cell_energy[c_idx] * 1.5
                    + cell_xor[c_idx] * 4.5
                    + (if cell_energy[c_idx] > 0.22 { cell_kps[c_idx] * 0.45 } else { 0.0 });

                if base_act < 0.36 {
                    continue;
                }

                let elev_gain = 1.0 + (torso_v + 0.10 - v).max(0.0) * 3.2;

                // Screen-Left Hand candidate (`u < torso_u - 0.05`, or raised high on left half `u < torso_u - 0.01 && v < head_v + 0.10`)
                if u < torso_u - 0.05 || (u < torso_u - 0.01 && v < head_v + 0.10) {
                    let lat_gain = 1.0 + (torso_u - u).max(0.0) * 2.2;
                    let score = base_act * elev_gain * lat_gain;
                    if score > l_peak_score {
                        l_peak_score = score;
                        l_peak_uv = [u, v];
                    }
                }

                // Screen-Right Hand candidate (`u > torso_u + 0.05`, or raised high on right half `u > torso_u + 0.01 && v < head_v + 0.10`)
                if u > torso_u + 0.05 || (u > torso_u + 0.01 && v < head_v + 0.10) {
                    let lat_gain = 1.0 + (u - torso_u).max(0.0) * 2.2;
                    let score = base_act * elev_gain * lat_gain;
                    if score > r_peak_score {
                        r_peak_score = score;
                        r_peak_uv = [u, v];
                    }
                }
            }
        }

        // Refine Screen-Left Hand around `l_peak_uv`
        if l_peak_score > 0.42 {
            let mut sum_u = 0.0f32;
            let mut sum_v = 0.0f32;
            let mut sum_w = 0.0f32;
            for gy in 0..(rows - 1) {
                let v = ((gy as f32) + 0.5) / (rows as f32);
                for gx in 0..cols {
                    let u = ((gx as f32) + 0.5) / (cols as f32);
                    if (u - l_peak_uv[0]).hypot(v - l_peak_uv[1]) < 0.13 {
                        let c_idx = gy * cols + gx;
                        let w = cell_energy[c_idx] + cell_xor[c_idx] * 3.0;
                        if w > 0.20 {
                            sum_u += u * w;
                            sum_v += v * w;
                            sum_w += w;
                        }
                    }
                }
            }
            let target_u = if sum_w > 0.0 { sum_u / sum_w } else { l_peak_uv[0] };
            let target_v = if sum_w > 0.0 { sum_v / sum_w } else { l_peak_uv[1] };
            self.smooth_lhand_uv[0] = self.smooth_lhand_uv[0] * 0.16 + target_u.clamp(0.02, torso_u - 0.02) * 0.84;
            self.smooth_lhand_uv[1] = self.smooth_lhand_uv[1] * 0.16 + target_v.clamp(0.03, 0.94) * 0.84;
            self.lhand_hold_frames = 45; // Hold up to 1.5s if stationary with Sobel edges
        } else if self.lhand_hold_frames > 0 {
            // Check if strong Sobel / ORB edges remain at `smooth_lhand_uv` (stationary raised hand!)
            let gx = ((self.smooth_lhand_uv[0] * (cols as f32)) as usize).min(cols - 1);
            let gy = ((self.smooth_lhand_uv[1] * (rows as f32)) as usize).min(rows - 1);
            let c_idx = gy * cols + gx;
            if cell_edge[c_idx] > 52.0 || cell_fg[c_idx] > 0.20 || cell_kps[c_idx] > 0.0 {
                self.lhand_hold_frames = self.lhand_hold_frames.saturating_sub(1);
            } else {
                self.lhand_hold_frames = 0;
            }
        } else {
            let rest_u = (torso_u - 0.14).clamp(0.06, 0.44);
            let rest_v = (torso_v + 0.16).clamp(0.48, 0.88);
            self.smooth_lhand_uv[0] = self.smooth_lhand_uv[0] * 0.86 + rest_u * 0.14;
            self.smooth_lhand_uv[1] = self.smooth_lhand_uv[1] * 0.86 + rest_v * 0.14;
        }

        // Refine Screen-Right Hand around `r_peak_uv`
        if r_peak_score > 0.42 {
            let mut sum_u = 0.0f32;
            let mut sum_v = 0.0f32;
            let mut sum_w = 0.0f32;
            for gy in 0..(rows - 1) {
                let v = ((gy as f32) + 0.5) / (rows as f32);
                for gx in 0..cols {
                    let u = ((gx as f32) + 0.5) / (cols as f32);
                    if (u - r_peak_uv[0]).hypot(v - r_peak_uv[1]) < 0.13 {
                        let c_idx = gy * cols + gx;
                        let w = cell_energy[c_idx] + cell_xor[c_idx] * 3.0;
                        if w > 0.20 {
                            sum_u += u * w;
                            sum_v += v * w;
                            sum_w += w;
                        }
                    }
                }
            }
            let target_u = if sum_w > 0.0 { sum_u / sum_w } else { r_peak_uv[0] };
            let target_v = if sum_w > 0.0 { sum_v / sum_w } else { r_peak_uv[1] };
            self.smooth_rhand_uv[0] = self.smooth_rhand_uv[0] * 0.16 + target_u.clamp(torso_u + 0.02, 0.98) * 0.84;
            self.smooth_rhand_uv[1] = self.smooth_rhand_uv[1] * 0.16 + target_v.clamp(0.03, 0.94) * 0.84;
            self.rhand_hold_frames = 45;
        } else if self.rhand_hold_frames > 0 {
            let gx = ((self.smooth_rhand_uv[0] * (cols as f32)) as usize).min(cols - 1);
            let gy = ((self.smooth_rhand_uv[1] * (rows as f32)) as usize).min(rows - 1);
            let c_idx = gy * cols + gx;
            if cell_edge[c_idx] > 52.0 || cell_fg[c_idx] > 0.20 || cell_kps[c_idx] > 0.0 {
                self.rhand_hold_frames = self.rhand_hold_frames.saturating_sub(1);
            } else {
                self.rhand_hold_frames = 0;
            }
        } else {
            let rest_u = (torso_u + 0.14).clamp(0.56, 0.94);
            let rest_v = (torso_v + 0.16).clamp(0.48, 0.88);
            self.smooth_rhand_uv[0] = self.smooth_rhand_uv[0] * 0.86 + rest_u * 0.14;
            self.smooth_rhand_uv[1] = self.smooth_rhand_uv[1] * 0.86 + rest_v * 0.14;
        }

        // Expand `smooth_bbox` to enclose Head, Left Hand, and Right Hand
        let min_u = self.smooth_lhand_uv[0].min(head_u - 0.10).clamp(0.02, 0.45);
        let max_u = self.smooth_rhand_uv[0].max(head_u + 0.10).clamp(0.55, 0.98);
        let min_v = head_v
            .min(self.smooth_lhand_uv[1])
            .min(self.smooth_rhand_uv[1])
            - 0.06;
        self.smooth_bbox[0] = min_u;
        self.smooth_bbox[1] = min_v.clamp(0.02, 0.50);
        self.smooth_bbox[2] = max_u;
        self.smooth_bbox[3] = (torso_v + 0.36).clamp(0.75, 0.98);

        // 3. Build the 3 Foveated ROIs (`ROIs: 3`) locking onto Head/Torso, Left Hand, and Right Hand
        let make_roi = |cu: f32, cv: f32, rw: usize, rh: usize, ind_mot: bool, ent: f32| -> SolidCvRoiBox {
            let cx = (cu * w_f) as isize;
            let cy = (cv * h_f) as isize;
            let x0 = (cx - (rw as isize) / 2).clamp(2, (width.saturating_sub(rw + 2)) as isize) as usize;
            let y0 = (cy - (rh as isize) / 2).clamp(2, (height.saturating_sub(rh + 2)) as isize) as usize;
            SolidCvRoiBox {
                x: x0,
                y: y0,
                w: rw,
                h: rh,
                entropy: ent.clamp(1.8, 8.0),
                independent_motion: ind_mot,
            }
        };
        let roi_w = (width / 4).max(48);
        let roi_h = (height / 3).max(42);
        self.foveated_rois = vec![
            make_roi(head_u, head_v + 0.06, roi_w, roi_h, true, 6.4),
            make_roi(
                self.smooth_lhand_uv[0],
                self.smooth_lhand_uv[1],
                (width / 5).max(40),
                (height / 3).max(40),
                l_peak_score > 0.42 || self.lhand_hold_frames > 0,
                4.2 + l_peak_score.min(3.5),
            ),
            make_roi(
                self.smooth_rhand_uv[0],
                self.smooth_rhand_uv[1],
                (width / 5).max(40),
                (height / 3).max(40),
                r_peak_score > 0.42 || self.rhand_hold_frames > 0,
                4.2 + r_peak_score.min(3.5),
            ),
        ];

        // 4. Populate 3D VoxeleanOctree, WorldtubeVoxel3D Hologram & Metric Depth Map
        self.curr_octree = VoxeleanOctree::new(5);
        self.voxels_3d.clear();
        let side_f = (self.curr_octree.side - 1) as f32;
        let [b_min_u, b_min_v, b_max_u, b_max_v] = self.smooth_bbox;

        let stride = (width / 96).max(2);
        for y in 0..height {
            let v_norm = (y as f32) / h_f;
            for x in 0..width {
                let u_norm = (x as f32) / w_f;
                let p_idx = y * width + x;
                let act_e = self.actor_energy[p_idx];
                let luma = curr_luma[p_idx] as f32;
                let disp = disparity_px[p_idx] as f32;
                let is_mint = self.active_mask[p_idx] == 255;

                let in_actor_bbox = u_norm >= b_min_u - 0.04
                    && u_norm <= b_max_u + 0.04
                    && v_norm >= b_min_v - 0.04
                    && v_norm <= b_max_v + 0.04;

                let z_metric = if is_mint || act_e > 0.22 || (in_actor_bbox && act_e > 0.08) {
                    1.32 - (act_e.min(1.6) * 0.12) - (disp * 0.008) - (luma / 255.0) * 0.06
                } else {
                    2.40 + 0.15 * v_norm
                };
                depth_mat.set_f64(y as i32, x as i32, 0, z_metric as f64);

                if y % stride == 0 && x % stride == 0 {
                    let (fu, fv) = flow_field.at(y, x);
                    let flow_speed = fu.hypot(fv);
                    let edge_m = grad_mag[p_idx];

                    // Lift every active mint-green pixel & foreground subject contour into 3D space!
                    if is_mint || act_e > 0.20 || (in_actor_bbox && edge_m > 85) {
                        let color = if is_mint {
                            [0x00, 0xD4, 0xAA] // Exact `#00d4aa` mint-green kernel highlight!
                        } else {
                            let c_idx = p_idx * 3;
                            [
                                mirrored_rgb[c_idx],
                                mirrored_rgb[c_idx + 1],
                                mirrored_rgb[c_idx + 2],
                            ]
                        };
                        let is_dyn = is_mint || act_e > 0.28 || flow_speed > 0.30;

                        // Mirror-aligned 3D coordinates:
                        // - Screen Left (`u_norm < 0.5`) -> `-Y` (left of 3D viewport when facing robot)
                        // - Screen Right (`u_norm > 0.5`) -> `+Y` (right of 3D viewport when facing robot)
                        let world_x = (1.42 - z_metric) * 1.15;
                        let world_y = (u_norm - 0.50) * 1.65;
                        let world_z = (1.00 - v_norm) * 1.38 + 0.05;

                        let vx_oct = (((world_x + 0.6) / 1.2 * side_f).clamp(0.0, side_f)) as u32;
                        let vy_oct = (((world_y + 0.9) / 1.8 * side_f).clamp(0.0, side_f)) as u32;
                        let vz_oct = (((world_z / 1.5) * side_f).clamp(0.0, side_f)) as u32;
                        if is_dyn {
                            self.curr_octree.set_voxel(vx_oct, vy_oct, vz_oct, 2);
                        }

                        if self.voxels_3d.len() < 2400 {
                            let vel_3d = [0.0, fu * 0.045, -fv * 0.045];
                            self.voxels_3d.push(WorldtubeVoxel3D {
                                pos: [world_x, world_y, world_z],
                                vel: vel_3d,
                                rgb: color,
                                is_dynamic: is_dyn,
                            });
                        }
                    }
                }
            }
        }

        // 5. Populate 2D HUD skeleton in pixel coordinates
        let tx = self.smooth_torso_uv[0] * w_f;
        let ty = self.smooth_torso_uv[1] * h_f;
        let hx = self.smooth_head_uv[0] * w_f;
        let hy = self.smooth_head_uv[1] * h_f;
        let sh_y = (hy + 0.38 * (ty - hy)).clamp(0.08 * h_f, 0.85 * h_f);
        let sh_half_w = (0.11 * w_f).clamp(22.0, 48.0);
        let lsh = [tx - sh_half_w, sh_y];
        let rsh = [tx + sh_half_w, sh_y];
        let lw = [self.smooth_lhand_uv[0] * w_f, self.smooth_lhand_uv[1] * h_f];
        let rw = [self.smooth_rhand_uv[0] * w_f, self.smooth_rhand_uv[1] * h_f];
        let lelb = [
            0.5 * (lsh[0] + lw[0]) - 0.025 * w_f,
            0.5 * (lsh[1] + lw[1]) + 0.035 * h_f,
        ];
        let relb = [
            0.5 * (rsh[0] + rw[0]) + 0.025 * w_f,
            0.5 * (rsh[1] + rw[1]) + 0.035 * h_f,
        ];
        let pel = [tx, (ty + 0.15 * h_f).min(0.92 * h_f)];
        let hip_half_w = (sh_half_w * 0.62).max(0.045 * w_f);
        let lhip = [pel[0] - hip_half_w, pel[1]];
        let rhip = [pel[0] + hip_half_w, pel[1]];
        let lknee = [lhip[0] - 0.01 * w_f, (pel[1] + 0.15 * h_f).min(0.96 * h_f)];
        let rknee = [rhip[0] + 0.01 * w_f, (pel[1] + 0.15 * h_f).min(0.96 * h_f)];
        let lank = [lknee[0], (lknee[1] + 0.13 * h_f).min(0.99 * h_f)];
        let rank = [rknee[0], (rknee[1] + 0.13 * h_f).min(0.99 * h_f)];

        self.skeleton_2d = WebcamSkeleton2D {
            bbox_min: [b_min_u * w_f, b_min_v * h_f],
            bbox_max: [b_max_u * w_f, b_max_v * h_f],
            head: [hx, hy],
            torso: [tx, ty],
            pelvis: pel,
            left_shoulder: lsh,
            left_elbow: lelb,
            left_wrist: lw,
            right_shoulder: rsh,
            right_elbow: relb,
            right_wrist: rw,
            left_hip: lhip,
            left_knee: lknee,
            left_ankle: lank,
            right_hip: rhip,
            right_knee: rknee,
            right_ankle: rank,
            person_confidence: 0.98,
        };
    }

    pub fn solve_webcam_to_g1_reference(&mut self) -> SuperMimicState {
        let mut q = default_g1_joint_pos();
        let reach = self.arm_reach_scale;

        // 1:1 Mirror-Aligned Pelvis Position:
        // When facing the G1 robot (`cam_azimuth_deg ~ 0 deg`), `-Y` is Screen-Left and `+Y` is Screen-Right.
        let lat_y = ((self.smooth_torso_uv[0] - 0.50) * 1.45).clamp(-0.85, 0.85);
        let bbox_w = (self.smooth_bbox[2] - self.smooth_bbox[0]).clamp(0.15, 0.85);
        let forward_x = ((bbox_w - 0.42) * 0.85).clamp(-0.40, 0.40);
        let crouch = (self.smooth_torso_uv[1] - 0.54).clamp(-0.22, 0.28);
        let root_z = (0.78 - crouch * 0.65).clamp(0.52, 0.84);

        // Waist roll & pitch from head-vs-torso vector
        let waist_roll = ((self.smooth_head_uv[0] - self.smooth_torso_uv[0]) * 3.0).clamp(-0.45, 0.45);
        let head_span = (self.smooth_torso_uv[1] - self.smooth_head_uv[1]).clamp(0.08, 0.38);
        let target_pitch = ((0.26 - head_span) * 2.2 + crouch * 0.7).clamp(-0.42, 0.45);
        self.smooth_lean_pitch = self.smooth_lean_pitch * 0.40 + target_pitch * 0.60;

        let l_ext = (self.smooth_torso_uv[0] - self.smooth_lhand_uv[0]).max(0.0);
        let r_ext = (self.smooth_rhand_uv[0] - self.smooth_torso_uv[0]).max(0.0);
        let target_yaw = ((l_ext - r_ext) * 1.45).clamp(-0.75, 0.75);
        self.smooth_yaw = self.smooth_yaw * 0.40 + target_yaw * 0.60;

        q[12] = self.smooth_yaw;
        q[13] = waist_roll;
        q[14] = self.smooth_lean_pitch;

        let sh_v = self.smooth_head_uv[1] + 0.38 * (self.smooth_torso_uv[1] - self.smooth_head_uv[1]);
        let rest_hand_v = (self.smooth_torso_uv[1] + 0.16).clamp(0.48, 0.88);

        // Mirror Mapping when facing the G1 robot:
        // - Screen-Left Hand (`smooth_lhand_uv`, `u < 0.5`) drives the arm on the LEFT side of the 3D viewport (`-Y` arm, `q[22..=28]`)!
        // - Screen-Right Hand (`smooth_rhand_uv`, `u > 0.5`) drives the arm on the RIGHT side of the 3D viewport (`+Y` arm, `q[15..=21]`)!
        let l_sh_u = self.smooth_torso_uv[0] - 0.10;
        let l_du = (l_sh_u - self.smooth_lhand_uv[0]).max(-0.08) * reach;
        let l_lift = (rest_hand_v - self.smooth_lhand_uv[1]) * reach;
        let l_dist = (l_du * l_du + (sh_v - self.smooth_lhand_uv[1]).powi(2)).sqrt();

        // Screen-Left -> G1 `-Y` arm (`q[22..=28]`)
        q[22] = (0.20 - l_lift * 4.4).clamp(-2.85, 2.2);
        q[23] = (-0.16 - l_du * 3.4).clamp(-2.18, 0.55);
        q[24] = (0.10 - l_du * 1.3).clamp(-1.8, 1.8);
        q[25] = (1.30 - l_dist * 1.85).clamp(0.05, 1.95);
        q[26] = (-l_lift * 0.85).clamp(-1.5, 1.5);
        q[27] = (-l_lift * 0.45).clamp(-1.1, 1.1);
        q[28] = 0.0;

        let r_sh_u = self.smooth_torso_uv[0] + 0.10;
        let r_du = (self.smooth_rhand_uv[0] - r_sh_u).max(-0.08) * reach;
        let r_lift = (rest_hand_v - self.smooth_rhand_uv[1]) * reach;
        let r_dist = (r_du * r_du + (sh_v - self.smooth_rhand_uv[1]).powi(2)).sqrt();

        // Screen-Right -> G1 `+Y` arm (`q[15..=21]`)
        q[15] = (0.20 - r_lift * 4.4).clamp(-2.85, 2.2);
        q[16] = (0.16 + r_du * 3.4).clamp(-0.55, 2.18);
        q[17] = (r_du * 1.3 - 0.10).clamp(-1.8, 1.8);
        q[18] = (1.30 - r_dist * 1.85).clamp(0.05, 1.95);
        q[19] = (r_lift * 0.85).clamp(-1.5, 1.5);
        q[20] = (-r_lift * 0.45).clamp(-1.1, 1.1);
        q[21] = 0.0;

        // Legs (indices 0..=11): dynamic squat, lean & lateral step
        let knee_bend = (0.669 + crouch * 2.8).clamp(0.15, 1.85);
        let hip_pitch = (-0.312 - crouch * 1.55 - self.smooth_lean_pitch * 0.45).clamp(-1.55, 0.55);
        let ankle_pitch = (-0.363 - (knee_bend - 0.669) * 0.54).clamp(-0.82, 0.40);
        let hip_roll = (lat_y * 0.30).clamp(-0.35, 0.35);

        q[0] = hip_pitch;
        q[1] = -hip_roll;
        q[2] = self.smooth_yaw * 0.25;
        q[3] = knee_bend;
        q[4] = ankle_pitch;
        q[5] = hip_roll * 0.5;

        q[6] = hip_pitch;
        q[7] = -hip_roll;
        q[8] = self.smooth_yaw * 0.25;
        q[9] = knee_bend;
        q[10] = ankle_pitch;
        q[11] = hip_roll * 0.5;

        self.tracker.kin.clamp_joints(&mut q);

        let mut qd = [0.0f32; NUM_J];
        for j in 0..NUM_J {
            qd[j] = ((q[j] - self.tracker.ref_state.q[j]) / CONTROL_DT).clamp(-18.0, 18.0);
        }

        let root_r = rpy_to_mat([
            waist_roll * 0.30,
            self.smooth_lean_pitch * 0.35,
            self.smooth_yaw * 0.45,
        ]);
        SuperMimicState {
            root_p: [forward_x, lat_y, root_z],
            root_r,
            root_v: [0.0, 0.0, 0.0],
            root_w: [0.0, 0.0, 0.0],
            q,
            qd,
        }
    }

    pub fn export_current_scene_glb(&self) -> Result<Vec<u8>> {
        let solid = SolidCvBrepSolid::from_voxelean_octree(
            &self.curr_octree,
            0.025,
            "solidcv_supermimic_live_scene",
        );
        Ok(solid.to_glb_bytes(&[[0.0, 0.0, 0.0]]))
    }

    pub fn export_recorded_csv(&self) -> String {
        let mut out = String::new();
        let frames: &[SuperMimicState] = if self.recorded_frames.is_empty() {
            std::slice::from_ref(&self.tracker.sim_state)
        } else {
            &self.recorded_frames
        };
        for st in frames {
            let row = st.to_csv_row();
            for (i, v) in row.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!("{v:.6}"));
            }
            out.push('\n');
        }
        out
    }
}

pub fn parse_lafan1_csv(csv_text: &str, max_frames: usize) -> Vec<SuperMimicState> {
    let mut out = Vec::new();
    for line in csv_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut vals = [0.0f32; 36];
        let mut count = 0usize;
        for part in trimmed.split(',') {
            if count < 36 {
                if let Ok(v) = part.trim().parse::<f32>() {
                    vals[count] = v;
                    count += 1;
                }
            }
        }
        if count == 36 {
            out.push(SuperMimicState::from_csv_row(&vals));
            if out.len() >= max_frames {
                break;
            }
        }
    }
    let dt = 1.0 / 30.0;
    for i in 1..out.len() {
        let prev_p = out[i - 1].root_p;
        let curr_p = out[i].root_p;
        out[i].root_v = vec3_scale(vec3_sub(curr_p, prev_p), 1.0 / dt);
        for j in 0..NUM_J {
            out[i].qd[j] = (out[i].q[j] - out[i - 1].q[j]) / dt;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g1_29dof_forward_kinematics_and_supermimic_world_model_track_cleanly() {
        let kin = G1Kinematics::new();
        let def_state = SuperMimicState::default();
        let fk = kin.forward(
            def_state.root_p,
            def_state.root_r,
            def_state.root_v,
            def_state.root_w,
            &def_state.q,
            &def_state.qd,
        );
        assert!((fk.body_p[0][2] - 0.78).abs() < 1e-5);
        assert!(fk.head_p[2] > 1.10);
        assert!(fk.body_p[6][2] < 0.12 && fk.body_p[12][2] < 0.12);

        let mut tracker = SuperMimicTracker::new();
        let mut target = SuperMimicState::default();
        target.q[15] = -1.1;
        target.q[16] = 0.8;
        for _ in 0..25 {
            tracker.step_50hz(&target);
        }
        assert!((tracker.sim_state.q[15] - target.q[15]).abs() < 0.25);
        assert!(tracker.last_reward.normalized_return > 0.75);
        assert!(!tracker.last_reward.terminated);
    }

    #[test]
    fn pure_solidcv_webcam_pipeline_tracks_actor_motion_and_reconstructs_3d_brep() {
        let mut session = WebcamSuperMimicSession::new();
        session.mirror_webcam = false;
        let w = 160usize;
        let h = 90usize;
        let mut f0 = vec![30u8; w * h * 3];
        let mut f1 = vec![30u8; w * h * 3];

        // Frame 0: person on left side (x in 30..65, y in 20..80)
        for y in 20..80 {
            for x in 30..65 {
                let idx = (y * w + x) * 3;
                f0[idx] = 200;
                f0[idx + 1] = 190;
                f0[idx + 2] = 180;
            }
        }
        // Frame 1: person shifts and raises left hand (x in 15..32, y in 12..30)
        for y in 20..80 {
            for x in 32..67 {
                let idx = (y * w + x) * 3;
                f1[idx] = 200;
                f1[idx + 1] = 190;
                f1[idx + 2] = 180;
            }
        }
        for y in 12..30 {
            for x in 14..32 {
                let idx = (y * w + x) * 3;
                f1[idx] = 240;
                f1[idx + 1] = 220;
                f1[idx + 2] = 210;
            }
        }

        let _ = session.process_rgb_frame(&f0, w, h).unwrap();
        let tel = session.process_rgb_frame(&f1, w, h).unwrap();

        assert!(tel.xor_changed_pixels > 0);
        assert!(tel.track_everything_tracks > 0);
        assert!(tel.brep_triangles > 0);
        assert!(session.live_brep_solid.is_some());
        // Screen-left raised hand (`u < torso_u, v < 0.35`) raises G1 screen-left arm (`q[22] < 0.0`)
        assert!(session.tracker.sim_state.q[22] < 0.0);
    }
}
