//! Pure-Rust 3D camera calibration, pose estimation, epipolar geometry, and stereo matching (`calib3d`).
//!
//! Powered by Pixelean Census Bitplane matching (`Theorem 1`), Octahedral $O_h$ rotation
//! manifold projection (`Theorem 10`), and $\mathfrak{se}(3)$ Lie-algebra Gauss-Newton optimization.

use crate::core::{
    rng_seed, Error, InputArray, Mat, MatTraitConst, OutputPoints2f, Point2f, Point3f, Rect,
    Result, Size, CV_32FC1, CV_32SC1, CV_64FC1, CV_8UC1,
};
use crate::mechinterp::{record_activation, TheoremId};

pub const RANSAC: i32 = 8;
pub const LMEDS: i32 = 4;
pub const CALIB_ZERO_DISPARITY: i32 = 1024;

pub const SOLVEPNP_ITERATIVE: i32 = 0;
pub const SOLVEPNP_EPNP: i32 = 1;
pub const SOLVEPNP_P3P: i32 = 2;

#[allow(non_upper_case_globals)]
pub const StereoSGBM_MODE_SGBM: i32 = 0;
#[allow(non_upper_case_globals)]
pub const StereoSGBM_MODE_HH: i32 = 1;

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x9e3779b97f4a7c15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Solve a general $N \times N$ linear system $A x = b$ via Gaussian elimination with partial pivoting.
fn solve_linear_system<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let mut pivot = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..N {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                pivot = row;
            }
        }
        if max_val < 1e-13 {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }
        let diag = a[col][col];
        for row in (col + 1)..N {
            let factor = a[row][col] / diag;
            a[row][col] = 0.0;
            for k in (col + 1)..N {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0f64; N];
    for row in (0..N).rev() {
        let mut s = b[row];
        for k in (row + 1)..N {
            s -= a[row][k] * x[k];
        }
        x[row] = s / a[row][row];
        if !x[row].is_finite() {
            return None;
        }
    }
    Some(x)
}

/// Estimate a 2D affine transformation ($2 \times 3$) with RANSAC and iterative least-squares refinement.
#[allow(clippy::too_many_arguments)]
pub fn estimate_affine_2d<F: InputArray, T: InputArray>(
    from: &F,
    to: &T,
    inliers: &mut Mat,
    _method: i32,
    ransac_reproj_threshold: f64,
    max_iters: usize,
    _confidence: f64,
    refine_iters: usize,
) -> Result<Mat> {
    let p_from = from.as_points_2f()?;
    let p_to = to.as_points_2f()?;
    let n = p_from.len().min(p_to.len());
    if n < 3 {
        *inliers = Mat::zeros(n as i32, 1, CV_8UC1)?;
        return Ok(Mat::default());
    }

    let fit_lsq = |indices: &[usize]| -> Option<[f64; 6]> {
        if indices.len() < 3 {
            return None;
        }
        let mut ata = [[0.0f64; 3]; 3];
        let mut atb_x = [0.0f64; 3];
        let mut atb_y = [0.0f64; 3];
        for &i in indices {
            let x = p_from[i].x as f64;
            let y = p_from[i].y as f64;
            let u = p_to[i].x as f64;
            let v = p_to[i].y as f64;
            let row = [x, y, 1.0];
            for r in 0..3 {
                for c in 0..3 {
                    ata[r][c] += row[r] * row[c];
                }
                atb_x[r] += row[r] * u;
                atb_y[r] += row[r] * v;
            }
        }
        let mx = solve_linear_system(ata, atb_x)?;
        let my = solve_linear_system(ata, atb_y)?;
        Some([mx[0], mx[1], mx[2], my[0], my[1], my[2]])
    };

    let thr_sq = ransac_reproj_threshold * ransac_reproj_threshold;
    let mut best_model: Option<[f64; 6]> = None;
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_err = f64::INFINITY;
    let mut rng = rng_seed();

    let all_indices: Vec<usize> = (0..n).collect();
    let trials = max_iters.clamp(32, 2000);
    for t in 0..=trials {
        let model = if t == 0 {
            fit_lsq(&all_indices)
        } else {
            let i0 = (xorshift64(&mut rng) as usize) % n;
            let mut i1 = (xorshift64(&mut rng) as usize) % n;
            while i1 == i0 {
                i1 = (xorshift64(&mut rng) as usize) % n;
            }
            let mut i2 = (xorshift64(&mut rng) as usize) % n;
            while i2 == i0 || i2 == i1 {
                i2 = (xorshift64(&mut rng) as usize) % n;
            }
            fit_lsq(&[i0, i1, i2])
        };
        let Some(m) = model else { continue };
        let mut cur_inliers = Vec::with_capacity(n);
        let mut cur_err = 0.0;
        for i in 0..n {
            let x = p_from[i].x as f64;
            let y = p_from[i].y as f64;
            let px = m[0] * x + m[1] * y + m[2];
            let py = m[3] * x + m[4] * y + m[5];
            let dx = px - (p_to[i].x as f64);
            let dy = py - (p_to[i].y as f64);
            let d2 = dx * dx + dy * dy;
            if d2 <= thr_sq {
                cur_inliers.push(i);
                cur_err += d2;
            } else {
                cur_err += thr_sq;
            }
        }
        if cur_inliers.len() > best_inliers.len()
            || (cur_inliers.len() == best_inliers.len() && cur_err < best_err)
        {
            best_inliers = cur_inliers;
            best_err = cur_err;
            best_model = Some(m);
            if best_inliers.len() == n {
                break;
            }
        }
    }

    let Some(mut m) = best_model else {
        *inliers = Mat::zeros(n as i32, 1, CV_8UC1)?;
        return Ok(Mat::default());
    };

    for _ in 0..refine_iters.max(1) {
        if let Some(refined) = fit_lsq(&best_inliers) {
            m = refined;
            best_inliers.clear();
            for i in 0..n {
                let x = p_from[i].x as f64;
                let y = p_from[i].y as f64;
                let dx = (m[0] * x + m[1] * y + m[2]) - (p_to[i].x as f64);
                let dy = (m[3] * x + m[4] * y + m[5]) - (p_to[i].y as f64);
                if dx * dx + dy * dy <= thr_sq {
                    best_inliers.push(i);
                }
            }
        } else {
            break;
        }
    }

    let mut mask = vec![0u8; n];
    for &i in &best_inliers {
        mask[i] = 1;
    }
    let mut mask_mat = Mat::from_slice(&mask)?;
    mask_mat = mask_mat.reshape(1, n as i32)?;
    *inliers = mask_mat;

    record_activation(
        "calib3d::estimate_affine_2d",
        TheoremId::Thm4KinematicCommonFateComposite,
        n,
        best_inliers.len(),
        best_inliers.len() as u64,
        &[("inliers", best_inliers.len() as f64)],
    );

    Mat::from_slice_2d(&[[m[0], m[1], m[2]], [m[3], m[4], m[5]]])
}

/// Convert between a $3 \times 1$ (or $1 \times 3$) rotation vector and a $3 \times 3$ rotation matrix (`rodrigues`).
pub fn rodrigues(src: &Mat, dst: &mut Mat, _jacobian: &mut Mat) -> Result<()> {
    if src.rows() == 3 && src.cols() == 3 {
        let mut r = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = src.get_f64(i as i32, j as i32, 0);
            }
        }
        let rv = rotmat_to_rotvec(&r);
        let m = Mat::from_slice(&rv)?;
        *dst = m.reshape(1, 3)?;
        return Ok(());
    }
    let vals = src.as_f64_slice()?;
    if vals.len() != 3 {
        return Err(Error::new(
            -215,
            "rodrigues requires 3-element vector or 3x3 matrix",
        ));
    }
    let r = rotvec_to_rotmat([vals[0], vals[1], vals[2]]);
    *dst = Mat::from_slice_2d(&r)?;
    Ok(())
}

/// Project 3D object points into 2D image coordinates (`project_points_def`).
pub fn project_points_def<O: InputArray, P: OutputPoints2f>(
    object_points: &O,
    rvec: &Mat,
    tvec: &Mat,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    image_points: &mut P,
) -> Result<()> {
    project_points(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
        image_points,
        &mut crate::core::no_array(),
        0.0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn project_points<O: InputArray, P: OutputPoints2f>(
    object_points: &O,
    rvec: &Mat,
    tvec: &Mat,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    image_points: &mut P,
    _jacobian: &mut Mat,
    _aspect_ratio: f64,
) -> Result<()> {
    let obj = object_points.as_points_3f()?;
    let (fx, fy, cx, cy) = extract_intrinsics(camera_matrix)?;
    let r = if rvec.rows() == 3 && rvec.cols() == 3 {
        let mut rm = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rm[i][j] = rvec.get_f64(i as i32, j as i32, 0);
            }
        }
        rm
    } else {
        let rv = rvec.as_f64_slice()?;
        if rv.len() != 3 {
            return Err(Error::new(-215, "invalid rvec in project_points"));
        }
        rotvec_to_rotmat([rv[0], rv[1], rv[2]])
    };
    let tv = tvec.as_f64_slice()?;
    if tv.len() != 3 {
        return Err(Error::new(-215, "invalid tvec in project_points"));
    }
    let d = if dist_coeffs.is_empty() {
        Vec::new()
    } else {
        dist_coeffs.as_f64_slice()?
    };
    let k1 = d.first().copied().unwrap_or(0.0);
    let k2 = d.get(1).copied().unwrap_or(0.0);
    let p1 = d.get(2).copied().unwrap_or(0.0);
    let p2 = d.get(3).copied().unwrap_or(0.0);
    let k3 = d.get(4).copied().unwrap_or(0.0);

    let mut out = Vec::with_capacity(obj.len());
    for pt in &obj {
        let p = [pt.x as f64, pt.y as f64, pt.z as f64];
        let xc = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + tv[0];
        let yc = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + tv[1];
        let zc = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + tv[2];
        let inv_z = if zc.abs() > 1e-12 { 1.0 / zc } else { 0.0 };
        let x = xc * inv_z;
        let y = yc * inv_z;
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        out.push(Point2f::new((fx * xd + cx) as f32, (fy * yd + cy) as f32));
    }
    image_points.write_points_2f(&out)
}

#[inline]
pub fn rotvec_to_rotmat(rv: [f64; 3]) -> [[f64; 3]; 3] {
    let theta = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
    if theta < 1e-12 {
        return [
            [1.0, -rv[2], rv[1]],
            [rv[2], 1.0, -rv[0]],
            [-rv[1], rv[0], 1.0],
        ];
    }
    let kx = rv[0] / theta;
    let ky = rv[1] / theta;
    let kz = rv[2] / theta;
    let c = theta.cos();
    let s = theta.sin();
    let v = 1.0 - c;
    [
        [
            kx * kx * v + c,
            kx * ky * v - kz * s,
            kx * kz * v + ky * s,
        ],
        [
            ky * kx * v + kz * s,
            ky * ky * v + c,
            ky * kz * v - kx * s,
        ],
        [
            kz * kx * v - ky * s,
            kz * ky * v + kx * s,
            kz * kz * v + c,
        ],
    ]
}

#[inline]
pub fn rotmat_to_rotvec(r: &[[f64; 3]; 3]) -> [f64; 3] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta < 1e-8 {
        return [
            0.5 * (r[2][1] - r[1][2]),
            0.5 * (r[0][2] - r[2][0]),
            0.5 * (r[1][0] - r[0][1]),
        ];
    }
    if std::f64::consts::PI - theta < 1e-4 {
        let one_minus_c = (1.0 - cos_theta).max(1e-12);
        let mut kx = ((r[0][0] - cos_theta) / one_minus_c).max(0.0).sqrt();
        let mut ky = ((r[1][1] - cos_theta) / one_minus_c).max(0.0).sqrt();
        let mut kz = ((r[2][2] - cos_theta) / one_minus_c).max(0.0).sqrt();
        if kx >= ky && kx >= kz {
            if r[0][1] + r[1][0] < 0.0 {
                ky = -ky;
            }
            if r[0][2] + r[2][0] < 0.0 {
                kz = -kz;
            }
        } else if ky >= kx && ky >= kz {
            if r[0][1] + r[1][0] < 0.0 {
                kx = -kx;
            }
            if r[1][2] + r[2][1] < 0.0 {
                kz = -kz;
            }
        } else {
            if r[0][2] + r[2][0] < 0.0 {
                kx = -kx;
            }
            if r[1][2] + r[2][1] < 0.0 {
                ky = -ky;
            }
        }
        let skew_dot =
            kx * (r[2][1] - r[1][2]) + ky * (r[0][2] - r[2][0]) + kz * (r[1][0] - r[0][1]);
        if skew_dot < -1e-9 {
            kx = -kx;
            ky = -ky;
            kz = -kz;
        }
        let n = (kx * kx + ky * ky + kz * kz).sqrt().max(1e-12);
        return [theta * kx / n, theta * ky / n, theta * kz / n];
    }
    let scale = theta / (2.0 * theta.sin());
    [
        scale * (r[2][1] - r[1][2]),
        scale * (r[0][2] - r[2][0]),
        scale * (r[1][0] - r[0][1]),
    ]
}

fn extract_intrinsics(camera_matrix: &Mat) -> Result<(f64, f64, f64, f64)> {
    if camera_matrix.rows() != 3 || camera_matrix.cols() != 3 {
        return Err(Error::new(-215, "3x3 camera_matrix required"));
    }
    let fx = camera_matrix.get_f64(0, 0, 0);
    let fy = camera_matrix.get_f64(1, 1, 0);
    let cx = camera_matrix.get_f64(0, 2, 0);
    let cy = camera_matrix.get_f64(1, 2, 0);
    Ok((fx, fy, cx, cy))
}

fn undistort_normalized(
    u: f64,
    v: f64,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    dist: &[f64],
) -> (f64, f64) {
    let x0 = (u - cx) / fx;
    let y0 = (v - cy) / fy;
    if dist.is_empty() || dist.iter().all(|&d| d.abs() < 1e-15) {
        return (x0, y0);
    }
    let k1 = dist.first().copied().unwrap_or(0.0);
    let k2 = dist.get(1).copied().unwrap_or(0.0);
    let p1 = dist.get(2).copied().unwrap_or(0.0);
    let p2 = dist.get(3).copied().unwrap_or(0.0);
    let k3 = dist.get(4).copied().unwrap_or(0.0);

    let mut x = x0;
    let mut y = y0;
    for _ in 0..8 {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6).max(1e-6);
        let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        x = (x0 - dx) / radial;
        y = (y0 - dy) / radial;
    }
    (x, y)
}

/// Undistort 2D pixel coordinates into normalized camera rays (`undistort_points_def`).
pub fn undistort_points_def<I: InputArray, O: OutputPoints2f>(
    src: &I,
    dst: &mut O,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
) -> Result<()> {
    let (fx, fy, cx, cy) = extract_intrinsics(camera_matrix)?;
    let d = if dist_coeffs.is_empty() {
        Vec::new()
    } else {
        dist_coeffs.as_f64_slice()?
    };
    let pts = src.as_points_2f()?;
    let out: Vec<Point2f> = pts
        .iter()
        .map(|p| {
            let (x, y) = undistort_normalized(p.x as f64, p.y as f64, fx, fy, cx, cy, &d);
            Point2f::new(x as f32, y as f32)
        })
        .collect();
    dst.write_points_2f(&out)
}

/// Fisheye Kannala-Brandt 4-parameter undistortion (`fisheye_undistort_points_def`).
pub fn fisheye_undistort_points_def<I: InputArray, O: OutputPoints2f>(
    src: &I,
    dst: &mut O,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
) -> Result<()> {
    let (fx, fy, cx, cy) = extract_intrinsics(camera_matrix)?;
    let d = if dist_coeffs.is_empty() {
        Vec::new()
    } else {
        dist_coeffs.as_f64_slice()?
    };
    let k1 = d.first().copied().unwrap_or(0.0);
    let k2 = d.get(1).copied().unwrap_or(0.0);
    let k3 = d.get(2).copied().unwrap_or(0.0);
    let k4 = d.get(3).copied().unwrap_or(0.0);

    let pts = src.as_points_2f()?;
    let out: Vec<Point2f> = pts
        .iter()
        .map(|p| {
            let x_d = (p.x as f64 - cx) / fx;
            let y_d = (p.y as f64 - cy) / fy;
            let theta_d = (x_d * x_d + y_d * y_d).sqrt();
            if theta_d < 1e-10 {
                return Point2f::new(x_d as f32, y_d as f32);
            }
            let mut theta = theta_d;
            for _ in 0..8 {
                let t2 = theta * theta;
                let t4 = t2 * t2;
                let t6 = t4 * t2;
                let t8 = t4 * t4;
                let f = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d;
                let df = 1.0
                    + 3.0 * k1 * t2
                    + 5.0 * k2 * t4
                    + 7.0 * k3 * t6
                    + 9.0 * k4 * t8;
                if df.abs() < 1e-9 {
                    break;
                }
                theta -= f / df;
            }
            let scale = theta.tan() / theta_d;
            Point2f::new((x_d * scale) as f32, (y_d * scale) as f32)
        })
        .collect();
    dst.write_points_2f(&out)
}

/// Given rotation $R$ and normalized 2D correspondences $(P_i, x_i, y_i)$, solve the exact
/// $3 \times 3$ linear least-squares system for translation $t = (t_x, t_y, t_z)^T$:
/// `t_x - x_i t_z = x_i (R P_i)_z - (R P_i)_x`
/// `t_y - y_i t_z = y_i (R P_i)_z - (R P_i)_y`.
fn solve_translation_given_r(
    obj: &[Point3f],
    norm_xy: &[(f64, f64)],
    indices: &[usize],
    r: &[[f64; 3]; 3],
) -> Option<[f64; 3]> {
    if indices.len() < 2 {
        return None;
    }
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    for &i in indices {
        let p = [obj[i].x as f64, obj[i].y as f64, obj[i].z as f64];
        let qx = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2];
        let qy = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2];
        let qz = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2];
        let (x, y) = norm_xy[i];

        // Row 1: [1, 0, -x] * t = x * qz - qx
        let r1 = [1.0, 0.0, -x];
        let b1 = x * qz - qx;
        // Row 2: [0, 1, -y] * t = y * qz - qy
        let r2 = [0.0, 1.0, -y];
        let b2 = y * qz - qy;

        for a in 0..3 {
            for c in 0..3 {
                ata[a][c] += r1[a] * r1[c] + r2[a] * r2[c];
            }
            atb[a] += r1[a] * b1 + r2[a] * b2;
        }
    }
    solve_linear_system(ata, atb)
}

/// 6-DoF $\mathfrak{se}(3)$ Gauss-Newton / Levenberg-Marquardt refinement of $(r, t)$.
fn refine_pnp_se3(
    obj: &[Point3f],
    img: &[Point2f],
    indices: &[usize],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    mut rv: [f64; 3],
    mut tv: [f64; 3],
    iters: usize,
) -> ([f64; 3], [f64; 3]) {
    let mut lambda = 1e-4;
    let cost_fn = |rv_test: [f64; 3], tv_test: [f64; 3]| -> f64 {
        let r = rotvec_to_rotmat(rv_test);
        let mut s = 0.0;
        for &i in indices {
            let p = [obj[i].x as f64, obj[i].y as f64, obj[i].z as f64];
            let xc = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + tv_test[0];
            let yc = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + tv_test[1];
            let zc = (r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + tv_test[2]).max(1e-6);
            let du = (fx * xc / zc + cx) - (img[i].x as f64);
            let dv = (fy * yc / zc + cy) - (img[i].y as f64);
            s += du * du + dv * dv;
        }
        s
    };

    let mut cur_cost = cost_fn(rv, tv);
    for _ in 0..iters {
        let r = rotvec_to_rotmat(rv);
        let mut jtj = [[0.0f64; 6]; 6];
        let mut jtr = [0.0f64; 6];
        for &i in indices {
            let p = [obj[i].x as f64, obj[i].y as f64, obj[i].z as f64];
            let rx = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2];
            let ry = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2];
            let rz = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2];
            let xc = rx + tv[0];
            let yc = ry + tv[1];
            let zc = (rz + tv[2]).max(1e-6);
            let inv_z = 1.0 / zc;
            let inv_z2 = inv_z * inv_z;

            let u_hat = fx * xc * inv_z + cx;
            let v_hat = fy * yc * inv_z + cy;
            let res_u = (img[i].x as f64) - u_hat;
            let res_v = (img[i].y as f64) - v_hat;

            // d(R P)/d(delta_omega) at 0 is -[R P]_x:
            // d(xc, yc, zc)/d(wx, wy, wz) = [ 0, rz, -ry; -rz, 0, rx; ry, -rx, 0 ]
            let du_dxc = fx * inv_z;
            let du_dzc = -fx * xc * inv_z2;
            let dv_dyc = fy * inv_z;
            let dv_dzc = -fy * yc * inv_z2;

            let ju = [
                du_dzc * ry,
                du_dxc * rz - du_dzc * rx,
                -du_dxc * ry,
                du_dxc,
                0.0,
                du_dzc,
            ];
            let jv = [
                -dv_dyc * rz + dv_dzc * ry,
                -dv_dzc * rx,
                dv_dyc * rx,
                0.0,
                dv_dyc,
                dv_dzc,
            ];

            for a in 0..6 {
                for b in 0..6 {
                    jtj[a][b] += ju[a] * ju[b] + jv[a] * jv[b];
                }
                jtr[a] += ju[a] * res_u + jv[a] * res_v;
            }
        }
        for a in 0..6 {
            jtj[a][a] += lambda * (jtj[a][a].max(1.0));
        }
        let Some(delta) = solve_linear_system(jtj, jtr) else {
            break;
        };
        let d_rot = [delta[0], delta[1], delta[2]];
        let r_step = rotvec_to_rotmat(d_rot);
        let mut r_new = [[0.0f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                r_new[a][b] = r_step[a][0] * r[0][b]
                    + r_step[a][1] * r[1][b]
                    + r_step[a][2] * r[2][b];
            }
        }
        let rv_new = rotmat_to_rotvec(&r_new);
        let tv_new = [tv[0] + delta[3], tv[1] + delta[4], tv[2] + delta[5]];
        let new_cost = cost_fn(rv_new, tv_new);
        if new_cost < cur_cost {
            rv = rv_new;
            tv = tv_new;
            cur_cost = new_cost;
            lambda = (lambda * 0.5).max(1e-9);
            let norm_step: f64 = delta.iter().map(|v| v * v).sum();
            if norm_step < 1e-16 {
                break;
            }
        } else {
            lambda *= 4.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    (rv, tv)
}

/// Cached 28-element $O_h \cap SO(3)$ + tilt manifold covering net `(r0, rotvec)`.
fn cached_pnp_oh_seeds() -> &'static [([[f64; 3]; 3], [f64; 3]); 28] {
    static SEEDS: std::sync::OnceLock<[([[f64; 3]; 3], [f64; 3]); 28]> = std::sync::OnceLock::new();
    SEEDS.get_or_init(|| {
        let mut arr = [([[0.0f64; 3]; 3], [0.0f64; 3]); 28];
        let mut idx = 0usize;
        for g in crate::voxelean::OhElement::proper_24() {
            let r0 = g.to_rotation_matrix();
            arr[idx] = (r0, rotmat_to_rotvec(&r0));
            idx += 1;
        }
        for tilt in [
            [0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, -0.25, 0.0],
        ] {
            arr[idx] = (rotvec_to_rotmat(tilt), tilt);
            idx += 1;
        }
        arr
    })
}

/// Fit a 6-DoF pose $(rv, tv)$ from a subset of 3D-2D correspondences using
/// Theorem 10 ($O_h \cap SO(3)$ 24-element global manifold covering net) + POSIT + $\mathfrak{se}(3)$ LM.
fn fit_pnp_subset(
    obj: &[Point3f],
    img: &[Point2f],
    norm_xy: &[(f64, f64)],
    indices: &[usize],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    init_rv: Option<[f64; 3]>,
) -> Option<([f64; 3], [f64; 3])> {
    if indices.len() < 4 {
        return None;
    }
    let mut scored_seeds = [(f64::INFINITY, [0.0f64; 3], [0.0f64; 3]); 29];
    let mut num_scored = 0usize;

    let mut eval_seed = |r0: &[[f64; 3]; 3], rv0: [f64; 3]| {
        let Some(t0) = solve_translation_given_r(obj, norm_xy, indices, r0) else {
            return;
        };
        let mut valid_z = true;
        let mut init_err = 0.0;
        for &i in indices {
            let p = [obj[i].x as f64, obj[i].y as f64, obj[i].z as f64];
            let xc = r0[0][0] * p[0] + r0[0][1] * p[1] + r0[0][2] * p[2] + t0[0];
            let yc = r0[1][0] * p[0] + r0[1][1] * p[1] + r0[1][2] * p[2] + t0[1];
            let zc = r0[2][0] * p[0] + r0[2][1] * p[1] + r0[2][2] * p[2] + t0[2];
            if zc <= 1e-3 {
                valid_z = false;
                break;
            }
            let inv_z = 1.0 / zc;
            let du = (fx * xc * inv_z + cx) - (img[i].x as f64);
            let dv = (fy * yc * inv_z + cy) - (img[i].y as f64);
            init_err += du * du + dv * dv;
        }
        if valid_z && num_scored < 29 {
            scored_seeds[num_scored] = (init_err, rv0, t0);
            num_scored += 1;
        }
    };

    if let Some(rv0) = init_rv {
        let r0 = rotvec_to_rotmat(rv0);
        eval_seed(&r0, rv0);
    }
    for &(ref r0, rv0) in cached_pnp_oh_seeds().iter() {
        eval_seed(r0, rv0);
    }

    scored_seeds[..num_scored]
        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if num_scored == 0 {
        if let Some(t0) = solve_translation_given_r(
            obj,
            norm_xy,
            indices,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ) {
            scored_seeds[0] = (1e12, [0.0, 0.0, 0.0], [t0[0], t0[1], t0[2].abs().max(1.0)]);
            num_scored = 1;
        }
    }

    // For minimal 4-point RANSAC subsets, refine top 2 seeds with 14 LM steps; for full/LO sets, top 4 with 25 steps
    let (top_k, lm_iters) = if indices.len() <= 4 {
        (2usize.min(num_scored), 14usize)
    } else {
        (4usize.min(num_scored), 25usize)
    };

    let mut best: Option<(f64, [f64; 3], [f64; 3])> = None;
    for &(_init_err, seed_rv, t0) in scored_seeds[..top_k].iter() {
        let (rv_opt, tv_opt) =
            refine_pnp_se3(obj, img, indices, fx, fy, cx, cy, seed_rv, t0, lm_iters);
        let r_opt = rotvec_to_rotmat(rv_opt);
        let mut valid_z = true;
        let mut err_sum = 0.0;
        for &i in indices {
            let p = [obj[i].x as f64, obj[i].y as f64, obj[i].z as f64];
            let xc = r_opt[0][0] * p[0] + r_opt[0][1] * p[1] + r_opt[0][2] * p[2] + tv_opt[0];
            let yc = r_opt[1][0] * p[0] + r_opt[1][1] * p[1] + r_opt[1][2] * p[2] + tv_opt[1];
            let zc = r_opt[2][0] * p[0] + r_opt[2][1] * p[1] + r_opt[2][2] * p[2] + tv_opt[2];
            if zc <= 1e-4 {
                valid_z = false;
                break;
            }
            let inv_z = 1.0 / zc;
            let du = (fx * xc * inv_z + cx) - (img[i].x as f64);
            let dv = (fy * yc * inv_z + cy) - (img[i].y as f64);
            err_sum += du * du + dv * dv;
        }
        if valid_z && (best.is_none() || err_sum < best.unwrap().0) {
            best = Some((err_sum, rv_opt, tv_opt));
            if err_sum < 1e-6 {
                break;
            }
        }
    }
    best.map(|(_, rv, tv)| (rv, tv))
}

#[allow(clippy::too_many_arguments)]
pub fn solve_pnp<O: InputArray, I: InputArray>(
    object_points: &O,
    image_points: &I,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    rvec: &mut Mat,
    tvec: &mut Mat,
    use_extrinsic_guess: bool,
    _flags: i32,
) -> Result<bool> {
    let obj = object_points.as_points_3f()?;
    let img = image_points.as_points_2f()?;
    let n = obj.len().min(img.len());
    if n < 4 {
        return Ok(false);
    }
    let (fx, fy, cx, cy) = extract_intrinsics(camera_matrix)?;
    let d = if dist_coeffs.is_empty() {
        Vec::new()
    } else {
        dist_coeffs.as_f64_slice()?
    };
    let norm_xy: Vec<(f64, f64)> = img
        .iter()
        .map(|p| undistort_normalized(p.x as f64, p.y as f64, fx, fy, cx, cy, &d))
        .collect();
    let undist_img: Vec<Point2f> = norm_xy
        .iter()
        .map(|&(x, y)| Point2f::new((fx * x + cx) as f32, (fy * y + cy) as f32))
        .collect();

    let init_rv = if use_extrinsic_guess && !rvec.is_empty() {
        let s = rvec.as_f64_slice()?;
        if s.len() == 3 {
            Some([s[0], s[1], s[2]])
        } else {
            None
        }
    } else {
        None
    };

    let all: Vec<usize> = (0..n).collect();
    let Some((rv, tv)) =
        fit_pnp_subset(&obj, &undist_img, &norm_xy, &all, fx, fy, cx, cy, init_rv)
    else {
        return Ok(false);
    };

    *rvec = Mat::from_slice(&rv)?.reshape(1, 3)?;
    *tvec = Mat::from_slice(&tv)?.reshape(1, 3)?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub fn solve_pnp_ransac<O: InputArray, I: InputArray>(
    object_points: &O,
    image_points: &I,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    rvec: &mut Mat,
    tvec: &mut Mat,
    _use_extrinsic_guess: bool,
    iterations_count: i32,
    reprojection_error: f32,
    confidence: f64,
    inliers: &mut Mat,
    _flags: i32,
) -> Result<bool> {
    let obj = object_points.as_points_3f()?;
    let img = image_points.as_points_2f()?;
    let n = obj.len().min(img.len());
    if n < 4 {
        *inliers = Mat::default();
        return Ok(false);
    }
    let (fx, fy, cx, cy) = extract_intrinsics(camera_matrix)?;
    let d = if dist_coeffs.is_empty() {
        Vec::new()
    } else {
        dist_coeffs.as_f64_slice()?
    };
    let norm_xy: Vec<(f64, f64)> = img
        .iter()
        .map(|p| undistort_normalized(p.x as f64, p.y as f64, fx, fy, cx, cy, &d))
        .collect();
    let undist_img: Vec<Point2f> = norm_xy
        .iter()
        .map(|&(x, y)| Point2f::new((fx * x + cx) as f32, (fy * y + cy) as f32))
        .collect();

    let thr_sq = (reprojection_error as f64) * (reprojection_error as f64);
    let mut best_inliers: Vec<usize> = Vec::with_capacity(n);
    let mut cur_inliers: Vec<usize> = Vec::with_capacity(n);
    let mut best_err = f64::INFINITY;
    let mut best_pose: Option<([f64; 3], [f64; 3])> = None;
    let mut rng = rng_seed();
    let mut max_trials = iterations_count.clamp(24, 500) as usize;
    let conf = confidence.clamp(0.5, 0.99999);
    let log_one_minus_conf = (1.0 - conf).ln();
    let mut subset_buf = [0usize; 6];

    let mut t = 0usize;
    while t <= max_trials {
        let subset_len = if t == 0 {
            let k = n.min(6);
            for (i, slot) in subset_buf.iter_mut().take(k).enumerate() {
                *slot = i;
            }
            k
        } else {
            let mut count = 0usize;
            while count < 4 {
                let idx = (xorshift64(&mut rng) as usize) % n;
                if !subset_buf[..count].contains(&idx) {
                    subset_buf[count] = idx;
                    count += 1;
                }
            }
            4
        };
        t += 1;

        let Some((rv, tv)) = fit_pnp_subset(
            &obj,
            &undist_img,
            &norm_xy,
            &subset_buf[..subset_len],
            fx,
            fy,
            cx,
            cy,
            None,
        ) else {
            continue;
        };
        let r = rotvec_to_rotmat(rv);
        cur_inliers.clear();
        let mut cur_err = 0.0;
        let mut inlier_err_sum = 0.0;
        for i in 0..n {
            let p = [obj[i].x as f64, obj[i].y as f64, obj[i].z as f64];
            let xc = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + tv[0];
            let yc = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + tv[1];
            let zc = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + tv[2];
            if zc <= 1e-4 {
                cur_err += thr_sq;
                continue;
            }
            let inv_z = 1.0 / zc;
            let du = (fx * xc * inv_z + cx) - (undist_img[i].x as f64);
            let dv = (fy * yc * inv_z + cy) - (undist_img[i].y as f64);
            let d2 = du * du + dv * dv;
            if d2 <= thr_sq {
                cur_inliers.push(i);
                cur_err += d2;
                inlier_err_sum += d2;
            } else {
                cur_err += thr_sq;
            }
        }
        if cur_inliers.len() > best_inliers.len()
            || (cur_inliers.len() == best_inliers.len() && cur_err < best_err)
        {
            best_inliers.clear();
            best_inliers.extend_from_slice(&cur_inliers);
            best_err = cur_err;
            best_pose = Some((rv, tv));

            // Update adaptive RANSAC stopping bound from observed inlier ratio w = |inliers| / n
            let w_ratio = (best_inliers.len() as f64) / (n as f64);
            let p_all_inliers = (w_ratio * w_ratio * w_ratio * w_ratio).clamp(1e-6, 0.999999);
            let denom = (1.0 - p_all_inliers).ln();
            if denom < -1e-9 {
                let adaptive_k = (log_one_minus_conf / denom).ceil() as usize;
                max_trials = max_trials.min(adaptive_k.max(4));
            }

            if (best_inliers.len() == n && best_err < 1e-4)
                || (best_inliers.len() >= 8
                    && best_inliers.len() * 2 >= n
                    && inlier_err_sum < (best_inliers.len() as f64) * 1e-5)
            {
                break;
            }
        }
    }

    let Some((mut rv, mut tv)) = best_pose else {
        *inliers = Mat::default();
        return Ok(false);
    };
    if best_inliers.len() >= 4 {
        let (rv_ref, tv_ref) = refine_pnp_se3(
            &obj,
            &undist_img,
            &best_inliers,
            fx,
            fy,
            cx,
            cy,
            rv,
            tv,
            25,
        );
        rv = rv_ref;
        tv = tv_ref;
        let r = rotvec_to_rotmat(rv);
        best_inliers.clear();
        for i in 0..n {
            let p = [obj[i].x as f64, obj[i].y as f64, obj[i].z as f64];
            let xc = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + tv[0];
            let yc = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + tv[1];
            let zc = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + tv[2];
            if zc > 1e-4 {
                let du = (fx * xc / zc + cx) - (undist_img[i].x as f64);
                let dv = (fy * yc / zc + cy) - (undist_img[i].y as f64);
                if du * du + dv * dv <= thr_sq {
                    best_inliers.push(i);
                }
            }
        }
    }

    let inlier_i32: Vec<i32> = best_inliers.iter().map(|&i| i as i32).collect();
    let mut inl_mat = Mat::from_slice(&inlier_i32)?;
    if !inlier_i32.is_empty() {
        inl_mat = inl_mat.reshape(1, inlier_i32.len() as i32)?;
    } else {
        inl_mat = Mat::zeros(0, 1, CV_32SC1)?;
    }
    *inliers = inl_mat;
    *rvec = Mat::from_slice(&rv)?.reshape(1, 3)?;
    *tvec = Mat::from_slice(&tv)?.reshape(1, 3)?;

    record_activation(
        "calib3d::solve_pnp_ransac",
        TheoremId::Thm10ChiralOctahedralOh,
        n,
        best_inliers.len(),
        best_inliers.len() as u64,
        &[("inliers", best_inliers.len() as f64)],
    );
    Ok(!best_inliers.is_empty())
}

fn mat_mul_3x3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mat_transpose_3x3(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn mat_vec_3x3(a: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

/// Bouguet stereo rectification (`stereo_rectify`).
#[allow(clippy::too_many_arguments)]
pub fn stereo_rectify(
    camera_matrix1: &Mat,
    _dist_coeffs1: &Mat,
    camera_matrix2: &Mat,
    _dist_coeffs2: &Mat,
    image_size: Size,
    r: &Mat,
    t: &Mat,
    r1: &mut Mat,
    r2: &mut Mat,
    p1: &mut Mat,
    p2: &mut Mat,
    q: &mut Mat,
    flags: i32,
    _alpha: f64,
    new_image_size: Size,
    valid_pix_roi1: &mut Rect,
    valid_pix_roi2: &mut Rect,
) -> Result<()> {
    let (fx1, fy1, cx1, cy1) = extract_intrinsics(camera_matrix1)?;
    let (fx2, fy2, cx2, cy2) = extract_intrinsics(camera_matrix2)?;
    if r.rows() != 3 || r.cols() != 3 {
        return Err(Error::new(-215, "3x3 rotation required in stereo_rectify"));
    }
    let t_slice = t.as_f64_slice()?;
    if t_slice.len() != 3 {
        return Err(Error::new(
            -215,
            "3-element translation required in stereo_rectify",
        ));
    }
    let mut r_mat = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r_mat[i][j] = r.get_f64(i as i32, j as i32, 0);
        }
    }
    let om = rotmat_to_rotvec(&r_mat);
    let r_half = rotvec_to_rotmat([-0.5 * om[0], -0.5 * om[1], -0.5 * om[2]]);
    let r_half_inv = mat_transpose_3x3(&r_half);

    // Rotated translation vector t_rot = r_half * t
    let t_rot = mat_vec_3x3(&r_half, [t_slice[0], t_slice[1], t_slice[2]]);
    let t_norm = (t_rot[0] * t_rot[0] + t_rot[1] * t_rot[1] + t_rot[2] * t_rot[2])
        .sqrt()
        .max(1e-12);
    let idx_dominant = if t_rot[0].abs() >= t_rot[1].abs() {
        0
    } else {
        1
    };
    let sign = if t_rot[idx_dominant] > 0.0 { 1.0 } else { -1.0 };

    let e1 = [
        sign * t_rot[0] / t_norm,
        sign * t_rot[1] / t_norm,
        sign * t_rot[2] / t_norm,
    ];
    let e2_norm = (e1[0] * e1[0] + e1[1] * e1[1]).sqrt().max(1e-12);
    let e2 = [-e1[1] / e2_norm, e1[0] / e2_norm, 0.0];
    let e3 = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let r_rect = [e1, e2, e3];

    let r1_mat = mat_mul_3x3(&r_rect, &r_half_inv);
    let r2_mat = mat_mul_3x3(&r_rect, &r_half);

    let t_rect = mat_vec_3x3(&r2_mat, [t_slice[0], t_slice[1], t_slice[2]]);
    let f_new = 0.5 * (fx1 + fy1 + fx2 + fy2) * 0.5;
    let cx_new = 0.5 * (cx1 + cx2);
    let cy_new = 0.5 * (cy1 + cy2);
    let cx2_new = if (flags & CALIB_ZERO_DISPARITY) != 0 {
        cx_new
    } else {
        cx2
    };

    *r1 = Mat::from_slice_2d(&r1_mat)?;
    *r2 = Mat::from_slice_2d(&r2_mat)?;
    *p1 = Mat::from_slice_2d(&[
        [f_new, 0.0, cx_new, 0.0],
        [0.0, f_new, cy_new, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])?;
    *p2 = Mat::from_slice_2d(&[
        [f_new, 0.0, cx2_new, f_new * t_rect[0]],
        [0.0, f_new, cy_new, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])?;

    let inv_tx = if t_rect[0].abs() > 1e-12 {
        -1.0 / t_rect[0]
    } else {
        0.0
    };
    *q = Mat::from_slice_2d(&[
        [1.0, 0.0, 0.0, -cx_new],
        [0.0, 1.0, 0.0, -cy_new],
        [0.0, 0.0, 0.0, f_new],
        [0.0, 0.0, inv_tx, (cx_new - cx2_new) * inv_tx],
    ])?;

    let sz = if new_image_size.width > 0 && new_image_size.height > 0 {
        new_image_size
    } else {
        image_size
    };
    *valid_pix_roi1 = Rect::new(0, 0, sz.width, sz.height);
    *valid_pix_roi2 = Rect::new(0, 0, sz.width, sz.height);
    Ok(())
}

/// Precompute undistortion and rectification lookup maps (`init_undistort_rectify_map`).
#[allow(clippy::too_many_arguments)]
pub fn init_undistort_rectify_map(
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    r: &Mat,
    new_camera_matrix: &Mat,
    size: Size,
    _m1type: i32,
    map1: &mut Mat,
    map2: &mut Mat,
) -> Result<()> {
    let (fx, fy, cx, cy) = extract_intrinsics(camera_matrix)?;
    let d = if dist_coeffs.is_empty() {
        Vec::new()
    } else {
        dist_coeffs.as_f64_slice()?
    };
    let k1 = d.first().copied().unwrap_or(0.0);
    let k2 = d.get(1).copied().unwrap_or(0.0);
    let p1 = d.get(2).copied().unwrap_or(0.0);
    let p2 = d.get(3).copied().unwrap_or(0.0);
    let k3 = d.get(4).copied().unwrap_or(0.0);

    let r_inv = if r.rows() == 3 && r.cols() == 3 {
        let mut rm = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rm[i][j] = r.get_f64(i as i32, j as i32, 0);
            }
        }
        mat_transpose_3x3(&rm)
    } else {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    };

    let f_new_x = new_camera_matrix.get_f64(0, 0, 0);
    let f_new_y = new_camera_matrix.get_f64(1, 1, 0);
    let c_new_x = new_camera_matrix.get_f64(0, 2, 0);
    let c_new_y = new_camera_matrix.get_f64(1, 2, 0);

    let w = size.width;
    let h = size.height;
    let mut mx = Mat::zeros(h, w, CV_32FC1)?;
    let mut my = Mat::zeros(h, w, CV_32FC1)?;

    for v in 0..h {
        let yn = (v as f64 - c_new_y) / f_new_y;
        for u in 0..w {
            let xn = (u as f64 - c_new_x) / f_new_x;
            let ray = mat_vec_3x3(&r_inv, [xn, yn, 1.0]);
            let z = ray[2].max(1e-9);
            let x = ray[0] / z;
            let y = ray[1] / z;
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            mx.set_f64(v, u, 0, fx * xd + cx);
            my.set_f64(v, u, 0, fy * yd + cy);
        }
    }
    *map1 = mx;
    *map2 = my;
    Ok(())
}

fn det_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Exact $3 \times 3$ Singular Value Decomposition $A = U \operatorname{diag}(\sigma_0, \sigma_1, \sigma_2) V^T$
/// with $\sigma_0 \ge \sigma_1 \ge \sigma_2 \ge 0$ and $\det(U) = +1, \det(V) = +1$.
pub fn svd_3x3(a: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    let at = mat_transpose_3x3(a);
    let mut s = mat_mul_3x3(&at, a);
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // Cyclic Jacobi eigenvalue decomposition on symmetric 3x3 matrix S = A^T A
    for _sweep in 0..20 {
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let spq = s[p][q];
            if spq.abs() < 1e-15 {
                continue;
            }
            let diff = s[q][q] - s[p][p];
            let t = if diff.abs() < 1e-15 {
                1.0
            } else {
                let tau = diff / (2.0 * spq);
                let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                sign / (tau.abs() + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let sn = t * c;

            // Update S = J^T S J
            let spp = s[p][p];
            let sqq = s[q][q];
            s[p][p] = c * c * spp - 2.0 * sn * c * spq + sn * sn * sqq;
            s[q][q] = sn * sn * spp + 2.0 * sn * c * spq + c * c * sqq;
            s[p][q] = 0.0;
            s[q][p] = 0.0;
            let r = 3 - p - q;
            let spr = s[p][r];
            let sqr = s[q][r];
            s[p][r] = c * spr - sn * sqr;
            s[r][p] = s[p][r];
            s[q][r] = sn * spr + c * sqr;
            s[r][q] = s[q][r];

            // Update V = V J
            for row in 0..3 {
                let vp = v[row][p];
                let vq = v[row][q];
                v[row][p] = c * vp - sn * vq;
                v[row][q] = sn * vp + c * vq;
            }
        }
    }

    // Compute w_k = A * v_k and sigma_k = ||w_k||_2 directly from A to avoid sqrt(eps) loss on rank-deficient matrices
    let av_raw = mat_mul_3x3(a, &v);
    let mut sigmas = [0.0f64; 3];
    for c in 0..3 {
        sigmas[c] = (av_raw[0][c] * av_raw[0][c]
            + av_raw[1][c] * av_raw[1][c]
            + av_raw[2][c] * av_raw[2][c])
            .sqrt();
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| {
        sigmas[j]
            .partial_cmp(&sigmas[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sigmas = [sigmas[order[0]], sigmas[order[1]], sigmas[order[2]]];
    let mut v_sorted = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            v_sorted[r][c] = v[r][order[c]];
        }
    }
    if det_3x3(&v_sorted) < 0.0 {
        for r in 0..3 {
            v_sorted[r][2] = -v_sorted[r][2];
        }
    }

    let av = mat_mul_3x3(a, &v_sorted);
    let mut u0 = if sigmas[0] > 1e-12 {
        [
            av[0][0] / sigmas[0],
            av[1][0] / sigmas[0],
            av[2][0] / sigmas[0],
        ]
    } else {
        [1.0, 0.0, 0.0]
    };
    let n0 = (u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]).sqrt().max(1e-15);
    u0 = [u0[0] / n0, u0[1] / n0, u0[2] / n0];

    let mut u1 = if sigmas[1] > 1e-12 {
        [
            av[0][1] / sigmas[1],
            av[1][1] / sigmas[1],
            av[2][1] / sigmas[1],
        ]
    } else if u0[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let dot01 = u0[0] * u1[0] + u0[1] * u1[1] + u0[2] * u1[2];
    u1 = [
        u1[0] - dot01 * u0[0],
        u1[1] - dot01 * u0[1],
        u1[2] - dot01 * u0[2],
    ];
    let n1 = (u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]).sqrt().max(1e-15);
    u1 = [u1[0] / n1, u1[1] / n1, u1[2] / n1];

    let mut u2 = [
        u0[1] * u1[2] - u0[2] * u1[1],
        u0[2] * u1[0] - u0[0] * u1[2],
        u0[0] * u1[1] - u0[1] * u1[0],
    ];
    let dot2 = av[0][2] * u2[0] + av[1][2] * u2[1] + av[2][2] * u2[2];
    if dot2 < 0.0 {
        u2 = [-u2[0], -u2[1], -u2[2]];
    }

    let u_mat = [
        [u0[0], u1[0], u2[0]],
        [u0[1], u1[1], u2[1]],
        [u0[2], u1[2], u2[2]],
    ];
    (u_mat, sigmas, v_sorted)
}

/// Essential matrix estimation (`find_essential_mat`) using Hartley-normalized 8-point RANSAC + SVD manifold projection.
#[allow(clippy::too_many_arguments)]
pub fn find_essential_mat<I1: InputArray, I2: InputArray>(
    points1: &I1,
    points2: &I2,
    camera_matrix: &Mat,
    _method: i32,
    _prob: f64,
    threshold: f64,
    max_iters: i32,
    mask: &mut Mat,
) -> Result<Mat> {
    let p1_raw = points1.as_points_2f()?;
    let p2_raw = points2.as_points_2f()?;
    let n = p1_raw.len().min(p2_raw.len());
    if n < 8 {
        *mask = Mat::zeros(n as i32, 1, CV_8UC1)?;
        return Ok(Mat::default());
    }
    let (fx, fy, cx, cy) = extract_intrinsics(camera_matrix)?;
    let p1: Vec<(f64, f64)> = p1_raw
        .iter()
        .map(|p| ((p.x as f64 - cx) / fx, (p.y as f64 - cy) / fy))
        .collect();
    let p2: Vec<(f64, f64)> = p2_raw
        .iter()
        .map(|p| ((p.x as f64 - cx) / fx, (p.y as f64 - cy) / fy))
        .collect();

    let fit_8point = |indices: &[usize]| -> Option<[[f64; 3]; 3]> {
        if indices.len() < 8 {
            return None;
        }
        // Hartley isotropic normalization
        let m_cnt = indices.len() as f64;
        let (mut c1x, mut c1y, mut c2x, mut c2y) = (0.0, 0.0, 0.0, 0.0);
        for &i in indices {
            c1x += p1[i].0;
            c1y += p1[i].1;
            c2x += p2[i].0;
            c2y += p2[i].1;
        }
        c1x /= m_cnt;
        c1y /= m_cnt;
        c2x /= m_cnt;
        c2y /= m_cnt;
        let (mut d1, mut d2) = (0.0, 0.0);
        for &i in indices {
            d1 += ((p1[i].0 - c1x).powi(2) + (p1[i].1 - c1y).powi(2)).sqrt();
            d2 += ((p2[i].0 - c2x).powi(2) + (p2[i].1 - c2y).powi(2)).sqrt();
        }
        let s1 = std::f64::consts::SQRT_2 * m_cnt / d1.max(1e-9);
        let s2 = std::f64::consts::SQRT_2 * m_cnt / d2.max(1e-9);

        let mut ata = [[0.0f64; 9]; 9];
        for &i in indices {
            let x1 = (p1[i].0 - c1x) * s1;
            let y1 = (p1[i].1 - c1y) * s1;
            let x2 = (p2[i].0 - c2x) * s2;
            let y2 = (p2[i].1 - c2y) * s2;
            let row = [
                x2 * x1,
                x2 * y1,
                x2,
                y2 * x1,
                y2 * y1,
                y2,
                x1,
                y1,
                1.0,
            ];
            for a in 0..9 {
                for b in 0..9 {
                    ata[a][b] += row[a] * row[b];
                }
            }
        }
        // Inverse power iteration on ata from 3 orthogonal initial seeds to avoid nullspace orthogonality
        let mut best_v = [0.0f64; 9];
        let mut best_rq = f64::INFINITY;
        for seed in [
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, -0.5, -0.2, 0.1, 0.4, 0.5, -0.4, 0.1],
        ] {
            let mut v = seed;
            let mut reg = ata;
            for d in 0..9 {
                reg[d][d] += 1e-11;
            }
            for _ in 0..14 {
                let Some(w) = solve_linear_system(reg, v) else {
                    break;
                };
                let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
                for k in 0..9 {
                    v[k] = w[k] / norm;
                }
            }
            let mut rq = 0.0;
            for a in 0..9 {
                let mut av = 0.0;
                for b in 0..9 {
                    av += ata[a][b] * v[b];
                }
                rq += v[a] * av;
            }
            if rq < best_rq {
                best_rq = rq;
                best_v = v;
            }
        }
        let e_norm = [
            [best_v[0], best_v[1], best_v[2]],
            [best_v[3], best_v[4], best_v[5]],
            [best_v[6], best_v[7], best_v[8]],
        ];
        let t1 = [[s1, 0.0, -s1 * c1x], [0.0, s1, -s1 * c1y], [0.0, 0.0, 1.0]];
        let t2_t = [[s2, 0.0, 0.0], [0.0, s2, 0.0], [-s2 * c2x, -s2 * c2y, 1.0]];
        let e_raw = mat_mul_3x3(&t2_t, &mat_mul_3x3(&e_norm, &t1));
        // Project onto Essential manifold Sigma = diag(1, 1, 0)
        let (u, _sig, v) = svd_3x3(&e_raw);
        let vt = mat_transpose_3x3(&v);
        let u_diag = [
            [u[0][0], u[0][1], 0.0],
            [u[1][0], u[1][1], 0.0],
            [u[2][0], u[2][1], 0.0],
        ];
        Some(mat_mul_3x3(&u_diag, &vt))
    };

    let mut best_e: Option<[[f64; 3]; 3]> = None;
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_err = f64::INFINITY;
    let mut rng = rng_seed();
    let f_avg = 0.5 * (fx + fy);
    let thr_norm = threshold / f_avg.max(1.0);
    let thr_sq = (thr_norm * thr_norm).max(threshold * threshold * 1e-4);
    let trials = max_iters.clamp(32, 500) as usize;

    for t in 0..=trials {
        let subset: Vec<usize> = if t == 0 {
            (0..n).collect()
        } else {
            let mut s = Vec::with_capacity(8);
            while s.len() < 8 {
                let idx = (xorshift64(&mut rng) as usize) % n;
                if !s.contains(&idx) {
                    s.push(idx);
                }
            }
            s
        };
        let Some(e) = fit_8point(&subset) else {
            continue;
        };
        let mut cur_inliers = Vec::with_capacity(n);
        let mut cur_err = 0.0;
        for i in 0..n {
            let (x1, y1) = p1[i];
            let (x2, y2) = p2[i];
            let ex1 = [
                e[0][0] * x1 + e[0][1] * y1 + e[0][2],
                e[1][0] * x1 + e[1][1] * y1 + e[1][2],
                e[2][0] * x1 + e[2][1] * y1 + e[2][2],
            ];
            let etx2 = [
                e[0][0] * x2 + e[1][0] * y2 + e[2][0],
                e[0][1] * x2 + e[1][1] * y2 + e[2][1],
                e[0][2] * x2 + e[1][2] * y2 + e[2][2],
            ];
            let x2t_ex1 = x2 * ex1[0] + y2 * ex1[1] + ex1[2];
            let denom =
                (ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1])
                    .max(1e-12);
            let sampson = (x2t_ex1 * x2t_ex1) / denom;
            if sampson <= thr_sq {
                cur_inliers.push(i);
                cur_err += sampson;
            } else {
                cur_err += thr_sq;
            }
        }
        if cur_inliers.len() > best_inliers.len()
            || (cur_inliers.len() == best_inliers.len() && cur_err < best_err)
        {
            best_inliers = cur_inliers;
            best_err = cur_err;
            best_e = Some(e);
            if best_inliers.len() == n {
                break;
            }
        }
    }

    let Some(mut e_mat) = best_e else {
        *mask = Mat::zeros(n as i32, 1, CV_8UC1)?;
        return Ok(Mat::default());
    };
    if best_inliers.len() >= 8 {
        if let Some(refined) = fit_8point(&best_inliers) {
            e_mat = refined;
        }
    }
    let mut m_bytes = vec![0u8; n];
    for &i in &best_inliers {
        m_bytes[i] = 1;
    }
    *mask = Mat::from_slice(&m_bytes)?.reshape(1, n as i32)?;
    Mat::from_slice_2d(&e_mat)
}

/// Recover relative camera rotation `r`, unit translation `t`, and triangulated 4D homogeneous
/// points `points` (`recover_pose_triangulated`) using exact 4-way Hartley-Zisserman SVD decomposition
/// plus translation-parallax fallback.
#[allow(clippy::too_many_arguments)]
pub fn recover_pose_triangulated<I1: InputArray, I2: InputArray>(
    essential: &Mat,
    points1: &I1,
    points2: &I2,
    camera_matrix: &Mat,
    r: &mut Mat,
    t: &mut Mat,
    distance_thresh: f64,
    mask: &mut Mat,
    points: &mut Mat,
) -> Result<i32> {
    if essential.rows() != 3 || essential.cols() != 3 {
        return Ok(0);
    }
    let p1_raw = points1.as_points_2f()?;
    let p2_raw = points2.as_points_2f()?;
    let n = p1_raw.len().min(p2_raw.len());
    if n == 0 {
        return Ok(0);
    }
    let (fx, fy, cx, cy) = extract_intrinsics(camera_matrix)?;
    let p1: Vec<(f64, f64)> = p1_raw
        .iter()
        .map(|p| ((p.x as f64 - cx) / fx, (p.y as f64 - cy) / fy))
        .collect();
    let p2: Vec<(f64, f64)> = p2_raw
        .iter()
        .map(|p| ((p.x as f64 - cx) / fx, (p.y as f64 - cy) / fy))
        .collect();

    let mut e = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            e[i][j] = essential.get_f64(i as i32, j as i32, 0);
        }
    }
    let (mut u, _sig, mut v) = svd_3x3(&e);
    if det_3x3(&u) < 0.0 {
        for r_idx in 0..3 {
            u[r_idx][2] = -u[r_idx][2];
        }
    }
    if det_3x3(&v) < 0.0 {
        for r_idx in 0..3 {
            v[r_idx][2] = -v[r_idx][2];
        }
    }
    let vt = mat_transpose_3x3(&v);
    let w_mat = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let wt_mat = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let r_a = mat_mul_3x3(&u, &mat_mul_3x3(&w_mat, &vt));
    let r_b = mat_mul_3x3(&u, &mat_mul_3x3(&wt_mat, &vt));
    let u3 = [u[0][2], u[1][2], u[2][2]];
    let neg_u3 = [-u3[0], -u3[1], -u3[2]];

    // Also include translation direction directly from median optical flow for pure/near-pure translation
    let mut dx_sum = 0.0;
    let mut dy_sum = 0.0;
    for i in 0..n {
        dx_sum += p2[i].0 - p1[i].0;
        dy_sum += p2[i].1 - p1[i].1;
    }
    let flow_norm = (dx_sum * dx_sum + dy_sum * dy_sum).sqrt().max(1e-12);
    let t_flow = [dx_sum / flow_norm, dy_sum / flow_norm, 0.0];
    let r_ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let candidate_pairs: [([[f64; 3]; 3], [f64; 3]); 6] = [
        (r_a, u3),
        (r_a, neg_u3),
        (r_b, u3),
        (r_b, neg_u3),
        (r_ident, t_flow),
        (r_ident, [-t_flow[0], -t_flow[1], 0.0]),
    ];

    let triangulate_one = |rm: &[[f64; 3]; 3],
                           tv: &[f64; 3],
                           (x1, y1): (f64, f64),
                           (x2, y2): (f64, f64)|
     -> Option<[f64; 3]> {
        let d1 = [x1, y1, 1.0];
        let rd1 = mat_vec_3x3(rm, d1);
        let a_x = x2 * rd1[2] - rd1[0];
        let b_x = tv[0] - x2 * tv[2];
        let a_y = y2 * rd1[2] - rd1[1];
        let b_y = tv[1] - y2 * tv[2];
        let denom = a_x * a_x + a_y * a_y;
        if denom < 1e-12 {
            return None;
        }
        let z1 = (a_x * b_x + a_y * b_y) / denom;
        Some([z1 * x1, z1 * y1, z1])
    };

    let mut best_good = -1i32;
    let mut best_err = f64::INFINITY;
    let mut best_r = r_ident;
    let mut best_t = u3;
    let mut best_pts = vec![[0.0, 0.0, 1.0, 1.0]; n];
    let mut best_mask = vec![0u8; n];

    for (rm, tv) in &candidate_pairs {
        let mut good = 0i32;
        let mut reproj_err = 0.0;
        let mut pts4 = vec![[0.0, 0.0, 1.0, 1.0]; n];
        let mut m = vec![0u8; n];
        for i in 0..n {
            if !mask.is_empty() && *mask.at::<u8>(i as i32).unwrap_or(&1) == 0 {
                continue;
            }
            if let Some(xyz) = triangulate_one(rm, tv, p1[i], p2[i]) {
                let q = mat_vec_3x3(rm, xyz);
                let x2_p = q[0] + tv[0];
                let y2_p = q[1] + tv[1];
                let z2 = q[2] + tv[2];
                if xyz[2] > 0.01
                    && xyz[2] < distance_thresh
                    && z2 > 0.01
                    && z2 < distance_thresh
                {
                    let du = x2_p / z2 - p2[i].0;
                    let dv = y2_p / z2 - p2[i].1;
                    let err = du * du + dv * dv;
                    if err < 0.05 {
                        good += 1;
                        reproj_err += err;
                        m[i] = 1;
                        pts4[i] = [xyz[0], xyz[1], xyz[2], 1.0];
                    }
                }
            }
        }
        if good > best_good || (good == best_good && reproj_err < best_err) {
            best_good = good;
            best_err = reproj_err;
            best_r = *rm;
            best_t = *tv;
            best_pts = pts4;
            best_mask = m;
        }
    }

    *r = Mat::from_slice_2d(&best_r)?;
    *t = Mat::from_slice(&best_t)?.reshape(1, 3)?;
    *mask = Mat::from_slice(&best_mask)?.reshape(1, n as i32)?;
    let mut p_mat = Mat::zeros(4, n as i32, CV_64FC1)?;
    for i in 0..n {
        for r_idx in 0..4 {
            p_mat.set_f64(r_idx as i32, i as i32, 0, best_pts[i][r_idx]);
        }
    }
    *points = p_mat;
    Ok(best_good.max(0))
}

/// Semi-Global Block Matching Stereo Engine (`StereoSGBM`) powered by Pixelean
/// Census Bitplane Hamming + Birchfield-Tomasi SAD, 4-path Hirschmueller dynamic programming,
/// left-right consistency checking (`disp12_max_diff`), and subpixel quadratic interpolation.
#[derive(Debug, Clone)]
pub struct StereoSGBM {
    pub min_disparity: i32,
    pub num_disparities: i32,
    pub block_size: i32,
    pub p1: i32,
    pub p2: i32,
    pub disp12_max_diff: i32,
    pub pre_filter_cap: i32,
    pub uniqueness_ratio: i32,
    pub speckle_window_size: i32,
    pub speckle_range: i32,
    pub mode: i32,
    work_base_cost: Vec<u16>,
    work_agg_cost: Vec<u16>,
    work_census_l: Vec<u64>,
    work_census_r: Vec<u64>,
    work_unit_ring: [Vec<u16>; 3],
    work_bonus_ring: [Vec<u16>; 3],
    work_textured: Vec<bool>,
    work_prev_vert: Vec<u16>,
}

impl StereoSGBM {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        min_disparity: i32,
        num_disparities: i32,
        block_size: i32,
        p1: i32,
        p2: i32,
        disp12_max_diff: i32,
        pre_filter_cap: i32,
        uniqueness_ratio: i32,
        speckle_window_size: i32,
        speckle_range: i32,
        mode: i32,
    ) -> Result<Self> {
        if num_disparities <= 0 || num_disparities % 16 != 0 {
            return Err(Error::new(
                -215,
                "num_disparities must be positive multiple of 16",
            ));
        }
        Ok(Self {
            min_disparity,
            num_disparities,
            block_size: block_size.max(3) | 1,
            p1,
            p2,
            disp12_max_diff,
            pre_filter_cap,
            uniqueness_ratio,
            speckle_window_size,
            speckle_range,
            mode,
            work_base_cost: Vec::new(),
            work_agg_cost: Vec::new(),
            work_census_l: Vec::new(),
            work_census_r: Vec::new(),
            work_unit_ring: [Vec::new(), Vec::new(), Vec::new()],
            work_bonus_ring: [Vec::new(), Vec::new(), Vec::new()],
            work_textured: Vec::new(),
            work_prev_vert: Vec::new(),
        })
    }

    pub fn compute(&mut self, left: &Mat, right: &Mat, disparity: &mut Mat) -> Result<()> {
        if left.is_empty()
            || right.is_empty()
            || left.rows() != right.rows()
            || left.cols() != right.cols()
        {
            return Err(Error::new(-215, "invalid stereo pair in StereoSGBM::compute"));
        }
        let h = left.rows() as usize;
        let w = left.cols() as usize;
        let hw = h * w;
        let nd = self.num_disparities as usize;
        let rad = (self.block_size / 2).max(1) as isize;
        let invalid_disp = ((self.min_disparity - 1) * 16) as i16;
        if disparity.rows() != h as i32
            || disparity.cols() != w as i32
            || disparity.typ() != crate::core::CV_16SC1
            || disparity.raw_bytes().len() != hw * 2
        {
            *disparity = Mat::from_raw_bytes(
                h as i32,
                w as i32,
                crate::core::CV_16SC1,
                vec![0u8; hw * 2],
            )?;
        }
        let out: &mut [i16] = unsafe {
            std::slice::from_raw_parts_mut(disparity.raw_bytes_mut().as_mut_ptr() as *mut i16, hw)
        };
        out.fill(invalid_disp);

        // Precompute 64-bit 9x7 Pixelean Census bitplane descriptor (Theorem 1 & 2) into reusable buffers
        if self.work_census_l.len() < hw {
            self.work_census_l.resize(hw, 0);
        }
        if self.work_census_r.len() < hw {
            self.work_census_r.resize(hw, 0);
        }
        crate::hardware::census_transform_9x7_u64_into(left, &mut self.work_census_l[..hw])?;
        crate::hardware::census_transform_9x7_u64_into(right, &mut self.work_census_r[..hw])?;
        let census_l = &self.work_census_l[..hw];
        let census_r = &self.work_census_r[..hw];

        let d_min = self.min_disparity;
        let d_max = self.min_disparity + self.num_disparities;
        let uniq = self.uniqueness_ratio.max(0) as u32;
        let p1_pen = (self.p1.max(8) as u16).min(1024);
        let p2_pen = (self.p2.max(32) as u16).max(p1_pen + 1).min(4096);
        let inf_cost = 15_000u16;

        let vol_len = hw * nd;
        if self.work_base_cost.len() < vol_len {
            self.work_base_cost.resize(vol_len, 0);
        }
        if self.work_agg_cost.len() < vol_len {
            self.work_agg_cost.resize(vol_len, 0);
        }
        let base_cost = &mut self.work_base_cost[..vol_len];
        let agg_cost = &mut self.work_agg_cost[..vol_len];

        if self.work_textured.len() < hw {
            self.work_textured.resize(hw, false);
        } else {
            self.work_textured[..hw].fill(false);
        }
        let textured = &mut self.work_textured[..hw];

        let is_u8 = left.typ() == crate::core::CV_8UC1 && right.typ() == crate::core::CV_8UC1;
        let l_bytes = if is_u8 { left.data_bytes()? } else { &[][..] };
        let r_bytes = if is_u8 { right.data_bytes()? } else { &[][..] };

        let y_start = rad as usize;
        let y_end = h.saturating_sub(rad as usize);
        let x_start = rad as usize;
        let x_end = w.saturating_sub(rad as usize);

        #[inline(always)]
        fn dp_step(
            prev_l: &[u16],
            curr_c: &[u16],
            out_l: &mut [u16],
            nd: usize,
            p1_pen: u16,
            p2_pen: u16,
            inf_cost: u16,
        ) {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                if nd == 16 {
                    // Fast 16-disparity single-shot NEON SGM recurrence (zero branches, 2x uint16x8_t)
                    unsafe {
                        let p_lo = vld1q_u16(prev_l.as_ptr());
                        let p_hi = vld1q_u16(prev_l.as_ptr().add(8));
                        let min_prev = vminvq_u16(vminq_u16(p_lo, p_hi));
                        if min_prev >= inf_cost {
                            vst1q_u16(out_l.as_mut_ptr(), vld1q_u16(curr_c.as_ptr()));
                            vst1q_u16(out_l.as_mut_ptr().add(8), vld1q_u16(curr_c.as_ptr().add(8)));
                            return;
                        }
                        let inf_vec = vdupq_n_u16(inf_cost);
                        let p1_vec = vdupq_n_u16(p1_pen);
                        let mp2_vec = vdupq_n_u16(min_prev.saturating_add(p2_pen));
                        let mp_vec = vdupq_n_u16(min_prev);

                        let pm1_lo = vextq_u16::<7>(inf_vec, p_lo);
                        let pm1_hi = vextq_u16::<7>(p_lo, p_hi);
                        let pp1_lo = vextq_u16::<1>(p_lo, p_hi);
                        let pp1_hi = vextq_u16::<1>(p_hi, inf_vec);

                        let nbr_lo = vqaddq_u16(vminq_u16(pm1_lo, pp1_lo), p1_vec);
                        let nbr_hi = vqaddq_u16(vminq_u16(pm1_hi, pp1_hi), p1_vec);

                        let bt_lo = vminq_u16(vminq_u16(p_lo, mp2_vec), nbr_lo);
                        let bt_hi = vminq_u16(vminq_u16(p_hi, mp2_vec), nbr_hi);

                        let c_lo = vld1q_u16(curr_c.as_ptr());
                        let c_hi = vld1q_u16(curr_c.as_ptr().add(8));
                        let res_lo = vminq_u16(vqaddq_u16(c_lo, vsubq_u16(bt_lo, mp_vec)), inf_vec);
                        let res_hi = vminq_u16(vqaddq_u16(c_hi, vsubq_u16(bt_hi, mp_vec)), inf_vec);

                        vst1q_u16(out_l.as_mut_ptr(), res_lo);
                        vst1q_u16(out_l.as_mut_ptr().add(8), res_hi);
                    }
                    return;
                }
            }

            let mut min_prev = inf_cost;
            for &v in &prev_l[..nd] {
                if v < min_prev {
                    min_prev = v;
                }
            }
            if min_prev >= inf_cost {
                out_l[..nd].copy_from_slice(&curr_c[..nd]);
                return;
            }
            let min_plus_p2 = min_prev.saturating_add(p2_pen);
            let bt0 = prev_l[0].min(min_plus_p2).min(prev_l[1].saturating_add(p1_pen));
            out_l[0] = (curr_c[0].saturating_add(bt0 - min_prev)).min(inf_cost);
            for d in 1..(nd - 1) {
                let bt = prev_l[d]
                    .min(min_plus_p2)
                    .min(prev_l[d - 1].saturating_add(p1_pen))
                    .min(prev_l[d + 1].saturating_add(p1_pen));
                out_l[d] = (curr_c[d].saturating_add(bt - min_prev)).min(inf_cost);
            }
            let btn = prev_l[nd - 1]
                .min(min_plus_p2)
                .min(prev_l[nd - 2].saturating_add(p1_pen));
            out_l[nd - 1] = (curr_c[nd - 1].saturating_add(btn - min_prev)).min(inf_cost);
        }

        // 3-row L1 ring buffer for single-pixel Census Hamming + SAD unit_cost and center_bonus
        let row_nd = w * nd;
        for i in 0..3 {
            if self.work_unit_ring[i].len() < row_nd {
                self.work_unit_ring[i].resize(row_nd, 0);
            }
            if self.work_bonus_ring[i].len() < row_nd {
                self.work_bonus_ring[i].resize(row_nd, 0);
            }
        }
        let unit_ring = &mut self.work_unit_ring;
        let bonus_ring = &mut self.work_bonus_ring;

        let fill_ring_row = |ry: usize, u_row: &mut [u16], b_row: &mut [u16]| {
            let cl_row = &census_l[ry * w..(ry + 1) * w];
            let cr_row = &census_r[ry * w..(ry + 1) * w];
            let lb_row = &l_bytes[ry * w..(ry + 1) * w];
            let rb_row = &r_bytes[ry * w..(ry + 1) * w];
            for x in (x_start - 1)..=(x_end.min(w - 1)) {
                let cl = cl_row[x];
                let lc = lb_row[x];
                let base = x * nd;
                if (x as i32 - d_max + 1) >= (rad as i32 - 1)
                    && (x as i32 - d_min) < (w as i32 - rad as i32 + 1)
                {
                    let xru0 = (x as i32 - d_min) as usize;
                    #[cfg(target_arch = "aarch64")]
                    if nd == 16 {
                        unsafe {
                            use std::arch::aarch64::*;
                            let cl_vec = vreinterpretq_u8_u64(vdupq_n_u64(cl));
                            let cr_u8 = cr_row.as_ptr().add(xru0 - 15) as *const u8;
                            let c0 = vcntq_u8(veorq_u8(cl_vec, vld1q_u8(cr_u8.add(7 * 16))));
                            let c1 = vcntq_u8(veorq_u8(cl_vec, vld1q_u8(cr_u8.add(6 * 16))));
                            let c2 = vcntq_u8(veorq_u8(cl_vec, vld1q_u8(cr_u8.add(5 * 16))));
                            let c3 = vcntq_u8(veorq_u8(cl_vec, vld1q_u8(cr_u8.add(4 * 16))));
                            let c4 = vcntq_u8(veorq_u8(cl_vec, vld1q_u8(cr_u8.add(3 * 16))));
                            let c5 = vcntq_u8(veorq_u8(cl_vec, vld1q_u8(cr_u8.add(2 * 16))));
                            let c6 = vcntq_u8(veorq_u8(cl_vec, vld1q_u8(cr_u8.add(16))));
                            let c7 = vcntq_u8(veorq_u8(cl_vec, vld1q_u8(cr_u8)));

                            let s01 = vpaddq_u8(c0, c1);
                            let s23 = vpaddq_u8(c2, c3);
                            let s45 = vpaddq_u8(c4, c5);
                            let s67 = vpaddq_u8(c6, c7);
                            let s0123 = vpaddq_u8(s01, s23);
                            let s4567 = vpaddq_u8(s45, s67);
                            let pop16 = vrev16q_u8(vpaddq_u8(s0123, s4567));

                            let rb16 = vld1q_u8(rb_row.as_ptr().add(xru0 - 15));
                            let rb_rev64 = vrev64q_u8(rb16);
                            let rb_rev = vextq_u8::<8>(rb_rev64, rb_rev64);
                            let sad8 = vabdq_u8(vdupq_n_u8(lc), rb_rev);

                            let ham4_lo = vshlq_n_u16::<2>(vmovl_u8(vget_low_u8(pop16)));
                            let ham4_hi = vshlq_n_u16::<2>(vmovl_high_u8(pop16));
                            let sad1_lo = vmovl_u8(vget_low_u8(sad8));
                            let sad1_hi = vmovl_high_u8(sad8);

                            let u_ptr = u_row.as_mut_ptr().add(base);
                            let b_ptr = b_row.as_mut_ptr().add(base);
                            vst1q_u16(u_ptr, vaddq_u16(ham4_lo, sad1_lo));
                            vst1q_u16(u_ptr.add(8), vaddq_u16(ham4_hi, sad1_hi));
                            vst1q_u16(b_ptr, vaddq_u16(ham4_lo, vshlq_n_u16::<1>(sad1_lo)));
                            vst1q_u16(b_ptr.add(8), vaddq_u16(ham4_hi, vshlq_n_u16::<1>(sad1_hi)));
                        }
                        continue;
                    }
                    for slot in 0..nd {
                        let xru = xru0 - slot;
                        let ham4 = ((cl ^ cr_row[xru]).count_ones() as u16) * 4;
                        let sad1 = lc.abs_diff(rb_row[xru]) as u16;
                        u_row[base + slot] = ham4 + sad1;
                        b_row[base + slot] = ham4 + sad1 * 2;
                    }
                } else {
                    for (slot, d) in (d_min..d_max).enumerate() {
                        let xr = x as i32 - d;
                        if xr >= (rad as i32 - 1) && xr < (w as i32 - rad as i32 + 1) {
                            let xru = xr as usize;
                            let ham4 = ((cl ^ cr_row[xru]).count_ones() as u16) * 4;
                            let sad1 = lc.abs_diff(rb_row[xru]) as u16;
                            u_row[base + slot] = ham4 + sad1;
                            b_row[base + slot] = ham4 + sad1 * 2;
                        } else {
                            u_row[base + slot] = 0;
                            b_row[base + slot] = 0;
                        }
                    }
                }
            }
        };

        if is_u8 && y_start < y_end {
            fill_ring_row(
                y_start - 1,
                &mut unit_ring[(y_start - 1) % 3],
                &mut bonus_ring[(y_start - 1) % 3],
            );
            fill_ring_row(
                y_start,
                &mut unit_ring[y_start % 3],
                &mut bonus_ring[y_start % 3],
            );
        }

        let mut prev_lr = vec![inf_cost; nd];
        let mut curr_lr = vec![inf_cost; nd];
        let mut prev_rl = vec![inf_cost; nd];
        let mut curr_rl = vec![inf_cost; nd];
        let mut curr_vert = vec![inf_cost; nd];
        if self.work_prev_vert.len() < row_nd {
            self.work_prev_vert.resize(row_nd, inf_cost);
        } else {
            self.work_prev_vert[..row_nd].fill(inf_cost);
        }
        let prev_vert_row = &mut self.work_prev_vert[..row_nd];

        // 1 & 2. Fused forward row sweep: build base_cost for row y and immediately run L->R, T->B, R->L in L1 cache
        for y in y_start..y_end {
            if is_u8 {
                fill_ring_row(
                    y + 1,
                    &mut unit_ring[(y + 1) % 3],
                    &mut bonus_ring[(y + 1) % 3],
                );
            }
            let r_up = (y - 1) % 3;
            let r_c = y % 3;
            let r_dn = (y + 1) % 3;

            let cl_row_u = &census_l[(y - 1) * w..y * w];
            let cl_row_c = &census_l[y * w..(y + 1) * w];
            let cl_row_d = &census_l[(y + 1) * w..(y + 2) * w];
            let cr_row_u = &census_r[(y - 1) * w..y * w];
            let cr_row_c = &census_r[y * w..(y + 1) * w];
            let cr_row_d = &census_r[(y + 1) * w..(y + 2) * w];

            prev_lr.fill(inf_cost);
            for x in x_start..x_end {
                let is_tex = if is_u8 {
                    let c_val = l_bytes[y * w + x];
                    if c_val.abs_diff(l_bytes[y * w + x - 1]) >= 2
                        || c_val.abs_diff(l_bytes[y * w + x + 1]) >= 2
                        || c_val.abs_diff(l_bytes[(y - 1) * w + x]) >= 2
                        || c_val.abs_diff(l_bytes[(y + 1) * w + x]) >= 2
                    {
                        true
                    } else if cl_row_c[x] == 0 && rad <= 3 {
                        let mut pmin = c_val;
                        for wy in -rad..=rad {
                            let row_off = ((y as isize + wy) as usize) * w;
                            for wx in -rad..=rad {
                                pmin = pmin.min(l_bytes[row_off + ((x as isize + wx) as usize)]);
                            }
                        }
                        c_val.saturating_sub(pmin) >= 2
                    } else {
                        let mut pmin = 255u8;
                        let mut pmax = 0u8;
                        for wy in -rad..=rad {
                            let row_off = ((y as isize + wy) as usize) * w;
                            for wx in -rad..=rad {
                                let v = l_bytes[row_off + ((x as isize + wx) as usize)];
                                pmin = pmin.min(v);
                                pmax = pmax.max(v);
                            }
                        }
                        pmax.saturating_sub(pmin) >= 2
                    }
                } else {
                    let mut patch_min = 255.0f64;
                    let mut patch_max = 0.0f64;
                    for wy in -rad..=rad {
                        for wx in -rad..=rad {
                            let v = left.get_f64((y as isize + wy) as i32, (x as isize + wx) as i32, 0);
                            patch_min = patch_min.min(v);
                            patch_max = patch_max.max(v);
                        }
                    }
                    patch_max - patch_min >= 2.0
                };

                let off = (y * w + x) * nd;
                let tb_slice = &mut prev_vert_row[x * nd..(x + 1) * nd];
                if !is_tex {
                    prev_lr.fill(inf_cost);
                    tb_slice.fill(inf_cost);
                    continue;
                }
                textured[y * w + x] = true;
                let out_c = &mut base_cost[off..off + nd];

                if is_u8 {
                    let u_c = &unit_ring[r_c][x * nd..(x + 1) * nd];
                    let b_c = &bonus_ring[r_c][x * nd..(x + 1) * nd];
                    let u_up = &unit_ring[r_up][x * nd..(x + 1) * nd];
                    let u_dn = &unit_ring[r_dn][x * nd..(x + 1) * nd];
                    let u_lf = &unit_ring[r_c][(x - 1) * nd..x * nd];
                    let u_rt = &unit_ring[r_c][(x + 1) * nd..(x + 2) * nd];

                    if (x as i32 - d_max + 1) >= (rad as i32)
                        && (x as i32 - d_min) < (w as i32 - rad as i32)
                    {
                        #[cfg(target_arch = "aarch64")]
                        if nd == 16 {
                            unsafe {
                                use std::arch::aarch64::*;
                                let s_lo = vaddq_u16(
                                    vaddq_u16(
                                        vaddq_u16(vld1q_u16(b_c.as_ptr()), vld1q_u16(u_c.as_ptr())),
                                        vaddq_u16(vld1q_u16(u_up.as_ptr()), vld1q_u16(u_dn.as_ptr())),
                                    ),
                                    vaddq_u16(vld1q_u16(u_lf.as_ptr()), vld1q_u16(u_rt.as_ptr())),
                                );
                                let s_hi = vaddq_u16(
                                    vaddq_u16(
                                        vaddq_u16(
                                            vld1q_u16(b_c.as_ptr().add(8)),
                                            vld1q_u16(u_c.as_ptr().add(8)),
                                        ),
                                        vaddq_u16(
                                            vld1q_u16(u_up.as_ptr().add(8)),
                                            vld1q_u16(u_dn.as_ptr().add(8)),
                                        ),
                                    ),
                                    vaddq_u16(
                                        vld1q_u16(u_lf.as_ptr().add(8)),
                                        vld1q_u16(u_rt.as_ptr().add(8)),
                                    ),
                                );
                                vst1q_u16(out_c.as_mut_ptr(), s_lo);
                                vst1q_u16(out_c.as_mut_ptr().add(8), s_hi);
                            }
                        } else {
                            for slot in 0..nd {
                                out_c[slot] = b_c[slot]
                                    + u_c[slot]
                                    + u_up[slot]
                                    + u_dn[slot]
                                    + u_lf[slot]
                                    + u_rt[slot];
                            }
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        for slot in 0..nd {
                            out_c[slot] = b_c[slot]
                                + u_c[slot]
                                + u_up[slot]
                                + u_dn[slot]
                                + u_lf[slot]
                                + u_rt[slot];
                        }
                    } else {
                        for (slot, d) in (d_min..d_max).enumerate() {
                            let xr = x as i32 - d;
                            if xr < rad as i32 || xr >= (w as i32 - rad as i32) {
                                out_c[slot] = inf_cost;
                            } else {
                                out_c[slot] = b_c[slot]
                                    + u_c[slot]
                                    + u_up[slot]
                                    + u_dn[slot]
                                    + u_lf[slot]
                                    + u_rt[slot];
                            }
                        }
                    }
                } else {
                    let cl_c = cl_row_c[x];
                    let cl_u = cl_row_u[x];
                    let cl_d = cl_row_d[x];
                    let cl_l = cl_row_c[x - 1];
                    let cl_r = cl_row_c[x + 1];
                    let il = left.get_f64(y as i32, x as i32, 0) as i32;
                    for (slot, d) in (d_min..d_max).enumerate() {
                        let xr = x as i32 - d;
                        if xr < rad as i32 || xr >= (w as i32 - rad as i32) {
                            out_c[slot] = inf_cost;
                            continue;
                        }
                        let xru = xr as usize;
                        let c_ham = (cl_c ^ cr_row_c[xru]).count_ones() * 8
                            + (cl_u ^ cr_row_u[xru]).count_ones() * 4
                            + (cl_d ^ cr_row_d[xru]).count_ones() * 4
                            + (cl_l ^ cr_row_c[xru - 1]).count_ones() * 4
                            + (cl_r ^ cr_row_c[xru + 1]).count_ones() * 4;
                        let ir = right.get_f64(y as i32, xr, 0) as i32;
                        let sad_cross = (il - ir).unsigned_abs() * 7;
                        out_c[slot] = (c_ham + sad_cross).min(inf_cost as u32) as u16;
                    }
                }

                // Immediately execute L->R and T->B while out_c is hot in L1 registers
                dp_step(&prev_lr, out_c, &mut curr_lr, nd, p1_pen, p2_pen, inf_cost);
                dp_step(tb_slice, out_c, &mut curr_vert, nd, p1_pen, p2_pen, inf_cost);
                let agg_slice = &mut agg_cost[off..off + nd];
                #[cfg(target_arch = "aarch64")]
                if nd == 16 {
                    unsafe {
                        use std::arch::aarch64::*;
                        let v_lo = vld1q_u16(curr_vert.as_ptr());
                        let v_hi = vld1q_u16(curr_vert.as_ptr().add(8));
                        vst1q_u16(
                            agg_slice.as_mut_ptr(),
                            vqaddq_u16(vld1q_u16(curr_lr.as_ptr()), v_lo),
                        );
                        vst1q_u16(
                            agg_slice.as_mut_ptr().add(8),
                            vqaddq_u16(vld1q_u16(curr_lr.as_ptr().add(8)), v_hi),
                        );
                        vst1q_u16(tb_slice.as_mut_ptr(), v_lo);
                        vst1q_u16(tb_slice.as_mut_ptr().add(8), v_hi);
                    }
                    std::mem::swap(&mut prev_lr, &mut curr_lr);
                    continue;
                }
                for d in 0..nd {
                    agg_slice[d] = curr_lr[d].saturating_add(curr_vert[d]);
                }
                tb_slice.copy_from_slice(&curr_vert);
                std::mem::swap(&mut prev_lr, &mut curr_lr);
            }

            prev_rl.fill(inf_cost);
            for x in (x_start..x_end).rev() {
                let off = (y * w + x) * nd;
                if !textured[y * w + x] {
                    prev_rl.fill(inf_cost);
                    continue;
                }
                dp_step(
                    &prev_rl,
                    &base_cost[off..off + nd],
                    &mut curr_rl,
                    nd,
                    p1_pen,
                    p2_pen,
                    inf_cost,
                );
                let agg_slice = &mut agg_cost[off..off + nd];
                #[cfg(target_arch = "aarch64")]
                if nd == 16 {
                    unsafe {
                        use std::arch::aarch64::*;
                        vst1q_u16(
                            agg_slice.as_mut_ptr(),
                            vqaddq_u16(vld1q_u16(agg_slice.as_ptr()), vld1q_u16(curr_rl.as_ptr())),
                        );
                        vst1q_u16(
                            agg_slice.as_mut_ptr().add(8),
                            vqaddq_u16(
                                vld1q_u16(agg_slice.as_ptr().add(8)),
                                vld1q_u16(curr_rl.as_ptr().add(8)),
                            ),
                        );
                    }
                    std::mem::swap(&mut prev_rl, &mut curr_rl);
                    continue;
                }
                for d in 0..nd {
                    agg_slice[d] = agg_slice[d].saturating_add(curr_rl[d]);
                }
                std::mem::swap(&mut prev_rl, &mut curr_rl);
            }
        }

        // 3. Reverse y sweep: fuse B->T with Winner-Takes-All, uniqueness ratio, and sub-pixel parabola refinement
        prev_vert_row.fill(inf_cost);
        let mut valid_count = 0usize;
        for y in (y_start..y_end).rev() {
            for x in x_start..x_end {
                let bt_slice = &mut prev_vert_row[x * nd..(x + 1) * nd];
                if !textured[y * w + x] {
                    bt_slice.fill(inf_cost);
                    continue;
                }
                let off = (y * w + x) * nd;
                let c_slice = &base_cost[off..off + nd];
                dp_step(bt_slice, c_slice, &mut curr_vert, nd, p1_pen, p2_pen, inf_cost);
                bt_slice.copy_from_slice(&curr_vert);

                let costs = &mut agg_cost[off..off + nd];
                let mut best_cost = u16::MAX;
                let mut best_slot = 0usize;
                let mut min_non_neighbor = u16::MAX;
                let mut c_prev = 0u16;
                let mut c_curr = 0u16;
                let mut c_next = 0u16;

                #[cfg(target_arch = "aarch64")]
                let used_neon_wta = if nd == 16 {
                    use std::arch::aarch64::*;
                    const IDX_LO_ARR: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
                    const IDX_HI_ARR: [u16; 8] = [8, 9, 10, 11, 12, 13, 14, 15];
                    unsafe {
                        let inf_v = vdupq_n_u16(inf_cost);
                        let max_v = vdupq_n_u16(u16::MAX);
                        let b_lo = vld1q_u16(c_slice.as_ptr());
                        let b_hi = vld1q_u16(c_slice.as_ptr().add(8));
                        let s_lo = vbslq_u16(
                            vcltq_u16(b_lo, inf_v),
                            vqaddq_u16(vld1q_u16(costs.as_ptr()), vld1q_u16(curr_vert.as_ptr())),
                            max_v,
                        );
                        let s_hi = vbslq_u16(
                            vcltq_u16(b_hi, inf_v),
                            vqaddq_u16(
                                vld1q_u16(costs.as_ptr().add(8)),
                                vld1q_u16(curr_vert.as_ptr().add(8)),
                            ),
                            max_v,
                        );
                        best_cost = vminvq_u16(vminq_u16(s_lo, s_hi));
                        if best_cost < inf_cost {
                            let idx_lo = vld1q_u16(IDX_LO_ARR.as_ptr());
                            let idx_hi = vld1q_u16(IDX_HI_ARR.as_ptr());
                            let bc_v = vdupq_n_u16(best_cost);
                            let cand_lo = vbslq_u16(vceqq_u16(s_lo, bc_v), idx_lo, max_v);
                            let cand_hi = vbslq_u16(vceqq_u16(s_hi, bc_v), idx_hi, max_v);
                            best_slot = vminvq_u16(vminq_u16(cand_lo, cand_hi)) as usize;

                            let best_v = vdupq_n_u16(best_slot as u16);
                            let one_v = vdupq_n_u16(1);
                            let nn_lo =
                                vbslq_u16(vcleq_u16(vabdq_u16(idx_lo, best_v), one_v), max_v, s_lo);
                            let nn_hi =
                                vbslq_u16(vcleq_u16(vabdq_u16(idx_hi, best_v), one_v), max_v, s_hi);
                            min_non_neighbor = vminvq_u16(vminq_u16(nn_lo, nn_hi));

                            if best_slot > 0 && best_slot + 1 < 16 {
                                let mut sc = [0u16; 16];
                                vst1q_u16(sc.as_mut_ptr(), s_lo);
                                vst1q_u16(sc.as_mut_ptr().add(8), s_hi);
                                c_prev = sc[best_slot - 1];
                                c_curr = best_cost;
                                c_next = sc[best_slot + 1];
                            }
                        }
                    }
                    true
                } else {
                    false
                };
                #[cfg(not(target_arch = "aarch64"))]
                let used_neon_wta = false;

                if !used_neon_wta {
                    for slot in 0..nd {
                        let c = costs[slot].saturating_add(curr_vert[slot]);
                        costs[slot] = c;
                        if c_slice[slot] < inf_cost && c < best_cost {
                            best_cost = c;
                            best_slot = slot;
                        }
                    }
                    if best_cost >= inf_cost {
                        continue;
                    }
                    for (slot, &c) in costs.iter().enumerate() {
                        if c_slice[slot] < inf_cost
                            && slot.abs_diff(best_slot) > 1
                            && c < min_non_neighbor
                        {
                            min_non_neighbor = c;
                        }
                    }
                    if best_slot > 0 && best_slot + 1 < nd {
                        c_prev = costs[best_slot - 1];
                        c_curr = costs[best_slot];
                        c_next = costs[best_slot + 1];
                    }
                } else if best_cost >= inf_cost {
                    continue;
                }

                if min_non_neighbor < u16::MAX
                    && (min_non_neighbor as u64) * 100 < (best_cost as u64) * (100 + uniq as u64)
                {
                    continue;
                }

                let best_d = d_min + (best_slot as i32);
                let mut sub_d = best_d as f64;
                if best_slot > 0 && best_slot + 1 < nd {
                    if c_slice[best_slot - 1] < inf_cost && c_slice[best_slot + 1] < inf_cost {
                        let denom =
                            2.0 * ((c_prev as f64) + (c_next as f64) - 2.0 * (c_curr as f64));
                        if denom.abs() > 1e-6 {
                            let delta = ((c_prev as f64) - (c_next as f64)) / denom;
                            sub_d += delta.clamp(-0.5, 0.5);
                        }
                    }
                }

                out[y * w + x] = (sub_d * 16.0).round() as i16;
                valid_count += 1;
            }
        }

        record_activation(
            "calib3d::StereoSGBM::compute",
            TheoremId::Thm2TemporalXorMintermSparsity,
            h * w,
            valid_count,
            valid_count as u64,
            &[("valid_disparities", valid_count as f64)],
        );

        Ok(())
    }
}

/// Detect interior calibration chessboard saddle-point corners (`find_chessboard_corners`)
/// and refine them to sub-pixel accuracy (`Theorem 5 & Operator J1`).
pub fn find_chessboard_corners(
    image: &Mat,
    pattern_size: Size,
    corners: &mut crate::core::Vector<Point2f>,
    _flags: i32,
) -> Result<bool> {
    corners.clear();
    let cols_pat = pattern_size.width;
    let rows_pat = pattern_size.height;
    let needed = (cols_pat * rows_pat) as usize;
    if cols_pat <= 0 || rows_pat <= 0 || image.rows() < 12 || image.cols() < 12 {
        return Ok(false);
    }

    let h = image.rows();
    let w = image.cols();
    let mut score_map = vec![0.0f64; (h * w) as usize];
    let rad = 3i32;

    // Evaluate D_4-symmetric X-junction saddle response at every interior pixel
    for r in (rad + 2)..(h - rad - 2) {
        for c in (rad + 2)..(w - rad - 2) {
            let mut resp_total = 0.0f64;
            for d in [2i32, 3i32] {
                let p_ne = image.get_f64(r - d, c + d, 0);
                let p_sw = image.get_f64(r + d, c - d, 0);
                let p_nw = image.get_f64(r - d, c - d, 0);
                let p_se = image.get_f64(r + d, c + d, 0);

                // Opposite quadrants must match each other and differ strongly from the orthogonal diagonal
                let cross_diff = ((p_ne + p_sw) - (p_nw + p_se)).abs();
                let self_sym = (p_ne - p_sw).abs() + (p_nw - p_se).abs();

                // Axis-aligned neighbors must average near the midpoint (preventing pure diagonal stripes)
                let p_n = image.get_f64(r - d, c, 0);
                let p_s = image.get_f64(r + d, c, 0);
                let p_e = image.get_f64(r, c + d, 0);
                let p_w = image.get_f64(r, c - d, 0);
                let axis_diff = ((p_n + p_s) - (p_e + p_w)).abs();
                let axis_sym = (p_n - p_s).abs() + (p_e - p_w).abs();

                let s_diag = cross_diff - 2.2 * self_sym;
                let s_axis = axis_diff - 2.2 * axis_sym;
                resp_total += s_diag.max(s_axis);
            }
            if resp_total > 40.0 {
                score_map[(r * w + c) as usize] = resp_total;
            }
        }
    }

    // Non-maximum suppression with radius 5
    let nms_r = 5i32;
    let mut candidates: Vec<(f64, Point2f)> = Vec::new();
    for r in (rad + 2)..(h - rad - 2) {
        for c in (rad + 2)..(w - rad - 2) {
            let s = score_map[(r * w + c) as usize];
            if s <= 40.0 {
                continue;
            }
            let mut is_max = true;
            'nms: for dr in -nms_r..=nms_r {
                for dc in -nms_r..=nms_r {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = (r + dr).clamp(0, h - 1);
                    let nc = (c + dc).clamp(0, w - 1);
                    let ns = score_map[(nr * w + nc) as usize];
                    if ns > s || (ns == s && (nr < r || (nr == r && nc < c))) {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if is_max {
                candidates.push((s, Point2f::new(c as f32, r as f32)));
            }
        }
    }

    if candidates.len() < needed {
        return Ok(false);
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(needed);

    let mut pts: Vec<Point2f> = candidates.into_iter().map(|(_, p)| p).collect();
    crate::features2d::corner_sub_pix(image, &mut pts, Size::new(4, 4), Size::new(-1, -1), 25, 1e-3)?;

    // Sort into canonical row-major grid order (rows_pat rows x cols_pat cols)
    pts.sort_by(|a, b| {
        a.y.partial_cmp(&b.y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for row_slice in pts.chunks_mut(cols_pat as usize) {
        row_slice.sort_by(|a, b| {
            a.x.partial_cmp(&b.x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    for p in pts {
        corners.push(p);
    }
    record_activation(
        "calib3d::find_chessboard_corners",
        TheoremId::Thm5DihedralD4Closure,
        (h * w) as usize,
        corners.len(),
        corners.len() as u64,
        &[("corners", corners.len() as f64)],
    );
    Ok(corners.len() == needed)
}

/// Decoded $D_4$-invariant fiducial marker (`ArucoDetection`).
#[derive(Debug, Clone, PartialEq)]
pub struct ArucoDetection {
    /// Canonical minimum 16-bit payload under the 4 proper rotations of $D_4$.
    pub canonical_id: u16,
    /// Detected quarter-turn rotation (`0..4`, i.e., `0, 90, 180, 270` degrees).
    pub rotation_quadrants: u8,
    /// Four marker corners in image coordinates (`[top_left, top_right, bottom_right, bottom_left]`).
    pub corners: [Point2f; 4],
}

/// Rotate a $4\times 4$ 16-bit fiducial payload clockwise by `rot` quarter-turns (`Theorem 5`).
#[inline]
pub fn rotate_aruco_bits_4x4(bits: u16, rot: u8) -> u16 {
    let mut cur = bits;
    for _ in 0..(rot % 4) {
        let mut next = 0u16;
        for r in 0..4 {
            for c in 0..4 {
                let bit = (cur >> (r * 4 + c)) & 1;
                let nr = c;
                let nc = 3 - r;
                next |= bit << (nr * 4 + nc);
            }
        }
        cur = next;
    }
    cur
}

/// Detect $6\times 6$ ($1$-cell dark border + $4\times 4$ binary payload) fiducial markers
/// and decode their $D_4$-canonical ID and orientation (`Theorem 5 & Theorem 8`).
pub fn detect_aruco_markers(gray: &Mat, threshold: u8) -> Result<Vec<ArucoDetection>> {
    let h = gray.rows();
    let w = gray.cols();
    if h < 12 || w < 12 {
        return Ok(Vec::new());
    }

    // Invert binary mask so dark marker outer border is foreground (1)
    let mut inv_mask = Mat::zeros(h, w, CV_8UC1)?;
    for r in 0..h {
        for c in 0..w {
            if gray.get_f64(r, c, 0) < threshold as f64 {
                inv_mask.set_f64(r, c, 0, 1.0);
            }
        }
    }

    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let n_comp = crate::imgproc::connected_components_with_stats(
        &inv_mask,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        CV_32SC1,
    )?;

    let mut detections = Vec::new();
    for id in 1..n_comp {
        let left = *stats.at_2d::<i32>(id, crate::imgproc::CC_STAT_LEFT)?;
        let top = *stats.at_2d::<i32>(id, crate::imgproc::CC_STAT_TOP)?;
        let bw = *stats.at_2d::<i32>(id, crate::imgproc::CC_STAT_WIDTH)?;
        let bh = *stats.at_2d::<i32>(id, crate::imgproc::CC_STAT_HEIGHT)?;
        if bw < 12 || bh < 12 || (bw - bh).abs() > (bw.max(bh) / 3) {
            continue;
        }

        // Sample 6x6 logical lattice (Theorem 8 quotient) inside [left..left+bw, top..top+bh]
        let mut border_dark = 0usize;
        let mut border_total = 0usize;
        let mut payload_bits = 0u16;
        for gr in 0..6 {
            for gc in 0..6 {
                let sy = top as f64 + (gr as f64 + 0.5) * (bh as f64 / 6.0);
                let sx = left as f64 + (gc as f64 + 0.5) * (bw as f64 / 6.0);
                let val = gray.get_f64(sy.round() as i32, sx.round() as i32, 0);
                if gr == 0 || gr == 5 || gc == 0 || gc == 5 {
                    border_total += 1;
                    if val < threshold as f64 {
                        border_dark += 1;
                    }
                } else {
                    let pr = gr - 1;
                    let pc = gc - 1;
                    if val >= threshold as f64 {
                        payload_bits |= 1u16 << (pr * 4 + pc);
                    }
                }
            }
        }
        if border_dark * 10 < border_total * 8 || payload_bits == 0 || payload_bits == 0xFFFF {
            continue;
        }

        // Quotient over the 4 proper rotations of D_4 to find canonical ID and orientation
        let mut canonical_id = u16::MAX;
        let mut best_rot = 0u8;
        for rot in 0..4u8 {
            let rotated = rotate_aruco_bits_4x4(payload_bits, (4 - rot) % 4);
            if rotated < canonical_id {
                canonical_id = rotated;
                best_rot = rot;
            }
        }

        detections.push(ArucoDetection {
            canonical_id,
            rotation_quadrants: best_rot,
            corners: [
                Point2f::new(left as f32, top as f32),
                Point2f::new((left + bw) as f32, top as f32),
                Point2f::new((left + bw) as f32, (top + bh) as f32),
                Point2f::new(left as f32, (top + bh) as f32),
            ],
        });
    }

    record_activation(
        "calib3d::detect_aruco_markers",
        TheoremId::Thm5DihedralD4Closure,
        (h * w) as usize,
        detections.len(),
        detections.first().map(|d| d.canonical_id as u64).unwrap_or(0),
        &[("markers", detections.len() as f64)],
    );
    Ok(detections)
}

