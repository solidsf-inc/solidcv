//! Pure-Rust video analysis, pyramidal optical flow, and worldtube tracking (`video`).
//!
//! Accelerates pyramidal Lucas-Kanade (`calc_optical_flow_pyr_lk`) with:
//! - **Theorem 2**: Pixelean Temporal XOR Minterm Carrier (`\Delta_t = F_t \oplus F_{t-1}`)
//!   to bypass zero-change static patches at the finest pyramid level.
//! - **Theorem 11 & 20**: Schur-Feshbach dyadic wavelet pyramid downfolding (`1 - 2^{-D}`).
//! - **Theorem 18**: Causal Light-Cone & CFL Horizon Pruning.

use crate::core::{
    Error, InputArray, Mat, MatTraitConst, OutputF32, OutputPoints2f, OutputStatusU8, Point2f,
    Result, Size, TermCriteria, TermCriteria_Type,
};
use crate::imgproc::pyr_down_schur_feshbach;
use crate::mechinterp::{current_ablation_config, record_activation, TheoremId};
use crate::pixelean::XorMintermCarrier;
use crate::worldtube::CflLightConePruner;

#[inline]
fn sample_bilinear_gray(img: &Mat, x: f64, y: f64) -> f64 {
    let w = img.cols();
    let h = img.rows();
    if w <= 0 || h <= 0 {
        return 0.0;
    }
    let xc = x.clamp(0.0, (w - 1) as f64);
    let yc = y.clamp(0.0, (h - 1) as f64);
    let x0 = xc.floor() as i32;
    let y0 = yc.floor() as i32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = xc - (x0 as f64);
    let fy = yc - (y0 as f64);

    let v00 = img.get_f64(y0, x0, 0);
    let v10 = img.get_f64(y0, x1, 0);
    let v01 = img.get_f64(y1, x0, 0);
    let v11 = img.get_f64(y1, x1, 0);

    (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) + fy * ((1.0 - fx) * v01 + fx * v11)
}

/// Pyramidal Lucas-Kanade optical flow (`calc_optical_flow_pyr_lk`) accelerated by
/// Pixelean XOR Minterm Carrier (`Theorem 2`), Schur-Feshbach Wavelet Pyramids (`Theorem 20`),
/// and Causal Light-Cone CFL Pruning (`Theorem 18`).
#[allow(clippy::too_many_arguments)]
pub fn calc_optical_flow_pyr_lk<P, N, S, E>(
    prev_img: &Mat,
    next_img: &Mat,
    prev_pts: &P,
    next_pts: &mut N,
    status: &mut S,
    err: &mut E,
    win_size: Size,
    max_level: i32,
    criteria: TermCriteria,
    _flags: i32,
    min_eig_threshold: f64,
) -> Result<()>
where
    P: InputArray,
    N: OutputPoints2f,
    S: OutputStatusU8,
    E: OutputF32,
{
    if prev_img.is_empty() || next_img.is_empty() {
        return Err(Error::new(-215, "empty frame in calc_optical_flow_pyr_lk"));
    }
    if prev_img.rows() != next_img.rows() || prev_img.cols() != next_img.cols() {
        return Err(Error::new(
            -215,
            "frame size mismatch in calc_optical_flow_pyr_lk",
        ));
    }
    let pts0 = prev_pts.as_points_2f()?;
    let n_pts = pts0.len();
    if n_pts == 0 {
        next_pts.write_points_2f(&[])?;
        status.write_status_u8(&[])?;
        err.write_f32(&[])?;
        return Ok(());
    }

    let levels = max_level.clamp(0, 6) as usize;
    let mut prev_pyr = Vec::with_capacity(levels + 1);
    let mut next_pyr = Vec::with_capacity(levels + 1);
    prev_pyr.push(prev_img.clone());
    next_pyr.push(next_img.clone());
    for _ in 0..levels {
        let p_down = pyr_down_schur_feshbach(prev_pyr.last().unwrap())?;
        let n_down = pyr_down_schur_feshbach(next_pyr.last().unwrap())?;
        prev_pyr.push(p_down);
        next_pyr.push(n_down);
    }

    let ablation = current_ablation_config();
    let xor_carrier = XorMintermCarrier::from_gray_pair(prev_img, next_img, 0);
    let cfl = CflLightConePruner::new(
        2,
        win_size.width.max(win_size.height) * (1 << levels),
        (prev_img.rows() as usize) * (prev_img.cols() as usize),
    );

    let rx = (win_size.width.max(3) / 2) as i32;
    let ry = (win_size.height.max(3) / 2) as i32;
    let win_area = ((2 * rx + 1) * (2 * ry + 1)) as f64;
    let max_iter = if (criteria.typ & (TermCriteria_Type::COUNT as i32)) != 0 {
        criteria.max_count.clamp(1, 100) as usize
    } else {
        30
    };
    let eps = if (criteria.typ & (TermCriteria_Type::EPS as i32)) != 0 {
        criteria.epsilon.clamp(1e-5, 1.0)
    } else {
        0.01
    };

    let mut out_pts = vec![Point2f::new(0.0, 0.0); n_pts];
    let mut out_status = vec![1u8; n_pts];
    let mut out_err = vec![0.0f32; n_pts];
    let mut xor_skipped = 0usize;

    for (idx, &p0) in pts0.iter().enumerate() {
        // Fast Pixelean XOR minterm check: if the entire window around p0 is bit-identical
        // between prev_img and next_img, the optical flow at p0 is identically (0, 0) with 0 error!
        if !ablation.disable_xor_carrier
            && !xor_carrier.patch_has_activity(
                p0.x.round() as i32,
                p0.y.round() as i32,
                rx.max(ry) + 2,
            )
        {
            // Verify local structure tensor has sufficient texture eigenvalue
            let mut gxx = 0.0;
            let mut gyy = 0.0;
            let mut gxy = 0.0;
            for wy in -ry..=ry {
                for wx in -rx..=rx {
                    let px = p0.x as f64 + wx as f64;
                    let py = p0.y as f64 + wy as f64;
                    let ix =
                        0.5 * (sample_bilinear_gray(prev_img, px + 1.0, py)
                            - sample_bilinear_gray(prev_img, px - 1.0, py));
                    let iy =
                        0.5 * (sample_bilinear_gray(prev_img, px, py + 1.0)
                            - sample_bilinear_gray(prev_img, px, py - 1.0));
                    gxx += ix * ix;
                    gyy += iy * iy;
                    gxy += ix * iy;
                }
            }
            let min_eig = 0.5
                * ((gxx + gyy) - ((gxx - gyy) * (gxx - gyy) + 4.0 * gxy * gxy).sqrt())
                / (win_area * 255.0 * 255.0);
            if min_eig >= min_eig_threshold {
                out_pts[idx] = p0;
                out_status[idx] = 1;
                out_err[idx] = 0.0;
                xor_skipped += 1;
                continue;
            }
        }

        let mut dx = 0.0f64;
        let mut dy = 0.0f64;
        let mut tracked = true;

        for lvl in (0..=levels).rev() {
            let scale = 1.0 / ((1usize << lvl) as f64);
            let px0 = (p0.x as f64) * scale;
            let py0 = (p0.y as f64) * scale;
            dx *= 2.0;
            dy *= 2.0;

            let img_a = &prev_pyr[lvl];
            let img_b = &next_pyr[lvl];
            let w_lvl = img_a.cols() as f64;
            let h_lvl = img_a.rows() as f64;

            if px0 < 1.0 || py0 < 1.0 || px0 >= w_lvl - 1.0 || py0 >= h_lvl - 1.0 {
                if lvl == 0 {
                    tracked = false;
                }
                continue;
            }

            // Precompute template patch intensities & spatial gradients on img_a
            let win_w = (2 * rx + 1) as usize;
            let win_h = (2 * ry + 1) as usize;
            let mut tpl_i = Vec::with_capacity(win_w * win_h);
            let mut tpl_ix = Vec::with_capacity(win_w * win_h);
            let mut tpl_iy = Vec::with_capacity(win_w * win_h);
            let mut gxx = 0.0f64;
            let mut gyy = 0.0f64;
            let mut gxy = 0.0f64;

            for wy in -ry..=ry {
                for wx in -rx..=rx {
                    let x = px0 + (wx as f64);
                    let y = py0 + (wy as f64);
                    let i_val = sample_bilinear_gray(img_a, x, y);
                    let ix = 0.5
                        * (sample_bilinear_gray(img_a, x + 1.0, y)
                            - sample_bilinear_gray(img_a, x - 1.0, y));
                    let iy = 0.5
                        * (sample_bilinear_gray(img_a, x, y + 1.0)
                            - sample_bilinear_gray(img_a, x, y - 1.0));
                    tpl_i.push(i_val);
                    tpl_ix.push(ix);
                    tpl_iy.push(iy);
                    gxx += ix * ix;
                    gyy += iy * iy;
                    gxy += ix * iy;
                }
            }

            let det = gxx * gyy - gxy * gxy;
            let min_eig = 0.5
                * ((gxx + gyy) - ((gxx - gyy) * (gxx - gyy) + 4.0 * gxy * gxy).sqrt())
                / (win_area * 255.0 * 255.0);
            if det <= 1e-7 || min_eig < min_eig_threshold {
                if lvl == 0 {
                    tracked = false;
                }
                continue;
            }
            let inv_det = 1.0 / det;

            for _iter in 0..max_iter {
                let cur_x = px0 + dx;
                let cur_y = py0 + dy;
                if cur_x < -rx as f64
                    || cur_y < -ry as f64
                    || cur_x >= w_lvl + rx as f64
                    || cur_y >= h_lvl + ry as f64
                {
                    tracked = false;
                    break;
                }

                let mut bx = 0.0f64;
                let mut by = 0.0f64;
                let mut k = 0usize;
                for wy in -ry..=ry {
                    for wx in -rx..=rx {
                        let j_val = sample_bilinear_gray(
                            img_b,
                            cur_x + (wx as f64),
                            cur_y + (wy as f64),
                        );
                        let diff = tpl_i[k] - j_val;
                        bx += diff * tpl_ix[k];
                        by += diff * tpl_iy[k];
                        k += 1;
                    }
                }

                let step_x = inv_det * (gyy * bx - gxy * by);
                let step_y = inv_det * (-gxy * bx + gxx * by);
                dx += step_x;
                dy += step_y;

                if step_x * step_x + step_y * step_y < eps * eps {
                    break;
                }
            }
        }

        let final_x = (p0.x as f64) + dx;
        let final_y = (p0.y as f64) + dy;
        let w0 = prev_img.cols() as f64;
        let h0 = prev_img.rows() as f64;

        if !tracked
            || !final_x.is_finite()
            || !final_y.is_finite()
            || final_x < 0.0
            || final_y < 0.0
            || final_x >= w0
            || final_y >= h0
            || !cfl.within_horizon_2d(dx, dy)
        {
            out_pts[idx] = Point2f::new(final_x as f32, final_y as f32);
            out_status[idx] = 0;
            out_err[idx] = 1000.0;
            continue;
        }

        // Compute final mean L1 patch residual at level 0
        let mut l1_sum = 0.0f64;
        for wy in -ry..=ry {
            for wx in -rx..=rx {
                let ia = sample_bilinear_gray(prev_img, p0.x as f64 + wx as f64, p0.y as f64 + wy as f64);
                let ib = sample_bilinear_gray(next_img, final_x + wx as f64, final_y + wy as f64);
                l1_sum += (ia - ib).abs();
            }
        }
        let patch_err = (l1_sum / win_area) as f32;
        out_pts[idx] = Point2f::new(final_x as f32, final_y as f32);
        out_status[idx] = 1;
        out_err[idx] = patch_err;
    }

    record_activation(
        "video::calc_optical_flow_pyr_lk",
        TheoremId::Thm2TemporalXorMintermSparsity,
        n_pts,
        n_pts.saturating_sub(xor_skipped),
        (n_pts - xor_skipped) as u64,
        &[
            ("total_points", n_pts as f64),
            ("xor_skipped", xor_skipped as f64),
        ],
    );

    next_pts.write_points_2f(&out_pts)?;
    status.write_status_u8(&out_status)?;
    err.write_f32(&out_err)?;
    Ok(())
}

pub use crate::realitycomb::calc_optical_flow_farneback;

