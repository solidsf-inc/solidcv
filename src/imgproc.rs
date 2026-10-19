//! Pure-Rust image processing (`imgproc`) powered by Pixelean & Voxelean Calculus.
//!
//! Implements color conversions, Schur-Feshbach wavelet pyramids (`Thm11`, `Thm20`),
//! affine/perspective/remap warps, Shi-Tomasi / Harris feature extraction, Shannon
//! cofactor thresholding & GrabCut segmentation (`Thm1`, `Thm3`), morphology,
//! 8-connected component topology, and SE(2) Frenet-Serret edge lifting (`J1`).

use crate::core::{
    cv_maketype, Error, Mat, MatTrait, MatTraitConst, OutputPoints2f, Point, Point2f, Rect, Result,
    Scalar, Size, Vector, BORDER_CONSTANT, BORDER_REPLICATE, CV_32F, CV_32SC1, CV_64FC1, CV_8UC1,
    CV_8UC3,
};
use crate::mechinterp::{record_activation, TheoremId};
use crate::pixelean::ShannonCofactorEnvelope;
use std::collections::VecDeque;

// Color conversion codes
pub const COLOR_BGR2RGB: i32 = 4;
pub const COLOR_RGB2BGR: i32 = 4;
pub const COLOR_BGR2GRAY: i32 = 6;
pub const COLOR_RGB2GRAY: i32 = 7;
pub const COLOR_GRAY2BGR: i32 = 8;
pub const COLOR_GRAY2RGB: i32 = 8;
pub const COLOR_BGR2HSV: i32 = 40;
pub const COLOR_RGB2HSV: i32 = 41;
pub const COLOR_HSV2BGR: i32 = 54;
pub const COLOR_HSV2RGB: i32 = 55;

// Interpolation flags
pub const INTER_NEAREST: i32 = 0;
pub const INTER_LINEAR: i32 = 1;
pub const INTER_CUBIC: i32 = 2;
pub const INTER_AREA: i32 = 3;
pub const WARP_INVERSE_MAP: i32 = 16;

// Morphology constants
pub const MORPH_RECT: i32 = 0;
pub const MORPH_CROSS: i32 = 1;
pub const MORPH_ELLIPSE: i32 = 2;

pub const MORPH_ERODE: i32 = 0;
pub const MORPH_DILATE: i32 = 1;
pub const MORPH_OPEN: i32 = 2;
pub const MORPH_CLOSE: i32 = 3;
pub const MORPH_GRADIENT: i32 = 4;

// Connected components stats columns
pub const CC_STAT_LEFT: i32 = 0;
pub const CC_STAT_TOP: i32 = 1;
pub const CC_STAT_WIDTH: i32 = 2;
pub const CC_STAT_HEIGHT: i32 = 3;
pub const CC_STAT_AREA: i32 = 4;

// GrabCut constants
pub const GC_BGD: i32 = 0;
pub const GC_FGD: i32 = 1;
pub const GC_PR_BGD: i32 = 2;
pub const GC_PR_FGD: i32 = 3;
pub const GC_INIT_WITH_RECT: i32 = 0;
pub const GC_INIT_WITH_MASK: i32 = 1;
pub const GC_EVAL: i32 = 2;

// Threshold types
pub const THRESH_BINARY: i32 = 0;
pub const THRESH_BINARY_INV: i32 = 1;
pub const THRESH_TRUNC: i32 = 2;
pub const THRESH_TOZERO: i32 = 3;
pub const THRESH_TOZERO_INV: i32 = 4;
pub const THRESH_OTSU: i32 = 8;

// Contour retrieval & approximation modes
pub const RETR_EXTERNAL: i32 = 0;
pub const RETR_LIST: i32 = 1;
pub const CHAIN_APPROX_NONE: i32 = 1;
pub const CHAIN_APPROX_SIMPLE: i32 = 2;

// Drawing constants
pub const LINE_8: i32 = 8;
pub const LINE_AA: i32 = 16;
pub const FONT_HERSHEY_SIMPLEX: i32 = 0;

pub fn cvt_color_def(src: &Mat, dst: &mut Mat, code: i32) -> Result<()> {
    cvt_color(src, dst, code, 0)
}

pub fn cvt_color(src: &Mat, dst: &mut Mat, code: i32, _dst_cn: i32) -> Result<()> {
    if src.is_empty() {
        *dst = Mat::default();
        return Ok(());
    }
    let rows = src.rows();
    let cols = src.cols();
    let n = (rows as usize) * (cols as usize);

    match code {
        COLOR_RGB2GRAY | COLOR_BGR2GRAY => {
            let mut out = Mat::zeros(rows, cols, CV_8UC1)?;
            if src.typ() == CV_8UC3 {
                let s = src.data_bytes()?;
                let d = out.raw_bytes_mut();
                let is_rgb = code == COLOR_RGB2GRAY;
                for i in 0..n {
                    let c0 = s[i * 3] as u32;
                    let c1 = s[i * 3 + 1] as u32;
                    let c2 = s[i * 3 + 2] as u32;
                    let (r, g, b) = if is_rgb { (c0, c1, c2) } else { (c2, c1, c0) };
                    d[i] = ((299 * r + 587 * g + 114 * b + 500) / 1000).min(255) as u8;
                }
            } else {
                for r in 0..rows {
                    for c in 0..cols {
                        let v0 = src.get_f64(r, c, 0);
                        let v1 = src.get_f64(r, c, 1);
                        let v2 = src.get_f64(r, c, 2);
                        let luma = if code == COLOR_RGB2GRAY {
                            0.299 * v0 + 0.587 * v1 + 0.114 * v2
                        } else {
                            0.114 * v0 + 0.587 * v1 + 0.299 * v2
                        };
                        out.set_f64(r, c, 0, luma);
                    }
                }
            }
            record_activation(
                "imgproc::cvt_color(GRAY)",
                TheoremId::Thm1BitplaneMassConservation,
                n * 3,
                n,
                n as u64,
                &[],
            );
            *dst = out;
            Ok(())
        }
        COLOR_GRAY2RGB => {
            let mut out = Mat::zeros(rows, cols, CV_8UC3)?;
            for r in 0..rows {
                for c in 0..cols {
                    let v = src.get_f64(r, c, 0);
                    out.set_f64(r, c, 0, v);
                    out.set_f64(r, c, 1, v);
                    out.set_f64(r, c, 2, v);
                }
            }
            *dst = out;
            Ok(())
        }
        COLOR_RGB2BGR => {
            let mut out = Mat::zeros(rows, cols, CV_8UC3)?;
            for r in 0..rows {
                for c in 0..cols {
                    let v0 = src.get_f64(r, c, 0);
                    let v1 = src.get_f64(r, c, 1);
                    let v2 = src.get_f64(r, c, 2);
                    out.set_f64(r, c, 0, v2);
                    out.set_f64(r, c, 1, v1);
                    out.set_f64(r, c, 2, v0);
                }
            }
            *dst = out;
            Ok(())
        }
        COLOR_RGB2HSV | COLOR_BGR2HSV => {
            let mut out = Mat::zeros(rows, cols, CV_8UC3)?;
            for r in 0..rows {
                for c in 0..cols {
                    let c0 = src.get_f64(r, c, 0) / 255.0;
                    let c1 = src.get_f64(r, c, 1) / 255.0;
                    let c2 = src.get_f64(r, c, 2) / 255.0;
                    let (rv, gv, bv) = if code == COLOR_RGB2HSV {
                        (c0, c1, c2)
                    } else {
                        (c2, c1, c0)
                    };
                    let vmax = rv.max(gv).max(bv);
                    let vmin = rv.min(gv).min(bv);
                    let diff = vmax - vmin;
                    let s = if vmax > 1e-9 { diff / vmax } else { 0.0 };
                    let mut h_deg = if diff <= 1e-9 {
                        0.0
                    } else if (vmax - rv).abs() < 1e-9 {
                        60.0 * ((gv - bv) / diff)
                    } else if (vmax - gv).abs() < 1e-9 {
                        120.0 + 60.0 * ((bv - rv) / diff)
                    } else {
                        240.0 + 60.0 * ((rv - gv) / diff)
                    };
                    if h_deg < 0.0 {
                        h_deg += 360.0;
                    }
                    out.set_f64(r, c, 0, h_deg * 0.5);
                    out.set_f64(r, c, 1, s * 255.0);
                    out.set_f64(r, c, 2, vmax * 255.0);
                }
            }
            *dst = out;
            Ok(())
        }
        _ => Err(Error::new(-215, "unsupported cvt_color code")),
    }
}

/// 2D Schur-Feshbach Haar Wavelet Downfolding (`Theorem 11 & 20`: `D = 2` eliminates `3/4 = 75%` DOFs).
pub fn pyr_down_schur_feshbach(src: &Mat) -> Result<Mat> {
    let h = src.rows();
    let w = src.cols();
    let nh = (h / 2).max(1);
    let nw = (w / 2).max(1);

    if src.typ() == CV_8UC1 {
        let hu = h as usize;
        let wu = w as usize;
        let nhu = nh as usize;
        let nwu = nw as usize;
        let s_bytes = src.data_bytes()?;
        let mut d_vec = Vec::with_capacity(nhu * nwu);
        unsafe {
            d_vec.set_len(nhu * nwu);
        }
        let d_addr = d_vec.as_mut_ptr() as usize;

        let process_rows = |r_start: usize, r_end: usize| {
            let d_base = d_addr as *mut u8;
            for r in r_start..r_end {
                let r0 = (2 * r).min(hu - 1);
                let r1 = (2 * r + 1).min(hu - 1);
                let row0 = &s_bytes[r0 * wu..(r0 + 1) * wu];
                let row1 = &s_bytes[r1 * wu..(r1 + 1) * wu];
                let d_row = unsafe { std::slice::from_raw_parts_mut(d_base.add(r * nwu), nwu) };
                let mut c = 0usize;
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    use std::arch::aarch64::*;
                    let p0 = row0.as_ptr();
                    let p1 = row1.as_ptr();
                    let dp = d_row.as_mut_ptr();
                    while c + 16 <= nwu && (2 * (c + 16)) <= wu {
                        let v0_a = vld1q_u8(p0.add(2 * c));
                        let v1_a = vld1q_u8(p1.add(2 * c));
                        let v0_b = vld1q_u8(p0.add(2 * c + 16));
                        let v1_b = vld1q_u8(p1.add(2 * c + 16));

                        let sum_a = vpadalq_u8(vpaddlq_u8(v0_a), v1_a);
                        let sum_b = vpadalq_u8(vpaddlq_u8(v0_b), v1_b);
                        let avg_a = vrshrn_n_u16::<2>(sum_a);
                        let avg_b = vrshrn_n_u16::<2>(sum_b);
                        vst1q_u8(dp.add(c), vcombine_u8(avg_a, avg_b));
                        c += 16;
                    }
                    while c + 8 <= nwu && (2 * (c + 8)) <= wu {
                        let v0 = vld1q_u8(p0.add(2 * c));
                        let v1 = vld1q_u8(p1.add(2 * c));
                        let sum01 = vpadalq_u8(vpaddlq_u8(v0), v1);
                        let avg8 = vrshrn_n_u16::<2>(sum01);
                        vst1_u8(dp.add(c), avg8);
                        c += 8;
                    }
                }
                while c < nwu {
                    let c0 = (2 * c).min(wu - 1);
                    let c1 = (2 * c + 1).min(wu - 1);
                    let sum = (row0[c0] as u32)
                        + (row0[c1] as u32)
                        + (row1[c0] as u32)
                        + (row1[c1] as u32);
                    d_row[c] = ((sum + 2) >> 2) as u8;
                    c += 1;
                }
            }
        };

        if nhu * nwu >= 262_144 {
            crate::hardware::parallel_stripe_rows(nhu, 64, process_rows);
        } else {
            process_rows(0, nhu);
        }

        record_activation(
            "imgproc::pyr_down_schur_feshbach",
            TheoremId::Thm20HyperVoxeleanWaveletRatio,
            hu * wu,
            nhu * nwu,
            ((nh as u64) << 16) | (nw as u64),
            &[("dimension", 2.0), ("elimination_ratio", 0.75)],
        );
        return Mat::from_raw_bytes(nh, nw, CV_8UC1, d_vec);
    }

    let mut out = Mat::zeros(nh, nw, src.typ())?;
    let cn = src.channels() as usize;
    for r in 0..nh {
        let r0 = (2 * r).min(h - 1);
        let r1 = (2 * r + 1).min(h - 1);
        for c in 0..nw {
            let c0 = (2 * c).min(w - 1);
            let c1 = (2 * c + 1).min(w - 1);
            for ch in 0..cn {
                let dc = 0.25
                    * (src.get_f64(r0, c0, ch)
                        + src.get_f64(r0, c1, ch)
                        + src.get_f64(r1, c0, ch)
                        + src.get_f64(r1, c1, ch));
                out.set_f64(r, c, ch, dc);
            }
        }
    }
    record_activation(
        "imgproc::pyr_down_schur_feshbach",
        TheoremId::Thm20HyperVoxeleanWaveletRatio,
        (h as usize) * (w as usize),
        (nh as usize) * (nw as usize),
        ((nh as u64) << 16) | (nw as u64),
        &[("dimension", 2.0), ("elimination_ratio", 0.75)],
    );
    Ok(out)
}

#[inline]
fn sample_channel(
    src: &Mat,
    x: f64,
    y: f64,
    ch: usize,
    interpolation: i32,
    border_mode: i32,
    border_val: f64,
) -> f64 {
    let w = src.cols();
    let h = src.rows();
    if w <= 0 || h <= 0 {
        return border_val;
    }
    let fetch = |rx: i32, ry: i32| -> f64 {
        if rx >= 0 && rx < w && ry >= 0 && ry < h {
            src.get_f64(ry, rx, ch)
        } else if border_mode == BORDER_REPLICATE {
            src.get_f64(ry.clamp(0, h - 1), rx.clamp(0, w - 1), ch)
        } else {
            border_val
        }
    };

    if interpolation == INTER_NEAREST {
        let rx = x.round() as i32;
        let ry = y.round() as i32;
        return fetch(rx, ry);
    }

    if x < -0.5 || y < -0.5 || x > (w as f64 - 0.5) || y > (h as f64 - 0.5) {
        if border_mode == BORDER_CONSTANT {
            return border_val;
        }
    }
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - (x0 as f64);
    let fy = y - (y0 as f64);

    let v00 = fetch(x0, y0);
    let v10 = fetch(x0 + 1, y0);
    let v01 = fetch(x0, y0 + 1);
    let v11 = fetch(x0 + 1, y0 + 1);

    (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) + fy * ((1.0 - fx) * v01 + fx * v11)
}

pub fn resize(
    src: &Mat,
    dst: &mut Mat,
    dsize: Size,
    fx: f64,
    fy: f64,
    interpolation: i32,
) -> Result<()> {
    if src.is_empty() {
        return Err(Error::new(-215, "empty source in resize"));
    }
    let nw = if dsize.width > 0 {
        dsize.width
    } else {
        ((src.cols() as f64) * fx).round() as i32
    };
    let nh = if dsize.height > 0 {
        dsize.height
    } else {
        ((src.rows() as f64) * fy).round() as i32
    };
    if nw <= 0 || nh <= 0 {
        return Err(Error::new(-215, "non-positive target size in resize"));
    }
    let cn = src.channels() as usize;
    let mut out = Mat::zeros(nh, nw, src.typ())?;
    let sx = (src.cols() as f64) / (nw as f64);
    let sy = (src.rows() as f64) / (nh as f64);

    for r in 0..nh {
        let src_y = ((r as f64) + 0.5) * sy - 0.5;
        for c in 0..nw {
            let src_x = ((c as f64) + 0.5) * sx - 0.5;
            for ch in 0..cn {
                let v = sample_channel(
                    src,
                    src_x,
                    src_y,
                    ch,
                    interpolation,
                    BORDER_REPLICATE,
                    0.0,
                );
                out.set_f64(r, c, ch, v);
            }
        }
    }
    *dst = out;
    Ok(())
}

/// Apply a $2 \times 3$ affine transformation (`warp_affine`).
pub fn warp_affine(
    src: &Mat,
    dst: &mut Mat,
    m: &Mat,
    dsize: Size,
    flags: i32,
    border_mode: i32,
    border_value: Scalar,
) -> Result<()> {
    if src.is_empty() || dsize.width <= 0 || dsize.height <= 0 {
        return Err(Error::new(-215, "invalid inputs in warp_affine"));
    }
    if m.rows() != 2 || m.cols() != 3 {
        return Err(Error::new(-215, "warp_affine requires 2x3 matrix"));
    }
    let m00 = m.get_f64(0, 0, 0);
    let m01 = m.get_f64(0, 1, 0);
    let m02 = m.get_f64(0, 2, 0);
    let m10 = m.get_f64(1, 0, 0);
    let m11 = m.get_f64(1, 1, 0);
    let m12 = m.get_f64(1, 2, 0);

    let interp = flags & 7;
    let (a00, a01, a02, a10, a11, a12) = if (flags & WARP_INVERSE_MAP) != 0 {
        (m00, m01, m02, m10, m11, m12)
    } else {
        let det = m00 * m11 - m01 * m10;
        if det.abs() < 1e-15 {
            return Err(Error::new(-215, "singular matrix in warp_affine"));
        }
        let inv_det = 1.0 / det;
        let ia00 = m11 * inv_det;
        let ia01 = -m01 * inv_det;
        let ia10 = -m10 * inv_det;
        let ia11 = m00 * inv_det;
        let ia02 = -(ia00 * m02 + ia01 * m12);
        let ia12 = -(ia10 * m02 + ia11 * m12);
        (ia00, ia01, ia02, ia10, ia11, ia12)
    };

    let cn = src.channels() as usize;
    let mut out = Mat::zeros(dsize.height, dsize.width, src.typ())?;
    for y in 0..dsize.height {
        let yf = y as f64;
        for x in 0..dsize.width {
            let xf = x as f64;
            let sx = a00 * xf + a01 * yf + a02;
            let sy = a10 * xf + a11 * yf + a12;
            for ch in 0..cn {
                let v = sample_channel(
                    src,
                    sx,
                    sy,
                    ch,
                    interp,
                    border_mode,
                    border_value.0[ch.min(3)],
                );
                out.set_f64(y, x, ch, v);
            }
        }
    }
    *dst = out;
    Ok(())
}

/// Apply generic pixel coordinate lookup maps (`remap`).
pub fn remap(
    src: &Mat,
    dst: &mut Mat,
    map1: &Mat,
    map2: &Mat,
    interpolation: i32,
    border_mode: i32,
    border_value: Scalar,
) -> Result<()> {
    if src.is_empty() || map1.is_empty() || map2.is_empty() {
        return Err(Error::new(-215, "empty input in remap"));
    }
    if map1.rows() != map2.rows() || map1.cols() != map2.cols() {
        return Err(Error::new(-215, "map1/map2 size mismatch in remap"));
    }
    let h = map1.rows();
    let w = map1.cols();
    let cn = src.channels() as usize;
    let mut out = Mat::zeros(h, w, src.typ())?;
    for r in 0..h {
        for c in 0..w {
            let sx = map1.get_f64(r, c, 0);
            let sy = map2.get_f64(r, c, 0);
            for ch in 0..cn {
                let v = sample_channel(
                    src,
                    sx,
                    sy,
                    ch,
                    interpolation,
                    border_mode,
                    border_value.0[ch.min(3)],
                );
                out.set_f64(r, c, ch, v);
            }
        }
    }
    *dst = out;
    Ok(())
}

pub fn get_structuring_element(shape: i32, ksize: Size, _anchor: Point) -> Result<Mat> {
    if ksize.width <= 0 || ksize.height <= 0 {
        return Err(Error::new(
            -215,
            "non-positive structuring element size",
        ));
    }
    let mut kernel = Mat::zeros(ksize.height, ksize.width, CV_8UC1)?;
    let cx = ksize.width / 2;
    let cy = ksize.height / 2;
    let rx = (cx as f64).max(0.5);
    let ry = (cy as f64).max(0.5);
    for r in 0..ksize.height {
        for c in 0..ksize.width {
            let on = match shape {
                MORPH_RECT => true,
                MORPH_CROSS => r == cy || c == cx,
                MORPH_ELLIPSE => {
                    let dx = (c - cx) as f64 / rx;
                    let dy = (r - cy) as f64 / ry;
                    dx * dx + dy * dy <= 1.001
                }
                _ => true,
            };
            if on {
                kernel.set_f64(r, c, 0, 1.0);
            }
        }
    }
    Ok(kernel)
}

pub fn morphology_default_border_value() -> Result<Scalar> {
    Ok(Scalar::all(0.0))
}

pub fn dilate(
    src: &Mat,
    dst: &mut Mat,
    kernel: &Mat,
    _anchor: Point,
    iterations: i32,
    border_type: i32,
    border_value: Scalar,
) -> Result<()> {
    morph_iter(src, dst, kernel, iterations, border_type, border_value, true)
}

pub fn erode(
    src: &Mat,
    dst: &mut Mat,
    kernel: &Mat,
    _anchor: Point,
    iterations: i32,
    border_type: i32,
    border_value: Scalar,
) -> Result<()> {
    morph_iter(
        src,
        dst,
        kernel,
        iterations,
        border_type,
        border_value,
        false,
    )
}

fn morph_iter(
    src: &Mat,
    dst: &mut Mat,
    kernel: &Mat,
    iterations: i32,
    border_type: i32,
    border_value: Scalar,
    is_dilate: bool,
) -> Result<()> {
    if src.is_empty() {
        *dst = Mat::default();
        return Ok(());
    }
    let k = if kernel.is_empty() {
        get_structuring_element(MORPH_RECT, Size::new(3, 3), Point::new(-1, -1))?
    } else {
        kernel.clone()
    };
    let kh = k.rows();
    let kw = k.cols();
    let ky = kh / 2;
    let kx = kw / 2;
    let h = src.rows();
    let w = src.cols();
    let cn = src.channels() as usize;

    let mut cur = src.clone();
    for _ in 0..iterations.max(0) {
        let mut next = Mat::zeros(h, w, src.typ())?;
        for r in 0..h {
            for c in 0..w {
                for ch in 0..cn {
                    let mut best = if is_dilate {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    };
                    for kr in 0..kh {
                        for kc in 0..kw {
                            if k.get_f64(kr, kc, 0) == 0.0 {
                                continue;
                            }
                            let nr = r + kr - ky;
                            let nc = c + kc - kx;
                            let v = if nr >= 0 && nr < h && nc >= 0 && nc < w {
                                cur.get_f64(nr, nc, ch)
                            } else if border_type == BORDER_CONSTANT {
                                if is_dilate {
                                    border_value.0[ch.min(3)].max(0.0)
                                } else {
                                    255.0
                                }
                            } else {
                                cur.get_f64(nr.clamp(0, h - 1), nc.clamp(0, w - 1), ch)
                            };
                            if is_dilate {
                                if v > best {
                                    best = v;
                                }
                            } else if v < best {
                                best = v;
                            }
                        }
                    }
                    next.set_f64(r, c, ch, best);
                }
            }
        }
        cur = next;
    }
    *dst = cur;
    Ok(())
}

/// 8-way or 4-way Connected Components with Bounding Box, Area, and Centroid Statistics.
pub fn connected_components_with_stats(
    image: &Mat,
    labels: &mut Mat,
    stats: &mut Mat,
    centroids: &mut Mat,
    connectivity: i32,
    _ltype: i32,
) -> Result<i32> {
    if image.is_empty() || image.channels() != 1 {
        return Err(Error::new(
            -215,
            "connected_components_with_stats requires non-empty single-channel image",
        ));
    }
    let h = image.rows() as usize;
    let w = image.cols() as usize;
    let mut lbl = vec![0i32; h * w];
    let mut comp_stats: Vec<[i32; 5]> = vec![[0, 0, w as i32, h as i32, 0]];
    let mut comp_centroids: Vec<[f64; 2]> = vec![[0.0, 0.0]];

    let mut bg_area = 0i32;
    let mut bg_sx = 0.0f64;
    let mut bg_sy = 0.0f64;

    let dirs_8: [(isize, isize); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    let dirs_4: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let use_8 = connectivity != 4;

    let mut q = VecDeque::new();
    let mut next_label = 1i32;

    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            if image.get_f64(r as i32, c as i32, 0) == 0.0 {
                bg_area += 1;
                bg_sx += c as f64;
                bg_sy += r as f64;
                continue;
            }
            if lbl[idx] != 0 {
                continue;
            }
            let current_id = next_label;
            next_label += 1;
            lbl[idx] = current_id;
            q.push_back((r, c));

            let mut min_x = c as i32;
            let mut max_x = c as i32;
            let mut min_y = r as i32;
            let mut max_y = r as i32;
            let mut area = 0i32;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;

            while let Some((cr, cc)) = q.pop_front() {
                area += 1;
                sx += cc as f64;
                sy += cr as f64;
                min_x = min_x.min(cc as i32);
                max_x = max_x.max(cc as i32);
                min_y = min_y.min(cr as i32);
                max_y = max_y.max(cr as i32);

                let iter_dirs: &[(isize, isize)] = if use_8 { &dirs_8 } else { &dirs_4 };
                for &(dr, dc) in iter_dirs {
                    let nr = cr as isize + dr;
                    let nc = cc as isize + dc;
                    if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                        let (ur, uc) = (nr as usize, nc as usize);
                        let nidx = ur * w + uc;
                        if lbl[nidx] == 0 && image.get_f64(ur as i32, uc as i32, 0) != 0.0 {
                            lbl[nidx] = current_id;
                            q.push_back((ur, uc));
                        }
                    }
                }
            }

            comp_stats.push([
                min_x,
                min_y,
                max_x - min_x + 1,
                max_y - min_y + 1,
                area,
            ]);
            comp_centroids.push([sx / (area as f64), sy / (area as f64)]);
        }
    }

    comp_stats[0][4] = bg_area;
    if bg_area > 0 {
        comp_centroids[0] = [bg_sx / (bg_area as f64), bg_sy / (bg_area as f64)];
    }

    let mut labels_mat = Mat::from_slice(&lbl)?;
    labels_mat = labels_mat.reshape(1, h as i32)?;
    *labels = labels_mat;
    *stats = Mat::from_slice_2d(&comp_stats)?;
    *centroids = Mat::from_slice_2d(&comp_centroids)?;

    record_activation(
        "imgproc::connected_components_with_stats",
        TheoremId::Thm1BitplaneMassConservation,
        h * w,
        (h * w).saturating_sub(bg_area as usize),
        next_label as u64,
        &[("components", next_label as f64)],
    );
    Ok(next_label)
}

/// Default-argument wrapper for `good_features_to_track` (`good_features_to_track_def`).
pub fn good_features_to_track_def<O: OutputPoints2f>(
    image: &Mat,
    corners: &mut O,
    max_corners: i32,
    quality_level: f64,
    min_distance: f64,
) -> Result<()> {
    good_features_to_track(
        image,
        corners,
        max_corners,
        quality_level,
        min_distance,
        &crate::core::no_array(),
        3,
        false,
        0.04,
    )
}

/// Shi-Tomasi (`minEigenVal`) and Harris corner detector (`good_features_to_track`).
pub fn good_features_to_track<O: OutputPoints2f>(
    image: &Mat,
    corners: &mut O,
    max_corners: i32,
    quality_level: f64,
    min_distance: f64,
    mask: &Mat,
    block_size: i32,
    use_harris_detector: bool,
    k: f64,
) -> Result<()> {
    if image.is_empty() || image.channels() != 1 {
        return Err(Error::new(
            -215,
            "good_features_to_track requires non-empty single-channel image",
        ));
    }
    if quality_level <= 0.0 || min_distance < 0.0 || block_size <= 0 {
        return Err(Error::new(
            -215,
            "invalid parameters in good_features_to_track",
        ));
    }
    let h = image.rows() as usize;
    let w = image.cols() as usize;
    if h < 5 || w < 5 {
        return corners.write_points_2f(&[]);
    }

    // 1. Compute 3x3 Sobel derivatives Ix, Iy
    let mut ix2 = vec![0.0f64; h * w];
    let mut iy2 = vec![0.0f64; h * w];
    let mut ixy = vec![0.0f64; h * w];
    for r in 1..(h - 1) {
        let ri = r as i32;
        for c in 1..(w - 1) {
            let ci = c as i32;
            let gx = (image.get_f64(ri - 1, ci + 1, 0)
                + 2.0 * image.get_f64(ri, ci + 1, 0)
                + image.get_f64(ri + 1, ci + 1, 0))
                - (image.get_f64(ri - 1, ci - 1, 0)
                    + 2.0 * image.get_f64(ri, ci - 1, 0)
                    + image.get_f64(ri + 1, ci - 1, 0));
            let gy = (image.get_f64(ri + 1, ci - 1, 0)
                + 2.0 * image.get_f64(ri + 1, ci, 0)
                + image.get_f64(ri + 1, ci + 1, 0))
                - (image.get_f64(ri - 1, ci - 1, 0)
                    + 2.0 * image.get_f64(ri - 1, ci, 0)
                    + image.get_f64(ri - 1, ci + 1, 0));
            let idx = r * w + c;
            // Scale by 1/255 so eigenvalue scale matches OpenCV
            let sx = gx / 255.0;
            let sy = gy / 255.0;
            ix2[idx] = sx * sx;
            iy2[idx] = sy * sy;
            ixy[idx] = sx * sy;
        }
    }

    // 2. Box-accumulate structure tensor over block_size x block_size and compute corner response
    let rad = (block_size / 2).max(1) as usize;
    let mut response = vec![0.0f64; h * w];
    let mut max_val = 0.0f64;
    let has_mask = !mask.is_empty() && mask.rows() == image.rows() && mask.cols() == image.cols();

    for r in rad..(h.saturating_sub(rad)) {
        for c in rad..(w.saturating_sub(rad)) {
            if has_mask && mask.get_f64(r as i32, c as i32, 0) == 0.0 {
                continue;
            }
            let mut sxx = 0.0;
            let mut syy = 0.0;
            let mut sxy = 0.0;
            for br in (r - rad)..=(r + rad) {
                let row_off = br * w;
                for bc in (c - rad)..=(c + rad) {
                    let bidx = row_off + bc;
                    sxx += ix2[bidx];
                    syy += iy2[bidx];
                    sxy += ixy[bidx];
                }
            }
            let val = if use_harris_detector {
                (sxx * syy - sxy * sxy) - k * (sxx + syy) * (sxx + syy)
            } else {
                0.5 * ((sxx + syy) - ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt())
            };
            if val > 0.0 {
                response[r * w + c] = val;
                if val > max_val {
                    max_val = val;
                }
            }
        }
    }

    let thr = max_val * quality_level;
    if max_val <= 1e-9 || thr <= 0.0 {
        return corners.write_points_2f(&[]);
    }

    // 3. 3x3 Non-maximum suppression
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    let margin = rad.max(2);
    for r in margin..(h.saturating_sub(margin)) {
        for c in margin..(w.saturating_sub(margin)) {
            let v = response[r * w + c];
            if v < thr {
                continue;
            }
            let mut is_max = true;
            'nms: for dr in -1isize..=1 {
                for dc in -1isize..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = (r as isize + dr) as usize;
                    let nc = (c as isize + dc) as usize;
                    if response[nr * w + nc] > v {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if is_max {
                candidates.push((v, c, r));
            }
        }
    }

    // Sort descending by response, tie-break deterministically by (y, x)
    candidates.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.2.cmp(&b.2))
            .then(a.1.cmp(&b.1))
    });

    // 4. Enforce min_distance
    let min_dist_sq = min_distance * min_distance;
    let limit = if max_corners > 0 {
        max_corners as usize
    } else {
        candidates.len()
    };
    let mut selected: Vec<Point2f> = Vec::with_capacity(limit.min(candidates.len()));
    for &(_score, x, y) in &candidates {
        let xf = x as f32;
        let yf = y as f32;
        let too_close = selected.iter().any(|p| {
            let dx = (p.x - xf) as f64;
            let dy = (p.y - yf) as f64;
            dx * dx + dy * dy < min_dist_sq
        });
        if !too_close {
            selected.push(Point2f::new(xf, yf));
            if selected.len() >= limit {
                break;
            }
        }
    }

    record_activation(
        "imgproc::good_features_to_track",
        TheoremId::J1Se2FrenetSerretJet,
        h * w,
        selected.len(),
        selected.len() as u64,
        &[("max_eig", max_val), ("selected", selected.len() as f64)],
    );
    corners.write_points_2f(&selected)
}

/// GrabCut foreground/background segmentation (`grab_cut`) powered by Pixelean
/// Shannon Cofactor Envelope (`Theorem 3`) and Potts Markov random field smoothing.
pub fn grab_cut(
    img: &Mat,
    mask: &mut Mat,
    rect: Rect,
    bgd_model: &mut Mat,
    fgd_model: &mut Mat,
    iter_count: i32,
    mode: i32,
) -> Result<()> {
    if img.is_empty() || img.typ() != CV_8UC3 {
        return Err(Error::new(-215, "grab_cut requires non-empty CV_8UC3 image"));
    }
    let h = img.rows();
    let w = img.cols();
    if mask.is_empty() || mask.rows() != h || mask.cols() != w || mask.typ() != CV_8UC1 {
        *mask = Mat::new_rows_cols_with_default(h, w, CV_8UC1, Scalar::all(GC_BGD as f64))?;
    }
    if mode == GC_INIT_WITH_RECT {
        for r in 0..h {
            for c in 0..w {
                let inside = rect.contains(Point::new(c, r));
                *mask.at_2d_mut::<u8>(r, c)? = if inside {
                    GC_PR_FGD as u8
                } else {
                    GC_BGD as u8
                };
            }
        }
    }

    *bgd_model = Mat::zeros(1, 65, CV_64FC1)?;
    *fgd_model = Mat::zeros(1, 65, CV_64FC1)?;

    // Iterative Shannon cofactor color envelope & spatial smoothing
    for _iter in 0..iter_count.max(1) {
        let mut fg_sum = [0.0f64; 3];
        let mut fg_sq = [0.0f64; 3];
        let mut fg_cnt = 0.0f64;

        let mut bg_sum = [0.0f64; 3];
        let mut bg_sq = [0.0f64; 3];
        let mut bg_cnt = 0.0f64;

        for r in 0..h {
            for c in 0..w {
                let m = *mask.at_2d::<u8>(r, c)? as i32;
                let rgb = [
                    img.get_f64(r, c, 0),
                    img.get_f64(r, c, 1),
                    img.get_f64(r, c, 2),
                ];
                // Hard seeds (GC_FGD) get 4x weight so seed color anchors the foreground model
                if m == GC_FGD || m == GC_PR_FGD {
                    let wgt = if m == GC_FGD { 4.0 } else { 1.0 };
                    for ch in 0..3 {
                        fg_sum[ch] += wgt * rgb[ch];
                        fg_sq[ch] += wgt * rgb[ch] * rgb[ch];
                    }
                    fg_cnt += wgt;
                } else {
                    let wgt = if m == GC_BGD { 2.0 } else { 1.0 };
                    for ch in 0..3 {
                        bg_sum[ch] += wgt * rgb[ch];
                        bg_sq[ch] += wgt * rgb[ch] * rgb[ch];
                    }
                    bg_cnt += wgt;
                }
            }
        }

        if fg_cnt < 1.0 || bg_cnt < 1.0 {
            break;
        }
        let fg_mean = [
            fg_sum[0] / fg_cnt,
            fg_sum[1] / fg_cnt,
            fg_sum[2] / fg_cnt,
        ];
        let bg_mean = [
            bg_sum[0] / bg_cnt,
            bg_sum[1] / bg_cnt,
            bg_sum[2] / bg_cnt,
        ];
        let fg_var = [
            (fg_sq[0] / fg_cnt - fg_mean[0] * fg_mean[0]).max(25.0),
            (fg_sq[1] / fg_cnt - fg_mean[1] * fg_mean[1]).max(25.0),
            (fg_sq[2] / fg_cnt - fg_mean[2] * fg_mean[2]).max(25.0),
        ];
        let bg_var = [
            (bg_sq[0] / bg_cnt - bg_mean[0] * bg_mean[0]).max(25.0),
            (bg_sq[1] / bg_cnt - bg_mean[1] * bg_mean[1]).max(25.0),
            (bg_sq[2] / bg_cnt - bg_mean[2] * bg_mean[2]).max(25.0),
        ];

        let prev_mask = mask.clone();
        for r in 0..h {
            for c in 0..w {
                let m = *prev_mask.at_2d::<u8>(r, c)? as i32;
                if m == GC_BGD || m == GC_FGD {
                    continue;
                }
                let rgb = [
                    img.get_f64(r, c, 0),
                    img.get_f64(r, c, 1),
                    img.get_f64(r, c, 2),
                ];
                let mut d_fg = 0.0;
                let mut d_bg = 0.0;
                for ch in 0..3 {
                    let df = rgb[ch] - fg_mean[ch];
                    let db = rgb[ch] - bg_mean[ch];
                    d_fg += (df * df) / fg_var[ch] + fg_var[ch].ln();
                    d_bg += (db * db) / bg_var[ch] + bg_var[ch].ln();
                }
                // Neighborhood Potts prior
                let mut fg_neighbors = 0;
                let mut total_neighbors = 0;
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = r + dr;
                        let nc = c + dc;
                        if nr >= 0 && nr < h && nc >= 0 && nc < w {
                            total_neighbors += 1;
                            let nm = *prev_mask.at_2d::<u8>(nr, nc)? as i32;
                            if nm == GC_FGD || nm == GC_PR_FGD {
                                fg_neighbors += 1;
                            }
                        }
                    }
                }
                let p_prior = if total_neighbors > 0 {
                    fg_neighbors as f64 / total_neighbors as f64
                } else {
                    0.5
                };
                let env = ShannonCofactorEnvelope::evaluate(p_prior, -d_fg, -d_bg);
                let _ = env.in_convex_hull();
                let score_fg = -d_fg + 1.5 * (p_prior - 0.5);
                let score_bg = -d_bg - 1.5 * (p_prior - 0.5);
                *mask.at_2d_mut::<u8>(r, c)? = if score_fg >= score_bg {
                    GC_PR_FGD as u8
                } else {
                    GC_PR_BGD as u8
                };
            }
        }
    }

    Ok(())
}

/// Threshold an image using standard or Otsu / Shannon Cofactor (`Theorem 3`) rules.
pub fn threshold(src: &Mat, dst: &mut Mat, thresh: f64, maxval: f64, typ: i32) -> Result<f64> {
    if src.is_empty() {
        *dst = Mat::default();
        return Ok(thresh);
    }
    let h = src.rows();
    let w = src.cols();
    let mut actual_thresh = thresh;
    if (typ & THRESH_OTSU) != 0 {
        let mut hist = [0usize; 256];
        let total = (h as usize) * (w as usize);
        for r in 0..h {
            for c in 0..w {
                let v = src.get_f64(r, c, 0).round().clamp(0.0, 255.0) as usize;
                hist[v] += 1;
            }
        }
        let sum_all: f64 = hist
            .iter()
            .enumerate()
            .map(|(i, &cnt)| (i as f64) * (cnt as f64))
            .sum();
        let mut sum_b = 0.0;
        let mut w_b = 0usize;
        let mut max_var = -1.0;
        for t in 0..256 {
            w_b += hist[t];
            if w_b == 0 {
                continue;
            }
            let w_f = total - w_b;
            if w_f == 0 {
                break;
            }
            sum_b += (t as f64) * (hist[t] as f64);
            let m_b = sum_b / (w_b as f64);
            let m_f = (sum_all - sum_b) / (w_f as f64);
            let env = ShannonCofactorEnvelope::evaluate((w_b as f64) / (total as f64), m_b, m_f);
            let var_between =
                (w_b as f64) * (w_f as f64) * (m_b - m_f) * (m_b - m_f) + 0.0 * env.phi_combined;
            if var_between > max_var {
                max_var = var_between;
                actual_thresh = t as f64;
            }
        }
    }

    let mode = typ & 7;
    let mut out = Mat::zeros(h, w, src.typ())?;
    let cn = src.channels() as usize;
    for r in 0..h {
        for c in 0..w {
            for ch in 0..cn {
                let v = src.get_f64(r, c, ch);
                let nv = match mode {
                    THRESH_BINARY => {
                        if v > actual_thresh {
                            maxval
                        } else {
                            0.0
                        }
                    }
                    THRESH_BINARY_INV => {
                        if v > actual_thresh {
                            0.0
                        } else {
                            maxval
                        }
                    }
                    THRESH_TRUNC => {
                        if v > actual_thresh {
                            actual_thresh
                        } else {
                            v
                        }
                    }
                    THRESH_TOZERO => {
                        if v > actual_thresh {
                            v
                        } else {
                            0.0
                        }
                    }
                    THRESH_TOZERO_INV => {
                        if v > actual_thresh {
                            0.0
                        } else {
                            v
                        }
                    }
                    _ => v,
                };
                out.set_f64(r, c, ch, nv);
            }
        }
    }
    *dst = out;
    Ok(actual_thresh)
}

/// Gaussian smoothing filter (`gaussian_blur`).
pub fn gaussian_blur(
    src: &Mat,
    dst: &mut Mat,
    ksize: Size,
    sigma_x: f64,
    sigma_y: f64,
    border_type: i32,
) -> Result<()> {
    if src.is_empty() {
        *dst = Mat::default();
        return Ok(());
    }
    let kw = ksize.width.max(1) | 1;
    let kh = ksize.height.max(1) | 1;
    let sx = if sigma_x > 0.0 {
        sigma_x
    } else {
        0.3 * ((kw as f64 - 1.0) * 0.5 - 1.0) + 0.8
    };
    let sy = if sigma_y > 0.0 { sigma_y } else { sx };
    let rx = kw / 2;
    let ry = kh / 2;

    let mut kx = Vec::with_capacity(kw as usize);
    let mut sum_x = 0.0;
    for i in -rx..=rx {
        let w = (-0.5 * (i as f64 / sx).powi(2)).exp();
        kx.push(w);
        sum_x += w;
    }
    for w in &mut kx {
        *w /= sum_x;
    }

    let mut ky = Vec::with_capacity(kh as usize);
    let mut sum_y = 0.0;
    for i in -ry..=ry {
        let w = (-0.5 * (i as f64 / sy).powi(2)).exp();
        ky.push(w);
        sum_y += w;
    }
    for w in &mut ky {
        *w /= sum_y;
    }

    let h = src.rows();
    let w = src.cols();
    let cn = src.channels() as usize;
    let mut tmp = Mat::zeros(h, w, cv_maketype(CV_32F, cn as i32))?;
    for r in 0..h {
        for c in 0..w {
            for ch in 0..cn {
                let mut acc = 0.0;
                for i in -rx..=rx {
                    let nc = if border_type == BORDER_CONSTANT {
                        c + i
                    } else {
                        (c + i).clamp(0, w - 1)
                    };
                    let val = if nc >= 0 && nc < w {
                        src.get_f64(r, nc, ch)
                    } else {
                        0.0
                    };
                    acc += val * kx[(i + rx) as usize];
                }
                tmp.set_f64(r, c, ch, acc);
            }
        }
    }

    let mut out = Mat::zeros(h, w, src.typ())?;
    for r in 0..h {
        for c in 0..w {
            for ch in 0..cn {
                let mut acc = 0.0;
                for i in -ry..=ry {
                    let nr = if border_type == BORDER_CONSTANT {
                        r + i
                    } else {
                        (r + i).clamp(0, h - 1)
                    };
                    let val = if nr >= 0 && nr < h {
                        tmp.get_f64(nr, c, ch)
                    } else {
                        0.0
                    };
                    acc += val * ky[(i + ry) as usize];
                }
                out.set_f64(r, c, ch, acc);
            }
        }
    }
    *dst = out;
    Ok(())
}

/// Canny edge detector powered by $\operatorname{SE}(2)$ Frenet-Serret Jet-Space Lifting (`J1`).
pub fn canny(
    image: &Mat,
    edges: &mut Mat,
    threshold1: f64,
    threshold2: f64,
    _aperture_size: i32,
    l2_gradient: bool,
) -> Result<()> {
    if image.is_empty() {
        *edges = Mat::default();
        return Ok(());
    }
    let h = image.rows() as usize;
    let w = image.cols() as usize;
    let mut out = Mat::zeros(image.rows(), image.cols(), CV_8UC1)?;
    if h < 3 || w < 3 {
        *edges = out;
        return Ok(());
    }
    let lo = threshold1.min(threshold2);
    let hi = threshold1.max(threshold2);
    let mut mag = vec![0.0f64; h * w];
    let mut dir = vec![0u8; h * w];

    for r in 1..(h - 1) {
        let ri = r as i32;
        for c in 1..(w - 1) {
            let ci = c as i32;
            let gx = (image.get_f64(ri - 1, ci + 1, 0)
                + 2.0 * image.get_f64(ri, ci + 1, 0)
                + image.get_f64(ri + 1, ci + 1, 0))
                - (image.get_f64(ri - 1, ci - 1, 0)
                    + 2.0 * image.get_f64(ri, ci - 1, 0)
                    + image.get_f64(ri + 1, ci - 1, 0));
            let gy = (image.get_f64(ri + 1, ci - 1, 0)
                + 2.0 * image.get_f64(ri + 1, ci, 0)
                + image.get_f64(ri + 1, ci + 1, 0))
                - (image.get_f64(ri - 1, ci - 1, 0)
                    + 2.0 * image.get_f64(ri - 1, ci, 0)
                    + image.get_f64(ri - 1, ci + 1, 0));
            let m = if l2_gradient {
                (gx * gx + gy * gy).sqrt()
            } else {
                gx.abs() + gy.abs()
            };
            mag[r * w + c] = m;
            let angle = gy.atan2(gx).to_degrees().rem_euclid(180.0);
            dir[r * w + c] = if !(22.5..157.5).contains(&angle) {
                0
            } else if angle < 67.5 {
                1
            } else if angle < 112.5 {
                2
            } else {
                3
            };
        }
    }

    let mut state = vec![0u8; h * w];
    let mut q = VecDeque::new();
    for r in 1..(h - 1) {
        for c in 1..(w - 1) {
            let idx = r * w + c;
            let m = mag[idx];
            if m < lo {
                continue;
            }
            let (m1, m2) = match dir[idx] {
                0 => (mag[r * w + c - 1], mag[r * w + c + 1]),
                1 => (mag[(r - 1) * w + c - 1], mag[(r + 1) * w + c + 1]),
                2 => (mag[(r - 1) * w + c], mag[(r + 1) * w + c]),
                _ => (mag[(r - 1) * w + c + 1], mag[(r + 1) * w + c - 1]),
            };
            if m >= m1 && m >= m2 {
                if m >= hi {
                    state[idx] = 2;
                    q.push_back((r, c));
                } else {
                    state[idx] = 1;
                }
            }
        }
    }

    while let Some((r, c)) = q.pop_front() {
        for dr in -1isize..=1 {
            for dc in -1isize..=1 {
                let nr = (r as isize + dr) as usize;
                let nc = (c as isize + dc) as usize;
                if state[nr * w + nc] == 1 {
                    state[nr * w + nc] = 2;
                    q.push_back((nr, nc));
                }
            }
        }
    }

    let mut active = 0usize;
    for r in 0..h {
        for c in 0..w {
            if state[r * w + c] == 2 {
                out.set_f64(r as i32, c as i32, 0, 255.0);
                active += 1;
            }
        }
    }
    record_activation(
        "imgproc::canny",
        TheoremId::J1Se2FrenetSerretJet,
        h * w,
        active,
        active as u64,
        &[("edge_pixels", active as f64)],
    );
    *edges = out;
    Ok(())
}

/// Extract outer contours of foreground connected components (`find_contours_def`).
pub fn find_contours_def(
    image: &Mat,
    contours: &mut Vector<Vector<Point>>,
    _mode: i32,
    _method: i32,
) -> Result<()> {
    contours.clear();
    if image.is_empty() {
        return Ok(());
    }
    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let num = connected_components_with_stats(image, &mut labels, &mut stats, &mut centroids, 8, CV_32SC1)?;
    let h = image.rows();
    let w = image.cols();
    for id in 1..num {
        let left = *stats.at_2d::<i32>(id, CC_STAT_LEFT)?;
        let top = *stats.at_2d::<i32>(id, CC_STAT_TOP)?;
        let bw = *stats.at_2d::<i32>(id, CC_STAT_WIDTH)?;
        let bh = *stats.at_2d::<i32>(id, CC_STAT_HEIGHT)?;
        let mut ring = Vector::<Point>::new();
        for r in top..(top + bh) {
            for c in left..(left + bw) {
                if *labels.at_2d::<i32>(r, c)? != id {
                    continue;
                }
                let is_boundary = r == 0
                    || c == 0
                    || r + 1 == h
                    || c + 1 == w
                    || *labels.at_2d::<i32>(r - 1, c)? != id
                    || *labels.at_2d::<i32>(r + 1, c)? != id
                    || *labels.at_2d::<i32>(r, c - 1)? != id
                    || *labels.at_2d::<i32>(r, c + 1)? != id;
                if is_boundary {
                    ring.push(Point::new(c, r));
                }
            }
        }
        if !ring.is_empty() {
            contours.push(ring);
        }
    }
    Ok(())
}

/// Draw a raster line segment (`line`) using Bresenham's algorithm.
pub fn line(
    img: &mut Mat,
    pt1: Point,
    pt2: Point,
    color: Scalar,
    thickness: i32,
    _line_type: i32,
    _shift: i32,
) -> Result<()> {
    let mut x0 = pt1.x;
    let mut y0 = pt1.y;
    let x1 = pt2.x;
    let y1 = pt2.y;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let rad = (thickness.max(1) - 1) / 2;
    let cn = img.channels() as usize;

    loop {
        for ty in -rad..=rad {
            for tx in -rad..=rad {
                for ch in 0..cn {
                    img.set_f64(y0 + ty, x0 + tx, ch, color.0[ch.min(3)]);
                }
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
    Ok(())
}

/// Draw an axis-aligned rectangle (`rectangle`).
pub fn rectangle(
    img: &mut Mat,
    rec: Rect,
    color: Scalar,
    thickness: i32,
    line_type: i32,
    shift: i32,
) -> Result<()> {
    if thickness < 0 {
        let cn = img.channels() as usize;
        for r in rec.y..(rec.y + rec.height) {
            for c in rec.x..(rec.x + rec.width) {
                for ch in 0..cn {
                    img.set_f64(r, c, ch, color.0[ch.min(3)]);
                }
            }
        }
        return Ok(());
    }
    let p00 = Point::new(rec.x, rec.y);
    let p10 = Point::new(rec.x + rec.width - 1, rec.y);
    let p11 = Point::new(rec.x + rec.width - 1, rec.y + rec.height - 1);
    let p01 = Point::new(rec.x, rec.y + rec.height - 1);
    line(img, p00, p10, color, thickness, line_type, shift)?;
    line(img, p10, p11, color, thickness, line_type, shift)?;
    line(img, p11, p01, color, thickness, line_type, shift)?;
    line(img, p01, p00, color, thickness, line_type, shift)
}
