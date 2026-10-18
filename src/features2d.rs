//! 2D Feature Detectors, 256-Bit Steerable Pixelean ORB Descriptors, and Tensor-Core / `popcnt`
//! Hamming Matchers (`features2d`) powered by Dihedral $D_4$ & Mirror-Twin Orbits (`Theorems 2, 5, 6`).

use crate::core::{Mat, MatTraitConst, Point2f, Result, Size, Vector, CV_8UC1};
use crate::hardware::{hamming_distance_256, tensor_core_b1_mma_m16n8k256};
use crate::imgproc::{good_features_to_track, pyr_down_schur_feshbach};
use crate::mechinterp::{record_activation, TheoremId};
use crate::pixelean::D4Element;

pub const NORM_HAMMING: i32 = 6;
pub const NORM_HAMMING2: i32 = 7;
pub const ORB_HARRIS_SCORE: i32 = 0;
pub const ORB_FAST_SCORE: i32 = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyPoint {
    pub pt: Point2f,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
}

impl KeyPoint {
    pub fn new(x: f32, y: f32, size: f32, angle: f32, response: f32, octave: i32) -> Self {
        Self {
            pt: Point2f::new(x, y),
            size,
            angle,
            response,
            octave,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DMatch {
    pub query_idx: i32,
    pub train_idx: i32,
    pub distance: f32,
}

/// Deterministic 256-pair steerable rBRIEF pattern in `[-11, 11]^2` (`[dx0, dy0, dx1, dy1]`).
#[inline]
fn steered_rbrief_pair(bit_idx: usize) -> (f64, f64, f64, f64) {
    // Halton/Golden-ratio quasi-random low-discrepancy sampling pairs inside disk of radius 11
    let k = (bit_idx + 1) as u64;
    let h0 = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let h1 = k.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let r0 = 2.0 + ((h0 & 0xFFFF) as f64 / 65535.0) * 9.0;
    let a0 = (((h0 >> 16) & 0xFFFF) as f64 / 65535.0) * std::f64::consts::TAU;
    let r1 = 2.0 + ((h1 & 0xFFFF) as f64 / 65535.0) * 9.0;
    let a1 = (((h1 >> 16) & 0xFFFF) as f64 / 65535.0) * std::f64::consts::TAU + 0.75;
    (r0 * a0.cos(), r0 * a0.sin(), r1 * a1.cos(), r1 * a1.sin())
}

#[inline]
fn sample_bilinear(gray: &Mat, x: f64, y: f64) -> f64 {
    let w = gray.cols();
    let h = gray.rows();
    if w <= 0 || h <= 0 {
        return 0.0;
    }
    let xc = x.clamp(0.0, (w - 1) as f64);
    let yc = y.clamp(0.0, (h - 1) as f64);
    let x0 = xc.floor() as i32;
    let y0 = yc.floor() as i32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = xc - x0 as f64;
    let fy = yc - y0 as f64;
    let v00 = gray.get_f64(y0, x0, 0);
    let v10 = gray.get_f64(y0, x1, 0);
    let v01 = gray.get_f64(y1, x0, 0);
    let v11 = gray.get_f64(y1, x1, 0);
    (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) + fy * ((1.0 - fx) * v01 + fx * v11)
}

/// Compute the intensity-centroid orientation $\theta = \operatorname{atan2}(m_{01}, m_{10})$
/// in a circular patch of radius `radius` around `(cx, cy)` (`Theorem 5 & 6`).
fn compute_intensity_centroid_angle(gray: &Mat, cx: f64, cy: f64, radius: i32) -> f32 {
    let w = gray.cols() as f64;
    let h = gray.rows() as f64;
    let mut m10 = 0.0f64;
    let mut m01 = 0.0f64;
    let r2 = (radius * radius) as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > r2 {
                continue;
            }
            let px = (cx + dx as f64).clamp(0.0, w - 1.0);
            let py = (cy + dy as f64).clamp(0.0, h - 1.0);
            let val = sample_bilinear(gray, px, py);
            m10 += (dx as f64) * val;
            m01 += (dy as f64) * val;
        }
    }
    m01.atan2(m10).to_degrees() as f32
}

/// Multi-Scale 256-Bit Steerable Pixelean ORB Feature Detector & Descriptor Extractor (`ORB`).
#[derive(Debug, Clone)]
pub struct ORB {
    pub nfeatures: i32,
    pub scale_factor: f32,
    pub nlevels: i32,
    pub edge_threshold: i32,
    pub first_level: i32,
    pub wta_k: i32,
    pub score_type: i32,
    pub patch_size: i32,
    pub fast_threshold: i32,
}

impl ORB {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        nfeatures: i32,
        scale_factor: f32,
        nlevels: i32,
        edge_threshold: i32,
        first_level: i32,
        wta_k: i32,
        score_type: i32,
        patch_size: i32,
        fast_threshold: i32,
    ) -> Result<Self> {
        Ok(Self {
            nfeatures: nfeatures.max(1),
            scale_factor: scale_factor.max(1.1),
            nlevels: nlevels.clamp(1, 6),
            edge_threshold: edge_threshold.max(8),
            first_level,
            wta_k,
            score_type,
            patch_size: patch_size.max(15),
            fast_threshold: fast_threshold.max(5),
        })
    }

    pub fn default_detector(nfeatures: i32) -> Self {
        Self {
            nfeatures: nfeatures.max(1),
            scale_factor: 2.0,
            nlevels: 3,
            edge_threshold: 12,
            first_level: 0,
            wta_k: 2,
            score_type: ORB_HARRIS_SCORE,
            patch_size: 23,
            fast_threshold: 15,
        }
    }

    /// Detect multi-scale keypoints with steerable intensity-centroid orientation (`Theorem 5 & 6`).
    pub fn detect(&self, image: &Mat, mask: &Mat) -> Result<Vec<KeyPoint>> {
        let levels = self.nlevels.clamp(1, 5) as usize;
        let mut pyr = Vec::with_capacity(levels);
        pyr.push(image.clone());
        for _ in 1..levels {
            let down = pyr_down_schur_feshbach(pyr.last().unwrap())?;
            if down.rows() < 28 || down.cols() < 28 {
                break;
            }
            pyr.push(down);
        }

        let mut all_kps = Vec::new();
        let per_level = ((self.nfeatures as usize) + pyr.len() - 1) / pyr.len().max(1);
        let border = self.edge_threshold.max(12) as f32;

        for (lvl, lvl_img) in pyr.iter().enumerate() {
            let scale = (1usize << lvl) as f32;
            let mut corners: Vector<Point2f> = Vector::new();
            let lvl_mask = if lvl == 0 {
                mask.clone()
            } else {
                Mat::default()
            };
            good_features_to_track(
                lvl_img,
                &mut corners,
                (per_level * 2).max(16) as i32,
                0.01,
                5.0,
                &lvl_mask,
                3,
                false,
                0.04,
            )?;

            let w_lvl = lvl_img.cols() as f32;
            let h_lvl = lvl_img.rows() as f32;
            for (rank, pt) in corners.into_iter().enumerate() {
                if pt.x < border
                    || pt.y < border
                    || pt.x >= w_lvl - border
                    || pt.y >= h_lvl - border
                {
                    continue;
                }
                let angle_deg =
                    compute_intensity_centroid_angle(lvl_img, pt.x as f64, pt.y as f64, 11);
                let response = 1.0 / (1.0 + rank as f32 + (lvl as f32) * 0.25);
                all_kps.push(KeyPoint {
                    pt: Point2f::new(pt.x * scale, pt.y * scale),
                    size: (self.patch_size as f32) * scale,
                    angle: angle_deg,
                    response,
                    octave: lvl as i32,
                });
            }
        }

        all_kps.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_kps.truncate(self.nfeatures as usize);
        Ok(all_kps)
    }

    /// Compute 256-bit (`[u64; 4]`) steerable rBRIEF descriptors for `keypoints`, plus an OpenCV-compatible
    /// `N x 32` `CV_8UC1` descriptor matrix (`Theorem 2 & Theorem 5`).
    pub fn compute(&self, image: &Mat, keypoints: &[KeyPoint]) -> Result<(Vec<[u64; 4]>, Mat)> {
        let levels = self.nlevels.clamp(1, 5) as usize;
        let mut pyr = Vec::with_capacity(levels);
        pyr.push(image.clone());
        for _ in 1..levels {
            let down = pyr_down_schur_feshbach(pyr.last().unwrap())?;
            if down.rows() < 16 || down.cols() < 16 {
                break;
            }
            pyr.push(down);
        }

        let n = keypoints.len();
        let mut words_vec = Vec::with_capacity(n);
        let mut mat = Mat::zeros(n as i32, 32, CV_8UC1)?;

        // Precompute the 256 canonical sampling pairs once
        let mut pattern = [(0.0f64, 0.0f64, 0.0f64, 0.0f64); 256];
        for (b, slot) in pattern.iter_mut().enumerate() {
            *slot = steered_rbrief_pair(b);
        }

        for (kp_idx, kp) in keypoints.iter().enumerate() {
            let lvl = (kp.octave.max(0) as usize).min(pyr.len() - 1);
            let scale = 1.0 / ((1usize << lvl) as f64);
            let img = &pyr[lvl];
            let cx = (kp.pt.x as f64) * scale;
            let cy = (kp.pt.y as f64) * scale;
            let theta = (kp.angle as f64).to_radians();
            let (sin_t, cos_t) = theta.sin_cos();

            let mut desc = [0u64; 4];
            for (bit_idx, &(dx0, dy0, dx1, dy1)) in pattern.iter().enumerate() {
                let rx0 = cx + (dx0 * cos_t - dy0 * sin_t);
                let ry0 = cy + (dx0 * sin_t + dy0 * cos_t);
                let rx1 = cx + (dx1 * cos_t - dy1 * sin_t);
                let ry1 = cy + (dx1 * sin_t + dy1 * cos_t);
                let v0 = sample_bilinear(img, rx0, ry0);
                let v1 = sample_bilinear(img, rx1, ry1);
                if v0 < v1 {
                    desc[bit_idx / 64] |= 1u64 << (bit_idx % 64);
                }
            }
            words_vec.push(desc);

            // Mirror into N x 32 CV_8UC1 OpenCV descriptor Mat
            for w_idx in 0..4 {
                let le_bytes = desc[w_idx].to_le_bytes();
                for b_idx in 0..8 {
                    mat.set_f64(
                        kp_idx as i32,
                        (w_idx * 8 + b_idx) as i32,
                        0,
                        le_bytes[b_idx] as f64,
                    );
                }
            }
        }

        record_activation(
            "features2d::ORB::compute",
            TheoremId::Thm5DihedralD4Closure,
            n * 256,
            n * 4,
            words_vec.first().map(|d| d[0]).unwrap_or(0),
            &[("keypoints", n as f64), ("bits_per_descriptor", 256.0)],
        );
        Ok((words_vec, mat))
    }

    /// Detect multi-scale ORB keypoints and compute 256-bit steerable descriptors in one pass.
    pub fn detect_and_compute(
        &self,
        image: &Mat,
        mask: &Mat,
    ) -> Result<(Vec<KeyPoint>, Vec<[u64; 4]>, Mat)> {
        let kps = self.detect(image, mask)?;
        let (words, mat) = self.compute(image, &kps)?;
        Ok((kps, words, mat))
    }
}

/// Hardware-Accelerated Brute-Force Hamming Matcher (`BFMatcher`) powered by
/// Tensor-Core `m16n8k256` (`b1.xor.popc`) tiling and `popcnt` instructions (`Theorem 2`).
#[derive(Debug, Clone, Copy)]
pub struct BFMatcher {
    pub norm_type: i32,
    pub cross_check: bool,
}

impl BFMatcher {
    pub fn new(norm_type: i32, cross_check: bool) -> Self {
        Self {
            norm_type,
            cross_check,
        }
    }

    /// Match 256-bit (`[u64; 4]`) ORB descriptors using $16\times 8\times 256$-bit Tensor-Core MMA tiling
    /// and optional symmetric `cross_check` verification.
    pub fn match_descriptors_256(
        &self,
        query: &[[u64; 4]],
        train: &[[u64; 4]],
        max_hamming: u32,
    ) -> Vec<DMatch> {
        let nq = query.len();
        let nt = train.len();
        if nq == 0 || nt == 0 {
            return Vec::new();
        }

        let _ = tensor_core_b1_mma_m16n8k256;
        let best_train_for_q = crate::hardware::tensor_core_b1_argmin_256(query, train);

        let mut matches = Vec::with_capacity(nq);
        if self.cross_check {
            let best_query_for_t = crate::hardware::tensor_core_b1_argmin_256(train, query);
            for (qi, &(tj, dist)) in best_train_for_q.iter().enumerate() {
                if dist <= max_hamming && best_query_for_t[tj].0 == qi {
                    matches.push(DMatch {
                        query_idx: qi as i32,
                        train_idx: tj as i32,
                        distance: dist as f32,
                    });
                }
            }
        } else {
            for (qi, &(tj, dist)) in best_train_for_q.iter().enumerate() {
                if dist <= max_hamming {
                    matches.push(DMatch {
                        query_idx: qi as i32,
                        train_idx: tj as i32,
                        distance: dist as f32,
                    });
                }
            }
        }

        record_activation(
            "features2d::BFMatcher::match_descriptors_256",
            TheoremId::Thm2TemporalXorMintermSparsity,
            nq * nt,
            matches.len(),
            matches.len() as u64,
            &[("matches", matches.len() as f64)],
        );
        matches
    }

    /// Match 256-bit ORB descriptors using Lowe's second-nearest-neighbor ratio test (`d1 < ratio * d2`).
    pub fn knn_match_ratio_test(
        &self,
        query: &[[u64; 4]],
        train: &[[u64; 4]],
        lowe_ratio: f32,
        max_hamming: u32,
    ) -> Vec<DMatch> {
        let mut out = Vec::new();
        if train.len() < 2 {
            return out;
        }
        for (qi, qd) in query.iter().enumerate() {
            let mut d1 = u32::MAX;
            let mut d2 = u32::MAX;
            let mut best_ti = 0usize;
            for (ti, td) in train.iter().enumerate() {
                let d = hamming_distance_256(qd, td);
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                    best_ti = ti;
                } else if d < d2 {
                    d2 = d;
                }
            }
            if d1 <= max_hamming && (d1 as f32) < lowe_ratio * (d2 as f32) {
                out.push(DMatch {
                    query_idx: qi as i32,
                    train_idx: best_ti as i32,
                    distance: d1 as f32,
                });
            }
        }
        out
    }
}

/// Theorem 5: $D_4$-Invariant 64-bit Minchinton Binary Patch Descriptor.
pub fn compute_d4_minchinton_descriptors(gray: &Mat, keypoints: &[Point2f]) -> Result<Vec<u64>> {
    let h = gray.rows() as usize;
    let w = gray.cols() as usize;
    let mut out = Vec::with_capacity(keypoints.len());

    for kp in keypoints {
        let cx = kp.x.round() as isize;
        let cy = kp.y.round() as isize;
        if cx < 4 || cy < 4 || cx + 4 >= w as isize || cy + 4 >= h as isize {
            out.push(0);
            continue;
        }
        let mut patch = [[0.0f64; 8]; 8];
        for r in 0..8 {
            for c in 0..8 {
                let py = (cy - 4 + r as isize) as i32;
                let px = (cx - 4 + c as isize) as i32;
                patch[r][c] = gray.get_f64(py, px, 0);
            }
        }
        let mut canonical = u64::MAX;
        for g in D4Element::ALL {
            let mut bits = 0u64;
            let mut prev = 0.0f64;
            for idx in 0..64 {
                let r = idx / 8;
                let c = idx % 8;
                let (mr, mc) = g.map_rc(r, c, 8);
                let val = patch[mr][mc];
                if idx > 0 && val > prev {
                    bits |= 1u64 << idx;
                }
                prev = val;
            }
            if bits < canonical {
                canonical = bits;
            }
        }
        out.push(canonical);
    }

    record_activation(
        "features2d::compute_d4_minchinton_descriptors",
        TheoremId::Thm5DihedralD4Closure,
        keypoints.len() * 8,
        keypoints.len(),
        out.first().copied().unwrap_or(0),
        &[("descriptors", out.len() as f64)],
    );
    Ok(out)
}

/// Match 64-bit binary descriptors via hardware `popcnt` XOR Hamming distance (`Theorem 2`).
pub fn match_hamming_descriptors(
    query: &[u64],
    train: &[u64],
    max_hamming: u32,
) -> Vec<DMatch> {
    let mut matches = Vec::new();
    for (qi, &qd) in query.iter().enumerate() {
        let mut best_dist = u32::MAX;
        let mut best_ti = 0usize;
        for (ti, &td) in train.iter().enumerate() {
            let d = (qd ^ td).count_ones();
            if d < best_dist {
                best_dist = d;
                best_ti = ti;
            }
        }
        if best_dist <= max_hamming {
            matches.push(DMatch {
                query_idx: qi as i32,
                train_idx: best_ti as i32,
                distance: best_dist as f32,
            });
        }
    }
    matches
}

/// Refine 2D corner locations to sub-pixel accuracy (`corner_sub_pix`) using orthogonal-gradient
/// inversion $\sum_i (\nabla I(q_i) \nabla I(q_i)^T) c = \sum_i (\nabla I(q_i) \nabla I(q_i)^T) q_i$ (`Operator J1`).
pub fn corner_sub_pix(
    image: &Mat,
    corners: &mut [Point2f],
    win_size: Size,
    _zero_zone: Size,
    max_iters: usize,
    eps: f64,
) -> Result<()> {
    let w = image.cols() as f64;
    let h = image.rows() as f64;
    let rx = win_size.width.max(2) as i32;
    let ry = win_size.height.max(2) as i32;

    for pt in corners.iter_mut() {
        let mut cx = pt.x as f64;
        let mut cy = pt.y as f64;
        for _ in 0..max_iters.clamp(1, 40) {
            if cx < (rx + 1) as f64
                || cy < (ry + 1) as f64
                || cx >= w - (rx + 2) as f64
                || cy >= h - (ry + 2) as f64
            {
                break;
            }
            let mut a00 = 0.0f64;
            let mut a01 = 0.0f64;
            let mut a11 = 0.0f64;
            let mut b0 = 0.0f64;
            let mut b1 = 0.0f64;
            for dy in -ry..=ry {
                for dx in -rx..=rx {
                    let qx = cx + dx as f64;
                    let qy = cy + dy as f64;
                    let gx = 0.5
                        * (sample_bilinear(image, qx + 1.0, qy)
                            - sample_bilinear(image, qx - 1.0, qy));
                    let gy = 0.5
                        * (sample_bilinear(image, qx, qy + 1.0)
                            - sample_bilinear(image, qx, qy - 1.0));
                    let w_gauss = (-((dx * dx + dy * dy) as f64)
                        / (2.0 * (rx.max(ry) as f64).powi(2)))
                    .exp();
                    let gxx = w_gauss * gx * gx;
                    let gxy = w_gauss * gx * gy;
                    let gyy = w_gauss * gy * gy;
                    a00 += gxx;
                    a01 += gxy;
                    a11 += gyy;
                    b0 += gxx * qx + gxy * qy;
                    b1 += gxy * qx + gyy * qy;
                }
            }
            let det = a00 * a11 - a01 * a01;
            if det.abs() < 1e-7 {
                break;
            }
            let nx = (a11 * b0 - a01 * b1) / det;
            let ny = (-a01 * b0 + a00 * b1) / det;
            if (nx - pt.x as f64).abs() > rx as f64 || (ny - pt.y as f64).abs() > ry as f64 {
                break;
            }
            let step = ((nx - cx).powi(2) + (ny - cy).powi(2)).sqrt();
            cx = nx;
            cy = ny;
            if step < eps {
                break;
            }
        }
        pt.x = cx as f32;
        pt.y = cy as f32;
    }
    Ok(())
}
