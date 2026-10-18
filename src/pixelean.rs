//! Pillar I: 2D Pixelean Switching, Kinematic Objecthood, Construct Factoring & Spectral Operators.
//!
//! Directly implements Theorems 1-8 of `solidSF.Unified.PixeleanVoxeleanCalculus` plus
//! the `J1` SE(2) Frenet-Serret Jet-Space and `J2` `sfFFT` Spectral Phase Correlation kernels.

use crate::core::{Mat, MatTraitConst, Result};
use crate::mechinterp::{
    current_ablation_config, record_activation, record_circuit_node, TheoremId,
};
use serde::{Deserialize, Serialize};

/// Theorem 1: Bitplane Partition & Conservative Mass Transfer (`M_fg + M_bg = N`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitplanePartition {
    pub width: usize,
    pub height: usize,
    /// 8 intensity bitplanes (`b = 0..8`), each packed in 64-bit words.
    pub bitplanes: [Vec<u64>; 8],
    pub fg_words: Vec<u64>,
    pub fg_mass: usize,
    pub bg_mass: usize,
}

impl BitplanePartition {
    /// Decompose a grayscale `Mat` into 8 packed bitplanes and a foreground threshold mask.
    pub fn from_gray_mat(gray: &Mat, threshold: u8) -> Result<Self> {
        let w = gray.cols().max(0) as usize;
        let h = gray.rows().max(0) as usize;
        let n = w * h;
        let n_words = (n + 63) / 64;
        let mut bitplanes: [Vec<u64>; 8] = std::array::from_fn(|_| vec![0u64; n_words]);
        let mut fg_words = vec![0u64; n_words];
        let ablation = current_ablation_config();

        for r in 0..h {
            for c in 0..w {
                let idx = r * w + c;
                let word = idx / 64;
                let bit = idx % 64;
                let mut val = gray.get_f64(r as i32, c as i32, 0).round().clamp(0.0, 255.0) as u8;
                for &ab in &ablation.ablated_bitplanes {
                    if ab < 8 {
                        val &= !(1u8 << ab);
                    }
                }
                for b in 0..8 {
                    if (val >> b) & 1 != 0 {
                        bitplanes[b][word] |= 1u64 << bit;
                    }
                }
                if val > threshold {
                    fg_words[word] |= 1u64 << bit;
                }
            }
        }

        let fg_mass: usize = fg_words.iter().map(|w| w.count_ones() as usize).sum();
        let bg_mass = n.saturating_sub(fg_mass);
        let hash = fg_words
            .iter()
            .enumerate()
            .fold(0x9e3779b97f4a7c15u64, |acc, (i, &w)| {
                acc ^ w.wrapping_mul((i as u64 + 1).wrapping_mul(0xbf58476d1ce4e5b9))
            });

        record_activation(
            "pixelean::BitplanePartition::from_gray_mat",
            TheoremId::Thm1BitplaneMassConservation,
            n,
            fg_mass,
            hash,
            &[
                ("m_fg", fg_mass as f64),
                ("m_bg", bg_mass as f64),
                ("mass_sum_exact", (fg_mass + bg_mass == n) as u8 as f64),
            ],
        );
        record_circuit_node(
            "BitplanePartition",
            "fg_bg_partition",
            hash,
            fg_mass as f64,
            vec![],
        );

        Ok(Self {
            width: w,
            height: h,
            bitplanes,
            fg_words,
            fg_mass,
            bg_mass,
        })
    }

    /// Theorem 1 invariant: conservative transfer `(fg' - fg) + (bg' - bg) == 0`.
    pub fn conservative_transfer_delta(&self, next: &Self) -> (isize, isize) {
        let dfg = next.fg_mass as isize - self.fg_mass as isize;
        let dbg = next.bg_mass as isize - self.bg_mass as isize;
        debug_assert_eq!(dfg + dbg, 0);
        (dfg, dbg)
    }

    /// Reconstruct an 8-bit grayscale vector after any bitplane ablation.
    pub fn reconstruct_u8(&self) -> Vec<u8> {
        let n = self.width * self.height;
        let mut out = vec![0u8; n];
        for idx in 0..n {
            let word = idx / 64;
            let bit = idx % 64;
            let mut v = 0u8;
            for b in 0..8 {
                if (self.bitplanes[b][word] >> bit) & 1 != 0 {
                    v |= 1u8 << b;
                }
            }
            out[idx] = v;
        }
        out
    }
}

/// Theorem 2: Temporal XOR Minterm Sparsity & Carrier Compression (`\Delta_t = F_t \oplus F_{t-1}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XorMintermCarrier {
    pub width: usize,
    pub height: usize,
    pub diff_words: Vec<u64>,
    pub k_diff: usize,
    pub total_pixels: usize,
}

impl XorMintermCarrier {
    /// Compute the temporal XOR minterm carrier between two bitplane partitions or masks.
    pub fn from_partitions(prev: &BitplanePartition, curr: &BitplanePartition) -> Self {
        let n = prev.width * prev.height;
        let diff_words: Vec<u64> = prev
            .fg_words
            .iter()
            .zip(&curr.fg_words)
            .map(|(&a, &b)| a ^ b)
            .collect();
        let k_diff: usize = diff_words.iter().map(|w| w.count_ones() as usize).sum();
        let hash = diff_words
            .iter()
            .fold(0x517cc1b727220a95u64, |acc, &w| acc.rotate_left(7) ^ w);

        record_activation(
            "pixelean::XorMintermCarrier::from_partitions",
            TheoremId::Thm2TemporalXorMintermSparsity,
            n,
            k_diff,
            hash,
            &[
                ("k_diff", k_diff as f64),
                (
                    "linear_ratio",
                    if n > 0 { k_diff as f64 / n as f64 } else { 0.0 },
                ),
                (
                    "quadratic_ratio",
                    if n > 0 {
                        (k_diff as f64 * k_diff as f64) / (n as f64 * n as f64)
                    } else {
                        0.0
                    },
                ),
            ],
        );
        Self {
            width: prev.width,
            height: prev.height,
            diff_words,
            k_diff,
            total_pixels: n,
        }
    }

    /// Compute a tile-level active mask from two grayscale `Mat` frames (`|I_t - I_{t-1}| > tol`)
    /// using hardware NEON / SWAR 64-pixel vector unrolling.
    pub fn from_gray_pair(prev: &Mat, curr: &Mat, diff_tol: u8) -> Self {
        let w = prev.cols().min(curr.cols()).max(0) as usize;
        let h = prev.rows().min(curr.rows()).max(0) as usize;
        let n = w * h;
        let n_words = (n + 63) / 64;
        let (diff_words, k_diff) =
            if let (Ok(pa), Ok(pb)) = (prev.data_bytes(), curr.data_bytes()) {
                let limit = n.min(pa.len()).min(pb.len());
                crate::hardware::simd_xor_diff_carrier_u8(&pa[..limit], &pb[..limit], diff_tol)
            } else {
                (vec![0u64; n_words], 0usize)
            };
        record_activation(
            "pixelean::XorMintermCarrier::from_gray_pair",
            TheoremId::Thm2TemporalXorMintermSparsity,
            n,
            k_diff,
            k_diff as u64,
            &[(
                "k_diff_over_n",
                if n > 0 { k_diff as f64 / n as f64 } else { 0.0 },
            )],
        );
        Self {
            width: w,
            height: h,
            diff_words,
            k_diff,
            total_pixels: n,
        }
    }

    /// Check if any pixel within `[x - r ..= x + r] x [y - r ..= y + r]` changed in the XOR carrier
    /// using 64-bit SWAR word masks per row.
    #[inline]
    pub fn patch_has_activity(&self, cx: i32, cy: i32, radius: i32) -> bool {
        if self.k_diff == 0 || self.width == 0 || self.height == 0 {
            return false;
        }
        let y0 = (cy - radius).max(0) as usize;
        let y1 = (cy + radius).min(self.height as i32 - 1).max(0) as usize;
        let x0 = (cx - radius).max(0) as usize;
        let x1 = (cx + radius).min(self.width as i32 - 1).max(0) as usize;
        for y in y0..=y1 {
            let start_bit = y * self.width + x0;
            let end_bit = y * self.width + x1;
            let w0 = start_bit >> 6;
            let b0 = start_bit & 63;
            let w1 = end_bit >> 6;
            let b1 = end_bit & 63;
            if w0 == w1 {
                let mask = (u64::MAX << b0) & (u64::MAX >> (63 - b1));
                if (self.diff_words[w0] & mask) != 0 {
                    return true;
                }
            } else {
                let first_mask = u64::MAX << b0;
                if (self.diff_words[w0] & first_mask) != 0 {
                    return true;
                }
                for w_idx in (w0 + 1)..w1 {
                    if self.diff_words[w_idx] != 0 {
                        return true;
                    }
                }
                let last_mask = u64::MAX >> (63 - b1);
                if (self.diff_words[w1] & last_mask) != 0 {
                    return true;
                }
            }
        }
        false
    }
}

/// Theorem 3: Shannon Cofactor Expansion & Convex Envelope Bound.
/// `\Phi = p * \Phi_{pos} + (1 - p) * \Phi_{neg}` lies strictly inside `[\min, \max]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShannonCofactorEnvelope {
    pub p: f64,
    pub phi_pos: f64,
    pub phi_neg: f64,
    pub phi_combined: f64,
}

impl ShannonCofactorEnvelope {
    pub fn evaluate(p: f64, phi_pos: f64, phi_neg: f64) -> Self {
        let p_clamped = p.clamp(0.0, 1.0);
        let phi_combined = p_clamped * phi_pos + (1.0 - p_clamped) * phi_neg;
        record_activation(
            "pixelean::ShannonCofactorEnvelope::evaluate",
            TheoremId::Thm3ShannonCofactorEnvelope,
            2,
            1,
            phi_combined.to_bits(),
            &[
                ("p", p_clamped),
                ("phi_pos", phi_pos),
                ("phi_neg", phi_neg),
                ("phi_combined", phi_combined),
            ],
        );
        Self {
            p: p_clamped,
            phi_pos,
            phi_neg,
            phi_combined,
        }
    }

    pub fn in_convex_hull(&self) -> bool {
        let lo = self.phi_pos.min(self.phi_neg) - 1e-12;
        let hi = self.phi_pos.max(self.phi_neg) + 1e-12;
        self.phi_combined >= lo && self.phi_combined <= hi
    }
}

/// Theorem 4: Kinematic Common-Fate Composite Assembly (`A_total = A_core \uplus A_comp`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonFateComposite {
    pub area_core: f64,
    pub area_comp: f64,
    pub velocity: [f64; 2],
    pub iou_monochrome: f64,
    pub iou_composite: f64,
}

impl CommonFateComposite {
    pub fn assemble(area_core: f64, area_comp: f64, velocity: [f64; 2]) -> Self {
        let total = (area_core + area_comp).max(1e-9);
        let iou_monochrome = area_core / total;
        let iou_composite = 1.0;
        record_activation(
            "pixelean::CommonFateComposite::assemble",
            TheoremId::Thm4KinematicCommonFateComposite,
            total as usize,
            area_core as usize,
            total.to_bits(),
            &[
                ("iou_monochrome", iou_monochrome),
                ("iou_composite", iou_composite),
                ("iou_gain", iou_composite - iou_monochrome),
            ],
        );
        Self {
            area_core,
            area_comp,
            velocity,
            iou_monochrome,
            iou_composite,
        }
    }
}

/// Theorem 5: Dihedral $D_4 = \mathbb{Z}_4 \rtimes \mathbb{Z}_2$ Symmetry Element (8 elements).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct D4Element {
    /// Quarter-turns mod 4 (`0..4`).
    pub rot: u8,
    /// Horizontal reflection (`false` or `true`).
    pub reflect: bool,
}

impl D4Element {
    pub const ALL: [D4Element; 8] = [
        Self {
            rot: 0,
            reflect: false,
        },
        Self {
            rot: 1,
            reflect: false,
        },
        Self {
            rot: 2,
            reflect: false,
        },
        Self {
            rot: 3,
            reflect: false,
        },
        Self {
            rot: 0,
            reflect: true,
        },
        Self {
            rot: 1,
            reflect: true,
        },
        Self {
            rot: 2,
            reflect: true,
        },
        Self {
            rot: 3,
            reflect: true,
        },
    ];

    /// Group composition in $D_4 = \mathbb{Z}_4 \rtimes \mathbb{Z}_2$.
    pub const fn compose(self, other: Self) -> Self {
        let shift = if self.reflect {
            (4 - (other.rot % 4)) % 4
        } else {
            other.rot % 4
        };
        Self {
            rot: (self.rot + shift) % 4,
            reflect: self.reflect ^ other.reflect,
        }
    }

    pub const fn inverse(self) -> Self {
        if self.reflect {
            self
        } else {
            Self {
                rot: (4 - (self.rot % 4)) % 4,
                reflect: false,
            }
        }
    }

    /// Map `(r, c)` inside a square `n x n` patch under this $D_4$ element.
    #[inline]
    pub const fn map_rc(self, r: usize, c: usize, n: usize) -> (usize, usize) {
        let (r0, c0) = if self.reflect { (r, n - 1 - c) } else { (r, c) };
        match self.rot % 4 {
            0 => (r0, c0),
            1 => (c0, n - 1 - r0),
            2 => (n - 1 - r0, n - 1 - c0),
            _ => (n - 1 - c0, r0),
        }
    }
}

/// Theorem 6: Coupled Mirror-Twin Jacobian (`J = diag(s_r, s_c)`, `s_r^2 = s_c^2 = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirrorTwinJacobian {
    pub s_r: i32,
    pub s_c: i32,
}

impl MirrorTwinJacobian {
    pub fn new(s_r: i32, s_c: i32) -> Self {
        let sr = if s_r < 0 { -1 } else { 1 };
        let sc = if s_c < 0 { -1 } else { 1 };
        Self { s_r: sr, s_c: sc }
    }

    #[inline]
    pub const fn det(&self) -> i32 {
        self.s_r * self.s_c
    }

    #[inline]
    pub const fn transform_delta(&self, dr: i32, dc: i32) -> (i32, i32) {
        (self.s_r * dr, self.s_c * dc)
    }
}

/// Theorem 7: Visual Field Construct Factoring (`(1 + x_1)(1 + x_2) -> 1 + x_1 + x_2`).
#[inline]
pub fn construct_factor_savings(x1: u64, x2: u64) -> (u64, u64, u64) {
    let unfactored = (1 + x1) * (1 + x2);
    let factored = 1 + x1 + x2;
    let saved_cross_term = x1 * x2;
    record_activation(
        "pixelean::construct_factor_savings",
        TheoremId::Thm7ConstructFactoring,
        unfactored as usize,
        factored as usize,
        saved_cross_term,
        &[("saved_cross_term", saved_cross_term as f64)],
    );
    (unfactored, factored, saved_cross_term)
}

/// Theorem 8: 1px-Gridded Logical Lattice Stride Quotient Isomorphism.
/// `R = (H(s) + gap) / (s + gap)` where `H(s) = R * s + (R - 1) * gap`.
#[inline]
pub fn logical_lattice_quotient(pixel_extent: usize, cell_size: usize, gap: usize) -> Option<usize> {
    let stride = cell_size + gap;
    if stride == 0 || (pixel_extent + gap) % stride != 0 {
        return None;
    }
    let logical_r = (pixel_extent + gap) / stride;
    record_activation(
        "pixelean::logical_lattice_quotient",
        TheoremId::Thm8LogicalLatticeQuotient,
        pixel_extent * pixel_extent,
        logical_r * logical_r,
        logical_r as u64,
        &[("stride", stride as f64), ("logical_r", logical_r as f64)],
    );
    Some(logical_r)
}

/// Operator `J1`: $\operatorname{SE}(2)$ Frenet-Serret Jet-Space Sample `(x, y, cos_theta, sin_theta, kappa)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Se2JetSample {
    pub x: f64,
    pub y: f64,
    pub cos_theta: f64,
    pub sin_theta: f64,
    pub curvature_kappa: f64,
    pub gradient_mag: f64,
}

/// Lift a 2D grayscale `Mat` into $\operatorname{SE}(2)$ Frenet-Serret Jet-Space (`Operator J1`).
pub fn lift_se2_frenet_serret_jets(gray: &Mat, grad_threshold: f64) -> Vec<Se2JetSample> {
    let h = gray.rows() as usize;
    let w = gray.cols() as usize;
    if h < 3 || w < 3 {
        return Vec::new();
    }
    let mut jets = Vec::new();
    for r in 1..(h - 1) {
        for c in 1..(w - 1) {
            let ri = r as i32;
            let ci = c as i32;
            let raw_gx = (gray.get_f64(ri - 1, ci + 1, 0)
                + 2.0 * gray.get_f64(ri, ci + 1, 0)
                + gray.get_f64(ri + 1, ci + 1, 0))
                - (gray.get_f64(ri - 1, ci - 1, 0)
                    + 2.0 * gray.get_f64(ri, ci - 1, 0)
                    + gray.get_f64(ri + 1, ci - 1, 0));
            let raw_gy = (gray.get_f64(ri + 1, ci - 1, 0)
                + 2.0 * gray.get_f64(ri + 1, ci, 0)
                + gray.get_f64(ri + 1, ci + 1, 0))
                - (gray.get_f64(ri - 1, ci - 1, 0)
                    + 2.0 * gray.get_f64(ri - 1, ci, 0)
                    + gray.get_f64(ri - 1, ci + 1, 0));
            let raw_mag = (raw_gx * raw_gx + raw_gy * raw_gy).sqrt();
            if raw_mag < grad_threshold {
                continue;
            }
            // Exact pixel-step first derivatives (Sobel kernel sum is 8)
            let ix = raw_gx * 0.125;
            let iy = raw_gy * 0.125;
            let mag = raw_mag * 0.125;
            // Unit tangent vector (-iy, ix) / |grad I|
            let cos_theta = -iy / mag;
            let sin_theta = ix / mag;
            // Second derivatives for level-set curvature kappa = (I_xx I_y^2 - 2 I_xy I_x I_y + I_yy I_x^2) / |grad I|^3
            let ic = gray.get_f64(ri, ci, 0);
            let ixx = gray.get_f64(ri, ci + 1, 0) - 2.0 * ic + gray.get_f64(ri, ci - 1, 0);
            let iyy = gray.get_f64(ri + 1, ci, 0) - 2.0 * ic + gray.get_f64(ri - 1, ci, 0);
            let ixy = 0.25
                * (gray.get_f64(ri + 1, ci + 1, 0)
                    - gray.get_f64(ri + 1, ci - 1, 0)
                    - gray.get_f64(ri - 1, ci + 1, 0)
                    + gray.get_f64(ri - 1, ci - 1, 0));
            let denom = (ix * ix + iy * iy).powf(1.5).max(1e-9);
            let kappa = (ixx * iy * iy - 2.0 * ixy * ix * iy + iyy * ix * ix) / denom;
            jets.push(Se2JetSample {
                x: c as f64,
                y: r as f64,
                cos_theta,
                sin_theta,
                curvature_kappa: kappa,
                gradient_mag: raw_mag,
            });
        }
    }
    record_activation(
        "pixelean::lift_se2_frenet_serret_jets",
        TheoremId::J1Se2FrenetSerretJet,
        w * h,
        jets.len(),
        jets.len() as u64,
        &[("jet_count", jets.len() as f64)],
    );
    jets
}

/// Operator `J2`: Pure-Rust Radix-2 2D `sfFFT` Spectral Phase Correlation `(dx, dy, peak_response)`.
/// Recovers integer and sub-pixel translations `(dx, dy)` using zero-mean Hann windowing and
/// cyclic parabolic sub-pixel peak interpolation.
pub fn sffft_phase_correlation(prev: &Mat, curr: &Mat) -> Result<(f64, f64, f64)> {
    let h = prev.rows().min(curr.rows()).max(1) as usize;
    let w = prev.cols().min(curr.cols()).max(1) as usize;
    let ny = h.next_power_of_two().min(256);
    let nx = w.next_power_of_two().min(256);

    let eff_h = ny.min(h);
    let eff_w = nx.min(w);
    let n_pix = (eff_h * eff_w).max(1) as f64;
    let mut mean_a = 0.0f64;
    let mut mean_b = 0.0f64;
    for r in 0..eff_h {
        for c in 0..eff_w {
            mean_a += prev.get_f64(r as i32, c as i32, 0);
            mean_b += curr.get_f64(r as i32, c as i32, 0);
        }
    }
    mean_a /= n_pix;
    mean_b /= n_pix;

    let mut fa = vec![(0.0f64, 0.0f64); ny * nx];
    let mut fb = vec![(0.0f64, 0.0f64); ny * nx];
    for r in 0..eff_h {
        let wy = if eff_h > 4 {
            let s = (std::f64::consts::PI * (r as f64 + 0.5) / (eff_h as f64)).sin();
            0.25 + 0.75 * s * s
        } else {
            1.0
        };
        for c in 0..eff_w {
            let wx = if eff_w > 4 {
                let s = (std::f64::consts::PI * (c as f64 + 0.5) / (eff_w as f64)).sin();
                0.25 + 0.75 * s * s
            } else {
                1.0
            };
            let w_hann = wx * wy;
            fa[r * nx + c] = ((prev.get_f64(r as i32, c as i32, 0) - mean_a) * w_hann, 0.0);
            fb[r * nx + c] = ((curr.get_f64(r as i32, c as i32, 0) - mean_b) * w_hann, 0.0);
        }
    }
    fft2d_in_place(&mut fa, ny, nx, false);
    fft2d_in_place(&mut fb, ny, nx, false);

    // Cross-power spectrum R = (F_a^* * F_b) / |F_a^* * F_b|
    let mut cross = vec![(0.0f64, 0.0f64); ny * nx];
    for i in 0..(ny * nx) {
        let (ar, ai) = fa[i];
        let (br, bi) = fb[i];
        // conj(A) * B = (ar - i ai)(br + i bi) = (ar*br + ai*bi) + i(ar*bi - ai*br)
        let cr = ar * br + ai * bi;
        let ci = ar * bi - ai * br;
        let mag = (cr * cr + ci * ci).sqrt();
        if mag > 1e-12 {
            cross[i] = (cr / mag, ci / mag);
        }
    }
    fft2d_in_place(&mut cross, ny, nx, true);

    let mut best_val = -1.0f64;
    let mut best_r = 0usize;
    let mut best_c = 0usize;
    for r in 0..ny {
        for c in 0..nx {
            let v = cross[r * nx + c].0;
            if v > best_val {
                best_val = v;
                best_r = r;
                best_c = c;
            }
        }
    }

    // Cyclic sub-pixel peak refinement around (best_r, best_c)
    let sub_r = if ny >= 4 {
        let r_prev = (best_r + ny - 1) % ny;
        let r_next = (best_r + 1) % ny;
        let v_prev = cross[r_prev * nx + best_c].0.max(0.0);
        let v_curr = best_val.max(1e-12);
        let v_next = cross[r_next * nx + best_c].0.max(0.0);
        let denom = v_prev + v_curr + v_next;
        if denom > 1e-9 {
            ((v_next - v_prev) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let sub_c = if nx >= 4 {
        let c_prev = (best_c + nx - 1) % nx;
        let c_next = (best_c + 1) % nx;
        let v_prev = cross[best_r * nx + c_prev].0.max(0.0);
        let v_curr = best_val.max(1e-12);
        let v_next = cross[best_r * nx + c_next].0.max(0.0);
        let denom = v_prev + v_curr + v_next;
        if denom > 1e-9 {
            ((v_next - v_prev) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let base_dy = if best_r > ny / 2 {
        best_r as f64 - ny as f64
    } else {
        best_r as f64
    };
    let base_dx = if best_c > nx / 2 {
        best_c as f64 - nx as f64
    } else {
        best_c as f64
    };
    let dy = base_dy + sub_r;
    let dx = base_dx + sub_c;

    record_activation(
        "pixelean::sffft_phase_correlation",
        TheoremId::J2SfFftPhaseCorrelation,
        ny * nx,
        1,
        best_val.to_bits(),
        &[("dx", dx), ("dy", dy), ("peak", best_val)],
    );
    Ok((dx, dy, best_val))
}

fn fft1d_in_place(buf: &mut [(f64, f64)], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    let mut len = 2;
    let sign = if inverse { 1.0 } else { -1.0 };
    while len <= n {
        let half = len / 2;
        let ang = sign * 2.0 * std::f64::consts::PI / (len as f64);
        let wlen = (ang.cos(), ang.sin());
        for i in (0..n).step_by(len) {
            let mut w = (1.0, 0.0);
            for k in 0..half {
                let u = buf[i + k];
                let v = buf[i + k + half];
                let tv = (v.0 * w.0 - v.1 * w.1, v.0 * w.1 + v.1 * w.0);
                buf[i + k] = (u.0 + tv.0, u.1 + tv.1);
                buf[i + k + half] = (u.0 - tv.0, u.1 - tv.1);
                w = (w.0 * wlen.0 - w.1 * wlen.1, w.0 * wlen.1 + w.1 * wlen.0);
            }
        }
        len <<= 1;
    }
    if inverse {
        let scale = 1.0 / (n as f64);
        for x in buf.iter_mut() {
            x.0 *= scale;
            x.1 *= scale;
        }
    }
}

fn fft2d_in_place(grid: &mut [(f64, f64)], rows: usize, cols: usize, inverse: bool) {
    for r in 0..rows {
        fft1d_in_place(&mut grid[r * cols..(r + 1) * cols], inverse);
    }
    let mut col_buf = vec![(0.0, 0.0); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = grid[r * cols + c];
        }
        fft1d_in_place(&mut col_buf, inverse);
        for r in 0..rows {
            grid[r * cols + c] = col_buf[r];
        }
    }
}
