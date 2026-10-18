//! Hard mathematical, geometric, and adversarial verification suite for `solidcv`.
//!
//! Validates all 24 Pixelean, Voxelean, and Temporal Hyper-Voxelean theorems,
//! operators `J1`, `J2`, `A1`, `A2`, global $SO(3)$ PnP under large rotations and
//! heavy outliers, $3\times 3$ SVD + Hartley-normalized Essential matrix pose recovery,
//! sub-pixel pyramidal optical flow, dynamic-programming StereoSGBM, and
//! mechanistic interpretability (`mechinterp`) counterfactual ablations.

use solidcv::{
    calib3d::{self, StereoSGBM},
    core::{
        self, Mat, Point, Point2f, Point3f, Rect, Scalar, Size, TermCriteria, Vector, CV_64FC1,
        CV_8UC1, CV_8UC3,
    },
    dnn, imgproc,
    mechinterp::{reset_mind, with_counterfactual_ablation, AblationConfig},
    objdetect,
    pixelean::{self, BitplanePartition, D4Element, MirrorTwinJacobian, XorMintermCarrier},
    prelude::*,
    video,
    voxelean::{OhElement, SchurFeshbachOctreeWavelet, StarStencil7PointGfp, VoxelComplex3D},
    worldtube::{
        self, GrandSynthesisCertificate, HyperVoxeleanWavelet4D, UnbentRayTransferOperator,
        WorldtubeTrajectoryFactor,
    },
};
use std::collections::BTreeSet;

#[test]
fn test_01_d4_cayley_table_and_rectangular_multichannel_actions() {
    reset_mind();
    let elems = D4Element::ALL;
    assert_eq!(elems.len(), 8);
    let id = D4Element {
        rot: 0,
        reflect: false,
    };

    // Verify full 8x8 Cayley closure, identity, and inverses
    let mut set = BTreeSet::new();
    for &a in &elems {
        set.insert((a.rot, a.reflect));
        let inv = a.inverse();
        let prod_r = a.compose(inv);
        let prod_l = inv.compose(a);
        assert_eq!(prod_r, id);
        assert_eq!(prod_l, id);
        for &b in &elems {
            let c = a.compose(b);
            assert!(c.rot < 4);
        }
    }
    assert_eq!(set.len(), 8);

    // Test rotate, flip, and transpose on a non-square 11x17 CV_8UC3 matrix
    let mut m = Mat::zeros(11, 17, CV_8UC3).unwrap();
    for r in 0..11 {
        for c in 0..17 {
            m.set_f64(r, c, 0, (r * 13 + c * 7) as f64);
            m.set_f64(r, c, 1, (r * 5 + c * 11) as f64);
            m.set_f64(r, c, 2, (r ^ c) as f64);
        }
    }

    // 4 clockwise 90-degree rotations return to exact original 11x17 matrix
    let mut r1 = Mat::default();
    let mut r2 = Mat::default();
    let mut r3 = Mat::default();
    let mut r4 = Mat::default();
    core::rotate(&m, &mut r1, core::ROTATE_90_CLOCKWISE).unwrap();
    assert_eq!((r1.rows(), r1.cols()), (17, 11));
    core::rotate(&r1, &mut r2, core::ROTATE_90_CLOCKWISE).unwrap();
    assert_eq!((r2.rows(), r2.cols()), (11, 17));
    core::rotate(&r2, &mut r3, core::ROTATE_90_CLOCKWISE).unwrap();
    assert_eq!((r3.rows(), r3.cols()), (17, 11));
    core::rotate(&r3, &mut r4, core::ROTATE_90_CLOCKWISE).unwrap();
    assert_eq!(r4.raw_bytes(), m.raw_bytes());

    // ROTATE_90_COUNTERCLOCKWISE == r3
    let mut r_ccw = Mat::default();
    core::rotate(&m, &mut r_ccw, core::ROTATE_90_COUNTERCLOCKWISE).unwrap();
    assert_eq!(r_ccw.raw_bytes(), r3.raw_bytes());

    // ROTATE_180 == flip(-1) == r2
    let mut f_both = Mat::default();
    core::flip(&m, &mut f_both, -1).unwrap();
    assert_eq!(f_both.raw_bytes(), r2.raw_bytes());

    // transpose(flip_vertical(m)) == ROTATE_90_CLOCKWISE
    let mut f_vert = Mat::default();
    let mut tr = Mat::default();
    core::flip(&m, &mut f_vert, 0).unwrap();
    core::transpose(&f_vert, &mut tr).unwrap();
    assert_eq!(tr.raw_bytes(), r1.raw_bytes());

    // Mirror-Twin Jacobian (Theorem 6)
    let j_horiz = MirrorTwinJacobian::new(1, -1);
    assert_eq!(j_horiz.det(), -1);
    let j_rot180 = MirrorTwinJacobian::new(-1, -1);
    assert_eq!(j_rot180.det(), 1);
}

#[test]
fn test_02_oh_48_group_axioms_chiral_vs_achiral_and_rotvec_roundtrip() {
    let all48 = OhElement::all_48();
    let proper24 = OhElement::proper_24();
    assert_eq!(all48.len(), 48);
    assert_eq!(proper24.len(), 24);

    let improper_count = all48.iter().filter(|g| g.det() == -1).count();
    assert_eq!(improper_count, 24);

    let ident = OhElement {
        perm: 0,
        flip_mask: 0,
    };

    // Verify 48x48 Cayley group closure and inverse identity
    for g in &all48 {
        let inv = g.inverse();
        assert_eq!(g.compose(&inv), ident);
        assert_eq!(inv.compose(g), ident);
        for h in &all48 {
            let gh = g.compose(h);
            assert_eq!(gh.det(), g.det() * h.det());
        }
    }

    // Verify rotmat_to_rotvec -> rotvec_to_rotmat exact roundtrip across all 24 proper rotations
    // (including the nine 180-degree pi rotations!)
    for g in &proper24 {
        let rm = g.to_rotation_matrix();
        let rv = calib3d::rotmat_to_rotvec(&rm);
        let rm_back = calib3d::rotvec_to_rotmat(rv);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (rm[i][j] - rm_back[i][j]).abs() < 1e-9,
                    "rotvec roundtrip mismatch for {:?}: got {:?} vs {:?}",
                    g,
                    rm_back,
                    rm
                );
            }
        }
    }

    // Chiral 3D right-handed vs left-handed screw polycube inside an 8x8x8 grid
    let right_screw: Vec<([u32; 3], u8)> = vec![
        ([2, 2, 2], 1),
        ([3, 2, 2], 1),
        ([3, 3, 2], 1),
        ([3, 3, 3], 1),
    ];
    let rot_g = proper24[11]; // Non-trivial proper 3D rotation
    let rotated_screw: Vec<([u32; 3], u8)> = right_screw
        .iter()
        .map(|&(pt, v)| (rot_g.apply_grid(pt, 8), v))
        .collect();
    // Reflect along x-axis (det == -1) to create left-handed enantiomorph
    let mirror_g = OhElement {
        perm: 0,
        flip_mask: 1,
    };
    assert_eq!(mirror_g.det(), -1);
    let left_screw: Vec<([u32; 3], u8)> = right_screw
        .iter()
        .map(|&(pt, v)| (mirror_g.apply_grid(pt, 8), v))
        .collect();

    // Under chiral canonicalization (SO(3), 24 proper rotations):
    // right_screw and rotated_screw have IDENTICAL canonical hash, while left_screw differs!
    let (h_right_chiral, _) = OhElement::canonical_zobrist(&right_screw, 8, true);
    let (h_rot_chiral, _) = OhElement::canonical_zobrist(&rotated_screw, 8, true);
    let (h_left_chiral, _) = OhElement::canonical_zobrist(&left_screw, 8, true);
    assert_eq!(h_right_chiral, h_rot_chiral);
    assert_ne!(h_right_chiral, h_left_chiral);

    // Under full achiral O_h canonicalization (all 48 elements):
    // right_screw and left_screw unify to the exact same canonical hash!
    let (h_right_full, _) = OhElement::canonical_zobrist(&right_screw, 8, false);
    let (h_left_full, _) = OhElement::canonical_zobrist(&left_screw, 8, false);
    assert_eq!(h_right_full, h_left_full);
}

#[test]
fn test_03_schur_feshbach_2d_3d_4d_wavelet_exact_invertibility_and_parseval() {
    reset_mind();

    // 3D Octree Wavelet (Theorem 11: 1 + 7 = 8)
    for trial in 0..32 {
        let mut block = [0.0f64; 8];
        for i in 0..8 {
            block[i] = ((trial * 17 + i * 31) as f64 * 0.37).sin() * 50.0 + 12.5;
        }
        let wav3 = SchurFeshbachOctreeWavelet::decompose_block_2x2x2(&block);
        let raw_energy = block.iter().map(|v| v * v).sum::<f64>() * 0.125;
        assert!(
            (wav3.total_energy - raw_energy).abs() < 1e-11,
            "3D Parseval mismatch: {} vs {}",
            wav3.total_energy,
            raw_energy
        );
        let rec3 = wav3.reconstruct_block_2x2x2();
        for i in 0..8 {
            assert!(
                (rec3[i] - block[i]).abs() < 1e-11,
                "3D wavelet inverse mismatch at {}: {} vs {}",
                i,
                rec3[i],
                block[i]
            );
        }
    }

    // 4D Spacetime Hyper-Voxelean Wavelet (Theorem 20: 1 + 15 = 16)
    for trial in 0..32 {
        let mut hyper = [0.0f64; 16];
        for i in 0..16 {
            hyper[i] = ((trial * 23 + i * 19) as f64 * 0.41).cos() * 80.0 - 7.25;
        }
        let wav4 = HyperVoxeleanWavelet4D::decompose_hyperblock_2x2x2x2(&hyper);
        let raw_energy = hyper.iter().map(|v| v * v).sum::<f64>() * 0.0625;
        assert!(
            (wav4.total_energy - raw_energy).abs() < 1e-11,
            "4D Parseval mismatch: {} vs {}",
            wav4.total_energy,
            raw_energy
        );
        let rec4 = wav4.reconstruct_hyperblock_2x2x2x2();
        for i in 0..16 {
            assert!(
                (rec4[i] - hyper[i]).abs() < 1e-11,
                "4D wavelet inverse mismatch at {}: {} vs {}",
                i,
                rec4[i],
                hyper[i]
            );
        }
    }
}

#[test]
fn test_04_discrete_exterior_calculus_chain_complex_and_homology_betti_zoo() {
    // 1. Single voxel: V0=8, E1=12, F2=6, C3=1, beta=(1,0,0), chi=1
    let single = [(0, 0, 0)];
    let c_single = VoxelComplex3D::from_occupied_voxels(&single);
    assert_eq!(
        (c_single.v0, c_single.e1, c_single.f2, c_single.c3),
        (8, 12, 6, 1)
    );
    assert_eq!(
        (c_single.beta0, c_single.beta1, c_single.beta2),
        (1, 0, 0)
    );
    assert_eq!(c_single.euler_characteristic(), 1);
    assert_eq!(
        VoxelComplex3D::verify_chain_complex_boundary_zero(&single),
        (0, 0)
    );

    // 2. 2x2x2 solid cube: V0=27, E1=54, F2=36, C3=8, beta=(1,0,0), chi=1
    let mut cube2 = Vec::new();
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                cube2.push((x, y, z));
            }
        }
    }
    let c_cube2 = VoxelComplex3D::from_occupied_voxels(&cube2);
    assert_eq!(
        (c_cube2.v0, c_cube2.e1, c_cube2.f2, c_cube2.c3),
        (27, 54, 36, 8)
    );
    assert_eq!((c_cube2.beta0, c_cube2.beta1, c_cube2.beta2), (1, 0, 0));
    assert_eq!(c_cube2.euler_characteristic(), 1);
    assert_eq!(
        VoxelComplex3D::verify_chain_complex_boundary_zero(&cube2),
        (0, 0)
    );

    // 3. 3x3x3 hollow shell (center (1,1,1) removed): 26 voxels, 1 enclosed cavity -> beta=(1,0,1), chi=2
    let mut shell = Vec::new();
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                if (x, y, z) != (1, 1, 1) {
                    shell.push((x, y, z));
                }
            }
        }
    }
    let c_shell = VoxelComplex3D::from_occupied_voxels(&shell);
    assert_eq!(
        (c_shell.beta0, c_shell.beta1, c_shell.beta2),
        (1, 0, 1)
    );
    assert_eq!(c_shell.euler_characteristic(), 2);
    assert_eq!(c_shell.betti_euler(), 2);
    assert_eq!(
        VoxelComplex3D::verify_chain_complex_boundary_zero(&shell),
        (0, 0)
    );

    // 4. 3x3x1 square torus ring (center (1,1,0) open): 8 voxels, 1 tunnel -> beta=(1,1,0), chi=0
    let mut torus1 = Vec::new();
    for x in 0..3 {
        for y in 0..3 {
            if (x, y) != (1, 1) {
                torus1.push((x, y, 0));
            }
        }
    }
    let c_torus1 = VoxelComplex3D::from_occupied_voxels(&torus1);
    assert_eq!(
        (c_torus1.beta0, c_torus1.beta1, c_torus1.beta2),
        (1, 1, 0)
    );
    assert_eq!(c_torus1.euler_characteristic(), 0);
    assert_eq!(
        VoxelComplex3D::verify_chain_complex_boundary_zero(&torus1),
        (0, 0)
    );

    // 5. Genus-2 double torus (5x3x1 figure-8 with holes at (1,1,0) and (3,1,0)): beta=(1,2,0), chi=-1
    let mut torus2 = Vec::new();
    for x in 0..5 {
        for y in 0..3 {
            if (x, y) != (1, 1) && (x, y) != (3, 1) {
                torus2.push((x, y, 0));
            }
        }
    }
    let c_torus2 = VoxelComplex3D::from_occupied_voxels(&torus2);
    assert_eq!(
        (c_torus2.beta0, c_torus2.beta1, c_torus2.beta2),
        (1, 2, 0)
    );
    assert_eq!(c_torus2.euler_characteristic(), -1);
    assert_eq!(c_torus2.betti_euler(), -1);
    assert_eq!(
        VoxelComplex3D::verify_chain_complex_boundary_zero(&torus2),
        (0, 0)
    );

    // 6. Disconnected assembly: hollow shell at [0..3]^3 + isolated voxel at (10, 10, 10) -> beta=(2,0,1), chi=3
    let mut disconn = shell.clone();
    disconn.push((10, 10, 10));
    let c_disconn = VoxelComplex3D::from_occupied_voxels(&disconn);
    assert_eq!(
        (c_disconn.beta0, c_disconn.beta1, c_disconn.beta2),
        (2, 0, 1)
    );
    assert_eq!(c_disconn.euler_characteristic(), 3);
}

#[test]
fn test_05_se2_frenet_serret_jets_circle_curvature_and_sffft_phase_correlation() {
    // 1. Operator J1: SE(2) Frenet-Serret Jet-Space Lifting on smooth radial disks
    // For I(x, y) = 255 - 2 * r^2 where r = sqrt((x-cx)^2 + (y-cy)^2), the level sets are circles
    // of radius r, and the analytical level-set curvature is |kappa| = 1 / r.
    let mut disk = Mat::zeros(64, 64, CV_64FC1).unwrap();
    let (cx, cy) = (32.0, 32.0);
    for r in 0..64 {
        for c in 0..64 {
            let dx = (c as f64) - cx;
            let dy = (r as f64) - cy;
            let r2 = dx * dx + dy * dy;
            disk.set_f64(r, c, 0, 1000.0 - r2);
        }
    }
    let jets = pixelean::lift_se2_frenet_serret_jets(&disk, 40.0);
    assert!(!jets.is_empty());
    let mut rel_err_sum = 0.0;
    let mut cnt = 0usize;
    for j in &jets {
        let rad = ((j.x - cx).powi(2) + (j.y - cy).powi(2)).sqrt();
        if (8.0..=20.0).contains(&rad) {
            let expected_kappa = 1.0 / rad;
            let measured = j.curvature_kappa.abs();
            rel_err_sum += (measured - expected_kappa).abs() / expected_kappa;
            cnt += 1;
        }
    }
    assert!(cnt > 50);
    let mean_rel_err = rel_err_sum / (cnt as f64);
    assert!(
        mean_rel_err < 0.05,
        "expected mean curvature relative error < 5%, got {:.4}",
        mean_rel_err
    );

    // 2. Operator J2: 2D Radix-2 sfFFT Phase Correlation across positive and negative shifts
    for &(true_dx, true_dy) in &[(5isize, -3isize), (-6, 4), (0, 0), (7, 5)] {
        let mut prev = Mat::zeros(64, 64, CV_8UC1).unwrap();
        let mut curr = Mat::zeros(64, 64, CV_8UC1).unwrap();
        for r in 16..40 {
            for c in 16..40 {
                let val = (50 + ((r * 7) ^ (c * 13)) % 180) as f64;
                prev.set_f64(r, c, 0, val);
                let nr = (r as isize + true_dy) as i32;
                let nc = (c as isize + true_dx) as i32;
                curr.set_f64(nr, nc, 0, val);
            }
        }
        let (dx, dy, peak) = pixelean::sffft_phase_correlation(&prev, &curr).unwrap();
        assert!(
            (dx - true_dx as f64).abs() < 0.35 && (dy - true_dy as f64).abs() < 0.35,
            "sfFFT shift mismatch for ({}, {}): got ({:.3}, {:.3}) peak={:.3}",
            true_dx,
            true_dy,
            dx,
            dy,
            peak
        );
    }
}

#[test]
fn test_06_unbent_ray_matrix_exponentiation_and_snell_fermat_geodesic() {
    // 1. Operator A2 & Theorem 22: Unbent Ray Matrix Exponentiation over GF(65521)
    let p = 65521u64;
    let mat = vec![
        3, 11, 5, 2, //
        7, 19, 13, 23, //
        29, 31, 37, 41, //
        43, 47, 53, 59,
    ];
    let op = UnbentRayTransferOperator::new(4, p, mat);
    let v0 = vec![101, 202, 303, 404];
    for &depth_d in &[1u64, 7, 64, 513] {
        let (fast_state, squaring_steps) = op.propagate_binary_exp(&v0, depth_d);
        let seq_state = op.propagate_sequential(&v0, depth_d);
        assert_eq!(
            fast_state, seq_state,
            "binary matrix exponentiation mismatch at depth {}",
            depth_d
        );
        if depth_d == 513 {
            assert!(squaring_steps <= 10);
        }
    }

    // 2. Theorem 23: Snell-Fermat refraction geodesic solver
    for &(n1, n2) in &[(1.0, 1.5), (1.33, 2.42), (1.8, 1.1)] {
        let (x_star, l_star, l_straight, snell_res) =
            worldtube::snell_fermat_refraction(n1, n2, 4.0, 3.0, 3.0);
        assert!(
            snell_res < 1e-11,
            "Snell law residual too large: {}",
            snell_res
        );
        assert!(
            l_star < l_straight,
            "Fermat path {} not strictly shorter than straight path {}",
            l_star,
            l_straight
        );
        assert!(x_star.abs() > 0.1 && x_star.abs() < 4.0);
    }

    // 3. Theorem 16 & Operator A1: 7-Point Star Stencil over GF(p)
    let stencil = StarStencil7PointGfp {
        p_modulus: 65521,
        c0: 1234,
        cx_pos: 5678,
        cx_neg: 9101,
        cy_pos: 1121,
        cy_neg: 3141,
        cz_pos: 5161,
        cz_neg: 7181,
    };
    let recovered = StarStencil7PointGfp::identify_from_impulse(
        65521,
        stencil.c0 + 65521,
        stencil.cx_pos + 2 * 65521,
        stencil.cx_neg,
        stencil.cy_pos,
        stencil.cy_neg,
        stencil.cz_pos,
        stencil.cz_neg,
    );
    assert_eq!(recovered, stencil);
}

#[test]
fn test_07_subpixel_pyramidal_lucas_kanade_and_xor_minterm_fastpath() {
    reset_mind();
    let w = 80i32;
    let h = 80i32;
    let mut img0 = Mat::zeros(h, w, CV_8UC1).unwrap();
    let mut img1 = Mat::zeros(h, w, CV_8UC1).unwrap();

    // Synthesize a smooth high-contrast multi-frequency 2D texture patch around (40, 40)
    // and an identical static corner patch around (14, 14).
    let true_dx = 1.65f64;
    let true_dy = -1.25f64;
    let synth = |x: f64, y: f64| -> f64 {
        let gx = ((x - 40.0) / 7.5).exp_m1();
        let _ = gx;
        let dx = x - 40.0;
        let dy = y - 40.0;
        let env = (-(dx * dx + dy * dy) / 95.0).exp();
        let wave = 0.5
            + 0.25 * (dx * 0.55).cos() * (dy * 0.45).cos()
            + 0.25 * ((dx + dy) * 0.35).sin();
        (20.0 + 210.0 * env * wave).clamp(0.0, 255.0)
    };

    for r in 0..h {
        for c in 0..w {
            if r < 24 && c < 24 {
                // Static corner region (bit-identical between img0 and img1)
                let v = if (6..16).contains(&r) && (6..16).contains(&c) {
                    200.0
                } else {
                    20.0
                };
                img0.set_f64(r, c, 0, v);
                img1.set_f64(r, c, 0, v);
            } else {
                img0.set_f64(r, c, 0, synth(c as f64, r as f64));
                img1.set_f64(r, c, 0, synth(c as f64 - true_dx, r as f64 - true_dy));
            }
        }
    }

    let mut prev_pts = Vector::<Point2f>::new();
    prev_pts.push(Point2f::new(6.0, 6.0)); // Static corner
    prev_pts.push(Point2f::new(38.0, 38.0)); // Moving smooth texture
    prev_pts.push(Point2f::new(42.0, 41.0)); // Moving smooth texture

    let mut next_pts = Vector::<Point2f>::new();
    let mut status = Vector::<u8>::new();
    let mut err = Vector::<f32>::new();
    let criteria = TermCriteria::new(3, 30, 0.005).unwrap();
    video::calc_optical_flow_pyr_lk(
        &img0,
        &img1,
        &prev_pts,
        &mut next_pts,
        &mut status,
        &mut err,
        Size::new(15, 15),
        2,
        criteria,
        0,
        1e-4,
    )
    .unwrap();

    // Point 0 is in the bit-identical static region -> exact (0, 0) displacement via Theorem 2 XOR fastpath
    assert_eq!(status.get(0).unwrap(), 1);
    let p0_next = next_pts.get(0).unwrap();
    assert!((p0_next.x - 6.0).abs() < 1e-6 && (p0_next.y - 6.0).abs() < 1e-6);

    // Points 1 and 2 track the sub-pixel shift (true_dx, true_dy)
    for idx in 1..=2 {
        assert_eq!(status.get(idx).unwrap(), 1);
        let p_in = prev_pts.get(idx).unwrap();
        let p_out = next_pts.get(idx).unwrap();
        let est_dx = (p_out.x - p_in.x) as f64;
        let est_dy = (p_out.y - p_in.y) as f64;
        assert!(
            (est_dx - true_dx).abs() < 0.25 && (est_dy - true_dy).abs() < 0.25,
            "subpixel LK error at pt {}: got ({:.3}, {:.3}) vs ({:.3}, {:.3})",
            idx,
            est_dx,
            est_dy,
            true_dx,
            true_dy
        );
    }
}

#[test]
fn test_08_global_so3_solve_pnp_and_ransac_under_large_rotations_and_outliers() {
    let k_mat = Mat::from_slice_2d(&[
        [520.0f64, 0.0, 320.0],
        [0.0, 520.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    .unwrap();
    let dist = Mat::from_slice(&[0.04f64, -0.02, 0.001, -0.001, 0.0]).unwrap();

    // Non-coplanar 14-point 3D constellation
    let obj_slice = [
        Point3f::new(-0.8, -0.6, 0.2),
        Point3f::new(0.9, -0.5, -0.1),
        Point3f::new(0.7, 0.8, 0.3),
        Point3f::new(-0.7, 0.7, -0.2),
        Point3f::new(0.0, 0.0, 0.6),
        Point3f::new(-0.4, 0.3, 0.5),
        Point3f::new(0.5, -0.2, 0.4),
        Point3f::new(-0.3, -0.4, -0.3),
        Point3f::new(0.3, 0.5, -0.4),
        Point3f::new(0.6, 0.1, 0.2),
        Point3f::new(-0.6, -0.1, 0.1),
        Point3f::new(0.1, -0.7, 0.3),
        Point3f::new(-0.2, 0.8, 0.0),
        Point3f::new(0.4, 0.4, 0.5),
    ];
    let mut obj_pts = Vector::<Point3f>::new();
    for &p in &obj_slice {
        obj_pts.push(p);
    }

    // Test both a moderate rotation and a HUGE 135-degree (2.35 rad) 3D rotation
    // that would defeat any local-only Gauss-Newton solver without Theorem 10 O_h seeding!
    for &(gt_rv, gt_tv) in &[
        ([0.22f64, -0.31, 0.18], [0.25f64, -0.15, 3.2]),
        ([1.45f64, -1.10, 0.95], [-0.30f64, 0.20, 3.8]),
    ] {
        let rvec_gt = Mat::from_slice(&gt_rv).unwrap().reshape(1, 3).unwrap();
        let tvec_gt = Mat::from_slice(&gt_tv).unwrap().reshape(1, 3).unwrap();
        let mut clean_img = Vector::<Point2f>::new();
        calib3d::project_points_def(&obj_pts, &rvec_gt, &tvec_gt, &k_mat, &dist, &mut clean_img)
            .unwrap();

        // 1. Clean solve_pnp with lens distortion
        let mut rvec_est = Mat::default();
        let mut tvec_est = Mat::default();
        let ok = calib3d::solve_pnp(
            &obj_pts,
            &clean_img,
            &k_mat,
            &dist,
            &mut rvec_est,
            &mut tvec_est,
            false,
            calib3d::SOLVEPNP_ITERATIVE,
        )
        .unwrap();
        assert!(ok);
        for k in 0..3 {
            assert!(
                (rvec_est.get_f64(k, 0, 0) - gt_rv[k as usize]).abs() < 1e-2,
                "solve_pnp rvec[{}] mismatch for {:?}: got {}",
                k,
                gt_rv,
                rvec_est.get_f64(k, 0, 0)
            );
            assert!(
                (tvec_est.get_f64(k, 0, 0) - gt_tv[k as usize]).abs() < 1e-2,
                "solve_pnp tvec[{}] mismatch for {:?}: got {}",
                k,
                gt_tv,
                tvec_est.get_f64(k, 0, 0)
            );
        }

        // 2. Corrupt 5 of the 14 points (35% gross outliers) and test solve_pnp_ransac
        let mut noisy_img = clean_img.clone();
        noisy_img.set(2, Point2f::new(15.0, 470.0)).unwrap();
        noisy_img.set(5, Point2f::new(610.0, 12.0)).unwrap();
        noisy_img.set(8, Point2f::new(590.0, 460.0)).unwrap();
        noisy_img.set(11, Point2f::new(5.0, 5.0)).unwrap();
        noisy_img.set(13, Point2f::new(320.0, 10.0)).unwrap();

        let mut rvec_ran = Mat::default();
        let mut tvec_ran = Mat::default();
        let mut inliers = Mat::default();
        let ok_ran = calib3d::solve_pnp_ransac(
            &obj_pts,
            &noisy_img,
            &k_mat,
            &dist,
            &mut rvec_ran,
            &mut tvec_ran,
            false,
            200,
            3.0,
            0.99,
            &mut inliers,
            calib3d::SOLVEPNP_ITERATIVE,
        )
        .unwrap();
        assert!(ok_ran);
        assert_eq!(inliers.rows(), 9, "expected exactly 9 inliers out of 14");
        for k in 0..3 {
            assert!(
                (tvec_ran.get_f64(k, 0, 0) - gt_tv[k as usize]).abs() < 2e-2,
                "solve_pnp_ransac tvec[{}] mismatch: got {} vs {}",
                k,
                tvec_ran.get_f64(k, 0, 0),
                gt_tv[k as usize]
            );
        }
    }
}

#[test]
fn test_09_svd_3x3_hartley_essential_matrix_and_twisted_pair_pose_recovery() {
    // 1. Verify svd_3x3 across 40 diverse 3x3 matrices (invertible, negative det, rank-2, rank-1)
    for trial in 0..40 {
        let mut a = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] = ((trial * 19 + r * 7 + c * 13) as f64 * 0.63).sin() * 10.0;
            }
        }
        if trial % 4 == 0 {
            // Force exact rank-2 matrix (like an Essential matrix)
            a[2][0] = a[0][0] - 0.5 * a[1][0];
            a[2][1] = a[0][1] - 0.5 * a[1][1];
            a[2][2] = a[0][2] - 0.5 * a[1][2];
        }
        let (u, sig, v) = calib3d::svd_3x3(&a);
        assert!(sig[0] >= sig[1] - 1e-12 && sig[1] >= sig[2] - 1e-12);
        for i in 0..3 {
            for j in 0..3 {
                let recon = u[i][0] * sig[0] * v[j][0]
                    + u[i][1] * sig[1] * v[j][1]
                    + u[i][2] * sig[2] * v[j][2];
                assert!(
                    (recon - a[i][j]).abs() < 1e-9,
                    "SVD reconstruction mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    recon,
                    a[i][j]
                );
            }
        }
    }

    // 2. Test find_essential_mat + recover_pose_triangulated under combined 3D rotation + translation
    let k_mat = Mat::from_slice_2d(&[
        [500.0f64, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    .unwrap();
    let gt_rv = [0.08f64, -0.14, 0.06];
    let gt_r = calib3d::rotvec_to_rotmat(gt_rv);
    let raw_t = [0.60f64, -0.20, 0.15];
    let t_norm = (raw_t[0] * raw_t[0] + raw_t[1] * raw_t[1] + raw_t[2] * raw_t[2]).sqrt();
    let gt_t = [raw_t[0] / t_norm, raw_t[1] / t_norm, raw_t[2] / t_norm];

    let mut pts1 = Vector::<Point2f>::new();
    let mut pts2 = Vector::<Point2f>::new();
    for i in 0..20 {
        let x = -1.2 + (i % 5) as f64 * 0.6 + ((i * 7) % 3) as f64 * 0.11;
        let y = -0.9 + (i / 5) as f64 * 0.6 + ((i * 11) % 3) as f64 * 0.09;
        let z = 3.0 + ((i * 13) % 7) as f64 * 0.45;
        let u1 = 500.0 * (x / z) + 320.0;
        let v1 = 500.0 * (y / z) + 240.0;

        let x2 = gt_r[0][0] * x + gt_r[0][1] * y + gt_r[0][2] * z + gt_t[0];
        let y2 = gt_r[1][0] * x + gt_r[1][1] * y + gt_r[1][2] * z + gt_t[1];
        let z2 = gt_r[2][0] * x + gt_r[2][1] * y + gt_r[2][2] * z + gt_t[2];
        let u2 = 500.0 * (x2 / z2) + 320.0;
        let v2 = 500.0 * (y2 / z2) + 240.0;

        pts1.push(Point2f::new(u1 as f32, v1 as f32));
        pts2.push(Point2f::new(u2 as f32, v2 as f32));
    }

    let mut mask = Mat::default();
    let e_mat =
        calib3d::find_essential_mat(&pts1, &pts2, &k_mat, calib3d::RANSAC, 0.999, 1.5, 200, &mut mask)
            .unwrap();
    assert_eq!((e_mat.rows(), e_mat.cols()), (3, 3));

    let mut r_rec = Mat::default();
    let mut t_rec = Mat::default();
    let mut tri_pts = Mat::default();
    let good = calib3d::recover_pose_triangulated(
        &e_mat,
        &pts1,
        &pts2,
        &k_mat,
        &mut r_rec,
        &mut t_rec,
        50.0,
        &mut mask,
        &mut tri_pts,
    )
    .unwrap();
    assert_eq!(good, 20);

    let dot_t = t_rec.get_f64(0, 0, 0) * gt_t[0]
        + t_rec.get_f64(1, 0, 0) * gt_t[1]
        + t_rec.get_f64(2, 0, 0) * gt_t[2];
    assert!(
        dot_t > 0.98,
        "recovered unit translation direction cosine {} < 0.98",
        dot_t
    );
}

#[test]
fn test_10_multipath_stereo_sgbm_rectification_and_core_imgproc_suite() {
    // 1. Multi-depth StereoSGBM with 2-pass dynamic programming
    let h = 48i32;
    let w = 64i32;
    let mut left = Mat::zeros(h, w, CV_8UC1).unwrap();
    let mut right = Mat::zeros(h, w, CV_8UC1).unwrap();
    for r in 4..(h - 4) {
        for c in 12..(w - 4) {
            let disp = if r < 24 { 4 } else { 8 };
            let tex = (40 + ((r * 19 + c * 31) ^ (r * 7)) % 190) as f64;
            left.set_f64(r, c, 0, tex);
            right.set_f64(r, c - disp, 0, tex);
        }
    }
    let mut sgbm =
        StereoSGBM::create(0, 16, 5, 24, 96, 1, 31, 10, 50, 2, calib3d::StereoSGBM_MODE_SGBM)
            .unwrap();
    let mut disp_mat = Mat::default();
    sgbm.compute(&left, &right, &mut disp_mat).unwrap();
    let d_top = *disp_mat.at_2d::<i16>(14, 32).unwrap() as f64 / 16.0;
    let d_bot = *disp_mat.at_2d::<i16>(34, 32).unwrap() as f64 / 16.0;
    assert!(
        (d_top - 4.0).abs() < 0.5,
        "expected top disparity ~4.0, got {}",
        d_top
    );
    assert!(
        (d_bot - 8.0).abs() < 0.5,
        "expected bottom disparity ~8.0, got {}",
        d_bot
    );

    // 2. Core operations: roi, split, merge, in_range, min_max_loc, mean_std_dev, add_weighted
    let roi = Mat::roi(&left, Rect::new(16, 8, 20, 16)).unwrap();
    assert_eq!((roi.rows(), roi.cols()), (16, 20));
    let mut min_v = 0.0;
    let mut max_v = 0.0;
    let mut min_p = Point::default();
    let mut max_p = Point::default();
    core::min_max_loc(
        &roi,
        Some(&mut min_v),
        Some(&mut max_v),
        Some(&mut min_p),
        Some(&mut max_p),
        &core::no_array(),
    )
    .unwrap();
    assert!(max_v > min_v);

    let mut mean = Scalar::default();
    let mut stddev = Scalar::default();
    core::mean_std_dev(&roi, &mut mean, &mut stddev).unwrap();
    assert!(mean.0[0] > 20.0 && stddev.0[0] > 5.0);

    let mut planes = Vector::<Mat>::new();
    planes.push(roi.clone());
    planes.push(roi.clone());
    planes.push(roi.clone());
    let mut merged = Mat::default();
    core::merge(&planes, &mut merged).unwrap();
    assert_eq!(merged.channels(), 3);
    let mut split_out = Vector::<Mat>::new();
    core::split(&merged, &mut split_out).unwrap();
    assert_eq!(split_out.len(), 3);
    assert_eq!(split_out.get(1).unwrap().raw_bytes(), roi.raw_bytes());

    let mut mask_range = Mat::default();
    core::in_range(
        &roi,
        Scalar::new(100.0, 0.0, 0.0, 0.0),
        Scalar::new(255.0, 0.0, 0.0, 0.0),
        &mut mask_range,
    )
    .unwrap();
    assert!(core::count_non_zero(&mask_range).unwrap() > 0);

    // 3. Canny + find_contours_def + Otsu threshold + DNN blob + Objdetect
    let mut otsu = Mat::default();
    let thr_val = imgproc::threshold(
        &left,
        &mut otsu,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )
    .unwrap();
    assert!(thr_val > 0.0);

    let mut edges = Mat::default();
    imgproc::canny(&otsu, &mut edges, 50.0, 150.0, 3, false).unwrap();
    let mut contours: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours_def(
        &otsu,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
    )
    .unwrap();
    assert!(!contours.is_empty());

    let blob = dnn::blob_from_image(
        &merged,
        1.0 / 255.0,
        Size::new(16, 16),
        Scalar::all(0.0),
        false,
        false,
    )
    .unwrap();
    assert_eq!(blob.total(), 3 * 16 * 16);

    let det = objdetect::ConstructFactoredDetector::new(8, 2000, 120);
    let boxes = det.detect_multi_scale(&left).unwrap();
    assert!(!boxes.is_empty());
}

#[test]
fn test_11_worldtube_and_counterfactual_mechinterp_ablation() {
    reset_mind();
    let mut f0 = Mat::zeros(96, 72, CV_8UC1).unwrap();
    let mut f1 = Mat::zeros(96, 72, CV_8UC1).unwrap();
    for r in 16..48 {
        for c in 16..48 {
            f0.set_f64(r, c, 0, 180.0);
            f1.set_f64(r, c + 1, 0, 180.0);
        }
    }
    let p0 = BitplanePartition::from_gray_mat(&f0, 64).unwrap();
    let p1 = BitplanePartition::from_gray_mat(&f1, 64).unwrap();
    let xor = XorMintermCarrier::from_partitions(&p0, &p1);
    let xor_sparsity_gain = 1.0 - (xor.k_diff as f64 / xor.total_pixels as f64);
    assert!(xor_sparsity_gain > 0.85);
    assert!((worldtube::hyper_voxelean_wavelet_ratio(3) - 0.875).abs() < 1e-12);
    let wt = WorldtubeTrajectoryFactor::factor_3d(64, 6, [1.0, 0.0, 0.0]);
    let cert = GrandSynthesisCertificate::certify(3, 6, 64);
    assert!(wt.factored_cost < wt.unfactored_cost);
    assert!(cert.worldtube_compression_ratio > 0.75);

    let mut frame = Mat::zeros(32, 32, CV_8UC1).unwrap();
    for r in 8..20 {
        for c in 8..20 {
            frame.set_f64(r, c, 0, 128.0);
        }
    }
    let normal = BitplanePartition::from_gray_mat(&frame, 64).unwrap();
    assert_eq!(normal.fg_mass, 144);

    let (ablated, probe_summary) = with_counterfactual_ablation(
        AblationConfig {
            ablated_bitplanes: vec![7],
            ..Default::default()
        },
        || BitplanePartition::from_gray_mat(&frame, 64).unwrap(),
    );
    assert_eq!(ablated.fg_mass, 0);
    assert!(probe_summary.total_events > 0);
}

#[test]
fn test_12_hardware_neon_ptx_metal_wgsl_slabs_and_tensor_core_b1_mma() {
    use solidcv::hardware::{
        benchmark_hardware_kernels, grace_allocator_telemetry, simd_bitplane_pack_u8,
        simd_sobel_3x3_u8, simd_xor_diff_carrier_u8, tensor_core_b1_mma_m16n8k256,
        GraceCoherentSlab, ResidencyPolicy, APPLE_METAL_MSL_SOURCE, CUDA_PTX_KERNEL_SOURCE,
        WEBGPU_WGSL_KERNEL_SOURCE,
    };

    // 1. Verify 16 KB Apple Metal NoCopy + 128-byte NVIDIA GB300/GB10 TMA slab alignment
    let slab = GraceCoherentSlab::<u64>::allocate(512, ResidencyPolicy::KeepOnCardHbm3e).unwrap();
    assert!(slab.is_128b_tensor_core_aligned());
    assert!(slab.is_16k_metal_nocopy_aligned());
    let (res_bytes, alloc_cnt, _topo) = grace_allocator_telemetry();
    assert!(res_bytes >= 16384 && alloc_cnt >= 1);

    // 2. Verify embedded PTX 8.5 (b1.xor.popc + f16.f32 MMA), Apple Metal MSL 3.1 (simd_ballot + simdgroup_float8x8), and WebGPU WGSL
    assert!(CUDA_PTX_KERNEL_SOURCE.contains("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc"));
    assert!(CUDA_PTX_KERNEL_SOURCE.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
    assert!(APPLE_METAL_MSL_SOURCE.contains("simd_ballot"));
    assert!(APPLE_METAL_MSL_SOURCE.contains("simdgroup_multiply_accumulate"));
    assert!(WEBGPU_WGSL_KERNEL_SOURCE.contains("countOneBits"));

    // 3. Verify exact parity between NEON/SWAR SIMD kernels and scalar ground truth
    let mut img = Mat::zeros(37, 53, CV_8UC1).unwrap();
    for r in 0..37 {
        for c in 0..53 {
            img.set_f64(r, c, 0, ((r * 19 + c * 31 + (r ^ c) * 7) & 0xFF) as f64);
        }
    }
    let bytes = img.data_bytes().unwrap();
    let (words, fg) = simd_bitplane_pack_u8(bytes, 120);
    let scalar_fg = bytes.iter().filter(|&&b| b >= 120).count();
    assert_eq!(fg, scalar_fg);
    let word_ones: usize = words.iter().map(|w| w.count_ones() as usize).sum();
    assert_eq!(word_ones, scalar_fg);

    let (gx, gy) = simd_sobel_3x3_u8(&img).unwrap();
    for r in 1..36 {
        for c in 1..52 {
            let idx = (r * 53 + c) as usize;
            let exp_gx = (img.get_f64(r - 1, c + 1, 0)
                + 2.0 * img.get_f64(r, c + 1, 0)
                + img.get_f64(r + 1, c + 1, 0))
                - (img.get_f64(r - 1, c - 1, 0)
                    + 2.0 * img.get_f64(r, c - 1, 0)
                    + img.get_f64(r + 1, c - 1, 0));
            let exp_gy = (img.get_f64(r + 1, c - 1, 0)
                + 2.0 * img.get_f64(r + 1, c, 0)
                + img.get_f64(r + 1, c + 1, 0))
                - (img.get_f64(r - 1, c - 1, 0)
                    + 2.0 * img.get_f64(r - 1, c, 0)
                    + img.get_f64(r - 1, c + 1, 0));
            assert_eq!(gx[idx] as f64, exp_gx);
            assert_eq!(gy[idx] as f64, exp_gy);
        }
    }

    // 4. Verify Tensor-Core m16n8k256 1-bit XOR+POPCNT tile MMA
    let mut tile_a = [[0u64; 4]; 16];
    let mut tile_b = [[0u64; 4]; 8];
    for i in 0..16 {
        tile_a[i] = [i as u64, (i as u64) << 8, !0u64, 0xAAAA_5555_AAAA_5555];
    }
    for j in 0..8 {
        tile_b[j] = [j as u64, (j as u64) << 8, !0u64, 0xAAAA_5555_AAAA_5555];
    }
    let c_mma = tensor_core_b1_mma_m16n8k256(&tile_a, &tile_b);
    for i in 0..8 {
        assert_eq!(c_mma[i][i], 0);
    }

    let (_, kd) = simd_xor_diff_carrier_u8(bytes, bytes, 0);
    assert_eq!(kd, 0);

    let report = benchmark_hardware_kernels(128, 128).unwrap();
    assert!(report.bitplane_gpix_per_sec > 0.05);
    assert!(report.orb_256_mpairs_per_sec > 5.0);
    assert_eq!(report.ptx_kernels_ready, 4);
    assert!(report.slab_128b_aligned);
}

#[test]
fn test_13_multiscale_256bit_pixelean_orb_and_tensor_core_bfmatcher() {
    use solidcv::features2d::{BFMatcher, NORM_HAMMING, ORB};

    let mut img0 = Mat::zeros(96, 96, CV_8UC1).unwrap();
    let mut img1 = Mat::zeros(96, 96, CV_8UC1).unwrap();
    let patches = [
        (24i32, 24i32, 210.0, 40.0),
        (64, 28, 240.0, 20.0),
        (30, 66, 180.0, 60.0),
        (68, 64, 250.0, 10.0),
    ];
    for &(cx, cy, hi, lo) in &patches {
        for dy in -6..=6 {
            for dx in -6..=6 {
                let v = if (dx + dy * 2) % 3 == 0 { hi } else { lo };
                img0.set_f64(cy + dy, cx + dx, 0, v);
                img1.set_f64(cy + dy + 3, cx + dx + 4, 0, v);
            }
        }
    }

    let orb = ORB::default_detector(32);
    let mask = Mat::default();
    let (kps0, desc0, mat0) = orb.detect_and_compute(&img0, &mask).unwrap();
    let (kps1, desc1, mat1) = orb.detect_and_compute(&img1, &mask).unwrap();
    assert!(!kps0.is_empty() && !kps1.is_empty());
    assert_eq!(mat0.cols(), 32);
    assert_eq!(mat1.cols(), 32);

    let matcher = BFMatcher::new(NORM_HAMMING, true);
    let matches = matcher.match_descriptors_256(&desc0, &desc1, 64);
    assert!(!matches.is_empty());

    let mut inlier_shifts = 0usize;
    for m in &matches {
        let p0 = kps0[m.query_idx as usize].pt;
        let p1 = kps1[m.train_idx as usize].pt;
        if (p1.x - p0.x - 4.0).abs() <= 2.5 && (p1.y - p0.y - 3.0).abs() <= 2.5 {
            inlier_shifts += 1;
        }
    }
    assert!(inlier_shifts >= 1);
}

#[test]
fn test_14_d4_chessboard_subpixel_saddle_corners_and_aruco_fiducial_decoder() {
    // 1. Synthesize a 4x3 interior-corner checkerboard (5x4 squares of size 14x14 px)
    let sq = 14i32;
    let margin = 10i32;
    let w = margin * 2 + 5 * sq;
    let h = margin * 2 + 4 * sq;
    let mut board = Mat::new_rows_cols_with_default(h, w, CV_8UC1, Scalar::all(128.0)).unwrap();
    for sr in 0..4 {
        for sc in 0..5 {
            let val = if (sr + sc) % 2 == 0 { 230.0 } else { 25.0 };
            for y in (margin + sr * sq)..(margin + (sr + 1) * sq) {
                for x in (margin + sc * sq)..(margin + (sc + 1) * sq) {
                    board.set_f64(y, x, 0, val);
                }
            }
        }
    }

    let mut corners = Vector::<Point2f>::new();
    let found = calib3d::find_chessboard_corners(&board, Size::new(4, 3), &mut corners, 0).unwrap();
    assert!(found);
    assert_eq!(corners.len(), 12);

    for r in 0..3 {
        for c in 0..4 {
            let gt_x = (margin + (c + 1) * sq) as f32 - 0.5;
            let gt_y = (margin + (r + 1) * sq) as f32 - 0.5;
            let det = corners.get((r * 4 + c) as usize).unwrap();
            assert!(
                (det.x - gt_x).abs() < 1.2 && (det.y - gt_y).abs() < 1.2,
                "chessboard corner ({},{}) mismatch: got ({}, {}), expected ({}, {})",
                r,
                c,
                det.x,
                det.y,
                gt_x,
                gt_y
            );
        }
    }

    // 2. Synthesize a 6x6 ArUco fiducial marker rotated across all 4 proper D_4 orientations
    let base_payload: u16 = 0b0001_0100_1011_0010;
    let mut canonical_ids = Vec::new();
    for rot in 0..4u8 {
        let rotated_bits = calib3d::rotate_aruco_bits_4x4(base_payload, rot);
        let mut marker_img =
            Mat::new_rows_cols_with_default(64, 64, CV_8UC1, Scalar::all(240.0)).unwrap();
        let cell = 6i32;
        let m0 = 14i32;
        for gr in 0..6 {
            for gc in 0..6 {
                let is_bright = if gr == 0 || gr == 5 || gc == 0 || gc == 5 {
                    false
                } else {
                    let bit_idx = (gr - 1) * 4 + (gc - 1);
                    ((rotated_bits >> bit_idx) & 1) != 0
                };
                let v = if is_bright { 235.0 } else { 15.0 };
                for dy in 0..cell {
                    for dx in 0..cell {
                        marker_img.set_f64(m0 + gr * cell + dy, m0 + gc * cell + dx, 0, v);
                    }
                }
            }
        }
        let dets = calib3d::detect_aruco_markers(&marker_img, 128).unwrap();
        assert_eq!(dets.len(), 1);
        canonical_ids.push((dets[0].canonical_id, dets[0].rotation_quadrants));
    }
    // All 4 rotated markers must decode to the exact same D_4 canonical ID and distinct rotations!
    for i in 1..4 {
        assert_eq!(canonical_ids[i].0, canonical_ids[0].0);
        assert_ne!(canonical_ids[i].1, canonical_ids[0].1);
    }
}

#[test]
fn test_15_illumination_invariant_census64_sgbm_and_reality_comber_egomotion_split() {
    use solidcv::realitycomb::{comb_dense_velocity_field, IntegralLattice, RealityComber};

    // 1. Test IntegralLattice O(1) box stats & Sauvola adaptive threshold
    let mut doc = Mat::zeros(48, 48, CV_8UC1).unwrap();
    for r in 0..48 {
        for c in 0..48 {
            let illum_grad = 60.0 + (r + c) as f64 * 1.5;
            let fg = if (12..36).contains(&r) && (12..36).contains(&c) && (r + c) % 4 == 0 {
                -45.0
            } else {
                0.0
            };
            doc.set_f64(r, c, 0, (illum_grad + fg).clamp(0.0, 255.0));
        }
    }
    let lattice = IntegralLattice::from_gray(&doc).unwrap();
    let (mean, std) = lattice.rect_mean_std(10, 10, 38, 38);
    assert!(mean > 80.0 && std > 5.0);
    let bin = lattice.adaptive_sauvola_threshold(&doc, 5, 0.2, 128.0).unwrap();
    assert!(core::count_non_zero(&bin).unwrap() > 100);

    // 2. Test comb_dense_velocity_field with mostly static frame + moving patch
    let mut f0 = Mat::zeros(64, 64, CV_8UC1).unwrap();
    let mut f1 = Mat::zeros(64, 64, CV_8UC1).unwrap();
    for r in 0..64 {
        for c in 0..64 {
            let bg = (((r * 7) ^ (c * 11)) & 0x3F) as f64 + 30.0;
            f0.set_f64(r, c, 0, bg);
            f1.set_f64(r, c, 0, bg);
        }
    }
    for r in 24..40 {
        for c in 24..40 {
            let tex = 140.0 + ((r - 24) * 5 + (c - 24) * 3) as f64;
            f0.set_f64(r, c, 0, tex);
            f1.set_f64(r + 1, c + 2, 0, tex);
        }
    }
    let flow = comb_dense_velocity_field(&f0, &f1, 2, 3).unwrap();
    assert!(
        flow.skipped_static_blocks > flow.active_blocks,
        "expected static blocks skipped ({}) > active ({})",
        flow.skipped_static_blocks,
        flow.active_blocks
    );

    // 3. Test RealityComber ego-motion compensation & static-vs-moving worldtube split
    let k_mat = Mat::from_slice_2d(&[
        [450.0f64, 0.0, 160.0],
        [0.0, 450.0, 120.0],
        [0.0, 0.0, 1.0],
    ])
    .unwrap();
    let cam_rv = [0.04f64, -0.07, 0.03];
    let cam_r = calib3d::rotvec_to_rotmat(cam_rv);
    let cam_t = [0.12f64, -0.05, 0.08];

    let mut prev_world = Vec::new();
    let mut curr_img = Vec::new();
    let mut curr_cam = Vec::new();

    // 10 static world landmarks + 2 independently moving object points
    for i in 0..12 {
        let wx = -0.8 + (i % 4) as f64 * 0.55;
        let wy = -0.5 + (i / 4) as f64 * 0.50;
        let wz = 2.6 + ((i * 3) % 5) as f64 * 0.25;
        prev_world.push(Point3f::new(wx as f32, wy as f32, wz as f32));

        // Points 10 and 11 belong to an independently moving object (+0.35m in world X)
        let (true_wx, true_wy, true_wz) = if i >= 10 {
            (wx + 0.35, wy + 0.15, wz)
        } else {
            (wx, wy, wz)
        };
        let cx = cam_r[0][0] * true_wx + cam_r[0][1] * true_wy + cam_r[0][2] * true_wz + cam_t[0];
        let cy = cam_r[1][0] * true_wx + cam_r[1][1] * true_wy + cam_r[1][2] * true_wz + cam_t[1];
        let cz = cam_r[2][0] * true_wx + cam_r[2][1] * true_wy + cam_r[2][2] * true_wz + cam_t[2];
        curr_cam.push(Point3f::new(cx as f32, cy as f32, cz as f32));
        curr_img.push(Point2f::new(
            (450.0 * (cx / cz) + 160.0) as f32,
            (450.0 * (cy / cz) + 120.0) as f32,
        ));
    }

    let mut comber = RealityComber::new(4, 15, 0.08).unwrap();
    let receipt = comber
        .comb_observation(&f1, &k_mat, &prev_world, &curr_img, &curr_cam)
        .unwrap();
    assert!(receipt.slab_16k_metal_aligned);
    assert!(receipt.slab_128b_tensor_core_aligned);
    assert_eq!(receipt.static_inlier_count, 10);
    assert_eq!(receipt.moving_objects.len(), 1);
    assert_eq!(receipt.moving_objects[0].observation_count, 2);
    assert!((receipt.moving_objects[0].velocity_world_m_per_frame[0] - 0.35).abs() < 0.05);
}

#[test]
fn test_16_vlm_patch_token_pruning_sae_b1_circuit_matcher_and_64d_glassbox_signature() {
    use solidcv::mechinterp::{
        prune_vlm_patch_tokens, GlassBoxSignature64D, SaeBinaryAtom, SaeBinaryCircuitMatcher,
    };

    // 1. VLM ViT-16 (224x224 -> 14x14 = 196 patches) token pruning: only patch (5, 7) changes
    let mut vlm_f0 = Mat::new_rows_cols_with_default(224, 224, CV_8UC1, Scalar::all(40.0)).unwrap();
    let mut vlm_f1 = vlm_f0.clone();
    for r in (5 * 16)..(6 * 16) {
        for c in (7 * 16)..(8 * 16) {
            vlm_f1.set_f64(r, c, 0, 220.0);
        }
    }
    let prune = prune_vlm_patch_tokens(&vlm_f0, &vlm_f1, 16, 15, 8).unwrap();
    assert_eq!(prune.total_patches, 196);
    assert_eq!(prune.active_patches, 1);
    assert_eq!(prune.cached_patches, 195);
    assert_eq!(prune.active_patch_coords, vec![(5, 7)]);
    assert!(prune.token_savings_ratio > 0.99);
    assert!(!prune.kv_cache_reuse_mask[5 * 14 + 7]);
    assert!(prune.kv_cache_reuse_mask[0]);

    // 2. 256-bit Tensor-Core (b1.xor.popc m16n8k256) Sparse Autoencoder (SAE) binary circuit matcher
    let mut dict = Vec::new();
    for fid in 0..24u32 {
        let mut vec_f32 = vec![-1.0f32; 512];
        for k in 0..64usize {
            let idx = ((fid as usize * 37 + k * 7) % 512) as usize;
            vec_f32[idx] = 2.5;
        }
        dict.push(SaeBinaryAtom::from_f32_slice(
            fid,
            &format!("sae_circuit_{}", fid),
            &vec_f32,
            0.0,
        ));
    }
    let matcher = SaeBinaryCircuitMatcher::new(dict.clone());
    let queries = dict[4..20].to_vec();
    let matched = matcher.match_residual_batch(&queries);
    assert_eq!(matched.len(), 16);
    for (idx, m) in matched.iter().enumerate() {
        let (fid, label, dist, agreement) = m.as_ref().unwrap();
        assert_eq!(*fid, (idx + 4) as u32);
        assert_eq!(label, &format!("sae_circuit_{}", idx + 4));
        assert_eq!(*dist, 0);
        assert!((*agreement - 1.0).abs() < 1e-12);
    }

    // 3. 64-D L2-normalized Glass-Box visual & topological signature
    let sig0 = GlassBoxSignature64D::from_gray_mat(&vlm_f1, 100).unwrap();
    let sig1 = GlassBoxSignature64D::from_gray_mat(&vlm_f1, 100).unwrap();
    assert_eq!(sig0.vector.len(), 64);
    assert!((sig0.cosine_similarity(&sig1) - 1.0).abs() < 1e-5);
    let _ = vlm_f0.set_f64(0, 0, 0, 40.0);
}

#[test]
fn test_17_uplift_se2_zoom_worldtube_super_resolution_and_foveated_vlm_rois() {
    use solidcv::imgproc::pyr_down_schur_feshbach;
    use solidcv::uplift::{
        foveated_vlm_roi_uplift, worldtube_burst_super_resolve, zoom_se2_mass_conserving,
        FoveatedUpliftConfig,
    };

    // 1. Verify SE(2) Mass-Conserving 2x & 4x Zoom is the exact right-inverse of pyr_down_schur_feshbach
    let mut lr = Mat::zeros(64, 64, CV_8UC1).unwrap();
    for r in 0..64i32 {
        for c in 0..64i32 {
            let v = if r * 2 + c > 64 { 215.0 } else { 35.0 };
            lr.set_f64(r, c, 0, v);
        }
    }
    let (up2x, rep2) = zoom_se2_mass_conserving(&lr, 2).unwrap();
    assert_eq!(up2x.rows(), 128);
    assert_eq!(up2x.cols(), 128);
    assert_eq!(rep2.conservation_rate, 1.0);
    assert_eq!(rep2.max_gibbs_overshoot, 0);
    assert_eq!(rep2.upscaled_mass, rep2.parent_mass * 4);
    let down = pyr_down_schur_feshbach(&up2x).unwrap();
    assert_eq!(down.data_bytes().unwrap(), lr.data_bytes().unwrap());

    // 2. Verify Worldtube Multi-Frame Sub-Pixel Super-Resolution ("Drizzle" + IBP)
    let mut gt = Mat::zeros(64, 64, CV_8UC1).unwrap();
    for r in 0..64i32 {
        for c in 0..64i32 {
            let v = if (r >= 8 && r < 56) && (c >= 8 && c < 56) {
                if (r + c) % 2 == 0 { 220.0 } else { 30.0 }
            } else {
                100.0
            };
            gt.set_f64(r, c, 0, v);
        }
    }
    let shifts = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)];
    let mut burst = Vec::new();
    for &(dx, dy) in &shifts {
        let ox = (dx * 2.0) as i32;
        let oy = (dy * 2.0) as i32;
        let mut f = Mat::zeros(32, 32, CV_8UC1).unwrap();
        for r in 0..32i32 {
            for c in 0..32i32 {
                let v = *gt.at_2d::<u8>(2 * r + oy, 2 * c + ox).unwrap() as f64;
                f.set_f64(r, c, 0, v);
            }
        }
        burst.push(f);
    }
    let (sr, sr_rep) = worldtube_burst_super_resolve(&burst, Some(&shifts), 2).unwrap();
    assert_eq!(sr.rows(), 64);
    assert_eq!(sr.cols(), 64);
    assert!(sr_rep.high_freq_energy_gain > 1.35);

    // 3. Verify Foveated VLM ROI Crop-and-Zoom Uplift
    let mut scene = Mat::new_rows_cols_with_default(256, 256, CV_8UC1, Scalar::all(45.0)).unwrap();
    for r in 32..64 {
        for c in 160..192 {
            let v = if (r + c) % 2 == 0 { 245.0 } else { 15.0 };
            scene.set_f64(r, c, 0, v);
        }
    }
    let cfg = FoveatedUpliftConfig {
        global_downsample_levels: 2,
        roi_size: 32,
        max_foveal_rois: 2,
        vit_patch_size: 16,
        temporal_diff_tol: 8,
    };
    let plan = foveated_vlm_roi_uplift(&scene, None, &cfg).unwrap();
    assert_eq!(plan.foveal_rois.len(), 2);
    assert!(plan.token_compression_ratio > 0.90);
}
