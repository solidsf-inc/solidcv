//! `solidcv`: Pure-Rust Pixelean, Voxelean, and Temporal Hyper-Voxelean
//! Computer Vision, Stereo, Fiducial, and 3D/4D Reconstruction Engine (`solidSF`).
//!
//! Replaces external C++ `opencv` bindings and subprocess harnesses with zero-FFI,
//! hardware-accelerated Rust operators (Apple Silicon NEON + Metal 3.1 + WebGPU WGSL,
//! NVIDIA Grace-Blackwell GB300 / GB10 PTX 8.5+ Tensor Cores) grounded in the 24 theorems
//! of the Unified Pixelean-Voxelean Calculus and instrumented with first-class
//! Mechanistic Interpretability (`mechinterp`) telemetry and counterfactual ablation hooks.

pub mod calib3d;
pub mod cloud_ws;
pub mod core;
pub mod dnn;
pub mod features2d;
pub mod hardware;
pub mod imgproc;
pub mod mechinterp;
pub mod objdetect;
pub mod pixelean;
pub mod prelude;
pub mod realitycomb;
pub mod uplift;
pub mod video;
pub mod voxelean;
pub mod worldtube;

pub use crate as opencv;
pub use crate::core::{Error, Result};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Mat, Point, Point2f, Point3f, Rect, Scalar, Size, TermCriteria, Vector};
    use crate::mechinterp::{reset_mind, with_counterfactual_ablation, AblationConfig};
    use crate::prelude::*;

    #[test]
    fn pillar_i_pixelean_theorems_1_to_8_and_spectral_operators_hold() {
        reset_mind();
        // Theorem 1 & 2: Bitplane partition & Temporal XOR Minterm Sparsity
        let mut frame0 = Mat::zeros(32, 32, core::CV_8UC1).unwrap();
        let mut frame1 = Mat::zeros(32, 32, core::CV_8UC1).unwrap();
        for r in 8..16 {
            for c in 8..16 {
                frame0.set_f64(r, c, 0, 200.0);
                frame1.set_f64(r, c + 1, 0, 200.0);
            }
        }
        let p0 = pixelean::BitplanePartition::from_gray_mat(&frame0, 100).unwrap();
        let p1 = pixelean::BitplanePartition::from_gray_mat(&frame1, 100).unwrap();
        assert_eq!(p0.fg_mass + p0.bg_mass, 32 * 32);
        assert_eq!(p1.fg_mass + p1.bg_mass, 32 * 32);
        let (dfg, dbg) = p0.conservative_transfer_delta(&p1);
        assert_eq!(dfg + dbg, 0);

        let xor = pixelean::XorMintermCarrier::from_partitions(&p0, &p1);
        assert_eq!(xor.k_diff, 16); // Only 8 entering + 8 leaving pixels changed out of 1024!
        assert!(xor.k_diff < xor.total_pixels);
        assert!((xor.k_diff * xor.k_diff) < (xor.total_pixels * xor.total_pixels));

        // Theorem 3: Shannon Cofactor Envelope
        let env = pixelean::ShannonCofactorEnvelope::evaluate(0.35, 12.0, -4.0);
        assert!(env.in_convex_hull());

        // Theorem 4: Kinematic Common-Fate Composite Assembly
        let comp = pixelean::CommonFateComposite::assemble(48.0, 16.0, [2.0, 1.0]);
        assert!((comp.iou_composite - 1.0).abs() < 1e-12);
        assert!(comp.iou_composite > comp.iou_monochrome);

        // Theorem 5 & 6: Dihedral D_4 Closure & Mirror-Twin Jacobian
        for g in pixelean::D4Element::ALL {
            let inv = g.inverse();
            let id = g.compose(inv);
            assert_eq!(id.rot, 0);
            assert!(!id.reflect);
        }
        let jac = pixelean::MirrorTwinJacobian::new(-1, 1);
        assert_eq!(jac.det(), -1);
        assert_eq!(jac.transform_delta(3, -5), (-3, -5));

        // Theorem 7 & 8: Construct Factoring & Logical Lattice Quotient
        let (unfac, fac, saved) = pixelean::construct_factor_savings(12, 15);
        assert_eq!(unfac - fac, saved);
        assert_eq!(saved, 180);
        assert_eq!(pixelean::logical_lattice_quotient(27, 3, 1), Some(7));

        // Operator J1 & J2: SE(2) Frenet-Serret Jets & sfFFT Phase Correlation
        let jets = pixelean::lift_se2_frenet_serret_jets(&frame0, 50.0);
        assert!(!jets.is_empty());
        let (dx, dy, peak) = pixelean::sffft_phase_correlation(&frame0, &frame1).unwrap();
        assert!((dx - 1.0).abs() <= 1.0 && dy.abs() <= 1.0 && peak > 0.1);
    }

    #[test]
    fn pillar_ii_voxelean_theorems_9_to_16_hold() {
        reset_mind();
        // Theorem 9: 3D Voxelean Octree Zobrist Localization
        let mut oct0 = voxelean::VoxeleanOctree::new(4);
        oct0.set_voxel(3, 5, 7, 1);
        oct0.set_voxel(10, 2, 4, 2);
        let mut oct1 = oct0.clone();
        oct1.set_voxel(3, 5, 7, 3);
        let (changed, inspected, full_vol) = oct0.diff_localized(&oct1);
        assert_eq!(changed, vec![(3, 5, 7)]);
        assert!(inspected < full_vol);

        // Theorem 10: 3D Chiral Octahedral Group O_h (48 elements, 24 proper rotations)
        let all_oh = voxelean::OhElement::all_48();
        assert_eq!(all_oh.len(), 48);
        let proper_count = all_oh.iter().filter(|g| g.is_proper_rotation()).count();
        assert_eq!(proper_count, 24);
        let (h_full, _) =
            voxelean::OhElement::canonical_zobrist(&[([1, 2, 3], 1)], 8, false);
        let (h_chiral, _) =
            voxelean::OhElement::canonical_zobrist(&[([1, 2, 3], 1)], 8, true);
        assert!(h_full <= h_chiral);

        // Theorem 11: 3D Schur-Feshbach Octree Wavelet Split (1 + 7 = 8)
        let block = [10.0, 12.0, 14.0, 16.0, 20.0, 22.0, 24.0, 26.0];
        let wav = voxelean::SchurFeshbachOctreeWavelet::decompose_block_2x2x2(&block);
        assert!((wav.dc_coarse - 18.0).abs() < 1e-12);
        assert!(wav.total_energy > wav.retained_energy);

        // Theorem 12: Discrete Exterior Calculus & Euler-Poincare Betti Identity
        // Build a 3x3x1 hollow ring of 8 voxels (1-tunnel torus -> beta0=1, beta1=1, beta2=0, chi=0)
        let mut ring = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                if !(x == 1 && y == 1) {
                    ring.push((x, y, 0));
                }
            }
        }
        let complex = voxelean::VoxelComplex3D::from_occupied_voxels(&ring);
        assert_eq!(complex.euler_characteristic(), complex.betti_euler());
        assert_eq!(complex.beta0, 1);
        assert_eq!(complex.beta1, 1);
        assert_eq!(complex.beta2, 0);
        assert_eq!(complex.euler_characteristic(), 0);

        // Theorem 13, 14, 15, 16
        let (d1, d2, bflux) = voxelean::gauss_divergence_two_cell_flux(5.0, 3.0, -2.0);
        assert!((d1 + d2 - bflux).abs() < 1e-12);
        assert!((bflux - 3.0).abs() < 1e-12);
        let (b2, b3) = voxelean::landauer_trihedral_barrier(4.5);
        assert!(b3 > b2);
        let rays = voxelean::split_triaxial_ray_flux(100.0, 0.25, 0.4);
        assert!((rays[0] + rays[1] + rays[2] - 100.0).abs() < 1e-12);
        let star = voxelean::StarStencil7PointGfp::identify_from_impulse(251, 9, 1, 2, 3, 4, 5, 6);
        assert_eq!(star.c0, 9);
    }

    #[test]
    fn pillar_iii_worldtube_theorems_17_to_24_hold() {
        reset_mind();
        let cont = worldtube::WorldtubeContinuity::evaluate(100.0, 15.0, 15.0, 0.0, 0.0);
        assert!(cont.is_conservative());
        assert!((cont.m_t1 - 100.0).abs() < 1e-12);

        let cfl = worldtube::CflLightConePruner::new(3, 2, 1000);
        assert_eq!(cfl.cone_volume, 125);
        assert!(cfl.cone_volume < cfl.full_volume);

        let wt = worldtube::WorldtubeTrajectoryFactor::factor_3d(64, 10, [1.0, 0.0, -1.0]);
        assert_eq!(wt.unfactored_cost, 640);
        assert_eq!(wt.factored_cost, 67);

        assert!((worldtube::hyper_voxelean_wavelet_ratio(2) - 0.75).abs() < 1e-12);
        assert!((worldtube::hyper_voxelean_wavelet_ratio(3) - 0.875).abs() < 1e-12);
        assert!((worldtube::hyper_voxelean_wavelet_ratio(4) - 0.9375).abs() < 1e-12);

        assert_eq!(worldtube::hyperoctahedral_order(1), 2);
        assert_eq!(worldtube::hyperoctahedral_order(2), 8);
        assert_eq!(worldtube::hyperoctahedral_order(3), 48);
        assert_eq!(worldtube::hyperoctahedral_order(4), 384);

        let (unbent, unfac) = worldtube::unbent_ray_exponentiation_costs(3, 4);
        assert_eq!(unbent, 6);
        assert_eq!(unfac, 1296);

        let (dv_naive, dv_macro) = worldtube::bent_geodesic_renormalization(2.0, 3.0, 2.5, 2.0);
        assert!(dv_naive > 0.0 && dv_macro < 0.0);

        let cert = worldtube::GrandSynthesisCertificate::certify(3, 12, 50);
        assert_eq!(cert.symmetry_group_order, 48);
        assert!(cert.worldtube_compression_ratio > 0.9);
    }

    #[test]
    fn counterfactual_mechinterp_ablation_tracks_causal_circuit_deltas() {
        reset_mind();
        let mut gray = Mat::zeros(16, 16, core::CV_8UC1).unwrap();
        for r in 4..12 {
            for c in 4..12 {
                gray.set_f64(r, c, 0, 128.0 + 4.0); // bit 7 and bit 2 set
            }
        }
        let baseline = pixelean::BitplanePartition::from_gray_mat(&gray, 64).unwrap();
        assert_eq!(baseline.fg_mass, 64);

        // Counterfactual intervention: ablate bitplane 7 (value 128)
        let (ablated_part, summary) = with_counterfactual_ablation(
            AblationConfig {
                ablated_bitplanes: vec![7],
                ..Default::default()
            },
            || pixelean::BitplanePartition::from_gray_mat(&gray, 64).unwrap(),
        );
        assert_eq!(ablated_part.fg_mass, 0);
        assert!(summary.total_events >= 1);
        assert!(summary.theorem_counts.contains_key("THM-01-BITPLANE-MASS"));
    }

    #[test]
    fn full_cv_pipeline_optical_flow_pnp_stereo_and_grabcut_work_in_head() {
        reset_mind();
        let (w, h) = (96, 72);
        let mut img0 = Mat::zeros(h, w, core::CV_8UC1).unwrap();
        let mut img1 = Mat::zeros(h, w, core::CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                let v0 = (((x * 37) ^ (y * 53)) % 180 + 30) as f64;
                img0.set_f64(y, x, 0, v0);
                let sx = (x - 2).clamp(0, w - 1);
                let sy = (y - 1).clamp(0, h - 1);
                let v1 = (((sx * 37) ^ (sy * 53)) % 180 + 30) as f64;
                img1.set_f64(y, x, 0, v1);
            }
        }

        let mut corners = Vector::<Point2f>::new();
        imgproc::good_features_to_track(
            &img0,
            &mut corners,
            64,
            0.05,
            5.0,
            &core::no_array(),
            3,
            false,
            0.04,
        )
        .unwrap();
        assert!(corners.len() >= 16);

        let mut next_pts = Vector::<Point2f>::new();
        let mut status = Vector::<u8>::new();
        let mut err = Vector::<f32>::new();
        let criteria = TermCriteria::new(3, 20, 0.02).unwrap();
        video::calc_optical_flow_pyr_lk(
            &img0,
            &img1,
            &corners,
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

        let mut dxs = Vec::new();
        for i in 0..corners.len() {
            if status.get(i).unwrap() == 1 {
                dxs.push(next_pts.get(i).unwrap().x - corners.get(i).unwrap().x);
            }
        }
        dxs.sort_by(f32::total_cmp);
        assert!((dxs[dxs.len() / 2] - 2.0).abs() < 0.25);

        // PnP RANSAC test
        let obj = Vector::<Point3f>::from_slice(&[
            Point3f::new(-1.0, -1.0, 0.0),
            Point3f::new(1.0, -1.0, 0.2),
            Point3f::new(-1.0, 1.0, 0.1),
            Point3f::new(1.0, 1.0, 0.3),
            Point3f::new(-0.5, -0.4, 1.0),
            Point3f::new(0.7, -0.2, 0.8),
            Point3f::new(-0.3, 0.7, 1.2),
            Point3f::new(0.5, 0.5, 1.4),
        ]);
        let mut uv = Vector::<Point2f>::new();
        for p in &obj {
            uv.push(Point2f::new(
                (p.x + 0.3) * 100.0 / (p.z + 4.0) + 48.0,
                (p.y - 0.2) * 100.0 / (p.z + 4.0) + 36.0,
            ));
        }
        let k = Mat::from_slice_2d(&[[100.0, 0.0, 48.0], [0.0, 100.0, 36.0], [0.0, 0.0, 1.0]]).unwrap();
        let (mut rvec, mut tvec, mut inl) = (Mat::default(), Mat::default(), Mat::default());
        assert!(calib3d::solve_pnp_ransac(
            &obj,
            &uv,
            &k,
            &core::no_array(),
            &mut rvec,
            &mut tvec,
            false,
            100,
            1.0,
            0.99,
            &mut inl,
            calib3d::SOLVEPNP_EPNP,
        )
        .unwrap());
        assert_eq!(inl.total(), 8);
        assert!((*tvec.at::<f64>(0).unwrap() - 0.3).abs() < 1e-4);
        assert!((*tvec.at::<f64>(1).unwrap() + 0.2).abs() < 1e-4);
        assert!((*tvec.at::<f64>(2).unwrap() - 4.0).abs() < 1e-4);

        // GrabCut & Connected Components test
        let mut rgb = Mat::new_rows_cols_with_default(32, 32, core::CV_8UC3, Scalar::new(20.0, 20.0, 25.0, 0.0)).unwrap();
        imgproc::rectangle(
            &mut rgb,
            Rect::new(10, 10, 12, 12),
            Scalar::new(240.0, 220.0, 30.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )
        .unwrap();
        let mut gc_mask = Mat::default();
        let mut bgd = Mat::default();
        let mut fgd = Mat::default();
        imgproc::grab_cut(
            &rgb,
            &mut gc_mask,
            Rect::new(8, 8, 16, 16),
            &mut bgd,
            &mut fgd,
            3,
            imgproc::GC_INIT_WITH_RECT,
        )
        .unwrap();
        assert_eq!(
            *gc_mask.at_2d::<u8>(15, 15).unwrap(),
            imgproc::GC_PR_FGD as u8
        );
        let _ = Point::new(0, 0);

        let summary = mechinterp::mind_summary();
        assert!(summary.total_events >= 5);
    }

    #[test]
    fn cloud_ws_and_zobrist_feistel_bounty_seal_hold() {
        let unlocked = cloud_ws::unlock_vault_seal(0x534F_4C49_4453_4621);
        assert!(unlocked.contains("andy@solidsf.com"));
        assert!(unlocked.contains("fly you out to San Francisco"));

        // Verify solidSF Credits conversion and 30% markup ceiling rounding
        assert_eq!(cloud_ws::purchase_cents_to_microusd(1_000).unwrap(), 10_000_000);
        assert_eq!(cloud_ws::retail_debit_microusd(0.000010, 30), Some(13));

        let server = cloud_ws::SolidCvLoopbackWsServer::spawn_ephemeral(3).unwrap();
        let mut client = cloud_ws::SolidCvWsClient::connect(&server.ws_url, Some("test-key")).unwrap();

        let r1 = realitycomb::dispatch_over_ws(&mut client, 256, 256, 12).unwrap();
        assert!(r1.ok);
        assert_eq!(r1.engine, cloud_ws::WsEngine::RealityComber);
        assert_eq!(r1.credits_charged_microusd, 15);
        assert_eq!(r1.credits_remaining_microusd, 10_000_000 - 15);
        assert_eq!(r1.billing_status, "debited");

        let r2 = voxelean::dispatch_over_ws(&mut client, 64, 0xAAAA, 0xBBBB).unwrap();
        assert!(r2.ok);
        assert_eq!(r2.engine, cloud_ws::WsEngine::VoxeleanOctree);
        assert_eq!(r2.credits_charged_microusd, 5);
        assert_eq!(r2.credits_remaining_microusd, 10_000_000 - 20);

        let r3 = worldtube::dispatch_over_ws(&mut client, 64, 2.0).unwrap();
        assert!(r3.ok);
        assert_eq!(r3.engine, cloud_ws::WsEngine::Worldtube4D);
        assert_eq!(r3.credits_charged_microusd, 10);
        assert_eq!(r3.credits_remaining_microusd, 10_000_000 - 30);

        server.join();

        // Verify fail-closed 402 Payment Required when credits are depleted
        let low_server = cloud_ws::SolidCvLoopbackWsServer::spawn_with_balance(2, 15).unwrap();
        let mut low_client =
            cloud_ws::SolidCvWsClient::connect(&low_server.ws_url, Some("low-key")).unwrap();
        let ok_call = realitycomb::dispatch_over_ws(&mut low_client, 128, 128, 4).unwrap();
        assert_eq!(ok_call.credits_remaining_microusd, 0);
        let depleted = realitycomb::dispatch_over_ws(&mut low_client, 128, 128, 4);
        assert!(depleted.is_err());
        let err = depleted.unwrap_err();
        assert_eq!(err.code, 402);
        assert!(err.message.contains("https://solidsf.com/compute-credits"));
        low_server.join();
    }
}
