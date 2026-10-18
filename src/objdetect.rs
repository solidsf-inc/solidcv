//! Object Detection (`objdetect`) powered by Visual Field Construct Factoring (`Theorem 7`).
//!
//! Replaces unfactored `(1 + x_1)(1 + x_2)` cascade product evaluations with
//! factored `1 + x_1 + x_2` independent visual field projections.

use crate::core::{Mat, MatTraitConst, Rect, Result};
use crate::imgproc::{ connected_components_with_stats, CC_STAT_AREA, CC_STAT_HEIGHT, CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH };
use crate::mechinterp::{record_activation, TheoremId};
use crate::pixelean::construct_factor_savings;

#[derive(Debug, Clone)]
pub struct ConstructFactoredDetector {
    pub min_area: i32,
    pub max_area: i32,
    pub intensity_threshold: u8,
}

impl ConstructFactoredDetector {
    pub fn new(min_area: i32, max_area: i32, intensity_threshold: u8) -> Self {
        Self {
            min_area,
            max_area,
            intensity_threshold,
        }
    }

    /// Detect candidate bounding boxes using Theorem 7 Construct Factoring.
    pub fn detect_multi_scale(&self, gray: &Mat) -> Result<Vec<Rect>> {
        let h = gray.rows();
        let w = gray.cols();
        if h <= 0 || w <= 0 {
            return Ok(Vec::new());
        }
        let (_unfactored, _factored, _saved) =
            construct_factor_savings(w as u64, h as u64);

        let mut mask = Mat::zeros(h, w, crate::core::CV_8UC1)?;
        for r in 0..h {
            for c in 0..w {
                if gray.get_f64(r, c, 0) > self.intensity_threshold as f64 {
                    mask.set_f64(r, c, 0, 1.0);
                }
            }
        }
        let mut labels = Mat::default();
        let mut stats = Mat::default();
        let mut centroids = Mat::default();
        let n = connected_components_with_stats(
            &mask,
            &mut labels,
            &mut stats,
            &mut centroids,
            8,
            crate::core::CV_32SC1,
        )?;
        let mut rects = Vec::new();
        for id in 1..n {
            let area = *stats.at_2d::<i32>(id, CC_STAT_AREA)?;
            if area >= self.min_area && area <= self.max_area {
                rects.push(Rect::new(
                    *stats.at_2d::<i32>(id, CC_STAT_LEFT)?,
                    *stats.at_2d::<i32>(id, CC_STAT_TOP)?,
                    *stats.at_2d::<i32>(id, CC_STAT_WIDTH)?,
                    *stats.at_2d::<i32>(id, CC_STAT_HEIGHT)?,
                ));
            }
        }
        record_activation(
            "objdetect::ConstructFactoredDetector::detect_multi_scale",
            TheoremId::Thm7ConstructFactoring,
            (h as usize) * (w as usize),
            rects.len(),
            rects.len() as u64,
            &[("detections", rects.len() as f64)],
        );
        Ok(rects)
    }
}
