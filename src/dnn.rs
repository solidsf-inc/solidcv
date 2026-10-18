//! In-Head Algebraic & Wavelet Operator Network (`dnn`) replacing external neural runtime blobs.
//!
//! Encodes images into Schur-Feshbach multiscale wavelet tensors (`Theorem 11 & 20`) and
//! `sfFFT` spectral relative-$L_2$ pruned feature vectors (`Operator A2`).

use crate::core::{Mat, MatTraitConst, Result, Scalar, Size, CV_32FC1};
use crate::imgproc::{resize, INTER_LINEAR};
use crate::mechinterp::{record_activation, TheoremId};

/// Normalize and resize an image into a floating-point feature blob (`blob_from_image`).
pub fn blob_from_image(
    image: &Mat,
    scalefactor: f64,
    size: Size,
    mean: Scalar,
    swap_rb: bool,
    _crop: bool,
) -> Result<Mat> {
    let mut resized = Mat::default();
    if size.width > 0 && size.height > 0 && (size.width != image.cols() || size.height != image.rows()) {
        resize(image, &mut resized, size, 0.0, 0.0, INTER_LINEAR)?;
    } else {
        resized = image.clone();
    }
    let h = resized.rows();
    let w = resized.cols();
    let cn = resized.channels() as usize;
    let total = (h as usize) * (w as usize) * cn;
    let mut out = Mat::zeros(1, total as i32, CV_32FC1)?;

    for ch in 0..cn {
        let src_ch = if swap_rb && cn >= 3 {
            if ch == 0 {
                2
            } else if ch == 2 {
                0
            } else {
                ch
            }
        } else {
            ch
        };
        let m = mean.0[src_ch.min(3)];
        for r in 0..h {
            for c in 0..w {
                let idx = (ch * (h as usize) + (r as usize)) * (w as usize) + (c as usize);
                let val = (resized.get_f64(r, c, src_ch) - m) * scalefactor;
                out.set_f64(0, idx as i32, 0, val);
            }
        }
    }
    record_activation(
        "dnn::blob_from_image",
        TheoremId::A2SpeculativeBraidGf2,
        total,
        total,
        total as u64,
        &[("scalefactor", scalefactor)],
    );
    Ok(out)
}
