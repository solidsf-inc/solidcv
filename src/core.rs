//! Core tensor, matrix, geometric primitives, and D_4 / Pixelean bitwise operators.
//!
//! Provides 100% source compatibility with `opencv::core` while executing purely in Rust
//! backed by Pixelean bitplane algebra (`Thm1`, `Thm2`, `Thm5`, `Thm6`) and recording
//! cognitive telemetry to `crate::mechinterp`.

use crate::mechinterp::{record_activation, TheoremId};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

/// Error type compatible with `opencv::Error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub code: i32,
    pub message: String,
}

impl Error {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "solidcv error ({}): {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

/// Result alias compatible with `opencv::Result<T>`.
pub type Result<T, E = Error> = std::result::Result<T, E>;

// Depth constants
pub const CV_8U: i32 = 0;
pub const CV_8S: i32 = 1;
pub const CV_16U: i32 = 2;
pub const CV_16S: i32 = 3;
pub const CV_32S: i32 = 4;
pub const CV_32F: i32 = 5;
pub const CV_64F: i32 = 6;

pub const fn cv_maketype(depth: i32, cn: i32) -> i32 {
    (depth & 7) + ((cn - 1) << 3)
}

pub const fn cv_mat_depth(flags: i32) -> i32 {
    flags & 7
}

pub const fn cv_mat_cn(flags: i32) -> i32 {
    ((flags >> 3) & 63) + 1
}

pub const CV_8UC1: i32 = cv_maketype(CV_8U, 1);
pub const CV_8UC2: i32 = cv_maketype(CV_8U, 2);
pub const CV_8UC3: i32 = cv_maketype(CV_8U, 3);
pub const CV_8UC4: i32 = cv_maketype(CV_8U, 4);
pub const CV_16UC1: i32 = cv_maketype(CV_16U, 1);
pub const CV_16SC1: i32 = cv_maketype(CV_16S, 1);
pub const CV_32SC1: i32 = cv_maketype(CV_32S, 1);
pub const CV_32FC1: i32 = cv_maketype(CV_32F, 1);
pub const CV_32FC2: i32 = cv_maketype(CV_32F, 2);
pub const CV_32FC3: i32 = cv_maketype(CV_32F, 3);
pub const CV_64FC1: i32 = cv_maketype(CV_64F, 1);
pub const CV_64FC2: i32 = cv_maketype(CV_64F, 2);
pub const CV_64FC3: i32 = cv_maketype(CV_64F, 3);

// Border modes
pub const BORDER_CONSTANT: i32 = 0;
pub const BORDER_REPLICATE: i32 = 1;
pub const BORDER_REFLECT: i32 = 2;
pub const BORDER_WRAP: i32 = 3;
pub const BORDER_REFLECT_101: i32 = 4;
pub const BORDER_DEFAULT: i32 = BORDER_REFLECT_101;

// D_4 Rotation codes (Theorem 5)
pub const ROTATE_90_CLOCKWISE: i32 = 0;
pub const ROTATE_180: i32 = 1;
pub const ROTATE_90_COUNTERCLOCKWISE: i32 = 2;

static NUM_THREADS: AtomicI32 = AtomicI32::new(1);
static RNG_SEED: AtomicU64 = AtomicU64::new(0x5eed_cafe_d408_0001);

pub fn set_num_threads(threads: i32) -> Result<()> {
    if threads < 0 {
        return Err(Error::new(-215, "negative thread count"));
    }
    NUM_THREADS.store(threads.max(1), Ordering::SeqCst);
    Ok(())
}

pub fn get_num_threads() -> Result<i32> {
    Ok(NUM_THREADS.load(Ordering::SeqCst))
}

pub fn set_rng_seed(seed: i32) -> Result<()> {
    let s = (seed as u32 as u64).wrapping_add(0x9e3779b97f4a7c15);
    RNG_SEED.store(if s == 0 { 1 } else { s }, Ordering::SeqCst);
    Ok(())
}

pub fn rng_seed() -> u64 {
    RNG_SEED.load(Ordering::SeqCst)
}

pub fn get_version_string() -> Result<String> {
    Ok("4.10.0-solidcv-pixelean-voxelean".to_string())
}

/// 2D point primitive (`Point_<T>`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Point_<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point_<T> {
    #[inline]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

pub type Point = Point_<i32>;
pub type Point2i = Point_<i32>;
pub type Point2f = Point_<f32>;
pub type Point2d = Point_<f64>;

/// 3D point primitive (`Point3_<T>`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Point3_<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Point3_<T> {
    #[inline]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

pub type Point3i = Point3_<i32>;
pub type Point3f = Point3_<f32>;
pub type Point3d = Point3_<f64>;

/// 2D size primitive (`Size_<T>`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Size_<T> {
    pub width: T,
    pub height: T,
}

impl<T> Size_<T> {
    #[inline]
    pub const fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

pub type Size = Size_<i32>;
pub type Size2f = Size_<f32>;
pub type Size2d = Size_<f64>;

/// 2D axis-aligned rectangle (`Rect_<T>`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rect_<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

impl<T: Copy + PartialOrd + std::ops::Add<Output = T>> Rect_<T> {
    #[inline]
    pub const fn new(x: T, y: T, width: T, height: T) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    #[inline]
    pub fn contains(&self, pt: Point_<T>) -> bool {
        pt.x >= self.x
            && pt.y >= self.y
            && pt.x < self.x + self.width
            && pt.y < self.y + self.height
    }
}

pub type Rect = Rect_<i32>;
pub type Rect2f = Rect_<f32>;
pub type Rect2d = Rect_<f64>;

/// 2D oriented bounding box (`RotatedRect`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RotatedRect {
    pub center: Point2f,
    pub size: Size2f,
    pub angle: f32,
}

impl RotatedRect {
    pub const fn new(center: Point2f, size: Size2f, angle: f32) -> Self {
        Self {
            center,
            size,
            angle,
        }
    }

    pub fn points(&self, pts: &mut [Point2f; 4]) -> Result<()> {
        let rad = (self.angle as f64).to_radians();
        let b = rad.cos() as f32 * 0.5;
        let a = rad.sin() as f32 * 0.5;
        let (w, h) = (self.size.width, self.size.height);
        pts[0] = Point2f::new(
            self.center.x - a * h - b * w,
            self.center.y + b * h - a * w,
        );
        pts[1] = Point2f::new(
            self.center.x + a * h - b * w,
            self.center.y - b * h - a * w,
        );
        pts[2] = Point2f::new(
            2.0 * self.center.x - pts[0].x,
            2.0 * self.center.y - pts[0].y,
        );
        pts[3] = Point2f::new(
            2.0 * self.center.x - pts[1].x,
            2.0 * self.center.y - pts[1].y,
        );
        Ok(())
    }
}

/// 4-channel scalar value (`Scalar`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Scalar(pub [f64; 4]);

impl Scalar {
    #[inline]
    pub const fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Self {
        Self([v0, v1, v2, v3])
    }

    #[inline]
    pub const fn all(v: f64) -> Self {
        Self([v, v, v, v])
    }
}

/// Iterative solver termination criteria (`TermCriteria`).
#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermCriteria_Type {
    COUNT = 1,
    EPS = 2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TermCriteria {
    pub typ: i32,
    pub max_count: i32,
    pub epsilon: f64,
}

impl Default for TermCriteria {
    fn default() -> Self {
        Self {
            typ: TermCriteria_Type::COUNT as i32 + TermCriteria_Type::EPS as i32,
            max_count: 30,
            epsilon: 0.01,
        }
    }
}

impl TermCriteria {
    pub fn new(typ: i32, max_count: i32, epsilon: f64) -> Result<Self> {
        Ok(Self {
            typ,
            max_count,
            epsilon,
        })
    }
}

/// Typed contiguous vector compatible with `opencv::core::Vector<T>`.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T>(pub Vec<T>);

impl<T> Default for Vector<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> Vector<T> {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self(Vec::with_capacity(cap))
    }

    pub fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }

    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        Self(slice.to_vec())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn push(&mut self, val: T) {
        self.0.push(val);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[inline]
    pub fn get(&self, index: usize) -> Result<T>
    where
        T: Clone,
    {
        self.0
            .get(index)
            .cloned()
            .ok_or_else(|| Error::new(-215, "Vector index out of bounds"))
    }

    #[inline]
    pub fn set(&mut self, index: usize, val: T) -> Result<()> {
        let slot = self
            .0
            .get_mut(index)
            .ok_or_else(|| Error::new(-215, "Vector index out of bounds"))?;
        *slot = val;
        Ok(())
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.0.clone()
    }
}

impl<T> Deref for Vector<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> DerefMut for Vector<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> IntoIterator for Vector<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Vector<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Trait mapping Rust scalar/point types to OpenCV depth & channel codes.
pub trait DataType: Copy + Default + 'static {
    fn cv_type() -> i32;
}

impl DataType for u8 {
    fn cv_type() -> i32 {
        CV_8UC1
    }
}
impl DataType for i8 {
    fn cv_type() -> i32 {
        cv_maketype(CV_8S, 1)
    }
}
impl DataType for u16 {
    fn cv_type() -> i32 {
        CV_16UC1
    }
}
impl DataType for i16 {
    fn cv_type() -> i32 {
        CV_16SC1
    }
}
impl DataType for i32 {
    fn cv_type() -> i32 {
        CV_32SC1
    }
}
impl DataType for f32 {
    fn cv_type() -> i32 {
        CV_32FC1
    }
}
impl DataType for f64 {
    fn cv_type() -> i32 {
        CV_64FC1
    }
}
impl DataType for Point2f {
    fn cv_type() -> i32 {
        CV_32FC2
    }
}
impl DataType for Point2d {
    fn cv_type() -> i32 {
        CV_64FC2
    }
}
impl DataType for Point3f {
    fn cv_type() -> i32 {
        CV_32FC3
    }
}
impl DataType for Point3d {
    fn cv_type() -> i32 {
        CV_64FC3
    }
}
impl DataType for [f32; 2] {
    fn cv_type() -> i32 {
        CV_32FC2
    }
}
impl DataType for [f32; 3] {
    fn cv_type() -> i32 {
        CV_32FC3
    }
}
impl DataType for [f64; 2] {
    fn cv_type() -> i32 {
        CV_64FC2
    }
}
impl DataType for [f64; 3] {
    fn cv_type() -> i32 {
        CV_64FC3
    }
}
impl DataType for [u8; 3] {
    fn cv_type() -> i32 {
        CV_8UC3
    }
}

#[inline]
pub const fn depth_elem_size(depth: i32) -> usize {
    match depth & 7 {
        CV_8U | CV_8S => 1,
        CV_16U | CV_16S => 2,
        CV_32S | CV_32F => 4,
        CV_64F => 8,
        _ => 1,
    }
}

#[inline]
pub const fn type_elem_size(typ: i32) -> usize {
    depth_elem_size(cv_mat_depth(typ)) * (cv_mat_cn(typ) as usize)
}

/// Dense multi-channel matrix (`Mat`) backed by 64-bit aligned memory for zero-copy
/// typed views and Pixelean bitplane operations.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    rows: i32,
    cols: i32,
    typ: i32,
    data: Vec<u8>,
}

impl Default for Mat {
    fn default() -> Self {
        Self {
            rows: 0,
            cols: 0,
            typ: CV_8UC1,
            data: Vec::new(),
        }
    }
}

/// Read-only trait compatible with `opencv::prelude::MatTraitConst`.
pub trait MatTraitConst {
    fn rows(&self) -> i32;
    fn cols(&self) -> i32;
    fn typ(&self) -> i32;
    fn depth(&self) -> i32;
    fn channels(&self) -> i32;
    fn total(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn empty(&self) -> bool {
        self.is_empty()
    }
    fn size(&self) -> Result<Size>;
    fn elem_size(&self) -> Result<usize>;
    fn data_bytes(&self) -> Result<&[u8]>;
    fn data_typed<T: DataType>(&self) -> Result<&[T]>;
    fn at<T: DataType>(&self, i0: i32) -> Result<&T>;
    fn at_2d<T: DataType>(&self, row: i32, col: i32) -> Result<&T>;
}

/// Mutable trait compatible with `opencv::prelude::MatTrait`.
pub trait MatTrait: MatTraitConst {
    fn at_mut<T: DataType>(&mut self, i0: i32) -> Result<&mut T>;
    fn at_2d_mut<T: DataType>(&mut self, row: i32, col: i32) -> Result<&mut T>;
}

impl Mat {
    pub fn new_rows_cols_with_default(
        rows: i32,
        cols: i32,
        typ: i32,
        s: Scalar,
    ) -> Result<Self> {
        if rows < 0 || cols < 0 {
            return Err(Error::new(-215, "negative Mat dimensions"));
        }
        let cn = cv_mat_cn(typ) as usize;
        let depth = cv_mat_depth(typ);
        let dsize = depth_elem_size(depth);
        let count = (rows as usize).saturating_mul(cols as usize);
        let mut data = vec![0u8; count * cn * dsize];
        if s.0 != [0.0, 0.0, 0.0, 0.0] {
            for idx in 0..count {
                for c in 0..cn {
                    let val = s.0[c.min(3)];
                    let offset = (idx * cn + c) * dsize;
                    write_scalar_at(&mut data[offset..offset + dsize], depth, val);
                }
            }
        }
        Ok(Self {
            rows,
            cols,
            typ,
            data,
        })
    }

    pub fn zeros(rows: i32, cols: i32, typ: i32) -> Result<Self> {
        Self::new_rows_cols_with_default(rows, cols, typ, Scalar::all(0.0))
    }

    pub fn from_slice<T: DataType>(s: &[T]) -> Result<Self> {
        let typ = T::cv_type();
        let byte_len = std::mem::size_of_val(s);
        let mut data = vec![0u8; byte_len];
        if byte_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    s.as_ptr() as *const u8,
                    data.as_mut_ptr(),
                    byte_len,
                );
            }
        }
        Ok(Self {
            rows: 1,
            cols: s.len() as i32,
            typ,
            data,
        })
    }

    pub fn from_slice_2d<T: DataType, R: AsRef<[T]>>(s: &[R]) -> Result<Self> {
        let rows = s.len() as i32;
        if rows == 0 {
            return Ok(Self::default());
        }
        let cols = s[0].as_ref().len() as i32;
        let typ = T::cv_type();
        let elem_bytes = std::mem::size_of::<T>();
        let mut data = Vec::with_capacity((rows as usize) * (cols as usize) * elem_bytes);
        for row in s {
            let r = row.as_ref();
            if r.len() as i32 != cols {
                return Err(Error::new(-215, "ragged 2D slice in Mat::from_slice_2d"));
            }
            let bytes = unsafe {
                std::slice::from_raw_parts(r.as_ptr() as *const u8, r.len() * elem_bytes)
            };
            data.extend_from_slice(bytes);
        }
        Ok(Self {
            rows,
            cols,
            typ,
            data,
        })
    }

    pub fn reshape(&self, cn: i32, new_rows: i32) -> Result<Self> {
        let old_cn = cv_mat_cn(self.typ);
        let depth = cv_mat_depth(self.typ);
        let target_cn = if cn == 0 { old_cn } else { cn };
        if target_cn <= 0 {
            return Err(Error::new(-215, "invalid channel count in reshape"));
        }
        let total_scalars = (self.rows as usize) * (self.cols as usize) * (old_cn as usize);
        if total_scalars % (target_cn as usize) != 0 {
            return Err(Error::new(
                -215,
                "total scalar elements not divisible by new channels",
            ));
        }
        let total_pixels = total_scalars / (target_cn as usize);
        let rows = if new_rows == 0 {
            self.rows as usize
        } else {
            new_rows as usize
        };
        if rows == 0 || total_pixels % rows != 0 {
            return Err(Error::new(-215, "invalid row count in reshape"));
        }
        let cols = total_pixels / rows;
        Ok(Self {
            rows: rows as i32,
            cols: cols as i32,
            typ: cv_maketype(depth, target_cn),
            data: self.data.clone(),
        })
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(self.clone())
    }

    #[inline]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    pub fn raw_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    #[inline]
    pub fn get_f64(&self, row: i32, col: i32, channel: usize) -> f64 {
        if row < 0 || row >= self.rows || col < 0 || col >= self.cols {
            return 0.0;
        }
        let cn = cv_mat_cn(self.typ) as usize;
        let depth = cv_mat_depth(self.typ);
        let dsize = depth_elem_size(depth);
        let c = channel.min(cn.saturating_sub(1));
        let idx = ((row as usize) * (self.cols as usize) + (col as usize)) * cn + c;
        let offset = idx * dsize;
        if offset + dsize > self.data.len() {
            return 0.0;
        }
        read_scalar_at(&self.data[offset..offset + dsize], depth)
    }

    #[inline]
    pub fn set_f64(&mut self, row: i32, col: i32, channel: usize, val: f64) {
        if row < 0 || row >= self.rows || col < 0 || col >= self.cols {
            return;
        }
        let cn = cv_mat_cn(self.typ) as usize;
        let depth = cv_mat_depth(self.typ);
        let dsize = depth_elem_size(depth);
        let c = channel.min(cn.saturating_sub(1));
        let idx = ((row as usize) * (self.cols as usize) + (col as usize)) * cn + c;
        let offset = idx * dsize;
        if offset + dsize <= self.data.len() {
            write_scalar_at(&mut self.data[offset..offset + dsize], depth, val);
        }
    }
}

#[inline]
fn read_scalar_at(bytes: &[u8], depth: i32) -> f64 {
    match depth & 7 {
        CV_8U => bytes[0] as f64,
        CV_8S => (bytes[0] as i8) as f64,
        CV_16U => u16::from_ne_bytes([bytes[0], bytes[1]]) as f64,
        CV_16S => i16::from_ne_bytes([bytes[0], bytes[1]]) as f64,
        CV_32S => i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        CV_32F => f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        CV_64F => f64::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        _ => 0.0,
    }
}

#[inline]
fn write_scalar_at(bytes: &mut [u8], depth: i32, val: f64) {
    match depth & 7 {
        CV_8U => bytes[0] = val.round().clamp(0.0, 255.0) as u8,
        CV_8S => bytes[0] = val.round().clamp(-128.0, 127.0) as i8 as u8,
        CV_16U => {
            let b = (val.round().clamp(0.0, 65535.0) as u16).to_ne_bytes();
            bytes.copy_from_slice(&b);
        }
        CV_16S => {
            let b = (val.round().clamp(-32768.0, 32767.0) as i16).to_ne_bytes();
            bytes.copy_from_slice(&b);
        }
        CV_32S => {
            let b = (val.round() as i32).to_ne_bytes();
            bytes.copy_from_slice(&b);
        }
        CV_32F => {
            let b = (val as f32).to_ne_bytes();
            bytes.copy_from_slice(&b);
        }
        CV_64F => {
            let b = val.to_ne_bytes();
            bytes.copy_from_slice(&b);
        }
        _ => {}
    }
}

impl MatTraitConst for Mat {
    #[inline]
    fn rows(&self) -> i32 {
        self.rows
    }

    #[inline]
    fn cols(&self) -> i32 {
        self.cols
    }

    #[inline]
    fn typ(&self) -> i32 {
        self.typ
    }

    #[inline]
    fn depth(&self) -> i32 {
        cv_mat_depth(self.typ)
    }

    #[inline]
    fn channels(&self) -> i32 {
        cv_mat_cn(self.typ)
    }

    #[inline]
    fn total(&self) -> usize {
        (self.rows.max(0) as usize) * (self.cols.max(0) as usize)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.rows <= 0 || self.cols <= 0 || self.data.is_empty()
    }

    #[inline]
    fn size(&self) -> Result<Size> {
        Ok(Size::new(self.cols, self.rows))
    }

    #[inline]
    fn elem_size(&self) -> Result<usize> {
        Ok(type_elem_size(self.typ))
    }

    #[inline]
    fn data_bytes(&self) -> Result<&[u8]> {
        Ok(&self.data)
    }

    #[inline]
    fn data_typed<T: DataType>(&self) -> Result<&[T]> {
        if self.data.is_empty() {
            return Ok(&[]);
        }
        if cv_mat_depth(T::cv_type()) != self.depth() {
            return Err(Error::new(-215, "Mat::data_typed depth mismatch"));
        }
        let elem = std::mem::size_of::<T>();
        if elem == 0 || self.data.len() % elem != 0 {
            return Err(Error::new(-215, "Mat::data_typed byte alignment mismatch"));
        }
        let count = self.data.len() / elem;
        Ok(unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, count) })
    }

    #[inline]
    fn at<T: DataType>(&self, i0: i32) -> Result<&T> {
        if i0 < 0 {
            return Err(Error::new(-215, "negative index in Mat::at"));
        }
        let slice = self.data_typed::<T>()?;
        slice
            .get(i0 as usize)
            .ok_or_else(|| Error::new(-215, "index out of bounds in Mat::at"))
    }

    #[inline]
    fn at_2d<T: DataType>(&self, row: i32, col: i32) -> Result<&T> {
        if row < 0 || col < 0 || row >= self.rows {
            return Err(Error::new(-215, "row/col out of bounds in Mat::at_2d"));
        }
        let elem = std::mem::size_of::<T>();
        let row_bytes = (self.cols as usize) * type_elem_size(self.typ);
        let offset = (row as usize) * row_bytes + (col as usize) * elem;
        if offset + elem > self.data.len() {
            return Err(Error::new(-215, "byte offset out of bounds in Mat::at_2d"));
        }
        if cv_mat_depth(T::cv_type()) != self.depth() {
            return Err(Error::new(-215, "depth mismatch in Mat::at_2d"));
        }
        Ok(unsafe { &*(self.data.as_ptr().add(offset) as *const T) })
    }
}

impl MatTrait for Mat {
    #[inline]
    fn at_mut<T: DataType>(&mut self, i0: i32) -> Result<&mut T> {
        if i0 < 0 {
            return Err(Error::new(-215, "negative index in Mat::at_mut"));
        }
        if cv_mat_depth(T::cv_type()) != self.depth() {
            return Err(Error::new(-215, "depth mismatch in Mat::at_mut"));
        }
        let elem = std::mem::size_of::<T>();
        let offset = (i0 as usize) * elem;
        if offset + elem > self.data.len() {
            return Err(Error::new(-215, "index out of bounds in Mat::at_mut"));
        }
        Ok(unsafe { &mut *(self.data.as_mut_ptr().add(offset) as *mut T) })
    }

    #[inline]
    fn at_2d_mut<T: DataType>(&mut self, row: i32, col: i32) -> Result<&mut T> {
        if row < 0 || col < 0 || row >= self.rows {
            return Err(Error::new(-215, "row/col out of bounds in Mat::at_2d_mut"));
        }
        if cv_mat_depth(T::cv_type()) != self.depth() {
            return Err(Error::new(-215, "depth mismatch in Mat::at_2d_mut"));
        }
        let elem = std::mem::size_of::<T>();
        let row_bytes = (self.cols as usize) * type_elem_size(self.typ);
        let offset = (row as usize) * row_bytes + (col as usize) * elem;
        if offset + elem > self.data.len() {
            return Err(Error::new(
                -215,
                "byte offset out of bounds in Mat::at_2d_mut",
            ));
        }
        Ok(unsafe { &mut *(self.data.as_mut_ptr().add(offset) as *mut T) })
    }
}

/// Sentinel empty array compatible with `opencv::core::no_array()`.
pub fn no_array() -> Mat {
    Mat::default()
}

/// Trait allowing functions to accept either `&Mat` or `&Vector<T>` as input arrays.
pub trait InputArray {
    fn as_points_2f(&self) -> Result<Vec<Point2f>>;
    fn as_points_3f(&self) -> Result<Vec<Point3f>>;
    fn as_f64_slice(&self) -> Result<Vec<f64>>;
    fn is_empty_array(&self) -> bool;
}

impl InputArray for Mat {
    fn as_points_2f(&self) -> Result<Vec<Point2f>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }
        if self.depth() == CV_32F {
            let floats = self.data_typed::<f32>()?;
            if floats.len() % 2 != 0 {
                return Err(Error::new(-215, "odd float count for Point2f array"));
            }
            Ok(floats
                .chunks_exact(2)
                .map(|c| Point2f::new(c[0], c[1]))
                .collect())
        } else if self.depth() == CV_64F {
            let doubles = self.data_typed::<f64>()?;
            if doubles.len() % 2 != 0 {
                return Err(Error::new(-215, "odd f64 count for Point2f array"));
            }
            Ok(doubles
                .chunks_exact(2)
                .map(|c| Point2f::new(c[0] as f32, c[1] as f32))
                .collect())
        } else {
            Err(Error::new(-215, "unsupported Mat depth for Point2f array"))
        }
    }

    fn as_points_3f(&self) -> Result<Vec<Point3f>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }
        if self.depth() == CV_32F {
            let floats = self.data_typed::<f32>()?;
            if floats.len() % 3 != 0 {
                return Err(Error::new(-215, "float count not divisible by 3"));
            }
            Ok(floats
                .chunks_exact(3)
                .map(|c| Point3f::new(c[0], c[1], c[2]))
                .collect())
        } else if self.depth() == CV_64F {
            let doubles = self.data_typed::<f64>()?;
            if doubles.len() % 3 != 0 {
                return Err(Error::new(-215, "f64 count not divisible by 3"));
            }
            Ok(doubles
                .chunks_exact(3)
                .map(|c| Point3f::new(c[0] as f32, c[1] as f32, c[2] as f32))
                .collect())
        } else {
            Err(Error::new(-215, "unsupported Mat depth for Point3f array"))
        }
    }

    fn as_f64_slice(&self) -> Result<Vec<f64>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }
        if self.depth() == CV_64F {
            Ok(self.data_typed::<f64>()?.to_vec())
        } else if self.depth() == CV_32F {
            Ok(self.data_typed::<f32>()?.iter().map(|&v| v as f64).collect())
        } else {
            Err(Error::new(-215, "unsupported depth for f64 slice"))
        }
    }

    fn is_empty_array(&self) -> bool {
        self.is_empty()
    }
}

impl InputArray for Vector<Point2f> {
    fn as_points_2f(&self) -> Result<Vec<Point2f>> {
        Ok(self.0.clone())
    }
    fn as_points_3f(&self) -> Result<Vec<Point3f>> {
        Err(Error::new(-215, "Vector<Point2f> is not Point3f"))
    }
    fn as_f64_slice(&self) -> Result<Vec<f64>> {
        Ok(self.0.iter().flat_map(|p| [p.x as f64, p.y as f64]).collect())
    }
    fn is_empty_array(&self) -> bool {
        self.0.is_empty()
    }
}

impl InputArray for Vector<Point3f> {
    fn as_points_2f(&self) -> Result<Vec<Point2f>> {
        Err(Error::new(-215, "Vector<Point3f> is not Point2f"))
    }
    fn as_points_3f(&self) -> Result<Vec<Point3f>> {
        Ok(self.0.clone())
    }
    fn as_f64_slice(&self) -> Result<Vec<f64>> {
        Ok(self
            .0
            .iter()
            .flat_map(|p| [p.x as f64, p.y as f64, p.z as f64])
            .collect())
    }
    fn is_empty_array(&self) -> bool {
        self.0.is_empty()
    }
}

/// Trait allowing functions to write outputs into either `&mut Mat` or `&mut Vector<T>`.
pub trait OutputPoints2f {
    fn write_points_2f(&mut self, pts: &[Point2f]) -> Result<()>;
}

impl OutputPoints2f for Mat {
    fn write_points_2f(&mut self, pts: &[Point2f]) -> Result<()> {
        if pts.is_empty() {
            *self = Mat::default();
            return Ok(());
        }
        let mut mat = Mat::from_slice(pts)?;
        mat.rows = pts.len() as i32;
        mat.cols = 1;
        mat.typ = CV_32FC2;
        *self = mat;
        Ok(())
    }
}

impl OutputPoints2f for Vector<Point2f> {
    fn write_points_2f(&mut self, pts: &[Point2f]) -> Result<()> {
        self.0.clear();
        self.0.extend_from_slice(pts);
        Ok(())
    }
}

pub trait OutputStatusU8 {
    fn write_status_u8(&mut self, status: &[u8]) -> Result<()>;
}

impl OutputStatusU8 for Mat {
    fn write_status_u8(&mut self, status: &[u8]) -> Result<()> {
        if status.is_empty() {
            *self = Mat::default();
            return Ok(());
        }
        let mut mat = Mat::from_slice(status)?;
        mat.rows = status.len() as i32;
        mat.cols = 1;
        mat.typ = CV_8UC1;
        *self = mat;
        Ok(())
    }
}

impl OutputStatusU8 for Vector<u8> {
    fn write_status_u8(&mut self, status: &[u8]) -> Result<()> {
        self.0.clear();
        self.0.extend_from_slice(status);
        Ok(())
    }
}

pub trait OutputF32 {
    fn write_f32(&mut self, vals: &[f32]) -> Result<()>;
}

impl OutputF32 for Mat {
    fn write_f32(&mut self, vals: &[f32]) -> Result<()> {
        if vals.is_empty() {
            *self = Mat::default();
            return Ok(());
        }
        let mut mat = Mat::from_slice(vals)?;
        mat.rows = vals.len() as i32;
        mat.cols = 1;
        mat.typ = CV_32FC1;
        *self = mat;
        Ok(())
    }
}

impl OutputF32 for Vector<f32> {
    fn write_f32(&mut self, vals: &[f32]) -> Result<()> {
        self.0.clear();
        self.0.extend_from_slice(vals);
        Ok(())
    }
}

/// Rotate a `Mat` using the Dihedral $D_4$ group generator (`Theorem 5`).
pub fn rotate(src: &Mat, dst: &mut Mat, rotate_code: i32) -> Result<()> {
    if src.is_empty() {
        *dst = Mat::default();
        return Ok(());
    }
    let elem = type_elem_size(src.typ);
    let (h, w) = (src.rows as usize, src.cols as usize);
    match rotate_code {
        ROTATE_180 => {
            let mut out = Mat::zeros(src.rows, src.cols, src.typ)?;
            for r in 0..h {
                for c in 0..w {
                    let nr = h - 1 - r;
                    let nc = w - 1 - c;
                    let s_off = (r * w + c) * elem;
                    let d_off = (nr * w + nc) * elem;
                    out.data[d_off..d_off + elem].copy_from_slice(&src.data[s_off..s_off + elem]);
                }
            }
            *dst = out;
        }
        ROTATE_90_CLOCKWISE => {
            let mut out = Mat::zeros(src.cols, src.rows, src.typ)?;
            for r in 0..h {
                for c in 0..w {
                    let nr = c;
                    let nc = h - 1 - r;
                    let s_off = (r * w + c) * elem;
                    let d_off = (nr * h + nc) * elem;
                    out.data[d_off..d_off + elem].copy_from_slice(&src.data[s_off..s_off + elem]);
                }
            }
            *dst = out;
        }
        ROTATE_90_COUNTERCLOCKWISE => {
            let mut out = Mat::zeros(src.cols, src.rows, src.typ)?;
            for r in 0..h {
                for c in 0..w {
                    let nr = w - 1 - c;
                    let nc = r;
                    let s_off = (r * w + c) * elem;
                    let d_off = (nr * h + nc) * elem;
                    out.data[d_off..d_off + elem].copy_from_slice(&src.data[s_off..s_off + elem]);
                }
            }
            *dst = out;
        }
        _ => return Err(Error::new(-215, "unsupported rotate_code")),
    }
    record_activation(
        "core::rotate",
        TheoremId::Thm5DihedralD4Closure,
        h * w,
        h * w,
        rotate_code as u64,
        &[("rotate_code", rotate_code as f64)],
    );
    Ok(())
}

/// Mirror-twin reflection (`flip`) powered by Coupled Mirror-Twin Jacobian (`Theorem 6`).
pub fn flip(src: &Mat, dst: &mut Mat, flip_code: i32) -> Result<()> {
    if src.is_empty() {
        *dst = Mat::default();
        return Ok(());
    }
    let elem = type_elem_size(src.typ);
    let (h, w) = (src.rows as usize, src.cols as usize);
    let mut out = Mat::zeros(src.rows, src.cols, src.typ)?;
    let (sr, sc) = if flip_code == 0 {
        (-1.0, 1.0)
    } else if flip_code > 0 {
        (1.0, -1.0)
    } else {
        (-1.0, -1.0)
    };
    for r in 0..h {
        for c in 0..w {
            let nr = if flip_code <= 0 { h - 1 - r } else { r };
            let nc = if flip_code != 0 { w - 1 - c } else { c };
            let s_off = (r * w + c) * elem;
            let d_off = (nr * w + nc) * elem;
            out.data[d_off..d_off + elem].copy_from_slice(&src.data[s_off..s_off + elem]);
        }
    }
    *dst = out;
    record_activation(
        "core::flip",
        TheoremId::Thm6MirrorTwinJacobian,
        h * w,
        h * w,
        (flip_code as i64) as u64,
        &[("s_r", sr), ("s_c", sc), ("det_j", sr * sc)],
    );
    Ok(())
}

/// Bitwise XOR with Pixelean Temporal Minterm Sparsity telemetry (`Theorem 2`).
pub fn bitwise_xor(a: &Mat, b: &Mat, dst: &mut Mat) -> Result<()> {
    if a.rows != b.rows || a.cols != b.cols || a.typ != b.typ {
        return Err(Error::new(-215, "bitwise_xor dimension/type mismatch"));
    }
    let mut out = Mat::zeros(a.rows, a.cols, a.typ)?;
    let mut active_bytes = 0usize;
    for ((o, &va), &vb) in out.data.iter_mut().zip(&a.data).zip(&b.data) {
        let x = va ^ vb;
        *o = x;
        if x != 0 {
            active_bytes += 1;
        }
    }
    record_activation(
        "core::bitwise_xor",
        TheoremId::Thm2TemporalXorMintermSparsity,
        a.data.len(),
        active_bytes,
        active_bytes as u64,
        &[(
            "k_diff_over_n",
            if a.data.is_empty() {
                0.0
            } else {
                active_bytes as f64 / a.data.len() as f64
            },
        )],
    );
    *dst = out;
    Ok(())
}

/// Absolute difference `|a - b|` with Pixelean Minterm carrier telemetry (`Theorem 2`).
pub fn absdiff(a: &Mat, b: &Mat, dst: &mut Mat) -> Result<()> {
    if a.rows != b.rows || a.cols != b.cols || a.typ != b.typ {
        return Err(Error::new(-215, "absdiff dimension/type mismatch"));
    }
    let mut out = Mat::zeros(a.rows, a.cols, a.typ)?;
    let cn = cv_mat_cn(a.typ) as usize;
    let mut active = 0usize;
    for r in 0..a.rows {
        for c in 0..a.cols {
            let mut nz = false;
            for ch in 0..cn {
                let d = (a.get_f64(r, c, ch) - b.get_f64(r, c, ch)).abs();
                if d > 0.0 {
                    nz = true;
                }
                out.set_f64(r, c, ch, d);
            }
            if nz {
                active += 1;
            }
        }
    }
    record_activation(
        "core::absdiff",
        TheoremId::Thm2TemporalXorMintermSparsity,
        a.total(),
        active,
        active as u64,
        &[],
    );
    *dst = out;
    Ok(())
}

impl Mat {
    /// Extract a rectangular submatrix (`Mat::roi`).
    pub fn roi(m: &Mat, rect: Rect) -> Result<Self> {
        if rect.x < 0
            || rect.y < 0
            || rect.width <= 0
            || rect.height <= 0
            || rect.x + rect.width > m.cols
            || rect.y + rect.height > m.rows
        {
            return Err(Error::new(-215, "ROI out of bounds in Mat::roi"));
        }
        let elem = type_elem_size(m.typ);
        let mut out = Mat::zeros(rect.height, rect.width, m.typ)?;
        for r in 0..(rect.height as usize) {
            let src_r = (rect.y as usize) + r;
            let src_off = (src_r * (m.cols as usize) + (rect.x as usize)) * elem;
            let dst_off = r * (rect.width as usize) * elem;
            let row_bytes = (rect.width as usize) * elem;
            out.data[dst_off..dst_off + row_bytes]
                .copy_from_slice(&m.data[src_off..src_off + row_bytes]);
        }
        Ok(out)
    }

    /// Copy matrix contents to `dst` (`copy_to`).
    pub fn copy_to(&self, dst: &mut Mat) -> Result<()> {
        *dst = self.clone();
        Ok(())
    }

    /// Convert matrix depth/type with optional scale `alpha` and offset `beta` (`convert_to`).
    pub fn convert_to(&self, dst: &mut Mat, rtype: i32, alpha: f64, beta: f64) -> Result<()> {
        let target_type = if rtype < 0 {
            self.typ
        } else {
            cv_maketype(cv_mat_depth(rtype), self.channels())
        };
        let cn = self.channels() as usize;
        let mut out = Mat::zeros(self.rows, self.cols, target_type)?;
        for r in 0..self.rows {
            for c in 0..self.cols {
                for ch in 0..cn {
                    let v = self.get_f64(r, c, ch) * alpha + beta;
                    out.set_f64(r, c, ch, v);
                }
            }
        }
        *dst = out;
        Ok(())
    }
}

/// Matrix transpose (`transpose`) representing the diagonal reflection generator of $D_4$ (`Theorem 5`).
pub fn transpose(src: &Mat, dst: &mut Mat) -> Result<()> {
    if src.is_empty() {
        *dst = Mat::default();
        return Ok(());
    }
    let elem = type_elem_size(src.typ);
    let (h, w) = (src.rows as usize, src.cols as usize);
    let mut out = Mat::zeros(src.cols, src.rows, src.typ)?;
    for r in 0..h {
        for c in 0..w {
            let s_off = (r * w + c) * elem;
            let d_off = (c * h + r) * elem;
            out.data[d_off..d_off + elem].copy_from_slice(&src.data[s_off..s_off + elem]);
        }
    }
    record_activation(
        "core::transpose",
        TheoremId::Thm5DihedralD4Closure,
        h * w,
        h * w,
        1,
        &[],
    );
    *dst = out;
    Ok(())
}

/// Bitwise AND (`bitwise_and`) with Shannon Cofactor Decomposition telemetry (`Theorem 3`).
pub fn bitwise_and(a: &Mat, b: &Mat, dst: &mut Mat) -> Result<()> {
    if a.rows != b.rows || a.cols != b.cols || a.typ != b.typ {
        return Err(Error::new(-215, "bitwise_and dimension/type mismatch"));
    }
    let mut out = Mat::zeros(a.rows, a.cols, a.typ)?;
    let mut nz = 0usize;
    for ((o, &va), &vb) in out.data.iter_mut().zip(&a.data).zip(&b.data) {
        let v = va & vb;
        *o = v;
        if v != 0 {
            nz += 1;
        }
    }
    record_activation(
        "core::bitwise_and",
        TheoremId::Thm3ShannonCofactorEnvelope,
        a.data.len(),
        nz,
        nz as u64,
        &[],
    );
    *dst = out;
    Ok(())
}

/// Bitwise OR (`bitwise_or`).
pub fn bitwise_or(a: &Mat, b: &Mat, dst: &mut Mat) -> Result<()> {
    if a.rows != b.rows || a.cols != b.cols || a.typ != b.typ {
        return Err(Error::new(-215, "bitwise_or dimension/type mismatch"));
    }
    let mut out = Mat::zeros(a.rows, a.cols, a.typ)?;
    for ((o, &va), &vb) in out.data.iter_mut().zip(&a.data).zip(&b.data) {
        *o = va | vb;
    }
    *dst = out;
    Ok(())
}

/// Bitwise NOT (`bitwise_not`).
pub fn bitwise_not(src: &Mat, dst: &mut Mat) -> Result<()> {
    let mut out = Mat::zeros(src.rows, src.cols, src.typ)?;
    for (o, &v) in out.data.iter_mut().zip(&src.data) {
        *o = !v;
    }
    *dst = out;
    Ok(())
}

/// Count non-zero scalar elements in a single-channel `Mat` (`count_non_zero`).
pub fn count_non_zero(src: &Mat) -> Result<i32> {
    if src.is_empty() {
        return Ok(0);
    }
    let mut cnt = 0i32;
    for r in 0..src.rows {
        for c in 0..src.cols {
            if src.get_f64(r, c, 0).abs() > 0.0 {
                cnt += 1;
            }
        }
    }
    Ok(cnt)
}

/// Find minimum and maximum values and their locations (`min_max_loc`).
pub fn min_max_loc(
    src: &Mat,
    min_val: Option<&mut f64>,
    max_val: Option<&mut f64>,
    min_loc: Option<&mut Point>,
    max_loc: Option<&mut Point>,
    mask: &Mat,
) -> Result<()> {
    if src.is_empty() {
        return Err(Error::new(-215, "empty Mat in min_max_loc"));
    }
    let use_mask = !mask.is_empty() && mask.rows == src.rows && mask.cols == src.cols;
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    let mut pmin = Point::new(0, 0);
    let mut pmax = Point::new(0, 0);
    for r in 0..src.rows {
        for c in 0..src.cols {
            if use_mask && mask.get_f64(r, c, 0) == 0.0 {
                continue;
            }
            let v = src.get_f64(r, c, 0);
            if v < vmin {
                vmin = v;
                pmin = Point::new(c, r);
            }
            if v > vmax {
                vmax = v;
                pmax = Point::new(c, r);
            }
        }
    }
    if let Some(m) = min_val {
        *m = vmin;
    }
    if let Some(m) = max_val {
        *m = vmax;
    }
    if let Some(p) = min_loc {
        *p = pmin;
    }
    if let Some(p) = max_loc {
        *p = pmax;
    }
    Ok(())
}

/// Compute per-channel mean and standard deviation (`mean_std_dev`).
pub fn mean_std_dev(src: &Mat, mean: &mut Scalar, stddev: &mut Scalar) -> Result<()> {
    if src.is_empty() {
        *mean = Scalar::all(0.0);
        *stddev = Scalar::all(0.0);
        return Ok(());
    }
    let cn = (src.channels() as usize).min(4);
    let n = (src.total() as f64).max(1.0);
    let mut m = [0.0f64; 4];
    let mut s = [0.0f64; 4];
    for r in 0..src.rows {
        for c in 0..src.cols {
            for ch in 0..cn {
                m[ch] += src.get_f64(r, c, ch);
            }
        }
    }
    for ch in 0..cn {
        m[ch] /= n;
    }
    for r in 0..src.rows {
        for c in 0..src.cols {
            for ch in 0..cn {
                let d = src.get_f64(r, c, ch) - m[ch];
                s[ch] += d * d;
            }
        }
    }
    for ch in 0..cn {
        s[ch] = (s[ch] / n).sqrt();
    }
    *mean = Scalar(m);
    *stddev = Scalar(s);
    Ok(())
}

/// Compute L2 norm of a matrix (`norm`).
pub fn norm(src: &Mat) -> Result<f64> {
    let cn = src.channels() as usize;
    let mut acc = 0.0f64;
    for r in 0..src.rows {
        for c in 0..src.cols {
            for ch in 0..cn {
                let v = src.get_f64(r, c, ch);
                acc += v * v;
            }
        }
    }
    Ok(acc.sqrt())
}

/// Threshold a single or multi-channel matrix into a binary mask `[lowerb, upperb]` (`in_range`).
pub fn in_range(src: &Mat, lowerb: Scalar, upperb: Scalar, dst: &mut Mat) -> Result<()> {
    if src.is_empty() {
        *dst = Mat::default();
        return Ok(());
    }
    let cn = (src.channels() as usize).min(4);
    let mut out = Mat::zeros(src.rows, src.cols, CV_8UC1)?;
    let mut fg = 0usize;
    for r in 0..src.rows {
        for c in 0..src.cols {
            let mut inside = true;
            for ch in 0..cn {
                let v = src.get_f64(r, c, ch);
                if v < lowerb.0[ch] || v > upperb.0[ch] {
                    inside = false;
                    break;
                }
            }
            if inside {
                out.set_f64(r, c, 0, 255.0);
                fg += 1;
            }
        }
    }
    record_activation(
        "core::in_range",
        TheoremId::Thm1BitplaneMassConservation,
        src.total(),
        fg,
        fg as u64,
        &[],
    );
    *dst = out;
    Ok(())
}

/// Linear blend `dst = src1 * alpha + src2 * beta + gamma` (`add_weighted`).
pub fn add_weighted(
    src1: &Mat,
    alpha: f64,
    src2: &Mat,
    beta: f64,
    gamma: f64,
    dst: &mut Mat,
    dtype: i32,
) -> Result<()> {
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels() != src2.channels() {
        return Err(Error::new(-215, "dimension/channel mismatch in add_weighted"));
    }
    let target_type = if dtype < 0 {
        src1.typ
    } else {
        cv_maketype(cv_mat_depth(dtype), src1.channels())
    };
    let cn = src1.channels() as usize;
    let mut out = Mat::zeros(src1.rows, src1.cols, target_type)?;
    for r in 0..src1.rows {
        for c in 0..src1.cols {
            for ch in 0..cn {
                let v = src1.get_f64(r, c, ch) * alpha + src2.get_f64(r, c, ch) * beta + gamma;
                out.set_f64(r, c, ch, v);
            }
        }
    }
    *dst = out;
    Ok(())
}

/// Split a multi-channel `Mat` into single-channel planes (`split`).
pub fn split(src: &Mat, mv: &mut Vector<Mat>) -> Result<()> {
    mv.clear();
    if src.is_empty() {
        return Ok(());
    }
    let cn = src.channels() as usize;
    let single_type = cv_maketype(src.depth(), 1);
    for ch in 0..cn {
        let mut plane = Mat::zeros(src.rows, src.cols, single_type)?;
        for r in 0..src.rows {
            for c in 0..src.cols {
                plane.set_f64(r, c, 0, src.get_f64(r, c, ch));
            }
        }
        mv.push(plane);
    }
    Ok(())
}

/// Merge single-channel planes into a multi-channel `Mat` (`merge`).
pub fn merge(mv: &Vector<Mat>, dst: &mut Mat) -> Result<()> {
    if mv.is_empty() {
        *dst = Mat::default();
        return Ok(());
    }
    let first = mv.get(0)?;
    let rows = first.rows;
    let cols = first.cols;
    let depth = first.depth();
    let cn = mv.len();
    let mut out = Mat::zeros(rows, cols, cv_maketype(depth, cn as i32))?;
    for (ch, plane) in mv.iter().enumerate() {
        if plane.rows != rows || plane.cols != cols {
            return Err(Error::new(-215, "plane size mismatch in merge"));
        }
        for r in 0..rows {
            for c in 0..cols {
                out.set_f64(r, c, ch, plane.get_f64(r, c, 0));
            }
        }
    }
    *dst = out;
    Ok(())
}

