use crate::any_scalar::BackendScalar;
use crate::tensor_element::TensorElement;
use anyhow::{anyhow, ensure, Result};
use num_complex::{Complex32, Complex64};
use std::ops::Mul;
use std::sync::Arc;
use tenferro::{DType, TensorScalar};

/// Dispatch a compact-payload body once over every scalar type [`Storage`] holds.
///
/// The four erased arms live here and nowhere else; `$ty` names the payload
/// scalar type inside `$body`, so dtype-agnostic storage methods are written
/// exactly once. Do not name `$ty` after an enclosing generic parameter: the
/// expansion introduces a local type alias with that name.
macro_rules! storage_payload_dispatch {
    ($repr:expr, $ty:ident => |$payload:ident| $body:expr) => {
        match $repr {
            StorageRepr::F32($payload) => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = f32;
                $body
            }
            StorageRepr::F64($payload) => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = f64;
                $body
            }
            StorageRepr::C32($payload) => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = Complex32;
                $body
            }
            StorageRepr::C64($payload) => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = Complex64;
                $body
            }
        }
    };
}

/// Dispatch a two-operand body over payloads that share one scalar type.
///
/// Elementwise storage algebra is only defined for matching scalar kinds, so
/// the four same-dtype arms are written once and every other combination falls
/// through to `else <expr>`.
macro_rules! storage_payload_same_dtype_dispatch {
    ($left:expr, $right:expr, $ty:ident => |$a:ident, $b:ident| $body:expr, else $fallback:expr) => {
        match ($left, $right) {
            (StorageRepr::F32($a), StorageRepr::F32($b)) => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = f32;
                $body
            }
            (StorageRepr::F64($a), StorageRepr::F64($b)) => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = f64;
                $body
            }
            (StorageRepr::C32($a), StorageRepr::C32($b)) => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = Complex32;
                $body
            }
            (StorageRepr::C64($a), StorageRepr::C64($b)) => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = Complex64;
                $body
            }
            _ => $fallback,
        }
    };
}

/// Dispatch a two-payload body over every scalar-type pair.
///
/// The nesting is intentional: each payload keeps its own concrete scalar type,
/// so cross-dtype algebra (mixed promotion) is written once instead of in
/// sixteen hand-written arms.
macro_rules! storage_payload_pair_dispatch {
    ($left:expr, $right:expr, $left_ty:ident, $right_ty:ident => |$a:ident, $b:ident| $body:expr) => {
        storage_payload_dispatch!($left, $left_ty => |$a| {
            storage_payload_dispatch!($right, $right_ty => |$b| $body)
        })
    };
}

/// Dispatch a body over a computed result dtype, binding `$ty` to it.
///
/// Used where promotion decides the result scalar type before the payload can
/// be built, so the arithmetic is written once and narrowed or widened by
/// [`StorageScalar::from_widened_parts`].
macro_rules! storage_dtype_dispatch {
    ($dtype:expr, $ty:ident => $body:expr, else $fallback:expr) => {
        match $dtype {
            DType::F32 => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = f32;
                $body
            }
            DType::F64 => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = f64;
                $body
            }
            DType::C32 => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = Complex32;
                $body
            }
            DType::C64 => {
                #[allow(unused)] // the body names the payload type only when it needs it
                type $ty = Complex64;
                $body
            }
            _ => $fallback,
        }
    };
}

pub(crate) use storage_dtype_dispatch;

/// Named result of a non-finite scan over stored scalar values.
///
/// The two flags are independent: a payload can contain both NaN and infinity,
/// and `Default` is the all-finite result. Use it instead of a positional
/// `(has_nan, has_infinity)` pair so call sites name the flag they read.
///
/// This reports what the scan found; it does not classify, replace, or mask the
/// values. See [`Storage::payload_nonfinite_flags`] for the compact-payload
/// scan and `tensor4all_core` for the native-tensor scan over a different
/// representation.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::NonfiniteFlags;
///
/// let finite = NonfiniteFlags::default();
/// assert!(!finite.has_nan && !finite.has_infinity);
///
/// let flags = NonfiniteFlags {
///     has_nan: true,
///     has_infinity: false,
/// };
/// assert!(flags.has_nan);
/// assert_ne!(flags, finite);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NonfiniteFlags {
    /// At least one scanned real or imaginary component is NaN.
    pub has_nan: bool,
    /// At least one scanned real or imaginary component is infinite.
    pub has_infinity: bool,
}

/// Trait for scalar types that can be stored in [`Storage`].
///
/// This enables generic constructors such as [`Storage::from_dense_col_major`]
/// and [`Storage::from_diag_col_major`], and it is the single seam through
/// which dtype-erased storage code reaches a concrete payload. Implemented for
/// `f64`, `Complex64`, `f32`, and `Complex32`.
///
/// Compact storage never widens a payload: `f32` stays `f32` and `Complex32`
/// stays `Complex32` in memory and on every accessor. The associated items
/// describe the real/complex structure so that dtype-agnostic scans can be
/// written once; explicit width promotion happens only in the storage algebra
/// that documents it ([`Storage::axpby`], [`Storage::scale`]).
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{Storage, StorageScalar};
///
/// // Using the generic constructor -- scalar type is inferred from data
/// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0], &[3]).unwrap();
/// assert!(s.is_f64());
///
/// use num_complex::Complex64;
/// let c = Storage::from_dense_col_major(
///     vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
///     &[2],
/// ).unwrap();
/// assert!(c.is_c64());
///
/// // Narrow payloads keep their dtype instead of widening.
/// let narrow = Storage::from_dense_col_major(vec![1.0_f32, 2.0], &[2]).unwrap();
/// assert!(narrow.is_f32());
/// assert_eq!(f32::KIND, "f32");
/// assert_eq!(Complex64::KIND, "Complex64");
/// ```
pub trait StorageScalar: TensorElement + Default + PartialEq + num_traits::Num {
    /// Display name of this scalar type in diagnostics (`"f32"`, `"Complex64"`, ...).
    const KIND: &'static str;
    /// Whether values carry an imaginary component.
    const IS_COMPLEX: bool;
    /// The complex counterpart of this scalar type (`f32` -> `Complex32`).
    type Complex: StorageScalar;

    /// Widen this value to `(real, imaginary)` parts for dtype-agnostic scans.
    fn widened_parts(self) -> (f64, f64);

    /// Narrow widened parts back to this scalar type.
    fn from_widened_parts(re: f64, im: f64) -> Self;

    /// Real part as this scalar's real counterpart.
    fn real_part(self) -> Self::Real;

    /// Imaginary part as this scalar's real counterpart (`0` for real scalars).
    fn imag_part(self) -> Self::Real;

    /// Embed this value, including its imaginary part, in [`Self::Complex`].
    fn into_complex(self) -> Self::Complex;

    /// The complex conjugate (`self` for real scalars).
    fn conjugate(self) -> Self;

    /// Wrap a compact payload of this scalar type as a [`Storage`].
    fn wrap_payload(payload: StructuredStorage<Self>) -> Storage;

    /// Copy this value into a dynamic [`BackendScalar`].
    fn to_backend_scalar(self) -> BackendScalar;

    /// Copy the compact payload of this scalar type, rejecting other dtypes.
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] when `storage` holds a
    /// different scalar type.
    fn payload_vec(storage: &Storage) -> StorageResult<Vec<Self>>;

    /// Borrow the compact payload of this scalar type when it is contiguous.
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] when `storage` holds a
    /// different scalar type.
    fn payload_view(storage: &Storage) -> StorageResult<Option<&[Self]>>;

    /// Materialize the logical dense values of this scalar type.
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] when `storage` holds a
    /// different scalar type, and
    /// [`StorageError::InvalidStructuredStorage`] when `logical_dims` does not
    /// match the stored logical dimensions or the logical element count
    /// overflows `usize`.
    fn dense_values(storage: &Storage, logical_dims: &[usize]) -> StorageResult<Vec<Self>>;

    /// Build a dense [`Storage`] from column-major data.
    /// # Errors
    ///
    /// Returns an error when the shape product would overflow `usize` or
    /// `data.len()` does not match the expected shape.
    ///
    fn build_dense_storage(data: Vec<Self>, logical_dims: &[usize]) -> Result<Storage> {
        let label = format!("dense {} payload", Self::KIND);
        Storage::validate_dense_len(&data, logical_dims, &label)?;
        Ok(Self::wrap_payload(StructuredStorage::from_dense_col_major(
            data,
            logical_dims,
        )?))
    }
    /// Build a diagonal [`Storage`] from diagonal payload data.
    /// # Errors
    ///
    /// Returns an error when the diagonal payload length does not match the
    /// length required for `logical_rank` or a stride product would overflow
    /// `usize`.
    ///
    fn build_diag_storage(diag_data: Vec<Self>, logical_rank: usize) -> Result<Storage> {
        Ok(Self::wrap_payload(StructuredStorage::from_diag_col_major(
            diag_data,
            logical_rank,
        )?))
    }
    /// Build a structured [`Storage`] from explicit payload metadata.
    /// # Errors
    ///
    /// Returns an error when `axis_classes` is not in canonical
    /// first-appearance order, when the `payload_dims` or `strides` ranks do
    /// not match `axis_classes`, when a stride is negative, when a size
    /// product would overflow `usize`, or when `data.len()` does not match the
    /// required payload length.
    ///
    fn build_structured_storage(
        data: Vec<Self>,
        payload_dims: Vec<usize>,
        strides: Vec<isize>,
        axis_classes: Vec<usize>,
    ) -> Result<Storage> {
        Ok(Self::wrap_payload(StructuredStorage::new(
            data,
            payload_dims,
            strides,
            axis_classes,
        )?))
    }
}

/// Implement [`StorageScalar`] for one concrete scalar type.
///
/// Only the per-dtype algebra (real/imaginary extraction, complex embedding,
/// conjugation, widening) is spelled per type; payload construction and
/// dispatch are shared.
macro_rules! impl_storage_scalar {
    ($ty:ty, $variant:ident, $complex:ty, $kind:literal, $is_complex:literal,
     copy_operation: $copy_operation:literal, view_operation: $view_operation:literal,
     dense_operation: $dense_operation:literal,
     widened: $widened:expr, from_widened: $from_widened:expr,
     real: $real:expr, imag: $imag:expr, into_complex: $into_complex:expr,
     conj: $conj:expr) => {
        impl StorageScalar for $ty {
            const KIND: &'static str = $kind;
            const IS_COMPLEX: bool = $is_complex;
            type Complex = $complex;

            fn widened_parts(self) -> (f64, f64) {
                ($widened)(self)
            }

            fn from_widened_parts(re: f64, im: f64) -> Self {
                ($from_widened)(re, im)
            }

            fn real_part(self) -> Self::Real {
                ($real)(self)
            }

            fn imag_part(self) -> Self::Real {
                ($imag)(self)
            }

            fn into_complex(self) -> Self::Complex {
                ($into_complex)(self)
            }

            fn conjugate(self) -> Self {
                ($conj)(self)
            }

            fn wrap_payload(payload: StructuredStorage<Self>) -> Storage {
                Storage(StorageRepr::$variant(payload))
            }

            fn to_backend_scalar(self) -> BackendScalar {
                BackendScalar::from_value(self)
            }

            fn payload_vec(storage: &Storage) -> StorageResult<Vec<Self>> {
                match &storage.0 {
                    StorageRepr::$variant(payload) => Ok(payload.payload_col_major_vec()),
                    _ => Err(StorageError::ScalarKindMismatch {
                        expected: Self::KIND,
                        actual: storage.scalar_kind(),
                        operation: $copy_operation,
                    }),
                }
            }

            fn payload_view(storage: &Storage) -> StorageResult<Option<&[Self]>> {
                match &storage.0 {
                    StorageRepr::$variant(payload) => {
                        Ok(payload.payload_col_major_view_if_contiguous())
                    }
                    _ => Err(StorageError::ScalarKindMismatch {
                        expected: Self::KIND,
                        actual: storage.scalar_kind(),
                        operation: $view_operation,
                    }),
                }
            }

            fn dense_values(storage: &Storage, logical_dims: &[usize]) -> StorageResult<Vec<Self>> {
                match &storage.0 {
                    StorageRepr::$variant(payload) => checked_dense_payload(payload, logical_dims),
                    _ => Err(StorageError::ScalarKindMismatch {
                        expected: Self::KIND,
                        actual: storage.scalar_kind(),
                        operation: $dense_operation,
                    }),
                }
            }
        }
    };
}

impl_storage_scalar!(
    f32,
    F32,
    Complex32,
    "f32",
    false,
    copy_operation: "copying f32 payload",
    view_operation: "borrowing f32 payload",
    dense_operation: "materializing dense f32 values",
    widened: |value: f32| (f64::from(value), 0.0),
    from_widened: |re: f64, _im: f64| re as f32,
    real: |value: f32| value,
    imag: |_value: f32| 0.0,
    into_complex: |value: f32| Complex32::new(value, 0.0),
    conj: |value: f32| value
);

impl_storage_scalar!(
    f64,
    F64,
    Complex64,
    "f64",
    false,
    copy_operation: "copying f64 payload",
    view_operation: "borrowing f64 payload",
    dense_operation: "materializing dense f64 values",
    widened: |value: f64| (value, 0.0),
    from_widened: |re: f64, _im: f64| re,
    real: |value: f64| value,
    imag: |_value: f64| 0.0,
    into_complex: |value: f64| Complex64::new(value, 0.0),
    conj: |value: f64| value
);

impl_storage_scalar!(
    Complex32,
    C32,
    Complex32,
    "Complex32",
    true,
    copy_operation: "copying c32 payload",
    view_operation: "borrowing c32 payload",
    dense_operation: "materializing dense c32 values",
    widened: |value: Complex32| (f64::from(value.re), f64::from(value.im)),
    from_widened: |re: f64, im: f64| Complex32::new(re as f32, im as f32),
    real: |value: Complex32| value.re,
    imag: |value: Complex32| value.im,
    into_complex: |value: Complex32| value,
    conj: |value: Complex32| value.conj()
);

impl_storage_scalar!(
    Complex64,
    C64,
    Complex64,
    "Complex64",
    true,
    copy_operation: "copying c64 payload",
    view_operation: "borrowing c64 payload",
    dense_operation: "materializing dense c64 values",
    widened: |value: Complex64| (value.re, value.im),
    from_widened: |re: f64, im: f64| Complex64::new(re, im),
    real: |value: Complex64| value.re,
    imag: |value: Complex64| value.im,
    into_complex: |value: Complex64| value,
    conj: |value: Complex64| value.conj()
);

pub(crate) fn col_major_strides(dims: &[usize]) -> Result<Vec<isize>> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1isize;
    for &dim in dims {
        strides.push(stride);
        let dim = isize::try_from(dim)
            .map_err(|_| anyhow!("column-major stride overflow for dims {dims:?}"))?;
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| anyhow!("column-major stride overflow for dims {dims:?}"))?;
    }
    Ok(strides)
}

fn validate_canonical_axis_classes(axis_classes: &[usize]) -> Result<()> {
    let mut next_class = 0usize;
    for &class_id in axis_classes {
        ensure!(
            class_id <= next_class,
            "axis_classes must be canonical first-appearance labels, got {axis_classes:?}"
        );
        if class_id == next_class {
            next_class = next_class
                .checked_add(1)
                .ok_or_else(|| anyhow!("axis class index overflow"))?;
        }
    }
    Ok(())
}

fn required_storage_len(dims: &[usize], strides: &[isize]) -> Result<usize> {
    ensure!(
        dims.len() == strides.len(),
        "payload dims {:?} and strides {:?} must have the same rank",
        dims,
        strides
    );

    // Validate the compact payload product even when a zero stride would make
    // the referenced backing span look small. Iteration must never begin with
    // a wrapped payload length.
    let payload_len = dims.iter().try_fold(1usize, |length, &dim| {
        length
            .checked_mul(dim)
            .ok_or_else(|| anyhow!("payload length overflow for dims {dims:?}"))
    })?;

    let mut max_offset = 0usize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        ensure!(
            stride >= 0,
            "negative strides are not supported in StructuredStorage: {strides:?}"
        );
        let stride = usize::try_from(stride)
            .map_err(|_| anyhow!("payload stride overflow for dims {dims:?}"))?;
        if dim > 1 {
            let span = (dim - 1)
                .checked_mul(stride)
                .ok_or_else(|| anyhow!("payload stride overflow for dims {dims:?}"))?;
            max_offset = max_offset
                .checked_add(span)
                .ok_or_else(|| anyhow!("payload stride overflow for dims {dims:?}"))?;
        }
    }
    if payload_len == 0 {
        return Ok(0);
    }
    max_offset
        .checked_add(1)
        .ok_or_else(|| anyhow!("payload stride overflow for dims {dims:?}"))
}

fn logical_dims_from_axis_classes(payload_dims: &[usize], axis_classes: &[usize]) -> Vec<usize> {
    axis_classes
        .iter()
        .map(|&class_id| payload_dims[class_id])
        .collect()
}

fn col_major_multi_index(mut linear: usize, dims: &[usize]) -> Vec<usize> {
    let mut index = Vec::with_capacity(dims.len());
    for &dim in dims {
        if dim == 0 {
            index.push(0);
        } else {
            index.push(linear % dim);
            linear /= dim;
        }
    }
    index
}

fn offset_from_strides(index: &[usize], strides: &[isize]) -> Option<usize> {
    index
        .iter()
        .zip(strides.iter())
        .try_fold(0usize, |offset, (&value, &stride)| {
            let stride = usize::try_from(stride).ok()?;
            let term = value.checked_mul(stride)?;
            offset.checked_add(term)
        })
}

/// Structured tensor snapshot storage.
///
/// `data` and `strides` describe the payload tensor, while `axis_classes`
/// describes how logical axes map onto payload axes. Logical flat-buffer
/// semantics are column-major. A strided payload may contain unused backing
/// gaps; reductions and nonfinite scans visit only entries addressed by the
/// payload dimensions and strides.
///
/// A **dense** tensor has `axis_classes = [0, 1, ..., rank-1]` (each logical
/// axis maps to a distinct payload axis). A **diagonal** tensor has
/// `axis_classes = [0, 0, ..., 0]` (all logical axes share one payload axis),
/// storing only the diagonal entries.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::StructuredStorage;
///
/// // Dense 2x3 storage, column-major: [[1,3,5],[2,4,6]]
/// let dense = StructuredStorage::from_dense_col_major(
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3],
/// ).unwrap();
/// assert!(dense.is_dense());
/// assert!(!dense.is_diag());
/// assert_eq!(dense.logical_rank(), 2);
/// assert_eq!(dense.logical_dims(), vec![2, 3]);
///
/// // Diagonal 3x3 storage
/// let diag = StructuredStorage::from_diag_col_major(vec![1.0, 2.0, 3.0], 2).unwrap();
/// assert!(diag.is_diag());
/// assert_eq!(diag.logical_dims(), vec![3, 3]);
/// assert_eq!(diag.len(), 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredStorage<T> {
    data: Vec<T>,
    payload_dims: Vec<usize>,
    strides: Vec<isize>,
    axis_classes: Vec<usize>,
}

impl<T> StructuredStorage<T> {
    /// Creates a structured payload snapshot from explicit payload metadata.
    ///
    /// `payload_dims` and `strides` describe the compressed payload tensor,
    /// while `axis_classes` maps logical axes onto payload axes in canonical
    /// first-appearance order.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `axis_classes` is not in canonical first-appearance form
    /// - `payload_dims` rank does not match `axis_classes`
    /// - `strides` rank does not match `payload_dims`
    /// - `data` length does not match the required storage length
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// // Dense 2x3 with explicit column-major strides
    /// let s = StructuredStorage::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     vec![2, 3],     // payload_dims
    ///     vec![1, 2],     // column-major strides
    ///     vec![0, 1],     // axis_classes: each axis is independent
    /// ).unwrap();
    /// assert!(s.is_dense());
    /// assert_eq!(s.len(), 6);
    /// ```
    pub fn new(
        data: Vec<T>,
        payload_dims: Vec<usize>,
        strides: Vec<isize>,
        axis_classes: Vec<usize>,
    ) -> Result<Self> {
        let required_len =
            Storage::validate_structured_metadata(&payload_dims, &strides, &axis_classes)?;
        ensure!(
            data.len() == required_len,
            "payload storage len {} does not match required len {} for dims {:?} and strides {:?}",
            data.len(),
            required_len,
            payload_dims,
            strides
        );
        Ok(Self {
            data,
            payload_dims,
            strides,
            axis_classes,
        })
    }

    /// Creates a dense structured snapshot from column-major logical data.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Operation`] when a column-major stride overflows
    /// `usize` or when the payload length does not match `logical_dims`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    /// assert!(s.is_dense());
    /// assert_eq!(s.data(), &[10.0, 20.0, 30.0, 40.0]);
    /// ```
    pub fn from_dense_col_major(data: Vec<T>, logical_dims: &[usize]) -> StorageResult<Self> {
        let payload_dims = logical_dims.to_vec();
        let strides = col_major_strides(&payload_dims)?;
        let axis_classes = (0..logical_dims.len()).collect();
        Self::new(data, payload_dims, strides, axis_classes).map_err(StorageError::from)
    }

    /// Creates a diagonal structured snapshot from column-major diagonal data.
    ///
    /// The resulting tensor has `logical_rank` axes, each of size `diag_data.len()`.
    /// Only the diagonal entries are stored.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Operation`] when a stride product overflows
    /// `usize` or when the required payload length does not match
    /// `diag_data.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0, 3.0], 2).unwrap();
    /// assert!(d.is_diag());
    /// assert_eq!(d.logical_dims(), vec![3, 3]);
    /// assert_eq!(d.data(), &[1.0, 2.0, 3.0]);
    /// ```
    pub fn from_diag_col_major(diag_data: Vec<T>, logical_rank: usize) -> StorageResult<Self> {
        let payload_dims = if logical_rank == 0 {
            vec![]
        } else {
            vec![diag_data.len()]
        };
        let strides = col_major_strides(&payload_dims)?;
        let axis_classes = vec![0; logical_rank];
        Self::new(diag_data, payload_dims, strides, axis_classes).map_err(StorageError::from)
    }

    /// Returns the payload data buffer as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    /// assert_eq!(s.data(), &[1.0, 2.0]);
    /// ```
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns the payload tensor dimensions.
    ///
    /// For dense tensors, this equals the logical dimensions. For diagonal
    /// tensors, this is a single-element slice `[n]` where `n` is the diagonal
    /// length.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![0.0; 6], &[2, 3]).unwrap();
    /// assert_eq!(s.payload_dims(), &[2, 3]);
    ///
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 3).unwrap();
    /// assert_eq!(d.payload_dims(), &[2]);
    /// ```
    pub fn payload_dims(&self) -> &[usize] {
        &self.payload_dims
    }

    /// Returns the payload tensor strides.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// // Column-major 2x3: strides are [1, 2]
    /// let s = StructuredStorage::from_dense_col_major(vec![0.0; 6], &[2, 3]).unwrap();
    /// assert_eq!(s.strides(), &[1, 2]);
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns the canonical logical-to-payload axis classes.
    ///
    /// Each entry maps a logical axis to a payload axis index. Repeated values
    /// indicate axes that share the same payload dimension (e.g., diagonal).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let dense = StructuredStorage::from_dense_col_major(vec![0.0; 4], &[2, 2]).unwrap();
    /// assert_eq!(dense.axis_classes(), &[0, 1]);
    ///
    /// let diag = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2).unwrap();
    /// assert_eq!(diag.axis_classes(), &[0, 0]);
    /// ```
    pub fn axis_classes(&self) -> &[usize] {
        &self.axis_classes
    }

    /// Returns the logical dimensions derived from `payload_dims` and `axis_classes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0, 3.0], 3).unwrap();
    /// assert_eq!(d.logical_dims(), vec![3, 3, 3]);
    /// ```
    pub fn logical_dims(&self) -> Vec<usize> {
        logical_dims_from_axis_classes(&self.payload_dims, &self.axis_classes)
    }

    /// Returns the logical rank (number of logical axes).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![0.0; 6], &[2, 3]).unwrap();
    /// assert_eq!(s.logical_rank(), 2);
    /// ```
    pub fn logical_rank(&self) -> usize {
        self.axis_classes.len()
    }

    /// Returns `true` when the logical tensor is dense (each logical axis maps
    /// to a unique payload axis).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    /// assert!(s.is_dense());
    ///
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2).unwrap();
    /// assert!(!d.is_dense());
    /// ```
    pub fn is_dense(&self) -> bool {
        self.axis_classes
            .iter()
            .copied()
            .eq(0..self.axis_classes.len())
    }

    /// Returns `true` when the logical tensor is diagonal (rank >= 2 and all
    /// logical axes map to the same payload axis).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2).unwrap();
    /// assert!(d.is_diag());
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    /// assert!(!s.is_diag());
    /// ```
    pub fn is_diag(&self) -> bool {
        self.logical_rank() >= 2 && self.axis_classes.iter().all(|&class_id| class_id == 0)
    }

    /// Returns the payload buffer length.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let dense = StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// assert_eq!(dense.len(), 3);
    ///
    /// let diag = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2).unwrap();
    /// assert_eq!(diag.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the payload buffer is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let empty = StructuredStorage::from_dense_col_major(Vec::<f64>::new(), &[0]).unwrap();
    /// assert!(empty.is_empty());
    ///
    /// let non_empty = StructuredStorage::from_dense_col_major(vec![1.0], &[1]).unwrap();
    /// assert!(!non_empty.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a borrowed view when the logical tensor is dense and the
    /// payload is already stored contiguously in column-major order.
    ///
    /// Returns `None` for diagonal or non-contiguous payloads.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// assert_eq!(s.dense_col_major_view_if_contiguous(), Some(&[1.0, 2.0, 3.0][..]));
    ///
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2).unwrap();
    /// assert_eq!(d.dense_col_major_view_if_contiguous(), None);
    /// ```
    pub fn dense_col_major_view_if_contiguous(&self) -> Option<&[T]> {
        if self.is_dense()
            && matches!(col_major_strides(&self.payload_dims), Ok(strides) if strides == self.strides)
        {
            Some(&self.data)
        } else {
            None
        }
    }

    /// Returns a borrowed compact-payload view when the payload is already
    /// stored contiguously in column-major order.
    pub fn payload_col_major_view_if_contiguous(&self) -> Option<&[T]> {
        if matches!(col_major_strides(&self.payload_dims), Ok(strides) if strides == self.strides) {
            Some(&self.data)
        } else {
            None
        }
    }
}

impl<T: Clone> StructuredStorage<T> {
    /// Materializes the payload tensor as a contiguous column-major buffer.
    ///
    /// If the payload is already column-major, returns a clone.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// assert_eq!(s.payload_col_major_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn payload_col_major_vec(&self) -> Vec<T> {
        let payload_len = self
            .payload_dims
            .iter()
            .try_fold(1usize, |length, &dim| length.checked_mul(dim))
            // `StructuredStorage::new` validates this invariant before any
            // instance can be constructed; return an empty vector rather than
            // wrapping if an invalid value ever reaches this private state.
            .unwrap_or(0);
        if payload_len == 0 {
            return Vec::new();
        }
        if matches!(col_major_strides(&self.payload_dims), Ok(strides) if strides == self.strides) {
            return self.data.clone();
        }

        let mut output = Vec::with_capacity(payload_len);
        for linear in 0..payload_len {
            let index = col_major_multi_index(linear, &self.payload_dims);
            let Some(offset) = offset_from_strides(&index, &self.strides) else {
                return Vec::new();
            };
            output.push(self.data[offset].clone());
        }
        output
    }

    /// Returns a copy of the storage with logical axes permuted.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Operation`] when `perm` is not a permutation of
    /// the logical axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// // Diagonal 3x3x3 tensor; permute axes (identity for diag is always valid)
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0, 3.0], 3).unwrap();
    /// let p = d.permute_logical_axes(&[2, 0, 1]).unwrap();
    /// // Diagonal: all axes share the same dimension, so dims stay the same
    /// assert_eq!(p.logical_dims(), vec![3, 3, 3]);
    /// assert!(p.is_diag());
    /// ```
    pub fn permute_logical_axes(&self, perm: &[usize]) -> StorageResult<Self> {
        if perm.len() != self.axis_classes.len() {
            return Err(StorageError::from(anyhow::anyhow!(
                "logical permutation length {} must match logical rank {}",
                perm.len(),
                self.axis_classes.len()
            )));
        }
        let mut seen = vec![false; self.axis_classes.len()];
        let axis_classes = perm
            .iter()
            .map(|&index| {
                if index >= self.axis_classes.len() {
                    return Err(anyhow::anyhow!(
                        "logical permutation axis {index} is out of range for rank {}",
                        self.axis_classes.len()
                    ));
                }
                if seen[index] {
                    return Err(anyhow::anyhow!("logical permutation repeats axis {index}"));
                }
                seen[index] = true;
                Ok(self.axis_classes[index])
            })
            .collect::<Result<Vec<_>>>()
            .map_err(StorageError::from)?;
        Self::new(
            self.data.clone(),
            self.payload_dims.clone(),
            self.strides.clone(),
            axis_classes,
        )
        .map_err(StorageError::from)
    }
}

impl<T: Copy> StructuredStorage<T> {
    /// Wrap this payload as a compact [`Storage`] of the same scalar type.
    ///
    /// The payload keeps its dtype and metadata; nothing is widened or
    /// re-ordered. This is the construction seam used by dtype-generic storage
    /// algebra so each operation returns storage of its own scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let payload = StructuredStorage::from_dense_col_major(vec![1.0_f32, 2.0], &[2]).unwrap();
    /// let storage = payload.into_storage();
    /// assert!(storage.is_f32());
    /// assert_eq!(storage.len(), 2);
    /// ```
    pub fn into_storage(self) -> Storage
    where
        T: StorageScalar,
    {
        T::wrap_payload(self)
    }

    fn payload_scalar_at(&self, payload_coords: &[usize]) -> StorageResult<T> {
        if payload_coords.len() != self.payload_dims.len() {
            return Err(StorageError::InvalidStructuredStorage(format!(
                "payload coordinate rank {} does not match payload rank {}",
                payload_coords.len(),
                self.payload_dims.len()
            )));
        }
        let offset = payload_coords
            .iter()
            .zip(self.payload_dims.iter())
            .zip(self.strides.iter())
            .try_fold(0usize, |offset, ((&coordinate, &dim), &stride)| {
                if coordinate >= dim {
                    return Err(StorageError::InvalidStructuredStorage(format!(
                        "payload coordinate {coordinate} is out of bounds for dim {dim}"
                    )));
                }
                let stride = usize::try_from(stride).map_err(|_| {
                    StorageError::InvalidStructuredStorage("negative stride".into())
                })?;
                let term = coordinate.checked_mul(stride).ok_or_else(|| {
                    StorageError::InvalidStructuredStorage("payload offset overflow".into())
                })?;
                offset.checked_add(term).ok_or_else(|| {
                    StorageError::InvalidStructuredStorage("payload offset overflow".into())
                })
            })?;
        self.data.get(offset).copied().ok_or_else(|| {
            StorageError::InvalidStructuredStorage("payload offset outside storage".into())
        })
    }

    /// Maps payload elements while preserving payload metadata and axis classes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let doubled = s.map_copy(|x| x * 2.0);
    /// assert_eq!(doubled.data(), &[2.0, 4.0, 6.0]);
    /// ```
    pub fn map_copy<U>(&self, mut f: impl FnMut(T) -> U) -> StructuredStorage<U> {
        StructuredStorage {
            data: self.data.iter().copied().map(&mut f).collect(),
            payload_dims: self.payload_dims.clone(),
            strides: self.strides.clone(),
            axis_classes: self.axis_classes.clone(),
        }
    }
}

impl<T: Copy + Default> StructuredStorage<T> {
    /// Returns the checked product of the logical dimensions, rejecting
    /// overflow. The fallible dense exporters rely on this so that a logical
    /// product that overflows `usize` fails closed instead of being treated
    /// as an empty tensor.
    fn checked_logical_len(&self) -> StorageResult<usize> {
        let dims = self.logical_dims();
        dims.iter()
            .try_fold(1usize, |length, &dim| length.checked_mul(dim))
            .ok_or_else(|| {
                StorageError::InvalidStructuredStorage(format!(
                    "logical dims product overflow for {dims:?}"
                ))
            })
    }

    /// Materializes the logical tensor as a contiguous column-major dense buffer.
    ///
    /// Repeated entries in `axis_classes` encode equality constraints between
    /// logical axes. Logical indices that violate those constraints are
    /// structural zeros in the dense materialization.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidStructuredStorage`] if the logical
    /// dimension product overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// // Diagonal [1, 2] in 2x2 becomes [1, 0, 0, 2] column-major
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2).unwrap();
    /// assert_eq!(d.logical_dense_col_major_vec().unwrap(), vec![1.0, 0.0, 0.0, 2.0]);
    /// ```
    pub fn logical_dense_col_major_vec(&self) -> StorageResult<Vec<T>> {
        let logical_dims = self.logical_dims();
        let logical_len = self.checked_logical_len()?;
        if logical_len == 0 {
            return Ok(Vec::new());
        }
        if let Some(view) = self.dense_col_major_view_if_contiguous() {
            return Ok(view.to_vec());
        }
        if self.is_dense() {
            return Ok(self.payload_col_major_vec());
        }

        let mut payload_index = vec![0usize; self.payload_dims.len()];
        Ok((0..logical_len)
            .map(|linear| {
                let logical_index = col_major_multi_index(linear, &logical_dims);
                self.value_at_with_dims(&logical_index, &logical_dims, &mut payload_index)
                    .unwrap_or_default()
            })
            .collect())
    }

    fn for_each_payload_value(&self, mut f: impl FnMut(T)) {
        if let Some(view) = self.payload_col_major_view_if_contiguous() {
            for &value in view {
                f(value);
            }
            return;
        }
        let payload_len = self
            .payload_dims
            .iter()
            .try_fold(1usize, |length, &dim| length.checked_mul(dim))
            .unwrap_or(0);
        let mut payload_index = vec![0usize; self.payload_dims.len()];
        for _ in 0..payload_len {
            let Some(offset) = offset_from_strides(&payload_index, &self.strides) else {
                return;
            };
            f(self.data[offset]);
            let mut carry = true;
            for (coordinate, &dim) in payload_index.iter_mut().zip(self.payload_dims.iter()) {
                if !carry {
                    break;
                }
                *coordinate += 1;
                if *coordinate == dim {
                    *coordinate = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    fn value_at_with_dims(
        &self,
        logical_index: &[usize],
        logical_dims: &[usize],
        payload_index: &mut [usize],
    ) -> StorageResult<T> {
        if logical_index.len() != logical_dims.len() {
            return Err(StorageError::InvalidStructuredStorage(format!(
                "logical index rank {} does not match rank {}",
                logical_index.len(),
                logical_dims.len()
            )));
        }
        if payload_index.len() < self.payload_dims.len() {
            return Err(StorageError::InvalidStructuredStorage(format!(
                "payload scratch rank {} is smaller than {}",
                payload_index.len(),
                self.payload_dims.len()
            )));
        }
        payload_index[..self.payload_dims.len()].fill(usize::MAX);
        for ((&value, &dim), &class_id) in logical_index
            .iter()
            .zip(logical_dims.iter())
            .zip(self.axis_classes.iter())
        {
            if value >= dim {
                return Err(StorageError::InvalidStructuredStorage(format!(
                    "logical index {value} is out of bounds for dim {dim}"
                )));
            }
            if payload_index[class_id] == usize::MAX {
                payload_index[class_id] = value;
            } else if payload_index[class_id] != value {
                return Ok(T::default());
            }
        }
        let offset = payload_index[..self.payload_dims.len()]
            .iter()
            .zip(self.strides.iter())
            .try_fold(0usize, |offset, (&value, &stride)| {
                let stride = usize::try_from(stride).map_err(|_| {
                    StorageError::InvalidStructuredStorage("negative stride".into())
                })?;
                let term = value.checked_mul(stride).ok_or_else(|| {
                    StorageError::InvalidStructuredStorage("payload offset overflow".into())
                })?;
                offset.checked_add(term).ok_or_else(|| {
                    StorageError::InvalidStructuredStorage("payload offset overflow".into())
                })
            })?;
        self.data.get(offset).copied().ok_or_else(|| {
            StorageError::InvalidStructuredStorage("payload offset outside storage".into())
        })
    }
}

/// Storage backend for tensor data.
///
/// Public callers interact with this opaque wrapper through constructors and
/// high-level query/materialization methods.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::Storage;
///
/// // Dense 2x3 matrix stored column-major: [[1,2,3],[4,5,6]]
/// let data = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
/// let s = Storage::from_dense_col_major(data, &[2, 3]).unwrap();
/// assert!(s.is_f64());
/// assert!(!s.is_complex());
///
/// // Diagonal storage: 2x2 identity-like diagonal
/// let diag = Storage::new_diag(vec![1.0_f64, 2.0]).unwrap();
/// assert!(diag.is_f64());
/// ```
#[derive(Debug, Clone)]
pub struct Storage(pub(crate) StorageRepr);

/// Classifies the compact layout used by [`Storage`].
///
/// Use this to distinguish dense logical payloads from diagonal/copy payloads
/// and general structured payloads without exposing the internal storage enum.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{Storage, StorageKind};
///
/// let dense = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
/// assert_eq!(dense.storage_kind(), StorageKind::Dense);
///
/// let diag = Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap();
/// assert_eq!(diag.storage_kind(), StorageKind::Diagonal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKind {
    /// Logical dense payload layout.
    Dense,
    /// Diagonal or copy-tensor payload layout.
    Diagonal,
    /// General structured payload layout with repeated axis classes.
    Structured,
}

/// Errors returned by storage payload and elementwise operations.
///
/// Use this to distinguish scalar-kind mismatches, length mismatches, and
/// invalid structured-storage metadata from general backend failures.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{Storage, StorageError};
///
/// let storage = Storage::from_dense_col_major(vec![1.0_f64], &[1]).unwrap();
/// let err = storage.payload_c64_col_major_vec().unwrap_err();
/// assert!(matches!(err, StorageError::ScalarKindMismatch { .. }));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// The storage scalar kind did not match the requested operation.
    #[error("expected {expected} storage when {operation}, got {actual}")]
    ScalarKindMismatch {
        /// The scalar kind that the caller requested.
        expected: &'static str,
        /// The scalar kind actually stored.
        actual: &'static str,
        /// Human-readable operation description.
        operation: &'static str,
    },
    /// Two storages had different payload lengths for an elementwise operation.
    #[error("storage lengths must match for {operation}: {left} != {right}")]
    LengthMismatch {
        /// Name of the operation being performed.
        operation: &'static str,
        /// Left-hand payload length.
        left: usize,
        /// Right-hand payload length.
        right: usize,
    },
    /// Structured storage metadata was invalid after an operation.
    #[error("invalid structured storage: {0}")]
    InvalidStructuredStorage(String),
    /// The requested operation does not support the provided storage kinds.
    #[error("storage types are not supported for {operation}: {left} vs {right}")]
    OperationNotSupported {
        /// Name of the operation being performed.
        operation: &'static str,
        /// Left-hand storage kind.
        left: &'static str,
        /// Right-hand storage kind.
        right: &'static str,
    },
    /// A storage operation failed with an internal diagnostic.
    #[error("storage operation failed: {source}")]
    Operation {
        /// Original internal diagnostic, preserving its source chain.
        #[source]
        source: anyhow::Error,
    },
}

impl From<anyhow::Error> for StorageError {
    fn from(source: anyhow::Error) -> Self {
        Self::Operation { source }
    }
}

/// Result type returned by storage methods that can fail with [`StorageError`].
pub type StorageResult<T> = std::result::Result<T, StorageError>;

#[derive(Debug, Clone)]
pub(crate) enum StorageRepr {
    /// Storage with f32 elements.
    F32(StructuredStorage<f32>),
    /// Storage with f64 elements.
    F64(StructuredStorage<f64>),
    /// Storage with Complex32 elements.
    C32(StructuredStorage<Complex32>),
    /// Storage with Complex64 elements.
    C64(StructuredStorage<Complex64>),
}

impl StorageRepr {
    fn dtype(&self) -> DType {
        storage_payload_dispatch!(self, Payload => |_payload| <Payload as TensorScalar>::dtype())
    }

    fn scalar_kind(&self) -> &'static str {
        storage_payload_dispatch!(self, Payload => |_payload| <Payload as StorageScalar>::KIND)
    }

    fn is_complex(&self) -> bool {
        storage_payload_dispatch!(self, Payload => |_payload| <Payload as StorageScalar>::IS_COMPLEX)
    }
}

/// Types that can be computed as the result of a reduction over `Storage`.
///
/// This lets callers write `let s: T = tensor.sum();` without matching on
/// the storage variant. Implemented for `f64`, `Complex64`, `f32`, and
/// `Complex32`; every reduction widens compact payload values to the requested
/// accumulator, so a narrow payload can be summed without changing its stored
/// dtype.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{Storage, SumFromStorage};
///
/// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0], &[3]).unwrap();
/// let total: f64 = f64::sum_from_storage(&s);
/// assert!((total - 6.0).abs() < 1e-10);
///
/// let narrow = Storage::from_dense_col_major(vec![1.0_f32, 2.0], &[2]).unwrap();
/// assert_eq!(f32::sum_from_storage(&narrow), 3.0_f32);
/// assert_eq!(f64::sum_from_storage(&narrow), 3.0_f64);
/// ```
pub trait SumFromStorage: Sized {
    /// Compute the sum of all elements in the storage.
    fn sum_from_storage(storage: &Storage) -> Self;
}

impl SumFromStorage for f64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        let mut sum = 0.0_f64;
        storage.for_each_payload_parts(|re, _im| sum += re);
        sum
    }
}

impl SumFromStorage for Complex64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        let mut sum = Complex64::new(0.0, 0.0);
        storage.for_each_payload_parts(|re, im| sum += Complex64::new(re, im));
        sum
    }
}

impl SumFromStorage for f32 {
    fn sum_from_storage(storage: &Storage) -> Self {
        let mut sum = 0.0_f32;
        storage.for_each_payload_parts(|re, _im| sum += re as f32);
        sum
    }
}

impl SumFromStorage for Complex32 {
    fn sum_from_storage(storage: &Storage) -> Self {
        let mut sum = Complex32::new(0.0, 0.0);
        storage.for_each_payload_parts(|re, im| sum += Complex32::new(re as f32, im as f32));
        sum
    }
}

/// Lattice width of a floating dtype, or `None` for non-floating dtypes.
fn float_lattice_width(dtype: DType) -> Option<u8> {
    match dtype {
        DType::F32 | DType::C32 => Some(32),
        DType::F64 | DType::C64 => Some(64),
        _ => None,
    }
}

/// Least upper bound dtype of a set of floating dtypes.
///
/// The lattice is `f32 < f64`, `Complex32 < Complex64`, and real `<` complex, so
/// the result takes the widest real component and the complex component as soon
/// as any input is complex. Returns `None` when an input is not floating point.
fn promote_float_dtypes(dtypes: impl IntoIterator<Item = DType>) -> Option<DType> {
    let mut width = 0_u8;
    let mut complex = false;
    for dtype in dtypes {
        width = width.max(float_lattice_width(dtype)?);
        complex |= matches!(dtype, DType::C32 | DType::C64);
    }
    match (width, complex) {
        (32, false) => Some(DType::F32),
        (64, false) => Some(DType::F64),
        (32, true) => Some(DType::C32),
        (64, true) => Some(DType::C64),
        _ => None,
    }
}

/// Rebuild a compact payload from copied metadata and replacement values.
///
/// The metadata comes from an already validated payload, so this construction
/// cannot introduce invalid strides or lengths; it is used by the infallible
/// elementwise operators.
fn rebuild_payload<T: StorageScalar>(
    payload_dims: &[usize],
    strides: &[isize],
    axis_classes: &[usize],
    data: Vec<T>,
) -> Storage {
    T::wrap_payload(StructuredStorage {
        data,
        payload_dims: payload_dims.to_vec(),
        strides: strides.to_vec(),
        axis_classes: axis_classes.to_vec(),
    })
}

/// Zip two same-scalar payloads, preserving the left payload's metadata.
fn zip_payloads<T, U>(
    left: &StructuredStorage<T>,
    right: &StructuredStorage<T>,
    operation: &'static str,
    mut combine: impl FnMut(T, T) -> U,
) -> StorageResult<Storage>
where
    T: StorageScalar,
    U: StorageScalar,
{
    if left.len() != right.len() {
        return Err(StorageError::LengthMismatch {
            operation,
            left: left.len(),
            right: right.len(),
        });
    }
    let values: Vec<U> = left
        .data()
        .iter()
        .copied()
        .zip(right.data().iter().copied())
        .map(|(x, y)| combine(x, y))
        .collect();
    StructuredStorage::new(
        values,
        left.payload_dims().to_vec(),
        left.strides().to_vec(),
        left.axis_classes().to_vec(),
    )
    .map(StructuredStorage::into_storage)
    .map_err(StorageError::from)
}

/// Materialize the logical dense values of one payload, checking the shape.
fn checked_dense_payload<T: StorageScalar>(
    payload: &StructuredStorage<T>,
    logical_dims: &[usize],
) -> StorageResult<Vec<T>> {
    let structured_dims = payload.logical_dims();
    if structured_dims != logical_dims {
        return Err(StorageError::InvalidStructuredStorage(format!(
            "logical dims {:?} do not match stored logical dims {:?}",
            logical_dims, structured_dims
        )));
    }
    payload.logical_dense_col_major_vec()
}

impl Storage {
    #[cfg(test)]
    pub(crate) fn from_repr(repr: StorageRepr) -> Self {
        Self(repr)
    }

    /// Visit every referenced compact payload value as widened `(real, imaginary)` parts.
    ///
    /// Stride gaps and padding in a structured backing buffer are not tensor
    /// values and are excluded. This is the single dtype-agnostic payload scan:
    /// reductions and elementwise algebra are written once against it.
    fn for_each_payload_parts(&self, mut visit: impl FnMut(f64, f64)) {
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            payload.for_each_payload_value(|value| {
                let (re, im) = <Payload as StorageScalar>::widened_parts(value);
                visit(re, im);
            })
        })
    }

    /// Visit every referenced compact payload entry's magnitude as `f64`.
    ///
    /// Real payloads visit `|x|`; complex payloads visit `hypot(re, im)`, and a
    /// NaN component propagates as `NaN` rather than being masked by the other
    /// component. Narrow payloads widen per value for the scan only.
    fn for_each_payload_magnitude(&self, mut visit: impl FnMut(f64)) {
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            payload.for_each_payload_value(|value| {
                let (re, im) = <Payload as StorageScalar>::widened_parts(value);
                visit(if <Payload as StorageScalar>::IS_COMPLEX {
                    if re.is_nan() || im.is_nan() {
                        f64::NAN
                    } else {
                        re.hypot(im)
                    }
                } else {
                    re.abs()
                });
            })
        })
    }

    fn invalid_storage_error(err: anyhow::Error) -> StorageError {
        StorageError::from(err)
    }

    #[cfg(test)]
    pub(crate) fn repr(&self) -> &StorageRepr {
        &self.0
    }

    fn validate_dense_len<T>(data: &[T], logical_dims: &[usize], label: &str) -> Result<()> {
        let expected_len = logical_dims.iter().try_fold(1usize, |length, &dim| {
            length
                .checked_mul(dim)
                .ok_or_else(|| anyhow!("{label} dimension product overflows for {logical_dims:?}"))
        })?;
        ensure!(
            data.len() == expected_len,
            "{label} len {} does not match logical dims {:?} (expected {})",
            data.len(),
            logical_dims,
            expected_len
        );
        Ok(())
    }

    /// Create dense storage from column-major logical values (generic over scalar type).
    ///
    /// The scalar type is inferred from the `data` argument.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape product would overflow `usize` or
    /// `data.len()` does not match the expected shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// // 2x2 matrix, column-major: [[1,3],[2,4]]
    /// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// assert!(s.is_f64());
    /// assert!(s.is_dense());
    /// assert_eq!(s.len(), 4);
    /// ```
    pub fn from_dense_col_major<T: StorageScalar>(
        data: Vec<T>,
        logical_dims: &[usize],
    ) -> Result<Self> {
        T::build_dense_storage(data, logical_dims)
    }

    /// Create diagonal storage from column-major diagonal payload values (generic over scalar type).
    ///
    /// Creates a rank-2 diagonal storage by default. The scalar type is
    /// inferred from `diag_data`.
    ///
    /// # Errors
    ///
    /// Returns an error when the diagonal payload length does not match the
    /// length required for `logical_rank` or a stride product would overflow
    /// `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_diag_col_major(vec![1.0_f64, 2.0, 3.0], 2).unwrap();
    /// assert!(s.is_diag());
    /// assert!(s.is_f64());
    /// assert_eq!(s.len(), 3);
    /// ```
    pub fn from_diag_col_major<T: StorageScalar>(
        diag_data: Vec<T>,
        logical_rank: usize,
    ) -> Result<Self> {
        T::build_diag_storage(diag_data, logical_rank)
    }

    /// Create a new 1D zero-initialized dense storage (generic over scalar type).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::new_dense::<f64>(5).unwrap();
    /// assert!(s.is_dense());
    /// assert_eq!(s.len(), 5);
    /// assert!((s.max_abs()).abs() < 1e-10);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Operation`] when the dense storage construction
    /// or allocation fails.
    pub fn new_dense<T: StorageScalar + Default>(size: usize) -> StorageResult<Self> {
        Self::from_dense_col_major(vec![T::default(); size], &[size])
            .map_err(Self::invalid_storage_error)
    }

    /// Create a new diagonal storage with the given diagonal data (generic over scalar type).
    ///
    /// # Errors
    ///
    /// Returns an error if diagonal metadata is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::new_diag(vec![1.0_f64, 2.0, 3.0]).unwrap();
    /// assert!(s.is_diag());
    /// assert!(s.is_f64());
    /// ```
    pub fn new_diag<T: StorageScalar>(diag_data: Vec<T>) -> StorageResult<Self> {
        Self::from_diag_col_major(diag_data, 2).map_err(Self::invalid_storage_error)
    }

    /// Create a new structured storage (generic over scalar type).
    ///
    /// # Errors
    ///
    /// Returns an error when `axis_classes` is not in canonical
    /// first-appearance order, when the `payload_dims` or `strides` ranks do
    /// not match `axis_classes`, when a stride is negative, when a size
    /// product would overflow `usize`, or when `data.len()` does not match the
    /// required payload length.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// // Diagonal-like structured storage: axis_classes = [0, 0]
    /// let s = Storage::new_structured(
    ///     vec![1.0_f64, 2.0],
    ///     vec![2],         // payload_dims
    ///     vec![1],         // strides
    ///     vec![0, 0],      // axis_classes: both axes map to payload axis 0
    /// ).unwrap();
    /// assert!(s.is_diag());
    /// ```
    pub fn new_structured<T: StorageScalar>(
        data: Vec<T>,
        payload_dims: Vec<usize>,
        strides: Vec<isize>,
        axis_classes: Vec<usize>,
    ) -> Result<Self> {
        T::build_structured_storage(data, payload_dims, strides, axis_classes)
    }

    /// Validate structured-storage metadata and return the required payload length.
    ///
    /// The metadata must be internally consistent: canonical axis classes, a
    /// payload rank implied by the axis classes, matching dim/stride ranks,
    /// non-negative strides, and size products that fit in `usize`. The
    /// returned length is exactly what a constructed [`Storage`] would
    /// require, so untrusted payload lengths can be rejected before any
    /// allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata is inconsistent or its size products
    /// overflow `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let required = Storage::validate_structured_metadata(
    ///     &[2],
    ///     &[1],
    ///     &[0, 0],
    /// ).unwrap();
    /// assert_eq!(required, 2);
    ///
    /// assert!(Storage::validate_structured_metadata(&[2], &[-1], &[0]).is_err());
    /// ```
    pub fn validate_structured_metadata(
        payload_dims: &[usize],
        strides: &[isize],
        axis_classes: &[usize],
    ) -> Result<usize> {
        validate_canonical_axis_classes(axis_classes)?;
        let payload_rank = axis_classes.iter().try_fold(0usize, |rank, &class_id| {
            let required_rank = class_id
                .checked_add(1)
                .ok_or_else(|| anyhow!("axis class index overflow"))?;
            Ok::<_, anyhow::Error>(rank.max(required_rank))
        })?;
        ensure!(
            payload_dims.len() == payload_rank,
            "payload rank {} does not match axis_classes {:?}",
            payload_dims.len(),
            axis_classes
        );
        ensure!(
            strides.len() == payload_dims.len(),
            "payload dims {:?} and strides {:?} must have the same rank",
            payload_dims,
            strides
        );
        required_storage_len(payload_dims, strides)
    }

    /// Check if this storage is logically dense.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    /// assert!(s.is_dense());
    ///
    /// let d = Storage::new_diag(vec![1.0_f64, 2.0]).unwrap();
    /// assert!(!d.is_dense());
    /// ```
    pub fn is_dense(&self) -> bool {
        storage_payload_dispatch!(&self.0, Payload => |value| value.is_dense())
    }

    /// Check if this storage is a Diag storage type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let d = Storage::new_diag(vec![1.0_f64, 2.0]).unwrap();
    /// assert!(d.is_diag());
    /// ```
    pub fn is_diag(&self) -> bool {
        storage_payload_dispatch!(&self.0, Payload => |value| value.is_diag())
    }

    /// Returns the compact layout class for this storage.
    ///
    /// The return value is metadata-only and never materializes dense logical
    /// values. Use it to choose whether to read compact payload metadata or
    /// dense logical values.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{Storage, StorageKind};
    ///
    /// let structured = Storage::new_structured(
    ///     vec![1.0_f64, 2.0],
    ///     vec![2],
    ///     vec![1],
    ///     vec![0, 0],
    /// ).unwrap();
    /// assert_eq!(structured.storage_kind(), StorageKind::Diagonal);
    /// ```
    pub fn storage_kind(&self) -> StorageKind {
        if self.is_dense() {
            StorageKind::Dense
        } else if self.is_diag() {
            StorageKind::Diagonal
        } else {
            StorageKind::Structured
        }
    }

    /// Returns the logical tensor dimensions represented by this storage.
    ///
    /// The dimensions are derived from payload dimensions and `axis_classes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let diag = Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap();
    /// assert_eq!(diag.logical_dims(), vec![2, 2]);
    /// ```
    pub fn logical_dims(&self) -> Vec<usize> {
        storage_payload_dispatch!(&self.0, Payload => |value| value.logical_dims())
    }

    /// Returns the logical tensor rank represented by this storage.
    ///
    /// This equals `axis_classes().len()`, not necessarily `payload_dims().len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let diag = Storage::from_diag_col_major(vec![1.0_f64, 2.0], 3).unwrap();
    /// assert_eq!(diag.logical_rank(), 3);
    /// assert_eq!(diag.payload_dims(), &[2]);
    /// ```
    pub fn logical_rank(&self) -> usize {
        storage_payload_dispatch!(&self.0, Payload => |value| value.logical_rank())
    }

    /// Returns the compact payload dimensions.
    ///
    /// For dense storage these match logical dimensions. For diagonal storage
    /// this is rank-1 even when the logical tensor has multiple axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let diag = Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap();
    /// assert_eq!(diag.payload_dims(), &[2]);
    /// ```
    pub fn payload_dims(&self) -> &[usize] {
        storage_payload_dispatch!(&self.0, Payload => |value| value.payload_dims())
    }

    /// Returns the compact payload strides.
    ///
    /// Strides are measured in stored scalar elements and describe the compact
    /// payload buffer, not the logical dense tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let dense = Storage::from_dense_col_major(vec![0.0_f64; 6], &[2, 3]).unwrap();
    /// assert_eq!(dense.payload_strides(), &[1, 2]);
    /// ```
    pub fn payload_strides(&self) -> &[isize] {
        storage_payload_dispatch!(&self.0, Payload => |value| value.strides())
    }

    /// Returns logical-axis equivalence classes for this storage.
    ///
    /// Repeated class labels mean the corresponding logical axes share one
    /// payload axis. Dense storage has `[0, 1, ...]`; diagonal storage has
    /// repeated zero labels.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let diag = Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap();
    /// assert_eq!(diag.axis_classes(), &[0, 0]);
    /// ```
    pub fn axis_classes(&self) -> &[usize] {
        storage_payload_dispatch!(&self.0, Payload => |value| value.axis_classes())
    }

    /// Returns the number of stored compact payload elements.
    ///
    /// For dense storage this equals the logical dense length. For diagonal and
    /// structured storage this is the compact payload length.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let diag = Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap();
    /// assert_eq!(diag.payload_len(), 2);
    /// ```
    pub fn payload_len(&self) -> usize {
        self.len()
    }

    /// Copies the compact `f32` payload in column-major payload order.
    ///
    /// This does not materialize logical dense values and never widens: a
    /// narrow payload stays `f32`.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] if the storage scalar type
    /// is not `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let diag = Storage::from_diag_col_major(vec![1.0_f32, 2.0], 2).unwrap();
    /// assert_eq!(diag.payload_f32_col_major_vec().unwrap(), vec![1.0_f32, 2.0]);
    /// ```
    pub fn payload_f32_col_major_vec(&self) -> StorageResult<Vec<f32>> {
        <f32 as StorageScalar>::payload_vec(self)
    }

    /// Borrows the compact `f32` payload when it is already contiguous in
    /// column-major payload order.
    ///
    /// # Errors
    ///
    /// Returns `StorageError::ScalarKindMismatch` when the storage is not
    /// f32-backed (a dtype mismatch).
    pub fn payload_f32_col_major_view_if_contiguous(&self) -> StorageResult<Option<&[f32]>> {
        <f32 as StorageScalar>::payload_view(self)
    }

    /// Copies the compact `f64` payload in column-major payload order.
    ///
    /// This does not materialize logical dense values. For diagonal storage the
    /// returned vector contains only diagonal payload values.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] if the storage scalar type
    /// is not `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let diag = Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap();
    /// assert_eq!(diag.payload_f64_col_major_vec().unwrap(), vec![1.0, 2.0]);
    /// ```
    pub fn payload_f64_col_major_vec(&self) -> StorageResult<Vec<f64>> {
        <f64 as StorageScalar>::payload_vec(self)
    }

    /// Borrows the compact `f64` payload when it is already contiguous in
    /// column-major payload order.
    ///
    /// # Errors
    ///
    /// Returns `StorageError::ScalarKindMismatch` when the storage is not
    /// f64-backed (a dtype mismatch).
    pub fn payload_f64_col_major_view_if_contiguous(&self) -> StorageResult<Option<&[f64]>> {
        <f64 as StorageScalar>::payload_view(self)
    }

    /// Copies the compact `Complex32` payload in column-major payload order.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] if the storage scalar type
    /// is not `Complex32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex32;
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let data = vec![Complex32::new(1.0, -2.0), Complex32::new(0.5, 0.0)];
    /// let diag = Storage::from_diag_col_major(data.clone(), 2).unwrap();
    /// assert_eq!(diag.payload_c32_col_major_vec().unwrap(), data);
    /// ```
    pub fn payload_c32_col_major_vec(&self) -> StorageResult<Vec<Complex32>> {
        <Complex32 as StorageScalar>::payload_vec(self)
    }

    /// Borrows the compact `Complex32` payload when it is already contiguous in
    /// column-major payload order.
    ///
    /// # Errors
    ///
    /// Returns `StorageError::ScalarKindMismatch` when the storage is not
    /// c32-backed (a dtype mismatch).
    pub fn payload_c32_col_major_view_if_contiguous(&self) -> StorageResult<Option<&[Complex32]>> {
        <Complex32 as StorageScalar>::payload_view(self)
    }

    /// Copies the compact `Complex64` payload in column-major payload order.
    ///
    /// This does not materialize logical dense values. Complex payloads are
    /// returned as native Rust `Complex64` values.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] if the storage scalar type
    /// is not `Complex64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    /// let diag = Storage::from_diag_col_major(data.clone(), 2).unwrap();
    /// assert_eq!(diag.payload_c64_col_major_vec().unwrap(), data);
    /// ```
    pub fn payload_c64_col_major_vec(&self) -> StorageResult<Vec<Complex64>> {
        <Complex64 as StorageScalar>::payload_vec(self)
    }

    /// Borrows the compact `Complex64` payload when it is already contiguous in
    /// column-major payload order.
    ///
    /// # Errors
    ///
    /// Returns `StorageError::ScalarKindMismatch` when the storage is not
    /// c64-backed (a dtype mismatch).
    pub fn payload_c64_col_major_view_if_contiguous(&self) -> StorageResult<Option<&[Complex64]>> {
        <Complex64 as StorageScalar>::payload_view(self)
    }

    /// Check if this storage uses f64 scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(vec![1.0_f64], &[1]).unwrap();
    /// assert!(s.is_f64());
    /// assert!(!s.is_c64());
    /// ```
    pub fn is_f64(&self) -> bool {
        self.dtype() == DType::F64
    }

    /// Check if this storage uses f32 scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(vec![1.0_f32], &[1]).unwrap();
    /// assert!(s.is_f32());
    /// assert!(!s.is_f64());
    /// ```
    pub fn is_f32(&self) -> bool {
        self.dtype() == DType::F32
    }

    /// Check if this storage uses Complex64 scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// use num_complex::Complex64;
    ///
    /// let s = Storage::from_dense_col_major(
    ///     vec![Complex64::new(1.0, 0.0)], &[1],
    /// ).unwrap();
    /// assert!(s.is_c64());
    /// ```
    pub fn is_c64(&self) -> bool {
        self.dtype() == DType::C64
    }

    /// Check if this storage uses Complex32 scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex32;
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(
    ///     vec![Complex32::new(1.0, -1.0)], &[1],
    /// ).unwrap();
    /// assert!(s.is_c32());
    /// assert!(s.is_complex());
    /// ```
    pub fn is_c32(&self) -> bool {
        self.dtype() == DType::C32
    }

    /// Check if this storage uses complex scalar type.
    ///
    /// This is `Complex64` or `Complex32`; it is not an alias for
    /// [`is_c64()`](Self::is_c64).
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex32;
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(vec![Complex32::new(1.0, 0.0)], &[1]).unwrap();
    /// assert!(s.is_complex());
    ///
    /// let r = Storage::from_dense_col_major(vec![1.0_f64], &[1]).unwrap();
    /// assert!(!r.is_complex());
    /// ```
    pub fn is_complex(&self) -> bool {
        self.0.is_complex()
    }

    /// Returns the stored scalar dtype.
    ///
    /// The dtype is metadata only and never depends on payload contents. It is
    /// the value a publication or conversion boundary should compare against
    /// instead of assuming the payload is `f64` or `Complex64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// use tenferro::DType;
    ///
    /// let dense = Storage::from_dense_col_major(vec![1.0_f32, 2.0], &[2]).unwrap();
    /// assert_eq!(dense.dtype(), DType::F32);
    ///
    /// let complex = Storage::from_diag_col_major(
    ///     vec![num_complex::Complex64::new(1.0, 0.0)], 2,
    /// ).unwrap();
    /// assert_eq!(complex.dtype(), DType::C64);
    /// ```
    pub fn dtype(&self) -> DType {
        self.0.dtype()
    }

    /// Returns this storage's scalar-kind label used in diagnostics.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(vec![1.0_f32], &[1]).unwrap();
    /// assert_eq!(s.scalar_kind(), "f32");
    /// ```
    pub fn scalar_kind(&self) -> &'static str {
        self.0.scalar_kind()
    }

    /// Get the length of the storage payload (number of stored elements).
    ///
    /// For dense storage this equals the product of logical dimensions.
    /// For diagonal storage this equals the diagonal length.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0], &[3]).unwrap();
    /// assert_eq!(s.len(), 3);
    ///
    /// let d = Storage::new_diag(vec![1.0_f64, 2.0]).unwrap();
    /// assert_eq!(d.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        storage_payload_dispatch!(&self.0, Payload => |value| value.len())
    }

    /// Check if the storage is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::new_dense::<f64>(0).unwrap();
    /// assert!(s.is_empty());
    ///
    /// let s2 = Storage::new_dense::<f64>(3).unwrap();
    /// assert!(!s2.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sum all referenced compact payload entries, converting to type `T`.
    ///
    /// Stride gaps and padding in a structured backing buffer are not tensor
    /// values and are excluded from the reduction.
    ///
    /// # Example
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// let s = Storage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// assert_eq!(s.sum::<f64>(), 6.0);
    /// ```
    pub fn sum<T: SumFromStorage>(&self) -> T {
        T::sum_from_storage(self)
    }

    /// Maximum absolute value over all referenced compact payload entries.
    ///
    /// Stride gaps and padding are excluded. For real storage this is `max(|x|)`, and for complex storage this is
    /// `max(hypot(re, im))`. NaN payloads, including a NaN real or imaginary
    /// component paired with infinity, propagate to a NaN result; positive
    /// infinity is preserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(vec![-3.0_f64, 1.0, 2.0], &[3]).unwrap();
    /// assert!((s.max_abs() - 3.0).abs() < 1e-10);
    /// ```
    pub fn max_abs(&self) -> f64 {
        fn fold_nan_propagating(current: f64, value: f64) -> f64 {
            if current.is_nan() || value.is_nan() {
                f64::NAN
            } else {
                current.max(value)
            }
        }

        let mut current = 0.0_f64;
        self.for_each_payload_magnitude(|magnitude| {
            current = fold_nan_propagating(current, magnitude);
        });
        current
    }

    /// Scan the compact payload without copying it.
    ///
    /// The returned flags describe the stored payload, not the logical dense
    /// tensor. Structural zeros cannot introduce non-finite values, so this is
    /// sufficient for metric validation while preserving compact memory use.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let storage = Storage::from_dense_col_major(vec![1.0_f64, f64::INFINITY], &[2]).unwrap();
    /// let flags = storage.payload_nonfinite_flags();
    /// assert!(!flags.has_nan);
    /// assert!(flags.has_infinity);
    /// ```
    pub fn payload_nonfinite_flags(&self) -> NonfiniteFlags {
        let mut flags = NonfiniteFlags::default();
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            payload.for_each_payload_value(|value| {
                let (re, im) = <Payload as StorageScalar>::widened_parts(value);
                flags.has_nan |= re.is_nan() || im.is_nan();
                flags.has_infinity |= re.is_infinite() || im.is_infinite();
            })
        });
        flags
    }

    /// Return one compact payload value as a dynamic [`BackendScalar`].
    ///
    /// `payload_coords` are column-major payload coordinates, not logical
    /// coordinates. Repeated logical axis classes are represented once in the
    /// payload, so this lookup performs only rank-bounded stride arithmetic and
    /// never allocates coordinates or scans the payload.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when the coordinate rank or bounds are invalid
    /// or the compact metadata points outside the backing payload.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let real = Storage::from_diag_col_major(vec![2.0_f64, 3.0], 2).unwrap();
    /// assert_eq!(real.scalar_at(&[1]).unwrap().real(), 3.0);
    ///
    /// let complex = Storage::from_dense_col_major(
    ///     vec![Complex64::new(1.0, -2.0)], &[1],
    /// ).unwrap();
    /// assert_eq!(complex.scalar_at(&[0]).unwrap().as_c64(),
    ///     Some(Complex64::new(1.0, -2.0)));
    /// ```
    pub fn scalar_at(&self, payload_coords: &[usize]) -> StorageResult<BackendScalar> {
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            payload
                .payload_scalar_at(payload_coords)
                .map(<Payload as StorageScalar>::to_backend_scalar)
        })
    }

    /// Materialize dense logical values as a column-major `f64` buffer.
    ///
    /// For diagonal storage, off-diagonal entries are filled with zero.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] if the storage is complex,
    /// and [`StorageError::InvalidStructuredStorage`] if `logical_dims` does
    /// not match the stored logical dimensions or the logical element count
    /// overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let dense = s.to_dense_f64_col_major_vec(&[2, 2]).unwrap();
    /// assert_eq!(dense, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn to_dense_f64_col_major_vec(&self, logical_dims: &[usize]) -> StorageResult<Vec<f64>> {
        <f64 as StorageScalar>::dense_values(self, logical_dims)
    }

    /// Materialize dense logical values as a column-major `f32` buffer.
    ///
    /// For diagonal storage, off-diagonal entries are filled with zero.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] unless the storage is
    /// `f32`-backed, and [`StorageError::InvalidStructuredStorage`] if
    /// `logical_dims` does not match the stored logical dimensions or the
    /// logical element count overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let diag = Storage::new_diag(vec![1.0_f32, 2.0]).unwrap();
    /// assert_eq!(diag.to_dense_f32_col_major_vec(&[2, 2]).unwrap(), vec![1.0, 0.0, 0.0, 2.0]);
    /// ```
    pub fn to_dense_f32_col_major_vec(&self, logical_dims: &[usize]) -> StorageResult<Vec<f32>> {
        <f32 as StorageScalar>::dense_values(self, logical_dims)
    }

    /// Materialize dense logical values as a column-major `Complex64` buffer.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] unless the storage is
    /// `Complex64`-backed, and [`StorageError::InvalidStructuredStorage`] if
    /// `logical_dims` does not match the stored logical dimensions or the
    /// logical element count overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    /// let s = Storage::from_dense_col_major(data.clone(), &[2]).unwrap();
    /// let dense = s.to_dense_c64_col_major_vec(&[2]).unwrap();
    /// assert_eq!(dense, data);
    /// ```
    pub fn to_dense_c64_col_major_vec(
        &self,
        logical_dims: &[usize],
    ) -> StorageResult<Vec<Complex64>> {
        <Complex64 as StorageScalar>::dense_values(self, logical_dims)
    }

    /// Materialize dense logical values as a column-major `Complex32` buffer.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ScalarKindMismatch`] unless the storage is
    /// `Complex32`-backed, and [`StorageError::InvalidStructuredStorage`] if
    /// `logical_dims` does not match the stored logical dimensions or the
    /// logical element count overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex32;
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let data = vec![Complex32::new(1.0, -1.0), Complex32::new(2.0, 0.5)];
    /// let s = Storage::from_dense_col_major(data.clone(), &[2]).unwrap();
    /// assert_eq!(s.to_dense_c32_col_major_vec(&[2]).unwrap(), data);
    /// ```
    pub fn to_dense_c32_col_major_vec(
        &self,
        logical_dims: &[usize],
    ) -> StorageResult<Vec<Complex32>> {
        <Complex32 as StorageScalar>::dense_values(self, logical_dims)
    }

    /// Convert this storage to dense storage.
    ///
    /// For Diag storage, creates a Dense storage with diagonal elements set
    /// and off-diagonal elements as zero. For Dense storage, returns a copy.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidStructuredStorage`] if `dims` does not
    /// match the stored logical dimensions or the dense storage cannot be built.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let d = Storage::new_diag(vec![1.0_f64, 2.0]).unwrap();
    /// let dense = d.to_dense_storage(&[2, 2]).unwrap();
    /// assert!(dense.is_dense());
    /// let vals = dense.to_dense_f64_col_major_vec(&[2, 2]).unwrap();
    /// assert_eq!(vals, vec![1.0, 0.0, 0.0, 2.0]);
    /// ```
    pub fn to_dense_storage(&self, dims: &[usize]) -> StorageResult<Storage> {
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            let values = checked_dense_payload(payload, dims)?;
            Storage::from_dense_col_major(values, dims).map_err(Self::invalid_storage_error)
        })
    }

    /// Permute the storage data according to the given permutation.
    ///
    /// The `_dims` parameter is currently unused (reserved for future use).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// // Diagonal 2x2 tensor, permute axes (identity perm for diag is valid)
    /// let d = Storage::new_diag(vec![1.0_f64, 2.0]).unwrap();
    /// let t = d.permute_storage(&[2, 2], &[1, 0]).unwrap();
    /// assert!(t.is_diag());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Operation`] when `perm` is not a permutation of
    /// the logical axes.
    pub fn permute_storage(&self, _dims: &[usize], perm: &[usize]) -> StorageResult<Storage> {
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            Ok(payload.permute_logical_axes(perm)?.into_storage())
        })
    }

    /// Extract real part from Complex64 storage as f64 storage.
    /// For f64 storage, returns a copy (clone).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    /// let s = Storage::from_dense_col_major(data, &[2]).unwrap();
    /// let re = s.extract_real_part();
    /// assert!(re.is_f64());
    /// assert_eq!(re.to_dense_f64_col_major_vec(&[2]).unwrap(), vec![1.0, 3.0]);
    /// ```
    pub fn extract_real_part(&self) -> Storage {
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            payload
                .map_copy(<Payload as StorageScalar>::real_part)
                .into_storage()
        })
    }

    /// Extract imaginary part from complex storage as the matching real storage.
    ///
    /// `Complex64` yields `f64` storage and `Complex32` yields `f32` storage.
    /// For real storage, returns zero storage of the same real dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    /// let s = Storage::from_dense_col_major(data, &[2]).unwrap();
    /// let im = s.extract_imag_part(&[2]);
    /// assert!(im.is_f64());
    /// assert_eq!(im.to_dense_f64_col_major_vec(&[2]).unwrap(), vec![2.0, 4.0]);
    /// ```
    pub fn extract_imag_part(&self, _dims: &[usize]) -> Storage {
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            payload
                .map_copy(<Payload as StorageScalar>::imag_part)
                .into_storage()
        })
    }

    /// Convert real storage to its complex counterpart (zero imaginary part).
    ///
    /// `f64` becomes `Complex64` and `f32` becomes `Complex32`; complex storage
    /// returns a clone of the same dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    /// let c = s.to_complex_storage();
    /// assert!(c.is_c64());
    ///
    /// let narrow = Storage::from_dense_col_major(vec![1.0_f32], &[1]).unwrap();
    /// assert!(narrow.to_complex_storage().is_c32());
    /// ```
    pub fn to_complex_storage(&self) -> Storage {
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            payload
                .map_copy(<Payload as StorageScalar>::into_complex)
                .into_storage()
        })
    }

    /// Complex conjugate of all elements.
    ///
    /// For real storage, returns a clone (conjugate of a real number is itself).
    /// For complex storage, conjugates each element without changing the dtype.
    ///
    /// This is inspired by the `conj` operation in ITensorMPS.jl.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)];
    /// let storage = Storage::from_dense_col_major(data, &[2]).unwrap();
    /// let conj_storage = storage.conj();
    ///
    /// let result = conj_storage.to_dense_c64_col_major_vec(&[2]).unwrap();
    /// assert_eq!(result[0], Complex64::new(1.0, -2.0));
    /// assert_eq!(result[1], Complex64::new(3.0, 4.0));
    /// ```
    pub fn conj(&self) -> Self {
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            payload
                .map_copy(<Payload as StorageScalar>::conjugate)
                .into_storage()
        })
    }

    /// Combine two real storages into their complex counterpart.
    ///
    /// `real_storage` becomes the real part and `imag_storage` the imaginary
    /// part, so `f64 + f64` yields `Complex64` storage and `f32 + f32` yields
    /// `Complex32` storage. Nothing is widened to a wider dtype.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::LengthMismatch`] if the payload lengths differ,
    /// [`StorageError::OperationNotSupported`] if the two storages do not share
    /// one real scalar kind, and
    /// [`StorageError::InvalidStructuredStorage`] if the result layout cannot
    /// be built.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// use num_complex::Complex64;
    ///
    /// let re = Storage::from_dense_col_major(vec![1.0_f64, 3.0], &[2]).unwrap();
    /// let im = Storage::from_dense_col_major(vec![2.0_f64, 4.0], &[2]).unwrap();
    /// let c = Storage::combine_to_complex(&re, &im).unwrap();
    /// assert!(c.is_c64());
    /// let vals = c.to_dense_c64_col_major_vec(&[2]).unwrap();
    /// assert_eq!(vals[0], Complex64::new(1.0, 2.0));
    /// assert_eq!(vals[1], Complex64::new(3.0, 4.0));
    ///
    /// let narrow_re = Storage::from_dense_col_major(vec![1.0_f32], &[1]).unwrap();
    /// let narrow_im = Storage::from_dense_col_major(vec![2.0_f32], &[1]).unwrap();
    /// assert!(Storage::combine_to_complex(&narrow_re, &narrow_im)
    ///     .unwrap()
    ///     .is_c32());
    /// ```
    pub fn combine_to_complex(
        real_storage: &Storage,
        imag_storage: &Storage,
    ) -> StorageResult<Storage> {
        match (&real_storage.0, &imag_storage.0) {
            (StorageRepr::F32(real), StorageRepr::F32(imag)) => {
                zip_payloads(real, imag, "combine_to_complex", |re, im| {
                    Complex32::new(re, im)
                })
            }
            (StorageRepr::F64(real), StorageRepr::F64(imag)) => {
                zip_payloads(real, imag, "combine_to_complex", |re, im| {
                    Complex64::new(re, im)
                })
            }
            _ => Err(StorageError::OperationNotSupported {
                operation: "combine_to_complex",
                left: real_storage.scalar_kind(),
                right: imag_storage.scalar_kind(),
            }),
        }
    }

    /// Add two storages element-wise, returning `Result` on error instead of panicking.
    ///
    /// Both storages must have the same scalar type and payload length; mixed
    /// scalar kinds are refused rather than promoted.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::LengthMismatch`] if the lengths differ,
    /// [`StorageError::OperationNotSupported`] if the scalar types differ, and
    /// [`StorageError::InvalidStructuredStorage`] if the result layout cannot be
    /// built.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let a = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    /// let b = Storage::from_dense_col_major(vec![3.0_f64, 4.0], &[2]).unwrap();
    /// let c = a.try_add(&b).unwrap();
    /// assert_eq!(c.to_dense_f64_col_major_vec(&[2]).unwrap(), vec![4.0, 6.0]);
    ///
    /// let narrow = Storage::from_dense_col_major(vec![0.5_f32, 0.5], &[2]).unwrap();
    /// let sum = narrow.try_add(&narrow).unwrap();
    /// assert!(sum.is_f32());
    /// assert_eq!(sum.to_dense_f32_col_major_vec(&[2]).unwrap(), vec![1.0, 1.0]);
    /// ```
    pub fn try_add(&self, other: &Storage) -> StorageResult<Storage> {
        storage_payload_same_dtype_dispatch!(&self.0, &other.0, Payload => |left, right| {
            zip_payloads(left, right, "addition", |x, y| x + y)
        },
        else Err(StorageError::OperationNotSupported {
            operation: "addition",
            left: self.scalar_kind(),
            right: other.scalar_kind(),
        }))
    }

    /// Try to subtract two storages element-wise.
    ///
    /// Both storages must have the same scalar type and payload length; mixed
    /// scalar kinds are refused rather than promoted.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::LengthMismatch`] if the lengths differ,
    /// [`StorageError::OperationNotSupported`] if the scalar types differ, and
    /// [`StorageError::InvalidStructuredStorage`] if the result layout cannot be
    /// built.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let a = Storage::from_dense_col_major(vec![5.0_f64, 7.0], &[2]).unwrap();
    /// let b = Storage::from_dense_col_major(vec![1.0_f64, 3.0], &[2]).unwrap();
    /// let c = a.try_sub(&b).unwrap();
    /// assert_eq!(c.to_dense_f64_col_major_vec(&[2]).unwrap(), vec![4.0, 4.0]);
    /// ```
    pub fn try_sub(&self, other: &Storage) -> StorageResult<Storage> {
        storage_payload_same_dtype_dispatch!(&self.0, &other.0, Payload => |left, right| {
            zip_payloads(left, right, "subtraction", |x, y| x - y)
        },
        else Err(StorageError::OperationNotSupported {
            operation: "subtraction",
            left: self.scalar_kind(),
            right: other.scalar_kind(),
        }))
    }

    /// Scale storage by a dynamic scalar value.
    ///
    /// The payload keeps its width and gains a complex component when the
    /// scalar is complex (`f64` -> `Complex64`, `f32` -> `Complex32`), which is
    /// exactly [`scale_target_dtype`]. Real scalars never widen a narrow
    /// payload.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{BackendScalar, Storage};
    ///
    /// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0], &[3]).unwrap();
    /// let scaled = s.scale(&BackendScalar::new_real(2.0));
    /// assert_eq!(scaled.to_dense_f64_col_major_vec(&[3]).unwrap(), vec![2.0, 4.0, 6.0]);
    ///
    /// let narrow = Storage::from_dense_col_major(vec![1.0_f32, 2.0], &[2]).unwrap();
    /// let complex = narrow.scale(&BackendScalar::new_complex(0.0, 1.0));
    /// assert!(complex.is_c32());
    /// ```
    pub fn scale(&self, scalar: &crate::BackendScalar) -> Storage {
        self * scalar.clone()
    }

    /// Compute linear combination: `a * self + b * other`.
    ///
    /// Promotion has one rule, shared with [`Storage::scale`]: the two payload
    /// dtypes promote along the lattice `f32 < f64`, `Complex32 < Complex64`,
    /// real `<` complex, while a scalar never widens the payload dtype - it only
    /// adds a complex component at the promoted width. So `f32 + f32` with
    /// `f64` scalars stays `f32`, `f32 + f64` becomes `f64`, and a complex
    /// scalar turns an `f32` result into `Complex32`.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::LengthMismatch`] if the lengths differ,
    /// [`StorageError::OperationNotSupported`] when an operand dtype is not a
    /// supported floating or complex floating dtype, and
    /// [`StorageError::InvalidStructuredStorage`] if the result layout cannot be
    /// built.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{BackendScalar, Storage};
    ///
    /// let x = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    /// let y = Storage::from_dense_col_major(vec![3.0_f64, 4.0], &[2]).unwrap();
    /// let a = BackendScalar::new_real(2.0);
    /// let b = BackendScalar::new_real(3.0);
    /// // result = 2*[1,2] + 3*[3,4] = [11, 16]
    /// let result = x.axpby(&a, &y, &b).unwrap();
    /// assert_eq!(result.to_dense_f64_col_major_vec(&[2]).unwrap(), vec![11.0, 16.0]);
    ///
    /// // Real scalars do not widen a narrow payload.
    /// let narrow = Storage::from_dense_col_major(vec![1.0_f32], &[1]).unwrap();
    /// assert!(narrow.axpby(&a, &narrow, &b).unwrap().is_f32());
    ///
    /// // A complex scalar adds complexity at the payload width.
    /// use num_complex::Complex32;
    /// let mixed = narrow
    ///     .axpby(
    ///         &BackendScalar::from(Complex32::new(0.0, 1.0)),
    ///         &narrow,
    ///         &BackendScalar::from(1.0_f32),
    ///     )
    ///     .unwrap();
    /// assert!(mixed.is_c32());
    /// assert_eq!(
    ///     mixed.to_dense_c32_col_major_vec(&[1]).unwrap(),
    ///     vec![Complex32::new(1.0, 1.0)],
    /// );
    /// ```
    pub fn axpby(
        &self,
        a: &crate::BackendScalar,
        other: &Storage,
        b: &crate::BackendScalar,
    ) -> StorageResult<Storage> {
        if self.len() != other.len() {
            return Err(StorageError::LengthMismatch {
                operation: "axpby",
                left: self.len(),
                right: other.len(),
            });
        }

        // One promotion rule with `scale`: the payloads promote along the lattice, then the scalars may
        // add a complex component at that width but never widen it.
        let payload_dtype =
            promote_float_dtypes([self.dtype(), other.dtype()]).ok_or_else(|| {
                StorageError::OperationNotSupported {
                    operation: "axpby",
                    left: self.scalar_kind(),
                    right: other.scalar_kind(),
                }
            })?;
        let target = scale_target_dtype(payload_dtype, scalar_dtype(a))
            .and_then(|dtype| scale_target_dtype(dtype, scalar_dtype(b)))
            .map_err(|_| StorageError::OperationNotSupported {
                operation: "axpby",
                left: self.scalar_kind(),
                right: other.scalar_kind(),
            })?;
        let (a_re, a_im) = (a.real(), a.imag());
        let (b_re, b_im) = (b.real(), b.imag());

        storage_dtype_dispatch!(target, Result => {
            let values: Vec<Result> =
                storage_payload_pair_dispatch!(&self.0, &other.0, Left, Right => |left, right| {
                    left.data()
                        .iter()
                        .zip(right.data().iter())
                        .map(|(x, y)| {
                            let (x_re, x_im) = <Left as StorageScalar>::widened_parts(*x);
                            let (y_re, y_im) = <Right as StorageScalar>::widened_parts(*y);
                            <Result as StorageScalar>::from_widened_parts(
                                a_re * x_re - a_im * x_im + b_re * y_re - b_im * y_im,
                                a_re * x_im + a_im * x_re + b_re * y_im + b_im * y_re,
                            )
                        })
                        .collect()
                });
            StructuredStorage::new(
                values,
                self.payload_dims().to_vec(),
                self.payload_strides().to_vec(),
                self.axis_classes().to_vec(),
            )
            .map(StructuredStorage::into_storage)
            .map_err(StorageError::from)
        }, else Err(StorageError::OperationNotSupported {
            operation: "axpby",
            left: self.scalar_kind(),
            right: other.scalar_kind(),
        }))
    }
}

/// Result dtype of scaling a compact payload by a dynamic scalar.
///
/// The payload keeps its width and gains a complex component only when the
/// scalar is complex: `f64` scaled by a `Complex32` or `Complex64` scalar stays
/// `Complex64`, and `f32` scaled by a complex scalar stays `Complex32`. This is
/// the compact-storage counterpart of the eager scaling rule in
/// `tensor4all-core`, which delegates here so both surfaces agree.
///
/// # Errors
///
/// Returns [`StorageError::OperationNotSupported`] when either dtype is not a
/// supported floating or complex floating dtype.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::scale_target_dtype;
/// use tenferro::DType;
///
/// assert_eq!(scale_target_dtype(DType::F64, DType::C32).unwrap(), DType::C64);
/// assert_eq!(scale_target_dtype(DType::F32, DType::C64).unwrap(), DType::C32);
/// assert_eq!(scale_target_dtype(DType::C32, DType::F64).unwrap(), DType::C32);
/// assert_eq!(scale_target_dtype(DType::F64, DType::F64).unwrap(), DType::F64);
/// assert!(scale_target_dtype(DType::I32, DType::F64).is_err());
/// ```
pub fn scale_target_dtype(payload: DType, scalar: DType) -> StorageResult<DType> {
    let width =
        float_lattice_width(payload).ok_or_else(|| StorageError::OperationNotSupported {
            operation: "scaling",
            left: dtype_label(payload),
            right: dtype_label(scalar),
        })?;
    if float_lattice_width(scalar).is_none() {
        return Err(StorageError::OperationNotSupported {
            operation: "scaling",
            left: dtype_label(payload),
            right: dtype_label(scalar),
        });
    }
    let complex =
        matches!(payload, DType::C32 | DType::C64) || matches!(scalar, DType::C32 | DType::C64);
    Ok(match (width, complex) {
        (32, false) => DType::F32,
        (64, false) => DType::F64,
        (32, true) => DType::C32,
        _ => DType::C64,
    })
}

fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::Bool => "bool",
        DType::C32 => "Complex32",
        DType::C64 => "Complex64",
        DType::External(_) => "external",
    }
}

fn scalar_dtype(scalar: &BackendScalar) -> DType {
    scalar.as_native().dtype()
}

/// Helper to get a mutable reference to storage, cloning if needed (COW).
///
/// Uses `Arc::make_mut` semantics: if the `Arc` has only one strong reference,
/// returns a mutable reference to the existing allocation. Otherwise clones
/// the inner value first.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tensor4all_tensorbackend::{make_mut_storage, Storage};
///
/// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
/// let mut arc = Arc::new(s);
/// let s_mut = make_mut_storage(&mut arc);
/// // s_mut is now a mutable reference to Storage
/// assert!(s_mut.is_f64());
/// ```
pub fn make_mut_storage(arc: &mut Arc<Storage>) -> &mut Storage {
    Arc::make_mut(arc)
}

/// Get the minimum dimension from a slice of dimensions.
///
/// Returns 1 for an empty slice. This is used for DiagTensor where all
/// indices must have the same dimension.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::min_dim;
///
/// assert_eq!(min_dim(&[2, 3, 4]), 2);
/// assert_eq!(min_dim(&[5, 5, 5]), 5);
/// assert_eq!(min_dim(&[]), 1);
/// ```
pub fn min_dim(dims: &[usize]) -> usize {
    dims.iter().copied().min().unwrap_or(1)
}

/// Contract two storage tensors along specified axes.
///
/// All storage is StructuredStorage; contraction is delegated to the native
/// tenferro backend. This is the primary tensor contraction entry point at
/// the storage layer.
///
/// # Arguments
///
/// * `storage_a` - First tensor storage
/// * `dims_a` - Dimensions of the first tensor
/// * `axes_a` - Axes of the first tensor to contract
/// * `storage_b` - Second tensor storage
/// * `dims_b` - Dimensions of the second tensor
/// * `axes_b` - Axes of the second tensor to contract
/// * `result_dims` - Dimensions of the result tensor (empty for scalar result)
///
/// # Returns
/// A new `Storage` containing the contracted result.
///
/// # Errors
///
/// Returns [`StorageError::InvalidStructuredStorage`] if the axis lists differ
/// in length, an axis is out of range, contracted dimensions do not match, or
/// the native backend rejects the contraction.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{contract_storage, Storage};
///
/// // Matrix-vector multiply: A(2x3) * v(3) -> result(2)
/// let a = Storage::from_dense_col_major(
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3],
/// ).unwrap();
/// let v = Storage::from_dense_col_major(vec![1.0, 1.0, 1.0], &[3]).unwrap();
/// let result = contract_storage(&a, &[2, 3], &[1], &v, &[3], &[0], &[2]).unwrap();
/// // Row sums: [1+3+5, 2+4+6] = [9, 12]
/// let vals = result.to_dense_f64_col_major_vec(&[2]).unwrap();
/// assert!((vals[0] - 9.0).abs() < 1e-10);
/// assert!((vals[1] - 12.0).abs() < 1e-10);
/// ```
pub fn contract_storage(
    storage_a: &Storage,
    dims_a: &[usize],
    axes_a: &[usize],
    storage_b: &Storage,
    dims_b: &[usize],
    axes_b: &[usize],
    result_dims: &[usize],
) -> StorageResult<Storage> {
    try_contract_storage(
        storage_a,
        dims_a,
        axes_a,
        storage_b,
        dims_b,
        axes_b,
        result_dims,
    )
}

fn try_contract_storage(
    storage_a: &Storage,
    dims_a: &[usize],
    axes_a: &[usize],
    storage_b: &Storage,
    dims_b: &[usize],
    axes_b: &[usize],
    result_dims: &[usize],
) -> StorageResult<Storage> {
    if axes_a.len() != axes_b.len() {
        return Err(StorageError::InvalidStructuredStorage(format!(
            "contract axes lengths must match: {} != {}",
            axes_a.len(),
            axes_b.len()
        )));
    }

    for (&a_axis, &b_axis) in axes_a.iter().zip(axes_b.iter()) {
        let Some(&a_dim) = dims_a.get(a_axis) else {
            return Err(StorageError::InvalidStructuredStorage(format!(
                "contract axis {a_axis} is out of range for left dims {dims_a:?}"
            )));
        };
        let Some(&b_dim) = dims_b.get(b_axis) else {
            return Err(StorageError::InvalidStructuredStorage(format!(
                "contract axis {b_axis} is out of range for right dims {dims_b:?}"
            )));
        };
        if a_dim != b_dim {
            return Err(StorageError::InvalidStructuredStorage(format!(
                "contracted dimensions must match: dims_a[{a_axis}] = {a_dim} != dims_b[{b_axis}] = {b_dim}"
            )));
        }
    }

    crate::tenferro_bridge::contract_storage_native(
        storage_a,
        dims_a,
        axes_a,
        storage_b,
        dims_b,
        axes_b,
        result_dims,
    )
    .map_err(|err| StorageError::InvalidStructuredStorage(err.to_string()))
}

/// Multiply storage by a real `f64` scalar.
///
/// The payload keeps its dtype: a narrow payload is not widened, and complex
/// payloads scale both components.
impl Mul<f64> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: f64) -> Self::Output {
        storage_payload_dispatch!(&self.0, Payload => |payload| {
            payload
                .map_copy(|value| {
                    let (re, im) = <Payload as StorageScalar>::widened_parts(value);
                    <Payload as StorageScalar>::from_widened_parts(re * scalar, im * scalar)
                })
                .into_storage()
        })
    }
}

/// Multiply storage by a `Complex64` scalar.
///
/// A real payload is promoted to its complex counterpart of the same width
/// (`f64` -> `Complex64`, `f32` -> `Complex32`); complex payloads keep their
/// dtype.
impl Mul<Complex64> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: Complex64) -> Self::Output {
        let target = if self.is_complex() {
            self.dtype()
        } else {
            scale_target_dtype(self.dtype(), DType::C64)
                .unwrap_or_else(|error| crate::invariant_violated("complex scaling", error))
        };
        storage_dtype_dispatch!(target, Result => {
            let values: Vec<Result> = storage_payload_dispatch!(&self.0, Source => |payload| {
                payload
                    .data()
                    .iter()
                    .map(|value| {
                        let (re, im) = <Source as StorageScalar>::widened_parts(*value);
                        <Result as StorageScalar>::from_widened_parts(
                            re * scalar.re - im * scalar.im,
                            re * scalar.im + im * scalar.re,
                        )
                    })
                    .collect()
            });
            rebuild_payload(
                self.payload_dims(),
                self.payload_strides(),
                self.axis_classes(),
                values,
            )
        }, else crate::invariant_violated(
            "complex scaling",
            anyhow!("unsupported storage dtype {:?}", self.dtype()),
        ))
    }
}

/// Multiply storage by a dynamic [`BackendScalar`].
///
/// A real scalar delegates to the real-scaling implementation, so a complex payload's components scale
/// independently and IEEE values propagate as they always have (`inf * 1` stays `inf` instead of turning
/// into `inf * 0 = NaN` through the complex product). A complex scalar goes through the complex path,
/// which keeps the payload's width and gains a complex component on a real payload.
impl Mul<BackendScalar> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: BackendScalar) -> Self::Output {
        if scalar.is_complex() {
            self * Complex64::from(scalar)
        } else {
            self * scalar.real()
        }
    }
}

#[cfg(test)]
mod tests;
