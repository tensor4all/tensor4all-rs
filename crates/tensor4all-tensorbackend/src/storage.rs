use anyhow::{anyhow, ensure, Result};
use num_complex::{Complex64, ComplexFloat};
use std::ops::{Add, Mul};
use std::sync::Arc;

/// Trait for scalar types that can be stored in [`Storage`].
///
/// This enables generic constructors such as [`Storage::from_dense_col_major`]
/// and [`Storage::from_diag_col_major`]. Implemented for `f64` and `Complex64`.
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
/// ```
pub trait StorageScalar: Clone + Send + Sync + 'static {
    /// Build a dense [`Storage`] from column-major data.
    fn build_dense_storage(data: Vec<Self>, logical_dims: &[usize]) -> Result<Storage>;
    /// Build a diagonal [`Storage`] from diagonal payload data.
    fn build_diag_storage(diag_data: Vec<Self>, logical_rank: usize) -> Result<Storage>;
    /// Build a structured [`Storage`] from explicit payload metadata.
    fn build_structured_storage(
        data: Vec<Self>,
        payload_dims: Vec<usize>,
        strides: Vec<isize>,
        axis_classes: Vec<usize>,
    ) -> Result<Storage>;
}

impl StorageScalar for f64 {
    fn build_dense_storage(data: Vec<Self>, logical_dims: &[usize]) -> Result<Storage> {
        Storage::validate_dense_len(&data, logical_dims, "dense f64 payload")?;
        Ok(Storage::from_repr(StorageRepr::F64(
            StructuredStorage::from_dense_col_major(data, logical_dims),
        )))
    }
    fn build_diag_storage(diag_data: Vec<Self>, logical_rank: usize) -> Result<Storage> {
        Ok(Storage::from_repr(StorageRepr::F64(
            StructuredStorage::from_diag_col_major(diag_data, logical_rank),
        )))
    }
    fn build_structured_storage(
        data: Vec<Self>,
        payload_dims: Vec<usize>,
        strides: Vec<isize>,
        axis_classes: Vec<usize>,
    ) -> Result<Storage> {
        Ok(Storage::from_repr(StorageRepr::F64(
            StructuredStorage::new(data, payload_dims, strides, axis_classes)?,
        )))
    }
}

impl StorageScalar for Complex64 {
    fn build_dense_storage(data: Vec<Self>, logical_dims: &[usize]) -> Result<Storage> {
        Storage::validate_dense_len(&data, logical_dims, "dense c64 payload")?;
        Ok(Storage::from_repr(StorageRepr::C64(
            StructuredStorage::from_dense_col_major(data, logical_dims),
        )))
    }
    fn build_diag_storage(diag_data: Vec<Self>, logical_rank: usize) -> Result<Storage> {
        Ok(Storage::from_repr(StorageRepr::C64(
            StructuredStorage::from_diag_col_major(diag_data, logical_rank),
        )))
    }
    fn build_structured_storage(
        data: Vec<Self>,
        payload_dims: Vec<usize>,
        strides: Vec<isize>,
        axis_classes: Vec<usize>,
    ) -> Result<Storage> {
        Ok(Storage::from_repr(StorageRepr::C64(
            StructuredStorage::new(data, payload_dims, strides, axis_classes)?,
        )))
    }
}

pub(crate) fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1isize;
    for &dim in dims {
        strides.push(stride);
        stride = stride
            .checked_mul(dim as isize)
            .unwrap_or_else(|| panic!("column-major stride overflow for dims {dims:?}"));
    }
    strides
}

fn validate_canonical_axis_classes(axis_classes: &[usize]) -> Result<()> {
    let mut next_class = 0usize;
    for &class_id in axis_classes {
        ensure!(
            class_id <= next_class,
            "axis_classes must be canonical first-appearance labels, got {axis_classes:?}"
        );
        if class_id == next_class {
            next_class += 1;
        }
    }
    Ok(())
}

fn required_storage_len(dims: &[usize], strides: &[isize]) -> Result<usize> {
    if dims.is_empty() {
        return Ok(1);
    }
    if dims.contains(&0) {
        return Ok(0);
    }
    ensure!(
        dims.len() == strides.len(),
        "payload dims {:?} and strides {:?} must have the same rank",
        dims,
        strides
    );

    let mut max_offset = 0usize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        ensure!(
            stride >= 0,
            "negative strides are not supported in StructuredStorage: {strides:?}"
        );
        if dim > 1 {
            max_offset = max_offset
                .checked_add((dim - 1) * usize::try_from(stride).unwrap_or(usize::MAX))
                .ok_or_else(|| anyhow!("payload stride overflow for dims {dims:?}"))?;
        }
    }
    Ok(max_offset + 1)
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

fn offset_from_strides(index: &[usize], strides: &[isize]) -> usize {
    index
        .iter()
        .zip(strides.iter())
        .map(|(&value, &stride)| value * usize::try_from(stride).unwrap_or(usize::MAX))
        .sum()
}

/// Structured tensor snapshot storage.
///
/// `data` and `strides` describe the payload tensor, while `axis_classes`
/// describes how logical axes map onto payload axes. Logical flat-buffer
/// semantics are column-major.
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
/// );
/// assert!(dense.is_dense());
/// assert!(!dense.is_diag());
/// assert_eq!(dense.logical_rank(), 2);
/// assert_eq!(dense.logical_dims(), vec![2, 3]);
///
/// // Diagonal 3x3 storage
/// let diag = StructuredStorage::from_diag_col_major(vec![1.0, 2.0, 3.0], 2);
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
        validate_canonical_axis_classes(&axis_classes)?;
        ensure!(
            payload_dims.len()
                == axis_classes
                    .iter()
                    .copied()
                    .max()
                    .map(|value| value + 1)
                    .unwrap_or(0),
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
        let required_len = required_storage_len(&payload_dims, &strides)?;
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
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal the product of `logical_dims`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    /// assert!(s.is_dense());
    /// assert_eq!(s.data(), &[10.0, 20.0, 30.0, 40.0]);
    /// ```
    pub fn from_dense_col_major(data: Vec<T>, logical_dims: &[usize]) -> Self {
        let payload_dims = logical_dims.to_vec();
        let strides = col_major_strides(&payload_dims);
        let axis_classes = (0..logical_dims.len()).collect();
        Self::new(data, payload_dims, strides, axis_classes)
            .unwrap_or_else(|err| panic!("StructuredStorage::from_dense_col_major failed: {err}"))
    }

    /// Creates a diagonal structured snapshot from column-major diagonal data.
    ///
    /// The resulting tensor has `logical_rank` axes, each of size `diag_data.len()`.
    /// Only the diagonal entries are stored.
    ///
    /// # Panics
    ///
    /// Panics if `logical_rank` is zero and data is non-empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0, 3.0], 2);
    /// assert!(d.is_diag());
    /// assert_eq!(d.logical_dims(), vec![3, 3]);
    /// assert_eq!(d.data(), &[1.0, 2.0, 3.0]);
    /// ```
    pub fn from_diag_col_major(diag_data: Vec<T>, logical_rank: usize) -> Self {
        let payload_dims = if logical_rank == 0 {
            vec![]
        } else {
            vec![diag_data.len()]
        };
        let strides = col_major_strides(&payload_dims);
        let axis_classes = vec![0; logical_rank];
        Self::new(diag_data, payload_dims, strides, axis_classes)
            .unwrap_or_else(|err| panic!("StructuredStorage::from_diag_col_major failed: {err}"))
    }

    /// Returns the payload data buffer as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0], &[2]);
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
    /// let s = StructuredStorage::from_dense_col_major(vec![0.0; 6], &[2, 3]);
    /// assert_eq!(s.payload_dims(), &[2, 3]);
    ///
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 3);
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
    /// let s = StructuredStorage::from_dense_col_major(vec![0.0; 6], &[2, 3]);
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
    /// let dense = StructuredStorage::from_dense_col_major(vec![0.0; 4], &[2, 2]);
    /// assert_eq!(dense.axis_classes(), &[0, 1]);
    ///
    /// let diag = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2);
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
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0, 3.0], 3);
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
    /// let s = StructuredStorage::from_dense_col_major(vec![0.0; 6], &[2, 3]);
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
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0], &[2]);
    /// assert!(s.is_dense());
    ///
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2);
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
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2);
    /// assert!(d.is_diag());
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0], &[2]);
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
    /// let dense = StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]);
    /// assert_eq!(dense.len(), 3);
    ///
    /// let diag = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2);
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
    /// let empty = StructuredStorage::from_dense_col_major(Vec::<f64>::new(), &[0]);
    /// assert!(empty.is_empty());
    ///
    /// let non_empty = StructuredStorage::from_dense_col_major(vec![1.0], &[1]);
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
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]);
    /// assert_eq!(s.dense_col_major_view_if_contiguous(), Some(&[1.0, 2.0, 3.0][..]));
    ///
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2);
    /// assert_eq!(d.dense_col_major_view_if_contiguous(), None);
    /// ```
    pub fn dense_col_major_view_if_contiguous(&self) -> Option<&[T]> {
        if self.is_dense() && self.strides == col_major_strides(&self.payload_dims) {
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
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// assert_eq!(s.payload_col_major_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn payload_col_major_vec(&self) -> Vec<T> {
        let payload_len: usize = self.payload_dims.iter().product();
        if payload_len == 0 {
            return Vec::new();
        }
        if self.strides == col_major_strides(&self.payload_dims) {
            return self.data.clone();
        }

        (0..payload_len)
            .map(|linear| {
                let index = col_major_multi_index(linear, &self.payload_dims);
                let offset = offset_from_strides(&index, &self.strides);
                self.data[offset].clone()
            })
            .collect()
    }

    /// Returns a copy of the storage with logical axes permuted.
    ///
    /// # Panics
    ///
    /// Panics if `perm.len()` does not equal the logical rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// // Diagonal 3x3x3 tensor; permute axes (identity for diag is always valid)
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0, 3.0], 3);
    /// let p = d.permute_logical_axes(&[2, 0, 1]);
    /// // Diagonal: all axes share the same dimension, so dims stay the same
    /// assert_eq!(p.logical_dims(), vec![3, 3, 3]);
    /// assert!(p.is_diag());
    /// ```
    pub fn permute_logical_axes(&self, perm: &[usize]) -> Self {
        assert_eq!(
            perm.len(),
            self.axis_classes.len(),
            "logical permutation length {} must match logical rank {}",
            perm.len(),
            self.axis_classes.len()
        );
        let axis_classes = perm.iter().map(|&index| self.axis_classes[index]).collect();
        Self::new(
            self.data.clone(),
            self.payload_dims.clone(),
            self.strides.clone(),
            axis_classes,
        )
        .unwrap_or_else(|err| panic!("StructuredStorage::permute_logical_axes failed: {err}"))
    }
}

impl<T: Copy> StructuredStorage<T> {
    /// Maps payload elements while preserving payload metadata and axis classes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// let s = StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]);
    /// let doubled = s.map_copy(|x| x * 2.0);
    /// assert_eq!(doubled.data(), &[2.0, 4.0, 6.0]);
    /// ```
    pub fn map_copy<U>(&self, mut f: impl FnMut(T) -> U) -> StructuredStorage<U> {
        StructuredStorage::new(
            self.data.iter().copied().map(&mut f).collect(),
            self.payload_dims.clone(),
            self.strides.clone(),
            self.axis_classes.clone(),
        )
        .unwrap_or_else(|err| panic!("StructuredStorage::map_copy failed: {err}"))
    }
}

impl<T: Copy + Default> StructuredStorage<T> {
    /// Materializes the logical tensor as a contiguous column-major dense buffer.
    ///
    /// Repeated entries in `axis_classes` encode equality constraints between
    /// logical axes. Logical indices that violate those constraints are
    /// structural zeros in the dense materialization.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::StructuredStorage;
    ///
    /// // Diagonal [1, 2] in 2x2 becomes [1, 0, 0, 2] column-major
    /// let d = StructuredStorage::from_diag_col_major(vec![1.0, 2.0], 2);
    /// assert_eq!(d.logical_dense_col_major_vec(), vec![1.0, 0.0, 0.0, 2.0]);
    /// ```
    pub fn logical_dense_col_major_vec(&self) -> Vec<T> {
        let logical_dims = self.logical_dims();
        let logical_len: usize = logical_dims.iter().product();
        if logical_len == 0 {
            return Vec::new();
        }
        if let Some(view) = self.dense_col_major_view_if_contiguous() {
            return view.to_vec();
        }
        if self.is_dense() {
            return self.payload_col_major_vec();
        }

        let payload_rank = self.payload_dims.len();
        (0..logical_len)
            .map(|linear| {
                let logical_index = col_major_multi_index(linear, &logical_dims);
                let mut payload_index = vec![0usize; payload_rank];
                let mut seen = vec![false; payload_rank];
                for (&value, &class_id) in logical_index.iter().zip(self.axis_classes.iter()) {
                    if seen[class_id] {
                        if payload_index[class_id] != value {
                            return T::default();
                        }
                    } else {
                        payload_index[class_id] = value;
                        seen[class_id] = true;
                    }
                }
                let offset = offset_from_strides(&payload_index, &self.strides);
                self.data[offset]
            })
            .collect()
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
/// let diag = Storage::new_diag(vec![1.0_f64, 2.0]);
/// assert!(diag.is_f64());
/// ```
#[derive(Debug, Clone)]
pub struct Storage(pub(crate) StorageRepr);

#[derive(Debug, Clone)]
pub(crate) enum StorageRepr {
    /// Storage with f64 elements.
    F64(StructuredStorage<f64>),
    /// Storage with Complex64 elements.
    C64(StructuredStorage<Complex64>),
}

/// Types that can be computed as the result of a reduction over `Storage`.
///
/// This lets callers write `let s: T = tensor.sum();` without matching on
/// the storage variant. Implemented for `f64` and `Complex64`.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{Storage, SumFromStorage};
///
/// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0], &[3]).unwrap();
/// let total: f64 = f64::sum_from_storage(&s);
/// assert!((total - 6.0).abs() < 1e-10);
/// ```
pub trait SumFromStorage: Sized {
    /// Compute the sum of all elements in the storage.
    fn sum_from_storage(storage: &Storage) -> Self;
}

impl SumFromStorage for f64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        match &storage.0 {
            StorageRepr::F64(v) => v.data().iter().copied().sum(),
            StorageRepr::C64(v) => v.data().iter().map(|z| z.re).sum(),
        }
    }
}

impl SumFromStorage for Complex64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        match &storage.0 {
            StorageRepr::F64(v) => Complex64::new(v.data().iter().copied().sum(), 0.0),
            StorageRepr::C64(v) => v.data().iter().copied().sum(),
        }
    }
}

// AnyScalar is now in its own module
pub use crate::any_scalar::AnyScalar;

impl Storage {
    pub(crate) fn from_repr(repr: StorageRepr) -> Self {
        Self(repr)
    }

    #[cfg(test)]
    pub(crate) fn repr(&self) -> &StorageRepr {
        &self.0
    }

    fn validate_dense_len<T>(data: &[T], logical_dims: &[usize], label: &str) -> Result<()> {
        let expected_len: usize = logical_dims.iter().product();
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
    /// Currently infallible for valid data, but returns `Result` for consistency.
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
    /// let s = Storage::new_dense::<f64>(5);
    /// assert!(s.is_dense());
    /// assert_eq!(s.len(), 5);
    /// assert!((s.max_abs()).abs() < 1e-10);
    /// ```
    pub fn new_dense<T: StorageScalar + Default>(size: usize) -> Self {
        Self::from_dense_col_major(vec![T::default(); size], &[size])
            .unwrap_or_else(|err| panic!("Storage::new_dense failed: {err}"))
    }

    /// Create a new diagonal storage with the given diagonal data (generic over scalar type).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::new_diag(vec![1.0_f64, 2.0, 3.0]);
    /// assert!(s.is_diag());
    /// assert!(s.is_f64());
    /// ```
    pub fn new_diag<T: StorageScalar>(diag_data: Vec<T>) -> Self {
        Self::from_diag_col_major(diag_data, 2)
            .unwrap_or_else(|err| panic!("Storage::new_diag failed: {err}"))
    }

    /// Create a new structured storage (generic over scalar type).
    ///
    /// # Errors
    ///
    /// Returns an error if the structured metadata is inconsistent (see
    /// [`StructuredStorage::new`] for details).
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

    /// Create dense f64 storage from column-major logical values.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len()` does not match the product of `logical_dims`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_f64_col_major(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// assert!(s.is_f64());
    /// assert!(s.is_dense());
    /// ```
    pub fn from_dense_f64_col_major(data: Vec<f64>, logical_dims: &[usize]) -> Result<Self> {
        Self::validate_dense_len(&data, logical_dims, "dense f64 payload")?;
        Ok(Self::from_repr(StorageRepr::F64(
            StructuredStorage::from_dense_col_major(data, logical_dims),
        )))
    }

    /// Create dense Complex64 storage from column-major logical values.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len()` does not match the product of `logical_dims`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    /// let s = Storage::from_dense_c64_col_major(data, &[2]).unwrap();
    /// assert!(s.is_c64());
    /// assert!(s.is_dense());
    /// ```
    pub fn from_dense_c64_col_major(data: Vec<Complex64>, logical_dims: &[usize]) -> Result<Self> {
        Self::validate_dense_len(&data, logical_dims, "dense c64 payload")?;
        Ok(Self::from_repr(StorageRepr::C64(
            StructuredStorage::from_dense_col_major(data, logical_dims),
        )))
    }

    /// Create diagonal f64 storage from column-major diagonal payload values.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_diag_f64_col_major(vec![1.0, 2.0], 2).unwrap();
    /// assert!(s.is_diag());
    /// assert!(s.is_f64());
    /// ```
    pub fn from_diag_f64_col_major(diag_data: Vec<f64>, logical_rank: usize) -> Result<Self> {
        Ok(Self::from_repr(StorageRepr::F64(
            StructuredStorage::from_diag_col_major(diag_data, logical_rank),
        )))
    }

    /// Create diagonal Complex64 storage from column-major diagonal payload values.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    /// let s = Storage::from_diag_c64_col_major(data, 2).unwrap();
    /// assert!(s.is_diag());
    /// assert!(s.is_c64());
    /// ```
    pub fn from_diag_c64_col_major(diag_data: Vec<Complex64>, logical_rank: usize) -> Result<Self> {
        Ok(Self::from_repr(StorageRepr::C64(
            StructuredStorage::from_diag_col_major(diag_data, logical_rank),
        )))
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
    /// let d = Storage::new_diag(vec![1.0_f64, 2.0]);
    /// assert!(!d.is_dense());
    /// ```
    pub fn is_dense(&self) -> bool {
        match &self.0 {
            StorageRepr::F64(value) => value.is_dense(),
            StorageRepr::C64(value) => value.is_dense(),
        }
    }

    /// Check if this storage is a Diag storage type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let d = Storage::new_diag(vec![1.0_f64, 2.0]);
    /// assert!(d.is_diag());
    /// ```
    pub fn is_diag(&self) -> bool {
        match &self.0 {
            StorageRepr::F64(value) => value.is_diag(),
            StorageRepr::C64(value) => value.is_diag(),
        }
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
        matches!(&self.0, StorageRepr::F64(_))
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
        matches!(&self.0, StorageRepr::C64(_))
    }

    /// Check if this storage uses complex scalar type.
    ///
    /// This is an alias for [`is_c64()`](Self::is_c64).
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
    /// assert!(s.is_complex());
    ///
    /// let r = Storage::from_dense_col_major(vec![1.0_f64], &[1]).unwrap();
    /// assert!(!r.is_complex());
    /// ```
    pub fn is_complex(&self) -> bool {
        self.is_c64()
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
    /// let d = Storage::new_diag(vec![1.0_f64, 2.0]);
    /// assert_eq!(d.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        match &self.0 {
            StorageRepr::F64(v) => v.len(),
            StorageRepr::C64(v) => v.len(),
        }
    }

    /// Check if the storage is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::new_dense::<f64>(0);
    /// assert!(s.is_empty());
    ///
    /// let s2 = Storage::new_dense::<f64>(3);
    /// assert!(!s2.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sum all elements, converting to type `T`.
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

    /// Maximum absolute value over all stored elements.
    ///
    /// For real storage this is `max(|x|)`, and for complex storage this is
    /// `max(norm(z))`.
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
        match &self.0 {
            StorageRepr::F64(v) => v.data().iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
            StorageRepr::C64(v) => v.data().iter().map(|z| z.norm()).fold(0.0_f64, f64::max),
        }
    }

    /// Materialize dense logical values as a column-major `f64` buffer.
    ///
    /// For diagonal storage, off-diagonal entries are filled with zero.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage is complex or `logical_dims` does not
    /// match the stored logical dimensions.
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
    pub fn to_dense_f64_col_major_vec(&self, logical_dims: &[usize]) -> Result<Vec<f64>, String> {
        match &self.0 {
            StorageRepr::F64(v) => {
                let structured_dims = v.logical_dims();
                if structured_dims != logical_dims {
                    return Err(format!(
                        "logical dims {:?} do not match StructuredF64 logical dims {:?}",
                        logical_dims, structured_dims
                    ));
                }
                Ok(v.logical_dense_col_major_vec())
            }
            StorageRepr::C64(_) => {
                Err("expected f64 storage when materializing dense f64 values".to_string())
            }
        }
    }

    /// Materialize dense logical values as a column-major `Complex64` buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage is real or `logical_dims` does not
    /// match the stored logical dimensions.
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
    ) -> Result<Vec<Complex64>, String> {
        match &self.0 {
            StorageRepr::C64(v) => {
                let structured_dims = v.logical_dims();
                if structured_dims != logical_dims {
                    return Err(format!(
                        "logical dims {:?} do not match StructuredC64 logical dims {:?}",
                        logical_dims, structured_dims
                    ));
                }
                Ok(v.logical_dense_col_major_vec())
            }
            StorageRepr::F64(_) => {
                Err("expected Complex64 storage when materializing dense c64 values".to_string())
            }
        }
    }

    /// Convert this storage to dense storage.
    ///
    /// For Diag storage, creates a Dense storage with diagonal elements set
    /// and off-diagonal elements as zero. For Dense storage, returns a copy.
    ///
    /// # Panics
    ///
    /// Panics if `dims` does not match the stored logical dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let d = Storage::new_diag(vec![1.0_f64, 2.0]);
    /// let dense = d.to_dense_storage(&[2, 2]);
    /// assert!(dense.is_dense());
    /// let vals = dense.to_dense_f64_col_major_vec(&[2, 2]).unwrap();
    /// assert_eq!(vals, vec![1.0, 0.0, 0.0, 2.0]);
    /// ```
    pub fn to_dense_storage(&self, dims: &[usize]) -> Storage {
        if self.is_f64() {
            let values = self
                .to_dense_f64_col_major_vec(dims)
                .unwrap_or_else(|err| panic!("Storage::to_dense_storage failed: {err}"));
            Storage::from_dense_col_major(values, dims)
                .unwrap_or_else(|err| panic!("Storage::to_dense_storage failed: {err}"))
        } else {
            let values = self
                .to_dense_c64_col_major_vec(dims)
                .unwrap_or_else(|err| panic!("Storage::to_dense_storage failed: {err}"));
            Storage::from_dense_col_major(values, dims)
                .unwrap_or_else(|err| panic!("Storage::to_dense_storage failed: {err}"))
        }
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
    /// let d = Storage::new_diag(vec![1.0_f64, 2.0]);
    /// let t = d.permute_storage(&[2, 2], &[1, 0]);
    /// assert!(t.is_diag());
    /// ```
    pub fn permute_storage(&self, _dims: &[usize], perm: &[usize]) -> Storage {
        match &self.0 {
            StorageRepr::F64(v) => {
                Storage::from_repr(StorageRepr::F64(v.permute_logical_axes(perm)))
            }
            StorageRepr::C64(v) => {
                Storage::from_repr(StorageRepr::C64(v.permute_logical_axes(perm)))
            }
        }
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
        match &self.0 {
            StorageRepr::F64(v) => Storage::from_repr(StorageRepr::F64(v.clone())),
            StorageRepr::C64(v) => Storage::from_repr(StorageRepr::F64(v.map_copy(|z| z.re))),
        }
    }

    /// Extract imaginary part from Complex64 storage as f64 storage.
    /// For f64 storage, returns zero storage.
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
        match &self.0 {
            StorageRepr::F64(v) => Storage::from_repr(StorageRepr::F64(v.map_copy(|_| 0.0))),
            StorageRepr::C64(v) => Storage::from_repr(StorageRepr::F64(v.map_copy(|z| z.im))),
        }
    }

    /// Convert f64 storage to Complex64 storage (real part only, imaginary part is zero).
    /// For Complex64 storage, returns a clone.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    /// let c = s.to_complex_storage();
    /// assert!(c.is_c64());
    /// ```
    pub fn to_complex_storage(&self) -> Storage {
        match &self.0 {
            StorageRepr::F64(v) => {
                Storage::from_repr(StorageRepr::C64(v.map_copy(|x| Complex64::new(x, 0.0))))
            }
            StorageRepr::C64(v) => Storage::from_repr(StorageRepr::C64(v.clone())),
        }
    }

    /// Complex conjugate of all elements.
    ///
    /// For real (f64) storage, returns a clone (conjugate of real is identity).
    /// For complex (Complex64) storage, conjugates each element.
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
        match &self.0 {
            StorageRepr::F64(v) => Storage::from_repr(StorageRepr::F64(v.clone())),
            StorageRepr::C64(v) => Storage::from_repr(StorageRepr::C64(v.map_copy(|z| z.conj()))),
        }
    }

    /// Combine two f64 storages into Complex64 storage.
    ///
    /// `real_storage` becomes the real part, `imag_storage` becomes the imaginary part.
    /// Formula: `real + i * imag`.
    ///
    /// # Panics
    ///
    /// Panics if either storage is not f64, or if their lengths differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Storage;
    /// use num_complex::Complex64;
    ///
    /// let re = Storage::from_dense_col_major(vec![1.0_f64, 3.0], &[2]).unwrap();
    /// let im = Storage::from_dense_col_major(vec![2.0_f64, 4.0], &[2]).unwrap();
    /// let c = Storage::combine_to_complex(&re, &im);
    /// assert!(c.is_c64());
    /// let vals = c.to_dense_c64_col_major_vec(&[2]).unwrap();
    /// assert_eq!(vals[0], Complex64::new(1.0, 2.0));
    /// assert_eq!(vals[1], Complex64::new(3.0, 4.0));
    /// ```
    pub fn combine_to_complex(real_storage: &Storage, imag_storage: &Storage) -> Storage {
        match (&real_storage.0, &imag_storage.0) {
            (StorageRepr::F64(real), StorageRepr::F64(imag)) => {
                assert_eq!(real.len(), imag.len(), "Storage lengths must match");
                let complex_vec: Vec<Complex64> = real
                    .data()
                    .iter()
                    .zip(imag.data().iter())
                    .map(|(&r, &i)| Complex64::new(r, i))
                    .collect();
                Storage::from_repr(StorageRepr::C64(
                    StructuredStorage::new(
                        complex_vec,
                        real.payload_dims().to_vec(),
                        real.strides().to_vec(),
                        real.axis_classes().to_vec(),
                    )
                    .unwrap_or_else(|err| panic!("Storage::combine_to_complex failed: {err}")),
                ))
            }
            _ => panic!("Both storages must be f64 for combine_to_complex"),
        }
    }

    /// Add two storages element-wise, returning `Result` on error instead of panicking.
    ///
    /// Both storages must have the same type and length.
    ///
    /// # Errors
    ///
    /// Returns an error if storage types or lengths don't match.
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
    /// ```
    pub fn try_add(&self, other: &Storage) -> Result<Storage, String> {
        match (&self.0, &other.0) {
            (StorageRepr::F64(a), StorageRepr::F64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for addition: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let sum_vec: Vec<f64> = a
                    .data()
                    .iter()
                    .zip(b.data().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Ok(Storage::from_repr(StorageRepr::F64(
                    StructuredStorage::new(
                        sum_vec,
                        a.payload_dims().to_vec(),
                        a.strides().to_vec(),
                        a.axis_classes().to_vec(),
                    )
                    .map_err(|err| err.to_string())?,
                )))
            }
            (StorageRepr::C64(a), StorageRepr::C64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for addition: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let sum_vec: Vec<Complex64> = a
                    .data()
                    .iter()
                    .zip(b.data().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Ok(Storage::from_repr(StorageRepr::C64(
                    StructuredStorage::new(
                        sum_vec,
                        a.payload_dims().to_vec(),
                        a.strides().to_vec(),
                        a.axis_classes().to_vec(),
                    )
                    .map_err(|err| err.to_string())?,
                )))
            }
            _ => Err(format!(
                "Storage types must match for addition: {:?} vs {:?}",
                std::mem::discriminant(&self.0),
                std::mem::discriminant(&other.0)
            )),
        }
    }

    /// Try to subtract two storages element-wise.
    ///
    /// # Errors
    ///
    /// Returns an error if the storages have different types or lengths.
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
    pub fn try_sub(&self, other: &Storage) -> Result<Storage, String> {
        match (&self.0, &other.0) {
            (StorageRepr::F64(a), StorageRepr::F64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for subtraction: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let diff_vec: Vec<f64> = a
                    .data()
                    .iter()
                    .zip(b.data().iter())
                    .map(|(&x, &y)| x - y)
                    .collect();
                Ok(Storage::from_repr(StorageRepr::F64(
                    StructuredStorage::new(
                        diff_vec,
                        a.payload_dims().to_vec(),
                        a.strides().to_vec(),
                        a.axis_classes().to_vec(),
                    )
                    .map_err(|err| err.to_string())?,
                )))
            }
            (StorageRepr::C64(a), StorageRepr::C64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for subtraction: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let diff_vec: Vec<Complex64> = a
                    .data()
                    .iter()
                    .zip(b.data().iter())
                    .map(|(&x, &y)| x - y)
                    .collect();
                Ok(Storage::from_repr(StorageRepr::C64(
                    StructuredStorage::new(
                        diff_vec,
                        a.payload_dims().to_vec(),
                        a.strides().to_vec(),
                        a.axis_classes().to_vec(),
                    )
                    .map_err(|err| err.to_string())?,
                )))
            }
            _ => Err(format!(
                "Storage types must match for subtraction: {:?} vs {:?}",
                std::mem::discriminant(&self.0),
                std::mem::discriminant(&other.0)
            )),
        }
    }

    /// Scale storage by a scalar value.
    ///
    /// If the scalar is complex but the storage is real, the storage is promoted to complex.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{AnyScalar, Storage};
    ///
    /// let s = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0], &[3]).unwrap();
    /// let scaled = s.scale(&AnyScalar::new_real(2.0));
    /// assert_eq!(scaled.to_dense_f64_col_major_vec(&[3]).unwrap(), vec![2.0, 4.0, 6.0]);
    /// ```
    pub fn scale(&self, scalar: &crate::AnyScalar) -> Storage {
        self * scalar.clone()
    }

    /// Compute linear combination: `a * self + b * other`.
    ///
    /// # Errors
    ///
    /// Returns an error if the storages have different types or lengths.
    /// If any scalar is complex, the result is promoted to complex.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{AnyScalar, Storage};
    ///
    /// let x = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    /// let y = Storage::from_dense_col_major(vec![3.0_f64, 4.0], &[2]).unwrap();
    /// let a = AnyScalar::new_real(2.0);
    /// let b = AnyScalar::new_real(3.0);
    /// // result = 2*[1,2] + 3*[3,4] = [11, 16]
    /// let result = x.axpby(&a, &y, &b).unwrap();
    /// assert_eq!(result.to_dense_f64_col_major_vec(&[2]).unwrap(), vec![11.0, 16.0]);
    /// ```
    pub fn axpby(
        &self,
        a: &crate::AnyScalar,
        other: &Storage,
        b: &crate::AnyScalar,
    ) -> Result<Storage, String> {
        // First check lengths match
        if self.len() != other.len() {
            return Err(format!(
                "Storage lengths must match for axpby: {} != {}",
                self.len(),
                other.len()
            ));
        }

        // Determine if we need complex output
        let needs_complex = a.is_complex()
            || b.is_complex()
            || matches!(&self.0, StorageRepr::C64(_))
            || matches!(&other.0, StorageRepr::C64(_));

        if needs_complex {
            // Promote everything to complex
            let a_c: Complex64 = a.clone().into();
            let b_c: Complex64 = b.clone().into();

            let (result, payload_dims, strides, axis_classes): (
                Vec<Complex64>,
                Vec<usize>,
                Vec<isize>,
                Vec<usize>,
            ) = match (&self.0, &other.0) {
                (StorageRepr::F64(x), StorageRepr::F64(y)) => (
                    x.data()
                        .iter()
                        .zip(y.data().iter())
                        .map(|(&xi, &yi)| {
                            a_c * Complex64::new(xi, 0.0) + b_c * Complex64::new(yi, 0.0)
                        })
                        .collect(),
                    x.payload_dims().to_vec(),
                    x.strides().to_vec(),
                    x.axis_classes().to_vec(),
                ),
                (StorageRepr::F64(x), StorageRepr::C64(y)) => (
                    x.data()
                        .iter()
                        .zip(y.data().iter())
                        .map(|(&xi, &yi)| a_c * Complex64::new(xi, 0.0) + b_c * yi)
                        .collect(),
                    x.payload_dims().to_vec(),
                    x.strides().to_vec(),
                    x.axis_classes().to_vec(),
                ),
                (StorageRepr::C64(x), StorageRepr::F64(y)) => (
                    x.data()
                        .iter()
                        .zip(y.data().iter())
                        .map(|(&xi, &yi)| a_c * xi + b_c * Complex64::new(yi, 0.0))
                        .collect(),
                    x.payload_dims().to_vec(),
                    x.strides().to_vec(),
                    x.axis_classes().to_vec(),
                ),
                (StorageRepr::C64(x), StorageRepr::C64(y)) => (
                    x.data()
                        .iter()
                        .zip(y.data().iter())
                        .map(|(&xi, &yi)| a_c * xi + b_c * yi)
                        .collect(),
                    x.payload_dims().to_vec(),
                    x.strides().to_vec(),
                    x.axis_classes().to_vec(),
                ),
            };
            Ok(Storage::from_repr(StorageRepr::C64(
                StructuredStorage::new(result, payload_dims, strides, axis_classes)
                    .map_err(|err| err.to_string())?,
            )))
        } else {
            // All real
            if !a.is_real() || !b.is_real() {
                return Err(format!(
                    "expected real scalars in real axpby branch: a={a}, b={b}"
                ));
            }
            let a_f = a.real();
            let b_f = b.real();

            match (&self.0, &other.0) {
                (StorageRepr::F64(x), StorageRepr::F64(y)) => {
                    let result: Vec<f64> = x
                        .data()
                        .iter()
                        .zip(y.data().iter())
                        .map(|(&xi, &yi)| a_f * xi + b_f * yi)
                        .collect();
                    Ok(Storage::from_repr(StorageRepr::F64(
                        StructuredStorage::new(
                            result,
                            x.payload_dims().to_vec(),
                            x.strides().to_vec(),
                            x.axis_classes().to_vec(),
                        )
                        .map_err(|err| err.to_string())?,
                    )))
                }
                _ => Err(format!(
                    "axpby not supported for storage types: {:?} vs {:?}",
                    std::mem::discriminant(&self.0),
                    std::mem::discriminant(&other.0)
                )),
            }
        }
    }
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
/// use tensor4all_tensorbackend::mindim;
///
/// assert_eq!(mindim(&[2, 3, 4]), 2);
/// assert_eq!(mindim(&[5, 5, 5]), 5);
/// assert_eq!(mindim(&[]), 1);
/// ```
pub fn mindim(dims: &[usize]) -> usize {
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
/// # Panics
///
/// Panics if the contracted dimensions don't match, or if the storage types
/// are incompatible.
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
/// let result = contract_storage(&a, &[2, 3], &[1], &v, &[3], &[0], &[2]);
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
) -> Storage {
    // Verify that contracted dimensions match
    for (a_axis, b_axis) in axes_a.iter().zip(axes_b.iter()) {
        assert_eq!(
            dims_a[*a_axis], dims_b[*b_axis],
            "Contracted dimensions must match: dims_a[{}] = {} != dims_b[{}] = {}",
            a_axis, dims_a[*a_axis], b_axis, dims_b[*b_axis]
        );
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
    .unwrap_or_else(|err| panic!("contract_storage failed: {err}"))
}

/// Add two storages element-wise.
/// Both storages must have the same type and length.
///
/// # Panics
///
/// Panics if storage types don't match or lengths differ.
///
/// # Note
///
/// **Prefer using [`Storage::try_add`]** which returns a `Result` instead of panicking.
/// This trait implementation is kept for convenience but may panic on invalid inputs.
impl Add<&Storage> for &Storage {
    type Output = Storage;

    fn add(self, rhs: &Storage) -> Self::Output {
        self.try_add(rhs)
            .unwrap_or_else(|err| panic!("Storage addition failed: {err}"))
    }
}

/// Multiply storage by a scalar (f64).
/// For Complex64 storage, multiplies each element by the scalar (treated as real).
impl Mul<f64> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: f64) -> Self::Output {
        match &self.0 {
            StorageRepr::F64(v) => Storage::from_repr(StorageRepr::F64(v.map_copy(|x| x * scalar))),
            StorageRepr::C64(v) => Storage::from_repr(StorageRepr::C64(
                v.map_copy(|z| z * Complex64::new(scalar, 0.0)),
            )),
        }
    }
}

/// Multiply storage by a scalar (Complex64).
impl Mul<Complex64> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: Complex64) -> Self::Output {
        match &self.0 {
            StorageRepr::F64(v) => Storage::from_repr(StorageRepr::C64(
                v.map_copy(|x| Complex64::new(x, 0.0) * scalar),
            )),
            StorageRepr::C64(v) => Storage::from_repr(StorageRepr::C64(v.map_copy(|z| z * scalar))),
        }
    }
}

/// Multiply storage by a scalar (AnyScalar).
/// May promote f64 storage to Complex64 when scalar is complex.
impl Mul<AnyScalar> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: AnyScalar) -> Self::Output {
        if scalar.is_complex() {
            let z: Complex64 = scalar.into();
            self * z
        } else {
            self * scalar.real()
        }
    }
}

#[cfg(test)]
mod tests;
