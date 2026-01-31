//! Layer 1: ITensor (TensorDynLen) HDF5 read/write (ITensors.jl compatible).

use anyhow::{bail, Context, Result};
use hdf5::types::VarLenUnicode;
use hdf5::Group;
use ndarray::ArrayD;
use num_complex::Complex64;
use std::str::FromStr;
use tensor4all_core::defaults::tensordynlen::TensorAccess;
use tensor4all_core::TensorDynLen;

use crate::index;

/// Convert row-major flat data to column-major flat data.
///
/// Row-major (C order): last axis varies fastest.
/// Column-major (Fortran order): first axis varies fastest.
///
/// We use ndarray's axis reversal to perform the conversion.
fn row_major_to_col_major_f64(data: &[f64], dims: &[usize]) -> Vec<f64> {
    if dims.is_empty() || data.len() <= 1 {
        return data.to_vec();
    }
    // Create an ndarray in C order (row-major) with given shape
    let arr = ArrayD::from_shape_vec(dims.to_vec(), data.to_vec())
        .expect("Shape mismatch in row_major_to_col_major");
    // Reverse axes to get column-major order, then collect
    let arr_f = arr.reversed_axes();
    arr_f.iter().copied().collect()
}

/// Convert column-major flat data to row-major flat data.
fn col_major_to_row_major_f64(data: &[f64], dims: &[usize]) -> Vec<f64> {
    if dims.is_empty() || data.len() <= 1 {
        return data.to_vec();
    }
    // Column-major data with dims [d0, d1, ..., d_{n-1}] means
    // it was stored with reversed dims in row-major perspective.
    let reversed_dims: Vec<usize> = dims.iter().rev().copied().collect();
    let arr = ArrayD::from_shape_vec(reversed_dims, data.to_vec())
        .expect("Shape mismatch in col_major_to_row_major");
    let arr_c = arr.reversed_axes();
    arr_c.iter().copied().collect()
}

/// Convert row-major flat data to column-major flat data (Complex64).
fn row_major_to_col_major_c64(data: &[Complex64], dims: &[usize]) -> Vec<Complex64> {
    if dims.is_empty() || data.len() <= 1 {
        return data.to_vec();
    }
    let arr = ArrayD::from_shape_vec(dims.to_vec(), data.to_vec())
        .expect("Shape mismatch in row_major_to_col_major_c64");
    let arr_f = arr.reversed_axes();
    arr_f.iter().copied().collect()
}

/// Convert column-major flat data to row-major flat data (Complex64).
fn col_major_to_row_major_c64(data: &[Complex64], dims: &[usize]) -> Vec<Complex64> {
    if dims.is_empty() || data.len() <= 1 {
        return data.to_vec();
    }
    let reversed_dims: Vec<usize> = dims.iter().rev().copied().collect();
    let arr = ArrayD::from_shape_vec(reversed_dims, data.to_vec())
        .expect("Shape mismatch in col_major_to_row_major_c64");
    let arr_c = arr.reversed_axes();
    arr_c.iter().copied().collect()
}

/// Write a [`TensorDynLen`] as an ITensors.jl `ITensor` to an HDF5 group.
///
/// Schema:
/// ```text
/// <group>/
///   @type = "ITensor"
///   @version = 1
///   inds/          (IndexSet)
///   storage/       (Dense{Float64} or Dense{ComplexF64})
/// ```
pub(crate) fn write_itensor(group: &Group, tensor: &TensorDynLen) -> Result<()> {
    // Type and version attributes
    let type_attr = group.new_attr::<VarLenUnicode>().shape(()).create("type")?;
    type_attr
        .as_writer()
        .write_scalar(&VarLenUnicode::from_str("ITensor")?)?;

    let version_attr = group.new_attr::<i64>().shape(()).create("version")?;
    version_attr.as_writer().write_scalar(&1i64)?;

    // Write indices
    let inds_group = group.create_group("inds")?;
    index::write_index_set(&inds_group, tensor.indices())?;

    // Write storage
    let storage_group = group.create_group("storage")?;
    let dims = tensor.dims();

    if tensor.is_f64() {
        let data = tensor.to_vec_f64().context("Failed to extract f64 data")?;
        let col_major_data = row_major_to_col_major_f64(&data, &dims);

        let type_attr = storage_group
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create("type")?;
        type_attr
            .as_writer()
            .write_scalar(&VarLenUnicode::from_str("Dense{Float64}")?)?;

        let version_attr = storage_group
            .new_attr::<i64>()
            .shape(())
            .create("version")?;
        version_attr.as_writer().write_scalar(&1i64)?;

        let data_ds = storage_group
            .new_dataset::<f64>()
            .shape([col_major_data.len()])
            .create("data")?;
        data_ds.as_writer().write(&col_major_data)?;
    } else if tensor.is_complex() {
        let data = tensor.to_vec_c64().context("Failed to extract c64 data")?;
        let col_major_data = row_major_to_col_major_c64(&data, &dims);

        let type_attr = storage_group
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create("type")?;
        type_attr
            .as_writer()
            .write_scalar(&VarLenUnicode::from_str("Dense{ComplexF64}")?)?;

        let version_attr = storage_group
            .new_attr::<i64>()
            .shape(())
            .create("version")?;
        version_attr.as_writer().write_scalar(&1i64)?;

        // Store as native HDF5 compound type (compatible with ITensors.jl)
        let data_ds = storage_group
            .new_dataset::<Complex64>()
            .shape([col_major_data.len()])
            .create("data")?;
        data_ds.as_writer().write(&col_major_data)?;
    } else {
        bail!("Unsupported storage type for HDF5 serialization");
    }

    Ok(())
}

/// Read a [`TensorDynLen`] from an ITensors.jl `ITensor` in an HDF5 group.
pub(crate) fn read_itensor(group: &Group) -> Result<TensorDynLen> {
    // Read indices
    let inds_group = group.group("inds")?;
    let indices = index::read_index_set(&inds_group)?;
    let dims: Vec<usize> = indices.iter().map(|idx| idx.dim).collect();

    // Read storage
    let storage_group = group.group("storage")?;
    let storage_type_str = crate::compat::read_string_attr_by_name(&storage_group, "type")?;

    if storage_type_str.contains("Dense{Float64}") {
        let data_ds = storage_group.dataset("data")?;
        let col_major_data: Vec<f64> = data_ds
            .as_reader()
            .read_1d()
            .context("Failed to read f64 data")?
            .to_vec();
        let row_major_data = col_major_to_row_major_f64(&col_major_data, &dims);
        Ok(TensorDynLen::from_dense_f64(indices, row_major_data))
    } else if storage_type_str.contains("Dense{ComplexF64}") {
        let data_ds = storage_group.dataset("data")?;
        // Read as native HDF5 compound type (Complex64)
        let col_major_data: Vec<Complex64> = data_ds
            .as_reader()
            .read_1d()
            .context("Failed to read complex data")?
            .to_vec();
        let row_major_data = col_major_to_row_major_c64(&col_major_data, &dims);
        Ok(TensorDynLen::from_dense_c64(indices, row_major_data))
    } else {
        bail!(
            "Unsupported storage type: {}. Only Dense{{Float64}} and Dense{{ComplexF64}} are supported.",
            storage_type_str
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_col_major_roundtrip_f64() {
        let dims = vec![2, 3];
        // Row-major: [[0,1,2],[3,4,5]] → flat: [0,1,2,3,4,5]
        let row_major = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let col_major = row_major_to_col_major_f64(&row_major, &dims);
        // Column-major: [0,3,1,4,2,5]
        assert_eq!(col_major, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

        let back = col_major_to_row_major_f64(&col_major, &dims);
        assert_eq!(back, row_major);
    }

    #[test]
    fn test_row_col_major_3d() {
        let dims = vec![2, 3, 4];
        let n: usize = dims.iter().product();
        let row_major: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let col_major = row_major_to_col_major_f64(&row_major, &dims);
        let back = col_major_to_row_major_f64(&col_major, &dims);
        assert_eq!(back, row_major);
    }

    #[test]
    fn test_scalar_tensor() {
        let data = vec![42.0];
        let dims: Vec<usize> = vec![];
        let col = row_major_to_col_major_f64(&data, &dims);
        assert_eq!(col, vec![42.0]);
    }
}
