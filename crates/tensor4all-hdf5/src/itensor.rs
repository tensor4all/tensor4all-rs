//! Layer 1: ITensor (TensorDynLen) HDF5 read/write (ITensors.jl compatible).

use anyhow::{bail, Context, Result};
use hdf5::Group;
use num_complex::Complex64;
use tensor4all_core::defaults::tensordynlen::TensorAccess;
use tensor4all_core::TensorDynLen;

use crate::index;
use crate::layout;
use crate::schema;

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
    schema::write_type_version(group, "ITensor", 1)?;

    // Write indices
    let inds_group = group.create_group("inds")?;
    index::write_index_set(&inds_group, tensor.indices())?;

    // Write storage
    let storage_group = group.create_group("storage")?;
    let dims = tensor.dims();

    if tensor.is_f64() {
        let data = tensor.to_vec_f64().context("Failed to extract f64 data")?;
        let col_major_data = layout::row_major_to_col_major(&data, &dims)?;

        schema::write_type_version(&storage_group, "Dense{Float64}", 1)?;

        let data_ds = storage_group
            .new_dataset::<f64>()
            .shape([col_major_data.len()])
            .create("data")?;
        data_ds.as_writer().write(&col_major_data)?;
    } else if tensor.is_complex() {
        let data = tensor.to_vec_c64().context("Failed to extract c64 data")?;
        let col_major_data = layout::row_major_to_col_major(&data, &dims)?;

        schema::write_type_version(&storage_group, "Dense{ComplexF64}", 1)?;

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
    schema::require_type_version(group, "ITensor", 1)?;

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
        let row_major_data = layout::col_major_to_row_major(&col_major_data, &dims)?;
        Ok(TensorDynLen::from_dense_f64(indices, row_major_data))
    } else if storage_type_str.contains("Dense{ComplexF64}") {
        let data_ds = storage_group.dataset("data")?;
        // Read as native HDF5 compound type (Complex64)
        let col_major_data: Vec<Complex64> = data_ds
            .as_reader()
            .read_1d()
            .context("Failed to read complex data")?
            .to_vec();
        let row_major_data = layout::col_major_to_row_major(&col_major_data, &dims)?;
        Ok(TensorDynLen::from_dense_c64(indices, row_major_data))
    } else {
        bail!(
            "Unsupported storage type: {}. Only Dense{{Float64}} and Dense{{ComplexF64}} are supported.",
            storage_type_str
        );
    }
}
