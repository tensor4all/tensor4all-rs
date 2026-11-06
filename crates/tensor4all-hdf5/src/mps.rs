//! Layer 2: MPS (TensorTrain) HDF5 read/write (ITensorMPS.jl compatible).
//!
//! MPS is simply metadata (length, llim, rlim) + a sequence of ITensors,
//! so this module is a thin wrapper around [`crate::itensor`].

use crate::backend::{self, Group};
use anyhow::{Context, Result};
use tensor4all_itensorlike::{CanonicalForm, TensorTrain};

use crate::index;
use crate::itensor;
use crate::schema;

fn read_i32(name: &'static str, value: i64, length: usize) -> Result<i32> {
    i32::try_from(value).with_context(|| {
        format!("HDF5 dataset {name} does not fit in i32, got {value} for MPS length {length}")
    })
}

/// Convert ITensorMPS.jl's 1-based `llim`/`rlim` to the 0-based orthogonality
/// center of a [`TensorTrain`].
///
/// ITensorMPS records the sites that may be non-orthogonal as `llim + 1..=rlim - 1`
/// (1-based), with `llim = 0`, `rlim = length + 1` when nothing is known. A
/// single such site at 1-based `c` (`llim = c - 1`, `rlim = c + 1`) is the
/// center `c - 1` here. Any other valid range (no center, or a range of several
/// sites) loads as no center, which claims no orthogonality.
///
/// `llim = -1` together with `rlim = length + 1` is a second spelling of "no
/// center" that some writers emit; it also loads as no center.
fn center_from_itensor_limits(llim: i32, rlim: i32, length: usize) -> Result<Option<usize>> {
    let length_i32 = i32::try_from(length).map_err(|_| {
        anyhow::anyhow!(
            "invalid HDF5 MPS orthogonality limits: llim={llim}, rlim={rlim}, length={length}; length does not fit in i32"
        )
    })?;
    let max_rlim = length_i32.checked_add(1).ok_or_else(|| {
        anyhow::anyhow!(
            "invalid HDF5 MPS orthogonality limits: llim={llim}, rlim={rlim}, length={length}; length + 1 overflows i32"
        )
    })?;
    // Some writers spell "no orthogonality center" as `llim = -1` with
    // `rlim = length + 1`; the fixed inputs of tensor4all-benchmark case 4
    // (release `vertex-bse-inputs-v1`) do. ITensorMPS.jl writes `llim = 0` for
    // the same state, so accept both spellings and claim no center.
    if llim == -1 && rlim == max_rlim {
        return Ok(None);
    }
    if !(0 <= llim && llim < rlim && rlim <= max_rlim) {
        anyhow::bail!(
            "invalid HDF5 MPS orthogonality limits: llim={llim}, rlim={rlim}, length={length}; expected 0 <= llim < rlim <= {max_rlim} (ITensorMPS 1-based limits)"
        );
    }
    // `rlim <= length + 1` makes the center `llim` at most `length - 1`.
    Ok((llim.checked_add(2) == Some(rlim)).then_some(llim as usize))
}

/// ITensorMPS.jl's 1-based `(llim, rlim)` for a [`TensorTrain`]: the center at
/// 0-based `c` is the 1-based site `c + 1`, so `(c, c + 2)`; no center is
/// `(0, length + 1)`.
fn itensor_limits(tt: &TensorTrain) -> (i64, i64) {
    match tt.ortho_center() {
        Some(center) => (center as i64, center as i64 + 2),
        None => (0, tt.len() as i64 + 1),
    }
}

const CANONICAL_FORM_ATTR: &str = "canonical_form";

/// Write the optional `canonical_form` Int32 attribute (absent for `None`).
pub(crate) fn write_canonical_form(group: &Group, form: Option<CanonicalForm>) -> Result<()> {
    if let Some(form) = form {
        let canonical_form_attr = group
            .new_attr::<i32>()
            .shape(())
            .create(CANONICAL_FORM_ATTR)?;
        canonical_form_attr
            .as_writer()
            .write_scalar(&form.to_i32())?;
    }

    Ok(())
}

/// Read the optional `canonical_form` Int32 attribute (`None` when absent).
pub(crate) fn read_canonical_form(group: &Group) -> Result<Option<CanonicalForm>> {
    if !backend::attribute_exists(group, CANONICAL_FORM_ATTR)
        .context("Failed to check MPS canonical_form attribute existence")?
    {
        return Ok(None);
    }

    let attr = group
        .attr(CANONICAL_FORM_ATTR)
        .context("Failed to open MPS canonical_form attribute")?;
    let value: i32 = attr
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS canonical_form")?;

    CanonicalForm::from_i32(value)
        .ok_or_else(|| anyhow::anyhow!("Invalid MPS canonical_form value: {}", value))
        .map(Some)
}

/// Write a [`TensorTrain`] as an ITensorMPS.jl `MPS` to an HDF5 group.
///
/// Site tensors are stored as 1-indexed subgroups (`MPS[1]`, `MPS[2]`, ...),
/// following the Julia convention. Each site tensor is written using
/// [`crate::itensor::write_itensor`].
///
/// The `canonical_form` attribute is a tensor4all-rs extension not present in
/// the ITensorMPS.jl schema. It is silently ignored when reading files that
/// lack it.
///
/// # HDF5 Schema
///
/// ```text
/// <group>/
///   @type = "MPS"
///   @version = 1
///   length: Int64
///   llim: Int64   (ITensorMPS 1-based: center c (0-based) is written as c)
///   rlim: Int64   (c + 2; no center is llim = 0, rlim = length + 1)
///   @canonical_form: Int32?  (tensor4all-rs extension; absent for non-canonical MPS)
///   MPS[1]/ ...   (ITensor, 1-indexed)
///   MPS[2]/ ...
/// ```
pub(crate) fn write_mps(group: &Group, tt: &TensorTrain) -> Result<()> {
    schema::write_type_version(group, "MPS", 1)?;

    // Metadata datasets
    let length = tt.len() as i64;
    let length_ds = group.new_dataset::<i64>().shape(()).create("length")?;
    length_ds.as_writer().write_scalar(&length)?;

    // ITensorMPS.jl's limits are 1-based; see `itensor_limits`.
    let (llim, rlim) = itensor_limits(tt);
    let llim_ds = group.new_dataset::<i64>().shape(()).create("llim")?;
    llim_ds.as_writer().write_scalar(&llim)?;

    let rlim_ds = group.new_dataset::<i64>().shape(()).create("rlim")?;
    rlim_ds.as_writer().write_scalar(&rlim)?;

    write_canonical_form(group, tt.canonical_form())?;

    // Write each tensor (1-indexed, Julia convention)
    let tensors = tt.tensors();
    for (i, tensor) in tensors.iter().enumerate() {
        let name = format!("MPS[{}]", i + 1);
        let tensor_group = group.create_group(&name)?;
        itensor::write_itensor(&tensor_group, tensor)?;
    }

    Ok(())
}

/// Read a [`TensorTrain`] from an ITensorMPS.jl `MPS` in an HDF5 group.
///
/// Validates the `@type` and `@version` attributes, then reads site tensors
/// from 1-indexed subgroups. The `canonical_form` attribute is read if present
/// (tensor4all-rs extension), otherwise defaults to `None`.
pub(crate) fn read_mps(group: &Group) -> Result<TensorTrain> {
    schema::require_type_version(group, "MPS", 1)?;

    let length: i64 = group
        .dataset("length")?
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS length")?;
    let length = index::read_nonnegative_usize("length", length)?;

    let llim: i64 = group
        .dataset("llim")?
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS llim")?;
    let llim = read_i32("llim", llim, length)?;

    let rlim: i64 = group
        .dataset("rlim")?
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS rlim")?;
    let rlim = read_i32("rlim", rlim, length)?;
    let center = center_from_itensor_limits(llim, rlim, length)?;
    let canonical_form = read_canonical_form(group)?;

    index::validate_expected_child_groups(
        group,
        length,
        "MPS",
        "MPS[",
        Some(']'),
        &["length", "llim", "rlim"],
    )?;
    let mut tensors = Vec::with_capacity(length);
    for ordinal in 1..=length {
        let name = index::expected_child_name("MPS[", ordinal, Some(']'));
        let tensor_group = group
            .group(&name)
            .with_context(|| format!("Failed to open HDF5 MPS child group {name}"))?;
        tensors.push(itensor::read_itensor(&tensor_group)?);
    }

    // `with_ortho` takes TensorTrain's 0-based limits: center `c` is
    // `(c - 1, c + 1)`, no center is `(-1, length + 1)` (see `TensorTrain::llim`).
    let (llim, rlim) = match center {
        Some(center) => (center as i32 - 1, center as i32 + 1),
        None => (-1, length as i32 + 1),
    };
    Ok(TensorTrain::with_ortho(
        tensors,
        llim,
        rlim,
        canonical_form,
    )?)
}
