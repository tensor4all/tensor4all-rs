//! Layer 0: Index / TagSet HDF5 read/write (ITensors.jl compatible).

use anyhow::{Context, Result};
use hdf5::types::VarLenUnicode;
use hdf5::Group;
use std::str::FromStr;
use tensor4all_core::index::{DynId, DynIndex, Index, TagSet};
use tensor4all_core::tagset::TagSetLike;
use tensor4all_core::ConjState;

/// Convert a [`ConjState`] to ITensors.jl `Arrow` integer.
fn conj_state_to_dir(state: ConjState) -> i64 {
    match state {
        ConjState::Undirected => 0,
        ConjState::Ket => -1, // Arrow.In
        ConjState::Bra => 1,  // Arrow.Out
    }
}

/// Convert an ITensors.jl `Arrow` integer to [`ConjState`].
fn dir_to_conj_state(dir: i64) -> ConjState {
    match dir {
        -1 => ConjState::Ket,
        1 => ConjState::Bra,
        _ => ConjState::Undirected,
    }
}

/// Convert a [`TagSet`] to a comma-separated string (ITensors.jl format).
fn tagset_to_string(tags: &TagSet) -> String {
    let tag_strs: Vec<String> = TagSetLike::iter(tags).map(|s| s.to_string()).collect();
    tag_strs.join(",")
}

/// Write a TagSet to an HDF5 group (ITensors.jl compatible).
///
/// Schema:
/// ```text
/// <group>/
///   @type = "TagSet"
///   @version = 1
///   tags: String  (comma-separated)
/// ```
pub(crate) fn write_tagset(group: &Group, tags: &TagSet) -> Result<()> {
    let type_attr = group.new_attr::<VarLenUnicode>().shape(()).create("type")?;
    type_attr
        .as_writer()
        .write_scalar(&VarLenUnicode::from_str("TagSet")?)?;

    let version_attr = group.new_attr::<i64>().shape(()).create("version")?;
    version_attr.as_writer().write_scalar(&1i64)?;

    let tag_string = tagset_to_string(tags);
    let ds = group
        .new_dataset::<VarLenUnicode>()
        .shape(())
        .create("tags")?;
    ds.as_writer()
        .write_scalar(&VarLenUnicode::from_str(&tag_string)?)?;

    Ok(())
}

/// Read a TagSet from an HDF5 group.
pub(crate) fn read_tagset(group: &Group) -> Result<TagSet> {
    let ds = group.dataset("tags")?;
    let s = crate::compat::read_string_dataset(&ds)?;
    if s.is_empty() {
        Ok(TagSet::new())
    } else {
        TagSet::from_str(&s)
            .map_err(|e| anyhow::anyhow!("Failed to parse TagSet from HDF5: {:?}", e))
    }
}

/// Write a DynIndex to an HDF5 group (ITensors.jl compatible).
///
/// Schema:
/// ```text
/// <group>/
///   @type = "Index"
///   @version = 1
///   @space_type = "Int"
///   id: UInt64
///   dim: Int64
///   dir: Int64       (In=-1, Out=1, Neither=0)
///   plev: Int64      (always 0)
///   tags/            (TagSet group)
/// ```
pub(crate) fn write_index(group: &Group, index: &DynIndex) -> Result<()> {
    // Attributes
    let type_attr = group.new_attr::<VarLenUnicode>().shape(()).create("type")?;
    type_attr
        .as_writer()
        .write_scalar(&VarLenUnicode::from_str("Index")?)?;

    let version_attr = group.new_attr::<i64>().shape(()).create("version")?;
    version_attr.as_writer().write_scalar(&1i64)?;

    let space_type_attr = group
        .new_attr::<VarLenUnicode>()
        .shape(())
        .create("space_type")?;
    space_type_attr
        .as_writer()
        .write_scalar(&VarLenUnicode::from_str("Int")?)?;

    // Datasets
    let id_ds = group.new_dataset::<u64>().shape(()).create("id")?;
    id_ds.as_writer().write_scalar(&index.id.0)?;

    let dim_ds = group.new_dataset::<i64>().shape(()).create("dim")?;
    dim_ds.as_writer().write_scalar(&(index.dim as i64))?;

    let dir = conj_state_to_dir(index.conj_state());
    let dir_ds = group.new_dataset::<i64>().shape(()).create("dir")?;
    dir_ds.as_writer().write_scalar(&dir)?;

    let plev_ds = group.new_dataset::<i64>().shape(()).create("plev")?;
    plev_ds.as_writer().write_scalar(&0i64)?;

    // Tags subgroup
    let tags_group = group.create_group("tags")?;
    write_tagset(&tags_group, &index.tags)?;

    Ok(())
}

/// Read a DynIndex from an HDF5 group.
pub(crate) fn read_index(group: &Group) -> Result<DynIndex> {
    let id: u64 = group
        .dataset("id")?
        .as_reader()
        .read_scalar()
        .context("Failed to read index id")?;

    let dim: i64 = group
        .dataset("dim")?
        .as_reader()
        .read_scalar()
        .context("Failed to read index dim")?;

    let _dir: i64 = group
        .dataset("dir")?
        .as_reader()
        .read_scalar()
        .context("Failed to read index dir")?;

    // plev is read but ignored (always 0 in tensor4all-rs)
    let _plev: i64 = group
        .dataset("plev")?
        .as_reader()
        .read_scalar()
        .context("Failed to read index plev")?;

    let tags_group = group.group("tags")?;
    let tags = read_tagset(&tags_group)?;

    // Note: dir_to_conj_state(_dir) would give us ConjState, but DynIndex
    // is always Undirected. We store dir for round-trip fidelity but
    // DynIndex doesn't use it internally.
    let _conj_state = dir_to_conj_state(_dir);

    Ok(Index::new_with_tags(DynId(id), dim as usize, tags))
}

/// Write an IndexSet to an HDF5 group (ITensors.jl compatible).
///
/// Schema:
/// ```text
/// <group>/
///   @type = "IndexSet"
///   @version = 1
///   length: Int64
///   index_1/ ...
///   index_2/ ...
/// ```
pub(crate) fn write_index_set(group: &Group, indices: &[DynIndex]) -> Result<()> {
    let type_attr = group.new_attr::<VarLenUnicode>().shape(()).create("type")?;
    type_attr
        .as_writer()
        .write_scalar(&VarLenUnicode::from_str("IndexSet")?)?;

    let version_attr = group.new_attr::<i64>().shape(()).create("version")?;
    version_attr.as_writer().write_scalar(&1i64)?;

    let length_ds = group.new_dataset::<i64>().shape(()).create("length")?;
    length_ds
        .as_writer()
        .write_scalar(&(indices.len() as i64))?;

    for (i, index) in indices.iter().enumerate() {
        let name = format!("index_{}", i + 1); // 1-indexed
        let index_group = group.create_group(&name)?;
        write_index(&index_group, index)?;
    }

    Ok(())
}

/// Read an IndexSet from an HDF5 group.
pub(crate) fn read_index_set(group: &Group) -> Result<Vec<DynIndex>> {
    let length: i64 = group
        .dataset("length")?
        .as_reader()
        .read_scalar()
        .context("Failed to read IndexSet length")?;

    let mut indices = Vec::with_capacity(length as usize);
    for i in 0..length {
        let name = format!("index_{}", i + 1);
        let index_group = group.group(&name)?;
        indices.push(read_index(&index_group)?);
    }

    Ok(indices)
}

// conj_state_to_dir and dir_to_conj_state are used in index module only
// but kept accessible for potential future use via ConjState field on DynIndex.

use tensor4all_core::index_like::IndexLike;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conj_state_dir_roundtrip() {
        assert_eq!(
            dir_to_conj_state(conj_state_to_dir(ConjState::Undirected)),
            ConjState::Undirected
        );
        assert_eq!(
            dir_to_conj_state(conj_state_to_dir(ConjState::Ket)),
            ConjState::Ket
        );
        assert_eq!(
            dir_to_conj_state(conj_state_to_dir(ConjState::Bra)),
            ConjState::Bra
        );
    }

    #[test]
    fn test_tagset_to_string() {
        let tags = TagSet::from_str("Site,n=1").unwrap();
        let s = tagset_to_string(&tags);
        // Tags are sorted, so order may differ
        assert!(s.contains("Site"));
        assert!(s.contains("n=1"));
    }
}
