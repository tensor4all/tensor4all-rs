# tensor4all-hdf5

HDF5 serialization for tensor4all-rs, compatible with ITensors.jl / ITensorMPS.jl file formats.

**Thread safety.** The HDF5 C library is not thread-safe. All public
`save_*` / `append_*` / `load_*` calls are serialized through one
process-wide lock, so the crate is safe to call concurrently by construction
(even on distinct files). The crate never touches the process environment. HDF5 OS file locking is
**your** choice, made before the first HDF5 call in the process: set
`HDF5_USE_FILE_LOCKING=FALSE` if you re-open a just-written path in the same
process (otherwise the OS lock can outlive `H5Fclose` and the reopen can fail
with `errno = 35`); leave it unset to keep cross-process write protection. See
the crate docs, "File locking". Direct use of the re-exported low-level hdf5-rt passthroughs bypasses the
crate lock.

## Key Types

- `save_itensor()` / `load_itensor()` — read/write `IdxTensor` as ITensors.jl `ITensor`
- `save_mps()` / `load_mps()` — read/write `TensorTrain` as ITensorMPS.jl `MPS`
- `save_treetn()` / `append_treetn()` / `load_treetn()` — read/write a general
  `TreeTN<IdxTensor, V>` (`String` or `usize` node names) using the
  tensor4all-rs `TreeTN` v1 schema: one `node_i/` ITensor subgroup per node
  with a `node_name` attribute; topology is recovered on load from shared
  index identity via `TreeTN::from_tensors`

## TreeTN schema (v1)

```text
<group>/
  @type = "TreeTN"
  @version = 1
  node_count: Int64
  node_1/ ... node_N/
    @node_name: VarLenUnicode
    (ITensor — same per-node schema as save_itensor)
```

Edges are not stored explicitly: bond connections are recovered on load from
shared `Index` identity, exactly as the tree was assembled originally.

## Feature Flags

- `link` (default) — compile-time HDF5 linking
- `runtime-loading` — dlopen for FFI environments

## Documentation

- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_hdf5/)
