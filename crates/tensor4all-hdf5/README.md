# tensor4all-hdf5

HDF5 serialization for tensor4all-rs, compatible with ITensors.jl / ITensorMPS.jl file formats.

## Key Types

- `save_itensor()` / `load_itensor()` — read/write `TensorDynLen` as ITensors.jl `ITensor`
- `save_mps()` / `load_mps()` — read/write `TensorTrain` as ITensorMPS.jl `MPS`

## Feature Flags

- `link` (default) — compile-time HDF5 linking
- `runtime-loading` — dlopen for FFI environments

## Documentation

- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_hdf5/)
