# Component Provenance

This file records which tensor4all-rs components build on which external
projects, and the nature of each relationship. We maintain it because modern
development, human or AI-assisted, makes it easy to write one codebase while
referencing another, and the scientific contribution of the referenced
projects deserves visible credit beyond what licenses require. When a new
component references external code or designs, add it here in the same PR.

Relationship vocabulary:

- **Port**: reimplementation of a specific upstream library, following its
  public API and algorithms.
- **Derived (license)**: contains code translated closely enough from the
  upstream to carry its license terms.
- **Inspired**: data structures or API design modeled on the upstream;
  implementation independent.
- **Compatible**: interoperates with the upstream's conventions or file
  formats; validated against it.
- **Backend**: delegates execution to the upstream.

| Component | Role | Builds on | Relationship |
| --- | --- | --- | --- |
| `tensor4all-core` | Index system, dynamic-rank tensors, contraction, factorizations | [ITensors.jl](https://github.com/ITensor/ITensors.jl) | Inspired (Index semantics designed fully compatible with ITensors.jl) |
| `tensor4all-tensorbackend` | Scalars, storage, dense linear algebra | [tenferro-rs](https://github.com/tensor4all/tenferro-rs) | Backend |
| `tensor4all-simplett` | Simple TT/MPS with compression | — | Original |
| `tensor4all-itensorlike` | TensorTrain API with orthogonality tracking | [ITensors.jl](https://github.com/ITensor/ITensors.jl) / ITensorMPS.jl | Inspired |
| `tensor4all-treetn` | Tree tensor networks: canonicalization, DMRG, TDVP, linsolve, GSE | [ITensorNetworks.jl](https://github.com/ITensor/ITensorNetworks.jl), [NamedGraphs.jl](https://github.com/mtfishman/NamedGraphs.jl), KrylovKit.jl | Inspired (data structures); Derived (Apache-2.0) for TDVP sweep plans; Compatible (KrylovKit solver conventions) |
| `tensor4all-tcicore` | rrLU / MatrixLUCI / cross-interpolation infrastructure | TCI algorithm literature ([SciPost Phys. 18, 104 (2025)](https://doi.org/10.21468/SciPostPhys.18.3.104)) | Original implementation of published algorithms |
| `tensor4all-tensorci` | Tensor cross interpolation (TCI1/TCI2) | [TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl) | Compatible (validated for parity) |
| `tensor4all-treetci` | Tree tensor cross interpolation | [TreeTCI.jl](https://github.com/tensor4all/TreeTCI.jl) by Ryo Watanabe | Port |
| `tensor4all-quanticstci` | Quantics TCI function interpolation | [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl) | Port |
| `tensor4all-interpolativeqtt` | Interpolative QTT construction | [InterpolativeQTT.jl](https://github.com/tensor4all/InterpolativeQTT.jl) | Port |
| `tensor4all-quanticstransform` | Quantics operators (shift, flip, QFT, affine) | [Quantics.jl](https://github.com/tensor4all/Quantics.jl) | Port (validated against v0.4.7) |
| `tensor4all-aci` | Alternating cross interpolation elementwise ops | [AlternatingCrossInterpolation.jl](https://github.com/tensor4all/AlternatingCrossInterpolation.jl) by Marc K. Ritter | Port |
| `tensor4all-partitionedtt` | Partitioned TT over subdomains | — | Original |
| `tensor4all-hdf5` | HDF5 serialization | [ITensors.jl](https://github.com/ITensor/ITensors.jl) / ITensorMPS.jl file formats | Compatible (round-trip validated) |
| `tensor4all-capi` | C FFI surface for language bindings | — | Original (consumed by [Tensor4all.jl](https://github.com/tensor4all/Tensor4all.jl)) |

License-bearing derivations are declared in the affected crate
(`tensor4all-treetn`: `LICENSE-APACHE`, SPDX `MIT AND Apache-2.0`).
Citation guidance for research use is in the README's "How to Cite" section
and `CITATION.cff`.
