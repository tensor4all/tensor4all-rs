# Provenance and Citation Policy

This document records which tensor4all-rs components build on which external
projects, which implemented algorithms originate in which publications, and
how we ask research users to cite them. We maintain it because modern
development, human or AI-assisted, makes it easy to write one codebase while
referencing another, and the scientific contribution of the referenced
projects deserves visible credit beyond what licenses require. When a new
component references external code, designs, or published algorithms, add it
here in the same PR.

## Citation policy

- Cite tensor4all-rs itself using [`CITATION.cff`](../CITATION.cff) (GitHub's
  "Cite this repository" button). A software paper is planned and will become
  the preferred citation once available.
- Additionally cite the original papers of the algorithms your work relies on
  (see "Algorithm origins" below), and the library papers of upstream
  projects whose designs the components you use are ported from or build on
  (see the component table below and the README's "How to Cite" section).

## Component provenance

This table records intellectual provenance (designs, algorithms, and code
lineage), not the Cargo dependency graph; see each crate's `Cargo.toml` for
actual dependencies. Relationship vocabulary:

- **Port**: reimplementation of a specific upstream library, following its
  public API and algorithms.
- **Derived (license)**: contains code translated closely enough from the
  upstream to carry its license terms.
- **Inspired**: data structures or API design modeled on the upstream;
  implementation independent.
- **Compatible**: interoperates with the upstream's conventions or file
  formats; validated against it.
- **Backend**: delegates execution to the upstream (in this one case the
  upstream is also an actual dependency).

| Component | Role | Design/code provenance | Relationship |
| --- | --- | --- | --- |
| `tensor4all-core` | Index system, dynamic-rank tensors, contraction, factorizations | [ITensors.jl](https://github.com/ITensor/ITensors.jl) | Inspired (Index semantics designed fully compatible with ITensors.jl) |
| `tensor4all-tensorbackend` | Scalars, storage, dense linear algebra | [tenferro-rs](https://github.com/tensor4all/tenferro-rs) | Backend |
| `tensor4all-simplett` | Simple TT/MPS with compression | — | Original |
| `tensor4all-itensorlike` | TensorTrain API with orthogonality tracking | [ITensors.jl](https://github.com/ITensor/ITensors.jl) / ITensorMPS.jl | Inspired |
| `tensor4all-treetn` | Tree tensor networks: canonicalization, DMRG, TDVP, linsolve, GSE | [ITensorNetworks.jl](https://github.com/ITensor/ITensorNetworks.jl), [NamedGraphs.jl](https://github.com/mtfishman/NamedGraphs.jl), KrylovKit.jl | Inspired (data structures); Derived (Apache-2.0) for TDVP sweep plans; Compatible (KrylovKit solver conventions) |
| `tensor4all-tcicore` | rrLU / MatrixLUCI / cross-interpolation infrastructure | [TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl) | Port |
| `tensor4all-tensorci` | Tensor cross interpolation (TCI1/TCI2) | [TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl) | Compatible (validated for parity) |
| `tensor4all-treetci` | Tree tensor cross interpolation | [TreeTCI.jl](https://github.com/tensor4all/TreeTCI.jl) by Ryo Watanabe | Port |
| `tensor4all-quanticstci` | Quantics TCI function interpolation | [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl) | Port |
| `tensor4all-interpolativeqtt` | Interpolative QTT construction | [InterpolativeQTT.jl](https://github.com/tensor4all/InterpolativeQTT.jl) | Port |
| `tensor4all-quanticstransform` | Quantics operators (shift, flip, QFT, affine) | [Quantics.jl](https://github.com/tensor4all/Quantics.jl) | Port (validated against v0.4.7) |
| `tensor4all-aci` | Alternating cross interpolation elementwise ops | [AlternatingCrossInterpolation.jl](https://github.com/tensor4all/AlternatingCrossInterpolation.jl) by Marc K. Ritter | Port |
| `tensor4all-partitionedtt` | Partitioned TT over subdomains | — | Original |
| `tensor4all-hdf5` | HDF5 serialization | [ITensors.jl](https://github.com/ITensor/ITensors.jl) / ITensorMPS.jl file formats | Compatible (round-trip validated) |
| `tensor4all-capi` | C FFI surface for language bindings | — | Original (consumed by [Tensor4all.jl](https://github.com/tensor4all/Tensor4all.jl)) |

## Algorithm origins

Library provenance above is distinct from the scientific origin of the
algorithms themselves. Where an implemented algorithm has an identifiable
original publication, research using that component should cite it. This
list is best-effort; corrections and additions are welcome.

| Algorithm | Component(s) | Original references |
| --- | --- | --- |
| DMRG | `treetn` | S. R. White, [Phys. Rev. Lett. 69, 2863 (1992)](https://doi.org/10.1103/PhysRevLett.69.2863); [Phys. Rev. B 48, 10345 (1993)](https://doi.org/10.1103/PhysRevB.48.10345) |
| TDVP | `treetn` | J. Haegeman et al., [Phys. Rev. Lett. 107, 070601 (2011)](https://doi.org/10.1103/PhysRevLett.107.070601); [Phys. Rev. B 94, 165116 (2016)](https://doi.org/10.1103/PhysRevB.94.165116) |
| Global subspace expansion (GSE) | `treetn` | M. Yang and S. R. White, [Phys. Rev. B 102, 094315 (2020)](https://doi.org/10.1103/PhysRevB.102.094315) |
| Tensor cross interpolation / TT-cross | `tcicore`, `tensorci`, `treetci` | I. Oseledets and E. Tyrtyshnikov, [Linear Algebra Appl. 432, 70 (2010)](https://doi.org/10.1016/j.laa.2009.07.024); Y. Núñez Fernández et al., [Phys. Rev. X 12, 041018 (2022)](https://doi.org/10.1103/PhysRevX.12.041018); [SciPost Phys. 18, 104 (2025)](https://doi.org/10.21468/SciPostPhys.18.3.104) |
| Quantics representation | `quanticstci`, `quanticstransform` | I. Oseledets, [SIAM J. Matrix Anal. Appl. 31, 2130 (2010)](https://doi.org/10.1137/090757861); B. Khoromskij, [Constr. Approx. 34, 257 (2011)](https://doi.org/10.1007/s00365-011-9131-1) |
| Affine transformations on quantics TT | `quanticstransform` | S. Rohshap, M. K. Ritter, H. Shinaoka, J. von Delft, M. Wallerberger, A. Kauch, [Phys. Rev. Research 7, 023087 (2025)](https://doi.org/10.1103/PhysRevResearch.7.023087), Appendix B |
| Quantum Fourier transform as low-rank MPO | `quanticstransform` | J. Chen and M. Lindsey, "Direct interpolative construction of the discrete Fourier transform as a matrix product operator", Appl. Comput. Harmon. Anal. (2025), [arXiv:2404.03182](https://arxiv.org/abs/2404.03182) |
| Interpolative QTT construction | `interpolativeqtt` | M. Lindsey, [arXiv:2311.12554](https://arxiv.org/abs/2311.12554) |
| Alternating cross interpolation (ACI) | `aci` | M. K. Ritter et al., [arXiv:2604.00037](https://arxiv.org/abs/2604.00037) |

License-bearing derivations are declared in the affected crate
(`tensor4all-treetn`: `LICENSE-APACHE`, SPDX `MIT AND Apache-2.0`).
Citation guidance for research use is in the README's "How to Cite" section
and `CITATION.cff`.
