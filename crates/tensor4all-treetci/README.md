# tensor4all-treetci

Tree Tensor Cross Interpolation for `tensor4all-rs`.

This crate is intended as a Rust port of
[TreeTCI.jl](https://github.com/tensor4all/TreeTCI.jl). The upstream Julia
package currently lists:

- Ryo Watanabe <https://github.com/Ryo-wtnb11>

The Rust port should preserve that attribution explicitly.

Tree-specific pivot updates in this crate use `matrixluci` as the low-level
pivot-selection substrate. `matrixluci` is tensor4all-owned code, with a dense
path informed by a port of `faer` full-pivoting LU ideas and primitives.
