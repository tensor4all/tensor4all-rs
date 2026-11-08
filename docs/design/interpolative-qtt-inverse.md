# Interpolative QTT inverse

[Issue #1175](https://github.com/tensor4all/tensor4all-rs/issues/1175) selects
a separate exact entry point for uncompressed single-scale trains.

`invert_qtt` keeps its gauge-independent two-sample reconstruction, including
on compressed trains. `invert_qtt_uncompressed` requires a single-scale train
built with `tolerance = 0.0` and `max_bond_dim = None`, with no later compression
or change of bond basis. Both return the same multiresolution matrix layout
and share the coarse-level restriction.

In the uncompressed basis, contracting all but the last core already gives
the parent polynomial's node values. The exact inverse returns these values
directly. It checks the supplied nodes against the canonical construction grid,
then checks the final bond dimension and final core's samples against the
canonical construction basis before contraction. A supplied basis rebuilt from
identical nodes is accepted even if its computed weights round differently.
These checks reject common basis/gauge mismatches; they do not prove
construction history.

Automatic selection from bond dimensions or the final core is insufficient:
`SimpleTensorTrain` records no compression provenance and permits direct core
mutation. A gauge change can preserve bond dimensions, and a change acting in
the final sampling map's nullspace can preserve that map while changing the
parent polynomial. Adding provenance to the general train type would extend
well beyond inverse interpolation. The explicit entry point leaves that
precondition with callers without changing the general train representation.

Exact recovery means no additional two-sample blending error, up to floating-
point rounding. It does not remove interpolation or coarse restriction error.
Small-grid polynomial tests cover every finest node, with separate compressed
round-trip tests preserving the general decoder's contract.
