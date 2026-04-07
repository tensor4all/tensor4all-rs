# Concepts

This page introduces the core ideas behind tensor4all-rs in plain language.
No deep mathematical background is assumed.

---

## Tensor Train (TT / MPS)

A *tensor train* (TT) represents a high-dimensional tensor as a chain of
smaller, 3-index tensors.  Instead of storing all entries of a size-`d^N`
tensor (which grows exponentially with `N`), a TT stores `N` tensors each of
modest size, multiplied together to reproduce any entry on demand.  The size of
the shared "bond" indices between adjacent tensors is called the *bond
dimension* `m`; larger `m` means a more accurate approximation at the cost of
more memory and compute.  In quantum physics the same structure is called a
*Matrix Product State* (MPS).

```
A[0] ---- A[1] ---- A[2] ---- ... ---- A[N-1]
  |          |          |                  |
 i_0        i_1        i_2             i_{N-1}
```

Vertical lines are *site indices* (also called physical indices) — one per
tensor, labeling the dimension of the original tensor at that position.
Horizontal lines are *link indices* (bond indices) — internal indices that
connect neighboring tensors and whose size is the bond dimension.

---

## Tensor Cross Interpolation (TCI)

*Tensor Cross Interpolation* approximates a high-dimensional function
`f(i_0, i_1, ..., i_{N-1})` by evaluating it at an adaptively chosen subset of
points and fitting a tensor train.  The idea generalises the CUR matrix
decomposition — which approximates a matrix using selected rows and columns —
to arbitrary numbers of dimensions.  The algorithm automatically identifies the
most informative "pivots" (evaluation points), so it never needs to query the
function at every grid point.  For smooth or structured functions, the number
of required evaluations can be orders of magnitude smaller than the full grid.
The output is a tensor train whose accuracy is controlled by a relative
tolerance threshold (`rtol`).

---

## Quantics Tensor Train (QTT)

*Quantics* (also called *quantized tensor train*) is a technique for
representing functions of continuous variables with a tensor train.  A function
`f(x)` sampled on a uniform grid of `2^R` points in `[0, 1)` is reshaped into
an `R`-site tensor train where every site index has dimension 2.  Each site
corresponds to one bit of the binary representation of the grid index.  Smooth
functions have low bond dimension in this representation, giving exponential
compression relative to storing all `2^R` values.  Combining QTT with TCI
("Quantics TCI") allows efficient approximation of high-dimensional continuous
integrands without ever forming the full grid.

```
bit R-1        bit R-2                bit 0
  |              |          ...          |
 B[0] --------- B[1] ----- ... ------- B[R-1]
```

Each tensor `B[k]` has two physical legs of dimension 2 (one per variable
dimension in the multivariate case) and one or two bond legs.

---

## Tree Tensor Network (TreeTN)

A *Tree Tensor Network* generalises the tensor train to tree-shaped graphs.
Each node of the tree holds a tensor, and each edge corresponds to a shared
(bond) index between the two tensors at its endpoints.  Contracting along any
edge multiplies those two tensors and merges their free indices.  The tensor
train is the special case where the tree is a simple path graph.  Tree
structures can capture correlations that are not naturally captured by a chain,
making them useful for problems with hierarchical or multi-scale structure.  In
tensor4all-rs, `TreeTensorNetwork<V>` (where `V` is a vertex label type) is the
primary data structure representing both TT/MPS and more general tree networks.

```
         T[root]
        /        \
    T[a]          T[b]
    /   \            \
T[c]   T[d]         T[e]
  |      |             |
 i_c    i_d           i_e
```

Vertical lines are site indices; lines along the tree edges are bond indices.

---

## Key Terminology

| Term | Meaning |
|------|---------|
| **Bond dimension** | The size of a link (bond) index shared between two adjacent tensors.  Controls the accuracy vs. cost trade-off. |
| **Site index** | A physical or external index at a single tensor site, corresponding to one degree of freedom of the original tensor. |
| **Link index** | An internal index connecting two neighbouring tensors; its size is the bond dimension. |
| **Truncation tolerance (`rtol`)** | Relative error threshold used during SVD-based compression.  Singular values smaller than `rtol * sigma_max` are discarded. |
| **MPS / MPO** | Matrix Product State / Matrix Product Operator — physics names for TT with one or two site indices per tensor, respectively. |
| **Pivot** | In TCI, a selected multi-index at which the function is evaluated to refine the tensor train approximation. |
