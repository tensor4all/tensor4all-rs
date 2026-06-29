# Chain MPS Global Subspace Expansion

This note explains the chain/MPS global subspace expansion (GSE) algorithm used
by RydbergToolkit.jl before translating it to TreeTN GSE-TDVP.

The goal is not to define the final tensor4all-rs API. The goal is to make the
chain algorithm explicit enough that an MPS reader can see exactly what bond
space is enlarged, why the original state is preserved, and which details are
nontrivial when moving from a chain to a general tree.

## Source Snapshot

This analysis is based on CodingThrust/RydbergToolkit.jl at commit
`9896452ce9f9164f67f761c88b64583aacd72ff9`.

Related tensor4all-rs planning issue:
[#539](https://github.com/tensor4all/tensor4all-rs/issues/539).

Relevant files:

| File | Role |
|---|---|
| [`src/gsetdvp/krylov.jl`](https://github.com/CodingThrust/RydbergToolkit.jl/blob/9896452ce9f9164f67f761c88b64583aacd72ff9/src/gsetdvp/krylov.jl) | Builds the global Krylov reference MPS list. |
| [`src/gsetdvp/expand.jl`](https://github.com/CodingThrust/RydbergToolkit.jl/blob/9896452ce9f9164f67f761c88b64583aacd72ff9/src/gsetdvp/expand.jl) | Expands each MPS bond from the reference states. |
| [`src/gsetdvp/tdvp.jl`](https://github.com/CodingThrust/RydbergToolkit.jl/blob/9896452ce9f9164f67f761c88b64583aacd72ff9/src/gsetdvp/tdvp.jl) | Runs TDVP and inserts GSE between TDVP steps when `krylov_dim > 0`. |
| [`test/gsetdvp/krylov.jl`](https://github.com/CodingThrust/RydbergToolkit.jl/blob/9896452ce9f9164f67f761c88b64583aacd72ff9/test/gsetdvp/krylov.jl) | Compares GSE references and expansion with ITensorMPS. |
| [`test/gsetdvp/expand.jl`](https://github.com/CodingThrust/RydbergToolkit.jl/blob/9896452ce9f9164f67f761c88b64583aacd72ff9/test/gsetdvp/expand.jl) | Checks state preservation, bond growth, and final orthogonality center. |

The source comment in `expand.jl` cites Yang and White, "Time-dependent
variational principle with ancillary Krylov subspace", Phys. Rev. B 102, 094315
(2020).

## Two Different Krylov Uses

There are two Krylov-related pieces that should not be conflated.

1. Local TDVP exponential: TDVP evolves a one-site or two-site local tensor by
   applying `exp(dt H_eff)` in a small effective space. RydbergToolkit can use
   KrylovKit or ExponentialUtilities for this local `expmv`. tensor4all-rs
   already has Hermitian Krylov `expmv` machinery used by the existing TreeTN
   TDVP.
2. Global subspace expansion: GSE builds extra MPS reference states such as
   `H psi`, `H^2 psi`, ... and uses them only to enlarge the variational bond
   spaces before TDVP continues.

This document is about the second item: how the bond basis is enlarged.

## MPS Convention

RydbergToolkit stores an MPS tensor as

`A[j][left_bond, physical, right_bond]`.

For a chain of `N` sites:

```text
A[1] -- A[2] -- ... -- A[j-1] == A[j] -- ... -- A[N]
                         ^
                         bond expanded by the j-th sweep step
```

The expansion sweep canonicalizes the target state and all reference states to
site `N`, then sweeps from `j = N` down to `j = 2`. At the step for site `j`,
the bond between sites `j-1` and `j` is enlarged. Equivalently, the basis for
the right block `j..N` seen across that cut is enlarged.

Let

`A[j]` have shape `(chi_left, d_j, chi_right)`.

At the `j` step the tensor is reshaped to a matrix

```text
M_j[alpha, (s_j, beta)] = A[j][alpha, s_j, beta]

rows:    alpha in the left bond space of site j
columns: local basis for site j times the already processed right block
```

The SVD of `M_j` gives the current row basis for the right block. In the
RydbergToolkit helper, `truncated_svd` returns `Vt`, so the current basis is a
matrix `B_j` whose rows are orthonormal basis vectors in the column space
`C^(d_j * chi_right)`.

## Reference State Generation

`global_krylov_subspace(psi, H; krylovdim)` builds reference states by repeated
MPO application:

| Reference index | State represented by the code |
|---|---|
| `1` | normalized compressed `H psi` |
| `2` | normalized compressed `H (H psi)` |
| `k` | normalized compressed `H^k psi` |

The implementation initializes `cur_reference = psi`, then repeatedly applies
`H` to the current reference. The original `psi` itself is not inserted into
the `references` vector.

Each MPO application uses full compression with `maxdim = maxlinkdim(psi) + 1`
and `atol = 1e-13` by default, then normalizes the result. The `+1` cap is an
important part of this particular implementation: the references are not meant
to be unrestricted high-rank Krylov vectors. They are low-rank probes carrying
directions just outside the current MPS manifold.

## Per-Bond Expansion Step

For a fixed site `j`, RydbergToolkit performs the following operations.

### 1. Extract the Current Basis

Reshape the target tensor:

`M_j = reshape(A[j], chi_left, d_j * chi_right)`.

Compute a truncated SVD:

`M_j = U S B_j`.

Here `B_j` is the returned `Vt` block. Its rows form the current right-block
basis for the bond `(j-1, j)`. With nonzero `atol`, numerically small old
directions may be dropped before new directions are added.

### 2. Build a Reference Density Matrix

For each reference MPS `Phi_r`, reshape its current tensor at site `j` in the
same way:

`R_r = reshape(Phi_r[j], left_ref, d_j * chi_right)`.

Then accumulate

`rho_j = sum_r R_r^* R_r`.

This traces out the left/environment side of each reference and leaves a
density matrix on the same local right-block space where `B_j` lives:

```text
left side traced out
        |
        v
R_r^* R_r  acts on  (site j) x (right block basis)
```

RydbergToolkit normalizes `rho_j` by its trace before projection. That makes the
absolute tolerance less sensitive to the norm and number of reference states.

### 3. Remove the Already Represented Directions

The intended projector is the projector onto the complement of the row span of
`B_j`:

`P_j = I - projector(rowspan(B_j))`.

The projected density matrix is

`rho_missing = P_j rho_j P_j`.

This matrix contains only the reference-state weight that cannot already be
represented by the current bond basis.

Implementation detail: the Julia code stores basis vectors as rows from `Vt`
and forms the projector with `transpose(B_j) * conj(B_j)`. Several nearby
comments in `expand.jl` mark conjugation and index order as items to verify.
With a conventional column-vector bra/ket convention, the row-space projector
would usually be written as `adjoint(B_j) * B_j`. A Rust translation should make
the chosen convention explicit and test complex-valued MPS cases, rather than
copying the expression without checking the bra/ket orientation.

### 4. Diagonalize the Missing Density

If `norm(rho_missing) > atol`, RydbergToolkit diagonalizes the Hermitian matrix
`rho_missing` and keeps eigenvectors whose eigenvalues exceed `atol`:

`C_j = significant eigenvectors of rho_missing`.

Since Julia's `eigen` returns eigenvectors as columns, the code transposes them
before appending them to the row-basis matrix:

`E_j = stack_rows(B_j, C_j)`.

If `rho_missing` is negligible, no new directions are added and `E_j = B_j`.

The new bond dimension after this step is

`rank(B_j) + number_of_kept_missing_directions`.

There is no explicit maximum expansion dimension in `expand.jl`; the only
filter is the eigenvalue tolerance.

### 5. Replace the Right Tensor and Absorb Coefficients Left

Reshape the expanded basis matrix:

`E_j -> E_tensor[j]` with shape `(new_chi_left, d_j, chi_right)`.

Then replace the two tensors around the cut by

```text
old:  A[j-1] -- A[j]
new:  A_new[j-1] -- E_tensor[j]
```

where `A_new[j-1]` is obtained by contracting the old two-site block with the
conjugate expanded basis on site `j` and the right bond.

Conceptually:

```text
two-site block T = contract(A[j-1], A[j])

A_new[j-1][..., ell] = <E_j[ell] | T>
A_new[j]             = E_j
```

After this, the orthogonality center has moved from `j` to `j-1`.

## Why the Target State Is Preserved

The target state is not intentionally changed by GSE. Only its allowed bond
space is enlarged.

Before expansion, the right tensor `A[j]` lies completely in the row span of
`B_j`:

`A[j] = coefficients * B_j`.

After expansion, `E_j` contains all rows of `B_j` plus extra rows from the
projected reference density. Therefore the old tensor can be represented in the
larger basis exactly, up to SVD/eigensolver tolerance:

```text
old right basis:       B_j
expanded right basis:  [ B_j ]
                       [ C_j ]

old state coefficients in C_j directions: zero
```

The left tensor `A_new[j-1]` receives those coefficients. The added basis
vectors therefore create empty variational directions for `psi`; they do not
inject the reference states into `psi` immediately. Subsequent TDVP updates can
then place amplitude in those new directions.

This is why RydbergToolkit tests check that the overlap between the original
and expanded target MPS is approximately one, while the maximum link dimension
increases and the final orthogonality center is site `1`.

## What Happens to Reference States During the Sweep

The reference states are mutable work buffers. At each cut, each reference is
projected into the same expanded basis `E_j` so that the next cut to the left
can build a density matrix in a compatible gauge.

For the next iteration, the important tensor is the updated
`reference.data[j - 1]`: it contains the reference coefficients in the expanded
right-block basis. The code also writes a tail tensor at `reference.data[j]`,
but that line is marked with an index-order FIXME. Since the sweep never needs
that already processed tensor again for density construction, a translation
should treat the references as internal moving-center work buffers. If a Rust
API returns expanded references, it should add separate validity tests for
their tensor order and canonical form.

## TDVP Integration in RydbergToolkit

In `tdvp.jl`, GSE is enabled when `krylov_dim > 0`. The implementation:

1. Performs an initial TDVP step.
2. For each later time step, builds global Krylov references from the current
   MPS.
3. Calls `expand!` on the current MPS.
4. Normalizes the expanded MPS if requested.
5. Runs the next TDVP step, rebuilding environments instead of reusing the old
   TDVP cache.

The cache rebuild is essential: every expanded bond changes tensor shapes and
invalidates projected-operator environments.

## Mental Model

GSE is a density-matrix basis-completion sweep.

```text
Current TDVP state psi:

    ... -- A[j-1] == A[j] -- ...
              chi_old

Krylov references H psi, H^2 psi, ... show useful missing directions:

    rho_ref on (site j) x (right block)
       |
       | remove directions already in psi basis
       v
    rho_missing
       |
       | keep dominant eigenvectors
       v
    extra basis rows C_j

Expanded state:

    ... -- A_new[j-1] == E[j] -- ...
              chi_old + chi_extra
```

The state vector represented by `psi` is preserved, but TDVP now has a larger
local tangent/variational space available on later updates.

## Implications for a TreeTN Translation

The chain algorithm has a single natural "right block" at each cut. A TreeTN
translation must replace that with a directed-edge view.

For an oriented edge `(parent, child)` in a rooted tree, the chain analogue is:

```text
parent side  ==  child tensor plus child-side subtrees
             ^
             expanded bond
```

The child tensor should be reshaped as

```text
rows:    bond index toward the parent
columns: physical indices at child plus all child-side bond indices
```

assuming the child-side subtrees have already been orthogonalized into
isometric bases. Then the same density-projection-eigenvector logic can enlarge
the parent-child bond.

The nontrivial parts are:

| Chain assumption | TreeTN issue |
|---|---|
| There is one left and one right direction. | Every bond needs an explicit orientation relative to a root. |
| Site `j` has columns `(physical, right_bond)`. | A tree node has `(physical indices, all child bonds except parent)`. |
| A right-to-left sweep gives a natural order. | The sweep should follow a postorder over directed edges. |
| Reference density uses one local tensor after the right block is canonical. | The reference work buffers must maintain compatible gauges on all already processed child subtrees. |
| Bond update touches adjacent tensors `j-1` and `j`. | A TreeTN update touches the two endpoint tensors and must replace the shared bond index consistently. |
| TDVP environments are chain arrays. | TreeTN projected-operator caches must invalidate all messages affected by expanded directed edges. |

For issue-level planning, the key implementation target is therefore not "add
KrylovKit". It is an edge-oriented TreeTN basis-expansion primitive that:

1. builds low-rank reference states such as `H psi`, `H^2 psi`, ... without
   hidden full materialization;
2. sweeps directed edges in a canonical gauge;
3. computes projected reference density matrices on local child-side spaces;
4. expands shared bond indices while preserving the target state;
5. updates reference work buffers consistently enough for the next edge; and
6. hands the expanded state to the existing TreeTN TDVP with invalidated
   environments.
