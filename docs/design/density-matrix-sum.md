# Density-matrix sums of TreeTNs

`TreeTN::density_matrix_sum(terms, center, cutoff, max_bond_dim)` builds a
compressed sum without an exact direct-sum intermediate. The terms have the
same named topology, full site-index identities, and scalar kind. The output
is unitary toward `center`; the result also reports `discarded_weight`, the
sum of absolute discarded density eigenvalue tails.

## Algorithm and existing machinery

Root the tree at `center` using the existing canonicalization edge order.
The existing sum-fit overlap environments supply the exact metric on the
opposite side of each edge, including every pair of summands. Separate bra
bond labels prevent accidental contraction of corresponding bond legs.

Process nodes toward the center. Child bases have already been selected;
contract their coefficient maps into each original local core, producing
`A_i(q, a_i)`, where `q` contains the site's physical indices and retained
child bonds. The density matrix is

`rho(q, q') = sum_ij A_i(q, a_i) E_ij(a_i, a_j) conj(A_j(q', a_j))`.

Keep the leading eigenvectors whose discarded eigenvalue tail is at most
`cutoff * trace(rho)`, subject to `max_bond_dim`. Contract their conjugates
into each `A_i` to pass coefficient maps to the parent. At the root, sum
the projected cores. No whole-network dense tensor or summed input bond is
formed. Exact opposite-side overlap metrics make a preliminary QR sweep
unnecessary, and the retained eigenvectors already give canonical output;
there is no second canonicalization pass.

## Error contract

In exact arithmetic, `||sum_i t_i - output||_F^2 <= discarded_weight`.
The subtree projectors are nested on ancestral edges and commute on
disjoint subtrees. Sequentially applying them gives orthogonal losses.
Each actual loss is bounded by the local density tail, which is evaluated
with the original exterior rather than an already projected exterior.

Without a binding bond cap, every tail is at most `cutoff * ||sum_i t_i||_F^2`;
therefore the relative error is at most `sqrt((R - 1) * cutoff)`. A cap can
violate that requested cutoff; the reported absolute tail includes cap losses.
The algorithm is not an SVD implementation: forming density matrices squares
the singular spectrum and loses relative accuracy near machine precision.
Roundoff, especially under cancellation between large terms, can invalidate
the exact-arithmetic estimate. Equal cutoffs need not produce the same ranks
as a two-direction SVD truncation on arbitrary data.

## Cost and scope

For chains with `R` sites, `N` equal-bond-`D` terms, local dimension `d`, and
kept bond `k`, pair environments require `O(R N^2 d D^3)` work and retain
`O(R N^2 D^2)` elements. At each node the density dimension is `m = d k`;
forming it takes `O(N^2 (m D^2 + m^2 D))` work and diagonalizing it takes
`O(m^3)`, with `O(m^2)` local density storage. This is quadratic in `N`,
not the linear target count of variational fitting. It may be slower than
the N-ary sum at small input bonds.

On a branching tree `m` is the product of the site's physical dimensions
and all retained child-bond dimensions. Local density matrices and pair
contractions can consequently be expensive at high-degree nodes. The cap
limits each output bond, not the number of local density entries.

The exact `SumEngine` contract is unchanged. This method is an explicit
approximate entry point, usable on its own or as the initial network for
`fit_sum_with_report`.
