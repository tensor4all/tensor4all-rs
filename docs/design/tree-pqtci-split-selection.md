# Tree pQTCI split selection and overpatching design review

## Status

Design notes for M5 of
[`tree-adaptive-patching-roadmap.md`](./tree-adaptive-patching-roadmap.md).
Nothing here is approved for implementation: the proposals and open questions
below need the user's review. The only M5 data so far is the fixed-depth
exploration in
[`2026-10-03-m5-fixed-depth-exploration.md`](../../benchmarks/results/2026-10-03-m5-fixed-depth-exploration.md).

## Current seam

`adaptive_interpolation::Driver::split` chooses the first unfixed site in the
validated `SiteLayout::split_order`. That is a site-coordinate split: one
`DynIndex` is fixed to each local value, the patch cache is partitioned among
children, and the corresponding full site index stays in the resulting
TreeTN. `patch_order` can restrict and order these sites.

The existing `PatchSplitStrategy::{Sequential, ExactParameterGain}` belongs
to patched algebra. `ExactParameterGain` projects and truncates a concrete
`SubDomainTreeTN` for each candidate and compares its logical parameter
count. It is not a split strategy for the pQTCI driver, whose candidate
children do not yet have interpolants. A pQTCI option needs its own name and
contract if one is added.

The M1 `InterpolationOutcome::pivots` are full points intended to seed later
engine runs. They contain no edge-local scores or function values.

## The paper's heuristic

Section 3.7 and Appendix C (Algorithm 1) of
[Grosso et al., v3](https://arxiv.org/abs/2602.22372v3) choose the next split
site of a TT patch as follows. Take the first bond `l_max` of maximal bond
dimension. Combine every row pivot `i` of its left pivot set with every
column pivot `j` of its right pivot set into a full index `i ⊕ j`. For each
candidate site `l` and local value `v`, overwrite site `l` with `v`, evaluate
the function to form the modified pivot matrix `P(l, v)`, and accumulate
`S_l = Σ_v rank_τ(P(l, v))²`. The chosen site minimizes `S_l` (the paper
normalizes by `rank_τ(P_lmax)²`, which does not change the minimizer). The
heuristic runs per patch, so different patches at the same level can split at
different sites.

The paper is distributed under CC BY-NC-SA 4.0. An implementation that ports
its pseudocode must record the derivation under the repository's
[provenance policy](../PROVENANCE_AND_CITATION_POLICY.md) and check license
compatibility first.

## Proposed tree generalization (not reviewed)

Use the largest-rank **edge** in place of the largest-rank bond: the edge
splits the tree into two node sets, the engine's pivots are projected onto
each side, and the row and column projections are crossed into full points.
Overwriting a candidate site changes the row or column coordinate on the side
that holds it. Ties are broken by canonical edge order, then by the validated
`split_order`; candidates stay within `patch_order`.

What this needs before code is written:

- an amendment of the M1 pivot contract so per-edge pivot sets can be used
  this way, and a rule for engines that return no pivots;
- `rank_τ` defined consistently with the selected error norm, computed
  through the existing tensor factorization seam (no local SVD, no
  unconfigured backend);
- a bound on the probe work `Σ_l d_l · n_rows · n_cols` and on matrix memory;
- a provenance and license review (above).

## Open questions for the user

1. **Selector.** Adopt the tree generalization above as an opt-in pQTCI
   option, keep `Sequential` as the default until measurements justify a
   change?
2. **Minimum patch size.** The roadmap lists a minimum patch size option. What
   should happen when a patch fails its tolerance and every remaining split
   would go below the minimum: accept it with its measured error reported,
   stop the whole run with an error, or something else?
3. **Sibling merging.** The roadmap lists sibling reconstruction with
   independent parent-region verification. Its acceptance rule, how it is
   reported, and whether it belongs in the driver or as a post-processing
   step are not decided.
4. **Capped outcomes and early exit** (M3 open question 4, deferred to M5):
   whether a `BondCapReached` patch may be accepted on its measured error,
   together with stopping the engine at the first saturated sweep.
5. **Corner-localized misses.** Split rules and child candidate sets must
   take them into account
   ([known limitation](./tree-patching-error-contract.md#known-limitation-corner-localized-misses));
   the fix itself belongs to the M9 global review.

## Measurement requirements

Any M5 measurement that is to support a decision must:

- run the adaptive pQTCI driver on workloads at downstream scale: realized
  ranks of roughly 30–200 (bond caps around 200, as in gw-rs), TreeTCI
  tolerance around `1e-4`, and enough quantics bits that patches stay far
  larger than a few grid points. Runs at bond caps of a few units or tens are
  smoke tests only;
- compare policies at matched accuracy, never at one shared tolerance;
- include at least one branched tree (a node of degree three or more), a
  chain control, a localized workload, and a delocalized workload;
- pin the source revision, workloads, seeds, candidate set, error norm, and
  cost metric before timing, and record split-probe evaluations separately
  from interpolation evaluations;
- report overpatching as total logical parameters over the unpatched
  interpolant's parameters, with the permitted margin fixed in advance.

The fixed-depth exploration does not meet these requirements (unknown
revision, static partitions, sampled max-norm criterion, single runs); it only
guides the choice of workloads.
