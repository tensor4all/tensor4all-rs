# The two-site tour's return visit in `TreeTN::truncate` (#1186 part 2)

Evidence for the question the issue asks about `truncate`: the return half of the
two-site Euler tour takes 46-47 % of the sweep, so what does it buy, and what can
be removed from it without changing the result.

Harness: `crates/tensor4all-quanticstransform/examples/repro_1186_truncate_two_visit.rs`
(public API only, so the same file builds against a checkout with and without the
change). It lives next to the operator it applies: the state is a comb TreeTN
after a per-branch zip-up apply of the transposed quantics Fourier transform,
which is the state the issue measures.

States:

- **A** - three branches of 20 binary sites, bond `min(2^(i+1), 128)`, per-leg
  amplitude weight `exp(-0.1 * bond index)`, canonicalized at the junction
  (state A of the issue thread). Bonds 128 / 5754.
- **B1** - A with the transposed 20-site quantics Fourier operator applied to
  branch 0 by `ApplyOptions::zipup()` at rtol 1e-8 (state B1 of the thread).
  Bonds 595 / 9723.
- **B2** - B1 truncated once at rtol 1e-9, i.e. a state that is already at the
  ranks the sweep picks (state B2 of the thread). Bonds 270 / 5959.
- **IFT** - the reporter's configuration: a comb with bond cap 160 and weight
  `exp(-0.2 * bond index)`, all three branches zip-up applied at rtol 1e-8.
  Bonds 307 / 9888.

Revisions: base = `a1a02af6` (`origin/main`), head = the PR branch. Host: the
L3 domain 0-3 is the measured cell, the measured process is pinned there with
`taskset -c 0-3` and `RAYON_NUM_THREADS=4`; the shell that launches the runs stays
on CPUs 60-63. Load average 26 / 17 / 12 during the campaign, so the absolute
times are host-loaded; every cell alternates base and head, and each side
reproduces bit for bit, so the ratio is what the numbers support, not the third
digit.

## Reproduction

The two binaries are two checkouts: the base is `origin/main` with the example
file copied into `crates/tensor4all-quanticstransform/examples/`, the head is the
PR branch, and both are built with their own `cargo build --release`. Build each
with `--example repro_1186_truncate_two_visit` and run the resulting
`target/release/examples/repro_1186_truncate_two_visit`.

```bash
ex=repro_1186_truncate_two_visit
pin() { taskset -c 0-3 env RAYON_NUM_THREADS=4 "$@"; }

# base checkout, then head checkout, alternating; $B and $H are the two binaries
pin "$B" diag B1 1e-9
pin "$B" run  B1 1e-9 base /tmp/b-base.bin
pin "$H" run  B1 1e-9 head /tmp/b-head.bin
pin "$B" run  B1 1e-15 ref /tmp/b-ref.bin
pin "$B" dims B1 1e-9
pin "$H" dims B1 1e-9
```

`run` prints the truncation wall time, the bond dimensions before and after and a
fingerprint; the amplitude files are what the accuracy column compares (relative
L2; the tree inner product cannot resolve differences below about 1e-8).

## What the return visit does, per edge (`diag`, base, rtol 1e-9)

`visit 1` is the outbound pass, `visit 2` the return pass. "units" are bond
dimensions removed (`bond before - bond after`); "SVD s" is the time inside the
two-site factorization.

| state | bonds before -> after | units visit 1 | units visit 2 | discarded weight visit 1 | discarded weight visit 2 | SVD s visit 1 | SVD s visit 2 | visit 2 share |
|---|---|---|---|---|---|---|---|---|
| A | 128/5754 -> 112/4105 | 1649 | **0** | 1.2e-9 | 1.2e-15 | 2.673 | 1.891 | 41 % |
| B1 | 595/9723 -> 270/5959 | 3762 | **2** | 2.6e-9 | 6.1e-10 | 4.178 | 2.802 | 40 % |
| B2 | 270/5959 -> 270/5959 | 0 | **0** | 1.4e-15 | 1.4e-15 | 2.699 | 2.725 | 50 % |
| IFT | 307/9888 -> 135/5254 | 4632 | **2** | 2.6e-9 | 7.9e-10 | 2.612 | 1.503 | 36 % |

"Discarded weight" is the largest relative Frobenius norm of the discarded tail
over the visits of that direction, summed from the discarded squares; the cutoff
is 1e-9, so the outbound visits discard at the cutoff and the return visits
discard one to six orders of magnitude less.

The two units the return pass removes, named:

| state | edge | bond | outbound visit removed | return visit removed |
|---|---|---|---|---|
| B1 | 36-39 | 265 | 314 | 1 |
| B1 | 42-45 | 256 | 337 | 1 |
| IFT | 28-31 | 129 | 158 | 1 |
| IFT | 40-43 | 135 | 165 | 1 |

43 or 45 of the 60 return visits (per state) drop singular values of the block at
all - typically 40-180 of them, at 1e-10 to 1e-15 relative weight - yet only the
four rows above move the bond. Counting dropped singular values instead of dropped
*rank* therefore says the return visit truncates everywhere; counting rank says it
truncates at two edges per application state.

## The candidate skip rule

The rule "skip the return visit on an edge whose outbound visit removed no rank"
is exactly result-preserving on this data: in all four states, the return visits
that remove rank sit on edges whose outbound visit removed 314-337 units, so they
are not in the skip set.

| state | return steps the rule skips | share of the return SVD time | share of the sweep SVD time | skipped visits that would still remove rank |
|---|---|---|---|---|---|
| A | 15 / 60 | 0.2 % | 0.1 % | 0 |
| B1 | 18 / 60 | 23.0 % | 9.2 % | 0 |
| B2 | 60 / 60 | 100 % | 50.2 % | 0 |
| IFT | 21 / 60 | 11.0 % | 4.0 % | 0 |

So the rule saves 0.1 % (A), 9 % (B1) and 4 % (IFT) of the sweep on the states the
issue measures, and 50 % only on a state that has already been compressed once.
It is also not a *proof*: it rests on the observation that a bond which was full
rank on the outbound visit stays full rank on the return visit, and the counter-
example would be a state where truncating the sub-tree below an edge lowers the
rank across that edge.

## The change that shipped

The return visit's time is the two-site factorization, and any correct
implementation owes that factorization for the gauge move, so nothing was
skipped. What did change is its cost: `TruncateUpdater` conditions the left
operand with an exact QR first, so the truncating SVD runs on the `bond x
columns` remainder instead of the full block. `A = Q R` with `Q` isometric gives
`AB = Q (R B)`, so the singular values, the retained rank and the truncated value
are the block's own; only the work shrinks. It applies to both directions of the
tour and only when the bond is smaller than both matricization dimensions.

Truncate wall time, two runs per side, base and head alternating:

| state | base | head | speedup | bonds (both) | units (both) |
|---|---|---|---|---|---|
| A | 5.153 s / 5.145 s | 2.304 s / 2.293 s | 2.24x | 128/5754 -> 112/4105 | 1649 |
| B1 | 8.743 s / 8.761 s | 5.846 s / 5.873 s | 1.50x | 595/9723 -> 270/5959 | 3764 |
| B2 | 5.844 s / 5.822 s | 3.389 s / 3.387 s | 1.72x | 270/5959 -> 270/5959 | 0 |
| IFT | 4.863 s / 4.860 s | 3.862 s / 3.871 s | 1.26x | 307/9888 -> 135/5254 | 4634 |

Accuracy against a base run at rtol 1e-15, relative L2 over 8192 sampled
amplitudes:

| state | base vs ref | head vs ref | head vs base |
|---|---|---|---|
| A | 6.106e-9 | 6.106e-9 | 9.97e-14 |
| B1 | 9.457e-9 | 9.457e-9 | 9.93e-14 |
| B2 | 2.101e-14 | 2.782e-13 | 2.765e-13 |
| IFT | 1.310e-8 | 1.310e-8 | 6.69e-14 |

The accuracy columns are **not recomputable from this repository**: they come from
an offline comparison of the amplitude files that `run` writes, and neither those
files nor the comparison script were kept. They are reported as measured, not as
evidence a reader can regenerate; what a reader can check is the agreement of the
two binaries' `dims` and `diag` output, which is in the JSON.

`dims <kind> 1e-9` prints every bond dimension before and after a truncation, and
the two binaries agree on the whole vector in all four states, so it is not only
the maximum and the sum that match. The number of removed units is identical too,
and the state agrees to 1e-13, which is the different rounding of an equivalent
factorization, not a different truncation. The raw output of the timing runs and
of the `diag` classification is in
[the evidence JSON](2026-11-10-treetn-truncate-return-visit.json)
(`campaign-*.txt`, `r2-diag-*.txt`), and every number in the table above is read
off those captures. The reference runs (`run <kind> 1e-15`) and the offline
comparison that produces the accuracy table are **not** in the JSON: they are
reproducible from the commands recorded there, but their output was not kept.

## Not measured

- A state where truncating a sub-tree lowers the rank across an edge above it;
  that is the case the skip rule would get wrong, and none of the four states here
  produces one.
- Wall times on an idle host: the campaign ran at load average 12-26, so the
  speedups are host-loaded. The bond dimensions and units are exact regardless.
- The reporter's original `chi = 160, decay = 0.05` comb and their seed; the IFT
  state here uses seed 12345 and decay 0.2. Their table's "largest bond after the
  transform" is 138 for decay 0.2, this host reaches 307 for the same decay, and
  the difference is not explained by the seed (seed 7 gives 304); part 1 (#1197)
  and the API changes since `e7861ba2` are the likely causes and were not chased.
- GPU validation: not performed. `truncate_in` reaches the same updater through
  the context-scoped route, so the conditioning sits on a CUDA-reachable path; the
  host used here has no GPU and that path was not exercised.
- A zero or near-roundoff cutoff. The comparison `sigma > cutoff * sigma_max` then
  no longer separates the same set of singular values in the two routes: the
  conditioned SVD cannot retain more than the current bond, while the direct one
  can keep positive roundoff directions beyond it. On these states a zero cutoff
  makes the sweep grow the bonds until it becomes unusable (a run was abandoned
  after an hour), so no numbers were produced; the conditioned route is the one
  that keeps the bond from growing.
