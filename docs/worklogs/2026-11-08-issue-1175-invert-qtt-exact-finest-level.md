# invert_qtt's exact finest level: when it applies

## Decisions

- The finest level is reconstructed in the parent basis (exact at every node) only
  when the train is still the interpolation construction's uncompressed form, and
  the blend otherwise. `invert_qtt` decides from the train itself; there is no flag
  and no second entry point (tensor4all-rs#1175).
- The criterion is three conditions, not the theoretical
  `rank(i) == min(2^i, 2^(K-i))` ranks of a generic `2^K` vector: this
  construction's bonds saturate at `basis.len()` (11 for 12 sites at degree 10), so
  that formula never holds for any train the crate builds.
  1. every link at `basis.len()` — the "no bond was truncated" half, which also
     catches an earlier truncated bond;
  2. the finest core proportional to `L_alpha(sigma / 2)` — this is what makes the
     last bond index name a local basis function. `SimpleTensorTrain::scale_mut`
     multiplies the last core only, and applying it twice perturbs the core by a
     couple of ulps, so the comparison is proportional with a `1e-14` relative
     bound and an exact zero;
  3. the reconstruction agrees with the train's own samples where two subintervals
     meet. The basis is cardinal at `t = 0`, so the train's `t = 0` sample of the
     next subinterval is `current[cell + 1, 0]`, and the reconstructed `t = 1` value
     of a subinterval has to equal it. A gauge that rescales a coefficient column of
     an earlier core leaves every bond, every stored sample, and the finest core
     canonical, and only this check catches it.
- Rejected: a bit-exact core comparison (fails a legitimate double `scale_mut`), and
  a loose relative tolerance on the core alone (admits a perturbation that the
  parent coefficients amplify; the boundary check is what bounds the output).
- Rejected: verifying the whole chain structurally against the canonical cores.
  Columns whose basis functions vanish at the sampled points stay invisible, so the
  freedom is in the train, not in the check.

## Verification conclusions and constraints

- Both counterexamples are pinned as regressions: a sample-preserving gauge on an
  earlier core falls back to the blend (`invert_qtt_gauge_scaled_coefficient_column_does_not_reach_the_exact_route`),
  and a twice-rescaled train keeps the exact route
  (`invert_qtt_repeatedly_rescaled_train_keeps_the_exact_finest_level`).
- Untouched constructions have a boundary residual of exactly `0` (degree 2, 3, 6 and
  10; 2 to 16 sites), which leaves the `1e-12` bound twelve orders above the
  measured value.
- Against the directly evaluated parent interpolant for a quadratic (which the
  degree-`d` interpolant reproduces, so the reference carries no error of its own),
  the exact route is `1.81e-15` at degree 10 / 12 sites and `6.14e-16` at degree 2 /
  8 sites; the blend is `1.19e-7` and `3.05e-5`.
- Limit (maintainer-ruled contract, narrowed after the final review, PR #1198): the
  check detects truncation, rescaling and any rewrite that moves the reconstruction
  off the stored samples — `t = 0`, `t = 1/2` and the shared endpoints. A rewrite whose
  component vanishes at all of those points is **not** detectable: for degree >= 3 a
  basis function such as `L_1` vanishes there, so doubling its coefficient leaves every
  sample, every bond, the finest core and the boundary residual untouched while the
  exact route reports the rewritten coefficient (`0.1464`-node value `2` instead of
  `1` at 2 sites, degree 4). The train stores two samples per subinterval and carries
  no information there, so no comparison inside `invert_qtt` can reject it; the route
  is a contract about a train that is the direct output of this crate's constructors.
  Pinned by `invert_qtt_sample_preserving_interior_rewrite_is_documented_as_undetectable`,
  which fails if that documented behaviour changes.
- Unverified: nothing outside `tensor4all-interpolativeqtt` changes, and no claim
  here covers trains that were hand-built to be sample-equivalent on purpose.
