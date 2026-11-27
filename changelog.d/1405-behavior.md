- TreeTCI's default stop rule is the per-bond rule again (#1405), the one of
  TensorCrossInterpolation.jl, TreeTCI.jl and xfac: the recorded error is the largest
  bond pivot error, and the global search accepts a pivot only above
  `tolerance * tol_margin_global_search` (scaled by the sample magnitude when
  `scale_tolerance` is on). The sampled stop of #1357, which also records the residuals
  the global search measured and keeps observed errors above `tolerance`, stays available
  as `TreeTciOptions { stop_rule: StopRule::Sampled, .. }` (and
  `QtciOptions::with_stop_rule`). On the 16-site quantics chain of the treetci
  full-run golden the sampled stop takes 1.74x the oracle evaluations of the per-bond
  stop (22073 against 12720, six iterations against four); on the 16-site Gaussian
  chain of #1371 it took 3.14x, 3.37x and 2.18x at tolerances 1e-4, 1e-6 and 1e-8
  (six seeds summed), for the same true error. The unit-endpoint probes and subtree
  restarts of #1357, the third search stage of #1433 and the #1434 proposer change stay
  on both rules; under the per-bond rule their finds are accepted only above the
  margin. Runs with the global search enabled can therefore stop earlier, with
  different `errors`, than on the previous release; to keep the previous behaviour, set
  `StopRule::Sampled` (Python: `crossinterpolate(..., stop_rule="sampled")`). The public
  `find_global_pivots*` functions keep the sampled acceptance, and
  `TreeTciInterpolator` always runs `StopRule::Sampled`, because the adaptive drivers
  of tensor4all-partitionedtreetn accept a patch on its error estimate.
