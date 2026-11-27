- `tensor4all-treeaci`: a `Converged` stop no longer rests on per-edge residuals that
  later commits of the same pass made stale (#1451). Every commit rewrites two output
  cores, so after a pass every edge except the one committed last holds the residual
  of an output that no longer exists. On a pass that meets every other stop condition,
  those residuals are now re-measured as the output's largest error over each edge's
  current local matrix, the stop is refused when one exceeds the tolerance, and that
  pass's `max_errors` entry holds the re-measured maximum. On the diagnosed run (G0
  r = 5, cttn, tolerance 1e-3, `min_sweeps = 1`, seed 1, with the rule applied to
  `29baf4a9`, where it reproduces; #1450 changed that seed's trajectory) the pass-3
  entry becomes 4.73 instead of a stale 9.5e-4, and the run stops after pass 4 at a
  dense error of 8.0e-4 instead of `Converged` at 4.73 after pass 3. Over seeds 1-400
  at `min_sweeps = 1` the wrong stops (`Converged` above 1e-2) fall from 13 / 6 / 32
  to 0 on the cttn / swap / nblock layouts, at 2.11x / 1.60x / 2.81x the oracle
  points; with the default `min_sweeps = 2` at 1.04x / 1.37x / 1.33x, with every
  cttn and swap run keeping its termination, pass count and dense error. Each pass
  that reaches the stop test pays about one forward pass of operator evaluations.
  With `min_sweeps = 1`, 9 of 400 cttn runs now end `MaxSweeps` on the known
  local-error plateau (dense error 1.01e-3) instead of stopping at up to 5.8e-3.
  `TreeAciDiagnostics::evaluated_points` includes the re-measurements.
