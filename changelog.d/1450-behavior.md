- `tensor4all-treeaci`: the return pass also re-commits, children first, every
  off-spine bond whose candidate set a Guard injection enlarged, before the
  spine commit that reads it (#1450). Before, it re-committed only the spine,
  so a spine commit could select an injected candidate that its factor then
  dropped, and that bond left the pivot basis (bond axis `k` = the `k`-th
  retained candidate). With `max_sweeps = min_sweeps = 1` the cleanup return
  pass returned such a tree: on G0 r = 5 (comb tree, seed 42, tolerance 1e-3)
  max|M - I| was 0.76 and is now 0, at 9,196 oracle points instead of 3,556.
  Runs whose Guard injects after a forward pass change their trajectory; on the
  G0 r = 5-8 comb and swap workloads the median oracle points at matched
  accuracy changed by 1.00-1.17x per workload. Runs without such an injection are unchanged.
