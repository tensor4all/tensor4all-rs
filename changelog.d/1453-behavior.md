- `tensor4all-treeaci`: a run no longer canonicalizes its output (#1449), so
  `tree_elementwise`, `tree_elementwise_batched*`, `hadamard_many` and
  `TreeAciInterpolator` results now report `tree.canonical_form() == None`
  instead of `Some(CanonicalForm::CI)`; the canonical region is still the node
  of the last commit. The CI canonicalization after the first pass re-selected
  its own pivots and moved bonds out of the basis that later commits read (bond
  axis `k` = the `k`-th retained candidate), and the spine-only return pass then
  committed against them: on the #1449 reproducer a run stopped `Converged` at
  the true ranks with a recorded error of 4.4e-16 and a dense relative error of
  1.58. Measured on the #1449 reproducers (the full and the reduced tree, Guard
  on and off, 11 seeds each; see #1453), no run now stops at a false
  `Converged`; the committed regression test pins the reduced tree with the
  Guard off and seed 1. This is not established in general: a reverse cleanup
  pass that consumed Guard-injected candidates could still return a tree
  outside the pivot basis (#1450, fixed in the entry below). An initial guess's canonical form is cleared at
  initialization instead of being passed through to the result. Callers that
  need a canonical form canonicalize the result themselves. The rustdoc of
  `TreeAciResult::max_errors` and `TreeAciOptions::tolerance` now states what
  the recorded per-edge residual measures (the output's error at commit time
  over that commit's local matrix, while the bonds it reads are in that basis)
  and what it does not bound (the dense error, later commits, pass-1
  residuals).
