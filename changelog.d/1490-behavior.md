- TreeTCI adds an opt-in `TreeTciOptions::rank_slack` (default `0.0`) and a
  positive `ncheck_history` (default `3`) for its stopping windows (#1264).
  Both result types now expose `nglobal_pivots` per iteration. Defaults preserve
  the existing stopping behavior; a nonzero slack needs independent accuracy
  validation. `TreeTciInterpolator::MIN_MAX_ITER` is removed: its constructor
  requires `max_iter >= ncheck_history` instead. Quantics wrappers retain the
  default window and zero slack.
