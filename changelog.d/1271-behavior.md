- `PivotSearchStrategy::Rook` is now documented and implemented as a heuristic
  strategy (#1271, #1282). A bond update stops when the current rook pivot is
  below tolerance, and the trailing `pivot_errors` entry (and so the TCI2 bond
  error) is that last rook pivot, not an upper bound on the remaining residual;
  `TCI2Termination::Converged` under `Rook` does not prove the tolerance. The
  exact remaining-residual scan that v0.3.0 ran before every `Rook` tolerance
  stop and at a rank cap (#1258, #1090) is removed, and `Rook` walks now start
  from the bond's previous pivots. `PivotSearchStrategy::Full` stays the
  default and remains certified; use it when a bound is required. The same
  contract applies to `matrix_luci_factors_from_blocks`.
