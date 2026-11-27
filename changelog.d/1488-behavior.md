- Reverts #1484: `TCI2Termination::Converged` under `PivotSearchStrategy::Rook`
  is again not certified by an exact `Full` sweep. `Rook` is a fast, inexact
  heuristic by design; use `Full` when the tolerance must hold (#1257).
