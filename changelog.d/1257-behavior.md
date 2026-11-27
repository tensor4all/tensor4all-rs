- `PivotSearchStrategy::Rook` runs two more rook walks, from columns spread
  over the remaining block, before it accepts a tolerance stop, and takes the
  first pivot they find above the threshold (#1257). On the #1256 ridge and a
  4-site ridge without global pivots, single-walk stops returned `Converged`
  with true errors of 1.3e-6 and 0.39; both now reach the tolerance. The stop
  is still heuristic. A rook walk also moves only to a strictly larger
  magnitude now: on a roundoff-level residual it used to cycle until its step
  limit (rows + columns + 1 steps), because the same entry computed from its
  row and from its column differs in the last bits.
