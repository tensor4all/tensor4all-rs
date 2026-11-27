- `PreparedContraction::new` rejects shared labels with inconsistent dimensions
  during planning, including retained labels whose identities match but extents
  differ. Previously such invalid prepared plans could be constructed, while
  ordinary dense contraction already rejected the inconsistency during execution.
