- `PreparedContraction::new` rejects shared labels with inconsistent dimensions
  during planning, including retained labels whose identities match but extents
  differ. Previously such invalid prepared plans could be constructed, while
  ordinary dense contraction already rejected the inconsistency during execution.
  The explicit-session `contract_with_options_on_session` route now rejects the
  same invalid metadata before materializing operands on the session, with an
  earlier shape-mismatch message instead of a downstream session error.
