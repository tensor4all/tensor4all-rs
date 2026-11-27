- TreeTCI streams the third-stage leaf-side enumeration instead of retaining
  every deduplicated full-site point (#1441). Sparse cursors preserve its
  lexicographic order and equal-residual pivot ties, while only one bounded
  evaluation batch and at most `max_nglobal_pivot` accepted points are kept.
