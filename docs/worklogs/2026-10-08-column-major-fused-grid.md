# Column-major fused (left, site) grid (#821)

## Decisions

- `Tensor3Ops::as_left_matrix` / `as_right_matrix` (and the `try_` forms) and
  `Tensor4Ops::as_left_matrix` / `as_right_matrix` / `as_center_matrix` are
  the column-major reshapes of the fused axes, so each emitted buffer equals
  the tensor's flat column-major buffer. The public order changes from the
  previous site-major / right-major fusion; no compatibility shim is kept.
- TCI1 encodes the same grid in four places that must agree: the pivot-set
  enumeration (`build_pi_i_set` has `left` fastest, `build_pi_j_set` has
  `site` fastest), the `MatrixCI` row/column positions, the site-tensor
  decoders (`update_site_tensor_from_matrix`, `tensor_from_left_matrix`), and
  `conversion::split_indices`. All were changed together. `positions_in_set`
  looks values up by identity and needed no change.
- The private `tensor3_to_left/right_matrix` helpers in `canonical.rs`,
  `vidal.rs` and `compression.rs` were already self-consistent and are left
  alone.

## Verification conclusions and constraints

- Element-level tests pin the column-major positions for all five public
  helpers, the pivot-set enumeration, and the round trips through
  `tensor_from_left_matrix`, `update_site_tensor_from_matrix` and
  `split_indices`. `crossinterpolate1` reproduces a function with unequal local
  dimensions on its full grid.
- Tests of `tensor4all-simplett`, `tensor4all-tensorci`,
  `tensor4all-partitionedtt` and `tensor4all-treetci` pass; no tolerance was
  changed.
