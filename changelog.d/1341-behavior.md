- `tensor4all-simplett`: `evaluate`, `sum`, `norm2`, `norm`, `log_norm`,
  `inner_product`, and `inner_product_with_session` now refuse a tensor train
  whose first core has a non-unit leading bond (#1303, #1340), the
  left-boundary sibling of the trailing-bond guard from #913. They previously
  read only the `l == 0` component and returned its projection; the
  inner-product routes summed every leading component instead.
