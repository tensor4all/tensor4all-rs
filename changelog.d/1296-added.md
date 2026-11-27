- `tensor4all-treetn` adds `apply_linear_operator_certified` and
  `apply_linear_operator_certified_to_indices`, with `CertifiedApplyOptions`,
  `CertifiedApplication` and `CertifiedApplyError` (#1286). On a chain state
  they truncate during the zip-up sweep and return a proved bound on
  `||A x - y||`, at most the caller's absolute budget, without forming the exact
  image. The exact factors the certificate needs are estimated before they are
  built, and an estimate above the memory limit (4 GiB by default) is refused.
