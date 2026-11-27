- For 0.4.0 (unreleased), remove core's `contract_owned`,
  `contract_owned_with_options`, their `defaults` exports, and the doc-hidden
  `tensor4all_core::legacy::einsum_native_tensors_owned` bridge. Replace
  `contract_owned(tensors)?` with `let refs: Vec<_> = tensors.iter().collect();
  contract(&refs)?`; replace `contract_owned_with_options(tensors, options)?`
  with the same references and `contract_with_options(&refs, options)?`.
  These calls borrow the caller's tensors without cloning their payloads.
  Existing backend-level native-tensor callers replace the legacy bridge with
  `tensor4all_tensorbackend::einsum_native_tensors_owned_in(&context, operands,
  output_ids)?`; both tensorbackend-owned einsum APIs remain. The pinned usage
  survey found only core implementation/wrapper and seven test call sites, with
  no matching names in the surveyed tenferro, tenferro-sci or Julia snapshots;
  this does not establish usage in private or uninspected downstream code.
  Full survey and migrations: `docs/design/core-contraction-api.md`.
