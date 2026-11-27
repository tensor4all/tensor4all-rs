- `tensor4all-interpolativeqtt` returns TreeTN chains with ordered physical indices
  for ordinary and patched interpolation (#1334). Pass the tree and its site
  indices to `invert_qtt::<T>` for binary chains. The interpolation crate no longer
  depends on SimpleTT.
