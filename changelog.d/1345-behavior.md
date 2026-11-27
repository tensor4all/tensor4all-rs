- `tensor4all-treetn` no longer enables `simplett-bridge` by default (#1335).
  Enable it explicitly for SimpleTT conversions and the bridge's chain helpers.
  Native M6 benchmark builds also exclude legacy comparisons and Fourier
  construction unless their benchmark features are explicitly selected.
