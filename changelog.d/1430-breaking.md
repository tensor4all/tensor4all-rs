- `tensor4all-aci`: `AciError` has a new `Backend { operation, source }` variant (#1382).
  A matrix product of the ACI sweep that the execution context refuses (for example a
  reentrant session), that the engine rejects, or whose operand shapes disagree is now
  reported as `AciError::Backend` with the original error as `source`, instead of
  `AciError::InvalidInitialGuess`, which is kept for problems with the initial guess.
  An exhaustive `match` on `AciError` needs the new arm. `elementwise_batched_with_rng_in`
  now runs every factorization and matrix product of the workload on the caller's
  context, including the batched frame updates, local block setup and global-pivot
  environments that previously took the process-global route; the context-free entries
  run the same code path on the process-global context. In a build without
  `global-defaults` they return `AciError::Backend` for a run that needs a sweep, after
  input and option validation; the single-site case needs no context and still succeeds.

