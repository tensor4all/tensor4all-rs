# TreeTCI candidate random streams (#824)

## Contract

- The canonical proposer implementation is `candidates_with_rng<T, R>`;
  candidate generation and sampling consume the supplied `R: Rng + ?Sized`
  directly. The seeded `candidates` convenience method delegates to it with
  `ChaCha8Rng::seed_from_u64(seed())`.
- High-level optimizer and interpolation calls construct two named ChaCha8
  streams once per call: candidate draws use the proposer seed, global starts
  use `TreeTciOptions::seed`. This preserves the two independent seed controls.
  One shared helper owns high-level construction.
- Caller-RNG optimizer and interpolation calls share the supplied stream
  between candidates and global searches. Proposer and option seeds are ignored
  there. Reusing the same generator advances it across calls; no RNG state is
  hidden in a proposer or derived from edges, ranks or snapshot history.
- The default proposer consumes no draws. Stream routing dispatches once at
  edge/search boundaries, not per tensor element.

## Compatibility and verification

- Fixed-seed trajectories intentionally change from the unstable
  DefaultHasher/SmallRng derivation. Custom proposers implement the canonical
  caller-stream method and can override `seed` for high-level calls. No
  compatibility shim or deprecated implementation is retained.
- Direct-reference tests compare random candidate draws, ordered truncation,
  successive calls and the exact remaining RNG state. Optimizer instrumentation
  checks every edge pass receives the same caller-owned stream. Deterministic
  proposers leave it untouched when global search is disabled; randomized
  proposers advance it even in that configuration.
- The seed convention is distinct from continued-state convergence/history
  semantics, handled in #833. High-level calls restart their generators;
  caller-stream calls preserve the sequence if the caller reuses the generator.
