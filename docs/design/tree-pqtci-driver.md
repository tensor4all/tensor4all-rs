# Sequential tree pQTCI driver

## Status

Approved and implemented in M2 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md). It
builds on the M1 contract in
[tree-interpolation-engine-seam.md](./tree-interpolation-engine-seam.md). The
implementation consists of:

- the public helper `tensor4all_treetn::interpolation::validate_layout`, which
  `InterpolationProblem::new` now calls;
- the module `tensor4all_partitionedtreetn::adaptive_interpolation` (the
  driver, its packed-key cache, candidate sampling, and re-embedding) and the
  crate-internal assembly `PartitionedTreeTN::from_disjoint_subdomains`;
- the driver-local dense test engine and the TreeTCI end-to-end tests;
- the provenance record, `LICENSE-TCIALGORITHMS-MIT`, and the public-surface
  updates listed below.

The decisions taken during implementation are recorded under
[Implementation decisions](#implementation-decisions).

## Goal

Adaptive patched interpolation of a function on an arbitrary tree: run an
interpolation engine on the whole domain, and wherever the engine does not
converge below the bond cap, fix the next site in a given order and retry on
each child region. The result is a `PartitionedTreeTN` with disjoint patches.
This is the producer of every patch that later milestones measure or consume,
so it favours correctness and determinism over speed; parallel execution is
M7 and split-site selection beyond a fixed order is M5.

## Findings that shape the design

Verified against the M1 branch state.

- The M1 contract (`tensor4all_treetn::interpolation`) returns an outcome whose
  network carries only the active site indices. `Converged` means the error
  criterion holds with the final rank strictly below the cap, so a patch with
  a nonzero value on several nodes never converges with a cap of one.
  All-zero initial samples return `AllSamplesZero`; non-finite initial
  samples return `Evaluator`. The contract requires at least one active site.
  M1 pivots cover the active sites only.
- `SubDomainTreeTN::new(tree, projector)` requires every projector index to be
  a site index of `tree` and masks projected indices eagerly; the
  crate-internal `from_masked_data` accepts data that is already masked.
- `tensor4all_core::outer_product` promotes mixed dtypes, and
  `IdxTensor::onehot` always builds `f64`. A one-hot factor must therefore be
  built from the patch scalar type `T`, not with `onehot`, or a patch of a
  different dtype becomes inhomogeneous and fails with `DTypeMismatch`.
- `tensor4all_core::CachedFunction` requires an infallible, `Send + Sync`
  point function and a `'static` batch function over `&[Vec<I>]`, and its keys
  stop at 1024 bits. It cannot wrap a fallible column-major evaluator, and its
  entries can only be cleared all at once. Core's key-width selection is
  crate-private, and `tensor4all-partitionedtreetn` has no wide-integer
  dependency.
- The topology and site checks of the M1 contract (tree shape, node set equal
  to the `node_sites` keys, distinct full site identities, positive
  dimensions, at least one site) currently run only inside
  `InterpolationProblem::new`.
- `PartitionedTreeTN` treats an absent patch as zero and requires disjoint
  projectors, not coverage; `from_subdomains(vec![])` is valid and later
  operations on an empty partition return `Empty`. `from_subdomains` checks all
  patch pairs for overlap.
- `PatchingOptions::patch_order` in the same crate accepts a partial order.
- `tensor4all-partitionedtreetn` has no random-number dependency.
  REPOSITORY_RULES.md requires a named RNG algorithm in seed-based production
  code and a caller-owned `&mut R` API for randomized algorithms.
- Design lineage and provenance: the chain driver
  `tensor4all-partitionedtt::adaptiveinterpolate` states that its queue, split
  flow, and pivot recycling derive from TCIAlgorithms.jl. The approved records
  ([partitioned-treetn.md](./partitioned-treetn.md), the M1 seam record) state
  that the M2 driver derives its queue from that lineage and carries the
  derivation notice and `LICENSE-TCIALGORITHMS-MIT`. The chain crate is not a
  verification baseline.

## Public surface

A new module `tensor4all_partitionedtreetn::adaptive_interpolation`:

```rust
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchedInterpolationOptions {
    pub rtol: f64,
    pub reference_scale: Option<f64>,
    pub max_bond_dim: usize,
    pub patch_order: Vec<DynIndex>,
    pub n_initial_pivots: usize,
    pub recycle_pivots: bool,
    pub seed: u64,
    pub max_patches: Option<usize>,
}
// Built with `PatchedInterpolationOptions::new(max_bond_dim)`, which sets the
// defaults below, plus `with_*` builders; there is no `Default` because the
// bond cap has no sensible default.

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchRecord {
    pub projector: Projector,
    pub termination: InterpolationTermination,
    pub error_estimate: f64,
    pub max_sample_magnitude: f64,
    pub max_bond_dim: usize,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchedInterpolationReport {
    pub reference_scale: f64,
    pub accepted: Vec<PatchRecord>,
    pub zero_projectors: Vec<Projector>,
    pub splits: usize,
    pub function_evaluations: usize,
    pub cache_hits: usize,
}

#[derive(Debug)]
pub struct PatchedInterpolationResult<V> {
    pub partition: PartitionedTreeTN<V>,
    pub report: PatchedInterpolationReport,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PatchedInterpolationError {
    InvalidInput { message: String },
    Interpolation { projector: Projector, source: InterpolationError },
    Partition { source: PartitionedTreeTNError },
    NoSplitIndexLeft { projector: Projector },
    ResourceLimit { resource: &'static str, limit: usize },
}

pub fn patched_interpolate<T, V, E, F>(
    engine: &E,
    topology: NodeNameNetwork<V>,
    node_sites: BTreeMap<V, Vec<DynIndex>>,
    initial_pivots: ColMajorArray<usize>,
    evaluate: F,
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError>
where
    T: /* scalar bounds of IdxTensor data and magnitudes */,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T> + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>> + Send + Sync;
```

The exact scalar bound set is the minimum needed for magnitudes, zero tests,
one-hot factors, and dense construction, fixed in implementation. `E: Sync`
and `F: Send + Sync` are required now so that M7 adds no bound.

### Options

| Field | Meaning | Default and guidance |
|---|---|---|
| `rtol` | Relative tolerance; the engine's absolute tolerance is `rtol * reference_scale`. `0` is allowed and splits until every patch is exact | `1e-8` |
| `reference_scale` | Scale for every patch. `None` pins it to the largest magnitude of the root patch's candidate samples, a sampled lower bound on `max |f|`; for localized functions this makes the tolerance tighter than intended, so passing a known scale is recommended | `None` |
| `max_bond_dim` | Bond cap per patch, at least 2 (a cap of one could only accept patches nonzero on a single node) | required; a smaller cap means more, smaller patches |
| `patch_order` | Sites fixed when a patch splits, in order; partial orders are allowed. Empty means the derived site order | empty |
| `n_initial_pivots` | Target number of distinct initial pivots per patch, at least 1 | `5` |
| `recycle_pivots` | Seed children with the parent's pivots | `false` |
| `seed` | Root seed | `0` |
| `max_patches` | Limit on processed patches (accepted, zero, fully fixed, and split alike); `Some(0)` is invalid | `None` |

`rtol` and `max_bond_dim` trade off: a tighter tolerance or a smaller cap
produces more patches.

### Error criterion

Acceptance uses the engine's sampled pivot-error criterion against
`rtol * reference_scale`. It is not a certified bound and makes no L2 claim;
rustdoc states this. M3 adds a user-selectable error norm with the measured
L2 error as the default (Decision 3); the options and report types are
`#[non_exhaustive]` so that M3 extends them without silently changing the
meaning of `rtol` and `reference_scale`.

**Superseded in M3** ([tree-patching-error-contract.md](./tree-patching-error-contract.md)).
This criterion is now `ErrorNorm::SampledMax`: `rtol` moved into
`tolerance: ErrorTolerance { rtol, atol }`, `reference_scale` became
`ErrorNorm::SampledMax { max_reference }` with engine tolerance
`max(atol, rtol * max_reference)`, `PatchRecord::error_estimate` became
`engine_error_estimate`, and `report.reference_scale` / `zero_projectors`
became `report.norm` / `zero_patches`. `ErrorNorm::sampled_max()` reproduces
the M2 outputs (checked against frozen golden outputs); the default is the
driver-measured L2 norm. The rest of this record describes M2 as built.

## Algorithm

1. **Validate** the inputs before any evaluation. The topology and sites are
   checked by a public helper added to the M1 module,
   `tensor4all_treetn::interpolation::validate_layout(&topology, &node_sites)`,
   which `InterpolationProblem::new` also calls, so the driver and the
   contract share one validator and no fast path can skip it. Its
   `InterpolationError::InvalidProblem` is reported as
   `PatchedInterpolationError::InvalidInput` with the same message. Then:
   `patch_order` entries are distinct site indices of
   the problem (full identity); `rtol` is finite and nonnegative;
   `reference_scale`, if given, is finite and positive; `max_bond_dim >= 2`;
   `n_initial_pivots >= 1`; `max_patches` is not `Some(0)`; `initial_pivots`
   has one row per site and coordinates in range (zero columns are allowed).
   The site order is `InterpolationProblem::derive_site_order(&node_sites)`.
2. **Queue.** Start with the root patch (empty projector) and process patches
   in FIFO order, counting every processed patch against `max_patches`
   (`ResourceLimit` when exceeded).
3. **Cache.** Each patch owns an evaluation cache keyed by its active
   coordinates packed into a variable-length key (a boxed slice of `u64`
   words, each holding as many coordinates as fit), so any domain size is
   supported without a width limit or a new dependency. When a patch splits,
   its cache is partitioned among its children in one pass; when a patch is
   accepted or found to be zero, its cache is dropped. An evaluator error is
   returned immediately; nothing is cached for it.
4. **Candidates.** Keep, in order and without duplicates, the user pivots and
   (if enabled) the recycled parent pivots that are compatible with the
   patch's projector; then add random points inside the patch up to
   `n_initial_pivots`. The number of points in a patch is computed with
   saturating arithmetic (it only needs to be compared with the target).
   Random candidates use bounded rejection attempts and then fall back to the
   first unused points in column-major order. Each random coordinate in
   `0..d` is drawn from the SplitMix64 stream with Lemire's multiply-shift
   method with rejection (unbiased), which fixes the reproducible stream.
5. **Sample checks.** Every value the driver samples (candidates and the
   exact values of step 8) is checked for finiteness before it is used; a
   non-finite value is reported as
   `Interpolation { projector, source: InterpolationError::Evaluator }`.
6. **Reference scale.** If not given, it is pinned from the root patch and
   reused for every patch: from all exact values when the root itself is
   handled exactly by step 8, otherwise from its candidate samples. An exact
   root never needs the scale, and an all-zero exact root returns an empty
   partition with its projector in `zero_projectors`; if no `reference_scale`
   was given, `report.reference_scale` is `0.0` in that case. A non-exact root with
   all candidate samples exactly zero cannot pin a scale and returns
   `InvalidInput` with the remedy to pass `reference_scale` or pivots in the
   support.
7. **Zero screening.** Patches handled exactly by step 8 are screened on all
   their values instead of candidates. For other patches, if every candidate
   sample is exactly zero,
   the patch is a zero patch: its projector is recorded in
   `zero_projectors` and it is omitted from the partition. Exact zero matches
   M1's `AllSamplesZero` rule (the chain lineage used a `1e-30` threshold).
   This is a finite-sampling policy; sparse functions need pivots in their
   support. An engine that still returns `AllSamplesZero` after screening is
   propagated as an `Interpolation` error.
8. **Exact small patches.** A patch with no active site is one value; a patch
   with exactly one active site has `d` values. Both are evaluated directly and
   built as a network with dimension-one links: the values on the node that
   carries the active site (or on the smallest node name if none), and a
   one-hot factor for every fixed site on its own node. No engine call. The
   exact values are evaluated before zero screening: if all are exactly zero
   the patch is a zero patch, otherwise it is accepted. Its `PatchRecord` has
   `termination = Converged`, `error_estimate = 0`, the largest evaluated
   magnitude, and `max_bond_dim = 1`.
9. **Interpolate.** Otherwise build an `InterpolationProblem` with the active
   sites, the candidates in active coordinates,
   `absolute_tolerance = rtol * reference_scale`, the bond cap, and the engine
   seed of the patch. The problem's evaluator inserts the fixed coordinates and
   goes through the patch cache.
10. **Accept or split.** (Superseded in part: M3 added the L2 measurement,
    and the M5 patch-size bounds of 2026-10-04 accept small capped patches
    and retain patches at a minimum size; see
    [tree-pqtci-patch-size-bounds.md](./tree-pqtci-patch-size-bounds.md).)
    `Converged` is accepted. Any other verdict splits the
    patch at the next site of `patch_order` that is not fixed yet, one child per
    coordinate; if no site of `patch_order` is left, the driver returns
    `NoSplitIndexLeft`. With `recycle_pivots`, the outcome's pivots (active
    sites only) are completed with the patch's fixed coordinates and passed to
    the children, which keep those compatible with their split coordinate.
11. **Re-embed.** An accepted outcome network gets every fixed site
    re-attached to its original node by an outer product with a one-hot vector
    built from `T`, then is wrapped with the crate-internal `from_masked_data`
    (the data is already masked). This helper is shared by all engines.
12. **Order.** `report.accepted` and `report.zero_projectors` are sorted by
    patch path (lexicographic over the (position, coordinate) pairs), a
    canonical order independent of processing order, so M7 can guarantee
    identical reports.
13. **Assemble** the accepted patches with a crate-internal constructor that
    skips the pairwise overlap check, because the queue produces disjoint
    projectors by construction; a debug assertion keeps the check in tests.
    An all-zero run returns an empty partition.

The partition stays in the eager (masked) form required by the current
`partitionedtreetn` invariant; M4 decides whether that changes.

## Randomness and determinism

The driver's generator is SplitMix64, a named algorithm implemented in the
driver, so no dependency is added (roadmap Decision 1). Each patch derives two
sub-seeds, one for candidate sampling and one for the engine, by mixing the
root seed with the patch path, encoded as pairs of (position in the derived
site order, coordinate); the path never uses `DynIndex` IDs, so the encoding
survives the adaptive split sites of M5. Tests that need random data use
`ChaCha8Rng`, through dev-only dependencies on the workspace `rand` and
`rand_chacha`;
the library itself gains no dependency.

Exception to the caller-owned `&mut R` rule: a single caller stream would make
each patch's randomness depend on the processing order, so parallel execution
(M7) could not reproduce sequential results. The driver therefore offers only
the seed API and documents this exception in rustdoc.

For a fixed seed, a deterministic evaluator, and a deterministic engine, the
partition and report are identical across runs.

## Provenance

Following the approved records, the driver is derived from the TCIAlgorithms.jl
lineage through `tensor4all-partitionedtt`. The M2 PR:

- adds the derivation notice to the module header and copies
  `LICENSE-TCIALGORITHMS-MIT` into `crates/tensor4all-partitionedtreetn/`;
- adds a "Derived (MIT)" row for `tensor4all-partitionedtreetn` adaptive
  interpolation to `docs/PROVENANCE_AND_CITATION_POLICY.md` and rewrites the
  statement there that the crate does not contain adaptive interpolation.

## Public-surface updates in the M2 PR

- `tensor4all_treetn::interpolation::validate_layout` (new public helper, with
  rustdoc and an asserted example); `InterpolationProblem::new` delegates its
  topology and site checks to it without changing behavior. The M1 record
  [tree-interpolation-engine-seam.md](./tree-interpolation-engine-seam.md)
  lists the helper in its contract and validation sections.
- `crates/tensor4all-partitionedtreetn/README.md` and the crate docs in
  `src/lib.rs`, which say the crate provides no TCI or sampled-zero inference.
- `docs/book/src/guides/partitioned-treetn.md` (same statement; add the entry
  point) and `docs/book/src/architecture.md`.
- `skills/use-tensor4all-rs/SKILL.md` and `references/crates.md`, and
  `llms.txt`.

## Tests

- A driver-local test engine in `tensor4all-partitionedtreetn`: it builds the
  exact network from dense samples (supporting nodes without active sites) and
  reports `BondCapReached` above a configured rank, so splitting is tested
  without a real engine.
- TreeTCI end to end through a path-only dev-dependency on
  `tensor4all-treetci` (as the crate already does for
  `tensor4all-quanticstransform`).
- Topologies: a chain, a branched tree with a node of degree three or more,
  a node with several sites, a node without sites in the caller's topology,
  and a single-node topology. Splits at a leaf, an internal node, the junction,
  and a multi-site node, including a node whose sites all become fixed.
- A function with localized features on a small quantics grid whose monolithic
  rank exceeds the cap; the assembled partition is compared with a dense
  reference (materialize once, subtract, `maxabs`) with the bound
  `10 * rtol * reference_scale`, recorded as a test constant.
- A vanishing region: zero projectors are reported, the accepted and zero
  projectors are disjoint, and together they cover the domain.
- The exact small-patch paths, a complex scalar type with homogeneous
  re-embedded dtypes, recycling on and off, and determinism across two runs
  compared by projector key.
- Evaluation counts show no duplicate evaluation of a point.
- Errors: `NoSplitIndexLeft` (partial order), `ResourceLimit`, an unpinnable
  scale, a driver-side evaluator failure and a non-finite sample (including in
  the exact small-patch path), and each `InvalidInput` branch, including every
  `validate_layout` rejection reached through a root with at most one site.
- A cache over a domain wider than 128 bits (for example three variables of
  43 bits) works.
- A one-site patch whose local dimension exceeds `n_initial_pivots` and is
  nonzero at a single coordinate is accepted, not screened as zero, both below
  the root and as the root without `reference_scale` (the scale is pinned from
  the exact values); an all-zero exact root returns an empty partition and,
  without `reference_scale`, reports `reference_scale == 0.0`.
- Random candidate coordinates follow the documented SplitMix64 and Lemire
  mapping (a fixed seed gives a fixed candidate list).
- Reports are in canonical path order.
- Index identity: sites sharing an ID but differing in prime level or tags.
- Rustdoc: runnable, asserted examples for every public item and `# Errors`
  naming the variants.

## Non-goals

- Parallel execution (M7), error norms and certified bounds (M3), the patch
  representation decision (M4), and split strategies other than a fixed order
  (M5).
- Retiring or changing `tensor4all-partitionedtt`.

## Names

The review kept the proposed names: `adaptive_interpolation`,
`patched_interpolate`, and the `Patched*` types, chosen to avoid
`AdaptiveInterpolationResult` in the chain crate.

## Implementation decisions

- **Scalar bounds.** `T: tensor4all_core::CommonScalar + TensorElement`.
  Magnitudes and the exact-zero test use `CommonScalar::abs_val` (the
  hypotenuse for complex values). A sampled value is rejected with one of two
  messages: "non-finite value" when a component is infinite or NaN (checked
  per component as `value * 0 == 0`), and "magnitude overflows" when every
  component is finite but `abs_val` is not (a complex value near the largest
  float in both parts). Both are `Interpolation { source: Evaluator }`, so
  every magnitude the driver uses (the pinned reference scale, zero
  screening, the maximum sample) and every value the engine receives has a
  finite magnitude. One-hot factors and exact networks are built with
  `IdxTensor::from_dense::<T>`.
- **Where values are checked.** The per-patch cache is the only path to the
  evaluator, so the count and finiteness checks of step 5 sit there and also
  cover the samples the engine requests, not only the candidates and exact
  values. A failure is returned to the engine as an evaluator error, which the
  M1 contract reports as `InterpolationError::Evaluator`. The cache also
  rejects an engine batch with the wrong number of rows or an out-of-range
  coordinate the same way, before the evaluator sees it. Nothing is cached for
  a failed batch.
- **Engine outcomes are checked, not trusted.** An accepted outcome must carry
  the problem's nodes and edges and exactly the active sites of every node
  (full identity and dimension); returned pivots are checked (one row per
  active site, in-range coordinates) only when `recycle_pivots` is on. An
  outcome reported as `Converged` whose re-embedded patch has a bond dimension
  at or above `max_bond_dim` is rejected, because the M1 contract defines
  `Converged` as strictly below the cap; it is not treated as a split. A
  mismatch is `Interpolation { projector, source: InterpolationError::Engine }`.
- **`patch_order` identity.** An entry that matches a site's identity (ID,
  tags, prime level) but not its dimension is rejected as `InvalidInput`, as
  the rest of the crate rejects equal-identity aliases with another dimension.
- **Seeds and keys.** `mix(x)` is the first SplitMix64 output of the state
  `x`. A patch absorbs its path into `mix(root_seed ^ PATH_DOMAIN)` pair by
  pair (`s = mix(s ^ position)`, then `s = mix(s ^ coordinate)`), and its
  candidate and engine sub-seeds are `mix(s ^ CANDIDATE_STREAM)` and
  `mix(s ^ ENGINE_STREAM)` with the constants in
  `src/adaptive_interpolation/sampling.rs`. Random candidates get
  `20 * missing + 100` attempts before the column-major fallback. A cache key
  gives each coordinate `bits(d - 1)` bits (none for `d = 1`) without
  straddling a word. Unit tests pin these streams against an independent
  implementation.
- **No patch without an active site.** Only patches with at least two active
  sites split, so every child keeps at least one; the no-active-site case of
  step 8 is unreachable through the driver. The exact path enumerates the
  points of a patch without distinguishing the two cases, and the network
  builder for it is unit-tested directly.
- **Counters.** `function_evaluations` counts the points passed to the
  evaluator; `cache_hits` counts requested points served without it, including
  repeats inside one batch.
- **Cache keys and lookups.** Step 3's variable-length key is stored inline
  when it has at most two words: a cache whose layout needs zero or one word
  keys a `u64` map, two words a `u128` map, and only three or more words a
  boxed slice. A lookup checks and packs the requested point into one buffer
  reused across the batch and borrows it, so a cache hit allocates nothing;
  the points new to a batch are kept in a map from key to their index among
  the new points, and their keys move into the cache once the values pass the
  checks. Every map hashes with a local unseeded word hasher (a
  rotate-xor-multiply fold and the MurmurHash3 64-bit finalizer) instead of
  SipHash, with no new dependency; the keys come from the driver and the
  engine, so HashDoS resistance is not needed. No output depends on the
  iteration order of a cache: a split or an insertion produces the same
  key-to-value mapping in any order, and the reports use only counts.
- **Cache split.** A child keeps its parent's packing with the bits of the
  split coordinate cleared, so a split moves every entry with its key words
  (for a boxed key, in place) instead of decoding and re-encoding it. Only
  when a compact packing of the child's coordinates would need fewer words is
  the child re-encoded into it, so a root wider than two words reaches the
  inline keys after enough splits; the unit test
  `cache_split_re_encodes_only_when_a_word_is_freed` covers that path (the
  129-site integration test accepts its root without a split). A child key
  therefore no longer equals the compact packing of its own coordinates,
  which no caller relies on; the counters and the
  determinism of the run are unchanged. Measured on the branched quantics
  tree workload of the tree-patching runner (L2, `rtol = 1e-4`, `eta = 0.3`,
  cap 32, release, one pinned core, all thread variables set to 1): `R = 7`
  went from 32.8 s to 15.6 s and `R = 8` from 95.9 s to 46.7 s, with
  identical stored tensors, projectors, and counts; the cache cost per
  requested point fell from about 216 ns to 69 ns (`R = 7`) and from 253 ns
  to 84 ns (`R = 8`), and splitting from 2.2 s to 0.12 s and from 9.1 s to
  0.49 s.
- **Assembly.** `from_disjoint_subdomains` still checks topology, site space,
  and dtype against the first patch (linear in the patch count) and skips only
  the pairwise overlap check. Violated internal invariants, which the queue
  rules out, are reported as `Partition` errors rather than panics.
- **Tolerance overflow.** `rtol * reference_scale` can overflow for finite
  inputs; `InterpolationProblem::new` then rejects the root problem, reported
  as `Interpolation { source: InvalidProblem }`, since a pinned scale is only
  known after the root samples.
- **Tests.** The domain wider than 128 bits is a chain of 129 binary sites
  (three 43-bit variables) run with a test engine that builds the exact
  rank-one network of a product function from fibers: TreeTCI took about
  300 s for one run on that chain in a debug build.
- **Determinism, verified.** Two runs with the same seed store bitwise
  identical patches: per projector and per node, the same legs in the same
  positional order (sites by position, bonds by neighbor and dimension) and
  the same raw column-major data, with an identical report. The tests assert
  this on the degree-three trees with the dense test engine and with TreeTCI.
  A one-off manual check, with no committed test, also found identical
  digests of the same comparison in three separate test processes. The claim
  covers the report and the stored node tensors only. The stored patches are
  `TreeTN`s built through `TreeTN::from_tensors`. Issue
  [#791](https://github.com/tensor4all/tensor4all-rs/issues/791) was subsequently
  fixed by #793: site/edge construction and dense output axis order are stable.
  `site_space` remains a set with unspecified iteration order. The M3 extension
  adds a separate cached-evaluator contraction-path determinism prerequisite;
  see [the error contract](./tree-patching-error-contract.md#determinism).
