# Shared sampling across the components of `quanticscrossinterpolate_multicomponent` (#1463)

The multicomponent entry point interpolated each output component with its own
memo, so a point sampled by several components was requested from the callback
once per component, although one callback call answers all components of a
point (#828/#747 deferred this). The component runs of one call now share a
single memo.

**Design.** One `MultiIndexCache<Arc<[V]>>` is created before the component
loop, keyed by the quantics multi-index of the one `DiscretizedGrid` all
components use, and dropped with the call. Each value block holds all
`n_components` values of a point; the payload declared to the cache is
`n_components * size_of::<V>()` plus the two counter words of the `Arc`, so the
default 256 MiB limit bounds tables and payload together. The seam is internal:
`site_evaluator` stores a whole-point value `W` and projects each run's own
`V` out of it, and `quantics_component_run` (pivot preparation and TCI run) is
shared by the single-component entry points and the component runs
(`SharedSampling` in `quantics_tci/shared_sampling.rs`). No public API changed.
A request sends only its distinct misses, in first-occurrence order and one
batch, to the callback; the value count is checked inside the miss closure, so a
wrong count or a callback error inserts nothing and aborts the call with the
same `QuanticsTCIError::Operation` as before.

**Identical fits.** Pivot choice, ranks, errors and random draws depend only on
the values and the options, and the only draws from the caller's stream are the
random initial pivots, made per component before any evaluation. Tensor trains,
ranks, errors and the final stream position are bit-identical to the old
per-component loop, which the tests keep as a local reference. Only the
callback sees a difference: the first component's calls are unchanged, later
ones send fewer or no batches. The callback must therefore be a pure function of
the point (documented on the entry points).

**Maintainer decisions (contract review on #1463).** The stats stay `pub(crate)`
(`SharedSampling::stats`, no `cache_stats()`). The purity precondition is on the
public docstrings and now also says the callback must be independent of batching
and call order (the memo sends one batch for several components, so a callback
that branches on `n_points` or on how often it is called behaves differently).
The 256 MiB limit is the crate-level `pub(crate) const
MULTICOMPONENT_SHARED_MEMO_BYTES` (value `DEFAULT_RETAINED_BYTE_LIMIT`), named in
prose in the docstring and not caller-tunable. A component that fails after
sampling aborts all components; today's code returned that component's own error
without saying which component failed, so the message now names it
(`component <k> of <n>: ...`, same variant, original error kept in the source
chain).

**Evaluated values.** Each call of the joint callback evaluates all components
of a point. Before sharing, component run `i` requested its sampled set `S_i` and
discarded the other components' values: `n_components * sum|S_i|` values. With the
memo it is `n_components * |union S_i|`. 2x3: 7524 -> 1926 values; 8x8:
840704 -> 27392; three components with partial overlap: 5850 -> 2856.

**Source-call counts** (points requested from the callback; no wall times;
`matrix_valued_source_call_counts` in `batched/tests/shared_sampling.rs`). Two
variables of 5 bits, `tolerance = 1e-8`, `max_bond_dim = 8`, no random pivots,
entries that are sums of two exponentials (quantics rank at most two):

| Output shape | Components | Per-component runs | Shared memo | Saving |
|---|---:|---:|---:|---:|
| 2x3 | 6 | 1254 | 321 | 3.9x |
| 8x8 | 64 | 13136 | 428 | 30.7x |

The counts are pinned exactly because this target is platform-stable: the search
converges far below the tolerance, so pivot choices do not hinge on rounding
noise. The test repeats each case with every value multiplied by
`1 + 1e-13 * noise` (far above one-ulp libm differences) and requires the same
counts. A first version of this test used sines of growing frequency under the
bond cap: that target is not low-rank within the tolerance, the pivots are
picked from rounding noise, and the counts differ between aarch64 (2133 -> 878)
and x86_64 CI (2201 -> 890); a relative perturbation of 1e-16 already moved them.
The per-component side moves as well, so this is solver sensitivity to the
target's last bits, not the sharing code. The shared count equals the number of
distinct points the components sample.

The saving grows with the number of components, as in the matrix-valued
experiment (`docs/experiments/issue-1463`): large for wide matrices, modest for
2x3.

Three components of different structure (rank-one product, low-rank sum,
localized bump) on the same grid, with component-dependent random initial pivots
(`n_random_init_pivot = 2`): the components sample 552, 560 and 838 points,
their union is 952 and the per-component total is 1950, so the callback receives
952 points. The components share part of their points (at least the origin
pivot) and each also samples points the others do not. With no random pivots the
counts are 448, 448, 771 sampled; union 864; per-component total 1667.

**Tests** (`batched/tests/shared_sampling.rs`, `f64` and `Complex64`): source
calls equal the union of the per-component sets and are fewer than their sum;
exact equality of every core, rank, error and the stream position against the
local per-component reference, through the internal seam and through the
public entry point; partial overlap of the sampled sets; one component equals
the plain batch fit; short, long and failing callback results on the first and a
later component insert nothing from the failed batch and abort; an all-zero
component aborts as before; payload accounting (`retained_bytes == tables + len *
payload`) and tiny limits (evictions, dropped inserts) with identical fits; a
grid wider than the cache key is rejected before any callback call.
