# TreeTN AD Coverage Expansion Design

**Issue:** `#375`

## Goal

Broaden TreeTN AD integration coverage with a shared real/complex test harness, then use that harness to validate representative TreeTN state operations under the documented forward and reverse complex conventions.

## Approaches

### Approach 1: Bespoke per-operation tests

Add separate `f64` and `Complex64` tests for each operation with local helpers in each test.

Pros:
- Fastest to start
- Lowest short-term abstraction cost

Cons:
- Duplicates test logic aggressively
- Makes convention drift between real and complex paths likely
- Scales poorly once operator and contraction paths are added

### Approach 2: Generic scalar harness with thin wrappers

Add a small test-only scalar trait that centralizes:
- sample data generation
- finite-difference perturbations
- forward-mode readout
- reverse-mode real-valued loss construction
- dense/scalar comparison helpers

Then write each operation test body once and expose thin `f64` / `Complex64` entrypoints.

Pros:
- Matches `AGENTS.md` generic test guidance
- Keeps real/complex conventions explicit in one place
- Makes it cheap to add more representative operations later

Cons:
- Slightly more upfront design work
- Not every operation fits the exact same backward harness

### Approach 3: Jump directly to higher-level algorithmic/operator paths

Build the generic harness and immediately cover `apply_linear_operator`, `contract_fit`, and `contract_zipup` in the first pass.

Pros:
- Maximizes user-facing algorithm coverage quickly

Cons:
- Larger failure surface
- Harder to separate harness bugs from algorithm bugs
- More likely to turn one issue into multiple debugging threads

## Recommendation

Use Approach 2 first, and defer Approach 3 to follow-up work inside the same issue.

The first implementation batch should cover representative TreeTN state operations:
- `to_dense`
- `canonicalize`
- `truncate`
- `add`
- `evaluate`
- `swap_site_indices`
- `inner` where the AD convention remains unambiguous

Higher-level algorithmic/operator coverage should come only after the generic harness is stable.

## Complex AD Convention

TreeTN integration tests should follow `../chainrules-rs`:
- forward-mode complex checks use standard JVP on `C ~= R^2`
- reverse-mode complex checks use conjugate-Wirtinger behavior for real-valued losses

That means complex backward tests must avoid arbitrary complex-valued losses. The test harness should instead build explicitly real-valued objectives such as norm-like or squared-residual losses.

## Test Shape

Use a shared small TreeTN fixture that is easy to perturb elementwise and easy to materialize densely. Favor fixed, deterministic topologies over randomized fixtures for the first pass.

For each covered operation:
- forward-mode: verify tangent preservation and/or finite-difference agreement
- reverse-mode: verify gradient propagation and finite-difference agreement when the operation returns a tensor and the loss can be made real-valued

## Non-Goals

This pass does not attempt to fully cover:
- `apply_linear_operator`
- `contract_fit`
- `contract_zipup`
- `square_linsolve`

Those remain follow-up scope once the generic state-operation harness is in place.
