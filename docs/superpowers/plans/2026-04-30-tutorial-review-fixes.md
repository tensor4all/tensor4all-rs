# Tutorial Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the April 30 tutorial review findings into concrete mdBook and repository-rule fixes so online tutorials stay beginner-friendly, executable, and synchronized with `docs/tutorial-code`.

**Architecture:** Keep the live online tutorial source in `docs/book/src/tutorials/`. Keep runnable full demos and generated artifacts in `docs/tutorial-code/`. mdBook code blocks should use public tensor4all-rs APIs, while `docs/tutorial-code/src/*_common.rs` files may be used as implementation references.

**Tech Stack:** Rust mdBook snippets, `mdbook test` via `./scripts/test-mdbook.sh`, Cargo release tests for `tensor4all-tutorial-code`, repository policy markdown.

---

## Files And Responsibilities

- `docs/book/src/tutorials/computations-with-qtt/partial-fourier2d.md`: fix the largest code/prose mismatch by showing an interleaved two-variable grid and the x-site mapping explicitly.
- `docs/book/src/tutorials/quantics-basics/qtt-definite-integrals.md`: replace the weak positivity assertion with an analytic integral assertion.
- `docs/book/src/tutorials/computations-with-qtt/elementwise-product.md`: replace the structural-only assertion with a pointwise product evaluation assertion.
- `docs/book/src/tutorials/computations-with-qtt/affine-transformation.md`: replace the output-exists assertion with explicit output-structure assertions.
- `docs/book/src/tutorials/computations-with-qtt/fourier-transform.md`: add a stronger Fourier-output assertion and mention bit-reversed output.
- `AGENTS.md`: add the agent-facing tutorial maintenance rule.
- `REPOSITORY_RULES.md`: add the durable tutorial maintenance rule.
- `docs/tutorial-code/README.md`: update the tutorial order to all nine live online tutorials and clarify legacy tutorial markdown status.
- `scripts/refresh-tutorial-artifacts.sh`: add one deliberate command for regenerating tutorial data, plots, copying PNGs, and verifying mdBook examples.

Do not change test tolerances unless the user explicitly approves. Do not add `ignore` or `no_run` to mdBook Rust blocks.

## Task 1: Strengthen The Definite Integral Tutorial Assertion

**Files:**
- Modify: `docs/book/src/tutorials/quantics-basics/qtt-definite-integrals.md`

- [x] **Step 1: Inspect the current snippet**

Run:

```bash
sed -n '1,90p' docs/book/src/tutorials/quantics-basics/qtt-definite-integrals.md
```

Expected: the snippet contains `let integral = qtt.integral()?;` followed by `assert!(integral > 0.0);`.

- [x] **Step 2: Replace the weak assertion**

Replace:

```rust
let integral = qtt.integral()?;
assert!(integral > 0.0);
```

with:

```rust
let integral = qtt.integral()?;
let exact = 3.0;
assert!((integral - exact).abs() < 8e-2);
```

Rationale: the analytic integral of `x^2` on `[-1, 2]` is `3.0`; the tolerance matches the existing `tensor4all-tutorial-code` verification threshold for this grid approximation and is strong enough to catch a broken call.

- [x] **Step 3: Verify the mdBook snippet**

Run:

```bash
./scripts/test-mdbook.sh
```

Expected: command exits with status 0.

## Task 2: Fix The 2D Partial Fourier Tutorial Code/Prose Mismatch

**Files:**
- Modify: `docs/book/src/tutorials/computations-with-qtt/partial-fourier2d.md`
- Reference only: `docs/tutorial-code/src/qtt_partial_fourier2d_common.rs`

- [x] **Step 1: Inspect the real runnable workflow**

Run:

```bash
rg -n "build_input_grid|build_partial_fourier_operator|rename_operator_nodes|expand_operator_to_interleaved_state|transform_x_dimension|x_site_node_mapping" docs/tutorial-code/src/qtt_partial_fourier2d_common.rs
```

Expected: all listed helper names appear. Use this file as the source of truth for the algorithmic shape.

- [x] **Step 2: Replace the misleading prose before the code block**

Replace the paragraph below `## Key API Pieces` with:

```markdown
For an interleaved two-variable QTT, the state nodes are ordered
`x0, t0, x1, t1, ...`. A one-dimensional Fourier MPO has only `x` nodes, so the
operator nodes must be renamed onto the even state nodes before the operator is
expanded with identity tensors on the `t` nodes. The runnable source linked
above performs that expansion in `transform_x_dimension`.
```

- [x] **Step 3: Replace the current code block with a public-API skeleton that matches the prose**

Use a short block that compiles and names the required mapping. Keep it on public APIs; do not import `tensor4all_tutorial_code` into mdBook snippets.

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_quanticstci::{
#     quanticscrossinterpolate, DiscretizedGrid, QtciOptions, UnfoldingScheme,
# };
# use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
# use tensor4all_simplett::AbstractTensorTrain;
let bits = 4;
let grid = DiscretizedGrid::builder(&[bits, bits])
    .with_variable_names(&["x", "t"])
    .with_lower_bound(&[-4.0, 0.0])
    .with_upper_bound(&[4.0, 1.0])
    .include_endpoint(true)
    .with_unfolding_scheme(UnfoldingScheme::Interleaved)
    .build()?;

let f = |coords: &[f64]| -> f64 {
    let x = coords[0];
    let t = coords[1];
    (-0.5 * x * x).exp() * (2.0 * std::f64::consts::PI * t).cos()
};
let options = QtciOptions::default()
    .with_nrandominitpivot(0)
    .with_unfoldingscheme(UnfoldingScheme::Interleaved)
    .with_verbosity(0);
let pivots = vec![vec![1_i64, 1_i64], vec![8, 8]];
let (state, _ranks, _errors) = quanticscrossinterpolate(&grid, f, Some(pivots), options)?;

let operator = quantics_fourier_operator(bits, FourierOptions::forward())?;
let x_site_mapping: Vec<_> = (0..bits).map(|site| (site, 2 * site)).collect();

assert_eq!(state.tensor_train().len(), 2 * bits);
assert_eq!(operator.mpo.node_count(), bits);
assert_eq!(x_site_mapping, vec![(0, 0), (1, 2), (2, 4), (3, 6)]);
# Ok(())
# }
```

- [x] **Step 4: Add one sentence after the code block**

Add:

```markdown
The full source then renames the operator nodes with this mapping, expands the
operator with identity tensors on the odd `t` nodes, aligns the resulting
operator to the state, and applies it.
```

- [x] **Step 5: Verify the page**

Run:

```bash
./scripts/test-mdbook.sh
```

Expected: command exits with status 0.

## Task 3: Strengthen Advanced Tutorial Assertions

**Files:**
- Modify: `docs/book/src/tutorials/computations-with-qtt/elementwise-product.md`
- Modify: `docs/book/src/tutorials/computations-with-qtt/affine-transformation.md`
- Modify: `docs/book/src/tutorials/computations-with-qtt/fourier-transform.md`

- [x] **Step 1: Elementwise product assertion**

In `elementwise-product.md`, keep `assert_eq!(product.node_count(), 3);` only if a numerical assertion is added immediately before it.

Add this import to the hidden imports:

```rust
# use tensor4all_core::ColMajorArrayRef;
```

Add this assertion near the end of the code block:

```rust
let shape = [site_indices_a.len(), 1];
let site_values = ColMajorArrayRef::new(&[0usize, 1, 1], &shape);
let value = product.evaluate_at(&site_indices_a, site_values)?;
let expected = f_a(&[4_i64]) * f_b(&[4_i64]);
assert!((value[0].real() - expected).abs() < 1e-8);
```

The site values `[0, 1, 1]` are the three-bit quantics representation of the
one-based grid index `4`.

- [x] **Step 2: Affine transformation assertion**

In `affine-transformation.md`, replace the final assertion:

```rust
let external = TensorIndex::external_indices(&result);
assert_eq!(external.len(), 3);
```

with:

```rust
let external = TensorIndex::external_indices(&result);
assert_eq!(external.len(), 3);
assert!(result.node_count() >= state.node_count());
```

This keeps the snippet short while checking the transformed state exposes the
expected number of external output indices and does not collapse the TreeTN
structure.

- [x] **Step 3: Fourier transform assertion and note**

In `fourier-transform.md`, add this paragraph after the code block:

```markdown
The quantics Fourier operator follows the bit-reversed output convention
described in the Quantum Fourier Transform guide.
```

Then replace:

```rust
assert!(result.node_count() > 0);
```

with:

```rust
assert_eq!(result.node_count(), bits);
```

- [x] **Step 4: Verify all three pages**

Run:

```bash
./scripts/test-mdbook.sh
```

Expected: command exits with status 0.

## Task 4: Add Explicit Tutorial Maintenance Rules

**Files:**
- Modify: `AGENTS.md`
- Modify: `REPOSITORY_RULES.md`
- Modify: `docs/tutorial-code/README.md`

- [x] **Step 1: Add an agent-facing rule to `AGENTS.md`**

Under `### Public Surface Drift`, add:

```markdown
### Online Tutorial Synchronization

- The live online tutorials are in `docs/book/src/tutorials/`.
- Runnable tutorial demos live in `docs/tutorial-code/src/bin/` with shared
  helpers in `docs/tutorial-code/src/`.
- When changing public APIs, tutorial code, generated tutorial CSV/PNG
  artifacts, or examples quoted by the online tutorials, update the live mdBook
  tutorial page in the same branch.
- Do not update only `docs/tutorial-code/docs/tutorials/`; those markdown files
  are legacy/reference material unless this policy is changed explicitly.
```

- [x] **Step 2: Add the durable rule to `REPOSITORY_RULES.md`**

Under `## Documentation Examples`, add:

```markdown
## Online Tutorial Synchronization

- `docs/book/src/tutorials/` is the live source for online tutorial prose.
- `docs/tutorial-code/src/bin/` and `docs/tutorial-code/src/` are the runnable
  source for tutorial demos and shared tutorial helpers.
- Any change to tutorial APIs, tutorial code, generated tutorial artifacts, or
  public APIs used by tutorials must check and update the corresponding live
  mdBook page before the branch is complete.
- Legacy markdown under `docs/tutorial-code/docs/tutorials/` must not be treated
  as the online source of truth.
```

- [x] **Step 3: Update `docs/tutorial-code/README.md` tutorial order**

Replace the seven-item suggested order with links to the nine live mdBook pages:

```markdown
Suggested online order:

1. [QTT of a Scalar Function](../book/src/tutorials/quantics-basics/qtt-scalar-function.md)
2. [QTT on a Physical Interval](../book/src/tutorials/quantics-basics/qtt-physical-interval.md)
3. [Definite Integrals](../book/src/tutorials/quantics-basics/qtt-definite-integrals.md)
4. [Sweep over Bit Depth](../book/src/tutorials/quantics-basics/sweep-bit-depth.md)
5. [Multivariate Functions](../book/src/tutorials/quantics-basics/multivariate-functions.md)
6. [Elementwise Product](../book/src/tutorials/computations-with-qtt/elementwise-product.md)
7. [Affine Transformation](../book/src/tutorials/computations-with-qtt/affine-transformation.md)
8. [Fourier Transform](../book/src/tutorials/computations-with-qtt/fourier-transform.md)
9. [2D Partial Fourier Transform](../book/src/tutorials/computations-with-qtt/partial-fourier2d.md)
```

Add after the list:

```markdown
The markdown files under `docs/tutorial-code/docs/tutorials/` are legacy
reference material. The online documentation is generated from
`docs/book/src/tutorials/`.
```

- [x] **Step 4: Verify the policy text**

Run:

```bash
rg -n "Online Tutorial Synchronization|docs/book/src/tutorials|legacy/reference" AGENTS.md REPOSITORY_RULES.md docs/tutorial-code/README.md
```

Expected: the new policy appears in all three files.

## Task 5: Add A Full Tutorial Artifact Refresh Script

**Files:**
- Create: `scripts/refresh-tutorial-artifacts.sh`

- [x] **Step 1: Create the script**

Create `scripts/refresh-tutorial-artifacts.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/docs/tutorial-code"

cargo run --release --bin qtt_function
cargo run --release --bin qtt_interval
cargo run --release --bin qtt_integral
cargo run --release --bin qtt_integral_sweep
cargo run --release --bin qtt_r_sweep
cargo run --release --bin qtt_multivariate
cargo run --release --bin qtt_elementwise_product
cargo run --release --bin qtt_affine
cargo run --release --bin qtt_fourier
cargo run --release --bin qtt_partial_fourier2d

julia --project=docs/plotting docs/plotting/qtt_function_plot.jl
julia --project=docs/plotting docs/plotting/qtt_interval_plot.jl
julia --project=docs/plotting docs/plotting/qtt_integral_sweep_plot.jl
julia --project=docs/plotting docs/plotting/qtt_r_sweep_plot.jl
julia --project=docs/plotting docs/plotting/qtt_multivariate_plot.jl
julia --project=docs/plotting docs/plotting/qtt_elementwise_product_plot.jl
julia --project=docs/plotting docs/plotting/qtt_affine_plot.jl
julia --project=docs/plotting docs/plotting/qtt_fourier_plot.jl
julia --project=docs/plotting docs/plotting/qtt_partial_fourier2d_plot.jl

cd "$repo_root"

cp docs/tutorial-code/docs/plots/qtt_function_vs_qtt.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_function_bond_dims.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_interval_function_vs_qtt.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_interval_bond_dims.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_integral_sweep.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_r_sweep_samples.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_r_sweep_runtime.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_multivariate_values.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_multivariate_error.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_multivariate_bond_dims.png docs/book/src/tutorials/quantics-basics/

cp docs/tutorial-code/docs/plots/qtt_elementwise_product_factors.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_elementwise_product_product.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_elementwise_product_bond_dims.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_affine_values.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_affine_error.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_affine_bond_dims.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_affine_operator_bond_dims.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_fourier_transform.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_fourier_bond_dims.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_fourier_operator_bond_dims.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_partial_fourier2d_values.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_partial_fourier2d_error.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_partial_fourier2d_bond_dims.png docs/book/src/tutorials/computations-with-qtt/

./scripts/test-mdbook.sh
cargo test --release -p tensor4all-tutorial-code

git status --short docs/tutorial-code/docs/data docs/tutorial-code/docs/plots docs/book/src/tutorials
```

- [x] **Step 2: Make it executable**

Run:

```bash
chmod +x scripts/refresh-tutorial-artifacts.sh
```

Expected: `test -x scripts/refresh-tutorial-artifacts.sh` succeeds.

- [x] **Step 3: Do not run the full refresh automatically**

The full refresh can be slow and needs Julia dependencies. Only run it if the user asks. For this task, verify shell syntax only:

```bash
bash -n scripts/refresh-tutorial-artifacts.sh
```

Expected: command exits with status 0.

## Task 6: Final Verification

**Files:**
- All changed files from Tasks 1-5

- [x] **Step 1: Format**

Run:

```bash
cargo fmt --all
```

Expected: command exits with status 0.

- [x] **Step 2: Check mdBook snippets**

Run:

```bash
./scripts/test-mdbook.sh
```

Expected: command exits with status 0.

- [x] **Step 3: Check tutorial-code tests**

Run:

```bash
cargo test --release -p tensor4all-tutorial-code
```

Expected: command exits with status 0.

- [x] **Step 4: Check changed files**

Run:

```bash
git status --short
git diff --stat
```

Expected: changed files are limited to tutorial markdown, repository policy markdown, the new refresh script, and any artifacts the user explicitly asked to refresh.

- [x] **Step 5: Commit only with user approval**

Repository rules say never push or create a PR without user approval. Also do not commit unless the user asked for a commit.
