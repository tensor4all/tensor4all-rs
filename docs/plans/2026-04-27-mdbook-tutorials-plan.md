# Tutorial Code and mdBook Tutorials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the runnable Rust QTT tutorials into `docs/tutorial-code/` and add mdBook tutorial pages that stay consistent with that local code.

**Architecture:** `docs/tutorial-code/` becomes a local Rust tutorial crate with binaries, helpers, tests, data, plots, and refresh scripts. `docs/book/src/tutorials/` contains the online tutorial pages and selected PNGs copied from the local tutorial-code artifacts. The local tutorial code is the source of truth; mdBook pages explain and quote small tested pieces of it.

**Tech Stack:** Cargo, Rust 2021, local `path` dependencies to workspace crates, mdBook 0.5.2, Markdown, Julia/CairoMakie only for optional plot regeneration.

**Key decisions (see design doc `docs/plans/2026-04-27-mdbook-tutorials-design.md`):**
- Use flat `docs/tutorial-code/` rather than preserving the old `rust-Tensor4all` name.
- Convert all `tensor4all-*` dependencies to local `path` dependencies.
- Keep `tt_basics.rs` as optional runnable prelude code, but exclude `tt_basics_tutorial.md` from online navigation.
- mdBook Rust snippets must compile through `./scripts/test-mdbook.sh` and include assertions.
- Push, PR creation, and auto-merge require explicit user approval and are not part of the default execution.

---

### Task 0: Prepare branch and API reference

**Files:**
- Generate locally, do not commit: `docs/api/*.md`
- Read: `README.md`, `REPOSITORY_RULES.md`, relevant `docs/api/*.md`

- [ ] **Step 1: Fetch the current remote base**

Run:

```bash
git fetch origin
```

Expected: command completes successfully.

- [ ] **Step 2: Create or switch to a feature branch from `origin/main`**

Run:

```bash
git switch -c codex/tutorial-code-mdbook origin/main
```

Expected: new branch `codex/tutorial-code-mdbook` is checked out.

- [ ] **Step 3: Generate the API reference**

Run:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

Expected: generated API markdown appears under `docs/api/`. The directory is
ignored by git and is used only as a local reference unless repository policy
changes.

- [ ] **Step 4: Read the relevant API docs**

Read generated files for the crates used by the tutorials before touching
tutorial code:

```bash
ls docs/api
rg -n "quanticscrossinterpolate|quanticscrossinterpolate_discrete|DiscretizedGrid|QtciOptions|integral|evaluate|tensor_train|link_dims|affine_operator|quantics_fourier_operator|apply_linear_operator" docs/api
```

Expected: the implementation uses only API names present in these generated
docs, unless source inspection later confirms an undocumented public item.

---

### Task 1: Import tutorial project into `docs/tutorial-code/`

**Files:**
- Create: `docs/tutorial-code/`
- Copy from: public source repository <https://github.com/sdirnboeck/rust-Tensor4all>

- [ ] **Step 1: Create the target directory**

Run:

```bash
mkdir -p docs/tutorial-code
```

Expected: `docs/tutorial-code/` exists.

- [ ] **Step 2: Clone the public tutorial source**

Clone the public tutorial repository into a temporary location:

```bash
git clone --depth 1 https://github.com/sdirnboeck/rust-Tensor4all /tmp/rust-Tensor4all-tutorial-source
```

Expected: `/tmp/rust-Tensor4all-tutorial-source/Cargo.toml` exists.

- [ ] **Step 3: Copy the tutorial project into this repo**

Copy the tutorial project contents into `docs/tutorial-code/`, excluding build
output and VCS metadata:

```bash
rsync -a \
  --exclude target \
  --exclude .git \
  --exclude .DS_Store \
  /tmp/rust-Tensor4all-tutorial-source/ \
  docs/tutorial-code/
```

Expected: `docs/tutorial-code/Cargo.toml`, `docs/tutorial-code/src/`,
`docs/tutorial-code/tests/`, `docs/tutorial-code/docs/`, and
`docs/tutorial-code/scripts/` exist.

- [ ] **Step 4: Inspect imported files**

Run:

```bash
rg --files docs/tutorial-code | sort
```

Expected: the imported project contains the Rust binaries, helper modules,
tests, data, plots, plotting scripts, and README from the old learning project.

---

### Task 2: Convert the tutorial crate to local workspace dependencies

**Files:**
- Modify: `docs/tutorial-code/Cargo.toml`
- Modify if needed: `docs/tutorial-code/Cargo.lock`
- Modify if needed: `docs/tutorial-code/README.md`

- [ ] **Step 1: Inspect existing dependencies**

Run:

```bash
sed -n '1,220p' docs/tutorial-code/Cargo.toml
rg -n "git =|tensor4all-rs|rust-Tensor4all|playground" docs/tutorial-code
```

Expected: any old git dependencies or old-project wording are identified.

- [ ] **Step 2: Replace `tensor4all-*` git dependencies with local paths**

Update `docs/tutorial-code/Cargo.toml` so all `tensor4all-*` crates point at the
workspace:

```toml
tensor4all-core = { path = "../../crates/tensor4all-core" }
tensor4all-simplett = { path = "../../crates/tensor4all-simplett" }
tensor4all-treetn = { path = "../../crates/tensor4all-treetn" }
tensor4all-quanticstci = { path = "../../crates/tensor4all-quanticstci" }
tensor4all-quanticstransform = { path = "../../crates/tensor4all-quanticstransform" }
```

Only include crates actually used by the tutorial project. Keep non-tensor4all
dependencies on workspace versions where practical.

- [ ] **Step 3: Remove old external-repo wording from local docs**

Edit `docs/tutorial-code/README.md` and any imported tutorial-support docs so
they describe `docs/tutorial-code/` as the local runnable source for the
mdBook tutorials. Remove wording that makes the old external repository the
source of truth.

- [ ] **Step 4: Build the tutorial crate**

Run:

```bash
cargo test --manifest-path docs/tutorial-code/Cargo.toml --release
```

Expected: tutorial crate tests pass. If compilation fails because of API drift,
fix the tutorial code using the generated `docs/api/` as the first reference.

- [ ] **Step 5: Commit**

Run:

```bash
cargo fmt --all
git add docs/tutorial-code
git commit -m "docs: import runnable QTT tutorial code"
```

Expected: commit succeeds after formatting.

---

### Task 3: Update tutorial code for repository consistency

**Files:**
- Modify as needed: `docs/tutorial-code/src/**/*.rs`
- Modify as needed: `docs/tutorial-code/tests/**/*.rs`
- Modify as needed: `docs/tutorial-code/scripts/*.sh`
- Modify as needed: `docs/tutorial-code/docs/dev/*.md`

- [ ] **Step 1: Search for drift-prone wording and paths**

Run:

```bash
rg -n "playground|external repo|rust-Tensor4all|sdirnboeck|github.com/sdirnboeck|cargo run --bin|TENSOR4ALL_RS_PATH|docs/data|docs/plots" docs/tutorial-code
```

Expected: each hit is reviewed. Keep valid local paths; rewrite stale external
source-of-truth wording.

- [ ] **Step 2: Check binary output paths**

Inspect:

```bash
sed -n '1,220p' docs/tutorial-code/src/output_paths.rs
rg -n "CARGO_MANIFEST_DIR|TENSOR4ALL_DATA_DIR|TENSOR4ALL_PLOTS_DIR|docs/data|docs/plots" docs/tutorial-code/src docs/tutorial-code/tests
```

Expected: generated CSVs and plots go under `docs/tutorial-code/docs/data/`
and `docs/tutorial-code/docs/plots/` by default, with test overrides where
needed.

- [ ] **Step 3: Run smoke tests for representative binaries**

Run at least the lightweight binaries that exercise the core paths:

```bash
cargo run --manifest-path docs/tutorial-code/Cargo.toml --release --bin qtt_function
cargo run --manifest-path docs/tutorial-code/Cargo.toml --release --bin qtt_interval
cargo run --manifest-path docs/tutorial-code/Cargo.toml --release --bin qtt_integral
```

Expected: each command exits successfully and writes expected local data files
under `docs/tutorial-code/docs/data/`.

- [ ] **Step 4: Keep `tt_basics` local but outside online navigation**

Confirm `docs/tutorial-code/src/bin/tt_basics.rs` may still build, but no
online mdBook page or `SUMMARY.md` entry is planned for `tt_basics_tutorial.md`.

Run:

```bash
cargo run --manifest-path docs/tutorial-code/Cargo.toml --release --bin tt_basics
```

Expected: binary exits successfully, or any failure is fixed without adding the
basic TT tutorial to the online mdBook Tutorials section.

- [ ] **Step 5: Commit**

Run:

```bash
cargo fmt --all
git add docs/tutorial-code
git commit -m "docs: align QTT tutorial code with workspace"
```

Expected: commit succeeds after the tutorial crate builds and representative
binaries run.

---

### Task 4: Add mdBook tutorial directories and copy local plots

**Files:**
- Create: `docs/book/src/tutorials/quantics-basics/`
- Create: `docs/book/src/tutorials/computations-with-qtt/`
- Copy PNGs from: `docs/tutorial-code/docs/plots/`

- [ ] **Step 1: Create mdBook tutorial directories**

Run:

```bash
mkdir -p docs/book/src/tutorials/quantics-basics
mkdir -p docs/book/src/tutorials/computations-with-qtt
```

Expected: both directories exist.

- [ ] **Step 2: Copy quantics-basics plots from local tutorial artifacts**

Run:

```bash
cp docs/tutorial-code/docs/plots/qtt_function_vs_qtt.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_function_bond_dims.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_interval_function_vs_qtt.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_interval_bond_dims.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_integral_sweep.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_r_sweep_error.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_r_sweep_samples.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_r_sweep_runtime.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_multivariate_values.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_multivariate_error.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_multivariate_bond_dims.png docs/book/src/tutorials/quantics-basics/
```

Expected: all listed PNG files exist in the mdBook quantics-basics directory.

- [ ] **Step 3: Copy computations-with-qtt plots from local tutorial artifacts**

Run:

```bash
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
```

Expected: all listed PNG files exist in the mdBook computations-with-qtt
directory.

- [ ] **Step 4: Commit**

Run:

```bash
git add docs/book/src/tutorials
git commit -m "docs: add QTT tutorial plots to mdBook"
```

Expected: commit succeeds.

---

### Task 5: Write quantics-basics mdBook tutorials

**Files:**
- Create: `docs/book/src/tutorials/quantics-basics/qtt-scalar-function.md`
- Create: `docs/book/src/tutorials/quantics-basics/qtt-physical-interval.md`
- Create: `docs/book/src/tutorials/quantics-basics/qtt-definite-integrals.md`
- Create: `docs/book/src/tutorials/quantics-basics/sweep-bit-depth.md`
- Create: `docs/book/src/tutorials/quantics-basics/multivariate-functions.md`
- Source reference: `docs/tutorial-code/docs/tutorials/*.md`
- Runnable source: `docs/tutorial-code/src/bin/*.rs`

- [ ] **Step 1: Write `qtt-scalar-function.md`**

Adapt `docs/tutorial-code/docs/tutorials/qtt_function_tutorial.md`.

Required content:

- short introduction for a new `tensor4all-rs` user
- link to `../../../../tutorial-code/src/bin/qtt_function.rs` from the mdBook
  page location, or a verified correct relative link
- explanation that `quanticscrossinterpolate_discrete` uses 1-based discrete
  grid indices
- Rust snippet using `quanticscrossinterpolate_discrete`, `QtciOptions`,
  `evaluate`, and an assertion
- figures `qtt_function_vs_qtt.png` and `qtt_function_bond_dims.png`
- one-sentence explanation of bond dimension

- [ ] **Step 2: Write `qtt-physical-interval.md`**

Adapt `docs/tutorial-code/docs/tutorials/qtt_interval_tutorial.md`.

Required content:

- explain `DiscretizedGrid` as the map from integer grid points to physical
  coordinates
- Rust snippet using `quanticscrossinterpolate`, `DiscretizedGrid`,
  `QtciOptions`, `evaluate`, and an assertion
- figures `qtt_interval_function_vs_qtt.png` and
  `qtt_interval_bond_dims.png`

- [ ] **Step 3: Write `qtt-definite-integrals.md`**

Adapt `docs/tutorial-code/docs/tutorials/qtt_integral_tutorial.md`.

Required content:

- explain that `integral()` returns a Riemann-sum approximation on a continuous
  `DiscretizedGrid`
- Rust snippet using `integral()` and an assertion comparing to an analytic
  value with an explicit tolerance
- figure `qtt_integral_sweep.png`

- [ ] **Step 4: Write `sweep-bit-depth.md`**

Adapt `docs/tutorial-code/docs/tutorials/qtt_r_sweep_tutorial.md`.

Required content:

- explain bit depth as the number of binary sites per variable
- explain the accuracy/cost trade-off in plain language
- Rust snippet that builds a small QTT for one chosen bit depth and asserts a
  sample value
- figures `qtt_r_sweep_samples.png`, `qtt_r_sweep_error.png`, and
  `qtt_r_sweep_runtime.png`

- [ ] **Step 5: Write `multivariate-functions.md`**

Adapt `docs/tutorial-code/docs/tutorials/qtt_multivariate_tutorial.md`.

Required content:

- explain grouped and interleaved layouts briefly
- Rust snippet using the current public multivariate QTT API and an assertion
- figures `qtt_multivariate_values.png`, `qtt_multivariate_error.png`, and
  `qtt_multivariate_bond_dims.png`

- [ ] **Step 6: Run mdBook snippet tests**

Run:

```bash
./scripts/test-mdbook.sh
```

Expected: all snippets, including the new quantics-basics snippets, compile and
pass their assertions.

- [ ] **Step 7: Commit**

Run:

```bash
git add docs/book/src/tutorials/quantics-basics
git commit -m "docs: add quantics basics tutorials"
```

Expected: commit succeeds after `./scripts/test-mdbook.sh` passes.

---

### Task 6: Write computations-with-qtt mdBook tutorials

**Files:**
- Create: `docs/book/src/tutorials/computations-with-qtt/elementwise-product.md`
- Create: `docs/book/src/tutorials/computations-with-qtt/affine-transformation.md`
- Create: `docs/book/src/tutorials/computations-with-qtt/fourier-transform.md`
- Create: `docs/book/src/tutorials/computations-with-qtt/partial-fourier2d.md`
- Source reference: `docs/tutorial-code/docs/tutorials/*.md`
- Runnable source: `docs/tutorial-code/src/bin/*.rs`

- [ ] **Step 1: Write `elementwise-product.md`**

Adapt `docs/tutorial-code/docs/tutorials/qtt_elementwise_product_tutorial.md`.

Required content:

- explain why the example moves from QTTs to `TreeTN`
- use only high-level public APIs
- include a Rust snippet with an assertion
- include figures `qtt_elementwise_product_factors.png`,
  `qtt_elementwise_product_product.png`, and
  `qtt_elementwise_product_bond_dims.png`

- [ ] **Step 2: Write `affine-transformation.md`**

Adapt `docs/tutorial-code/docs/tutorials/qtt_affine_tutorial.md`.

Required content:

- explain the coordinate lookup in plain language before using "pullback"
- include a Rust snippet using current affine transform APIs and an assertion
- include figures `qtt_affine_values.png`, `qtt_affine_error.png`,
  `qtt_affine_bond_dims.png`, and `qtt_affine_operator_bond_dims.png`

- [ ] **Step 3: Write `fourier-transform.md`**

Adapt `docs/tutorial-code/docs/tutorials/qtt_fourier_tutorial.md`.

Required content:

- cross-reference `../../guides/quantics.md` or the verified correct relative
  path from this page
- include a Rust snippet using `quantics_fourier_operator` and an assertion
- include figures `qtt_fourier_transform.png`, `qtt_fourier_bond_dims.png`,
  and `qtt_fourier_operator_bond_dims.png`

- [ ] **Step 4: Write `partial-fourier2d.md`**

Adapt `docs/tutorial-code/docs/tutorials/qtt_partial_fourier2d_tutorial.md`.

Required content:

- explain "partial Fourier transform" as transforming one coordinate while
  leaving the other coordinate unchanged
- cross-reference the Quantics Transform guide
- include a Rust snippet with an assertion
- include figures `qtt_partial_fourier2d_values.png`,
  `qtt_partial_fourier2d_error.png`, and
  `qtt_partial_fourier2d_bond_dims.png`

- [ ] **Step 5: Run mdBook snippet tests**

Run:

```bash
./scripts/test-mdbook.sh
```

Expected: all snippets, including the new computations-with-qtt snippets,
compile and pass their assertions.

- [ ] **Step 6: Commit**

Run:

```bash
git add docs/book/src/tutorials/computations-with-qtt
git commit -m "docs: add computations with QTT tutorials"
```

Expected: commit succeeds after `./scripts/test-mdbook.sh` passes.

---

### Task 7: Update mdBook navigation

**Files:**
- Modify: `docs/book/src/SUMMARY.md`

- [ ] **Step 1: Insert Tutorials section before Conventions**

Insert after the `[Guides]()` section and before `[Conventions](conventions.md)`:

```markdown
- [Tutorials]()
  - [Quantics Basics]()
    - [QTT of a Scalar Function](tutorials/quantics-basics/qtt-scalar-function.md)
    - [QTT on a Physical Interval](tutorials/quantics-basics/qtt-physical-interval.md)
    - [Definite Integrals](tutorials/quantics-basics/qtt-definite-integrals.md)
    - [Sweep over Bit Depth](tutorials/quantics-basics/sweep-bit-depth.md)
    - [Multivariate Functions](tutorials/quantics-basics/multivariate-functions.md)
  - [Computations with QTT]()
    - [Elementwise Product](tutorials/computations-with-qtt/elementwise-product.md)
    - [Affine Transformation](tutorials/computations-with-qtt/affine-transformation.md)
    - [Fourier Transform](tutorials/computations-with-qtt/fourier-transform.md)
    - [2D Partial Fourier Transform](tutorials/computations-with-qtt/partial-fourier2d.md)
```

Do not add an entry for `tt_basics_tutorial.md`.

- [ ] **Step 2: Build mdBook**

Run:

```bash
mdbook build docs/book
```

Expected: no errors, and `docs/book/book/tutorials/` contains generated HTML.

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/book/src/SUMMARY.md
git commit -m "docs: add Tutorials section to mdBook"
```

Expected: commit succeeds after the mdBook build passes.

---

### Task 8: Final verification

- [ ] **Step 1: Format**

Run:

```bash
cargo fmt --all
cargo fmt --all -- --check
```

Expected: formatting succeeds.

- [ ] **Step 2: Re-run API dump**

Run:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

Expected: command succeeds. Do not manually edit generated `docs/api` files.

- [ ] **Step 3: Test tutorial crate**

Run:

```bash
cargo test --manifest-path docs/tutorial-code/Cargo.toml --release
```

Expected: all tutorial-code tests pass.

- [ ] **Step 4: Test mdBook snippets**

Run:

```bash
./scripts/test-mdbook.sh
```

Expected: all mdBook Rust snippets compile and pass assertions.

- [ ] **Step 5: Build mdBook**

Run:

```bash
mdbook build docs/book
```

Expected: mdBook build succeeds.

- [ ] **Step 6: Check for stale external source-of-truth wording**

Run:

```bash
rg -n "sdirnboeck|rust-Tensor4all|raw.githubusercontent|external repo|playground|QuanticsTciBuilder|no_run|ignore" docs/tutorial-code docs/book/src/tutorials docs/book/src/SUMMARY.md
```

Expected: no stale hits. Any remaining hit must be deliberately justified in
the final notes.

- [ ] **Step 7: Review branch contents**

Run:

```bash
git status --short
git log --oneline origin/main..HEAD
```

Expected: only intentional tutorial-code, mdBook, and generated API-reference
changes are present.

---

### Task 9: Optional publish steps after explicit user approval

Do not run this task unless the user explicitly approves pushing and opening a
PR.

- [ ] **Step 1: Push branch**

Run:

```bash
git push -u origin codex/tutorial-code-mdbook
```

- [ ] **Step 2: Create PR**

Use GitHub tooling only after approval. Suggested title:

```text
docs: add runnable QTT tutorial code and mdBook tutorials
```

Suggested body:

```markdown
## Summary

Adds runnable QTT tutorial code under `docs/tutorial-code/` and adds a matching
Tutorials section to the mdBook documentation.

## Verification

- `cargo fmt --all -- --check`
- `cargo test --manifest-path docs/tutorial-code/Cargo.toml --release`
- `./scripts/test-mdbook.sh`
- `mdbook build docs/book`
```

- [ ] **Step 3: Enable auto-merge only if requested**

Enable auto-merge only after the user explicitly asks for it and the branch is
synchronized with the current `origin/main`.
