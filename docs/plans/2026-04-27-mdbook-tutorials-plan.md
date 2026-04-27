# mdBook Tutorials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Tutorials" section to the mdBook documentation with adapted tutorial pages and their plots.

**Architecture:** New `docs/book/src/tutorials/` directory with two subdirectories (`quantics-basics/`, `computations-with-qtt/`), each containing adapted `.md` files and `.png` plots. Updated `SUMMARY.md` adds a `[Tutorials]()` section parallel to `[Guides]()`. Exact number of tutorials may change based on feedback.

**Tech Stack:** mdBook 0.5.2, Markdown, full ` ```rust ` blocks with hidden lines, PNG plots.

**Key decisions (see design doc `docs/plans/2026-04-27-mdbook-tutorials-design.md`):**
- Code blocks use full ` ```rust ` with hidden lines (`# `), same pattern as existing guides
- Plots keep original underscore filenames from reference repo
- Cross-references to the Quantics Transform guide added in Fourier tutorials
- Commit in logical batches (not per-file)

---

### Task 1: Create directory structure and download plots

**Files:**
- Create: `docs/book/src/tutorials/quantics-basics/`
- Create: `docs/book/src/tutorials/computations-with-qtt/`
- Download: 24 `.png` files from the public reference repo

**Reference repo:** <https://github.com/sdirnboeck/rust-Tensor4all>
Plot sources are under `docs/plots/` in that repo. Use the raw GitHub URLs to
download each file individually.

- [ ] **Step 1: Create directories**

```bash
mkdir -p docs/book/src/tutorials/quantics-basics
mkdir -p docs/book/src/tutorials/computations-with-qtt
```

- [ ] **Step 2: Download quantics-basics plots**

```bash
BASE="https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/plots"
DST="docs/book/src/tutorials/quantics-basics"
curl -sS "$BASE/qtt_function_vs_qtt.png"           -o "$DST/qtt_function_vs_qtt.png"
curl -sS "$BASE/qtt_function_bond_dims.png"         -o "$DST/qtt_function_bond_dims.png"
curl -sS "$BASE/qtt_interval_function_vs_qtt.png"   -o "$DST/qtt_interval_function_vs_qtt.png"
curl -sS "$BASE/qtt_interval_bond_dims.png"         -o "$DST/qtt_interval_bond_dims.png"
curl -sS "$BASE/qtt_integral_sweep.png"             -o "$DST/qtt_integral_sweep.png"
curl -sS "$BASE/qtt_r_sweep_error.png"              -o "$DST/qtt_r_sweep_error.png"
curl -sS "$BASE/qtt_r_sweep_samples.png"            -o "$DST/qtt_r_sweep_samples.png"
curl -sS "$BASE/qtt_r_sweep_runtime.png"            -o "$DST/qtt_r_sweep_runtime.png"
curl -sS "$BASE/qtt_multivariate_values.png"        -o "$DST/qtt_multivariate_values.png"
curl -sS "$BASE/qtt_multivariate_error.png"         -o "$DST/qtt_multivariate_error.png"
curl -sS "$BASE/qtt_multivariate_bond_dims.png"     -o "$DST/qtt_multivariate_bond_dims.png"
```

- [ ] **Step 3: Download computations-with-qtt plots**

```bash
BASE="https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/plots"
DST="docs/book/src/tutorials/computations-with-qtt"
curl -sS "$BASE/qtt_elementwise_product_factors.png"    -o "$DST/qtt_elementwise_product_factors.png"
curl -sS "$BASE/qtt_elementwise_product_product.png"    -o "$DST/qtt_elementwise_product_product.png"
curl -sS "$BASE/qtt_elementwise_product_bond_dims.png"  -o "$DST/qtt_elementwise_product_bond_dims.png"
curl -sS "$BASE/qtt_affine_values.png"                  -o "$DST/qtt_affine_values.png"
curl -sS "$BASE/qtt_affine_error.png"                   -o "$DST/qtt_affine_error.png"
curl -sS "$BASE/qtt_affine_bond_dims.png"               -o "$DST/qtt_affine_bond_dims.png"
curl -sS "$BASE/qtt_affine_operator_bond_dims.png"      -o "$DST/qtt_affine_operator_bond_dims.png"
curl -sS "$BASE/qtt_fourier_transform.png"              -o "$DST/qtt_fourier_transform.png"
curl -sS "$BASE/qtt_fourier_bond_dims.png"              -o "$DST/qtt_fourier_bond_dims.png"
curl -sS "$BASE/qtt_fourier_operator_bond_dims.png"     -o "$DST/qtt_fourier_operator_bond_dims.png"
curl -sS "$BASE/qtt_partial_fourier2d_values.png"       -o "$DST/qtt_partial_fourier2d_values.png"
curl -sS "$BASE/qtt_partial_fourier2d_error.png"        -o "$DST/qtt_partial_fourier2d_error.png"
curl -sS "$BASE/qtt_partial_fourier2d_bond_dims.png"    -o "$DST/qtt_partial_fourier2d_bond_dims.png"
```

- [ ] **Step 4: Commit**

```bash
git add docs/book/src/tutorials/
git commit -m "feat(docs): add tutorial directory structure and plots"
```

---

### Task 2: Write quantics-basics tutorials

**Reference repo:** <https://github.com/sdirnboeck/rust-Tensor4all>
Source tutorials are under `docs/tutorials/` in that repo. Open the raw
file at the corresponding URL to read the content to adapt.

| mdBook file | Source (raw URL) |
|-------------|------------------|
| `qtt-scalar-function.md` | <https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/tutorials/qtt_function_tutorial.md> |
| `qtt-physical-interval.md` | <https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/tutorials/qtt_interval_tutorial.md> |
| `qtt-definite-integrals.md` | <https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/tutorials/qtt_integral_tutorial.md> |
| `sweep-bit-depth.md` | <https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/tutorials/qtt_r_sweep_tutorial.md> |
| `multivariate-functions.md` | <https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/tutorials/qtt_multivariate_tutorial.md> |

**Template per page:**
- Title + Introduction (from original, remove Julia references)
- What the example computes (from original)
- Key API pieces — full ` ```rust ` blocks with hidden lines (`# `), compilable under `mdbook test`
- Figures — PNGs embedded with `![](./filename.png)`
- How to read the plots (from original)
- Cross-references: none needed for quantics-basics tutorials

**Removed** from originals: "Files in this example", "Source code",
"Code organization", "Pseudocode", all Julia sections, `cargo run`
instructions.

- [ ] **Step 1: Write qtt-scalar-function.md**

Adapt from `qtt_function_tutorial.md`.

- [ ] **Step 2: Write qtt-physical-interval.md**

Adapt from `qtt_interval_tutorial.md`.

- [ ] **Step 3: Write qtt-definite-integrals.md**

Adapt from `qtt_integral_tutorial.md`.

- [ ] **Step 4: Write sweep-bit-depth.md**

Adapt from `qtt_r_sweep_tutorial.md`.

- [ ] **Step 5: Write multivariate-functions.md**

Adapt from `qtt_multivariate_tutorial.md`.

- [ ] **Step 6: Commit all quantics-basics tutorials**

```bash
git add docs/book/src/tutorials/quantics-basics/*.md
git commit -m "feat(docs): add quantics-basics tutorials"
```

---

### Task 3: Write computations-with-qtt tutorials

**Reference repo:** same as Task 2.

| mdBook file | Source (raw URL) | Cross-reference |
|-------------|------------------|-----------------|
| `elementwise-product.md` | <https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/tutorials/qtt_elementwise_product_tutorial.md> | — |
| `affine-transformation.md` | <https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/tutorials/qtt_affine_tutorial.md> | — |
| `fourier-transform.md` | <https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/tutorials/qtt_fourier_tutorial.md> | See the [Quantics Transform](../guides/quantics.md) guide |
| `partial-fourier2d.md` | <https://raw.githubusercontent.com/sdirnboeck/rust-Tensor4all/main/docs/tutorials/qtt_partial_fourier2d_tutorial.md> | See the [Quantics Transform](../guides/quantics.md) guide |

Same template as Task 2.

- [ ] **Step 1: Write elementwise-product.md**

Adapt from `qtt_elementwise_product_tutorial.md`.

- [ ] **Step 2: Write affine-transformation.md**

Adapt from `qtt_affine_tutorial.md`.

- [ ] **Step 3: Write fourier-transform.md**

Adapt from `qtt_fourier_tutorial.md`.
Add cross-reference to Quantics Transform guide.

- [ ] **Step 4: Write partial-fourier2d.md**

Adapt from `qtt_partial_fourier2d_tutorial.md`.
Add cross-reference to Quantics Transform guide.

- [ ] **Step 5: Commit all computations-with-qtt tutorials**

```bash
git add docs/book/src/tutorials/computations-with-qtt/*.md
git commit -m "feat(docs): add computations-with-qtt tutorials"
```

---

### Task 4: Update SUMMARY.md

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

- [ ] **Step 2: Commit**

```bash
git add docs/book/src/SUMMARY.md
git commit -m "feat(docs): add Tutorials section to SUMMARY"
```

---

### Task 5: Build mdBook and verify

- [ ] **Step 1: Build mdBook**

```bash
mdbook build docs/book
```

Expected: no errors, output in `docs/book/book/`

- [ ] **Step 2: Run mdBook tests (CI equivalent)**

```bash
./scripts/test-mdbook.sh
```

Expected: all Rust code blocks compile and assertions pass.

- [ ] **Step 3: Verify the build output**

```bash
ls docs/book/book/tutorials/quantics-basics/
ls docs/book/book/tutorials/computations-with-qtt/
```

Expected: both directories exist with `.html` files.

---

### Task 6: Pre-PR checks

- [ ] **Step 1: Format check**

```bash
cargo fmt --all -- --check
```

No Rust code was changed — this is a documentation-only change. Expected: no issues.

- [ ] **Step 2: Verify only this branch is touched**

```bash
git log --oneline origin/main..HEAD
```

Expected: only commits from this plan, no merge commits.

---

### Task 7: Push and create PR

- [ ] **Step 1: Push branch**

```bash
git push -u origin add-mdbook-tutorials
```

- [ ] **Step 2: Open PR**

```bash
gh pr create \
  --base main \
  --title "feat(docs): add Tutorials section to mdBook" \
  --body "$(cat <<'EOF'
## Summary

Adds a new "Tutorials" section to the mdBook documentation covering
Quantics Basics and Computations with QTT.

## Changes

- New `docs/book/src/tutorials/` directory with two subsections:
  - `quantics-basics/` — tutorials and plots
  - `computations-with-qtt/` — tutorials and plots
- Updated `SUMMARY.md` with Tutorials section

## Notes

- Tutorials show full ` ```rust ` code blocks with hidden lines, compilable via `mdbook test`
- No executables or binaries in this repo — tutorials are self-contained
- Cross-references to the Quantics Transform guide in Fourier tutorials
EOF
)"
```

- [ ] **Step 3: Enable auto-merge**

```bash
gh pr merge --auto --squash --delete-branch
```
