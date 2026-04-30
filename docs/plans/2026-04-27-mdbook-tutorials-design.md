# Tutorial Code and mdBook Tutorials Design

## Goal

Move the runnable Rust QTT tutorial project from the public tutorial repository
into `tensor4all-rs` under `docs/tutorial-code/`, keep it building against the
current workspace crates, and derive the online mdBook tutorials from that local
source.

The result should give readers two consistent entry points:

- runnable Rust tutorial code in `docs/tutorial-code/`
- readable online tutorial pages in `docs/book/src/tutorials/`

The online tutorials must not drift from the runnable code. The local tutorial
code is the source of truth for examples, generated CSV data, and plots.

## Reader Profile

The target reader is a master's student who has basic numerical-computing
background and has learned the rough idea of QTTs, but has not used
`tensor4all-rs` before.

Tutorial prose should avoid unexplained jargon. Terms that are likely to be new
should get a short local explanation the first time they matter. For example,
"bond dimension" may be followed by a short phrase such as "the internal size
that controls how much information the QTT carries between neighboring sites."

## Source Material

The initial source material is the public tutorial repository:

<https://github.com/sdirnboeck/rust-Tensor4all>

During migration, clone or fetch that repository and copy from the checked-out
files rather than downloading individual raw GitHub URLs. After migration, the
source of truth becomes:

```text
docs/tutorial-code/
```

## Repository Placement

Use a flat `docs/tutorial-code/` directory for the Rust tutorial project:

```text
docs/tutorial-code/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── bin/
│   │   ├── qtt_function.rs
│   │   ├── qtt_interval.rs
│   │   ├── qtt_integral.rs
│   │   ├── qtt_integral_sweep.rs
│   │   ├── qtt_r_sweep.rs
│   │   ├── qtt_multivariate.rs
│   │   ├── qtt_elementwise_product.rs
│   │   ├── qtt_affine.rs
│   │   ├── qtt_fourier.rs
│   │   ├── qtt_partial_fourier2d.rs
│   │   └── tt_basics.rs
│   └── ...
├── tests/
├── docs/
│   ├── data/
│   ├── plots/
│   └── plotting/
└── scripts/
```

`tt_basics.rs` may remain as optional runnable prelude code, but
`tt_basics_tutorial.md` must not be added to the online mdBook Tutorials
section.

## Dependency Policy

`docs/tutorial-code/Cargo.toml` must use local `path` dependencies on
`crates/tensor4all-*`. It must not depend on `tensor4all-rs` through git URLs.
This keeps the tutorials aligned with the repository version being built.

Examples:

```toml
tensor4all-quanticstci = { path = "../../crates/tensor4all-quanticstci" }
tensor4all-quanticstransform = { path = "../../crates/tensor4all-quanticstransform" }
tensor4all-treetn = { path = "../../crates/tensor4all-treetn" }
```

## API Verification

Before adapting code or writing mdBook snippets, generate and consult the local
API reference:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

Read the relevant generated files in `docs/api/` first. Use source files only
when the generated API reference is insufficient.

Do not use APIs that are not present in the current public surface. In
particular, mdBook snippets should use the current public API names such as
`quanticscrossinterpolate`, `quanticscrossinterpolate_discrete`,
`DiscretizedGrid`, `QtciOptions`, `evaluate`, `sum`, and `integral` as
appropriate. Do not introduce placeholder APIs such as `QuanticsTciBuilder`
unless that type exists in the generated API reference.

## mdBook Organization

Add a top-level `Tutorials` section to `docs/book/src/SUMMARY.md`, parallel to
`Guides`:

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

Do not add a separate Tutorials landing page unless a later review finds that
the navigation needs one.

## mdBook File Organization

```text
docs/book/src/tutorials/
├── quantics-basics/
│   ├── qtt-scalar-function.md
│   ├── qtt_function_vs_qtt.png
│   ├── qtt_function_bond_dims.png
│   ├── qtt-physical-interval.md
│   ├── qtt_interval_function_vs_qtt.png
│   ├── qtt_interval_bond_dims.png
│   ├── qtt-definite-integrals.md
│   ├── qtt_integral_sweep.png
│   ├── sweep-bit-depth.md
│   ├── qtt_r_sweep_samples.png
│   ├── qtt_r_sweep_error.png
│   ├── qtt_r_sweep_runtime.png
│   ├── multivariate-functions.md
│   ├── qtt_multivariate_values.png
│   ├── qtt_multivariate_error.png
│   └── qtt_multivariate_bond_dims.png
└── computations-with-qtt/
    ├── elementwise-product.md
    ├── qtt_elementwise_product_factors.png
    ├── qtt_elementwise_product_product.png
    ├── qtt_elementwise_product_bond_dims.png
    ├── affine-transformation.md
    ├── qtt_affine_values.png
    ├── qtt_affine_error.png
    ├── qtt_affine_bond_dims.png
    ├── qtt_affine_operator_bond_dims.png
    ├── fourier-transform.md
    ├── qtt_fourier_transform.png
    ├── qtt_fourier_bond_dims.png
    ├── qtt_fourier_operator_bond_dims.png
    ├── partial-fourier2d.md
    ├── qtt_partial_fourier2d_values.png
    ├── qtt_partial_fourier2d_error.png
    └── qtt_partial_fourier2d_bond_dims.png
```

Markdown files use kebab-case. Plot files may keep the existing underscore
names so they match the generated tutorial artifacts.

## Learning Progression

**Quantics Basics** builds from one-dimensional function encoding to multiple
dimensions:

1. *QTT of a Scalar Function* - encode a simple analytic function, evaluate it
   on the grid, compare values, and inspect bond dimensions.
2. *QTT on a Physical Interval* - map a QTT grid to a real interval with
   `DiscretizedGrid`.
3. *Definite Integrals* - call `integral()` on the interval QTT and explain
   that it is a Riemann-sum approximation.
4. *Sweep over Bit Depth* - vary the number of quantics bits and show the
   accuracy/cost trade-off.
5. *Multivariate Functions* - build a two-dimensional QTT and compare grouped
   and interleaved layouts, with a short explanation of both layouts.

**Computations with QTT** assumes the reader has seen the basics:

6. *Elementwise Product* - build two QTTs, convert to `TreeTN`, and multiply
   them through the public tensor-network APIs.
7. *Affine Transformation* - apply a 2D affine coordinate lookup. Explain
   "pullback" in plain language if the term is used.
8. *Fourier Transform* - build a QTT for a Gaussian and apply the quantics
   Fourier operator.
9. *2D Partial Fourier Transform* - apply Fourier along one coordinate of a 2D
   QTT.

## Online Tutorial Page Template

Every mdBook tutorial page should contain:

| Section | Content |
|---------|---------|
| Title + introduction | Short, learner-oriented summary |
| Runnable source | Link to the local source file in `docs/tutorial-code/src/bin/` |
| What the example computes | Concrete result and why it matters |
| Key API pieces | Small Rust snippets using the current public API |
| Figures | PNG plots copied from `docs/tutorial-code/docs/plots/` |
| How to read the plots | Plain-language interpretation |

Remove or rewrite material that only made sense in the old external project:

- references to "playground" or an external source of truth
- raw GitHub source-code links as the primary source
- Julia plotting commands as required workflow
- long pseudocode blocks when the runnable Rust source is available locally

Julia plotting may remain documented as an optional artifact-regeneration path
inside `docs/tutorial-code/`, but it should not distract from the Rust tutorial
flow in the mdBook pages.

## Code Block Policy

Rust code blocks in mdBook pages must follow repository rules:

- use full ` ```rust ` blocks with hidden lines (`# `) for imports and wrappers
- compile under `./scripts/test-mdbook.sh`
- include assertions that verify correctness
- avoid `ignore` and `no_run`
- use only public APIs from the current generated API reference

Example shape:

````markdown
```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_quanticstci::{
#     quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme,
# };
let sizes = [8usize];
let f = |idx: &[i64]| -> f64 {
    let x = (idx[0] as f64 - 1.0) / sizes[0] as f64;
    x.cosh()
};
let options = QtciOptions::default()
    .with_unfoldingscheme(UnfoldingScheme::Interleaved)
    .with_verbosity(0);
let (qtci, _ranks, _errors) =
    quanticscrossinterpolate_discrete::<f64, _>(&sizes, f, None, options)?;
let value = qtci.evaluate(&[1])?;
assert!((value - 1.0).abs() < 1e-8);
# Ok(())
# }
```
````

## Artifact Policy

The runnable tutorial code may generate CSV files and plots under
`docs/tutorial-code/docs/data/` and `docs/tutorial-code/docs/plots/`.

The mdBook should embed selected PNGs copied into `docs/book/src/tutorials/...`
so the published documentation is self-contained. The implementation plan must
include an explicit step that refreshes or copies these PNGs from the local
tutorial-code artifacts, not from the old external repository.

## Verification

The implementation is not complete until these pass:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo run -p api-dump --release -- . -o docs/api
cargo test --manifest-path docs/tutorial-code/Cargo.toml --release
./scripts/test-mdbook.sh
mdbook build docs/book
```

For artifact-regeneration changes, also run the relevant tutorial binaries and
plot-refresh script from `docs/tutorial-code/`.

Do not push, open a PR, or enable auto-merge without explicit user approval.
