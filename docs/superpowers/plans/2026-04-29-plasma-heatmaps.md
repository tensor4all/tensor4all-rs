# Switch Tutorial Heatmaps to `:plasma` Colormap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace CairoMakie's default (`:viridis`) colormap with `:plasma` in all `heatmap!` calls across 3 Julia plotting scripts, keeping the existing CSS `invert(1) hue-rotate(180deg)` dark-mode rule unchanged.

**Architecture:** Three independent Julia scripts (`docs/tutorial-code/docs/plotting/qtt_*.jl`) each generate PNG/PDF heatmaps from CSV data. The colormap is a keyword argument on `heatmap!()`. No Rust code, no CSS, and no CSV data changes. Only 7 `heatmap!` call sites across 3 files need a `colormap = :plasma` addition.

**Tech Stack:** Julia + CairoMakie (plots), Rust (data generation, unchanged), mdBook (static site, unchanged)

**Affected PNGs (6 files):**
| Script | Heatmap PNGs |
|--------|-------------|
| `qtt_affine_plot.jl` | `qtt_affine_values.png`, `qtt_affine_error.png` |
| `qtt_multivariate_plot.jl` | `qtt_multivariate_values.png`, `qtt_multivariate_error.png` |
| `qtt_partial_fourier2d_plot.jl` | `qtt_partial_fourier2d_values.png`, `qtt_partial_fourier2d_error.png` |

**Not affected:** Bond-dimension line plots (`*_bond_dims.png`, `*_operator_bond_dims.png`) and all non-heatmap plots are unchanged.

---

### Task 1: Edit `qtt_affine_plot.jl`

**Files:**
- Modify: `docs/tutorial-code/docs/plotting/qtt_affine_plot.jl:87` and `:105`

- [ ] **Step 1: Add `colormap = :plasma` to the values heatmap loop (line 87)**

Change:
```julia
hm = heatmap!(ax, xs, ys, matrices[i]; colorrange)
```
To:
```julia
hm = heatmap!(ax, xs, ys, matrices[i]; colorrange, colormap = :plasma)
```

- [ ] **Step 2: Add `colormap = :plasma` to the errors heatmap loop (line 105)**

Change:
```julia
hm = heatmap!(ax, xs, ys, matrices[i])
```
To:
```julia
hm = heatmap!(ax, xs, ys, matrices[i]; colormap = :plasma)
```

---

### Task 2: Edit `qtt_multivariate_plot.jl`

**Files:**
- Modify: `docs/tutorial-code/docs/plotting/qtt_multivariate_plot.jl:67` and `:84`

- [ ] **Step 1: Add `colormap = :plasma` to the values heatmap loop (line 67)**

Change:
```julia
hm = heatmap!(ax, xs, ys, matrices[i])
```
To:
```julia
hm = heatmap!(ax, xs, ys, matrices[i]; colormap = :plasma)
```

- [ ] **Step 2: Add `colormap = :plasma` to the errors heatmap loop (line 84)**

Change:
```julia
hm = heatmap!(ax, xs, ys, matrices[i])
```
To:
```julia
hm = heatmap!(ax, xs, ys, matrices[i]; colormap = :plasma)
```

---

### Task 3: Edit `qtt_partial_fourier2d_plot.jl`

**Files:**
- Modify: `docs/tutorial-code/docs/plotting/qtt_partial_fourier2d_plot.jl:75`, `:84`, `:106`

- [ ] **Step 1: Add `colormap = :plasma` to the analytic heatmap (line 75)**

Change:
```julia
hm1 = heatmap!(ax1, ks, ts, analytic')
```
To:
```julia
hm1 = heatmap!(ax1, ks, ts, analytic'; colormap = :plasma)
```

- [ ] **Step 2: Add `colormap = :plasma` to the QTT heatmap (line 84)**

Change:
```julia
hm2 = heatmap!(ax2, ks, ts, qtt')
```
To:
```julia
hm2 = heatmap!(ax2, ks, ts, qtt'; colormap = :plasma)
```

- [ ] **Step 3: Add `colormap = :plasma` to the error heatmap (line 106)**

Change:
```julia
hm = heatmap!(ax, ks, ts, errors')
```
To:
```julia
hm = heatmap!(ax, ks, ts, errors'; colormap = :plasma)
```

---

### Task 4: Verify Julia scripts parse correctly

**Prerequisites:** Julia + CairoMakie installed, dependencies instantiated:
```bash
cd docs/tutorial-code && julia --project=docs/plotting -e 'using Pkg; Pkg.instantiate()'
```

- [ ] **Step 1: Dry-run parse each modified script (syntax check, no execution)**

Run:
```bash
cd docs/tutorial-code
julia --project=docs/plotting -e 'include("docs/plotting/qtt_affine_plot.jl"); println("affine OK")'
julia --project=docs/plotting -e 'include("docs/plotting/qtt_multivariate_plot.jl"); println("multivariate OK")'
julia --project=docs/plotting -e 'include("docs/plotting/qtt_partial_fourier2d_plot.jl"); println("partial_fourier2d OK")'
```

Expected: Each prints `<name> OK` with no error. (This will also regenerate the plots since `main()` is called at the bottom of each script.)

- [ ] **Step 2: Verify all 6 PNGs exist and are non-empty**

Run:
```bash
ls -lh docs/tutorial-code/docs/plots/qtt_affine_values.png docs/tutorial-code/docs/plots/qtt_affine_error.png docs/tutorial-code/docs/plots/qtt_multivariate_values.png docs/tutorial-code/docs/plots/qtt_multivariate_error.png docs/tutorial-code/docs/plots/qtt_partial_fourier2d_values.png docs/tutorial-code/docs/plots/qtt_partial_fourier2d_error.png
```

Expected: All 6 files listed with non-zero size.

---

### Task 5: Copy PNGs to book source and rebuild

- [ ] **Step 1: Copy heatmap PNGs to book source directories**

```bash
cp docs/tutorial-code/docs/plots/qtt_affine_values.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_affine_error.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_multivariate_values.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_multivariate_error.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_partial_fourier2d_values.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_partial_fourier2d_error.png docs/book/src/tutorials/computations-with-qtt/
```

- [ ] **Step 2: Rebuild the mdBook**

```bash
cd docs/book && mdbook build
```

Expected: exit code 0, no warnings.

- [ ] **Step 3: Run mdbook tests**

```bash
cd ../.. && ./scripts/test-mdbook.sh
```

Expected: exit code 0.

---

### Task 6: Visual verification

- [ ] **Step 1: Open the built book in a browser**

```bash
open docs/book/book/index.html
```

- [ ] **Step 2: Navigate to each tutorial with heatmaps and visually verify**

Pages to check:
- `tutorials/quantics-basics/multivariate-functions.html` -- 5 heatmaps (values + errors)
- `tutorials/computations-with-qtt/affine-transformation.html` -- 5 heatmaps (values + errors)
- `tutorials/computations-with-qtt/partial-fourier2d.html` -- 3 heatmaps (values + errors)

- [ ] **Step 3: Toggle dark themes (Coal, Navy, Ayu) and verify**

Switch each theme via the theme picker (paintbrush icon) and confirm heatmaps remain readable and the `:plasma` colormap with CSS `invert(1) hue-rotate(180deg)` produces good-looking results on dark backgrounds.
