# Tutorial Review — April 29, 2026

Comprehensive review of all 9 mdBook tutorials covering: code consistency,
beginner-friendliness, documentation gaps, and maintenance processes.

## 1. Pseudocode vs Rust Binary Consistency

All 9 tutorials pass. API names, signatures, domain bounds, and function
expressions are consistent between markdown code blocks and Rust binaries.

| Tutorial | Status |
|----------|--------|
| qtt-scalar-function | OK |
| qtt-physical-interval | OK |
| qtt-definite-integrals | OK |
| sweep-bit-depth | OK |
| multivariate-functions | OK (minor: dual `UnfoldingScheme` is redundant but not an error) |
| elementwise-product | OK |
| affine-transformation | OK |
| fourier-transform | OK |
| partial-fourier2d | OK |

## 2. Beginner-Friendliness Scores

| # | Tutorial | Score | Key Issues |
|---|----------|-------|------------|
| 1 | Scalar Function | 3/5 | 6 API types in one block, no inline comments |
| 2 | Physical Interval | 3.5/5 | `include_endpoint` unexplained |
| 3 | Definite Integrals | 3.5/5 | Weak assertion (`integral > 0.0`), forward-reference to sweep tutorial |
| 4 | Sweep Bit Depth | 3/5 | Bit-shift notation may confuse non-Rust readers |
| 5 | Multivariate Functions | 3/5 | Dual `UnfoldingScheme` confusing; 2D pivot format unexplained |
| **6** | **Elementwise Product** | **2/5** | Massive conceptual jump: TreeTN, `partial_contract`, `diagonal_pairs`, `center` — all in one page, none explained |
| **7** | **Affine Transformation** | **2/5** | MPO, pullback, `Fused` scheme, boundary conditions — all jargon, zero comments |
| 8 | Fourier Transform | 3.5/5 | Workflow now familiar; bit-reversed output not mentioned |
| **9** | **Partial Fourier 2D** | **2/5** | **Code–prose mismatch**: code shows flat quantics but prose describes interleaved 2D. Code does not implement the partial transform described. |

**Systemic issue:** All 9 code blocks have zero inline comments.

**Complexity flow:**

```
T1 ████░░░░  QTT construction (6 new API types)
T2 ███░░░░░  + grid mapping
T3 ██░░░░░░  + integral
T4 ██░░░░░░  + sweep loop
T5 █████░░░  ++ 2D, layouts
T6 ████████  ++++ TreeTN, contractions, center (UNDISCLOSED JUMP)
T7 ████████  +++ MPO, pullback, Fused (UNDISCLOSED JUMP)
T8 █████░░░  ++ Fourier (pattern now familiar)
T9 ████████  ++++ Partial apply, site mapping (CODE BUG)
```

## 3. Documentation Gaps

### Present

- `docs/plans/2026-04-27-mdbook-tutorials-design.md`: "The online tutorials must not drift from the runnable code."
- `docs/tutorial-code/README.md`: Refresh process documented
- `REPOSITORY_RULES.md` & `AGENTS.md`: "Public Surface Drift" rules cover mdBook snippets
- CI: `./scripts/test-mdbook.sh` runs in `deploy-docs.yml` on push to main

### Missing

1. **No `TUTORIALS.md` at root** or section in `README.md` stating the tutorial
   maintenance policy
2. **No `scripts/refresh-tutorial-artifacts.sh`** — runs all binaries, Julia
   plots, copies PNGs, rebuilds book
3. **No checklist** for updating a tutorial after library API changes
4. **Two sets of tutorial markdown**: `docs/tutorial-code/docs/tutorials/` (legacy)
   and `docs/book/src/tutorials/` (live) with no documented policy
5. **No "how to add a new tutorial" guide**
6. **No explicit statement** that `deploy-docs.yml` auto-publishes the live book
   from current source on every push to main

## 4. Recommended Fixes

### Critical — Tutorial Content

- **Tutorial 6**: Add 5-6 sentences of prose explaining TreeTN, `diagonal_pairs`,
  `partial_contract`, and `center`. Add 3-4 inline comments in code block.
- **Tutorial 9**: Either rewrite code to show actual interleaved partial
  transform, or update prose to match simplified code.
- **Tutorial 7**: Add 3-4 inline comments explaining `Fused`, `transpose()`,
  boundary conditions, `align_to_state`.

### High — Documentation

- Create `TUTORIALS.md` in repo root or extend `AGENTS.md` Public Surface Drift
- Document the full tutorial refresh workflow
- Decide fate of legacy `docs/tutorial-code/docs/tutorials/` markdown files

### Medium — Automation

- Create `scripts/refresh-tutorial-artifacts.sh`
- Add tutorial sync requirements to issue templates

## 5. Repository Rules Check

All existing rules are followed:
- `AGENTS.md` doc comment requirements → applied in library code
- `REPOSITORY_RULES.md` Public Surface Drift → tutorials were just audited
- `./scripts/test-mdbook.sh` → passes (22/22)
- `cargo test --release -p tensor4all-tutorial-code` → passes (26/26)
- `cargo fmt --all` → clean
- `cargo clippy -p tensor4all-tutorial-code` → clean
