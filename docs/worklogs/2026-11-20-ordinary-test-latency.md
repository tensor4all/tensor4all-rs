# Ordinary test latency

## Decisions

- Keep ordinary tests deterministic and small without changing assertion
  tolerances. The project-first addition regression now pins the Volume and Norm
  cases that actually split and reach a discard-free leaf pass, rather than
  searching a parameter grid and conditionally asserting.
- Keep structural TreeACI oracle coverage in the ordinary suite, including
  scalar and multi-axis nodes, paths, branching, site-free nodes, both index
  orientations, both maps, and all four scalar kinds. Preserve the original
  multi-seed geometry sweep as heavy evidence.
- Use a low-degree asymmetric mixed polynomial for the ordinary three-dimensional
  QTT oracle, retaining all three core positions and all 512 grid points. Preserve
  the degree-15 exponential approximation as heavy evidence.
- Put the cache timing measurement and full tutorial runs in the heavy suite.
  The ordinary tutorial test independently verifies all four values of a small
  sine interpolation. The nightly selector also reaches private heavy unit-test
  modules, which need internal measurement helpers.
- The tiny GSE and coefficient-gradient fixtures remain unchanged. Profiling
  located their latency in the dependency compiler: identity layout rewrites
  scanned the whole graph, and conjugation handling repeatedly allocated slot
  tables and copied shape metadata. Repair both at their owner and backport
  only the compiler change onto the existing SDK version. A pin-only Hataori
  backport keeps the adapter and direct SDK dependencies on one Rust type
  identity without bringing unrelated Hataori changes into this branch.

## Verification conclusions and constraints

- Reduced TreeACI, QTT, project-first addition, and tutorial checks passed with
  existing numerical tolerances. Both preserved heavy tutorial runs also passed.
- Production negative controls were rejected: incorrect single-node batch
  packing by all four TreeACI dtypes, broadcasting the first refinement digit
  across QTT axes, splitting a truncated probe instead of the exact parent, and
  replacing the tutorial sine target with cosine. The split mutation reproduces
  the historical error of approximately `7.72e-5`.
- Independent fixture review found no actionable defects. The original broad
  oracle and exponential witnesses remain available to nightly execution.
- The final dependency graph resolves all 14 SDK crates at one revision, with
  package versions and checksums unchanged. The current upstream compiler passes
  94 checks and its strict local PR gate; its byte-identical downstream backport
  passes 93 checks in release mode. All 2,072 ordinary checks across the five
  affected crates and all eight moved heavy witnesses pass on the final pins.
- A complete three-pair release rerun meets the original local primary and
  load/noise checks: GSE falls from about 2.41 s to 0.34 s, and coefficient-AD
  linsolve from about 1.29 s to 0.18 s. Both fixtures and all tolerances remain
  unchanged. The first layout-only experiment remains inconclusive; complete
  evidence is retained with the upstream compiler change. These unpinned macOS
  measurements do not establish the upstream pinned-core host-validity gate.
- Strict local Clippy on Homebrew Rust 1.98 reports pre-existing
  `nonminimal_bool` diagnostics in unchanged core/treetn code. The focused
  changed-crate lint check passes with only that diagnostic allowed; no lint
  suppression was added to source. Exact-head hosted CI remains pending.
- Local disk pressure required removing only task-owned completed validation
  artifacts; no other worktree's artifacts were removed.
