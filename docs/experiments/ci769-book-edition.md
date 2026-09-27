# Book harness edition evaluation for issue #769

## Scope

The Cargo `book-tests` package and standalone mdBook configuration are separate
Rust edition settings. The workspace default was already 2024, but both book
settings still explicitly selected 2021 before this evaluation. Only those book
settings are evaluated here; the deliberately retained 2021 library crates are
not migrated.

The Cargo harness embeds the root README and ten guide pages. A source inventory
finds 47 Rust examples and no `compile_fail` examples in those eleven files. The
standalone mdBook traversal additionally includes tutorials and other published
chapters; the Cargo README example is outside its chapter set. Both execution
paths remain, with their assertions and source snippets unchanged.

## Method

The local host is Apple M4 with ten logical CPUs and 24 GiB RAM, macOS 27.0,
Rust 1.98.1. Dependencies resolve from the unchanged workspace lockfile. Cargo
uses release mode and two build jobs. `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` equal one;
the effective default CPU backend thread count is checked through its public
API. Cargo doctests receive `--test-threads=1`; standalone mdBook uses
`RUST_TEST_THREADS=1`. No CPU affinity is imposed.

Each Cargo edition has an initial warmup and three retained samples; standalone
mdBook has one warmup and one retained reference sample per edition. The Cargo wall
clock is split at its `Finished release profile` message: dependency/package
preparation before that message and rustdoc compilation/execution after it.
The initial 2021 invocation needs a different dependency feature union from the
preceding integration tests, so it rebuilds some dependencies. Its build cost is
reported separately and is excluded from the warm-edition comparison.

Standalone tests use the hosted pinned mdBook 0.5.2 binary. The machine's existing
mdBook installation is not used for this comparison. Both standalone editions
reuse the same fresh Cargo rustdoc log through `scripts/test-mdbook.sh`, so the
comparison does not include another Cargo preparation build. Preparation and
chapter execution remain distinguishable in that script's output.

## Results

### Cargo harness

| Edition | Initial Cargo preparation | Initial post-build | Warm median preparation | Warm median post-build |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 545.694 s | 45.745 s | 0.157 s | 36.686 s |
| 2024 | 0.200 s | 3.335 s | 0.131 s | 3.309 s |

The warm post-build median decreased by 90.98%. The 2024 rustdoc output explicitly
reports merged doctest compilation (about 2.27 s in the last retained sample).
This is a local preparation/compilation improvement; it is not a claim about the
full hosted CI critical path. Rust 2024 [combines eligible doctests into fewer
executables](https://doc.rust-lang.org/edition-guide/rust-2024/rustdoc-doctests.html),
while examples continue to execute in separate processes.

Every Cargo invocation passed all 47 examples, with zero ignored tests and zero
compile-fail examples. Before comparing case IDs, the 2021 synthetic
`src/lib.rs` line numbers were normalized by subtracting the zero-based
`#[doc = include_str!(...)]` attribute offset. The resulting `(module, Markdown
line)` sets are identical to the 2024 included-file locations. All snippet
bodies and assertions are unchanged.

The large initial 2021 preparation is recorded, not attributed to the edition.
The earlier integration-test dependency closure enabled `either` features
`std,use_std`; the book closure used no `either` features. Both `rayon` variants
had the same direct features, profile, rustflags, and configuration, but their
`either` dependency fingerprints differed. The resulting dependency hashes
propagated through numerical dependencies, requiring new compilation. Direct
features alone therefore do not establish that a compiled artifact is reusable.
The three warm comparisons avoid this initial feature-union rebuild.

### Standalone mdBook

| Edition | Warmup | Retained sample |
| --- | ---: | ---: |
| 2021 | 70.784 s | 59.500 s |
| 2024 | 35.902 s | 35.258 s |

Both editions completed successfully. The retained 2024 observation is 40.74%
shorter than the retained 2021 observation.

The wrapper reports zero seconds of Cargo preparation for each standalone run:
it reuses the same fresh resolved extern log. These runs include chapter-level
rustdoc compilation and execution. With only one retained sample per edition,
the timings are reference observations, not a statistical performance claim.

All 30 traversed chapter paths match between editions. They contain 65 runnable
Rust fences and no `compile_fail`, `ignore`, or `no_run` fences. Successful
mdBook output reports chapters rather than individual passing cases, so 65 is
a static source inventory, not an independently emitted runtime test count.
No snippet, tolerance, assertion, chapter, or Cargo example was removed.

## Decision and reproducibility

Adopt workspace edition 2024 for `book-tests` and explicitly select 2024 in the
standalone book configuration. Both independent paths pass locally; no book
compatibility exemption remains. The deliberately held library editions and
both complementary documentation test paths remain unchanged. Full hosted
workspace checks still validate the branch separately.

Raw sample values, canonical Cargo case IDs, chapter paths, and fence locations
are in [ci769-book-edition.json](ci769-book-edition.json). Trial 0 is the excluded
warmup. Cargo command:

```sh
cargo test --locked --doc --release -p book-tests -j 2 -vv -- --test-threads=1
```

Standalone command is `./scripts/test-mdbook.sh`, with mdBook 0.5.2 first on
`PATH`, `TENSOR4ALL_CARGO_PROFILE=release`, and `TENSOR4ALL_RUSTDOC_LOG` pointing
to a successful verbose Cargo book-test log. Apply the thread environment above
to both commands. Local diagnostic logs were retained as
`/tmp/t4a-book-{2021,2024}-{0,1,2,3}.log` and
`/tmp/t4a-mdbook-{2021,2024}-{0,1}.log`; they are disposable diagnostics, while the
committed JSON contains the comparison data.

