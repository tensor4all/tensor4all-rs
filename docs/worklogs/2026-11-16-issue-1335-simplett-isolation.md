# TreeTN interoperability and M6 dependency isolation

## Decisions

- Make TreeTN's existing SimpleTT bridge opt-in. Keep a single conversion
  implementation and weak CPU-provider forwarding; native chain construction
  and compression remain the implementation supplied by #1334.
- Keep the benchmark package and numerical harnesses intact. Feature-gate
  Fourier construction, historical SimpleTT/TCI comparisons, and the
  ITensors-style facade separately instead of splitting or copying kernels.
  The Fourier feature uses the native constructor landed in #1333, with legacy
  transforms disabled. This does not audit untracked M6a programs elsewhere.
- Check resolved normal/build packages and activated features from independent
  downstream library roots and package-selected benchmark graphs. Include
  benchmark dev edges, and positively check intentional legacy opt-ins, so
  neither workspace feature unification nor a source-text check can hide a leak.
- Build cached-evaluator measurement inputs through the native chain API,
  preserving the old RNG draw order and column-major values. Keep the actual
  TTCache comparison benchmark behind the bridge feature.
- Tutorial and book-test packages opt into the bridge explicitly for their
  legacy Quantics TCI states, instead of relying on transitive feature
  unification. Their live tutorial pages describe that state-conversion seam.
- The durable feature contract is recorded in
  [build profiles](../design/build-profiles.md#treetn-interoperability-and-benchmark-isolation).

## Verification conclusions and constraints

- The default direct TreeTN consumer and isolated M6 library consumers resolve
  without SimpleTT or its bridge. Combined consumers cover all four CPU
  providers at the graph level. Native benchmark graphs exclude unrelated
  Fourier and legacy stacks; Fourier-only graphs exclude SimpleTT transforms.
- No numerical kernel, tolerance, stop condition, or benchmark measurement
  protocol changed. No speedup claim or full numerical benchmark rerun is
  warranted by this packaging change.
- Existing cached-evaluator tests remain available without the bridge. The
  small fixture regression compares native construction with the previous
  SimpleTT fixture's full dense values; explicit bridge regressions and
  opt-in target compilation preserve migration coverage. Removed duplicate
  setup code was reviewed for shared-helper coverage impact.
- Workspace and isolated default/opt-in target compilation, strict Clippy,
  focused native/bridge numerical regressions, default and bridge doctests,
  mdBook examples, and deterministic repository boundary checks passed.
- Runtime validation is on Linux with the default faer provider. Other CPU
  providers are resolved-graph checks only; hardware-specific execution and
  comprehensive hosted CI remain separate validation.
