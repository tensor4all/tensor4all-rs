# Issue #1371 oracle evaluation comparison

This comparison answers the request to measure retained-pivot seeding before
choosing the final meaning of `tolerance`:
[maintainer reply](https://github.com/tensor4all/tensor4all-rs/issues/1371#issuecomment-6969875677).
The target, initial pivot, tolerances and seeds match Simon's 18-run Gaussian
table. The results do not establish a final API decision.

## Variants

- `baseline501ebb4a.csv`: commit
  `501ebb4a109c4f64386c6d2db7c15ef132c041c6`, with its original local LU and
  stopping behavior.
- `retained-only.csv`: candidate search and known-sample validation with the
  original local LU threshold. The measurement-only
  [patch](retained-only.patch) disables both the per-edge local budget and the
  absolute-truncation recovery phase. Its iteration-limit outcomes measure the
  unresolved stall; this ablation is not a completed floor-handling proposal.
- `candidate.csv`: the full experimental candidate, including per-edge local
  budgets, recovery and explicit floor/budget verdicts.

The latter two variants replay known samples and seed confirmation from distinct
retained full points. Consequently this comparison includes the cost and
stopping consequences of remembering known violations, rather than testing
seeding in isolation while discarding previously observed errors.

Each CSV contains all six seeds at each of `1e-4`, `1e-6` and `1e-8`, including
termination, iteration count, maximum rank, successful oracle point count,
reported error and a dense whole-result error. The table in
[summary.md](summary.md) sums oracle counts over the six seeds. Source and
result hashes are in [provenance.json](provenance.json).

## Interpretation

Search plus known-sample validation with the original LU threshold does not
remove the stall: three seeds hit the iteration limit at 1e-4, three at 1e-6,
and all six at 1e-8. The full candidate converges within four or five iterations
with dense errors below tolerance in all 18 runs. It still costs 1.327x to
1.454x the baseline oracle evaluations. Maximum ranks increase from 17/21/23
to 19/22/25 at the three tolerances, respectively.

This does not isolate seeding from history validation, and it does not show
that seeding alone eliminates the originally reported 2.2x to 3.4x gap. The
complete candidate changes local truncation too. Its additional accuracy and
rank growth have a cost; the final tolerance policy remains pending.

## Reproduction

Use the identical
[example source](../../../../crates/tensor4all-treetci/examples/issue1371_evaluation_cost.rs)
in separate baseline, candidate and ablation checkouts, with separate Cargo
targets. Copy that example into the baseline checkout. Apply `retained-only.patch`
to the ablation checkout. Keep the compiler, default faer provider and thread
configuration identical across variants.

For each checkout, build first:

```bash
ai-run build --mem 6G --jobs 2 -- cargo build --locked --release -p tensor4all-treetci --example issue1371_evaluation_cost
```

Then collect its CSV from the finished binary:

```bash
ai-run build --mem 2G --jobs 2 -- target/release/examples/issue1371_evaluation_cost > variant.csv
```

These are deterministic functional counters, collected under the build gate.
No elapsed times or speedups are reported. `evaluated_points` includes repeated
requests because oracle memoization is disabled; dense verification runs after
interpolation and is excluded from that counter. Known-sample replay evaluates
the approximation without calling the oracle. Its computational overhead is
therefore not represented by the oracle count.

Dense errors also expose unsampled misses. A reported sampled maximum remains
a diagnostic, not an exhaustive error certificate. These Linux results need
not reproduce another compiler, CPU provider or machine's exact trajectory.
