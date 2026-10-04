# Tree RSI review fixes and branch scope

## Decisions

- The RSI rewrite is synchronized with fetched main `b881f39d9e72e32c43b9841fe10433fa5f63b167` before fixes. The withdrawn predecessor is not imported.
- Fix only RSI and its required context/row-interpolation seams. Reject negative/nonfinite LU controls and nonfinite input at the new owned context facades, without changing the existing pivot algorithm or default tolerance. Row-only residual diagnostics serve RSI; legacy full-factor diagnostics retain main's behavior.
- Keep unrelated LU work separate. The inherited scale/pivot problem from review R2 is already covered by [issue #779](https://github.com/tensor4all/tensor4all-rs/issues/779), with an existing independent candidate. General dense/rook/full-factor residual corrections were removed from the RSI draft and preserved on a separate local branch. No ACI or TCI implementation fix is included here.
- Benchmark workers have immutable source/build receipts. Capture local dependency files, original worker/build source, resolved lock, Cargo configuration and compiler/build controls before and after compilation; execute Cargo's reported executable from its archive. Changed source or missing receipts require rebuilding before scheduling. Historical runs retain their identities and cannot acquire new receipts retroactively.
- Completeness means equality with the explicit Cartesian case/options, algorithm, seed, phase and block schedule. Summarization and artifact verification share this check. Trace verification compares actual baseline outcomes as well as its recorded schedule and immutable worker.
- Error documentation and panic-audit baseline maintenance cover the newly added APIs and moved assertions only. No new panic suppression, test-tolerance change or coverage-threshold change is introduced.

## Verification conclusions and constraints

- Debug tests and doctests for RSI/core/backend passed (1,663 tests, five pre-existing ignored). Focused optimized core pivot and row-facade tests and all 26 RSI executable tests passed, covering the shared-engine extraction and numerical/boundary paths. The affected TreeTN/TreeTCI suite also passed (1,097 tests, six pre-existing ignored).
- Clippy including missing error/panic documentation lints, public-error documentation, the compiler-backed panic audit, crate boundaries, and the explicit faer feature build passed. The complete mdBook guide test and all 1,095 workspace doctests passed. Rustdoc generated successfully; its six remaining core warnings concern unchanged main code.
- Python suites passed: seven small-harness tests, five main-comparison tests and 23 paper-coverage tests. Regressions reject stale/missing worker evidence, altered source/configuration, wrong schedules despite matching counts, missing source manifests and false trace parity.
- Fresh release workers on the synchronized source were built. An independent nonconstant two-site product reconstructed exactly through both providers; the receipt and schedule verifier accepted the evidence. Editing the original worker source then caused rejection before scheduling, and the source was restored.
- An optional full TreeACI run exercised 176 tests before its unrelated r=9 GW convergence regression was interrupted after nine minutes in debug mode. No ACI implementation or test tolerance was changed; this extra run is not a complete TreeACI validation.
- No broad performance rerun or speedup claim is made. Historical unreceipted measurements are not evidence for this source. RSI still requires independent output-error validation, including around the inherited issue #779; zero local tolerance/residual does not certify global accuracy. Downstream GW and unsupported nonlinear operations remain outside this branch.

The final scoped review and staged deterministic repository-rules preview found no remaining RSI blocker. The hosted/LLM rules gate has not run locally (no local API credential); no PR is created by this push.

The earlier [whole-branch review](2026-10-04-tree-rsi-whole-branch-review.md) records the original findings; this record describes their RSI-scoped disposition.
