# Coverage obligations and reproducibility boundaries

This file defines benchmark obligations and validation boundaries; it is not
a success report. No measured-results synopsis is committed. Use raw completed
runs and independently recomputed errors under ignored `target/` when checking
any result.

| Source / gap | Required coverage | Validation / constraint |
|---|---|---|
| Paper IV.B, Fig.5 | Author spin-1 DMRG n20, input chi10/15/20/25/30, rank curves | Independent norm/Hzz, Born + uniform samples; bounded QR/SVD reference where feasible |
| Paper IV.B, Figs.6–7 | Author n50, chi20/40/60/80/100 | Native main TreeACI versus branch RSI; physical observables are not a global error certificate |
| High-rank physical input | Author n50 chi150 | Actual core dimensions checked; never trust file names |
| DMRG validation calibration | Author n10 chi20 | All 3^10 entries; compare sample and observable metrics to full error |
| Paper IV.C, Fig.8 | n25 Gaussian separations .4/.6, .25/.75, .1/.9, sigma .15 | Formula holdout plus independent bounded product/QR/SVD reference |
| Paper IV.C, Fig.9 | n25 narrow Gaussian .49/.51, sigma .01 | Verify input accuracy and initialization failure independently of product algorithm |
| Paper IV.C, Fig.10 | 2/3/4-input Gaussian products, cap15 | Genuine simultaneous product calls; disclose that repeated operands are separately sketched by current API |
| Paper IV.C, Fig.11 | n25 oscillatory formulas, ranks5–30, p0/5/10 | Error-versus-rank/oversampling curves; additional 3/4-input cases from public Python plots |
| Paper V.A, Fig.13 | Original Dxx data, input cap20, output cap5/10/20/30 | Separate input discretization/compression error from product error |
| Paper V.B, Fig.14 | Complex frequency-domain product and convolution | Exact original functions/operator parameters are not specified by the paper/public scripts; any substitute is labeled a separate validation |
| Paper VI, Fig.15 | ReLU input formula and rank curve | Current Rust RSI has no general nonlinear map API; author map experiment cannot be counted as Rust support |
| Public author GPE extension | Original complex 91-vortex wavefunction | Verify conjugation, probability normalization and held-out values; published code opens 13 sites, unlike current two-site sketch construction |
| Tree extension | Branching high-rank complex data, root changes, >2 operands | Full bounded oracle and actual edge ranks; empirical results are not a general tree error theorem |
| Timing repeatability | Fixed repeat blocks on physical and function workloads | Report every observation, variability and unstable status; no cherry-picked speedups |
| Intermediate ranks | Initial/committed/injected ACI bonds, constructed RSI bonds | Diagnostic replay compared to untouched main; distinguish local matrix dimensions from bond dimensions |
| gw-rs | Current API compatibility and independent G0/Pi/Sigma/iteration checks | Old wrappers call removed `tree_rsi_elementwise` and use local pivots as error; stale reports are not acceptance |

## Source limitations already established

- The paper reports Julia/ITensor timings; that implementation was not released
  in the cited revision. Public Python replay cannot reproduce those exact
  Julia timings or establish that its implementation is efficient.
- Public plot scripts contain literal numeric arrays. They are not raw run logs.
- The public sketch function accepts `seed` but does not seed NumPy; its
  `normal(0,10)` distribution differs from the paper's standard normal.
- The author's LU elimination loop runs to the matrix's smaller dimension
  (or cutoff) before extracting `maxdim` factors. With `eps=0`, a small
  requested output rank does not bound its elimination work. It is not an
  appropriate performance implementation to copy blindly.
- Author DMRG uses `eps=0` and `floor(cap/2)+10`; the primary Rust comparison
  uses local relative tolerance1e-12 and its d-aware width heuristic. These
  settings are recorded separately and sensitivity cases do not replace the
  original observations.
- `psi_maxdim200_n50.h5` contains n50/chi50, and
  `psi_maxdim50_n50.h5` contains n10/chi50. Both are excluded from the claimed
  n50/chi200 or n50/chi50 workload. Their metadata are retained in the ignored
  dataset inventory.
- The public n20 DMRG test materializes 3^20 entries. This benchmark prohibits
  that allocation; one f64 array would consume about 26 GiB.
- Historical batches used main commits `dcc91f58` and `9ad67f2`. Current
  protocols pin `b881f39d`, including the main RNG changes. Earlier batches
  lack the required build receipts and cannot establish current behavior or
  timing. They retain their original identities and are not relabeled.
- `RankLimited` with a small independently measured error is not inherently
  anomalous. It is a stopping reason, not evidence of a defect.

Generated source snapshots, datasets, tensors, traces and reports belong only
under ignored `target/tree-rsi/paper-coverage/`. Do not commit them.
