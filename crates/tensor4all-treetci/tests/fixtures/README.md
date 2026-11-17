# Three-branch rank-deficiency fixtures

`issue1352_m12_bits.txt` and `issue1352_m14_bits.txt` freeze the target matrices
from [Simon's issue #1352 reproducer](https://github.com/tensor4all/tensor4all-rs/issues/1352#issuecomment-6939597508).
They contain `g(a, c)` for `0 <= a, c < 32`, in column-major order: one
column (fixed `c`, increasing `a`) per line. Each hexadecimal word is an IEEE
754 binary64 bit pattern. Tests decode the bits exactly; no RNG is involved
in constructing the target at test time.

The source uses rand 0.9.5 `StdRng`, seed 42, to generate orthonormal factors
with one-pass Gram-Schmidt. Both factors have the Gaussian envelope centered
at 16 with width 4; the left factor is additionally zero at every fourth
entry. The matrices have 12 terms with decay 0.3 and 14 terms with decay 0.2,
respectively. Each case restarts the source RNG: the right factors of the two
cases differ, so the 14-term case is not a truncation of the 12-term case.
The frozen matrices were generated using the source's arithmetic on Linux
x86_64 with rustc 1.98.1.

The test target is zero unless the middle branch has coordinate zero. Its
full 32-by-32-by-32 reference is small enough to materialize once. The
fixture's generation provenance is retained here rather than committing a
second linear-algebra implementation or relying on `StdRng` in tests.
