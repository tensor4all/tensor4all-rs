# Rank-deficiency fixtures from issue #1352

`issue1352_m12_bits.txt` and `issue1352_m14_bits.txt` are byte-for-byte copies
of `crates/tensor4all-treetci/tests/fixtures/`, kept here so this crate stays
independent. They freeze the target matrices from
[Simon's issue #1352 reproducer](https://github.com/tensor4all/tensor4all-rs/issues/1352#issuecomment-6939597508):
`g(a, c)` for `0 <= a, c < 32`, in column-major order, one column (fixed `c`,
increasing `a`) per line. Each hexadecimal word is an IEEE 754 binary64 bit
pattern, decoded exactly; no RNG is involved in constructing the target at
test time.

The source uses rand 0.9.5 `StdRng`, seed 42, to generate orthonormal factors
with one-pass Gram-Schmidt. Both factors have the Gaussian envelope centered
at 16 with width 4; the left factor is additionally zero at every fourth
entry. The matrices have 12 terms with decay 0.3 and 14 terms with decay 0.2.
They were generated on Linux x86_64 with rustc 1.98.1.

`tests/issue1358_chain_global_residual.rs` evaluates
`f(a, b, c) = g(a, c)` if `b = 0`, else `0`, on 5-bit binary encodings of `a`,
`b` and `c`, laid out on a chain (#1358). The full 2^15 reference is small
enough to materialize once per run.
