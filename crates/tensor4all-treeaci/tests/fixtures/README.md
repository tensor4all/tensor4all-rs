# Unit-junction fixture

`issue1352_reopened_bits.txt` is a byte-identical copy of
`crates/tensor4all-treetci/tests/fixtures/issue1352_reopened_bits.txt`: the
frozen 32-by-32 table `g[z][x]` of Simon's reopened report in
[#1352](https://github.com/tensor4all/tensor4all-rs/issues/1352#issuecomment-6967309986),
column-major, one IEEE 754 binary64 bit pattern per hexadecimal word, preserving
exact zeros. It is copied rather than read across crates so that this crate's
tests stay self-contained. `tests/support/unit_junction.rs` builds the #1435
three-branch input from it.
