# Partial apply: skipping the untouched zip-up nodes (#1186)

Evidence for the pass-through a partial `apply_linear_operator` uses for the
nodes its operator does not cover.

Harness: `crates/tensor4all-treetn/examples/benchmark_partial_apply_untouched.rs`
(public API only, so the same file builds against a checkout with and without the
pass-through). State: a comb, a degree-three scalar hub with three chains of 20
binary sites, bond dimensions `min(2^(i+1), 160)` and a per-leg amplitude weight
`exp(-0.1 * bond dimension)`; operator: a random MPO of bond dimension 14 over a
prefix of one branch, applied with `ApplyOptions::zipup()`, `center = hub`,
`rtol = 1e-8`.

Revisions: before = `8b756648` (`origin/main`, the base of the PR branch), after =
the PR branch head. Host: 32-core Linux, `RAYON_NUM_THREADS=1`, pinned to cores
8-11 with `taskset`, load average 0.9-2.0 during the runs.

## Reproduction

```bash
# after: the PR branch
cargo run --release -p tensor4all-treetn --example benchmark_partial_apply_untouched
# before: any checkout of origin/main 8b756648 with the same file copied into
# crates/tensor4all-treetn/examples/
RAYON_NUM_THREADS=1 taskset -c 8-11 \
  ./target/release/examples/benchmark_partial_apply_untouched
```

## Output, before (`8b756648`), three runs

```text
13-site comb (4 sites per branch, bonds <= 16): max |zipup - exact local| = 1.9190590211230242e-22
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 20 sites of one branch, 40 nodes skipped): 47.652 s for 3 applies (15.884 s each)
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 3 sites of one branch, 57 nodes skipped): 0.230 s for 3 applies (0.077 s each)
13-site comb (4 sites per branch, bonds <= 16): max |zipup - exact local| = 1.9190590211230242e-22
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 20 sites of one branch, 40 nodes skipped): 47.714 s for 3 applies (15.905 s each)
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 3 sites of one branch, 57 nodes skipped): 0.231 s for 3 applies (0.077 s each)
13-site comb (4 sites per branch, bonds <= 16): max |zipup - exact local| = 1.9190590211230242e-22
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 20 sites of one branch, 40 nodes skipped): 47.559 s for 3 applies (15.853 s each)
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 3 sites of one branch, 57 nodes skipped): 0.230 s for 3 applies (0.077 s each)
```

## Output, after (PR branch), three runs

```text
13-site comb (4 sites per branch, bonds <= 16): max |zipup - exact local| = 1.9852334701272664e-22
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 20 sites of one branch, 40 nodes skipped): 47.704 s for 3 applies (15.901 s each)
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 3 sites of one branch, 57 nodes skipped): 0.025 s for 3 applies (0.008 s each)
13-site comb (4 sites per branch, bonds <= 16): max |zipup - exact local| = 1.9852334701272664e-22
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 20 sites of one branch, 40 nodes skipped): 47.664 s for 3 applies (15.888 s each)
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 3 sites of one branch, 57 nodes skipped): 0.025 s for 3 applies (0.008 s each)
13-site comb (4 sites per branch, bonds <= 16): max |zipup - exact local| = 1.9852334701272664e-22
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 20 sites of one branch, 40 nodes skipped): 47.761 s for 3 applies (15.920 s each)
61-site comb (3 branches x 20 sites, bonds <= 160, bond-14 MPO on 3 sites of one branch, 57 nodes skipped): 0.025 s for 3 applies (0.008 s each)
```

## Reading

- Nodes skipped per apply: 40 of the 61 nodes when the operator covers a whole
  branch (the hub is a gap node too, but it receives the operator branch's
  remainder and is still factorized), and 57 of 61 when it covers three sites.
  The skip is allowed here because the comb is canonical at `hub` and the apply
  asks only for a relative policy (`rtol = 1e-8`) with no bond cap; a bond cap
  always keeps the untouched nodes on the factorization path.
  The three unit tests in `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs`
  assert the counts on a small comb (4 of 8 nodes) and on a chain (0), and the
  small comb's untouched node tensors against the input, bit for bit.
- The saving is proportional to how much of the per-node work sits in the
  untouched nodes, which depends on the shape. Covering a whole branch leaves the
  operator's own branch dominating the apply: 15.85-15.91 s before against
  15.89-15.92 s after per apply, i.e. no measurable change. Covering three sites of
  the same state makes the untouched nodes the whole job: 0.077 s before against
  0.008 s after per apply, about 9.6x.
- Both runs agree with the exact local (`naive`) apply on the 13-site comb to
  `1.9e-22` (before) and `2.0e-22` (after); the difference between them is the
  skipped node's rounding, not an accuracy change.
- The issue's 18-26 % per-node figure is their pipeline's accounting; this
  harness measures wall time on a different shape and does not reproduce it.
