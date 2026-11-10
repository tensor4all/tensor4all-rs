# SimpleTT key encode cost (#1224)

`#1146` moved `TTCache` onto core's bit-packed `IndexKey`. Its quiet-host run at
`06ec80f6` measured pure-hit `n=300` at 2.32–2.40× the original, with `encode`
taking 1623.8 ns of a 1930.4 ns call. This directory records the fix and the
candidates that were measured and not taken.

## Fix

The fix is in `crates/tensor4all-core/src/index_key/`. The 129–512-bit arms
shifted a `bnum` value once per component, touching every digit each time. They
now pack into `u64` digits on the stack through a per-indexer digit layout
(component range per digit, plus fields that straddle a digit boundary) and
convert once. The 65–128-bit arm uses the same packer. The bounds check is a
branch-free scan that locates the bad position only on failure. Key values,
equality and hashing are unchanged, and the boxed wide arms stay boxed.

### Where this was measured

On `main`, no library code calls `FlatIndexer` yet: `TTCache` still has its own
private indexer, and `#1146` is what moves it onto core's keys. The paired
before/after below was therefore measured with the core change applied on top
of the `#1146` stack. `before` is `#1146` at `06ec80f6`; `after` is `06b75544`,
which is `06ec80f6` plus this core change, the encode benchmark and nothing
else. The core diff in this directory's PR is the same change rebased onto
`main`. It differs only in keeping `main`'s offsets for singleton fields;
resetting those to zero is `#1146`'s own fix. The encode-only numbers measure
`FlatIndexer` alone, so they apply to `main` directly.

## Host and protocol

- **Host: evo2**, AMD Ryzen AI MAX+ 395, 32 logical CPUs, Linux 7.0.0-42,
  rustc 1.99.1 (LLVM 23.1.1). **This is not the `#1146` host** (evo, Ryzen AI 9
  HX 470), which is not available for benchmarks. The before/after comparison
  below is therefore made on evo2: the `before` run reproduces the `#1146`
  regression there.
- Driver: `benchmarks/rust/run_simplett_index_key_pairs.py` from `#1146`. Each
  copy changes only the `final` arm SHA, as the `#1146` measurement did:
  `before` = `06ec80f6…`, `after` = `06b7554442450b70c8bdef068bb2f7b3a0c6f0b2`,
  `memo` = `2985d117…`. `original` = `097e2927`, `shared` = `65278707`.
- Arguments as in `#1146`: `--cpu 4 --idle-seconds 60 --idle-max-load 2.0`,
  driver process pinned to the other L3 domain (`taskset -c 8-15`; CPU 4 is in
  `0-7,16-23`). Three rotated rounds × three arms × 30 cells = 270 cells per run,
  20 Criterion samples, 0.2 s warm-up, 0.5 s measurement (the harness's
  settings), all thread variables 1.
- Every run passed the idle gate on its first attempt and all noise gates. See
  `*/idle-check.json` and `*/summary.json`.

### A run that was discarded

The first `before` and `memo` runs shared one `--target-dir` with the `after`
run. All three runs recorded the same `final` binary SHA-256 (`b98f57e0…`).
`git archive` stamps exported files with the commit time, which is older than
the artifacts already built, so cargo reused the `after` binary for the other
two revisions. Those runs are not kept here. They were redone with one target
directory per run, and their `final` binaries are distinct: `89bf2bac…` for
`before` and `fd0e8080…` for `memo`. The `after` run was the first build into a
fresh target directory. The driver's identical-executables guard only compares
arms within one run, so it cannot detect this. The driver lives in `#1146`,
and the fix for it is there too (`d7a4c35a`): each arm builds in
`<--target-dir>/<arm>/<full SHA>`, with a regression test.

## Encode alone (`crates/tensor4all-core/benches/index_key_encode.rs`)

Pinned to CPU 4 behind an idle gate (load 1.79 → 1.92, limit 2.0). Three rotated
rounds; 0.5 s warm-up, 2 s measurement, 100 samples. The table gives the median
of the per-round Criterion means in ns (all rounds are in `encode/summary.json`).
`before` = `c1647a52` (`06ec80f6` plus this benchmark file only).

| binary components (= bits) | before | after | after/before | old `Vec<usize>` key + hash (after binary) |
|---:|---:|---:|---:|---:|
| 16 | 19.9 | 18.1 | 0.906 | 19.8 |
| 50 | 45.5 | 37.3 | 0.820 | 35.0 |
| 64 | 58.8 | 44.2 | 0.752 | 40.1 |
| 65 | 112.2 | 48.6 | 0.433 | 42.1 |
| 100 | 170.5 | 67.5 | 0.396 | 70.4 |
| 128 | 219.0 | 83.2 | 0.380 | 75.9 |
| 129 | 911.8 | 100.7 | 0.110 | 77.3 |
| **150** | **1068.6** | **113.8** | **0.106** | 142.4 |
| 200 | 1392.1 | 142.7 | 0.103 | 153.4 |
| 256 | 1733.8 | 172.0 | 0.099 | 184.4 |
| 257 | 2004.3 | 170.5 | 0.085 | 204.4 |
| 300 | 2326.7 | 196.2 | 0.084 | 246.9 |
| 513 (limbs, not changed) | 977.1 | 919.2 | 0.941 | 373.9 |

## Paired runs (driver)

Median of the three per-round paired ratios, with the per-round minimum and
maximum.

| route / case | before: final/original | after: final/original | after: final/shared |
|---|---:|---:|---:|
| left_hit/n300_chi1 | 2.4129 [2.3238, 2.4347] | **0.6172** [0.6138, 0.6214] | 0.6234 |
| left_hit/n300_chi8 | 2.3972 [2.3302, 2.4423] | **0.6027** [0.5931, 0.6220] | 0.6264 |
| right_hit/n300_chi1 | 2.3797 [2.3463, 2.3886] | **0.6156** [0.5931, 0.6164] | 0.6314 |
| right_hit/n300_chi8 | 2.4115 [2.3922, 2.4147] | **0.6088** [0.5888, 0.6179] | 0.6249 |
| hit, n32/n100 (8 cells) | 0.7442–0.8666 | 0.6866–0.8531 | 0.6760–0.8538 |
| cold_fixed (6 cells) | 0.5291–0.7808 | 0.3506–0.7612 | 0.3386–0.7508 |
| warm_auto (6 cells) | 0.6594–0.8654 | 0.3328–0.8000 | 0.1565–0.5881 |
| warm_fixed (6 cells) | 0.9095–1.9363 | 0.5197–1.0366 | 0.2872–0.9053 |

Driver decision: `before` FAIL (batch vs original 1.9363, hit geomean vs shared
1.1325); `after` **PASS** (batch vs original 1.0366, batch vs shared 0.9053, hit
geomean vs shared 0.6941). Max noise in `after`: load 1.118, two-sided CI
half-width 1.21%, repetition CV 7.55%.

Two `warm_fixed` cells are still slower than original in `after`:
`n100_chi1_batch1024` at 1.0366 [1.0282, 1.0408] and `n100_chi8_batch1024` at
1.0326 [1.0243, 1.0371]. Both are within the 1.10 gate, and both are faster
than `before` (1.1426 and 1.1375). Full 30-cell tables: `*/summary.json`, and
`table.py <dir>` prints them.

## Candidates not taken

- **Memoise the last key per length** (`2985d117`, a local experiment commit on `06ec80f6`, not pushed). The pure-hit
  cells are the fastest of any arm, at 0.4195–0.4462 of original for `n=300`,
  because the benchmark repeats one index. The other routes pay for it:
  `warm_fixed/n300` stays at 1.9001/1.9123, and cold batches are slower than
  `before`. For example, `cold_fixed/n100_chi1_batch64` takes 179.6572 µs
  against 160.4517 µs, and `cold_fixed/n300_chi1_batch64` takes 1268.5857 µs
  against 1167.6421 µs. Both are cross-run comparisons; each miss copies the
  index. The larger stack frame also overflows the stack in
  `batch_cache_supports_keys_wider_than_1024_bits`. A memo only helps a caller
  that repeats the identical index, and TCI-style batches do not.
- **`Vec<usize>` key for some widths.** These are encode-only medians, in ns,
  with the fix applied: packed `encode` against building and hashing the old
  `Vec<usize>` key.
  - The packed key is cheaper at 16 bits (18.1 against 19.8), 100 bits (67.5
    against 70.4) and every width from 150 to 300 bits (150 bits: 113.8 against
    142.4; 300 bits: 196.2 against 246.9).
  - The packed key is slightly more expensive at 50 bits (37.3 against 35.0), 64
    (44.2 against 40.1), 65 (48.6 against 42.1) and 128 (83.2 against 75.9), and
    more so at 129 (100.7 against 77.3).
  - At 513 bits it is far more expensive: the limb arm, which this change does
    not touch, takes 919.2 against 373.9.

  So no single threshold separates the two key types on 16–300 bits: the
  `Vec` key wins on 50–129 bits and loses on either side of that range. Where
  it wins, the difference is at most 23.4 ns per lookup. The cache-level hit
  cells in that range (`n100`, 50-bit keys) already run at 0.6866–0.7087 of the
  `original` arm, which uses the `Vec` key, in the `after` run. On that
  evidence a second key type is not taken for widths up to 512 bits. Above 512
  bits the `Vec` key is cheaper in this encode-only measurement, but no
  `TTCache` case in the harness reaches that width, so this run decides nothing
  there.
- **Remove the allocation (unboxed 256-bit arm)** (`0a0f9889`, a local experiment commit, not pushed). Encode at 150
  bits falls from 113.8 to 99.7 ns, a saving of 14.1 ns, or about 4% of the
  0.333 µs hit. In return `IndexKey` would exceed the 48-byte bound that
  `the_key_enum_stays_small` enforces, and every narrow key in a map would pay
  for the larger footprint. 257–512 bits would still allocate.

## Not measured

- The `#1146` host (evo). The earlier development measurements taken there are
  not used.
- The limb arm above 512 bits (unchanged): its encode is 919.2 ns at 513 bits,
  against 373.9 ns for the `Vec` key. No `TTCache` case in the harness reaches
  that width.
- Memory, multi-threaded throughput, and the harness's other groups.
