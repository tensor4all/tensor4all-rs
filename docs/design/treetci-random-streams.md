# TreeTCI random stream ownership

The built-in random proposers and global finder use explicitly named
`ChaCha8Rng` streams at seeded entry points. `DefaultProposer` is deterministic
and consumes no draws. Candidate sampling retains its ordered subset and
previous-pass pivot retention rules.

| Entry point | Candidate draws | Global search starts |
| --- | --- | --- |
| `proposer.candidates(state, edge)` | A fresh ChaCha8 stream from `proposer.seed()` | No global search |
| `proposer.candidates_with_rng(state, edge, rng)` | Directly from `rng`; proposer seed ignored | No global search |
| `optimize_with_proposer` / `crossinterpolate2` | One ChaCha8 stream per call from `proposer.seed()` | A separate ChaCha8 stream per call from `options.seed`; entropy only for enabled searches with no seed |
| Caller-RNG optimizer / interpolation entry points | Directly from the supplied stream | Directly from that same supplied stream |

High-level candidate and global streams are independent so both seed controls
remain meaningful. Caller-stream entry points ignore both seed controls and
never derive a private generator. Reuse the same RNG across calls when the draw
sequence must continue; high-level seeded calls restart their generators.

There is no edge/rank/history hash in stream construction. The optimizer visits
edges in its established order; random proposers draw in candidate-generation
order, and global searches then draw their starts. Changing a proposer can
therefore change later global starts on a shared caller stream. High-level
separate streams isolate global starts from candidate draw consumption.

Fixed-seed trajectories changed in #824 from the prior unstable
`DefaultHasher`/`SmallRng` scheme. The named generator pins the algorithm;
this is not a guarantee that all sampling or numerical behavior will remain
bitwise identical through dependency/algorithm upgrades. Custom proposers
implement `candidates_with_rng` and obey the same caller-stream ownership
contract; `seed` controls only high-level seeded calls.
