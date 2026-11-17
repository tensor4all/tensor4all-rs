# Same-library native scaling on clean downstream-pinned dependencies

Resource gate: **CLEAN**, exit **0**; all **130** independent processes and numerical checks completed.

[Frozen protocol](2026-11-17-clean-pins-protocol.md) · [All samples and source audit](2026-11-17-clean-pins-record.json) · [Original ai-run report](2026-11-17-clean-pins-ai-run.json).

Production base `86b27b91d03c0b7027be5547c44c45829220d8de`; harness `85cd96b326b5fe146330ae60cb710623fdb735f5`. Production code and numerical dependency identities are unchanged from the downstream lock. Tenferro is the downstream Git pin `32a4e98c` (0.10), Hataori `9d655157`, and threadteam registry 0.5.1; no local dependency override or repair is used. This is not a latest-upstream comparison.

Four physical cores are CPUs 8,10,12,14. Eight and sixteen compute participants intentionally oversubscribe those cores. Each process constructs its resources outside timing, primes for two seconds, runs a checked warmup, then times one public QTT API call. Each scalar has its own pool-free one-thread baseline. These are internal modes of the same library, not competing implementations.

## Real evaluator

| Mode | Lanes P | Threads/engine B | Total compute participants | Wall median ms | CPU median ms | Speedup / single | Paired 95% speedup interval | Relative MAD | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| engine | 1 | 1 | 1 | 111.429 | 111.420 | 1.000× | [1.000, 1.000] | 0.37% | no_clear_difference |
| engine | 1 | 2 | 2 | 59.468 | 118.408 | 1.874× | [1.870, 1.915] | 0.31% | faster |
| engine | 1 | 4 | 4 | 32.795 | 124.156 | 3.398× | [3.392, 3.440] | 0.16% | faster |
| engine | 1 | 8 | 8 | 39.792 | 137.799 | 2.800× | [2.495, 2.965] | 3.10% | faster |
| engine | 1 | 16 | 16 | 48.817 | 167.698 | 2.283× | [2.100, 2.748] | 6.18% | faster |
| lanes | 1 | 1 | 1 | 112.516 | 112.484 | 0.990× | [0.984, 1.003] | 0.64% | no_clear_difference |
| lanes | 2 | 1 | 2 | 62.046 | 106.778 | 1.796× | [1.763, 1.863] | 0.73% | faster |
| lanes | 4 | 1 | 4 | 35.422 | 99.992 | 3.146× | [3.062, 3.274] | 1.46% | faster |
| lanes | 8 | 1 | 8 | 38.610 | 104.313 | 2.886× | [2.292, 3.078] | 4.12% | faster |
| lanes | 16 | 1 | 16 | 39.706 | 102.672 | 2.806× | [2.764, 2.984] | 1.81% | faster |
| mixed | 2 | 2 | 4 | 87.533 | 121.096 | 1.273× | [1.250, 1.382] | 1.80% | faster |
| mixed | 4 | 2 | 8 | 102.539 | 122.779 | 1.087× | [1.082, 1.106] | 0.66% | no_clear_difference |
| mixed | 8 | 2 | 16 | 109.153 | 121.520 | 1.021× | [1.015, 1.036] | 1.36% | no_clear_difference |

## Complex evaluator

| Mode | Lanes P | Threads/engine B | Total compute participants | Wall median ms | CPU median ms | Speedup / single | Paired 95% speedup interval | Relative MAD | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| engine | 1 | 1 | 1 | 113.079 | 113.070 | 1.000× | [1.000, 1.000] | 0.75% | no_clear_difference |
| engine | 1 | 2 | 2 | 60.636 | 120.699 | 1.865× | [1.848, 1.901] | 0.22% | faster |
| engine | 1 | 4 | 4 | 33.593 | 129.447 | 3.366× | [3.341, 3.404] | 0.79% | faster |
| engine | 1 | 8 | 8 | 42.624 | 142.059 | 2.653× | [2.377, 2.925] | 6.89% | faster |
| engine | 1 | 16 | 16 | 49.500 | 167.977 | 2.284× | [2.006, 2.345] | 1.80% | faster |
| lanes | 1 | 1 | 1 | 114.853 | 114.838 | 0.985× | [0.971, 0.997] | 0.39% | no_clear_difference |
| lanes | 2 | 1 | 2 | 62.585 | 110.664 | 1.807× | [1.796, 1.819] | 0.13% | faster |
| lanes | 4 | 1 | 4 | 36.062 | 109.584 | 3.136× | [3.084, 3.258] | 0.56% | faster |
| lanes | 8 | 1 | 8 | 38.934 | 103.995 | 2.904× | [2.867, 3.030] | 0.20% | faster |
| lanes | 16 | 1 | 16 | 40.412 | 107.133 | 2.798× | [2.725, 3.084] | 1.90% | faster |
| mixed | 2 | 2 | 4 | 84.684 | 121.943 | 1.335× | [1.294, 1.379] | 0.60% | faster |
| mixed | 4 | 2 | 8 | 103.794 | 123.748 | 1.089× | [1.078, 1.114] | 0.42% | no_clear_difference |
| mixed | 8 | 2 | 16 | 109.857 | 123.864 | 1.029× | [1.008, 1.043] | 0.83% | no_clear_difference |

## Correctness, placement and limits

Every warmup and timed output passed the independent 16×16 grid oracle at the original 1e-4 tolerance, exact grid ownership and canonical prefix order. Warmup/time results agree; callback counts, evaluator points, prefixes, core bytes and all grid values agree across configurations within each scalar (existing 1e-10 diagnostic comparison tolerance). The full values and every raw observation remain in the record.

Constructor masks narrowed in **50/130** processes. This is recorded behavior of the unmodified constructors, not silently repaired by the harness. Initial, per-constructor and warmup callback masks are retained for every process. The configurations request resource budgets inside one shared process CPU allocation; they do not promise disjoint per-engine domains.

All 30 mixed-mode processes observed only these callback masks: `['10', '8']`. Their first B2 construction narrowed the creator from the four-core allocation to CPU8, and subsequent constructors retained CPU8. This directly records loss of the intended allocation in the clean pinned stack; it is not evidence about latest upstream. It also prevents interpreting P×B as evidence that all four physical cores actually executed evaluator work.

The process-CPU clock brackets the primary wall timer. Callback counters and affinity queries are absent from the timed callback monomorphization. Setup, priming, warmup and dense verification are outside primary timing. No previous withdrawn sample is reused; no local repair, MPI speedup or latest-upstream result is inferred. More than 15% relative MAD is marked imprecise while retaining every sample.
