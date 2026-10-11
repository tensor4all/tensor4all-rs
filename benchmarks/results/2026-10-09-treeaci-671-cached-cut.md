# TreeACI #671: cached-cut repair and real R=10 attribution

Related: [#671](https://github.com/tensor4all/tensor4all-rs/issues/671), [#854](https://github.com/tensor4all/tensor4all-rs/issues/854).
[Repair and closure scope](../../docs/worklogs/2026-10-09-treeaci-671-cached-cut.md).

## Production comparison protocol

Baseline: pristine merged `fb6ab269931fed3324edbaa3c64bc7714861ad3c`, built in an isolated target. Candidate code: `462252a93184122cda8da0cf84d00371fc228d60`. Benchmark and lockfile are unchanged; separate executable paths preserve both builds. Source stamps, hashes, release/default CPU backend, fixture, seed and tolerance are checked. Diagnostics are disabled. CPU 2, all six controlled Rayon/BLAS/Tenferro variables one; AMD Ryzen 9 6900HX, virtualized Linux, Rust 1.98.1. No compilation or other numerical workload overlaps measurements.

The existing 120-case branch-cost protocol covers f64/Complex64, physical dimension 2, degrees 2/3/4, four bond profiles, ACI and cold/warm queries at 8/32 points. An independent A/A noise study (five alternating pairs, five repetitions) is **INCONCLUSIVE**: degree 2/profile 3/f64/cold/batch 8 exceeds the unchanged relative-MAD 0.20 gate. It is preserved. A complete, separately declared nine-pair/nine-repetition confirmation, with the same oracle/dispersion/load/frequency gates, has zero failures. Its maximum paired-ratio interval upper bound is 1.219462; tiny cold calls remain noisy. No formal regression bound is declared.

The candidate comparison uses nine alternating pairs and nine repetitions for every case. It has zero validity/oracle failures and a **DESCRIPTIVE** verdict. All 1,080 paired records have exactly matching evaluated-point counts, maximum oracle errors, fixture dimensions and frame/arena cache observations. Maximum relative oracle error is 2.655255541087604e-15. Timing ratios describe these workloads and do not establish a universal speedup or formal non-regression certificate. Cold negative controls retain their variation; none are removed.

Fixture SHA-256: `a6e19eadddd88724bcadf3576c6666027cc2eb5d11e55b242ac2e34d2fe64a14`. Lockfile SHA-256: `ead4c85a9ded2807a08b246bee41c3684d87109eaf0dac2951e2469ae7b14520`.

| Executable | SHA-256 |
| --- | --- |
| 671-baseline-branch | `ce4cfa70a962d35125808b9276956337c9614170dd0cdbc04e0ec44075d6ac5b` |
| 671-candidate-branch | `7ac2a5be3a379a6596a9b30e80747cd4a6f3d9ec94f59122d2c81e1f52d0224d` |
| 671-baseline-cache | `08a78f4fdae0b648db81643b8dde23a3f5313bd8d0230a1a2679f4a8c1b07c89` |
| 671-candidate-cache | `33d74d477bb54309993783c02ab3595f063d774fd92ba4368f536d062de9151d` |
| 671-replay-no-diagnostics | `069c7e852b6f8eec80f7ab122055bdbb4853677ce228fec952f1fc460b7bba32` |
| 671-candidate-replay-no-diagnostics | `d6cc862640dc3020bef543deeecf53405455ea8a44f25182b9269adb597ef65b` |

## Branch-cost aggregate ratios

Ratios are candidate/baseline elapsed time. Lower means less elapsed time.

| Degree | Mode | Cases | Median ratio | Minimum | Maximum |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | aci | 8 | 0.507601 | 0.448466 | 0.560263 |
| 2 | cold | 16 | 0.997126 | 0.733204 | 1.031593 |
| 2 | warm | 16 | 0.441695 | 0.379475 | 0.472875 |
| 3 | aci | 8 | 0.680338 | 0.565408 | 0.799226 |
| 3 | cold | 16 | 1.003351 | 0.990090 | 1.013374 |
| 3 | warm | 16 | 0.403940 | 0.383199 | 0.415506 |
| 4 | aci | 8 | 0.673301 | 0.583957 | 0.769724 |
| 4 | cold | 16 | 0.995059 | 0.972848 | 1.023056 |
| 4 | warm | 16 | 0.351413 | 0.341886 | 0.364502 |

## Complete branch-cost case summaries

Times are microseconds. Intervals bootstrap the nine paired ratios and are descriptive. MAD is relative median absolute deviation of per-pair medians.

| Degree | Profile | Scalar | Mode | Batch | Baseline us | Candidate us | Ratio | 95% interval | Baseline MAD | Candidate MAD |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 2 | 0 | f64 | aci | 1 | 3563.585 | 1993.395 | 0.560263 | [0.557326, 0.564336] | 0.009480 | 0.008218 |
| 2 | 0 | f64 | cold | 8 | 18.144 | 18.445 | 1.005999 | [0.995023, 1.026598] | 0.007165 | 0.007102 |
| 2 | 0 | f64 | warm | 8 | 5.420 | 2.415 | 0.445387 | [0.434317, 0.451397] | 0.009225 | 0.012008 |
| 2 | 0 | f64 | cold | 32 | 29.796 | 30.146 | 1.001362 | [0.982850, 1.037346] | 0.008088 | 0.021595 |
| 2 | 0 | f64 | warm | 32 | 11.291 | 5.230 | 0.463154 | [0.454256, 0.480611] | 0.007174 | 0.013384 |
| 3 | 0 | f64 | aci | 1 | 5686.833 | 3202.627 | 0.565408 | [0.557380, 0.569619] | 0.008592 | 0.008512 |
| 3 | 0 | f64 | cold | 8 | 21.100 | 20.980 | 0.990090 | [0.963471, 1.044389] | 0.008104 | 0.014824 |
| 3 | 0 | f64 | warm | 8 | 6.272 | 2.494 | 0.397640 | [0.392391, 0.417952] | 0.014349 | 0.016038 |
| 3 | 0 | f64 | cold | 32 | 34.154 | 34.244 | 1.000599 | [0.985361, 1.030509] | 0.023160 | 0.019624 |
| 3 | 0 | f64 | warm | 32 | 13.195 | 5.360 | 0.404464 | [0.400573, 0.416868] | 0.012126 | 0.020522 |
| 4 | 0 | f64 | aci | 1 | 6359.508 | 4038.918 | 0.630359 | [0.624519, 0.641117] | 0.008266 | 0.012202 |
| 4 | 0 | f64 | cold | 8 | 24.977 | 24.606 | 0.972848 | [0.968227, 0.996355] | 0.012051 | 0.011379 |
| 4 | 0 | f64 | warm | 8 | 7.504 | 2.565 | 0.341886 | [0.338167, 0.351325] | 0.007996 | 0.007797 |
| 4 | 0 | f64 | cold | 32 | 41.147 | 42.620 | 1.023056 | [1.006180, 1.077568] | 0.011179 | 0.028695 |
| 4 | 0 | f64 | warm | 32 | 15.830 | 5.681 | 0.358647 | [0.349116, 0.369082] | 0.011371 | 0.021123 |
| 2 | 1 | f64 | aci | 1 | 4248.843 | 2210.683 | 0.521079 | [0.518793, 0.522776] | 0.006315 | 0.005044 |
| 2 | 1 | f64 | cold | 8 | 19.907 | 20.188 | 0.997539 | [0.977278, 1.028748] | 0.020646 | 0.011393 |
| 2 | 1 | f64 | warm | 8 | 5.530 | 2.415 | 0.440145 | [0.417668, 0.447855] | 0.018083 | 0.012422 |
| 2 | 1 | f64 | cold | 32 | 32.672 | 32.772 | 1.004598 | [0.982187, 1.008866] | 0.007315 | 0.016508 |
| 2 | 1 | f64 | warm | 32 | 11.391 | 5.320 | 0.466217 | [0.457128, 0.479814] | 0.003599 | 0.007519 |
| 3 | 1 | f64 | aci | 1 | 6950.367 | 4209.569 | 0.608886 | [0.603477, 0.617005] | 0.002864 | 0.002804 |
| 3 | 1 | f64 | cold | 8 | 22.523 | 22.352 | 1.006768 | [0.967715, 1.027934] | 0.020424 | 0.004966 |
| 3 | 1 | f64 | warm | 8 | 6.422 | 2.535 | 0.395102 | [0.388097, 0.402201] | 0.007786 | 0.016174 |
| 3 | 1 | f64 | cold | 32 | 36.639 | 36.940 | 1.008215 | [0.977814, 1.031952] | 0.010399 | 0.017055 |
| 3 | 1 | f64 | warm | 32 | 13.305 | 5.470 | 0.411124 | [0.402878, 0.417763] | 0.011274 | 0.007313 |
| 4 | 1 | f64 | aci | 1 | 8862.670 | 5787.643 | 0.651066 | [0.645595, 0.660881] | 0.007471 | 0.006803 |
| 4 | 1 | f64 | cold | 8 | 25.789 | 25.718 | 1.007583 | [0.970589, 1.012890] | 0.017837 | 0.025702 |
| 4 | 1 | f64 | warm | 8 | 7.504 | 2.625 | 0.349009 | [0.340688, 0.359679] | 0.016125 | 0.019048 |
| 4 | 1 | f64 | cold | 32 | 43.571 | 43.712 | 0.999319 | [0.985665, 1.007543] | 0.007138 | 0.016288 |
| 4 | 1 | f64 | warm | 32 | 15.980 | 5.621 | 0.356312 | [0.347683, 0.363660] | 0.015707 | 0.016189 |
| 2 | 2 | f64 | aci | 1 | 5126.612 | 2444.112 | 0.478162 | [0.472076, 0.485728] | 0.006248 | 0.012461 |
| 2 | 2 | f64 | cold | 8 | 32.481 | 32.241 | 0.983871 | [0.854301, 1.039217] | 0.024968 | 0.045346 |
| 2 | 2 | f64 | warm | 8 | 5.941 | 2.455 | 0.411547 | [0.393470, 0.418826] | 0.026931 | 0.024440 |
| 2 | 2 | f64 | cold | 32 | 62.798 | 64.782 | 1.031593 | [0.962874, 1.080347] | 0.034778 | 0.027847 |
| 2 | 2 | f64 | warm | 32 | 12.544 | 5.640 | 0.443245 | [0.430485, 0.460660] | 0.021524 | 0.033688 |
| 3 | 2 | f64 | aci | 1 | 9152.045 | 6479.092 | 0.704538 | [0.697676, 0.711358] | 0.004828 | 0.008203 |
| 3 | 2 | f64 | cold | 8 | 25.839 | 25.868 | 0.992290 | [0.954266, 1.019688] | 0.015171 | 0.010901 |
| 3 | 2 | f64 | warm | 8 | 6.413 | 2.525 | 0.398409 | [0.384556, 0.416001] | 0.016997 | 0.011881 |
| 3 | 2 | f64 | cold | 32 | 62.187 | 62.548 | 1.009709 | [0.982378, 1.028511] | 0.012237 | 0.004333 |
| 3 | 2 | f64 | warm | 32 | 14.217 | 5.831 | 0.411744 | [0.400811, 0.419592] | 0.014138 | 0.008575 |
| 4 | 2 | f64 | aci | 1 | 12883.023 | 8960.614 | 0.695537 | [0.688411, 0.697223] | 0.003853 | 0.007223 |
| 4 | 2 | f64 | cold | 8 | 30.247 | 30.387 | 1.020010 | [0.996338, 1.035479] | 0.011935 | 0.009544 |
| 4 | 2 | f64 | warm | 8 | 7.594 | 2.635 | 0.347331 | [0.342619, 0.351116] | 0.011851 | 0.015180 |
| 4 | 2 | f64 | cold | 32 | 69.360 | 70.212 | 1.007390 | [0.990298, 1.023659] | 0.018930 | 0.012562 |
| 4 | 2 | f64 | warm | 32 | 16.932 | 5.901 | 0.348313 | [0.336799, 0.352255] | 0.012993 | 0.011862 |
| 2 | 3 | f64 | aci | 1 | 6825.731 | 3049.759 | 0.448466 | [0.443543, 0.450673] | 0.007410 | 0.004635 |
| 2 | 3 | f64 | cold | 8 | 92.985 | 69.301 | 0.733204 | [0.652288, 1.283576] | 0.085229 | 0.124789 |
| 2 | 3 | f64 | warm | 8 | 7.454 | 2.895 | 0.387780 | [0.379315, 0.404104] | 0.008049 | 0.013817 |
| 2 | 3 | f64 | cold | 32 | 99.397 | 98.575 | 0.994647 | [0.924343, 1.011870] | 0.019659 | 0.024895 |
| 2 | 3 | f64 | warm | 32 | 13.806 | 5.711 | 0.411658 | [0.403520, 0.426899] | 0.016732 | 0.021012 |
| 3 | 3 | f64 | aci | 1 | 12637.852 | 9548.499 | 0.751710 | [0.746122, 0.764009] | 0.009963 | 0.004330 |
| 3 | 3 | f64 | cold | 8 | 59.491 | 59.291 | 0.996638 | [0.944579, 1.037440] | 0.011615 | 0.006089 |
| 3 | 3 | f64 | warm | 8 | 7.554 | 2.895 | 0.385605 | [0.369395, 0.395358] | 0.014562 | 0.010363 |
| 3 | 3 | f64 | cold | 32 | 76.934 | 76.965 | 1.002979 | [0.971482, 1.031240] | 0.015507 | 0.006237 |
| 3 | 3 | f64 | warm | 32 | 14.487 | 5.831 | 0.402777 | [0.397667, 0.405604] | 0.005522 | 0.007031 |
| 4 | 3 | f64 | aci | 1 | 17301.523 | 12832.247 | 0.744864 | [0.734935, 0.752677] | 0.001034 | 0.007131 |
| 4 | 3 | f64 | cold | 8 | 58.650 | 58.049 | 0.990090 | [0.974837, 1.010364] | 0.011628 | 0.022429 |
| 4 | 3 | f64 | warm | 8 | 8.616 | 3.016 | 0.349417 | [0.335598, 0.358171] | 0.015088 | 0.019894 |
| 4 | 3 | f64 | cold | 32 | 80.652 | 79.138 | 0.990040 | [0.953386, 0.998978] | 0.017768 | 0.012548 |
| 4 | 3 | f64 | warm | 32 | 17.252 | 6.011 | 0.348481 | [0.337874, 0.358419] | 0.012752 | 0.004991 |
| 2 | 0 | c64 | aci | 1 | 3946.033 | 2201.085 | 0.556211 | [0.550179, 0.557797] | 0.005883 | 0.006837 |
| 2 | 0 | c64 | cold | 8 | 19.637 | 19.196 | 0.996924 | [0.954626, 1.004768] | 0.036716 | 0.016722 |
| 2 | 0 | c64 | warm | 8 | 5.631 | 2.525 | 0.443772 | [0.434331, 0.450791] | 0.016161 | 0.015842 |
| 2 | 0 | c64 | cold | 32 | 30.978 | 31.439 | 1.007452 | [0.974598, 1.030829] | 0.004197 | 0.021343 |
| 2 | 0 | c64 | warm | 32 | 11.231 | 5.300 | 0.472875 | [0.464707, 0.484148] | 0.019678 | 0.007547 |
| 3 | 0 | c64 | aci | 1 | 5977.831 | 3508.862 | 0.588073 | [0.580929, 0.599770] | 0.007944 | 0.006744 |
| 3 | 0 | c64 | cold | 8 | 21.750 | 21.841 | 1.003724 | [0.980591, 1.032613] | 0.007770 | 0.014239 |
| 3 | 0 | c64 | warm | 8 | 6.362 | 2.605 | 0.407953 | [0.403312, 0.418631] | 0.012575 | 0.011516 |
| 3 | 0 | c64 | cold | 32 | 35.266 | 34.885 | 1.007861 | [0.988671, 1.027894] | 0.021040 | 0.017257 |
| 3 | 0 | c64 | warm | 32 | 13.285 | 5.490 | 0.415506 | [0.405937, 0.425369] | 0.014302 | 0.005464 |
| 4 | 0 | c64 | aci | 1 | 8010.760 | 4676.234 | 0.583957 | [0.583300, 0.585743] | 0.006986 | 0.006468 |
| 4 | 0 | c64 | cold | 8 | 25.828 | 25.678 | 0.996128 | [0.976121, 1.006270] | 0.007705 | 0.014020 |
| 4 | 0 | c64 | warm | 8 | 7.755 | 2.715 | 0.346032 | [0.338885, 0.356349] | 0.006447 | 0.011050 |
| 4 | 0 | c64 | cold | 32 | 43.041 | 42.691 | 0.992440 | [0.982604, 1.020465] | 0.013499 | 0.012930 |
| 4 | 0 | c64 | warm | 32 | 15.860 | 5.721 | 0.364502 | [0.358328, 0.368754] | 0.013871 | 0.010488 |
| 2 | 1 | c64 | aci | 1 | 4566.849 | 2414.086 | 0.529686 | [0.524118, 0.537500] | 0.007001 | 0.008395 |
| 2 | 1 | c64 | cold | 8 | 21.140 | 21.139 | 0.993346 | [0.968788, 1.013324] | 0.015184 | 0.015185 |
| 2 | 1 | c64 | warm | 8 | 5.700 | 2.545 | 0.437210 | [0.436275, 0.448109] | 0.015965 | 0.019646 |
| 2 | 1 | c64 | cold | 32 | 34.325 | 34.445 | 0.997328 | [0.986164, 1.023230] | 0.015178 | 0.006097 |
| 2 | 1 | c64 | warm | 32 | 11.542 | 5.360 | 0.464391 | [0.459136, 0.477475] | 0.020880 | 0.009328 |
| 3 | 1 | c64 | aci | 1 | 7001.855 | 4615.061 | 0.656137 | [0.651812, 0.661252] | 0.003398 | 0.004273 |
| 3 | 1 | c64 | cold | 8 | 23.464 | 23.634 | 1.004270 | [0.997360, 1.028073] | 0.011976 | 0.005120 |
| 3 | 1 | c64 | warm | 8 | 6.473 | 2.635 | 0.407076 | [0.391484, 0.423425] | 0.012205 | 0.003795 |
| 3 | 1 | c64 | cold | 32 | 37.200 | 37.360 | 1.005367 | [0.985880, 1.022903] | 0.006720 | 0.010974 |
| 3 | 1 | c64 | warm | 32 | 13.305 | 5.440 | 0.412448 | [0.399537, 0.420853] | 0.012777 | 0.011029 |
| 4 | 1 | c64 | aci | 1 | 10269.685 | 6489.820 | 0.629508 | [0.621636, 0.638697] | 0.003688 | 0.012611 |
| 4 | 1 | c64 | cold | 8 | 26.971 | 27.271 | 1.006856 | [0.960217, 1.027694] | 0.023433 | 0.014301 |
| 4 | 1 | c64 | warm | 8 | 7.664 | 2.665 | 0.354671 | [0.344179, 0.360886] | 0.006524 | 0.011257 |
| 4 | 1 | c64 | cold | 32 | 44.704 | 44.944 | 1.011496 | [0.984667, 1.014947] | 0.008970 | 0.007587 |
| 4 | 1 | c64 | warm | 32 | 15.689 | 5.731 | 0.362738 | [0.356402, 0.366476] | 0.005673 | 0.024429 |
| 2 | 2 | c64 | aci | 1 | 5303.344 | 2626.454 | 0.494123 | [0.491202, 0.496967] | 0.005016 | 0.001518 |
| 2 | 2 | c64 | cold | 8 | 34.034 | 33.303 | 0.967500 | [0.955747, 1.000606] | 0.030616 | 0.012912 |
| 2 | 2 | c64 | warm | 8 | 6.012 | 2.475 | 0.411774 | [0.398284, 0.415843] | 0.009980 | 0.016162 |
| 2 | 2 | c64 | cold | 32 | 69.391 | 68.148 | 0.982093 | [0.922381, 1.013601] | 0.024672 | 0.004710 |
| 2 | 2 | c64 | warm | 32 | 12.574 | 5.581 | 0.444949 | [0.428583, 0.451726] | 0.006362 | 0.008959 |
| 3 | 2 | c64 | aci | 1 | 9068.387 | 6858.502 | 0.754880 | [0.746851, 0.771771] | 0.003792 | 0.007111 |
| 3 | 2 | c64 | cold | 8 | 27.441 | 27.592 | 0.994334 | [0.984750, 1.006960] | 0.017856 | 0.019970 |
| 3 | 2 | c64 | warm | 8 | 6.553 | 2.635 | 0.398217 | [0.387149, 0.414424] | 0.013582 | 0.015180 |
| 3 | 2 | c64 | cold | 32 | 69.030 | 68.649 | 0.993883 | [0.990410, 1.031176] | 0.015674 | 0.005113 |
| 3 | 2 | c64 | warm | 32 | 14.407 | 5.831 | 0.404453 | [0.399734, 0.414451] | 0.008329 | 0.017150 |
| 4 | 2 | c64 | aci | 1 | 12828.832 | 9786.675 | 0.760086 | [0.749215, 0.775018] | 0.004697 | 0.008499 |
| 4 | 2 | c64 | cold | 8 | 32.612 | 32.300 | 0.983970 | [0.978810, 1.025197] | 0.008034 | 0.006161 |
| 4 | 2 | c64 | warm | 8 | 7.654 | 2.746 | 0.359039 | [0.353410, 0.362992] | 0.010452 | 0.014931 |
| 4 | 2 | c64 | cold | 32 | 75.512 | 75.672 | 0.993989 | [0.983804, 1.025421] | 0.008634 | 0.009673 |
| 4 | 2 | c64 | warm | 32 | 16.902 | 6.051 | 0.353670 | [0.351084, 0.359580] | 0.010650 | 0.016526 |
| 2 | 3 | c64 | aci | 1 | 6762.224 | 3219.519 | 0.475674 | [0.470821, 0.480487] | 0.004709 | 0.004985 |
| 2 | 3 | c64 | cold | 8 | 93.265 | 96.621 | 1.013757 | [0.892568, 1.302263] | 0.091846 | 0.024260 |
| 2 | 3 | c64 | warm | 8 | 7.925 | 3.056 | 0.379475 | [0.372572, 0.390108] | 0.012744 | 0.009817 |
| 2 | 3 | c64 | cold | 32 | 120.416 | 119.525 | 0.992929 | [0.975068, 1.027167] | 0.011062 | 0.016432 |
| 2 | 3 | c64 | warm | 32 | 13.866 | 5.761 | 0.420856 | [0.410971, 0.429703] | 0.014496 | 0.010588 |
| 3 | 3 | c64 | aci | 1 | 13775.158 | 11053.115 | 0.799226 | [0.791794, 0.820740] | 0.004886 | 0.011464 |
| 3 | 3 | c64 | cold | 8 | 67.978 | 68.008 | 0.999119 | [0.966971, 1.025207] | 0.012813 | 0.011940 |
| 3 | 3 | c64 | warm | 8 | 7.824 | 3.016 | 0.383199 | [0.379007, 0.392324] | 0.010097 | 0.009947 |
| 3 | 3 | c64 | cold | 32 | 87.775 | 89.177 | 1.013374 | [0.987978, 1.041497] | 0.004227 | 0.018312 |
| 3 | 3 | c64 | warm | 32 | 14.798 | 5.931 | 0.403427 | [0.392029, 0.409189] | 0.013583 | 0.020233 |
| 4 | 3 | c64 | aci | 1 | 19134.210 | 14834.571 | 0.769724 | [0.757648, 0.776975] | 0.007169 | 0.007109 |
| 4 | 3 | c64 | cold | 8 | 69.671 | 68.859 | 0.992025 | [0.970658, 1.027674] | 0.008339 | 0.008438 |
| 4 | 3 | c64 | warm | 8 | 8.817 | 3.106 | 0.353408 | [0.349472, 0.365856] | 0.012476 | 0.019317 |
| 4 | 3 | c64 | cold | 32 | 93.836 | 93.856 | 0.991930 | [0.972356, 1.004613] | 0.009399 | 0.013457 |
| 4 | 3 | c64 | warm | 32 | 17.233 | 6.021 | 0.348517 | [0.343532, 0.355849] | 0.009923 | 0.011626 |

## Phase and CPU observations

One feature-enabled current-main Pi replay uses distinct operand namespaces and records actual dimensions, cache states and batches. Message/Guard and frame time are exclusive; query time is inclusive and must not be added to them. These instrumented single-replay observations are descriptive and cannot certify uninstrumented wall time or necessity of a cost. Tensor element count is a size proxy, not a FLOP certificate.

| Topology/node | Degree | Bonds | Local elements | Guard hits/misses | Guard points/calls | Guard ns/point | Frame points/calls | Frame ns/point |
| --- | ---: | --- | ---: | --- | --- | ---: | --- | ---: |
| comb/input:0:0 | 2 | [44, 44] | 3872 | 24/5374 | 5398/5342 | 7400.008 | 1552/20 | 832.775 |
| comb/input:0:1 | 3 | [11, 44, 47] | 45496 | 16/5388 | 5404/5348 | 73894.410 | 105044/30 | 756.031 |
| nblock/input:0:7 | 2 | [134, 143] | 38324 | 3/12227 | 12230/12164 | 62837.593 | 5260/26 | 1448.477 |

The Comb hub has 11.75 times the core size of the reference degree-2 node, roughly 9.99 times message time per component assignment, and a very different frame-batch distribution. Both message paths are almost entirely misses under the product wrapper's 1 MiB aggregate message budget. W uses the upstream default 256 MiB aggregate message budget. This explains why a pure warm-query improvement cannot be extrapolated to Pi/Sigma stage speedups; oracle-point counts alone do not control tensor shape, cache state or batching. These observations do not prove that every remaining cost is optimal.

A software CPU profile of the prior 16-site/bond-256 hinted typed warm chain control (same TreeTN source) retains 1,621 samples inside the warm edge-cut window. Self-cost buckets include edge-cut contraction 18.08%, rooted assignment building 12.95%, memory-copy symbols 10.49%, SipHash write 8.08%, whole-message typed decode/collection 5.74%, and IndexKey hashing 4.87%. The later unrelated fixture-initialization tail is excluded by its recorded timestamps. Optimized unwinding loses some callers, so this is descriptive sampling, not complete causal accounting. The independently red allocation and actual-message regressions establish the specific avoidable work removed here.

## Real R=10 paired replay

Real SGW fixtures: R=10, T=0.1, mu=0.5, U=2. Historical G0/W/Pi tree checkpoints are self-describing and copied into new run directories; current-main transforms reconstruct Pi/Sigma operands. Loading/transforms and independent validation stay outside operation timing. The unary W map is `U*U*Pi/(1-(U*Pi)^2)`. The production wrappers retain original tolerance (absolute 1e-4), seed, Guard, sweep/rank and topology-specific memory settings. Pi/Sigma use a 1 MiB aggregate message cache; W uses the default 256 MiB.

One complete untimed warm-up per revision/topology/stage precedes three alternating revision pairs for all six cases. Every timed output is checked at the same 1,086 fixed points (seed 671, 1,024 uniform points plus zero/one and single-bit-flip witnesses), via generic contraction rather than the cached raw kernels. The unchanged accuracy gate is finite values and max absolute sample residual <=1e-3, the existing tenfold global margin. It does not certify a full physical grid or a stricter 1e-4 global error.

All 36 measured calls and 12 warm-ups have exactly matching samples, maximum sample error, last reported error, ranks, sweeps, point counts and termination relative to the baseline. Load/frequency/dispersion and execution gates have zero failures. This new experiment's predeclared termination gate requires exact preservation of baseline status, including NBlock W MaxSweeps, and its verdict is **DESCRIPTIVE**. The earlier Converged-only baseline matrix remains **INCONCLUSIVE** with three NBlock W termination failures; no cell is removed and no gate is retroactively relaxed. This is behavior preservation and cost measurement, not W convergence validation.

| Topology | Stage | Baseline s | Candidate s | Paired ratio | Sweeps | Points | Max rank | Termination | Max sample absolute error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| comb | pi_rtau | 5.145944 | 5.169584 | 1.004594 | 10 | 8738200 | 101 | Converged | 1.925137824841e-05 |
| nblock | pi_rtau | 16.169322 | 16.017642 | 0.990619 | 13 | 7850817 | 207 | Converged | 3.375872041919e-05 |
| comb | W | 0.491359 | 0.477285 | 0.967402 | 10 | 1444904 | 41 | Converged | 1.082974549759e-04 |
| nblock | W | 0.938080 | 0.911355 | 0.971084 | 20 | 2187304 | 76 | MaxSweeps | 1.157482548583e-04 |
| comb | sigma_rtau | 2.277785 | 2.296344 | 1.008148 | 7 | 2879586 | 104 | Converged | 4.613615089458e-05 |
| nblock | sigma_rtau | 6.303831 | 6.303748 | 0.993403 | 9 | 3109146 | 163 | Converged | 4.494522605971e-05 |

Pi/Sigma paired ratios range from 0.990619 to 1.008148; no substantial production speedup is established. W ratios are 0.967402 and 0.971084. Three pairs are limited descriptive evidence and no formal regression/speedup bound is asserted. The candidate Comb/NBlock per-oracle-point ratios are:

| Stage | Candidate Comb/NBlock time per evaluated point |
| --- | ---: |
| pi_rtau | 0.289968 |
| W | 0.792794 |
| sigma_rtau | 0.393323 |

The original fixed 5--6.5x branching signature is absent in these current-main replays. Different ranks, cache states and batches prevent assigning a universal topology factor from point counts. W's preserved MaxSweeps is a separate quality investigation, not a performance failure hidden by the table.

### Input checkpoint hashes

These are experiment inputs, not newly committed binary fixtures.

| Relative experiment path | SHA-256 |
| --- | --- |
| comb/state/prepared/g_rtau.field | `ef43d2ec8bb4461104252e1d598df93cd1f35b6979334daec3cb6bdabdeb7500` |
| comb/state/prepared/g_rtau.h5 | `a03034b374dfcde5f515039a642ac55fe7a1358bd0450572ac5fdc74bdc322ab` |
| comb/state/prepared/g_reflected.h5 | `c6424a1df885c4fbd8c7b891a0753d2b064cd0f94c83af85f8e795b89e2cbbfe` |
| comb/state/prepared/g_reflected.field | `ef43d2ec8bb4461104252e1d598df93cd1f35b6979334daec3cb6bdabdeb7500` |
| comb/state/prepared/w_rtau.h5 | `917cfabdff1d2d66f42b5a3c807d5e0a4a4b859082bc922d9e40b8f97c4d7838` |
| comb/state/prepared/w_rtau.field | `5a26befcffed5f85e310b2cc6abc702018f5db403f3348ee6b80a61ec050680a` |
| comb/state/stages/04_Pi.h5 | `65191e4fc9e9d8dcdcbf777c0cdd7f9e67e06e6558221fec2dbce81d4b60c892` |
| comb/state/stages/04_Pi.field | `be42d45bc749955a455a70c731075a30c7261cbe9fa39bc8772c714b21b44cf0` |
| nblock/state/prepared/g_rtau.field | `ef43d2ec8bb4461104252e1d598df93cd1f35b6979334daec3cb6bdabdeb7500` |
| nblock/state/prepared/g_rtau.h5 | `168893808b40e8af6bd7f5e7ed73c1e6cb21a721ae21695c72583f8a990dbdaf` |
| nblock/state/prepared/g_reflected.h5 | `a903bf68ee59df12a4cc05e700b6b443d3ed86ee49c69624a5b190f9c528c663` |
| nblock/state/prepared/g_reflected.field | `ef43d2ec8bb4461104252e1d598df93cd1f35b6979334daec3cb6bdabdeb7500` |
| nblock/state/prepared/w_rtau.h5 | `53940d75298b4e762aa4afedc76f467612e0bac4e42b07177338e9ffa9d7a2bf` |
| nblock/state/prepared/w_rtau.field | `5a26befcffed5f85e310b2cc6abc702018f5db403f3348ee6b80a61ec050680a` |
| nblock/state/stages/04_Pi.h5 | `96eef4fe3b7b412c2bcddec6d641b5f7f1601836d092ce9b02b3586c4ce244fc` |
| nblock/state/stages/04_Pi.field | `be42d45bc749955a455a70c731075a30c7261cbe9fa39bc8772c714b21b44cf0` |

## Chain controls

All 24 original TTCache/default vertex/hinted erased/hinted typed cold/warm controls run at both revisions, with the same 16-site f64 chain, bonds 64/128/256 and 64-point Cartesian batch. Every route independently checks TTCache value parity before its timing. Release, diagnostics off, CPU 2 and six thread variables one; 10 samples, one-second warm-up and one-second measurement (Criterion extends short measurement windows to complete the declared ten samples). All executions and value checks pass.

One complete baseline run precedes one complete candidate run. This is a descriptive sequential comparison; Criterion's displayed within-run significance labels are not used as a formal causal or non-regression verdict. Unchanged controls vary by a few percent. Every case and both estimate intervals remain below.

| Route | Bond | Baseline us [interval] | Candidate us [interval] | Estimate ratio |
| --- | ---: | --- | --- | ---: |
| ttcache_cold | 64 | 2717.300 [2709.300, 2722.000] | 2768.900 [2736.300, 2812.200] | 1.018989 |
| ttcache_warm | 64 | 9.095 [9.075, 9.120] | 9.229 [9.094, 9.493] | 1.014700 |
| treetn_cold | 64 | 1156.000 [1153.400, 1159.000] | 1126.800 [1113.600, 1145.300] | 0.974740 |
| treetn_warm | 64 | 362.740 [361.770, 363.550] | 354.590 [353.480, 355.450] | 0.977532 |
| treetn_around_split_cold | 64 | 1098.900 [1093.500, 1106.000] | 1051.500 [1049.100, 1053.800] | 0.956866 |
| treetn_around_split_warm | 64 | 154.190 [153.830, 154.580] | 77.595 [77.438, 77.793] | 0.503243 |
| treetn_typed_around_split_cold | 64 | 1035.900 [1032.100, 1040.800] | 989.460 [986.480, 991.700] | 0.955169 |
| treetn_typed_around_split_warm | 64 | 87.760 [87.231, 88.484] | 13.463 [13.433, 13.512] | 0.153407 |
| ttcache_cold | 128 | 10971.000 [10953.000, 10998.000] | 10944.000 [10898.000, 10982.000] | 0.997539 |
| ttcache_warm | 128 | 11.228 [11.203, 11.250] | 11.188 [11.171, 11.214] | 0.996437 |
| treetn_cold | 128 | 4396.800 [4371.800, 4421.200] | 4184.300 [4174.800, 4192.000] | 0.951669 |
| treetn_warm | 128 | 1037.100 [1034.100, 1040.800] | 997.200 [994.970, 998.880] | 0.961527 |
| treetn_around_split_cold | 128 | 3937.400 [3920.100, 3973.400] | 3684.400 [3680.200, 3689.300] | 0.935744 |
| treetn_around_split_warm | 128 | 181.210 [180.660, 181.760] | 81.847 [81.694, 82.039] | 0.451669 |
| treetn_typed_around_split_cold | 128 | 3825.400 [3813.000, 3833.600] | 3660.700 [3626.200, 3724.500] | 0.956946 |
| treetn_typed_around_split_warm | 128 | 121.440 [121.190, 121.690] | 17.180 [17.165, 17.206] | 0.141469 |
| ttcache_cold | 256 | 46342.000 [46294.000, 46396.000] | 44433.000 [44335.000, 44549.000] | 0.958806 |
| ttcache_warm | 256 | 16.666 [16.599, 16.721] | 16.865 [16.816, 16.915] | 1.011940 |
| treetn_cold | 256 | 22083.000 [21948.000, 22253.000] | 21962.000 [21837.000, 22083.000] | 0.994521 |
| treetn_warm | 256 | 3816.400 [3805.300, 3830.400] | 3633.200 [3626.400, 3638.500] | 0.951997 |
| treetn_around_split_cold | 256 | 20383.000 [20305.000, 20430.000] | 20862.000 [20548.000, 21290.000] | 1.023500 |
| treetn_around_split_warm | 256 | 283.960 [283.320, 284.610] | 93.275 [93.097, 93.484] | 0.328479 |
| treetn_typed_around_split_cold | 256 | 20217.000 [20092.000, 20410.000] | 20384.000 [20244.000, 20554.000] | 1.008260 |
| treetn_typed_around_split_warm | 256 | 213.740 [213.410, 214.070] | 28.210 [28.164, 28.248] | 0.131983 |

At bond 256, the hinted typed estimate changes from 213.740 to 28.210 us; candidate TTCache is 16.865 us, leaving a 1.673x gap. The default vertex route still contracts its center core and is retained as an explicitly different control. Erased output still pays its dynamic wrapper cost. Component encoding, hashing, validation and scalar dispatch explain implementation work that remains; this experiment does not establish a universal lower bound for it.

## Final ownership cleanup and verification

The final measured code is `66f7bda6b7a84dfa5c3f78701e41fba77b7556cd`: it additionally borrows the cold/partial route's caller-owned selected environment instead of cloning it, and adds right-column dtype/end-offset rejection coverage. The preceding primary paired results are explicitly for `462252a9`; they are not silently relabeled as final-source timing certificates.

All 96 feature-enabled cached-evaluator tests and 205 TreeACI unit tests pass after this cleanup, with deny-warning Clippy. The submission subsequently moves the existing checked work-count multiplication ahead of result allocation and assembly, so overflow is rejected before that work. This boundary-order change leaves successful arithmetic and work unchanged; the measurements retain their actual `66f7bda6` source stamp. All 96 feature-enabled cached-evaluator tests and deny-warning Clippy pass again after this move. All six final-source R=10 numerical replays again exactly preserve the original baseline samples, residuals, point/sweep counts, ranks and termination (including NBlock W MaxSweeps). Their operation times are recorded but not promoted to a new paired speedup claim.

The final-source nine-pair/nine-repeat full 120-case experiment is **INCONCLUSIVE**: degree 2/profile 3/f64/cold/batch 8 has candidate relative MAD 0.239826. A separately declared complete confirmation (12 pairs, 15 repetitions, unchanged oracle/load/frequency/MAD gates) remains **INCONCLUSIVE** in the same case, at candidate MAD 0.204599. Both reports are retained; no case or gate is removed, and no further rerun is used to fish for a passing result. All numerical/effort checks pass in both full matrices. No final-source universal wall-time or formal non-regression claim follows. The confirmed actual-message and independently observed allocation invariants establish the repair's scope.

| Final experiment | Verdict | Failing case | Baseline MAD | Candidate MAD | Ratio | 95% interval |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 671-final-paired-120 | INCONCLUSIVE | [2, 3, 2, 'f64', 'cold', 8] | 0.031036 | 0.239826 | 1.088584 | [1.016399, 1.593150] |
| 671-final-confirmation-120 | INCONCLUSIVE | [2, 3, 2, 'f64', 'cold', 8] | 0.148995 | 0.204599 | 1.029673 | [0.774338, 1.443304] |

### Complete final confirmation case summaries

Times are microseconds; all 120 cases of the final confirmation are retained, including the failed validity case. Ratios and intervals are descriptive observations inside an INCONCLUSIVE experiment.

| Degree | Profile | Scalar | Mode | Batch | Baseline us | Final us | Ratio | 95% interval | Baseline MAD | Final MAD |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2 | 0 | f64 | aci | 1 | 3549.948 | 1981.713 | 0.557377 | [0.554427, 0.559619] | 0.005715 | 0.006609 |
| 2 | 0 | f64 | cold | 8 | 17.948 | 17.543 | 0.975028 | [0.966539, 0.986129] | 0.008107 | 0.011429 |
| 2 | 0 | f64 | warm | 8 | 5.380 | 2.365 | 0.445630 | [0.431503, 0.448549] | 0.013011 | 0.012476 |
| 2 | 0 | f64 | cold | 32 | 29.516 | 29.255 | 0.997416 | [0.977548, 1.007643] | 0.024800 | 0.011485 |
| 2 | 0 | f64 | warm | 32 | 11.061 | 5.120 | 0.465033 | [0.454616, 0.469472] | 0.012205 | 0.008790 |
| 3 | 0 | f64 | aci | 1 | 5571.249 | 3214.575 | 0.573950 | [0.573194, 0.579645] | 0.005664 | 0.005953 |
| 3 | 0 | f64 | cold | 8 | 20.694 | 20.664 | 0.995655 | [0.975966, 1.005802] | 0.009181 | 0.014518 |
| 3 | 0 | f64 | warm | 8 | 6.191 | 2.474 | 0.393473 | [0.388436, 0.400388] | 0.008802 | 0.014144 |
| 3 | 0 | f64 | cold | 32 | 33.758 | 33.704 | 1.002050 | [0.988170, 1.014311] | 0.005495 | 0.008901 |
| 3 | 0 | f64 | warm | 32 | 13.069 | 5.375 | 0.415555 | [0.404878, 0.418347] | 0.007651 | 0.010326 |
| 4 | 0 | f64 | aci | 1 | 6376.131 | 4020.032 | 0.630116 | [0.621744, 0.633265] | 0.006400 | 0.003425 |
| 4 | 0 | f64 | cold | 8 | 24.101 | 23.930 | 0.995681 | [0.982303, 1.015825] | 0.009958 | 0.008567 |
| 4 | 0 | f64 | warm | 8 | 7.383 | 2.590 | 0.350862 | [0.341961, 0.359848] | 0.012122 | 0.011776 |
| 4 | 0 | f64 | cold | 32 | 40.721 | 40.682 | 0.997369 | [0.983441, 1.009347] | 0.013040 | 0.013298 |
| 4 | 0 | f64 | warm | 32 | 15.700 | 5.640 | 0.361171 | [0.357901, 0.363786] | 0.006083 | 0.007092 |
| 2 | 1 | f64 | aci | 1 | 4244.757 | 2219.681 | 0.523723 | [0.522269, 0.525513] | 0.003272 | 0.003787 |
| 2 | 1 | f64 | cold | 8 | 19.832 | 19.572 | 0.984360 | [0.971505, 1.003885] | 0.010337 | 0.007664 |
| 2 | 1 | f64 | warm | 8 | 5.495 | 2.384 | 0.435701 | [0.425520, 0.440337] | 0.010828 | 0.021183 |
| 2 | 1 | f64 | cold | 32 | 32.060 | 31.870 | 0.995266 | [0.984428, 1.006644] | 0.008125 | 0.009758 |
| 2 | 1 | f64 | warm | 32 | 11.206 | 5.159 | 0.459450 | [0.454423, 0.466758] | 0.003123 | 0.007947 |
| 3 | 1 | f64 | aci | 1 | 6918.854 | 4255.569 | 0.613884 | [0.609696, 0.618736] | 0.003252 | 0.005001 |
| 3 | 1 | f64 | cold | 8 | 21.851 | 21.691 | 0.980890 | [0.970566, 1.001844] | 0.009611 | 0.009013 |
| 3 | 1 | f64 | warm | 8 | 6.297 | 2.459 | 0.390286 | [0.386579, 0.396209] | 0.008020 | 0.016263 |
| 3 | 1 | f64 | cold | 32 | 35.928 | 36.008 | 0.996520 | [0.988809, 1.012134] | 0.005163 | 0.006957 |
| 3 | 1 | f64 | warm | 32 | 12.959 | 5.405 | 0.416349 | [0.409108, 0.418776] | 0.006135 | 0.009251 |
| 4 | 1 | f64 | aci | 1 | 8863.478 | 5824.538 | 0.656681 | [0.652708, 0.659352] | 0.004668 | 0.003440 |
| 4 | 1 | f64 | cold | 8 | 25.308 | 24.861 | 0.975274 | [0.951714, 0.988985] | 0.011874 | 0.015908 |
| 4 | 1 | f64 | warm | 8 | 7.293 | 2.585 | 0.351539 | [0.346572, 0.357929] | 0.017824 | 0.015474 |
| 4 | 1 | f64 | cold | 32 | 42.044 | 42.495 | 1.015665 | [1.009117, 1.020127] | 0.005126 | 0.012731 |
| 4 | 1 | f64 | warm | 32 | 15.614 | 5.580 | 0.359096 | [0.354370, 0.361517] | 0.006437 | 0.006183 |
| 2 | 2 | f64 | aci | 1 | 5125.671 | 2459.231 | 0.479085 | [0.473538, 0.480492] | 0.007538 | 0.004803 |
| 2 | 2 | f64 | cold | 8 | 31.564 | 31.159 | 0.993153 | [0.854362, 1.086500] | 0.029844 | 0.048541 |
| 2 | 2 | f64 | warm | 8 | 5.951 | 2.405 | 0.405754 | [0.397732, 0.409202] | 0.006722 | 0.010605 |
| 2 | 2 | f64 | cold | 32 | 60.564 | 62.422 | 1.036141 | [0.995783, 1.057211] | 0.009098 | 0.021675 |
| 2 | 2 | f64 | warm | 32 | 12.444 | 5.490 | 0.440711 | [0.436852, 0.449536] | 0.008077 | 0.010929 |
| 3 | 2 | f64 | aci | 1 | 9123.923 | 6540.133 | 0.717667 | [0.710637, 0.722447] | 0.003452 | 0.006218 |
| 3 | 2 | f64 | cold | 8 | 25.332 | 25.122 | 0.989375 | [0.974281, 0.998377] | 0.014625 | 0.014947 |
| 3 | 2 | f64 | warm | 8 | 6.362 | 2.519 | 0.396187 | [0.387698, 0.400961] | 0.015718 | 0.007938 |
| 3 | 2 | f64 | cold | 32 | 61.025 | 61.400 | 1.009330 | [0.989456, 1.016645] | 0.010348 | 0.009959 |
| 3 | 2 | f64 | warm | 32 | 14.086 | 5.751 | 0.407037 | [0.401201, 0.414644] | 0.004969 | 0.011302 |
| 4 | 2 | f64 | aci | 1 | 12926.957 | 9055.658 | 0.698167 | [0.692086, 0.703938] | 0.003185 | 0.008447 |
| 4 | 2 | f64 | cold | 8 | 29.526 | 29.154 | 0.991158 | [0.979219, 1.003785] | 0.007299 | 0.005677 |
| 4 | 2 | f64 | warm | 8 | 7.399 | 2.585 | 0.345259 | [0.340388, 0.352100] | 0.015543 | 0.009865 |
| 4 | 2 | f64 | cold | 32 | 68.729 | 68.855 | 1.002055 | [0.985915, 1.009496] | 0.009043 | 0.008743 |
| 4 | 2 | f64 | warm | 32 | 16.626 | 5.871 | 0.351323 | [0.349954, 0.357029] | 0.008150 | 0.008516 |
| 2 | 3 | f64 | aci | 1 | 6823.626 | 3020.323 | 0.442059 | [0.436976, 0.443386] | 0.006482 | 0.009135 |
| 2 | 3 | f64 | cold | 8 | 69.325 | 74.775 | 1.029673 | [0.774338, 1.443304] | 0.148995 | 0.204599 |
| 2 | 3 | f64 | warm | 8 | 7.359 | 2.816 | 0.383534 | [0.372088, 0.392635] | 0.016308 | 0.021488 |
| 2 | 3 | f64 | cold | 32 | 95.650 | 94.281 | 1.006717 | [0.933622, 1.073771] | 0.017909 | 0.080069 |
| 2 | 3 | f64 | warm | 32 | 13.605 | 5.756 | 0.423432 | [0.414537, 0.425770] | 0.010694 | 0.008687 |
| 3 | 3 | f64 | aci | 1 | 12686.537 | 9690.182 | 0.766616 | [0.753453, 0.770402] | 0.007991 | 0.004907 |
| 3 | 3 | f64 | cold | 8 | 59.441 | 59.722 | 1.000721 | [0.977563, 1.030618] | 0.014409 | 0.016694 |
| 3 | 3 | f64 | warm | 8 | 7.439 | 2.916 | 0.388673 | [0.385979, 0.395565] | 0.008738 | 0.013891 |
| 3 | 3 | f64 | cold | 32 | 75.291 | 74.731 | 1.000043 | [0.988165, 1.015708] | 0.017964 | 0.008986 |
| 3 | 3 | f64 | warm | 32 | 14.387 | 5.751 | 0.400870 | [0.398754, 0.407617] | 0.006985 | 0.011302 |
| 4 | 3 | f64 | aci | 1 | 17296.095 | 12987.838 | 0.750303 | [0.744258, 0.752674] | 0.006441 | 0.008008 |
| 4 | 3 | f64 | cold | 8 | 57.248 | 58.455 | 1.020212 | [0.999479, 1.032620] | 0.012088 | 0.014652 |
| 4 | 3 | f64 | warm | 8 | 8.506 | 2.991 | 0.351951 | [0.348756, 0.354735] | 0.008229 | 0.008358 |
| 4 | 3 | f64 | cold | 32 | 79.109 | 78.252 | 1.006100 | [0.989215, 1.024759] | 0.011528 | 0.015552 |
| 4 | 3 | f64 | warm | 32 | 16.957 | 5.941 | 0.351114 | [0.347970, 0.353197] | 0.006811 | 0.004124 |
| 2 | 0 | c64 | aci | 1 | 3969.140 | 2154.534 | 0.545082 | [0.540844, 0.549642] | 0.008387 | 0.004836 |
| 2 | 0 | c64 | cold | 8 | 19.096 | 18.770 | 0.984779 | [0.981784, 0.990791] | 0.005787 | 0.008817 |
| 2 | 0 | c64 | warm | 8 | 5.505 | 2.475 | 0.449772 | [0.435589, 0.460420] | 0.006358 | 0.020202 |
| 2 | 0 | c64 | cold | 32 | 30.653 | 30.427 | 0.994773 | [0.986315, 1.004402] | 0.011598 | 0.008874 |
| 2 | 0 | c64 | warm | 32 | 11.181 | 5.285 | 0.473049 | [0.466203, 0.476228] | 0.007602 | 0.014098 |
| 3 | 0 | c64 | aci | 1 | 6002.540 | 3487.673 | 0.579577 | [0.575055, 0.587344] | 0.004519 | 0.009070 |
| 3 | 0 | c64 | cold | 8 | 21.530 | 21.706 | 1.007480 | [0.998133, 1.021143] | 0.005829 | 0.007809 |
| 3 | 0 | c64 | warm | 8 | 6.337 | 2.585 | 0.403331 | [0.399422, 0.410331] | 0.006312 | 0.011605 |
| 3 | 0 | c64 | cold | 32 | 34.906 | 34.565 | 0.992987 | [0.976489, 1.005640] | 0.013336 | 0.009287 |
| 3 | 0 | c64 | warm | 32 | 13.170 | 5.470 | 0.413712 | [0.410593, 0.420486] | 0.005315 | 0.011792 |
| 4 | 0 | c64 | aci | 1 | 8087.098 | 4612.992 | 0.571003 | [0.566926, 0.574763] | 0.004070 | 0.007107 |
| 4 | 0 | c64 | cold | 8 | 25.563 | 25.012 | 0.986666 | [0.962442, 1.007766] | 0.016058 | 0.012014 |
| 4 | 0 | c64 | warm | 8 | 7.624 | 2.715 | 0.355394 | [0.352091, 0.362659] | 0.006624 | 0.007366 |
| 4 | 0 | c64 | cold | 32 | 42.445 | 42.510 | 1.001090 | [0.984840, 1.019806] | 0.010508 | 0.015326 |
| 4 | 0 | c64 | warm | 32 | 15.850 | 5.761 | 0.359548 | [0.358075, 0.361385] | 0.010757 | 0.010503 |
| 2 | 1 | c64 | aci | 1 | 4566.901 | 2379.781 | 0.519392 | [0.515098, 0.524949] | 0.002792 | 0.005012 |
| 2 | 1 | c64 | cold | 8 | 20.859 | 20.764 | 0.993691 | [0.981497, 1.009805] | 0.010811 | 0.006767 |
| 2 | 1 | c64 | warm | 8 | 5.630 | 2.495 | 0.441575 | [0.439904, 0.444701] | 0.008969 | 0.010020 |
| 2 | 1 | c64 | cold | 32 | 33.648 | 33.608 | 0.991828 | [0.975213, 1.013841] | 0.013537 | 0.015800 |
| 2 | 1 | c64 | warm | 32 | 11.377 | 5.285 | 0.465712 | [0.459042, 0.471428] | 0.007911 | 0.005676 |
| 3 | 1 | c64 | aci | 1 | 7021.730 | 4593.453 | 0.649850 | [0.648206, 0.656739] | 0.005441 | 0.003099 |
| 3 | 1 | c64 | cold | 8 | 22.703 | 22.488 | 0.992696 | [0.975718, 0.997098] | 0.011915 | 0.005092 |
| 3 | 1 | c64 | warm | 8 | 6.427 | 2.575 | 0.396897 | [0.388423, 0.407267] | 0.012447 | 0.021558 |
| 3 | 1 | c64 | cold | 32 | 37.020 | 36.924 | 0.995646 | [0.991484, 1.017261] | 0.007712 | 0.005430 |
| 3 | 1 | c64 | warm | 32 | 13.275 | 5.515 | 0.414751 | [0.412806, 0.421320] | 0.010169 | 0.006346 |
| 4 | 1 | c64 | aci | 1 | 10292.612 | 6416.069 | 0.621096 | [0.617732, 0.627524] | 0.001916 | 0.005265 |
| 4 | 1 | c64 | cold | 8 | 26.790 | 26.269 | 0.983371 | [0.972865, 0.997158] | 0.007503 | 0.011059 |
| 4 | 1 | c64 | warm | 8 | 7.514 | 2.645 | 0.353949 | [0.345073, 0.360431] | 0.010048 | 0.013233 |
| 4 | 1 | c64 | cold | 32 | 43.928 | 44.008 | 0.994127 | [0.985201, 1.021716] | 0.009686 | 0.010135 |
| 4 | 1 | c64 | warm | 32 | 15.730 | 5.721 | 0.362010 | [0.360080, 0.366542] | 0.008328 | 0.008740 |
| 2 | 2 | c64 | aci | 1 | 5323.594 | 2609.759 | 0.491541 | [0.483479, 0.497726] | 0.008659 | 0.008756 |
| 2 | 2 | c64 | cold | 8 | 33.348 | 32.822 | 0.987078 | [0.976953, 0.999368] | 0.007947 | 0.007312 |
| 2 | 2 | c64 | warm | 8 | 6.006 | 2.434 | 0.406548 | [0.401325, 0.416835] | 0.018230 | 0.008012 |
| 2 | 2 | c64 | cold | 32 | 67.782 | 67.797 | 0.990348 | [0.980184, 1.013679] | 0.009981 | 0.015281 |
| 2 | 2 | c64 | warm | 32 | 12.414 | 5.620 | 0.448390 | [0.441690, 0.456841] | 0.010916 | 0.008006 |
| 3 | 2 | c64 | aci | 1 | 9095.968 | 6834.143 | 0.752063 | [0.746443, 0.753574] | 0.003714 | 0.007177 |
| 3 | 2 | c64 | cold | 8 | 27.035 | 26.555 | 0.993115 | [0.971974, 1.004628] | 0.005197 | 0.013387 |
| 3 | 2 | c64 | warm | 8 | 6.437 | 2.595 | 0.401268 | [0.397066, 0.409570] | 0.007768 | 0.009443 |
| 3 | 2 | c64 | cold | 32 | 68.083 | 69.025 | 1.016078 | [0.975222, 1.022210] | 0.008541 | 0.008490 |
| 3 | 2 | c64 | warm | 32 | 14.212 | 5.831 | 0.410220 | [0.405472, 0.411903] | 0.007740 | 0.009432 |
| 4 | 2 | c64 | aci | 1 | 12943.371 | 9790.928 | 0.759664 | [0.755882, 0.768909] | 0.004962 | 0.009307 |
| 4 | 2 | c64 | cold | 8 | 32.035 | 31.709 | 0.987913 | [0.981063, 1.018976] | 0.007976 | 0.012788 |
| 4 | 2 | c64 | warm | 8 | 7.524 | 2.625 | 0.349180 | [0.343593, 0.352755] | 0.011962 | 0.017143 |
| 4 | 2 | c64 | cold | 32 | 74.615 | 74.930 | 1.006282 | [0.994508, 1.014410] | 0.006178 | 0.004604 |
| 4 | 2 | c64 | warm | 32 | 16.922 | 6.031 | 0.357677 | [0.354032, 0.362722] | 0.006826 | 0.008208 |
| 2 | 3 | c64 | aci | 1 | 6760.315 | 3188.025 | 0.470191 | [0.466905, 0.475633] | 0.004026 | 0.003984 |
| 2 | 3 | c64 | cold | 8 | 89.778 | 92.003 | 0.988757 | [0.957777, 1.078697] | 0.077892 | 0.085155 |
| 2 | 3 | c64 | warm | 8 | 7.934 | 2.996 | 0.376957 | [0.373558, 0.380629] | 0.013233 | 0.013351 |
| 2 | 3 | c64 | cold | 32 | 120.256 | 121.043 | 1.011260 | [0.968195, 1.022630] | 0.014910 | 0.017878 |
| 2 | 3 | c64 | warm | 32 | 13.746 | 5.731 | 0.418718 | [0.412416, 0.425649] | 0.010549 | 0.008899 |
| 3 | 3 | c64 | aci | 1 | 13786.522 | 11009.247 | 0.799717 | [0.793079, 0.809998] | 0.001949 | 0.006881 |
| 3 | 3 | c64 | cold | 8 | 66.951 | 65.900 | 1.000520 | [0.969740, 1.013681] | 0.016161 | 0.009507 |
| 3 | 3 | c64 | warm | 8 | 7.654 | 2.995 | 0.390271 | [0.383505, 0.402196] | 0.004573 | 0.015023 |
| 3 | 3 | c64 | cold | 32 | 86.894 | 86.773 | 0.997636 | [0.985117, 1.015441] | 0.008533 | 0.008488 |
| 3 | 3 | c64 | warm | 32 | 14.773 | 5.876 | 0.402928 | [0.397915, 0.408946] | 0.009815 | 0.006807 |
| 4 | 3 | c64 | aci | 1 | 19223.667 | 14861.392 | 0.772013 | [0.766773, 0.781761] | 0.004150 | 0.005750 |
| 4 | 3 | c64 | cold | 8 | 69.305 | 68.684 | 0.994297 | [0.983798, 1.000808] | 0.006789 | 0.008532 |
| 4 | 3 | c64 | warm | 8 | 8.741 | 3.066 | 0.350417 | [0.343540, 0.363499] | 0.008065 | 0.019573 |
| 4 | 3 | c64 | cold | 32 | 93.280 | 92.329 | 0.989867 | [0.982471, 1.004853] | 0.008914 | 0.009006 |
| 4 | 3 | c64 | warm | 32 | 17.203 | 5.971 | 0.348022 | [0.343835, 0.351096] | 0.008167 | 0.011723 |

### Final-source chain controls

Both final-source and baseline 24-case chain controls pass all independent value checks and executions. The same sequential ten-sample/one-second protocol and limitations as the primary chain controls apply.

| Route | Bond | Baseline us [interval] | Final us [interval] | Estimate ratio |
| --- | ---: | --- | --- | ---: |
| ttcache_cold | 64 | 2728.800 [2709.700, 2765.000] | 2688.000 [2680.200, 2701.100] | 0.985048 |
| ttcache_warm | 64 | 9.038 [9.015, 9.064] | 9.129 [9.101, 9.162] | 1.010002 |
| treetn_cold | 64 | 1163.000 [1157.700, 1167.700] | 1122.200 [1115.800, 1127.900] | 0.964918 |
| treetn_warm | 64 | 362.130 [360.950, 363.480] | 358.040 [356.520, 359.290] | 0.988706 |
| treetn_around_split_cold | 64 | 1112.300 [1105.900, 1118.800] | 1057.600 [1054.900, 1061.600] | 0.950823 |
| treetn_around_split_warm | 64 | 152.410 [151.980, 152.680] | 76.378 [76.072, 76.644] | 0.501135 |
| treetn_typed_around_split_cold | 64 | 1047.700 [1045.500, 1050.300] | 996.780 [993.800, 1000.500] | 0.951398 |
| treetn_typed_around_split_warm | 64 | 88.532 [88.354, 88.641] | 13.477 [13.444, 13.518] | 0.152227 |
| ttcache_cold | 128 | 11208.000 [11131.000, 11338.000] | 11040.000 [11009.000, 11072.000] | 0.985011 |
| ttcache_warm | 128 | 11.209 [11.193, 11.235] | 11.285 [11.253, 11.333] | 1.006780 |
| treetn_cold | 128 | 4457.200 [4449.200, 4470.900] | 4266.100 [4230.200, 4313.000] | 0.957126 |
| treetn_warm | 128 | 1037.700 [1033.900, 1040.700] | 1007.800 [1005.000, 1010.900] | 0.971186 |
| treetn_around_split_cold | 128 | 3955.000 [3948.100, 3960.700] | 3762.000 [3748.800, 3773.300] | 0.951201 |
| treetn_around_split_warm | 128 | 179.240 [178.900, 179.570] | 80.414 [80.268, 80.623] | 0.448639 |
| treetn_typed_around_split_cold | 128 | 3914.800 [3897.300, 3933.400] | 3694.900 [3686.900, 3700.400] | 0.943829 |
| treetn_typed_around_split_warm | 128 | 117.330 [117.010, 117.770] | 17.466 [17.414, 17.560] | 0.148862 |
| ttcache_cold | 256 | 45152.000 [45065.000, 45229.000] | 45192.000 [45086.000, 45317.000] | 1.000886 |
| ttcache_warm | 256 | 17.036 [16.958, 17.095] | 17.042 [17.005, 17.074] | 1.000352 |
| treetn_cold | 256 | 23484.000 [23388.000, 23574.000] | 22918.000 [22809.000, 23081.000] | 0.975898 |
| treetn_warm | 256 | 3735.800 [3729.900, 3744.300] | 3736.800 [3728.900, 3741.800] | 1.000268 |
| treetn_around_split_cold | 256 | 22419.000 [22279.000, 22611.000] | 21955.000 [21529.000, 22667.000] | 0.979303 |
| treetn_around_split_warm | 256 | 284.300 [283.820, 284.740] | 91.442 [91.285, 91.548] | 0.321639 |
| treetn_typed_around_split_cold | 256 | 22138.000 [22043.000, 22233.000] | 21412.000 [21233.000, 21566.000] | 0.967206 |
| treetn_typed_around_split_warm | 256 | 215.510 [215.000, 215.960] | 28.393 [28.301, 28.589] | 0.131748 |

Final executable SHA-256:
- 671-final-branch: `b33202a0b7614eba1aa90422b9f8577391e9edaafd1fa8803d1562e14160fd3f`.
- 671-final-cache: `e8bba40ba558ef9fa625a4e823c5a4701f1162dc6908375fc69d4a196c228aa6`.
- 671-final-replay: `a7736c78654cd0ba71c49cff11cadf105cd0d9d22149b75a2472bd8a3c46f435`.
- 671-w-curve-final: `2b06c5505ad54d1df1ae03d32eb8ac130210ebdc1ada90a2aed9133456ba1625`.

## W prefix quality investigation

The W prefix probe varies only max_sweeps from the original minimum of 2 up to 20, stopping when Converged. All other settings, input checkpoints, operator and independent 1,086-point samples remain fixed. Exact max-error/rank history-prefix checks pass. This is a quality probe, not a timed performance experiment.

After the last NBlock Guard injection (pass 3), every local error is below the absolute 1e-4 cutoff and Guard reports no new pivots; from pass 5 through 20 at least one edge grows every pass. Growth resets the schedule's rank-stability counter to one, below the required two; a following growth-free pass would suffice, but none occurs in this interval. The late maximum rank fluctuates between 72 and 76 while sample residuals plateau around 1.1e-4--1.4e-4. No complete returned rank vector repeats after pass 4, so identity with #784's exact DMRG cycle is not established. The additional witness belongs in #784's investigation; no stopping policy is changed.

[Complete returned edge-rank data](2026-10-09-treeaci-w-prefix.csv). Growth counts compare returned output vectors; immediately after Guard injection they also reflect cleanup, so late growth interpretation uses the injection-free passes.

| Topology | Cap/sweeps | Last local error | Last reported rank | Last Guard pivots | Returned edges grew | Max sample absolute error | Termination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| comb | 2 | 9.856656420320e-05 | 26 | 5 | - | 1.006809839193e-02 | MaxSweeps |
| comb | 3 | 9.571021916194e-05 | 39 | 0 | 0 | 1.006809839193e-02 | MaxSweeps |
| comb | 4 | 9.729251557425e-05 | 39 | 2 | 15 | 2.831648848205e-04 | MaxSweeps |
| comb | 5 | 9.899434421943e-05 | 40 | 0 | 0 | 2.831648848205e-04 | MaxSweeps |
| comb | 6 | 9.899434421943e-05 | 40 | 0 | 4 | 9.807076113976e-05 | MaxSweeps |
| comb | 7 | 9.891619934133e-05 | 41 | 0 | 2 | 1.133932676231e-04 | MaxSweeps |
| comb | 8 | 9.770738201851e-05 | 41 | 0 | 1 | 1.168584514667e-04 | MaxSweeps |
| comb | 9 | 9.894471716589e-05 | 39 | 0 | 5 | 1.046102625418e-04 | MaxSweeps |
| comb | 10 | 9.868108593846e-05 | 39 | 0 | 0 | 1.082974549759e-04 | Converged |
| nblock | 2 | 9.961137249632e-05 | 16 | 5 | - | 1.449757379253e-03 | MaxSweeps |
| nblock | 3 | 9.697831633053e-05 | 34 | 4 | 16 | 9.299162913352e-05 | MaxSweeps |
| nblock | 4 | 9.923134337238e-05 | 65 | 0 | 0 | 9.299162913352e-05 | MaxSweeps |
| nblock | 5 | 9.987168808647e-05 | 76 | 0 | 11 | 1.245762223976e-04 | MaxSweeps |
| nblock | 6 | 9.939638031481e-05 | 76 | 0 | 5 | 1.082045808346e-04 | MaxSweeps |
| nblock | 7 | 9.909947473963e-05 | 76 | 0 | 7 | 1.187530331379e-04 | MaxSweeps |
| nblock | 8 | 9.909947473985e-05 | 76 | 0 | 3 | 1.125025307937e-04 | MaxSweeps |
| nblock | 9 | 9.932016215786e-05 | 75 | 0 | 6 | 1.374330516633e-04 | MaxSweeps |
| nblock | 10 | 9.815517212999e-05 | 72 | 0 | 3 | 1.207079186802e-04 | MaxSweeps |
| nblock | 11 | 9.967596139892e-05 | 76 | 0 | 4 | 1.260116790075e-04 | MaxSweeps |
| nblock | 12 | 9.967596139933e-05 | 76 | 0 | 8 | 1.077801402746e-04 | MaxSweeps |
| nblock | 13 | 9.993083955952e-05 | 73 | 0 | 5 | 1.446869149615e-04 | MaxSweeps |
| nblock | 14 | 9.996519522050e-05 | 73 | 0 | 4 | 1.167596009044e-04 | MaxSweeps |
| nblock | 15 | 9.996519522188e-05 | 73 | 0 | 4 | 1.196598343398e-04 | MaxSweeps |
| nblock | 16 | 9.708048629906e-05 | 75 | 0 | 6 | 1.164274886893e-04 | MaxSweeps |
| nblock | 17 | 9.930463406382e-05 | 74 | 0 | 6 | 1.214158027340e-04 | MaxSweeps |
| nblock | 18 | 9.913307523669e-05 | 73 | 0 | 8 | 1.201954084054e-04 | MaxSweeps |
| nblock | 19 | 9.766672039604e-05 | 76 | 0 | 3 | 1.210239459207e-04 | MaxSweeps |
| nblock | 20 | 9.989455250730e-05 | 73 | 0 | 1 | 1.157482548583e-04 | MaxSweeps |

## Limits

The source repair eliminates independently verified redundant work and transient bond-sized copies. Component keys, lookup, validation and dtype decoding still have necessary general-evaluator costs. It neither changes rank/update/Guard policy nor repairs NBlock W MaxSweeps, #784 or the unreproduced #794 report. The original Converged-only real R=10 matrix remains INCONCLUSIVE; a separate before/after experiment preserves baseline termination exactly. No RSI algorithm is benchmarked.
