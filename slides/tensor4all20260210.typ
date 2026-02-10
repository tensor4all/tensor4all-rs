// tensor4all-rs / Tensor4all.jl Presentation Slides
// Author: Hiroshi Shinaoka

#set page(
  paper: "presentation-16-9",
  margin: (x: 2cm, y: 1.5cm),
  numbering: "1",
  footer: context [
    #h(1fr)
    #text(size: 14pt, fill: gray)[#counter(page).display()]
  ],
)

#set text(
  font: "Hiragino Sans",
  size: 18pt,
)

#let title-slide(title, subtitle: none, author: none, date: none, body: none, acknowledgements: none, url: none) = {
  set align(center + horizon)
  block[
    #text(size: 48pt, weight: "bold")[#title]
    #if subtitle != none {
      v(0.5em)
      text(size: 28pt, fill: gray.darken(20%))[#subtitle]
    }
    #if author != none {
      v(1.5em)
      text(size: 24pt)[#author]
    }
    #if date != none {
      v(0.5em)
      text(size: 20pt, fill: gray)[#date]
    }
    #if body != none {
      v(1.0em)
      body
    }
    #if acknowledgements != none {
      v(0.8em)
      text(size: 16pt, fill: gray)[#acknowledgements]
    }
    #if url != none {
      v(1em)
      text(size: 16pt, fill: rgb("#2563eb"))[#link(url)[#url]]
    }
  ]
}

#let slide(title, body) = {
  pagebreak()
  block(width: 100%, height: 100%)[
    #text(size: 28pt, weight: "bold", fill: rgb("#2563eb"))[#title]
    #v(0.8em)
    #line(length: 100%, stroke: 0.5pt + rgb("#2563eb"))
    #v(1em)
    #body
  ]
}

#let code-block(code) = {
  block(
    fill: rgb("#f1f5f9"),
    inset: 16pt,
    radius: 6pt,
    width: 100%,
  )[
    #text(font: "Menlo", size: 14pt)[#code]
  ]
}

// =====================================================================
// Slide 0: Title
// =====================================================================
#title-slide(
  "Tensor4all: a pure-Rust tensor stack",
  author: "Hiroshi Shinaoka",
  date: "2026-02-10",
  acknowledgements: "Acknowledgements: Jing-Guo Liu, Satoshi Terasaki, Ken Inayoshi",
)

// =====================================================================
// Slide 1: Goal
// =====================================================================
#slide("Goal")[
  - Build a *pure-Rust tensor stack* for physics & tensor network computation
  - Provide the minimal core needed for research:
    - Complex128 (first-class, day one)
    - Optimized einsum (contraction tree + GEMM)
    - TN algorithms (MPS/TT, TreeTN, TCI)
    - AD support (reverse / forward / HVP)
  - Distribute as *installable packages* for Julia / Python
    - C-API + runtime loading (no Rust toolchain required by end users — _in the future_)
]

// =====================================================================
// Slide 2: Architecture — 4 layers
// =====================================================================
#slide("Architecture: 4 layers")[
  #text(size: 14pt)[
    #table(
      columns: (1.2fr, 3.8fr),
      stroke: 0.5pt,
      inset: 6pt,
      [*Layer*], [*What it provides (and repos)*],
      [4. Bindings], [*Compatibility & distribution*\
        Tensor4all.jl · Python bindings (planned)],
      [3. Tensor networks], [*TT/MPS, TreeTN, TCI, C-API, HDF5 serialization*\
        tensor4all-rs],
      [2. Tensor layer], [*Dense / BlockSparse / Diag + AD integration*\
        ndtensors-rs (proof of concept)],
      [1. Backend], [*Arrays + linalg + einsum*\
        mdarray · mdarray-linalg · strided-rs],
    )
  ]

  #v(0.4em)
  #text(size: 14pt)[
    Interop (for Julia/Python): hdf5-rt, rsmpi-rt, cblas-inject, lapack-inject
  ]
]

// =====================================================================
// Slide 3: Recent progress — 3 items
// =====================================================================
#slide("Recent progress")[
  1. *Tensor4all.jl: installable Julia package*
    - `Pkg.add` + `Pkg.build` auto-compiles the Rust backend via C-FFI
    - Runtime library sharing (HDF5, MPI) solved with hdf5-rt / rsmpi-rt

  #v(0.3em)

  2. *strided-rs: Rust einsum on par with OMEinsum.jl*
    - Contraction-tree optimizer + cache-optimized kernels
    - Benchmarked on 8 standardized einsum instances (same contraction path)

  #v(0.3em)

  3. *ndtensors-rs: proof of concept*
    - Tensor types (Dense / BlockSparse / Diag) + AD prototypes in Rust
    - C-API callable from Julia / Python
]

// =====================================================================
// Slide 4a: Tensor4all.jl — build mechanism
// =====================================================================
#slide("1.1 Tensor4all.jl: build mechanism")[
  - *RustToolChain.jl* (by Satoshi Terasaki) provides `cargo` to Julia
    - No system Rust installation needed — Julia artifact handles it

  #v(0.3em)

  - `Pkg.build("Tensor4all")` workflow:
    + Find Rust source (env var / sibling dir / GitHub clone)
    + `cargo build -p tensor4all-capi --release`
    + Copy the shared library (`.so` / `.dylib`) into `deps/`
    + Julia loads the C-FFI library via `Libdl.dlopen`

  #v(0.3em)

  - *Future:* distribute pre-built binary artifacts (no build step)
  - *Now:* local cargo build — very convenient for development & debugging
    - Edit Rust code → `Pkg.build` → test immediately in Julia
]

// =====================================================================
// Slide 4b: Sharing runtime libraries (HDF5 / MPI)
// =====================================================================
#slide("1.2 Sharing runtime libraries")[
  *Key constraint:* Julia/Python and Rust must share the same C libraries at runtime

  #v(0.3em)

  - *hdf5-rt* — runtime HDF5 loading via `dlopen`
    - Reuses the HDF5 library already loaded by HDF5.jl / h5py

  #v(0.3em)

  - *rsmpi-rt* — MPI via MPItrampoline / MPIABI + `dlopen`
    - Shares `MPI_COMM_WORLD` with MPI.jl / mpi4py

  #v(0.3em)

  - *cblas-inject* / *lapack-inject* — BLAS/LAPACK via C-pointer injection
    - Inject function pointers (e.g. `dgemm`, `zgesvd`) from host language at runtime
    - Rust side calls through function pointers — no link-time BLAS dependency
]

// =====================================================================
// Slide 4c: Tensor4all.jl — usage example
// =====================================================================
#slide("1.3 Tensor4all.jl: usage example")[
  #code-block[
    ```julia
    using Pkg
    Pkg.add(url="https://github.com/tensor4all/Tensor4all.jl")
    Pkg.build("Tensor4all")
    ```
  ]

  #v(0.3em)

  #code-block[
    ```julia
    using Tensor4all

    i = Index(2)
    j = Index(3; tags="Site,n=1")
    A = Tensor([i, j], rand(2, 3))
    B = Tensor([j, i], rand(3, 2))

    C = contract(A, B)           # einsum contraction
    U, S, V = svd(A, [i])        # SVD with index selection
    arr = Array(C, [i, i'])      # export to Julia array
    ```
  ]

  #v(0.3em)
  #text(size: 14pt)[
    TODO: more examples — MPS construction, TCI, TreeTN
  ]
]

// =====================================================================
// Slide 5: strided-rs
// =====================================================================
#slide("2. strided-rs: Rust einsum on par with OMEinsum.jl")[
  - GitHub: #link("https://github.com/tensor4all/strided-rs")[github.com/tensor4all/strided-rs]
  #v(0.3em)

  #text(size: 14pt)[
    #table(
      columns: (1.6fr, 3.4fr),
      stroke: 0.5pt,
      inset: 6pt,
      [*Crate*], [*Role*],
      [`strided-view`], [Dynamic-rank strided views (zero-copy permute/reshape)],
      [`strided-kernel`], [Cache-optimized map / reduce / broadcast (L1-tiled)],
      [`strided-einsum2`], [Binary einsum (GEMM-based building block)],
      [`strided-opteinsum`], [N-ary einsum + contraction-tree optimizer],
    )
  ]

  #v(0.3em)
  - Two GEMM backends: *faer* (pure Rust) or *OpenBLAS*
  - Allows arbitrary strides (more general than mdarray): can be integrated into the mdtensor ecosystem in the future?
]

// =====================================================================
// Slide 6: Benchmark suite
// =====================================================================
#slide("Benchmark suite")[
  - GitHub: #link("https://github.com/tensor4all/strided-rs-benchmark-suite")[github.com/tensor4all/strided-rs-benchmark-suite]
  - Based on #link("https://benchmark.einsum.org/")[einsum benchmark] (168 standardized problems / 7 categories)
  - JSON metadata only — tensors generated at runtime (zero-filled)

  #v(0.3em)

  - Runners:
    - Rust: `strided-opteinsum` (faer + OpenBLAS backends)
    - Julia: OMEinsum.jl (same pre-computed contraction path — fair kernel-level comparison)

  #v(0.3em)

  - Instance categories: graphical models, language models, matrix chains, MPS, MERA
  - Thread control: `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, `JULIA_NUM_THREADS`
]

// =====================================================================
// Slide 7: Results (1 thread)
// =====================================================================
#slide("Results: 1 thread (opt_flops)")[
  #text(size: 14pt)[
    Apple Silicon M2 · median of 5 runs (2 warmup) · 2026-02-10
  ]

  #v(0.4em)

  #text(size: 12pt)[
    #table(
      columns: (2.9fr, 1.2fr, 1.4fr, 1.6fr),
      stroke: 0.5pt,
      inset: 6pt,
      [*Instance*], [*faer (ms)*], [*OpenBLAS (ms)*], [*OMEinsum (ms)*],
      [gm_queen5_5_3.wcsp], [3668], [4370], [-],
      [lm_batch_likelihood_brackets_4_4d], [*16.6*], [19.2], [20.3],
      [lm_batch_likelihood_sentence_3_12d], [*44.1*], [48.3], [53.2],
      [lm_batch_likelihood_sentence_4_4d], [*18.5*], [20.2], [19.9],
      [str_matrix_chain_multiplication_100], [11.3], [*10.4*], [15.6],
      [str_mps_varying_inner_product_200], [16.8], [18.1], [*16.5*],
      [str_nw_mera_closed_120], [*1107*], [1118], [1118],
      [str_nw_mera_open_26], [*709*], [710], [789],
    )
  ]
]

// =====================================================================
// Slide 8: Results (4 threads)
// =====================================================================
#slide("Results: 4 threads (opt_flops)")[
  #text(size: 14pt)[
    Apple Silicon M2 · median of 5 runs (2 warmup) · 2026-02-10
  ]

  #v(0.4em)

  #text(size: 12pt)[
    #table(
      columns: (2.9fr, 1.2fr, 1.4fr, 1.6fr),
      stroke: 0.5pt,
      inset: 6pt,
      [*Instance*], [*faer (ms)*], [*OpenBLAS (ms)*], [*OMEinsum (ms)*],
      [gm_queen5_5_3.wcsp], [3597], [4016], [-],
      [lm_batch_likelihood_brackets_4_4d], [*14.8*], [16.9], [22.7],
      [lm_batch_likelihood_sentence_3_12d], [*23.3*], [25.7], [35.6],
      [lm_batch_likelihood_sentence_4_4d], [*16.0*], [17.7], [20.9],
      [str_matrix_chain_multiplication_100], [*8.7*], [9.1], [16.7],
      [str_mps_varying_inner_product_200], [18.1], [21.8], [*15.0*],
      [str_nw_mera_closed_120], [380], [377], [*370*],
      [str_nw_mera_open_26], [227], [*226*], [266],
    )
  ]
]

// =====================================================================
// Slide 9: Benchmark notes
// =====================================================================
#slide("Benchmark notes")[
  - `-` = instance skipped (e.g. duplicate axis labels not yet supported)
  - strided-rs and OMEinsum.jl use the *same contraction path* — pure kernel-level comparison
  - *faer* = pure Rust GEMM #h(1em) *OpenBLAS* = vendor BLAS via `cblas-sys`
  - Row-major (NumPy benchmark) ↔ column-major (strided-rs) conversion is metadata-only

  #v(0.4em)

  - *Takeaway:* strided-rs (faer) is competitive with or faster than OMEinsum.jl (OpenBLAS) on most instances, especially for small-tensor workloads (LM, matrix chain)
]

// =====================================================================
// Slide 10: ndtensors-rs — proof of concept
// =====================================================================
#slide("3. ndtensors-rs: proof of concept")[
  - GitHub: #link("https://github.com/tensor4all/ndtensors-rs")[github.com/tensor4all/ndtensors-rs]
  - Experimental Rust port of NDTensors.jl — feasibility study

  #v(0.3em)

  - *What it demonstrates:*
    - Storage types: Dense, BlockSparse, Diag, Combiner
    - Ops: contract (GEMM), permute/reshape, SVD/QR/eigen
    - AD prototypes: backward (tape), forward (dual), HVP (FoR)
    - C-API: callable from Julia, Python, C++

  #v(0.3em)

  - *Key idea:* compatibility without rewriting Julia
    - Inject BLAS/LAPACK function pointers from Julia at runtime
    - One C-API enables multi-language backends
  - Status: POC complete — design feeds into *tenet* (next steps)
]

// =====================================================================
// Slide 11: Next — tenet
// =====================================================================
#slide("Next: tenet (GPU + Burn + Wirtinger AD)")[
  - *tenet*: physics-oriented tensor computing framework built on strided-rs

  #v(0.3em)

  #text(size: 14pt)[
    #table(
      columns: (1.6fr, 3.4fr),
      stroke: 0.5pt,
      inset: 6pt,
      [*Component*], [*Role*],
      [`tenet-core`], [GPU dispatch — cuBLAS / hipBLAS via dlopen · CubeCL JIT kernels],
      [`tenet-ad`], [Wirtinger-aware reverse-mode AD (Complex128)],
      [`tenet-block`], [Generic block-sparse tensor (symmetry-agnostic)],
      [`tenet-burn`], [Burn interop — reuse NN modules (Linear, Attention, …)],
    )
  ]

  #v(0.3em)

  - Primary types: f64, Complex128 (scientific computing)
  - GPU: NVIDIA (cudarc) + AMD ROCm — zero build-time GPU deps (all dlopen)
  - No Apple Metal GPU (no f64 ALUs) — Apple Silicon uses CPU via Accelerate
]

// =====================================================================
// Slide 12: Summary
// =====================================================================
#slide("Summary")[
  #text(size: 16pt)[
    #table(
      columns: (0.4fr, 1.6fr, 3fr),
      stroke: 0.5pt,
      inset: 6pt,
      [*\#*], [*What*], [*Status*],
      [1], [*Tensor4all.jl*], [`Pkg.add` + `Pkg.build` works · hdf5-rt / rsmpi-rt for runtime sharing],
      [2], [*strided-rs*], [Einsum on par with OMEinsum.jl · benchmarked on 8 instances],
      [3], [*ndtensors-rs*], [POC complete — Dense/BlockSparse/Diag + AD + C-API],
    )
  ]

  #v(0.5em)

  - *Next:* tenet — GPU acceleration, Wirtinger AD, Burn bridge
  - *Goal:* a Rust-native, `cargo add`-able alternative to the libtorch subset used in physics
]
