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
    #text(size: 36pt, weight: "bold", fill: rgb("#2563eb"))[#title]
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

// Slide 1: Title
#title-slide(
  "Tensor4all: a pure-Rust tensor stack",
  author: "Hiroshi Shinaoka",
  date: "2026-02-10",
  acknowledgements: "Acknowledgements: Jing-Guo Liu, Satoshi Terasaki, Ken Inayoshi",
)

// Slide 3: Stack overview
#slide("Architecture: 4 layers")[
  #text(size: 14pt)[
    #table(
      columns: (1.2fr, 3.8fr),
      stroke: 0.5pt,
      inset: 6pt,
      [*Layer*], [*What it provides (and repos)*],
      [4. Bindings], [*Compatibility & distribution (optional)*\
        Tensor4all.jl, Python: tensor4all-rs/python/tensor4all],
      [3. Tensor networks], [*TT/MPS, TreeTN, TCI, C-API, HDF5 serialization*\
        tensor4all-rs],
      [2. Tensor layer], [*Dense / BlockSparse / Diag + AD integration*\
        ndtensors-rs],
      [1. Backend], [*Arrays + linalg + einsum frontend (powered by strided-rs (New!))*\
        mdarray, mdarray-linalg, mdarray-opteinsum, strided-rs (New!)],
    )
  ]

  #v(0.4em)
  #text(size: 14pt)[
    Interop (for Julia/Python): hdf5-rt, rsmpi-rt.
  ]
]

// Slide 4: Quick usage
#slide("Minimal usage (Julia)")[
  - Install + build (auto-compiles the Rust backend):

  #code-block[
    ```julia
    using Pkg
    Pkg.add("Tensor4all")
    Pkg.build("Tensor4all")
    ```
  ]

  #v(0.5em)

  #code-block[
    ```julia
    using Tensor4all

    i = Index(2)
    j = Index(3; tags="Site,n=1")
    t = Tensor([i, j], rand(2, 3))

    dims(t)
    arr = Array(t, [j, i])
    ```
  ]
]

// Slide 6: Interop for bindings
#slide("Interop for bindings: reuse HDF5/MPI runtimes")[
  - *HDF5*: `hdf5-rt` (runtime loading via dlopen)
    - GitHub: #link("https://github.com/tensor4all/hdf5-rt")[https://github.com/tensor4all/hdf5-rt]
    - Reuse the HDF5 library already loaded by HDF5.jl / h5py

  #v(0.5em)

  - *MPI*: `rsmpi-rt` (MPItrampoline/MPIABI + dlopen)
    - GitHub: #link("https://github.com/tensor4all/rsmpi-rt")[https://github.com/tensor4all/rsmpi-rt]
    - Goal: run alongside MPI.jl / mpi4py, sharing `MPI_COMM_WORLD`
    - Runtime: set `MPI_RT_LIB` to the MPIwrapper library path
]

// Slide 7: Layer 3
#slide("Layer 3: tensor4all-rs (tensor networks)")[
  - GitHub: #link("https://github.com/tensor4all/tensor4all-rs")[https://github.com/tensor4all/tensor4all-rs]
  - Provides: TT/MPS, TreeTN, TCI, C-API, and HDF5 serialization
  - Designed as a Rust-native library; also consumable from Julia/Python via the C-API
]

// Slide 8: Layer 2 (overview)
#slide("Layer 2: the tensor layer (types + AD foundation)")[
  - Storage flavors: Dense / Diagonal / BlockSparse
  - AD: host-language AD + native Rust AD prototypes
  - Main repo: ndtensors-rs (#link("https://github.com/tensor4all/ndtensors-rs")[github.com/tensor4all/ndtensors-rs])
]

// Slide 9: ndtensors-rs
#slide("ndtensors-rs (experimental): goals & approach")[
  - GitHub: #link("https://github.com/tensor4all/ndtensors-rs")[https://github.com/tensor4all/ndtensors-rs]
  - Unofficial experimental Rust port of NDTensors.jl (feasibility study)
  - Target stack:
    - ITensors.jl → NDTensors.jl (Julia wrapper) → `ndtensors-rs` (Rust via C API)
  - Key idea: *compatibility without rewriting Julia*
    - Inject BLAS/LAPACK function pointers from Julia at runtime (C-pointer injection)
    - Select backends dynamically at runtime (e.g., CPU vs accelerator)
  - One C API enables multi-language backends (Julia / Python / C++)
]

// Slide 10: ndtensors-rs details
#slide("ndtensors-rs: AD + implementation status (high level)")[
  #text(size: 16pt)[
    - Automatic differentiation (multi-host):
      - Julia: ChainRules.jl rrule/frule
      - Python: JAX `custom_vjp/jvp`, PyTorch `autograd.Function`
      - Native Rust prototypes: backward (tape), forward (dual), HVP (FoR)

    #v(0.4em)

    - Implementation status (selected):
      - Storage/tensors: Dense, BlockSparse, Diag, Combiner
      - Ops: contract (GEMM), permute/reshape, norms, SVD/QR/eigen
      - C API: f64 DenseTensor basics; complex / block-sparse is WIP
  ]
]

// Slide 11: Layer 1
#slide("Layer 1: backend (arrays + linalg + einsum)")[
  - `mdarray`: core multidimensional arrays
    - GitHub: #link("https://github.com/fre-hu/mdarray")[https://github.com/fre-hu/mdarray]
  - `mdarray-linalg`: SVD/QR/LU/eigen backends
    - GitHub: #link("https://github.com/grothesque/mdarray-linalg")[https://github.com/grothesque/mdarray-linalg]
  - `mdarray-opteinsum` (developed by us): N-ary einsum for mdarray, powered by `strided-opteinsum` (in `strided-rs` (*New!*))
    - Repo: #link("https://github.com/tensor4all/strided-rs/tree/main/mdarray-opteinsum")[strided-rs/mdarray-opteinsum]
]

// Slide 12: strided-rs
#slide("strided-rs: an einsum engine in Rust")[
  - GitHub: #link("https://github.com/tensor4all/strided-rs")[https://github.com/tensor4all/strided-rs]
  - A Rust workspace for strided tensor views, kernels, and einsum
  - Components:
    - `strided-view`: dynamic-rank strided views
    - `strided-kernel`: cache-optimized map/reduce/broadcast
    - `strided-einsum2`: binary einsum (`einsum2_into`) with a GEMM backend
    - `strided-opteinsum`: N-ary einsum + contraction-order optimization
  - Two GEMM backends (benchmark-suite): `faer` (pure Rust) or OpenBLAS
]

// Slide 13: einsum2
#slide("strided-einsum2: fast binary einsum")[
  - A focused kernel for *two-input* einsum (GEMM-based)
  - Intended as the building block for N-ary contraction trees
  - Works on strided tensors/views (zero-copy transpose/permute metadata)
]

// Slide 14: Benchmark suite
#slide("Benchmark suite: fair Rust vs Julia comparison")[
  - GitHub: #link("https://github.com/tensor4all/strided-rs-benchmark-suite")[https://github.com/tensor4all/strided-rs-benchmark-suite]
  - Based on einsum benchmark (168 standardized problems / 7 categories)
  - Stores only metadata as JSON; tensors are generated at runtime (zero-filled)
  - Runners:
    - Rust: `strided-opteinsum` (faer + OpenBLAS)
    - Julia: OMEinsum.jl + TensorOperations.jl
  - Fairness: strided-rs and OMEinsum.jl follow the same pre-computed contraction path
  - Threads: `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, `JULIA_NUM_THREADS`
]

// Slide 15: Results (1 thread)
#slide("Results (1 thread, opt_flops)")[
  #text(size: 14pt)[
    Environment: Apple Silicon M4. Median time (ms) of 5 runs (2 warmup).
  ]

  #v(0.4em)

  #text(size: 14pt)[
    #table(
      columns: (2.8fr, 1.2fr, 1.4fr, 1.6fr),
      stroke: 0.5pt,
      inset: 6pt,
      [*Instance*], [*faer (ms)*], [*OpenBLAS (ms)*], [*OMEinsum (ms)*],
      [lm_batch_likelihood_brackets_4_4d], [*18.764*], [20.782], [20.838],
      [lm_batch_likelihood_sentence_3_12d], [*50.148*], [55.970], [61.733],
      [lm_batch_likelihood_sentence_4_4d], [21.200], [*21.036*], [24.490],
      [str_matrix_chain_multiplication_100], [13.497], [*11.450*], [19.548],
    )
  ]

  #v(0.4em)
  #text(size: 14pt)[
    TensorOperations.jl (matrix chain): 69.269 ms
  ]
]

// Slide 16: Results (4 threads)
#slide("Results (4 threads, opt_flops)")[
  #text(size: 14pt)[
    Environment: Apple Silicon M4. Median time (ms) of 5 runs (2 warmup).
  ]

  #v(0.4em)

  #text(size: 14pt)[
    #table(
      columns: (2.8fr, 1.2fr, 1.4fr, 1.6fr),
      stroke: 0.5pt,
      inset: 6pt,
      [*Instance*], [*faer (ms)*], [*OpenBLAS (ms)*], [*OMEinsum (ms)*],
      [lm_batch_likelihood_brackets_4_4d], [*14.946*], [16.315], [19.981],
      [lm_batch_likelihood_sentence_3_12d], [*25.458*], [28.477], [44.182],
      [lm_batch_likelihood_sentence_4_4d], [*16.360*], [17.363], [18.960],
      [str_matrix_chain_multiplication_100], [9.821], [*8.051*], [14.033],
    )
  ]

  #v(0.4em)
  #text(size: 14pt)[
    TensorOperations.jl (matrix chain): 25.583 ms
  ]
]

// Slide 17: Notes
#slide("Notes")[
  - `-` in the benchmark tables indicates TensorOperations.jl could not handle the LM instances
  - strided-rs vs OMEinsum.jl uses the same contraction path (kernel-level comparison)
  - `faer` = pure Rust GEMM; OpenBLAS = `cblas-sys`
  - Row-major (NumPy) ↔ column-major (strided-rs) conversion is metadata-only
]

// Slide 18: Wrap-up
#slide("Wrap-up")[
  #text(size: 14pt)[
    #table(
      columns: (1.2fr, 3.8fr),
      stroke: 0.5pt,
      inset: 6pt,
      [*Layer*], [*What it provides (and crates)*],
      [4. Bindings], [*Compatibility & distribution (optional)*\
        Tensor4all.jl, Python: tensor4all-rs/python/tensor4all],
      [3. Tensor networks], [*TT/MPS, TreeTN, TCI, C-API, HDF5 serialization*\
        tensor4all-rs],
      [2. Tensor layer], [*Dense / BlockSparse / Diag + AD integration*\
        ndtensors-rs],
      [1. Backend], [*Arrays + linalg + einsum frontend (powered by strided-rs (New!))*\
        mdarray, mdarray-linalg, mdarray-opteinsum, strided-rs (New!)],
    )
  ]
]
