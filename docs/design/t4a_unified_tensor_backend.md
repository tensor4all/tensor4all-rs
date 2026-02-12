# t4a: Unified Tensor Backend Design Plan

> **Note on naming**: `t4a` is a **working name** (short for tensor4all). The
> final project name is undecided. One candidate is
> **[tensoratu](https://gricad-gitlab.univ-grenoble-alpes.fr/theorypheliqs/tensoratu)**,
> proposed by the Grenoble team (TheoryPheliqs) as a tensor toolkit with
> hybrid indexing for tensor networks. Adopting `tensoratu` would change the
> crate prefix from `t4a-*` to `tensoratu-*` (e.g., `tensoratu-view`,
> `tensoratu-omeinsum`). Throughout this document, `t4a-*` should be read
> as a placeholder for whatever prefix is ultimately chosen.

## Context

Three independent Rust projects exist in tensor4all:
- **strided-rs**: Cache-optimized strided array kernels (view, map/reduce, einsum)
- **omeinsum-rs**: Einsum with tropical algebra, gradient support, GPU dispatch
- **ndtensors-rs**: Tensor types with storage hierarchy, linear algebra, autograd
- **tensor4all-rs**: Tensor network algorithms (TCI, Quantics, MPS) with ad-hoc tensor backend

These have significant overlap (3 einsum implementations, 3 scalar trait definitions, 3 dense storage types) yet critical gaps. The goal is to unify into a coherent, reusable tensor backend library **t4a-\*** that:

1. Integrates strided-rs and omeinsum-rs components directly (not as external dependencies)
2. Provides unified CPU/GPU dispatch for both element-wise ops and einsum
3. Supports complex numbers natively
4. Supports custom scalar types (tropical semiring, etc.) with pluggable GEMM backends
5. Exposes VJP/JVP through C API for Julia ChainRules.jl
6. Can optionally bridge to Burn for NN workloads

**Key design principles**:
- **Unified workspace**: All dense array infrastructure lives in one workspace (`t4a-rs`). No external strided-rs dependency.
- **Builder pattern for API stability**: All configuration (backend selection, options) uses builder pattern. New fields can be added to `BackendConfig` and other config types without breaking existing call sites. This is preferred over positional arguments or enum-heavy APIs.
- **Unified backend dispatch**: CPU/GPU dispatch is managed by `t4a-backend` (`BackendConfig`). Computation crates query this config — not the tensor layer above.
- **t4a-view is pure type design**: "strided" is about data layout, not operations. The view crate has no GPU dependencies.

---

## Crate Structure

```
t4a-rs/ (workspace) ── Dense array foundation ────────────
│
├── t4a-scalar           # Scalar trait hierarchy (ScalarBase, ElementOp, Scalar)
├── t4a-view             # StridedArrayView/Mut (zero-copy strided views over &[T])
├── t4a-buffer           # DataBuffer<T>: CPU Vec<T> / GPU CudaSlice<T> (Arc-based COW)
├── t4a-algebra          # Semiring/Algebra traits, tropical types (MaxPlus, MinPlus, MaxMul)
├── t4a-backend          # Unified backend dispatch (BackendConfig builder, global default)
├── t4a-mapreduce        # map/reduce/broadcast (CPU: cache-optimized, GPU: CubeCL)
├── t4a-omeinsum         # Einsum engine (CPU: GEMM backends, GPU: cuTENSOR)
│                        #   Binary contraction + N-ary optimizer (omeco)
│                        #   Respects OMEinsum.jl naming
├── t4a-tensor           # Tensor<T> = DataBuffer + sizes + strides + offset
│                        #   User-facing API, delegates to t4a-mapreduce / t4a-omeinsum
├── t4a-linalg           # SVD, QR, eigen, polar (CPU: faer, GPU: cuSOLVER)
├── t4a-autograd         # TrackedTensor, DualTensor, VJP/JVP
├── t4a-capi             # C FFI (tensor ops + VJP/JVP for ChainRules.jl)
└── burn-t4a             # Burn Backend bridge [OPTIONAL, for NN only]

t4a-structured-rs/ (workspace) ── Structured tensor types ──
│
├── t4a-blocksparse      # BlockSparseTensor (single DataBuffer + block offsets)
├── t4a-diag             # DiagTensor (1D Tensor of diagonal elements)
└── t4a-graded           # GradedTensor (future: quantum number sectors)

tensor4all-rs/ (workspace) ── Tensor network algorithms ────
│
├── TCI, Quantics, MPS, ...
└── depends on t4a-rs + t4a-structured-rs
```

### Dependency Graph

```
t4a-scalar
    │
    ├──────────────────────────────┐
    ↓                              ↓
t4a-view                      t4a-buffer
    │                              │
    ├──── t4a-algebra ←────────────┤
    │         │                    │
    ↓         ↓                    ↓
t4a-backend ←──────────────────────┘  (BackendConfig, global default)
    │
    ├──────────────────────────────┐
    ↓                              ↓
t4a-mapreduce                 t4a-omeinsum
(queries t4a-backend)         (queries t4a-backend)
    │                              │
    └──────────────┬───────────────┘
                   ↓
              t4a-tensor
                   │
         ┌─────────┼───────────┐
         ↓         ↓           ↓
   t4a-linalg  t4a-autograd  t4a-capi
   (← faer)

[optional]
burn-t4a ← t4a-tensor, burn-backend

[separate workspace: t4a-structured-rs]
t4a-blocksparse ← t4a-tensor
t4a-diag        ← t4a-tensor
t4a-graded      ← t4a-blocksparse (future)
```

### Origin of Each Crate

| t4a crate | Origin | What changes |
|-----------|--------|--------------|
| t4a-scalar | strided-traits | Adds `Scalar` (division, complex), `RealScalar` |
| t4a-view | strided-view | Rename only |
| t4a-buffer | New | CPU/GPU buffer abstraction |
| t4a-algebra | omeinsum-rs (Algebra traits) | Standalone crate for Semiring/tropical types |
| t4a-backend | New | Unified dispatch config (BackendConfig builder) |
| t4a-mapreduce | strided-kernel | Rename + add GPU dispatch via CubeCL |
| t4a-omeinsum | strided-einsum2 + strided-opteinsum + omeinsum-rs | Merge all einsum into one, add GPU dispatch |
| t4a-tensor | New | Tensor<T> API over DataBuffer |
| t4a-linalg | ndtensors-rs (linalg) | Port SVD/QR/eigen |
| t4a-autograd | ndtensors-rs (autodiff) | Port TrackedTensor/DualTensor |
| t4a-capi | ndtensors-rs (capi) + tensor4all-rs (capi) | Port C FFI |
| burn-t4a | New | Burn Backend bridge |
| **t4a-structured-rs (separate workspace):** | | |
| t4a-blocksparse | ndtensors-rs (blocksparse) | Port with single-buffer layout |
| t4a-diag | ndtensors-rs (diag) | Port DiagTensor |
| t4a-graded | New (future) | Quantum number graded tensors |

---

## Type Hierarchy

```
t4a-rs:
    t4a-buffer: DataBuffer<T>  ← 1D flat memory (CPU Vec<T> or GPU CudaSlice<T>)
        │
    t4a-tensor: Tensor<T>     ← strides + sizes + offset over DataBuffer
                                 = "Dense tensor". The fundamental primitive.

t4a-structured-rs (separate workspace, depends on t4a-tensor):
    t4a-diag: DiagTensor<T>
        diagonal elements stored as 1D Tensor<T>
    t4a-blocksparse: BlockSparseTensor<T>
        single DataBuffer + block offsets (ITensors.jl pattern)
        each block is a Tensor<T> view into the shared buffer
    t4a-graded: GradedTensor<T, S: Sector> (future)
        BlockSparseTensor with sector-labeled block indices
```

---

## Phase 1: Dense Array Foundation

### t4a-scalar

```rust
// t4a-scalar/src/lib.rs

/// Minimal scalar trait for map/reduce/einsum.
/// Copy + Send + Sync + basic arithmetic.
pub trait ScalarBase: Copy + Send + Sync + Add<Output=Self> + Mul<Output=Self>
    + Zero + One + PartialEq + Debug {}

/// Element operation applied lazily on access (Identity, Conj, Transpose, Adjoint).
pub trait ElementOpApply: ScalarBase {
    fn conjugate(self) -> Self;
    fn transpose(self) -> Self;
    fn adjoint(self) -> Self;
}

/// Rich scalar for linalg and standard numeric types.
pub trait Scalar: ScalarBase + ElementOpApply
    + Div<Output=Self> + Sub<Output=Self> + Neg<Output=Self> + SubAssign + DivAssign
{
    type Real: RealScalar;
    fn conjugate(self) -> Self;
    fn real_part(self) -> Self::Real;
    fn imag_part(self) -> Self::Real;
    fn abs_squared(self) -> Self::Real;
    fn from_real(re: Self::Real) -> Self;
    fn is_complex() -> bool;
}

pub trait RealScalar: Scalar<Real = Self> + PartialOrd {
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn to_f64(self) -> f64;
    fn from_f64(val: f64) -> Self;
}
```

**Implementations**:

| Trait | Types |
|-------|-------|
| `ScalarBase` | f32, f64, Complex\<f32\>, Complex\<f64\>, i32, i64, u32, u64 |
| `ElementOpApply` | f32, f64, Complex\<f32\>, Complex\<f64\> |
| `Scalar` | f32, f64, Complex\<f32\>, Complex\<f64\> |
| `RealScalar` | f32, f64 |

Integer types implement `ScalarBase` only — sufficient for einsum via naive
backend and map/reduce operations. They do not implement `Scalar` (no
conjugate, division semantics differ) or `ElementOpApply`.

**Custom types**: Tropical semiring only needs `ScalarBase`. Types with custom GEMM implement `BgemmBackend<T>`.

### t4a-view

Renamed from strided-view. No functional changes.

```rust
pub struct StridedArrayView<'a, T, const N: usize, Op = Identity> { ... }
pub struct StridedArrayViewMut<'a, T, const N: usize, Op = Identity> { ... }
```

- Borrows `&'a [T]` — pure CPU, no GPU dependencies
- Const-generic rank `N`
- Zero-copy: slice, reshape, permute, transpose
- Lazy element operations via `Op` type parameter (Identity, Conj, Transpose, Adjoint)

### t4a-buffer

```rust
pub struct CpuBuffer<T: ScalarBase> {
    data: Vec<T>,
}

#[cfg(feature = "cuda")]
pub struct GpuBuffer<T: ScalarBase> {
    data: cudarc::driver::CudaSlice<T>,
    device: Arc<cudarc::driver::CudaDevice>,
}

pub enum DataBuffer<T: ScalarBase> {
    Cpu(Arc<CpuBuffer<T>>),
    #[cfg(feature = "cuda")]
    Gpu(Arc<GpuBuffer<T>>),
}
```

Shared ownership via `Arc`. COW via `Arc::make_mut`.

### t4a-algebra

Semiring/Algebra abstraction from omeinsum-rs, as a standalone crate:

```rust
pub trait Semiring: ScalarBase {
    fn sem_zero() -> Self;
    fn sem_one() -> Self;
    fn sem_add(self, rhs: Self) -> Self;
    fn sem_mul(self, rhs: Self) -> Self;
}

/// Standard linear algebra: sem_add = +, sem_mul = ×
pub struct Standard<T>(pub T);

/// Tropical (max-plus): sem_add = max, sem_mul = +
pub struct MaxPlus<T>(pub T);

/// Tropical (min-plus): sem_add = min, sem_mul = +
pub struct MinPlus<T>(pub T);

/// Tropical (max-mul): sem_add = max, sem_mul = ×
pub struct MaxMul<T>(pub T);
```

Also provides:
- Argmax tracking for tropical backward pass
- Algebra trait extending Semiring with optional backward/gradient support

### t4a-backend

Unified backend dispatch configuration. All computation crates (t4a-mapreduce,
t4a-omeinsum, t4a-linalg) query this crate for backend selection instead of
managing dispatch independently.

**BackendConfig builder** (Rust builder pattern for stable, extensible API):

```rust
/// Backend configuration. Built via builder pattern.
/// New fields can be added without breaking existing code.
#[derive(Clone, Debug)]
pub struct BackendConfig {
    device: ComputeDevice,
    gemm: GemmBackend,
    linalg: LinalgBackend,
    mapreduce: MapReduceBackend,
}

#[derive(Clone, Debug, Default)]
pub enum ComputeDevice {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),  // device_id
    Auto,
}

#[derive(Clone, Debug, Default)]
pub enum GemmBackend {
    #[default]
    Faer,
    Naive,
    // Future extensions (added without breaking API thanks to builder pattern):
    // Blas,          // OpenBLAS / MKL via cblas-inject or system CBLAS
    // TropicalGemm,  // SIMD-optimized tropical algebra GEMM
    // #[cfg(feature = "cuda")]
    // CuTensor,      // GPU via cuTENSOR
    Auto,
}

#[derive(Clone, Debug, Default)]
pub enum LinalgBackend {
    #[default]
    Faer,
    #[cfg(feature = "cuda")]
    CuSolver,
    Auto,
}

#[derive(Clone, Debug, Default)]
pub enum MapReduceBackend {
    #[default]
    Cpu,
    #[cfg(feature = "cubecl")]
    CubeCL,
    Auto,
}

impl BackendConfig {
    pub fn new() -> Self { Self::default() }

    pub fn device(mut self, device: ComputeDevice) -> Self {
        self.device = device; self
    }
    pub fn gemm(mut self, gemm: GemmBackend) -> Self {
        self.gemm = gemm; self
    }
    pub fn linalg(mut self, linalg: LinalgBackend) -> Self {
        self.linalg = linalg; self
    }
    pub fn mapreduce(mut self, mr: MapReduceBackend) -> Self {
        self.mapreduce = mr; self
    }
}
```

**Two-tier dispatch: global default + per-call override**:

```rust
// 1. Set global default (once at startup)
t4a_backend::set_global_default(
    BackendConfig::new()
        .device(ComputeDevice::Cpu)
        .gemm(GemmBackend::Faer)
);

// 2. Simple API — uses global default
einsum(&[&a, &b], &labels)?;
reduce(&src, &dst, |a, b| a + b)?;

// 3. Per-call override via _with variant
let cfg = BackendConfig::new().gemm(GemmBackend::Blas);
einsum_with(&[&a, &b], &labels, &cfg)?;
```

Thread-local scoped override is intentionally omitted — it is error-prone
in async/multithreaded Rust. The two-tier model (global + per-call) is
sufficient and predictable.

**`Auto` resolution**: When a field is `Auto`, t4a-backend resolves it based
on context (e.g., `ComputeDevice::Auto` → GPU if any input is on GPU, else CPU).
Resolution happens once at call entry, producing a fully-resolved `BackendConfig`.

### t4a-mapreduce

Absorbs strided-kernel. Adds GPU dispatch.

**CPU path** (from strided-kernel):
- Cache-optimized kernels: dimension fusion, stride-based reordering, L1 tiled iteration
- `map_into`, `zip_map2_into`, `zip_map3_into`, `zip_map4_into`
- `reduce`, `reduce_axis`, `mapreducedim_into`
- `broadcast_into`, `CaptureArgs`
- Contiguous fast paths bypass blocking for direct iteration

**GPU path** (feature-gated: `cubecl`):
- Custom GPU kernels via [CubeCL](https://github.com/tracel-ai/cubecl)
- CubeCL compiles Rust kernel code to CUDA/WebGPU/etc.
- Map/reduce/broadcast dispatched by `DataBuffer` variant

**Dispatch via t4a-backend**: queries `BackendConfig::mapreduce()` to select
CPU or GPU path. Per-call override via `_with` variants.

### t4a-omeinsum

Merges strided-einsum2 + strided-opteinsum + omeinsum-rs. Named after OMEinsum.jl.

**Pluggable backend trait hierarchy** — two levels of abstraction:

```rust
/// Level 1: Batched GEMM (minimal custom backend)
/// t4a-omeinsum handles permute → reshape → bgemm pipeline
trait BgemmBackend<T: ScalarBase> {
    fn bgemm(/* alpha, a, b, beta, c, batch, m, n, k, strides */) -> Result<()>;
}

/// Level 2: Direct tensor contraction (cuTENSOR-like)
/// Backend handles the entire contraction — no permute/reshape needed
trait TensordotBackend<T: ScalarBase> {
    fn tensordot(
        a: /* buffer + sizes + strides */,
        b: /* buffer + sizes + strides */,
        contracted: &[(usize, usize)],
        out: /* buffer */,
    ) -> Result<()>;
}
```

**Dispatch priority** (t4a-omeinsum resolves per contraction):
1. `TensordotBackend` available? → direct tensor contraction (cuTENSOR, etc.)
2. `BgemmBackend` available? → permute → reshape → bgemm (faer, custom GEMM)
3. Neither? → naive loop fallback (any `ScalarBase`)

**Built-in implementations**:

| Backend | Implements | Types | Notes |
|---|---|---|---|
| FaerBackend | `BgemmBackend<T>` | f32, f64, Complex32, Complex64 | Default CPU GEMM |
| NaiveBackend | (fallback) | any `ScalarBase` | Loop-based, no GEMM |
| CuTensorBackend | `TensordotBackend<T>` | f32, f64, Complex32, Complex64 | GPU, future |

**User extension points**:
- **Custom CPU GEMM**: implement `BgemmBackend<MyType>` — t4a-omeinsum
  orchestrates the permute/reshape/bgemm pipeline automatically
- **Custom GPU/accelerator**: implement `TensordotBackend<MyType>` — bypasses
  the reshape-to-GEMM path entirely for maximum performance
- **No implementation needed**: `ScalarBase` types fallback to naive loop

**CPU performance optimizations** (from strided-rs):
- Element-wise bypass: all-batch (no contraction) → `zip_map2_into` instead of GEMM
- Single-tensor fast paths: direct trace loop, partial trace, zero-copy permutation
- Fusability-aware reshape-to-GEMM: `try_fuse_group` avoids unnecessary copies
- Buffer pool: intermediate tensors recycled across contraction steps

**N-ary optimization**: omeco contraction tree optimizer (Greedy, TreeSA)

**Algebra support**: dispatch GEMM backend based on algebra and scalar type:
- `Standard<f64>` / `Standard<f32>` / `Standard<Complex64>` → faer GEMM
- `Standard<i32>` / `Standard<i64>` / `Standard<u32>` / `Standard<u64>` → naive loop
- `MaxPlus<f64>` → tropical-gemm (SIMD, future)
- Custom → user-provided `BgemmBackend`

**Dispatch via t4a-backend**: queries `BackendConfig::gemm()` and
`BackendConfig::device()` to select CPU GEMM backend or GPU cuTENSOR.
Per-call override via `einsum_with` / `einsum2_with`.

**`Auto` resolution** (when `GemmBackend::Auto` or `ComputeDevice::Auto`):
1. If any input is on GPU → cuTENSOR (future)
2. If N-ary (>2 inputs) → omeco contraction order optimization, then pairwise dispatch
3. If `T: Scalar` (f32/f64/Complex) → Faer
4. If `T: ScalarBase` + `BgemmBackend<T>` → `einsum2_with_backend_into`
5. Fallback: naive loop (integers, tropical, custom ScalarBase types)

**Initial CPU backend**: faer only (pure Rust, no external dependencies).
BLAS backends (cblas-inject, system CBLAS) and tropical-gemm are deferred to
later phases. The builder pattern (`BackendConfig`) ensures these can be added
as new `GemmBackend` variants without breaking existing call sites.

**Future GEMM backends** (added via feature flags):
- `blas`: system CBLAS (OpenBLAS, MKL)
- `cblas-inject`: runtime-injected CBLAS (e.g., from Julia's OpenBLAS)
- `tropical-gemm`: SIMD-optimized tropical algebra GEMM
- Custom: implement `BgemmBackend<T>`

### t4a-tensor

```rust
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    sizes: SmallVec<[usize; 6]>,
    strides: SmallVec<[isize; 6]>,
    storage_offset: usize,
}
```

**Bridge to t4a-view** (CPU path):
```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn as_strided_view(&self) -> StridedArrayView<'_, T, N, Identity>;
    pub fn as_strided_view_mut(&mut self) -> StridedArrayViewMut<'_, T, N>;  // triggers COW
}
```

**Zero-copy view operations**: permute, transpose, slice, select, expand, flip, diagonal — all modify metadata only.

**Element-wise operations** (delegates to t4a-mapreduce):
- `map`, `zip_map`, `reduce`, `sum`, `fill`
- Arithmetic operators: Add, Sub, Mul (element-wise)

**Einsum** (delegates to t4a-omeinsum):
```rust
// Simple API — uses global default BackendConfig
pub fn einsum<T: ScalarBase>(
    inputs: &[&Tensor<T>],
    input_labels: &[&[i32]],
    output_labels: &[i32],
) -> Result<Tensor<T>>;

// Per-call override
pub fn einsum_with<T: ScalarBase>(
    inputs: &[&Tensor<T>],
    input_labels: &[&[i32]],
    output_labels: &[i32],
    backend: &BackendConfig,
) -> Result<Tensor<T>>;
```

---

## Phase 2: t4a-linalg

All decomposition functions:
1. Call `tensor.contiguous_col_major()` (faer is column-major)
2. Create `faer::MatRef` from raw pointer + strides
3. Call faer's decomposition
4. Wrap result back into Tensor

**Operations**: svd, svd_truncated, qr, qr_positive, ql, eigen_hermitian, eigen, polar, matrix_exp.

**N-D tensor decomposition**: specify left_dims for "row" side, reshape to 2D, decompose, reshape back.

**Trait bound**: `T: Scalar + faer::ComplexField` (enforced here, not in t4a-scalar/tensor).

**Dispatch via t4a-backend**: queries `BackendConfig::linalg()` to select
CPU (faer) or GPU (cuSOLVER). Same two-tier pattern:

```rust
// Simple API — uses global default
pub fn svd<T: Scalar>(tensor: &Tensor<T>) -> Result<SvdResult<T>>;

// Per-call override
pub fn svd_with<T: Scalar>(tensor: &Tensor<T>, backend: &BackendConfig) -> Result<SvdResult<T>>;
```

**Source files to create**:
- `t4a-linalg/src/lib.rs` — public API
- `t4a-linalg/src/faer_bridge.rs` — Tensor<T> ↔ faer::MatRef conversion
- `t4a-linalg/src/svd.rs`, `qr.rs`, `eigen.rs`, `polar.rs`

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/backend/faer_interop.rs` — faer bridge pattern
- `ndtensors-rs/crates/ndtensors/src/linalg/` — SVD, QR implementations

**Verification**: Cross-validate SVD/QR results against ndtensors-rs.

---

## Phase 3: t4a-autograd (Primary AD System)

This is the **default** AD system for t4a users. Burn's autodiff is only for NN workloads.

### Reverse-Mode (Backward)

Adapted from `ndtensors-rs/crates/ndtensors/src/autodiff/`:

```rust
pub struct TrackedTensor<T: Scalar> {
    tensor: Tensor<T>,
    node: Option<NodeRef>,
    requires_grad: bool,
}

pub trait GradFn<T: Scalar>: Debug + Send {
    fn backward(&self, grad_output: &Tensor<T>) -> Vec<(NodeId, Tensor<T>)>;
    fn inputs(&self) -> Vec<NodeId>;
}

pub fn backward<T: Scalar>(loss: &TrackedTensor<T>) -> Result<Gradients<T>>;
```

### Forward-Mode (JVP)

```rust
pub struct DualTensor<T: Scalar> {
    primal: Tensor<T>,
    tangent: Option<Tensor<T>>,
}
```

### Contraction VJP/JVP

```rust
// VJP: grad_A = contract(grad_C, labels_gc, B, labels_b_vjp)
//      grad_B = contract(A, labels_a_vjp, grad_C, labels_gc)
pub fn contract_vjp<T: Scalar>(
    a: &Tensor<T>, labels_a: &[i32],
    b: &Tensor<T>, labels_b: &[i32],
    grad_output: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)>;

// JVP: dC = contract(dA, B) + contract(A, dB)  (Leibniz rule)
pub fn dual_contract<T: Scalar>(
    a: &DualTensor<T>, labels_a: &[i32],
    b: &DualTensor<T>, labels_b: &[i32],
) -> Result<DualTensor<T>>;
```

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/autodiff/` — backward pass, graph, TrackedTensor
- `ndtensors-rs/crates/ndtensors/src/contract/naive.rs:222` — contract_vjp
- `ndtensors-rs/crates/ndtensors/src/autodiff/ops/dual_contract.rs` — JVP

**Verification**:
- Numerical gradient checks (finite difference vs AD)
- Forward-mode vs reverse-mode consistency
- Complex-valued contraction gradients (Wirtinger calculus)

---

## Phase 4: t4a-capi (C FFI for ChainRules.jl)

### ChainRules.jl Integration Design

Julia's ChainRules.jl defines:
- `rrule(f, args...)` → `(result, pullback)` — reverse-mode rule
- `frule((Δself, Δargs...), f, args...)` → `(result, Δresult)` — forward-mode rule

t4a-capi exposes the VJP/JVP primitives that Julia wraps as ChainRules rules:

```c
// Opaque types
typedef struct t4a_tensor_f64 t4a_tensor_f64;
typedef struct t4a_tensor_c64 t4a_tensor_c64;

// === Core tensor lifecycle ===
t4a_tensor_f64* t4a_tensor_f64_from_data(const double* data, const size_t* shape, size_t ndim);
void t4a_tensor_f64_release(t4a_tensor_f64* tensor);
const double* t4a_tensor_f64_data_ptr(const t4a_tensor_f64* tensor);
size_t t4a_tensor_f64_ndim(const t4a_tensor_f64* tensor);

// === Contraction ===
t4a_tensor_f64* t4a_contract_f64(
    const t4a_tensor_f64* a, const int32_t* labels_a,
    const t4a_tensor_f64* b, const int32_t* labels_b,
    int* status);

// === ChainRules: VJP (for rrule pullback) ===
int t4a_contract_vjp_f64(
    const t4a_tensor_f64* a, const int32_t* labels_a, size_t ndim_a,
    const t4a_tensor_f64* b, const int32_t* labels_b, size_t ndim_b,
    const t4a_tensor_f64* grad_c,
    t4a_tensor_f64** grad_a_out,
    t4a_tensor_f64** grad_b_out);

// === ChainRules: JVP (for frule) ===
t4a_tensor_f64* t4a_contract_jvp_f64(
    const t4a_tensor_f64* a, const int32_t* labels_a, size_t ndim_a,
    const t4a_tensor_f64* b, const int32_t* labels_b, size_t ndim_b,
    const t4a_tensor_f64* da,   // nullable (zero tangent)
    const t4a_tensor_f64* db,   // nullable (zero tangent)
    int* status);

// Duplicate for _c64 variants
```

**Uses integer labels** (i32, matching ndtensors-rs convention: negative = contracted, positive = output), not string notation, for C API ergonomics.

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors-capi/src/lib.rs` — C API patterns
- `tensor4all-rs/crates/tensor4all-capi/src/` — opaque type patterns, status codes

---

## Phase 5: t4a-structured-rs (separate workspace)

Structured tensor types built on top of `Tensor<T>`. These live in a
**separate workspace** (`t4a-structured-rs`) so they can be used by
projects other than tensor4all-rs without pulling in application-level
dependencies.

### t4a-diag

```rust
pub struct DiagTensor<T: ScalarBase> {
    diag: Tensor<T>,          // 1D Tensor storing diagonal elements
    full_sizes: Vec<usize>,   // logical shape of the full tensor
}
```

### t4a-blocksparse

Follows the **ITensors.jl/NDTensors pattern**: all blocks in a **single contiguous `DataBuffer`** with block offset mapping.

```rust
pub struct BlockSparseTensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    block_offsets: HashMap<BlockIndex, usize>,
    block_sizes: HashMap<BlockIndex, Vec<usize>>,
    full_sizes: Vec<usize>,
}

pub type BlockIndex = SmallVec<[usize; 4]>;
```

Each block accessed as a `Tensor<T>` view into the shared buffer (zero-copy).

### t4a-graded (future)

Quantum number graded tensors. BlockSparseTensor with sector-labeled
block indices and fusion-rule-constrained block structure.

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/operations/blocksparse.rs`
- `ndtensors-rs/crates/ndtensors/src/contract/blocksparse.rs`

---

## Phase 6: burn-t4a (Optional Burn Backend for NN)

**Only needed when**: Users want Burn's NN modules (conv, attention, optimizers) with t4a tensors.

```rust
#[derive(Debug, Clone)]
pub enum T4aBurnTensor {
    F64(t4a_tensor::Tensor<f64>),
    F32(t4a_tensor::Tensor<f32>),
    I64(t4a_tensor::Tensor<i64>),
    I32(t4a_tensor::Tensor<i32>),
    Bool(t4a_tensor::Tensor<bool>),
}

impl Backend for BurnT4a { ... }
```

| Burn Method | t4a Implementation |
|---|---|
| `float_add` | `t4a_mapreduce::zip_map2_into` |
| `float_matmul` | `t4a_omeinsum::einsum2_into` |
| `float_exp/sin/...` | `t4a_mapreduce::map_into` |
| `float_reshape/permute/slice` | Metadata-only (zero-copy) |
| `float_sum/sum_dim` | `t4a_mapreduce::reduce` / `reduce_axis` |

For NN autograd: `Autodiff<BurnT4a>` (Burn's decorator pattern).
For tensor network AD: Use t4a-autograd directly.

---

## GPU Strategy

### Design Principle

All CPU/GPU dispatch is managed through **t4a-backend** (`BackendConfig`).
Computation crates query `BackendConfig` and route accordingly:

| Crate | BackendConfig field | CPU | GPU |
|-------|---------------------|-----|-----|
| t4a-mapreduce | `mapreduce()` | Cache-optimized kernels | CubeCL custom kernels |
| t4a-omeinsum | `gemm()` + `device()` | faer, BLAS, naive, tropical-gemm | cuTENSOR |
| t4a-linalg | `linalg()` | faer | cuSOLVER |

Users control dispatch via global default or per-call `_with` variants — not
by feature flags or buffer introspection at the call site.

### CubeCL for map/reduce/broadcast

[CubeCL](https://github.com/tracel-ai/cubecl) compiles Rust kernel code to CUDA/WebGPU. This enables GPU map/reduce/broadcast without hand-written CUDA:

```rust
// Example: element-wise map kernel (CubeCL)
#[cube(launch)]
fn map_kernel<T: Numeric>(input: &Tensor<T>, output: &mut Tensor<T>) {
    let idx = ABSOLUTE_POS;
    output[idx] = some_op(input[idx]);
}
```

Feature-gated: `cubecl` feature in t4a-mapreduce.

### cuTENSOR for einsum

- Native N-ary tensor contraction with optimization
- Accepts strided tensors (no need to make contiguous)
- Supports f32, f64, Complex32, Complex64
- Feature-gated: `cuda` feature in t4a-omeinsum

### GPU buffer (t4a-buffer)

```toml
# t4a-buffer/Cargo.toml
[features]
default = []
cuda = ["dep:cudarc"]
```

GPU tensors participate in the same computation graph as CPU tensors (t4a-autograd).

---

## Custom Scalar Type Support

### Three-Tier Dispatch

| Tier | Trait Bound | Backend | Types |
|---|---|---|---|
| Optimized (tensordot) | `TensordotBackend<T>` | Direct contraction | GPU (cuTENSOR), custom accelerators |
| Optimized (GEMM) | `BgemmBackend<T>` | permute→reshape→bgemm | f32, f64, Complex (faer), custom GEMM |
| Generic | `ScalarBase` only | naive loop | Integers, tropical, boolean, any custom type |

Users extend the system by implementing traits at the appropriate level:
- **Simple custom type**: implement `ScalarBase` → naive loop works automatically
- **Custom type with fast GEMM**: also implement `BgemmBackend<T>`
- **Custom accelerator**: implement `TensordotBackend<T>` for full control

### Algebra-Aware Dispatch

t4a-omeinsum dispatches backend based on algebra and scalar type:
- `Standard<f64>` / `Standard<f32>` / `Standard<Complex64>` → faer `BgemmBackend`
- `Standard<i32>` / `Standard<i64>` / `Standard<u32>` / `Standard<u64>` → naive loop
- `MaxPlus<f64>` → tropical-gemm `BgemmBackend` (future)
- GPU tensors → cuTENSOR `TensordotBackend` (future)
- Custom algebras → user-provided `BgemmBackend` or `TensordotBackend`

---

## Migration Strategy (tensor4all-rs)

### Replace tensor4all-rs's tensorbackend with t4a

1. Replace `tensor4all-tensorbackend::Storage` with t4a types (`Tensor<T>` from t4a-rs, `DiagTensor<T>`, `BlockSparseTensor<T>` from t4a-structured-rs)
2. Replace `DenseStorage<T>` (mdarray-based) with `Tensor<T>`
3. Adapt `TensorLike` trait to use t4a Tensor
4. Remove mdarray dependency from tensor4all-rs core

This happens **after t4a core is stable**.

---

## Verification Plan

### After Phase 1 (core):
```bash
cd t4a-rs && cargo test -p t4a-scalar -p t4a-view -p t4a-buffer -p t4a-algebra -p t4a-backend -p t4a-mapreduce -p t4a-omeinsum -p t4a-tensor
```
- Unit tests for all Tensor operations (permute, slice, reshape, etc.)
- Zero-copy verification: assert same Arc pointer after view ops
- Tropical semiring contraction test (t4a-algebra + naive backend)
- Integer type einsum test (i32, i64 via naive backend)
- **Custom type extensibility tests** (CRITICAL for maintaining system extensibility):
  - Define a test-only custom scalar type (e.g., `ModInt<P>` — integers mod P)
    that implements only `ScalarBase` → verify einsum works via naive fallback
  - Define a test-only custom `BgemmBackend<ModInt<P>>` → verify einsum uses
    the custom GEMM path (permute→reshape→bgemm)
  - Define a test-only custom `TensordotBackend<ModInt<P>>` → verify einsum
    bypasses the GEMM path and calls tensordot directly
  - These tests guarantee that downstream users can extend the system without
    touching t4a internals
- Benchmark: t4a einsum vs current tensor4all-rs mdarray-einsum

### After Phase 2 (linalg):
```bash
cargo test -p t4a-linalg
```
- Cross-validate SVD/QR results against ndtensors-rs
- Complex SVD test

### After Phase 3 (autograd):
```bash
cargo test -p t4a-autograd
```
- Numerical gradient checks (finite difference vs reverse-mode AD)
- Forward-mode vs reverse-mode consistency for contraction
- Complex-valued gradient test (Wirtinger calculus)

### After Phase 4 (C API):
```bash
cargo test -p t4a-capi
# From Julia:
julia -e 'using Tensor4all; Tensor4all.test_chainrules()'
```
- Round-trip test: Julia → C API → Rust → C API → Julia
- ChainRules.jl integration test with Zygote.jl

---

## Key Files Reference

| Component | Source | Destination |
|---|---|---|
| ScalarBase trait | `strided-rs/strided-traits/src/scalar.rs` | t4a-scalar |
| ElementOp | `strided-rs/strided-traits/src/element_op.rs` | t4a-scalar |
| StridedArrayView | `strided-rs/strided-view/src/view.rs` | t4a-view |
| map/reduce kernels | `strided-rs/strided-kernel/src/` | t4a-mapreduce |
| einsum2_into | `strided-rs/strided-einsum2/src/lib.rs` | t4a-omeinsum |
| einsum2_naive_into | `strided-rs/strided-einsum2/src/lib.rs` | t4a-omeinsum |
| BgemmBackend | `strided-rs/strided-einsum2/src/backend.rs` | t4a-omeinsum |
| opteinsum | `strided-rs/strided-opteinsum/src/lib.rs` | t4a-omeinsum |
| Algebra traits | `omeinsum-rs/src/` | t4a-algebra |
| BackendConfig | New | t4a-backend |
| tropical-gemm | `omeinsum-rs` (dependency) | t4a-omeinsum |
| faer bridge | `ndtensors-rs/.../faer_interop.rs` | t4a-linalg |
| contract_vjp | `ndtensors-rs/.../contract/naive.rs` | t4a-autograd |
| TrackedTensor | `ndtensors-rs/.../autodiff/tensor.rs` | t4a-autograd |
| backward pass | `ndtensors-rs/.../autodiff/backward.rs` | t4a-autograd |
| C API patterns | `tensor4all-rs/crates/tensor4all-capi/src/` | t4a-capi |

---

## Future Considerations

### Complex-valued differentiation rules for linear algebra

Complex SVD, QR, eigen decompositions require non-trivial backward rules (Wirtinger calculus, structured perturbation theory). Key references:

- **[BackwardsLinalg.jl](https://github.com/GiggleLiu/BackwardsLinalg.jl)**: Reference implementations for SVD, QR, Cholesky, eigen backward rules in the complex case.
- **[MatrixFactorizations.jl](https://github.com/JuliaLinearAlgebra/MatrixFactorizations.jl)**: Extended factorizations with ChainRules.jl integration.

### JAX / PyTorch integration via C-FFI

t4a-capi should support integration with JAX and PyTorch via ctypes (no PyO3). Pattern already implemented in ndtensors-rs:

```
JAX / PyTorch → Python wrapper → ctypes FFI → t4a-capi → Rust
```

- JAX: `jax.custom_vjp` + `jax.pure_callback()`
- PyTorch: `torch.autograd.Function` with custom forward/backward

**Existing code to port**: `ndtensors-rs/python/ndtensors_rs/src/ndtensors_rs/`

### Insights from ITensor Julia ecosystem

| Aspect | ITensor Julia | t4a | Notes |
|---|---|---|---|
| Sparse storage | DOK-of-Arrays | Single DataBuffer + offset map | t4a is GPU-friendly |
| Axis fusion | FusionStyle dispatch | Not yet designed | Critical for quantum number tensors |
| Lazy evaluation | MapBroadcast.jl | Not planned | Low priority |

Key takeaways:
1. **FusionStyle hierarchy**: Needed for quantum number tensors (future). Keep blocksparse API extensible.
2. **SparseArraysBase pattern**: Consider `SparseArray` trait for t4a-blocksparse and t4a-diag.
3. **GradedArrays.jl**: Future `GradedTensor<T, S: Sector>` for symmetry-exploiting tensor networks.

---

## Relationship with mdarray / mdarray-linalg

### Two complementary layers

| | mdarray / mdarray-linalg | t4a-* |
|---|---|---|
| Role | **numpy equivalent** — general-purpose multidimensional array | **PyTorch equivalent** — high-performance tensor library |
| Memory | Owned `Array<T, D>` | `DataBuffer<T>` (Arc-based COW, CPU/GPU) |
| GPU | No | CubeCL, cuTENSOR, cuSOLVER |
| Autodiff | No | t4a-autograd (VJP/JVP) |
| Einsum | No | t4a-omeinsum (contraction tree optimization) |
| Dispatch | Direct function calls | BackendConfig (runtime backend selection) |
| Use cases | Lightweight, embeddable, general numerics | Tensor networks, ML, large-scale scientific computing |

Both are needed. mdarray is a foundational array library (simple, no heavy
dependencies); t4a builds a richer tensor ecosystem on top of different
abstractions.

### Why t4a does NOT go through mdarray-linalg

t4a-linalg and mdarray-linalg are **parallel** (both call faer directly),
not **serial** (t4a does not call mdarray-linalg):

```
faer (SVD, QR, eigen)
    ↑                ↑
t4a-linalg       mdarray-linalg-faer
(Tensor<T>       (Array<T, D>
 → MatRef)        → MatRef)
```

Reasons:
1. **No conversion overhead**: `Tensor<T>` bridges to `faer::MatRef` directly
   via raw pointer + strides. Going through mdarray would add unnecessary
   `Tensor<T>` → `Array<T>` → faer → `Array<T>` → `Tensor<T>` round-trips.
2. **GPU dispatch**: t4a-linalg dispatches to cuSOLVER via `BackendConfig`.
   mdarray-linalg has no GPU concept.
3. **BackendConfig integration**: Runtime backend selection is a t4a concern
   that doesn't exist in mdarray-linalg's design.
4. **Different memory models**: `DataBuffer<T>` (Arc, COW, CPU/GPU) vs
   mdarray's owned `Vec<T>` storage. The bridge abstractions are different.

### mdarray as a dependency

strided-view (now t4a-view) currently depends on mdarray for the base `Array`
type used in some constructors and tests. This dependency is lightweight and
may be retained for interop convenience, but t4a's core data path
(`DataBuffer<T>` → `Tensor<T>` → faer) does not go through mdarray.
