# t4a: Unified Tensor Backend Design Plan

## Context

Three independent Rust projects exist in tensor4all:
- **strided-rs**: Cache-optimized strided array kernels (view, map/reduce, einsum)
- **ndtensors-rs**: Tensor types with storage hierarchy, linear algebra, autograd
- **tensor4all-rs**: Tensor network algorithms (TCI, Quantics, MPS) with ad-hoc tensor backend

These have significant overlap (3 einsum implementations, 3 scalar trait definitions, 3 dense storage types) yet critical gaps. The goal is to unify into a coherent, reusable tensor backend library **t4a-\*** that:

1. Is a standalone tensor library with built-in AD (default path)
2. Can optionally bridge to Burn for NN workloads
3. Supports complex numbers natively
4. Exposes VJP/JVP through C API for Julia ChainRules.jl
5. Supports custom scalar types (tropical semiring, etc.) with pluggable GEMM backends

**Key design principle**: t4a-* is self-contained. Burn is only used when NN modules are needed.

---

## Crate Structure

```
t4a/ (workspace)
├── t4a-scalar          # Scalar trait hierarchy
├── t4a-buffer          # Raw data buffer: DataBuffer (Arc-based COW, CPU/GPU)
├── t4a-tensor          # Tensor type (sizes + strides + offset over DataBuffer)
├── t4a-linalg          # SVD, QR, eigen, polar (via faer)
├── t4a-autograd        # TrackedTensor, DualTensor, computation graph [PRIMARY AD]
├── t4a-blocksparse     # BlockSparseTensor (single DataBuffer + block offsets)
├── t4a-diag            # DiagTensor (1D Tensor of diagonal elements)
├── t4a-capi            # C FFI (tensor ops + VJP/JVP for ChainRules.jl)
└── burn-t4a            # Burn Backend bridge [OPTIONAL, for NN only]
```

### Dependency Graph

```
strided-traits (ScalarBase, ElementOp)
    |
strided-view (StridedView, StridedArray)
    |
strided-kernel (map/reduce/broadcast)      strided-einsum2 (binary einsum + GEMM)
    |                                           |
    +------------ t4a-scalar ← num-traits, num-complex
                      |
                  t4a-buffer
                      |
                  t4a-tensor ← strided-view, strided-kernel, strided-einsum2, strided-opteinsum
                      |
          +-----------+-----------+
          |           |           |
    t4a-linalg    t4a-autograd   t4a-blocksparse
    (← faer)          |          t4a-diag
                      |
                  t4a-capi ← t4a-tensor, t4a-autograd, t4a-linalg

    [optional]
    burn-t4a ← t4a-tensor, burn-backend
    burn-complex ← burn-backend (decorator for complex in Burn)
```

---

## Type Hierarchy

```
t4a-buffer: DataBuffer<T>  ← 1D flat memory (CPU Vec<T> or GPU CudaSlice<T>)
    │
t4a-tensor: Tensor<T>     ← strided view over DataBuffer (sizes + strides + offset)
    │                        = "Dense tensor". The fundamental primitive.
    │
    ├── t4a-diag: DiagTensor<T>
    │     diagonal elements stored as 1D Tensor<T>
    │
    └── t4a-blocksparse: BlockSparseTensor<T>
          single DataBuffer + block offsets (ITensors.jl pattern)
          each block is a Tensor<T> view into the shared buffer
```

**Naming convention**: `DataBuffer` is the raw 1D memory buffer (replaces the confusing name "Storage" which could be mistaken for DenseStorage/DiagStorage/BlockSparseStorage in tensor4all-rs). `Tensor<T>` is a strided view over a `DataBuffer`, equivalent to "dense tensor".

---

## Phase 1: t4a-scalar + t4a-buffer + t4a-tensor

### t4a-scalar

Re-export `strided_traits::ScalarBase` as `t4a_scalar::ScalarBase` (option (a) from design doc — avoids orphan rule issues).

```rust
// t4a-scalar/src/lib.rs
pub use strided_traits::ScalarBase;   // minimal: Copy+Send+Sync+Mul+Add+Zero+One+PartialEq
pub use strided_traits::ElementOpApply;

pub trait Scalar: ScalarBase + Div<Output=Self> + DivAssign + Sub<Output=Self> + Neg<Output=Self> + SubAssign {
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

**Implementations**: f32, f64, Complex<f32>, Complex<f64>.

**Bridge**: ScalarBase aligns with strided-rs's ScalarBase. faer::ComplexField is enforced only at t4a-linalg level.

**Custom types**: Tropical semiring only needs `ScalarBase` (no `Scalar` or GEMM required). Uses `einsum2_naive_into`.

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

Shared ownership via `Arc` inside enum variants. COW via `Arc::make_mut`.

### t4a-tensor

```rust
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,            // flat 1D memory (CPU or GPU)
    sizes: SmallVec<[usize; 6]>,
    strides: SmallVec<[isize; 6]>,   // isize for flip() support
    storage_offset: usize,
}
```

**Bridge to strided-rs**:
```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn as_strided_view(&self) -> StridedView<'_, T, Identity>;
    pub fn as_strided_view_mut(&mut self) -> StridedViewMut<'_, T>;  // triggers COW
}
```

**Zero-copy view operations**: permute, transpose, slice, select, expand, flip, diagonal — all modify metadata only.

**Element-wise operations** (via strided-kernel):
- `map`, `zip_map`, `reduce`, `sum`, `fill`
- Arithmetic operators: Add, Sub, Mul (element-wise)

**Contraction / Einsum** — dispatched by buffer location and scalar type:

| Buffer | Scalar Tier | Backend | Function |
|---|---|---|---|
| CPU | `ScalarBase` (any) | loop-based | `strided_einsum2::einsum2_naive_into` |
| CPU | `Scalar` (f32/f64/Complex) | GEMM (faer/BLAS) | `strided_einsum2::einsum2_into` |
| CPU | N-ary | opt-einsum optimizer | `strided_opteinsum::einsum` |
| GPU | `Scalar` (f32/f64/Complex) | **cuTENSOR** | `cudarc::cutensor` |

**Unified einsum entry point**:
```rust
pub fn einsum<T: ScalarBase>(
    inputs: &[&Tensor<T>],
    input_labels: &[&[i32]],
    output_labels: &[i32],
) -> Result<Tensor<T>>;
```

Dispatch logic:
1. If **any** input is on GPU → all inputs moved to GPU, use cuTENSOR
2. If all inputs are on CPU and `T: Scalar` → use GEMM-accelerated `einsum2_into`
3. If all inputs are on CPU and `T: ScalarBase` only → use `einsum2_naive_into`
4. For N-ary (>2 inputs) on CPU → use `strided_opteinsum` for contraction order optimization, then pairwise dispatch

**cuTENSOR integration** (GPU einsum):
- cuTENSOR natively supports N-ary tensor contraction with optimization
- Accepts strided tensors (no need to make contiguous first)
- Supports f32, f64, Complex32, Complex64
- Contraction path optimization built-in (like opt-einsum)
- Feature-gated: `cuda` feature in t4a-buffer propagates to t4a-tensor

**CPU GEMM backends**: strided-einsum2 already supports pluggable backends via cargo features:
- `faer` (default): pure-Rust GEMM via faer
- `blas`: system CBLAS
- `blas-inject`: injected CBLAS (e.g., from Julia's OpenBLAS)

These features propagate through t4a-tensor's Cargo.toml.

**Source files to create**:
- `t4a-scalar/src/lib.rs` — trait hierarchy
- `t4a-scalar/src/impls.rs` — f32, f64, Complex impls
- `t4a-buffer/src/lib.rs` — DataBuffer<T>, CpuBuffer<T>, GpuBuffer<T>
- `t4a-tensor/src/lib.rs` — Tensor<T>, view ops, bridge to strided-rs
- `t4a-tensor/src/ops.rs` — element-wise ops via strided-kernel
- `t4a-tensor/src/einsum.rs` — contraction wrappers (Scalar tier + ScalarBase tier)
- `t4a-tensor/src/contiguous.rs` — contiguous/col-major materialization

**Existing code to reuse**:
- `strided-rs/strided-traits/src/scalar.rs` — ScalarBase definition
- `strided-rs/strided-view/src/view.rs` — StridedView type (bridge target)
- `strided-rs/strided-kernel/src/lib.rs` — map_into, zip_map2_into, reduce, sum, dot
- `strided-rs/strided-einsum2/src/lib.rs:183` — einsum2_into
- `strided-rs/strided-einsum2/src/lib.rs:287` — einsum2_naive_into
- `strided-rs/strided-opteinsum/src/lib.rs:46` — einsum (N-ary)

**Verification**:
- `cargo test` passes all unit tests
- Benchmark: zero-copy permute chain (5 permutes) vs data-copying permute
- Benchmark: einsum via strided-rs vs current tensor4all-rs mdarray-einsum
- Test: tropical semiring contraction via einsum_naive

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

Port from ndtensors-rs (`crates/ndtensors/src/contract/naive.rs:189-400`):

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

**Source files to create**:
- `t4a-autograd/src/lib.rs`
- `t4a-autograd/src/graph.rs` — ComputationGraph, Node, NodeId
- `t4a-autograd/src/tracked.rs` — TrackedTensor
- `t4a-autograd/src/dual.rs` — DualTensor
- `t4a-autograd/src/backward.rs` — backward pass (topological sort + gradient propagation)
- `t4a-autograd/src/ops/contract.rs` — contract_vjp, dual_contract
- `t4a-autograd/src/gradients.rs` — Gradients container

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/autodiff/backward.rs` — backward pass logic
- `ndtensors-rs/crates/ndtensors/src/autodiff/graph.rs` — ComputationGraph, GradFn
- `ndtensors-rs/crates/ndtensors/src/autodiff/tensor.rs` — TrackedTensor
- `ndtensors-rs/crates/ndtensors/src/autodiff/ops/dual_contract.rs` — JVP for contraction
- `ndtensors-rs/crates/ndtensors/src/contract/naive.rs:222` — contract_vjp

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

**Julia side** (in Tensor4all.jl):
```julia
function ChainRules.rrule(::typeof(contract), A, labels_a, B, labels_b)
    C = contract(A, labels_a, B, labels_b)
    function contract_pullback(ΔC)
        grad_a, grad_b = ccall((:t4a_contract_vjp_f64, libt4a), ...)
        return NoTangent(), grad_a, NoTangent(), grad_b, NoTangent()
    end
    return C, contract_pullback
end

function ChainRules.frule((_, ΔA, _, ΔB, _), ::typeof(contract), A, labels_a, B, labels_b)
    C = contract(A, labels_a, B, labels_b)
    ΔC = ccall((:t4a_contract_jvp_f64, libt4a), ...)
    return C, ΔC
end
```

**Uses integer labels** (i32, matching ndtensors-rs convention: negative = contracted, positive = output), not string notation, for C API ergonomics.

**Source files to create**:
- `t4a-capi/src/lib.rs`
- `t4a-capi/src/tensor.rs` — opaque tensor types + lifecycle
- `t4a-capi/src/contract.rs` — contraction + VJP/JVP
- `t4a-capi/src/linalg.rs` — SVD, QR exports

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors-capi/src/lib.rs` — C API patterns
- `tensor4all-rs/crates/tensor4all-capi/src/` — opaque type patterns, status codes, macros

---

## Phase 5: t4a-blocksparse + t4a-diag

Higher-level tensor types built on top of `Tensor<T>` (which is the fundamental primitive).

### t4a-diag

```rust
pub struct DiagTensor<T: ScalarBase> {
    diag: Tensor<T>,          // 1D Tensor storing diagonal elements
    full_sizes: Vec<usize>,   // logical shape of the full tensor
}
```

O(n) diagonal operations. Contraction with dense tensors avoids materializing the full matrix.

### t4a-blocksparse

Follows the **ITensors.jl/NDTensors pattern**: all blocks stored in a **single contiguous `DataBuffer`** with block offset mapping.

```rust
pub struct BlockSparseTensor<T: ScalarBase> {
    buffer: DataBuffer<T>,                           // single flat buffer for ALL blocks
    block_offsets: HashMap<BlockIndex, usize>,        // block → offset into buffer
    block_sizes: HashMap<BlockIndex, Vec<usize>>,     // block → shape per block
    full_sizes: Vec<usize>,                           // logical shape of the full tensor
}

pub type BlockIndex = SmallVec<[usize; 4]>;  // N-dimensional block index
```

**Memory layout** (matching ITensors.jl `BlockSparse{ElT,VecT,N}`):
```
buffer: [block_1_elems..., block_2_elems..., block_3_elems...]
         ↑ offset=0        ↑ offset=n₁       ↑ offset=n₁+n₂

block_offsets:
  (0,1) → 0
  (1,0) → n₁
  (2,1) → n₁+n₂
```

Each block is accessed as a `Tensor<T>` view into the shared buffer (zero-copy):
```rust
impl<T: ScalarBase> BlockSparseTensor<T> {
    pub fn block_view(&self, index: &BlockIndex) -> Option<Tensor<T>>;
}
```

**Advantages of single-buffer layout**:
- Cache locality for sequential block access
- Single allocation / deallocation
- Compatible with GPU: entire buffer can reside in `GpuBuffer<T>`
- New blocks appended without reorganizing existing data

**Contraction**: block-wise GEMM using block sparsity structure (only non-zero block pairs contracted).

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/operations/blocksparse.rs`
- `ndtensors-rs/crates/ndtensors/src/operations/diag.rs`
- `ndtensors-rs/crates/ndtensors/src/contract/blocksparse.rs`
- ITensors.jl reference: `~/git/ITensors.jl/NDTensors/src/blocksparse/blocksparse.jl`

---

## Phase 6: burn-t4a (Optional Burn Backend for NN)

**Only needed when**: Users want to use Burn's NN modules (conv, attention, optimizers) with t4a tensors.

### Architecture

```rust
// burn-t4a/src/lib.rs

/// Dynamic-dtype tensor enum (follows burn-ndarray's NdArrayTensor pattern)
#[derive(Debug, Clone)]
pub enum T4aBurnTensor {
    F64(t4a_tensor::Tensor<f64>),
    F32(t4a_tensor::Tensor<f32>),
    I64(t4a_tensor::Tensor<i64>),
    I32(t4a_tensor::Tensor<i32>),
    // ... other dtypes
    Bool(t4a_tensor::Tensor<bool>),  // bool does NOT impl ScalarBase; needs special handling
}

#[derive(Clone, Copy, Default, Debug)]
pub struct BurnT4a;

impl Backend for BurnT4a {
    type Device = T4aDevice;
    type FloatTensorPrimitive = T4aBurnTensor;
    type FloatElem = f64;
    type IntTensorPrimitive = T4aBurnTensor;
    type IntElem = i64;
    type BoolTensorPrimitive = T4aBurnTensor;
    type BoolElem = bool;
    // ...
}
```

### FloatTensorOps mapping

| Burn Method | t4a Implementation |
|---|---|
| `float_add` | `strided_kernel::zip_map2_into(dst, a, b, \|a,b\| a+b)` |
| `float_matmul` | `strided_einsum2::einsum2_into` |
| `float_exp/sin/cos/...` | `strided_kernel::map_into(dst, src, f64::exp)` |
| `float_reshape` | Metadata-only (zero-copy) |
| `float_permute` | Metadata-only (zero-copy) |
| `float_slice` | Adjust offset+strides (zero-copy) |
| `float_sum` | `strided_kernel::reduce` |
| `float_sum_dim` | `strided_kernel::reduce_axis` |

### Autograd for NN: Use burn-autodiff

When NN autograd is needed: `Autodiff<BurnT4a>` (Burn's decorator pattern).
This wraps BurnT4a's FloatTensorPrimitive with graph nodes automatically.

For tensor network AD (contraction VJP/JVP): Use t4a-autograd directly (not through Burn).

### burn-complex (Complex support in Burn)

Per `github.com/shinaoka/burn/issues/1`:
- Decorator pattern with `ComplexTensorBackend` trait
- `ComplexTensorOps` with ~60 methods
- Interleaved memory layout `[re0, im0, re1, im1, ...]`
- burn-t4a can provide optimized implementation using native `Tensor<Complex64>`

**Source files**:
- `burn-t4a/src/backend.rs` — Backend impl
- `burn-t4a/src/tensor.rs` — T4aBurnTensor enum + TensorMetadata
- `burn-t4a/src/ops/float.rs` — FloatTensorOps (~80 methods)
- `burn-t4a/src/ops/int.rs` — IntTensorOps (~40 methods)
- `burn-t4a/src/ops/bool.rs` — BoolTensorOps (~30 methods)
- `burn-t4a/src/ops/modules.rs` — ModuleOps (conv, pool, attention)
- `burn-complex/src/lib.rs` — Complex decorator

---

## Custom Scalar Type Support

### Two-Tier Architecture

| Tier | Trait Bound | GEMM | Einsum Function | Types |
|---|---|---|---|---|
| Generic | `ScalarBase` | No | `einsum2_naive_into` | Tropical, log-semiring, boolean, custom |
| Optimized | `Scalar` + backend | Yes (faer/BLAS) | `einsum2_into` | f32, f64, Complex32, Complex64 |

### Custom GEMM Backend

strided-einsum2 already supports this via cargo features:
```toml
[features]
default = ["faer"]
faer = ["dep:faer"]
blas = ["dep:cblas-sys"]
blas-inject = ["dep:cblas-inject"]  # Julia's OpenBLAS via runtime injection
```

To add a NEW custom GEMM backend:
1. Implement `BgemmBackend` trait in strided-einsum2
2. Add new feature flag
3. t4a-tensor propagates the feature

For truly custom scalar types (no GEMM at all):
```rust
// User code
use t4a_tensor::Tensor;
use t4a_tensor::einsum_naive;

let a = Tensor::<Tropical>::from_vec(data_a, &[2, 3]);
let b = Tensor::<Tropical>::from_vec(data_b, &[3, 4]);
let c = einsum_naive("ij,jk->ik", &[&a, &b])?;
```

---

## Migration Strategy (tensor4all-rs)

### Phase 6b: Replace tensor4all-rs's tensorbackend with t4a

1. Replace `tensor4all-tensorbackend::Storage` with t4a types (`Tensor<T>`, `DiagTensor<T>`, `BlockSparseTensor<T>`)
2. Replace `DenseStorage<T>` (mdarray-based) with `Tensor<T>` (strided-view over `DataBuffer`)
3. Adapt `TensorLike` trait to use t4a Tensor
4. Remove mdarray dependency from tensor4all-rs core

This phase is OUT OF SCOPE for this plan — it happens after t4a is stable.

---

## Verification Plan

### After Phase 1 (core):
```bash
cd t4a && cargo test -p t4a-scalar -p t4a-buffer -p t4a-tensor
```
- Unit tests for all Tensor operations (permute, slice, reshape, etc.)
- Zero-copy verification: assert same Arc pointer after view ops
- Tropical semiring contraction test
- Benchmark: strided-rs einsum vs mdarray-einsum

### After Phase 2 (linalg):
```bash
cargo test -p t4a-linalg
```
- Cross-validate SVD/QR results against ndtensors-rs
- Verify truncated SVD rank selection
- Complex SVD test

### After Phase 3 (autograd):
```bash
cargo test -p t4a-autograd
```
- Numerical gradient checks (finite diff vs reverse-mode AD)
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

### After Phase 6 (burn-t4a):
```bash
cargo test -p burn-t4a
# Run Burn's backend test suite
cargo test -p burn-backend-tests --features t4a
```
- Pass Burn's standard backend test suite
- Autodiff<BurnT4a> works for simple NN model

---

## Key Files Reference

| Component | Source | Purpose |
|---|---|---|
| ScalarBase trait | `strided-rs/strided-traits/src/scalar.rs:10` | Re-export as t4a-scalar::ScalarBase |
| StridedView | `strided-rs/strided-view/src/view.rs:88` | Bridge target for Tensor<T> |
| Kernel ops | `strided-rs/strided-kernel/src/lib.rs` | map_into, reduce, sum, dot |
| einsum2_into | `strided-rs/strided-einsum2/src/lib.rs:183` | GEMM-accelerated contraction |
| einsum2_naive | `strided-rs/strided-einsum2/src/lib.rs:287` | Generic contraction (any ScalarBase) |
| opteinsum | `strided-rs/strided-opteinsum/src/lib.rs:46` | N-ary einsum |
| faer bridge | `ndtensors-rs/crates/ndtensors/src/backend/faer_interop.rs` | Tensor↔faer conversion |
| contract_vjp | `ndtensors-rs/crates/ndtensors/src/contract/naive.rs:222` | VJP implementation |
| dual_contract | `ndtensors-rs/crates/ndtensors/src/autodiff/ops/dual_contract.rs:55` | JVP implementation |
| TrackedTensor | `ndtensors-rs/crates/ndtensors/src/autodiff/tensor.rs` | Reverse-mode AD tensor |
| backward pass | `ndtensors-rs/crates/ndtensors/src/autodiff/backward.rs:45` | Topological sort + grad propagation |
| C API patterns | `tensor4all-rs/crates/tensor4all-capi/src/` | Opaque types, status codes |
| Burn Backend | `~/git/burn/crates/burn-backend/src/backend/base.rs` | Backend + AutodiffBackend traits |
| NdArrayTensor | `~/git/burn/crates/burn-ndarray/src/tensor.rs:23` | Reference enum pattern |
| Burn complex | `github.com/shinaoka/burn/issues/1` | ComplexTensorBackend design |

---

## Future Considerations

### Complex-valued differentiation rules for linear algebra

Complex SVD, QR, eigen decompositions require non-trivial backward rules (Wirtinger calculus, structured perturbation theory). Key references:

- **[BackwardsLinalg.jl](https://github.com/GiggleLiu/BackwardsLinalg.jl)**: Reference implementations of backward rules for SVD, QR, Cholesky, eigen, etc. in the complex case. Includes handling of degenerate singular values and correct conjugation patterns.
- **[MatrixFactorizations.jl](https://github.com/JuliaLinearAlgebra/MatrixFactorizations.jl)**: Extended matrix factorizations (QL, positive QR, etc.) with ChainRules.jl integration. Provides frule/rrule implementations that serve as correctness references.

These should be consulted when implementing VJP/JVP for t4a-linalg operations (Phase 2 + Phase 3 intersection), especially for:
- SVD backward with degenerate/near-degenerate singular values
- QR backward for complex matrices (sign conventions differ from real case)
- Eigendecomposition backward with repeated eigenvalues
- Truncated SVD gradient (requires careful handling of discarded singular vectors)

### GEMM-capable custom scalar types

The current two-tier architecture (ScalarBase → naive einsum, Scalar → GEMM einsum) assumes custom types cannot use GEMM. A third tier should be considered:

| Tier | Trait Bound | GEMM | Einsum Function | Types |
|---|---|---|---|---|
| Generic | `ScalarBase` | No | `einsum2_naive_into` | Tropical, log-semiring, boolean |
| Custom GEMM | `ScalarBase` + `GemmScalar` | User-provided | `einsum2_into` with custom backend | Interval arithmetic, AD scalars, multiprecision |
| Optimized | `Scalar` + built-in backend | Yes (faer/BLAS) | `einsum2_into` | f32, f64, Complex32, Complex64 |

This enables users to provide their own GEMM implementation for custom types that have matrix multiplication semantics but are not standard floating-point types. Examples:
- **Interval arithmetic** (e.g., `Interval<f64>`): GEMM is mathematically valid, just needs custom implementation
- **AD scalars** (e.g., dual numbers `Dual<f64>`): element-wise GEMM is correct
- **Multiprecision** (e.g., `BigFloat`): GEMM via naive loops or specialized libraries

The `GemmScalar` trait would extend `ScalarBase` with a GEMM kernel registration mechanism, allowing strided-einsum2 to dispatch to user-provided GEMM at the einsum level without modifying the core library.

### JAX / PyTorch integration via C-FFI

t4a-capi should support integration with JAX and PyTorch, enabling Rust-accelerated tensor contraction with autodiff support in Python ML frameworks. **This pattern is already implemented in ndtensors-rs** and should be ported to t4a.

#### Existing implementation in ndtensors-rs

**Architecture** (no PyO3 — pure ctypes over C API):

```
JAX / PyTorch
    ↓
Python wrapper (jax_ops.py / torch_ops.py)
    ↓
jax.pure_callback / torch.autograd.Function
    ↓
ctypes FFI (_lib.py)
    ↓
C API (ndtensors-capi)
    ↓
Rust (ndtensors)
```

**JAX integration** (`ndtensors-rs/python/ndtensors_rs/src/ndtensors_rs/jax_ops.py`):
- `jax.custom_vjp` with `nondiff_argnums` for labels
- Forward: `jax.pure_callback()` → ctypes → Rust `contract()`
- Backward: `jax.pure_callback()` → ctypes → Rust `contract_vjp()`
- JIT-compatible via `pure_callback` with `ShapeDtypeStruct`

**PyTorch integration** (`ndtensors-rs/python/ndtensors_rs/src/ndtensors_rs/torch_ops.py`):
- `torch.autograd.Function` subclass with custom forward/backward
- Forward: `ctx.save_for_backward()` → ctypes → Rust `contract()`
- Backward: ctypes → Rust `contract_vjp()` → gradients on same device

**Key design decisions**:
- ctypes (not PyO3) for zero build dependency on Python
- `cdylib` crate type for shared library
- Opaque pointer wrapping `Box<Tensor<f64>>`, Python `__del__` calls release
- `catch_unwind()` at FFI boundary for panic safety
- Status codes for error propagation (not exceptions)
- numpy as data interchange format (column-major via `to_numpy()` / `from_numpy()`)

#### t4a-capi plan

Port the Python wrapper layer to use t4a-capi instead of ndtensors-capi:

**Files to create**:
- `python/t4a/src/t4a/_lib.py` — ctypes declarations for t4a-capi
- `python/t4a/src/t4a/tensor.py` — TensorF64 / TensorC64 wrapper
- `python/t4a/src/t4a/ops.py` — contract, contract_vjp, contract_jvp
- `python/t4a/src/t4a/jax_ops.py` — `jax.custom_vjp` wrapper
- `python/t4a/src/t4a/torch_ops.py` — `torch.autograd.Function` wrapper
- `python/t4a/pyproject.toml` — hatchling build with custom Rust compile hook

**Existing code to reuse** (direct port):
- `ndtensors-rs/python/ndtensors_rs/src/ndtensors_rs/jax_ops.py`
- `ndtensors-rs/python/ndtensors_rs/src/ndtensors_rs/torch_ops.py`
- `ndtensors-rs/python/ndtensors_rs/src/ndtensors_rs/_lib.py`
- `ndtensors-rs/python/ndtensors_rs/src/ndtensors_rs/tensor.py`
- `ndtensors-rs/python/ndtensors_rs/src/ndtensors_rs/ops.py`
- `ndtensors-rs/python/ndtensors_rs/src/ndtensors_rs/_status.py`
- `ndtensors-rs/python/ndtensors_rs/hatch_build.py`

**Extensions beyond ndtensors-rs**:
- Complex tensor support (TensorC64) for JAX/PyTorch
- JVP support for JAX (`jax.custom_jvp`) via `t4a_contract_jvp_f64`
- SVD/QR with VJP/JVP for differentiable linear algebra from Python

### GPU array support via cudarc

t4a-tensor should support GPU (CUDA) device arrays, enabling GEMM and SVD on GPU. This extends the scope beyond CPU-only.

#### Motivation

- Tensor network contraction (einsum) and linear algebra (SVD, QR) are the dominant cost in TCI/Quantics algorithms
- GPU acceleration for these operations provides significant speedup for large tensors
- cudarc is already used in the tensor4all ecosystem (`extern/scirs2`, version 0.18, CUDA 12.0)
- Burn's CUDA backend also uses cudarc, confirming its viability

#### Design: GPU buffer variant

`DataBuffer<T>` (defined in t4a-buffer) already includes `Cpu` and `Gpu` variants (see Phase 1). The `cuda` feature gate enables the `Gpu` variant with `cudarc::driver::CudaSlice<T>`.

**Tensor<T>** remains the same (sizes + strides + offset over DataBuffer), but operations dispatch based on buffer backend. BlockSparseTensor also benefits: its single `DataBuffer` can reside on GPU.

#### GPU operations via cudarc

| Operation | CUDA Library | cudarc Module |
|---|---|---|
| **Einsum (contraction)** | **cuTENSOR** | `cudarc::cutensor` |
| SVD | cuSOLVER (cusolverDn) | `cudarc::cusolver` |
| QR | cuSOLVER (cusolverDn) | `cudarc::cusolver` |
| Element-wise ops | Custom CUDA kernels | `cudarc::driver` + NVRTC |

**Note**: GPU einsum uses cuTENSOR (not cuBLAS GEMM). cuTENSOR provides native N-ary tensor contraction with built-in contraction path optimization, strided tensor support, and complex dtype support — making it the natural GPU counterpart to strided-einsum2 + strided-opteinsum on CPU. See the unified einsum dispatch table in Phase 1.

#### Memory management

- **RAII**: `CudaSlice<T>` auto-frees on drop
- **Host↔Device transfer**: `htod_sync_copy()` / `dtoh_sync_copy()` for data movement
- **Stream-ordered**: Asynchronous operations via `CudaStream`
- **COW**: Same `Arc`-based COW pattern as CPU storage; `Arc::make_mut` triggers device-side copy

#### Feature gating

```toml
# t4a-buffer/Cargo.toml
[features]
default = []
cuda = ["dep:cudarc"]

[dependencies]
cudarc = { version = "0.18", optional = true, features = ["cuda-12000", "driver", "cublas", "cusolver"] }

[target.'cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))'.dependencies]
cudarc = { version = "0.18", optional = true }
```

Platform-conditional (x86_64 Linux/Windows only), matching the pattern in `extern/scirs2`.

#### Integration with autograd

GPU tensors participate in the same computation graph as CPU tensors:
- `TrackedTensor<T>` wraps `Tensor<T>` regardless of storage location
- VJP/JVP operations dispatch to GPU when inputs are on GPU
- Gradient tensors are allocated on the same device as the primal tensors
- `backward()` traverses the graph; each `GradFn` operates on the device of its inputs

#### Phasing

This is a follow-up to the core CPU implementation (Phases 1–4). Suggested order:
1. GPU buffer + host↔device transfer (t4a-buffer `cuda` feature)
2. GEMM on GPU via cuBLAS (t4a-tensor einsum GPU path)
3. SVD/QR on GPU via cuSOLVER (t4a-linalg GPU path)
4. Autograd with GPU tensors (t4a-autograd, no new graph logic needed)
5. Custom CUDA kernels for element-wise ops (NVRTC compilation)
