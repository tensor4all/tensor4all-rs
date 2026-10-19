//! Cached batch evaluation for tree tensor networks.

use crate::error::TreeTNOperationError;
use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

/// Temporary counters for issue #671's root-cause investigation: whether the
/// branch contraction kernel's per-physical-value BLAS setup (building the
/// `left` intermediate in [`TreeTNCachedEvaluator::grouped_branch_message_contraction`])
/// is a fixed cost that amortizes poorly against Guard's typically small
/// per-call point counts, unlike the chain kernel
/// ([`TreeTNCachedEvaluator::grouped_chain_message_contraction`]), which has
/// an explicit `CHAIN_BLAS_MIN_GROUP_POINTS`/`groups.len() > 8` safeguard the
/// branch kernel does not. Not on the hot path unless `diagnostics` is
/// enabled.
#[cfg(feature = "diagnostics")]
pub(crate) mod contraction_diagnostics {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static BRANCH_SETUP_NS: AtomicU64 = AtomicU64::new(0);
    // [AI Supplied] Diagnostic-only split around the two representation
    // copies that precede the already-instrumented branch kernel.
    pub static BRANCH_CHILD_DECODE_NS: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_CHILD_GATHER_NS: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_CHILD_DECODE_VALUES: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_CHILD_GATHER_VALUES: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_SETUP_VALUES: AtomicU64 = AtomicU64::new(0);
    // [AI Supplied] Counts the evaluator-owned immutable branch-slice cache
    // decisions separately from the GEMM dispatch counters.
    pub static BRANCH_PREPARED_SLICE_HITS: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_PREPARED_SLICE_MISSES: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_PREPARED_SLICE_REFUSALS: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_PREPARED_SLICE_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_MATMUL_NS: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_ACCUMULATE_NS: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_BLAS_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_BLAS_POINTS: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_BLAS_GROUPS: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_SCALAR_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_SCALAR_POINTS: AtomicU64 = AtomicU64::new(0);

    /// Scratch counters included in [`summary`] for whether the
    /// contiguous-read fast path's `parent_axis == raw's fastest axis`
    /// condition is the common case among calls that reach the BLAS path, or
    /// whether `child_axis_1`/`child_axis_2` land there instead -- deciding
    /// whether generalizing the fast path to those two axes is worth the
    /// added complexity. One increment per BLAS dispatch; the axis assignment
    /// is the same for every group in a kernel invocation.
    pub static BRANCH_FAST_AXIS_IS_PARENT: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_FAST_AXIS_IS_CHILD1: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_FAST_AXIS_IS_CHILD2: AtomicU64 = AtomicU64::new(0);
    pub static BRANCH_FAST_AXIS_IS_PHYSICAL: AtomicU64 = AtomicU64::new(0);

    pub static CHAIN_CONTRACT_NS: AtomicU64 = AtomicU64::new(0);
    pub static CHAIN_BLAS_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static CHAIN_BLAS_POINTS: AtomicU64 = AtomicU64::new(0);
    pub static CHAIN_SCALAR_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static CHAIN_SCALAR_POINTS: AtomicU64 = AtomicU64::new(0);

    pub fn add(counter: &AtomicU64, elapsed: std::time::Duration) {
        counter.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        use super::diagnostics::{nanos, record_kernel, KernelDiagnostics};
        let ns = nanos(elapsed);
        let mut delta = KernelDiagnostics::default();
        if std::ptr::eq(counter, &BRANCH_SETUP_NS) {
            delta.setup_ns = ns;
        }
        if std::ptr::eq(counter, &BRANCH_MATMUL_NS) {
            delta.matmul_ns = ns;
        }
        if std::ptr::eq(counter, &BRANCH_ACCUMULATE_NS) {
            delta.accumulate_ns = ns;
        }
        if std::ptr::eq(counter, &BRANCH_CHILD_GATHER_NS) {
            delta.gather_ns = ns;
        }
        record_kernel(delta);
    }

    pub fn inc(counter: &AtomicU64, by: usize) {
        counter.fetch_add(by as u64, Ordering::Relaxed);
        use super::diagnostics::{record_kernel, KernelDiagnostics};
        let n = by as u64;
        let mut delta = KernelDiagnostics::default();
        if std::ptr::eq(counter, &BRANCH_BLAS_CALLS) || std::ptr::eq(counter, &CHAIN_BLAS_CALLS) {
            delta.matmul_calls = n;
        }
        if std::ptr::eq(counter, &BRANCH_SCALAR_POINTS)
            || std::ptr::eq(counter, &CHAIN_SCALAR_POINTS)
        {
            delta.scalar_points = n;
        }
        if std::ptr::eq(counter, &BRANCH_PREPARED_SLICE_HITS) {
            delta.prepared_hits = n;
        }
        if std::ptr::eq(counter, &BRANCH_PREPARED_SLICE_MISSES) {
            delta.prepared_misses = n;
        }
        if std::ptr::eq(counter, &BRANCH_PREPARED_SLICE_REFUSALS) {
            delta.prepared_refusals = n;
        }
        record_kernel(delta);
    }

    pub fn reset_all() {
        for counter in [
            &BRANCH_SETUP_NS,
            &BRANCH_CHILD_DECODE_NS,
            &BRANCH_CHILD_GATHER_NS,
            &BRANCH_CHILD_DECODE_VALUES,
            &BRANCH_CHILD_GATHER_VALUES,
            &BRANCH_SETUP_VALUES,
            &BRANCH_PREPARED_SLICE_HITS,
            &BRANCH_PREPARED_SLICE_MISSES,
            &BRANCH_PREPARED_SLICE_REFUSALS,
            &BRANCH_PREPARED_SLICE_BYTES,
            &BRANCH_MATMUL_NS,
            &BRANCH_ACCUMULATE_NS,
            &BRANCH_BLAS_CALLS,
            &BRANCH_BLAS_POINTS,
            &BRANCH_BLAS_GROUPS,
            &BRANCH_SCALAR_CALLS,
            &BRANCH_SCALAR_POINTS,
            &BRANCH_FAST_AXIS_IS_PARENT,
            &BRANCH_FAST_AXIS_IS_CHILD1,
            &BRANCH_FAST_AXIS_IS_CHILD2,
            &BRANCH_FAST_AXIS_IS_PHYSICAL,
            &CHAIN_CONTRACT_NS,
            &CHAIN_BLAS_CALLS,
            &CHAIN_BLAS_POINTS,
            &CHAIN_SCALAR_CALLS,
            &CHAIN_SCALAR_POINTS,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    /// Renders every counter as one human-readable line.
    pub fn summary() -> String {
        format!(
            "branch: blas_calls={} blas_points={} blas_groups={} child_decode_ns={} child_gather_ns={} \
             child_decode_values={} child_gather_values={} setup_values={} prepared_slice[hits={} misses={} refusals={} bytes={}] setup_ns={} matmul_ns={} accumulate_ns={} \
             scalar_calls={} scalar_points={} fast_axis[parent={} child1={} child2={} physical={}] \
             | chain: blas_calls={} blas_points={} contract_ns={} \
             scalar_calls={} scalar_points={}",
            BRANCH_BLAS_CALLS.load(Ordering::Relaxed),
            BRANCH_BLAS_POINTS.load(Ordering::Relaxed),
            BRANCH_BLAS_GROUPS.load(Ordering::Relaxed),
            BRANCH_CHILD_DECODE_NS.load(Ordering::Relaxed),
            BRANCH_CHILD_GATHER_NS.load(Ordering::Relaxed),
            BRANCH_CHILD_DECODE_VALUES.load(Ordering::Relaxed),
            BRANCH_CHILD_GATHER_VALUES.load(Ordering::Relaxed),
            BRANCH_SETUP_VALUES.load(Ordering::Relaxed),
            BRANCH_PREPARED_SLICE_HITS.load(Ordering::Relaxed),
            BRANCH_PREPARED_SLICE_MISSES.load(Ordering::Relaxed),
            BRANCH_PREPARED_SLICE_REFUSALS.load(Ordering::Relaxed),
            BRANCH_PREPARED_SLICE_BYTES.load(Ordering::Relaxed),
            BRANCH_SETUP_NS.load(Ordering::Relaxed),
            BRANCH_MATMUL_NS.load(Ordering::Relaxed),
            BRANCH_ACCUMULATE_NS.load(Ordering::Relaxed),
            BRANCH_SCALAR_CALLS.load(Ordering::Relaxed),
            BRANCH_SCALAR_POINTS.load(Ordering::Relaxed),
            BRANCH_FAST_AXIS_IS_PARENT.load(Ordering::Relaxed),
            BRANCH_FAST_AXIS_IS_CHILD1.load(Ordering::Relaxed),
            BRANCH_FAST_AXIS_IS_CHILD2.load(Ordering::Relaxed),
            BRANCH_FAST_AXIS_IS_PHYSICAL.load(Ordering::Relaxed),
            CHAIN_BLAS_CALLS.load(Ordering::Relaxed),
            CHAIN_BLAS_POINTS.load(Ordering::Relaxed),
            CHAIN_CONTRACT_NS.load(Ordering::Relaxed),
            CHAIN_SCALAR_CALLS.load(Ordering::Relaxed),
            CHAIN_SCALAR_POINTS.load(Ordering::Relaxed),
        )
    }
}

use anyhow::{bail, ensure, Context, Result};
use num_complex::{Complex32, Complex64};
use tensor4all_core::{
    contract_with_options, contract_with_options_on_session,
    index_key::{FlatIndexer, IndexKey, KeyBuilder},
    index_select_on_session, permute_indices_on_session, select_indices_on_session,
    stack_along_new_index_on_session, AnyScalar, ColMajorArrayRef, ContractionOptions, DynIndex,
    IdxTensor, IndexLike, TensorIndex, TensorLike,
};
use tensor4all_tensorbackend::explicit::Session;
use tensor4all_tensorbackend::{
    mat_mul, mat_mul_owned, BlasMul, Matrix, StorageScalar, TensorElement,
};

#[cfg(feature = "diagnostics")]
use super::diagnostics;
use super::evaluator::value_at;
use super::TreeTN;

type KeyId = usize;
type EnvironmentCache<V> = HashMap<V, StackedMessage>;
type CacheBuildResult<V> = (Vec<ComponentBatch<V>>, EnvironmentCache<V>);
type ParentMap<V> = HashMap<V, Option<V>>;
type DirectedComponentLayouts<V> = HashMap<(V, V), Arc<DirectedComponentLayout<V>>>;

/// Execution route for one cached-evaluation call: the process-global route or
/// a caller-held explicit session.
///
/// Every numerical leaf of the evaluator (grouped GEMMs, retained-index
/// contractions, selection, stacking, gathering and permutation) goes through
/// this one seam, so the evaluation body is shared and only the leaf route
/// changes. The global variant is exactly the compatibility route; the session
/// variant never enters the process-global backend or an eager runtime.
struct CachedEvaluationRoute<'route, 'session> {
    session: Option<&'route mut Session<'session>>,
}

impl<'route, 'session> CachedEvaluationRoute<'route, 'session> {
    fn global() -> Self {
        Self { session: None }
    }

    fn on_session(session: &'route mut Session<'session>) -> Self {
        Self {
            session: Some(session),
        }
    }

    fn mat_mul_owned<T: BlasMul>(
        &mut self,
        left: Matrix<T>,
        right: Matrix<T>,
    ) -> Result<Matrix<T>> {
        match self.session.as_mut() {
            None => mat_mul_owned(left, right).map_err(anyhow::Error::from),
            Some(session) => session.mat_mul(&left, &right).map_err(anyhow::Error::from),
        }
    }

    fn mat_mul<T: BlasMul>(&mut self, left: &Matrix<T>, right: &Matrix<T>) -> Result<Matrix<T>> {
        match self.session.as_mut() {
            None => mat_mul(left, right).map_err(anyhow::Error::from),
            Some(session) => session.mat_mul(left, right).map_err(anyhow::Error::from),
        }
    }

    fn contract_nary(
        &mut self,
        operands: &[&IdxTensor],
        options: ContractionOptions<'_>,
    ) -> Result<IdxTensor> {
        match self.session.as_mut() {
            None => contract_with_options(operands, options).map_err(anyhow::Error::from),
            Some(session) => contract_with_options_on_session(session, operands, options)
                .map_err(anyhow::Error::from),
        }
    }

    fn select_indices(
        &mut self,
        tensor: &IdxTensor,
        selected_indices: &[DynIndex],
        positions: &[usize],
    ) -> Result<IdxTensor> {
        match self.session.as_mut() {
            None => tensor
                .select_indices(selected_indices, positions)
                .map_err(anyhow::Error::from),
            Some(session) => {
                select_indices_on_session(session, tensor, selected_indices, positions)
                    .map_err(anyhow::Error::from)
            }
        }
    }

    fn stack_along_new_index(
        &mut self,
        tensors: &[&IdxTensor],
        new_index: DynIndex,
        axis: isize,
    ) -> Result<IdxTensor> {
        match self.session.as_mut() {
            None => IdxTensor::stack_along_new_index(tensors, new_index, axis)
                .map_err(anyhow::Error::from),
            Some(session) => stack_along_new_index_on_session(session, tensors, new_index, axis)
                .map_err(anyhow::Error::from),
        }
    }

    fn index_select(
        &mut self,
        tensor: &IdxTensor,
        source_index: &DynIndex,
        target_index: DynIndex,
        positions: &[usize],
    ) -> Result<IdxTensor> {
        match self.session.as_mut() {
            None => tensor
                .index_select(source_index, target_index, positions)
                .map_err(anyhow::Error::from),
            Some(session) => {
                index_select_on_session(session, tensor, source_index, target_index, positions)
                    .map_err(anyhow::Error::from)
            }
        }
    }

    fn permute_indices(
        &mut self,
        tensor: &IdxTensor,
        new_indices: &[DynIndex],
    ) -> Result<IdxTensor> {
        match self.session.as_mut() {
            None => tensor
                .permute_indices(new_indices)
                .map_err(anyhow::Error::from),
            Some(session) => permute_indices_on_session(session, tensor, new_indices)
                .map_err(anyhow::Error::from),
        }
    }

    fn tensor_from_cached_values(
        &mut self,
        indices: Vec<DynIndex>,
        values: Vec<CachedScalar>,
    ) -> Result<IdxTensor> {
        match self.session.as_mut() {
            None => tensor_from_cached_values(indices, values),
            Some(_) => tensor_from_cached_values_compact(indices, values),
        }
    }
}

/// Minimum scalar multiply count before the backend setup cost is amortized
/// by the grouped chain kernel. Smaller contractions keep the existing scalar
/// loop, which is faster for the tiny messages common at low bond dimension.
const CHAIN_BLAS_WORK_THRESHOLD: usize = 4096;
/// Minimum number of columns in every physical-value group before paying for
/// a backend matrix multiplication. This avoids turning the common one- or
/// two-point floating-zone callback into a collection of matrix-vector calls.
const CHAIN_BLAS_MIN_GROUP_POINTS: usize = 4;

#[derive(Clone, Copy, Debug)]
struct ChainContractionSpec {
    strides: [usize; 3],
    physical_axis: usize,
    parent_axis: usize,
    child_axis: usize,
    parent_dim: usize,
    child_dim: usize,
}

/// Minimum scalar multiply count before the backend setup cost is amortized
/// by the grouped branch kernel. Unlike the chain kernel, there is no
/// separate minimum-group-points gate: a branch step's per-point work is
/// O(parent_dim * child_dim_1 * child_dim_2), an extra bond-dimension factor
/// over the chain kernel's O(parent_dim * child_dim), so at realistic bond
/// dimensions `scalar_work` alone already justifies BLAS even for a single
/// point per group (see `grouped_branch_message_contraction`).
const BRANCH_BLAS_WORK_THRESHOLD: usize = 4096;

/// Largest prepared slice the arbitrary-degree branch kernel will materialize
/// for one physical group, in scalars.
///
/// That slice is `parent_dim * prod(child_dims)`, which grows as `chi^z` in
/// the node's coordination number `z` -- the same exponent a tree contraction
/// carries in general (Tindall, Stoudenmire and Levy, arXiv:2410.03572v3,
/// Sec. 2, the paragraph after Eq. (1): a TTN contraction "can be done in
/// `O(nL chi^z)` time, where `z` is the maximum co-ordination number"). The
/// arithmetic is therefore unavoidable for such a node, but *buffering* it is
/// not, so beyond this size the flat kernel runs instead and allocates
/// nothing. At 4 Mi scalars this is 32 MiB of `f64` or 64 MiB of `Complex64`.
const MULTI_BRANCH_MAX_PREPARED_ELEMENTS: usize = 1 << 22;

// [AI Supplied] Test-only A/B seam for independently checking whether the
// existing generic `contract_with_options` path can replace the specialized
// raw message kernels without relying on historical worklog claims.
#[cfg(test)]
fn raw_message_kernels_disabled_for_test() -> bool {
    std::env::var_os("T4A_TREETN_DISABLE_RAW_MESSAGES").is_some()
}

#[derive(Clone, Copy, Debug)]
struct BranchContractionSpec {
    strides: [usize; 4],
    physical_axis: usize,
    parent_axis: usize,
    child_axis_1: usize,
    child_axis_2: usize,
    parent_dim: usize,
    child_dim_1: usize,
    child_dim_2: usize,
}

struct BranchMessageBatch<'a, T> {
    spec: BranchContractionSpec,
    raw: &'a [T],
    physical_values: &'a [usize],
    child1_columns: &'a [T],
    child2_columns: &'a [T],
}

#[derive(Clone, Debug)]
struct RawCenterComponent<T> {
    axis: usize,
    dim: usize,
    point_to_assignment: Vec<usize>,
    values: Vec<T>,
}

#[derive(Clone, Debug)]
struct SiteEntry {
    index: DynIndex,
    input_position: usize,
    local_axis: usize,
}

#[derive(Clone, Debug)]
struct MessageCacheLayout {
    input_positions: Vec<usize>,
    indexer: FlatIndexer,
}

/// Immutable metadata for one directed tree component. The component is the
/// side containing `from` after removing the edge `(from, to)`; `child_nodes`
/// records the checked append order used to compose its key.
#[derive(Clone, Debug)]
struct DirectedComponentLayout<V> {
    layout: MessageCacheLayout,
    child_nodes: Vec<V>,
}

#[derive(Clone, Debug)]
struct EvaluatorLayout<V> {
    entries_by_node: HashMap<V, Vec<SiteEntry>>,
    local_layouts_by_node: HashMap<V, MessageCacheLayout>,
    n_indices: usize,
}

#[derive(Default)]
struct KeyInterner<T>
where
    T: Clone + Eq + Hash,
{
    ids: HashMap<T, KeyId>,
}

impl<T> KeyInterner<T>
where
    T: Clone + Eq + Hash,
{
    fn intern(&mut self, key: T) -> KeyId {
        let next = self.ids.len();
        *self.ids.entry(key).or_insert(next)
    }
}

#[derive(Clone, Debug)]
struct AssignmentBatch {
    point_to_assignment: Vec<usize>,
    first_points: Vec<usize>,
    keys: Vec<IndexKey>,
}

#[derive(Clone, Debug)]
struct ComponentBatch<V> {
    neighbor: V,
    point_to_assignment: Vec<usize>,
}

struct EdgeCutAssembly<'a, V> {
    values: ColMajorArrayRef<'a, usize>,
    cut_batch: &'a ComponentBatch<V>,
    cut_environment: &'a StackedMessage,
    center_assignment_batch: &'a AssignmentBatch,
    center_message: &'a StackedMessage,
    bond_dim: usize,
}

#[derive(Clone, Debug)]
struct StackedMessage {
    assignment_index: DynIndex,
    tensor: Option<IdxTensor>,
    raw_values: Option<Vec<CachedScalar>>,
}

#[derive(Clone, Copy, Debug)]
enum CachedScalar {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}

impl CachedScalar {
    fn kind(&self) -> ScalarKind {
        match self {
            Self::F32(_) => ScalarKind::F32,
            Self::F64(_) => ScalarKind::F64,
            Self::C32(_) => ScalarKind::C32,
            Self::C64(_) => ScalarKind::C64,
        }
    }

    fn into_any(self) -> AnyScalar {
        match self {
            Self::F32(value) => AnyScalar::from_value(value),
            Self::F64(value) => AnyScalar::from_value(value),
            Self::C32(value) => AnyScalar::from_value(value),
            Self::C64(value) => AnyScalar::from_value(value),
        }
    }

    /// Name of the dtype this value was stored under.
    fn stored_dtype_name(self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::C32(_) => "c32",
            Self::C64(_) => "c64",
        }
    }

    /// Widens a real value to `f64`.
    ///
    /// Returns `None` for a complex value: dropping an imaginary part is the
    /// one conversion the dynamic wrapper also refuses.
    fn as_real(self) -> Option<f64> {
        match self {
            Self::F32(value) => Some(f64::from(value)),
            Self::F64(value) => Some(value),
            Self::C32(_) | Self::C64(_) => None,
        }
    }

    /// Widens any value to `Complex64`; a real value gains a zero imaginary
    /// part, exactly as the dynamic wrapper's complex accessor does.
    fn as_complex(self) -> Complex64 {
        match self {
            Self::F32(value) => Complex64::new(f64::from(value), 0.0),
            Self::F64(value) => Complex64::new(value, 0.0),
            Self::C32(value) => Complex64::new(f64::from(value.re), f64::from(value.im)),
            Self::C64(value) => value,
        }
    }
}

/// Canonical dtype name for a typed batch request.
///
/// The four supported element types get the same short names the evaluator
/// uses for stored payloads; anything else falls back to its Rust type name so
/// the error still identifies the request.
fn requested_dtype_name<T: TensorElement>() -> &'static str {
    let requested = TypeId::of::<T>();
    if requested == TypeId::of::<f32>() {
        "f32"
    } else if requested == TypeId::of::<f64>() {
        "f64"
    } else if requested == TypeId::of::<Complex32>() {
        "c32"
    } else if requested == TypeId::of::<Complex64>() {
        "c64"
    } else {
        std::any::type_name::<T>()
    }
}

/// Converts a batch of stored evaluator scalars into the requested element
/// type without constructing one dynamic rank-zero tensor per result.
///
/// The conversion policy is the dynamic wrapper's: precision changes within a
/// kind are permitted, a real payload widens into a complex request, and a
/// complex payload is never silently narrowed into a real request.
fn cached_values_into_typed<T: TensorElement>(
    values: Vec<CachedScalar>,
) -> std::result::Result<Vec<T>, EvaluatedScalarKindMismatch> {
    let requested = requested_dtype_name::<T>();
    let mismatch = |value: CachedScalar| EvaluatedScalarKindMismatch {
        stored: value.stored_dtype_name(),
        requested,
    };
    let requested_id = TypeId::of::<T>();
    let typed: Box<dyn Any> = if requested_id == TypeId::of::<f64>() {
        let mut decoded = Vec::with_capacity(values.len());
        for value in values {
            decoded.push(value.as_real().ok_or_else(|| mismatch(value))?);
        }
        Box::new(decoded)
    } else if requested_id == TypeId::of::<f32>() {
        let mut decoded = Vec::with_capacity(values.len());
        for value in values {
            decoded.push(value.as_real().ok_or_else(|| mismatch(value))? as f32);
        }
        Box::new(decoded)
    } else if requested_id == TypeId::of::<Complex64>() {
        Box::new(
            values
                .into_iter()
                .map(CachedScalar::as_complex)
                .collect::<Vec<Complex64>>(),
        )
    } else if requested_id == TypeId::of::<Complex32>() {
        Box::new(
            values
                .into_iter()
                .map(|value| {
                    let value = value.as_complex();
                    Complex32::new(value.re as f32, value.im as f32)
                })
                .collect::<Vec<Complex32>>(),
        )
    } else {
        let stored = values
            .first()
            .map_or("f64", |value| value.stored_dtype_name());
        return Err(EvaluatedScalarKindMismatch { stored, requested });
    };
    typed
        .downcast::<Vec<T>>()
        .map(|values| *values)
        .map_err(|_| EvaluatedScalarKindMismatch {
            stored: "f64",
            requested,
        })
}

/// The requested element type of a typed batch cannot represent the values the
/// evaluated `TreeTN` stores.
///
/// This is the one dtype rule
/// [`TreeTNCachedEvaluator::evaluate_batched_typed`] enforces: a complex
/// payload is never silently narrowed into a real request. It is wrapped in a
/// [`TreeTNOperationError`], so a caller that needs to distinguish a dtype
/// mismatch from a shape or backend failure can downcast the error source.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
/// use tensor4all_treetn::{
///     CachedEvaluatorOptions, EvaluatedScalarKindMismatch, EvaluationHint, TreeTN,
///     TreeTNCachedEvaluator,
/// };
///
/// let s = DynIndex::new_dyn(2);
/// let tensor = IdxTensor::from_dense(
///     vec![s.clone()],
///     vec![num_complex::Complex64::new(1.0, 2.0), num_complex::Complex64::new(3.0, 4.0)],
/// )?;
/// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
/// let values = [0usize, 1usize];
/// let mut evaluator = TreeTNCachedEvaluator::new(
///     &tree,
///     &[s],
///     CachedEvaluatorOptions { center: Some(0), ..Default::default() },
/// )?;
/// let points = ColMajorArrayRef::new(&values, &[1, 2])?;
///
/// // A complex tree cannot answer a real request.
/// let error = evaluator
///     .evaluate_batched_typed::<f64>(points, EvaluationHint::default())
///     .unwrap_err();
/// let mismatch = error
///     .source
///     .downcast_ref::<EvaluatedScalarKindMismatch>()
///     .expect("dtype mismatch");
/// assert_eq!(mismatch.stored, "c64");
/// assert_eq!(mismatch.requested, "f64");
///
/// // The same batch answers a complex request exactly.
/// let complex = evaluator
///     .evaluate_batched_typed::<num_complex::Complex64>(points, EvaluationHint::default())?;
/// assert_eq!(complex, vec![
///     num_complex::Complex64::new(1.0, 2.0),
///     num_complex::Complex64::new(3.0, 4.0),
/// ]);
/// # Ok::<(), anyhow::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("cannot decode a {stored} TreeTN value as {requested}")]
pub struct EvaluatedScalarKindMismatch {
    /// Short dtype name of the payload the evaluated `TreeTN` stores:
    /// `"f32"`, `"f64"`, `"c32"`, or `"c64"`.
    pub stored: &'static str,
    /// Short dtype name requested by the typed batch call.
    pub requested: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ScalarKind {
    F32,
    F64,
    C32,
    C64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CachedEvaluationStats {
    subtree_environment_count: usize,
    directed_message_count: usize,
    batched_message_contract_count: usize,
    batched_center_contract_count: usize,
    message_cache_hits: usize,
    message_cache_misses: usize,
    message_cache_key_count: usize,
    message_cache_logical_bytes: usize,
    message_cache_owned_bytes_estimate: usize,
    /// [AI Supplied] Test-only work count for the warm edge-cut final dot
    /// product. Kept in the private stats record so the complexity gate does
    /// not alter the public evaluator API.
    warm_edge_cut_assembly_visits: usize,
}

#[derive(Clone, Debug)]
struct RootedMessagePlan<V> {
    children: HashMap<V, Vec<V>>,
    postorder: Vec<V>,
    parent: ParentMap<V>,
}

#[derive(Debug)]
struct ComponentCostIndex<V> {
    neighbors: HashMap<V, Vec<V>>,
    directed_counts: HashMap<(V, V), usize>,
    node_costs: Option<HashMap<V, usize>>,
}

impl<V> ComponentCostIndex<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    /// # Errors
    ///
    /// Returns an error when `build_layout` rejects `tree`/`indices`, or when
    /// `values` fails the shape or per-entry value checks.
    ///
    fn new(
        tree: &TreeTN<IdxTensor, V>,
        indices: &[DynIndex],
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Self> {
        let layout = build_layout(tree, indices)?;
        Self::from_layout(tree, &layout, values)
    }

    fn from_layout(
        tree: &TreeTN<IdxTensor, V>,
        layout: &EvaluatorLayout<V>,
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Self> {
        validate_values_shape(values, layout.n_indices, "ComponentCostIndex::new")?;
        let n_points = values.shape()[1];

        let neighbors = sorted_neighbors(tree);
        if neighbors.is_empty() {
            return Ok(Self {
                neighbors,
                directed_counts: HashMap::new(),
                node_costs: None,
            });
        }

        let mut node_names: Vec<V> = neighbors.keys().cloned().collect();
        node_names.sort();
        let root = node_names[0].clone();

        let (parent, order) = rooted_tree(&neighbors, &root)?;
        let mut local_interner = KeyInterner::<Vec<usize>>::default();
        let mut local_keys: HashMap<V, Vec<KeyId>> = HashMap::with_capacity(node_names.len());
        for node in &node_names {
            let entries = layout
                .entries_by_node
                .get(node)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let mut keys = Vec::with_capacity(n_points);
            for point in 0..n_points {
                let key = entries
                    .iter()
                    .map(|entry| {
                        value_at(
                            values,
                            entry.input_position,
                            point,
                            "ComponentCostIndex::new",
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                validate_entry_values(entries, &key, "ComponentCostIndex::new")?;
                keys.push(local_interner.intern(key));
            }
            local_keys.insert(node.clone(), keys);
        }

        let mut component_interner = KeyInterner::<Vec<KeyId>>::default();
        let mut directed_keys: HashMap<V, HashMap<V, Vec<KeyId>>> =
            HashMap::with_capacity(neighbors.len());

        for node in order.iter().rev() {
            let Some(parent_node) = parent.get(node).and_then(Clone::clone) else {
                continue;
            };
            let node_neighbors = neighbors.get(node).ok_or_else(|| {
                anyhow::anyhow!("ComponentCostIndex::new: missing neighbors for {:?}", node)
            })?;
            let incoming = node_neighbors
                .iter()
                .filter(|neighbor| *neighbor != &parent_node)
                .map(|neighbor| {
                    directed_keys
                        .get(neighbor)
                        .and_then(|by_target| by_target.get(node))
                        .with_context(|| {
                            format!(
                                "ComponentCostIndex::new: missing child key {:?}->{:?}",
                                neighbor, node
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let node_local_keys = local_keys.get(node).ok_or_else(|| {
                anyhow::anyhow!("ComponentCostIndex::new: missing local keys for {:?}", node)
            })?;
            let keys = intern_component_keys(
                node_local_keys,
                &incoming,
                n_points,
                &mut component_interner,
            );
            directed_keys
                .entry(node.clone())
                .or_default()
                .insert(parent_node, keys);
        }

        for node in &order {
            let node_neighbors = neighbors.get(node).ok_or_else(|| {
                anyhow::anyhow!("ComponentCostIndex::new: missing neighbors for {:?}", node)
            })?;
            for child in node_neighbors.iter().filter(|neighbor| {
                parent.get(*neighbor).and_then(Clone::clone) == Some(node.clone())
            }) {
                let incoming = node_neighbors
                    .iter()
                    .filter(|neighbor| *neighbor != child)
                    .map(|neighbor| {
                        directed_keys
                            .get(neighbor)
                            .and_then(|by_target| by_target.get(node))
                            .with_context(|| {
                                format!(
                                    "ComponentCostIndex::new: missing incoming key {:?}->{:?}",
                                    neighbor, node
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let node_local_keys = local_keys.get(node).ok_or_else(|| {
                    anyhow::anyhow!("ComponentCostIndex::new: missing local keys for {:?}", node)
                })?;
                let keys = intern_component_keys(
                    node_local_keys,
                    &incoming,
                    n_points,
                    &mut component_interner,
                );
                directed_keys
                    .entry(node.clone())
                    .or_default()
                    .insert(child.clone(), keys);
            }
        }

        let mut directed_counts = HashMap::with_capacity(tree.edge_count() * 2);
        for (source, targets) in directed_keys {
            for (target, keys) in targets {
                let count = keys.into_iter().collect::<HashSet<_>>().len();
                directed_counts.insert((source.clone(), target), count);
            }
        }

        Ok(Self {
            neighbors,
            directed_counts,
            node_costs: None,
        })
    }

    fn all_nodes(&self) -> Vec<V> {
        let mut nodes: Vec<V> = self.neighbors.keys().cloned().collect();
        nodes.sort();
        nodes
    }

    fn component_count(&self, edge: &(V, V)) -> Option<usize> {
        self.directed_counts.get(edge).copied()
    }

    fn center_cost(&self, center: &V) -> Result<usize> {
        if let Some(node_costs) = &self.node_costs {
            return node_costs.get(center).copied().ok_or_else(|| {
                anyhow::anyhow!("center {:?} is not present in cost index", center)
            });
        }
        let neighbors = self
            .neighbors
            .get(center)
            .ok_or_else(|| anyhow::anyhow!("center {:?} is not present in cost index", center))?;
        neighbors.iter().try_fold(0usize, |acc, neighbor| {
            self.component_count(&(neighbor.clone(), center.clone()))
                .map(|count| acc + count)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing component cost for directed edge {:?}->{:?}",
                        neighbor,
                        center
                    )
                })
        })
    }

    #[cfg(test)]
    fn from_parts_for_test(
        mut neighbors: HashMap<V, Vec<V>>,
        node_costs: HashMap<V, usize>,
    ) -> Self {
        for neighbor_list in neighbors.values_mut() {
            neighbor_list.sort();
        }
        Self {
            neighbors,
            directed_counts: HashMap::new(),
            node_costs: Some(node_costs),
        }
    }
}

fn intern_component_keys(
    local_keys: &[KeyId],
    incoming: &[&Vec<KeyId>],
    n_points: usize,
    interner: &mut KeyInterner<Vec<KeyId>>,
) -> Vec<KeyId> {
    let mut keys = Vec::with_capacity(n_points);
    for point in 0..n_points {
        let mut tuple = Vec::with_capacity(1 + incoming.len());
        tuple.push(local_keys[point]);
        for incoming_keys in incoming {
            tuple.push(incoming_keys[point]);
        }
        keys.push(interner.intern(tuple));
    }
    keys
}

impl<V> RootedMessagePlan<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    /// # Errors
    ///
    /// Returns an error when `center` is not a node of `tree`.
    ///
    fn new(tree: &TreeTN<IdxTensor, V>, center: &V) -> Result<Self> {
        Self::from_neighbors(&sorted_neighbors(tree), center)
    }

    /// Builds the rooted plan from an already-sorted neighbour table.
    ///
    /// The table is pure topology, so a caller holding one immutable copy
    /// does not rebuild it for every centre.
    fn from_neighbors(neighbors: &HashMap<V, Vec<V>>, center: &V) -> Result<Self> {
        let (parent, order) = rooted_tree(neighbors, center)?;

        let mut children = HashMap::<V, Vec<V>>::new();
        for node in neighbors.keys() {
            children.insert(node.clone(), Vec::new());
        }
        for (node, parent_node) in &parent {
            if let Some(parent_node) = parent_node {
                children
                    .get_mut(parent_node)
                    .ok_or_else(|| anyhow::anyhow!("missing rooted parent {:?}", parent_node))?
                    .push(node.clone());
            }
        }
        for node_children in children.values_mut() {
            node_children.sort();
        }

        let postorder = order
            .into_iter()
            .rev()
            .filter(|node| node != center)
            .collect::<Vec<_>>();

        Ok(Self {
            children,
            postorder,
            parent,
        })
    }
}

/// Options controlling cached batch evaluation for [`TreeTN`].
///
/// Use this to pin the contraction center or to configure the greedy automatic
/// center search. When in doubt, leave all fields at their defaults.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::CachedEvaluatorOptions;
///
/// let options = CachedEvaluatorOptions::<usize>::default();
/// assert!(options.center.is_none());
/// assert!(options.initial_centers.is_empty());
/// assert!(options.max_greedy_steps_per_start.is_none());
/// assert_eq!(options.max_batch_points, None);
/// assert_eq!(options.message_cache_max_bytes, 8 * 1024 * 1024);
/// assert_eq!(options.branch_slice_cache_max_bytes, 32 * 1024 * 1024);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedEvaluatorOptions<V> {
    /// Operand label for opt-in per-node diagnostics, e.g. `"input:0"`.
    /// Defaults to `"tree"`; assign distinct labels to distinct evaluators
    /// sharing a diagnostics window. Has no effect without the feature.
    #[cfg(feature = "diagnostics")]
    pub diagnostic_namespace: String,
    /// Fixed center node for evaluation.
    ///
    /// When set, greedy center search is skipped. Use this when the caller
    /// already knows where repeated batch structure is concentrated.
    pub center: Option<V>,
    /// Candidate starting centers for greedy automatic center search.
    ///
    /// Empty means all nodes are eligible as starts. Supplying a short list can
    /// reduce center-search overhead for large trees.
    pub initial_centers: Vec<V>,
    /// Maximum number of greedy moves from each initial center.
    ///
    /// `None` means no explicit step limit; the search stops at a local minimum.
    pub max_greedy_steps_per_start: Option<usize>,
    /// Explicit maximum points processed together in one internal batch.
    ///
    /// `None` (recommended) chooses at most 256 points, reducing the limit
    /// for large local tensors using a 32 MiB local-buffer estimate. This is
    /// a sizing heuristic, not a bound on total RSS. `Some(n)` must be positive
    /// and fixes the limit to `n`; `Some(usize::MAX)` processes the whole input
    /// together and can require very large buffers. Input and returned scalars
    /// still occupy O(number of points) memory.
    pub max_batch_points: Option<usize>,
    /// Maximum logical payload bytes retained by each directed message cache.
    ///
    /// A value of `0` disables retention while preserving the same evaluation
    /// results. The default is 8 MiB per directed edge. Use `usize::MAX` only
    /// when unbounded retention is intentional; the budget excludes key/map
    /// metadata and spare allocation capacity.
    pub message_cache_max_bytes: usize,
    /// Maximum logical payload bytes retained by the evaluator's prepared
    /// branch physical-slice cache.
    ///
    /// A value of `0` disables retention while preserving the same numerical
    /// result: each branch slice is prepared for the current group and then
    /// released. The default is 32 MiB per scalar kind. Use `usize::MAX` only
    /// when retaining every visited physical slice is intentional.
    /// Setting a finite value bounds retained slice payloads across directed
    /// orientations and scalar kinds; cache metadata is not included in this
    /// logical payload budget.
    pub branch_slice_cache_max_bytes: usize,
}

impl<V> Default for CachedEvaluatorOptions<V> {
    fn default() -> Self {
        Self {
            #[cfg(feature = "diagnostics")]
            diagnostic_namespace: "tree".to_owned(),
            center: None,
            initial_centers: Vec::new(),
            max_greedy_steps_per_start: None,
            max_batch_points: None,
            message_cache_max_bytes: 8 * 1024 * 1024,
            branch_slice_cache_max_bytes: 32 * 1024 * 1024,
        }
    }
}

/// Per-call knowledge a caller can supply to `evaluate_batched_with_hint`.
///
/// Non-exhaustive so later hints can be added without breaking callers.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::EvaluationHint;
///
/// assert!(EvaluationHint::<usize>::default().center.is_none());
/// assert_eq!(EvaluationHint::around(3usize).center, Some(3));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EvaluationHint<V> {
    /// The node this batch varies around, if the caller knows it.
    ///
    /// `None` keeps the evaluator's existing centre, chosen from options or by
    /// greedy search on the first batch.
    pub center: Option<V>,
}

impl<V> Default for EvaluationHint<V> {
    /// An empty hint, which leaves the evaluator's centre selection unchanged.
    ///
    /// Written out rather than derived: `derive(Default)` would demand
    /// `V: Default`, but a hint that names no node needs nothing of `V`.
    fn default() -> Self {
        Self { center: None }
    }
}

impl<V> EvaluationHint<V> {
    /// A hint naming the node this batch varies around.
    pub fn around(center: V) -> Self {
        Self {
            center: Some(center),
        }
    }
}

/// Result of greedy center search for cached TreeTN evaluation.
///
/// The result records the selected node, its estimated cost, and the path taken
/// by greedy descent from the chosen start.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::CenterSearchResult;
///
/// let result = CenterSearchResult {
///     center: 2_usize,
///     cost: 7,
///     path: vec![0, 1, 2],
/// };
/// assert_eq!(result.center, 2);
/// assert_eq!(result.cost, 7);
/// assert_eq!(result.path.last(), Some(&2));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CenterSearchResult<V> {
    /// Selected center node.
    pub center: V,
    /// Estimated cache cost at `center`.
    pub cost: usize,
    /// Greedy descent path that produced `center`.
    pub path: Vec<V>,
}

/// Greedy local search for TreeTN cached-evaluation centers.
///
/// This type is intentionally separate from [`TreeTNCachedEvaluator`] so future
/// center-selection algorithms can share the same cost model.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::GreedyCenterSearch;
///
/// let search = GreedyCenterSearch::<usize>::default();
/// assert!(search.max_steps().is_none());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GreedyCenterSearch<V> {
    max_steps: Option<usize>,
    _marker: std::marker::PhantomData<V>,
}

impl<V> Default for GreedyCenterSearch<V> {
    fn default() -> Self {
        Self {
            max_steps: None,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<V> GreedyCenterSearch<V> {
    /// Creates a greedy center search with an optional step limit.
    ///
    /// `max_steps` limits the number of edge moves from each start. `None`
    /// searches until no neighbor has lower cost.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::GreedyCenterSearch;
    ///
    /// let search = GreedyCenterSearch::<usize>::with_max_steps(Some(3));
    /// assert_eq!(search.max_steps(), Some(3));
    /// ```
    pub fn with_max_steps(max_steps: Option<usize>) -> Self {
        Self {
            max_steps,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the optional greedy-step limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::GreedyCenterSearch;
    ///
    /// let search = GreedyCenterSearch::<usize>::with_max_steps(None);
    /// assert_eq!(search.max_steps(), None);
    /// ```
    pub fn max_steps(&self) -> Option<usize> {
        self.max_steps
    }
}

impl<V> GreedyCenterSearch<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    fn search(
        &self,
        cost_index: &ComponentCostIndex<V>,
        starts: &[V],
    ) -> Result<CenterSearchResult<V>> {
        let owned_starts;
        let starts = if starts.is_empty() {
            owned_starts = cost_index.all_nodes();
            owned_starts.as_slice()
        } else {
            starts
        };
        if starts.is_empty() {
            bail!("GreedyCenterSearch::search: cost index has no nodes");
        }

        let mut best: Option<CenterSearchResult<V>> = None;
        for start in starts {
            if !cost_index.neighbors.contains_key(start) {
                bail!(
                    "GreedyCenterSearch::search: initial center {:?} is not present in TreeTN",
                    start
                );
            }
            let result = self.descend_from(cost_index, start)?;
            match &best {
                None => best = Some(result),
                Some(current) => {
                    if (result.cost, result.center.clone()) < (current.cost, current.center.clone())
                    {
                        best = Some(result);
                    }
                }
            }
        }

        best.ok_or_else(|| anyhow::anyhow!("GreedyCenterSearch::search: no start centers"))
    }

    fn descend_from(
        &self,
        cost_index: &ComponentCostIndex<V>,
        start: &V,
    ) -> Result<CenterSearchResult<V>> {
        let mut center = start.clone();
        let mut cost = cost_index.center_cost(&center)?;
        let mut path = vec![center.clone()];
        let mut steps = 0usize;

        loop {
            if self.max_steps.is_some_and(|max_steps| steps >= max_steps) {
                break;
            }

            let mut candidates = Vec::new();
            let neighbors = cost_index.neighbors.get(&center).ok_or_else(|| {
                anyhow::anyhow!(
                    "GreedyCenterSearch::descend_from: center {:?} is not present in cost index",
                    center
                )
            })?;
            for neighbor in neighbors {
                candidates.push((cost_index.center_cost(neighbor)?, neighbor.clone()));
            }
            candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let Some((next_cost, next_center)) = candidates.into_iter().next() else {
                break;
            };
            if next_cost >= cost {
                break;
            }

            center = next_center;
            cost = next_cost;
            path.push(center.clone());
            steps += 1;
        }

        Ok(CenterSearchResult { center, cost, path })
    }
}

/// Immutable evaluation plan shared by [`TreeTNCachedEvaluator`] instances.
///
/// A plan holds everything that depends only on a tree's *topology* and its
/// physical index list: which physical index sits on which node, the
/// column-major key layout of every directed component, the sorted node and
/// neighbour tables, and the rooted traversal plans discovered so far. None of
/// it depends on the tensors' numerical values, bond dimensions, or dtype, so
/// a caller that repeatedly rebuilds an evaluator for the *same* tree shape
/// with *changed* tensors -- the TreeACI global guard rebuilding its output
/// evaluator after pivot injection -- can keep the plan and rebuild only the
/// numerical message caches.
///
/// [`TreeTNCachedEvaluator::new`] builds a private plan; use
/// [`TreeTNCachedEvaluator::with_plan`] to share one. Cloning a plan is an
/// `Arc` clone, and a shared plan is safe to use from several evaluators.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
/// use tensor4all_treetn::{
///     CachedEvaluatorOptions, CachedEvaluatorPlan, EvaluationHint, TreeTN,
///     TreeTNCachedEvaluator,
/// };
///
/// let a = DynIndex::new_dyn(2);
/// let b = DynIndex::new_dyn(2);
/// let bond = DynIndex::new_dyn(1);
/// let build = |scale: f64| -> anyhow::Result<TreeTN<IdxTensor, usize>> {
///     let left = IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![scale, 2.0 * scale])?;
///     let right = IdxTensor::from_dense(vec![bond.clone(), b.clone()], vec![1.0_f64, 10.0])?;
///     Ok(TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?)
/// };
///
/// let first = build(1.0)?;
/// let plan = CachedEvaluatorPlan::new(&first, &[a.clone(), b.clone()])?;
///
/// let values = [0usize, 0, 0, 1];
/// let points = ColMajorArrayRef::new(&values, &[2, 2])?;
/// let mut evaluator =
///     TreeTNCachedEvaluator::with_plan(&first, &plan, CachedEvaluatorOptions::default())?;
/// assert_eq!(
///     evaluator.evaluate_batched_typed::<f64>(points, EvaluationHint::default())?,
///     vec![1.0, 10.0]
/// );
///
/// // The same plan serves a tree with the same topology and different values.
/// let second = build(3.0)?;
/// let mut reused =
///     TreeTNCachedEvaluator::with_plan(&second, &plan, CachedEvaluatorOptions::default())?;
/// assert_eq!(
///     reused.evaluate_batched_typed::<f64>(points, EvaluationHint::default())?,
///     vec![3.0, 30.0]
/// );
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct CachedEvaluatorPlan<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    inner: Arc<CachedEvaluatorPlanInner<V>>,
}

impl<V> Clone for CachedEvaluatorPlan<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<V> Debug for CachedEvaluatorPlan<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CachedEvaluatorPlan")
            .field("nodes", &self.inner.sorted_node_names.len())
            .field("indices", &self.inner.indices.len())
            .field(
                "directed_components",
                &self.inner.directed_component_layouts.len(),
            )
            .finish()
    }
}

struct CachedEvaluatorPlanInner<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    indices: Vec<DynIndex>,
    layout: EvaluatorLayout<V>,
    directed_component_layouts: DirectedComponentLayouts<V>,
    /// Sorted neighbour table, built once instead of per batch call.
    neighbors: HashMap<V, Vec<V>>,
    /// Node names in the deterministic order the assignment builder walks.
    sorted_node_names: Vec<V>,
    /// Rooted traversal plans are pure topology, so they are memoized here
    /// and shared by every evaluator holding this plan.
    rooted_plans: std::sync::Mutex<HashMap<V, Arc<RootedMessagePlan<V>>>>,
}

impl<V> CachedEvaluatorPlan<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    /// Builds the immutable plan for `tree` and the requested physical indices.
    ///
    /// # Arguments
    ///
    /// * `tree` - The tree whose topology and physical index placement the
    ///   plan describes. Only its shape is read; tensor values, bond
    ///   dimensions, and dtype are not part of the plan.
    /// * `indices` - Every physical index this evaluator answers, in the
    ///   column order later batches use. Duplicates are rejected.
    ///
    /// # Returns
    ///
    /// A cheaply cloneable, shareable plan. Clones share one allocation.
    ///
    /// # Errors
    ///
    /// Returns a [`TreeTNOperationError`] when `tree` has no nodes, when
    /// `indices` does not cover exactly the tree's site indices, when an index
    /// is duplicated or is carried by no node, or when a component key layout
    /// needs more bits than this platform's key width and would overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorPlan, TreeTN};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![7])?;
    /// let plan = CachedEvaluatorPlan::new(&tree, &[s.clone()])?;
    /// assert_eq!(plan.indices(), &[s]);
    /// assert_eq!(plan.node_count(), 1);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        tree: &TreeTN<IdxTensor, V>,
        indices: &[DynIndex],
    ) -> std::result::Result<Self, TreeTNOperationError> {
        let layout = build_layout(tree, indices)?;
        let directed_component_layouts = build_directed_component_layouts(tree, &layout)?;
        let neighbors = sorted_neighbors(tree);
        let mut sorted_node_names = tree.node_names();
        sorted_node_names.sort();
        Ok(Self {
            inner: Arc::new(CachedEvaluatorPlanInner {
                indices: indices.to_vec(),
                layout,
                directed_component_layouts,
                neighbors,
                sorted_node_names,
                rooted_plans: std::sync::Mutex::new(HashMap::new()),
            }),
        })
    }

    /// The physical indices this plan was built for, in batch column order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorPlan, TreeTN};
    ///
    /// let s = DynIndex::new_dyn(3);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0, 3.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let plan = CachedEvaluatorPlan::new(&tree, &[s.clone()])?;
    /// assert_eq!(plan.indices().len(), 1);
    /// assert_eq!(plan.indices()[0], s);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn indices(&self) -> &[DynIndex] {
        &self.inner.indices
    }

    /// Number of nodes the plan describes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorPlan, TreeTN};
    ///
    /// let a = DynIndex::new_dyn(2);
    /// let b = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(1);
    /// let left = IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0_f64, 2.0])?;
    /// let right = IdxTensor::from_dense(vec![bond, b.clone()], vec![1.0_f64, 3.0])?;
    /// let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?;
    /// let plan = CachedEvaluatorPlan::new(&tree, &[a, b])?;
    /// assert_eq!(plan.node_count(), 2);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn node_count(&self) -> usize {
        self.inner.sorted_node_names.len()
    }

    /// Whether two handles share one plan allocation.
    ///
    /// Use it to assert that a rebuilt evaluator really reused a plan instead
    /// of silently building an equivalent one.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorPlan, TreeTN};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let plan = CachedEvaluatorPlan::new(&tree, &[s.clone()])?;
    /// let shared = plan.clone();
    /// let rebuilt = CachedEvaluatorPlan::new(&tree, &[s])?;
    /// assert!(plan.is_same_as(&shared));
    /// assert!(!plan.is_same_as(&rebuilt));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn is_same_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Checks that `tree` has the topology and physical index placement this
    /// plan was built for. Bond dimensions, tensor values, and dtype are
    /// deliberately not compared: those are exactly what a plan is allowed to
    /// outlive.
    fn ensure_matches(&self, tree: &TreeTN<IdxTensor, V>) -> Result<()> {
        let mut node_names = tree.node_names();
        node_names.sort();
        ensure!(
            node_names == self.inner.sorted_node_names,
            "TreeTNCachedEvaluator::with_plan: the plan describes different nodes than this tree"
        );
        let neighbors = sorted_neighbors(tree);
        ensure!(
            neighbors == self.inner.neighbors,
            "TreeTNCachedEvaluator::with_plan: the plan describes a different topology than this tree"
        );
        let site_index_count = tree.site_index_network().site_index_count();
        ensure!(
            site_index_count == self.inner.indices.len(),
            "TreeTNCachedEvaluator::with_plan: the plan has {} indices but this tree has {site_index_count} site indices",
            self.inner.indices.len()
        );
        for (node, entries) in &self.inner.layout.entries_by_node {
            for entry in entries {
                let index_node = tree
                    .site_index_network()
                    .find_node_by_index(&entry.index)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::with_plan: index {:?} is not a site index of this tree",
                            entry.index
                        )
                    })?;
                ensure!(
                    index_node == node,
                    "TreeTNCachedEvaluator::with_plan: index {:?} moved from node {:?} to node {:?}",
                    entry.index,
                    node,
                    index_node
                );
            }
        }
        Ok(())
    }

    /// Returns the rooted traversal plan for `center`, building and memoizing
    /// it on first use. The plan is pure topology, so it is shared by every
    /// evaluator holding this plan.
    fn rooted_plan_for_center(&self, center: &V) -> Result<Arc<RootedMessagePlan<V>>> {
        let mut rooted_plans = self.inner.rooted_plans.lock().map_err(|_| {
            anyhow::anyhow!("TreeTNCachedEvaluator: shared rooted-plan cache is poisoned")
        })?;
        if let Some(plan) = rooted_plans.get(center) {
            return Ok(Arc::clone(plan));
        }
        let plan = Arc::new(RootedMessagePlan::from_neighbors(
            &self.inner.neighbors,
            center,
        )?);
        rooted_plans.insert(center.clone(), Arc::clone(&plan));
        Ok(plan)
    }
}

/// Cached batch evaluator for [`TreeTN`].
///
/// Use this when many batch points share repeated assignments on subtrees. It
/// chooses a center node and caches contractions from neighboring components
/// into that center.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
/// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
///
/// let s = DynIndex::new_dyn(2);
/// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![4.0_f64, 6.0])?;
/// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
/// let values = [0usize, 1usize];
/// let shape = [1usize, 2usize];
/// let points = ColMajorArrayRef::new(&values, &shape).unwrap();
///
/// let mut evaluator = TreeTNCachedEvaluator::new(
///     &tree,
///     &[s],
///     CachedEvaluatorOptions::<usize>::default(),
/// )?;
/// let result = evaluator.evaluate_batched(points)?;
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0].real(), 4.0);
/// assert_eq!(result[1].real(), 6.0);
/// assert_eq!(evaluator.center(), Some(&0));
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct TreeTNCachedEvaluator<'a, V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    tree: &'a TreeTN<IdxTensor, V>,
    /// Immutable, dtype-independent topology, key layouts, and traversal
    /// plans. Shared with every other evaluator built for the same topology
    /// and index list, so replacing a tree's numerical values rebuilds only
    /// the message caches below.
    plan: CachedEvaluatorPlan<V>,
    options: CachedEvaluatorOptions<V>,
    center: Option<V>,
    batch_point_limit: usize,
    last_stats: CachedEvaluationStats,
    /// Run-scoped, per-directed-edge persistent message cache. Lives as long
    /// as this evaluator: an input-tree evaluator lives for a whole TreeACI
    /// run, so its cache does too; an output-tree evaluator that must be
    /// dropped when pivot injection changes the output is a caller-level
    /// concern (drop this evaluator and build a new one), not this field's.
    /// Keyed by the physical assignments in the node's rooted subtree. This
    /// is the minimal assignment set on which a directed message depends;
    /// changing a site in another component must not invalidate this entry.
    message_caches: HashMap<(V, V), PackedMessageCache<IndexKey, CachedScalar>>,
    /// Each node's parent-bond `DynIndex`, memoized: it never changes for a
    /// fixed rooting, but `TreeTN::edge_between`/`bond_index` are graph
    /// lookups that cost real time if repeated on every call.
    parent_bond_indices: HashMap<(V, V), DynIndex>,
    prepared_branch_slices_f64: PreparedBranchSliceCache<V, f64>,
    prepared_branch_slices_c64: PreparedBranchSliceCache<V, Complex64>,
    raw_messages: bool,
}

impl<'a, V> TreeTNCachedEvaluator<'a, V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    /// Creates a cached evaluator for `tree` and the requested physical indices.
    ///
    /// If `options.center` is set, that node is used directly. Otherwise, the
    /// first internal batch of [`Self::evaluate_batched`] chooses a center
    /// with greedy search using that batch's repeated-subtree structure.
    ///
    /// # Errors
    ///
    /// Returns a [`TreeTNOperationError`] on an index or topology mismatch,
    /// when a configured center is not a node, when `max_batch_points` is
    /// `Some(0)`, or when backend preparation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![5])?;
    /// let evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions { center: Some(5), ..Default::default() },
    /// )?;
    /// assert_eq!(evaluator.center(), Some(&5));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        tree: &'a TreeTN<IdxTensor, V>,
        indices: &[DynIndex],
        options: CachedEvaluatorOptions<V>,
    ) -> std::result::Result<Self, TreeTNOperationError> {
        let plan = CachedEvaluatorPlan::new(tree, indices)?;
        Self::with_plan(tree, &plan, options)
    }

    /// Creates a cached evaluator that reuses an existing
    /// [`CachedEvaluatorPlan`].
    ///
    /// Use this when the same tree *shape* is evaluated repeatedly while its
    /// tensors change, so the immutable topology, key layouts, and traversal
    /// plans are built once and only the numerical message caches are new.
    ///
    /// # Arguments
    ///
    /// * `tree` - The tree to evaluate. Its topology and physical index
    ///   placement must be the ones `plan` was built for; its bond dimensions,
    ///   values, and dtype are free to differ.
    /// * `plan` - The shared plan. It is cloned by `Arc`, so the new evaluator
    ///   keeps it alive and no layout work is repeated.
    /// * `options` - Same meaning as for [`Self::new`]; batch size, cache
    ///   budgets and centre selection are per evaluator, not per plan.
    ///
    /// # Returns
    ///
    /// An evaluator with an empty message cache that shares `plan`.
    ///
    /// # Errors
    ///
    /// Returns a [`TreeTNOperationError`] when `tree` does not carry the
    /// plan's node set, neighbour structure, or physical index placement, so
    /// that the plan and the tree mismatch, or when `options.center` or any
    /// entry of `options.initial_centers` is not a node of `tree`, or when
    /// `options.max_batch_points` is `Some(0)`. Bond
    /// dimensions, tensor values, and dtype are deliberately not checked,
    /// because the plan does not describe them.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_treetn::{
    ///     CachedEvaluatorOptions, CachedEvaluatorPlan, EvaluationHint, TreeTN,
    ///     TreeTNCachedEvaluator,
    /// };
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let first = TreeTN::<_, usize>::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![s.clone()], vec![4.0_f64, 6.0])?],
    ///     vec![0],
    /// )?;
    /// let second = TreeTN::<_, usize>::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![s.clone()], vec![-1.0_f64, 0.5])?],
    ///     vec![0],
    /// )?;
    ///
    /// let plan = CachedEvaluatorPlan::new(&first, &[s])?;
    /// let values = [0usize, 1];
    /// let points = ColMajorArrayRef::new(&values, &[1, 2])?;
    ///
    /// let mut a = TreeTNCachedEvaluator::with_plan(
    ///     &first,
    ///     &plan,
    ///     CachedEvaluatorOptions { center: Some(0), ..Default::default() },
    /// )?;
    /// let mut b = TreeTNCachedEvaluator::with_plan(
    ///     &second,
    ///     &plan,
    ///     CachedEvaluatorOptions { center: Some(0), ..Default::default() },
    /// )?;
    ///
    /// assert_eq!(
    ///     a.evaluate_batched_typed::<f64>(points, EvaluationHint::default())?,
    ///     vec![4.0, 6.0]
    /// );
    /// assert_eq!(
    ///     b.evaluate_batched_typed::<f64>(points, EvaluationHint::default())?,
    ///     vec![-1.0, 0.5]
    /// );
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn with_plan(
        tree: &'a TreeTN<IdxTensor, V>,
        plan: &CachedEvaluatorPlan<V>,
        options: CachedEvaluatorOptions<V>,
    ) -> std::result::Result<Self, TreeTNOperationError> {
        plan.ensure_matches(tree)?;
        if options.max_batch_points == Some(0) {
            return Err(TreeTNOperationError::from(anyhow::anyhow!(
                "TreeTNCachedEvaluator: max_batch_points must be positive"
            )));
        }
        if let Some(center) = &options.center {
            ensure_node_exists(tree, center, "TreeTNCachedEvaluator::new: center")?;
        }
        for initial_center in &options.initial_centers {
            ensure_node_exists(
                tree,
                initial_center,
                "TreeTNCachedEvaluator::new: initial center",
            )?;
        }
        let batch_point_limit = match options.max_batch_points {
            Some(limit) => limit,
            None => {
                // Bound the largest local tensor's per-point scalar buffer
                // estimate; small/raw paths keep larger groups to amortize setup.
                let largest_local = plan.inner.sorted_node_names.iter().try_fold(
                    1_usize,
                    |largest, node| -> Result<usize> {
                        let tensor = tensor_for_node(tree, node)?;
                        let elements = tensor
                            .indices()
                            .iter()
                            .fold(1_usize, |n, index| n.saturating_mul(index.dim()));
                        Ok(largest.max(elements))
                    },
                )?;
                let per_point = largest_local
                    .saturating_mul(std::mem::size_of::<CachedScalar>())
                    .max(1);
                (32 * 1024 * 1024 / per_point).clamp(1, 256)
            }
        };
        let center = options.center.clone();
        let branch_slice_cache_max_bytes = options.branch_slice_cache_max_bytes;
        Ok(Self {
            tree,
            plan: plan.clone(),
            options,
            center,
            batch_point_limit,
            last_stats: CachedEvaluationStats::default(),
            message_caches: HashMap::new(),
            parent_bond_indices: HashMap::new(),
            prepared_branch_slices_f64: PreparedBranchSliceCache::new(branch_slice_cache_max_bytes),
            prepared_branch_slices_c64: PreparedBranchSliceCache::new(branch_slice_cache_max_bytes),
            raw_messages: false,
        })
    }

    /// Returns the immutable plan this evaluator uses.
    ///
    /// Clone it to build another evaluator for the same topology with
    /// [`Self::with_plan`] instead of repeating the layout work.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s.clone()],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// )?;
    /// assert_eq!(evaluator.plan().indices(), &[s]);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn plan(&self) -> &CachedEvaluatorPlan<V> {
        &self.plan
    }

    /// Immutable physical layout of this evaluator's index list.
    fn layout(&self) -> &EvaluatorLayout<V> {
        &self.plan.inner.layout
    }

    /// Immutable per-directed-edge component layouts.
    fn directed_component_layouts(&self) -> &DirectedComponentLayouts<V> {
        &self.plan.inner.directed_component_layouts
    }

    /// Returns the selected center node, if one has been selected.
    ///
    /// A fixed center is available immediately after [`Self::new`]. An automatic
    /// center is selected during the first [`Self::evaluate_batched`] call.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let mut evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// )?;
    /// assert_eq!(evaluator.center(), None);
    /// let values = [0usize];
    /// let shape = [1usize, 1usize];
    /// let _ = evaluator.evaluate_batched(ColMajorArrayRef::new(&values, &shape).unwrap())?;
    /// assert_eq!(evaluator.center(), Some(&0));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn center(&self) -> Option<&V> {
        self.center.as_ref()
    }

    /// Evaluates all batch points using cached subtree environments.
    ///
    /// `values` must have shape `[indices.len(), n_points]` in column-major
    /// layout. The returned vector contains one scalar per column. Internal
    /// batches contain at most [`CachedEvaluatorOptions::max_batch_points`]
    /// points when set; the default selects up to 256 using local tensor sizes.
    /// Separate finite cache budgets bound retained
    /// payloads; input/output storage still scales with the number of points,
    /// and temporary storage also depends on network ranks.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] under the same conditions as
    /// [`Self::evaluate_batched_with_hint`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![4.0_f64, 6.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let values = [0usize, 1usize];
    /// let shape = [1usize, 2usize];
    /// let mut evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// )?;
    /// let result = evaluator.evaluate_batched(ColMajorArrayRef::new(&values, &shape).unwrap())?;
    /// assert_eq!(result.len(), 2);
    /// assert_eq!(result[0].real(), 4.0);
    /// assert_eq!(result[1].real(), 6.0);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_batched(
        &mut self,
        values: ColMajorArrayRef<'_, usize>,
    ) -> std::result::Result<Vec<AnyScalar>, TreeTNOperationError> {
        self.evaluate_batched_with_hint(values, EvaluationHint::default())
    }

    /// Evaluates a batch, optionally naming the node this batch varies around.
    ///
    /// A caller that scans one site while holding the rest fixed knows which
    /// node that is. Contracting around it makes every incoming message
    /// constant across the batch, so each is contracted once. Directed-message
    /// caches remain valid when successive calls name different centers. Once
    /// the directed messages for a hinted center are warm, evaluation may
    /// assemble the result from the two sides of one cut edge, with a final
    /// work count proportional to the batch size times that edge dimension.
    ///
    /// The hint applies to this call only and does not replace a centre already
    /// chosen by [`CachedEvaluatorOptions::center`] or by greedy search, so a
    /// caller that knows nothing keeps the existing behaviour.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when `values` does not have shape
    /// `[indices.len(), n_points]`, when the hinted node is not in the tree, or
    /// when the cached message contractions fail.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_treetn::{
    ///     CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator,
    /// };
    ///
    /// let a = DynIndex::new_dyn(2);
    /// let b = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(1);
    /// let left = IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0_f64, 2.0])?;
    /// let right = IdxTensor::from_dense(vec![bond, b.clone()], vec![1.0_f64, 10.0])?;
    /// let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?;
    ///
    /// let mut evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[a, b],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// )?;
    ///
    /// // Scan site 1 with site 0 held fixed.
    /// let values = [0usize, 0, 0, 1];
    /// let points = ColMajorArrayRef::new(&values, &[2, 2])?;
    /// let hinted = evaluator.evaluate_batched_with_hint(points, EvaluationHint::around(1))?;
    ///
    /// assert_eq!(hinted.len(), 2);
    /// assert_eq!(hinted[0].real(), 1.0);
    /// assert_eq!(hinted[1].real(), 10.0);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_batched_with_hint(
        &mut self,
        values: ColMajorArrayRef<'_, usize>,
        hint: EvaluationHint<V>,
    ) -> std::result::Result<Vec<AnyScalar>, TreeTNOperationError> {
        Ok(self
            .evaluate_batched_cached(values, hint)?
            .into_iter()
            .map(CachedScalar::into_any)
            .collect())
    }

    /// Evaluates a batch and returns typed scalars directly.
    ///
    /// This is the allocation-light counterpart of
    /// [`Self::evaluate_batched_with_hint`]: results stay in the evaluator's
    /// internal lightweight scalar representation and are converted once into
    /// `T`, so no dynamic rank-zero tensor is built per result. Use it
    /// whenever the caller already knows its element type; use the
    /// `AnyScalar` wrapper only when the dtype must stay dynamic.
    ///
    /// # Arguments
    ///
    /// * `values` - Physical coordinates with shape `[indices.len(),
    ///   n_points]` in **column-major** layout: column `p` holds point `p`'s
    ///   coordinate for each index of this evaluator's index list, in that
    ///   list's order. Duplicate and repeated columns are allowed and are
    ///   answered in the order given.
    /// * `hint` - Optional centre this batch varies around; see
    ///   [`Self::evaluate_batched_with_hint`]. [`EvaluationHint::default`]
    ///   keeps the evaluator's ordinary centre selection.
    ///
    /// # Returns
    ///
    /// One owned `T` per column of `values`, in the same order. The vector is
    /// freshly allocated and owned by the caller; the evaluator retains only
    /// its own message cache.
    ///
    /// # Errors
    ///
    /// Returns an error when `values` has the wrong shape for this evaluator's
    /// index list, when the hinted node is not in the tree, when a contraction
    /// fails, or when the tree's stored dtype cannot be decoded as `T`.
    /// Precision conversion within a kind (`f32` to `f64`) and widening a real
    /// payload into a complex request are permitted; narrowing a complex
    /// payload into a real request is not, and reports
    /// [`EvaluatedScalarKindMismatch`] as the error source.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_treetn::{
    ///     CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator,
    /// };
    ///
    /// let a = DynIndex::new_dyn(2);
    /// let b = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(1);
    /// let left = IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0_f64, 2.0])?;
    /// let right = IdxTensor::from_dense(vec![bond, b.clone()], vec![3.0_f64, 10.0])?;
    /// let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?;
    ///
    /// let mut evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[a, b],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// )?;
    ///
    /// // Column-major: each column is one point (site 0 value, site 1 value).
    /// let values = [0usize, 0, 1, 1];
    /// let points = ColMajorArrayRef::new(&values, &[2, 2])?;
    /// let typed = evaluator.evaluate_batched_typed::<f64>(points, EvaluationHint::around(1))?;
    /// assert_eq!(typed, vec![3.0, 20.0]);
    ///
    /// // A real tree widens into a complex request.
    /// let complex = evaluator
    ///     .evaluate_batched_typed::<num_complex::Complex64>(points, EvaluationHint::default())?;
    /// assert_eq!(complex[0], num_complex::Complex64::new(3.0, 0.0));
    /// assert_eq!(complex[1], num_complex::Complex64::new(20.0, 0.0));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_batched_typed<T: TensorElement>(
        &mut self,
        values: ColMajorArrayRef<'_, usize>,
        hint: EvaluationHint<V>,
    ) -> std::result::Result<Vec<T>, TreeTNOperationError> {
        let cached = self.evaluate_batched_cached(values, hint)?;
        cached_values_into_typed(cached)
            .map_err(|error| TreeTNOperationError::from(anyhow::Error::new(error)))
    }

    /// Evaluates a hinted batch and returns typed scalars on a caller-held explicit session.
    ///
    /// The session counterpart of [`Self::evaluate_batched_typed`], with the same
    /// precision conversion rules. Every numerical leaf (grouped GEMMs,
    /// retained-index contractions, selection, stacking, gathering and
    /// permutation) runs through `session` and publishes compact host storage,
    /// so nothing enters the process-global backend or an eager runtime. This is
    /// the only session batch wrapper: its results are typed scalars, because an
    /// `AnyScalar` result would adopt a rank-zero eager tensor.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] under the same conditions as
    /// [`Self::evaluate_batched_typed`], and when the supplied session rejects a
    /// leaf.
    pub fn evaluate_batched_typed_on_session<T: TensorElement>(
        &mut self,
        session: &mut Session<'_>,
        values: ColMajorArrayRef<'_, usize>,
        hint: EvaluationHint<V>,
    ) -> std::result::Result<Vec<T>, TreeTNOperationError> {
        let mut route = CachedEvaluationRoute::on_session(session);
        let cached = self.evaluate_batched_cached_with_route(values, hint, &mut route)?;
        cached_values_into_typed(cached)
            .map_err(|error| TreeTNOperationError::from(anyhow::Error::new(error)))
    }

    /// Shared batch evaluation returning the evaluator's internal lightweight
    /// scalars. Both public batch entry points convert this result: the
    /// compatibility wrapper into `AnyScalar` and the typed entry point
    /// directly into `T`.
    fn evaluate_batched_cached(
        &mut self,
        values: ColMajorArrayRef<'_, usize>,
        hint: EvaluationHint<V>,
    ) -> std::result::Result<Vec<CachedScalar>, TreeTNOperationError> {
        self.evaluate_batched_cached_with_route(values, hint, &mut CachedEvaluationRoute::global())
    }

    fn evaluate_batched_cached_with_route(
        &mut self,
        values: ColMajorArrayRef<'_, usize>,
        hint: EvaluationHint<V>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> std::result::Result<Vec<CachedScalar>, TreeTNOperationError> {
        #[cfg(feature = "diagnostics")]
        let query_started = std::time::Instant::now();
        validate_values_shape(
            values,
            self.layout().n_indices,
            "TreeTNCachedEvaluator::evaluate_batched",
        )?;
        if values.shape()[1] == 0 {
            self.last_stats = CachedEvaluationStats::default();
            return Ok(Vec::new());
        }
        let chunk_size = self.batch_point_limit.min(values.shape()[1]);
        let n_indices = self.layout().n_indices;
        let first_shape = [n_indices, chunk_size];
        let first = ColMajorArrayRef::new(&values.data()[..n_indices * chunk_size], &first_shape)
            .map_err(anyhow::Error::from)?;
        let hinted_center = hint.center.is_some();
        let center = match hint.center {
            Some(node) => {
                if self.tree.node_index(&node).is_none() {
                    return Err(TreeTNOperationError::from(anyhow::anyhow!(
                        "TreeTNCachedEvaluator: hinted centre {:?} is not a node of this tree",
                        node
                    )));
                }
                node
            }
            None => self.ensure_center(first)?.clone(),
        };
        let mut results = Vec::with_capacity(values.shape()[1]);
        let mut stats = CachedEvaluationStats::default();
        for start in (0..values.shape()[1]).step_by(chunk_size) {
            let end = start.saturating_add(chunk_size).min(values.shape()[1]);
            let shape = [n_indices, end - start];
            let chunk =
                ColMajorArrayRef::new(&values.data()[n_indices * start..n_indices * end], &shape)
                    .map_err(anyhow::Error::from)?;
            results.extend(self.evaluate_chunk(&center, chunk, hinted_center, route)?);
            stats.subtree_environment_count += self.last_stats.subtree_environment_count;
            stats.directed_message_count += self.last_stats.directed_message_count;
            stats.batched_message_contract_count += self.last_stats.batched_message_contract_count;
            stats.batched_center_contract_count += self.last_stats.batched_center_contract_count;
            stats.message_cache_hits += self.last_stats.message_cache_hits;
            stats.message_cache_misses += self.last_stats.message_cache_misses;
            stats.warm_edge_cut_assembly_visits += self.last_stats.warm_edge_cut_assembly_visits;
            stats.message_cache_key_count = self.last_stats.message_cache_key_count;
            stats.message_cache_logical_bytes = self.last_stats.message_cache_logical_bytes;
            stats.message_cache_owned_bytes_estimate =
                self.last_stats.message_cache_owned_bytes_estimate;
        }
        self.last_stats = stats;
        #[cfg(feature = "diagnostics")]
        {
            let (node, shape) = self.diagnostic_node(&center)?;
            diagnostics::record_query(
                &node,
                shape,
                query_started.elapsed(),
                values.shape()[1],
                self.diagnostic_cache(),
            );
        }
        Ok(results)
    }

    fn evaluate_chunk(
        &mut self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        hinted_center: bool,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> std::result::Result<Vec<CachedScalar>, TreeTNOperationError> {
        if hinted_center {
            if let Some(result) = self.try_contract_cached_edge_cut(center, values)? {
                return Ok(result);
            }
        }
        let (component_batches, environment_cache) =
            self.build_environment_cache(center, values, route)?;
        let results = if hinted_center {
            self.contract_edge_cut_or_center(
                center,
                values,
                &component_batches,
                &environment_cache,
                route,
            )
        } else {
            self.contract_center_for_points(
                center,
                values,
                &component_batches,
                &environment_cache,
                route,
            )
        }?;
        self.last_stats.batched_center_contract_count = 1;
        Ok(results)
    }

    #[cfg(feature = "diagnostics")]
    fn diagnostic_node(&self, node: &V) -> Result<(String, diagnostics::NodeShape)> {
        let bond_dims = self
            .tree
            .site_index_network()
            .neighbors(node)
            .map(|neighbor| {
                self.tree
                    .edge_between(node, &neighbor)
                    .and_then(|edge| self.tree.bond_index(edge))
                    .map(|index| index.dim())
                    .ok_or_else(|| anyhow::anyhow!("diagnostic node has no incident bond"))
            })
            .collect::<Result<Vec<_>>>()?;
        let physical_dim = self
            .layout()
            .entries_by_node
            .get(node)
            .into_iter()
            .flatten()
            .try_fold(1usize, |n, entry| {
                n.checked_mul(entry.index.dim())
                    .ok_or_else(|| anyhow::anyhow!("diagnostic physical dimension overflows usize"))
            })?;
        Ok((
            format!("{}:{node:?}", self.options.diagnostic_namespace),
            diagnostics::NodeShape {
                physical_dim,
                bond_dims,
            },
        ))
    }

    #[cfg(feature = "diagnostics")]
    fn diagnostic_cache(&self) -> diagnostics::CacheDiagnostics {
        let mut result = diagnostics::CacheDiagnostics {
            message_owned_bytes: hash_map_owned_bytes_estimate(&self.message_caches),
            prepared_entries: self.prepared_branch_slices_f64.slices.len()
                + self.prepared_branch_slices_c64.slices.len(),
            prepared_payload_bytes: self.prepared_branch_slices_f64.retained_bytes()
                + self.prepared_branch_slices_c64.retained_bytes(),
            ..Default::default()
        };
        for cache in self.message_caches.values() {
            result.message_entries = result.message_entries.saturating_add(cache.key_count());
            result.message_payload_bytes = result
                .message_payload_bytes
                .saturating_add(cache.logical_payload_bytes());
            // The outer map already accounts for the inline cache value.
            result.message_owned_bytes = result.message_owned_bytes.saturating_add(
                cache
                    .owned_retained_bytes_estimate()
                    .saturating_sub(std::mem::size_of_val(cache)),
            );
        }
        result.prepared_owned_bytes = result
            .prepared_payload_bytes
            .saturating_add(hash_map_owned_bytes_estimate(
                &self.prepared_branch_slices_f64.slices,
            ))
            .saturating_add(hash_map_owned_bytes_estimate(
                &self.prepared_branch_slices_c64.slices,
            ));
        result
    }

    fn ensure_center(&mut self, values: ColMajorArrayRef<'_, usize>) -> Result<&V> {
        if self.center.is_none() {
            let cost_index = ComponentCostIndex::from_layout(self.tree, self.layout(), values)?;
            let search =
                GreedyCenterSearch::<V>::with_max_steps(self.options.max_greedy_steps_per_start);
            let result = search.search(&cost_index, &self.options.initial_centers)?;
            self.center = Some(result.center);
        }
        self.center.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TreeTNCachedEvaluator::ensure_center: no center selected")
        })
    }

    fn build_environment_cache(
        &mut self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<CacheBuildResult<V>> {
        self.last_stats = CachedEvaluationStats::default();
        let plan = self.rooted_plan_for_center(center)?;
        self.raw_messages = self.can_use_raw_messages(center)?;
        let assignment_batches = self.build_message_assignment_batches(&plan, values)?;

        let mut messages = HashMap::<V, StackedMessage>::new();
        for neighbor in plan.children.get(center).cloned().unwrap_or_default() {
            let node_message = self.get_or_compute_node_message(
                &neighbor,
                values,
                &plan,
                &assignment_batches,
                &mut messages,
                route,
            )?;
            messages.insert(neighbor, node_message);
        }

        let mut component_batches = Vec::new();
        let mut cache = HashMap::new();
        let mut subtree_environment_count = 0usize;
        for neighbor in plan.children.get(center).cloned().unwrap_or_default() {
            let assignment_batch = assignment_batches.get(&neighbor).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing assignments for neighbor {:?}",
                    neighbor
                )
            })?;
            let environment = messages.remove(&neighbor).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing messages for neighbor {:?}",
                    neighbor
                )
            })?;
            subtree_environment_count += assignment_batch.first_points.len();
            cache.insert(neighbor.clone(), environment);
            component_batches.push(ComponentBatch {
                neighbor,
                point_to_assignment: assignment_batch.point_to_assignment.clone(),
            });
        }
        self.last_stats.subtree_environment_count = subtree_environment_count;
        #[cfg(test)]
        self.update_cache_storage_stats();
        Ok((component_batches, cache))
    }

    #[cfg(test)]
    fn update_cache_storage_stats(&mut self) {
        self.last_stats.message_cache_logical_bytes = self
            .message_caches
            .values()
            .map(PackedMessageCache::logical_payload_bytes)
            .sum();
        self.last_stats.message_cache_key_count = self
            .message_caches
            .values()
            .map(PackedMessageCache::key_count)
            .sum();
        self.last_stats.message_cache_owned_bytes_estimate = self
            .message_caches
            .values()
            .map(PackedMessageCache::owned_retained_bytes_estimate)
            .sum::<usize>()
            .saturating_add(hash_map_owned_bytes_estimate(&self.message_caches));
    }

    /// A hinted, fully cached cut needs neither the other center components
    /// nor stacked copies of its packed columns. Cold and partial hits keep
    /// the ordinary message-computation path and its admission accounting.
    fn try_contract_cached_edge_cut(
        &mut self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Option<Vec<CachedScalar>>> {
        let plan = self.rooted_plan_for_center(center)?;
        let Some(neighbor) = plan
            .children
            .get(center)
            .and_then(|children| children.first())
        else {
            return Ok(None);
        };
        let left_edge = (neighbor.clone(), center.clone());
        let right_edge = (center.clone(), neighbor.clone());
        let Some(left_cache) = self.message_caches.get(&left_edge) else {
            return Ok(None);
        };
        let Some(right_cache) = self.message_caches.get(&right_edge) else {
            return Ok(None);
        };
        if left_cache.key_count() == 0 || right_cache.key_count() == 0 {
            return Ok(None);
        }
        let bond_dim = self
            .parent_bond_indices
            .get(&left_edge)
            .ok_or_else(|| anyhow::anyhow!("cached cut is missing its bond index"))?
            .dim();
        if bond_dim == 0 {
            return Ok(None);
        }
        if left_cache.bond_dim != bond_dim || right_cache.bond_dim != bond_dim {
            return Err(anyhow::anyhow!(
                "cached cut column dimension does not match its bond"
            ));
        }
        let kind = tensor_scalar_kind(tensor_for_node(self.tree, center)?)?;
        if left_cache.columns.first().map(CachedScalar::kind) != Some(kind)
            || right_cache.columns.first().map(CachedScalar::kind) != Some(kind)
        {
            // Mixed real/complex networks use the existing promotion path.
            return Ok(None);
        }
        let left_batch = self.build_directed_assignment_batch(neighbor, center, values)?;
        let right_batch = self.build_directed_assignment_batch(center, neighbor, values)?;
        #[cfg(feature = "diagnostics")]
        let left_started = std::time::Instant::now();
        let Some(left_positions) = left_batch
            .keys
            .iter()
            .map(|key| left_cache.position(key))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        #[cfg(feature = "diagnostics")]
        let left_elapsed = left_started.elapsed();
        #[cfg(feature = "diagnostics")]
        let right_started = std::time::Instant::now();
        let Some(right_positions) = right_batch
            .keys
            .iter()
            .map(|key| right_cache.position(key))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        #[cfg(feature = "diagnostics")]
        let right_elapsed = right_started.elapsed();

        let n_points = values.shape()[1];
        let assembly_visits = n_points
            .checked_mul(bond_dim)
            .ok_or_else(|| anyhow::anyhow!("cached cut work count overflows usize"))?;
        let mut result = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let left_assignment = left_batch.point_to_assignment.get(point).ok_or_else(|| {
                anyhow::anyhow!("missing cached cut assignment for point {point}")
            })?;
            let right_assignment = right_batch.point_to_assignment.get(point).ok_or_else(|| {
                anyhow::anyhow!("missing cached reverse assignment for point {point}")
            })?;
            let left_position = left_positions
                .get(*left_assignment)
                .ok_or_else(|| anyhow::anyhow!("missing cached cut column for point {point}"))?;
            let right_position = right_positions.get(*right_assignment).ok_or_else(|| {
                anyhow::anyhow!("missing cached reverse column for point {point}")
            })?;
            let left = left_cache.checked_column(*left_position)?;
            let right = right_cache.checked_column(*right_position)?;
            result.push(cached_column_dot(kind, left, right)?);
        }
        // Tentative all-hit probes do not count. Commit hits only after both
        // directions and their checked assembly have succeeded.
        self.message_caches
            .get_mut(&left_edge)
            .ok_or_else(|| anyhow::anyhow!("cached cut disappeared"))?
            .record_hits(left_batch.keys.len());
        self.message_caches
            .get_mut(&right_edge)
            .ok_or_else(|| anyhow::anyhow!("cached reverse cut disappeared"))?
            .record_hits(right_batch.keys.len());
        self.last_stats = CachedEvaluationStats {
            subtree_environment_count: left_batch.keys.len(),
            directed_message_count: left_batch.keys.len() + right_batch.keys.len(),
            batched_message_contract_count: 2,
            batched_center_contract_count: 1,
            message_cache_hits: left_batch.keys.len() + right_batch.keys.len(),
            warm_edge_cut_assembly_visits: assembly_visits,
            ..Default::default()
        };
        #[cfg(test)]
        self.update_cache_storage_stats();
        #[cfg(feature = "diagnostics")]
        for (node, elapsed, hits) in [
            (neighbor, left_elapsed, left_batch.keys.len()),
            (center, right_elapsed, right_batch.keys.len()),
        ] {
            let (name, shape) = self.diagnostic_node(node)?;
            diagnostics::record_guard(
                &name,
                shape,
                diagnostics::PhaseMeasurement {
                    elapsed,
                    hits: hits as u64,
                    misses: 0,
                    kernel: Default::default(),
                },
            );
        }
        Ok(Some(result))
    }

    fn rooted_plan_for_center(&mut self, center: &V) -> Result<Arc<RootedMessagePlan<V>>> {
        self.plan.rooted_plan_for_center(center)
    }

    fn build_message_assignment_batches(
        &self,
        plan: &RootedMessagePlan<V>,
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<HashMap<V, AssignmentBatch>> {
        let n_points = values.shape()[1];
        let mut local_keys = HashMap::<V, Vec<IndexKey>>::new();

        // One scratch coordinate buffer for the whole call. Allocating it per
        // node and point was the assignment builder's dominant short-lived
        // allocation; the buffer never escapes this loop.
        let mut raw = Vec::new();
        for node in &self.plan.inner.sorted_node_names {
            let entries = self
                .layout()
                .entries_by_node
                .get(node)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let local_layout = self
                .layout()
                .local_layouts_by_node
                .get(node)
                .ok_or_else(|| anyhow::anyhow!("missing local layout for node {:?}", node))?;
            let mut keys = Vec::with_capacity(n_points);
            for point in 0..n_points {
                raw.clear();
                for entry in entries {
                    raw.push(value_at(
                        values,
                        entry.input_position,
                        point,
                        "TreeTNCachedEvaluator::evaluate_batched",
                    )?);
                }
                validate_entry_values(entries, &raw, "TreeTNCachedEvaluator::evaluate_batched")?;
                keys.push(
                    local_layout
                        .indexer
                        .encode(&raw)
                        .map_err(anyhow::Error::from)?,
                );
            }
            local_keys.insert(node.clone(), keys);
        }

        let mut assignment_batches = HashMap::<V, AssignmentBatch>::new();
        for node in &plan.postorder {
            let local_keys = local_keys.get(node).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing local keys for {:?}",
                    node
                )
            })?;
            let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
            let child_batches = children
                .iter()
                .map(|child| {
                    assignment_batches.get(child).ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::evaluate_batched: missing child assignments for {:?}",
                            child
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let parent = plan
                .parent
                .get(node)
                .and_then(Clone::clone)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::evaluate_batched: missing parent for {:?}",
                        node
                    )
                })?;
            let component_layout = self
                .directed_component_layouts()
                .get(&(node.clone(), parent))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::evaluate_batched: missing directed component layout for {:?}",
                        node
                    )
                })?;
            ensure!(
                component_layout.child_nodes == children,
                "TreeTNCachedEvaluator::evaluate_batched: component child order disagrees with rooted plan for {:?}",
                node
            );

            let mut assignment_ids = HashMap::<IndexKey, usize>::new();
            let mut first_points = Vec::new();
            let mut assignment_keys = Vec::new();
            let mut point_to_assignment = Vec::with_capacity(n_points);
            for (point, local_key) in local_keys.iter().enumerate().take(n_points) {
                let mut builder =
                    KeyBuilder::with_capacity_bits(component_layout.layout.indexer.width_bits())
                        .map_err(anyhow::Error::from)?;
                builder.push(local_key).map_err(anyhow::Error::from)?;
                for child_batch in &child_batches {
                    let child_assignment = child_batch.point_to_assignment[point];
                    let child_key = child_batch
                        .keys
                        .get(child_assignment)
                        .ok_or_else(|| anyhow::anyhow!("missing child key for point {point}"))?;
                    builder.push(child_key).map_err(anyhow::Error::from)?;
                }
                ensure!(
                    builder.width_bits() == component_layout.layout.indexer.width_bits(),
                    "TreeTNCachedEvaluator: composed key width does not match its component layout"
                );
                let key = builder.finish();
                let assignment_id = if let Some(&assignment_id) = assignment_ids.get(&key) {
                    assignment_id
                } else {
                    let assignment_id = assignment_keys.len();
                    assignment_ids.insert(key.clone(), assignment_id);
                    assignment_keys.push(key);
                    first_points.push(point);
                    assignment_id
                };
                point_to_assignment.push(assignment_id);
            }
            let assignment_batch = AssignmentBatch {
                point_to_assignment,
                first_points,
                keys: assignment_keys,
            };
            assignment_batches.insert(node.clone(), assignment_batch);
        }

        Ok(assignment_batches)
    }

    /// Builds the assignment batch for one directed component without walking
    /// the rest of a rooted tree. This is the warm-path form: when every key
    /// for the requested edge is already cached, the recursive message lookup
    /// needs only this endpoint's assignments and must not reconstruct any
    /// descendant assignment batches.
    fn build_directed_assignment_batch(
        &self,
        from: &V,
        to: &V,
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<AssignmentBatch> {
        let component_layout = self
            .directed_component_layouts()
            .get(&(from.clone(), to.clone()))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator: missing directed component layout for {:?}->{:?}",
                    from,
                    to
                )
            })?;
        let n_points = values.shape()[1];
        let mut assignment_ids = HashMap::<IndexKey, usize>::new();
        let mut first_points = Vec::new();
        let mut assignment_keys = Vec::new();
        let mut point_to_assignment = Vec::with_capacity(n_points);
        // One scratch coordinate buffer for the whole component batch; it is
        // re-encoded per point and never escapes this loop.
        let mut raw = Vec::with_capacity(component_layout.layout.input_positions.len());
        for point in 0..n_points {
            raw.clear();
            for &input_position in &component_layout.layout.input_positions {
                raw.push(value_at(
                    values,
                    input_position,
                    point,
                    "TreeTNCachedEvaluator::evaluate_batched",
                )?);
            }
            let key = component_layout
                .layout
                .indexer
                .encode(&raw)
                .map_err(anyhow::Error::from)?;
            let assignment_id = if let Some(&assignment_id) = assignment_ids.get(&key) {
                assignment_id
            } else {
                let assignment_id = assignment_keys.len();
                assignment_ids.insert(key.clone(), assignment_id);
                assignment_keys.push(key);
                first_points.push(point);
                assignment_id
            };
            point_to_assignment.push(assignment_id);
        }
        Ok(AssignmentBatch {
            point_to_assignment,
            first_points,
            keys: assignment_keys,
        })
    }

    fn can_use_raw_messages(&self, center: &V) -> Result<bool> {
        #[cfg(test)]
        if raw_message_kernels_disabled_for_test() {
            return Ok(false);
        }
        let neighbors = &self.plan.inner.neighbors;
        if !neighbors.contains_key(center) {
            return Ok(false);
        }
        // Coordination is no longer a condition: a non-center node with three
        // or more children is covered by
        // `try_compute_multi_branch_message_raw` and the center of any degree
        // by the raw center kernel. Refusing a single high-coordination node
        // used to push *every* message of the whole tree onto the generic
        // `IdxTensor` path, which measured ~20x per hinted evaluation on a
        // 13-site spider against a chain of the same size, independently of
        // bond dimension (tensor4all-rs #727).
        if self
            .layout()
            .entries_by_node
            .values()
            .any(|entries| entries.len() != 1)
        {
            return Ok(false);
        }
        let mut scalar_kind = None;
        for (node, node_neighbors) in neighbors {
            let tensor = tensor_for_node(self.tree, node)?;
            if tensor.indices().len() != node_neighbors.len() + 1 {
                return Ok(false);
            }
            let kind = tensor_scalar_kind(tensor)?;
            if let Some(previous) = scalar_kind {
                if previous != kind {
                    return Ok(false);
                }
            } else {
                scalar_kind = Some(kind);
            }
        }
        // The specialized raw kernels currently operate on the two 64-bit
        // scalar kinds. The generic contraction path below preserves f32/c32;
        // do not route those tensors through mismatched f64/c64 readers.
        Ok(matches!(
            scalar_kind,
            Some(ScalarKind::F64 | ScalarKind::C64)
        ))
    }

    /// Computes a leaf node's message directly from the tree tensor's raw
    /// data, bypassing `contract_with_options`/`IdxTensor` entirely.
    ///
    /// Root cause (see `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`):
    /// a contraction result is backend-resident/non-contiguous, so reading it
    /// back out via `IdxTensor::to_vec` falls through to an expensive
    /// session-based materialization (measured at 71.3% of miss-path time,
    /// 3x the contraction itself). A leaf node has no contraction to do at
    /// all -- its message is just its own tensor with the physical index
    /// fixed -- so this reads the tensor's already-host-resident data once
    /// and slices it directly, producing a plain `Vec<f64>` with no backend
    /// round-trip.
    ///
    /// Returns `Ok(None)` when `node` is not eligible for this fast path
    /// (more than one physical index, a complex-valued tensor, or a tensor
    /// shape other than the expected 2 axes for a leaf), so the caller can
    /// fall back to [`Self::compute_stacked_message`]. Ineligibility is not
    /// an error: most of this crate's existing tests exercise multi-physical
    /// or branched nodes this first slice does not yet cover.
    fn try_compute_leaf_message_raw(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        _route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Option<Vec<f64>>> {
        #[cfg(test)]
        if raw_message_kernels_disabled_for_test() {
            return Ok(None);
        }
        let entries = self
            .layout()
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };

        let tensor = tensor_for_node(self.tree, node)?;
        if tensor.is_complex() {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != 2 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let parent_axis = 1 - physical_axis;

        let dims = tensor.dims();
        let parent_dim = dims[parent_axis];
        // Column-major: stride of axis k is the product of the dims before it.
        let strides = [1usize, dims[0]];

        let raw = tensor.dense_host_values::<f64>()?;
        let mut out = Vec::with_capacity(parent_dim * points.len());
        for &point in points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_leaf_message_raw",
            )?;
            for parent_value in 0..parent_dim {
                let mut axis_values = [0usize; 2];
                axis_values[physical_axis] = physical_value;
                axis_values[parent_axis] = parent_value;
                let flat = axis_values[0] * strides[0] + axis_values[1] * strides[1];
                out.push(raw[flat]);
            }
        }
        Ok(Some(out))
    }

    /// Complex-valued counterpart of [`Self::try_compute_leaf_message_raw`].
    ///
    /// Keeping this path separate from the real-valued helper avoids changing
    /// the established f64 path while allowing SGW's complex tensors to skip
    /// the generic contraction/materialization fallback as well.
    fn try_compute_leaf_message_complex_raw(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        _route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Option<Vec<Complex64>>> {
        #[cfg(test)]
        if raw_message_kernels_disabled_for_test() {
            return Ok(None);
        }
        let entries = self
            .layout()
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };

        let tensor = tensor_for_node(self.tree, node)?;
        if !tensor.is_complex() {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != 2 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let parent_axis = 1 - physical_axis;

        let dims = tensor.dims();
        let parent_dim = dims[parent_axis];
        let strides = [1usize, dims[0]];
        let raw = tensor.dense_host_values::<Complex64>()?;
        let mut out = Vec::with_capacity(parent_dim * points.len());
        for &point in points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_leaf_message_complex_raw",
            )?;
            for parent_value in 0..parent_dim {
                let mut axis_values = [0usize; 2];
                axis_values[physical_axis] = physical_value;
                axis_values[parent_axis] = parent_value;
                let flat = axis_values[0] * strides[0] + axis_values[1] * strides[1];
                out.push(raw[flat]);
            }
        }
        Ok(Some(out))
    }

    /// Contracts the parent-message matrices for one chain node in groups of
    /// equal physical values. Large groups use the tensorbackend matrix
    /// multiply; small groups retain the scalar implementation so backend
    /// setup does not dominate low-rank calls.
    fn grouped_chain_message_contraction<T>(
        route: &mut CachedEvaluationRoute<'_, '_>,
        spec: ChainContractionSpec,
        raw: &[T],
        physical_values: &[usize],
        child_columns: &[T],
    ) -> Result<Vec<T>>
    where
        T: BlasMul + Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
    {
        let ChainContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_axis,
            parent_dim,
            child_dim,
        } = spec;
        let point_count = physical_values.len();
        let expected_child_values = point_count
            .checked_mul(child_dim)
            .ok_or_else(|| anyhow::anyhow!("chain child-message shape overflows usize"))?;
        anyhow::ensure!(
            child_columns.len() == expected_child_values,
            "chain child-message length {} does not match {} points x {} child values",
            child_columns.len(),
            point_count,
            child_dim
        );
        let output_len = point_count
            .checked_mul(parent_dim)
            .ok_or_else(|| anyhow::anyhow!("chain parent-message shape overflows usize"))?;

        if point_count < 2 * CHAIN_BLAS_MIN_GROUP_POINTS {
            #[cfg(feature = "diagnostics")]
            {
                contraction_diagnostics::inc(&contraction_diagnostics::CHAIN_SCALAR_CALLS, 1);
                contraction_diagnostics::inc(
                    &contraction_diagnostics::CHAIN_SCALAR_POINTS,
                    point_count,
                );
            }
            return scalar_chain_message_contraction(spec, raw, physical_values, child_columns);
        }

        let mut groups = HashMap::<usize, Vec<usize>>::new();
        for (point, &physical_value) in physical_values.iter().enumerate() {
            groups.entry(physical_value).or_default().push(point);
        }
        let scalar_work = parent_dim
            .checked_mul(child_dim)
            .and_then(|work| work.checked_mul(point_count))
            .ok_or_else(|| anyhow::anyhow!("chain contraction work estimate overflows usize"))?;
        if scalar_work < CHAIN_BLAS_WORK_THRESHOLD
            || groups.len() > 8
            || groups
                .values()
                .any(|points| points.len() < CHAIN_BLAS_MIN_GROUP_POINTS)
        {
            #[cfg(feature = "diagnostics")]
            {
                contraction_diagnostics::inc(&contraction_diagnostics::CHAIN_SCALAR_CALLS, 1);
                contraction_diagnostics::inc(
                    &contraction_diagnostics::CHAIN_SCALAR_POINTS,
                    point_count,
                );
            }
            return scalar_chain_message_contraction(spec, raw, physical_values, child_columns);
        }
        #[cfg(feature = "diagnostics")]
        let diag_start = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        contraction_diagnostics::inc(&contraction_diagnostics::CHAIN_BLAS_POINTS, point_count);

        let matrix_len = parent_dim
            .checked_mul(child_dim)
            .ok_or_else(|| anyhow::anyhow!("chain matrix shape overflows usize"))?;
        let mut output = vec![T::default(); output_len];
        for (physical_value, points) in groups {
            #[cfg(feature = "diagnostics")]
            let setup_started = std::time::Instant::now();
            let mut left = vec![T::default(); matrix_len];
            for child_value in 0..child_dim {
                for parent_value in 0..parent_dim {
                    let mut axis_values = [0usize; 3];
                    axis_values[physical_axis] = physical_value;
                    axis_values[parent_axis] = parent_value;
                    axis_values[child_axis] = child_value;
                    let flat = axis_values[0]
                        .checked_mul(strides[0])
                        .and_then(|value| {
                            value.checked_add(axis_values[1].checked_mul(strides[1])?)
                        })
                        .and_then(|value| {
                            value.checked_add(axis_values[2].checked_mul(strides[2])?)
                        })
                        .ok_or_else(|| anyhow::anyhow!("chain tensor offset overflows usize"))?;
                    let left_offset = parent_dim
                        .checked_mul(child_value)
                        .and_then(|value| value.checked_add(parent_value))
                        .ok_or_else(|| anyhow::anyhow!("chain matrix offset overflows usize"))?;
                    left[left_offset] = *raw.get(flat).ok_or_else(|| {
                        anyhow::anyhow!("chain tensor offset {flat} is out of bounds")
                    })?;
                }
            }

            let right_len = child_dim
                .checked_mul(points.len())
                .ok_or_else(|| anyhow::anyhow!("chain right matrix shape overflows usize"))?;
            let mut right = Vec::with_capacity(right_len);
            for &point in &points {
                let start = point
                    .checked_mul(child_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain child column offset overflows usize"))?;
                let end = start
                    .checked_add(child_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain child column end overflows usize"))?;
                right.extend_from_slice(child_columns.get(start..end).ok_or_else(|| {
                    anyhow::anyhow!("chain child column {start}..{end} is out of bounds")
                })?);
            }

            #[cfg(feature = "diagnostics")]
            contraction_diagnostics::inc(&contraction_diagnostics::CHAIN_BLAS_CALLS, 1);
            #[cfg(feature = "diagnostics")]
            diagnostics::record_kernel(diagnostics::KernelDiagnostics {
                setup_ns: diagnostics::nanos(setup_started.elapsed()),
                ..Default::default()
            });
            #[cfg(feature = "diagnostics")]
            let matmul_started = std::time::Instant::now();
            let product = route.mat_mul_owned(
                Matrix::from_col_major_vec(parent_dim, child_dim, left),
                Matrix::from_col_major_vec(child_dim, points.len(), right),
            )?;
            #[cfg(feature = "diagnostics")]
            diagnostics::record_kernel(diagnostics::KernelDiagnostics {
                matmul_ns: diagnostics::nanos(matmul_started.elapsed()),
                ..Default::default()
            });
            #[cfg(feature = "diagnostics")]
            let accumulate_started = std::time::Instant::now();
            for (column, &point) in points.iter().enumerate() {
                let destination = point
                    .checked_mul(parent_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain output offset overflows usize"))?;
                let source = column
                    .checked_mul(parent_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain product offset overflows usize"))?;
                let source_end = source
                    .checked_add(parent_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain product end overflows usize"))?;
                let destination_end = destination
                    .checked_add(parent_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain output end overflows usize"))?;
                output[destination..destination_end]
                    .copy_from_slice(&product.as_col_major_slice()[source..source_end]);
            }
            #[cfg(feature = "diagnostics")]
            diagnostics::record_kernel(diagnostics::KernelDiagnostics {
                accumulate_ns: diagnostics::nanos(accumulate_started.elapsed()),
                ..Default::default()
            });
        }
        #[cfg(feature = "diagnostics")]
        contraction_diagnostics::add(
            &contraction_diagnostics::CHAIN_CONTRACT_NS,
            diag_start.elapsed(),
        );
        Ok(output)
    }

    /// Prepares one physical branch slice as the column-major left operand
    /// used by the grouped branch GEMM.
    ///
    /// The returned matrix has shape `(parent_dim * child_dim_2) x
    /// child_dim_1`. Parent values remain contiguous inside each child-2
    /// block, preserving the existing reduction order.
    fn prepare_branch_slice<T>(
        spec: BranchContractionSpec,
        raw: &[T],
        physical_value: usize,
    ) -> Result<Matrix<T>>
    where
        T: Copy + Default,
    {
        let left_rows = spec
            .parent_dim
            .checked_mul(spec.child_dim_2)
            .ok_or_else(|| anyhow::anyhow!("branch matrix row count overflows usize"))?;
        let left_len = left_rows
            .checked_mul(spec.child_dim_1)
            .ok_or_else(|| anyhow::anyhow!("branch matrix size overflows usize"))?;
        let mut left = vec![T::default(); left_len];
        let physical_base = physical_value
            .checked_mul(spec.strides[spec.physical_axis])
            .ok_or_else(|| anyhow::anyhow!("branch tensor offset overflows usize"))?;

        if spec.strides[spec.child_axis_2] == 1 {
            // Read a contiguous child-2 run, then scatter it into the
            // parent-fast matrix layout. This is the old specialized read
            // path, now paid only when an owner cache misses.
            for c1 in 0..spec.child_dim_1 {
                let base_after_c1 =
                    physical_base
                        .checked_add(c1.checked_mul(spec.strides[spec.child_axis_1]).ok_or_else(
                            || anyhow::anyhow!("branch tensor offset overflows usize"),
                        )?)
                        .ok_or_else(|| anyhow::anyhow!("branch tensor offset overflows usize"))?;
                let left_c1_offset = left_rows
                    .checked_mul(c1)
                    .ok_or_else(|| anyhow::anyhow!("branch matrix offset overflows usize"))?;
                for parent in 0..spec.parent_dim {
                    let base = base_after_c1
                        .checked_add(
                            parent
                                .checked_mul(spec.strides[spec.parent_axis])
                                .ok_or_else(|| {
                                    anyhow::anyhow!("branch tensor offset overflows usize")
                                })?,
                        )
                        .ok_or_else(|| anyhow::anyhow!("branch tensor offset overflows usize"))?;
                    let end = base
                        .checked_add(spec.child_dim_2)
                        .ok_or_else(|| anyhow::anyhow!("branch tensor offset overflows usize"))?;
                    let slice = raw.get(base..end).ok_or_else(|| {
                        anyhow::anyhow!("branch tensor offset {base}..{end} is out of bounds")
                    })?;
                    let left_base = left_c1_offset
                        .checked_add(parent)
                        .ok_or_else(|| anyhow::anyhow!("branch matrix offset overflows usize"))?;
                    for (c2, &value) in slice.iter().enumerate() {
                        let offset = left_base
                            .checked_add(spec.parent_dim.checked_mul(c2).ok_or_else(|| {
                                anyhow::anyhow!("branch matrix offset overflows usize")
                            })?)
                            .ok_or_else(|| {
                                anyhow::anyhow!("branch matrix offset overflows usize")
                            })?;
                        left[offset] = value;
                    }
                }
            }
        } else {
            for c1 in 0..spec.child_dim_1 {
                let base_after_c1 =
                    physical_base
                        .checked_add(c1.checked_mul(spec.strides[spec.child_axis_1]).ok_or_else(
                            || anyhow::anyhow!("branch tensor offset overflows usize"),
                        )?)
                        .ok_or_else(|| anyhow::anyhow!("branch tensor offset overflows usize"))?;
                let left_c1_offset = left_rows
                    .checked_mul(c1)
                    .ok_or_else(|| anyhow::anyhow!("branch matrix offset overflows usize"))?;
                for c2 in 0..spec.child_dim_2 {
                    let base_after_c2 = base_after_c1
                        .checked_add(c2.checked_mul(spec.strides[spec.child_axis_2]).ok_or_else(
                            || anyhow::anyhow!("branch tensor offset overflows usize"),
                        )?)
                        .ok_or_else(|| anyhow::anyhow!("branch tensor offset overflows usize"))?;
                    let left_c2_offset = left_c1_offset
                        .checked_add(spec.parent_dim.checked_mul(c2).ok_or_else(|| {
                            anyhow::anyhow!("branch matrix offset overflows usize")
                        })?)
                        .ok_or_else(|| anyhow::anyhow!("branch matrix offset overflows usize"))?;
                    if spec.strides[spec.parent_axis] == 1 {
                        let end = base_after_c2.checked_add(spec.parent_dim).ok_or_else(|| {
                            anyhow::anyhow!("branch tensor offset overflows usize")
                        })?;
                        let slice = raw.get(base_after_c2..end).ok_or_else(|| {
                            anyhow::anyhow!(
                                "branch tensor offset {base_after_c2}..{end} is out of bounds"
                            )
                        })?;
                        left[left_c2_offset..left_c2_offset + spec.parent_dim]
                            .copy_from_slice(slice);
                    } else {
                        for parent in 0..spec.parent_dim {
                            let flat = base_after_c2
                                .checked_add(
                                    parent
                                        .checked_mul(spec.strides[spec.parent_axis])
                                        .ok_or_else(|| {
                                            anyhow::anyhow!("branch tensor offset overflows usize")
                                        })?,
                                )
                                .ok_or_else(|| {
                                    anyhow::anyhow!("branch tensor offset overflows usize")
                                })?;
                            left[left_c2_offset + parent] = *raw.get(flat).ok_or_else(|| {
                                anyhow::anyhow!("branch tensor offset {flat} is out of bounds")
                            })?;
                        }
                    }
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        contraction_diagnostics::inc(&contraction_diagnostics::BRANCH_SETUP_VALUES, left_len);
        Ok(Matrix::from_col_major_vec(
            left_rows,
            spec.child_dim_1,
            left,
        ))
    }

    /// Contracts a branch node's raw tensor data against two children's
    /// already-computed raw message columns, generalizing
    /// [`Self::grouped_chain_message_contraction`] from one child to two.
    ///
    /// Unlike the cartesian-product batching in `tensor4all-treeaci/frames.rs`'s
    /// analogous fix (`two_incoming_core_matrix_batched`), this evaluator's
    /// `points` are independent per-point assignments -- point `p` names one
    /// specific `(child1_assignment, child2_assignment)` pair, not every
    /// combination -- so only the first child's contraction reduces to a
    /// single shared-matrix `mat_mul_owned` call per physical-value group
    /// (the node's raw tensor slice at that physical value is the same for
    /// every point in the group, so all of the group's child-1 columns can
    /// be batched against it in one matmul). The second child's contraction
    /// cannot share a single matrix across points this way -- the
    /// intermediate from step one already differs per point -- so it is
    /// folded in via a vectorized accumulate loop over `child_dim_2`
    /// instead: still allocation-free, per-element-function-call-free array
    /// arithmetic, just not a single BLAS call.
    fn grouped_branch_message_contraction<T>(
        route: &mut CachedEvaluationRoute<'_, '_>,
        spec: BranchContractionSpec,
        raw: &[T],
        physical_values: &[usize],
        child1_columns: &[T],
        child2_columns: &[T],
    ) -> Result<Vec<T>>
    where
        T: BlasMul + Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
    {
        let BranchContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_axis_1,
            child_axis_2,
            parent_dim,
            child_dim_1,
            child_dim_2,
        } = spec;
        let point_count = physical_values.len();
        anyhow::ensure!(
            child1_columns.len() == point_count * child_dim_1,
            "branch child-1 message length {} does not match {} points x {} child values",
            child1_columns.len(),
            point_count,
            child_dim_1
        );
        anyhow::ensure!(
            child2_columns.len() == point_count * child_dim_2,
            "branch child-2 message length {} does not match {} points x {} child values",
            child2_columns.len(),
            point_count,
            child_dim_2
        );

        // Unlike the chain kernel (O(parent_dim * child_dim) work per point),
        // a branch step is O(parent_dim * child_dim_1 * child_dim_2) per
        // point -- an extra bond-dimension factor. At realistic bond
        // dimensions that alone already dwarfs BLAS's fixed per-call setup
        // cost even for a single point (a single Step-A `mat_mul_owned` call
        // is still a large, worthwhile matrix-vector product), so gating on
        // point/group count the way `CHAIN_BLAS_MIN_GROUP_POINTS` does is
        // the wrong heuristic here: it would force every small
        // floating-zone-walk batch (the common case) onto the scalar
        // fallback regardless of how large `scalar_work` actually is, which
        // was measured to *regress* wall time at bond=128 (see
        // `docs/worklogs/2026-08-22-treetn-branch-message-raw-path.md`).
        // `scalar_work` alone is the right gate.
        let scalar_work = parent_dim * child_dim_1 * child_dim_2 * point_count;
        if scalar_work < BRANCH_BLAS_WORK_THRESHOLD {
            #[cfg(feature = "diagnostics")]
            {
                contraction_diagnostics::inc(&contraction_diagnostics::BRANCH_SCALAR_CALLS, 1);
                contraction_diagnostics::inc(
                    &contraction_diagnostics::BRANCH_SCALAR_POINTS,
                    point_count,
                );
            }
            return scalar_branch_message_contraction(
                spec,
                raw,
                physical_values,
                child1_columns,
                child2_columns,
            );
        }

        let mut groups = HashMap::<usize, Vec<usize>>::new();
        for (point, &physical_value) in physical_values.iter().enumerate() {
            groups.entry(physical_value).or_default().push(point);
        }
        #[cfg(feature = "diagnostics")]
        let fast_axis_counter = if strides[parent_axis] == 1 {
            &contraction_diagnostics::BRANCH_FAST_AXIS_IS_PARENT
        } else if strides[child_axis_1] == 1 {
            &contraction_diagnostics::BRANCH_FAST_AXIS_IS_CHILD1
        } else if strides[child_axis_2] == 1 {
            &contraction_diagnostics::BRANCH_FAST_AXIS_IS_CHILD2
        } else {
            &contraction_diagnostics::BRANCH_FAST_AXIS_IS_PHYSICAL
        };
        #[cfg(feature = "diagnostics")]
        {
            contraction_diagnostics::inc(&contraction_diagnostics::BRANCH_BLAS_POINTS, point_count);
            contraction_diagnostics::inc(
                &contraction_diagnostics::BRANCH_BLAS_GROUPS,
                groups.len(),
            );
        }

        let mut output = vec![T::default(); point_count * parent_dim];
        for (physical_value, points) in groups {
            // Step A: fold in child 1 via one matmul for the whole group.
            // `left` is (parent_dim*child_dim_2) x child_dim_1, laid out so
            // child_dim_1 is the trailing (column) axis -- child2 rides
            // along in "rows", ordered slower than parent so a fixed c2
            // selects a contiguous parent_dim-length row block within each
            // column below.
            #[cfg(feature = "diagnostics")]
            let setup_start = std::time::Instant::now();
            // Loop-invariant-hoisted rewrite of the original per-(c1,c2,parent)
            // `axis_values` array + 4-term stride dot product: mathematically
            // identical to `physical_value*strides[physical_axis] +
            // c1*strides[child_axis_1] + c2*strides[child_axis_2] +
            // parent*strides[parent_axis]`, just accumulated incrementally
            // instead of recomputed from scratch on every one of the up to
            // `parent_dim*child_dim_1*child_dim_2` iterations -- issue #671's
            // downstream data showed this loop, not the BLAS call after it,
            // dominates a branch node's contraction time at realistic bond
            // dimensions (setup_ns >> matmul_ns).
            let left_len = parent_dim * child_dim_2 * child_dim_1;
            #[cfg(feature = "diagnostics")]
            contraction_diagnostics::inc(&contraction_diagnostics::BRANCH_SETUP_VALUES, left_len);
            let mut left = vec![T::default(); left_len];
            let physical_base = physical_value * strides[physical_axis];
            if strides[child_axis_2] == 1 {
                // `child2` happens to be `raw`'s fastest (contiguous) axis --
                // read the whole child_dim_2 run as one slice per (c1, parent)
                // instead of one bounds-checked element at a time. `child2` is
                // NOT `left`'s fastest write axis (`parent` is, per the layout
                // note above), so this scatters the contiguous read into a
                // parent_dim-strided write -- still a net win, since `raw`
                // (this node's own tensor) is what the earlier fast-axis
                // census (issue #671) found the reads on, not the much
                // smaller `left` buffer the writes land in. Same values, same
                // final `left` contents as the fully general branch below.
                for c1 in 0..child_dim_1 {
                    let base_after_c1 = physical_base + c1 * strides[child_axis_1];
                    let left_c1_offset = parent_dim * child_dim_2 * c1;
                    for parent in 0..parent_dim {
                        let base = base_after_c1 + parent * strides[parent_axis];
                        let end = base.checked_add(child_dim_2).ok_or_else(|| {
                            anyhow::anyhow!("branch tensor offset overflows usize")
                        })?;
                        let slice = raw.get(base..end).ok_or_else(|| {
                            anyhow::anyhow!("branch tensor offset {base}..{end} is out of bounds")
                        })?;
                        let left_base = left_c1_offset + parent;
                        for (c2, &value) in slice.iter().enumerate() {
                            left[left_base + parent_dim * c2] = value;
                        }
                    }
                }
            } else {
                for c1 in 0..child_dim_1 {
                    let base_after_c1 = physical_base + c1 * strides[child_axis_1];
                    let left_c1_offset = parent_dim * child_dim_2 * c1;
                    for c2 in 0..child_dim_2 {
                        let base_after_c2 = base_after_c1 + c2 * strides[child_axis_2];
                        let left_c2_offset = left_c1_offset + parent_dim * c2;
                        if strides[parent_axis] == 1 {
                            // `parent` happens to be `raw`'s fastest (contiguous)
                            // axis for this node's tensor -- read the whole
                            // parent_dim run as one slice instead of one
                            // bounds-checked element at a time. Same values,
                            // same write offsets as the general branch below;
                            // just lets the compiler emit a vectorized copy.
                            let end = base_after_c2.checked_add(parent_dim).ok_or_else(|| {
                                anyhow::anyhow!("branch tensor offset overflows usize")
                            })?;
                            let slice = raw.get(base_after_c2..end).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "branch tensor offset {base_after_c2}..{end} is out of bounds"
                                )
                            })?;
                            left[left_c2_offset..left_c2_offset + parent_dim]
                                .copy_from_slice(slice);
                        } else {
                            let mut flat = base_after_c2;
                            for parent in 0..parent_dim {
                                let left_offset = left_c2_offset + parent;
                                left[left_offset] = *raw.get(flat).ok_or_else(|| {
                                    anyhow::anyhow!("branch tensor offset {flat} is out of bounds")
                                })?;
                                flat += strides[parent_axis];
                            }
                        }
                    }
                }
            }
            let mut right = Vec::with_capacity(child_dim_1 * points.len());
            for &point in &points {
                let start = point * child_dim_1;
                right.extend_from_slice(&child1_columns[start..start + child_dim_1]);
            }
            #[cfg(feature = "diagnostics")]
            {
                contraction_diagnostics::add(
                    &contraction_diagnostics::BRANCH_SETUP_NS,
                    setup_start.elapsed(),
                );
            }
            #[cfg(feature = "diagnostics")]
            let matmul_start = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            {
                contraction_diagnostics::inc(&contraction_diagnostics::BRANCH_BLAS_CALLS, 1);
                contraction_diagnostics::inc(fast_axis_counter, 1);
            }
            let intermediate = route.mat_mul_owned(
                Matrix::from_col_major_vec(parent_dim * child_dim_2, child_dim_1, left),
                Matrix::from_col_major_vec(child_dim_1, points.len(), right),
            )?;
            let intermediate = intermediate.as_col_major_slice();
            #[cfg(feature = "diagnostics")]
            {
                contraction_diagnostics::add(
                    &contraction_diagnostics::BRANCH_MATMUL_NS,
                    matmul_start.elapsed(),
                );
            }

            // Step B: fold in child 2 via a vectorized accumulate over
            // child_dim_2 -- the intermediate's "rows" already interleave
            // parent (fast) and c2 (slow) per group column, so for a fixed
            // c2 the parent_dim-length slice at rows
            // [c2*parent_dim, (c2+1)*parent_dim) within each group-column is
            // contiguous.
            #[cfg(feature = "diagnostics")]
            let accumulate_start = std::time::Instant::now();
            for (column, &point) in points.iter().enumerate() {
                let column_base = column * parent_dim * child_dim_2;
                let destination = point * parent_dim;
                for c2 in 0..child_dim_2 {
                    let child2_value = child2_columns[point * child_dim_2 + c2];
                    let row_base = column_base + c2 * parent_dim;
                    for parent in 0..parent_dim {
                        output[destination + parent] +=
                            intermediate[row_base + parent] * child2_value;
                    }
                }
            }
            #[cfg(feature = "diagnostics")]
            {
                contraction_diagnostics::add(
                    &contraction_diagnostics::BRANCH_ACCUMULATE_NS,
                    accumulate_start.elapsed(),
                );
            }
        }
        Ok(output)
    }

    /// Runs the common grouped branch reduction when each physical group can
    /// obtain its already-prepared left operand through `group_gemm`.
    fn grouped_branch_message_from_gemm<T, F>(
        spec: BranchContractionSpec,
        physical_values: &[usize],
        child1_columns: &[T],
        child2_columns: &[T],
        mut group_gemm: F,
    ) -> Result<Vec<T>>
    where
        T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
        F: FnMut(usize, &[usize]) -> Result<Matrix<T>>,
    {
        let point_count = physical_values.len();
        let expected_child1_len = point_count
            .checked_mul(spec.child_dim_1)
            .ok_or_else(|| anyhow::anyhow!("branch child-1 message length overflows usize"))?;
        let expected_child2_len = point_count
            .checked_mul(spec.child_dim_2)
            .ok_or_else(|| anyhow::anyhow!("branch child-2 message length overflows usize"))?;
        anyhow::ensure!(
            child1_columns.len() == expected_child1_len,
            "branch child-1 message length {} does not match {} points x {} child values",
            child1_columns.len(),
            point_count,
            spec.child_dim_1
        );
        anyhow::ensure!(
            child2_columns.len() == expected_child2_len,
            "branch child-2 message length {} does not match {} points x {} child values",
            child2_columns.len(),
            point_count,
            spec.child_dim_2
        );

        let mut groups = HashMap::<usize, Vec<usize>>::new();
        for (point, &physical_value) in physical_values.iter().enumerate() {
            groups.entry(physical_value).or_default().push(point);
        }
        #[cfg(feature = "diagnostics")]
        let fast_axis_counter = if spec.strides[spec.parent_axis] == 1 {
            &contraction_diagnostics::BRANCH_FAST_AXIS_IS_PARENT
        } else if spec.strides[spec.child_axis_1] == 1 {
            &contraction_diagnostics::BRANCH_FAST_AXIS_IS_CHILD1
        } else if spec.strides[spec.child_axis_2] == 1 {
            &contraction_diagnostics::BRANCH_FAST_AXIS_IS_CHILD2
        } else {
            &contraction_diagnostics::BRANCH_FAST_AXIS_IS_PHYSICAL
        };
        #[cfg(feature = "diagnostics")]
        {
            contraction_diagnostics::inc(&contraction_diagnostics::BRANCH_BLAS_POINTS, point_count);
            contraction_diagnostics::inc(
                &contraction_diagnostics::BRANCH_BLAS_GROUPS,
                groups.len(),
            );
        }

        let output_len = point_count
            .checked_mul(spec.parent_dim)
            .ok_or_else(|| anyhow::anyhow!("branch output length overflows usize"))?;
        let mut output = vec![T::default(); output_len];
        for (physical_value, points) in groups {
            #[cfg(feature = "diagnostics")]
            let matmul_start = std::time::Instant::now();
            let intermediate = group_gemm(physical_value, &points)?;
            #[cfg(feature = "diagnostics")]
            {
                contraction_diagnostics::add(
                    &contraction_diagnostics::BRANCH_MATMUL_NS,
                    matmul_start.elapsed(),
                );
                contraction_diagnostics::inc(&contraction_diagnostics::BRANCH_BLAS_CALLS, 1);
                contraction_diagnostics::inc(fast_axis_counter, 1);
            }
            let expected_rows = spec
                .parent_dim
                .checked_mul(spec.child_dim_2)
                .ok_or_else(|| anyhow::anyhow!("branch GEMM row count overflows usize"))?;
            anyhow::ensure!(
                intermediate.nrows() == expected_rows && intermediate.ncols() == points.len(),
                "branch GEMM returned shape {}x{}, expected {}x{}",
                intermediate.nrows(),
                intermediate.ncols(),
                expected_rows,
                points.len()
            );
            let intermediate = intermediate.as_col_major_slice();

            // Keep the previous c2-major/parent-minor reduction order. The
            // prepared representation changes ownership and reuse only.
            #[cfg(feature = "diagnostics")]
            let accumulate_start = std::time::Instant::now();
            for (column, &point) in points.iter().enumerate() {
                let column_base = column * spec.parent_dim * spec.child_dim_2;
                let destination = point * spec.parent_dim;
                for c2 in 0..spec.child_dim_2 {
                    let child2_value = child2_columns[point * spec.child_dim_2 + c2];
                    let row_base = column_base + c2 * spec.parent_dim;
                    for parent in 0..spec.parent_dim {
                        output[destination + parent] +=
                            intermediate[row_base + parent] * child2_value;
                    }
                }
            }
            #[cfg(feature = "diagnostics")]
            contraction_diagnostics::add(
                &contraction_diagnostics::BRANCH_ACCUMULATE_NS,
                accumulate_start.elapsed(),
            );
        }
        Ok(output)
    }

    /// Contracts a branch with evaluator-owned prepared physical slices.
    fn grouped_branch_message_contraction_cached<T>(
        route: &mut CachedEvaluationRoute<'_, '_>,
        cache: &mut PreparedBranchSliceCache<V, T>,
        node: &V,
        parent: &V,
        scalar_kind: ScalarKind,
        batch: BranchMessageBatch<'_, T>,
    ) -> Result<Vec<T>>
    where
        T: BlasMul + Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
    {
        let BranchMessageBatch {
            spec,
            raw,
            physical_values,
            child1_columns,
            child2_columns,
        } = batch;
        let scalar_work = spec
            .parent_dim
            .checked_mul(spec.child_dim_1)
            .and_then(|value| value.checked_mul(spec.child_dim_2))
            .and_then(|value| value.checked_mul(physical_values.len()))
            .ok_or_else(|| anyhow::anyhow!("branch scalar work overflows usize"))?;
        if scalar_work < BRANCH_BLAS_WORK_THRESHOLD {
            #[cfg(feature = "diagnostics")]
            {
                contraction_diagnostics::inc(&contraction_diagnostics::BRANCH_SCALAR_CALLS, 1);
                contraction_diagnostics::inc(
                    &contraction_diagnostics::BRANCH_SCALAR_POINTS,
                    physical_values.len(),
                );
            }
            return scalar_branch_message_contraction(
                spec,
                raw,
                physical_values,
                child1_columns,
                child2_columns,
            );
        }

        Self::grouped_branch_message_from_gemm(
            spec,
            physical_values,
            child1_columns,
            child2_columns,
            |physical_value, points| {
                let key = (node.clone(), parent.clone(), physical_value, scalar_kind);
                let mut right = Vec::with_capacity(points.len() * spec.child_dim_1);
                for &point in points {
                    let start = point * spec.child_dim_1;
                    right.extend_from_slice(&child1_columns[start..start + spec.child_dim_1]);
                }
                cache.with_prepared_slice(
                    key,
                    || {
                        #[cfg(feature = "diagnostics")]
                        let setup_start = std::time::Instant::now();
                        let left = Self::prepare_branch_slice(spec, raw, physical_value);
                        #[cfg(feature = "diagnostics")]
                        if left.is_ok() {
                            contraction_diagnostics::add(
                                &contraction_diagnostics::BRANCH_SETUP_NS,
                                setup_start.elapsed(),
                            );
                        }
                        left
                    },
                    |left| {
                        let right =
                            Matrix::from_col_major_vec(spec.child_dim_1, points.len(), right);
                        route.mat_mul(left, &right)
                    },
                )
            },
        )
    }

    /// Computes an interior chain node's (exactly one child) message directly
    /// from raw data, generalizing [`Self::try_compute_leaf_message_raw`] the
    /// way `row_vector_times_matrix`
    /// (`crates/tensor4all-simplett/src/einsum_helper.rs`) generalizes a bare
    /// slice: contract the node's own raw tensor data against the child's
    /// already-computed message column without constructing an `IdxTensor`.
    ///
    /// The child's message must already be present in `messages` (true for
    /// any node reached in postorder) and must itself be real-valued and
    /// `IdxTensor`-backed by already-host-resident data -- true for every
    /// `StackedMessage` this evaluator produces, since both
    /// `IdxTensor::from_dense_any` (the cache-hit path) and a leaf's own
    /// slice-only tensor are host-resident by construction, unlike a
    /// `contract_with_options` result.
    ///
    /// Returns `Ok(None)` when `node` is not eligible (not exactly one
    /// physical index and one child, a complex-valued tensor, or an
    /// unexpected axis count), so the caller falls back to
    /// [`Self::compute_stacked_message`].
    #[allow(clippy::too_many_arguments)]
    fn try_compute_chain_message_raw(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Option<Vec<f64>>> {
        #[cfg(test)]
        if raw_message_kernels_disabled_for_test() {
            return Ok(None);
        }
        let entries = self
            .layout()
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
        let [child] = children else {
            return Ok(None);
        };

        let tensor = tensor_for_node(self.tree, node)?;
        if tensor.is_complex() {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != 3 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let Some(child_edge) = self.tree.edge_between(node, child) else {
            return Ok(None);
        };
        let Some(child_bond_index) = self.tree.bond_index(child_edge) else {
            return Ok(None);
        };
        let Some(child_axis) = tensor_indices
            .iter()
            .position(|idx| idx == child_bond_index)
        else {
            return Ok(None);
        };
        let Some(parent_axis) = (0..3).find(|&axis| axis != physical_axis && axis != child_axis)
        else {
            return Ok(None);
        };

        let dims = tensor.dims();
        let child_dim = dims[child_axis];
        let child_message = messages.get(child).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_chain_message_raw: missing message for child {:?}",
                child
            )
        })?;
        let child_assignment_batch = assignment_batches.get(child).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_chain_message_raw: missing assignment batch for child {:?}",
                child
            )
        })?;
        let Some(child_values) = gather_message_columns(
            child_message,
            child_dim,
            points,
            child_assignment_batch,
            |value| match value {
                CachedScalar::F64(value) => Some(*value),
                _ => None,
            },
            "TreeTNCachedEvaluator::try_compute_chain_message_raw",
        )?
        else {
            return Ok(None);
        };

        let parent_dim = dims[parent_axis];
        let strides = [
            1usize,
            dims[0],
            dims[0]
                .checked_mul(dims[1])
                .ok_or_else(|| anyhow::anyhow!("chain tensor strides overflow usize"))?,
        ];
        let spec = ChainContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_axis,
            parent_dim,
            child_dim,
        };
        let mut physical_values = Vec::with_capacity(points.len());
        for &point in points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_chain_message_raw",
            )?;
            physical_values.push(physical_value);
        }
        let child_columns = child_values;
        let raw = tensor.dense_host_values::<f64>()?;
        let result = Self::grouped_chain_message_contraction(
            route,
            spec,
            &raw,
            &physical_values,
            &child_columns,
        )?;
        Ok(Some(result))
    }

    /// Complex-valued counterpart of [`Self::try_compute_chain_message_raw`].
    ///
    /// This preserves the same postorder message and assignment-batch
    /// semantics as the real-valued helper, but performs the contraction with
    /// `Complex64` values directly in host memory.
    #[allow(clippy::too_many_arguments)]
    fn try_compute_chain_message_complex_raw(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Option<Vec<Complex64>>> {
        #[cfg(test)]
        if raw_message_kernels_disabled_for_test() {
            return Ok(None);
        }
        let entries = self
            .layout()
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
        let [child] = children else {
            return Ok(None);
        };

        let tensor = tensor_for_node(self.tree, node)?;
        if !tensor.is_complex() {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != 3 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let Some(child_edge) = self.tree.edge_between(node, child) else {
            return Ok(None);
        };
        let Some(child_bond_index) = self.tree.bond_index(child_edge) else {
            return Ok(None);
        };
        let Some(child_axis) = tensor_indices
            .iter()
            .position(|idx| idx == child_bond_index)
        else {
            return Ok(None);
        };
        let Some(parent_axis) = (0..3).find(|&axis| axis != physical_axis && axis != child_axis)
        else {
            return Ok(None);
        };

        let child_message = messages.get(child).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_chain_message_complex_raw: missing message for child {:?}",
                child
            )
        })?;
        let child_assignment_batch = assignment_batches.get(child).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_chain_message_complex_raw: missing assignment batch for child {:?}",
                child
            )
        })?;
        let dims = tensor.dims();
        let child_dim = dims[child_axis];
        let Some(child_values) = gather_message_columns(
            child_message,
            child_dim,
            points,
            child_assignment_batch,
            |value| match value {
                CachedScalar::C64(value) => Some(*value),
                _ => None,
            },
            "TreeTNCachedEvaluator::try_compute_chain_message_complex_raw",
        )?
        else {
            return Ok(None);
        };

        let parent_dim = dims[parent_axis];
        let strides = [
            1usize,
            dims[0],
            dims[0]
                .checked_mul(dims[1])
                .ok_or_else(|| anyhow::anyhow!("chain tensor strides overflow usize"))?,
        ];
        let spec = ChainContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_axis,
            parent_dim,
            child_dim,
        };
        let mut physical_values = Vec::with_capacity(points.len());
        for &point in points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_chain_message_complex_raw",
            )?;
            physical_values.push(physical_value);
        }
        let child_columns = child_values;
        let raw = tensor.dense_host_values::<Complex64>()?;
        let result = Self::grouped_chain_message_contraction(
            route,
            spec,
            &raw,
            &physical_values,
            &child_columns,
        )?;
        Ok(Some(result))
    }

    /// Computes a branch node's (exactly two children) message directly from
    /// raw data, generalizing [`Self::try_compute_chain_message_raw`] from
    /// one child to two via [`Self::grouped_branch_message_contraction`].
    ///
    /// Returns `Ok(None)` when `node` is not eligible (not exactly one
    /// physical index and exactly two children, a complex-valued tensor, or
    /// an unexpected axis count), so the caller falls back to
    /// [`Self::compute_stacked_message`].
    #[allow(clippy::too_many_arguments)]
    fn try_compute_branch_message_raw(
        &mut self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Option<Vec<f64>>> {
        #[cfg(test)]
        if raw_message_kernels_disabled_for_test() {
            return Ok(None);
        }
        let entries = self
            .layout()
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
        let [child_1, child_2] = children else {
            return Ok(None);
        };
        let Some(parent) = plan.parent.get(node).and_then(Clone::clone) else {
            return Ok(None);
        };

        let tensor = tensor_for_node(self.tree, node)?;
        if tensor.is_complex() {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != 4 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let Some(child_1_edge) = self.tree.edge_between(node, child_1) else {
            return Ok(None);
        };
        let Some(child_1_bond_index) = self.tree.bond_index(child_1_edge) else {
            return Ok(None);
        };
        let Some(child_axis_1) = tensor_indices
            .iter()
            .position(|idx| idx == child_1_bond_index)
        else {
            return Ok(None);
        };
        let Some(child_2_edge) = self.tree.edge_between(node, child_2) else {
            return Ok(None);
        };
        let Some(child_2_bond_index) = self.tree.bond_index(child_2_edge) else {
            return Ok(None);
        };
        let Some(child_axis_2) = tensor_indices
            .iter()
            .position(|idx| idx == child_2_bond_index)
        else {
            return Ok(None);
        };
        let Some(parent_axis) = (0..4)
            .find(|&axis| axis != physical_axis && axis != child_axis_1 && axis != child_axis_2)
        else {
            return Ok(None);
        };

        let child_1_message = messages.get(child_1).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_branch_message_raw: missing message for child {:?}",
                child_1
            )
        })?;
        let child_2_message = messages.get(child_2).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_branch_message_raw: missing message for child {:?}",
                child_2
            )
        })?;
        let child_1_assignment_batch = assignment_batches.get(child_1).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_branch_message_raw: missing assignment batch for child {:?}",
                child_1
            )
        })?;
        let child_2_assignment_batch = assignment_batches.get(child_2).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_branch_message_raw: missing assignment batch for child {:?}",
                child_2
            )
        })?;

        let dims = tensor.dims();
        let parent_dim = dims[parent_axis];
        let child_dim_1 = dims[child_axis_1];
        let child_dim_2 = dims[child_axis_2];
        let strides = [
            1usize,
            dims[0],
            dims[0]
                .checked_mul(dims[1])
                .ok_or_else(|| anyhow::anyhow!("branch tensor strides overflow usize"))?,
            dims[0]
                .checked_mul(dims[1])
                .and_then(|value| value.checked_mul(dims[2]))
                .ok_or_else(|| anyhow::anyhow!("branch tensor strides overflow usize"))?,
        ];
        let spec = BranchContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_axis_1,
            child_axis_2,
            parent_dim,
            child_dim_1,
            child_dim_2,
        };
        let Some(child_1_values) = gather_message_columns(
            child_1_message,
            child_dim_1,
            points,
            child_1_assignment_batch,
            |value| match value {
                CachedScalar::F64(value) => Some(*value),
                _ => None,
            },
            "TreeTNCachedEvaluator::try_compute_branch_message_raw child 1",
        )?
        else {
            return Ok(None);
        };
        let Some(child_2_values) = gather_message_columns(
            child_2_message,
            child_dim_2,
            points,
            child_2_assignment_batch,
            |value| match value {
                CachedScalar::F64(value) => Some(*value),
                _ => None,
            },
            "TreeTNCachedEvaluator::try_compute_branch_message_raw child 2",
        )?
        else {
            return Ok(None);
        };
        #[cfg(feature = "diagnostics")]
        let child_gather_started = std::time::Instant::now();
        let mut physical_values = Vec::with_capacity(points.len());
        for &point in points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_branch_message_raw",
            )?;
            physical_values.push(physical_value);
        }
        let child_1_columns = child_1_values;
        let child_2_columns = child_2_values;
        #[cfg(feature = "diagnostics")]
        {
            contraction_diagnostics::add(
                &contraction_diagnostics::BRANCH_CHILD_GATHER_NS,
                child_gather_started.elapsed(),
            );
            contraction_diagnostics::inc(
                &contraction_diagnostics::BRANCH_CHILD_GATHER_VALUES,
                child_1_columns.len() + child_2_columns.len(),
            );
        }
        let cache = &mut self.prepared_branch_slices_f64;
        let raw = tensor.dense_host_values::<f64>()?;
        let result = Self::grouped_branch_message_contraction_cached(
            route,
            cache,
            node,
            &parent,
            ScalarKind::F64,
            BranchMessageBatch {
                spec,
                raw: &raw,
                physical_values: &physical_values,
                child1_columns: &child_1_columns,
                child2_columns: &child_2_columns,
            },
        )?;
        Ok(Some(result))
    }

    /// Complex-valued counterpart of [`Self::try_compute_branch_message_raw`].
    #[allow(clippy::too_many_arguments)]
    fn try_compute_branch_message_complex_raw(
        &mut self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Option<Vec<Complex64>>> {
        #[cfg(test)]
        if raw_message_kernels_disabled_for_test() {
            return Ok(None);
        }
        let entries = self
            .layout()
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
        let [child_1, child_2] = children else {
            return Ok(None);
        };
        let Some(parent) = plan.parent.get(node).and_then(Clone::clone) else {
            return Ok(None);
        };

        let tensor = tensor_for_node(self.tree, node)?;
        if !tensor.is_complex() {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != 4 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let Some(child_1_edge) = self.tree.edge_between(node, child_1) else {
            return Ok(None);
        };
        let Some(child_1_bond_index) = self.tree.bond_index(child_1_edge) else {
            return Ok(None);
        };
        let Some(child_axis_1) = tensor_indices
            .iter()
            .position(|idx| idx == child_1_bond_index)
        else {
            return Ok(None);
        };
        let Some(child_2_edge) = self.tree.edge_between(node, child_2) else {
            return Ok(None);
        };
        let Some(child_2_bond_index) = self.tree.bond_index(child_2_edge) else {
            return Ok(None);
        };
        let Some(child_axis_2) = tensor_indices
            .iter()
            .position(|idx| idx == child_2_bond_index)
        else {
            return Ok(None);
        };
        let Some(parent_axis) = (0..4)
            .find(|&axis| axis != physical_axis && axis != child_axis_1 && axis != child_axis_2)
        else {
            return Ok(None);
        };

        let child_1_message = messages.get(child_1).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_branch_message_complex_raw: missing message for child {:?}",
                child_1
            )
        })?;
        let child_2_message = messages.get(child_2).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_branch_message_complex_raw: missing message for child {:?}",
                child_2
            )
        })?;
        let child_1_assignment_batch = assignment_batches.get(child_1).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_branch_message_complex_raw: missing assignment batch for child {:?}",
                child_1
            )
        })?;
        let child_2_assignment_batch = assignment_batches.get(child_2).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_branch_message_complex_raw: missing assignment batch for child {:?}",
                child_2
            )
        })?;

        let dims = tensor.dims();
        let parent_dim = dims[parent_axis];
        let child_dim_1 = dims[child_axis_1];
        let child_dim_2 = dims[child_axis_2];
        let strides = [
            1usize,
            dims[0],
            dims[0]
                .checked_mul(dims[1])
                .ok_or_else(|| anyhow::anyhow!("branch tensor strides overflow usize"))?,
            dims[0]
                .checked_mul(dims[1])
                .and_then(|value| value.checked_mul(dims[2]))
                .ok_or_else(|| anyhow::anyhow!("branch tensor strides overflow usize"))?,
        ];
        let spec = BranchContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_axis_1,
            child_axis_2,
            parent_dim,
            child_dim_1,
            child_dim_2,
        };
        let Some(child_1_values) = gather_message_columns(
            child_1_message,
            child_dim_1,
            points,
            child_1_assignment_batch,
            |value| match value {
                CachedScalar::C64(value) => Some(*value),
                _ => None,
            },
            "TreeTNCachedEvaluator::try_compute_branch_message_complex_raw child 1",
        )?
        else {
            return Ok(None);
        };
        let Some(child_2_values) = gather_message_columns(
            child_2_message,
            child_dim_2,
            points,
            child_2_assignment_batch,
            |value| match value {
                CachedScalar::C64(value) => Some(*value),
                _ => None,
            },
            "TreeTNCachedEvaluator::try_compute_branch_message_complex_raw child 2",
        )?
        else {
            return Ok(None);
        };
        #[cfg(feature = "diagnostics")]
        let child_gather_started = std::time::Instant::now();
        let mut physical_values = Vec::with_capacity(points.len());
        for &point in points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_branch_message_complex_raw",
            )?;
            physical_values.push(physical_value);
        }
        let child_1_columns = child_1_values;
        let child_2_columns = child_2_values;
        #[cfg(feature = "diagnostics")]
        {
            contraction_diagnostics::add(
                &contraction_diagnostics::BRANCH_CHILD_GATHER_NS,
                child_gather_started.elapsed(),
            );
            contraction_diagnostics::inc(
                &contraction_diagnostics::BRANCH_CHILD_GATHER_VALUES,
                child_1_columns.len() + child_2_columns.len(),
            );
        }
        let cache = &mut self.prepared_branch_slices_c64;
        let raw = tensor.dense_host_values::<Complex64>()?;
        let result = Self::grouped_branch_message_contraction_cached(
            route,
            cache,
            node,
            &parent,
            ScalarKind::C64,
            BranchMessageBatch {
                spec,
                raw: &raw,
                physical_values: &physical_values,
                child1_columns: &child_1_columns,
                child2_columns: &child_2_columns,
            },
        )?;
        Ok(Some(result))
    }

    /// Computes a rooted message for a node with three or more children,
    /// staying on the raw `Vec<T>` representation.
    ///
    /// The zero-, one-, and two-child cases keep their own kernels above;
    /// this is their arbitrary-degree generalization, and it is what lets
    /// [`Self::can_use_raw_messages`] admit a tree containing a node of
    /// coordination four or more at all. Before it existed, one such node
    /// forced *every* message of the whole tree onto the generic
    /// `IdxTensor` path.
    ///
    /// # Arguments
    ///
    /// * `kind` - the scalar kind `T` decodes; a node stored in another kind
    ///   returns `Ok(None)` so the caller falls back.
    /// * `decode` - reads one cached scalar as `T`, returning `None` for a
    ///   cached message of a different kind.
    ///
    /// # Returns
    ///
    /// `Ok(None)` whenever this node is not eligible (wrong child count,
    /// several physical indices, a foreign scalar kind, or a child message
    /// that cannot be read as `T`), which is not an error: the caller then
    /// uses the generic contraction.
    ///
    /// # Errors
    ///
    /// Returns an error when the tree is inconsistent with the plan, an
    /// offset overflows `usize`, or the backend rejects the contraction.
    #[allow(clippy::too_many_arguments)]
    fn try_compute_multi_branch_message_raw<T>(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
        kind: ScalarKind,
        decode: impl Fn(&CachedScalar) -> Option<T> + Copy,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Option<Vec<T>>>
    where
        T: TensorElement
            + StorageScalar
            + BlasMul
            + Copy
            + Default
            + std::ops::AddAssign
            + std::ops::Mul<Output = T>,
    {
        #[cfg(test)]
        if raw_message_kernels_disabled_for_test() {
            return Ok(None);
        }
        let entries = self
            .layout()
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
        if children.len() < 3 {
            return Ok(None);
        }
        if plan.parent.get(node).and_then(Clone::clone).is_none() {
            return Ok(None);
        }

        let tensor = tensor_for_node(self.tree, node)?;
        if tensor_scalar_kind(tensor)? != kind {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != children.len() + 2 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let mut child_axes = Vec::with_capacity(children.len());
        for child in children {
            let Some(edge) = self.tree.edge_between(node, child) else {
                return Ok(None);
            };
            let Some(bond) = self.tree.bond_index(edge) else {
                return Ok(None);
            };
            let Some(axis) = tensor_indices.iter().position(|idx| idx == bond) else {
                return Ok(None);
            };
            child_axes.push(axis);
        }
        let Some(parent_axis) = (0..tensor_indices.len())
            .find(|axis| *axis != physical_axis && !child_axes.contains(axis))
        else {
            return Ok(None);
        };

        let dims = tensor.dims();
        let mut strides = Vec::with_capacity(dims.len());
        let mut stride = 1usize;
        for &dim in &dims {
            strides.push(stride);
            stride = stride
                .checked_mul(dim)
                .ok_or_else(|| anyhow::anyhow!("multi-branch tensor strides overflow usize"))?;
        }
        let spec = MultiBranchContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_dims: child_axes.iter().map(|&axis| dims[axis]).collect(),
            child_axes,
            parent_dim: dims[parent_axis],
        };

        let mut child_columns = Vec::with_capacity(children.len());
        for (position, child) in children.iter().enumerate() {
            let message = messages.get(child).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::try_compute_multi_branch_message_raw: missing message for child {:?}",
                    child
                )
            })?;
            let assignment_batch = assignment_batches.get(child).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::try_compute_multi_branch_message_raw: missing assignment batch for child {:?}",
                    child
                )
            })?;
            let Some(columns) = gather_message_columns(
                message,
                spec.child_dims[position],
                points,
                assignment_batch,
                decode,
                "TreeTNCachedEvaluator::try_compute_multi_branch_message_raw child",
            )?
            else {
                return Ok(None);
            };
            child_columns.push(columns);
        }

        let mut physical_values = Vec::with_capacity(points.len());
        for &point in points {
            physical_values.push(value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_multi_branch_message_raw",
            )?);
        }

        let raw = tensor.dense_host_values::<T>()?;
        let result =
            multi_branch_message_contraction(route, &spec, &raw, &physical_values, &child_columns)?;
        Ok(Some(result))
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_generic_cached_message_values(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Vec<CachedScalar>> {
        let missing_message = self.compute_stacked_message(
            node,
            values,
            points,
            plan,
            assignment_batches,
            messages,
            route,
        )?;
        tensor_values_cached(
            missing_message
                .tensor
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("generic message did not materialize a tensor"))?,
        )
    }

    /// Computes `node`'s directed message toward its parent, consulting the
    /// per-node persistent cache first.
    ///
    /// Only the assignments genuinely missing from the cache are recomputed
    /// -- via a smaller call to [`Self::compute_stacked_message`] scoped to
    /// just those points -- and the result is merged with the cached columns
    /// for everything else, in the caller's original point order. A node
    /// whose whole batch is already cached skips computation entirely.
    ///
    /// Cache keys contain only the physical assignments in `node`'s rooted
    /// subtree. Sites in another component cannot affect this directed
    /// message and therefore do not belong in its cache key.
    fn get_or_compute_node_message(
        &mut self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &mut HashMap<V, StackedMessage>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<StackedMessage> {
        let assignment_batch = assignment_batches.get(node).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::evaluate_batched: missing assignments for {:?}",
                node
            )
        })?;
        self.last_stats.directed_message_count += assignment_batch.first_points.len();
        self.last_stats.batched_message_contract_count += 1;
        #[cfg(feature = "diagnostics")]
        let diag_start = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let diag_kernel_start = diagnostics::kernel_snapshot();
        #[cfg(feature = "diagnostics")]
        let (diag_node, diag_shape) = self.diagnostic_node(node)?;
        let points = assignment_batch.first_points.clone();
        let keys = assignment_batch.keys.clone();

        let Some(Some(parent)) = plan.parent.get(node) else {
            // No parent under this rooting: `node` is not the fixed centre but
            // has none, which should not happen for a postorder entry. Fall
            // back to the uncached path rather than fail the whole call.
            return self.compute_stacked_message(
                node,
                values,
                &points,
                plan,
                assignment_batches,
                messages,
                route,
            );
        };
        let parent = parent.clone();
        // A cached message is identified by its direction, not just its source
        // node. Re-rooting the tree changes some source -> parent directions,
        // while every unchanged direction remains an exact reusable message.
        let directed_edge = (node.clone(), parent.clone());
        let bond_index = match self.parent_bond_indices.get(&directed_edge) {
            Some(index) => index.clone(),
            None => {
                let edge = self.tree.edge_between(node, &parent).ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::evaluate_batched: no edge between {:?} and {:?}",
                        node,
                        parent
                    )
                })?;
                let index = self
                    .tree
                    .bond_index(edge)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::evaluate_batched: missing bond index between {:?} and {:?}",
                            node,
                            parent
                        )
                    })?
                    .clone();
                self.parent_bond_indices
                    .insert(directed_edge.clone(), index.clone());
                index
            }
        };
        let bond_dim = bond_index.dim();
        let assignment_index = DynIndex::new_dyn(keys.len());
        let message_cache_max_bytes = self.options.message_cache_max_bytes;

        // Split into hits and misses without computing anything yet. The
        // cache borrow must not outlive this block: `compute_stacked_message`
        // below needs `&self`, which conflicts with a live `&mut
        // self.message_caches` entry.
        let (hit_keys, missing_indices) = {
            let cache = self
                .message_caches
                .entry(directed_edge.clone())
                .or_insert_with(|| PackedMessageCache::new(bond_dim, message_cache_max_bytes));
            if let Some(positions) = cache.get_all_cached(&keys) {
                let mut data = Vec::with_capacity(bond_dim * keys.len());
                for position in positions {
                    data.extend_from_slice(cache.column(position));
                }
                let (tensor, raw_values) = if self.raw_messages {
                    (None, Some(data))
                } else {
                    (
                        Some(route.tensor_from_cached_values(
                            vec![bond_index, assignment_index.clone()],
                            data,
                        )?),
                        None,
                    )
                };
                self.last_stats.message_cache_hits += keys.len();
                #[cfg(feature = "diagnostics")]
                diagnostics::record_guard(
                    &diag_node,
                    diag_shape,
                    diagnostics::PhaseMeasurement {
                        elapsed: diag_start.elapsed(),
                        hits: keys.len() as u64,
                        misses: 0,
                        kernel: diagnostics::kernel_snapshot().since(diag_kernel_start),
                    },
                );
                return Ok(StackedMessage {
                    assignment_index,
                    tensor,
                    raw_values,
                });
            }
            let mut hit_keys = Vec::new();
            let mut missing_indices = Vec::new();
            for (i, key) in keys.iter().enumerate() {
                if cache.contains(key) {
                    hit_keys.push(key.clone());
                } else {
                    missing_indices.push(i);
                }
            }
            (hit_keys, missing_indices)
        };
        self.last_stats.message_cache_hits += hit_keys.len();
        self.last_stats.message_cache_misses += missing_indices.len();
        #[cfg(feature = "diagnostics")]
        let (diag_hits, diag_misses) = (hit_keys.len() as u64, missing_indices.len() as u64);
        if !hit_keys.is_empty() {
            // `entry().or_insert_with()` rather than `get_mut().expect(...)`:
            // the entry for `node` was inserted above and nothing removes
            // entries from `message_caches`. `or_insert_with`'s closure is
            // never invoked here, so this cannot fail rather than merely
            // being checked not to.
            let cache = self
                .message_caches
                .entry(directed_edge.clone())
                .or_insert_with(|| PackedMessageCache::new(bond_dim, message_cache_max_bytes));
            cache.record_hits(hit_keys.len());
        }

        // A parent cache hit returned above before this point, so descendants
        // are consulted only when at least one parent column is missing.
        let children = plan.children.get(node).cloned().unwrap_or_default();
        #[cfg(feature = "diagnostics")]
        let children_started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let children_kernel_start = diagnostics::kernel_snapshot();
        for child in children {
            if !messages.contains_key(&child) {
                let child_message = self.get_or_compute_node_message(
                    &child,
                    values,
                    plan,
                    assignment_batches,
                    messages,
                    route,
                )?;
                messages.insert(child, child_message);
            }
        }

        #[cfg(feature = "diagnostics")]
        let children_elapsed = children_started.elapsed();
        #[cfg(feature = "diagnostics")]
        let children_kernel = diagnostics::kernel_snapshot().since(children_kernel_start);

        // Compute only the missing points, as a batch of just that size.
        let missing_points = missing_indices
            .iter()
            .map(|&i| points[i])
            .collect::<Vec<_>>();
        let missing_keys = missing_indices
            .iter()
            .map(|&i| keys[i].clone())
            .collect::<Vec<_>>();
        let tensor_kind = tensor_scalar_kind(tensor_for_node(self.tree, node)?)?;
        let missing_values: Vec<CachedScalar> = match tensor_kind {
            ScalarKind::C64 => {
                let leaf = self.try_compute_leaf_message_complex_raw(
                    node,
                    values,
                    &missing_points,
                    route,
                )?;
                let chain = if leaf.is_some() {
                    leaf
                } else {
                    self.try_compute_chain_message_complex_raw(
                        node,
                        values,
                        &missing_points,
                        plan,
                        assignment_batches,
                        messages,
                        route,
                    )?
                };
                let branch = if chain.is_some() {
                    chain
                } else {
                    self.try_compute_branch_message_complex_raw(
                        node,
                        values,
                        &missing_points,
                        plan,
                        assignment_batches,
                        messages,
                        route,
                    )?
                };
                let raw_missing_values = if branch.is_some() {
                    branch
                } else {
                    self.try_compute_multi_branch_message_raw::<Complex64>(
                        node,
                        values,
                        &missing_points,
                        plan,
                        assignment_batches,
                        messages,
                        ScalarKind::C64,
                        |value| match value {
                            CachedScalar::C64(value) => Some(*value),
                            _ => None,
                        },
                        route,
                    )?
                };
                match raw_missing_values {
                    Some(raw) => raw.into_iter().map(CachedScalar::C64).collect(),
                    None => self.compute_generic_cached_message_values(
                        node,
                        values,
                        &missing_points,
                        plan,
                        assignment_batches,
                        messages,
                        route,
                    )?,
                }
            }
            ScalarKind::F64 => {
                let leaf =
                    self.try_compute_leaf_message_raw(node, values, &missing_points, route)?;
                let chain = if leaf.is_some() {
                    leaf
                } else {
                    self.try_compute_chain_message_raw(
                        node,
                        values,
                        &missing_points,
                        plan,
                        assignment_batches,
                        messages,
                        route,
                    )?
                };
                let branch = if chain.is_some() {
                    chain
                } else {
                    self.try_compute_branch_message_raw(
                        node,
                        values,
                        &missing_points,
                        plan,
                        assignment_batches,
                        messages,
                        route,
                    )?
                };
                let raw_missing_values = if branch.is_some() {
                    branch
                } else {
                    self.try_compute_multi_branch_message_raw::<f64>(
                        node,
                        values,
                        &missing_points,
                        plan,
                        assignment_batches,
                        messages,
                        ScalarKind::F64,
                        |value| match value {
                            CachedScalar::F64(value) => Some(*value),
                            _ => None,
                        },
                        route,
                    )?
                };
                match raw_missing_values {
                    Some(raw) => raw.into_iter().map(CachedScalar::F64).collect(),
                    None => self.compute_generic_cached_message_values(
                        node,
                        values,
                        &missing_points,
                        plan,
                        assignment_batches,
                        messages,
                        route,
                    )?,
                }
            }
            ScalarKind::F32 | ScalarKind::C32 => self.compute_generic_cached_message_values(
                node,
                values,
                &missing_points,
                plan,
                assignment_batches,
                messages,
                route,
            )?,
        };

        // Same reasoning as the `get_all_cached` call above: the entry for
        // `node` was inserted at the top of this function and nothing
        // between there and here removes it, so `entry().or_insert_with()`
        // cannot fail rather than merely being checked not to.
        let cache = self
            .message_caches
            .entry(directed_edge)
            .or_insert_with(|| PackedMessageCache::new(bond_dim, message_cache_max_bytes));
        let missing_slots = cache.get_or_compute_batch(&missing_keys, |request_keys| {
            request_keys
                .iter()
                .map(|request_key| {
                    let index = missing_keys.iter().position(|key| key == request_key).ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::evaluate_batched: missing key not found in this call's batch"
                        )
                    })?;
                    Ok(missing_values[index * bond_dim..(index + 1) * bond_dim].to_vec())
                })
                .collect::<Result<Vec<_>>>()
        })?;

        // Merge cached and uncached columns in the original point order. A
        // finite budget may return `CacheSlot::Uncached`; those values are
        // still valid for this call and must not be looked up again.
        let mut data = Vec::with_capacity(bond_dim * keys.len());
        let mut missing_slot_iter = missing_slots.into_iter();
        let mut missing_index_iter = missing_indices.into_iter().peekable();
        for (point_index, key) in keys.iter().enumerate() {
            if missing_index_iter.peek() == Some(&point_index) {
                missing_index_iter.next();
                let slot = missing_slot_iter.next().ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::evaluate_batched: missing cache slot for missing key"
                    )
                })?;
                match slot {
                    CacheSlot::Cached(position) => data.extend_from_slice(cache.column(position)),
                    CacheSlot::Uncached(column) => {
                        ensure!(
                            column.len() == bond_dim,
                            "TreeTNCachedEvaluator::evaluate_batched: uncached message column has length {}, expected {bond_dim}",
                            column.len()
                        );
                        data.extend_from_slice(&column);
                    }
                }
            } else {
                let position = cache.position(key).ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::evaluate_batched: cached key missing during merge"
                    )
                })?;
                data.extend_from_slice(cache.column(position));
            }
        }
        ensure!(
            missing_index_iter.next().is_none(),
            "TreeTNCachedEvaluator::evaluate_batched: missing point index after merge"
        );
        ensure!(
            missing_slot_iter.next().is_none(),
            "TreeTNCachedEvaluator::evaluate_batched: extra cache slots after merge"
        );
        let (tensor, raw_values) =
            if self.raw_messages {
                (None, Some(data))
            } else {
                (
                    Some(route.tensor_from_cached_values(
                        vec![bond_index, assignment_index.clone()],
                        data,
                    )?),
                    None,
                )
            };
        #[cfg(feature = "diagnostics")]
        diagnostics::record_guard(
            &diag_node,
            diag_shape,
            diagnostics::PhaseMeasurement {
                elapsed: diag_start.elapsed().saturating_sub(children_elapsed),
                hits: diag_hits,
                misses: diag_misses,
                kernel: diagnostics::kernel_snapshot()
                    .since(diag_kernel_start)
                    .since(children_kernel),
            },
        );
        Ok(StackedMessage {
            assignment_index,
            tensor,
            raw_values,
        })
    }

    /// Computes `node`'s directed message for exactly `points` (global point
    /// indices into `values`, in the order the result's assignment axis
    /// should carry) -- not necessarily the node's whole assignment batch, so
    /// a caller with a persistent cache can pass only the points it still
    /// needs to compute.
    #[allow(clippy::too_many_arguments)]
    fn compute_stacked_message(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<StackedMessage> {
        let assignment_index = DynIndex::new_dyn(points.len());
        let tensor = tensor_for_node(self.tree, node)?;
        let mut local_slices = Vec::with_capacity(points.len());
        for point in points.iter().copied() {
            let index_vals = self.index_vals_for_point(node, values, point)?;
            local_slices.push(slice_tensor(tensor, &index_vals, route).with_context(|| {
                format!(
                    "TreeTNCachedEvaluator::evaluate_batched: failed to slice message node {:?}",
                    node
                )
            })?);
        }
        let local_message =
            stack_tensors_with_assignment_index(&assignment_index, &local_slices, route)
                .with_context(|| {
                    format!(
                "TreeTNCachedEvaluator::evaluate_batched: failed to stack message node {:?}",
                node
            )
                })?;

        let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
        if children.is_empty() {
            return Ok(StackedMessage {
                assignment_index,
                tensor: Some(local_message),
                raw_values: None,
            });
        }

        let mut operands = Vec::with_capacity(1 + children.len());
        operands.push(local_message);
        for child in children {
            let child_assignment_batch = assignment_batches.get(child).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing child assignments for {:?}",
                    child
                )
            })?;
            let selected_assignments = points
                .iter()
                .map(|&point| {
                    child_assignment_batch
                        .point_to_assignment
                        .get(point)
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!("missing child assignment for point {point}")
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let child_message = messages.get(child).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing child message for {:?}",
                    child
                )
            })?;
            operands.push(gather_stacked_tensor(
                child_message
                    .tensor
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("child message did not materialize a tensor"))?,
                &child_message.assignment_index,
                &assignment_index,
                &selected_assignments,
                route,
            )?);
        }

        let retain = [assignment_index.clone()];
        let options = ContractionOptions::new().with_retain_indices(&retain);
        let operand_refs = operands.iter().collect::<Vec<_>>();
        let tensor = route.contract_nary(&operand_refs, options).context(
            "TreeTNCachedEvaluator::evaluate_batched: failed to contract batched directed message",
        )?;
        let tensor = ensure_assignment_axis_last(tensor, &assignment_index, route)?;
        Ok(StackedMessage {
            assignment_index,
            tensor: Some(tensor),
            raw_values: None,
        })
    }

    fn try_contract_leaf_center_from_raw(
        &self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        component: &ComponentBatch<V>,
        environment: &StackedMessage,
    ) -> Result<Option<Vec<CachedScalar>>> {
        let Some(raw_values) = environment.raw_values.as_ref() else {
            return Ok(None);
        };
        let entries = self
            .layout()
            .entries_by_node
            .get(center)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let center_tensor = tensor_for_node(self.tree, center)?;
        let center_indices = center_tensor.indices();
        if center_indices.len() != 2 {
            return Ok(None);
        }
        let Some(physical_axis) = center_indices
            .iter()
            .position(|index| index == &entry.index)
        else {
            return Ok(None);
        };
        let bond_axis = 1 - physical_axis;
        let bond_dim = center_tensor.dims()[bond_axis];
        if bond_dim == 0 || raw_values.len() % bond_dim != 0 {
            return Ok(None);
        }
        let assignment_dim = raw_values.len() / bond_dim;
        let center_dims = center_tensor.dims();
        let center_strides = [1usize, center_dims[0]];
        let n_points = values.shape()[1];

        match tensor_scalar_kind(center_tensor)? {
            ScalarKind::F32 | ScalarKind::C32 => return Ok(None),
            ScalarKind::F64 => {}
            ScalarKind::C64 => {
                let center_raw = center_tensor.dense_host_values::<Complex64>()?;
                let mut result = Vec::with_capacity(n_points);
                for point in 0..n_points {
                    let physical_value = value_at(
                        values,
                        entry.input_position,
                        point,
                        "TreeTNCachedEvaluator::try_contract_leaf_center_from_raw",
                    )?;
                    let assignment = component
                        .point_to_assignment
                        .get(point)
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!("missing centre assignment for point {point}")
                        })?;
                    ensure!(
                        assignment < assignment_dim,
                        "centre assignment {assignment} is out of bounds for dimension {assignment_dim}"
                    );
                    let mut sum = Complex64::new(0.0, 0.0);
                    for bond in 0..bond_dim {
                        let center_offset = physical_value * center_strides[physical_axis]
                            + bond * center_strides[bond_axis];
                        let environment_offset = assignment * bond_dim + bond;
                        let CachedScalar::C64(environment_value) = raw_values[environment_offset]
                        else {
                            return Ok(None);
                        };
                        sum += center_raw[center_offset] * environment_value;
                    }
                    result.push(CachedScalar::C64(sum));
                }
                return Ok(Some(result));
            }
        }

        let center_raw = center_tensor.dense_host_values::<f64>()?;
        let mut result = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_contract_leaf_center_from_raw",
            )?;
            let assignment = component
                .point_to_assignment
                .get(point)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing centre assignment for point {point}"))?;
            ensure!(
                assignment < assignment_dim,
                "centre assignment {assignment} is out of bounds for dimension {assignment_dim}"
            );
            let mut sum = 0.0;
            for bond in 0..bond_dim {
                let center_offset = physical_value * center_strides[physical_axis]
                    + bond * center_strides[bond_axis];
                let environment_offset = assignment * bond_dim + bond;
                let CachedScalar::F64(environment_value) = raw_values[environment_offset] else {
                    return Ok(None);
                };
                sum += center_raw[center_offset] * environment_value;
            }
            result.push(CachedScalar::F64(sum));
        }
        Ok(Some(result))
    }

    /// Evaluates a one-component centre on a path without constructing a
    /// backend contraction. A leaf centre has one physical axis and one bond
    /// axis, so each point is just a dot product of the sliced centre row with
    /// the cached incoming message column. Branching centres and nonstandard
    /// tensor layouts return `None` and retain the generic contraction path.
    fn try_contract_leaf_center_raw(
        &self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        component_batches: &[ComponentBatch<V>],
        environment_cache: &EnvironmentCache<V>,
    ) -> Result<Option<Vec<CachedScalar>>> {
        let [component] = component_batches else {
            return Ok(None);
        };
        let entries = self
            .layout()
            .entries_by_node
            .get(center)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let center_tensor = tensor_for_node(self.tree, center)?;
        let environment = environment_cache.get(&component.neighbor).ok_or_else(|| {
            anyhow::anyhow!("TreeTNCachedEvaluator::evaluate_batched: missing cached environment")
        })?;
        if environment.tensor.is_none() {
            return self.try_contract_leaf_center_from_raw(center, values, component, environment);
        }
        let center_indices = center_tensor.indices();
        let Some(environment_tensor) = environment.tensor.as_ref() else {
            return Ok(None);
        };
        let environment_indices = environment_tensor.indices();
        if center_indices.len() != 2 || environment_indices.len() != 2 {
            return Ok(None);
        }
        let Some(physical_axis) = center_indices
            .iter()
            .position(|index| index == &entry.index)
        else {
            return Ok(None);
        };
        let bond_axis = 1 - physical_axis;
        let bond_index = &center_indices[bond_axis];
        let Some(environment_bond_axis) = environment_indices
            .iter()
            .position(|index| index == bond_index)
        else {
            return Ok(None);
        };
        let assignment_axis = 1 - environment_bond_axis;
        let center_dims = center_tensor.dims();
        let environment_dims = environment_tensor.dims();
        let bond_dim = center_dims[bond_axis];
        if environment_dims[environment_bond_axis] != bond_dim {
            return Ok(None);
        }
        let assignment_dim = environment_dims[assignment_axis];
        let n_points = values.shape()[1];
        let center_strides = [1usize, center_dims[0]];
        let environment_strides = [1usize, environment_dims[0]];
        let physical_position = entry.input_position;

        let center_kind = tensor_scalar_kind(center_tensor)?;
        if center_kind != tensor_scalar_kind(environment_tensor)? {
            return Ok(None);
        }
        match center_kind {
            ScalarKind::F32 | ScalarKind::C32 => return Ok(None),
            ScalarKind::F64 => {}
            ScalarKind::C64 => {
                let center_raw = center_tensor.dense_host_values::<Complex64>()?;
                let environment_raw = environment_tensor.dense_host_values::<Complex64>()?;
                let mut result = Vec::with_capacity(n_points);
                for point in 0..n_points {
                    let physical_value = value_at(
                        values,
                        physical_position,
                        point,
                        "TreeTNCachedEvaluator::try_contract_leaf_center_raw",
                    )?;
                    let assignment = component
                        .point_to_assignment
                        .get(point)
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!("missing centre assignment for point {point}")
                        })?;
                    ensure!(
                        assignment < assignment_dim,
                        "centre assignment {assignment} is out of bounds for dimension {assignment_dim}"
                    );
                    let mut sum = Complex64::new(0.0, 0.0);
                    for bond in 0..bond_dim {
                        let center_offset = physical_value * center_strides[physical_axis]
                            + bond * center_strides[bond_axis];
                        let environment_offset = bond * environment_strides[environment_bond_axis]
                            + assignment * environment_strides[assignment_axis];
                        sum += center_raw[center_offset] * environment_raw[environment_offset];
                    }
                    result.push(CachedScalar::C64(sum));
                }
                return Ok(Some(result));
            }
        }

        let center_raw = center_tensor.dense_host_values::<f64>()?;
        let environment_raw = environment_tensor.dense_host_values::<f64>()?;
        let mut result = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let physical_value = value_at(
                values,
                physical_position,
                point,
                "TreeTNCachedEvaluator::try_contract_leaf_center_raw",
            )?;
            let assignment = component
                .point_to_assignment
                .get(point)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing centre assignment for point {point}"))?;
            ensure!(
                assignment < assignment_dim,
                "centre assignment {assignment} is out of bounds for dimension {assignment_dim}"
            );
            let mut sum = 0.0;
            for bond in 0..bond_dim {
                let center_offset = physical_value * center_strides[physical_axis]
                    + bond * center_strides[bond_axis];
                let environment_offset = bond * environment_strides[environment_bond_axis]
                    + assignment * environment_strides[assignment_axis];
                sum += center_raw[center_offset] * environment_raw[environment_offset];
            }
            result.push(CachedScalar::F64(sum));
        }
        Ok(Some(result))
    }

    /// Contracts a degree-2 or degree-3 center directly from raw core and
    /// directed-message buffers.
    ///
    /// Moving a floating-zone scan's center to its varying node makes every
    /// incoming component invariant across that batch. The message evaluator
    /// already represents those components as raw columns; keeping the center
    /// raw as well avoids falling back to the generic `IdxTensor` contraction
    /// solely because the varying node is not a leaf.
    fn try_contract_internal_center_raw(
        &self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        component_batches: &[ComponentBatch<V>],
        environment_cache: &EnvironmentCache<V>,
    ) -> Result<Option<Vec<CachedScalar>>> {
        if component_batches.len() < 2 {
            return Ok(None);
        }
        let entries = self
            .layout()
            .entries_by_node
            .get(center)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let center_tensor = tensor_for_node(self.tree, center)?;
        if center_tensor.indices().len() != component_batches.len() + 1 {
            return Ok(None);
        }
        let Some(physical_axis) = center_tensor
            .indices()
            .iter()
            .position(|index| index == &entry.index)
        else {
            return Ok(None);
        };
        let physical_values = (0..values.shape()[1])
            .map(|point| {
                value_at(
                    values,
                    entry.input_position,
                    point,
                    "TreeTNCachedEvaluator::try_contract_internal_center_raw",
                )
            })
            .collect::<Result<Vec<_>>>()?;

        match tensor_scalar_kind(center_tensor)? {
            ScalarKind::F32 | ScalarKind::C32 => return Ok(None),
            ScalarKind::F64 => {}
            ScalarKind::C64 => {
                let Some(components) = self.raw_center_components::<Complex64>(
                    center,
                    component_batches,
                    environment_cache,
                    |value| match value {
                        CachedScalar::C64(value) => Some(*value),
                        _ => None,
                    },
                )?
                else {
                    return Ok(None);
                };
                let core = center_tensor.dense_host_values::<Complex64>()?;
                let result = contract_raw_center(
                    &core,
                    &center_tensor.dims(),
                    physical_axis,
                    &physical_values,
                    &components,
                )?;
                return Ok(Some(result.into_iter().map(CachedScalar::C64).collect()));
            }
        }

        let Some(components) = self.raw_center_components::<f64>(
            center,
            component_batches,
            environment_cache,
            |value| match value {
                CachedScalar::F64(value) => Some(*value),
                _ => None,
            },
        )?
        else {
            return Ok(None);
        };
        let core = center_tensor.dense_host_values::<f64>()?;
        let result = contract_raw_center(
            &core,
            &center_tensor.dims(),
            physical_axis,
            &physical_values,
            &components,
        )?;
        Ok(Some(result.into_iter().map(CachedScalar::F64).collect()))
    }

    fn raw_center_components<T>(
        &self,
        center: &V,
        component_batches: &[ComponentBatch<V>],
        environment_cache: &EnvironmentCache<V>,
        convert: impl Fn(&CachedScalar) -> Option<T>,
    ) -> Result<Option<Vec<RawCenterComponent<T>>>> {
        let center_tensor = tensor_for_node(self.tree, center)?;
        let mut components = Vec::with_capacity(component_batches.len());
        for batch in component_batches {
            let edge = self
                .tree
                .edge_between(center, &batch.neighbor)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::evaluate_batched: no edge between {:?} and {:?}",
                        center,
                        batch.neighbor
                    )
                })?;
            let bond = self.tree.bond_index(edge).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing center bond index"
                )
            })?;
            let Some(axis) = center_tensor
                .indices()
                .iter()
                .position(|index| index == bond)
            else {
                return Ok(None);
            };
            let dim = bond.dim();
            let environment = environment_cache.get(&batch.neighbor).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing cached environment"
                )
            })?;
            let Some(raw_values) = environment.raw_values.as_ref() else {
                return Ok(None);
            };
            let Some(values) = raw_values.iter().map(&convert).collect::<Option<Vec<_>>>() else {
                return Ok(None);
            };
            ensure!(
                dim > 0 && values.len() % dim == 0,
                "raw center environment length {} is incompatible with bond dimension {dim}",
                values.len()
            );
            components.push(RawCenterComponent {
                axis,
                dim,
                point_to_assignment: batch.point_to_assignment.clone(),
                values,
            });
        }
        // Column-major storage makes the lowest-numbered bond axis the
        // smallest-stride one. Put it in the innermost contraction loop so a
        // center scan streams through the core instead of jumping between
        // distant columns at every scalar multiply.
        components.sort_by_key(|component| std::cmp::Reverse(component.axis));
        Ok(Some(components))
    }

    /// Uses an edge cut for a hinted batch and falls back to the existing
    /// vertex-center contraction when the tree has no edge. The cut is the
    /// first sorted neighbor of the hinted center; this makes the route
    /// deterministic while leaving the numerical contraction order of each
    /// directed message unchanged.
    ///
    /// [AI Supplied] The identity used here is re-derived from the tree
    /// contraction: after removing `(center, cut_neighbor)`, the two directed
    /// messages are vectors on the cut bond and their dot product is the full
    /// scalar. This is a tree generalization, not a claim about pseudocode in
    /// the ACI paper.
    fn contract_edge_cut_or_center(
        &mut self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        component_batches: &[ComponentBatch<V>],
        environment_cache: &EnvironmentCache<V>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Vec<CachedScalar>> {
        let Some(cut_batch) = component_batches.first() else {
            return self.contract_center_for_points(
                center,
                values,
                component_batches,
                environment_cache,
                route,
            );
        };
        let cut_neighbor = cut_batch.neighbor.clone();
        let cut_environment = environment_cache.get(&cut_neighbor).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator: missing cut environment for neighbor {:?}",
                cut_neighbor
            )
        })?;

        // Rooting at the other endpoint makes `center -> cut_neighbor` an
        // ordinary non-root message. The recursive message routine checks its
        // own cache before descending, so a fully warm call never touches the
        // other component messages.
        let reverse_plan = self.rooted_plan_for_center(&cut_neighbor)?;
        let reverse_edge = (center.clone(), cut_neighbor.clone());
        let reverse_assignments = if self.message_caches.contains_key(&reverse_edge) {
            let center_batch =
                self.build_directed_assignment_batch(center, &cut_neighbor, values)?;
            let fully_cached = self
                .message_caches
                .get(&reverse_edge)
                .is_some_and(|cache| center_batch.keys.iter().all(|key| cache.contains(key)));
            if fully_cached {
                HashMap::from([(center.clone(), center_batch)])
            } else {
                self.build_message_assignment_batches(&reverse_plan, values)?
            }
        } else {
            self.build_message_assignment_batches(&reverse_plan, values)?
        };
        let mut reverse_messages = HashMap::new();
        let center_message = self.get_or_compute_node_message(
            center,
            values,
            &reverse_plan,
            &reverse_assignments,
            &mut reverse_messages,
            route,
        )?;
        let center_assignment_batch = reverse_assignments.get(center).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator: missing reverse assignment batch for center {:?}",
                center
            )
        })?;

        let edge = self
            .tree
            .edge_between(center, &cut_neighbor)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator: no edge between {:?} and {:?}",
                    center,
                    cut_neighbor
                )
            })?;
        let bond_index = self.tree.bond_index(edge).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator: missing bond index for edge {:?}->{:?}",
                center,
                cut_neighbor
            )
        })?;
        let bond_dim = bond_index.dim();
        if bond_dim == 0 {
            return self.contract_center_for_points(
                center,
                values,
                component_batches,
                environment_cache,
                route,
            );
        }

        let scalar_kind = tensor_scalar_kind(tensor_for_node(self.tree, center)?)?;
        if stacked_message_scalar_kind(cut_environment)? != Some(scalar_kind)
            || stacked_message_scalar_kind(&center_message)? != Some(scalar_kind)
        {
            return self.contract_center_for_points(
                center,
                values,
                component_batches,
                environment_cache,
                route,
            );
        }
        let assembly = EdgeCutAssembly {
            values,
            cut_batch,
            cut_environment,
            center_assignment_batch,
            center_message: &center_message,
            bond_dim,
        };

        match scalar_kind {
            ScalarKind::F32 => self
                .contract_edge_cut_typed(&assembly, |value| match value {
                    CachedScalar::F32(value) => Some(*value),
                    _ => None,
                })
                .map(|values| values.into_iter().map(CachedScalar::F32).collect()),
            ScalarKind::F64 => self
                .contract_edge_cut_typed(&assembly, |value| match value {
                    CachedScalar::F64(value) => Some(*value),
                    _ => None,
                })
                .map(|values| values.into_iter().map(CachedScalar::F64).collect()),
            ScalarKind::C32 => self
                .contract_edge_cut_typed(&assembly, |value| match value {
                    CachedScalar::C32(value) => Some(*value),
                    _ => None,
                })
                .map(|values| values.into_iter().map(CachedScalar::C32).collect()),
            ScalarKind::C64 => self
                .contract_edge_cut_typed(&assembly, |value| match value {
                    CachedScalar::C64(value) => Some(*value),
                    _ => None,
                })
                .map(|values| values.into_iter().map(CachedScalar::C64).collect()),
        }
    }

    fn contract_edge_cut_typed<T>(
        &mut self,
        assembly: &EdgeCutAssembly<'_, V>,
        decode: impl Fn(&CachedScalar) -> Option<T>,
    ) -> Result<Vec<T>>
    where
        T: TensorElement
            + StorageScalar
            + Copy
            + Default
            + std::ops::AddAssign
            + std::ops::Mul<Output = T>,
    {
        let left_values = stacked_message_values_typed(assembly.cut_environment, &decode)?;
        let right_values = stacked_message_values_typed(assembly.center_message, &decode)?;
        ensure_typed_message_storage_shape(
            &left_values,
            assembly.cut_environment.assignment_index.dim(),
            assembly.bond_dim,
            "cut environment",
        )?;
        ensure_typed_message_storage_shape(
            &right_values,
            assembly.center_message.assignment_index.dim(),
            assembly.bond_dim,
            "reverse center message",
        )?;

        let n_points = assembly.values.shape()[1];
        let mut result = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let left_assignment = assembly
                .cut_batch
                .point_to_assignment
                .get(point)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing cut assignment for point {point}"))?;
            let right_assignment = assembly
                .center_assignment_batch
                .point_to_assignment
                .get(point)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing reverse assignment for point {point}"))?;
            let left_start = left_assignment
                .checked_mul(assembly.bond_dim)
                .ok_or_else(|| {
                    anyhow::anyhow!("cut assignment offset overflows usize for point {point}")
                })?;
            let right_start = right_assignment
                .checked_mul(assembly.bond_dim)
                .ok_or_else(|| {
                    anyhow::anyhow!("reverse assignment offset overflows usize for point {point}")
                })?;
            let left_end = left_start
                .checked_add(assembly.bond_dim)
                .ok_or_else(|| anyhow::anyhow!("cut column end overflows usize"))?;
            let right_end = right_start
                .checked_add(assembly.bond_dim)
                .ok_or_else(|| anyhow::anyhow!("reverse column end overflows usize"))?;
            let left = left_values.get(left_start..left_end).ok_or_else(|| {
                anyhow::anyhow!("cut message column is out of bounds for point {point}")
            })?;
            let right = right_values.get(right_start..right_end).ok_or_else(|| {
                anyhow::anyhow!("reverse message column is out of bounds for point {point}")
            })?;
            result.push(column_dot(left, right, |value| Some(*value))?);
            self.last_stats.warm_edge_cut_assembly_visits += assembly.bond_dim;
        }
        Ok(result)
    }

    fn contract_center_for_points(
        &self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        component_batches: &[ComponentBatch<V>],
        environment_cache: &EnvironmentCache<V>,
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Vec<CachedScalar>> {
        let n_points = values.shape()[1];
        if n_points == 0 {
            return Ok(Vec::new());
        }
        if let Some(result) =
            self.try_contract_leaf_center_raw(center, values, component_batches, environment_cache)?
        {
            return Ok(result);
        }
        let raw_internal_result = self.try_contract_internal_center_raw(
            center,
            values,
            component_batches,
            environment_cache,
        );
        if let Some(result) = raw_internal_result? {
            return Ok(result);
        }
        let center_entries = self
            .layout()
            .entries_by_node
            .get(center)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let center_tensor = tensor_for_node(self.tree, center)?;
        let point_index = DynIndex::new_dyn(n_points);
        let mut center_slices = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let center_index_vals = center_entries
                .iter()
                .map(|entry| {
                    let value = value_at(
                        values,
                        entry.input_position,
                        point,
                        "TreeTNCachedEvaluator::evaluate_batched",
                    )?;
                    Ok((entry.index.clone(), value))
                })
                .collect::<Result<Vec<_>>>()?;
            validate_index_vals(
                &center_index_vals,
                "TreeTNCachedEvaluator::evaluate_batched",
            )?;
            center_slices.push(
                slice_tensor(center_tensor, &center_index_vals, route).context(
                    "TreeTNCachedEvaluator::evaluate_batched: failed to slice center tensor",
                )?,
            );
        }

        let mut operands = Vec::with_capacity(1 + component_batches.len());
        operands.push(
            stack_tensors_with_assignment_index(&point_index, &center_slices, route).context(
                "TreeTNCachedEvaluator::evaluate_batched: failed to stack center tensor",
            )?,
        );

        for batch in component_batches {
            let environment = environment_cache.get(&batch.neighbor).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing cached environment"
                )
            })?;
            operands.push(
                gather_stacked_tensor(
                    environment.tensor.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::evaluate_batched: environment did not materialize a tensor"
                        )
                    })?,
                    &environment.assignment_index,
                    &point_index,
                    &batch.point_to_assignment,
                    route,
                )
                .context(
                    "TreeTNCachedEvaluator::evaluate_batched: failed to gather center environment",
                )?,
            );
        }

        let result_tensor = if operands.len() == 1 {
            operands.remove(0)
        } else {
            let retain = [point_index.clone()];
            let options = ContractionOptions::new().with_retain_indices(&retain);
            let operand_refs = operands.iter().collect::<Vec<_>>();
            route.contract_nary(&operand_refs, options).context(
                "TreeTNCachedEvaluator::evaluate_batched: failed to contract center batch",
            )?
        };
        let result_tensor = ensure_assignment_axis_last(result_tensor, &point_index, route)?;
        anyhow::ensure!(
            result_tensor.indices() == std::slice::from_ref(&point_index),
            "TreeTNCachedEvaluator::evaluate_batched: center contraction left non-scalar indices {:?}",
            result_tensor.indices()
        );

        tensor_values_cached(&result_tensor)
    }

    fn index_vals_for_point(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        point: usize,
    ) -> Result<Vec<(DynIndex, usize)>> {
        let Some(entries) = self.layout().entries_by_node.get(node) else {
            return Ok(Vec::new());
        };
        entries
            .iter()
            .map(|entry| {
                let value = value_at(
                    values,
                    entry.input_position,
                    point,
                    "TreeTNCachedEvaluator::evaluate_batched",
                )?;
                Ok((entry.index.clone(), value))
            })
            .collect()
    }

    #[cfg(test)]
    fn stats_for_test(&self) -> CachedEvaluationStats {
        self.last_stats.clone()
    }
}

fn tensor_from_cached_values(
    indices: Vec<DynIndex>,
    values: Vec<CachedScalar>,
) -> Result<IdxTensor> {
    if let Some(data) = values
        .iter()
        .map(|value| match value {
            CachedScalar::F32(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    {
        return Ok(IdxTensor::from_dense(indices, data)?);
    }
    if let Some(data) = values
        .iter()
        .map(|value| match value {
            CachedScalar::F64(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    {
        return Ok(IdxTensor::from_dense(indices, data)?);
    }
    if let Some(data) = values
        .iter()
        .map(|value| match value {
            CachedScalar::C32(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    {
        return Ok(IdxTensor::from_dense(indices, data)?);
    }
    if let Some(data) = values
        .iter()
        .map(|value| match value {
            CachedScalar::C64(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    {
        return Ok(IdxTensor::from_dense(indices, data)?);
    }
    Ok(IdxTensor::from_dense_any(
        indices,
        values.into_iter().map(CachedScalar::into_any).collect(),
    )?)
}

/// Publish cached message values as compact host storage without entering a runtime.
///
/// The session route uses this instead of [`tensor_from_cached_values`], whose
/// `IdxTensor::from_dense` publication adopts an eager value from the
/// process-global runtime. Values keep their own dtype, and the result is
/// compact, so later indexed operations on it stay runtime-free.
fn tensor_from_cached_values_compact(
    indices: Vec<DynIndex>,
    values: Vec<CachedScalar>,
) -> Result<IdxTensor> {
    if let Some(data) = values
        .iter()
        .map(|value| match value {
            CachedScalar::F32(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    {
        return IdxTensor::from_dense_compact(indices, data).map_err(anyhow::Error::from);
    }
    if let Some(data) = values
        .iter()
        .map(|value| match value {
            CachedScalar::F64(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    {
        return IdxTensor::from_dense_compact(indices, data).map_err(anyhow::Error::from);
    }
    if let Some(data) = values
        .iter()
        .map(|value| match value {
            CachedScalar::C32(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    {
        return IdxTensor::from_dense_compact(indices, data).map_err(anyhow::Error::from);
    }
    if let Some(data) = values
        .iter()
        .map(|value| match value {
            CachedScalar::C64(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    {
        return IdxTensor::from_dense_compact(indices, data).map_err(anyhow::Error::from);
    }
    Err(anyhow::anyhow!(
        "cached message contains mixed scalar kinds; a session message needs one dtype"
    ))
}

// [AI Supplied] Test-only observation seam for the Hiroshi #671 work-count
// invariant. Production builds compile out both storage and loop increments.
#[cfg(test)]
thread_local! {
    static RAW_CENTER_CORE_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// `(grouped, flat)` calls of the arbitrary-degree branch kernel, so a
    /// test can pin which route a configuration actually takes. Thread-local
    /// rather than atomic because the test harness runs tests in parallel
    /// threads of one process.
    static MULTI_BRANCH_ROUTES: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

#[cfg(test)]
fn record_multi_branch_route(grouped: bool) {
    let (grouped_calls, flat_calls) = MULTI_BRANCH_ROUTES.get();
    if grouped {
        MULTI_BRANCH_ROUTES.set((grouped_calls + 1, flat_calls));
    } else {
        MULTI_BRANCH_ROUTES.set((grouped_calls, flat_calls + 1));
    }
}

#[cfg(test)]
fn reset_multi_branch_routes_for_test() {
    MULTI_BRANCH_ROUTES.set((0, 0));
}

#[cfg(test)]
fn multi_branch_routes_for_test() -> (usize, usize) {
    MULTI_BRANCH_ROUTES.get()
}

#[cfg(test)]
fn reset_raw_center_core_visits_for_test() {
    RAW_CENTER_CORE_VISITS.set(0);
}

#[cfg(test)]
fn raw_center_core_visits_for_test() -> usize {
    RAW_CENTER_CORE_VISITS.get()
}

fn contract_raw_center<T>(
    core: &[T],
    dims: &[usize],
    physical_axis: usize,
    physical_values: &[usize],
    components: &[RawCenterComponent<T>],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    ensure!(
        components.len() >= 2,
        "raw internal center needs at least two components"
    );
    ensure!(
        physical_axis < dims.len(),
        "raw center physical axis is out of bounds"
    );
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in dims {
        strides.push(stride);
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("raw center strides overflow usize"))?;
    }
    ensure!(
        core.len() == stride,
        "raw center core length does not match its shape"
    );
    for component in components {
        ensure!(
            component.axis < dims.len() && dims[component.axis] == component.dim,
            "raw center component shape does not match its core axis"
        );
        ensure!(
            component.axis != physical_axis,
            "raw center component axis overlaps its physical axis"
        );
        ensure!(
            component.point_to_assignment.len() == physical_values.len(),
            "raw center component assignment count does not match point count"
        );
        ensure!(
            component.dim > 0 && component.values.len() % component.dim == 0,
            "raw center component values do not contain complete bond columns"
        );
        let assignment_count = component.values.len() / component.dim;
        ensure!(
            component
                .point_to_assignment
                .iter()
                .all(|&assignment| assignment < assignment_count),
            "raw center component assignment is out of bounds"
        );
    }
    for (position, component) in components.iter().enumerate() {
        ensure!(
            components[position + 1..]
                .iter()
                .all(|other| component.axis != other.axis),
            "raw center components overlap the same core axis"
        );
    }

    #[cfg(test)]
    let mut core_visits = 0usize;
    let mut output = Vec::with_capacity(physical_values.len());
    for (point, &physical) in physical_values.iter().enumerate() {
        ensure!(
            physical < dims[physical_axis],
            "raw center physical value is out of bounds"
        );
        let physical_offset = physical * strides[physical_axis];
        let sum = fold_raw_center_components(
            core,
            &strides,
            components,
            point,
            0,
            physical_offset,
            #[cfg(test)]
            &mut core_visits,
        );
        output.push(sum);
    }
    #[cfg(test)]
    RAW_CENTER_CORE_VISITS.set(core_visits);
    Ok(output)
}

/// Sums one point's center contraction over the components from `level` on.
///
/// The descent keeps the nesting the two- and three-component cases used
/// before it replaced them: each level multiplies its own component value
/// onto the sum accumulated by the levels inside it, so the arithmetic and
/// its order are unchanged where those cases still apply, and the same
/// nesting simply continues for a center of higher coordination. Components
/// arrive sorted by descending axis, so the innermost loop is the
/// smallest-stride one.
fn fold_raw_center_components<T>(
    core: &[T],
    strides: &[usize],
    components: &[RawCenterComponent<T>],
    point: usize,
    level: usize,
    offset: usize,
    #[cfg(test)] core_visits: &mut usize,
) -> T
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let component = &components[level];
    let assignment = component.point_to_assignment[point];
    let innermost = level + 1 == components.len();
    let mut sum = T::default();
    for bond in 0..component.dim {
        let value = component.values[assignment * component.dim + bond];
        let offset = offset + bond * strides[component.axis];
        let inner = if innermost {
            #[cfg(test)]
            {
                *core_visits += 1;
            }
            core[offset]
        } else {
            fold_raw_center_components(
                core,
                strides,
                components,
                point,
                level + 1,
                offset,
                #[cfg(test)]
                core_visits,
            )
        };
        sum += inner * value;
    }
    sum
}

/// One arbitrary-degree branch node's contraction geometry.
///
/// The counterpart of [`BranchContractionSpec`] for a node with any number of
/// rooted children. `child_axes` and `child_dims` are in rooted-child order,
/// so `child_axes[0]` is the child whose bond the grouped kernel folds with
/// its shared GEMM and the rest are folded per point afterwards.
#[derive(Clone, Debug)]
struct MultiBranchContractionSpec {
    strides: Vec<usize>,
    physical_axis: usize,
    parent_axis: usize,
    child_axes: Vec<usize>,
    parent_dim: usize,
    child_dims: Vec<usize>,
}

impl MultiBranchContractionSpec {
    /// Elements of the prepared left operand of one physical group.
    ///
    /// # Errors
    ///
    /// Returns an error when the product overflows `usize`.
    fn prepared_elements(&self) -> Result<usize> {
        self.child_dims
            .iter()
            .try_fold(self.parent_dim, |product, &dim| {
                product
                    .checked_mul(dim)
                    .ok_or_else(|| anyhow::anyhow!("multi-branch slice size overflows usize"))
            })
    }

    /// Destination weights of the prepared left operand, in child order.
    ///
    /// The prepared operand is column-major with `parent` fastest, then the
    /// children from last to second, and `child_axes[0]` as the trailing
    /// (column) axis. Folding the children in order therefore always reduces
    /// the slowest remaining axis, so every step reads a contiguous block and
    /// no intermediate transpose is needed -- the obstacle that kept the
    /// two-child kernel from being generalized when it was written (see
    /// `docs/worklogs/2026-08-22-treetn-branch-message-raw-path.md`).
    ///
    /// # Errors
    ///
    /// Returns an error when a weight overflows `usize`.
    fn destination_weights(&self) -> Result<Vec<usize>> {
        let mut weights = vec![0usize; self.child_dims.len()];
        let mut weight = self.parent_dim;
        for (position, &dim) in self.child_dims.iter().enumerate().skip(1).rev() {
            weights[position] = weight;
            weight = weight
                .checked_mul(dim)
                .ok_or_else(|| anyhow::anyhow!("multi-branch slice weight overflows usize"))?;
        }
        // `child_axes[0]` is the GEMM's column axis, so its weight is the row
        // count of the prepared operand.
        if let Some(first) = weights.first_mut() {
            *first = weight;
        }
        Ok(weights)
    }
}

/// Prepares one physical slice of an arbitrary-degree branch node as the
/// column-major left operand of the grouped GEMM.
///
/// The result has `parent_dim * prod(child_dims[1..])` rows and
/// `child_dims[0]` columns; see
/// [`MultiBranchContractionSpec::destination_weights`] for the layout and why
/// it is the one that makes every later fold contiguous.
///
/// # Errors
///
/// Returns an error when an offset overflows `usize` or falls outside `raw`.
fn prepare_multi_branch_slice<T>(
    spec: &MultiBranchContractionSpec,
    raw: &[T],
    physical_value: usize,
) -> Result<Vec<T>>
where
    T: Copy + Default,
{
    let length = spec.prepared_elements()?;
    let weights = spec.destination_weights()?;
    let mut left = vec![T::default(); length];
    let physical_base = physical_value
        .checked_mul(spec.strides[spec.physical_axis])
        .ok_or_else(|| anyhow::anyhow!("multi-branch tensor offset overflows usize"))?;
    let parent_stride = spec.strides[spec.parent_axis];
    let parent_dim = spec.parent_dim;
    let degree = spec.child_dims.len();
    let mut coordinates = vec![0usize; degree];
    loop {
        let mut source = physical_base;
        let mut destination = 0usize;
        for (position, &coordinate) in coordinates.iter().enumerate() {
            source = coordinate
                .checked_mul(spec.strides[spec.child_axes[position]])
                .and_then(|offset| source.checked_add(offset))
                .ok_or_else(|| anyhow::anyhow!("multi-branch tensor offset overflows usize"))?;
            destination = coordinate
                .checked_mul(weights[position])
                .and_then(|offset| destination.checked_add(offset))
                .ok_or_else(|| anyhow::anyhow!("multi-branch slice offset overflows usize"))?;
        }
        if parent_stride == 1 {
            let end = source
                .checked_add(parent_dim)
                .ok_or_else(|| anyhow::anyhow!("multi-branch tensor offset overflows usize"))?;
            let block = raw.get(source..end).ok_or_else(|| {
                anyhow::anyhow!("multi-branch tensor offset {source}..{end} is out of bounds")
            })?;
            left[destination..destination + parent_dim].copy_from_slice(block);
        } else {
            let mut flat = source;
            for parent in 0..parent_dim {
                left[destination + parent] = *raw.get(flat).ok_or_else(|| {
                    anyhow::anyhow!("multi-branch tensor offset {flat} is out of bounds")
                })?;
                flat = flat
                    .checked_add(parent_stride)
                    .ok_or_else(|| anyhow::anyhow!("multi-branch tensor offset overflows usize"))?;
            }
        }

        // Odometer over the child bonds, least significant last so the walk
        // covers every combination exactly once.
        let mut position = degree;
        loop {
            if position == 0 {
                return Ok(left);
            }
            position -= 1;
            coordinates[position] += 1;
            if coordinates[position] < spec.child_dims[position] {
                break;
            }
            coordinates[position] = 0;
        }
    }
}

/// Folds one point's remaining children into its parent-message column.
///
/// `column` enters as the prepared operand's rows for one point, laid out
/// `[parent, child_{q-1}, ..., child_1]` with `child_1` slowest, and each fold
/// reduces the slowest remaining axis in place of a transpose.
///
/// # Errors
///
/// Returns an error when a child's column is missing or the buffer length is
/// not divisible by the child dimension it is about to fold.
fn fold_multi_branch_children<T>(
    spec: &MultiBranchContractionSpec,
    mut column: Vec<T>,
    child_columns: &[Vec<T>],
    point: usize,
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    for (position, &dim) in spec.child_dims.iter().enumerate().skip(1) {
        anyhow::ensure!(
            dim > 0 && column.len().is_multiple_of(dim),
            "multi-branch fold buffer of length {} is incompatible with child dimension {dim}",
            column.len()
        );
        let block = column.len() / dim;
        let values = child_columns.get(position).ok_or_else(|| {
            anyhow::anyhow!("multi-branch fold is missing child {position}'s columns")
        })?;
        let base = point
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("multi-branch child offset overflows usize"))?;
        let mut folded = vec![T::default(); block];
        for coordinate in 0..dim {
            let value = *values
                .get(base + coordinate)
                .ok_or_else(|| anyhow::anyhow!("multi-branch child column is out of bounds"))?;
            let start = coordinate * block;
            for (target, source) in folded.iter_mut().zip(&column[start..start + block]) {
                *target += *source * value;
            }
        }
        column = folded;
    }
    Ok(column)
}

/// Contracts an arbitrary-degree branch node against every child's message
/// columns without materializing a prepared slice.
///
/// This is the memory-flat route: it walks the node's own tensor with a
/// strided odometer and accumulates directly into each point's parent column,
/// so its only allocation is the output itself. The grouped route below is
/// faster per flop but needs a `parent_dim * prod(child_dims)` scratch buffer,
/// which is why this one stays reachable at every size.
///
/// # Errors
///
/// Returns an error when a child column list has the wrong length or an
/// offset leaves `raw`.
fn scalar_multi_branch_message_contraction<T>(
    spec: &MultiBranchContractionSpec,
    raw: &[T],
    physical_values: &[usize],
    child_columns: &[Vec<T>],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let point_count = physical_values.len();
    let output_len = point_count
        .checked_mul(spec.parent_dim)
        .ok_or_else(|| anyhow::anyhow!("multi-branch output length overflows usize"))?;
    let mut output = vec![T::default(); output_len];
    let parent_stride = spec.strides[spec.parent_axis];
    for (point, &physical_value) in physical_values.iter().enumerate() {
        let physical_base = physical_value
            .checked_mul(spec.strides[spec.physical_axis])
            .ok_or_else(|| anyhow::anyhow!("multi-branch tensor offset overflows usize"))?;
        let destination = point * spec.parent_dim;
        accumulate_multi_branch_point(
            spec,
            raw,
            child_columns,
            point,
            0,
            physical_base,
            None,
            parent_stride,
            &mut output[destination..destination + spec.parent_dim],
        )?;
    }
    Ok(output)
}

/// Recursive Horner descent over one point's child bonds.
///
/// `weight` is the product of the child values chosen so far, or `None` at the
/// root so that a childless node copies its slice rather than multiplying by a
/// synthetic one.
///
/// # Errors
///
/// Returns an error when an offset overflows `usize` or leaves `raw`.
#[allow(clippy::too_many_arguments)]
fn accumulate_multi_branch_point<T>(
    spec: &MultiBranchContractionSpec,
    raw: &[T],
    child_columns: &[Vec<T>],
    point: usize,
    level: usize,
    offset: usize,
    weight: Option<T>,
    parent_stride: usize,
    output: &mut [T],
) -> Result<()>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    if level == spec.child_dims.len() {
        let mut flat = offset;
        for target in output.iter_mut() {
            let value = *raw.get(flat).ok_or_else(|| {
                anyhow::anyhow!("multi-branch tensor offset {flat} is out of bounds")
            })?;
            match weight {
                Some(weight) => *target += value * weight,
                None => *target += value,
            }
            flat = flat
                .checked_add(parent_stride)
                .ok_or_else(|| anyhow::anyhow!("multi-branch tensor offset overflows usize"))?;
        }
        return Ok(());
    }
    let dim = spec.child_dims[level];
    let stride = spec.strides[spec.child_axes[level]];
    let values = child_columns
        .get(level)
        .ok_or_else(|| anyhow::anyhow!("multi-branch contraction is missing child {level}"))?;
    let base = point
        .checked_mul(dim)
        .ok_or_else(|| anyhow::anyhow!("multi-branch child offset overflows usize"))?;
    for coordinate in 0..dim {
        let value = *values
            .get(base + coordinate)
            .ok_or_else(|| anyhow::anyhow!("multi-branch child column is out of bounds"))?;
        let next_weight = Some(match weight {
            Some(weight) => weight * value,
            None => value,
        });
        let next_offset = coordinate
            .checked_mul(stride)
            .and_then(|shift| offset.checked_add(shift))
            .ok_or_else(|| anyhow::anyhow!("multi-branch tensor offset overflows usize"))?;
        accumulate_multi_branch_point(
            spec,
            raw,
            child_columns,
            point,
            level + 1,
            next_offset,
            next_weight,
            parent_stride,
            output,
        )?;
    }
    Ok(())
}

/// Contracts an arbitrary-degree branch node one physical group at a time,
/// folding the first child with a shared GEMM.
///
/// Every point in a physical group multiplies the *same* slice of the node's
/// tensor, so that slice is prepared once and all of the group's first-child
/// columns are folded into it in a single matrix multiplication. The
/// remaining children differ per point already after that step, so they are
/// folded per point by [`fold_multi_branch_children`] -- the same two-stage
/// shape the two-child kernel uses, generalized by choosing a layout in which
/// every later fold reduces the slowest remaining axis.
///
/// # Errors
///
/// Returns an error when a child column list has the wrong length, an offset
/// leaves `raw`, or the backend rejects the multiplication.
fn grouped_multi_branch_message_contraction<T>(
    route: &mut CachedEvaluationRoute<'_, '_>,
    spec: &MultiBranchContractionSpec,
    raw: &[T],
    physical_values: &[usize],
    child_columns: &[Vec<T>],
) -> Result<Vec<T>>
where
    T: BlasMul + Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let point_count = physical_values.len();
    let output_len = point_count
        .checked_mul(spec.parent_dim)
        .ok_or_else(|| anyhow::anyhow!("multi-branch output length overflows usize"))?;
    let first_dim = *spec
        .child_dims
        .first()
        .ok_or_else(|| anyhow::anyhow!("multi-branch grouped kernel needs at least one child"))?;
    let rows = spec
        .prepared_elements()?
        .checked_div(first_dim)
        .ok_or_else(|| anyhow::anyhow!("multi-branch child dimension must be positive"))?;
    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for (point, &physical_value) in physical_values.iter().enumerate() {
        groups.entry(physical_value).or_default().push(point);
    }
    let mut output = vec![T::default(); output_len];
    let first_columns = child_columns
        .first()
        .ok_or_else(|| anyhow::anyhow!("multi-branch grouped kernel needs at least one child"))?;
    for (physical_value, points) in groups {
        #[cfg(feature = "diagnostics")]
        let setup_started = std::time::Instant::now();
        let left = prepare_multi_branch_slice(spec, raw, physical_value)?;
        let mut right = Vec::with_capacity(first_dim * points.len());
        for &point in &points {
            let start = point
                .checked_mul(first_dim)
                .ok_or_else(|| anyhow::anyhow!("multi-branch child offset overflows usize"))?;
            let end = start
                .checked_add(first_dim)
                .ok_or_else(|| anyhow::anyhow!("multi-branch child offset overflows usize"))?;
            right.extend_from_slice(
                first_columns
                    .get(start..end)
                    .ok_or_else(|| anyhow::anyhow!("multi-branch child column is out of bounds"))?,
            );
        }
        #[cfg(feature = "diagnostics")]
        diagnostics::record_kernel(diagnostics::KernelDiagnostics {
            setup_ns: diagnostics::nanos(setup_started.elapsed()),
            ..Default::default()
        });
        #[cfg(feature = "diagnostics")]
        let matmul_started = std::time::Instant::now();
        let intermediate = route.mat_mul_owned(
            Matrix::from_col_major_vec(rows, first_dim, left),
            Matrix::from_col_major_vec(first_dim, points.len(), right),
        )?;
        #[cfg(feature = "diagnostics")]
        diagnostics::record_kernel(diagnostics::KernelDiagnostics {
            matmul_ns: diagnostics::nanos(matmul_started.elapsed()),
            matmul_calls: 1,
            ..Default::default()
        });
        #[cfg(feature = "diagnostics")]
        let accumulate_started = std::time::Instant::now();
        let intermediate = intermediate.as_col_major_slice();
        for (column, &point) in points.iter().enumerate() {
            let start = column * rows;
            let folded = fold_multi_branch_children(
                spec,
                intermediate[start..start + rows].to_vec(),
                child_columns,
                point,
            )?;
            anyhow::ensure!(
                folded.len() == spec.parent_dim,
                "multi-branch fold produced {} values, expected {}",
                folded.len(),
                spec.parent_dim
            );
            let destination = point * spec.parent_dim;
            output[destination..destination + spec.parent_dim].copy_from_slice(&folded);
        }
        #[cfg(feature = "diagnostics")]
        diagnostics::record_kernel(diagnostics::KernelDiagnostics {
            accumulate_ns: diagnostics::nanos(accumulate_started.elapsed()),
            ..Default::default()
        });
    }
    Ok(output)
}

/// Routes one arbitrary-degree branch contraction to the grouped or the flat
/// kernel.
///
/// The grouped kernel is used when the group's arithmetic is large enough to
/// pay for a backend call *and* its prepared slice fits
/// [`MULTI_BRANCH_MAX_PREPARED_ELEMENTS`]; a node whose
/// `parent_dim * prod(child_dims)` cross would need more scratch than that
/// stays on the flat kernel, which allocates nothing beyond its output. Both
/// routes compute the same contraction and differ only in the order the sums
/// are accumulated.
///
/// # Errors
///
/// Propagates the selected kernel's failures.
fn multi_branch_message_contraction<T>(
    route: &mut CachedEvaluationRoute<'_, '_>,
    spec: &MultiBranchContractionSpec,
    raw: &[T],
    physical_values: &[usize],
    child_columns: &[Vec<T>],
) -> Result<Vec<T>>
where
    T: BlasMul + Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let prepared = spec.prepared_elements()?;
    let scalar_work = prepared
        .checked_mul(physical_values.len())
        .ok_or_else(|| anyhow::anyhow!("multi-branch work estimate overflows usize"))?;
    if scalar_work < BRANCH_BLAS_WORK_THRESHOLD || prepared > MULTI_BRANCH_MAX_PREPARED_ELEMENTS {
        #[cfg(feature = "diagnostics")]
        diagnostics::record_kernel(diagnostics::KernelDiagnostics {
            scalar_points: physical_values.len() as u64,
            ..Default::default()
        });
        #[cfg(test)]
        record_multi_branch_route(false);
        return scalar_multi_branch_message_contraction(spec, raw, physical_values, child_columns);
    }
    #[cfg(test)]
    record_multi_branch_route(true);
    grouped_multi_branch_message_contraction(route, spec, raw, physical_values, child_columns)
}

fn scalar_chain_message_contraction<T>(
    spec: ChainContractionSpec,
    raw: &[T],
    physical_values: &[usize],
    child_columns: &[T],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let ChainContractionSpec {
        strides,
        physical_axis,
        parent_axis,
        child_axis,
        parent_dim,
        child_dim,
    } = spec;
    let point_count = physical_values.len();
    let output_len = point_count
        .checked_mul(parent_dim)
        .ok_or_else(|| anyhow::anyhow!("chain parent-message shape overflows usize"))?;
    let mut output = vec![T::default(); output_len];
    for (point, &physical_value) in physical_values.iter().enumerate() {
        for parent_value in 0..parent_dim {
            let mut sum = T::default();
            for child_value in 0..child_dim {
                let mut axis_values = [0usize; 3];
                axis_values[physical_axis] = physical_value;
                axis_values[parent_axis] = parent_value;
                axis_values[child_axis] = child_value;
                let flat = axis_values[0]
                    .checked_mul(strides[0])
                    .and_then(|value| value.checked_add(axis_values[1].checked_mul(strides[1])?))
                    .and_then(|value| value.checked_add(axis_values[2].checked_mul(strides[2])?))
                    .ok_or_else(|| anyhow::anyhow!("chain tensor offset overflows usize"))?;
                let child_offset = point
                    .checked_mul(child_dim)
                    .and_then(|value| value.checked_add(child_value))
                    .ok_or_else(|| anyhow::anyhow!("chain child offset overflows usize"))?;
                sum += *raw
                    .get(flat)
                    .ok_or_else(|| anyhow::anyhow!("chain tensor offset is out of bounds"))?
                    * *child_columns.get(child_offset).ok_or_else(|| {
                        anyhow::anyhow!("chain child column offset is out of bounds")
                    })?;
            }
            let destination = point
                .checked_mul(parent_dim)
                .and_then(|value| value.checked_add(parent_value))
                .ok_or_else(|| anyhow::anyhow!("chain output offset overflows usize"))?;
            output[destination] = sum;
        }
    }
    Ok(output)
}

fn scalar_branch_message_contraction<T>(
    spec: BranchContractionSpec,
    raw: &[T],
    physical_values: &[usize],
    child1_columns: &[T],
    child2_columns: &[T],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let BranchContractionSpec {
        strides,
        physical_axis,
        parent_axis,
        child_axis_1,
        child_axis_2,
        parent_dim,
        child_dim_1,
        child_dim_2,
    } = spec;
    let point_count = physical_values.len();
    let mut output = vec![T::default(); point_count * parent_dim];
    for (point, &physical_value) in physical_values.iter().enumerate() {
        for parent in 0..parent_dim {
            let mut sum = T::default();
            for c1 in 0..child_dim_1 {
                let child1_value = child1_columns[point * child_dim_1 + c1];
                for c2 in 0..child_dim_2 {
                    let child2_value = child2_columns[point * child_dim_2 + c2];
                    let mut axis_values = [0usize; 4];
                    axis_values[physical_axis] = physical_value;
                    axis_values[parent_axis] = parent;
                    axis_values[child_axis_1] = c1;
                    axis_values[child_axis_2] = c2;
                    let flat = axis_values[0] * strides[0]
                        + axis_values[1] * strides[1]
                        + axis_values[2] * strides[2]
                        + axis_values[3] * strides[3];
                    sum += *raw.get(flat).ok_or_else(|| {
                        anyhow::anyhow!("branch tensor offset {flat} is out of bounds")
                    })? * child1_value
                        * child2_value;
                }
            }
            output[point * parent_dim + parent] = sum;
        }
    }
    Ok(output)
}

fn build_layout<V>(tree: &TreeTN<IdxTensor, V>, indices: &[DynIndex]) -> Result<EvaluatorLayout<V>>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    if tree.node_count() == 0 {
        bail!("TreeTNCachedEvaluator::new: network must have at least one node");
    }

    let total_site_indices = tree.site_index_network().site_index_count();
    anyhow::ensure!(
        indices.len() == total_site_indices,
        "TreeTNCachedEvaluator::new: indices.len() ({}) != total site indices ({})",
        indices.len(),
        total_site_indices
    );

    let mut seen = HashSet::with_capacity(indices.len());
    for index in indices {
        anyhow::ensure!(
            seen.insert(index.clone()),
            "TreeTNCachedEvaluator::new: duplicate index {:?}",
            index
        );
    }

    let mut entries_by_node: HashMap<V, Vec<SiteEntry>> = HashMap::new();
    let mut tensor_indices_by_node: HashMap<V, Vec<DynIndex>> = HashMap::new();
    for (input_position, index) in indices.iter().enumerate() {
        let node_name = tree
            .site_index_network()
            .find_node_by_index(index)
            .ok_or_else(|| {
                anyhow::anyhow!("TreeTNCachedEvaluator::new: unknown index {:?}", index)
            })?
            .clone();
        let tensor = tensor_for_node(tree, &node_name)?;
        let tensor_indices = tensor_indices_by_node
            .entry(node_name.clone())
            .or_insert_with(|| tensor.external_indices());
        let local_axis = tensor_indices
            .iter()
            .position(|tensor_index| tensor_index == index)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::new: site index {:?} is registered on node {:?} but not present in its tensor",
                    index,
                    node_name
                )
            })?;
        entries_by_node
            .entry(node_name)
            .or_default()
            .push(SiteEntry {
                index: index.clone(),
                input_position,
                local_axis,
            });
    }

    for entries in entries_by_node.values_mut() {
        entries.sort_by_key(|entry| entry.local_axis);
    }

    let mut local_layouts_by_node = HashMap::with_capacity(tree.node_count());
    let mut node_names = tree.node_names();
    node_names.sort();
    for node in node_names {
        let entries = entries_by_node.get(&node).map(Vec::as_slice).unwrap_or(&[]);
        let input_positions = entries
            .iter()
            .map(|entry| entry.input_position)
            .collect::<Vec<_>>();
        let dimensions = entries
            .iter()
            .map(|entry| entry.index.dim())
            .collect::<Vec<_>>();
        let indexer = FlatIndexer::try_new(&dimensions).map_err(anyhow::Error::from)?;
        local_layouts_by_node.insert(
            node,
            MessageCacheLayout {
                input_positions,
                indexer,
            },
        );
    }

    Ok(EvaluatorLayout {
        entries_by_node,
        local_layouts_by_node,
        n_indices: indices.len(),
    })
}

fn append_directed_component_layout<V>(
    layout: &EvaluatorLayout<V>,
    node: &V,
    blocked: &V,
    neighbors: &HashMap<V, Vec<V>>,
    input_positions: &mut Vec<usize>,
    dimensions: &mut Vec<usize>,
    visited: &mut HashSet<V>,
) -> Result<Vec<V>>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    let mut child_nodes = Vec::new();
    let mut stack = vec![(node.clone(), blocked.clone(), true)];
    while let Some((current, current_blocked, is_root)) = stack.pop() {
        ensure!(
            visited.insert(current.clone()),
            "TreeTNCachedEvaluator: directed component traversal encountered a cycle at {:?}",
            current
        );
        if let Some(entries) = layout.entries_by_node.get(&current) {
            for entry in entries {
                input_positions.push(entry.input_position);
                dimensions.push(entry.index.dim());
            }
        }
        let current_neighbors = neighbors
            .get(&current)
            .ok_or_else(|| anyhow::anyhow!("missing neighbors for node {:?}", current))?;
        let children = current_neighbors
            .iter()
            .filter(|neighbor| *neighbor != &current_blocked)
            .cloned()
            .collect::<Vec<_>>();
        if is_root {
            child_nodes.extend(children.iter().cloned());
        }
        for child in children.into_iter().rev() {
            stack.push((child, current.clone(), false));
        }
    }
    Ok(child_nodes)
}

fn build_directed_component_layouts<V>(
    tree: &TreeTN<IdxTensor, V>,
    layout: &EvaluatorLayout<V>,
) -> Result<DirectedComponentLayouts<V>>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    let neighbors = sorted_neighbors(tree);
    let mut node_names = tree.node_names();
    node_names.sort();
    let mut directed_layouts = HashMap::with_capacity(tree.edge_count() * 2);
    for node in node_names {
        let node_neighbors = neighbors
            .get(&node)
            .ok_or_else(|| anyhow::anyhow!("missing neighbors for node {:?}", node))?;
        for neighbor in node_neighbors {
            let mut input_positions = Vec::new();
            let mut dimensions = Vec::new();
            let mut visited = HashSet::new();
            let child_nodes = append_directed_component_layout(
                layout,
                &node,
                neighbor,
                &neighbors,
                &mut input_positions,
                &mut dimensions,
                &mut visited,
            )?;
            let indexer = FlatIndexer::try_new(&dimensions).map_err(anyhow::Error::from)?;
            directed_layouts.insert(
                (node.clone(), neighbor.clone()),
                Arc::new(DirectedComponentLayout {
                    layout: MessageCacheLayout {
                        input_positions,
                        indexer,
                    },
                    child_nodes,
                }),
            );
        }
    }
    Ok(directed_layouts)
}

fn validate_values_shape(
    values: ColMajorArrayRef<'_, usize>,
    n_indices: usize,
    context: &str,
) -> Result<()> {
    anyhow::ensure!(
        values.shape().len() == 2,
        "{context}: values must be 2D, got {}D",
        values.shape().len()
    );
    anyhow::ensure!(
        values.shape()[0] == n_indices,
        "{context}: row count {} does not match indices.len() {}",
        values.shape()[0],
        n_indices
    );
    Ok(())
}

/// Bounds-checks one point's local coordinates against its node's physical
/// indices.
///
/// This is the same contract as [`validate_index_vals`], including its
/// message, but it borrows the entries instead of cloning every index into a
/// temporary pair vector: the assignment builder calls it once per node and
/// point, so the temporary was one of the batch's dominant short-lived
/// allocations.
fn validate_entry_values(entries: &[SiteEntry], values: &[usize], context: &str) -> Result<()> {
    for (entry, value) in entries.iter().zip(values) {
        anyhow::ensure!(
            *value < entry.index.dim(),
            "{context}: coordinate {} is out of range for index {:?} with dim {}",
            value,
            entry.index,
            entry.index.dim()
        );
    }
    Ok(())
}

fn validate_index_vals<I>(index_vals: &[(I, usize)], context: &str) -> Result<()>
where
    I: IndexLike,
{
    for (index, value) in index_vals {
        anyhow::ensure!(
            *value < index.dim(),
            "{context}: coordinate {} is out of range for index {:?} with dim {}",
            value,
            index,
            index.dim()
        );
    }
    Ok(())
}

fn ensure_node_exists<V>(tree: &TreeTN<IdxTensor, V>, node: &V, context: &str) -> Result<()>
where
    V: Clone + Eq + Hash + Debug + Send + Sync,
{
    if tree.node_index(node).is_none() {
        bail!("{context} {:?} is not present in TreeTN", node);
    }
    Ok(())
}

fn tensor_for_node<'a, V>(tree: &'a TreeTN<IdxTensor, V>, node: &V) -> Result<&'a IdxTensor>
where
    V: Clone + Eq + Hash + Debug + Send + Sync,
{
    let node_idx = tree
        .node_index(node)
        .ok_or_else(|| anyhow::anyhow!("node {:?} is not present in TreeTN", node))?;
    tree.tensor(node_idx)
        .ok_or_else(|| anyhow::anyhow!("tensor for node {:?} is not present", node))
}

fn tensor_scalar_kind(tensor: &IdxTensor) -> Result<ScalarKind> {
    if tensor.is_f32() {
        Ok(ScalarKind::F32)
    } else if tensor.is_f64() {
        Ok(ScalarKind::F64)
    } else if tensor.is_c32() {
        Ok(ScalarKind::C32)
    } else if tensor.is_c64() {
        Ok(ScalarKind::C64)
    } else {
        bail!(
            "TreeTNCachedEvaluator: unsupported tensor scalar kind {:?}",
            tensor.storage_kind()
        )
    }
}

/// Gather only the requested message columns while borrowing tensor-backed
/// payloads for the duration of the gather.
///
/// Raw cached messages still need enum decoding, but the decoding is limited
/// to the requested columns. An `IdxTensor` message reads its compact host
/// payload through `dense_host_values`, so a session route does not enter the
/// process-global eager runtime for a read.
fn gather_message_columns<T>(
    message: &StackedMessage,
    child_dim: usize,
    points: &[usize],
    assignment_batch: &AssignmentBatch,
    decode: impl Fn(&CachedScalar) -> Option<T>,
    context: &str,
) -> Result<Option<Vec<T>>>
where
    T: TensorElement + Copy + StorageScalar,
{
    if let Some(raw_values) = message.raw_values.as_ref() {
        let mut columns = Vec::with_capacity(child_dim * points.len());
        for &point in points {
            let assignment = assignment_batch
                .point_to_assignment
                .get(point)
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!("{context}: missing child assignment for point {point}")
                })?;
            let start = assignment.checked_mul(child_dim).ok_or_else(|| {
                anyhow::anyhow!("{context}: child assignment offset overflows usize")
            })?;
            let end = start.checked_add(child_dim).ok_or_else(|| {
                anyhow::anyhow!("{context}: child assignment end overflows usize")
            })?;
            let values = raw_values
                .get(start..end)
                .ok_or_else(|| anyhow::anyhow!("{context}: child assignment is out of bounds"))?;
            for value in values {
                let Some(value) = decode(value) else {
                    return Ok(None);
                };
                columns.push(value);
            }
        }
        return Ok(Some(columns));
    }

    let Some(tensor) = message.tensor.as_ref() else {
        return Ok(None);
    };
    let raw_values = tensor.dense_host_values::<T>()?;
    let mut columns = Vec::with_capacity(child_dim * points.len());
    for &point in points {
        let assignment = assignment_batch
            .point_to_assignment
            .get(point)
            .copied()
            .ok_or_else(|| {
                anyhow::anyhow!("{context}: missing child assignment for point {point}")
            })?;
        let start = assignment
            .checked_mul(child_dim)
            .ok_or_else(|| anyhow::anyhow!("{context}: child assignment offset overflows usize"))?;
        let end = start
            .checked_add(child_dim)
            .ok_or_else(|| anyhow::anyhow!("{context}: child assignment end overflows usize"))?;
        columns.extend_from_slice(
            raw_values
                .get(start..end)
                .ok_or_else(|| anyhow::anyhow!("{context}: child assignment is out of bounds"))?,
        );
    }
    Ok(Some(columns))
}

fn slice_tensor(
    tensor: &IdxTensor,
    index_vals: &[(DynIndex, usize)],
    route: &mut CachedEvaluationRoute<'_, '_>,
) -> Result<IdxTensor> {
    if index_vals.is_empty() {
        return Ok(tensor.clone());
    }
    validate_index_vals(index_vals, "slice_tensor")?;
    let selected_indices = index_vals
        .iter()
        .map(|(index, _)| index.clone())
        .collect::<Vec<_>>();
    let positions = index_vals
        .iter()
        .map(|(_, position)| *position)
        .collect::<Vec<_>>();
    route.select_indices(tensor, &selected_indices, &positions)
}

fn tensor_values_any(tensor: &IdxTensor) -> Result<Vec<AnyScalar>> {
    if tensor.is_f32() {
        tensor
            .dense_host_values::<f32>()
            .map(|values| values.into_iter().map(AnyScalar::from_value).collect())
            .map_err(anyhow::Error::from)
    } else if tensor.is_f64() {
        tensor
            .dense_host_values::<f64>()
            .map(|values| values.into_iter().map(AnyScalar::from_value).collect())
            .map_err(anyhow::Error::from)
    } else if tensor.is_c32() {
        tensor
            .dense_host_values::<Complex32>()
            .map(|values| values.into_iter().map(AnyScalar::from_value).collect())
            .map_err(anyhow::Error::from)
    } else if tensor.is_c64() {
        tensor
            .dense_host_values::<Complex64>()
            .map(|values| values.into_iter().map(AnyScalar::from_value).collect())
            .map_err(anyhow::Error::from)
    } else {
        bail!(
            "TreeTNCachedEvaluator: unsupported tensor scalar kind {:?}",
            tensor.storage_kind()
        )
    }
}

fn stacked_message_scalar_kind(message: &StackedMessage) -> Result<Option<ScalarKind>> {
    if let Some(raw_values) = message.raw_values.as_ref() {
        return Ok(raw_values.first().map(|value| match value {
            CachedScalar::F32(_) => ScalarKind::F32,
            CachedScalar::F64(_) => ScalarKind::F64,
            CachedScalar::C32(_) => ScalarKind::C32,
            CachedScalar::C64(_) => ScalarKind::C64,
        }));
    }
    message.tensor.as_ref().map(tensor_scalar_kind).transpose()
}

// Both owned and borrowed assembly use the same ordered, unconjugated dot.
fn column_dot<S, T>(left: &[S], right: &[S], decode: impl Fn(&S) -> Option<T>) -> Result<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    if left.len() != right.len() {
        return Err(anyhow::anyhow!("edge-cut columns have different lengths"));
    }
    let mut sum = T::default();
    for (left, right) in left.iter().zip(right) {
        let left =
            decode(left).ok_or_else(|| anyhow::anyhow!("edge-cut column scalar kind mismatch"))?;
        let right =
            decode(right).ok_or_else(|| anyhow::anyhow!("edge-cut column scalar kind mismatch"))?;
        sum += left * right;
    }
    Ok(sum)
}

fn cached_column_dot(
    kind: ScalarKind,
    left: &[CachedScalar],
    right: &[CachedScalar],
) -> Result<CachedScalar> {
    match kind {
        ScalarKind::F32 => column_dot(left, right, |value| match value {
            CachedScalar::F32(value) => Some(*value),
            _ => None,
        })
        .map(CachedScalar::F32),
        ScalarKind::F64 => column_dot(left, right, |value| match value {
            CachedScalar::F64(value) => Some(*value),
            _ => None,
        })
        .map(CachedScalar::F64),
        ScalarKind::C32 => column_dot(left, right, |value| match value {
            CachedScalar::C32(value) => Some(*value),
            _ => None,
        })
        .map(CachedScalar::C32),
        ScalarKind::C64 => column_dot(left, right, |value| match value {
            CachedScalar::C64(value) => Some(*value),
            _ => None,
        })
        .map(CachedScalar::C64),
    }
}

fn stacked_message_values_typed<T>(
    message: &StackedMessage,
    decode: impl Fn(&CachedScalar) -> Option<T>,
) -> Result<Vec<T>>
where
    T: TensorElement + Copy + StorageScalar,
{
    if let Some(raw_values) = message.raw_values.as_ref() {
        return raw_values
            .iter()
            .map(|value| {
                decode(value).ok_or_else(|| {
                    anyhow::anyhow!("cached message scalar kind does not match edge assembly")
                })
            })
            .collect();
    }
    let tensor = message
        .tensor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("stacked message has no values"))?;
    tensor.dense_host_values::<T>().map_err(anyhow::Error::from)
}

fn ensure_typed_message_storage_shape<T>(
    values: &[T],
    assignment_dim: usize,
    bond_dim: usize,
    context: &str,
) -> Result<()> {
    let expected = assignment_dim
        .checked_mul(bond_dim)
        .ok_or_else(|| anyhow::anyhow!("{context} storage size overflows usize"))?;
    ensure!(
        values.len() == expected,
        "{context} storage has {} values, expected {expected}",
        values.len()
    );
    Ok(())
}

fn tensor_values_cached(tensor: &IdxTensor) -> Result<Vec<CachedScalar>> {
    if tensor.is_f32() {
        tensor
            .dense_host_values::<f32>()
            .map(|values| values.into_iter().map(CachedScalar::F32).collect())
            .map_err(anyhow::Error::from)
    } else if tensor.is_f64() {
        tensor
            .dense_host_values::<f64>()
            .map(|values| values.into_iter().map(CachedScalar::F64).collect())
            .map_err(anyhow::Error::from)
    } else if tensor.is_c32() {
        tensor
            .dense_host_values::<Complex32>()
            .map(|values| values.into_iter().map(CachedScalar::C32).collect())
            .map_err(anyhow::Error::from)
    } else if tensor.is_c64() {
        tensor
            .dense_host_values::<Complex64>()
            .map(|values| values.into_iter().map(CachedScalar::C64).collect())
            .map_err(anyhow::Error::from)
    } else {
        bail!(
            "TreeTNCachedEvaluator: unsupported tensor scalar kind {:?}",
            tensor.storage_kind()
        )
    }
}

fn stack_tensors_with_assignment_index(
    assignment_index: &DynIndex,
    tensors: &[IdxTensor],
    route: &mut CachedEvaluationRoute<'_, '_>,
) -> Result<IdxTensor> {
    anyhow::ensure!(
        !tensors.is_empty(),
        "stack_tensors_with_assignment_index requires at least one tensor"
    );
    anyhow::ensure!(
        assignment_index.dim() == tensors.len(),
        "assignment index dim {} does not match tensor count {}",
        assignment_index.dim(),
        tensors.len()
    );

    let tensor_refs = tensors.iter().collect::<Vec<_>>();
    route.stack_along_new_index(&tensor_refs, assignment_index.clone(), -1)
}

fn gather_stacked_tensor(
    stacked: &IdxTensor,
    source_assignment_index: &DynIndex,
    target_assignment_index: &DynIndex,
    selected_assignments: &[usize],
    route: &mut CachedEvaluationRoute<'_, '_>,
) -> Result<IdxTensor> {
    anyhow::ensure!(
        stacked.indices().last() == Some(source_assignment_index),
        "source assignment index must be the last stacked axis"
    );
    anyhow::ensure!(
        selected_assignments.len() == target_assignment_index.dim(),
        "selected assignment count {} does not match target assignment dim {}",
        selected_assignments.len(),
        target_assignment_index.dim()
    );

    route.index_select(
        stacked,
        source_assignment_index,
        target_assignment_index.clone(),
        selected_assignments,
    )
}

fn ensure_assignment_axis_last(
    tensor: IdxTensor,
    assignment_index: &DynIndex,
    route: &mut CachedEvaluationRoute<'_, '_>,
) -> Result<IdxTensor> {
    if tensor.indices().last() == Some(assignment_index) {
        return Ok(tensor);
    }
    anyhow::ensure!(
        tensor.indices().contains(assignment_index),
        "batched contraction result is missing assignment index {:?}",
        assignment_index
    );
    let mut new_order = Vec::with_capacity(tensor.indices().len());
    new_order.extend(
        tensor
            .indices()
            .iter()
            .filter(|index| *index != assignment_index)
            .cloned(),
    );
    new_order.push(assignment_index.clone());
    route.permute_indices(&tensor, &new_order)
}

fn sorted_neighbors<T, V>(tree: &TreeTN<T, V>) -> HashMap<V, Vec<V>>
where
    T: TensorLike,
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    let mut map = HashMap::new();
    let mut node_names = tree.node_names();
    node_names.sort();
    for node in node_names {
        let mut neighbors: Vec<V> = tree.site_index_network().neighbors(&node).collect();
        neighbors.sort();
        map.insert(node, neighbors);
    }
    map
}

fn rooted_tree<V>(neighbors: &HashMap<V, Vec<V>>, root: &V) -> Result<(ParentMap<V>, Vec<V>)>
where
    V: Clone + Eq + Hash + Ord + Debug,
{
    let mut parent = HashMap::<V, Option<V>>::new();
    let mut order = Vec::<V>::new();
    let mut stack = vec![(root.clone(), None)];
    while let Some((node, parent_node)) = stack.pop() {
        if parent.contains_key(&node) {
            continue;
        }
        parent.insert(node.clone(), parent_node.clone());
        order.push(node.clone());
        let mut children = neighbors
            .get(&node)
            .ok_or_else(|| anyhow::anyhow!("node {:?} is missing from neighbor map", node))?
            .iter()
            .filter(|neighbor| Some(*neighbor) != parent_node.as_ref())
            .cloned()
            .collect::<Vec<_>>();
        children.sort_by(|a, b| b.cmp(a));
        for child in children {
            stack.push((child, Some(node.clone())));
        }
    }

    anyhow::ensure!(
        parent.len() == neighbors.len(),
        "TreeTN topology is disconnected: reached {} of {} nodes",
        parent.len(),
        neighbors.len()
    );
    Ok((parent, order))
}

/// Key ownership accounting without scanning the map on cache hits.
trait MessageCacheKey: Eq + Hash + Clone {
    fn owned_heap_bytes(&self) -> usize;
}

impl MessageCacheKey for IndexKey {
    fn owned_heap_bytes(&self) -> usize {
        IndexKey::owned_heap_bytes(self)
    }
}

#[cfg(test)]
impl MessageCacheKey for u32 {
    fn owned_heap_bytes(&self) -> usize {
        0
    }
}

/// A run-scoped, append-only cache of packed message columns for one
/// directed edge.
///
/// Per Hiroshi's #646 review design: entries are never evicted individually
/// and there is no public `clear` -- the cache lives no longer than the
/// evaluator that owns it, so nothing outside can hold a handle that would
/// need clearing. Columns are stored contiguously in a single flat buffer
/// (column-major: column `i` occupies `columns[i * bond_dim .. (i+1) *
/// bond_dim]`) instead of one heap allocation per message. The configured
/// budget applies to logical column payload; observability separately reports
/// an owned-storage estimate that includes retained vector capacity, the
/// key map, and key-owned heap payloads.
struct PackedMessageCache<K, T> {
    bond_dim: usize,
    max_bytes: usize,
    positions: HashMap<K, usize>,
    columns: Vec<T>,
    key_heap_bytes: usize,
    hits: usize,
    misses: usize,
}

// A deterministic estimate for the control bytes and bucket bookkeeping of
// the standard-library hash table. HashMap does not expose its allocator
// layout, so this is deliberately documented as an estimate rather than an
// allocator-specific byte measurement.
const HASH_MAP_BUCKET_OVERHEAD_ESTIMATE_BYTES: usize = 16;

#[cfg(any(test, feature = "diagnostics"))]
fn hash_map_owned_bytes_estimate<K, V>(map: &HashMap<K, V>) -> usize {
    let per_bucket = std::mem::size_of::<K>()
        .saturating_add(std::mem::size_of::<V>())
        .saturating_add(HASH_MAP_BUCKET_OVERHEAD_ESTIMATE_BYTES);
    std::mem::size_of::<HashMap<K, V>>().saturating_add(map.capacity().saturating_mul(per_bucket))
}

/// Where one requested key's column ended up.
///
/// `Uncached` carries the computed values directly rather than a position,
/// since an over-budget entry is never written into `columns` -- there is
/// nothing in the packed buffer to point at.
#[derive(Debug, Clone, PartialEq)]
enum CacheSlot<T> {
    Cached(usize),
    Uncached(Vec<T>),
}

/// One evaluator-owned, column-major physical slice used as the left operand
/// of a branch GEMM.
struct PreparedBranchSlice<T> {
    matrix: Matrix<T>,
    payload_bytes: usize,
}

/// Bounded cache of immutable branch slices.
///
/// The key includes the directed orientation, physical coordinate, and scalar
/// kind. The cache charges only matrix payload bytes; metadata remains outside
/// the logical payload budget and is bounded by the number of inserted keys.
struct PreparedBranchSliceCache<V, T> {
    max_bytes: usize,
    retained_bytes: usize,
    slices: HashMap<(V, V, usize, ScalarKind), PreparedBranchSlice<T>>,
}

impl<V, T> PreparedBranchSliceCache<V, T>
where
    V: Clone + Eq + Hash,
{
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            retained_bytes: 0,
            slices: HashMap::new(),
        }
    }

    fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.slices.len()
    }

    fn with_prepared_slice<R, F, G>(
        &mut self,
        key: (V, V, usize, ScalarKind),
        prepare: F,
        use_slice: G,
    ) -> Result<R>
    where
        F: FnOnce() -> Result<Matrix<T>>,
        G: FnOnce(&Matrix<T>) -> Result<R>,
    {
        if self.slices.contains_key(&key) {
            #[cfg(feature = "diagnostics")]
            contraction_diagnostics::inc(&contraction_diagnostics::BRANCH_PREPARED_SLICE_HITS, 1);
            let slice = self
                .slices
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("prepared branch slice disappeared after lookup"))?;
            return use_slice(&slice.matrix);
        }

        #[cfg(feature = "diagnostics")]
        contraction_diagnostics::inc(&contraction_diagnostics::BRANCH_PREPARED_SLICE_MISSES, 1);
        let matrix = prepare()?;
        let payload_bytes = matrix
            .as_col_major_slice()
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| anyhow::anyhow!("prepared branch slice byte count overflows usize"))?;
        let can_retain = self
            .retained_bytes
            .checked_add(payload_bytes)
            .is_some_and(|bytes| bytes <= self.max_bytes);
        if can_retain {
            self.retained_bytes += payload_bytes;
            self.slices.insert(
                key.clone(),
                PreparedBranchSlice {
                    matrix,
                    payload_bytes,
                },
            );
            #[cfg(feature = "diagnostics")]
            contraction_diagnostics::inc(
                &contraction_diagnostics::BRANCH_PREPARED_SLICE_BYTES,
                payload_bytes,
            );
            let slice = self
                .slices
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("prepared branch slice missing after insertion"))?;
            use_slice(&slice.matrix)
        } else {
            #[cfg(feature = "diagnostics")]
            contraction_diagnostics::inc(
                &contraction_diagnostics::BRANCH_PREPARED_SLICE_REFUSALS,
                1,
            );
            use_slice(&matrix)
        }
    }
}

#[cfg(test)]
impl<V, T> Drop for PreparedBranchSliceCache<V, T> {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.retained_bytes,
            self.slices
                .values()
                .map(|slice| slice.payload_bytes)
                .sum::<usize>()
        );
    }
}

impl<K, T> PackedMessageCache<K, T>
where
    K: MessageCacheKey,
    T: Clone,
{
    fn new(bond_dim: usize, max_bytes: usize) -> Self {
        Self {
            bond_dim,
            max_bytes,
            positions: HashMap::new(),
            columns: Vec::new(),
            key_heap_bytes: 0,
            hits: 0,
            misses: 0,
        }
    }

    fn column(&self, position: usize) -> &[T] {
        &self.columns[position * self.bond_dim..(position + 1) * self.bond_dim]
    }

    fn checked_column(&self, position: usize) -> Result<&[T]> {
        let start = position
            .checked_mul(self.bond_dim)
            .ok_or_else(|| anyhow::anyhow!("cached column offset overflows usize"))?;
        let end = start
            .checked_add(self.bond_dim)
            .ok_or_else(|| anyhow::anyhow!("cached column end overflows usize"))?;
        self.columns
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("cached column is out of bounds"))
    }

    /// Looks up every key without computing anything.
    ///
    /// Returns `Some(positions)`, one per key in order, and counts each as a
    /// hit, only if every key is already cached; otherwise returns `None`
    /// without touching the hit/miss counters, so a caller that falls back to
    /// [`Self::get_or_compute_batch`] on a partial hit does not double-count.
    fn get_all_cached(&mut self, keys: &[K]) -> Option<Vec<usize>> {
        let positions = keys
            .iter()
            .map(|key| self.positions.get(key).copied())
            .collect::<Option<Vec<_>>>()?;
        self.hits += keys.len();
        Some(positions)
    }

    /// Records keys already confirmed as cached by a partial lookup.
    fn record_hits(&mut self, count: usize) {
        self.hits += count;
    }

    fn contains(&self, key: &K) -> bool {
        self.positions.contains_key(key)
    }

    /// Looks up one key's column position without touching the hit/miss
    /// counters -- for reassembling a merged result after the counted
    /// lookup/insert calls that drove the merge have already run.
    fn position(&self, key: &K) -> Option<usize> {
        self.positions.get(key).copied()
    }

    /// Logical payload bytes admitted by the cache budget.
    ///
    /// This intentionally uses the vector length, not capacity: it describes
    /// the message columns that are logically retained. Use
    /// [`Self::owned_retained_bytes_estimate`] for the storage estimate that
    /// also includes spare vector capacity and the key map.
    fn retained_bytes(&self) -> usize {
        self.logical_payload_bytes()
    }

    fn logical_payload_bytes(&self) -> usize {
        self.columns.len().saturating_mul(std::mem::size_of::<T>())
    }

    fn key_count(&self) -> usize {
        self.positions.len()
    }

    /// Estimates bytes owned by this cache, including vector capacity, key
    /// storage (including key-owned heap payloads), map entries/buckets, and
    /// the cache's fixed metadata.
    ///
    /// The `HashMap` bucket term uses
    /// [`HASH_MAP_BUCKET_OVERHEAD_ESTIMATE_BYTES`] because its allocator
    /// layout is an implementation detail of the standard library. This
    /// estimate is for observability and is not the logical admission budget.
    fn owned_retained_bytes_estimate(&self) -> usize {
        let key_bytes = std::mem::size_of::<K>();
        let value_bytes = std::mem::size_of::<usize>();
        let per_bucket = key_bytes
            .saturating_add(value_bytes)
            .saturating_add(HASH_MAP_BUCKET_OVERHEAD_ESTIMATE_BYTES);
        std::mem::size_of::<Self>()
            .saturating_add(
                self.columns
                    .capacity()
                    .saturating_mul(std::mem::size_of::<T>()),
            )
            .saturating_add(self.positions.capacity().saturating_mul(per_bucket))
            .saturating_add(self.key_heap_bytes)
    }

    fn hits(&self) -> usize {
        self.hits
    }

    fn misses(&self) -> usize {
        self.misses
    }

    /// Returns where each requested key's column ended up, in order.
    ///
    /// Follows Hiroshi's seven-step batch protocol: look up hits, dedupe
    /// misses, compute the missing columns as one batch via
    /// `compute_missing`, then append and commit as many as the byte budget
    /// admits together, so the cache stays consistent even if computation
    /// fails partway. Once the budget is exhausted, further misses are still
    /// computed and returned (`CacheSlot::Uncached`) but not retained --
    /// matching Hiroshi's "continue evaluating new messages without caching
    /// them" rather than evicting an already-cached entry to make room.
    fn get_or_compute_batch<F, E>(
        &mut self,
        keys: &[K],
        compute_missing: F,
    ) -> std::result::Result<Vec<CacheSlot<T>>, E>
    where
        F: FnOnce(&[K]) -> std::result::Result<Vec<Vec<T>>, E>,
        E: From<anyhow::Error>,
    {
        let mut missing_keys = Vec::new();
        let mut missing_seen = HashSet::new();
        for key in keys {
            if self.positions.contains_key(key) {
                self.hits += 1;
            } else {
                self.misses += 1;
                if missing_seen.insert(key.clone()) {
                    missing_keys.push(key.clone());
                }
            }
        }

        let mut computed_values: HashMap<K, Vec<T>> = HashMap::new();
        if !missing_keys.is_empty() {
            let computed = compute_missing(&missing_keys)?;
            if computed.len() != missing_keys.len() {
                return Err(anyhow::anyhow!(
                    "PackedMessageCache::get_or_compute_batch: compute_missing returned fewer columns ({} for {} keys)",
                    computed.len(),
                    missing_keys.len()
                )
                .into());
            }
            let bytes_per_column = self
                .bond_dim
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "PackedMessageCache::get_or_compute_batch: column byte count overflows usize"
                    )
                })?;
            for column in &computed {
                if column.len() != self.bond_dim {
                    return Err(anyhow::anyhow!(
                        "PackedMessageCache::get_or_compute_batch: computed column has length {}, expected {}",
                        column.len(),
                        self.bond_dim
                    )
                    .into());
                }
            }
            for (key, column) in missing_keys.into_iter().zip(computed) {
                let would_retain_bytes = self.logical_payload_bytes().checked_add(bytes_per_column);
                if would_retain_bytes.is_some_and(|bytes| bytes <= self.max_bytes) {
                    let position = self.columns.len() / self.bond_dim;
                    self.columns.extend(column);
                    self.key_heap_bytes =
                        self.key_heap_bytes.saturating_add(key.owned_heap_bytes());
                    self.positions.insert(key, position);
                } else {
                    computed_values.insert(key, column);
                }
            }
        }

        // Every key is either cached above or, when the byte budget refused
        // it, present in `computed_values` -- unless `compute_missing`
        // returned fewer columns than the `missing_keys` it was asked for,
        // which is a caller contract violation this type cannot prevent, so
        // it is reported as an error rather than assumed impossible.
        keys.iter()
            .map(|key| match self.positions.get(key) {
                Some(&position) => Ok(CacheSlot::Cached(position)),
                None => computed_values
                    .get(key)
                    .cloned()
                    .map(CacheSlot::Uncached)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "PackedMessageCache::get_or_compute_batch: compute_missing returned \
                             fewer columns than requested keys"
                        )
                        .into()
                    }),
            })
            .collect::<std::result::Result<Vec<_>, E>>()
    }
}

#[cfg(test)]
mod tests;
