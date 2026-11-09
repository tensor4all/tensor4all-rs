//! Naive cost of the treersi sketch on a CUDA context (issue #1158 step 1).
//!
//! Step 1 keeps the sketch blocks in host buffers and only dispatches each
//! contraction through the caller's context, so a CUDA run pays a transfer per
//! operand and per result. This harness prices that naive path against one
//! single-threaded CPU run and against an explicit download of the same input
//! cores, which is a lower bound on the traffic the product call performs.
//!
//! Run one measured cell at a time; every path is warmed before timing, CUDA is
//! synchronised, and one `RESULT` line per cell is printed. It is a record for
//! the issue, not a benchmark gate. The fixture refuses a cell whose derived
//! probe width no longer reaches the sketch path (chi above 512 at this depth).
//!
//! ```sh
//! cargo run --release -p tensor4all-benchmarks --features cuda \
//!     --example benchmark_treersi_cuda_step1 -- 256_f64
//! ```

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{CudaExecutionContext, DynIndex, ExecutionContext, IdxTensor, IndexLike};
use tensor4all_tensorbackend::{set_default_cpu_threads, CpuBackend, CpuExecutionContext};
use tensor4all_treersi::{hadamard_many_with_rng_in, TreeRsiOptions, TreeRsiScalar};
use tensor4all_treetn::TreeTN;

type Tree = TreeTN<IdxTensor, usize>;

/// Long enough that the deepest complements `2^(NODES-2)` exceed
/// `k = ceil(chi / d) + 5` for both measured chi, so the sketch and not only the
/// exact complements is exercised.
const NODES: usize = 11;
const PHYS: usize = 2;
const REPS: usize = 3;

/// A chain of `NODES` cores with bond dimension `bond_dim` and physical
/// dimension `PHYS`, values in `[-1, 1)` so the product is well scaled.
fn chain<T: Sample>(bond_dim: usize, seed: u64) -> Tree {
    let sites = (0..NODES)
        .map(|_| DynIndex::new_dyn(PHYS))
        .collect::<Vec<_>>();
    let bonds = (1..NODES)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let tensors = (0..NODES)
        .map(|node| {
            let mut indices = Vec::new();
            if node > 0 {
                indices.push(bonds[node - 1].clone());
            }
            indices.push(sites[node].clone());
            if node + 1 < NODES {
                indices.push(bonds[node].clone());
            }
            let len = indices.iter().map(|index| index.dim()).product();
            let data = (0..len)
                .map(|_| T::from_real(rng.random_range(-1.0..1.0)))
                .collect();
            IdxTensor::from_dense(indices, data).unwrap()
        })
        .collect::<Vec<_>>();
    TreeTN::from_tensors(tensors, (0..NODES).collect()).unwrap()
}

/// Fixture construction and the payload size of one element.
trait Sample: TreeRsiScalar {
    fn from_real(value: f64) -> Self;
    fn element_bytes() -> usize;
}

impl Sample for f64 {
    fn from_real(value: f64) -> Self {
        value
    }
    fn element_bytes() -> usize {
        std::mem::size_of::<f64>()
    }
}

impl Sample for Complex64 {
    fn from_real(value: f64) -> Self {
        Complex64::new(value, 0.25 * value)
    }
    fn element_bytes() -> usize {
        std::mem::size_of::<Complex64>()
    }
}

fn scoped_copy<T: TreeRsiScalar>(tree: &Tree, context: &ExecutionContext) -> Tree {
    let names = tree.node_names();
    let tensors = names
        .iter()
        .map(|name| {
            let tensor = tree.tensor(tree.node_index(name).unwrap()).unwrap();
            let indices = tensor.indices().to_vec();
            let values = tensor.to_vec::<T>().unwrap();
            IdxTensor::from_dense_in(context, indices, values).unwrap()
        })
        .collect::<Vec<_>>();
    TreeTN::from_tensors(tensors, names).unwrap()
}

fn core_elements(tree: &Tree) -> usize {
    tree.node_indices()
        .into_iter()
        .map(|node| {
            tree.tensor(node)
                .unwrap()
                .indices()
                .iter()
                .map(|index| index.dim())
                .product::<usize>()
        })
        .sum()
}

fn options<T: TreeRsiScalar>(bond_dim: usize) -> TreeRsiOptions<usize> {
    TreeRsiOptions {
        max_bond_dim: Some(bond_dim),
        root: Some(NODES - 1),
        rel_tol: T::epsilon() * 10.0,
        ..Default::default()
    }
}

fn measure<T: Sample>(bond_dim: usize, dtype: &str) {
    let host = chain::<T>(bond_dim, 7);
    let cpu = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).expect("one-thread CPU backend"),
    )));
    let cuda = Arc::new(CudaExecutionContext::new().expect("CUDA ordinal 0 must be available"));
    let cuda_context = ExecutionContext::Cuda(Arc::clone(&cuda));

    let cpu_inputs = (0..2)
        .map(|_| scoped_copy::<T>(&host, &cpu))
        .collect::<Vec<_>>();
    let cuda_inputs = (0..2)
        .map(|_| scoped_copy::<T>(&host, &cuda_context))
        .collect::<Vec<_>>();
    let options = options::<T>(bond_dim);

    // The product call must reach both input core sets on the host, so their
    // download is a lower bound on its traffic.
    let core_bytes = 2 * core_elements(&host) * T::element_bytes();
    let core_download = |reps: usize| {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let start = Instant::now();
            for tree in &cuda_inputs {
                let _ = tree.download(&cuda).expect("download");
            }
            cuda.synchronize().expect("synchronize");
            best = best.min(start.elapsed().as_secs_f64());
        }
        best
    };
    let _ = core_download(1);
    let core_download_s = core_download(REPS);

    let run = |inputs: &[Tree], context: &ExecutionContext, reps: usize| {
        let mut best = (f64::INFINITY, 0_usize);
        for _ in 0..reps {
            let mut rng = ChaCha8Rng::seed_from_u64(11);
            let start = Instant::now();
            let result = hadamard_many_with_rng_in::<T, _, _>(inputs, &options, &mut rng, context)
                .expect("product");
            if matches!(context, ExecutionContext::Cuda(_)) {
                cuda.synchronize().expect("synchronize");
            }
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed < best.0 {
                best = (elapsed, result.diagnostics.sketch_dim);
            }
        }
        best
    };
    let (_, k) = run(&cpu_inputs, &cpu, 1);
    let (_, cuda_k) = run(&cuda_inputs, &cuda_context, 1);
    assert_eq!(k, cuda_k, "both paths must use one sketch width");
    let (cpu_s, _) = run(&cpu_inputs, &cpu, REPS);
    let (cuda_s, _) = run(&cuda_inputs, &cuda_context, REPS);

    // The workload must reach the sketch, or the cell prices only the exact
    // complements.
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let sketched = hadamard_many_with_rng_in::<T, _, _>(&cpu_inputs, &options, &mut rng, &cpu)
        .expect("product")
        .diagnostics
        .edges
        .iter()
        .any(|edge| !edge.exact_columns);
    assert!(sketched, "the fixture must reach a sketched edge");

    println!(
        "RESULT chi={bond_dim} d={PHYS} dtype={dtype} nodes={NODES} k={k} reps={REPS} \
         cpu_1t_s={cpu_s:.4} cuda_s={cuda_s:.4} core_download_s={core_download_s:.4} \
         core_download_bytes={core_bytes} core_download_gbs={:.2} cuda_over_cpu={:.2}",
        core_bytes as f64 / core_download_s / 1e9,
        cuda_s / cpu_s
    );
}

fn main() {
    // Both arms must see one CPU thread, including the host rank decision a CUDA
    // run performs; this has to precede the first global-context use.
    set_default_cpu_threads(NonZeroUsize::new(1).expect("1 is nonzero"))
        .expect("the default CPU thread count is set before any global context is created");

    let case = std::env::args()
        .skip(1)
        .find(|argument| !argument.starts_with("--"))
        .unwrap_or_else(|| "256_f64".to_string());
    let (bond_dim, dtype) = case.split_once('_').expect("case is <chi>_<dtype>");
    let bond_dim: usize = bond_dim.parse().expect("chi is a positive integer");
    match dtype {
        "f64" => measure::<f64>(bond_dim, dtype),
        "c64" => measure::<Complex64>(bond_dim, dtype),
        other => panic!("unknown dtype {other}: use f64 or c64"),
    }
}
