// M6c benchmark: patched element-wise operations on the Matsubara G0 of
// gw-rs (docs/design/tree-patching-elementwise.md, implementation step 5).
//
// Workloads (2D Hubbard dispersion, mu = 0.5, interleaved quantics grid with
// `R` bits per variable) on one of two gw-rs topologies (`--topology`):
// `cttn`, the comb CTTN whose spine `kx1 - ky1 - n1` has a degree-three node
// at `ky1`, or `tt_nblock`, the chain of every `n` bit followed by `kx` and
// `ky` interleaved (gw-rs's topology at low temperature; the comb is G0's
// worst case at T = 0.01 for R >= 9):
//
// - `g0`: the nonlinear map `G0 = 1 / (mu + 2 cos kx + 2 cos ky + i omega_n)`
//   of three exact rank-2 inputs (`cos kx`, `cos ky`, `omega_n`), the TreeACI
//   Dyson construction of gw-rs.
// - `g0sq`: the product `G0 * G0` of an approximate G0 with itself (the
//   integrand of the static bubble at `q = 0`).
// - `g0sq-mixed`: the same product of two patched G0 built with different
//   caps and tolerances, so the two inputs are partitioned differently and
//   the driver must split the output down to their common refinement
//   (patched and `b0` methods only).
//
// Methods (one per process, so that peak RSS belongs to one method):
//
// - `patched-aci` / `patched-rsi`: `patched_elementwise` with the TreeACI or
//   tree RSI kernel (RSI for `g0sq` only);
// - `b0`: `patched_interpolate` with TreeTCI and an evaluator that evaluates
//   every input partition pointwise and applies `f`;
// - `unpatched-aci` / `unpatched-rsi`: the kernel on the whole domain.
//
// For `g0sq`, the patched and `b0` methods take the patched G0 of
// `patched-aci` (input cap and rtol given) and the unpatched ones the
// unpatched G0 of `unpatched-aci` (input tolerance given); building the input
// is reported but not timed with the operation. For `g0sq-mixed` the second
// factor is the patched G0 of `--in2-cap` and `--in2-rtol`.
//
// Accuracy is measured outside the timed region against the analytic
// function: the relative L2 error is estimated by importance sampling from
// the mixture `q = (|f|^2 / S^2 + 1 / V) / 2`, with `S` the exact L2 norm of
// the analytic function on the grid (summed over every grid point, using the
// symmetries of the dispersion). The mixture concentrates samples where
// `|f|` is large (low frequencies on the Fermi surface) and keeps a uniform
// half for absolute error floors. Every method of one `R` and seed uses the
// same points. For `g0sq` the error against `f` of the given inputs (the
// operation's own contract) is reported as well.
//
// Run (single thread, as every other harness):
//   cargo build --release -p tensor4all-benchmarks --example benchmark_m6c_elementwise
//   RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
//     /usr/bin/time -v ./target/release/examples/benchmark_m6c_elementwise \
//     g0 patched-aci --r 12 --cap 64 --rtol 1e-3
//
// Arguments: `<workload> <method>`, then `--r`, `--temperature`, `--cap`,
// `--rtol` (relative L2 tolerance of the patched methods), `--tol` (absolute
// max-norm tolerance of unpatched TreeACI), `--rsi-rtol` (tree RSI's
// LU pivot tolerance, patched and unpatched; not an accuracy control, since
// tree RSI estimates no error), `--rsi-cap` (unpatched RSI), `--in-cap`,
// `--in-rtol`, `--in-tol` (the G0 input of `g0sq`), `--scale-tolerance`
// (unpatched TreeACI: `--tol` relative to the sampled scale, as gw-rs runs it,
// instead of absolute), `--max-sweeps` (unpatched TreeACI's sweep limit,
// 50 unless given), `--topology`, `--in2-cap`, `--in2-rtol`
// (the second factor of `g0sq-mixed`), `--samples`, `--seed`, `--trace-calls`
// (patched TreeACI: one `engine_call` line per kernel and residual-TT run),
// `--trace-history` (patched TreeACI on `g0`: per-pass history of every
// kernel run below the cap, from a rerun),
// `--error-norm` (`l2` or `sampled-max`, patched methods), `--audit` (the
// driver's L2 audit, `true` unless given),
// and `--variants "cap=32;cap=128,rtol=1e-4"`, which runs the method once per
// `;`-separated set of overrides on the same inputs and samples (peak RSS
// then covers every variant), and `--exhaustive true`, which compares at
// every grid point (`R <= 7`) and adds the relative max error. Output: one
// JSON object per line.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_elementwise, patched_interpolate, PatchedInterpolationOptions,
    PatchedInterpolationResult, VerificationOptions,
};
use tensor4all_partitionedtreetn::{
    ErrorNorm, ErrorTolerance, L2Reference, PartitionedTreeTN, SubDomainTreeTN,
};
use tensor4all_tensorbackend::explicit::Session;
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_treeaci::{
    tree_elementwise_batched_with_rng_on_session, TreeAciInterpolator, TreeAciNode, TreeAciOptions,
    TreeAciTermination, TreeElementwiseBatch,
};
use tensor4all_treersi::{hadamard_many_with_rng_on_session, TreeRsiInterpolator, TreeRsiOptions};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::{
    ElementwiseInterpolator, ElementwiseOperator, EvalContext, InterpolationError,
    InterpolationOutcome, InterpolationProblem, InterpolationTermination, Parallelism, Product,
    TreeInterpolator,
};
use tensor4all_treetn::{NodeNameNetwork, TreeTN};

const MU: f64 = 0.5;

type Tree = TreeTN<IdxTensor, usize>;
type Partition = PartitionedTreeTN<usize>;

// ---------------------------------------------------------------------------
// Arguments

#[derive(Debug, Clone)]
struct Args {
    workload: String,
    method: String,
    r: usize,
    temperature: f64,
    cap: usize,
    rtol: f64,
    tol: f64,
    rsi_rtol: f64,
    rsi_cap: usize,
    in_cap: usize,
    in_rtol: f64,
    in_tol: f64,
    in2_cap: usize,
    in2_rtol: f64,
    samples: usize,
    seed: u64,
    /// Settings run one after another on the same inputs, each a list of
    /// `key=value` overrides of the arguments above (without `--`).
    variants: Vec<String>,
    /// Compare at every grid point instead of sampling (small `R` only).
    exhaustive: bool,
    /// Emit one line per accepted patch of the patched methods.
    dump_patches: bool,
    /// Emit one `engine_call` line per kernel and residual-TT run of the
    /// patched TreeACI method: active sites, tolerance, cap, termination,
    /// rank, wall time and the points evaluated.
    trace_calls: bool,
    /// Patched TreeACI on `g0` only: rerun every kernel run that ends below
    /// the cap with the same inputs, options and seed, and emit one
    /// `kernel_history` line with its per-pass ranks, errors and Guard
    /// pivots, which the adapter does not keep.
    trace_history: bool,
    /// Diagnostics (small `R` only): compare every TreeACI product run with
    /// the dense product of its restricted inputs.
    audit_kernel: bool,
    /// TreeACI's `global_tolerance_margin` (its default is 10).
    aci_margin: f64,
    /// `cttn` (comb, degree three at ky1) or `tt_nblock` (chain).
    topology: String,
    /// Unpatched TreeACI: `--tol` relative to the sampled magnitude (`true`,
    /// TreeACI's default and gw-rs g0_aci's) or absolute (`false`).
    scale_tolerance: bool,
    /// Unpatched TreeACI's sweep limit (`TreeAciOptions::max_sweeps`); 50
    /// unless given. A lower limit stops a run that cycles at the tolerance
    /// boundary (#784) after its rank has settled.
    max_sweeps: usize,
    /// `TreeAciInterpolator::with_tolerance_factor` of the patched TreeACI
    /// kernel: it runs at `tau / factor`, the acceptance stays at `tau`.
    /// Pinned to 1 unless given, as in the 2026-11-11 report, although the
    /// library's default is 3.
    engine_tol_factor: f64,
    /// Acceptance of the patched methods: `l2` (the driver measures every
    /// patch against `rtol` times the exact grid norm, the default) or
    /// `sampled-max` (`ErrorNorm::sampled_max`: the engine's own estimate
    /// against `rtol` times `max |f|` pinned from the root sample, with no
    /// measurement, as in Grosso et al.'s pQTCI). The same `rtol` is not the
    /// same accuracy under the two.
    error_norm: String,
    /// The driver's independent audit of every sampled L2 acceptance (the
    /// library default `true`); it never changes the partition.
    audit: bool,
}

fn set(args: &mut Args, key: &str, value: &str) -> Result<()> {
    match key {
        "--r" => args.r = value.parse()?,
        "--temperature" => args.temperature = value.parse()?,
        "--cap" => args.cap = value.parse()?,
        "--rtol" => args.rtol = value.parse()?,
        "--tol" => args.tol = value.parse()?,
        "--rsi-rtol" => args.rsi_rtol = value.parse()?,
        "--rsi-cap" => args.rsi_cap = value.parse()?,
        "--in-cap" => args.in_cap = value.parse()?,
        "--in-rtol" => args.in_rtol = value.parse()?,
        "--in-tol" => args.in_tol = value.parse()?,
        "--in2-cap" => args.in2_cap = value.parse()?,
        "--in2-rtol" => args.in2_rtol = value.parse()?,
        "--samples" => args.samples = value.parse()?,
        "--seed" => args.seed = value.parse()?,
        "--exhaustive" => args.exhaustive = value.parse()?,
        "--dump-patches" => args.dump_patches = value.parse()?,
        "--trace-calls" => args.trace_calls = value.parse()?,
        "--trace-history" => args.trace_history = value.parse()?,
        "--audit-kernel" => args.audit_kernel = value.parse()?,
        "--aci-margin" => args.aci_margin = value.parse()?,
        "--topology" => args.topology = value.to_string(),
        "--scale-tolerance" => args.scale_tolerance = value.parse()?,
        "--engine-tol-factor" => args.engine_tol_factor = value.parse()?,
        "--error-norm" => args.error_norm = value.to_string(),
        "--audit" => args.audit = value.parse()?,
        "--max-sweeps" => args.max_sweeps = value.parse()?,
        "--variants" => {
            args.variants = value.split(';').map(str::to_string).collect();
        }
        _ => bail!("unknown argument {key}"),
    }
    Ok(())
}

/// `args` with the overrides of one variant; only the method's own settings
/// may change, since the inputs are shared.
fn variant(args: &Args, overrides: &str) -> Result<Args> {
    let mut out = args.clone();
    for pair in overrides.split(',').filter(|p| !p.is_empty()) {
        let (key, value) = pair
            .split_once('=')
            .with_context(|| format!("variant entry {pair} is not key=value"))?;
        if !matches!(
            key,
            "cap"
                | "rtol"
                | "tol"
                | "rsi-rtol"
                | "rsi-cap"
                | "seed"
                | "engine-tol-factor"
                | "aci-margin"
                | "max-sweeps"
                | "error-norm"
                | "audit"
        ) {
            bail!("variant key {key} would change the inputs or the samples");
        }
        set(&mut out, &format!("--{key}"), value)?;
    }
    Ok(out)
}

fn parse_args() -> Result<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() < 2 {
        bail!("usage: benchmark_m6c_elementwise <g0|g0sq> <method> [--key value ...]");
    }
    let mut args = Args {
        workload: raw[0].clone(),
        method: raw[1].clone(),
        r: 12,
        temperature: 0.01,
        cap: 64,
        rtol: 1e-3,
        tol: 3e-3,
        rsi_rtol: 1e-14,
        rsi_cap: 256,
        in_cap: 64,
        in_rtol: 1e-3,
        in_tol: 3e-3,
        in2_cap: 32,
        in2_rtol: 1e-3,
        samples: 100_000,
        seed: 2026,
        variants: vec![String::new()],
        exhaustive: false,
        dump_patches: false,
        trace_calls: false,
        trace_history: false,
        audit_kernel: false,
        aci_margin: 10.0,
        engine_tol_factor: 1.0,
        max_sweeps: 50,
        error_norm: "l2".to_string(),
        audit: true,
        topology: "cttn".to_string(),
        scale_tolerance: false,
    };
    let mut rest = raw[2..].iter();
    while let Some(key) = rest.next() {
        let value = rest
            .next()
            .with_context(|| format!("missing value for {key}"))?;
        set(&mut args, key, value)?;
    }
    if !matches!(args.workload.as_str(), "g0" | "g0sq" | "g0sq-mixed") {
        bail!("workload must be g0, g0sq, or g0sq-mixed");
    }
    if args.trace_history && (args.method != "patched-aci" || args.workload != "g0") {
        bail!("--trace-history covers patched-aci on g0 only");
    }
    if args.workload == "g0sq-mixed" && args.method.starts_with("unpatched") {
        bail!(
            "g0sq-mixed compares differently partitioned inputs; unpatched inputs have one patch"
        );
    }
    Ok(args)
}

fn emit(value: Value) {
    println!("{value}");
}

// ---------------------------------------------------------------------------
// Grid and analytic function

/// The interleaved quantics grid: site `3 b + v` is bit `b` (MSB first) of
/// variable `v` (`kx`, `ky`, `n`); one site per node, node name = site.
struct Grid {
    r: usize,
    temperature: f64,
    sites: Vec<DynIndex>,
    edges: Vec<(usize, usize)>,
}

impl Grid {
    /// `topology` is `cttn` (gw-rs `cttn_seq_comb`) or `tt_nblock` (gw-rs
    /// `tt_chain_nblock`).
    fn new(r: usize, temperature: f64, topology: &str) -> Result<Self> {
        let sites = (0..3 * r).map(|_| DynIndex::new_dyn(2)).collect();
        let edges = match topology {
            // Comb CTTN: spine kx1 - ky1 - n1, each variable's remaining
            // bits on its own arm; ky1 has degree three.
            "cttn" => {
                let mut edges = vec![(0, 1), (1, 2)];
                for var in 0..3 {
                    edges.extend((1..r).map(|bit| (var + 3 * (bit - 1), var + 3 * bit)));
                }
                edges
            }
            // Chain: every n bit (MSB first), then kx and ky interleaved.
            "tt_nblock" => {
                let order: Vec<usize> = (0..r)
                    .map(|bit| 2 + 3 * bit)
                    .chain((0..r).flat_map(|bit| [3 * bit, 3 * bit + 1]))
                    .collect();
                order.windows(2).map(|pair| (pair[0], pair[1])).collect()
            }
            other => bail!("topology must be cttn or tt_nblock, not {other}"),
        };
        Ok(Self {
            r,
            temperature,
            sites,
            edges,
        })
    }

    fn size(&self) -> usize {
        1 << self.r
    }

    fn omega(&self, n: usize) -> f64 {
        2.0 * PI * self.temperature * (n as f64 - self.size() as f64 / 2.0 + 0.5)
    }

    fn a(&self, kx: usize, ky: usize) -> f64 {
        let size = self.size() as f64;
        MU + 2.0 * (2.0 * PI * kx as f64 / size).cos() + 2.0 * (2.0 * PI * ky as f64 / size).cos()
    }

    fn g0(&self, kx: usize, ky: usize, n: usize) -> Complex64 {
        1.0 / Complex64::new(self.a(kx, ky), self.omega(n))
    }

    /// The site values of grid point `(kx, ky, n)`.
    fn point(&self, coords: [usize; 3]) -> Vec<usize> {
        let mut point = vec![0; 3 * self.r];
        for bit in 0..self.r {
            for var in 0..3 {
                point[3 * bit + var] = (coords[var] >> (self.r - 1 - bit)) & 1;
            }
        }
        point
    }

    fn topology(&self) -> Result<NodeNameNetwork<usize>> {
        let mut topology = NodeNameNetwork::new();
        for node in 0..self.sites.len() {
            topology.add_node(node).map_err(|e| anyhow!("{e}"))?;
        }
        for &(a, b) in &self.edges {
            topology.add_edge(&a, &b).map_err(|e| anyhow!("{e}"))?;
        }
        Ok(topology)
    }

    fn node_sites(&self) -> BTreeMap<usize, Vec<DynIndex>> {
        self.sites
            .iter()
            .enumerate()
            .map(|(node, site)| (node, vec![site.clone()]))
            .collect()
    }

    /// Each variable's bits, MSB first.
    fn variables(&self) -> Vec<Vec<DynIndex>> {
        (0..3)
            .map(|var| {
                (0..self.r)
                    .map(|bit| self.sites[3 * bit + var].clone())
                    .collect()
            })
            .collect()
    }

    /// The three exact rank-2 inputs `cos kx`, `cos ky`, `omega_n`. The unique
    /// path between a variable's first and last bit carries a rank-2
    /// rotation (cosines) or affine (frequency) state; every other bond has
    /// dimension 1.
    fn inputs(&self) -> Result<Vec<Tree>> {
        let n_nodes = self.sites.len();
        let mut adjacency = vec![Vec::new(); n_nodes];
        for &(a, b) in &self.edges {
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
        let step = 2.0 * PI * self.temperature;
        let mut inputs = Vec::new();
        for var in 0..3 {
            let mut parents = vec![usize::MAX; n_nodes];
            parents[var] = var;
            let mut stack = vec![var];
            while let Some(node) = stack.pop() {
                for &next in &adjacency[node] {
                    if parents[next] == usize::MAX {
                        parents[next] = node;
                        stack.push(next);
                    }
                }
            }
            let mut path = Vec::new();
            let mut node = var + 3 * (self.r - 1);
            while node != var {
                path.push(node);
                node = parents[node];
            }
            path.push(var);
            path.reverse();
            let bonds: Vec<DynIndex> = self
                .edges
                .iter()
                .map(|&(a, b)| {
                    let on_path = path.windows(2).any(|pair| pair == [a, b] || pair == [b, a]);
                    DynIndex::new_dyn(if on_path { 2 } else { 1 })
                })
                .collect();
            let mut tensors = Vec::new();
            for (node, site) in self.sites.iter().enumerate() {
                let mut indices = vec![site.clone()];
                let (mut prev, mut next) = (None, None);
                let position = path.iter().position(|&p| p == node);
                for (edge, &(a, b)) in self.edges.iter().enumerate() {
                    if a != node && b != node {
                        continue;
                    }
                    let other = if a == node { b } else { a };
                    if bonds[edge].dim() == 2 {
                        if position.is_some_and(|pos| pos > 0 && path[pos - 1] == other) {
                            prev = Some(indices.len());
                        } else {
                            next = Some(indices.len());
                        }
                    }
                    indices.push(bonds[edge].clone());
                }
                let dims: Vec<usize> = indices.iter().map(IndexLike::dim).collect();
                let values: Vec<Complex64> = (0..dims.iter().product())
                    .map(|flat| {
                        if position.is_none() {
                            return Complex64::new(1.0, 0.0);
                        }
                        let mut rest = flat;
                        let coords: Vec<usize> = dims
                            .iter()
                            .map(|d| {
                                let x = rest % d;
                                rest /= d;
                                x
                            })
                            .collect();
                        let bit = node / 3;
                        let m = if node % 3 != var {
                            [1.0, 0.0, 0.0, 1.0]
                        } else if var < 2 {
                            let (s, c) = (coords[0] as f64 * PI / (1usize << bit) as f64).sin_cos();
                            [c, -s, s, c]
                        } else {
                            [
                                1.0,
                                coords[0] as f64 * step * (1usize << (self.r - 1 - bit)) as f64,
                                0.0,
                                1.0,
                            ]
                        };
                        let initial = if var < 2 { [1.0, 0.0] } else { [0.0, 1.0] };
                        let incoming = prev.map_or(initial, |axis| {
                            if coords[axis] == 0 {
                                [1.0, 0.0]
                            } else {
                                [0.0, 1.0]
                            }
                        });
                        let after = [
                            m[0] * incoming[0] + m[1] * incoming[1],
                            m[2] * incoming[0] + m[3] * incoming[1],
                        ];
                        let value = next.map_or_else(
                            || {
                                let offset = if var < 2 {
                                    0.0
                                } else {
                                    step * (0.5 - self.size() as f64 / 2.0)
                                };
                                after[0] + offset * after[1]
                            },
                            |axis| after[coords[axis]],
                        );
                        Complex64::new(value, 0.0)
                    })
                    .collect();
                tensors.push(IdxTensor::from_dense(indices, values)?);
            }
            inputs.push(TreeTN::from_tensors(tensors, (0..n_nodes).collect())?);
        }
        Ok(inputs)
    }
}

/// The Dyson map of the `g0` workload on a `[3, n_points]` batch.
/// Points evaluated by the patched Dyson map, so a traced kernel run of `g0`
/// reports how many points it evaluated (the library's `Product` is not
/// counted, so `g0sq` traces report 0).
static OPERATOR_POINTS: AtomicU64 = AtomicU64::new(0);

fn dyson(inputs: ColMajorArrayRef<'_, Complex64>, output: &mut [Complex64]) -> Result<()> {
    OPERATOR_POINTS.fetch_add(output.len() as u64, Ordering::Relaxed);
    let data = inputs.data();
    for (p, value) in output.iter_mut().enumerate() {
        let (cx, cy, w) = (data[3 * p], data[3 * p + 1], data[3 * p + 2]);
        *value = 1.0 / (MU + 2.0 * cx + 2.0 * cy + Complex64::i() * w);
    }
    Ok(())
}

/// The Dyson map in TreeACI's batch form (unpatched runs).
fn dyson_batch(
    batch: TreeElementwiseBatch<'_, Complex64>,
    output: &mut [Complex64],
) -> tensor4all_treeaci::Result<()> {
    for (p, value) in output.iter_mut().enumerate() {
        *value = 1.0
            / (MU
                + 2.0 * batch.get(0, p)?
                + 2.0 * batch.get(1, p)?
                + Complex64::i() * batch.get(2, p)?);
    }
    Ok(())
}

fn product_batch(
    batch: TreeElementwiseBatch<'_, Complex64>,
    output: &mut [Complex64],
) -> tensor4all_treeaci::Result<()> {
    for (p, value) in output.iter_mut().enumerate() {
        *value = batch.get(0, p)? * batch.get(1, p)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Exact norms and the sampling measure

/// `s[h]`, for the nonnegative frequency `h` (`omega = 2 pi T (h + 1/2)`), is
/// the sum of `|G0|^(2 power)` over all `kx, ky`; each frequency `-omega`
/// gives the same sum.
fn frequency_sums(grid: &Grid, power: i32) -> Vec<f64> {
    let size = grid.size();
    let half = size / 2;
    let cosines: Vec<(f64, f64)> = (0..=half)
        .map(|k| {
            let multiplicity = if k == 0 || k == half { 1.0 } else { 2.0 };
            ((2.0 * PI * k as f64 / size as f64).cos(), multiplicity)
        })
        .collect();
    let omegas2: Vec<f64> = (0..half)
        .map(|h| {
            let w = 2.0 * PI * grid.temperature * (h as f64 + 0.5);
            w * w
        })
        .collect();
    let mut sums = vec![0.0; half];
    for (i, &(ci, mi)) in cosines.iter().enumerate() {
        for &(cj, mj) in &cosines[i..] {
            let weight = if ci == cj { mi * mj } else { 2.0 * mi * mj };
            let a = MU + 2.0 * ci + 2.0 * cj;
            let a2 = a * a;
            for (sum, &w2) in sums.iter_mut().zip(&omegas2) {
                let inverse = 1.0 / (a2 + w2);
                *sum += weight * inverse.powi(power);
            }
        }
    }
    sums
}

struct Samples {
    /// Column-major `[3 R, n]` site values.
    points: Vec<usize>,
    /// Grid coordinates `(kx, ky, n)` of each point.
    coords: Vec<[usize; 3]>,
    /// The density of the mixture at each point.
    density: Vec<f64>,
    /// The exact L2 norm `S` of the analytic function on the grid.
    norm: f64,
}

/// Draw `count` points from `q = (|f|^2 / S^2 + 1 / V) / 2` with
/// `|f|^2 = |G0|^(2 power)`.
fn draw(grid: &Grid, power: i32, count: usize, seed: u64) -> Samples {
    let size = grid.size();
    let half = size / 2;
    let sums = frequency_sums(grid, power);
    let total: f64 = 2.0 * sums.iter().sum::<f64>();
    // CDF over all frequencies n in 0..size.
    let mass = |n: usize| {
        let h = if n >= half { n - half } else { half - 1 - n };
        sums[h]
    };
    let mut cdf = Vec::with_capacity(size);
    let mut acc = 0.0;
    for n in 0..size {
        acc += mass(n);
        cdf.push(acc / total);
    }
    let volume = (size as f64).powi(3);
    let modulus = |kx: usize, ky: usize, n: usize| grid.g0(kx, ky, n).norm_sqr().powi(power);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut coords = Vec::with_capacity(count);
    for _ in 0..count {
        if rng.random::<bool>() {
            coords.push([
                rng.random_range(0..size),
                rng.random_range(0..size),
                rng.random_range(0..size),
            ]);
            continue;
        }
        let u: f64 = rng.random();
        let n = cdf.partition_point(|&c| c < u).min(size - 1);
        let w2 = grid.omega(n).powi(2);
        // Rejection from uniform k: |G0|^2 <= 1 / omega^2.
        loop {
            let (kx, ky) = (rng.random_range(0..size), rng.random_range(0..size));
            let accept = (modulus(kx, ky, n) * w2.powi(power)).min(1.0);
            if rng.random::<f64>() < accept {
                coords.push([kx, ky, n]);
                break;
            }
        }
    }
    let density = coords
        .iter()
        .map(|&[kx, ky, n]| 0.5 * modulus(kx, ky, n) / total + 0.5 / volume)
        .collect();
    let points = coords.iter().flat_map(|&c| grid.point(c)).collect();
    Samples {
        points,
        coords,
        density,
        norm: total.sqrt(),
    }
}

/// Every grid point with the uniform density `1 / V`, so that the estimate
/// of [`relative_l2`] is the exact sum.
fn every_point(grid: &Grid, power: i32) -> Samples {
    let size = grid.size();
    let sums = frequency_sums(grid, power);
    let total: f64 = 2.0 * sums.iter().sum::<f64>();
    let volume = (size as f64).powi(3);
    let coords: Vec<[usize; 3]> = (0..size)
        .flat_map(|kx| (0..size).flat_map(move |ky| (0..size).map(move |n| [kx, ky, n])))
        .collect();
    Samples {
        points: coords.iter().flat_map(|&c| grid.point(c)).collect(),
        density: vec![1.0 / volume; coords.len()],
        coords,
        norm: total.sqrt(),
    }
}

/// The largest pointwise error relative to the largest `|exact|`.
fn relative_max(approx: &[Complex64], exact: &[Complex64]) -> f64 {
    let error = approx
        .iter()
        .zip(exact)
        .fold(0.0_f64, |m, (a, e)| m.max((a - e).norm()));
    let scale = exact.iter().fold(0.0_f64, |m, e| m.max(e.norm()));
    error / scale
}

/// The relative L2 error of `approx` against `exact` and its standard error.
fn relative_l2(samples: &Samples, approx: &[Complex64], exact: &[Complex64]) -> (f64, f64) {
    let terms: Vec<f64> = approx
        .iter()
        .zip(exact)
        .zip(&samples.density)
        .map(|((a, e), q)| (a - e).norm_sqr() / q)
        .collect();
    let n = terms.len() as f64;
    let mean = terms.iter().sum::<f64>() / n;
    let var = terms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let rel = mean.sqrt() / samples.norm;
    // delta method: se(sqrt(E)) = se(E) / (2 sqrt(E)).
    let se = if mean > 0.0 {
        (var / n).sqrt() / (2.0 * mean.sqrt()) / samples.norm
    } else {
        0.0
    };
    (rel, se)
}

fn evaluate(partition: &Partition, grid: &Grid, samples: &Samples) -> Result<Vec<Complex64>> {
    let n = samples.coords.len();
    let values = partition.evaluate_batched(
        &grid.sites,
        ColMajorArrayRef::new(&samples.points, &[grid.sites.len(), n])?,
    )?;
    Ok(values.into_iter().map(Complex64::from).collect())
}

// ---------------------------------------------------------------------------
// Statistics of a result

fn tree_params(tree: &Tree) -> Result<usize> {
    let mut params = 0;
    for name in tree.node_names() {
        let index = tree.node_index(&name).context("node")?;
        params += tree
            .tensor(index)
            .context("tensor")?
            .dims()
            .iter()
            .product::<usize>();
    }
    Ok(params)
}

fn partition_stats(partition: &Partition) -> Result<Value> {
    let mut params = 0;
    let mut max_rank = 0;
    for patch in partition.values() {
        params += tree_params(patch.data())?;
        max_rank = max_rank.max(patch.max_bond_dim());
    }
    Ok(json!({ "patches": partition.len(), "params": params, "max_rank": max_rank }))
}

/// One line per accepted patch: its depth, the engine's verdict and estimate,
/// and the measurement that accepted it, against the per-point allowance
/// `tau = delta / sqrt(V)`.
fn dump_patches(result: &PatchedInterpolationResult<usize>, grid: &Grid) {
    let report = &result.report;
    let volume = (grid.size() as f64).powi(3);
    let tau = report.norm.delta().map(|delta| delta / volume.sqrt());
    for record in &report.accepted {
        let acceptance = record.acceptance.as_ref();
        // Fixed sites as `[site, value]`, site `3 b + v` = bit b of variable v.
        let mut fixed: Vec<[usize; 2]> = record
            .projector
            .iter()
            .filter_map(|(index, &value)| {
                grid.sites
                    .iter()
                    .position(|s| s == index)
                    .map(|site| [site, value])
            })
            .collect();
        fixed.sort_unstable();
        emit(json!({
            "kind": "patch",
            "depth": record.projector.len(),
            "fixed": fixed,
            "termination": format!("{:?}", record.termination),
            "engine_error_estimate": record.engine_error_estimate,
            "max_sample_magnitude": record.max_sample_magnitude,
            "max_bond_dim": record.max_bond_dim,
            "retries_used": record.retries_used,
            "tau": tau,
            "method": acceptance.map(|m| format!("{:?}", m.method)),
            "rms": acceptance.map(|m| m.rms),
            "max_residual": acceptance.map(|m| m.max_residual),
            "residual_rms": acceptance.and_then(|m| m.residual_rms),
            "sample_rms": acceptance.and_then(|m| m.sample_rms),
            "points": acceptance.map(|m| m.points),
        }));
    }
}

fn report_stats<K, R>(
    result: &PatchedInterpolationResult<usize>,
    kernel: &Timed<K>,
    residual: &Timed<R>,
) -> Value {
    let report = &result.report;
    json!({
        "kernel": kernel.json(),
        "residual_engine": residual.json(),
        "accepted": report.accepted.len(),
        "zero_patches": report.zero_patches.len(),
        "splits": report.splits,
        "function_evaluations": report.function_evaluations,
        "measurement_evaluations": report.measurement_evaluations,
        "residual_evaluations": report.residual_evaluations,
        "verification_failures": report.verification_failures,
        "engine_retries": report.engine_retries,
        "global_l2": format!("{:?}", report.norm.l2_error().map(|e| &e.global)),
    })
}

// ---------------------------------------------------------------------------
// Engine timing

/// An engine that records the number and the wall time of its runs, so the
/// time of a patched run splits into the kernel, the residual-TT estimator,
/// and the rest of the driver.
struct Timed<E> {
    inner: E,
    nanos: AtomicU64,
    calls: AtomicUsize,
    /// `Some(role)` emits one `engine_call` line per run (`--trace-calls`).
    trace: Option<&'static str>,
}

impl<E> Timed<E> {
    fn new(inner: E) -> Self {
        Self {
            inner,
            nanos: AtomicU64::new(0),
            calls: AtomicUsize::new(0),
            trace: None,
        }
    }

    fn traced(mut self, role: &'static str, enabled: bool) -> Self {
        self.trace = enabled.then_some(role);
        self
    }

    /// The `engine_call` line of one run, if traced.
    fn trace_call<V>(
        &self,
        problem: &InterpolationProblem<V>,
        outcome: &Result<InterpolationOutcome<V>, InterpolationError>,
        seconds: f64,
        points: u64,
    ) where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    {
        let Some(role) = self.trace else { return };
        let (termination, rank, estimate) = match outcome {
            Ok(outcome) => (
                format!("{:?}", outcome.termination),
                outcome.network.link_dims().into_iter().max().unwrap_or(1),
                outcome.error_estimate,
            ),
            Err(error) => (format!("error: {error}"), 0, f64::NAN),
        };
        emit(json!({
            "kind": "engine_call",
            "role": role,
            "call": self.calls.load(Ordering::Relaxed),
            "active": problem.site_order().len(),
            "tolerance": problem.absolute_tolerance(),
            "cap": problem.max_bond_dim().map(|c| c.get()),
            "termination": termination,
            "rank": rank,
            "estimate": estimate,
            "seconds": seconds,
            "points": points,
        }));
    }

    fn time<R>(&self, run: impl FnOnce() -> R) -> R {
        let start = Instant::now();
        let result = run();
        self.nanos
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        result
    }

    fn json(&self) -> Value {
        json!({
            "calls": self.calls.load(Ordering::Relaxed),
            "seconds": self.nanos.load(Ordering::Relaxed) as f64 * 1e-9,
        })
    }
}

impl<T, Op, E: ElementwiseInterpolator<T, Op>> ElementwiseInterpolator<T, Op> for Timed<E> {
    fn interpolate_elementwise<V>(
        &self,
        problem: &InterpolationProblem<V>,
        inputs: &[TreeTN<IdxTensor, V>],
        operator: &Op,
        ctx: &mut EvalContext<'_>,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    {
        let before = OPERATOR_POINTS.load(Ordering::Relaxed);
        let start = Instant::now();
        let outcome = self.time(|| {
            self.inner
                .interpolate_elementwise(problem, inputs, operator, ctx)
        });
        let points = OPERATOR_POINTS.load(Ordering::Relaxed) - before;
        self.trace_call(problem, &outcome, start.elapsed().as_secs_f64(), points);
        outcome
    }
}

impl<T, E: TreeInterpolator<T>> TreeInterpolator<T> for Timed<E> {
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        ctx: &mut EvalContext<'_>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
        F: FnMut(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>>,
    {
        let mut evaluate = evaluate;
        let mut points = 0u64;
        let start = Instant::now();
        let outcome = self.time(|| {
            self.inner.interpolate(problem, ctx, |batch, ctx| {
                points += batch.shape().get(1).copied().unwrap_or(0) as u64;
                evaluate(batch, ctx)
            })
        });
        self.trace_call(problem, &outcome, start.elapsed().as_secs_f64(), points);
        outcome
    }
}

/// Singular values of the dense target `values` (column-major over the
/// problem's site order) across every edge of the problem's tree: one entry
/// per edge with the smaller side's dimension and the singular values above
/// the SVD's default relative cutoff (1e-12), largest first.
fn cut_spectra<V>(problem: &InterpolationProblem<V>, values: &[Complex64]) -> Result<Vec<Value>>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let sites = problem.site_order().to_vec();
    let tensor = IdxTensor::from_dense(sites.clone(), values.to_vec())?;
    let topology = problem.topology();
    let graph = topology.graph();
    let mut out = Vec::new();
    for edge in graph.edge_indices() {
        let (a, b) = graph.edge_endpoints(edge).context("edge")?;
        // The component of `a` without the edge.
        let mut side = vec![a];
        let mut stack = vec![a];
        while let Some(node) = stack.pop() {
            for next in graph.neighbors(node) {
                if (node == a && next == b) || side.contains(&next) {
                    continue;
                }
                side.push(next);
                stack.push(next);
            }
        }
        let mut left: Vec<DynIndex> = Vec::new();
        for node in &side {
            let name = topology.node_name(*node).context("node name")?;
            left.extend(problem.node_sites()[name].iter().cloned());
        }
        if left.is_empty() || left.len() == sites.len() {
            continue;
        }
        let (_, s, _) = tensor4all_core::svd::<Complex64>(&tensor, &left)?;
        let r = s.dims()[0];
        // The singular values are real whatever the scalar type.
        let diag = s.dense_host_values::<f64>()?;
        let singular: Vec<f64> = (0..r).map(|i| diag[i * r + i].abs()).collect();
        let left_dim: usize = left.iter().map(IndexLike::dim).product();
        let right_dim = values.len() / left_dim;
        out.push(json!({ "dims": [left_dim, right_dim], "singular": singular }));
    }
    Ok(out)
}

/// A product engine that, after every run, compares the network with the
/// dense product of the restricted inputs and emits the exact errors next to
/// the engine's own estimate and the problem's tolerance. Diagnostics only:
/// the dense product has `2^active` entries.
struct AuditedProduct<'a, E>(&'a E);

impl<E: ElementwiseInterpolator<Complex64, Product>> ElementwiseInterpolator<Complex64, Product>
    for AuditedProduct<'_, E>
{
    fn interpolate_elementwise<V>(
        &self,
        problem: &InterpolationProblem<V>,
        inputs: &[TreeTN<IdxTensor, V>],
        operator: &Product,
        ctx: &mut EvalContext<'_>,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    {
        let outcome = self
            .0
            .interpolate_elementwise(problem, inputs, operator, ctx)?;
        let sites = problem.site_order();
        let dense = |tree: &TreeTN<IdxTensor, V>| -> Result<Vec<Complex64>> {
            Ok(tree
                .to_dense()?
                .permute_indices(sites)?
                .dense_host_values::<Complex64>()?)
        };
        let audit = || -> Result<Value> {
            let mut exact = dense(&inputs[0])?;
            for input in &inputs[1..] {
                for (e, v) in exact.iter_mut().zip(dense(input)?) {
                    *e *= v;
                }
            }
            let approx = dense(&outcome.network)?;
            let (mut max_err, mut sum_sq, mut max_f) = (0.0_f64, 0.0_f64, 0.0_f64);
            for (a, e) in approx.iter().zip(&exact) {
                let err = (a - e).norm();
                max_err = max_err.max(err);
                sum_sq += err * err;
                max_f = max_f.max(e.norm());
            }
            let spectra = cut_spectra(problem, &exact)?;
            Ok(json!({
                "kind": "kernel_call",
                "cuts": spectra,
                "active": sites.len(),
                "tau": problem.absolute_tolerance(),
                "cap": problem.max_bond_dim().map(|c| c.get()),
                "termination": format!("{:?}", outcome.termination),
                "rank": outcome.network.link_dims().into_iter().max().unwrap_or(1),
                "engine_estimate": outcome.error_estimate,
                "max_sample_magnitude": outcome.max_sample_magnitude,
                "exact_max_err": max_err,
                "exact_rms_err": (sum_sq / exact.len() as f64).sqrt(),
                "max_f": max_f,
            }))
        };
        match audit() {
            Ok(line) => emit(line),
            Err(error) => emit(json!({ "kind": "kernel_call", "audit_error": error.to_string() })),
        }
        Ok(outcome)
    }
}

/// A TreeACI kernel that, after every run ending below the cap, reruns the
/// same problem through `tree_elementwise_batched_with_rng_on_session` with
/// the adapter's options and seed, and emits the per-pass history of
/// `TreeAciResult` that the adapter drops. `reproduced` says whether the
/// rerun ended exactly like the measured run (same final estimate, rank and
/// termination). The rerun is outside the kernel's timing. Diagnostics only.
struct HistoryProbe<'a> {
    kernel: &'a Timed<TreeAciInterpolator>,
    margin: f64,
}

impl<Op: ElementwiseOperator<Complex64>> ElementwiseInterpolator<Complex64, Op>
    for HistoryProbe<'_>
{
    fn interpolate_elementwise<V>(
        &self,
        problem: &InterpolationProblem<V>,
        inputs: &[TreeTN<IdxTensor, V>],
        operator: &Op,
        ctx: &mut EvalContext<'_>,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    {
        let outcome = self
            .kernel
            .interpolate_elementwise(problem, inputs, operator, ctx)?;
        if outcome.termination == InterpolationTermination::BondCapReached {
            return Ok(outcome);
        }
        let call = self.kernel.calls.load(Ordering::Relaxed);
        // The adapter's `run_options`, with the harness's base options.
        let options = TreeAciOptions {
            tolerance: problem.absolute_tolerance() / self.kernel.inner.tolerance_factor(),
            scale_tolerance: false,
            max_bond_dim: problem.max_bond_dim().map(|cap| cap.get()),
            rng_seed: problem.seed(),
            ..aci_options::<V>(self.margin)
        };
        let mut rng = ChaCha8Rng::seed_from_u64(problem.seed());
        let callback = |batch: TreeElementwiseBatch<'_, Complex64>,
                        output: &mut [Complex64]|
         -> tensor4all_treeaci::Result<()> {
            let shape = [batch.n_inputs(), batch.n_points()];
            ColMajorArrayRef::new(batch.as_col_major_slice(), &shape)
                .map_err(anyhow::Error::from)
                .and_then(|values| operator.apply(values, output))
                .map_err(|error| tensor4all_treeaci::TreeAciError::Callback {
                    message: error.to_string(),
                })
        };
        let line = match tree_elementwise_batched_with_rng_on_session(
            callback,
            inputs,
            &options,
            &mut rng,
            ctx.session(),
        ) {
            Ok(rerun) => {
                let rank = rerun
                    .diagnostics
                    .edge_ranks
                    .iter()
                    .map(|&(_, _, rank)| rank)
                    .max()
                    .unwrap_or(0);
                // Below the cap, as the measured run was; a later variant
                // of the non-exhaustive enum counts as not reproduced.
                let termination = match rerun.termination {
                    TreeAciTermination::Converged => Some(InterpolationTermination::Converged),
                    TreeAciTermination::MaxSweeps => Some(InterpolationTermination::IterationLimit),
                    _ => None,
                };
                let outcome_rank = outcome.network.link_dims().into_iter().max().unwrap_or(0);
                json!({
                    "kind": "kernel_history",
                    "call": call,
                    "active": problem.site_order().len(),
                    "termination": format!("{:?}", outcome.termination),
                    "engine_tolerance": options.tolerance,
                    "min_sweeps": options.min_sweeps,
                    "max_sweeps": options.max_sweeps,
                    "reproduced": rerun.max_errors.last().copied() == Some(outcome.error_estimate)
                        && rank == outcome_rank
                        && termination == Some(outcome.termination),
                    "max_ranks": rerun.max_ranks,
                    "max_errors": rerun.max_errors,
                    "global_pivots_found": rerun.global_pivots_found,
                })
            }
            Err(error) => json!({
                "kind": "kernel_history",
                "call": call,
                "rerun_error": error.to_string(),
            }),
        };
        emit(line);
        Ok(outcome)
    }
}

// ---------------------------------------------------------------------------
// Methods

fn patched_options(grid: &Grid, args: &Args, norm: f64) -> Result<PatchedInterpolationOptions> {
    let error_norm = match args.error_norm.as_str() {
        "l2" => ErrorNorm::l2(L2Reference::Given(norm)),
        "sampled-max" => ErrorNorm::sampled_max(),
        other => bail!("unknown --error-norm {other} (l2 or sampled-max)"),
    };
    Ok(PatchedInterpolationOptions::new(args.cap)
        .with_error_norm(error_norm)
        .with_verification(VerificationOptions::new().with_audit(args.audit))
        .with_tolerance(ErrorTolerance {
            rtol: args.rtol,
            atol: 0.0,
        })
        .with_patch_order(grid.sites.clone())
        .with_variables(grid.variables())
        .with_seed(args.seed))
}

fn aci_options<V: TreeAciNode>(margin: f64) -> TreeAciOptions<V> {
    TreeAciOptions {
        max_sweeps: 50,
        global_tolerance_margin: margin,
        max_working_bytes: 8usize << 30,
        max_frame_bytes: 8usize << 30,
        max_sample_arena_bytes: 8usize << 30,
        message_cache_max_bytes: 8usize << 30,
        ..Default::default()
    }
}

fn single_patch(tree: Tree) -> Result<Partition> {
    Ok(PartitionedTreeTN::from_subdomain(
        SubDomainTreeTN::from_treetn(tree)?,
    )?)
}

/// Run an element-wise operation of `inputs` by `method`; return the
/// partition and the method's own statistics.
fn run_method(
    args: &Args,
    grid: &Grid,
    inputs: &[Partition],
    norm: f64,
    product: bool,
) -> Result<(Partition, Value, f64)> {
    let mut backend = CpuBackend::with_threads(1)?;
    let options = patched_options(grid, args, norm)?;
    let refs: Vec<&Partition> = inputs.iter().collect();
    let start = Instant::now();
    let (partition, stats) = match args.method.as_str() {
        "patched-aci" => {
            let engine = Timed::new(
                TreeAciInterpolator::new(aci_options::<usize>(args.aci_margin))?
                    .with_tolerance_factor(args.engine_tol_factor)?,
            )
            .traced("kernel", args.trace_calls);
            let kernel = &engine;
            let residual =
                Timed::new(TreeTciInterpolator::default()).traced("residual", args.trace_calls);
            let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
            let result = if product && args.audit_kernel {
                patched_elementwise::<Complex64, _, _, _, _>(
                    &AuditedProduct(kernel),
                    &residual,
                    &refs,
                    &Product,
                    &mut ctx,
                    &options,
                )?
            } else if product {
                patched_elementwise::<Complex64, _, _, _, _>(
                    kernel, &residual, &refs, &Product, &mut ctx, &options,
                )?
            } else if args.trace_history {
                let probe = HistoryProbe {
                    kernel,
                    margin: args.aci_margin,
                };
                patched_elementwise::<Complex64, _, _, _, _>(
                    &probe, &residual, &refs, &dyson, &mut ctx, &options,
                )?
            } else {
                patched_elementwise::<Complex64, _, _, _, _>(
                    kernel, &residual, &refs, &dyson, &mut ctx, &options,
                )?
            };
            if args.dump_patches {
                dump_patches(&result, grid);
            }
            let stats = report_stats(&result, &engine, &residual);
            (result.partition, stats)
        }
        "patched-rsi" => {
            if !product {
                bail!("tree RSI computes products only");
            }
            let engine = Timed::new(TreeRsiInterpolator::new(TreeRsiOptions::<usize> {
                rel_tol: args.rsi_rtol,
                ..Default::default()
            })?);
            let residual = Timed::new(TreeTciInterpolator::default());
            let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
            let result = patched_elementwise::<Complex64, _, _, _, _>(
                &engine, &residual, &refs, &Product, &mut ctx, &options,
            )?;
            if args.dump_patches {
                dump_patches(&result, grid);
            }
            let stats = report_stats(&result, &engine, &residual);
            (result.partition, stats)
        }
        "b0" => {
            let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
            let n_sites = grid.sites.len();
            let pivot = grid.point([0, 0, grid.size() / 2]);
            let sites = grid.sites.clone();
            // The black-box engine also runs the residual estimator, so its
            // time covers both.
            let engine = Timed::new(TreeTciInterpolator::default());
            let residual = Timed::new(());
            let result = patched_interpolate::<Complex64, _, _, _>(
                &engine,
                grid.topology()?,
                grid.node_sites(),
                ColMajorArray::new(pivot, vec![n_sites, 1])?,
                &mut ctx,
                |batch: ColMajorArrayRef<'_, usize>,
                 _ctx: &mut EvalContext<'_>|
                 -> Result<Vec<Complex64>> {
                    let n_points = batch.shape()[1];
                    let mut columns = vec![Complex64::new(0.0, 0.0); refs.len() * n_points];
                    for (i, input) in refs.iter().enumerate() {
                        let values = input.evaluate_batched(&sites, batch)?;
                        for (p, value) in values.into_iter().enumerate() {
                            columns[i + refs.len() * p] = Complex64::from(value);
                        }
                    }
                    let mut output = vec![Complex64::new(0.0, 0.0); n_points];
                    let shape = [refs.len(), n_points];
                    if product {
                        for (p, value) in output.iter_mut().enumerate() {
                            *value = columns[2 * p] * columns[2 * p + 1];
                        }
                    } else {
                        dyson(ColMajorArrayRef::new(&columns, &shape)?, &mut output)?;
                    }
                    Ok(output)
                },
                &options,
            )?;
            if args.dump_patches {
                dump_patches(&result, grid);
            }
            let stats = report_stats(&result, &engine, &residual);
            (result.partition, stats)
        }
        "unpatched-aci" => {
            let trees: Vec<Tree> = inputs
                .iter()
                .map(|p| Ok(p.values().next().context("single patch")?.data().clone()))
                .collect::<Result<_>>()?;
            let options = TreeAciOptions {
                tolerance: args.tol,
                scale_tolerance: args.scale_tolerance,
                max_bond_dim: Some(4000),
                rng_seed: args.seed,
                max_sweeps: args.max_sweeps,
                ..aci_options(args.aci_margin)
            };
            let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
            let mut session = Session::new(backend.engine_mut());
            let result = if product {
                tree_elementwise_batched_with_rng_on_session(
                    product_batch,
                    &trees,
                    &options,
                    &mut rng,
                    &mut session,
                )?
            } else {
                tree_elementwise_batched_with_rng_on_session(
                    dyson_batch,
                    &trees,
                    &options,
                    &mut rng,
                    &mut session,
                )?
            };
            let stats = json!({
                "termination": format!("{:?}", result.termination),
                "sweeps": result.max_errors.len(),
                "last_error": result.max_errors.last(),
                "max_ranks": result.max_ranks,
                "max_errors": result.max_errors,
            });
            (single_patch(result.tree)?, stats)
        }
        "unpatched-rsi" => {
            if !product {
                bail!("tree RSI computes products only");
            }
            let trees: Vec<Tree> = inputs
                .iter()
                .map(|p| Ok(p.values().next().context("single patch")?.data().clone()))
                .collect::<Result<_>>()?;
            let options = TreeRsiOptions::<usize> {
                rel_tol: args.rsi_rtol,
                max_bond_dim: Some(args.rsi_cap),
                seed: args.seed,
                ..Default::default()
            };
            let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
            let mut session = Session::new(backend.engine_mut());
            let result = hadamard_many_with_rng_on_session::<Complex64, _, _>(
                &trees,
                &options,
                &mut rng,
                &mut session,
            )?;
            (single_patch(result.tree)?, json!({}))
        }
        other => bail!("unknown method {other}"),
    };
    let seconds = start.elapsed().as_secs_f64();
    Ok((partition, stats, seconds))
}

/// The G0 input of the `g0sq` workload, built by the patched or unpatched
/// TreeACI method.
fn g0_input(
    args: &Args,
    grid: &Grid,
    g0_norm: f64,
    cap: usize,
    rtol: f64,
) -> Result<(Partition, Value)> {
    let unpatched = args.method.starts_with("unpatched");
    let input_args = Args {
        workload: "g0".into(),
        method: if unpatched {
            "unpatched-aci"
        } else {
            "patched-aci"
        }
        .into(),
        cap,
        rtol,
        tol: args.in_tol,
        dump_patches: false,
        trace_calls: false,
        trace_history: false,
        audit_kernel: false,
        // The inputs keep TreeACI's default margin whatever the product uses.
        aci_margin: 10.0,
        engine_tol_factor: 1.0,
        // The inputs keep the L2 acceptance whatever the product uses.
        error_norm: "l2".into(),
        audit: true,
        ..args.clone()
    };
    let exact = grid
        .inputs()?
        .into_iter()
        .map(single_patch)
        .collect::<Result<Vec<_>>>()?;
    let (partition, stats, seconds) = run_method(&input_args, grid, &exact, g0_norm, false)?;
    let mut summary = partition_stats(&partition)?;
    summary["method"] = json!(input_args.method);
    summary["seconds"] = json!(seconds);
    summary["engine"] = stats;
    Ok((partition, summary))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let grid = Grid::new(args.r, args.temperature, &args.topology)?;
    let product = args.workload.starts_with("g0sq");
    let power = if product { 2 } else { 1 };
    emit(json!({
        "kind": "config",
        "workload": args.workload, "method": args.method, "r": args.r,
        "temperature": args.temperature, "mu": MU, "topology": args.topology,
        "scale_tolerance": args.scale_tolerance, "max_sweeps": args.max_sweeps,
        "error_norm": args.error_norm, "audit": args.audit,
        "cap": args.cap, "rtol": args.rtol, "tol": args.tol,
        "rsi_rtol": args.rsi_rtol, "rsi_cap": args.rsi_cap,
        "in_cap": args.in_cap, "in_rtol": args.in_rtol, "in_tol": args.in_tol,
        "in2_cap": args.in2_cap, "in2_rtol": args.in2_rtol,
        "samples": args.samples, "seed": args.seed, "variants": args.variants,
        "git_commit": option_env!("T4A_BENCH_GIT_COMMIT"),
    }));

    let t = Instant::now();
    if args.exhaustive && args.r > 7 {
        bail!("--exhaustive is limited to r <= 7");
    }
    let samples = if args.exhaustive {
        every_point(&grid, power)
    } else {
        draw(&grid, power, args.samples, args.seed ^ 0x5eed)
    };
    let g0_norm = if product {
        frequency_sums(&grid, 1)
            .iter()
            .sum::<f64>()
            .mul_add(2.0, 0.0)
            .sqrt()
    } else {
        samples.norm
    };
    emit(json!({
        "kind": "reference", "norm": samples.norm, "g0_norm": g0_norm,
        "seconds": t.elapsed().as_secs_f64(),
    }));

    let inputs = if product {
        let (g0, summary) = g0_input(&args, &grid, g0_norm, args.in_cap, args.in_rtol)?;
        emit(json!({ "kind": "input", "factor": 0, "summary": summary }));
        let second = if args.workload == "g0sq-mixed" {
            let (other, summary) = g0_input(&args, &grid, g0_norm, args.in2_cap, args.in2_rtol)?;
            emit(json!({ "kind": "input", "factor": 1, "summary": summary }));
            other
        } else {
            g0.clone()
        };
        let common = PartitionedTreeTN::common_refinement(&[&g0, &second])?;
        let shared = g0.projectors().filter(|p| second.contains(p)).count();
        emit(json!({
            "kind": "input_layout",
            "patches": [g0.len(), second.len()],
            "shared_projectors": shared,
            "common_refinement": common.len(),
        }));
        vec![g0, second]
    } else {
        grid.inputs()?
            .into_iter()
            .map(single_patch)
            .collect::<Result<Vec<_>>>()?
    };

    let analytic: Vec<Complex64> = samples
        .coords
        .iter()
        .map(|&[kx, ky, n]| {
            let g = grid.g0(kx, ky, n);
            if product {
                g * g
            } else {
                g
            }
        })
        .collect();
    // f of the given inputs at the sample points (products only).
    let target = if product {
        let first = evaluate(&inputs[0], &grid, &samples)?;
        let second = evaluate(&inputs[1], &grid, &samples)?;
        let target: Vec<Complex64> = first.iter().zip(&second).map(|(a, b)| a * b).collect();
        let (rel_in, se_in) = relative_l2(&samples, &target, &analytic);
        emit(json!({
            "kind": "input_accuracy",
            "rel_l2_inputs_squared": rel_in, "rel_l2_inputs_squared_se": se_in,
        }));
        Some(target)
    } else {
        None
    };

    for overrides in &args.variants {
        let run_args = variant(&args, overrides)?;
        let (partition, stats, seconds) =
            run_method(&run_args, &grid, &inputs, samples.norm, product)?;

        let t = Instant::now();
        let approx = evaluate(&partition, &grid, &samples)?;
        let (rel, se) = relative_l2(&samples, &approx, &analytic);
        let mut accuracy = json!({ "rel_l2": rel, "rel_l2_se": se });
        if args.exhaustive {
            accuracy["exhaustive"] = json!(true);
            accuracy["rel_max"] = json!(relative_max(&approx, &analytic));
        }
        if let Some(target) = &target {
            let (rel_op, se_op) = relative_l2(&samples, &approx, target);
            accuracy["rel_l2_vs_inputs"] = json!(rel_op);
            accuracy["rel_l2_vs_inputs_se"] = json!(se_op);
            if args.exhaustive {
                accuracy["rel_max_vs_inputs"] = json!(relative_max(&approx, target));
            }
        }
        accuracy["seconds"] = json!(t.elapsed().as_secs_f64());

        let mut result = partition_stats(&partition)?;
        result["kind"] = json!("result");
        result["variant"] = json!(overrides);
        result["seconds"] = json!(seconds);
        result["engine"] = stats;
        result["accuracy"] = accuracy;
        emit(result);
    }
    Ok(())
}
