// Recovered M5 fixed-depth partition exploration.
//
// This runs manually specified static partitions; it is not the adaptive
// pQTCI driver or a current-branch performance baseline.
//
// For each depth k, fix the first k sites of a patch order and run one
// uncapped TreeTCI (M1 engine `TreeTciInterpolator`) per patch, exactly as the
// M2/M3 driver builds a patch problem (same topology, fixed sites removed from
// `node_sites`, evaluator inserting the fixed coordinates). Absolute tolerance
// is `RTOL * max_ref` for every patch (the M2/M3 SampledMax engine tolerance
// with the analytic maximum as the reference).
//
// Usage: m5_partition <spectral|ridge|peaks> <chain|tree> <bits> <eta>
//        <depths, e.g. 0,1,2,3,4> <order: msb|xfirst> <points_per_stratum>
//        [max_iter]
// Prints JSONL to stdout, flushed per record.
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::io::Write;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
use tensor4all_treetci::{TreeTciInterpolator, TreeTciOptions};
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationProblem, TreeInterpolator,
};
use tensor4all_treetn::{NodeNameNetwork, TreeTN};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Name = String;
type Network = TreeTN<tensor4all_core::IdxTensor, Name>;

const RTOL: f64 = 1.0e-4;
const MU: f64 = 0.5;
const SEED: u64 = 7;
const VARS: [&str; 3] = ["x", "y", "z"];
const PEAK_CENTERS: [[f64; 3]; 4] = [
    [0.23, 0.61, 0.37],
    [0.71, 0.18, 0.84],
    [0.45, 0.92, 0.12],
    [0.88, 0.39, 0.55],
];

fn emit(value: serde_json::Value) {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = writeln!(lock, "{value}");
    let _ = lock.flush();
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn proc_status(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn reset_hwm() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Spectral,
    Ridge,
    Peaks,
    GRidge,
}

struct Fun {
    kind: Kind,
    eta: f64,
}

/// Range of cos(2 pi t) over the closed interval [lo, hi] within [0, 1).
fn cos_range(lo: f64, hi: f64) -> (f64, f64) {
    let tau = std::f64::consts::TAU;
    let (a, b) = ((tau * lo).cos(), (tau * hi).cos());
    let mut cmin = a.min(b);
    let mut cmax = a.max(b);
    if lo <= 0.5 && 0.5 <= hi {
        cmin = -1.0;
    }
    if lo <= 0.0 {
        cmax = 1.0;
    }
    (cmin, cmax)
}

fn dist_interval(t: f64, lo: f64, hi: f64) -> f64 {
    if t < lo {
        lo - t
    } else if t > hi {
        t - hi
    } else {
        0.0
    }
}

impl Fun {
    fn value(&self, r: [f64; 3]) -> f64 {
        let eta = self.eta;
        match self.kind {
            Kind::Spectral => {
                let tau = std::f64::consts::TAU;
                let e = -2.0 * r.iter().map(|c| (tau * c).cos()).sum::<f64>();
                eta / ((e - MU).powi(2) + eta * eta)
            }
            Kind::Ridge => {
                // squared perpendicular distance to the line x = y = z
                let s = r[0] + r[1] + r[2];
                let d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] - s * s / 3.0;
                eta / (d2.max(0.0) + eta * eta)
            }
            Kind::GRidge => {
                let s = r[0] + r[1] + r[2];
                let d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] - s * s / 3.0;
                (-d2.max(0.0) / (2.0 * eta * eta)).exp()
            }
            Kind::Peaks => PEAK_CENTERS
                .iter()
                .map(|c| {
                    let d2: f64 = (0..3).map(|i| (r[i] - c[i]).powi(2)).sum();
                    eta * eta / (d2 + eta * eta)
                })
                .sum(),
        }
    }

    /// Lower bound of sup over the closed box that is exact for spectral and
    /// ridge and the largest single-peak term for peaks.
    fn sup_box(&self, lo: [f64; 3], hi: [f64; 3]) -> f64 {
        let eta = self.eta;
        match self.kind {
            Kind::Spectral => {
                let mut emin = 0.0;
                let mut emax = 0.0;
                for i in 0..3 {
                    let (cmin, cmax) = cos_range(lo[i], hi[i]);
                    emin += -2.0 * cmax;
                    emax += -2.0 * cmin;
                }
                let d = dist_interval(MU, emin, emax);
                eta / (d * d + eta * eta)
            }
            Kind::Ridge | Kind::GRidge => {
                // min over t of sum_i dist(t, [lo_i, hi_i])^2 (convex in t)
                let g = |t: f64| -> f64 {
                    (0..3).map(|i| dist_interval(t, lo[i], hi[i]).powi(2)).sum()
                };
                let (mut a, mut b) = (0.0f64, 1.0f64);
                for _ in 0..200 {
                    let m1 = a + (b - a) / 3.0;
                    let m2 = b - (b - a) / 3.0;
                    if g(m1) <= g(m2) {
                        b = m2;
                    } else {
                        a = m1;
                    }
                }
                let d2 = g(0.5 * (a + b));
                if self.kind == Kind::GRidge {
                    (-d2 / (2.0 * eta * eta)).exp()
                } else {
                    eta / (d2 + eta * eta)
                }
            }
            Kind::Peaks => PEAK_CENTERS
                .iter()
                .map(|c| {
                    let d2: f64 = (0..3)
                        .map(|i| dist_interval(c[i], lo[i], hi[i]).powi(2))
                        .sum();
                    eta * eta / (d2 + eta * eta)
                })
                .fold(0.0, f64::max),
        }
    }

    fn feature_points(&self, bits: usize) -> Vec<[f64; 3]> {
        let n = (1u64 << bits) as f64;
        let mut pts = Vec::new();
        match self.kind {
            Kind::Spectral => {
                let tau = std::f64::consts::TAU;
                let m = 24;
                for a in 0..m {
                    for b in 0..m {
                        let u = (a as f64 + 0.37) / m as f64;
                        let v = (b as f64 + 0.61) / m as f64;
                        let cosine = -0.5 * MU - (tau * u).cos() - (tau * v).cos();
                        if cosine.abs() > 1.0 {
                            continue;
                        }
                        let t0 = cosine.acos() / tau;
                        for t in [t0, 1.0 - t0] {
                            pts.push([t, u, v]);
                            pts.push([u, t, v]);
                            pts.push([u, v, t]);
                        }
                    }
                }
            }
            Kind::Ridge | Kind::GRidge => {
                let m = 512usize.min(1 << bits);
                for a in 0..m {
                    let t = (a as f64 + 0.5) / m as f64;
                    pts.push([t, t, t]);
                }
            }
            Kind::Peaks => pts.extend(PEAK_CENTERS.iter().copied()),
        }
        // quantize to grid
        pts.iter()
            .map(|p| p.map(|c| ((c * n).floor()).min(n - 1.0) / n))
            .collect()
    }
}

struct Workload {
    bits: usize,
    topology: NodeNameNetwork<Name>,
    node_sites: BTreeMap<Name, Vec<DynIndex>>,
    /// Sites in the derived interpolation order of the full problem.
    sites: Vec<DynIndex>,
    /// (variable, bit) of every site in `sites`.
    vb: Vec<(usize, usize)>,
    /// node name of every site in `sites`.
    site_node: Vec<Name>,
    /// patch order as positions in `sites`.
    order: Vec<usize>,
    fun: Fun,
}

fn node_name(v: usize, b: usize) -> Name {
    format!("{}{b:02}", VARS[v])
}

fn build(kind: Kind, topo: &str, bits: usize, eta: f64, order: &str) -> Result<Workload> {
    let mut node_sites: BTreeMap<Name, Vec<DynIndex>> = BTreeMap::new();
    for b in 0..bits {
        for v in 0..3 {
            node_sites.insert(node_name(v, b), vec![DynIndex::new_dyn(2)]);
        }
    }
    let mut edges = Vec::new();
    match topo {
        "chain" => {
            let ord: Vec<Name> = (0..bits)
                .flat_map(|b| (0..3).map(move |v| node_name(v, b)))
                .collect();
            for w in ord.windows(2) {
                edges.push((w[0].clone(), w[1].clone()));
            }
        }
        "tree" => {
            node_sites.insert("r".to_string(), Vec::new());
            for v in 0..3 {
                edges.push(("r".to_string(), node_name(v, 0)));
                for b in 1..bits {
                    edges.push((node_name(v, b - 1), node_name(v, b)));
                }
            }
        }
        _ => return Err("topology".into()),
    }
    let mut topology = NodeNameNetwork::new();
    for node in node_sites.keys() {
        topology.add_node(node.clone())?;
    }
    for (l, r) in &edges {
        topology.add_edge(l, r)?;
    }
    let sites = InterpolationProblem::derive_site_order(&node_sites);
    let mut vb = Vec::new();
    let mut site_node = Vec::new();
    for s in &sites {
        let (name, _) = node_sites
            .iter()
            .find(|(_, ss)| ss.first() == Some(s))
            .ok_or("site")?;
        let v = VARS.iter().position(|p| name.starts_with(p)).ok_or("var")?;
        let b: usize = name[1..].parse()?;
        vb.push((v, b));
        site_node.push(name.clone());
    }
    let pos = |v: usize, b: usize| vb.iter().position(|&x| x == (v, b)).unwrap_or(usize::MAX);
    let order: Vec<usize> = match order {
        "msb" => (0..bits)
            .flat_map(|b| (0..3).map(move |v| (v, b)))
            .map(|(v, b)| pos(v, b))
            .collect(),
        // all bits of x first (MSB-first), then y, then z
        "xfirst" => (0..3)
            .flat_map(|v| (0..bits).map(move |b| (v, b)))
            .map(|(v, b)| pos(v, b))
            .collect(),
        _ => return Err("order".into()),
    };
    Ok(Workload {
        bits,
        topology,
        node_sites,
        sites,
        vb,
        site_node,
        order,
        fun: Fun { kind, eta },
    })
}

impl Workload {
    fn coords(&self, point: &[usize]) -> [f64; 3] {
        let mut c = [0.0; 3];
        for (&val, &(v, b)) in point.iter().zip(&self.vb) {
            c[v] += val as f64 * 0.5f64.powi(b as i32 + 1);
        }
        c
    }
    fn point_of(&self, r: [f64; 3]) -> Vec<usize> {
        let n = 1u64 << self.bits;
        let q = r.map(|c| ((c * n as f64).floor() as u64).min(n - 1));
        self.vb
            .iter()
            .map(|&(v, b)| ((q[v] >> (self.bits - 1 - b)) & 1) as usize)
            .collect()
    }
    fn key(point: &[usize]) -> u128 {
        point.iter().fold(0u128, |k, &x| (k << 1) | x as u128)
    }
    fn max_ref(&self) -> f64 {
        match self.fun.kind {
            Kind::Spectral | Kind::Ridge => 1.0 / self.fun.eta,
            Kind::GRidge => 1.0,
            Kind::Peaks => PEAK_CENTERS
                .iter()
                .map(|c| {
                    let p = self.point_of(*c);
                    self.fun.value(self.coords(&p))
                })
                .fold(0.0, f64::max),
        }
    }
    /// Patch box (closed, grid points only) for fixed values at order[0..k].
    fn patch_box(&self, fixed: &[Option<usize>]) -> ([f64; 3], [f64; 3]) {
        let mut lo = [0.0; 3];
        let mut m = [0usize; 3];
        for (pos, f) in fixed.iter().enumerate() {
            if let Some(val) = f {
                let (v, b) = self.vb[pos];
                lo[v] += *val as f64 * 0.5f64.powi(b as i32 + 1);
                m[v] = m[v].max(b + 1);
            }
        }
        let grid = 0.5f64.powi(self.bits as i32);
        let hi = [0, 1, 2].map(|v| lo[v] + 0.5f64.powi(m[v] as i32) - grid);
        (lo, hi)
    }
}

struct PatchResult {
    fixed: Vec<Option<usize>>,
    network: Option<Network>,
    active: Vec<DynIndex>,
    active_pos: Vec<usize>,
    rank: usize,
}

enum Axis {
    Site(usize),
    Child(usize),
    Parent,
}

/// Plain point evaluator of a tree network (dense node tensors, messages
/// toward a root); used only for the accuracy check.
struct DenseNet {
    /// nodes in BFS order from the root; children always come after parents
    dims: Vec<Vec<usize>>,
    data: Vec<Vec<f64>>,
    axes: Vec<Vec<Axis>>,
}

fn contract_axis(data: &[f64], dims: &[usize], axis: usize, v: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let before: usize = dims[..axis].iter().product();
    let d = dims[axis];
    let after: usize = dims[axis + 1..].iter().product();
    let mut out = vec![0.0; before * after];
    for a in 0..after {
        for k in 0..d {
            let w = v[k];
            if w == 0.0 {
                continue;
            }
            let base = (a * d + k) * before;
            let o = &mut out[a * before..(a + 1) * before];
            for (x, y) in o.iter_mut().zip(&data[base..base + before]) {
                *x += w * y;
            }
        }
    }
    let mut nd = dims.to_vec();
    nd.remove(axis);
    (out, nd)
}

impl DenseNet {
    fn new(net: &Network, active: &[DynIndex]) -> Result<Self> {
        let names = net.node_names();
        let mut tens = Vec::new();
        for n in &names {
            let idx = net.node_index(n).ok_or("node")?;
            let t = net.tensor(idx).ok_or("tensor")?;
            tens.push((t.indices().to_vec(), t.dims(), t.to_vec::<f64>()?));
        }
        let nn = names.len();
        // adjacency through shared non-site indices
        let shares = |a: usize, b: usize| -> Option<DynIndex> {
            tens[a]
                .0
                .iter()
                .find(|i| !active.contains(i) && tens[b].0.contains(i))
                .cloned()
        };
        let mut order = vec![0usize];
        let mut parent: Vec<Option<(usize, DynIndex)>> = vec![None; nn];
        let mut seen = vec![false; nn];
        seen[0] = true;
        let mut q = 0;
        while q < order.len() {
            let a = order[q];
            q += 1;
            for b in 0..nn {
                if !seen[b] {
                    if let Some(bond) = shares(a, b) {
                        seen[b] = true;
                        parent[b] = Some((a, bond));
                        order.push(b);
                    }
                }
            }
        }
        if order.len() != nn {
            return Err("disconnected".into());
        }
        let pos_in_order: Vec<usize> = {
            let mut p = vec![0; nn];
            for (i, &o) in order.iter().enumerate() {
                p[o] = i;
            }
            p
        };
        let mut dims = Vec::new();
        let mut data = Vec::new();
        let mut axes = Vec::new();
        for &o in &order {
            let (inds, d, v) = &tens[o];
            let mut ax = Vec::new();
            for i in inds {
                if let Some(s) = active.iter().position(|a| a == i) {
                    ax.push(Axis::Site(s));
                } else if parent[o].as_ref().map_or(false, |(_, b)| b == i) {
                    ax.push(Axis::Parent);
                } else {
                    let c = (0..nn)
                        .find(|&c| parent[c].as_ref().map_or(false, |(p, b)| *p == o && b == i))
                        .ok_or("bond without child")?;
                    ax.push(Axis::Child(pos_in_order[c]));
                }
            }
            dims.push(d.clone());
            data.push(v.clone());
            axes.push(ax);
        }
        Ok(Self { dims, data, axes })
    }

    fn eval(&self, point: &[usize]) -> f64 {
        let n = self.dims.len();
        let mut msg: Vec<Vec<f64>> = vec![Vec::new(); n];
        for i in (0..n).rev() {
            let mut cur = self.data[i].clone();
            let mut dims = self.dims[i].clone();
            for ax in (0..self.axes[i].len()).rev() {
                match self.axes[i][ax] {
                    Axis::Site(s) => {
                        let mut v = vec![0.0; dims[ax]];
                        v[point[s]] = 1.0;
                        let (c, d) = contract_axis(&cur, &dims, ax, &v);
                        cur = c;
                        dims = d;
                    }
                    Axis::Child(c) => {
                        let (c2, d) = contract_axis(&cur, &dims, ax, &msg[c]);
                        cur = c2;
                        dims = d;
                    }
                    Axis::Parent => {}
                }
            }
            msg[i] = cur;
        }
        msg[0].iter().sum()
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 7 {
        return Err("usage: <spectral|ridge|peaks> <chain|tree> <bits> <eta> <depths> <msb|xfirst> <pts_per_stratum> [max_iter]".into());
    }
    let kind = match args[0].as_str() {
        "spectral" => Kind::Spectral,
        "ridge" => Kind::Ridge,
        "peaks" => Kind::Peaks,
        "gridge" => Kind::GRidge,
        _ => return Err("function".into()),
    };
    let topo = args[1].clone();
    let bits: usize = args[2].parse()?;
    let eta: f64 = args[3].parse()?;
    let depths: Vec<usize> = args[4]
        .split(',')
        .map(|s| s.parse())
        .collect::<std::result::Result<_, _>>()?;
    let order_name = args[5].clone();
    let pts_per_stratum: usize = args[6].parse()?;
    let max_iter: usize = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(20);
    let rtol: f64 = args.get(8).map(|s| s.parse()).transpose()?.unwrap_or(RTOL);
    let w = build(kind, &topo, bits, eta, &order_name)?;
    let n_sites = w.sites.len();
    let max_ref = w.max_ref();
    let tol = rtol * max_ref;
    let engine = TreeTciInterpolator::new(TreeTciOptions {
        max_iter,
        ..Default::default()
    })?;
    let kmax = *depths.iter().max().unwrap_or(&0);
    emit(json!({
        "kind": "config", "function": args[0], "topology": topo, "bits": bits, "eta": eta,
        "sites": n_sites, "order": order_name, "depths": depths, "rtol": rtol,
        "max_ref": max_ref, "abs_tol": tol, "max_iter": max_iter,
        "order_sites": w.order.iter().take(kmax.max(1)).map(|&p| w.site_node[p].clone()).collect::<Vec<_>>(),
        "tolerance_def": "abs_tol = rtol * analytic max|f| for every patch; TreeTCI sampled pivot (max-norm) criterion; cap None",
    }));

    // test points: stratified over the depth-kmax patches
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0xACC);
    let mut test_points: Vec<Vec<usize>> = Vec::new();
    for p in 0..(1usize << kmax) {
        for _ in 0..pts_per_stratum {
            let mut pt: Vec<usize> = (0..n_sites).map(|_| rng.random_range(0..2)).collect();
            for j in 0..kmax {
                pt[w.order[j]] = (p >> (kmax - 1 - j)) & 1;
            }
            test_points.push(pt);
        }
    }
    let exact: Vec<f64> = test_points
        .iter()
        .map(|p| w.fun.value(w.coords(p)))
        .collect();
    let ref_l2: f64 = exact.iter().map(|v| v * v).sum::<f64>().sqrt();
    let feature_pts: Vec<Vec<usize>> = w
        .fun
        .feature_points(bits)
        .into_iter()
        .map(|r| w.point_of(r))
        .collect();

    for &k in &depths {
        let depth_start = Instant::now();
        let mut results: Vec<PatchResult> = Vec::new();
        let (mut sum_eval, mut sum_unique, mut sum_time) = (0usize, 0usize, 0.0f64);
        let (mut sum_r2, mut sum_r3, mut sum_params, mut sum_params_eager) =
            (0f64, 0f64, 0usize, 0usize);
        let (mut max_rank_feat, mut max_rank_nonfeat, mut n_feat, mut n_zero, mut n_not_conv) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut peak_hwm = 0u64;
        for p in 0..(1usize << k) {
            let mut fixed: Vec<Option<usize>> = vec![None; n_sites];
            for j in 0..k {
                fixed[w.order[j]] = Some((p >> (k - 1 - j)) & 1);
            }
            let projector: Vec<(String, usize)> = (0..k)
                .map(|j| (w.site_node[w.order[j]].clone(), (p >> (k - 1 - j)) & 1))
                .collect();
            let (lo, hi) = w.patch_box(&fixed);
            let sup = w.fun.sup_box(lo, hi) / max_ref;
            let feature = sup >= 0.5;
            if feature {
                n_feat += 1;
            }
            // active layout
            let mut node_sites = w.node_sites.clone();
            for (pos, f) in fixed.iter().enumerate() {
                if f.is_some() {
                    node_sites.insert(w.site_node[pos].clone(), Vec::new());
                }
            }
            let active = InterpolationProblem::derive_site_order(&node_sites);
            let active_pos: Vec<usize> = active
                .iter()
                .map(|s| w.sites.iter().position(|t| t == s).unwrap_or(usize::MAX))
                .collect();
            // initial pivots: compatible feature points (up to 8) + random to 5
            let compatible = |pt: &Vec<usize>| {
                fixed
                    .iter()
                    .zip(pt)
                    .all(|(f, &v)| f.map_or(true, |x| x == v))
            };
            let mut init: Vec<Vec<usize>> = Vec::new();
            let mut seen = HashSet::new();
            for fp in feature_pts.iter().filter(|q| compatible(q)) {
                if init.len() >= 8 {
                    break;
                }
                if seen.insert(Workload::key(fp)) {
                    init.push(fp.clone());
                }
            }
            let mut prng = ChaCha8Rng::seed_from_u64(mix(SEED ^ ((k as u64) << 40) ^ p as u64));
            while init.len() < 5 {
                let pt: Vec<usize> = (0..n_sites)
                    .map(|i| fixed[i].unwrap_or_else(|| prng.random_range(0..2)))
                    .collect();
                if seen.insert(Workload::key(&pt)) {
                    init.push(pt);
                }
            }
            let n_init = init.len();
            let piv: Vec<usize> = init
                .iter()
                .flat_map(|pt| active_pos.iter().map(|&i| pt[i]).collect::<Vec<_>>())
                .collect();
            let problem = InterpolationProblem::new(
                w.topology.clone(),
                node_sites,
                ColMajorArray::new(piv, vec![active.len(), n_init])?,
                tol,
                None::<NonZeroUsize>,
                mix(SEED ^ ((k as u64) << 32) ^ p as u64),
            )?;
            let requested = AtomicUsize::new(0);
            let unique: Mutex<HashSet<u128>> = Mutex::new(HashSet::new());
            let evaluate = |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
                let n = batch.shape()[1];
                requested.fetch_add(n, Ordering::Relaxed);
                let mut out = Vec::with_capacity(n);
                let mut full: Vec<usize> = fixed.iter().map(|f| f.unwrap_or(0)).collect();
                let mut u = unique.lock().map_err(|_| anyhow::anyhow!("poison"))?;
                for col in batch.data().chunks(active.len()) {
                    for (a, &i) in active_pos.iter().enumerate() {
                        full[i] = col[a];
                    }
                    u.insert(Workload::key(&full));
                    out.push(w.fun.value(w.coords(&full)));
                }
                Ok(out)
            };
            let rss_before = proc_status("VmRSS:");
            reset_hwm();
            let t0 = Instant::now();
            let outcome = engine.interpolate(&problem, evaluate);
            let secs = t0.elapsed().as_secs_f64();
            let hwm = proc_status("VmHWM:");
            peak_hwm = peak_hwm.max(hwm);
            let n_req = requested.load(Ordering::Relaxed);
            let n_uniq = unique.lock().map(|u| u.len()).unwrap_or(0);
            sum_eval += n_uniq;
            sum_unique += n_uniq;
            let _ = sum_unique;
            sum_time += secs;
            match outcome {
                Ok(out) => {
                    let link = out.network.link_dims();
                    let rank = link.iter().copied().max().unwrap_or(1);
                    let mut params = 0usize;
                    let mut params_eager = 0usize;
                    for name in out.network.node_names() {
                        let idx = out.network.node_index(&name).ok_or("node")?;
                        let size: usize = out
                            .network
                            .tensor(idx)
                            .ok_or("tensor")?
                            .dims()
                            .iter()
                            .product();
                        let nfixed = (0..n_sites)
                            .filter(|&i| fixed[i].is_some() && w.site_node[i] == name)
                            .count();
                        params += size;
                        params_eager += size << nfixed;
                    }
                    let conv = format!("{:?}", out.termination);
                    if conv != "Converged" {
                        n_not_conv += 1;
                    }
                    sum_r2 += (rank as f64).powi(2);
                    sum_r3 += (rank as f64).powi(3);
                    sum_params += params;
                    sum_params_eager += params_eager;
                    if feature {
                        max_rank_feat = max_rank_feat.max(rank);
                    } else {
                        max_rank_nonfeat = max_rank_nonfeat.max(rank);
                    }
                    emit(json!({
                        "kind": "patch", "depth": k, "patch": p, "projector": projector,
                        "box_lo": lo, "box_hi": hi, "sup_over_max": sup, "feature": feature,
                        "termination": conv, "rank": rank, "link_dims": link,
                        "params": params, "params_eager": params_eager,
                        "evals_unique": n_uniq, "evals_requested": n_req, "seconds": secs,
                        "hwm_kb": hwm, "rss_before_kb": rss_before,
                        "err_est_over_max": out.error_estimate / max_ref,
                        "max_sample_over_max": out.max_sample_magnitude / max_ref,
                        "n_initial_pivots": n_init,
                    }));
                    results.push(PatchResult {
                        fixed,
                        network: Some(out.network),
                        active,
                        active_pos,
                        rank,
                    });
                }
                Err(InterpolationError::AllSamplesZero) => {
                    n_zero += 1;
                    emit(json!({
                        "kind": "patch", "depth": k, "patch": p, "projector": projector,
                        "sup_over_max": sup, "feature": feature, "termination": "AllSamplesZero",
                        "rank": 0, "params": 0, "params_eager": 0, "evals_unique": n_uniq,
                        "evals_requested": n_req, "seconds": secs, "hwm_kb": hwm,
                    }));
                    results.push(PatchResult {
                        fixed,
                        network: None,
                        active,
                        active_pos,
                        rank: 0,
                    });
                }
                Err(e) => {
                    emit(
                        json!({"kind": "patch_error", "depth": k, "patch": p, "error": e.to_string()}),
                    );
                    return Err(Box::new(e));
                }
            }
        }
        // accuracy on stratified test points
        let t_acc = Instant::now();
        let mut approx = vec![0.0f64; test_points.len()];
        for (p, res) in results.iter().enumerate() {
            let _ = res.rank;
            let members: Vec<usize> = (0..test_points.len())
                .filter(|&i| (0..k).all(|j| test_points[i][w.order[j]] == (p >> (k - 1 - j)) & 1))
                .collect();
            debug_assert!(members.iter().all(|&i| res
                .fixed
                .iter()
                .zip(&test_points[i])
                .all(|(f, &v)| f.map_or(true, |x| x == v))));
            if let Some(net) = &res.network {
                if members.is_empty() {
                    continue;
                }
                let dn = DenseNet::new(net, &res.active)?;
                let vals: Vec<f64> = members
                    .iter()
                    .map(|&i| {
                        let pt: Vec<usize> =
                            res.active_pos.iter().map(|&a| test_points[i][a]).collect();
                        dn.eval(&pt)
                    })
                    .collect();
                for (&i, v) in members.iter().zip(vals) {
                    approx[i] = v;
                }
            }
        }
        let err_l2: f64 = exact
            .iter()
            .zip(&approx)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let err_max: f64 = exact
            .iter()
            .zip(&approx)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        emit(json!({
            "kind": "depth", "depth": k, "patches": 1usize << k, "feature_patches": n_feat,
            "zero_patches": n_zero, "not_converged": n_not_conv,
            "max_rank_feature": max_rank_feat, "max_rank_nonfeature": max_rank_nonfeat,
            "ranks": results.iter().map(|r| r.rank).collect::<Vec<_>>(),
            "sum_evals_unique": sum_eval, "sum_tci_seconds": sum_time,
            "sum_r2": sum_r2, "sum_r3": sum_r3, "sum_params": sum_params, "sum_params_eager": sum_params_eager,
            "peak_hwm_kb": peak_hwm, "test_points": test_points.len(),
            "rel_l2_sampled": err_l2 / ref_l2, "max_err_over_max": err_max / max_ref,
            "accuracy_seconds": t_acc.elapsed().as_secs_f64(), "depth_wall_seconds": depth_start.elapsed().as_secs_f64(),
        }));
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        emit(json!({"kind": "fatal", "error": e.to_string()}));
        std::process::exit(2);
    }
}
