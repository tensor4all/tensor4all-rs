//! The Matsubara G0 grid of the gw-rs workloads, shared by the M6c
//! element-wise benchmark and the M6a Fourier benchmark: the topologies
//! (`cttn`, `tt_nblock`), the analytic G0, and its three exact rank-2 inputs.
use std::collections::BTreeMap;
use std::f64::consts::PI;

use anyhow::{anyhow, bail, Result};
use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{NodeNameNetwork, TreeTN};

/// The chemical potential of the G0 workloads.
pub(crate) const MU: f64 = 0.5;

type Tree = TreeTN<IdxTensor, usize>;

/// The interleaved quantics grid: site `3 b + v` is bit `b` (MSB first) of
/// variable `v` (`kx`, `ky`, `n`); one site per node, node name = site.
pub(crate) struct Grid {
    pub(crate) r: usize,
    pub(crate) temperature: f64,
    pub(crate) sites: Vec<DynIndex>,
    pub(crate) edges: Vec<(usize, usize)>,
}

impl Grid {
    /// `topology` is `cttn` (gw-rs `cttn_seq_comb`) or `tt_nblock` (gw-rs
    /// `tt_chain_nblock`).
    pub(crate) fn new(r: usize, temperature: f64, topology: &str) -> Result<Self> {
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

    pub(crate) fn size(&self) -> usize {
        1 << self.r
    }

    pub(crate) fn omega(&self, n: usize) -> f64 {
        2.0 * PI * self.temperature * (n as f64 - self.size() as f64 / 2.0 + 0.5)
    }

    pub(crate) fn a(&self, kx: usize, ky: usize) -> f64 {
        let size = self.size() as f64;
        MU + 2.0 * (2.0 * PI * kx as f64 / size).cos() + 2.0 * (2.0 * PI * ky as f64 / size).cos()
    }

    pub(crate) fn g0(&self, kx: usize, ky: usize, n: usize) -> Complex64 {
        1.0 / Complex64::new(self.a(kx, ky), self.omega(n))
    }

    /// The site values of grid point `(kx, ky, n)`.
    pub(crate) fn point(&self, coords: [usize; 3]) -> Vec<usize> {
        let mut point = vec![0; 3 * self.r];
        for bit in 0..self.r {
            for var in 0..3 {
                point[3 * bit + var] = (coords[var] >> (self.r - 1 - bit)) & 1;
            }
        }
        point
    }

    pub(crate) fn topology(&self) -> Result<NodeNameNetwork<usize>> {
        let mut topology = NodeNameNetwork::new();
        for node in 0..self.sites.len() {
            topology.add_node(node).map_err(|e| anyhow!("{e}"))?;
        }
        for &(a, b) in &self.edges {
            topology.add_edge(&a, &b).map_err(|e| anyhow!("{e}"))?;
        }
        Ok(topology)
    }

    pub(crate) fn node_sites(&self) -> BTreeMap<usize, Vec<DynIndex>> {
        self.sites
            .iter()
            .enumerate()
            .map(|(node, site)| (node, vec![site.clone()]))
            .collect()
    }

    /// Each variable's bits, MSB first.
    pub(crate) fn variables(&self) -> Vec<Vec<DynIndex>> {
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
    pub(crate) fn inputs(&self) -> Result<Vec<Tree>> {
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

/// `s[h]`, for the nonnegative frequency `h` (`omega = 2 pi T (h + 1/2)`), is
/// the sum of `|G0|^(2 power)` over all `kx, ky`; each frequency `-omega`
/// gives the same sum.
pub(crate) fn frequency_sums(grid: &Grid, power: i32) -> Vec<f64> {
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
