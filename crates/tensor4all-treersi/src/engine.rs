//! Tree extension of RSI (arXiv:2602.17974v1, III.A.1–4).
//!
//! The original TT routines in Recursive-Sketched-Interpolation,
//! src/multiply_rsi.py and src/sketch.py, revision 153b25a8, were consulted
//! during the earlier implementation. This engine uses scale-carrying tree
//! messages and cached exact complements; it is not an author-code replay.
//! Interpolated frames remain exact slices, up to floating-point arithmetic.

use crate::{
    dense::{self, Dense},
    plan::{build_plan, checked_product, enforce_limit, node_tensor, Plan},
    scalar::{magnitude, product_underflows},
    scaled::{add_exponents, Scaled},
    Result, TreeRsiDiagnostics, TreeRsiEdgeReport, TreeRsiError, TreeRsiNode, TreeRsiOptions,
    TreeRsiResult, TreeRsiScalar,
};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use tensor4all_core::{
    matrix_luci_row_interpolation_owned_in, DynIndex, IdxTensor, IndexLike, RrLUOptions,
};
use tensor4all_tensorbackend::{CpuExecutionContext, ExecutionContext, Matrix};
use tensor4all_treetn::TreeTN;

type Blocks<T> = Vec<Option<Scaled<T>>>;

fn missing(message: &'static str) -> TreeRsiError {
    TreeRsiError::InternalInvariant { message }
}
fn block<T>(blocks: &[Option<Scaled<T>>], node: usize) -> Result<&Scaled<T>> {
    blocks
        .get(node)
        .and_then(Option::as_ref)
        .ok_or_else(|| missing("required message is missing"))
}

/// Every contraction checks its output block before allocation, normalizes
/// the result, and carries the input exponents into the result.
fn replace<T: TreeRsiScalar>(
    cpu: &CpuExecutionContext,
    input: Scaled<T>,
    axis: usize,
    weights: &Scaled<T>,
    limit: usize,
) -> Result<Scaled<T>> {
    let rank = weights.array.dims[0];
    let mut dims = input.array.dims.clone();
    dims[axis] = rank;
    enforce_limit(
        "axis contraction",
        checked_product(dims, "axis contraction")?,
        limit,
    )?;
    let exponent = add_exponents(input.exponent, weights.exponent)?;
    Scaled::new(
        dense::replace_axis(cpu, input.array, axis, &weights.array.data, rank)?,
        exponent,
    )
}

fn read_cores<T: TreeRsiScalar, V: TreeRsiNode>(
    plan: &Plan<V>,
    tree: &TreeTN<IdxTensor, V>,
    limit: usize,
) -> Result<Vec<Scaled<T>>> {
    let mut cores = Vec::with_capacity(plan.nodes.len());
    for (node, name) in plan.nodes.iter().enumerate() {
        let tensor = node_tensor(tree, name)?;
        let mut order = Vec::new();
        let mut dims = Vec::new();
        for &child in &plan.children[node] {
            let edge = tree
                .edge_between(name, &plan.nodes[child])
                .ok_or_else(|| missing("tree edge is missing"))?;
            let bond = tree
                .bond_index(edge)
                .ok_or_else(|| missing("tree bond is missing"))?;
            if bond.dim() == 0 {
                return Err(TreeRsiError::ZeroBond {
                    node: format!("{name:?}"),
                });
            }
            dims.push(bond.dim());
            order.push(bond.clone());
        }
        order.extend(plan.phys[node].indices.iter().cloned());
        dims.push(plan.phys[node].dim);
        if let Some(parent) = plan.parent[node] {
            let edge = tree
                .edge_between(name, &plan.nodes[parent])
                .ok_or_else(|| missing("parent edge is missing"))?;
            let bond = tree
                .bond_index(edge)
                .ok_or_else(|| missing("parent bond is missing"))?;
            if bond.dim() == 0 {
                return Err(TreeRsiError::ZeroBond {
                    node: format!("{name:?}"),
                });
            }
            dims.push(bond.dim());
            order.push(bond.clone());
        }
        if order.len() != tensor.indices().len() {
            return Err(missing("unclassified tensor index"));
        }
        enforce_limit(
            "input core",
            checked_product(dims.iter().copied(), "input core")?,
            limit,
        )?;
        let data = tensor
            .permute_indices(&order)
            .map_err(TreeRsiError::numerical)?
            .to_vec::<T>()
            .map_err(|e| TreeRsiError::ScalarKind {
                message: e.to_string(),
            })?;
        cores.push(Scaled::new(Dense { dims, data }, 0)?);
    }
    Ok(cores)
}

fn probes<T: TreeRsiScalar, V: TreeRsiNode, R: Rng + ?Sized>(
    plan: &Plan<V>,
    inputs: usize,
    options: &TreeRsiOptions<V>,
    rng: &mut R,
) -> Result<Vec<Blocks<T>>> {
    if options
        .probes
        .as_ref()
        .is_some_and(|p| p.n_inputs() != inputs)
    {
        return Err(TreeRsiError::InvalidProbes {
            message: "supply one probe map per input".into(),
        });
    }
    // Validate all supplied shapes before advancing the caller's RNG.
    for a in 0..inputs {
        for v in 0..plan.nodes.len() {
            if !plan.needs_probe[v] {
                continue;
            }
            let d = plan.phys[v].dim;
            enforce_limit(
                "probe",
                checked_product([d, plan.k], "probe")?,
                options.max_local_elements,
            )?;
            if let Some(set) = &options.probes {
                match set.probe(a, &plan.nodes[v]) {
                    Some(p)
                        if p.nrows() == d
                            && p.ncols() == plan.k
                            && p.as_col_major_slice().iter().all(|&x| {
                                x.is_finite()
                                    && magnitude(T::from_f64(x)).is_finite()
                                    && (x == 0.0 || magnitude(T::from_f64(x)) != 0.0)
                            }) => {}
                    None if d == 1 => {}
                    _ => {
                        return Err(TreeRsiError::InvalidProbes {
                            message: format!(
                            "input {a}, node {:?}: expected a finite, representable {d}x{} matrix",
                            plan.nodes[v], plan.k
                        ),
                        })
                    }
                }
            }
        }
    }
    (0..inputs)
        .map(|a| {
            (0..plan.nodes.len())
                .map(|v| {
                    if !plan.needs_probe[v] {
                        return Ok(None);
                    }
                    let d = plan.phys[v].dim;
                    let supplied = options
                        .probes
                        .as_ref()
                        .and_then(|p| p.probe(a, &plan.nodes[v]));
                    let data = if let Some(p) = supplied {
                        p.as_col_major_slice()
                            .iter()
                            .map(|&x| T::from_f64(x))
                            .collect()
                    } else if d == 1 {
                        vec![T::one(); plan.k]
                    } else {
                        (0..d * plan.k)
                            .map(|_| {
                                let x: f64 = StandardNormal.sample(&mut *rng);
                                T::from_f64(x)
                            })
                            .collect()
                    };
                    Ok(Some(Scaled::new(
                        Dense {
                            dims: vec![d, plan.k],
                            data,
                        },
                        0,
                    )?))
                })
                .collect()
        })
        .collect()
}

/// Product-state contraction with one shared probe label, performed by
/// backend GEMM/batched GEMM. A single scale per block preserves the original
/// sketch's relative column weights; independent per-column renormalization
/// would define a different sketch and pivot selection policy.
fn sketch<T: TreeRsiScalar>(
    cpu: &CpuExecutionContext,
    core: &Scaled<T>,
    mut factors: Vec<(usize, &Scaled<T>)>,
    k: usize,
    limit: usize,
) -> Result<Scaled<T>> {
    let first = factors
        .iter()
        .enumerate()
        .max_by_key(|(_, (axis, _))| (core.array.dims[*axis], *axis))
        .map(|(i, _)| i)
        .ok_or_else(|| missing("empty sketch factor list"))?;
    let (axis, weights) = factors.swap_remove(first);
    let dims = core
        .array
        .dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, d)| *d)
        .chain([k]);
    enforce_limit(
        "sketch block",
        checked_product(dims, "sketch block")?,
        limit,
    )?;
    let mut value = Scaled::new(
        dense::contract_axis_to_probe(cpu, &core.array, axis, &weights.array.data, k)?,
        add_exponents(core.exponent, weights.exponent)?,
    )?;
    let mut rest = factors
        .into_iter()
        .map(|(a, w)| (if a > axis { a - 1 } else { a }, w))
        .collect::<Vec<_>>();
    rest.sort_by_key(|(a, _)| std::cmp::Reverse(*a));
    for (axis, weights) in rest {
        value = Scaled::new(
            dense::contract_axis_diag_probe(cpu, value.array, axis, &weights.array.data)?,
            add_exponents(value.exponent, weights.exponent)?,
        )?;
    }
    Ok(value)
}

fn sketch_environments<T: TreeRsiScalar, V>(
    cpu: &CpuExecutionContext,
    plan: &Plan<V>,
    cores: &[Scaled<T>],
    probes: &Blocks<T>,
    limit: usize,
) -> Result<Blocks<T>> {
    let n = cores.len();
    let mut up: Blocks<T> = vec![None; n];
    let mut down: Blocks<T> = vec![None; n];
    let mut environments: Blocks<T> = vec![None; n];
    for &v in &plan.postorder {
        if !plan.needs_up[v] {
            continue;
        }
        let mut factors = plan.children[v]
            .iter()
            .enumerate()
            .map(|(axis, &c)| Ok((axis, block(&up, c)?)))
            .collect::<Result<Vec<_>>>()?;
        factors.push((plan.children[v].len(), block(probes, v)?));
        up[v] = Some(sketch(cpu, &cores[v], factors, plan.k, limit)?);
    }
    for &p in &plan.preorder {
        let children = &plan.children[p];
        let q = children.len();
        if !children
            .iter()
            .any(|&v| plan.needs_sketched_env[v] || plan.needs_down[v])
        {
            continue;
        }
        let d = plan.phys[p].dim;
        let rank_one = cores[p]
            .array
            .dims
            .iter()
            .enumerate()
            .all(|(axis, &dim)| axis == q || dim == 1)
            && d.checked_mul(plan.k).is_some_and(|size| size <= limit);
        if rank_one {
            // INVARIANT: rank-one bonds turn sibling contractions into
            // columnwise scalar products. Prefix/suffix reuse avoids degree^2
            // work on high-degree stars, without division through zero probes.
            let identity = Scaled::new(
                Dense {
                    dims: vec![1, plan.k],
                    data: vec![T::one(); plan.k],
                },
                0,
            )?;
            let mut prefix = vec![identity.clone()];
            for &v in children {
                let previous = prefix
                    .last()
                    .ok_or_else(|| missing("empty probe prefix"))?
                    .clone();
                prefix.push(if let Some(message) = &up[v] {
                    previous.multiply(message)?
                } else {
                    previous
                });
            }
            let mut suffix = if plan.parent[p].is_some() {
                block(&down, p)?.clone()
            } else {
                identity
            };
            for i in (0..q).rev() {
                let v = children[i];
                if plan.needs_sketched_env[v] || plan.needs_down[v] {
                    let weights = prefix[i].clone().multiply(&suffix)?;
                    let mut data = Vec::with_capacity(d * plan.k);
                    for &weight in &weights.array.data {
                        for &x in &cores[p].array.data {
                            let value = x * weight;
                            if product_underflows(x, weight) {
                                return Err(TreeRsiError::DynamicRange {
                                    stage: "rank-one sketch contraction",
                                });
                            }
                            data.push(value);
                        }
                    }
                    let env = Scaled::new(
                        Dense {
                            dims: vec![1, d, plan.k],
                            data,
                        },
                        add_exponents(cores[p].exponent, weights.exponent)?,
                    )?;
                    if plan.needs_down[v] {
                        let probe = block(probes, p)?;
                        down[v] = Some(Scaled::new(
                            dense::contract_axis_diag_probe(
                                cpu,
                                env.array.clone(),
                                1,
                                &probe.array.data,
                            )?,
                            add_exponents(env.exponent, probe.exponent)?,
                        )?);
                    }
                    if plan.needs_sketched_env[v] {
                        environments[v] = Some(env);
                    }
                }
                if let Some(message) = &up[v] {
                    suffix = suffix.multiply(message)?;
                }
            }
        } else {
            for &v in children {
                if !(plan.needs_sketched_env[v] || plan.needs_down[v]) {
                    continue;
                }
                let mut factors = children
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| **c != v)
                    .map(|(axis, &c)| Ok((axis, block(&up, c)?)))
                    .collect::<Result<Vec<_>>>()?;
                if plan.parent[p].is_some() {
                    factors.push((q + 1, block(&down, p)?));
                }
                if plan.needs_sketched_env[v] {
                    let env = sketch(cpu, &cores[p], factors, plan.k, limit)?;
                    if plan.needs_down[v] {
                        let probe = block(probes, p)?;
                        down[v] = Some(Scaled::new(
                            dense::contract_axis_diag_probe(
                                cpu,
                                env.array.clone(),
                                1,
                                &probe.array.data,
                            )?,
                            add_exponents(env.exponent, probe.exponent)?,
                        )?);
                    }
                    environments[v] = Some(env);
                } else {
                    factors.push((q, block(probes, p)?));
                    down[v] = Some(sketch(cpu, &cores[p], factors, plan.k, limit)?);
                }
            }
        }
    }
    Ok(environments)
}

/// Exact directed messages carry at most k physical assignments. This avoids
/// traversing/contracting an entire complement again at every exact edge.
struct Exact<T> {
    up: Blocks<T>,
    down: Blocks<T>,
}
impl<T: TreeRsiScalar> Exact<T> {
    fn build<V>(
        cpu: &CpuExecutionContext,
        plan: &Plan<V>,
        cores: &[Scaled<T>],
        limit: usize,
    ) -> Result<Self> {
        let mut cache = Self {
            up: vec![None; cores.len()],
            down: vec![None; cores.len()],
        };
        for &v in &plan.postorder {
            if !plan.exact_up[v] {
                continue;
            }
            let target = plan.children[v].len() + 1;
            let array = cache.partial(cpu, plan, cores, v, target, limit)?;
            cache.up[v] = Some(Self::close(array, target)?);
        }
        for &p in &plan.preorder {
            let children = &plan.children[p];
            let scalar = cores[p].array.dims.iter().all(|&d| d == 1)
                && children.iter().all(|&c| {
                    cache.up[c]
                        .as_ref()
                        .is_some_and(|m| m.array.data.len() == 1)
                })
                && (plan.parent[p].is_none()
                    || cache.down[p]
                        .as_ref()
                        .is_some_and(|m| m.array.data.len() == 1));
            if scalar && !children.is_empty() {
                // INVARIANT: all factors are scalars. Prefix/suffix products
                // compute every excluded-neighbor product in linear work,
                // including zeros (division by a message is never used).
                let mut first = cores[p].clone();
                first.array.dims = vec![1, 1];
                let mut prefix = vec![first];
                for &c in children {
                    let next = prefix
                        .last()
                        .ok_or_else(|| missing("empty scalar prefix"))?
                        .clone()
                        .multiply(block(&cache.up, c)?)?;
                    prefix.push(next);
                }
                let mut suffix = if plan.parent[p].is_some() {
                    block(&cache.down, p)?.clone()
                } else {
                    Scaled::new(
                        Dense {
                            dims: vec![1, 1],
                            data: vec![T::one()],
                        },
                        0,
                    )?
                };
                for i in (0..children.len()).rev() {
                    let v = children[i];
                    if plan.exact_down[v] {
                        cache.down[v] = Some(prefix[i].clone().multiply(&suffix)?);
                    }
                    suffix = suffix.multiply(block(&cache.up, v)?)?;
                }
            } else {
                for (target, &v) in children.iter().enumerate() {
                    if !plan.exact_down[v] {
                        continue;
                    }
                    let array = cache.partial(cpu, plan, cores, p, target, limit)?;
                    cache.down[v] = Some(Self::close(array, target)?);
                }
            }
        }
        Ok(cache)
    }

    fn partial<V>(
        &self,
        cpu: &CpuExecutionContext,
        plan: &Plan<V>,
        cores: &[Scaled<T>],
        v: usize,
        target: usize,
        limit: usize,
    ) -> Result<Scaled<T>> {
        let mut value = cores[v].clone();
        for (axis, &c) in plan.children[v].iter().enumerate() {
            if axis != target {
                value = replace(cpu, value, axis, block(&self.up, c)?, limit)?;
            }
        }
        let axis = plan.children[v].len() + 1;
        if plan.parent[v].is_some() && axis != target {
            value = replace(cpu, value, axis, block(&self.down, v)?, limit)?;
        }
        Ok(value)
    }

    fn close(value: Scaled<T>, target: usize) -> Result<Scaled<T>> {
        let mut order = (0..value.array.dims.len())
            .filter(|&a| a != target)
            .collect::<Vec<_>>();
        order.push(target);
        let bond = value.array.dims[target];
        let mut array = dense::permute(&value.array, &order);
        array.dims = vec![array.data.len() / bond, bond];
        Scaled::new(array, value.exponent)
    }

    fn environment<V>(
        &self,
        cpu: &CpuExecutionContext,
        plan: &Plan<V>,
        cores: &[Scaled<T>],
        v: usize,
        limit: usize,
    ) -> Result<Scaled<T>> {
        if let Some(message) = &self.down[v] {
            let rows = message.array.dims[0];
            let bond = message.array.dims[1];
            return Scaled::new(
                Dense {
                    dims: vec![bond, rows],
                    data: dense::transpose(&message.array.data, rows, bond),
                },
                message.exponent,
            );
        }
        let p = plan.parent[v].ok_or_else(|| missing("root has no complement"))?;
        let target = plan.children[p]
            .iter()
            .position(|&c| c == v)
            .ok_or_else(|| missing("child not in parent"))?;
        let value = self.partial(cpu, plan, cores, p, target, limit)?;
        let physical = plan.children[p].len();
        let mut order = vec![target, physical];
        order.extend((0..value.array.dims.len()).filter(|&a| a != target && a != physical));
        let mut array = dense::permute(&value.array, &order);
        let bond = array.dims[0];
        array.dims = vec![bond, array.data.len() / bond];
        Scaled::new(array, value.exponent)
    }
}

fn candidate<T: TreeRsiScalar, V>(
    cpu: &CpuExecutionContext,
    plan: &Plan<V>,
    cores: &[Scaled<T>],
    frames: &Blocks<T>,
    v: usize,
    limit: usize,
) -> Result<Scaled<T>> {
    let mut value = cores[v].clone();
    for (axis, &child) in plan.children[v].iter().enumerate() {
        value = replace(cpu, value, axis, block(frames, child)?, limit)?;
    }
    Ok(value)
}

pub(crate) fn run<T: TreeRsiScalar, V: TreeRsiNode, R: Rng + ?Sized>(
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeRsiOptions<V>,
    rng: &mut R,
    context: &ExecutionContext,
) -> Result<TreeRsiResult<V>> {
    let cpu = match context {
        ExecutionContext::Cpu(cpu) => cpu,
        // INVARIANT: the other context variant exists only with GPU features.
        #[allow(unreachable_patterns)]
        _ => {
            return Err(TreeRsiError::Context {
                message: "RSI requires a CPU execution context".into(),
            })
        }
    };
    for input in inputs {
        input
            .validate_context(context)
            .map_err(TreeRsiError::InputContext)?;
    }
    let plan = build_plan(inputs, options)?;
    let limit = options.max_local_elements;
    let cores = inputs
        .iter()
        .map(|x| read_cores::<T, V>(&plan, x, limit))
        .collect::<Result<Vec<_>>>()?;
    let probes = probes::<T, V, R>(&plan, inputs.len(), options, rng)?;
    let mut sketches = cores
        .iter()
        .zip(&probes)
        .map(|(c, p)| sketch_environments(cpu, &plan, c, p, limit))
        .collect::<Result<Vec<_>>>()?;
    let exact = cores
        .iter()
        .map(|c| Exact::build(cpu, &plan, c, limit))
        .collect::<Result<Vec<_>>>()?;
    let n = plan.nodes.len();
    let mut frames = vec![vec![None; n]; inputs.len()];
    let mut bonds: Vec<Option<DynIndex>> = vec![None; n];
    let mut output: Vec<Option<IdxTensor>> = vec![None; n];
    let mut edges = Vec::with_capacity(n - 1);
    for &v in plan.postorder.iter().chain(std::iter::once(&plan.root)) {
        let candidates = cores
            .iter()
            .zip(&frames)
            .map(|(c, f)| candidate(cpu, &plan, c, f, v, limit))
            .collect::<Result<Vec<_>>>()?;
        let mut product: Option<Scaled<T>> = None;
        let is_root = v == plan.root;
        let mut rows = 0;
        let mut columns = 1;
        for a in 0..inputs.len() {
            let c = &candidates[a];
            let value = if is_root {
                c.clone()
            } else {
                let chi = *c
                    .array
                    .dims
                    .last()
                    .ok_or_else(|| missing("empty candidate shape"))?;
                rows = c.array.data.len() / chi;
                let env = if plan.exact[v] {
                    exact[a].environment(cpu, &plan, &cores[a], v, limit)?
                } else {
                    sketches[a][v]
                        .take()
                        .ok_or_else(|| missing("missing sketch environment"))?
                };
                columns = env.array.data.len() / chi;
                enforce_limit(
                    "local matrix",
                    checked_product([rows, columns], "local matrix")?,
                    limit,
                )?;
                Scaled::new(
                    Dense {
                        dims: vec![rows, columns],
                        data: dense::mat_mul_cols(
                            cpu,
                            c.array.data.clone(),
                            rows,
                            chi,
                            env.array.data,
                            columns,
                        )?,
                    },
                    add_exponents(c.exponent, env.exponent)?,
                )?
            };
            product = Some(match product {
                Some(p) => p.multiply(&value)?,
                None => value,
            });
        }
        let product = product.ok_or(TreeRsiError::NoInputs)?;
        let mut indices = plan.children[v]
            .iter()
            .map(|&c| {
                bonds[c]
                    .clone()
                    .ok_or_else(|| missing("output child bond missing"))
            })
            .collect::<Result<Vec<_>>>()?;
        indices.extend(plan.phys[v].indices.iter().cloned());
        if is_root {
            output[v] = Some(
                IdxTensor::from_dense_in(context, indices, product.into_values()?)
                    .map_err(TreeRsiError::numerical)?,
            );
            continue;
        }
        let scale = product
            .array
            .data
            .iter()
            .map(|&x| magnitude(x))
            .fold(0.0_f64, f64::max);
        let cap = options
            .max_bond_dim
            .unwrap_or(usize::MAX)
            .min(rows)
            .min(columns);
        let id = matrix_luci_row_interpolation_owned_in(
            Matrix::try_from_col_major_vec(rows, columns, product.array.data)
                .map_err(TreeRsiError::numerical)?,
            Some(RrLUOptions {
                max_bond_dim: cap,
                rel_tol: options.rel_tol,
                abs_tol: 0.0,
                left_orthogonal: true,
            }),
            cpu,
        )
        .map_err(TreeRsiError::numerical)?;
        let (pivots, left) = if id.rows.is_empty() {
            let mut unit = vec![T::zero(); rows];
            unit[0] = T::one();
            (vec![0], unit)
        } else {
            (id.rows, id.interpolation.into_col_major_vec())
        };
        let rank = pivots.len();
        for a in 0..inputs.len() {
            let c = &candidates[a];
            let chi = *c
                .array
                .dims
                .last()
                .ok_or_else(|| missing("empty candidate shape"))?;
            frames[a][v] = Some(Scaled::new(
                Dense {
                    dims: vec![rank, chi],
                    data: dense::gather_rows(&c.array.data, rows, chi, &pivots),
                },
                c.exponent,
            )?);
            // Each child frame is used once by its parent; release it here.
            for &child in &plan.children[v] {
                frames[a][child] = None;
            }
        }
        let bond = DynIndex::new_dyn(rank);
        indices.push(bond.clone());
        bonds[v] = Some(bond);
        output[v] = Some(
            IdxTensor::from_dense_in(context, indices, left).map_err(TreeRsiError::numerical)?,
        );
        let p = plan.parent[v].ok_or_else(|| missing("non-root node has no parent"))?;
        edges.push(TreeRsiEdgeReport {
            child: plan.nodes[v].clone(),
            parent: plan.nodes[p].clone(),
            rank,
            rows,
            columns,
            exact_columns: plan.exact[v],
            rank_limit_reached: rank == cap,
            relative_pivot: if scale == 0.0 {
                0.0
            } else {
                id.pivot_magnitudes.last().copied().unwrap_or(0.0) / scale
            },
        });
    }
    let tensors = output
        .into_iter()
        .map(|x| x.ok_or_else(|| missing("output core missing")))
        .collect::<Result<Vec<_>>>()?;
    Ok(TreeRsiResult {
        tree: TreeTN::from_tensors(tensors, plan.nodes.clone())?,
        diagnostics: TreeRsiDiagnostics {
            root: plan.nodes[plan.root].clone(),
            sketch_dim: plan.k,
            edges,
            sketch_messages_per_input: plan.message_count(),
            exact_messages_per_input: plan.postorder.iter().filter(|&&v| plan.exact_up[v]).count()
                + plan
                    .postorder
                    .iter()
                    .filter(|&&v| plan.exact_down[v])
                    .count(),
        },
    })
}
