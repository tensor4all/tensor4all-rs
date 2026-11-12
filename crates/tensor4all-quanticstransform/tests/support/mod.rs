#![allow(dead_code)]
use num_complex::Complex64;
use std::{collections::HashSet, fmt::Debug, hash::Hash};
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_quanticstransform::AxisBit;
use tensor4all_treetn::{
    apply_linear_operator, bind_linear_operator_indices, ApplyOptions, LinearOperator,
    RestructureOptions, SiteIndexNetwork, TreeTN,
};

pub const ROUND_OFF: f64 = 2e-10;
pub fn values(n: usize) -> Vec<Complex64> {
    (0..1usize << n)
        .map(|j| Complex64::new(1.0 + (0.7 * j as f64).sin(), (0.3 * j as f64).cos()))
        .collect()
}
pub fn chain<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    sites: &[DynIndex],
    values: &[Complex64],
    names: &[V],
) -> TreeTN<IdxTensor, V> {
    let dense = IdxTensor::from_dense(sites.to_vec(), values.to_vec()).unwrap();
    let tree = TreeTN::from_tensors(vec![dense], vec![0usize]).unwrap();
    let mut target = SiteIndexNetwork::new();
    for (site, name) in sites.iter().zip(names) {
        target
            .add_node(name.clone(), HashSet::from([site.clone()]))
            .unwrap();
    }
    for pair in names.windows(2) {
        target.add_edge(&pair[0], &pair[1]).unwrap();
    }
    tree.restructure_to(&target, &RestructureOptions::default())
        .unwrap()
}
pub fn apply(
    operator: &LinearOperator<IdxTensor, usize>,
    state: &TreeTN<IdxTensor, usize>,
    sites: &[DynIndex],
) -> TreeTN<IdxTensor, usize> {
    let input: Vec<_> = sites
        .iter()
        .enumerate()
        .map(|(j, s)| {
            (
                operator.get_input_mapping(&j).unwrap().true_index.clone(),
                s.clone(),
            )
        })
        .collect();
    let output: Vec<_> = sites
        .iter()
        .enumerate()
        .map(|(j, s)| {
            (
                operator.get_output_mapping(&j).unwrap().true_index.clone(),
                s.clone(),
            )
        })
        .collect();
    let bound = bind_linear_operator_indices(operator, &input, &output).unwrap();
    apply_linear_operator(&bound, state, ApplyOptions::naive()).unwrap()
}
pub fn assert_dense(
    actual: &IdxTensor,
    sites: &[DynIndex],
    expected: &[Complex64],
    tolerance: f64,
) {
    let oracle = IdxTensor::from_dense(sites.to_vec(), expected.to_vec()).unwrap();
    let error = actual.sub(&oracle).unwrap().maxabs().unwrap();
    assert!(
        error <= tolerance,
        "max error {error:e} exceeds {tolerance:e}"
    );
}
/// Independent small dense DFT. Flat data is column-major over physical sites;
/// omitted sites are spectators, and output significance reverses per axis.
pub fn dft(
    input: &[Complex64],
    layout: &[Option<AxisBit>],
    sign: f64,
    normalize: bool,
) -> Vec<Complex64> {
    let mut axes = std::collections::BTreeMap::<usize, Vec<(usize, usize)>>::new();
    for (position, entry) in layout.iter().enumerate() {
        if let Some(entry) = entry {
            axes.entry(entry.axis)
                .or_default()
                .push((entry.bit, position));
        }
    }
    for bits in axes.values_mut() {
        bits.sort();
    }
    let selected = layout.iter().filter(|x| x.is_some()).count();
    let scale = if normalize {
        2.0_f64.powf(-0.5 * selected as f64)
    } else {
        1.0
    };
    (0..input.len())
        .map(|output| {
            input
                .iter()
                .enumerate()
                .filter(|(k, _)| {
                    layout
                        .iter()
                        .enumerate()
                        .all(|(j, b)| b.is_some() || ((k ^ output) >> j) & 1 == 0)
                })
                .map(|(k, value)| {
                    let mut phase = 0.0;
                    for bits in axes.values() {
                        let x = bits.iter().fold(0usize, |x, (_, j)| 2 * x + ((k >> j) & 1));
                        let y = bits
                            .iter()
                            .rev()
                            .fold(0usize, |y, (_, j)| 2 * y + ((output >> j) & 1));
                        phase += (x * y) as f64 / (1usize << bits.len()) as f64;
                    }
                    value * Complex64::from_polar(scale, sign * 2.0 * std::f64::consts::PI * phase)
                })
                .sum()
        })
        .collect()
}
#[cfg(feature = "partitioned")]
pub fn result_dense<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    result: &tensor4all_quanticstransform::MergeRefineResult<V>,
) -> IdxTensor {
    result
        .regions()
        .flat_map(|(_, terms)| terms)
        .map(|term| term.data().to_dense().unwrap())
        .reduce(|a, b| a.add(&b).unwrap())
        .expect("nonzero output")
}
#[cfg(feature = "partitioned")]
pub fn partition<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    tree: &TreeTN<IdxTensor, V>,
    prefixes: &[Vec<(DynIndex, usize)>],
) -> tensor4all_partitionedtreetn::PartitionedTreeTN<V> {
    use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    PartitionedTreeTN::from_subdomains(
        prefixes
            .iter()
            .map(|pairs| {
                SubDomainTreeTN::new(tree.clone(), Projector::from_pairs(pairs.clone()).unwrap())
                    .unwrap()
            })
            .collect(),
    )
    .unwrap()
}
