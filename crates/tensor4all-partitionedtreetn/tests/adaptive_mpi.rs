//! MPI smoke test of the distributed adaptive patch driver.
//!
//! This is its own test target because it initializes MPI, whose initialization
//! must happen before any other thread of the process uses it: run the binary
//! under a launcher, at world size one,
//!
//! ```sh
//! MPI_RT_LIB=/path/to/libmpiwrapper.so \
//!     mpiexec -n 1 target/debug/deps/adaptive_mpi-<hash> --nocapture
//! ```
//!
//! `scripts/check-mpi-smoke.sh` builds and runs it. The test asserts that the
//! distributed entries reproduce the serial ones at world size one: the
//! interpolation entry on the fixture the lanes tests use, and the element-wise
//! entry on a product of two partitions. MPI is initialized once per process,
//! so both runs share one test.

#[path = "adaptive_common/mod.rs"]
mod adaptive_common;

use adaptive_common::*;
use hataori_tenferro::TenferroLanes;
use std::collections::BTreeMap;
use tenferro_cpu::CpuEngine;
use tensor4all_core::IdxTensor;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_elementwise, patched_elementwise_mpi, patched_interpolate_mpi,
    PatchedInterpolationOptions,
};
use tensor4all_partitionedtreetn::{
    ErrorNorm, L2Reference, PartitionedTreeTN, Projector, SubDomainTreeTN,
};
use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::EvalContext;
use tensor4all_treetn::interpolation::{Parallelism, PointwiseElementwise, Product};
use tensor4all_treetn::NodeNameNetwork;
use tensor4all_treetn::TreeTN;

#[cfg(all(feature = "hataori-mpi", not(feature = "hataori-mpi-rsmpi-rt")))]
use mpi;
use mpi::traits::Communicator;
#[cfg(feature = "hataori-mpi-rsmpi-rt")]
use mpi_rt as mpi;

fn chain() -> (NodeNameNetwork<usize>, Vec<DynIndex>) {
    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let mut topology = NodeNameNetwork::new();
    for node in 0..3usize {
        topology.add_node(node).unwrap();
    }
    topology.add_edge(&0, &1).unwrap();
    topology.add_edge(&1, &2).unwrap();
    (topology, sites)
}

fn chain_f(point: &[usize]) -> f64 {
    1.0 / (1.0 + point.iter().sum::<usize>() as f64)
}

/// A rank-one network on the chain with node factors `factors`, masked onto
/// the patches `layout` of (site, value) pairs.
fn chain_partition(
    sites: &[DynIndex],
    factors: [[f64; 2]; 3],
    layout: &[&[(usize, usize)]],
) -> PartitionedTreeTN<usize> {
    let bonds = [DynIndex::new_dyn(1), DynIndex::new_dyn(1)];
    let tensor = |legs: Vec<DynIndex>, values: [f64; 2]| {
        IdxTensor::from_dense(legs, values.to_vec()).expect("tensor")
    };
    let tree = TreeTN::from_tensors(
        vec![
            tensor(vec![sites[0].clone(), bonds[0].clone()], factors[0]),
            tensor(
                vec![bonds[0].clone(), sites[1].clone(), bonds[1].clone()],
                factors[1],
            ),
            tensor(vec![bonds[1].clone(), sites[2].clone()], factors[2]),
        ],
        vec![0usize, 1, 2],
    )
    .expect("tree");
    PartitionedTreeTN::from_subdomains(
        layout
            .iter()
            .map(|pairs| {
                let projector = Projector::from_pairs(
                    pairs
                        .iter()
                        .map(|&(site, value)| (sites[site].clone(), value)),
                )
                .expect("projector");
                SubDomainTreeTN::new(tree.clone(), projector).expect("patch")
            })
            .collect(),
    )
    .expect("partition")
}

fn options() -> PatchedInterpolationOptions {
    let norm = (0..8)
        .map(|x| (1.0 / (1.0 + x as f64)).powi(2))
        .sum::<f64>()
        .sqrt();
    PatchedInterpolationOptions::new(2)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
        .with_tolerance(tol(1e-12))
}

#[test]
#[ignore = "initializes MPI: run it under a launcher, see scripts/check-mpi-smoke.sh"]
fn world_size_one_reproduces_the_serial_report() {
    // rsmpi-rt loads its runtime through this variable.
    assert!(
        std::env::var("MPI_RT_LIB").is_ok(),
        "set MPI_RT_LIB to the MPI wrapper library (see scripts/check-mpi-smoke.sh)"
    );
    let (universe, provided) =
        mpi::initialize_with_threading(mpi::environment::Threading::Funneled)
            .expect("MPI must not be initialized yet");
    assert!(provided >= mpi::environment::Threading::Funneled);
    let world = universe.world();
    let rank = world.rank();
    assert!(world.size() >= 1);

    let (topology, sites) = chain();
    let node_sites: BTreeMap<usize, Vec<DynIndex>> = (0..3)
        .map(|node| (node, vec![sites[node].clone()]))
        .collect();
    let evaluate = |batch: ColMajorArrayRef<'_, usize>, _ctx: &mut EvalContext<'_>| {
        Ok::<Vec<f64>, anyhow::Error>(batch.data().chunks(3).map(chain_f).collect())
    };
    let mut lanes = TenferroLanes::new(vec![CpuEngine::serial(), CpuEngine::serial()])
        .expect("one engine per lane");
    let distributed = patched_interpolate_mpi(
        &world,
        &mut lanes,
        0,
        &TreeTciInterpolator::default(),
        topology.clone(),
        node_sites.clone(),
        ColMajorArray::new(vec![0, 0, 0], vec![3, 1]).expect("pivots"),
        evaluate,
        &options(),
    )
    .expect("distributed run");

    // The element-wise entry: x split on site 0, y only on {site 1 = 0}.
    let x = chain_partition(
        &sites,
        [[1.0, 2.0], [1.0, 0.5], [3.0, 1.0]],
        &[&[(0, 0)], &[(0, 1)]],
    );
    let y = chain_partition(&sites, [[2.0, 1.0], [1.0, 1.0], [1.0, -1.0]], &[&[(1, 0)]]);
    let product_options = PatchedInterpolationOptions::new(4)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(10.0)))
        .with_tolerance(tol(1e-12));
    let mut lanes = TenferroLanes::new(vec![CpuEngine::serial(), CpuEngine::serial()])
        .expect("one engine per lane");
    let product = patched_elementwise_mpi::<f64, _, _, _, _, _>(
        &world,
        &mut lanes,
        0,
        &PointwiseElementwise(TreeTciInterpolator::default()),
        &TreeTciInterpolator::default(),
        &[&x, &y],
        &Product,
        &product_options,
    )
    .expect("distributed element-wise run");
    if rank != 0 {
        assert!(
            distributed.is_none() && product.is_none(),
            "only the root owns the distributed results"
        );
        drop(universe);
        return;
    }
    let product = product.expect("the root owns the element-wise result");
    let mut backend = CpuBackend::with_threads(1).expect("backend");
    let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
    let serial_product = patched_elementwise::<f64, _, _, _, _>(
        &PointwiseElementwise(TreeTciInterpolator::default()),
        &TreeTciInterpolator::default(),
        &[&x, &y],
        &Product,
        &mut ctx,
        &product_options,
    )
    .expect("serial element-wise run");
    assert_eq!(product.report.splits, serial_product.report.splits);
    assert_eq!(product.partition.len(), serial_product.partition.len());
    assert_eq!(
        product.report.zero_patches.len(),
        serial_product.report.zero_patches.len()
    );
    assert_eq!(
        product.report.function_evaluations,
        serial_product.report.function_evaluations
    );
    let distributed = distributed.expect("the root owns the result");
    assert_eq!(distributed.report.splits, 3);
    assert_eq!(distributed.partition.len(), 4);

    // The same problem on the serial entry of this rank.
    let problem = Problem::new(
        &[("a", &[2]), ("b", &[2]), ("c", &[2])],
        &[("a", "b"), ("b", "c")],
    );
    let serial = run(
        &TreeTciInterpolator::default(),
        &problem,
        &chain_f,
        &[vec![0, 0, 0]],
        &options(),
    )
    .expect("serial run");
    assert_eq!(
        distributed.report.function_evaluations,
        serial.report.function_evaluations
    );
    assert_eq!(distributed.report.splits, serial.report.splits);
    assert_eq!(
        distributed.report.accepted.len(),
        serial.report.accepted.len()
    );
    for (left, right) in distributed
        .report
        .accepted
        .iter()
        .zip(&serial.report.accepted)
    {
        assert_eq!(left.termination, right.termination);
        assert_eq!(left.max_bond_dim, right.max_bond_dim);
        assert_eq!(left.status, right.status);
        assert_eq!(
            left.acceptance.as_ref().map(|m| m.rms),
            right.acceptance.as_ref().map(|m| m.rms)
        );
    }
    drop(universe);
}
