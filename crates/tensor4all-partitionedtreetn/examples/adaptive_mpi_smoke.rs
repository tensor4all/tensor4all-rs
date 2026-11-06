//! Distributed adaptive patching at any world size: the same tree
//! interpolation as the serial entry, split over the ranks of an MPI world.
//!
//! Run it under a launcher:
//!
//! ```sh
//! MPI_RT_LIB=/path/to/libmpiwrapper.so \
//!     mpiexec --oversubscribe --bind-to none -n 2 \
//!     target/debug/examples/adaptive_mpi_smoke
//! ```
//!
//! The root asserts the report and the dense reconstruction of `f`; every other
//! rank asserts that it received no result.

use std::collections::BTreeMap;
use std::error::Error;

use hataori_tenferro::TenferroLanes;
use tenferro_cpu::CpuEngine;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_interpolate_mpi, GlobalL2Error, PatchedInterpolationOptions,
};
use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, L2Reference};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::EvalContext;
use tensor4all_treetn::NodeNameNetwork;

#[cfg(all(feature = "hataori-mpi", not(feature = "hataori-mpi-rsmpi-rt")))]
use mpi;
use mpi::traits::Communicator;
#[cfg(feature = "hataori-mpi-rsmpi-rt")]
use mpi_rt as mpi;

/// `f(x) = 1 / (1 + x)` on three binary sites, `x = b0 + 2 b1 + 4 b2`.
fn f(point: &[usize]) -> f64 {
    1.0 / (1.0 + point.iter().sum::<usize>() as f64)
}

fn main() -> Result<(), Box<dyn Error>> {
    let (universe, provided) =
        mpi::initialize_with_threading(mpi::environment::Threading::Funneled)
            .expect("MPI must not be initialized yet");
    assert!(
        provided >= mpi::environment::Threading::Funneled,
        "the distributed entry keeps MPI progress on the initialization thread"
    );
    let world = universe.world();
    let rank = world.rank();

    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let mut topology = NodeNameNetwork::new();
    for node in 0..3usize {
        topology.add_node(node)?;
    }
    topology.add_edge(&0, &1)?;
    topology.add_edge(&1, &2)?;
    let node_sites: BTreeMap<usize, Vec<DynIndex>> = (0..3)
        .map(|node| (node, vec![sites[node].clone()]))
        .collect();
    let norm = (0..8)
        .map(|x| (1.0 / (1.0 + x as f64)).powi(2))
        .sum::<f64>()
        .sqrt();
    let options = PatchedInterpolationOptions::new(2)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
        .with_tolerance(ErrorTolerance {
            rtol: 1e-12,
            atol: 0.0,
        });
    let evaluate = |batch: ColMajorArrayRef<'_, usize>, _ctx: &mut EvalContext<'_>| {
        Ok::<Vec<f64>, anyhow::Error>(batch.data().chunks(3).map(f).collect())
    };

    let mut lanes = TenferroLanes::new(vec![CpuEngine::serial(), CpuEngine::serial()])?;
    let result = patched_interpolate_mpi(
        &world,
        &mut lanes,
        0,
        &TreeTciInterpolator::default(),
        topology,
        node_sites,
        ColMajorArray::new(vec![0, 0, 0], vec![3, 1])?,
        evaluate,
        &options,
    )?;

    match (rank, result) {
        (0, Some(result)) => {
            // Three splits and four exact leaf patches; the error is certified.
            assert_eq!(result.report.splits, 3, "root splits twice");
            assert_eq!(result.partition.len(), 4, "four leaf patches");
            assert_eq!(result.report.function_evaluations, 8, "every point once");
            let error = result.report.norm.l2_error().expect("an L2 report");
            assert!(
                matches!(error.global, GlobalL2Error::Certified { rms_error, .. } if rms_error == 0.0),
                "the leaves are exact, so the run is certified"
            );
            // The partition reproduces `f` on the whole domain.
            let values: Vec<f64> = (0..8)
                .map(|x| f(&[(x & 1), ((x >> 1) & 1), ((x >> 2) & 1)]))
                .collect();
            let reference = IdxTensor::from_dense(sites.clone(), values)?;
            let dense = result.partition.to_treetn()?.contract_to_tensor()?;
            let difference = dense.sub(&reference)?.maxabs()?;
            assert!(difference <= 1e-12, "dense residual {difference:e}");
            if world.size() > 1 {
                eprintln!("world size {}: distributed run agrees", world.size());
            }
        }
        (0, None) => panic!("the root owns the distributed result"),
        (_, Some(_)) => panic!("only the root receives a result"),
        (_, None) => {}
    }
    drop(universe);
    Ok(())
}
