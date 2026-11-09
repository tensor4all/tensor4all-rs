//! One oriented one-site ALS step with residual enrichment, driven directly
//! through `SquareLinsolveUpdater`, so tests can inspect the state right after
//! enrichment. Shared by the `integration` and `linsolve_enrichment_policy`
//! test targets.
use std::collections::HashMap;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{
    apply_local_update_sweep, CanonicalizationOptions, IndexMapping, LinsolveOptions,
    LocalUpdateStep, LocalUpdateSweepPlan, ResidualEnrichment, SquareLinsolveUpdater, TreeTN,
};

pub(crate) type State = TreeTN<IdxTensor, usize>;

/// One oriented ALS step `0 -> 1` on two three-dimensional sites with `A = I`,
/// `a0 = 0`, `a1 = 1`, initial state `|00>` and RHS
/// `|00> + |11> + small |22>`. The source solve keeps `|0>`, so the returned
/// state is the state right after enrichment, before any solve at node 1.
pub(crate) fn enrich_once(small: f64, enrichment: ResidualEnrichment) -> (State, [DynIndex; 2]) {
    let sites = [DynIndex::new_dyn(3), DynIndex::new_dyn(3)];
    let unit = |index: &DynIndex, bond: &DynIndex| {
        let mut data = vec![0.0; 3];
        data[0] = 1.0;
        IdxTensor::from_dense(vec![index.clone(), bond.clone()], data).unwrap()
    };
    let link = DynIndex::new_dyn(1);
    let init = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![unit(&sites[0], &link), unit(&sites[1], &link)],
        vec![0, 1],
    )
    .unwrap();
    let diag = |index: &DynIndex, bond: &DynIndex, values: [f64; 3]| {
        let mut data = vec![0.0; 9];
        for (k, value) in values.into_iter().enumerate() {
            data[4 * k] = value;
        }
        IdxTensor::from_dense(vec![index.clone(), bond.clone()], data).unwrap()
    };
    let link = DynIndex::new_dyn(3);
    let rhs = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            diag(&sites[0], &link, [1.0, 1.0, 1.0]),
            diag(&sites[1], &link, [1.0, 1.0, small]),
        ],
        vec![0, 1],
    )
    .unwrap();
    let link = DynIndex::new_dyn(1);
    let mut input = HashMap::new();
    let mut output = HashMap::new();
    let mut identities = Vec::new();
    for (node, site) in sites.iter().enumerate() {
        let s_in = site.sim();
        let s_out = s_in.prime();
        let mut data = vec![0.0; 9];
        for k in 0..3 {
            data[4 * k] = 1.0;
        }
        identities.push(
            IdxTensor::from_dense(vec![s_out.clone(), s_in.clone(), link.clone()], data).unwrap(),
        );
        input.insert(
            node,
            IndexMapping {
                true_index: site.clone(),
                internal_index: s_in,
            },
        );
        output.insert(
            node,
            IndexMapping {
                true_index: site.clone(),
                internal_index: s_out,
            },
        );
    }
    let operator = TreeTN::<IdxTensor, usize>::from_tensors(identities, vec![0, 1]).unwrap();
    let options = LinsolveOptions::new(1)
        .with_nsite(1)
        .with_coefficients(0.0, 1.0)
        .with_gmres_tol(1e-14)
        .with_enrichment(enrichment);
    let mut updater =
        SquareLinsolveUpdater::with_index_mappings(operator, input, output, rhs, options);
    let mut state = init
        .canonicalize([0], CanonicalizationOptions::default())
        .unwrap();
    let plan = LocalUpdateSweepPlan {
        nsite: 1,
        steps: vec![LocalUpdateStep {
            nodes: vec![0, 1],
            new_center: 1,
        }],
    };
    apply_local_update_sweep(&mut state, &plan, &mut updater).unwrap();
    (state, sites)
}

pub(crate) fn bond_dim(state: &State) -> usize {
    state
        .bond_index(state.edge_between(&0, &1).unwrap())
        .unwrap()
        .dim()
}

/// Max-abs distance of the represented state from `|00>`.
pub(crate) fn distance_from_00(state: &State, sites: &[DynIndex; 2]) -> f64 {
    let mut data = vec![0.0; 9];
    data[0] = 1.0;
    let expected = IdxTensor::from_dense(sites.to_vec(), data).unwrap();
    state
        .contract_to_tensor()
        .unwrap()
        .sub(&expected)
        .unwrap()
        .maxabs()
        .unwrap()
}
