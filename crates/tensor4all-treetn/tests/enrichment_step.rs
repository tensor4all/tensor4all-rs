//! One oriented one-site ALS step with residual enrichment, driven directly
//! through `SquareLinsolveUpdater`, so tests can inspect the state right after
//! enrichment. Shared by the `integration` and `linsolve_enrichment_policy`
//! test targets.
use std::collections::HashMap;

use num_complex::Complex64;

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
    EnrichmentStep {
        rhs: [1.0, 1.0, small].map(Complex64::from),
        coefficients: (0.0, 1.0),
        operator_scale: 1.0,
    }
    .run(enrichment)
    .unwrap()
}

/// The step of [`enrich_once`] with RHS `|00> + rhs[1] |11> + rhs[2] |22>`
/// (`rhs[0]` scales `|00>`), coefficients `(a0, a1)` and `A` the product of
/// `operator_scale * I` on each site. The RHS is complex if any `rhs[k]` has
/// a nonzero imaginary part, and real otherwise.
pub(crate) struct EnrichmentStep {
    pub(crate) rhs: [Complex64; 3],
    pub(crate) coefficients: (f64, f64),
    pub(crate) operator_scale: f64,
}

impl EnrichmentStep {
    pub(crate) fn run(
        &self,
        enrichment: ResidualEnrichment,
    ) -> anyhow::Result<(State, [DynIndex; 2])> {
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
        let complex = self.rhs.iter().any(|value| value.im != 0.0);
        let diag = |index: &DynIndex, bond: &DynIndex, values: [Complex64; 3]| {
            let mut data = vec![Complex64::new(0.0, 0.0); 9];
            for (k, value) in values.into_iter().enumerate() {
                data[4 * k] = value;
            }
            let indices = vec![index.clone(), bond.clone()];
            if complex {
                IdxTensor::from_dense(indices, data).unwrap()
            } else {
                IdxTensor::from_dense(indices, data.iter().map(|v| v.re).collect::<Vec<_>>())
                    .unwrap()
            }
        };
        let link = DynIndex::new_dyn(3);
        let rhs = TreeTN::<IdxTensor, usize>::from_tensors(
            vec![
                diag(&sites[0], &link, [Complex64::new(1.0, 0.0); 3]),
                diag(&sites[1], &link, self.rhs),
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
                data[4 * k] = self.operator_scale;
            }
            identities.push(
                IdxTensor::from_dense(vec![s_out.clone(), s_in.clone(), link.clone()], data)
                    .unwrap(),
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
            .with_coefficients(self.coefficients.0, self.coefficients.1)
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
        apply_local_update_sweep(&mut state, &plan, &mut updater)?;
        Ok((state, sites))
    }
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
