use super::validate_retained_batch;
use crate::evaluation::Compatibility;
use crate::{to_treetn, GlobalIndexBatch, TreeTCI2, TreeTciGraph};
use tensor4all_treetn::TreeTNCachedEvaluator;

#[test]
fn invalid_retained_oracle_batches_fail_before_approximation_readout() {
    let mut state =
        TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::linear_chain(2).unwrap()).unwrap();
    state.add_global_pivots(&[vec![0, 0]]).unwrap();
    let tree = to_treetn(&state, |batch| Ok(vec![1.0; batch.n_points()]), None).unwrap();
    let sites: Vec<_> = (0..2)
        .map(|site| {
            tree.tensor(tree.node_index(&site).unwrap())
                .unwrap()
                .indices()[0]
                .clone()
        })
        .collect();
    let mut cache = TreeTNCachedEvaluator::new(&tree, &sites, Default::default()).unwrap();
    for fails in [false, true] {
        let mut target = |_batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
            if fails {
                anyhow::bail!("retained target failed");
            }
            Ok(Vec::new())
        };
        let mut execution = Compatibility::new(&mut target);
        let error = validate_retained_batch(
            &mut execution,
            &mut cache,
            &tree,
            &sites,
            &[0, 0],
            &mut 0.0,
            &mut Vec::new(),
            &mut |_, _, _, _, _| panic!("invalid target output must precede readout"),
            super::super::cached_walk_readout_on_session,
        )
        .unwrap_err();
        assert!(error.to_string().contains(if fails {
            "retained target failed"
        } else {
            "target returned 0 values for 1 retained pivots"
        }));
    }
}
