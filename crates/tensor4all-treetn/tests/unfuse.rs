use std::collections::HashSet;

use tensor4all_core::{DynIndex, IndexLike, LinearizationOrder, TensorDynLen};
use tensor4all_treetn::TreeTN;

#[test]
fn replace_site_index_with_indices_preserves_dense_tensor_values() {
    let fused = DynIndex::new_dyn(4);
    let tensor = TensorDynLen::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![tensor], vec![0]).unwrap();
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);

    let unfused = tn
        .replace_site_index_with_indices(
            &fused,
            &[left.clone(), right.clone()],
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();

    let dense = unfused.contract_to_tensor().unwrap();
    let expected =
        TensorDynLen::from_dense(vec![left.clone(), right.clone()], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
    assert!((&dense - &expected).maxabs() < 1.0e-12);

    let (site_indices, _) = unfused.all_site_indices().unwrap();
    let site_id_set: HashSet<_> = site_indices.iter().map(|idx| *idx.id()).collect();
    assert_eq!(site_indices.len(), 2);
    assert!(site_id_set.contains(left.id()));
    assert!(site_id_set.contains(right.id()));
}
