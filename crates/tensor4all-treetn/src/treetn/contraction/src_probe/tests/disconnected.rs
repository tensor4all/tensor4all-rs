use super::super::contract_components;
use super::*;
use num_complex::Complex64;
use tensor4all_core::{TensorContractionLike, TensorElement};

fn check_one_sided_probe_batch<S: TensorElement>(sample: impl Fn(f64) -> S) {
    let output = DynIndex::new_dyn(2);
    let left = DynIndex::new_dyn(2);
    // Same logical ID, distinct concrete legs must remain an outer product.
    let right = left.prime();
    let batch = DynIndex::new_link(3).unwrap();
    let a = IdxTensor::from_dense(
        vec![left.clone(), output.clone()],
        (1..=4).map(|v| sample(v as f64)).collect(),
    )
    .unwrap();
    let b = IdxTensor::from_dense(vec![right.clone()], vec![sample(2.0), sample(-3.0)]).unwrap();
    let probes = ProbeBank::from_seed(vec![output.clone()], 5, 1472).unwrap();
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    for (a, b) in [(&a, &b), (&b, &a)] {
        let actual = probed_site_pair_batch_range(
            a,
            b,
            std::slice::from_ref(&output),
            &probes,
            1,
            3,
            &batch,
            &context,
        )
        .unwrap();
        assert_eq!(actual.indices().last(), Some(&batch));
        assert!(actual.indices().contains(&left));
        assert!(actual.indices().contains(&right));
        // Independent single-column contractions are the probe oracle, including
        // a nonzero starting column. Stack once for a whole-tensor comparison.
        let columns: Vec<_> = (1..4)
            .map(|column| {
                probed_site_pair(a, b, std::slice::from_ref(&output), &probes, column).unwrap()
            })
            .collect();
        let expected = IdxTensor::stack_along_new_index(
            &columns.iter().collect::<Vec<_>>(),
            batch.clone(),
            -1,
        )
        .unwrap();
        let residual = actual.sub(&expected).unwrap().maxabs().unwrap();
        assert!(residual < 1e-12, "probe column residual {residual:e}");
    }
}

#[test]
fn one_sided_probe_batch_real() {
    check_one_sided_probe_batch(|v| v);
}

#[test]
fn one_sided_probe_batch_complex() {
    check_one_sided_probe_batch(|v| Complex64::new(v, v.sin()));
}

#[test]
fn retaining_join_contracts_components_before_outer_products() {
    let batch = DynIndex::new_dyn(3);
    let p = DynIndex::new_dyn(2);
    let q = DynIndex::new_dyn(2);
    let a = IdxTensor::from_dense(vec![p.clone(), batch.clone()], vec![1., 2., 3., 4., 5., 6.])
        .unwrap();
    let probe =
        IdxTensor::from_dense(vec![p, batch.clone()], vec![2., 3., 5., 7., 11., 13.]).unwrap();
    let b = IdxTensor::from_dense(vec![q.clone()], vec![2., 3.]).unwrap();
    let c = IdxTensor::from_dense(vec![q], vec![5., 7.]).unwrap();
    let scalar = IdxTensor::from_dense(vec![], vec![2.]).unwrap();
    let expected = IdxTensor::from_dense(vec![batch.clone()], vec![496., 2666., 8246.]).unwrap();
    // The first factor is outside the batch component; both other components
    // need contraction, and a rank-zero factor is its own component.
    for factors in [
        vec![&b, &a, &scalar, &probe, &c],
        vec![&a, &probe, &b, &c, &scalar],
    ] {
        let actual = contract_retaining(&factors, &batch).unwrap();
        let residual = actual.sub(&expected).unwrap().maxabs().unwrap();
        assert!(residual < 1e-12, "component join residual {residual:e}");
    }
}

#[test]
fn retaining_join_preserves_gradients_through_disconnected_factors() {
    let batch = DynIndex::new_dyn(2);
    let p = DynIndex::new_dyn(2);
    let q = DynIndex::new_dyn(2);
    let a = IdxTensor::from_dense(vec![p.clone(), batch.clone()], vec![1., 2., 3., 4.])
        .unwrap()
        .enable_grad()
        .unwrap();
    let probe = IdxTensor::from_dense(vec![p, batch.clone()], vec![2., 3., 5., 7.]).unwrap();
    let b = IdxTensor::from_dense(vec![q], vec![2., 3.])
        .unwrap()
        .enable_grad()
        .unwrap();
    let actual = contract_retaining(&[&a, &b, &probe], &batch).unwrap();
    let ones = IdxTensor::from_dense(actual.indices().to_vec(), vec![1.; 4]).unwrap();
    actual.contract_pair(&ones).unwrap().backward().unwrap();
    assert_eq!(
        a.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![10., 15., 25., 35.]
    );
    assert_eq!(
        b.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![51., 51.]
    );
}

#[test]
fn retaining_join_rejects_empty_or_missing_batch_inputs() {
    let batch = DynIndex::new_dyn(2);
    let other = IdxTensor::from_dense(vec![batch.prime()], vec![1., 2.]).unwrap();
    assert!(contract_retaining::<IdxTensor>(&[], &batch).is_err());
    assert!(contract_retaining(&[&other], &batch).is_err());
    assert!(contract_retaining(&[&other, &other], &batch).is_err());
}

#[test]
fn unbatched_component_join_keeps_outer_legs_and_contracts_shared_legs() {
    let p = DynIndex::new_dyn(2);
    let q = p.prime();
    let a = IdxTensor::from_dense(vec![p.clone()], vec![2., 3.]).unwrap();
    let b = IdxTensor::from_dense(vec![p], vec![5., 7.]).unwrap();
    let c = IdxTensor::from_dense(vec![q.clone()], vec![11., 13.]).unwrap();
    let expected = IdxTensor::from_dense(vec![q], vec![341., 403.]).unwrap();
    let actual = contract_components(&[&a, &c, &b], None).unwrap();
    let residual = actual.sub(&expected).unwrap().maxabs().unwrap();
    assert!(
        residual < 1e-12,
        "unbatched component residual {residual:e}"
    );
    assert!(contract_components::<IdxTensor>(&[], None).is_err());
    // Core supports a summed hyperedge: keep all three factors in one component.
    let cube_sum = contract_components(&[&a, &a, &a], None).unwrap();
    assert_eq!(cube_sum.to_vec::<f64>().unwrap(), vec![35.]);
}
