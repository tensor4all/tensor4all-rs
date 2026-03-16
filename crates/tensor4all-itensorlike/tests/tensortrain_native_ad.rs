use tensor4all_core::{forward_ad, Index, TensorDynLen};
use tensor4all_itensorlike::{CanonicalForm, TensorTrain, TruncateOptions};

fn make_two_site_tt(fw: &forward_ad::DualLevel<'_>) -> TensorTrain {
    let s0 = Index::new_dyn(2);
    let s1 = Index::new_dyn(2);
    let bond = Index::new_dyn(2);

    let t0_primal =
        TensorDynLen::from_dense(vec![s0.clone(), bond.clone()], vec![1.0, 0.0, 0.0, 2.0]).unwrap();
    let t0_tangent =
        TensorDynLen::from_dense(vec![s0, bond.clone()], vec![0.1, 0.0, 0.0, -0.2]).unwrap();
    let t1_primal =
        TensorDynLen::from_dense(vec![bond.clone(), s1.clone()], vec![3.0, 0.0, 0.0, 4.0]).unwrap();
    let t1_tangent = TensorDynLen::from_dense(vec![bond, s1], vec![0.3, 0.0, 0.0, -0.4]).unwrap();

    let t0 = fw.make_dual(&t0_primal, &t0_tangent).unwrap();
    let t1 = fw.make_dual(&t1_primal, &t1_tangent).unwrap();

    TensorTrain::new(vec![t0, t1]).unwrap()
}

#[test]
fn orthogonalize_preserves_forward_payload() {
    forward_ad::dual_level(|fw| {
        let mut tt = make_two_site_tt(fw);

        tt.orthogonalize_with(1, CanonicalForm::Unitary).unwrap();

        for site in 0..tt.len() {
            let tensor = tt.tensor(site);
            let (_primal, tangent) = fw.unpack_dual(tensor)?;
            assert!(tangent.is_some(), "site {site} lost tangent information");
        }

        Ok(())
    })
    .unwrap();
}

#[test]
fn truncate_preserves_forward_payload() {
    forward_ad::dual_level(|fw| {
        let mut tt = make_two_site_tt(fw);

        tt.truncate(&TruncateOptions::svd().with_max_rank(1))
            .unwrap();

        assert_eq!(tt.tensor(0).dims()[1], 1);
        assert_eq!(tt.tensor(1).dims()[0], 1);
        for site in 0..tt.len() {
            let tensor = tt.tensor(site);
            let (_primal, tangent) = fw.unpack_dual(tensor)?;
            assert!(tangent.is_some(), "site {site} lost tangent information");
        }

        Ok(())
    })
    .unwrap();
}
