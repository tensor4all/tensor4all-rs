//! Regression tests for tensor4all/tensor4all-rs#995 (reported by @simonb282):
//! truncating QR must drop the negligible rows of `R` wherever they sit, not keep a
//! prefix. A non-pivoted QR of a rank-deficient matrix can leave a near-zero row of
//! `R` before a large one; both fixtures below come from the report.

use tensor4all_core::{contract, qr, qr_with, DynIndex, IdxTensor, QrOptions};

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn qr_keeps_a_large_row_after_a_negligible_one() {
    // Column-major, dims [4, 2, 2, 2, 2, 2]; the left set is {index 0, index 5}.
    #[rustfmt::skip]
    let data: Vec<f64> = vec![
        -2.6457513110645907, 0.0, 0.0, 0.0,
        -2.6457513110645907, 0.0, 0.0, 0.0,
        -2.6457513110645907, 0.0, 0.0, 0.0,
        -2.6457513110645907, 0.0, 0.0, 0.0,
        -2.6457513110645907, 0.0, 0.0, 0.0,
        -2.6457513110645907, 0.0, 0.0, 0.0,
        -2.6457513110645907, 0.0, 0.0, 0.0,
        -2.6457513110645907, 0.0, 0.0, 0.0,
        -2.267786838055364, -0.9258200997725514, 0.0, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, -3.1676639673831405e-16, 0.0,
        -2.267786838055364, -0.9258200997725519, 0.954367967141531, -0.2986331918828533,
        -2.267786838055364, -0.9258200997725519, 0.954367967141531, -0.2986331918828533,
        -2.267786838055364, -0.9258200997725519, 0.954367967141531, -0.2986331918828533,
        -2.267786838055364, -0.9258200997725519, 0.954367967141531, -0.2986331918828533,
        -2.267786838055364, -0.9258200997725519, 0.954367967141531, -0.2986331918828533,
        -2.267786838055364, -0.9258200997725519, 0.954367967141531, -0.2986331918828533,
    ];
    let idx: Vec<DynIndex> = [4, 2, 2, 2, 2, 2]
        .iter()
        .map(|&d| DynIndex::new_dyn(d))
        .collect();
    let t = IdxTensor::from_dense(idx.clone(), data.clone()).unwrap();
    let left = [idx[0].clone(), idx[5].clone()];

    let (q, r) = qr::<f64>(&t, &left).unwrap();
    let back = contract(&[&q, &r]).unwrap().permute_indices(&idx).unwrap();
    let err = max_abs_diff(&back.to_vec::<f64>().unwrap(), &data);
    assert!(err < 1e-12, "Q*R differs from the input by {err:e}");

    // The retained bond is exactly the number of rows of the untruncated R that pass
    // the default threshold.
    let (_, r_full) = qr_with::<f64>(&t, &left, &QrOptions::new().with_rtol(0.0)).unwrap();
    let kb = r_full.dims()[0];
    let v = r_full.to_vec::<f64>().unwrap();
    let norms: Vec<f64> = (0..kb)
        .map(|b| {
            v.iter()
                .skip(b)
                .step_by(kb)
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let max = norms.iter().cloned().fold(0.0, f64::max);
    let passing = norms.iter().filter(|&&n| n >= 1e-15 * max).count();
    assert_eq!(q.dims()[2], passing, "row norms {norms:?}");
}

#[test]
fn qr_chain_of_a_box_function_reconstructs_every_step() {
    let r = 10;
    let f: Vec<f64> = (0..1usize << r)
        .map(|n| if (410..=615).contains(&n) { 1.0 } else { 0.0 })
        .collect();
    let s: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let order: Vec<DynIndex> = s.iter().rev().cloned().collect();
    let mut rest = IdxTensor::from_dense(order, f).unwrap();
    let mut left: Vec<DynIndex> = vec![];
    for (k, site) in s.iter().enumerate().take(r - 1) {
        let mut l = left.clone();
        l.push(site.clone());
        let idx = rest.indices().to_vec();
        let a = rest.to_vec::<f64>().unwrap();
        let (q, rr) = qr::<f64>(&rest, &l).unwrap();
        let back = contract(&[&q, &rr]).unwrap().permute_indices(&idx).unwrap();
        let err = max_abs_diff(&back.to_vec::<f64>().unwrap(), &a);
        println!("step {k}: Q dims {:?}, max abs {err:e}", q.dims());
        assert!(
            err < 1e-12,
            "step {k}: Q*R differs from the input by {err:e}"
        );
        left = vec![q.indices().last().unwrap().clone()];
        rest = rr;
    }
}
