//! Case B of #1435: binary junction whose pivots agree locally must unlock.

use num_complex::Complex64;
use tensor4all_treeaci::{tree_elementwise, TreeAciOptions, TreeAciScalar, TreeAciTermination};

use crate::unit_junction;

fn check<T: TreeAciScalar + From<f64>>() {
    let input = unit_junction::input_with_junction::<T>(true).unwrap();
    let reference = input.to_dense().unwrap();
    let scale = reference.maxabs().unwrap();
    for rng_seed in [1, 5, 9] {
        let options = TreeAciOptions {
            tolerance: 1e-10,
            rng_seed,
            ..TreeAciOptions::default()
        };
        let result =
            tree_elementwise::<T, _, _>(|values| values[0], std::slice::from_ref(&input), &options)
                .unwrap();
        let error = result
            .tree
            .to_dense()
            .unwrap()
            .sub(&reference)
            .unwrap()
            .maxabs()
            .unwrap()
            / scale;
        assert_eq!(result.termination, TreeAciTermination::Converged);
        assert_eq!(
            unit_junction::junction_ranks(&result.tree, 3 * (unit_junction::R - 1) + 1).unwrap(),
            [12, 1, 12]
        );
        assert!(
            error <= 1e-12,
            "seed {rng_seed}: dense relative error {error:e}"
        );
    }
}

#[test]
fn delta_junction_reaches_true_ranks_f64() {
    check::<f64>();
}

#[test]
fn delta_junction_reaches_true_ranks_c64() {
    check::<Complex64>();
}
