use std::fs;
use std::path::Path;

use rand::rng;
use tensor4all_core::index::{DynId, Index};
use tensor4all_core::{contract_connected, contract_multi, AllowedPairs, TensorDynLen};

fn assert_readme_uses_current_contraction_example(path: &Path) {
    let readme = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("failed to read {}: {err}", path.display());
    });

    assert!(
        readme.contains("contract_multi(&[&a, &b, &c], AllowedPairs::All)?"),
        "{} should pass tensor references to contract_multi in the all-pairs example",
        path.display()
    );
    assert!(
        readme.contains("contract_multi(&[&a, &b, &c], AllowedPairs::Specified(&pairs))?"),
        "{} should pass tensor references to contract_multi in the specified-pairs example",
        path.display()
    );
    assert!(
        readme.contains("contract_connected(&[&a, &b, &c], AllowedPairs::All)?"),
        "{} should pass tensor references to contract_connected",
        path.display()
    );
    assert!(
        !readme.contains("contract_multi(&[a.clone(), b.clone(), c.clone()], AllowedPairs::All)?"),
        "{} still contains the stale owned-tensor contract_multi example",
        path.display()
    );
    assert!(
        !readme.contains(
            "contract_multi(&[a.clone(), b.clone(), c.clone()], AllowedPairs::Specified(&pairs))?"
        ),
        "{} still contains the stale owned-tensor specified-pairs example",
        path.display()
    );
    assert!(
        !readme.contains("contract_connected(&[a, b, c], AllowedPairs::All)?"),
        "{} still contains the stale owned-tensor contract_connected example",
        path.display()
    );
}

#[test]
fn core_readme_uses_current_contraction_api() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert_readme_uses_current_contraction_example(&crate_root.join("README.md"));
}

#[test]
fn core_readme_contraction_flow_runs_against_public_api() {
    let i = Index::<DynId>::new_dyn_with_tag(2, "i").unwrap();
    let j = Index::<DynId>::new_dyn_with_tag(3, "j").unwrap();
    let k = Index::<DynId>::new_dyn_with_tag(4, "k").unwrap();
    let l = Index::<DynId>::new_dyn_with_tag(5, "l").unwrap();

    let mut rng = rng();
    let a = TensorDynLen::random::<f64, _>(&mut rng, vec![i.clone(), j.clone()]);
    let b = TensorDynLen::random::<f64, _>(&mut rng, vec![j.clone(), k.clone()]);
    let c = TensorDynLen::random::<f64, _>(&mut rng, vec![k.clone(), l.clone()]);

    let result_all = contract_multi(&[&a, &b, &c], AllowedPairs::All).unwrap();
    let pairs = [(0, 1), (1, 2)];
    let result_specified = contract_multi(&[&a, &b, &c], AllowedPairs::Specified(&pairs)).unwrap();
    let result_connected = contract_connected(&[&a, &b, &c], AllowedPairs::All).unwrap();

    assert_eq!(result_all.indices(), &[i.clone(), l.clone()]);
    assert_eq!(result_specified.indices(), &[i.clone(), l.clone()]);
    assert_eq!(result_connected.indices(), &[i, l]);
}
