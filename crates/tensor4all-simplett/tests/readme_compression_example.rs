use std::fs;
use std::path::Path;

use tensor4all_simplett::{AbstractTensorTrain, CompressionOptions, TensorTrain};

fn assert_readme_uses_current_compression_example(path: &Path) {
    let readme = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("failed to read {}: {err}", path.display());
    });

    assert!(
        readme.contains("CompressionOptions"),
        "{} should mention CompressionOptions in the SimpleTT example",
        path.display()
    );
    assert!(
        readme.contains("tt.compressed(&options)?"),
        "{} should call tt.compressed(&options)? in the SimpleTT example",
        path.display()
    );
    assert!(
        !readme.contains("tt.compressed(1e-10, Some(20))?"),
        "{} still contains the removed two-argument compressed API",
        path.display()
    );
}

#[test]
fn simplett_readmes_use_current_compression_api() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert_readme_uses_current_compression_example(&crate_root.join("README.md"));
    assert_readme_uses_current_compression_example(&crate_root.join("../../README.md"));
}

#[test]
fn simplett_readme_compression_flow_runs_against_public_api() {
    let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
    let value = tt.evaluate(&[0, 1, 2]).unwrap();
    let total = tt.sum();
    let options = CompressionOptions {
        tolerance: 1e-10,
        max_bond_dim: 20,
        ..Default::default()
    };
    let compressed = tt.compressed(&options).unwrap();

    assert_eq!(value, 1.0);
    assert_eq!(total, 24.0);
    assert_eq!(compressed.len(), 3);
}
