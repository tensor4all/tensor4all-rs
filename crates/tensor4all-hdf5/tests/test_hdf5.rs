use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use tensor4all_core::defaults::tensordynlen::TensorAccess;
use tensor4all_core::index::{DynId, DynIndex, Index, TagSet};
use tensor4all_core::TensorDynLen;
use tensor4all_hdf5::{load_itensor, load_mps, save_itensor, save_mps};
use tensor4all_itensorlike::TensorTrain;

fn temp_path(name: &str) -> String {
    let dir = std::env::temp_dir();
    dir.join(format!("tensor4all_hdf5_test_{}.h5", name))
        .to_string_lossy()
        .to_string()
}

/// Create a simple 2x3 f64 tensor with known data.
fn make_test_tensor_f64() -> TensorDynLen {
    let i1 = Index::new_dyn_with_tags(2, TagSet::from_str("Site,n=1").unwrap());
    let i2 = Index::new_dyn_with_tags(3, TagSet::from_str("Link,l=1").unwrap());
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    TensorDynLen::from_dense_f64(vec![i1, i2], data)
}

/// Create a simple 2x3 complex tensor with known data.
fn make_test_tensor_c64() -> TensorDynLen {
    let i1 = Index::new_dyn_with_tags(2, TagSet::from_str("Site,n=1").unwrap());
    let i2 = Index::new_dyn_with_tags(3, TagSet::from_str("Link,l=1").unwrap());
    let data: Vec<Complex64> = vec![
        Complex64::new(1.0, 0.1),
        Complex64::new(2.0, 0.2),
        Complex64::new(3.0, 0.3),
        Complex64::new(4.0, 0.4),
        Complex64::new(5.0, 0.5),
        Complex64::new(6.0, 0.6),
    ];
    TensorDynLen::from_dense_c64(vec![i1, i2], data)
}

#[test]
fn test_itensor_f64_roundtrip() {
    let path = temp_path("itensor_f64");
    let tensor = make_test_tensor_f64();

    save_itensor(&path, "tensor", &tensor).unwrap();
    let loaded = load_itensor(&path, "tensor").unwrap();

    // Check dimensions
    assert_eq!(tensor.dims(), loaded.dims());

    // Check index properties
    let orig_indices = tensor.indices();
    let loaded_indices = loaded.indices();
    assert_eq!(orig_indices.len(), loaded_indices.len());
    for (orig, loaded) in orig_indices.iter().zip(loaded_indices.iter()) {
        assert_eq!(orig.id.0, loaded.id.0);
        assert_eq!(orig.dim, loaded.dim);
        assert_eq!(orig.tags, loaded.tags);
    }

    // Check data
    let orig_data = tensor.to_vec_f64().unwrap();
    let loaded_data = loaded.to_vec_f64().unwrap();
    for (a, b) in orig_data.iter().zip(loaded_data.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-15);
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_itensor_c64_roundtrip() {
    let path = temp_path("itensor_c64");
    let tensor = make_test_tensor_c64();

    save_itensor(&path, "tensor", &tensor).unwrap();
    let loaded = load_itensor(&path, "tensor").unwrap();

    // Check dimensions
    assert_eq!(tensor.dims(), loaded.dims());

    // Check data
    let orig_data = tensor.to_vec_c64().unwrap();
    let loaded_data = loaded.to_vec_c64().unwrap();
    for (a, b) in orig_data.iter().zip(loaded_data.iter()) {
        assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-15);
        assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-15);
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_itensor_3d_roundtrip() {
    let path = temp_path("itensor_3d");
    let i1 = Index::new_dyn_with_tags(2, TagSet::from_str("Link,l=0").unwrap());
    let i2 = Index::new_dyn_with_tags(3, TagSet::from_str("Site,n=1").unwrap());
    let i3 = Index::new_dyn_with_tags(4, TagSet::from_str("Link,l=1").unwrap());
    let n = 2 * 3 * 4;
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let tensor = TensorDynLen::from_dense_f64(vec![i1, i2, i3], data.clone());

    save_itensor(&path, "tensor3d", &tensor).unwrap();
    let loaded = load_itensor(&path, "tensor3d").unwrap();

    assert_eq!(tensor.dims(), loaded.dims());
    let loaded_data = loaded.to_vec_f64().unwrap();
    for (a, b) in data.iter().zip(loaded_data.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-15);
    }

    std::fs::remove_file(&path).ok();
}

/// Create a simple 3-site MPS for testing.
fn make_test_mps() -> TensorTrain {
    // Site 0: (1, d0=2, chi01=3) → indices: [link_left_dummy, site0, link01]
    // Site 1: (chi01=3, d1=2, chi12=4) → indices: [link01, site1, link12]
    // Site 2: (chi12=4, d2=2, 1) → indices: [link12, site2, link_right_dummy]

    let site_tags: Vec<TagSet> = (0..3)
        .map(|n| TagSet::from_str(&format!("Site,n={}", n)).unwrap())
        .collect();

    // Link indices (shared between adjacent sites)
    let link01_id = 100u64;
    let link12_id = 200u64;

    let link_tags_01 = TagSet::from_str("Link,l=1").unwrap();
    let link_tags_12 = TagSet::from_str("Link,l=2").unwrap();

    // Dummy bond indices for boundary (dim=1)
    let left_dummy = DynIndex::new_dyn(1);
    let right_dummy = DynIndex::new_dyn(1);

    // Site indices
    let site0 = Index::new_dyn_with_tags(2, site_tags[0].clone());
    let site1 = Index::new_dyn_with_tags(2, site_tags[1].clone());
    let site2 = Index::new_dyn_with_tags(2, site_tags[2].clone());

    // Link indices (same id for shared bond)
    let link01_left = Index::new_with_tags(DynId(link01_id), 3, link_tags_01.clone());
    let link01_right = link01_left.clone();
    let link12_left = Index::new_with_tags(DynId(link12_id), 4, link_tags_12.clone());
    let link12_right = link12_left.clone();

    // Tensor 0: shape (1, 2, 3) = 6 elements
    let data0: Vec<f64> = (0..6).map(|i| i as f64 * 0.1).collect();
    let t0 = TensorDynLen::from_dense_f64(vec![left_dummy, site0, link01_left], data0);

    // Tensor 1: shape (3, 2, 4) = 24 elements
    let data1: Vec<f64> = (0..24).map(|i| i as f64 * 0.01).collect();
    let t1 = TensorDynLen::from_dense_f64(vec![link01_right, site1, link12_left], data1);

    // Tensor 2: shape (4, 2, 1) = 8 elements
    let data2: Vec<f64> = (0..8).map(|i| i as f64 * 0.05).collect();
    let t2 = TensorDynLen::from_dense_f64(vec![link12_right, site2, right_dummy], data2);

    TensorTrain::new(vec![t0, t1, t2]).unwrap()
}

#[test]
fn test_mps_roundtrip() {
    let path = temp_path("mps");
    let mps = make_test_mps();

    save_mps(&path, "mps", &mps).unwrap();
    let loaded = load_mps(&path, "mps").unwrap();

    // Check length
    assert_eq!(mps.len(), loaded.len());

    // Check each tensor
    let orig_tensors = mps.tensors();
    let loaded_tensors = loaded.tensors();
    for (i, (orig, loaded_t)) in orig_tensors.iter().zip(loaded_tensors.iter()).enumerate() {
        assert_eq!(orig.dims(), loaded_t.dims(), "Dims mismatch at site {}", i);

        // Check index IDs are preserved
        let orig_inds = orig.indices();
        let loaded_inds = loaded_t.indices();
        for (oi, li) in orig_inds.iter().zip(loaded_inds.iter()) {
            assert_eq!(oi.id.0, li.id.0, "Index ID mismatch at site {}", i);
            assert_eq!(oi.dim, li.dim, "Index dim mismatch at site {}", i);
        }

        // Check data
        let orig_data = orig.to_vec_f64().unwrap();
        let loaded_data = loaded_t.to_vec_f64().unwrap();
        for (a, b) in orig_data.iter().zip(loaded_data.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_mps_ortho_roundtrip() {
    let path = temp_path("mps_ortho");
    let mps = make_test_mps();

    // Check that llim/rlim survive roundtrip
    let llim = mps.llim();
    let rlim = mps.rlim();

    save_mps(&path, "mps", &mps).unwrap();
    let loaded = load_mps(&path, "mps").unwrap();

    assert_eq!(loaded.llim(), llim);
    assert_eq!(loaded.rlim(), rlim);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_type_mismatch_error() {
    // Write an ITensor, then try to load it as MPS → should get a clear error
    let path = temp_path("type_mismatch");
    let tensor = make_test_tensor_f64();
    save_itensor(&path, "obj", &tensor).unwrap();

    let err = load_mps(&path, "obj").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Expected HDF5 type 'MPS'"),
        "Expected type mismatch error, got: {}",
        msg
    );

    std::fs::remove_file(&path).ok();
}
