use super::*;

#[test]
fn test_site_mpo_creation() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

    assert_eq!(site_mpo.len(), 2);
    assert_eq!(site_mpo.center(), 0);
}

#[test]
fn test_site_mpo_move_center() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2), (2, 2)], 1.0);
    let mut site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

    site_mpo.set_center(2).unwrap();
    assert_eq!(site_mpo.center(), 2);

    site_mpo.set_center(1).unwrap();
    assert_eq!(site_mpo.center(), 1);
}

#[test]
fn test_site_mpo_invalid_center() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let result = SiteMPO::from_mpo(mpo, 5);
    assert!(result.is_err());
}

#[test]
fn test_from_mpo_empty() {
    let mpo = MPO::<f64>::constant(&[], 1.0);
    let result = SiteMPO::from_mpo(mpo, 0);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), MPOError::Empty));
}

#[test]
fn test_move_center_left_from_zero() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mut site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

    let result = site_mpo.move_center_left();
    assert!(result.is_err());
}

#[test]
fn test_move_center_right_from_last() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mut site_mpo = SiteMPO::from_mpo(mpo, 1).unwrap();

    let result = site_mpo.move_center_right();
    assert!(result.is_err());
}

#[test]
fn test_set_center_out_of_bounds() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mut site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

    let result = site_mpo.set_center(5);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        MPOError::InvalidCenter { center: 5, max: 2 }
    ));
}

#[test]
fn test_into_mpo_round_trip() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (3, 3)], 5.0);
    let original_sum = mpo.sum();
    let original_dims = mpo.site_dims();

    let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();
    let recovered = site_mpo.into_mpo();

    assert_eq!(recovered.site_dims(), original_dims);
    assert!((recovered.sum() - original_sum).abs() < 1e-10);
}

#[test]
fn test_link_dims_and_rank() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (3, 3), (2, 2)], 1.0);
    let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

    let lds = site_mpo.link_dims();
    assert_eq!(lds.len(), 2);
    for &d in &lds {
        assert_eq!(d, 1);
    }
    assert_eq!(site_mpo.rank(), 1);
}

#[test]
fn test_link_dims_single_site() {
    let mpo = MPO::<f64>::constant(&[(2, 2)], 1.0);
    let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

    assert!(site_mpo.link_dims().is_empty());
    assert_eq!(site_mpo.rank(), 1);
}

#[test]
fn test_site_dims() {
    let mpo = MPO::<f64>::constant(&[(2, 3), (4, 5)], 1.0);
    let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

    assert_eq!(site_mpo.site_dims(), vec![(2, 3), (4, 5)]);
}

#[test]
fn test_site_tensor_accessors() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (3, 3)], 1.0);
    let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

    let t0 = site_mpo.site_tensor(0);
    assert_eq!(t0.site_dim_1(), 2);
    assert_eq!(t0.site_dim_2(), 2);

    let t1 = site_mpo.site_tensor(1);
    assert_eq!(t1.site_dim_1(), 3);
    assert_eq!(t1.site_dim_2(), 3);

    assert_eq!(site_mpo.site_tensors().len(), 2);
}

#[test]
fn test_site_tensor_mut_accessors() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mut site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

    let t = site_mpo.site_tensor_mut(0);
    t.set4(0, 0, 0, 0, 99.0);
    assert_eq!(*site_mpo.site_tensor(0).get4(0, 0, 0, 0), 99.0);

    let tensors = site_mpo.site_tensors_mut();
    tensors[1].set4(0, 1, 1, 0, 42.0);
    assert_eq!(*site_mpo.site_tensor(1).get4(0, 1, 1, 0), 42.0);
}

#[test]
fn test_is_empty() {
    let mpo = MPO::<f64>::constant(&[(2, 2)], 1.0);
    let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();
    assert!(!site_mpo.is_empty());
}
