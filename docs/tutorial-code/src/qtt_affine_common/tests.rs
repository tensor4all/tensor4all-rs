use super::*;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn scratch_csv(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is after the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{name}-{nonce}.csv"))
}

#[test]
fn transformed_reference_applies_boundary_modes() {
    let bits = 3;
    let n = point_count(bits);

    let periodic = transformed_reference(n - 1, 2, bits, AffineBoundaryMode::Periodic);
    assert!((periodic - source_function(1, 2, n)).abs() < 1e-12);

    let open_outside = transformed_reference(n - 1, 2, bits, AffineBoundaryMode::Open);
    assert_eq!(open_outside, 0.0);

    let open_inside = transformed_reference(1, 2, bits, AffineBoundaryMode::Open);
    assert!((open_inside - source_function(3, 2, n)).abs() < 1e-12);
}

#[test]
fn bond_dim_rows_pad_missing_profiles() {
    assert_eq!(
        collect_bond_dims(&[2, 3], &[5], &[6], &[7, 11, 13]),
        vec![
            (1, Some(2), Some(5), Some(6), Some(7)),
            (2, Some(3), None, None, Some(11)),
            (3, None, None, None, Some(13)),
        ]
    );

    assert_eq!(
        collect_operator_bond_dims(&[2], &[4], &[3, 5]),
        vec![(1, Some(2), Some(4), Some(3)), (2, None, None, Some(5))]
    );
}

#[test]
fn csv_writers_keep_headers_and_empty_optional_cells() -> Result<(), Box<dyn Error>> {
    let samples_path = scratch_csv("affine-samples");
    write_samples_csv(
        &samples_path,
        &[AffineSamplePoint {
            x_index: 1,
            y_index: 2,
            x: 0,
            y: 1,
            source_u_periodic: 1,
            source_v: 1,
            source_exact: 0.5,
            periodic_exact: 0.25,
            periodic_qtt: 0.125,
            periodic_abs_error: 0.125,
            antiperiodic_exact: -0.25,
            antiperiodic_qtt: -0.125,
            antiperiodic_abs_error: 0.125,
            open_exact: 0.0,
            open_qtt: 0.0,
            open_abs_error: 0.0,
        }],
    )?;
    let sample_csv = fs::read_to_string(&samples_path)?;
    assert!(sample_csv.starts_with(
        "x_index,y_index,x,y,source_u_periodic,source_v,source_exact,periodic_exact,periodic_qtt,periodic_abs_error,antiperiodic_exact,antiperiodic_qtt,antiperiodic_abs_error,open_exact,open_qtt,open_abs_error\n"
    ));
    assert!(sample_csv.contains(",0.5000000000000000,0.2500000000000000,"));

    let bond_dims_path = scratch_csv("affine-bonds");
    write_bond_dims_csv(&bond_dims_path, &[(1, Some(2), None, Some(3), Some(4))])?;
    let bond_csv = fs::read_to_string(&bond_dims_path)?;
    assert_eq!(
        bond_csv,
        "bond_index,input_bond_dim,periodic_transformed_bond_dim,antiperiodic_transformed_bond_dim,open_transformed_bond_dim\n1,2,,3,4\n"
    );

    let operator_dims_path = scratch_csv("affine-operator-bonds");
    write_operator_bond_dims_csv(&operator_dims_path, &[(1, None, Some(6), Some(8))])?;
    let operator_csv = fs::read_to_string(&operator_dims_path)?;
    assert_eq!(
        operator_csv,
        "bond_index,periodic_operator_bond_dim,antiperiodic_operator_bond_dim,open_operator_bond_dim\n1,,6,8\n"
    );

    Ok(())
}
