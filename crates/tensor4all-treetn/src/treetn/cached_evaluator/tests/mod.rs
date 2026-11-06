//! Unit tests for the batched cached evaluator, split by concern.
//!
//! Shared scalar-kind plumbing and fixtures used by more than one concern live
//! here; each submodule holds the tests and helpers for a single concern.

use super::*;
use num_complex::{Complex32, Complex64};
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, TensorElement};

mod chunking;
mod components;
mod edge_cut;
mod evaluation;
mod grouped_contraction;
mod measurements;
mod message_cache;
mod raw_messages;
mod scalar_chain_kernel;
mod scalar_kinds;

trait CachedEvaluatorTestScalar: TensorElement + PartialEq + Debug {
    const DEBUG_DTYPE: &'static str;
    const TOLERANCE: f64;
    const IS_COMPLEX: bool;

    fn from_parts(real: f64, imag: f64) -> Self;

    /// Lifts a typed value back into the dynamic wrapper so a typed batch
    /// can be compared against the `AnyScalar` route without inventing a
    /// second conversion policy.
    fn into_any_scalar(self) -> AnyScalar;
}

impl CachedEvaluatorTestScalar for f32 {
    const DEBUG_DTYPE: &'static str = "f32";
    const TOLERANCE: f64 = 1.0e-5;
    const IS_COMPLEX: bool = false;

    fn from_parts(real: f64, _imag: f64) -> Self {
        real as f32
    }

    fn into_any_scalar(self) -> AnyScalar {
        AnyScalar::from_value(self)
    }
}

impl CachedEvaluatorTestScalar for f64 {
    const DEBUG_DTYPE: &'static str = "f64";
    const TOLERANCE: f64 = 1.0e-12;
    const IS_COMPLEX: bool = false;

    fn from_parts(real: f64, _imag: f64) -> Self {
        real
    }

    fn into_any_scalar(self) -> AnyScalar {
        AnyScalar::from_value(self)
    }
}

impl CachedEvaluatorTestScalar for Complex32 {
    const DEBUG_DTYPE: &'static str = "c32";
    const TOLERANCE: f64 = 1.0e-5;
    const IS_COMPLEX: bool = true;

    fn from_parts(real: f64, imag: f64) -> Self {
        Self::new(real as f32, imag as f32)
    }

    fn into_any_scalar(self) -> AnyScalar {
        AnyScalar::from_value(self)
    }
}

impl CachedEvaluatorTestScalar for Complex64 {
    const DEBUG_DTYPE: &'static str = "c64";
    const TOLERANCE: f64 = 1.0e-12;
    const IS_COMPLEX: bool = true;

    fn from_parts(real: f64, imag: f64) -> Self {
        Self::new(real, imag)
    }

    fn into_any_scalar(self) -> AnyScalar {
        AnyScalar::from_value(self)
    }
}

fn typed_three_node_chain<T: CachedEvaluatorTestScalar>(
) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let s0 = DynIndex::new_dyn(3);
    let b01 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(3);
    let s2 = DynIndex::new_dyn(3);

    let t0 = IdxTensor::from_dense(
        vec![s0.clone(), b01.clone()],
        vec![
            T::from_parts(1.0, 0.25),
            T::from_parts(-0.5, -0.75),
            T::from_parts(2.0, -1.0),
            T::from_parts(0.75, 0.5),
            T::from_parts(-1.25, 0.75),
            T::from_parts(0.125, -0.25),
        ],
    )
    .unwrap();
    let t1 = IdxTensor::from_dense(
        vec![b01, s1.clone(), b12.clone()],
        (0..12)
            .map(|value| T::from_parts(value as f64 * 0.5 - 1.0, (value as f64 + 1.0) * 0.125))
            .collect(),
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(
        vec![b12, s2.clone()],
        vec![
            T::from_parts(0.25, -0.5),
            T::from_parts(1.5, 0.75),
            T::from_parts(-0.75, 1.25),
            T::from_parts(2.0, -1.5),
            T::from_parts(0.5, 0.25),
            T::from_parts(-1.25, -0.75),
            T::from_parts(1.25, 0.5),
            T::from_parts(-0.25, 1.0),
            T::from_parts(0.875, -0.625),
        ],
    )
    .unwrap();

    let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();
    (tree, vec![s0, s1, s2])
}

fn typed_unequal_y_tree<T: CachedEvaluatorTestScalar>() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>)
{
    let sc = DynIndex::new_dyn(2);
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let b0 = DynIndex::new_dyn(2);
    let b1 = DynIndex::new_dyn(3);
    let b2 = DynIndex::new_dyn(4);

    let center = IdxTensor::from_dense(
        vec![sc.clone(), b0.clone(), b1.clone(), b2.clone()],
        (0..48)
            .map(|value| T::from_parts(value as f64 * 0.125 - 1.0, (value as f64 + 1.0) * 0.0625))
            .collect(),
    )
    .unwrap();
    let leaf0 = IdxTensor::from_dense(
        vec![b0, s0.clone()],
        vec![
            T::from_parts(1.0, 0.25),
            T::from_parts(-0.5, -0.5),
            T::from_parts(1.5, 0.75),
            T::from_parts(0.25, -1.0),
        ],
    )
    .unwrap();
    let leaf1 = IdxTensor::from_dense(
        vec![b1, s1.clone()],
        (0..6)
            .map(|value| T::from_parts(value as f64 * 0.25 + 0.5, -0.125))
            .collect(),
    )
    .unwrap();
    let leaf2 = IdxTensor::from_dense(
        vec![b2, s2.clone()],
        (0..8)
            .map(|value| T::from_parts(1.0 - value as f64 * 0.125, 0.5))
            .collect(),
    )
    .unwrap();

    let tree =
        TreeTN::<_, usize>::from_tensors(vec![center, leaf0, leaf1, leaf2], vec![0, 1, 2, 3])
            .unwrap();
    (tree, vec![sc, s0, s1, s2])
}

fn assert_typed_results<T: CachedEvaluatorTestScalar>(
    actual: &[AnyScalar],
    expected: &[AnyScalar],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        let scale = actual
            .real()
            .abs()
            .max(actual.imag().abs())
            .max(expected.real().abs())
            .max(expected.imag().abs())
            .max(1.0);
        assert!(
            (actual.real() - expected.real()).abs() <= T::TOLERANCE * scale
                && (actual.imag() - expected.imag()).abs() <= T::TOLERANCE * scale,
            "actual={actual:?} expected={expected:?}"
        );
        let debug = format!("{actual:?}");
        assert!(
            debug.contains(&format!("dtype: \"{}\"", T::DEBUG_DTYPE)),
            "cached result lost dtype {}: {actual:?}",
            T::DEBUG_DTYPE
        );
    }
}

fn varied_three_node_chain() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let s0 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);

    let t0 =
        IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let t1 = IdxTensor::from_dense(
        vec![b01, s1.clone(), b12.clone()],
        (0..8).map(|i| i as f64 + 1.0).collect(),
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(vec![b12, s2.clone()], vec![0.5_f64, 1.5, 2.5, 3.5]).unwrap();
    let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();
    (tree, vec![s0, s1, s2])
}

fn assert_scalars_close(actual: &[AnyScalar], expected: &[AnyScalar]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual.real() - expected.real()).abs() < 1.0e-12,
            "actual={actual:?} expected={expected:?}"
        );
        assert!(
            (actual.imag() - expected.imag()).abs() < 1.0e-12,
            "actual={actual:?} expected={expected:?}"
        );
    }
}

fn complex_three_node_chain() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let s0 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(
        vec![s0.clone(), b01.clone()],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-0.25, 1.5),
            Complex64::new(2.0, -0.75),
            Complex64::new(0.5, -1.0),
        ],
    )
    .unwrap();
    let t1 = IdxTensor::from_dense(
        vec![b01, s1.clone(), b12.clone()],
        (0..8)
            .map(|value| Complex64::new(value as f64 + 0.5, -(value as f64) * 0.25))
            .collect(),
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(
        vec![b12, s2.clone()],
        vec![
            Complex64::new(0.75, -0.5),
            Complex64::new(1.25, 0.25),
            Complex64::new(-1.0, 0.75),
            Complex64::new(2.5, -1.25),
        ],
    )
    .unwrap();
    let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();
    (tree, vec![s0, s1, s2])
}

fn five_node_chain() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let sites: Vec<DynIndex> = (0..5).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<DynIndex> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();

    let t0 =
        IdxTensor::from_dense(vec![sites[0].clone(), bonds[0].clone()], vec![1.0_f64; 4]).unwrap();
    let t1 = IdxTensor::from_dense(
        vec![bonds[0].clone(), sites[1].clone(), bonds[1].clone()],
        vec![1.0_f64; 8],
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(
        vec![bonds[1].clone(), sites[2].clone(), bonds[2].clone()],
        vec![1.0_f64; 8],
    )
    .unwrap();
    let t3 = IdxTensor::from_dense(
        vec![bonds[2].clone(), sites[3].clone(), bonds[3].clone()],
        vec![1.0_f64; 8],
    )
    .unwrap();
    let t4 =
        IdxTensor::from_dense(vec![bonds[3].clone(), sites[4].clone()], vec![1.0_f64; 4]).unwrap();

    let tree =
        TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2, t3, t4], vec![0, 1, 2, 3, 4]).unwrap();
    (tree, sites)
}

fn star_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let sc = DynIndex::new_dyn(2);
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let b0 = DynIndex::new_dyn(2);
    let b1 = DynIndex::new_dyn(2);
    let b2 = DynIndex::new_dyn(2);
    let center_data: Vec<f64> = (0..16).map(|value| value as f64 + 1.0).collect();
    let center = IdxTensor::from_dense(
        vec![sc.clone(), b0.clone(), b1.clone(), b2.clone()],
        center_data,
    )
    .unwrap();
    let leaf0 = IdxTensor::from_dense(vec![b0, s0.clone()], vec![1.0_f64, 0.5, 1.5, 2.0]).unwrap();
    let leaf1 =
        IdxTensor::from_dense(vec![b1, s1.clone()], vec![0.25_f64, 1.0, 1.25, 2.0]).unwrap();
    let leaf2 = IdxTensor::from_dense(vec![b2, s2.clone()], vec![2.0_f64, 1.0, 0.75, 1.5]).unwrap();
    let tree =
        TreeTN::<_, usize>::from_tensors(vec![center, leaf0, leaf1, leaf2], vec![0, 1, 2, 3])
            .unwrap();
    (tree, vec![sc, s0, s1, s2])
}

/// A hub of `arms` unequal bonds whose physical index sits *between* two
/// of them, with unequal leaf dimensions.
///
/// Rooted at leaf 1 the hub has `arms - 1` children, so `arms >= 4`
/// reaches the arbitrary-degree message kernel. Nothing about the fixture
/// is canonical: axis order, bond dimensions, and site dimensions all
/// differ, so a kernel that assumes any of them cannot pass by accident.
fn unequal_star_tree<T: CachedEvaluatorTestScalar>(
    arms: usize,
) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    assert!(arms >= 2, "a star needs at least two arms");
    let value = |offset: usize| {
        T::from_parts(
            (offset % 7) as f64 * 0.25 - 0.75,
            ((offset % 5) as f64) * 0.125 - 0.25,
        )
    };
    let hub_site = DynIndex::new_dyn(2);
    let sites = (0..arms)
        .map(|arm| DynIndex::new_dyn(2 + arm % 2))
        .collect::<Vec<_>>();
    // Bonds are large enough that a handful of points already crosses
    // the grouped kernel's work threshold, so both routes are reachable
    // through the evaluator rather than only through the primitives.
    let bonds = (0..arms)
        .map(|arm| DynIndex::new_dyn(4 + arm % 3))
        .collect::<Vec<_>>();

    let mut hub_indices = vec![bonds[0].clone(), hub_site.clone()];
    hub_indices.extend(bonds[1..].iter().cloned());
    let hub_len = hub_indices
        .iter()
        .map(|index| index.dim())
        .product::<usize>();
    let hub =
        IdxTensor::from_dense(hub_indices, (0..hub_len).map(value).collect::<Vec<T>>()).unwrap();

    let mut tensors = vec![hub];
    for arm in 0..arms {
        let leaf_len = bonds[arm].dim() * sites[arm].dim();
        tensors.push(
            IdxTensor::from_dense(
                vec![bonds[arm].clone(), sites[arm].clone()],
                (0..leaf_len)
                    .map(|offset| value(offset + arm + 3))
                    .collect::<Vec<T>>(),
            )
            .unwrap(),
        );
    }
    let tree = TreeTN::<_, usize>::from_tensors(tensors, (0..=arms).collect()).unwrap();
    let mut indices = vec![hub_site];
    indices.extend(sites);
    (tree, indices)
}

/// Distinct points, enumerated as an odometer over the site dimensions so
/// that `point_count` distinct assignments really do reach the kernel
/// (a cycling pattern repeats after `lcm(dims)` points and would silently
/// keep every batch under the grouped kernel's work threshold).
fn star_points(indices: &[DynIndex], point_count: usize) -> Vec<usize> {
    let capacity: usize = indices.iter().map(|index| index.dim()).product();
    assert!(
        point_count <= capacity,
        "the fixture has only {capacity} distinct points"
    );
    let mut values = Vec::with_capacity(indices.len() * point_count);
    for point in 0..point_count {
        let mut remaining = point;
        for index in indices {
            values.push(remaining % index.dim());
            remaining /= index.dim();
        }
    }
    values
}

fn four_arm_star_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let sc = DynIndex::new_dyn(2);
    let sites = (0..4).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let bonds = (0..4).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let center = IdxTensor::from_dense(
        vec![
            sc.clone(),
            bonds[0].clone(),
            bonds[1].clone(),
            bonds[2].clone(),
            bonds[3].clone(),
        ],
        vec![1.0_f64; 32],
    )
    .unwrap();
    let leaves = bonds
        .into_iter()
        .zip(sites.iter().cloned())
        .map(|(bond, site)| IdxTensor::from_dense(vec![bond, site], vec![1.0_f64; 4]))
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    let mut tensors = vec![center];
    tensors.extend(leaves);
    let tree = TreeTN::<_, usize>::from_tensors(tensors, vec![0, 1, 2, 3, 4]).unwrap();
    let mut indices = vec![sc];
    indices.extend(sites);
    (tree, indices)
}
