//! Patch ordering for `SubDomainTreeTN` (#1086).
//!
//! The first three components of a patch key come from index and projector
//! metadata. They cannot separate two patches that carry the same site
//! signature and the same projector coordinates but different payloads, so the
//! fourth component is a content digest of the patch payloads. These tests pin
//! that component: it must be decided by content, not by the order in which the
//! patches were built.

use std::sync::Arc;

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_tensorbackend::StructuredStorage;
use tensor4all_treetn::{TreeTN, TruncationOptions};

use crate::projector::Projector;
use crate::subdomain_tree_tn::{tree_content_digest, SubDomainTreeTN};

/// A one-node patch over `site`, with `values` as its payload.
fn patch(site: &DynIndex, values: Vec<f64>) -> SubDomainTreeTN<usize> {
    let tensor = IdxTensor::from_dense(vec![site.clone()], values).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    let projector = Projector::from_pairs([(site.clone(), 0)]).unwrap();
    SubDomainTreeTN::new(tree, projector).unwrap()
}

/// Two patches with equal index signature and equal projector coordinates are
/// ordered by their content, and swapping their construction order keeps that
/// order.
#[test]
fn patch_order_key_orders_equal_metadata_by_content() {
    let site = DynIndex::new_dyn(2);
    let low = patch(&site, vec![1.0, 2.0]);
    let high = patch(&site, vec![3.0, 4.0]);

    let (low_key, high_key) = (low.patch_order_key(), high.patch_order_key());
    assert_eq!(
        (&low_key.0, &low_key.1, &low_key.2),
        (&high_key.0, &high_key.1, &high_key.2),
        "the metadata components must tie for this test to be about content"
    );
    assert!(low_key.3.is_some() && high_key.3.is_some());
    assert_ne!(
        low_key.3, high_key.3,
        "different payloads must not share a digest"
    );

    // Same two patches, built in the opposite order.
    let high_again = patch(&site, vec![3.0, 4.0]);
    let low_again = patch(&site, vec![1.0, 2.0]);
    assert_eq!(low_key, low_again.patch_order_key());
    assert_eq!(high_key, high_again.patch_order_key());
    assert_eq!(
        low_key.cmp(high_key),
        low_again
            .patch_order_key()
            .cmp(high_again.patch_order_key())
    );
}

/// Equal payloads give equal keys, so a content-identical tie stays a tie.
#[test]
fn patch_order_key_is_equal_for_equal_content() {
    let site = DynIndex::new_dyn(2);
    let first = patch(&site, vec![1.0, 2.0]);
    let second = patch(&site, vec![1.0, 2.0]);
    assert_eq!(first.patch_order_key(), second.patch_order_key());
}

/// A payload that cannot be read without materializing it leaves the digest
/// unavailable instead of densifying it inside the fold that builds the key.
///
/// Projection rebuilds the payload contiguously, so this reaches the `None` path
/// through the fold directly: the only way a patch keeps such a payload is a
/// backend-resident one, which a CPU build cannot construct.
#[test]
fn tree_content_digest_is_none_for_an_unreadable_payload() {
    let site = DynIndex::new_dyn(3);
    // Column-major strides for `[3]` are `[1]`; `[4]` keeps a strided payload.
    let strided = StructuredStorage::new(
        (0..9).map(|i| i as f64).collect(),
        vec![3],
        vec![4],
        vec![0],
    )
    .unwrap();
    let tensor =
        IdxTensor::from_storage(vec![site.clone()], Arc::new(strided.into_storage())).unwrap();
    assert_eq!(tensor.content_digest(), None);

    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    assert_eq!(tree_content_digest(&tree, &[0usize]), None);

    let readable = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![1.0_f64; 3]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    assert!(tree_content_digest(&readable, &[0usize]).is_some());
}

/// Two patches whose site indices and node buffers match but whose node tensors
/// carry the same buffer under a different leg order are different networks and
/// must not share a key: the digest covers each physical leg's role.
///
/// The projectors are empty so projection does not mask either payload; both
/// patches therefore hold the identical buffers `[1, 2; 3, 4]` with the same
/// payload dimensions, and only the meaning of the axes differs.
#[test]
fn patch_order_key_distinguishes_leg_roles() {
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);

    let patch = |node0_legs: [DynIndex; 2], node1_legs: [DynIndex; 2]| {
        let node0 =
            IdxTensor::from_dense(node0_legs.to_vec(), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let node1 =
            IdxTensor::from_dense(node1_legs.to_vec(), vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
        let tree = TreeTN::from_tensors(vec![node0, node1], vec![0usize, 1]).unwrap();
        SubDomainTreeTN::new(tree, Projector::default()).unwrap()
    };

    let site_first = patch(
        [site_a.clone(), bond.clone()],
        [bond.clone(), site_b.clone()],
    );
    let bond_first = patch(
        [bond.clone(), site_a.clone()],
        [site_b.clone(), bond.clone()],
    );

    for (left, right) in [(&site_first, &bond_first), (&bond_first, &site_first)] {
        assert_eq!(
            (
                left.patch_order_key().0.clone(),
                left.patch_order_key().1.clone()
            ),
            (
                right.patch_order_key().0.clone(),
                right.patch_order_key().1.clone()
            ),
        );
        assert_eq!(left.patch_order_key().2, right.patch_order_key().2);
    }
    assert_ne!(site_first.patch_order_key(), bond_first.patch_order_key());
}

/// Truncation rewrites the payloads, so the cached content digest must be
/// recomputed rather than left describing the pre-truncation patch.
#[test]
fn truncate_refreshes_the_cached_content_digest() {
    let site_left = DynIndex::new_dyn(2);
    let site_right = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let node0 = IdxTensor::from_dense(
        vec![site_left.clone(), bond.clone()],
        vec![1.0_f64, 0.5, 0.25, 2.0],
    )
    .unwrap();
    let node1 = IdxTensor::from_dense(vec![bond, site_right.clone()], vec![0.3_f64, 1.0, 2.0, 0.7])
        .unwrap();
    let tree = TreeTN::from_tensors(vec![node0, node1], vec![0usize, 1]).unwrap();
    let mut patch = SubDomainTreeTN::new(tree, Projector::default()).unwrap();
    let before = patch.patch_order_key().3;
    assert!(before.is_some());

    patch
        .truncate(&0usize, TruncationOptions::new().with_max_bond_dim(1))
        .unwrap();

    assert_eq!(
        patch.patch_order_key().3,
        tree_content_digest(patch.data(), &[0usize, 1usize]),
        "the cached digest must describe the truncated payloads"
    );
    assert_ne!(patch.patch_order_key().3, before);
}
