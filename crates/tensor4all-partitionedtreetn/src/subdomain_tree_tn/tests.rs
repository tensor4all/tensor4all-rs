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

mod exact_sum;
use crate::subdomain_tree_tn::{
    sort_by_patch_order, tree_content_digest, SubDomainTreeTN, CONTENT_DIGEST_CALLS,
};

/// A one-node patch over `site`, with `values` as its payload.
fn patch(site: &DynIndex, values: Vec<f64>) -> SubDomainTreeTN<usize> {
    let tensor = IdxTensor::from_dense(vec![site.clone()], values).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    let projector = Projector::from_pairs([(site.clone(), 0)]).unwrap();
    SubDomainTreeTN::new(tree, projector).unwrap()
}

/// Two patches with equal index signature and equal projector coordinates tie on
/// metadata and are separated by content, and swapping their construction order
/// keeps that order.
#[test]
fn patch_order_orders_equal_metadata_by_content() {
    let site = DynIndex::new_dyn(2);
    let low = patch(&site, vec![1.0, 2.0]);
    let high = patch(&site, vec![3.0, 4.0]);

    assert_eq!(
        low.patch_order_key(),
        high.patch_order_key(),
        "the metadata keys must tie for this test to be about content"
    );
    assert!(low.content_digest().is_some() && high.content_digest().is_some());
    assert_ne!(
        low.content_digest(),
        high.content_digest(),
        "different payloads must not share a digest"
    );

    // Same two patches, built in the opposite order.
    let high_again = patch(&site, vec![3.0, 4.0]);
    let low_again = patch(&site, vec![1.0, 2.0]);
    assert_eq!(low.content_digest(), low_again.content_digest());
    assert_eq!(high.content_digest(), high_again.content_digest());

    // Ordering the two lists gives the same content order in both cases.
    let mut forward = vec![low, high];
    let mut backward = vec![low_again, high_again];
    sort_by_patch_order(
        &mut forward,
        |a, b| a.patch_order_key().cmp(b.patch_order_key()),
        SubDomainTreeTN::content_digest,
    );
    sort_by_patch_order(
        &mut backward,
        |a, b| a.patch_order_key().cmp(b.patch_order_key()),
        SubDomainTreeTN::content_digest,
    );
    assert_eq!(
        forward
            .iter()
            .map(SubDomainTreeTN::content_digest)
            .collect::<Vec<_>>(),
        backward
            .iter()
            .map(SubDomainTreeTN::content_digest)
            .collect::<Vec<_>>()
    );
}

/// Ordering an input whose metadata keys are distinct computes no content digest,
/// so patch construction and an untied sort stay free of the linear payload pass.
#[test]
fn sort_by_patch_order_computes_no_digest_without_metadata_ties() {
    let site = DynIndex::new_dyn(3);
    let mut patches: Vec<_> = (0..3)
        .map(|value| {
            let tensor =
                IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 2.0, 3.0]).unwrap();
            let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
            // Distinct projector coordinates give distinct metadata keys.
            let projector = Projector::from_pairs([(site.clone(), value)]).unwrap();
            SubDomainTreeTN::new(tree, projector).unwrap()
        })
        .collect();

    CONTENT_DIGEST_CALLS.with(|count| count.set(0));
    sort_by_patch_order(
        &mut patches,
        |a, b| a.patch_order_key().cmp(b.patch_order_key()),
        SubDomainTreeTN::content_digest,
    );
    assert_eq!(CONTENT_DIGEST_CALLS.with(std::cell::Cell::get), 0);
}

/// A metadata tie of two patches computes one digest per tied patch, and none for
/// a group that does not tie.
#[test]
fn sort_by_patch_order_computes_digests_for_tied_patches_only() {
    let site = DynIndex::new_dyn(2);
    let tie = |values: Vec<f64>| patch(&site, values);
    let mut patches = vec![
        tie(vec![1.0, 2.0]),
        tie(vec![3.0, 4.0]),
        // Distinct coordinates, so this one never joins the tie group.
        SubDomainTreeTN::new(
            TreeTN::from_tensors(
                vec![IdxTensor::from_dense(vec![site.clone()], vec![5.0_f64, 6.0]).unwrap()],
                vec![0usize],
            )
            .unwrap(),
            Projector::from_pairs([(site.clone(), 1)]).unwrap(),
        )
        .unwrap(),
    ];

    CONTENT_DIGEST_CALLS.with(|count| count.set(0));
    sort_by_patch_order(
        &mut patches,
        |a, b| a.patch_order_key().cmp(b.patch_order_key()),
        SubDomainTreeTN::content_digest,
    );
    assert_eq!(CONTENT_DIGEST_CALLS.with(std::cell::Cell::get), 2);
}

/// Equal payloads give equal content digests, so a content-identical tie stays a
/// tie.
#[test]
fn patch_order_is_equal_for_equal_content() {
    let site = DynIndex::new_dyn(2);
    let first = patch(&site, vec![1.0, 2.0]);
    let second = patch(&site, vec![1.0, 2.0]);
    assert_eq!(first.patch_order_key(), second.patch_order_key());
    assert_eq!(first.content_digest(), second.content_digest());
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
    assert_eq!(site_first.patch_order_key(), bond_first.patch_order_key());
    assert_ne!(
        site_first.content_digest(),
        bond_first.content_digest(),
        "the leg roles are part of the content digest"
    );
}

/// Truncation rewrites the payloads, and the digest is computed on demand, so it
/// always describes the current payloads.
#[test]
fn content_digest_follows_truncation() {
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
    let before = patch.content_digest();
    assert!(before.is_some());

    patch
        .truncate(&0usize, TruncationOptions::new().with_max_bond_dim(1))
        .unwrap();

    assert_eq!(
        patch.content_digest(),
        tree_content_digest(patch.data(), &[0usize, 1usize]),
        "the digest must describe the truncated payloads"
    );
    assert_ne!(patch.content_digest(), before);
}

/// A payload that cannot be read without materializing it leaves the digest
/// unavailable instead of densifying it inside the ordering path.
///
/// Projection rebuilds a payload contiguously, so this reaches the fold directly:
/// the only way a patch keeps such a payload is a backend-resident one, which a
/// CPU build cannot construct.
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
