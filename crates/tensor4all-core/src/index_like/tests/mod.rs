use super::*;

/// Minimal IndexLike implementation that uses the default plev() method.
/// Used to test coverage of the default trait implementations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TestIndex {
    id: u64,
    dim: usize,
    state: ConjState,
}

fn test_index(id: u64, dim: usize) -> TestIndex {
    TestIndex {
        id,
        dim,
        state: ConjState::Undirected,
    }
}

fn directed_test_index(id: u64, dim: usize, state: ConjState) -> TestIndex {
    TestIndex { id, dim, state }
}

impl IndexLike for TestIndex {
    type Id = u64;

    fn id(&self) -> &u64 {
        &self.id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn conj_state(&self) -> ConjState {
        self.state
    }

    fn conj(&self) -> Self {
        let state = match self.state {
            ConjState::Undirected => ConjState::Undirected,
            ConjState::Ket => ConjState::Bra,
            ConjState::Bra => ConjState::Ket,
        };
        TestIndex {
            state,
            ..self.clone()
        }
    }

    fn sim(&self) -> Self {
        TestIndex {
            id: self.id + 1000,
            ..self.clone()
        }
    }

    fn create_dummy_link_pair() -> (Self, Self) {
        (test_index(0, 1), test_index(0, 1))
    }

    fn new_link(dim: usize) -> anyhow::Result<Self> {
        Ok(test_index(10000, dim))
    }

    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    fn product_link(indices: &[Self]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !indices.is_empty(),
            "product_link requires at least one index"
        );
        let dim = indices.iter().try_fold(1usize, |acc, idx| {
            acc.checked_mul(idx.dim)
                .ok_or_else(|| anyhow::anyhow!("product link dimension overflow"))
        })?;
        Ok(test_index(9999, dim))
    }
}

#[test]
fn test_default_plev_is_zero() {
    let idx = test_index(1, 3);
    assert_eq!(idx.plev(), 0);
}

#[test]
fn test_default_is_contractable_with_plev() {
    let a = test_index(1, 3);
    let b = test_index(1, 3);
    // Same id, dim, and default plev=0: contractable
    assert!(a.is_contractable(&b));
}

#[test]
fn test_default_is_contractable_rejects_id_and_dim_mismatch() {
    let a = test_index(1, 3);
    let different_id = test_index(2, 3);
    let different_dim = test_index(1, 4);

    assert!(!a.is_contractable(&different_id));
    assert!(!a.is_contractable(&different_dim));
}

#[test]
fn test_default_is_contractable_supports_directed_pairs_only() {
    let ket = directed_test_index(1, 3, ConjState::Ket);
    let bra = directed_test_index(1, 3, ConjState::Bra);
    let undirected = test_index(1, 3);

    assert!(ket.is_contractable(&bra));
    assert!(bra.is_contractable(&ket));
    assert!(!ket.is_contractable(&undirected));
    assert!(!undirected.is_contractable(&bra));
}

#[test]
fn test_default_link_helpers_preserve_expected_structure() {
    let idx = directed_test_index(7, 5, ConjState::Ket);
    let conjugated = idx.conj();
    assert_eq!(conjugated.conj_state(), ConjState::Bra);
    assert_eq!(conjugated.id(), idx.id());
    assert_eq!(conjugated.dim(), idx.dim());

    let similar = idx.sim();
    assert_eq!(similar.dim(), idx.dim());
    assert_eq!(similar.conj_state(), idx.conj_state());
    assert_ne!(similar.id(), idx.id());

    let (left, right) = TestIndex::create_dummy_link_pair();
    assert_eq!(left.dim(), 1);
    assert!(left.is_contractable(&right));
}

#[test]
fn test_product_link_helper_checks_empty_and_overflow_inputs() {
    let a = test_index(1, 2);
    let b = test_index(2, 3);
    let product = TestIndex::product_link(&[a, b]).unwrap();
    assert_eq!(product.dim(), 6);
    assert_eq!(product.conj_state(), ConjState::Undirected);

    assert!(TestIndex::product_link(&[]).is_err());
    assert!(TestIndex::product_link(&[test_index(1, usize::MAX), test_index(2, 2)]).is_err());
}

#[test]
fn test_default_same_id() {
    let a = test_index(1, 3);
    let b = test_index(1, 5);
    let c = test_index(2, 3);
    assert!(a.same_id(&b));
    assert!(!a.same_id(&c));
}

#[test]
fn test_default_has_id() {
    let a = test_index(42, 3);
    assert!(a.has_id(&42));
    assert!(!a.has_id(&99));
}

#[test]
fn test_sort_indices_deterministic_pins_same_id_prime_pair() {
    use crate::defaults::DynIndex;

    let s = DynIndex::new_dyn(2);
    let s_prime = s.prime();
    assert!(s.same_id(&s_prime));
    assert_ne!(s, s_prime);

    // Primed index first (worst-case insertion order): the sort must pin
    // unprimed before primed by full-index order (plev 0 < plev 1), not
    // preserve insertion order.
    let mut indices = vec![s_prime.clone(), s.clone()];
    sort_indices_deterministic(&mut indices);
    assert_eq!(indices, vec![s.clone(), s_prime.clone()]);

    // Same canonical order regardless of input permutation.
    let mut indices = vec![s.clone(), s_prime.clone()];
    sort_indices_deterministic(&mut indices);
    assert_eq!(indices, vec![s, s_prime]);
}

#[test]
fn test_sort_indices_deterministic_distinguishes_same_id_tag_pair() {
    use crate::defaults::{DynId, DynIndex, TagSet};

    let site = DynIndex::new_with_tags(DynId(42), 2, TagSet::from_tags(&["Site"]).unwrap());
    let link = DynIndex::new_with_tags(DynId(42), 2, TagSet::from_tags(&["Link"]).unwrap());
    assert!(site.same_id(&link));
    assert_ne!(site, link);

    // The Debug tiebreak must give one deterministic order, not leave the
    // pair to insertion order.
    let mut a = vec![link.clone(), site.clone()];
    let mut b = vec![site.clone(), link.clone()];
    sort_indices_deterministic(&mut a);
    sort_indices_deterministic(&mut b);
    assert_eq!(a, b);
    assert!(a.contains(&site));
    assert!(a.contains(&link));
}
