use super::{common_ind_positions, common_ind_positions_hashed, common_ind_positions_linear};
use super::{prepare_contraction, prepare_contraction_pairs};
use crate::index::DefaultIndex as Index;
use crate::{ConjState, IndexLike};

/// Directed test index whose `conj()` toggles Ket <-> Bra AND remaps the
/// id (Ket(id) <-> Bra(id + 1000)). This is the case where id-keyed
/// contraction pairing silently drops contractable pairs (#615).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct DirectedIdIndex {
    id: u64,
    dim: usize,
    state: ConjState,
}

impl DirectedIdIndex {
    fn new(id: u64, dim: usize, state: ConjState) -> Self {
        Self { id, dim, state }
    }
}

impl IndexLike for DirectedIdIndex {
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
        let id = match (self.state, state) {
            (ConjState::Ket, ConjState::Bra) => self.id + 1000,
            (ConjState::Bra, ConjState::Ket) => self.id - 1000,
            _ => self.id,
        };
        Self {
            id,
            dim: self.dim,
            state,
        }
    }

    fn sim(&self) -> Self {
        Self {
            id: self.id + 10000,
            ..self.clone()
        }
    }

    fn create_dummy_link_pair() -> (Self, Self) {
        (
            Self::new(0, 1, ConjState::Undirected),
            Self::new(0, 1, ConjState::Undirected),
        )
    }

    fn new_link(dim: usize) -> anyhow::Result<Self> {
        Ok(Self::new(10000, dim, ConjState::Undirected))
    }

    fn product_link(indices: &[Self]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !indices.is_empty(),
            "product_link requires at least one index"
        );
        let dim = indices.iter().try_fold(1usize, |acc, idx| {
            acc.checked_mul(idx.dim)
                .ok_or_else(|| anyhow::anyhow!("product link dimension overflow"))
        })?;
        Ok(Self::new(9999, dim, ConjState::Undirected))
    }
}

#[test]
fn hashed_and_linear_agree_on_directed_conj_id_change() {
    // a: Ket ids 0,1,2; b: Bra partners 1000,1002 plus an unrelated Ket.
    // Only (0,0) and (2,2) are contractable; the id-keyed hashed path
    // would drop both because no b shares a's id.
    let a = vec![
        DirectedIdIndex::new(0, 2, ConjState::Ket),
        DirectedIdIndex::new(1, 2, ConjState::Ket),
        DirectedIdIndex::new(2, 2, ConjState::Ket),
    ];
    let b = vec![
        DirectedIdIndex::new(1000, 2, ConjState::Bra),
        DirectedIdIndex::new(5, 2, ConjState::Ket),
        DirectedIdIndex::new(1002, 2, ConjState::Bra),
    ];

    let linear = common_ind_positions_linear(&a, &b);
    assert_eq!(linear.as_slice(), &[(0, 0), (2, 2)]);
    let hashed = common_ind_positions_hashed(&a, &b);
    assert_eq!(hashed, linear);
}

#[test]
fn hashed_and_linear_agree_on_same_id_different_metadata() {
    // Same id, different prime level: not contractable; both paths must
    // reject the pair rather than pairing on id alone.
    let i = Index::new_dyn(2);
    let i_prime = i.prime();
    let j = Index::new_dyn(2);

    let a = vec![i.clone(), j.clone()];
    let b = vec![i_prime.clone(), j.clone()];
    let linear = common_ind_positions_linear(&a, &b);
    let hashed = common_ind_positions_hashed(&a, &b);
    assert_eq!(linear.as_slice(), &[(1, 1)]);
    assert_eq!(hashed, linear);
}

#[test]
fn common_ind_positions_dispatch_hash_path_matches_linear_reference() {
    // scan_work = 9*9 = 81 > LINEAR_CONTRACTION_SCAN_LIMIT (64) so the
    // public entry point dispatches to the hashed path.
    let lhs: Vec<_> = (0..9).map(|_| Index::new_dyn(2)).collect();
    let shared = lhs[7].clone();
    let mut rhs: Vec<_> = (0..9).map(|_| Index::new_dyn(2)).collect();
    rhs[5] = shared.clone();

    let dispatched = common_ind_positions(&lhs, &rhs);
    let reference = common_ind_positions_linear(&lhs, &rhs).into_vec();
    assert_eq!(dispatched, vec![(7, 5)]);
    assert_eq!(dispatched, reference);
}

#[test]
fn prepare_contraction_pairs_selects_exact_same_id_prime_index() {
    let i = Index::new_dyn(2);
    let i_prime = i.prime();
    let spec = prepare_contraction_pairs(
        &[i.clone(), i_prime.clone()],
        &[2, 2],
        std::slice::from_ref(&i_prime),
        &[2],
        &[(i_prime.clone(), i_prime.clone())],
    )
    .unwrap();

    assert_eq!(spec.axes_a.as_slice(), &[1]);
    assert_eq!(spec.axes_b.as_slice(), &[0]);
    assert_eq!(spec.result_indices.as_slice(), &[i]);
}

#[test]
fn prepare_contraction_large_rank_uses_hash_fallback_semantics() {
    let mut lhs: Vec<_> = (0..9).map(|_| Index::new_dyn(2)).collect();
    let shared = lhs[7].clone();
    let mut rhs: Vec<_> = (0..9).map(|_| Index::new_dyn(2)).collect();
    rhs[5] = shared;

    let lhs_dims = vec![2; lhs.len()];
    let rhs_dims = vec![2; rhs.len()];
    let spec = prepare_contraction(&lhs, &lhs_dims, &rhs, &rhs_dims).unwrap();

    assert_eq!(spec.axes_a.as_slice(), &[7]);
    assert_eq!(spec.axes_b.as_slice(), &[5]);
    assert_eq!(spec.result_indices.len(), lhs.len() + rhs.len() - 2);

    lhs.remove(7);
    rhs.remove(5);
    let mut expected = lhs;
    expected.extend(rhs);
    assert_eq!(spec.result_indices.as_slice(), expected.as_slice());
}
