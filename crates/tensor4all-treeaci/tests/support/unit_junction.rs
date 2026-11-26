//! Exact three-branch input whose junction has local dimension 1 (#1352, #1435).
//!
//! Three branches (x, y, z) of [`R`] binary sites join at a centre node whose
//! only site has dimension 1, so the centre has degree 3 and every other node
//! degree at most 2. Site `3 * level + branch` carries bit `R - 1 - level` of
//! its branch's coordinate, and each branch bond carries the integer prefix
//! read so far, so the input is exact with bonds up to `2^R = 32`.
//!
//! The target is `f(x, y, z) = M[x, z]` if `y == N / 2`, else 0, where `M` is
//! the frozen 32 x 32 table of #1352's reopened report
//! (`fixtures/issue1352_reopened_bits.txt`, column-major, one IEEE 754
//! binary64 bit pattern per word; the same file as TreeTCI's
//! `tests/fixtures/issue1352_reopened_bits.txt`). The true ranks of the bonds
//! into the centre are 12 (x), 1 (y) and 12 (z).

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

/// Binary sites per branch.
pub(crate) const R: usize = 5;
/// Grid points per branch coordinate.
pub(crate) const N: usize = 1 << R;
/// Name of the centre node, after the `3 * R` branch sites.
pub(crate) const CENTRE: usize = 3 * R;

type FixtureResult<T> = Result<T, Box<dyn std::error::Error>>;

/// The frozen table `M`, column-major.
fn table() -> FixtureResult<Vec<f64>> {
    let matrix = include_str!("../fixtures/issue1352_reopened_bits.txt")
        .split_whitespace()
        .map(|word| u64::from_str_radix(word, 16).map(f64::from_bits))
        .collect::<Result<Vec<_>, _>>()?;
    if matrix.len() != N * N {
        return Err(format!("fixture has {} entries, expected {}", matrix.len(), N * N).into());
    }
    Ok(matrix)
}

/// The exact input tree; node names are `0..=CENTRE`.
pub(crate) fn input() -> FixtureResult<TreeTN<IdxTensor, usize>> {
    input_with_junction::<f64>(false)
}

/// Exact fixture with a unit centre (false) or a binary delta junction (true).
pub(crate) fn input_with_junction<T: tensor4all_treeaci::TreeAciScalar + From<f64>>(
    delta_junction: bool,
) -> FixtureResult<TreeTN<IdxTensor, usize>> {
    let matrix = table()?;
    let sites: Vec<DynIndex> = (0..CENTRE).map(|_| DynIndex::new_dyn(2)).collect();
    // bonds[branch][level]: from `level` to `level + 1` (the centre after R - 1).
    let bonds: Vec<Vec<DynIndex>> = (0..3)
        .map(|_| {
            (0..R)
                .map(|level| DynIndex::new_dyn(1 << (level + 1)))
                .collect()
        })
        .collect();
    let mut tensors = Vec::new();
    let mut names = Vec::new();
    for level in 0..R {
        for (branch, branch_bonds) in bonds.iter().enumerate() {
            if delta_junction && level == R - 1 && branch == 1 {
                // T[x, y_prefix, s, z] = M[x, z] on 2*y_prefix + s = N/2.
                let half = N / 2;
                let mut data = vec![T::from(0.0); N * half * 2 * N];
                for x in 0..N {
                    for z in 0..N {
                        data[x + N * (N / 4 + half * 2 * z)] = T::from(matrix[x + N * z]);
                    }
                }
                tensors.push(IdxTensor::from_dense(
                    vec![
                        bonds[0][R - 1].clone(),
                        bonds[1][R - 2].clone(),
                        sites[3 * level + branch].clone(),
                        bonds[2][R - 1].clone(),
                    ],
                    data,
                )?);
                names.push(3 * level + branch);
                continue;
            }
            let in_dim = 1usize << level;
            let out_dim = 1usize << (level + 1);
            // T[in, s, out] = [out == 2 in + s], column-major.
            let mut data = vec![T::from(0.0); in_dim * 2 * out_dim];
            for i in 0..in_dim {
                for s in 0..2 {
                    data[i + in_dim * (s + 2 * (2 * i + s))] = T::from(1.0);
                }
            }
            let site = sites[3 * level + branch].clone();
            let out = branch_bonds[level].clone();
            let indices = if level == 0 {
                vec![site, out]
            } else {
                vec![branch_bonds[level - 1].clone(), site, out]
            };
            tensors.push(IdxTensor::from_dense(indices, data)?);
            names.push(3 * level + branch);
        }
    }
    if delta_junction {
        return Ok(TreeTN::from_tensors(tensors, names)?);
    }
    // C[x, y, z, 0] = M[x + N z] if y == N / 2, column-major.
    let mut centre = vec![T::from(0.0); N * N * N];
    for x in 0..N {
        for z in 0..N {
            centre[x + N * (N / 2) + N * N * z] = T::from(matrix[x + N * z]);
        }
    }
    tensors.push(IdxTensor::from_dense(
        vec![
            bonds[0][R - 1].clone(),
            bonds[1][R - 1].clone(),
            bonds[2][R - 1].clone(),
            DynIndex::new_dyn(1),
        ],
        centre,
    )?);
    names.push(CENTRE);
    Ok(TreeTN::from_tensors(tensors, names)?)
}

/// Bond dimensions of `tree` into the centre from the x, y and z branches.
pub(crate) fn centre_ranks(tree: &TreeTN<IdxTensor, usize>) -> FixtureResult<[usize; 3]> {
    junction_ranks(tree, CENTRE)
}

/// Bond dimensions into either fixture's junction from x, y and z.
pub(crate) fn junction_ranks(
    tree: &TreeTN<IdxTensor, usize>,
    junction: usize,
) -> FixtureResult<[usize; 3]> {
    let mut ranks = [0; 3];
    for (branch, rank) in ranks.iter_mut().enumerate() {
        let from = if junction != CENTRE && branch == 1 {
            3 * (R - 2) + 1
        } else {
            3 * (R - 1) + branch
        };
        let edge = tree
            .edge_between(&from, &junction)
            .ok_or("missing centre edge")?;
        *rank = tree.bond_index(edge).ok_or("missing centre bond")?.dim();
    }
    Ok(ranks)
}

/// `max |result - input| / max |input|` over the full grid.
pub(crate) fn dense_relative_error(
    result: &TreeTN<IdxTensor, usize>,
    input: &TreeTN<IdxTensor, usize>,
) -> FixtureResult<f64> {
    let reference = input.to_dense()?;
    Ok(result.to_dense()?.sub(&reference)?.maxabs()? / reference.maxabs()?)
}
