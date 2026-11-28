//! Original #1107 fixtures; seed offset zero preserves its exact random stream.

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};

use super::{Net, Result};

pub(super) const BOND: usize = 10;

pub(super) fn signed(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
}

pub(super) fn term(sites: &[DynIndex], family: &str, i: usize, input_seed: u64) -> Result<Net> {
    let seed = match family {
        "smooth" => 1000,
        "random" => 5000,
        "mixed" => 9000,
        _ => return Err("family must be smooth, random or mixed".into()),
    };
    let mut state = (seed as u64)
        .checked_add(i as u64)
        .and_then(|seed| seed.checked_add(input_seed))
        .ok_or("input seed offset overflows u64")?;
    let rates: Vec<_> = match family {
        "smooth" => (0..BOND)
            .map(|r| {
                10f64.powf(3.0 * r as f64 / (BOND - 1) as f64) * (1.0 + 0.1 * signed(&mut state))
            })
            .collect(),
        "mixed" => (0..BOND)
            .map(|_| 10f64.powf(1.5 * (signed(&mut state) + 1.0)))
            .collect(),
        _ => Vec::new(),
    };
    let bonds: Vec<_> = (1..sites.len()).map(|_| DynIndex::new_dyn(BOND)).collect();
    let mut tensors = Vec::with_capacity(sites.len());
    for j in 0..sites.len() {
        let left = if j == 0 { 1 } else { BOND };
        let right = if j + 1 == sites.len() { 1 } else { BOND };
        let mut indices = Vec::new();
        if j > 0 {
            indices.push(bonds[j - 1].clone());
        }
        indices.push(sites[j].clone());
        if j + 1 < sites.len() {
            indices.push(bonds[j].clone());
        }
        let mut data = Vec::with_capacity(left * 2 * right);
        // Column-major (left, site, right), matching #1107's random stream.
        for r in 0..right {
            for s in 0..2 {
                for l in 0..left {
                    let value = if family == "random" {
                        Complex64::new(signed(&mut state), signed(&mut state))
                    } else {
                        let rate = if j == 0 { rates[r] } else { rates[l] };
                        let value = if j == 0 || j + 1 == sites.len() || l == r {
                            (-rate * 0.5f64.powi(j as i32 + 1) * s as f64).exp()
                        } else {
                            0.0
                        };
                        Complex64::new(value, 0.0)
                    };
                    data.push(value);
                }
            }
        }
        tensors.push(IdxTensor::from_dense(indices, data)?);
    }
    Ok(Net::from_tensors(tensors, (0..sites.len()).collect())?)
}
