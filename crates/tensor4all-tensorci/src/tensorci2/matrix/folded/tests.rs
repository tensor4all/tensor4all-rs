use super::*;

/// Deterministic, non-trivially rounded values so a changed summation order
/// would show up in the low bits.
fn values(len: usize, seed: f64) -> Vec<f64> {
    (0..len)
        .map(|i| ((i as f64 + seed) * 0.7390851332151607).sin() / 3.0)
        .collect()
}

fn bits(data: &[f64]) -> Vec<u64> {
    data.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn gemm_slab_is_bit_identical_to_the_product_with_a_copied_slab() {
    for (left, site, right, m) in [(1, 1, 1, 1), (3, 2, 4, 5), (4, 3, 1, 2), (2, 5, 6, 7)] {
        let core = FoldedCore {
            left,
            site,
            right,
            data: values(left * site * right, 1.0),
        };
        let a = values(m * left, 2.0);
        for x in 0..site {
            let strided = core.gemm_slab(&a, m, left, x);
            let copied = col_major_gemm(&a, m, left, &core.slab(x), right);
            assert_eq!(
                bits(&strided),
                bits(&copied),
                "core {left}x{site}x{right}, x = {x}"
            );
        }
    }
}
