#![cfg(feature = "partitioned")]
#[path = "../examples/support/large_fourier.rs"]
mod large_fourier;

// The full 1D/2D R=10/20 grid has up to 2^40 entries, but these tests use
// interpolative construction and 35 cached sample points, never a dense DFT.
// 2D interleaved is absent: the scheduler's exact reference application needs
// input bond x operator bond (50 x 144 at R=10) and exceeds 14 GiB (#1279).
#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_gaussian_1d_r10_r20() {
    for bits in [10, 20] {
        let measured = large_fourier::run_case(bits, 1, false).unwrap();
        assert!(measured.reconstruction_bound > 0.0);
        assert!(measured.compressions > 0);
    }
}
#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_gaussian_2d_grouped_r10_r20() {
    for bits in [10, 20] {
        let measured = large_fourier::run_case(bits, 2, false).unwrap();
        assert!(measured.reconstruction_bound > 0.0);
        assert!(measured.compressions > 0);
    }
}
