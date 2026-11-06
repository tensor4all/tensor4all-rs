//! Paired full_tensor diagnostic: build this same source against baseline and candidate.
//! RAYON_NUM_THREADS fixes the provider budget; these scalar readout loops are sequential.
//! Each process warms up once, then prints five individual call times (allocation included,
//! construction, validation and destruction excluded). Run the two binaries alternately.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Instant;
use tensor4all_simplett::mpo::{tensor4_from_data, MPO};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};
use tensor4all_tensorbackend::default_cpu_execution_context;

fn measure(
    name: &str,
    samples: usize,
    mut export: impl FnMut() -> (Vec<f64>, Vec<usize>),
    expected: &[(usize, f64)],
) {
    for rep in 0..=samples {
        let start = Instant::now();
        let output = black_box(export());
        let elapsed = start.elapsed();
        if rep == 0 {
            for &(index, reference) in expected {
                let value = output.0[index];
                assert!(
                    (value - reference).abs() <= 1e-10 * reference.abs().max(1.0),
                    "{name}[{index}]: {value} != {reference}"
                );
            }
        } else {
            println!(
                "{name},{rep},{},{:.9}",
                output.0.len(),
                elapsed.as_secs_f64()
            );
        }
    }
}

fn main() {
    let threads: usize = std::env::var("RAYON_NUM_THREADS")
        .expect("set an explicit thread budget")
        .parse()
        .unwrap();
    let effective = default_cpu_execution_context().with_backend(|backend| backend.num_threads());
    assert_eq!(effective, threads);
    println!("# provider=default CPU (faer on Linux), configured_workers={effective}, readout=sequential");
    println!("case,rep,elements,seconds");
    let samples = if std::env::args().any(|arg| arg == "--test") {
        1
    } else {
        5
    };
    let constant = SimpleTensorTrain::<f64>::constant(&[2; 20], 1.0);
    measure(
        "tt_constant_20_bond1",
        samples,
        || constant.full_tensor().unwrap(),
        &[(0, 1.0), ((1 << 20) - 1, 1.0)],
    );

    let mut rng = ChaCha8Rng::seed_from_u64(1060);
    let tt = SimpleTensorTrain::new(
        (0..16)
            .map(|site| {
                let left = if site == 0 { 1 } else { 8 };
                let right = if site == 15 { 1 } else { 8 };
                let data = (0..left * 2 * right)
                    .map(|_| rng.random_range(-0.5..0.5))
                    .collect();
                tensor3_from_data(data, left, 2, right).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let expected: Vec<_> = [0, 1, 0x5555, 0xffff]
        .into_iter()
        .map(|flat| {
            let indices: Vec<_> = (0..16).map(|site| (flat >> site) & 1).collect();
            (flat, tt.evaluate(&indices).unwrap())
        })
        .collect();
    measure(
        "tt_random_16_bond8",
        samples,
        || tt.full_tensor().unwrap(),
        &expected,
    );

    let mpo = MPO::new(
        (0..6)
            .map(|site| {
                let left = if site == 0 { 1 } else { 8 };
                let right = if site == 5 { 1 } else { 8 };
                let data = (0..left * 4 * right)
                    .map(|_| rng.random_range(-0.5..0.5))
                    .collect();
                tensor4_from_data(data, left, 2, 2, right).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let expected: Vec<_> = [0, 1, 0x555, 0xfff]
        .into_iter()
        .map(|flat| {
            let indices: Vec<_> = (0..12).map(|site| (flat >> site) & 1).collect();
            (flat, mpo.evaluate(&indices).unwrap())
        })
        .collect();
    measure(
        "mpo_random_6_2x2_bond8",
        samples,
        || mpo.full_tensor().unwrap(),
        &expected,
    );
}
