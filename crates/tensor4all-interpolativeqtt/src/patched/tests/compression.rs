use super::*;

// Reference from the parent (#1310), retaining per-point Lagrange evaluation
// and the original left-to-right TT evaluate(). Never used by production.
fn compression_change_reference<T: InterpolativeScalar>(
    plan: &Plan<'_, T>,
    domain: &NInterval,
    children: &[(NInterval, Vec<T>)],
    compressed: &SimpleTensorTrain<T>,
) -> Result<f64> {
    let sites = compressed.len();
    let ndims = domain.ndims();
    // Match sampled node sets to children by their exact intervals: the
    // assembly splits this domain with the same `NInterval::split`, and every
    // box of the driver is safe, so each child was sampled exactly once.
    let pieces = domain.split()?;
    let node_values: Vec<&[T]> = pieces
        .iter()
        .map(|piece| {
            children
                .iter()
                .find(|(interval, _)| interval == piece)
                .map(|(_, values)| values.as_slice())
                .unwrap_or_else(|| {
                    tensor4all_tensorbackend::invariant_violated(
                        "patched compression check",
                        "a child box was not sampled on its Chebyshev nodes",
                    )
                })
        })
        .collect();
    let half = 1u64 << (sites - 1);
    let axis = compression_check_indices(&plan.test_points_1d, sites);
    let count = checked_power(axis.len(), ndims)?;
    let mut max_change = 0.0_f64;
    let mut quantics = vec![0; sites];
    let mut lagrange = vec![Vec::with_capacity(plan.basis.len()); ndims];
    for multi in product_indices(axis.len(), ndims, count) {
        let mut child = 0;
        for (d, &i) in multi.iter().enumerate() {
            let k = axis[i];
            let upper = k >= half;
            child |= usize::from(upper) << d;
            let local = (k - u64::from(upper) * half) as f64 / half as f64;
            lagrange[d].clear();
            for alpha in 0..plan.basis.len() {
                lagrange[d].push(plan.basis.evaluate(alpha, local)?);
            }
        }
        let mut interpolant = T::zero();
        for (node, &value) in plan.nodes.iter().zip(node_values[child]) {
            let weight: f64 = node
                .iter()
                .enumerate()
                .map(|(d, &alpha)| lagrange[d][alpha])
                .product();
            interpolant = interpolant + value * T::from_f64(weight);
        }
        for (level, site) in quantics.iter_mut().enumerate() {
            let shift = sites - 1 - level;
            *site = multi
                .iter()
                .enumerate()
                .map(|(d, &i)| (((axis[i] >> shift) & 1) as usize) << d)
                .sum();
        }
        let change = (compressed.evaluate(&quantics)? - interpolant).abs_val();
        max_change = max_change.max(change);
    }
    Ok(max_change)
}

fn fixture<T: InterpolativeScalar>(
    plan: &Plan<'_, T>,
    session: &mut Session<'_>,
    f: impl Fn(&[f64]) -> T,
) -> (Vec<(NInterval, Vec<T>)>, SimpleTensorTrain<T>) {
    let mut children = Vec::new();
    let cores = build_refined_cores(
        &mut |batch: SampleBatch<'_>| {
            let values: Vec<T> = batch
                .points()
                .data()
                .chunks_exact(plan.domain.ndims())
                .map(&f)
                .collect();
            if let SampleBatch::Nodes { interval, .. } = batch {
                children.push((interval.clone(), values.clone()));
            }
            Ok(values)
        },
        &plan.domain,
        plan.num_bits,
        &plan.basis,
        |_, _| true,
    )
    .unwrap();
    let mut train = SimpleTensorTrain::new(cores).unwrap();
    train
        .compress_svd_with_session(session, 1e-10, Some(4))
        .unwrap();
    (children, train)
}

fn target(p: &[f64]) -> f64 {
    let phase: f64 = p.iter().enumerate().map(|(d, &x)| (d + 1) as f64 * x).sum();
    (phase * phase).sin() + 0.2 * phase.exp()
}

fn compare<T: InterpolativeScalar>(f: impl Fn(&[f64]) -> T) {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());
    for (dims, sites, degree) in [(1, 2, 3), (1, 10, 4), (2, 3, 2), (2, 8, 3), (3, 4, 2)] {
        let options = PatchedInterpolativeOptions::default();
        let plan =
            Plan::<T>::new(&vec![-0.3; dims], &vec![0.9; dims], sites, degree, &options).unwrap();
        let (children, train) = fixture(&plan, &mut session, &f);
        let old = compression_change_reference(&plan, &plan.domain, &children, &train).unwrap();
        let new = compression_change(&plan, &plan.domain, &children, &train).unwrap();
        assert_eq!(
            old.to_bits(),
            new.to_bits(),
            "D={dims}, L={sites}, p={degree}: {old:e} vs {new:e}"
        );
    }
}

#[test]
fn cached_compression_change_is_bitwise_unchanged() {
    compare(target);
    compare(|p| Complex64::new(target(p), (p[0] - p[p.len() - 1]).cos()));
}

#[test]
#[ignore = "paired release timing experiment; run explicitly with pinned CPU and thread settings"]
fn compression_change_timing() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());
    println!("dims,sites,degree,rank,pair,iterations,reference_ns,candidate_ns,residual_bits");
    for (dims, sites, degree, iterations) in [
        (1, 4, 3, 100),
        (1, 12, 6, 100),
        (2, 4, 3, 10),
        (2, 12, 3, 10),
        (3, 4, 2, 3),
        (3, 10, 2, 3),
    ] {
        let options = PatchedInterpolativeOptions::default();
        let plan =
            Plan::<f64>::new(&vec![-0.3; dims], &vec![0.9; dims], sites, degree, &options).unwrap();
        let (children, train) = fixture(&plan, &mut session, target);
        let check = |reference| {
            let p = black_box(&plan);
            let c = black_box(&children);
            let t = black_box(&train);
            if reference {
                compression_change_reference(p, &p.domain, c, t)
            } else {
                compression_change(p, &p.domain, c, t)
            }
            .unwrap()
        };
        let reference = check(true);
        assert_eq!(reference.to_bits(), check(false).to_bits());
        // Equal time-based priming before every case, then warm both paths.
        let prime = Instant::now();
        while prime.elapsed() < Duration::from_secs(1) {
            black_box(1.0_f64.exp());
        }
        black_box(check(true));
        black_box(check(false));
        for pair in 0..11 {
            let mut times = [0; 2];
            for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                for _ in 0..iterations {
                    black_box(check(arm == 0));
                }
                times[arm] = start.elapsed().as_nanos();
            }
            println!(
                "{dims},{sites},{degree},{},{pair},{iterations},{},{},{}",
                train.rank(),
                times[0],
                times[1],
                reference.to_bits()
            );
        }
    }
}
