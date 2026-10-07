// Fresh-process RSS and throughput comparison for issue #805.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Instant;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mode = args.first().map(String::as_str).unwrap_or("bounded");
    let rank = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(17usize);
    let count = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1024usize);
    let generic = args.get(3).is_some_and(|s| s == "generic");
    let physical = (0..4).map(|_| DynIndex::new_dyn(128)).collect::<Vec<_>>();
    let bonds = (0..3).map(|_| DynIndex::new_dyn(rank)).collect::<Vec<_>>();
    // One physical value at the hub is enough; leaves supply changing points.
    let hub_site = DynIndex::new_dyn(2);
    let mut hub_indices = bonds.clone();
    if !generic {
        hub_indices.insert(0, hub_site.clone());
    }
    let mut tensors = vec![IdxTensor::from_dense(
        hub_indices,
        vec![1.0_f64; (if generic { 1 } else { 2 }) * rank * rank * rank],
    )?];
    for leaf in 0..3 {
        let data = (0..128)
            .flat_map(|site| {
                (0..rank).map(move |bond| {
                    (1.0 + site as f64 / 128.0 + bond as f64 / rank as f64) / rank as f64
                })
            })
            .collect();
        tensors.push(IdxTensor::from_dense(
            vec![bonds[leaf].clone(), physical[leaf + 1].clone()],
            data,
        )?);
    }
    let mut indices = physical[1..].to_vec();
    if !generic {
        indices.insert(0, hub_site);
    }
    let tree = TreeTN::from_tensors(tensors, vec![0_usize, 1, 2, 3])?;
    let mut rng = ChaCha8Rng::seed_from_u64(123456789);
    let values = (0..count)
        .flat_map(|_| {
            (0..indices.len())
                .map(|axis| rng.random_range(0..if !generic && axis == 0 { 2 } else { 128 }))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let shape = [indices.len(), count];
    let points = ColMajorArrayRef::new(&values, &shape)?;
    let mut options = CachedEvaluatorOptions {
        center: Some(if generic { 1 } else { 0 }),
        ..Default::default()
    };
    match mode {
        "bounded" => {}
        "chunk16" => options.max_batch_points = Some(16),
        "chunk64" => options.max_batch_points = Some(64),
        "chunk256" => options.max_batch_points = Some(256),
        "unchunked" => options.max_batch_points = Some(usize::MAX),
        "legacy" => {
            options.max_batch_points = Some(usize::MAX);
            options.message_cache_max_bytes = usize::MAX;
            options.branch_slice_cache_max_bytes = usize::MAX;
        }
        _ => anyhow::bail!("mode must be bounded, unchunked, or legacy"),
    }
    let mut evaluator = TreeTNCachedEvaluator::new(&tree, &indices, options)?;
    let started = Instant::now();
    let result = evaluator.evaluate_batched(black_box(points))?;
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    let mut max_relative_error = 0.0_f64;
    for (point, actual) in values.chunks_exact(indices.len()).zip(&result) {
        let offset = (rank - 1) as f64 / (2.0 * rank as f64);
        let expected = point[if generic { 0 } else { 1 }..]
            .iter()
            .map(|&site| 1.0 + site as f64 / 128.0 + offset)
            .product::<f64>();
        max_relative_error = max_relative_error.max((actual.real() - expected).abs() / expected);
    }
    assert!(
        max_relative_error < 1.0e-12,
        "relative error {max_relative_error}"
    );
    let status = std::fs::read_to_string("/proc/self/status")?;
    let rss = status
        .lines()
        .find(|s| s.starts_with("VmHWM:"))
        .unwrap_or("VmHWM: unavailable");
    println!("fixture={},mode={mode},rank={rank},points={count},elapsed_ms={elapsed:.6},max_relative_error={max_relative_error:.3e},{rss}", if generic { "generic" } else { "raw" });
    black_box(result);
    Ok(())
}
