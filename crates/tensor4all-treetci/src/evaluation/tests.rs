use super::*;

#[test]
fn failed_batches_are_retryable_without_retaining_partial_results() {
    for limit in [None, Some(0), Some(1024)] {
        let mut calls = 0;
        let evaluator = RunEvaluator::new(
            |batch: GlobalIndexBatch<'_>| {
                calls += 1;
                match calls {
                    1 => anyhow::bail!("oracle unavailable"),
                    2 => Ok(Vec::new()),
                    _ => Ok(batch.data().iter().map(|&i| 10.0 * i as f64).collect()),
                }
            },
            &[2],
            limit,
        )
        .unwrap();
        let batch = GlobalIndexBatch::new(&[1, 0, 1], 1, 3).unwrap();
        assert_eq!(
            evaluator.call(batch).unwrap_err().to_string(),
            "oracle unavailable"
        );
        if limit.is_some() {
            assert!(evaluator.call(batch).is_err());
        } else {
            // Uncached calls preserve the phase-specific length diagnostic
            // at the caller (initialization, edge update or materialization).
            assert!(evaluator.call(batch).unwrap().is_empty());
        }
        assert_eq!(evaluator.stats().cached_entries, 0);
        assert_eq!(evaluator.stats().evaluated_points, 0);
        assert_eq!(evaluator.call(batch).unwrap(), vec![10.0, 0.0, 10.0]);
        assert_eq!(evaluator.call(batch).unwrap(), vec![10.0, 0.0, 10.0]);
        let stats = evaluator.stats();
        assert_eq!(stats.requested_points, 12);
        assert_eq!(
            stats.evaluated_points,
            if limit.is_none() {
                6
            } else if limit == Some(0) {
                4
            } else {
                2
            }
        );
        assert_eq!(
            stats.cached_entries,
            if limit == Some(1024) { 2 } else { 0 }
        );
    }
}

#[test]
fn invalid_memo_coordinates_do_not_call_the_oracle() {
    let evaluator = RunEvaluator::<f64, _>::new(
        |_| panic!("invalid batch must not reach the target"),
        &[2],
        Some(1024),
    )
    .unwrap();
    for batch in [
        GlobalIndexBatch::new(&[0, 2], 1, 2).unwrap(),
        GlobalIndexBatch::new(&[0, 0], 2, 1).unwrap(),
    ] {
        assert!(evaluator.call(batch).is_err());
        assert_eq!(evaluator.stats().cached_entries, 0);
    }
}

#[test]
fn wide_index_spaces_remain_supported_when_memo_is_disabled() {
    let dims = vec![2; 1025];
    let evaluator =
        RunEvaluator::new(|batch| Ok(vec![3.0; batch.n_points()]), &dims, None).unwrap();
    let indices = vec![0; 1025];
    assert_eq!(
        evaluator
            .call(GlobalIndexBatch::new(&indices, 1025, 1).unwrap())
            .unwrap(),
        vec![3.0]
    );
    assert!(RunEvaluator::<f64, _>::new(|_| Ok(vec![]), &dims, Some(1024)).is_err());
}
