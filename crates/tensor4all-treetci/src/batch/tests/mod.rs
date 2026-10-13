use super::{checked_batch_len, evaluate_points_chunked, GlobalIndexBatch};
use std::cell::RefCell;

/// Encode each point as a number so values pin both content and order.
fn encode(batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<u64>> {
    Ok(batch
        .data()
        .chunks(batch.n_sites())
        .map(|point| point.iter().fold(0u64, |acc, &v| acc * 100 + v as u64))
        .collect())
}

fn run_chunked(n_points: usize, chunk_points: usize) -> (Vec<u64>, Vec<usize>) {
    let calls = RefCell::new(Vec::new());
    let evaluate = |batch: GlobalIndexBatch<'_>| {
        calls.borrow_mut().push(batch.n_points());
        encode(batch)
    };
    let mut next = 0usize;
    let values = evaluate_points_chunked(
        3,
        n_points,
        chunk_points,
        |point| {
            point.copy_from_slice(&[next % 7, next / 7, 42]);
            next += 1;
        },
        &evaluate,
        "test points",
    )
    .unwrap();
    (values, calls.into_inner())
}

#[test]
fn evaluate_points_chunked_matches_single_call_for_every_chunk_size() {
    let (reference, calls) = run_chunked(23, usize::MAX);
    assert_eq!(calls, vec![23]);
    let expected: Vec<u64> = (0..23u64)
        .map(|p| (p % 7) * 10_000 + (p / 7) * 100 + 42)
        .collect();
    assert_eq!(reference, expected);

    // A chunk size that does not divide the point count, one that does,
    // one point per call, and an exact single chunk.
    for (chunk, sizes) in [
        (5, vec![5, 5, 5, 5, 3]),
        (1, vec![1; 23]),
        (23, vec![23]),
        (24, vec![23]),
    ] {
        let (values, calls) = run_chunked(23, chunk);
        assert_eq!(values, reference, "chunk {chunk}");
        assert_eq!(calls, sizes, "chunk {chunk}");
    }
}

#[test]
fn evaluate_points_chunked_rejects_invalid_input() {
    let evaluate = |batch: GlobalIndexBatch<'_>| encode(batch);
    assert!(evaluate_points_chunked(2, 3, 0, |_| {}, &evaluate, "p").is_err());
    assert!(evaluate_points_chunked(0, 3, 2, |_| {}, &evaluate, "p").is_err());
    assert!(evaluate_points_chunked(2, 0, 2, |_| {}, &evaluate, "p").is_err());

    // A wrong value count in a later chunk is reported with its size.
    let short = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<u64>> {
        let n = batch.n_points();
        Ok(vec![0; if n == 3 { n } else { n - 1 }])
    };
    let error =
        evaluate_points_chunked(1, 5, 3, |point| point[0] = 0, &short, "test points").unwrap_err();
    assert_eq!(
        error.to_string(),
        "batch evaluator returned 1 values for 2 test points (points 3..5 of 5)"
    );
}

#[test]
fn checked_batch_len_accepts_valid_shape() {
    assert_eq!(checked_batch_len(2, 3).unwrap(), 6);
}

#[test]
fn checked_batch_len_rejects_overflow() {
    assert!(checked_batch_len(usize::MAX, 2).is_err());
}
