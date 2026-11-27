//! Exact production point generators, isolated from TreeTN evaluation and caches.
type MultiIndex = Vec<usize>;
include!("baseline.rs");
#[path = "candidate.rs"]
mod candidate;

fn consume(batch: &[MultiIndex], count: &mut usize, checksum: &mut u64) {
    for point in batch {
        *count += 1;
        for &coordinate in point {
            *checksum = (*checksum ^ coordinate as u64).wrapping_mul(1099511628211);
        }
    }
}

fn main() {
    let sites = 1001;
    let dims: Vec<_> = (0..sites).map(|i| if i == 0 { 1 } else { 2 }).collect();
    let sides: Vec<Vec<_>> = (0..200).map(|i| (1 + 5 * i..6 + 5 * i).collect()).collect();
    let mut anchors = vec![vec![0; sites]; 5];
    for (value, anchor) in anchors.iter_mut().enumerate() {
        for bit in 0..3 {
            anchor[1 + bit] = (value >> bit) & 1;
        }
    }
    let chunk = (1 << 20) / sites;
    let mut count = 0;
    let mut checksum = 14695981039346656037;
    match std::env::args().nth(1).as_deref() {
        Some("baseline") => {
            let points = enumeration_points(&anchors, &sides, &dims);
            for batch in points.chunks(chunk) {
                consume(batch, &mut count, &mut checksum);
            }
        }
        Some("candidate") => {
            let mut points = candidate::EnumerationPoints::new(&anchors, &sides, &dims);
            loop {
                let batch: Vec<_> = points.by_ref().take(chunk).collect();
                if batch.is_empty() { break; }
                consume(&batch, &mut count, &mut checksum);
            }
        }
        _ => panic!("expected baseline or candidate"),
    }
    assert_eq!(count, 30_877);
    println!("{{\"points\":{count},\"checksum\":{checksum},\"chunk_points\":{chunk}}}");
}
