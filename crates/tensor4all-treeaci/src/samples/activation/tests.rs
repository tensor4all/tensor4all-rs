use super::*;
use crate::{
    problem::prepare_problem,
    samples::SampleArena,
    test_support::{random_decaying_tree, SplitMix},
    TreeAciOptions,
};
use std::collections::HashSet;
use tensor4all_core::DynIndex;

fn problem(edges: &[(usize, usize)], n: usize) -> PreparedTreeProblem<usize> {
    let physical: Vec<_> = (0..n).map(|_| DynIndex::new_dyn(2)).collect();
    let input = random_decaying_tree::<f64>(edges, &physical, 1, 1.0, &mut SplitMix(1265));
    prepare_problem::<f64, _>(&[input], &TreeAciOptions::default()).unwrap()
}

fn inward(problem: &PreparedTreeProblem<usize>, center: usize) -> Vec<bool> {
    let mut neighbors = vec![vec![]; problem.node_order.len()];
    for edge in &problem.directed_edges {
        neighbors[edge.from].push(edge.to);
    }
    let mut distance = vec![usize::MAX; neighbors.len()];
    distance[center] = 0;
    let mut pending = vec![center];
    while let Some(node) = pending.pop() {
        for &next in &neighbors[node] {
            if distance[next] == usize::MAX {
                distance[next] = distance[node] + 1;
                pending.push(next);
            }
        }
    }
    problem
        .directed_edges
        .iter()
        .map(|edge| distance[edge.from] > distance[edge.to])
        .collect()
}

#[test]
fn every_small_point_capacity_mask_and_center_keep_distinct_useful_candidates_closed() {
    for edges in [vec![(0, 1), (1, 2), (2, 3)], vec![(0, 1), (0, 2), (0, 3)]] {
        let problem = problem(&edges, 4);
        for center in 0..4 {
            let closure = inward(&problem, center);
            for with_extra in [false, true] {
                let (mut arena, mut base) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
                let mut limits = [1; 3];
                if with_extra {
                    arena
                        .inject_global_point_masked(
                            &mut base,
                            &problem,
                            &[0, 1, 0, 0],
                            CandidateCapacity {
                                limits: &[2; 3],
                                required_closure: &closure,
                            },
                        )
                        .unwrap();
                    for (edge, limit) in limits.iter_mut().enumerate() {
                        *limit = base.ids[2 * edge].len().max(base.ids[2 * edge + 1].len());
                    }
                }
                for bits in 0..16 {
                    let point: Vec<_> = (0..4).map(|bit| (bits >> bit) & 1).collect();
                    for mask in 0..8 {
                        let point_limits: Vec<_> = (0..3)
                            .map(|edge| limits[edge] + usize::from(mask & (1 << edge) != 0))
                            .collect();
                        let mut actual = base.clone();
                        let mut samples = arena.clone();
                        let report = samples
                            .inject_global_point_masked(
                                &mut actual,
                                &problem,
                                &point,
                                CandidateCapacity {
                                    limits: &point_limits,
                                    required_closure: &closure,
                                },
                            )
                            .unwrap();
                        for (cut, ids) in actual.ids.iter().enumerate() {
                            assert!(ids.starts_with(&base.ids[cut]));
                            assert_eq!(
                                ids.iter().copied().collect::<HashSet<_>>().len(),
                                ids.len()
                            );
                            assert_eq!(ids.len(), base.ids[cut].len() + report.added_by_edge[cut]);
                            assert!(ids.len() <= point_limits[cut / 2]);
                            if closure[cut] {
                                for &id in ids {
                                    for &(child, sample) in
                                        &samples.record(cut, id).unwrap().incoming
                                    {
                                        assert!(actual.ids[child].contains(&sample));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn query_headroom_admits_a_distinct_candidate_and_generation_overflow_rolls_back() {
    let problem = problem(&[(0, 1)], 2);
    let (mut arena, mut candidates) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
    let before = candidates.clone();
    let records = arena.record_count();
    let budget = CandidateCapacity {
        limits: &[2],
        required_closure: &[true, false],
    };
    candidates.generation = u64::MAX;
    let overflowing = candidates.clone();
    assert!(matches!(
        arena.inject_global_point_masked(&mut candidates, &problem, &[0, 1], budget),
        Err(TreeAciError::SizeOverflow {
            context: "sample generation"
        })
    ));
    assert_eq!(candidates, overflowing);
    assert_eq!(arena.record_count(), records);
    candidates = before;
    let report = arena
        .inject_global_point_masked(&mut candidates, &problem, &[0, 1], budget)
        .unwrap();
    assert_eq!(report.total_added, 1);
    assert_eq!(candidates.ids, vec![vec![0], vec![0, 1]]);
}

#[test]
fn budget_shape_and_memory_fail_before_projection_or_candidate_mutation() {
    let mut problem = problem(&[(0, 1)], 2);
    let (mut arena, mut candidates) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
    let before = candidates.clone();
    let records = arena.record_count();
    for budget in [
        CandidateCapacity {
            limits: &[],
            required_closure: &[true; 2],
        },
        CandidateCapacity {
            limits: &[1],
            required_closure: &[],
        },
        CandidateCapacity {
            limits: &[0],
            required_closure: &[true; 2],
        },
    ] {
        assert!(arena
            .inject_global_point_masked(&mut candidates, &problem, &[1, 1], budget)
            .is_err());
    }
    problem.max_working_bytes = 1;
    assert!(matches!(
        arena.inject_global_point_masked(
            &mut candidates,
            &problem,
            &[1, 1],
            CandidateCapacity {
                limits: &[1],
                required_closure: &[true; 2]
            }
        ),
        Err(TreeAciError::ResourceLimit {
            resource: "working bytes",
            ..
        })
    ));
    assert_eq!(candidates, before);
    assert_eq!(arena.record_count(), records);
}
