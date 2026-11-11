//! Read-only investigation probe; compiled only into the retained audit binary.
use crate::{
    problem::PreparedTreeProblem,
    samples::{CandidateSets, PivotPairs, SampleArena},
    state::TreeAciState,
    Result, TreeAciNode, TreeAciScalar,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
static PASS: AtomicUsize = AtomicUsize::new(0);
pub(crate) fn set_pass(pass: usize) {
    PASS.store(pass, Ordering::Relaxed);
}
#[derive(Default, Debug)]
struct Counts {
    full_pc: usize,
    active_pc: usize,
    full_pp: usize,
    active_pp: usize,
    full_cc: usize,
    active_cc: usize,
    local_pc: usize,
    first: Vec<usize>,
    active_first: Vec<usize>,
    directions: Vec<[usize; 8]>,
}
fn inspect<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    arena: &SampleArena,
    candidates: &CandidateSets,
    pivots: &PivotPairs,
    center: usize,
    updated: Option<usize>,
) -> Result<Counts> {
    let n = problem.node_order.len();
    let mut adjacency = vec![Vec::new(); n];
    for e in &problem.directed_edges {
        adjacency[problem.node_positions[&e.from]].push(problem.node_positions[&e.to]);
    }
    let mut distance = vec![usize::MAX; n];
    distance[center] = 0;
    let mut stack = vec![center];
    while let Some(node) = stack.pop() {
        for &next in &adjacency[node] {
            if distance[next] == usize::MAX {
                distance[next] = distance[node] + 1;
                stack.push(next);
            }
        }
    }
    let candidate_sets: Vec<HashSet<usize>> = candidates
        .ids
        .iter()
        .map(|ids| ids.iter().copied().collect())
        .collect();
    let pivot_ids: Vec<Vec<usize>> = problem
        .directed_edges
        .iter()
        .map(|e| {
            pivots.per_edge[e.id / 2]
                .iter()
                .map(|&(a, b)| if e.id % 2 == 0 { a } else { b })
                .collect()
        })
        .collect();
    let pivot_sets: Vec<HashSet<usize>> = pivot_ids
        .iter()
        .map(|ids| ids.iter().copied().collect())
        .collect();
    let mut c = Counts::default();
    for e in &problem.directed_edges {
        let from = problem.node_positions[&e.from];
        let to = problem.node_positions[&e.to];
        let active = distance[from] > distance[to];
        let (mut pc, mut pp, mut cc) = (0, 0, 0);
        for &id in &pivot_ids[e.id] {
            for &(child_edge, child) in &arena.record(e.id, id)?.incoming {
                if !candidate_sets[child_edge].contains(&child) {
                    pc += 1;
                    let witness = vec![e.id, id, child_edge, child];
                    if c.first.is_empty() {
                        c.first = witness.clone();
                    }
                    if active && c.active_first.is_empty() {
                        c.active_first = witness;
                    }
                }
                if !pivot_sets[child_edge].contains(&child) {
                    pp += 1;
                }
            }
        }
        for &id in &candidates.ids[e.id] {
            for &(child_edge, child) in &arena.record(e.id, id)?.incoming {
                if !candidate_sets[child_edge].contains(&child) {
                    cc += 1;
                }
            }
        }
        c.full_pc += pc;
        c.full_pp += pp;
        c.full_cc += cc;
        if active {
            c.active_pc += pc;
            c.active_pp += pp;
            c.active_cc += cc;
        }
        if updated.is_some_and(|d| d / 2 == e.id / 2) {
            c.local_pc += pc;
        }
        c.directions.push([
            e.id,
            from,
            to,
            usize::from(active),
            pc,
            pp,
            cc,
            pivot_ids[e.id].len(),
        ]);
    }
    Ok(c)
}
pub(crate) fn record<T: TreeAciScalar, V: TreeAciNode>(
    state: &TreeAciState<'_, T, V>,
    stage: &str,
    updated: Option<usize>,
    stable: Option<usize>,
    guard: Option<usize>,
) -> Result<()> {
    assert_eq!(state.output.canonical_region().len(), 1);
    let node = state.output.canonical_region().iter().next().unwrap();
    let center = state.problem.node_positions[node];
    let c = inspect(
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.pivots,
        center,
        updated,
    )?;
    let mut prefix_mismatches = 0;
    let mut dimension_mismatches = 0;
    for e in &state.problem.directed_edges {
        let pivot_ids: Vec<_> = state.pivots.per_edge[e.id / 2]
            .iter()
            .map(|&(a, b)| if e.id % 2 == 0 { a } else { b })
            .collect();
        if !state.candidates.ids[e.id].starts_with(&pivot_ids) {
            prefix_mismatches += 1;
        }
        let graph = state.output.edge_between(&e.from, &e.to).unwrap();
        let dim = state.output.bond_index(graph).unwrap().dim();
        if dim != state.candidates.ids[e.id].len() {
            dimension_mismatches += 1;
        }
    }
    if stage == "after-guard" && c.active_cc > 0 {
        for d in &c.directions {
            if d[3] == 0 || d[6] == 0 {
                continue;
            }
            let edge = d[0];
            for &id in &state.candidates.ids[edge] {
                for &(child_edge, child) in &state.sample_arena.record(edge, id)?.incoming {
                    if state.candidates.ids[child_edge].contains(&child) {
                        continue;
                    }
                    let unique: HashSet<_> =
                        state.candidates.ids[child_edge].iter().copied().collect();
                    eprintln!("NESTING_CANDIDATE {{\"pass\":{},\"parent_edge\":{},\"parent_id\":{},\"child_edge\":{},\"child_id\":{},\"child_candidates\":{:?},\"child_unique_count\":{},\"child_rank\":{},\"child_algebraic_bound\":{}}}",PASS.load(Ordering::Relaxed),edge,id,child_edge,child,state.candidates.ids[child_edge],unique.len(),state.edge_ranks[child_edge/2],state.algebraic_edge_bounds[child_edge/2]);
                }
            }
        }
    }
    if stage == "initialize" {
        // Positive control: deleting a required child from a cloned candidate set
        // must be reported by both the full and center-inward checks.
        let mut active_control = false;
        let mut outward_control = false;
        for d in &c.directions {
            let edge = d[0];
            for &(a, b) in &state.pivots.per_edge[edge / 2] {
                let id = if edge % 2 == 0 { a } else { b };
                for &(incoming, child) in &state.sample_arena.record(edge, id)?.incoming {
                    if d[3] == 1 && !active_control {
                        let mut invalid = state.candidates.clone();
                        invalid.ids[incoming].retain(|&i| i != child);
                        let bad = inspect(
                            &state.problem,
                            &state.sample_arena,
                            &invalid,
                            &state.pivots,
                            center,
                            None,
                        )?;
                        assert!(bad.active_pc > c.active_pc && bad.full_pc > c.full_pc);
                        active_control = true;
                    }
                    if d[3] == 0 && c.directions[incoming][3] == 0 && !outward_control {
                        let mut invalid = state.candidates.clone();
                        invalid.ids[incoming].retain(|&i| i != child);
                        let bad = inspect(
                            &state.problem,
                            &state.sample_arena,
                            &invalid,
                            &state.pivots,
                            center,
                            None,
                        )?;
                        assert!(bad.full_pc > c.full_pc);
                        assert_eq!(bad.active_pc, c.active_pc);
                        outward_control = true;
                    }
                }
            }
        }
        assert!(active_control && outward_control);
        eprintln!("NESTING_CONTROL {{\"active_missing_detected\":true,\"outward_only_distinguished\":true}}");
    }
    eprintln!("NESTING {{\"pass\":{},\"stage\":{:?},\"updated\":{:?},\"center\":{},\"ranks\":{:?},\"stable\":{},\"guard\":{},\"full_pc\":{},\"active_pc\":{},\"full_pp\":{},\"active_pp\":{},\"full_cc\":{},\"active_cc\":{},\"local_pc\":{},\"prefix_mismatches\":{},\"dimension_mismatches\":{},\"first\":{:?},\"active_first\":{:?},\"directions\":{:?}}}",PASS.load(Ordering::Relaxed),stage,updated.map(|i|i as i64).unwrap_or(-1),center,state.edge_ranks,stable.map(|i|i as i64).unwrap_or(-1),guard.map(|i|i as i64).unwrap_or(-1),c.full_pc,c.active_pc,c.full_pp,c.active_pp,c.full_cc,c.active_cc,c.local_pc,prefix_mismatches,dimension_mismatches,c.first,c.active_first,c.directions);
    Ok(())
}
use tensor4all_core::IndexLike;
