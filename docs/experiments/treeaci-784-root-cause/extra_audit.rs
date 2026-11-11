pub(crate) fn edge_before<T: TreeAciScalar, V: TreeAciNode>(
    state: &TreeAciState<'_, T, V>,
    forward: usize,
    hints: usize,
) -> Result<()> {
    let reverse = state.problem.directed_edges[forward].reverse;
    let available = |edge: usize, id: usize| -> Result<bool> {
        Ok(state
            .sample_arena
            .record(edge, id)?
            .incoming
            .iter()
            .all(|&(d, i)| state.candidates.ids[d].contains(&i)))
    };
    let mut rows = 0;
    let mut cols = 0;
    for &(a, b) in &state.pivots.per_edge[forward / 2] {
        let (a, b) = if forward % 2 == 0 { (a, b) } else { (b, a) };
        rows += usize::from(available(forward, a)?);
        cols += usize::from(available(reverse, b)?);
    }
    eprintln!("ROOT_EDGE {{\"pass\":{},\"directed\":{},\"rank\":{},\"pivot_rank\":{},\"hints\":{},\"row_surviving\":{},\"col_surviving\":{}}}",PASS.load(Ordering::Relaxed),forward,state.edge_ranks[forward/2],state.pivots.rank(forward/2),hints,rows,cols);
    Ok(())
}

pub(crate) fn repair_guard_candidates<T: TreeAciScalar, V: TreeAciNode>(
    state: &mut TreeAciState<'_, T, V>,
) -> Result<()> {
    if std::env::var("T4A_ROOT_REPAIR").as_deref() != Ok("on") {
        return Ok(());
    }
    let node = state.output.canonical_region().iter().next().unwrap();
    let center = state.problem.node_positions[node];
    let before = inspect(
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.pivots,
        center,
        None,
    )?
    .active_cc;
    let mut repairs = 0;
    for _ in 0..state.candidates.ids.iter().map(Vec::len).sum::<usize>() {
        let c = inspect(
            &state.problem,
            &state.sample_arena,
            &state.candidates,
            &state.pivots,
            center,
            None,
        )?;
        let mut replacement = None;
        'scan: for d in &c.directions {
            if d[3] == 0 || d[6] == 0 {
                continue;
            }
            for &id in &state.candidates.ids[d[0]] {
                for &(child, missing) in &state.sample_arena.record(d[0], id)?.incoming {
                    let ids = &state.candidates.ids[child];
                    if ids.contains(&missing) {
                        continue;
                    }
                    for i in state.pivots.rank(child / 2)..ids.len() {
                        if ids[..i].contains(&ids[i]) {
                            replacement = Some((child, i, missing));
                            break 'scan;
                        }
                    }
                }
            }
        }
        let Some((child, slot, id)) = replacement else {
            break;
        };
        eprintln!(
            "ROOT_REPLACE {{\"pass\":{},\"child_cut\":{},\"slot\":{},\"old\":{},\"new\":{}}}",
            PASS.load(Ordering::Relaxed),
            child,
            slot,
            state.candidates.ids[child][slot],
            id
        );
        state.candidates.ids[child][slot] = id;
        repairs += 1;
    }
    if repairs > 0 {
        state.candidates.generation += 1;
        state.generation = state.candidates.generation;
    }
    let after = inspect(
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.pivots,
        center,
        None,
    )?
    .active_cc;
    eprintln!(
        "ROOT_REPAIR {{\"pass\":{},\"before\":{},\"after\":{},\"replacements\":{}}}",
        PASS.load(Ordering::Relaxed),
        before,
        after,
        repairs
    );
    Ok(())
}
