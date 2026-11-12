//! Temporary diagnostic instrumentation. Never part of a production dispatch.
use crate::{
    problem::PreparedTreeProblem,
    samples::{ComponentSample, SampleArena},
    state::TreeAciState,
    TreeAciNode, TreeAciScalar,
};
use serde_json::{json, Value};
use std::{
    any::Any,
    cell::RefCell,
    collections::HashMap,
    fs::File,
    hash::{Hash, Hasher},
    io::{BufWriter, Write},
};
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treetn::{
    CachedEvaluatorOptions, CachedEvaluatorPlan, EvaluationHint, TreeTNCachedEvaluator,
};

#[derive(Default)]
struct Context {
    pass: usize,
    step: usize,
    cut: usize,
    keys: HashMap<(usize, usize), u64>,
    plan: Option<Box<dyn Any>>,
}
thread_local! { static CONTEXT: RefCell<Context> = RefCell::new(Context::default()); }

fn emit(mut value: Value) {
    CONTEXT.with(|c| {
        let c = c.borrow();
        value["pass"] = json!(c.pass);
        value["step"] = json!(c.step);
        value["cut"] = json!(c.cut);
    });
    eprintln!("{value}");
}
fn component<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    arena: &SampleArena,
    cut: usize,
    id: usize,
    keys: &mut HashMap<(usize, usize), u64>,
) -> u64 {
    if let Some(&key) = keys.get(&(cut, id)) {
        return key;
    }
    let record = arena.record(cut, id).unwrap();
    let mut key = (record.local_coordinate as u64)
        << problem.node_positions[&problem.directed_edges[cut].from];
    for &(child, sample) in &record.incoming {
        key |= component(problem, arena, child, sample, keys);
    }
    keys.insert((cut, id), key);
    key
}
pub(crate) fn init<T: TreeAciScalar, V: TreeAciNode>(state: &TreeAciState<'_, T, V>) {
    assert!(state.problem.node_order.len() <= 60);
    assert!(state.problem.physical.iter().all(|p| p.local_dim == 2));
    let indices: Vec<_> = state
        .problem
        .physical
        .iter()
        .flat_map(|p| p.indices.iter().cloned())
        .collect();
    let plan = CachedEvaluatorPlan::new(&state.output, &indices).unwrap();
    CONTEXT.with(|c| {
        *c.borrow_mut() = Context {
            plan: Some(Box::new(plan)),
            ..Default::default()
        }
    });
}
pub(crate) fn pass(pass: usize) {
    CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        c.pass = pass;
        c.step = 0;
    });
}
pub(crate) fn state<T: TreeAciScalar, V: TreeAciNode>(state: &TreeAciState<'_, T, V>, kind: &str) {
    let (candidates, pivots) = CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        let keys = &mut c.keys;
        let candidates: Vec<Vec<_>> = state
            .candidates
            .ids
            .iter()
            .enumerate()
            .map(|(cut, ids)| {
                ids.iter()
                    .map(|&id| component(&state.problem, &state.sample_arena, cut, id, keys))
                    .collect()
            })
            .collect();
        let pivots: Vec<Vec<_>> = state
            .pivots
            .per_edge
            .iter()
            .enumerate()
            .map(|(edge, pairs)| {
                pairs
                    .iter()
                    .map(|&(a, b)| {
                        (
                            component(&state.problem, &state.sample_arena, 2 * edge, a, keys),
                            component(&state.problem, &state.sample_arena, 2 * edge + 1, b, keys),
                        )
                    })
                    .collect()
            })
            .collect();
        (candidates, pivots)
    });
    emit(json!({"kind":kind,"candidates":candidates,"pivots":pivots,"ranks":state.edge_ranks}));
}
pub(crate) fn before_edge<T: TreeAciScalar, V: TreeAciNode>(
    s: &TreeAciState<'_, T, V>,
    cut: usize,
) {
    CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        c.step += 1;
        c.cut = cut;
    });
    state(s, "before_edge");
}
fn sample_key(sample: &ComponentSample, cut: usize, positions: &[usize]) -> u64 {
    CONTEXT.with(|c| {
        let c = c.borrow();
        let mut key = (sample.local_coordinate as u64) << positions[cut];
        for &(child, id) in &sample.incoming {
            key |= c.keys[&(child, id)];
        }
        key
    })
}
pub(crate) fn matrix<T: TreeAciScalar, V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    cut: usize,
    rows: &[ComponentSample],
    cols: &[ComponentSample],
    values: &[T],
) {
    let positions: Vec<_> = problem
        .directed_edges
        .iter()
        .map(|e| problem.node_positions[&e.from])
        .collect();
    let row_keys: Vec<_> = rows
        .iter()
        .map(|s| sample_key(s, cut, &positions))
        .collect();
    let col_keys: Vec<_> = cols
        .iter()
        .map(|s| sample_key(s, problem.directed_edges[cut].reverse, &positions))
        .collect();
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    for &v in values {
        let (re, im) = v.widened_parts();
        re.to_bits().hash(&mut hash);
        im.to_bits().hash(&mut hash);
    }
    emit(
        json!({"kind":"matrix","rows":row_keys,"cols":col_keys,"matrix_hash":format!("{:016x}",hash.finish())}),
    );
    let target = std::env::var("T4A_DIAG_DUMP").unwrap_or_default();
    let selected = CONTEXT.with(|c| {
        let c = c.borrow();
        format!("{}:{}", c.pass, c.step) == target
    });
    if selected {
        let path = std::env::var("T4A_DIAG_DUMP_PATH").unwrap();
        let mut file = BufWriter::new(File::create(path).unwrap());
        file.write_all(&(rows.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&(cols.len() as u64).to_le_bytes()).unwrap();
        for &v in values {
            let (re, im) = v.widened_parts();
            file.write_all(&re.to_le_bytes()).unwrap();
            file.write_all(&im.to_le_bytes()).unwrap();
        }
    }
}
pub(crate) fn factors<T: TreeAciScalar>(f: &tensor4all_core::MatrixLuciFactors<T>, scale: f64) {
    emit(
        json!({"kind":"factors","rank":f.rank,"row_indices":f.row_indices,"col_indices":f.col_indices,"pivot_errors":f.pivot_errors,"sampled_scale":scale,
    "left_max":f.left.as_col_major_slice().iter().map(|&v|tensor4all_core::Scalar::abs_val(v)).fold(0.0,f64::max),
    "right_max":f.right.as_col_major_slice().iter().map(|&v|tensor4all_core::Scalar::abs_val(v)).fold(0.0,f64::max)}),
    );
}
pub(crate) fn guard(pivots: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let replay = std::env::var("T4A_DIAG_REPLAY").ok().and_then(|file| {
        let replay: Value = serde_json::from_reader(File::open(file).unwrap()).unwrap();
        let pass = CONTEXT.with(|c| c.borrow().pass);
        Some(
            replay
                .get(pass.to_string())?
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    p.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect()
                })
                .collect::<Vec<Vec<usize>>>(),
        )
    });
    emit(json!({"kind":"guard","natural":pivots,"injected":replay.as_ref().unwrap_or(&pivots)}));
    replay.unwrap_or(pivots)
}
pub(crate) fn after_edge<T: TreeAciScalar, V: TreeAciNode>(
    state: &TreeAciState<'_, T, V>,
    exec: &mut crate::exec::Exec<'_, '_>,
) {
    let enabled = std::env::var("T4A_DIAG_EDGE_PASSES").unwrap_or_default();
    let pass = CONTEXT.with(|c| c.borrow().pass);
    if enabled
        .split(',')
        .any(|p| p.parse::<usize>().ok() == Some(pass))
    {
        metrics(state, exec, "edge_error");
    }
}
pub(crate) fn metrics<T: TreeAciScalar, V: TreeAciNode>(
    state: &TreeAciState<'_, T, V>,
    exec: &mut crate::exec::Exec<'_, '_>,
    kind: &str,
) {
    if kind == "pass_error" && std::env::var("T4A_DIAG_PASS_METRICS").as_deref() == Ok("off") {
        return;
    }
    static DATA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    let data = DATA.get_or_init(|| {
        serde_json::from_reader(File::open(std::env::var("T4A_DIAG_VALIDATION").unwrap()).unwrap())
            .unwrap()
    });
    let points = data["points"].as_array().unwrap();
    let count = points.len();
    let n = data["importance_count"].as_u64().unwrap() as usize;
    let focus_full = std::env::var("T4A_DIAG_EDGE_FULL").unwrap_or_default();
    let selected = CONTEXT.with(|c| {
        let c = c.borrow();
        focus_full
            .split(',')
            .any(|v| v == format!("{}:{}", c.pass, c.step))
    });
    let importance_n = if kind == "edge_error" && !selected {
        256.min(n)
    } else {
        n
    };
    let sample_indices: Vec<usize> = (0..importance_n).chain(n..count).collect();
    let coordinates: Vec<_> = sample_indices
        .iter()
        .flat_map(|&i| {
            points[i]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
        })
        .collect();
    let plan = CONTEXT.with(|c| {
        c.borrow()
            .plan
            .as_ref()
            .unwrap()
            .downcast_ref::<CachedEvaluatorPlan<V>>()
            .unwrap()
            .clone()
    });
    let mut evaluator =
        TreeTNCachedEvaluator::with_plan(&state.output, &plan, CachedEvaluatorOptions::default())
            .unwrap();
    let actual = exec
        .evaluate::<T, V>(
            &mut evaluator,
            ColMajorArrayRef::new(
                &coordinates,
                &[state.problem.node_order.len(), sample_indices.len()],
            )
            .unwrap(),
            EvaluationHint::default(),
        )
        .unwrap();
    let norm = data["norm"].as_f64().unwrap();
    let peak = data["peak_scale"].as_f64().unwrap();
    let mut sum = 0.0;
    let mut peak_max = 0.0_f64;
    let mut peak_res = 0.0;
    let mut peak_ref = 0.0;
    for (&i, &a) in sample_indices.iter().zip(&actual) {
        let e = &data["expected"][i];
        let re = e[0].as_f64().unwrap();
        let im = e[1].as_f64().unwrap();
        let expected = T::from_widened_parts(re, im);
        let sq = tensor4all_core::Scalar::abs_sq(a - expected);
        if i < n {
            sum += sq / data["density"][i].as_f64().unwrap();
        } else {
            peak_max = peak_max.max(sq.sqrt());
            peak_res += sq;
            peak_ref += re * re + im * im;
        }
    }
    emit(
        json!({"kind":kind,"importance_samples":importance_n,"peak_samples":count-n,"importance_l2":(sum/importance_n as f64).sqrt()/norm,"normalized_max_error":peak_max/peak,"sampled_l2":(peak_res/peak_ref).sqrt()}),
    );
}

pub(crate) fn guard_starts(starts: &[Vec<usize>], errors: &[f64], nsearch: usize, threshold: f64) {
    emit(
        json!({"kind":"guard_starts","random_starts":&starts[..nsearch],"random_errors":&errors[..nsearch],"known_points":&starts[nsearch..],"known_errors":&errors[nsearch..],"threshold":threshold}),
    );
}
