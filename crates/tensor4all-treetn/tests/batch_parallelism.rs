//! The batch entrance takes the caller's `Parallelism` as a value.
//!
//! The expected bit patterns below were produced by `main` at `d41c7cd4`
//! (before the entrance took a `Parallelism`) with `max_batch_points = 1`, the
//! identity chunking, on the chain built here. Every route must reproduce them
//! bit for bit: the sequential value, and a pool of any budget, because the
//! shards are ranges of points and no point's arithmetic changes.

use num_complex::Complex64;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, IndexLike, TensorElement};
use tensor4all_treetn::interpolation::Parallelism;
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

/// Three-node chain, bond dimension 16, site dimensions `[2, 8, 2]`; the middle
/// node is an internal centre with two incoming components.
fn chain_tree<T: TensorElement>(
    value: impl Fn(usize) -> T + Copy,
) -> (Vec<DynIndex>, TreeTN<IdxTensor, usize>) {
    let sites: Vec<DynIndex> = [2usize, 8, 2]
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim))
        .collect();
    let bond0 = DynIndex::new_dyn(16);
    let bond1 = DynIndex::new_dyn(16);
    let mut offset = 0usize;
    let mut take = |count: usize| {
        let values = (0..count).map(|k| value(offset + k)).collect::<Vec<_>>();
        offset += count;
        values
    };
    let node0 = IdxTensor::from_dense(
        vec![sites[0].clone(), bond0.clone()],
        take(sites[0].dim() * bond0.dim()),
    )
    .expect("node 0");
    let node1 = IdxTensor::from_dense(
        vec![bond0.clone(), sites[1].clone(), bond1.clone()],
        take(16 * sites[1].dim() * 16),
    )
    .expect("node 1");
    let node2 = IdxTensor::from_dense(vec![bond1, sites[2].clone()], take(16 * sites[2].dim()))
        .expect("node 2");
    let tree =
        TreeTN::from_tensors(vec![node0, node1, node2], vec![0usize, 1, 2]).expect("chain tree");
    (sites, tree)
}

/// 99 points (not a multiple of the four-point lane block), scanning the
/// middle site.
fn points(sites: &[DynIndex]) -> (Vec<usize>, Vec<usize>) {
    let dims: Vec<usize> = sites.iter().map(IndexLike::dim).collect();
    let n = 99usize;
    let mut data = Vec::with_capacity(n * dims.len());
    for p in 0..n {
        data.push((p * 5 + 1) % dims[0]);
        data.push((p * 3) % dims[1]);
        data.push((p / 3) % dims[2]);
    }
    (data, vec![dims.len(), n])
}

fn real(k: usize) -> f64 {
    ((k * 37 % 101) as f64 - 50.0) / 17.0
}

fn complex(k: usize) -> Complex64 {
    Complex64::new(real(k), ((k * 53 % 89) as f64 - 44.0) / 13.0)
}

fn options(max_batch_points: Option<usize>) -> CachedEvaluatorOptions<usize> {
    CachedEvaluatorOptions {
        max_batch_points,
        ..CachedEvaluatorOptions::default()
    }
}

const MAIN_REAL_BITS: [u64; 99] = [
    4631086450597377226,
    4633009051303352432,
    13858784238779817838,
    4636981999271240277,
    4635032184925139369,
    13852643624115138617,
    13853996629010314452,
    4640824593902926017,
    4631086450597377226,
    4631207565618905072,
    13850039135525424306,
    4636981999271240277,
    4629910792698010033,
    13862479972854119972,
    13853996629010314452,
    13858185510051115478,
    4623107841001576756,
    4631207565618905072,
    13858784238779817838,
    13860755208150374356,
    4629910792698010033,
    13852643624115138617,
    13848950589920396328,
    13858185510051115478,
    4631086450597377226,
    4633009051303352432,
    13858784238779817838,
    4636981999271240277,
    4635032184925139369,
    13852643624115138617,
    13853996629010314452,
    4640824593902926017,
    4631086450597377226,
    4631207565618905072,
    13850039135525424306,
    4636981999271240277,
    4629910792698010033,
    13862479972854119972,
    13853996629010314452,
    13858185510051115478,
    4623107841001576756,
    4631207565618905072,
    13858784238779817838,
    13860755208150374356,
    4629910792698010033,
    13852643624115138617,
    13848950589920396328,
    13858185510051115478,
    4631086450597377226,
    4633009051303352432,
    13858784238779817838,
    4636981999271240277,
    4635032184925139369,
    13852643624115138617,
    13853996629010314452,
    4640824593902926017,
    4631086450597377226,
    4631207565618905072,
    13850039135525424306,
    4636981999271240277,
    4629910792698010033,
    13862479972854119972,
    13853996629010314452,
    13858185510051115478,
    4623107841001576756,
    4631207565618905072,
    13858784238779817838,
    13860755208150374356,
    4629910792698010033,
    13852643624115138617,
    13848950589920396328,
    13858185510051115478,
    4631086450597377226,
    4633009051303352432,
    13858784238779817838,
    4636981999271240277,
    4635032184925139369,
    13852643624115138617,
    13853996629010314452,
    4640824593902926017,
    4631086450597377226,
    4631207565618905072,
    13850039135525424306,
    4636981999271240277,
    4629910792698010033,
    13862479972854119972,
    13853996629010314452,
    13858185510051115478,
    4623107841001576756,
    4631207565618905072,
    13858784238779817838,
    13860755208150374356,
    4629910792698010033,
    13852643624115138617,
    13848950589920396328,
    13858185510051115478,
    4631086450597377226,
    4633009051303352432,
    13858784238779817838,
];

const MAIN_COMPLEX_BITS: [(u64, u64); 99] = [
    (4641113592167149344, 13865086993186608156),
    (13843510688181369520, 4637753344921789506),
    (4639470509233903968, 13866804956504574379),
    (4643520900030442357, 4637707234455405478),
    (4643239666298286721, 13857493871120419205),
    (4629888036297876524, 13860939418012506003),
    (4644078982611177683, 13867052061948871225),
    (4634982384133211925, 4643555279731381474),
    (4641113592167149344, 13865086993186608156),
    (4642704989143085350, 4631166018957484976),
    (13858802352673633052, 4636175131941716444),
    (4643520900030442357, 4637707234455405478),
    (4643054283333207301, 13849319722898501370),
    (13867318593452134096, 4627033835716666750),
    (4644078982611177683, 13867052061948871225),
    (13850035773272135410, 13868214022605065456),
    (4644274106953417733, 13864423841254875266),
    (4642704989143085350, 4631166018957484976),
    (4639470509233903968, 13866804956504574379),
    (13865596937213931022, 4641441405323929180),
    (4643054283333207301, 13849319722898501370),
    (4629888036297876524, 13860939418012506003),
    (13862463109508006015, 4644140979169556769),
    (13850035773272135410, 13868214022605065456),
    (4641113592167149344, 13865086993186608156),
    (13843510688181369520, 4637753344921789506),
    (4639470509233903968, 13866804956504574379),
    (4643520900030442357, 4637707234455405478),
    (4643239666298286721, 13857493871120419205),
    (4629888036297876524, 13860939418012506003),
    (4644078982611177683, 13867052061948871225),
    (4634982384133211925, 4643555279731381474),
    (4641113592167149344, 13865086993186608156),
    (4642704989143085350, 4631166018957484976),
    (13858802352673633052, 4636175131941716444),
    (4643520900030442357, 4637707234455405478),
    (4643054283333207301, 13849319722898501370),
    (13867318593452134096, 4627033835716666750),
    (4644078982611177683, 13867052061948871225),
    (13850035773272135410, 13868214022605065456),
    (4644274106953417733, 13864423841254875266),
    (4642704989143085350, 4631166018957484976),
    (4639470509233903968, 13866804956504574379),
    (13865596937213931022, 4641441405323929180),
    (4643054283333207301, 13849319722898501370),
    (4629888036297876524, 13860939418012506003),
    (13862463109508006015, 4644140979169556769),
    (13850035773272135410, 13868214022605065456),
    (4641113592167149344, 13865086993186608156),
    (13843510688181369520, 4637753344921789506),
    (4639470509233903968, 13866804956504574379),
    (4643520900030442357, 4637707234455405478),
    (4643239666298286721, 13857493871120419205),
    (4629888036297876524, 13860939418012506003),
    (4644078982611177683, 13867052061948871225),
    (4634982384133211925, 4643555279731381474),
    (4641113592167149344, 13865086993186608156),
    (4642704989143085350, 4631166018957484976),
    (13858802352673633052, 4636175131941716444),
    (4643520900030442357, 4637707234455405478),
    (4643054283333207301, 13849319722898501370),
    (13867318593452134096, 4627033835716666750),
    (4644078982611177683, 13867052061948871225),
    (13850035773272135410, 13868214022605065456),
    (4644274106953417733, 13864423841254875266),
    (4642704989143085350, 4631166018957484976),
    (4639470509233903968, 13866804956504574379),
    (13865596937213931022, 4641441405323929180),
    (4643054283333207301, 13849319722898501370),
    (4629888036297876524, 13860939418012506003),
    (13862463109508006015, 4644140979169556769),
    (13850035773272135410, 13868214022605065456),
    (4641113592167149344, 13865086993186608156),
    (13843510688181369520, 4637753344921789506),
    (4639470509233903968, 13866804956504574379),
    (4643520900030442357, 4637707234455405478),
    (4643239666298286721, 13857493871120419205),
    (4629888036297876524, 13860939418012506003),
    (4644078982611177683, 13867052061948871225),
    (4634982384133211925, 4643555279731381474),
    (4641113592167149344, 13865086993186608156),
    (4642704989143085350, 4631166018957484976),
    (13858802352673633052, 4636175131941716444),
    (4643520900030442357, 4637707234455405478),
    (4643054283333207301, 13849319722898501370),
    (13867318593452134096, 4627033835716666750),
    (4644078982611177683, 13867052061948871225),
    (13850035773272135410, 13868214022605065456),
    (4644274106953417733, 13864423841254875266),
    (4642704989143085350, 4631166018957484976),
    (4639470509233903968, 13866804956504574379),
    (13865596937213931022, 4641441405323929180),
    (4643054283333207301, 13849319722898501370),
    (4629888036297876524, 13860939418012506003),
    (13862463109508006015, 4644140979169556769),
    (13850035773272135410, 13868214022605065456),
    (4641113592167149344, 13865086993186608156),
    (13843510688181369520, 4637753344921789506),
    (4639470509233903968, 13866804956504574379),
];

fn real_bits(max_batch_points: Option<usize>, parallelism: Option<usize>) -> Vec<u64> {
    let (sites, tree) = chain_tree(real);
    let (data, shape) = points(&sites);
    let pts = ColMajorArrayRef::new(&data, &shape).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &sites, options(max_batch_points)).unwrap();
    let hint = EvaluationHint::around(1);
    let values = with_parallelism(parallelism, |p| {
        evaluator
            .evaluate_batched_typed_with_parallelism::<f64>(pts, hint, p)
            .unwrap()
    });
    values.iter().map(|x| x.to_bits()).collect()
}

fn complex_bits(max_batch_points: Option<usize>, parallelism: Option<usize>) -> Vec<(u64, u64)> {
    let (sites, tree) = chain_tree(complex);
    let (data, shape) = points(&sites);
    let pts = ColMajorArrayRef::new(&data, &shape).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &sites, options(max_batch_points)).unwrap();
    let hint = EvaluationHint::around(1);
    let values = with_parallelism(parallelism, |p| {
        evaluator
            .evaluate_batched_typed_with_parallelism::<Complex64>(pts, hint, p)
            .unwrap()
    });
    values
        .iter()
        .map(|x| (x.re.to_bits(), x.im.to_bits()))
        .collect()
}

/// `None` is the sequential value; `Some(budget)` is a pool of that many
/// threads handed in with that budget.
fn with_parallelism<R>(budget: Option<usize>, run: impl FnOnce(Parallelism<'_>) -> R) -> R {
    match budget {
        None => run(Parallelism::Serial),
        Some(budget) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(budget)
                .build()
                .unwrap();
            run(Parallelism::Pool {
                pool: &pool,
                budget,
            })
        }
    }
}

#[test]
fn chunk_one_serial_is_bit_identical_to_main() {
    assert_eq!(real_bits(Some(1), None), MAIN_REAL_BITS);
    assert_eq!(complex_bits(Some(1), None), MAIN_COMPLEX_BITS);
}

#[test]
fn chunk_one_pool_is_bit_identical_to_main() {
    for budget in [1, 2, 4] {
        assert_eq!(
            real_bits(Some(1), Some(budget)),
            MAIN_REAL_BITS,
            "budget {budget}"
        );
        assert_eq!(
            complex_bits(Some(1), Some(budget)),
            MAIN_COMPLEX_BITS,
            "budget {budget}"
        );
    }
}

#[test]
fn pool_matches_serial_bit_for_bit_across_budgets_and_chunks() {
    for chunk in [None, Some(7), Some(99)] {
        let serial_real = real_bits(chunk, None);
        let serial_complex = complex_bits(chunk, None);
        for budget in [1, 2, 3, 4, 8] {
            assert_eq!(
                real_bits(chunk, Some(budget)),
                serial_real,
                "chunk {chunk:?} budget {budget}"
            );
            assert_eq!(
                complex_bits(chunk, Some(budget)),
                serial_complex,
                "chunk {chunk:?} budget {budget}"
            );
        }
    }
}
