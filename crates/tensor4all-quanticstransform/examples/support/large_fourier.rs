//! Shared large-grid Gaussian validation for the example and heavy tests.
//! No full-grid materialization: work scales with TT ranks and a fixed sample set.
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{collections::HashSet, time::Instant};
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, SvdTruncationPolicy};
use tensor4all_interpolativeqtt::{interpolate_single_scale, InterpolativeQttOptions};
use tensor4all_partitionedtreetn::{
    reconstruction::{
        schedule_merge_refine, CoordinateGroup, ReconstructionTarget, SubsetOperatorOptions,
    },
    PartitionedTreeTN, Projector, SubDomainTreeTN,
};
use tensor4all_quanticstransform::{
    patched_fourier_transform, quantics_fourier_operator_multiaxis, AxisBit, FourierOptions,
    PatchedFourierOptions,
};
use tensor4all_simplett::SimpleTensorTrain;
use tensor4all_treetn::{
    tensor_train_to_treetn_with_names_and_site_indices, CachedEvaluatorOptions, EvaluationHint,
    RestructureOptions, SiteIndexNetwork, SwapOptions, TreeTN, TreeTNCachedEvaluator,
    TruncationOptions,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MODES: i32 = 12;
const SIGMA: f64 = 0.12;
const CONSTRUCTION_ALLOWANCE: f64 = 2e-7;

fn coefficient(mode: i32) -> f64 {
    (-2.0 * std::f64::consts::PI.powi(2) * SIGMA.powi(2) * (mode as f64).powi(2)).exp()
}
fn gaussian(x: f64, shift: f64) -> f64 {
    1.0 + 2.0
        * (1..=MODES)
            .map(|m| coefficient(m) * (2.0 * std::f64::consts::PI * m as f64 * (x - shift)).cos())
            .sum::<f64>()
}
fn spectrum(frequency: usize, n: usize, shift: f64) -> Complex64 {
    let signed = if frequency <= n / 2 {
        frequency as i64
    } else {
        frequency as i64 - n as i64
    };
    if signed.unsigned_abs() > MODES as u64 {
        return Complex64::new(0.0, 0.0);
    }
    Complex64::from_polar(
        coefficient(signed as i32),
        -2.0 * std::f64::consts::PI * signed as f64 * shift,
    )
}
fn policy(cap: usize) -> RestructureOptions {
    RestructureOptions::new().with_swap(SwapOptions {
        max_bond_dim: Some(cap),
        svd_policy: Some(SvdTruncationPolicy::new(1e-13)),
    })
}
fn sample(
    tree: &TreeTN<IdxTensor, usize>,
    indices: &[DynIndex],
    points: &[usize],
) -> Result<Vec<Complex64>> {
    let shape = [indices.len(), points.len() / indices.len()];
    let mut evaluator = TreeTNCachedEvaluator::new(
        tree,
        indices,
        CachedEvaluatorOptions {
            center: Some(indices.len() / 2),
            ..Default::default()
        },
    )?;
    Ok(evaluator.evaluate_batched_typed::<Complex64>(
        ColMajorArrayRef::new(points, &shape)?,
        EvaluationHint::default(),
    )?)
}

/// Which schedules a case runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Runs {
    /// `schedule_merge_refine` directly, timed on its own, then the front end.
    ScheduleAndFrontEnd,
    /// Only the front end, which also produces the sampled spectrum. Each
    /// schedule of the two-axis interleaved R = 10 case takes about 6 minutes
    /// with one thread (#1279, #1285); the example's `frontend-only` argument
    /// selects this mode to halve that.
    // The smoke-test and heavy targets include this module but never run this mode.
    #[allow(dead_code)]
    FrontEndOnly,
}

#[derive(Debug)]
pub(crate) struct Measurement {
    pub(crate) bits: usize,
    pub(crate) dimensions: usize,
    pub(crate) interleaved: bool,
    pub(crate) input_seconds: f64,
    pub(crate) operator_seconds: f64,
    /// `None` with [`Runs::FrontEndOnly`].
    pub(crate) schedule_seconds: Option<f64>,
    pub(crate) frontend_seconds: f64,
    pub(crate) operator_bond: usize,
    pub(crate) input_bond: usize,
    pub(crate) samples: usize,
    pub(crate) sample_error: f64,
    pub(crate) reconstruction_bound: f64,
    pub(crate) compressions: usize,
    pub(crate) output_regions: usize,
}

impl std::fmt::Display for Measurement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let schedule = self
            .schedule_seconds
            .map_or_else(|| "skipped".to_string(), |seconds| format!("{seconds:.6}"));
        write!(f, "bits={},dimensions={},interleaved={},input_seconds={:.6},operator_seconds={:.6},schedule_seconds={},frontend_seconds={:.6},operator_bond={},input_bond={},samples={},sample_error={:.6e},reconstruction_bound={:.6e},compressions={},output_regions={}",
            self.bits,self.dimensions,self.interleaved,self.input_seconds,self.operator_seconds,
            schedule,self.frontend_seconds,self.operator_bond,self.input_bond,self.samples,self.sample_error,
            self.reconstruction_bound,self.compressions,self.output_regions)
    }
}

/// A shifted periodic Gaussian defined by a finite Fourier series. Because
/// MODES < N/2, its normalized discrete DFT is exactly the listed coefficients;
/// there is no continuous-FT/quadrature/aliasing approximation in the oracle.
pub(crate) fn run_case(
    bits: usize,
    dimensions: usize,
    interleaved: bool,
    runs: Runs,
) -> Result<Measurement> {
    assert!((10..=20).contains(&bits));
    assert!((1..=2).contains(&dimensions));
    let start = Instant::now();
    let n = 1usize << bits;
    let shifts = [0.375, 0.3125];
    let axes: Vec<Vec<DynIndex>> = (0..dimensions)
        .map(|_| (0..bits).map(|_| DynIndex::new_dyn(2)).collect())
        .collect();
    let mut cores = Vec::new();
    for &shift in shifts.iter().take(dimensions) {
        let tt = interpolate_single_scale(
            |x| gaussian(x, shift) / (n as f64).sqrt(),
            0.0,
            1.0,
            bits,
            48,
            &InterpolativeQttOptions::default(),
        )?;
        cores.extend(tt.into_site_tensors());
    }
    // Concatenate independent rank-one-ended TTs: this is their tensor product.
    let tt = SimpleTensorTrain::new(cores)?;
    let grouped_indices: Vec<_> = axes.iter().flatten().cloned().collect();
    let mut tree = tensor_train_to_treetn_with_names_and_site_indices(
        &tt,
        (0..bits * dimensions).collect(),
        grouped_indices,
    )?;
    let layout: Vec<_> = if interleaved {
        (0..bits)
            .flat_map(|bit| (0..dimensions).map(move |axis| AxisBit { axis, bit }))
            .collect()
    } else {
        (0..dimensions)
            .flat_map(|axis| (0..bits).map(move |bit| AxisBit { axis, bit }))
            .collect()
    };
    let physical: Vec<_> = layout.iter().map(|b| axes[b.axis][b.bit].clone()).collect();
    if interleaved && dimensions > 1 {
        let mut target = SiteIndexNetwork::new();
        for (j, index) in physical.iter().enumerate() {
            target.add_node(j, HashSet::from([index.clone()]))?;
            if j > 0 {
                target.add_edge(&(j - 1), &j)?;
            }
        }
        tree = tree.restructure_to(&target, &policy(64))?;
    }
    let input_bond = tree.link_dims().into_iter().max().unwrap_or(1);
    let mut patches = Vec::new();
    for prefix in 0..1usize << dimensions {
        let projector = Projector::from_pairs(
            axes.iter()
                .enumerate()
                .map(|(axis, indices)| (indices[0].clone(), (prefix >> axis) & 1)),
        )?;
        let mut patch = SubDomainTreeTN::new(tree.clone(), projector)?;
        // Masking alone retains the global train's redundant bond spaces.
        // Real patch inputs are compressed local TTs (as in #1276).
        patch.truncate(
            &0,
            TruncationOptions::new().with_svd_policy(SvdTruncationPolicy::new(1e-13)),
        )?;
        patches.push(patch);
    }
    let state = PartitionedTreeTN::from_subdomains(patches)?;
    let input_bond = state
        .values()
        .map(|p| p.max_bond_dim())
        .max()
        .unwrap_or(input_bond);
    let input_seconds = start.elapsed().as_secs_f64();
    eprintln!("input ready: R={bits}, dimensions={dimensions}, interleaved={interleaved}, bond={input_bond}");
    let mut random = ChaCha8Rng::seed_from_u64(0x1277_1a26);
    let mut frequencies = vec![
        vec![0; dimensions],
        vec![1; dimensions],
        vec![n - 1; dimensions],
    ];
    for _ in 0..16 {
        frequencies.push(
            (0..dimensions)
                .map(|_| {
                    let mode = random.random_range(-5i32..=5);
                    if mode < 0 {
                        (n as i64 + mode as i64) as usize
                    } else {
                        mode as usize
                    }
                })
                .collect(),
        );
        frequencies.push((0..dimensions).map(|_| random.random_range(0..n)).collect());
    }
    let output_points: Vec<usize> = frequencies
        .iter()
        .flat_map(|frequency| layout.iter().map(move |b| (frequency[b.axis] >> b.bit) & 1))
        .collect();
    // Independently verify interpolation samples on the original MSB-first grid.
    let input_points: Vec<usize> = frequencies
        .iter()
        .flat_map(|frequency| {
            layout
                .iter()
                .map(move |b| (frequency[b.axis] >> (bits - 1 - b.bit)) & 1)
        })
        .collect();
    let mut input_values = vec![Complex64::new(0.0, 0.0); frequencies.len()];
    for patch in state.values() {
        for (total, value) in
            input_values
                .iter_mut()
                .zip(sample(patch.data(), &physical, &input_points)?)
        {
            *total += value;
        }
    }
    for (frequency, value) in frequencies.iter().zip(input_values) {
        let expected = frequency
            .iter()
            .enumerate()
            .map(|(axis, &x)| gaussian(x as f64 / n as f64, shifts[axis]) / (n as f64).sqrt())
            .product::<f64>();
        assert!(
            (value - expected).norm() * (n as f64).powf(dimensions as f64 / 2.0) < 5e-9,
            "interpolated Gaussian sample {value} differs from {expected}"
        );
    }
    let operator_start = Instant::now();
    let operator = quantics_fourier_operator_multiaxis(&layout, FourierOptions::forward())?;
    let operator_seconds = operator_start.elapsed().as_secs_f64();
    let operator_bond = operator.mpo().link_dims().into_iter().max().unwrap_or(1);
    eprintln!("operator ready: bond={operator_bond}, seconds={operator_seconds}");
    let mut options = PatchedFourierOptions::default();
    options.tolerance.rtol = 1e-6;
    options.merge_refine.target_bond_dim = Some(8);
    options.merge_refine.output_depth = Some(1);
    options.merge_refine.max_terms = 256;
    // Time the scheduler separately from the operator and target preparation.
    // Then exercise the public Fourier front end with the same policy.
    let schedule_seconds = if runs == Runs::ScheduleAndFrontEnd {
        let preimage = ReconstructionTarget::from_partition(&state)?;
        let mut schedule = options.merge_refine.clone();
        schedule.coordinate_groups = Some(axes.iter().cloned().map(CoordinateGroup::new).collect());
        let schedule_start = Instant::now();
        let scheduled = schedule_merge_refine(
            &preimage,
            &0,
            &operator,
            &physical,
            &SubsetOperatorOptions { unitary: true },
            options.tolerance,
            &schedule,
        )?;
        let schedule_seconds = schedule_start.elapsed().as_secs_f64();
        assert!(scheduled.report().error_bound <= scheduled.report().absolute_tolerance);
        eprintln!("scheduler finished in {schedule_seconds} seconds");
        Some(schedule_seconds)
    } else {
        None
    };
    drop(operator);
    let frontend_start = Instant::now();
    let result = patched_fourier_transform(&state, &0, &axes, FourierOptions::forward(), &options)?;
    let frontend_seconds = frontend_start.elapsed().as_secs_f64();
    eprintln!("front end finished in {frontend_seconds} seconds");
    let report = result.report();
    assert_eq!(report.applied_operator_count, 1usize << dimensions);
    assert!(report.error_bound <= report.absolute_tolerance);
    assert!(report.compression_attempts > 0);
    let mut actual = vec![Complex64::new(0.0, 0.0); frequencies.len()];
    for (_, terms) in result.regions() {
        for term in terms {
            for (total, value) in
                actual
                    .iter_mut()
                    .zip(sample(term.data(), &physical, &output_points)?)
            {
                *total += value;
            }
        }
    }
    let mut sample_error = 0.0_f64;
    for (frequency, value) in frequencies.iter().zip(actual) {
        let expected = frequency
            .iter()
            .enumerate()
            .map(|(axis, &r)| spectrum(r, n, shifts[axis]))
            .product::<Complex64>();
        sample_error = sample_error.max((value - expected).norm());
    }
    assert!(sample_error<=report.error_bound+CONSTRUCTION_ALLOWANCE,
        "R={bits}, dims={dimensions}, interleaved={interleaved}: sample error {sample_error:e}, reconstruction bound {:e}, separate construction/roundoff allowance {CONSTRUCTION_ALLOWANCE:e}",report.error_bound);
    let measurement = Measurement {
        bits,
        dimensions,
        interleaved,
        input_seconds,
        operator_seconds,
        schedule_seconds,
        frontend_seconds,
        operator_bond,
        input_bond,
        samples: frequencies.len(),
        sample_error,
        reconstruction_bound: report.error_bound,
        compressions: report.compressions,
        output_regions: result.regions().count(),
    };
    eprintln!("{measurement}");
    Ok(measurement)
}
