//! The explicit-only build (the default features, without `global-defaults`):
//! context-free entries refuse with a typed error instead of choosing an engine,
//! and values created in a caller's context keep working on that context's
//! engine (tensor4all-rs#1403).

use std::error::Error;

use rand::SeedableRng;
use std::sync::Arc;

use tensor4all_aci::{
    elementwise_batched, elementwise_batched_with_rng_in, AciError, AciOptions, ElementwiseBatch,
};
use tensor4all_core::{legacy, DynIndex, ExecutionContext, IdxTensor, IdxTensorError};
use tensor4all_itensorlike::{TensorTrain, TensorTrainError};
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
use tensor4all_tensorbackend::{
    with_runtime_engine, CpuBackend, CpuExecutionContext, ExecutionContextEntryError, Storage,
};

fn cpu_context(threads: usize) -> (Arc<CpuExecutionContext>, ExecutionContext) {
    let cpu = Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(threads).expect("a CPU backend"),
    ));
    (Arc::clone(&cpu), ExecutionContext::Cpu(cpu))
}

/// Whether `NoContext` appears anywhere in `error`'s source chain.
fn refused_for_no_context(error: &(dyn Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if matches!(
            error.downcast_ref::<ExecutionContextEntryError>(),
            Some(ExecutionContextEntryError::NoContext)
        ) {
            return true;
        }
        current = error.source();
    }
    false
}

/// A two-site real chain `a[s0, link] b[link, s1]` built in `context`, and its
/// norm² computed by hand.
fn chain(context: &ExecutionContext) -> (TensorTrain, f64) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);
    let link = DynIndex::new_dyn(2);
    // Column-major: a[s0, link], b[link, s1].
    let a_data = vec![1.0_f64, -2.0, 0.5, 3.0];
    let b_data = vec![0.25_f64, 1.5, -1.0, 2.0, 0.75, -0.5];
    let a = IdxTensor::from_dense_in(context, vec![s0, link.clone()], a_data.clone()).unwrap();
    let b = IdxTensor::from_dense_in(context, vec![link, s1], b_data.clone()).unwrap();

    // <psi|psi> = sum_{l,l'} (A^T A)[l,l'] (B B^T)[l,l'] for real data.
    let a_at = |s: usize, l: usize| a_data[s + 2 * l];
    let b_at = |l: usize, s: usize| b_data[l + 2 * s];
    let mut expected = 0.0;
    for l in 0..2 {
        for lp in 0..2 {
            let ata: f64 = (0..2).map(|s| a_at(s, l) * a_at(s, lp)).sum();
            let bbt: f64 = (0..3).map(|s| b_at(l, s) * b_at(lp, s)).sum();
            expected += ata * bbt;
        }
    }
    (TensorTrain::new(vec![a, b]).unwrap(), expected)
}

#[test]
fn this_package_builds_core_without_the_global_route() {
    // Compile-time fact of the `tensor4all-core` this package links: if a
    // dependency edge enabled `global-defaults`, every other test here would be
    // testing the wrong configuration.
    assert!(!legacy::GLOBAL_DEFAULTS);
}

#[test]
fn core_legacy_bridges_refuse_instead_of_choosing_an_engine() {
    assert!(matches!(
        legacy::eager_runtime(),
        Err(ExecutionContextEntryError::NoContext)
    ));
    assert!(matches!(
        legacy::default_cpu_execution_context(),
        Err(ExecutionContextEntryError::NoContext)
    ));
}

#[test]
fn backend_runtime_without_a_live_owner_refuses() {
    let (cpu, _context) = cpu_context(1);
    let runtime = cpu.eager_runtime().unwrap();
    drop(_context);
    drop(cpu);
    let mut ran = false;
    // The strict entry never chooses an engine (#1378) ...
    let entry = with_runtime_engine(&runtime, |_engine| ran = true);
    assert!(matches!(
        entry,
        Err(ExecutionContextEntryError::NoLiveOwner)
    ));
    // ... and the legacy bridge has no process-global context to fall back to.
    let bridged = tensor4all_tensorbackend::with_runtime_engine_or_global(&runtime, |_engine| {
        ran = true;
    });
    assert!(matches!(
        bridged,
        Err(ExecutionContextEntryError::NoContext)
    ));
    assert!(!ran, "no engine may be chosen for an owner-less runtime");
}

#[test]
fn owning_cpu_engine_is_the_creating_context() {
    let (_cpu, context) = cpu_context(2);
    let index = DynIndex::new_dyn(2);
    let owned = IdxTensor::from_dense_in(&context, vec![index], vec![1.0_f64, 2.0]).unwrap();
    assert_eq!(
        owned
            .with_owning_cpu_engine(|engine| engine.num_threads())
            .unwrap(),
        Some(2)
    );
}

#[test]
fn context_owned_chain_norm_runs_on_its_context() {
    let (_cpu, context) = cpu_context(2);
    let (tt, expected) = chain(&context);
    let actual = tt.norm_squared().unwrap();
    assert!(
        (actual - expected).abs() <= 1.0e-12 * expected,
        "norm² {actual:.16e}, expected {expected:.16e}"
    );
    assert!((tt.norm().unwrap() - expected.sqrt()).abs() <= 1.0e-12 * expected.sqrt());
}

#[test]
fn chain_whose_context_was_dropped_refuses_instead_of_a_host_fallback() {
    let (cpu, context) = cpu_context(1);
    let (tt, _expected) = chain(&context);
    drop(context);
    drop(cpu);
    let error = tt
        .norm_squared()
        .expect_err("no live context owns the chain");
    assert!(
        matches!(error, TensorTrainError::IdxTensor { .. }),
        "got {error:?}"
    );
    assert!(
        refused_for_no_context(&error),
        "expected ExecutionContextEntryError::NoContext in the chain, got {error:?}"
    );
}

#[test]
fn context_free_construction_and_contraction_refuse() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let error = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
        .expect_err("no runtime to adopt host data into");
    assert!(
        matches!(
            error,
            IdxTensorError::EngineEntry {
                source: ExecutionContextEntryError::NoContext
            }
        ),
        "got {error:?}"
    );

    // Values from two different contexts have no common engine; a
    // context-free contraction of a value whose context is gone refuses.
    let (cpu, context) = cpu_context(1);
    let a = IdxTensor::from_dense_in(&context, vec![i, j.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap();
    let b = IdxTensor::from_dense_in(&context, vec![j], vec![1.0_f64, -1.0]).unwrap();
    let product = tensor4all_core::contract(&[&a, &b]).unwrap();
    assert_eq!(product.to_vec::<f64>().unwrap(), vec![-2.0, -2.0]);
    drop(context);
    drop(cpu);
    let error = tensor4all_core::contract(&[&a, &b]).expect_err("the owning context is gone");
    assert!(refused_for_no_context(&error), "got {error:?}");
}

#[test]
fn host_storage_chain_norm_refuses_at_the_read_step() {
    // Host-only storage has no eager value and no owner. Reading it as a dense
    // vector already needs the implicit route, so the chain norm refuses before
    // any norm kernel is chosen: there is no host computation to fall back to.
    let s0 = DynIndex::new_dyn(2);
    let link = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);
    let host = |indices: Vec<DynIndex>, data: Vec<f64>, dims: &[usize]| {
        IdxTensor::from_storage(
            indices,
            Arc::new(Storage::from_dense_col_major(data, dims).unwrap()),
        )
        .unwrap()
    };
    let a = host(vec![s0, link.clone()], vec![1.0, -2.0, 0.5, 3.0], &[2, 2]);
    let b = host(
        vec![link, s1],
        vec![0.25, 1.5, -1.0, 2.0, 0.75, -0.5],
        &[2, 3],
    );
    let no_context = |error: &IdxTensorError| {
        matches!(
            error,
            IdxTensorError::EngineEntry {
                source: ExecutionContextEntryError::NoContext
            }
        )
    };
    let read = a
        .to_vec::<f64>()
        .expect_err("reading host storage needs a runtime");
    assert!(no_context(&read), "got {read:?}");

    let tt = TensorTrain::new(vec![a, b]).unwrap();
    match tt.norm_squared() {
        Err(TensorTrainError::IdxTensor { source }) if no_context(&source) => {}
        other => panic!("expected the typed NoContext refusal, got {other:?}"),
    }
}

/// The elementwise product of two ACI inputs, one value per point.
fn aci_product(batch: ElementwiseBatch<'_, f64>, output: &mut [f64]) -> tensor4all_aci::Result<()> {
    for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
        *value = batch.get(0, point)? * batch.get(1, point)?;
    }
    Ok(())
}

/// #1382: the context-free ACI entry needs the process-global context only for a run that sweeps.
/// Validation errors keep their own variants and the single-site case (no products at all) still
/// succeeds; a multi-site sweep refuses with `AciError::Backend` whose source is `NoContext`.
#[test]
fn aci_context_free_entry_refuses_only_runs_that_sweep() {
    let options = AciOptions::<f64>::default();

    let empty = elementwise_batched(aci_product, &[], &options).unwrap_err();
    assert!(matches!(empty, AciError::EmptyInputs), "got {empty:?}");

    let one_site = [
        SimpleTensorTrain::<f64>::constant(&[3], 2.0),
        SimpleTensorTrain::<f64>::constant(&[3], -1.5),
    ];
    let result = elementwise_batched(aci_product, &one_site, &options).unwrap();
    for point in 0..3 {
        assert_eq!(result.tensor_train.evaluate(&[point]).unwrap(), -3.0);
    }

    let three_sites = [
        SimpleTensorTrain::<f64>::constant(&[2, 2, 2], 2.0),
        SimpleTensorTrain::<f64>::constant(&[2, 2, 2], 3.0),
    ];
    let error = elementwise_batched(aci_product, &three_sites, &options).unwrap_err();
    assert!(matches!(error, AciError::Backend { .. }), "got {error:?}");
    assert!(refused_for_no_context(&error), "got {error:?}");
}

/// #1382: with a caller context the ACI sweep runs in this build, where any step that fell back to
/// the process-global route would have been refused.
#[test]
fn aci_context_entry_runs_without_the_global_route() {
    let (cpu, _) = cpu_context(1);
    let inputs = [
        SimpleTensorTrain::<f64>::constant(&[2, 2, 2], 2.0),
        SimpleTensorTrain::<f64>::constant(&[2, 2, 2], 3.0),
    ];
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let result = elementwise_batched_with_rng_in(
        aci_product,
        &inputs,
        &AciOptions::default(),
        &mut rng,
        &cpu,
    )
    .unwrap();
    for index in [[0, 0, 0], [1, 0, 1], [1, 1, 1]] {
        let value = result.tensor_train.evaluate(&index).unwrap();
        assert!((value - 6.0).abs() < 1e-12, "{index:?}: {value}");
    }
}
