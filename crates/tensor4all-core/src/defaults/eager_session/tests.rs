use std::error::Error;

use tensor4all_tensorbackend::CpuExecutionContextError;

use crate::IdxTensorError;

#[test]
fn context_error_conversion_keeps_the_typed_variant() {
    let error = IdxTensorError::from(CpuExecutionContextError::Reentered);
    assert!(matches!(
        error,
        IdxTensorError::EngineEntry {
            source: tensor4all_tensorbackend::ExecutionContextEntryError::Cpu(
                CpuExecutionContextError::Reentered
            )
        }
    ));
    assert!(error.source().is_some());
}

// ---------------------------------------------------------------------------
// Stage A2 (#1393): one engine-free `Runtime` shared by several `&mut` engines.
// ---------------------------------------------------------------------------

mod shared_runtime {
    use std::sync::atomic::{AtomicBool, Ordering};

    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use tenferro_ad::EagerTensor;
    use tenferro_einsum::{EagerSessionEinsumExt, EinsumSubscripts};
    use tensor4all_tensorbackend::{CpuBackend, CpuEngine, Runtime, RuntimeError};

    use super::super::with_eager_on;
    use crate::index_like::IndexLike;
    use crate::{DynIndex, IdxTensor, IdxTensorError};

    fn pooled(threads: usize) -> CpuEngine {
        CpuBackend::with_threads(threads).unwrap().into_engine()
    }

    fn dense(rt: &Runtime, rows: &DynIndex, cols: &DynIndex, offset: f64) -> IdxTensor {
        let data = (0..rows.dim() * cols.dim())
            .map(|k| ((k as f64 + offset) * 0.29).cos())
            .collect::<Vec<_>>();
        IdxTensor::from_dense_on(rt, vec![rows.clone(), cols.clone()], data).unwrap()
    }

    /// Host data that belongs to no runtime.
    fn host(i: &DynIndex, j: &DynIndex) -> IdxTensor {
        let storage = tensor4all_tensorbackend::Storage::from_dense_col_major(
            vec![1.0_f64, 2.0, 3.0, 4.0],
            &[i.dim(), j.dim()],
        )
        .unwrap();
        IdxTensor::from_storage(vec![i.clone(), j.clone()], std::sync::Arc::new(storage)).unwrap()
    }

    fn eager(tensor: &IdxTensor) -> &EagerTensor {
        tensor
            .eager_value()
            .expect("a runtime value has an eager value")
    }

    /// The input doubled on `exec`: an engine result, not a host constant.
    fn computed_on(rt: &Runtime, exec: &mut CpuEngine, input: &IdxTensor) -> IdxTensor {
        let value = eager(input);
        let doubled = with_eager_on("double", rt, &[input], exec, |session| {
            session.add(value, value)
        })
        .unwrap();
        IdxTensor::from_inner(input.indices().to_vec(), doubled).unwrap()
    }

    fn contract_on(rt: &Runtime, exec: &mut CpuEngine, a: &IdxTensor, b: &IdxTensor) -> Vec<u64> {
        let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
        let product = with_eager_on("contract", rt, &[a, b], exec, |session| {
            session.einsum_subscripts(&[eager(a), eager(b)], &subscripts)
        })
        .unwrap();
        assert_eq!(product.ctx_id(), rt.id(), "the result joins the runtime");
        product
            .value()
            .unwrap()
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    /// `CpuEngine::serial()` and a 4-thread engine share one runtime. A value
    /// made on each contracts on either engine, and the result equals, bit for
    /// bit, the same contraction of values made on that engine alone.
    #[test]
    fn values_from_two_engines_contract_on_either_engine_bitwise() {
        let rt = Runtime::new().unwrap();
        let (i, j, k) = (
            DynIndex::new_dyn(40),
            DynIndex::new_dyn(33),
            DynIndex::new_dyn(28),
        );
        let a_input = dense(&rt, &i, &j, 0.0);
        let b_input = dense(&rt, &j, &k, 5.0);
        let mut serial = CpuEngine::serial();
        let mut pool = pooled(4);

        let a_serial = computed_on(&rt, &mut serial, &a_input);
        let b_pooled = computed_on(&rt, &mut pool, &b_input);

        for (name, exec) in [("serial", &mut serial), ("pooled", &mut pool)] {
            let mixed = contract_on(&rt, exec, &a_serial, &b_pooled);
            let a_here = computed_on(&rt, exec, &a_input);
            let b_here = computed_on(&rt, exec, &b_input);
            let single = contract_on(&rt, exec, &a_here, &b_here);
            assert_eq!(mixed, single, "{name}: equal to the single-engine run");
        }
    }

    /// Operands of two runtimes are refused with `RuntimeMismatch` before the
    /// operation runs, so the RNG it would draw from is not advanced.
    #[test]
    fn values_from_different_runtimes_are_refused_before_any_rng_draw() {
        let rt = Runtime::new().unwrap();
        let other = Runtime::new().unwrap();
        let (i, j) = (DynIndex::new_dyn(3), DynIndex::new_dyn(2));
        let mine = dense(&rt, &i, &j, 0.0);
        let foreign = dense(&other, &i, &j, 1.0);

        let mut rng = ChaCha8Rng::seed_from_u64(1393);
        let mut untouched = rng.clone();
        let ran = AtomicBool::new(false);
        let error = with_eager_on(
            "random update",
            &rt,
            &[&mine, &foreign],
            &mut CpuEngine::serial(),
            |session| {
                ran.store(true, Ordering::SeqCst);
                let _ = rng.next_u64();
                session.add(eager(&mine), eager(&foreign))
            },
        )
        .expect_err("operands of two runtimes are refused");
        match error {
            IdxTensorError::RuntimeMismatch {
                operation,
                expected,
                found,
            } => {
                assert_eq!(operation, "random update");
                assert_eq!(expected, rt.id());
                assert_eq!(found, other.id());
            }
            other => panic!("expected RuntimeMismatch, got {other:?}"),
        }
        assert!(
            !ran.load(Ordering::SeqCst),
            "refused before the operation ran"
        );
        assert_eq!(
            rng.next_u64(),
            untouched.next_u64(),
            "the RNG did not advance"
        );
    }

    /// An engine the runtime has no modules for is refused at entry with the
    /// typed `UnsupportedEngine`, carried as the operation's source.
    #[test]
    fn an_engine_with_a_custom_runtime_id_is_refused_at_entry() {
        let rt = Runtime::new().unwrap();
        let (i, j) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
        let x = dense(&rt, &i, &j, 0.0);
        let custom = tenferro::EngineId::new("tensor4all-test.cpu.v1").unwrap();
        let mut exec = CpuEngine::serial().with_runtime_engine_id(custom);
        let error = with_eager_on("negate", &rt, &[&x], &mut exec, |session| {
            session.neg(eager(&x))
        })
        .expect_err("a custom-id engine is refused");
        let IdxTensorError::Operation { operation, source } = &error else {
            panic!("expected the typed refusal as the operation source, got {error:?}");
        };
        assert_eq!(*operation, "negate");
        assert!(matches!(
            source.downcast_ref::<RuntimeError>(),
            Some(RuntimeError::UnsupportedEngine { .. })
        ));
    }

    /// Host data with no eager value is readable by any CPU engine of the
    /// runtime: it belongs to no runtime, so it never causes a mismatch.
    #[test]
    fn host_operands_without_a_runtime_pass_validation_on_a_cpu_engine() {
        let rt = Runtime::new().unwrap();
        let (i, j) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
        let x = dense(&rt, &i, &j, 0.0);
        let host = host(&i, &j);
        assert!(host.eager_value().is_none());
        let doubled = with_eager_on("double", &rt, &[&x, &host], &mut pooled(2), |session| {
            session.add(eager(&x), eager(&x))
        })
        .unwrap();
        assert_eq!(doubled.ctx_id(), rt.id());
    }

    /// A CUDA engine cannot read a host value: `PlacementMismatch`, no upload.
    #[cfg(feature = "tenferro-cuda")]
    #[test]
    fn a_host_value_given_to_a_cuda_engine_is_a_placement_mismatch() {
        use tensor4all_tensorbackend::{CudaDeviceId, CudaEngine};

        let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0))
            .expect("CUDA ordinal 0 must be available");
        let rt = Runtime::for_cuda_engine(&mut gpu).unwrap();
        let (i, j) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
        let host_value = host(&i, &j);
        let ran = AtomicBool::new(false);
        let error = with_eager_on("negate", &rt, &[&host_value], &mut gpu, |_session| {
            ran.store(true, Ordering::SeqCst);
            Ok::<(), tenferro_ad::Error>(())
        })
        .expect_err("a CUDA engine does not read host memory");
        assert!(
            matches!(
                error,
                IdxTensorError::PlacementMismatch {
                    operation: "negate",
                    ..
                }
            ),
            "{error:?}"
        );
        assert!(!ran.load(Ordering::SeqCst), "refused before any work");
    }
}
