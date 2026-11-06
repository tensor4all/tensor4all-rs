use super::*;
use std::error::Error;

#[test]
fn source_chain_is_preserved_for_initialization() {
    let error = CudaExecutionContextError::Initialization {
        component: "backend",
        source: Arc::new(std::io::Error::other("driver unavailable")),
    };
    assert!(error.source().is_some());
    assert_eq!(error.source().unwrap().to_string(), "driver unavailable");
}

#[test]
fn metadata_rejections_do_not_need_a_cuda_context() {
    let host = Placement::default();
    assert!(validate_host_placement_metadata(&host, None).is_ok());
    assert!(!is_cuda_ordinal_zero_placement(&host));

    let cuda = Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: CUDA_ORDINAL as usize,
        }),
        cpu_affinity: None,
    };
    assert!(is_cuda_ordinal_zero_placement(&cuda));
    assert!(matches!(
        validate_host_placement_metadata(&cuda, None),
        Err(CudaExecutionContextError::UnsupportedInput { .. })
    ));
}

/// A runtime owned by a CUDA context resolves to that context's single engine,
/// and a nested entry on the same thread is the typed CUDA re-entry refusal.
#[cfg(feature = "global-defaults")]
#[test]
fn implicit_engine_route_resolves_the_owning_cuda_engine() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0");
    let runtime = context.eager_runtime().expect("eager runtime");

    let is_cuda_engine =
        crate::with_runtime_engine(&runtime, |engine| engine.as_any_mut().is::<CudaEngine>())
            .expect("implicit engine entry");
    assert!(is_cuda_engine, "the owning context's CUDA engine is used");

    let nested = context
        .with_session(|_| crate::with_runtime_engine(&runtime, |_| ()))
        .expect("outer session");
    assert!(matches!(
        nested,
        Err(crate::ExecutionContextEntryError::Cuda(
            CudaExecutionContextError::Reentered
        ))
    ));
}
