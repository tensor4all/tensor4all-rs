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
