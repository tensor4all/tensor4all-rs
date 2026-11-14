//! This separate process proves construction never initializes the global backend.
use std::{convert::Infallible, num::NonZeroUsize};
use tensor4all_interpolativeqtt::{interpolate_patched_nd, PatchedInterpolativeOptions};
use tensor4all_tensorbackend::set_default_cpu_threads;

#[test]
fn patched_construction_leaves_global_backend_uninitialized() {
    let options = PatchedInterpolativeOptions {
        max_bond_dim: Some(1),
        ..Default::default()
    };
    let evaluate = |p: tensor4all_core::ColMajorArrayRef<'_, f64>,
                    _: tensor4all_tensorbackend::Exec<'_>| {
        Ok::<_, Infallible>(p.data().iter().map(|x| 1.0 + x).collect())
    };
    let patches = interpolate_patched_nd(evaluate, &[0.0], &[1.0], 4, 3, &options).unwrap();
    assert!(patches.len() > 1);
    #[cfg(feature = "parallel")]
    {
        use tensor4all_interpolativeqtt::interpolate_patched_nd_on_lanes;
        use tensor4all_tensorbackend::CpuBackend;
        let mut lanes = hataori_tenferro::TenferroLanes::new(
            (0..2)
                .map(|_| CpuBackend::with_threads(2).unwrap().into_engine())
                .collect(),
        )
        .unwrap();
        let parallel =
            interpolate_patched_nd_on_lanes(&mut lanes, evaluate, &[0.0], &[1.0], 4, 3, &options)
                .unwrap();
        assert_eq!(patches.len(), parallel.len());
    }
    // This setter fails once a process-global backend has been initialized.
    set_default_cpu_threads(NonZeroUsize::new(1).unwrap()).unwrap();
}
