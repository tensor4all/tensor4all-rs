- The tenferro pin moves to tenferro-rs `5df83672` (tenferro 0.13.0 on threadteam 0.6.1,
  colmat 0.34, cpueinsum 0.13.0, tlinalg 0.12.0, tcontract 0.13.0; #1428). tenferro now
  has one faer-backed CPU backend (tenferro-rs#2609), so the CPU backend features
  `tenferro-cpu-platform`, `tenferro-cpu-faer`, `tenferro-system-blas` and
  `tenferro-provider-inject` are removed from every crate, together with the
  `tensor4all-cpu-platform` crate; there is no alias. macOS no longer uses Apple
  Accelerate. Drop these features from dependency lines (and the
  `default-features = false` that only served them). The direct dense dependency is
  `colmat` 0.34 (the renamed `t4a-faer`). `tensor4all_tensorbackend::explicit::detach`
  takes `&mut dyn tenferro::ExecutionEngine` instead of `&mut dyn Engine`, because
  tenferro materializes an eager value through an eager session.
