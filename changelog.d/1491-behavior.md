- `tensor4all-treersi`, `tenferro-cuda` builds (#1379): the lower-triangular
  solve of the local rank decision now runs on a single-thread CPU context
  owned by the run, not the process-global CPU context. It no longer contends
  with other users of the global backend; thread count changes from the global
  default to 1, which may alter the interpolation matrix in the last bits. The
  input-core readback still uses the global context.
