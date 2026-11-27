- `tensor4all-hdf5` no longer sets `HDF5_USE_FILE_LOCKING=FALSE` on first use (#1386).
  The library never touches the process environment now. If you save and then re-open
  the same path in one process (or do so from several threads), set
  `HDF5_USE_FILE_LOCKING=FALSE` yourself before the first HDF5 call in the process
  (shell, service unit, Cargo `[env]`, or `main` before threads start); otherwise the
  reopen can fail with `errno = 35`. Leaving it unset keeps HDF5's cross-process write
  protection.
