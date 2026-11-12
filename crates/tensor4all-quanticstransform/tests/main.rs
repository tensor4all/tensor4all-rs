//! Heavy integration tests for `tensor4all-quanticstransform`.
//! Each heavy file is a module so its test names start with `heavy_`, which
//! the scheduled heavy-tests workflow selects with `test(/^heavy_/)`. Ordinary
//! integration tests are separate `[[test]]` targets in `Cargo.toml`.

mod heavy_patched_fourier;
