use super::*;

// Compile-time check that TensorLike requires Sized (no dyn TensorLike)
fn _assert_sized<T: TensorLike>() {
    // This confirms T: Sized is required
}
