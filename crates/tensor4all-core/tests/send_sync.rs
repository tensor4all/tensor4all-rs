use tensor4all_core::TensorDynLen;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn eager_tensor_and_tensordynlen_are_send_sync() {
    assert_send_sync::<tenferro::EagerTensor<tenferro::CpuBackend>>();
    assert_send_sync::<TensorDynLen>();
}
