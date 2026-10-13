use super::*;

#[test]
fn external_kind_is_named_external() {
    assert_eq!(
        tensor_dtype_name(DType::External(core::any::TypeId::of::<u8>())),
        "external"
    );
}
