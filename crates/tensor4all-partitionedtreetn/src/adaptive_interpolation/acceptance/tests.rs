use super::*;

#[test]
fn measured_floors_are_not_capped_successes() {
    let policy = SizePolicy::new(
        &PatchedInterpolationOptions::new(4)
            .with_capped_patches(CappedPatches::AcceptUpTo { bits: 8 }),
    );
    assert!(policy.capped_eligible(InterpolationTermination::BondCapReached, 4));
    assert!(!policy.capped_eligible(InterpolationTermination::SampledErrorFloor, 4));
}
