use super::*;

#[test]
fn absent_settings_disable_all_profiles() {
    assert_eq!(
        CoreDiagnostics::from_lookup(|_| None),
        CoreDiagnostics {
            contract_profile: false,
            pairwise_contract_profile: false,
            gmres_profile: false,
        }
    );
}

#[test]
fn profile_flags_are_independent_and_use_presence() {
    for key in [
        "T4A_PROFILE_CONTRACT",
        "T4A_PROFILE_PAIRWISE_CONTRACT",
        "T4A_GMRES_OP_PROFILE",
    ] {
        for value in ["", "0", "false", "1"] {
            let config = CoreDiagnostics::from_lookup(|name| (name == key).then(|| value.into()));
            assert_eq!(config.contract_profile, key == "T4A_PROFILE_CONTRACT");
            assert_eq!(
                config.pairwise_contract_profile,
                key == "T4A_PROFILE_PAIRWISE_CONTRACT"
            );
            assert_eq!(config.gmres_profile, key == "T4A_GMRES_OP_PROFILE");
        }
    }
}

#[cfg(unix)]
#[test]
fn non_unicode_values_keep_the_original_flag_semantics() {
    use std::os::unix::ffi::OsStringExt;
    let config = CoreDiagnostics::from_lookup(|_| Some(OsString::from_vec(vec![0xff])));
    assert!(!config.contract_profile);
    assert!(!config.pairwise_contract_profile);
    assert!(config.gmres_profile);
}

#[test]
fn test_overrides_are_nested_and_restored_after_unwinding() {
    let original = core_diagnostics();
    with_test_diagnostics(
        |config| config.contract_profile = !original.contract_profile,
        || {
            let outer = core_diagnostics();
            assert_eq!(outer.contract_profile, !original.contract_profile);
            let result = std::panic::catch_unwind(|| {
                with_test_diagnostics(
                    |config| config.gmres_profile = !outer.gmres_profile,
                    || {
                        assert_eq!(core_diagnostics().gmres_profile, !outer.gmres_profile);
                        panic!("exercise override cleanup");
                    },
                );
            });
            assert!(result.is_err());
            assert_eq!(core_diagnostics(), outer);
        },
    );
    assert_eq!(core_diagnostics(), original);
}

#[test]
fn test_overrides_do_not_change_other_threads() {
    let original = core_diagnostics();
    with_test_diagnostics(
        |config| config.gmres_profile = !original.gmres_profile,
        || {
            assert_eq!(core_diagnostics().gmres_profile, !original.gmres_profile);
            assert_eq!(
                std::thread::spawn(core_diagnostics).join().unwrap(),
                original
            );
        },
    );
}
