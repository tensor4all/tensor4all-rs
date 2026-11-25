//! Immutable profiling settings shared by the core operation entry points.

use std::ffi::OsString;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoreDiagnostics {
    pub(crate) contract_profile: bool,
    pub(crate) pairwise_contract_profile: bool,
    pub(crate) gmres_profile: bool,
}

impl CoreDiagnostics {
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<OsString>) -> Self {
        Self {
            contract_profile: lookup("T4A_PROFILE_CONTRACT")
                .and_then(|value| value.into_string().ok())
                .is_some(),
            pairwise_contract_profile: lookup("T4A_PROFILE_PAIRWISE_CONTRACT")
                .and_then(|value| value.into_string().ok())
                .is_some(),
            // GMRES has always accepted non-Unicode values, unlike the other flags.
            gmres_profile: lookup("T4A_GMRES_OP_PROFILE").is_some(),
        }
    }
}

pub(crate) fn core_diagnostics() -> CoreDiagnostics {
    #[cfg(test)]
    if let Some(config) = TEST_CONFIG.with(std::cell::Cell::get) {
        return config;
    }
    static CONFIG: OnceLock<CoreDiagnostics> = OnceLock::new();
    *CONFIG.get_or_init(|| CoreDiagnostics::from_lookup(|key| std::env::var_os(key)))
}

#[cfg(test)]
thread_local! {
    static TEST_CONFIG: std::cell::Cell<Option<CoreDiagnostics>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_test_diagnostics<R>(
    configure: impl FnOnce(&mut CoreDiagnostics),
    run: impl FnOnce() -> R,
) -> R {
    struct Restore(Option<CoreDiagnostics>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_CONFIG.with(|slot| slot.set(self.0));
        }
    }

    let mut config = core_diagnostics();
    configure(&mut config);
    let _restore = Restore(TEST_CONFIG.with(|slot| slot.replace(Some(config))));
    run()
}

#[cfg(test)]
mod tests;
