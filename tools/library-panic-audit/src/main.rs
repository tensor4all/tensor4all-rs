use anyhow::{bail, Result};

mod audit;

use audit::{audit, Configuration};
use std::env;
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    root: Option<PathBuf>,
    baseline: Option<PathBuf>,
    configuration: Option<Configuration>,
}

struct RawArgs {
    root: Option<PathBuf>,
    baseline: Option<PathBuf>,
    config_name: Option<String>,
    packages: Vec<String>,
    no_default_features: bool,
    features: Option<String>,
}

fn parse_args() -> Result<Option<Args>> {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from<I>(arguments: I) -> Result<Option<Args>>
where
    I: IntoIterator<Item = String>,
{
    let mut raw = RawArgs {
        root: None,
        baseline: None,
        config_name: None,
        packages: Vec::new(),
        no_default_features: false,
        features: None,
    };
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--root" => {
                raw.root = Some(
                    arguments
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--root requires a path"))?
                        .into(),
                );
            }
            "--baseline" => {
                raw.baseline = Some(
                    arguments
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--baseline requires a path"))?
                        .into(),
                );
            }
            "--config-name" => {
                raw.config_name = Some(
                    arguments
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--config-name requires a name"))?,
                );
            }
            "--package" | "-p" => {
                raw.packages.push(
                    arguments
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--package requires a package name"))?,
                );
            }
            "--no-default-features" => raw.no_default_features = true,
            "--features" => {
                if raw.features.is_some() {
                    bail!("--features may only be given once");
                }
                raw.features = Some(
                    arguments
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--features requires a feature list"))?,
                );
            }
            "--help" | "-h" => return Ok(None),
            other => bail!("unknown argument: {other}"),
        }
    }
    let configuration = match raw.config_name {
        Some(name) => Some(Configuration::new(
            name,
            raw.packages,
            raw.no_default_features,
            raw.features,
        )?),
        None => {
            if !raw.packages.is_empty() || raw.no_default_features || raw.features.is_some() {
                bail!("--package, --no-default-features, and --features require --config-name");
            }
            None
        }
    };
    Ok(Some(Args {
        root: raw.root,
        baseline: raw.baseline,
        configuration,
    }))
}

fn print_help() {
    println!(
        "Usage: library-panic-audit [--root PATH] [--baseline PATH]\n\
         \x20      [--config-name NAME --package PACKAGE... \
         [--no-default-features] [--features LIST]]\n\n\
         Audit Cargo-selected production Rust targets for panic-style paths.\n\n\
         Without --config-name the audit covers the workspace's default features.\n\
         With --config-name it covers exactly the listed packages under that Cargo\n\
         feature selection, because the CPU backend features are mutually\n\
         exclusive; stale baseline entries are not reported for a subset\n\
         selection."
    );
}

fn main() {
    let result = (|| -> Result<i32> {
        let Some(args) = parse_args()? else {
            print_help();
            return Ok(0);
        };
        let root = args.root.unwrap_or(env::current_dir()?);
        let baseline = args
            .baseline
            .unwrap_or_else(|| root.join("scripts/library-panics-baseline.json"));
        let report = audit(&root, &baseline, args.configuration.as_ref())?;
        Ok(report.exit_code_and_print())
    })();

    match result {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("library panic audit configuration error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> Result<Option<Args>> {
        parse_args_from(arguments.iter().map(|argument| (*argument).to_owned()))
    }

    #[test]
    fn default_audit_has_no_configuration() {
        let args = parse(&[]).unwrap().unwrap();
        assert!(args.configuration.is_none());
        let args = parse(&["--root", "/tmp/root"]).unwrap().unwrap();
        assert!(args.configuration.is_none());
        assert_eq!(args.root, Some(PathBuf::from("/tmp/root")));
    }

    #[test]
    fn configuration_parses_cargo_selection_flags() {
        let args = parse(&[
            "--config-name",
            "system-blas",
            "-p",
            "tensor4all-core",
            "--package",
            "tensor4all-treetn",
            "--no-default-features",
            "--features",
            "tenferro-system-blas",
        ])
        .unwrap()
        .unwrap();
        let configuration = args.configuration.unwrap();
        assert_eq!(configuration.name, "system-blas");
        assert_eq!(
            configuration.packages,
            vec!["tensor4all-core".to_owned(), "tensor4all-treetn".to_owned()]
        );
        assert!(configuration.no_default_features);
        assert_eq!(
            configuration.features.as_deref(),
            Some("tenferro-system-blas")
        );
    }

    #[test]
    fn configuration_flags_without_a_name_are_rejected() {
        for arguments in [
            vec!["--package", "tensor4all-core"],
            vec!["--no-default-features"],
            vec!["--features", "diagnostics"],
        ] {
            let error = parse(&arguments).unwrap_err().to_string();
            assert!(error.contains("require --config-name"), "{error}");
        }
    }

    #[test]
    fn unknown_arguments_still_fail() {
        let error = parse(&["--workspace"]).unwrap_err().to_string();
        assert!(error.contains("unknown argument"), "{error}");
        let error = parse(&["--features", "a", "--features", "b"])
            .unwrap_err()
            .to_string();
        assert!(error.contains("only be given once"), "{error}");
    }
}
