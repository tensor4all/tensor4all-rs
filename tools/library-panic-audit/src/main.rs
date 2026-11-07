use anyhow::{bail, Result};

mod audit;

use audit::{audit, Configuration};
use std::env;
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    root: Option<PathBuf>,
    baseline: Option<PathBuf>,
    update_baseline: bool,
    configuration: Option<Configuration>,
}

struct RawArgs {
    root: Option<PathBuf>,
    baseline: Option<PathBuf>,
    update_baseline: bool,
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
        update_baseline: false,
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
            "--update-baseline" => raw.update_baseline = true,
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
    if raw.update_baseline && configuration.is_some() {
        bail!("--update-baseline cannot be combined with --config-name: a configuration audits a subset of the workspace");
    }
    Ok(Some(Args {
        root: raw.root,
        baseline: raw.baseline,
        update_baseline: raw.update_baseline,
        configuration,
    }))
}

fn print_help() {
    println!(
        "Usage: library-panic-audit [--root PATH] [--baseline PATH] [--update-baseline]\n\
         \x20      [--config-name NAME --package PACKAGE... \
         [--no-default-features] [--features LIST]]\n\n\
         Audit Cargo-selected production Rust targets for panic-style paths.\n\n\
         Without --config-name the audit covers the workspace's default features.\n\
         With --config-name it covers exactly the listed packages under that Cargo\n\
         feature selection, because the CPU backend features are mutually\n\
         exclusive; stale baseline entries are not reported for a subset\n\
         selection.\n\n\
         A baseline entry is keyed on (path, kind, enclosing item, occurrence\n\
         index), so code inserted above an assertion never invalidates it. A moved\n\
         entry is reported as information; an added or a deleted assertion fails.\n\
         --update-baseline rewrites the recorded lines and assertion text of the\n\
         current tree, prints what it changed, and refuses to run in CI or while\n\
         the set of keys is changing."
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
        if args.update_baseline {
            report.update_baseline(env::var("CI").ok().as_deref())?;
        }
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
mod tests;
