//! Sweep the integral error over several quantics resolutions.
//!
//! The main integral tutorial prints one terminal result. This companion writes
//! a tiny CSV table so Julia can plot how the integral error changes with `R`.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use tensor4all_tutorial_code::output_paths;
use tensor4all_tutorial_code::qtt_integral_sweep_utils::{
    print_sweep_row, write_sweep_csv, IntegralSweepRow,
};
use tensor4all_tutorial_code::qtt_interval_common::{
    build_interval_grid, build_interval_qtt, exact_integral, interval_target,
    IntervalTutorialConfig, DEFAULT_INTERVAL_CONFIG,
};

const R_VALUES: &[usize] = &[3, 4, 5, 6, 7, 8, 9, 10];

struct OutputSpec {
    csv_name: &'static str,
    plot_script_name: &'static str,
}

impl OutputSpec {
    fn new() -> Self {
        Self {
            csv_name: "qtt_integral_sweep.csv",
            plot_script_name: "qtt_integral_sweep_plot.jl",
        }
    }

    fn csv_path(&self, data_dir: &Path) -> PathBuf {
        data_dir.join(self.csv_name)
    }

    fn plot_script_path(&self, project_root: &Path) -> PathBuf {
        project_root
            .join("docs")
            .join("plotting")
            .join(self.plot_script_name)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = output_paths::data_dir();
    fs::create_dir_all(&data_dir)?;
    let output = OutputSpec::new();

    let mut rows = Vec::with_capacity(R_VALUES.len());

    for &r in R_VALUES {
        let config = IntervalTutorialConfig {
            bits: r,
            ..DEFAULT_INTERVAL_CONFIG
        };
        let exact = exact_integral(&config);
        let grid = build_interval_grid(&config)?;
        let (qtci, _ranks, _errors) = build_interval_qtt(&grid, interval_target, &config)?;
        let integral = qtci.integral()?;
        let row = IntegralSweepRow {
            r,
            npoints: 1usize << r,
            integral,
            exact_integral: exact,
            abs_error: (integral - exact).abs(),
            rank: qtci.rank(),
        };

        print_sweep_row(&row);
        rows.push(row);
    }

    let csv_path = output.csv_path(&data_dir);
    write_sweep_csv(&csv_path, &rows)?;
    println!("wrote {}", csv_path.display());
    println!(
        "next: run the Julia plotting script at {}",
        output.plot_script_path(project_root).display()
    );

    Ok(())
}
