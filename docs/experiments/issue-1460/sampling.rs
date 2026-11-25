//! Scalar reproducer based on the workload published in tensor4all-rs issue #1460.
use num_complex::Complex64 as C;
use std::{
    cell::RefCell,
    collections::HashMap,
    hint::black_box,
    io::{self, Write},
    time::Instant,
};
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorci::{crossinterpolate2, PivotSearchStrategy, TCI2Options};

fn value(x: &[usize]) -> C {
    let mut s = [0usize; 3];
    for &digit in x {
        for v in 0..3 {
            s[v] = 2 * s[v] + ((digit >> v) & 1);
        }
    }
    let n = (1usize << x.len()) - 1;
    if s.iter().sum::<usize>() > n {
        return C::new(0., 0.);
    }
    let t = s.map(|v| v as f64 * (8. / n as f64));
    let g = |t: f64| {
        C::new(0., -1.)
            * (-0.01 * t * t).exp()
            * (C::from_polar(1., t) + C::from_polar(1., 0.33 * t))
    };
    let d = |t: f64| {
        C::new(0., -1.)
            * (-t * t).exp()
            * (C::from_polar(1., 0.88 * t) + C::from_polar(1., 0.23 * t))
    };
    g(t[0]) * g(t[1]) * g(t[2]) * d(t[0] + t[1]) * d(t[1] + t[2])
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let sites: usize = args[1].parse()?;
    let strategy = match args[2].as_str() {
        "full" => PivotSearchStrategy::Full,
        "rook" => PivotSearchStrategy::Rook,
        strategy => anyhow::bail!("unknown strategy {strategy}"),
    };
    let mode = args[3].as_str();
    tensor4all_tensorbackend::set_default_cpu_threads(1usize.try_into()?)?;

    // Timing does not collect request traces, sample predictions, or allocation counts.
    let fit = |record: &mut dyn FnMut(u64)| -> anyhow::Result<_> {
        let cache = RefCell::new(HashMap::<u64, C>::new());
        let record = RefCell::new(record);
        let f = |index: &Vec<usize>| {
            let key = index
                .iter()
                .fold(0u64, |key, &digit| 8 * key + digit as u64);
            if mode == "trace" {
                assert_eq!(index.len(), sites);
                assert!(index.iter().all(|&digit| digit < 8));
                record.borrow_mut()(key);
            }
            *cache
                .borrow_mut()
                .entry(key)
                .or_insert_with(|| value(index))
        };
        let result = crossinterpolate2::<C, _, fn(&[Vec<usize>]) -> Vec<C>>(
            f,
            None,
            vec![8; sites],
            vec![vec![0; sites]],
            TCI2Options {
                tolerance: 1e-8,
                max_iter: 6,
                ncheck_history: 7,
                max_bond_dim: Some(128),
                pivot_search: strategy,
                nsearch: 0,
                max_nglobal_pivot: 0,
                strictly_nested: true,
                seed: Some(0),
                ..Default::default()
            },
        )?;
        let tt = result.tci.to_tensor_train()?;
        Ok((tt, result.ranks, result.errors, cache.into_inner().len()))
    };

    if mode == "time" {
        let warmup = Instant::now();
        while warmup.elapsed() < std::time::Duration::from_secs(1) {
            black_box(fit(&mut |_| {})?);
        }
        for _ in 0..9 {
            let start = Instant::now();
            let result = black_box(fit(&mut |_| {})?);
            let elapsed = start.elapsed();
            drop(result);
            println!("{}", elapsed.as_nanos());
        }
        return Ok(());
    }

    let mut calls = Vec::new();
    let (tt, ranks, errors, unique) = fit(&mut |key| calls.push(key))?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(out, "requests={} unique={unique}", calls.len())?;
    writeln!(out, "links={:?} ranks={ranks:?}", tt.link_dims())?;
    writeln!(
        out,
        "error_bits={:?}",
        errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>()
    )?;

    let mut state = 1024u64;
    let grid = 1usize << sites;
    let mut max_error = 0.0_f64;
    for probe in 0..4096 {
        let coordinates: [usize; 3] = std::array::from_fn(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as usize % if probe < 2048 { grid } else { grid / 8 }
        });
        let index: Vec<_> = (0..sites)
            .rev()
            .map(|shift| (0..3).map(|v| ((coordinates[v] >> shift) & 1) << v).sum())
            .collect();
        let actual = tt.evaluate(&index)?;
        max_error = max_error.max((actual - value(&index)).norm() / 32.0);
        writeln!(
            out,
            "{:016x} {:016x}",
            actual.re.to_bits(),
            actual.im.to_bits()
        )?;
    }
    assert!(max_error < 1e-7, "scaled residual {max_error}");
    writeln!(out, "residual={max_error:e}")?;
    for key in calls {
        writeln!(out, "{key:x}")?;
    }
    out.flush()?;
    Ok(())
}
