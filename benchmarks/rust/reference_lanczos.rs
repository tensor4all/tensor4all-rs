//! Standalone reference executable, deliberately outside the Cargo workspace.
//!
//! Linked by scripts/build-rsvd-reference.py to the exact faer artifacts of the
//! benchmark build. This does not add a faer dependency to any tensor4all crate.
//! Uses the public faer Lanczos partial-SVD API; no upstream source is copied.
//! API reference: t4a-faer 0.28.2, src/operator/eigen/mod.rs::{partial_svd,
//! partial_svd_scratch}; its private src/operator/svd/mod.rs driver requires
//! max_dim < min(m,n), so the comparison protocol uses matrices of at least 128.
use std::{error::Error, fs, time::Instant};

use dyn_stack::{MemBuffer, MemStack};
use faer::{
    matrix_free::eigen::{partial_svd, partial_svd_scratch, PartialEigenParams},
    Col, Mat, Par, Scale,
};

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("expected column-major input file")?;
    let input = fs::read(path)?;
    if input.len() < 24 {
        return Err("input header is truncated".into());
    }
    let read = |offset| -> Result<usize, Box<dyn Error>> {
        Ok(usize::try_from(u64::from_le_bytes(
            input[offset..offset + 8].try_into()?,
        ))?)
    };
    let (m, n, k) = (read(0)?, read(8)?, read(16)?);
    let len = m.checked_mul(n).ok_or("matrix shape overflow")?;
    if k == 0 || k > m.min(n) || input.len() != 24 + 8 * len {
        return Err("invalid matrix shape or rank".into());
    }
    let values: Vec<_> = input[24..]
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let a = Mat::from_fn(m, n, |i, j| values[i + m * j]);
    let mut v0 = Col::from_fn(n, |i| ((i + 1) as f64).sin() + 0.1);
    v0 /= v0.norm_l2();
    let params = PartialEigenParams::default();
    let requested_threads: usize = std::env::var("RAYON_NUM_THREADS")?.parse()?;
    let run = |mut par: Par<'_>| -> Result<(), Box<dyn Error>> {
        let req = partial_svd_scratch::<f64>(&a.as_ref(), k, par.shape(), params);
        // Identical time-based CPU priming in both decomposition executables.
        let prime = Instant::now();
        while prime.elapsed().as_secs_f64() < 1.0 {
            std::hint::spin_loop();
        }
        for repeat in 0..4 {
            let start = Instant::now();
            let mut memory = MemBuffer::new(req);
            let mut u = Mat::<f64>::zeros(m, k);
            let mut v = Mat::<f64>::zeros(n, k);
            let mut s = vec![0.; k];
            let info = partial_svd(
                u.as_mut(),
                v.as_mut(),
                &mut s,
                &a.as_ref(),
                v0.as_ref(),
                1e-10,
                par.reborrow(),
                MemStack::new(&mut memory),
                params,
            );
            let seconds = start.elapsed().as_secs_f64();
            if info.n_converged_eigen != k {
                return Err("Lanczos did not converge all requested triplets".into());
            }
            for (column, &value) in s.iter().enumerate() {
                let mut col = u.col_mut(column);
                col *= Scale(value);
            }
            let residual = (&a - &u * v.transpose()).norm_l2() / a.norm_l2();
            if repeat > 0 {
                println!(
                    "lanczos,{},{seconds},{residual},{}",
                    repeat - 1,
                    info.n_converged_eigen
                );
            }
        }
        Ok(())
    };
    if requested_threads == 1 {
        run(Par::Seq)
    } else {
        faer::team::with_team(requested_threads, run)
    }
}
