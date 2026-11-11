// Importance sampler retained from the M6a benchmark; see the investigation provenance.
use std::f64::consts::PI;
const MU: f64 = 0.5;
struct Grid {
    r: usize,
    temperature: f64,
}
impl Grid {
    fn size(&self) -> usize {
        1 << self.r
    }
    fn omega(&self, n: usize) -> f64 {
        2.0 * PI * self.temperature * (n as f64 - self.size() as f64 / 2.0 + 0.5)
    }
    fn g0(&self, x: usize, y: usize, n: usize) -> Complex64 {
        g0::exact_with_temperature(self.r, x, y, n, self.temperature)
    }
    fn point(&self, c: [usize; 3]) -> Vec<usize> {
        (0..self.r)
            .flat_map(|bit| c.map(|v| (v >> (self.r - 1 - bit)) & 1))
            .collect()
    }
}
fn frequency_sums(grid: &Grid, power: i32) -> Vec<f64> {
    let size = grid.size();
    let half = size / 2;
    let cosines: Vec<(f64, f64)> = (0..=half)
        .map(|k| {
            let multiplicity = if k == 0 || k == half { 1.0 } else { 2.0 };
            ((2.0 * PI * k as f64 / size as f64).cos(), multiplicity)
        })
        .collect();
    let omegas2: Vec<f64> = (0..half)
        .map(|h| {
            let w = 2.0 * PI * grid.temperature * (h as f64 + 0.5);
            w * w
        })
        .collect();
    let mut sums = vec![0.0; half];
    for (i, &(ci, mi)) in cosines.iter().enumerate() {
        for &(cj, mj) in &cosines[i..] {
            let weight = if ci == cj { mi * mj } else { 2.0 * mi * mj };
            let a = MU + 2.0 * ci + 2.0 * cj;
            let a2 = a * a;
            for (sum, &w2) in sums.iter_mut().zip(&omegas2) {
                let inverse = 1.0 / (a2 + w2);
                *sum += weight * inverse.powi(power);
            }
        }
    }
    sums
}

struct Samples {
    /// Column-major `[3 R, n]` site values.
    points: Vec<usize>,
    /// Grid coordinates `(kx, ky, n)` of each point.
    coords: Vec<[usize; 3]>,
    /// The density of the mixture at each point.
    density: Vec<f64>,
    /// The exact L2 norm `S` of the analytic function on the grid.
    norm: f64,
}

/// Draw `count` points from `q = (|f|^2 / S^2 + 1 / V) / 2` with
/// `|f|^2 = |G0|^(2 power)`.
fn draw(grid: &Grid, power: i32, count: usize, seed: u64) -> Samples {
    let size = grid.size();
    let half = size / 2;
    let sums = frequency_sums(grid, power);
    let total: f64 = 2.0 * sums.iter().sum::<f64>();
    // CDF over all frequencies n in 0..size.
    let mass = |n: usize| {
        let h = if n >= half { n - half } else { half - 1 - n };
        sums[h]
    };
    let mut cdf = Vec::with_capacity(size);
    let mut acc = 0.0;
    for n in 0..size {
        acc += mass(n);
        cdf.push(acc / total);
    }
    let volume = (size as f64).powi(3);
    let modulus = |kx: usize, ky: usize, n: usize| grid.g0(kx, ky, n).norm_sqr().powi(power);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut coords = Vec::with_capacity(count);
    for _ in 0..count {
        if rng.random::<bool>() {
            coords.push([
                rng.random_range(0..size),
                rng.random_range(0..size),
                rng.random_range(0..size),
            ]);
            continue;
        }
        let u: f64 = rng.random();
        let n = cdf.partition_point(|&c| c < u).min(size - 1);
        let w2 = grid.omega(n).powi(2);
        // Rejection from uniform k: |G0|^2 <= 1 / omega^2.
        loop {
            let (kx, ky) = (rng.random_range(0..size), rng.random_range(0..size));
            let accept = (modulus(kx, ky, n) * w2.powi(power)).min(1.0);
            if rng.random::<f64>() < accept {
                coords.push([kx, ky, n]);
                break;
            }
        }
    }
    let density = coords
        .iter()
        .map(|&[kx, ky, n]| 0.5 * modulus(kx, ky, n) / total + 0.5 / volume)
        .collect();
    let points = coords.iter().flat_map(|&c| grid.point(c)).collect();
    Samples {
        points,
        coords,
        density,
        norm: total.sqrt(),
    }
}

fn relative_l2(samples: &Samples, approx: &[Complex64], exact: &[Complex64]) -> (f64, f64) {
    let terms: Vec<f64> = approx
        .iter()
        .zip(exact)
        .zip(&samples.density)
        .map(|((a, e), q)| (a - e).norm_sqr() / q)
        .collect();
    let n = terms.len() as f64;
    let mean = terms.iter().sum::<f64>() / n;
    let var = terms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let rel = mean.sqrt() / samples.norm;
    // delta method: se(sqrt(E)) = se(E) / (2 sqrt(E)).
    let se = if mean > 0.0 {
        (var / n).sqrt() / (2.0 * mean.sqrt()) / samples.norm
    } else {
        0.0
    };
    (rel, se)
}
