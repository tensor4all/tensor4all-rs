//! Clean-pinned native scaling fixture. Numerical dependencies are unmodified.
//! Physical evaluator and oracle follow the earlier Green-function diagnostic;
//! resource construction uses only the downstream's existing public defaults.
use hataori_tenferro::TenferroLanes;
use num_complex::Complex64;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::{json, Value as Json};
use std::{
    collections::BTreeSet,
    convert::Infallible,
    fs,
    path::Path,
    process::Command,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};
use tensor4all_core::ColMajorArrayRef;
use tensor4all_interpolativeqtt::{
    interpolate_patched_nd, interpolate_patched_nd_on_lanes, InterpolativePatch,
    InterpolativeScalar, PatchedInterpolativeOptions,
};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, Exec};
const LOWER: [f64; 2] = [-6.0, 0.8];
const UPPER: [f64; 2] = [6.0, 1.6];
trait Value: InterpolativeScalar + Send + Sync + std::fmt::Debug {
    fn project(z: Complex64) -> Self;
    fn parts(self) -> [f64; 2];
}
impl Value for f64 {
    fn project(z: Complex64) -> Self {
        z.re
    }
    fn parts(self) -> [f64; 2] {
        [self, 0.0]
    }
}
impl Value for Complex64 {
    fn project(z: Complex64) -> Self {
        z
    }
    fn parts(self) -> [f64; 2] {
        [self.re, self.im]
    }
}
fn mask() -> String {
    fs::read_to_string("/proc/thread-self/status")
        .unwrap()
        .lines()
        .find_map(|s| s.strip_prefix("Cpus_allowed_list:"))
        .unwrap()
        .trim()
        .to_owned()
}
fn cpu_ns() -> u64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid writable timespec and Linux's process CPU clock id.
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut t) },
        0
    );
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
#[derive(Default)]
struct Counts {
    calls: AtomicUsize,
    points: AtomicUsize,
    masks: Mutex<BTreeSet<String>>,
}
fn reference(p: &[f64]) -> Complex64 {
    let z = Complex64::new(p[0], p[1]);
    (0..1024)
        .map(|k| {
            let energy = 2.0 * (std::f64::consts::TAU * (k as f64 + 0.5) / 1024.0).cos();
            let w = z - energy;
            Complex64::new(1.0, 0.0) / ((w - 2.0).sqrt() * (w + 2.0).sqrt())
        })
        .sum::<Complex64>()
        / 1024.0
}

fn energies(mesh: usize) -> Vec<f64> {
    let axis: Vec<_> = (0..mesh)
        .map(|k| 2.0 * (std::f64::consts::TAU * (k as f64 + 0.5) / mesh as f64).cos())
        .collect();
    axis.iter()
        .flat_map(|&x| axis.iter().map(move |&y| x + y))
        .collect()
}

fn integrate(p: &[f64], energies: &[f64]) -> Complex64 {
    let p = std::hint::black_box(p);
    let eta = p[1];
    let eta2 = eta * eta;
    let mut re = 0.0;
    let mut im = 0.0;
    for &energy in energies {
        let delta = p[0] - energy;
        let denominator = delta * delta + eta2;
        re += delta / denominator;
        im -= eta / denominator;
    }
    Complex64::new(re, im) / energies.len() as f64
}

fn evaluate<T: Value, const TRACK: bool>(
    p: ColMajorArrayRef<'_, f64>,
    exec: Exec<'_>,
    e: &[f64],
    counts: &Counts,
    b: usize,
) -> Result<Vec<T>, Infallible> {
    assert_eq!(exec.budget().get(), b);
    let n = p.shape()[1];
    if TRACK {
        counts.calls.fetch_add(1, Ordering::Relaxed);
        counts.points.fetch_add(n, Ordering::Relaxed);
    }
    let jobs = b.min(n);
    let mut slots: Vec<Vec<T>> = (0..jobs).map(|_| Vec::new()).collect();
    exec.fork_sized_slots(&vec![1; jobs], &mut slots, |job, out, _| {
        if TRACK {
            counts.masks.lock().unwrap().insert(mask());
        }
        out.extend(
            (n * job / jobs..n * (job + 1) / jobs)
                .map(|i| T::project(integrate(&p.data()[2 * i..2 * i + 2], e))),
        );
    })
    .unwrap();
    Ok(slots.into_iter().flatten().collect())
}
fn check<T: Value>(patches: &[InterpolativePatch]) -> Json {
    let mut visits = vec![0u32; 256];
    let mut values = vec![[0.0; 2]; 256];
    let mut max_error = 0.0_f64;
    let mut bytes = 0;
    for patch in patches {
        let depth = patch.prefix[0].len();
        assert_eq!(patch.site_indices.len(), 4 - depth);
        assert!(patch.prefix.iter().all(|a| a.len() == depth));
        assert!(patch.error_estimate <= 1e-4);
        let cores = patch.tree.chain_cores::<T>(&patch.site_indices).unwrap();
        bytes += cores
            .iter()
            .map(|c| c.len() * std::mem::size_of::<T>())
            .sum::<usize>();
        let dense = patch
            .tree
            .contract_to_tensor()
            .unwrap()
            .permute_indices(&patch.site_indices)
            .unwrap()
            .dense_host_values::<T>()
            .unwrap();
        for (mut flat, &value) in dense.iter().enumerate() {
            let mut indices: Vec<_> = patch
                .prefix
                .iter()
                .map(|a| a.iter().fold(0, |n, &b| 2 * n + b))
                .collect();
            for _ in 0..4 - depth {
                let site = flat % 4;
                flat >>= 2;
                for (d, index) in indices.iter_mut().enumerate() {
                    *index = 2 * *index + ((site >> d) & 1);
                }
            }
            let global = indices[0] + 16 * indices[1];
            visits[global] += 1;
            values[global] = value.parts();
            assert!(values[global].iter().all(|x| x.is_finite()));
            let physical = [
                LOWER[0] + (UPPER[0] - LOWER[0]) * indices[0] as f64 / 16.0,
                LOWER[1] + (UPPER[1] - LOWER[1]) * indices[1] as f64 / 16.0,
            ];
            max_error = max_error.max((value - T::project(reference(&physical))).abs_val());
        }
    }
    assert!(visits.iter().all(|&n| n == 1));
    assert!(max_error <= 1e-4, "oracle max error {max_error}");
    assert!(patches.windows(2).all(|p| p[0].prefix < p[1].prefix));
    json!({"patches":patches.len(),"prefixes":patches.iter().map(|p|p.prefix.clone()).collect::<Vec<_>>(),"core_bytes":bytes,"values":values,"max_error":max_error})
}
fn prime(engine: &mut tensor4all_tensorbackend::CpuEngine) {
    engine
        .with_exec(|exec| {
            let t = exec.token();
            let jobs = t.budget().get();
            let until = Instant::now() + Duration::from_secs(2);
            t.fork(jobs, |_, _| {
                while Instant::now() < until {
                    std::hint::spin_loop();
                }
            });
        })
        .unwrap();
}
fn run<T: Value>(family: &str, p: usize, b: usize) -> Json {
    let initial_mask = mask();
    let mut creator_masks = Vec::new();
    let mut make = || {
        let backend = CpuBackend::with_threads(b).unwrap();
        assert_eq!(backend.num_threads(), b);
        if b == 1 {
            assert!(backend.worker_cpus().is_empty());
        }
        creator_masks.push(mask());
        backend
    };
    let context = (family == "engine").then(|| CpuExecutionContext::from_backend(make()));
    let mut lanes = (family != "engine")
        .then(|| TenferroLanes::new((0..p).map(|_| make().into_engine()).collect()).unwrap());
    let after_construction = mask();
    let e = energies(128);
    let counts = Counts::default();
    let options = PatchedInterpolativeOptions {
        tolerance: 1e-4,
        max_bond_dim: Some(2),
        ..Default::default()
    };
    if let Some(c) = &context {
        c.with_session(prime).unwrap();
    }
    if let Some(l) = &mut lanes {
        hataori::map_in_lanes(l, vec![(); p], |_, engine| {
            prime(engine);
            Ok::<_, Infallible>(())
        })
        .unwrap();
    }
    let construct = |track: bool, local: &mut Option<TenferroLanes>| {
        let callback = |points: ColMajorArrayRef<'_, f64>, exec: Exec<'_>| {
            if track {
                evaluate::<T, true>(points, exec, &e, &counts, b)
            } else {
                evaluate::<T, false>(points, exec, &e, &counts, b)
            }
        };
        if let Some(c) = &context {
            interpolate_patched_nd(c, callback, &LOWER, &UPPER, 4, 3, &options).unwrap()
        } else {
            interpolate_patched_nd_on_lanes(
                local.as_mut().unwrap(),
                callback,
                &LOWER,
                &UPPER,
                4,
                3,
                &options,
            )
            .unwrap()
        }
    };
    let warm = construct(true, &mut lanes);
    let expected = check::<T>(&warm);
    drop(warm);
    // Select the untracked callback before the primary API call: its body has
    // no counters, clocks or affinity queries. CPU clocks bracket the wall timer.
    let callback = |points: ColMajorArrayRef<'_, f64>, exec: Exec<'_>| {
        evaluate::<T, false>(points, exec, &e, &counts, b)
    };
    let before_cpu = cpu_ns();
    let start = Instant::now();
    let patches = if let Some(c) = &context {
        interpolate_patched_nd(c, callback, &LOWER, &UPPER, 4, 3, &options).unwrap()
    } else {
        interpolate_patched_nd_on_lanes(
            lanes.as_mut().unwrap(),
            callback,
            &LOWER,
            &UPPER,
            4,
            3,
            &options,
        )
        .unwrap()
    };
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let process_cpu_ns = cpu_ns() - before_cpu;
    let actual = check::<T>(&patches);
    assert_eq!(expected, actual);
    json!({"wall_ms":wall_ms,"process_cpu_ns":process_cpu_ns,"initial_mask":initial_mask,"after_construction_mask":after_construction,
        "creator_masks":creator_masks,"callback_masks":counts.masks.into_inner().unwrap(),"calls":counts.calls.load(Ordering::Relaxed),
        "points":counts.points.load(Ordering::Relaxed),"check":actual})
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--cell") {
        let p = args[4].parse().unwrap();
        let b = args[5].parse().unwrap();
        let row = match args[3].as_str() {
            "real" => run::<f64>(&args[2], p, b),
            "complex" => run::<Complex64>(&args[2], p, b),
            _ => panic!("scalar"),
        };
        println!("CLEAN_PIN_SAMPLE {row}");
        return;
    }
    let out = Path::new(args.get(1).expect("output directory"));
    fs::create_dir_all(out).unwrap();
    assert!(
        !out.join("samples.json").exists(),
        "never overwrite a measured attempt"
    );
    let mut cases = Vec::new();
    for family in ["engine", "lanes"] {
        for n in [1, 2, 4, 8, 16] {
            let (p, b) = if family == "engine" { (1, n) } else { (n, 1) };
            cases.push((family, p, b));
        }
    }
    for p in [2, 4, 8] {
        cases.push(("mixed", p, 2));
    }
    let mut rng = StdRng::seed_from_u64(1347);
    let mut rows = Vec::new();
    let exe = std::env::current_exe().unwrap();
    for block in 0..5 {
        let mut order = Vec::new();
        for scalar in ["real", "complex"] {
            for &(family, p, b) in &cases {
                order.push((scalar, family, p, b));
            }
        }
        order.shuffle(&mut rng);
        for (scalar, family, p, b) in order {
            let total = p * b;
            let cpus = [8, 10, 12, 14][..total.min(4)]
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let label = format!("block-{block}-{scalar}-{family}-p{p}-b{b}");
            let result = Command::new("taskset")
                .args(["-c", &cpus])
                .arg(&exe)
                .args(["--cell", family, scalar, &p.to_string(), &b.to_string()])
                .env("RAYON_NUM_THREADS", "1")
                .env("OMP_NUM_THREADS", "1")
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env("VECLIB_MAXIMUM_THREADS", "1")
                .output()
                .unwrap();
            fs::write(
                out.join(format!("{label}.log")),
                [result.stdout.as_slice(), result.stderr.as_slice()].concat(),
            )
            .unwrap();
            assert!(result.status.success(), "child failed: {label}");
            let stdout = String::from_utf8(result.stdout).unwrap();
            let sample = stdout
                .lines()
                .find_map(|l| l.strip_prefix("CLEAN_PIN_SAMPLE "))
                .unwrap();
            let sample: Json = serde_json::from_str(sample).unwrap();
            rows.push(json!({"block":block,"scalar":scalar,"family":family,"p":p,"b":b,"total_threads":total,"requested_cpus":cpus,"sample":sample}));
            fs::write(
                out.join("samples.json"),
                serde_json::to_string_pretty(&rows).unwrap(),
            )
            .unwrap();
            println!("completed {}/130 {label}", rows.len());
        }
    }
    assert_eq!(rows.len(), 130);
    for scalar in ["real", "complex"] {
        let samples: Vec<_> = rows.iter().filter(|r| r["scalar"] == scalar).collect();
        let first = &samples[0]["sample"];
        for row in samples {
            let sample = &row["sample"];
            assert_eq!(sample["calls"], first["calls"], "unequal callback work");
            assert_eq!(sample["points"], first["points"], "unequal target work");
            for key in ["prefixes", "patches", "core_bytes"] {
                assert_eq!(sample["check"][key], first["check"][key], "unequal {key}");
            }
            for (actual, expected) in sample["check"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .zip(first["check"]["values"].as_array().unwrap())
            {
                for (a, b) in actual
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(expected.as_array().unwrap())
                {
                    assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= 1e-10);
                }
            }
        }
    }
    fs::write(
        out.join("validation.json"),
        "{\"complete\":true,\"independent_processes\":130,\"work_and_values_agree\":true}\n",
    )
    .unwrap();
}
