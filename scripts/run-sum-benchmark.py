#!/usr/bin/env python3
"""Reproducible Linux #1127 full-gate summation experiment (standard library only).

Run from a clean, committed checkout on a dedicated Linux host. Configure host
isolation yourself: every visible userspace thread must exclude the measured
CPU and its SMT siblings; observers must also exclude those protected CPUs.
The runner pins itself to observers, builds release once, freezes provenance,
and launches fresh one-CPU processes. It never changes other process affinities
or system configuration. /proc must expose every host process (no hidepid/PID
container namespace). Inaccessible audits invalidate the experiment. Inherited LD_PRELOAD and
LD_LIBRARY_PATH overrides are unsupported and rejected.

The launch affinity must include measured and observer CPUs. Ancestors/agents
stay on observers; explicitly widen only this launch, then it confines itself:
  taskset -c 0-2,4-7 python3 scripts/run-sum-benchmark.py --cpu 2
      --observer-cpus 0-1,4-7 --output /absolute/new-output
Exclude CPU2's SMT siblings from observers (actual topology varies).
--smoke checks plumbing only. --prepare-only records build/preflight without
measurements. Neither can PASS. --analyze DIR checks saved records without
building, host preflight, or Linux; it never changes the frozen protocol/raw data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import subprocess
import sys
import time

THREADS = ("RAYON_NUM_THREADS", "BLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "TENFERRO_NUM_THREADS")
METHODS = ("aci", "sequential", "balanced", "density", "exact", "exact_truncate")
CHECKSUM_FIELDS = ("sum_re", "sum_im", "sum_sq", "maxabs", "weighted_re", "weighted_im")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value, mode="x"):
    with Path(path).open(mode) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def source_digest(repo):
    result = hashlib.sha256()
    paths = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"]).split(b"\0")
    for name in sorted(filter(None, paths)):
        path = repo / os.fsdecode(name)
        if not path.is_file() or (path.is_symlink() and not path.resolve().is_relative_to(repo.resolve())):
            raise ValueError(f"cannot freeze tracked source {path}")
        result.update(name + b"\0" + digest(path).encode() + b"\0")
    return result.hexdigest()


def clean_checkout(repo):
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("clean committed checkout required (including untracked files)")
    return {"commit": git(repo, "rev-parse", "HEAD"), "source_sha256": source_digest(repo),
            "cargo_lock_sha256": digest(repo / "Cargo.lock")}


def cargo_config_snapshot(repo):
    cargo_home = Path(os.environ.get("CARGO_HOME") or Path.home() / ".cargo")
    if not cargo_home.is_absolute():
        cargo_home = repo / cargo_home
    directories = [cargo_home, *(parent / ".cargo" for parent in (repo, *repo.parents))]
    paths = {directory.joinpath(name).resolve() for directory in directories for name in ("config", "config.toml")}
    snapshot = {}
    for path in sorted(paths):
        try:
            snapshot[str(path)] = digest(path)
        except FileNotFoundError:
            snapshot[str(path)] = None
    return snapshot


def provenance(repo, binary, manifest):
    build_environment(dict(os.environ), "unrecorded")
    current = clean_checkout(repo) | {"cargo_config_sha256": cargo_config_snapshot(repo)}
    build = json.loads(Path(manifest).read_text())
    for key in ("commit", "source_sha256", "cargo_lock_sha256", "cargo_config_sha256"):
        if build.get(key) != current[key]:
            raise ValueError(f"build provenance mismatch: {key}")
    if (build.get("profile") != "release" or not build.get("provider")
            or not isinstance(build.get("features"), list) or not build.get("build_command")
            or build.get("binary_sha256") != digest(binary)):
        raise ValueError("invalid release build provenance or binary hash")
    return current | {"binary_sha256": digest(binary), "build_sha256": digest(manifest),
                      "binary": str(binary), "build": build}


def check_immutable(repo, binary, manifest, frozen):
    if provenance(repo, binary, manifest) != frozen:
        raise ValueError("checkout, build record or binary changed during experiment")


def make_protocol(repetitions=5, prime_seconds=2.0, smoke=False, **limits):
    cases = [dict(family="fourier", size=n, terms=2 ** n, input_seed=seed)
             for n, seed in ((4, 0), (5, 0), (6, 0), (6, 1), (6, 2))]
    cases += [dict(family=family, size=12, terms=n, input_seed=0)
              for family in ("smooth", "random", "mixed") for n in (8, 16, 32, 64)]
    if smoke:
        cases = [dict(family="fourier", size=2, terms=4, input_seed=0),
                 dict(family="mixed", size=4, terms=4, input_seed=0)]
    return dict(schema_version=1, cases=cases, repetitions=repetitions,
                prime_seconds=prime_seconds, smoke=smoke, prepare_only=False,
                relative_error_limit=1e-8, maxabs_error_limit=1e-8,
                cancellation_fraction_min=0.1, time_ratio_limit=0.8,
                max_relative_mad=0.20, max_idle_busy_fraction=0.02,
                max_external_busy_fraction=0.02, max_frequency_ratio=1.25,
                idle_seconds=0.5, timeout_seconds=900.0,
                bootstrap_samples=2000, bootstrap_seed=1127,
                selection="fastest median config with every repeat accurate; paired bootstrap reselects configs and pairwise order",
                thread_env=dict.fromkeys(THREADS, "1")) | limits


def configurations(case, prime_seconds):
    base = case | dict(aci_seed=0, tolerance=1e-10, cutoff=1e-20, prime_seconds=prime_seconds)
    for method in METHODS:
        values = (1e-8, 1e-10) if method == "aci" else ((None,) if method == "exact" else (1e-16, 1e-20))
        for value in values:
            config = base | {"method": method}
            if method == "aci":
                config["tolerance"] = value
            elif method != "exact":
                config["cutoff"] = value
            yield config


def schedule(protocol):
    for repetition in range(protocol["repetitions"]):
        for case in protocol["cases"]:
            arms = list(configurations(case, protocol["prime_seconds"]))
            # Reverse each round so an arm is not consistently first or last.
            if repetition % 2:
                arms.reverse()
            for config in arms:
                yield repetition, config


def parse_cpu_set(value):
    cpus = set()
    try:
        for part in value.split(","):
            bounds = part.split("-")
            if len(bounds) == 1:
                low = high = int(bounds[0])
            elif len(bounds) == 2:
                low, high = map(int, bounds)
            else:
                raise ValueError()
            if low < 0 or high < low:
                raise ValueError()
            cpus.update(range(low, high + 1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid CPU list: {value!r}") from error
    if not cpus:
        raise ValueError("CPU list must not be empty")
    return cpus


def validate_cpus(system, measured, observers, allowed, siblings):
    if system != "Linux":
        raise ValueError("Linux is required for measurement; use --analyze on other hosts")
    if measured < 0 or measured not in allowed or not observers or not observers <= allowed:
        raise ValueError("launch affinity must include measured + observer CPUs (use taskset on this launch only); nonempty observers required")
    if measured not in siblings or observers & siblings:
        raise ValueError("observers must exclude measured CPU and all its SMT siblings")


def status_affinity(path):
    for line in path.read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return parse_cpu_set(line.split(":", 1)[1].strip())
    raise ValueError(f"cannot verify affinity: {path}")


def audit_affinities(proc, observers, protected, worker_pid=None, worker_cpu=None):
    observations = []
    for folder in sorted(proc.iterdir()):
        if not folder.name.isdigit():
            continue
        try:
            command = (folder / "cmdline").read_bytes()
            if not command:
                # Empty argv also occurs in userspace; prove PF_KTHREAD or an
                # exited zombie from /proc/stat before excluding a task.
                fields = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
                if fields[0] == "Z" or int(fields[6]) & 0x00200000:
                    continue
            for task in (folder / "task").iterdir():
                affinity = status_affinity(task / "status")
                worker = int(folder.name) == worker_pid
                if worker and affinity != ({worker_cpu} if worker_cpu is not None else protected):
                    raise ValueError(f"worker thread {task.name} affinity {sorted(affinity)}")
                if not worker and affinity & protected:
                    raise ValueError(f"userspace thread {folder.name}/{task.name} can run on protected CPUs: {sorted(affinity)}")
                observations.append(dict(pid=int(folder.name), tid=int(task.name),
                                         executable=os.fsdecode(command.split(b"\0", 1)[0]),
                                         command_sha256=hashlib.sha256(command).hexdigest(),
                                         affinity=sorted(affinity)))
        except FileNotFoundError:
            if folder.exists():
                raise ValueError(f"cannot audit process/thread in {folder}")
        except PermissionError as error:
            raise ValueError(f"inaccessible /proc affinity audit: {folder}: {error}") from error
    if not observations:
        raise ValueError("no auditable userspace processes in /proc")
    return observations


def compact_affinity_audit(tasks, worker_pid, cpu, protected):
    workers = [task for task in tasks if task["pid"] == worker_pid]
    return dict(monotonic=time.monotonic(), tasks_sha256=hashlib.sha256(canonical(tasks).encode()).hexdigest(),
                task_count=len(tasks), worker_thread_count=len(workers),
                worker_cpus=sorted({core for task in workers for core in task["affinity"]}),
                other_protected_thread_count=sum(bool(set(task["affinity"]) & protected)
                                                 for task in tasks if task["pid"] != worker_pid))


def finite(value, name, minimum=0.0, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum or (positive and value == 0)):
        raise ValueError(f"invalid finite numeric field: {name}")


def validate_result(output, config, commit):
    try:
        result = json.loads(output)
        if not isinstance(result, dict):
            raise ValueError("worker result must be one JSON object")
        if (result["schema_version"] != 1 or result["config"] != config
                or result["build_commit"] != commit or result["release"] is not True
                or result["scalar"] != "c64" or result["thread_env"] != dict.fromkeys(THREADS, "1")
                or result["terms"] != config["terms"]
                or result["sites"] != config["size"] + (3 if config["family"] == "fourier" else 0)):
            raise ValueError("worker configuration/build/thread identity mismatch")
        finite(result["seconds"], "seconds", positive=True)
        if type(result["max_bond_dim"]) is not int or result["max_bond_dim"] < 1:
            raise ValueError("invalid max_bond_dim")
        for name in ("relative_error", "maxabs_relative_error", "maxabs_error",
                     "cancellation_fraction", "cancellation_maxabs_relative_error"):
            finite(result[name], name)
        if result["cancellation_fraction"] > 1:
            raise ValueError("invalid cancellation fraction")
        for name in ("reference", "checksum"):
            for field in CHECKSUM_FIELDS:
                value = result[name][field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"invalid {name}.{field}")
            finite(result[name]["sum_sq"], f"{name}.sum_sq")
            finite(result[name]["maxabs"], f"{name}.maxabs", positive=True)
        if config["method"] == "density":
            finite(result["discarded_weight"], "discarded_weight")
        elif result["discarded_weight"] is not None:
            raise ValueError("unexpected discarded weight")
        if (config["method"] == "aci" and not isinstance(result["aci"], dict)) or (config["method"] != "aci" and result["aci"] is not None):
            raise ValueError("invalid ACI diagnostics")
        # JSON permits NaN in Python; reject it in every nested diagnostic too.
        canonical(result)
        return result
    except (KeyError, TypeError, json.JSONDecodeError, OverflowError) as error:
        raise ValueError(f"malformed worker result: {error}") from error


def relative_mad(values):
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values) / median


def accurate(result, protocol):
    return (result["relative_error"] <= protocol["relative_error_limit"]
            and result["maxabs_relative_error"] <= protocol["maxabs_error_limit"]
            and result["cancellation_maxabs_relative_error"] <= protocol["maxabs_error_limit"])


def choose(arms, indices):
    selected = min(arms, key=lambda arm: statistics.median(arm["seconds"][i] for i in indices))
    return selected, statistics.median(selected["seconds"][i] for i in indices)


def ratio_intervals(pools, protocol):
    rng = random.Random(protocol["bootstrap_seed"])
    ratios = {"pairwise": [], "density": []}
    n = protocol["repetitions"]
    for _ in range(protocol["bootstrap_samples"]):
        indices = rng.choices(range(n), k=n)
        numerator = choose(pools["aci"], indices)[1]
        pairwise = min(choose(pools[method], indices)[1] for method in ("sequential", "balanced"))
        ratios["pairwise"].append(numerator / pairwise)
        ratios["density"].append(numerator / choose(pools["density"], indices)[1])
    intervals = {}
    for method, values in ratios.items():
        values.sort()
        intervals[method + "_ratio_ci95"] = [values[int(0.025 * len(values))],
                                             values[min(len(values) - 1, math.ceil(0.975 * len(values)) - 1)]]
    return intervals


def analyze(protocol, records):
    failures = list(protocol.get("validity_failures", []))
    defaults = make_protocol()
    if protocol["cases"] != defaults["cases"]:
        failures.append("frozen full case grid is required")
    if protocol["repetitions"] < 5 or protocol["prime_seconds"] < 2:
        failures.append("full gate requires >=5 repetitions and >=2s priming")
    for key in ("relative_error_limit", "maxabs_error_limit", "time_ratio_limit",
                "max_relative_mad", "max_idle_busy_fraction", "max_external_busy_fraction",
                "max_frequency_ratio"):
        if protocol[key] > defaults[key]:
            failures.append(f"full gate bound weakened: {key}")
    if protocol["cancellation_fraction_min"] < 0.1:
        failures.append("full cancellation gate weakened")
    if protocol["bootstrap_samples"] < 2000 or protocol["idle_seconds"] < 0.5:
        failures.append("full gate bootstrap/idle observation weakened")
    for key in ("prime_seconds", "relative_error_limit", "maxabs_error_limit", "time_ratio_limit",
                "max_relative_mad", "max_idle_busy_fraction", "max_external_busy_fraction", "idle_seconds"):
        finite(protocol[key], key, positive=True)
    expected = {(repetition, canonical(config)) for repetition, config in schedule(protocol)}
    indexed = {}
    for record in records:
        try:
            key = (record["repetition"], canonical(record["config"]))
            if key not in expected or key in indexed:
                raise ValueError("unexpected or duplicate sample")
            indexed[key] = record
            if record.get("error") or record.get("exit_code") != 0:
                raise ValueError(f"failed sample: {record.get('error', record.get('exit_code'))}")
            if record.get("validity_failures"):
                raise ValueError(f"host validity failure: {record['validity_failures']}")
            validate_result(canonical(record["result"]), record["config"], protocol["provenance"]["commit"])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(str(error))
    missing = len(expected - indexed.keys())
    if missing:
        failures.append(f"missing required samples: {missing}")
    if protocol.get("smoke") or protocol.get("prepare_only"):
        failures.append("smoke/prepare-only is not a full promotion experiment")
    summaries, principal, gate_failures = [], [], []
    for case in protocol["cases"]:
        pools = {method: [] for method in METHODS}
        references = set()
        cancellation_fractions = set()
        for config in configurations(case, protocol["prime_seconds"]):
            rows = [indexed.get((i, canonical(config))) for i in range(protocol["repetitions"])]
            if any(row is None or row.get("error") or "result" not in row for row in rows):
                continue
            try:
                results = [validate_result(canonical(row["result"]), config, protocol["provenance"]["commit"]) for row in rows]
                references.update(canonical(result["reference"]) for result in results)
                cancellation_fractions.update(result["cancellation_fraction"] for result in results)
                seconds = [result["seconds"] for result in results]
                dispersion = relative_mad(seconds)
                if dispersion > protocol["max_relative_mad"]:
                    failures.append(f"relative MAD {dispersion} exceeds bound: {canonical(config)}")
                arm = dict(config=config, seconds=seconds, median_seconds=statistics.median(seconds),
                           relative_mad=dispersion, accurate=all(accurate(r, protocol) for r in results),
                           max_relative_error=max(r["relative_error"] for r in results),
                           max_maxabs_relative_error=max(r["maxabs_relative_error"] for r in results),
                           minimum_cancellation_fraction=min(r["cancellation_fraction"] for r in results),
                           peak_rss_kib=max(row.get("peak_rss_kib", 0) for row in rows))
                summaries.append(arm)
                if arm["accurate"]:
                    pools[config["method"]].append(arm)
            except (ValueError, KeyError, TypeError) as error:
                failures.append(str(error))
        if len(references) > 1 or len(cancellation_fractions) > 1:
            failures.append(f"inconsistent input/oracle identity across methods/repetitions: {canonical(case)}")
        absent = [method for method, arms in pools.items() if not arms]
        if absent:
            failures.append(f"no complete accurate arm for {canonical(case)}: {absent}")
            continue
        selected = {method: choose(arms, range(protocol["repetitions"]))[0] for method, arms in pools.items()}
        if case["family"] == "fourier" and case["size"] == 6:
            row = dict(case=case, selected=selected) | ratio_intervals(pools, protocol)
            if min(arm["minimum_cancellation_fraction"] for arm in selected.values()) < protocol["cancellation_fraction_min"]:
                gate_failures.append(f"insufficient cancellation: {canonical(case)}")
            for comparison in ("pairwise", "density"):
                if row[comparison + "_ratio_ci95"][1] > protocol["time_ratio_limit"]:
                    gate_failures.append(f"insufficient speed versus {comparison}: {canonical(case)}")
            principal.append(row)
    if len(principal) != 3:
        failures.append("all three principal Fourier seeds are required")
    return dict(schema_version=1, status="INCONCLUSIVE" if failures else ("FAIL" if gate_failures else "PASS"),
                validity_failures=failures, gate_failures=gate_failures,
                expected_samples=len(expected), recorded_samples=len(records), arms=summaries,
                principal=principal)


def cpu_counters(cpus, proc=Path("/proc")):
    result = {}
    for line in (proc / "stat").read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith("cpu") and fields[0][3:].isdigit():
            cpu = int(fields[0][3:])
            if cpu in cpus:
                result[cpu] = list(map(int, fields[1:]))
    if result.keys() != cpus or any(len(row) < 8 for row in result.values()):
        raise ValueError("cannot read complete protected CPU counters")
    return result


def cpu_busy_fraction(before, after):
    # guest fields already contribute to user/nice; count only the first eight.
    delta = [right - left for left, right in zip(before[:8], after[:8])]
    total = sum(delta)
    if total <= 0 or min(delta) < 0:
        raise ValueError("unavailable or nonmonotonic CPU counters")
    return (total - delta[3] - delta[4]) / total


def frequencies(cpus):
    result = {}
    for cpu in sorted(cpus):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
        result[str(cpu)] = {}
        for key in ("scaling_cur_freq", "scaling_governor", "scaling_driver", "cpuinfo_max_freq"):
            path = root / key
            result[str(cpu)][key] = path.read_text().strip() if path.exists() else None
    return result


def frequency_values(snapshot, protected):
    expected = {str(cpu) for cpu in protected}
    if not isinstance(snapshot, dict) or set(snapshot) != expected:
        raise ValueError("complete protected-CPU frequency observations required")
    current = {}
    for cpu in sorted(expected):
        try:
            raw = snapshot[cpu]["scaling_cur_freq"]
            if isinstance(raw, bool):
                raise ValueError("boolean frequency")
            value = float(raw)
            finite(value, f"CPU {cpu} current frequency", positive=True)
            current[cpu] = value
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"positive numeric current frequency required for protected CPU {cpu}") from error
    return current


def host_noise_failures(record, protocol, frequency_bounds):
    protected = set(protocol["host"]["protected_cpus"])
    cpu = protocol["host"]["measured_cpu"]
    tick = protocol["host"]["clock_ticks_per_second"]
    finite(tick, "CPU clock ticks per second", positive=True)
    allowance = 2 / tick
    failures = []
    idle = record["idle_busy_fraction"]
    busy = {int(key): value for key, value in record["protected_busy_seconds"].items()}
    if set(map(int, idle)) != protected or set(busy) != protected:
        raise ValueError("complete protected-CPU idle and busy observations required")
    for value in idle.values():
        finite(value, "idle busy fraction")
        if value > protocol["max_idle_busy_fraction"]:
            failures.append("protected CPU idle gate failed")
    for value in busy.values():
        finite(value, "protected CPU busy seconds")
    wall = record["process_seconds"]
    budget = protocol["max_external_busy_fraction"] * wall
    external = max(0.0, busy[cpu] - record["user_seconds"] - record["system_seconds"] - allowance)
    if abs(record["external_busy_seconds"] - external) > 1e-9:
        failures.append("measured CPU external activity accounting mismatch")
    if external > budget:
        failures.append("external activity on measured CPU")
    for sibling in protected - {cpu}:
        if busy[sibling] > allowance + budget:
            failures.append(f"SMT sibling {sibling} busy during sample")
    limit = protocol["max_frequency_ratio"]
    finite(limit, "max_frequency_ratio", minimum=1.0)
    for endpoint in ("frequency_before", "frequency_after"):
        for core, value in frequency_values(record[endpoint], protected).items():
            low, high = frequency_bounds.get(core, (value, value))
            frequency_bounds[core] = (min(low, value), max(high, value))
    for core, (low, high) in frequency_bounds.items():
        if high / low > limit:
            failures.append(f"frequency drift on protected CPU {core}: run range {low:g}–{high:g} kHz exceeds {limit:g}x")
    return failures


def check_proc_visibility(proc=Path("/proc")):
    for line in (proc / "mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[1] == "/proc":
            if fields[2] != "proc" or any(option.startswith("hidepid=") and option != "hidepid=0" for option in fields[3].split(",")):
                raise ValueError("host-visible /proc without hidepid is required")
            break
    else:
        raise ValueError("cannot verify /proc mount visibility")
    # Linux initial PID namespace inode (PROC_PID_INIT_INO). A nested PID
    # namespace cannot prove that other host userspace stays off measured CPUs.
    # https://github.com/torvalds/linux/blob/master/include/uapi/linux/nsfs.h
    if (proc / "self/ns/pid").stat().st_ino != 4026531836:
        raise ValueError("initial host PID namespace required; container-only audit is insufficient")


def preflight(cpu, observers):
    if platform.system() != "Linux" or not hasattr(os, "sched_setaffinity") or not hasattr(os, "wait4"):
        raise ValueError("Linux with sched affinity and wait4 is required")
    cpu_root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    protected = parse_cpu_set((cpu_root / "thread_siblings_list").read_text().strip())
    initial_affinity = os.sched_getaffinity(0)
    validate_cpus(platform.system(), cpu, observers, initial_affinity, protected)
    check_proc_visibility()
    os.sched_setaffinity(0, observers)
    if os.sched_getaffinity(0) != observers:
        raise ValueError("runner observer affinity could not be set")
    observations = audit_affinities(Path("/proc"), observers, protected)
    topology = {}
    for sibling in sorted(protected):
        root = Path(f"/sys/devices/system/cpu/cpu{sibling}/topology")
        topology[str(sibling)] = {key: (root / key).read_text().strip()
                                 for key in ("thread_siblings_list", "core_id", "physical_package_id")}
    frequency = frequencies(protected)
    frequency_values(frequency, protected)
    return dict(measured_cpu=cpu, observer_cpus=sorted(observers), protected_cpus=sorted(protected),
                clock_ticks_per_second=os.sysconf("SC_CLK_TCK"),
                initial_affinity=sorted(initial_affinity), runner_affinity=sorted(os.sched_getaffinity(0)),
                affinity_audit=observations, topology=topology, kernel=platform.uname()._asdict(),
                cpuinfo=Path("/proc/cpuinfo").read_text(),
                proc_mounts=Path("/proc/mounts").read_text(),
                kernel_command_line=Path("/proc/cmdline").read_text(),
                frequencies=frequency,
                toolchain={name: subprocess.check_output([name, "-Vv" if name == "rustc" else "-V"], text=True).strip()
                           for name in ("rustc", "cargo")})


@contextmanager
def checkout_lock(repo):
    import fcntl
    lock = Path(git(repo, "rev-parse", "--git-path", "sum-benchmark.lock"))
    if not lock.is_absolute():
        lock = repo / lock
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another summation experiment owns this checkout lock") from error
        yield


def build_environment(inherited, commit):
    blocked = {key for key, value in inherited.items() if value and (
        key.startswith("CARGO_PROFILE_") or (key.startswith("CARGO_TARGET_") and key != "CARGO_TARGET_DIR")
        or key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
                   "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER",
                   "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTFLAGS"))}
    if blocked:
        raise ValueError(f"unsupported inherited build/profile overrides: {sorted(blocked)}")
    env = inherited | dict.fromkeys(THREADS, "1") | {"T4A_BENCH_GIT_COMMIT": commit}
    # Registry credentials remain available to Cargo, but never enter evidence.
    safe = (*THREADS, "T4A_BENCH_GIT_COMMIT", "CARGO_TARGET_DIR", "CARGO_BUILD_JOBS", "CARGO_INCREMENTAL",
            "RUSTUP_TOOLCHAIN", "CARGO_HOME", "RUSTUP_HOME")
    return env, {key: env[key] for key in safe if key in env}


def build_worker(repo, output):
    source = clean_checkout(repo) | {"cargo_config_sha256": cargo_config_snapshot(repo)}
    command = ["cargo", "build", "--locked", "--release", "-p", "tensor4all-benchmarks",
               "--example", "sum_benchmark", "--message-format=json"]
    env, recorded_environment = build_environment(dict(os.environ), source["commit"])
    with (output / "build.stdout").open("x") as stdout, (output / "build.stderr").open("x") as stderr:
        process = subprocess.run(command, cwd=repo, env=env, stdout=stdout, stderr=stderr, check=False)
    if process.returncode:
        raise ValueError(f"release build failed ({process.returncode}); see build.stderr")
    binaries = set()
    cargo_profile = None
    for line in (output / "build.stdout").read_text().splitlines():
        event = json.loads(line)
        if (event.get("reason") == "compiler-artifact" and event.get("target", {}).get("name") == "sum_benchmark"
                and "example" in event["target"].get("kind", []) and event.get("executable")):
            cargo_profile = event.get("profile", {})
            if (cargo_profile.get("debug_assertions") is not False or cargo_profile.get("opt_level") != "3"
                    or cargo_profile.get("overflow_checks") is not False):
                raise ValueError("Cargo profile is not the default optimized release profile")
            binaries.add(Path(event["executable"]).resolve())
    if len(binaries) != 1:
        raise ValueError("Cargo must identify exactly one sum_benchmark executable")
    binary = binaries.pop()
    if clean_checkout(repo) | {"cargo_config_sha256": cargo_config_snapshot(repo)} != source:
        raise ValueError("checkout or Cargo configuration changed during release build")
    build = source | dict(profile="release", provider="faer (Linux tenferro-cpu-platform default)",
                          features=["default"], build_command=command,
                          binary_sha256=digest(binary),
                          environment=recorded_environment, cargo_profile=cargo_profile,
                          build_stdout_sha256=digest(output / "build.stdout"),
                          build_stderr_sha256=digest(output / "build.stderr"))
    manifest = output / "build.json"
    write_json(manifest, build)
    return binary, manifest


def run_process(command, stem, env, timeout, before_exec=None, monitor=None, cwd=None):
    """Retain raw streams and reap exactly this PID with wait4 (Linux KiB RSS)."""
    failures, timed_out, status, usage = [], False, None, None
    started = time.monotonic()
    with Path(str(stem) + ".stdout").open("xb") as stdout, Path(str(stem) + ".stderr").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True, preexec_fn=before_exec)
        try:
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    break
                if time.monotonic() - started > timeout:
                    timed_out = True
                    failures.append(f"worker timeout after {timeout}s")
                    os.killpg(process.pid, signal.SIGKILL)
                    _, status, usage = os.wait4(process.pid, 0)
                    break
                if monitor is not None:
                    try:
                        monitor(process.pid)
                    except (ValueError, OSError) as error:
                        failures.append(str(error))
                        os.killpg(process.pid, signal.SIGKILL)
                        _, status, usage = os.wait4(process.pid, 0)
                        break
                time.sleep(0.05)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, usage = os.wait4(process.pid, 0)
            raise
        finally:
            if status is not None:
                process.returncode = os.waitstatus_to_exitcode(status)
    # Darwin reports bytes; the helper is unit-tested there, but measurement
    # preflight still requires Linux, whose ru_maxrss is KiB.
    peak = usage.ru_maxrss / (1024 if platform.system() == "Darwin" else 1)
    return dict(pid=process.pid, exit_code=process.returncode, peak_rss_kib=peak,
                user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                process_seconds=time.monotonic() - started, timed_out=timed_out,
                validity_failures=failures)


def observe(binary, config, repetition, index, output, protocol, repo, manifest, frequency_bounds):
    cpu = protocol["host"]["measured_cpu"]
    protected = set(protocol["host"]["protected_cpus"])
    observers = set(protocol["host"]["observer_cpus"])
    record = dict(index=index, repetition=repetition, config=config, validity_failures=[])
    stem = output / "raw" / f"{index:04d}"
    check_immutable(repo, binary, manifest, protocol["provenance"])
    idle_before = cpu_counters(protected)
    audit_before = audit_affinities(Path("/proc"), observers, protected)
    time.sleep(protocol["idle_seconds"])
    idle_after = cpu_counters(protected)
    idle_busy = {str(key): cpu_busy_fraction(idle_before[key], idle_after[key]) for key in protected}
    record["idle_busy_fraction"] = idle_busy
    record["affinity_audit_before"] = audit_before
    record["frequency_before"] = frequencies(protected)
    frequency_values(record["frequency_before"], protected)
    if any(value > protocol["max_idle_busy_fraction"] for value in idle_busy.values()):
        record["error"] = "protected CPU idle gate failed before launch"
        return record
    counters_before = cpu_counters(protected)
    worker_audits = []

    def monitor(pid):
        tasks = audit_affinities(Path("/proc"), observers, protected, worker_pid=pid, worker_cpu=cpu)
        # Audit every task at every poll; repeated full snapshots need not be retained.
        worker_audits.append(compact_affinity_audit(tasks, pid, cpu, protected))

    record.update(run_process([str(binary), canonical(config)], stem,
                             os.environ | protocol["thread_env"], protocol["timeout_seconds"],
                             before_exec=lambda: os.sched_setaffinity(0, {cpu}), monitor=monitor, cwd=repo))
    record["runtime_affinity_audits"] = worker_audits
    record["affinity_audit_after"] = audit_affinities(Path("/proc"), observers, protected)
    record["frequency_after"] = frequencies(protected)
    counters_after = cpu_counters(protected)
    tick = protocol["host"]["clock_ticks_per_second"]
    busy_seconds = {key: sum(right - left for i, (left, right) in enumerate(zip(counters_before[key][:8], counters_after[key][:8]))
                             if i not in (3, 4)) / tick for key in protected}
    child_cpu = record["user_seconds"] + record["system_seconds"]
    external = max(0.0, busy_seconds[cpu] - child_cpu - 2 / tick)
    record["protected_busy_seconds"] = busy_seconds
    record["external_busy_seconds"] = external
    record["validity_failures"].extend(host_noise_failures(record, protocol, frequency_bounds))
    record["stdout_sha256"] = digest(Path(str(stem) + ".stdout"))
    record["stderr_sha256"] = digest(Path(str(stem) + ".stderr"))
    if record["exit_code"]:
        record["error"] = f"worker exit {record['exit_code']} (raw stderr retained)"
    else:
        try:
            record["result"] = validate_result(Path(str(stem) + ".stdout").read_text(), config, protocol["provenance"]["commit"])
        except (ValueError, UnicodeError) as error:
            record["error"] = str(error)
    check_immutable(repo, binary, manifest, protocol["provenance"])
    return record


def verify_saved_host(protocol, records, completion):
    host = protocol["host"]
    cpu = host["measured_cpu"]
    protected, observers = set(host["protected_cpus"]), set(host["observer_cpus"])
    validate_cpus(host["kernel"]["system"], cpu, observers, set(host["initial_affinity"]), protected)
    if set(host["runner_affinity"]) != observers:
        raise ValueError("missing runner observer affinity proof")

    def audit(tasks, worker_pid=None):
        if not tasks:
            raise ValueError("missing host affinity audit")
        found = False
        for task in tasks:
            affinity = set(task["affinity"])
            if task["pid"] == worker_pid:
                found = True
                if affinity != {cpu}:
                    raise ValueError("saved worker thread affinity changed")
            elif affinity & protected:
                raise ValueError("saved userspace thread overlaps protected CPUs")
        return found

    audit(host["affinity_audit"])
    audit(completion["final_affinity_audit"])
    frequency_values(host["frequencies"], protected)
    frequency_bounds = {}
    for record in records:
        if record.get("error"):
            continue
        for key in ("peak_rss_kib", "process_seconds"):
            finite(record[key], key, positive=True)
        for key in ("user_seconds", "system_seconds", "external_busy_seconds"):
            finite(record[key], key)
        if record["timed_out"] or record["process_seconds"] < protocol["prime_seconds"]:
            raise ValueError("missing full per-child priming or terminated child")
        noise = host_noise_failures(record, protocol, frequency_bounds)
        if noise:
            raise ValueError(f"saved host noise gate failed: {noise}")
        audit(record["affinity_audit_before"])
        audit(record["affinity_audit_after"])
        observed, previous = False, -1.0
        for item in record["runtime_affinity_audits"]:
            finite(item["monotonic"], "runtime affinity timestamp")
            checksum = item["tasks_sha256"]
            if (not isinstance(checksum, str) or len(checksum) != 64
                    or any(char not in "0123456789abcdef" for char in checksum)):
                raise ValueError("invalid compact affinity snapshot digest")
            count, workers = item["task_count"], item["worker_thread_count"]
            if (type(count) is not int or type(workers) is not int or count < 1 or not 0 <= workers <= count
                    or item["other_protected_thread_count"] != 0 or item["monotonic"] < previous
                    or item["worker_cpus"] != ([cpu] if workers else [])):
                raise ValueError("invalid compact runtime affinity proof")
            previous = item["monotonic"]
            observed |= workers > 0
        if not observed:
            raise ValueError("missing measured worker thread affinity audit")


def verify_saved_build(protocol, build):
    provenance = protocol["provenance"]
    for key in ("commit", "source_sha256", "cargo_lock_sha256", "binary_sha256"):
        size = 40 if key == "commit" else 64
        value = provenance[key]
        if (not isinstance(value, str) or len(value) != size or any(c not in "0123456789abcdef" for c in value)
                or build[key] != value):
            raise ValueError(f"missing or inconsistent saved build provenance: {key}")
    configs = provenance["cargo_config_sha256"]
    if not isinstance(configs, dict) or not configs or configs != build["cargo_config_sha256"]:
        raise ValueError("missing or inconsistent saved Cargo configuration hashes")
    if (build["profile"] != "release" or not build["provider"] or build["features"] != ["default"]
            or build["build_command"] != ["cargo", "build", "--locked", "--release", "-p", "tensor4all-benchmarks",
                                          "--example", "sum_benchmark", "--message-format=json"]
            or build["cargo_profile"]["opt_level"] != "3"
            or build["cargo_profile"]["debug_assertions"] is not False
            or build["cargo_profile"]["overflow_checks"] is not False):
        raise ValueError("saved build is not the declared default optimized release")


def analyze_saved(output):
    protocol = json.loads((output / "protocol.json").read_text())
    records = []
    failures = []
    with (output / "samples.jsonl").open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                failures.append(f"malformed/truncated sample line {number}")
    try:
        completion = json.loads((output / "completion.json").read_text())
        failures.extend(completion["validity_failures"])
        if not completion.get("final_affinity_audit"):
            failures.append("missing final host affinity audit")
        if not completion["complete"] or completion["final_provenance"] != protocol["provenance"]:
            failures.append("incomplete run or missing immutable final provenance")
        for key, path in (("protocol_sha256", output / "protocol.json"), ("samples_sha256", output / "samples.jsonl")):
            if completion[key] != digest(path):
                failures.append(f"changed saved evidence: {path.name}")
        if protocol["provenance"]["build_sha256"] != digest(output / "build.json"):
            failures.append("changed/missing saved build provenance")
        verify_saved_build(protocol, json.loads((output / "build.json").read_text()))
        verify_saved_host(protocol, records, completion)
        for record in records:
            for stream in ("stdout", "stderr"):
                path = output / "raw" / f"{record['index']:04d}.{stream}"
                if record.get(stream + "_sha256") != digest(path):
                    failures.append(f"changed raw evidence: {path.name}")
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        failures.append(f"completion/evidence verification failed: {error}")
    protocol["validity_failures"] = protocol.get("validity_failures", []) + failures
    return analyze(protocol, records)


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--output", type=Path, help="new output directory; never overwritten")
    result.add_argument("--cpu", type=int, help="single measured logical CPU")
    result.add_argument("--observer-cpus", help="observer CPU list/ranges, e.g. 0-1,4-7")
    result.add_argument("--analyze", type=Path, metavar="DIR", help="regenerate summary from saved immutable evidence")
    result.add_argument("--prepare-only", action="store_true", help="build/freeze/preflight only; no timed samples")
    result.add_argument("--smoke", action="store_true", help="small plumbing grid; can never PASS the full gate")
    result.add_argument("--repetitions", type=int, default=5)
    for name, default in (("prime-seconds", 2.0), ("relative-error-limit", 1e-8),
                          ("maxabs-error-limit", 1e-8), ("time-ratio-limit", 0.8),
                          ("cancellation-fraction-min", 0.1), ("max-relative-mad", 0.20),
                          ("max-idle-busy-fraction", 0.02), ("max-external-busy-fraction", 0.02),
                          ("max-frequency-ratio", 1.25), ("idle-seconds", 0.5), ("timeout-seconds", 900.0)):
        result.add_argument("--" + name, type=float, default=default)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.analyze:
        try:
            summary = analyze_saved(args.analyze.resolve())
            write_json(args.analyze / "summary.json", summary, mode="w")
            print(summary["status"])
            return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[summary["status"]]
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"INCONCLUSIVE: cannot analyze saved run: {error}", file=sys.stderr)
            return 2
    output = None
    created_output = False
    protocol = None
    records = []
    try:
        if platform.system() != "Linux":
            raise ValueError("Linux is required for measurement; --analyze works on other platforms")
        build_environment(dict(os.environ), "unrecorded")
        if args.output is None or args.cpu is None or args.observer_cpus is None:
            raise ValueError("--output, --cpu and --observer-cpus are required for measurement")
        if args.repetitions < 1:
            raise ValueError("positive repetitions required")
        limits = {key: value for key, value in vars(args).items() if isinstance(value, float)}
        for key, value in limits.items():
            finite(value, key, positive=True)
        if args.max_frequency_ratio < 1 or any(value > 1 for key, value in limits.items() if key.endswith("fraction") or key == "cancellation_fraction_min"):
            raise ValueError("invalid frequency or fractional bound")
        observers = parse_cpu_set(args.observer_cpus)
        repo = Path(__file__).resolve().parent.parent
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()  # Exclusive creation preserves earlier evidence.
        created_output = True
        (output / "raw").mkdir()
        with checkout_lock(repo):
            host = preflight(args.cpu, observers)
            binary, manifest = build_worker(repo, output)
            protocol = make_protocol(args.repetitions, args.prime_seconds, args.smoke, **{key: value for key, value in limits.items() if key != "prime_seconds"})
            protocol.update(prepare_only=args.prepare_only, host=host,
                            started_utc=datetime.now(timezone.utc).isoformat(),
                            provenance=provenance(repo, binary, manifest))
            write_json(output / "protocol.json", protocol)
            frozen_protocol_hash = digest(output / "protocol.json")
            frequency_bounds = {}
            with (output / "samples.jsonl").open("x") as samples:
                if not args.prepare_only:
                    for index, (repetition, config) in enumerate(schedule(protocol)):
                        if digest(output / "protocol.json") != frozen_protocol_hash:
                            raise ValueError("frozen protocol changed during measurement")
                        try:
                            record = observe(binary, config, repetition, index, output, protocol, repo, manifest,
                                             frequency_bounds)
                        except (OSError, ValueError, subprocess.SubprocessError) as error:
                            record = dict(index=index, repetition=repetition, config=config, error=str(error), validity_failures=[str(error)])
                        records.append(record)
                        samples.write(canonical(record) + "\n")
                        samples.flush()
                        os.fsync(samples.fileno())
                        print(f"{index + 1}/{len(protocol['cases']) * 11 * args.repetitions} {config['family']} {config['method']} {'invalid' if record.get('error') or record['validity_failures'] else 'recorded'}", flush=True)
                        if record.get("error") or record["validity_failures"]:
                            break  # No selective retries; the missing remainder is inconclusive.
            check_immutable(repo, binary, manifest, protocol["provenance"])
            if digest(output / "protocol.json") != frozen_protocol_hash:
                raise ValueError("frozen protocol changed during measurement")
            final_audit = audit_affinities(Path("/proc"), observers, set(host["protected_cpus"]))
            write_json(output / "completion.json", dict(complete=not args.prepare_only and len(records) == len(list(schedule(protocol))),
                final_provenance=provenance(repo, binary, manifest), final_affinity_audit=final_audit,
                validity_failures=[], protocol_sha256=digest(output / "protocol.json"),
                samples_sha256=digest(output / "samples.jsonl"), completed_utc=datetime.now(timezone.utc).isoformat()))
            summary = analyze_saved(output)
            write_json(output / "summary.json", summary)
            print(summary["status"])
            return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[summary["status"]]
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        failure = dict(status="INCONCLUSIVE", error=str(error) or "interrupted",
                       recorded_samples=len(records), completed_utc=datetime.now(timezone.utc).isoformat())
        if created_output and not (output / "summary.json").exists():
            write_json(output / "failure.json", failure)
            write_json(output / "summary.json", failure)
        print(f"INCONCLUSIVE: {failure['error']}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
