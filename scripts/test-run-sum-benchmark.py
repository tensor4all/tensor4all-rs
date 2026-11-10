#!/usr/bin/env python3
"""Focused regression tests for the fail-closed Linux summation harness."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("run-sum-benchmark.py")
SPEC = importlib.util.spec_from_file_location("sum_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def worker_result(config, seconds=1.0, error=1e-12):
    checksum = dict(sum_re=1.0, sum_im=0.0, sum_sq=1.0, maxabs=1.0,
                    weighted_re=1.0, weighted_im=0.0)
    return dict(schema_version=1, config=config, build_commit="a" * 40,
                release=True, scalar="c64", sites=config["size"] + 3 if
                config["family"] == "fourier" else config["size"], terms=config["terms"],
                seconds=seconds, max_bond_dim=4, relative_error=error,
                maxabs_relative_error=error, maxabs_error=error,
                cancellation_fraction=0.5, cancellation_maxabs_relative_error=error,
                discarded_weight=0.0 if config["method"] == "density" else None,
                aci={} if config["method"] == "aci" else None,
                reference=checksum, checksum=checksum.copy(),
                thread_env={key: "1" for key in MODULE.THREADS})


def report_fixture(smoke=False):
    protocol = MODULE.make_protocol(repetitions=5, prime_seconds=2.0, smoke=smoke)
    protocol["provenance"] = {"commit": "a" * 40}
    records = []
    for repetition, config in MODULE.schedule(protocol):
        seconds = 0.5 if config["method"] == "aci" else 1.0
        records.append(dict(repetition=repetition, config=config,
                            result=worker_result(config, seconds), validity_failures=[],
                            peak_rss_kib=1024, exit_code=0))
    return protocol, records


class ProtocolTests(unittest.TestCase):
    def test_frozen_grid_and_alternation(self):
        protocol = MODULE.make_protocol()
        self.assertEqual(len(protocol["cases"]), 17)
        self.assertEqual(len(list(MODULE.schedule(protocol))), 935)
        self.assertEqual({(c["family"], c["size"], c["input_seed"]) for c in
                          protocol["cases"] if c["family"] == "fourier"},
                         {("fourier", 4, 0), ("fourier", 5, 0),
                          ("fourier", 6, 0), ("fourier", 6, 1), ("fourier", 6, 2)})
        rows = list(MODULE.schedule(protocol))
        first = [c["method"] for r, c in rows if r == 0][:11]
        second = [c["method"] for r, c in rows if r == 1][:11]
        self.assertEqual(second, list(reversed(first)))

    def test_cpu_constraints_and_smt_safety(self):
        for system, measured, observers, allowed, siblings in [
            ("Darwin", 2, {0}, {0, 2}, {2}),
            ("Linux", 2, {2}, {0, 2}, {2}),
            ("Linux", 2, set(), {0, 2}, {2}),
            ("Linux", 2, {0}, {0}, {2}),
            ("Linux", 2, {0}, {0, 2}, {0, 2}),
        ]:
            with self.subTest(system=system, observers=observers):
                with self.assertRaises(ValueError):
                    MODULE.validate_cpus(system, measured, observers, allowed, siblings)
        MODULE.validate_cpus("Linux", 2, {0, 1}, {0, 1, 2, 3}, {2, 3})
        self.assertEqual(MODULE.parse_cpu_set("0-2,5"), {0, 1, 2, 5})
        for bad in ("", "2-1", "-1", "x"):
            with self.assertRaises(ValueError):
                MODULE.parse_cpu_set(bad)

    def test_proc_audit_checks_all_threads_and_worker_affinity(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            for pid, cpus in ((100, "0-1"), (101, "2")):
                folder = proc / str(pid)
                (folder / "task" / str(pid)).mkdir(parents=True)
                (folder / "cmdline").write_bytes(b"agent\0")
                (folder / "task" / str(pid) / "status").write_text(
                    f"Name:\tagent\nCpus_allowed_list:\t{cpus}\n")
            self.assertEqual(len(MODULE.audit_affinities(proc, {0, 1}, {2}, 101)), 2)
            with self.assertRaises(ValueError):
                MODULE.audit_affinities(proc, {0, 1}, {2})
            thread = proc / "100/task/102"
            thread.mkdir()
            (thread / "status").write_text("Cpus_allowed_list:\t0-2\n")
            with self.assertRaises(ValueError):
                MODULE.audit_affinities(proc, {0, 1}, {2}, 101)


    def test_proc_audit_does_not_mistake_empty_userspace_for_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            folder = proc / "100"
            task = folder / "task/100"
            task.mkdir(parents=True)
            (folder / "cmdline").write_bytes(b"")
            (folder / "stat").write_text("100 (empty argv) S 1 1 1 0 0 0 0\n")
            (task / "status").write_text("Cpus_allowed_list:\t0-2\n")
            allowed = proc / "101/task/101"
            allowed.mkdir(parents=True)
            (proc / "101/cmdline").write_bytes(b"observer\0")
            (allowed / "status").write_text("Cpus_allowed_list:\t0-1\n")
            with self.assertRaises(ValueError):
                MODULE.audit_affinities(proc, {0, 1}, {2})

    def test_worker_output_is_strict_and_finite(self):
        config = next(MODULE.schedule(MODULE.make_protocol()))[1]
        good = worker_result(config)
        self.assertEqual(MODULE.validate_result(json.dumps(good), config, "a" * 40), good)
        for field, value in (("seconds", 0), ("seconds", float("nan")),
                             ("release", False), ("terms", 7),
                             ("build_commit", "unrecorded"), ("relative_error", -1),
                             ("reference", {}), ("thread_env", {})):
            with self.subTest(field=field, value=value):
                bad = good | {field: value}
                with self.assertRaises(ValueError):
                    MODULE.validate_result(json.dumps(bad), config, "a" * 40)
        for bad in ("{}", "[]", "garbage", json.dumps(good) + "\n{}"):
            with self.assertRaises(ValueError):
                MODULE.validate_result(bad, config, "a" * 40)


class AnalysisTests(unittest.TestCase):

    def test_saved_run_requires_completion_build_and_unchanged_raw_evidence(self):
        protocol, records = report_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            build = dict(commit="a" * 40, source_sha256="b" * 64, cargo_lock_sha256="c" * 64,
                         binary_sha256="d" * 64, cargo_config_sha256={"/fixture/.cargo/config.toml": None},
                         profile="release", provider="faer", features=["default"],
                         build_command=["cargo", "build", "--locked", "--release", "-p", "tensor4all-benchmarks",
                                        "--example", "sum_benchmark", "--message-format=json"],
                         cargo_profile=dict(opt_level="3", debug_assertions=False, overflow_checks=False))
            MODULE.write_json(root / "build.json", build)
            protocol["provenance"] |= {key: build[key] for key in ("source_sha256", "cargo_lock_sha256", "binary_sha256", "cargo_config_sha256")}
            protocol["provenance"]["build_sha256"] = MODULE.digest(root / "build.json")
            audit = [dict(pid=100, tid=100, affinity=[0])]
            protocol["host"] = dict(measured_cpu=2, observer_cpus=[0], protected_cpus=[2, 3],
                                    initial_affinity=[0, 2], runner_affinity=[0], affinity_audit=audit,
                                    kernel=dict(system="Linux"), clock_ticks_per_second=100,
                                    frequencies={str(cpu): {"scaling_cur_freq": "2000000"} for cpu in (2, 3)})
            for index, record in enumerate(records):
                record.update(index=index, pid=101, process_seconds=2.1, user_seconds=2.0, system_seconds=0.0,
                              timed_out=False, idle_busy_fraction={"2": 0.0, "3": 0.0},
                              external_busy_seconds=0.0, protected_busy_seconds={"2": 2.0, "3": 0.0},
                              affinity_audit_before=audit, affinity_audit_after=audit,
                              frequency_before={str(cpu): {"scaling_cur_freq": "2000000"} for cpu in (2, 3)},
                              frequency_after={str(cpu): {"scaling_cur_freq": "2000000"} for cpu in (2, 3)},
                              runtime_affinity_audits=[dict(monotonic=1.0, tasks_sha256="e" * 64,
                                  task_count=2, worker_thread_count=1, worker_cpus=[2], other_protected_thread_count=0)])
                for stream, text in (("stdout", json.dumps(record["result"])), ("stderr", "")):
                    path = root / "raw" / f"{index:04d}.{stream}"
                    path.write_text(text)
                    record[stream + "_sha256"] = MODULE.digest(path)
            MODULE.write_json(root / "protocol.json", protocol)
            (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            self.assertEqual(MODULE.analyze_saved(root)["status"], "INCONCLUSIVE")
            completion = dict(complete=True, final_provenance=protocol["provenance"], final_affinity_audit=[dict(pid=100, tid=100, affinity=[0])], validity_failures=[],
                protocol_sha256=MODULE.digest(root / "protocol.json"), samples_sha256=MODULE.digest(root / "samples.jsonl"))
            MODULE.write_json(root / "completion.json", completion)
            self.assertEqual(MODULE.analyze_saved(root)["status"], "PASS")
            for key, value in (
                ("frequency_before", {}),
                ("frequency_after", {"2": {"scaling_cur_freq": "2000000"}, "3": {"scaling_cur_freq": None}}),
                ("frequency_after", {"2": {"scaling_cur_freq": "2000000"}, "3": {"scaling_cur_freq": "4000000"}}),
                ("protected_busy_seconds", {"2": 2.0}),
                ("protected_busy_seconds", {"2": 2.0, "3": 1.0}),
                ("protected_busy_seconds", {"2": 20.0, "3": 0.0}),
                ("runtime_affinity_audits", [dict(monotonic=1.0, tasks_sha256="e" * 64, task_count=2,
                    worker_thread_count=1, worker_cpus=[3], other_protected_thread_count=0)]),
            ):
                with self.subTest(key=key, value=value), self.assertRaises((KeyError, ValueError)):
                    MODULE.verify_saved_host(protocol, [records[0] | {key: value}], completion)
            missing_audit = records[0].pop("runtime_affinity_audits")
            (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            completion["samples_sha256"] = MODULE.digest(root / "samples.jsonl")
            MODULE.write_json(root / "completion.json", completion, mode="w")
            self.assertEqual(MODULE.analyze_saved(root)["status"], "INCONCLUSIVE")
            records[0]["runtime_affinity_audits"] = missing_audit
            (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            completion["samples_sha256"] = MODULE.digest(root / "samples.jsonl")
            MODULE.write_json(root / "completion.json", completion, mode="w")
            (root / "raw/0000.stdout").write_text("corrupted")
            self.assertEqual(MODULE.analyze_saved(root)["status"], "INCONCLUSIVE")

    def test_all_gates_pass_and_smoke_is_not_promotion(self):
        protocol, records = report_fixture()
        summary = MODULE.analyze(protocol, records)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(len(summary["principal"]), 3)
        self.assertEqual(summary["principal"][0]["pairwise_ratio_ci95"], [0.5, 0.5])
        protocol, records = report_fixture(smoke=True)
        self.assertEqual(MODULE.analyze(protocol, records)["status"], "INCONCLUSIVE")

    def test_missing_failed_duplicate_and_no_accurate_arm_are_inconclusive(self):
        protocol, records = report_fixture()
        missing_density = [r for r in records if r["config"]["method"] != "density"]
        failed = copy.deepcopy(records)
        failed[0] = failed[0] | {"error": "worker failed", "exit_code": 1}
        inaccurate = copy.deepcopy(records)
        for record in inaccurate:
            if record["config"]["method"] == "balanced":
                record["result"]["relative_error"] = 1.0
        for rows in (records[:-1], missing_density, failed, records + [records[0]], inaccurate):
            with self.subTest(count=len(rows)):
                self.assertEqual(MODULE.analyze(protocol, rows)["status"], "INCONCLUSIVE")

    def test_fastest_accurate_configuration_requires_all_its_repeats(self):
        protocol, records = report_fixture()
        for record in records:
            if record["config"]["method"] == "aci":
                loose = record["config"]["tolerance"] == 1e-8
                record["result"]["seconds"] = 0.1 if loose else 0.6
                if loose and record["repetition"] == 4:
                    record["result"]["relative_error"] = 1.0
        summary = MODULE.analyze(protocol, records)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["principal"][0]["selected"]["aci"]["config"]["tolerance"], 1e-10)
        self.assertEqual(summary["principal"][0]["pairwise_ratio_ci95"], [0.6, 0.6])


    def test_reference_identity_and_cancellation_definition_are_shared(self):
        protocol, records = report_fixture()
        records[0]["result"]["reference"]["weighted_re"] = 100.0
        self.assertEqual(MODULE.analyze(protocol, records)["status"], "INCONCLUSIVE")
        protocol, records = report_fixture()
        records[0]["result"]["cancellation_fraction"] = 0.99
        self.assertEqual(MODULE.analyze(protocol, records)["status"], "INCONCLUSIVE")

    def test_fastest_pairwise_both_orders_and_insufficient_speed_fail(self):
        protocol, records = report_fixture()
        for record in records:
            if record["config"]["method"] == "balanced":
                record["result"]["seconds"] = 0.55
        self.assertEqual(MODULE.analyze(protocol, records)["status"], "FAIL")
        protocol, records = report_fixture()
        for record in records:
            record["result"]["cancellation_fraction"] = 0.05
        self.assertEqual(MODULE.analyze(protocol, records)["status"], "FAIL")


    def test_saved_analysis_rejects_weakened_or_truncated_protocol(self):
        protocol, records = report_fixture()
        protocol["cases"] = protocol["cases"][:1]
        records = [r for r in records if r["config"]["family"] == "fourier" and r["config"]["size"] == 4]
        self.assertEqual(MODULE.analyze(protocol, records)["status"], "INCONCLUSIVE")
        protocol, records = report_fixture()
        protocol["repetitions"] = 1
        records = [r for r in records if r["repetition"] == 0]
        self.assertEqual(MODULE.analyze(protocol, records)["status"], "INCONCLUSIVE")

    def test_host_noise_and_invalid_output_cannot_pass(self):
        protocol, records = report_fixture()
        records[0]["validity_failures"] = ["affinity changed"]
        self.assertEqual(MODULE.analyze(protocol, records)["status"], "INCONCLUSIVE")
        protocol, records = report_fixture()
        records[0]["result"]["checksum"]["sum_sq"] = float("inf")
        self.assertEqual(MODULE.analyze(protocol, records)["status"], "INCONCLUSIVE")
        protocol, records = report_fixture()
        for record in records:
            if record["repetition"] in (0, 1):
                record["result"]["seconds"] *= 10
        protocol["max_relative_mad"] = 0.01
        # Two outliers are retained; MAD is robust, but a deliberately spread
        # central triplet must invalidate the experiment.
        for record in records:
            record["result"]["seconds"] *= (1 + record["repetition"])
        self.assertEqual(MODULE.analyze(protocol, records)["status"], "INCONCLUSIVE")



class ProcessTests(unittest.TestCase):
    def test_wait4_raw_logs_thread_environment_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "worker.py"
            binary.write_text("import os, json, sys\n"
                "print(json.dumps({k: os.getenv(k) for k in sys.argv[1:]}))\n"
                "print('diagnostic retained', file=sys.stderr)\n")
            result = MODULE.run_process([sys.executable, str(binary), *MODULE.THREADS],
                root / "sample", os.environ | dict.fromkeys(MODULE.THREADS, "1"), timeout=5)
            self.assertEqual(result["exit_code"], 0)
            self.assertGreater(result["peak_rss_kib"], 0)
            self.assertEqual(json.loads((root / "sample.stdout").read_text()), dict.fromkeys(MODULE.THREADS, "1"))
            self.assertIn("diagnostic retained", (root / "sample.stderr").read_text())
            tasks = [dict(pid=100, tid=i, affinity=[0], executable="observer" + "x" * 100) for i in range(500)]
            tasks += [dict(pid=101, tid=501, affinity=[2]), dict(pid=101, tid=502, affinity=[2])]
            compact = MODULE.compact_affinity_audit(tasks, 101, 2, {2, 3})
            self.assertEqual(compact["task_count"], 502)
            self.assertEqual(compact["worker_thread_count"], 2)
            self.assertEqual(compact["worker_cpus"], [2])
            self.assertLess(len(json.dumps(compact)), 512)
            result = MODULE.run_process([sys.executable, "-c", "import time; time.sleep(10)"],
                root / "timeout", os.environ.copy(), timeout=0.05)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["exit_code"], 0)

    def test_idle_counter_accounting_and_external_noise(self):
        before = {2: [100, 0, 0, 900, 0, 0, 0, 0, 0, 0]}
        after = {2: [110, 0, 0, 990, 0, 0, 0, 0, 0, 0]}
        self.assertAlmostEqual(MODULE.cpu_busy_fraction(before[2], after[2]), 0.1)
        self.assertGreater(MODULE.cpu_busy_fraction(before[2], after[2]), 0.02)
        with self.assertRaises(ValueError):
            MODULE.cpu_busy_fraction(before[2], before[2])
        frequency = {"2": {"scaling_cur_freq": "2000000"}, "3": {"scaling_cur_freq": "2000000"}}
        self.assertEqual(MODULE.frequency_ratios(frequency, frequency, {2, 3}), {"2": 1.0, "3": 1.0})
        for missing in ({}, {"2": {"scaling_cur_freq": None}},
                        {"2": {"scaling_cur_freq": "0"}, "3": {"scaling_cur_freq": "2000000"}}):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                MODULE.frequency_ratios(missing, frequency, {2, 3})


class ProvenanceTests(unittest.TestCase):

    def test_build_environment_refuses_overrides_and_never_records_tokens(self):
        for key in ("CARGO_PROFILE_RELEASE_OPT_LEVEL", "CARGO_BUILD_TARGET", "RUSTFLAGS", "RUSTC_WRAPPER",
                    "LD_PRELOAD", "LD_LIBRARY_PATH"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.build_environment({key: "override"}, "a" * 40)
        env, recorded = MODULE.build_environment({"CARGO_REGISTRIES_PRIVATE_TOKEN": "secret",
                                                   "CARGO_TARGET_DIR": "/tmp/target"}, "a" * 40)
        self.assertEqual(env["CARGO_REGISTRIES_PRIVATE_TOKEN"], "secret")
        self.assertNotIn("secret", json.dumps(recorded))
        self.assertEqual(recorded["CARGO_TARGET_DIR"], "/tmp/target")

    def test_existing_output_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            (output / "evidence").write_text("original")
            with patch.object(MODULE.platform, "system", return_value="Linux"):
                code = MODULE.main(["--output", str(output), "--cpu", "2", "--observer-cpus", "0"])
            self.assertEqual(code, 2)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["evidence"])

    def test_checkout_binary_and_manifest_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as cargo_directory, \
                patch.dict(os.environ, {"CARGO_HOME": cargo_directory}):
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            source = repo / "Cargo.lock"
            source.write_text("version = 4\n")
            subprocess.run(["git", "-C", str(repo), "add", "Cargo.lock"], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                            "user.email=test@example.com", "commit", "-qm", "fixture"], check=True)
            binary = repo.parent / (repo.name + "-worker")
            manifest = repo.parent / (repo.name + "-build.json")
            try:
                binary.write_bytes(b"binary")
                commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
                manifest.write_text(json.dumps(dict(commit=commit, profile="release",
                    provider="faer", features=["tenferro-cpu-platform"],
                    build_command=["cargo", "build", "--release", "--locked"],
                    binary_sha256=MODULE.digest(binary), source_sha256=MODULE.source_digest(repo),
                    cargo_lock_sha256=MODULE.digest(source),
                    cargo_config_sha256=MODULE.cargo_config_snapshot(repo))))
                frozen = MODULE.provenance(repo, binary, manifest)
                self.assertEqual(frozen["commit"], commit)
                MODULE.check_immutable(repo, binary, manifest, frozen)
                config = Path(cargo_directory) / "config.toml"
                self.assertIsNone(frozen["cargo_config_sha256"][str(config.resolve())])
                original_manifest = manifest.read_text()
                config.write_text('[build]\njobs = 1\n# do-not-record\n')
                with self.assertRaises(ValueError):
                    MODULE.check_immutable(repo, binary, manifest, frozen)
                configured = json.loads(original_manifest)
                configured["cargo_config_sha256"] = MODULE.cargo_config_snapshot(repo)
                self.assertNotIn("do-not-record", json.dumps(configured))
                manifest.write_text(json.dumps(configured))
                frozen_configured = MODULE.provenance(repo, binary, manifest)
                config.write_text('[build]\njobs = 2\n# changed-config\n')
                with self.assertRaises(ValueError):
                    MODULE.check_immutable(repo, binary, manifest, frozen_configured)
                config.unlink()
                manifest.write_text(original_manifest)
                MODULE.check_immutable(repo, binary, manifest, frozen)
                binary.write_bytes(b"replaced")
                with self.assertRaises(ValueError):
                    MODULE.check_immutable(repo, binary, manifest, frozen)
                binary.write_bytes(b"binary")
                source.write_text("changed\n")
                with self.assertRaises(ValueError):
                    MODULE.provenance(repo, binary, manifest)
            finally:
                binary.unlink(missing_ok=True)
                manifest.unlink(missing_ok=True)

    def test_cli_help_and_platform_refusal(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--observer-cpus", result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-create"
            with patch.object(MODULE.platform, "system", return_value="Darwin"):
                self.assertEqual(MODULE.main(["--output", str(output), "--cpu", "2", "--observer-cpus", "0"]), 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
