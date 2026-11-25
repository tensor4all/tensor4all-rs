#!/usr/bin/env python3
"""Build and compare three locked workspace variants in a disposable checkout."""
import argparse
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess
import tempfile

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
CRATE = Path("crates/tensor4all-tensorci")
SOURCE = CRATE / "src/tensorci2.rs"
MANIFEST = CRATE / "Cargo.toml"
EXAMPLE = CRATE / "examples/issue1460_sampling.rs"
PRODUCTION = [SOURCE, MANIFEST, Path("Cargo.lock")]
CELLS = [(size, strategy) for size in (6, 10) for strategy in ("full", "rook")]


def git(*args):
    return subprocess.check_output(["git", "-C", ROOT, *args])


def instrument(source):
    # Match only index preparation immediately before these four f calls.
    pattern = re.compile(
        r"(?m)^( +)((?:let mut full_idx = i_multi\.clone\(\);\n"
        r" +full_idx\.extend\(j_multi\.iter\(\)\.cloned\(\)\);|"
        r"full_idx\.clear\(\);\n +full_idx\.extend_from_slice\(i_multi\);\n"
        r" +full_idx\.extend_from_slice\(j_multi\);))\n"
        r"(?= +(?:let val(?:ue)? = |pi1\[\[i, j\]\] = |p_mat\[\[i, j\]\] = )"
        r"f\(&full_idx\);)"
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 4:
        raise ValueError(f"expected four scalar preparation sites, got {len(matches)}")
    for slot, match in reversed(list(enumerate(matches))):
        indent, body = match.groups()
        baseline = body.startswith("let mut")
        binding = "let full_idx = " if baseline else ""
        lines = [f"{binding}allocation_probe::measure({slot}, || {{"]
        lines.extend("    " + line.lstrip() for line in body.splitlines())
        if baseline:
            lines.append("    full_idx")
        lines.append("});")
        source = source[:match.start()] + "".join(indent + line + "\n" for line in lines) + source[match.end():]
    return source + '\n#[path = "tensorci2/allocation_probe.rs"]\npub mod allocation_probe;\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="89a4c1a0fac426b31710d1e91bac6e76348ef25e")
    parser.add_argument("--target-dir", type=Path, help="Reuse an existing Cargo target directory")
    parser.add_argument("--release-tests", action="store_true", help="Validate the patched sampling and batch tests after the experiment")
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=False)
    base = git("rev-parse", args.base).decode().strip()
    changed = git("diff", base, "--", *map(str, PRODUCTION))
    source_diff = git("diff", base, "--", str(SOURCE))
    header, *hunks = re.split(rb"(?m)(?=^@@ )", source_diff)
    reuse_hunks = [hunk for hunk in hunks if b"+ " in hunk and b"full_idx.clear();" in hunk]
    if len(reuse_hunks) != 4:
        raise ValueError("expected four separate buffer-reuse hunks")
    reuse = header + b"".join(reuse_hunks)
    for name, patch in (("A", reuse), ("AB", changed)):
        (results / f"{name}.patch").write_bytes(patch)
    (results / "base.txt").write_text(base + "\n")
    (results / "rustc.txt").write_bytes(subprocess.check_output(["rustc", "-Vv"]))

    with tempfile.TemporaryDirectory(prefix="tci2-1460-") as temporary:
        work = Path(temporary)
        target = args.target_dir.resolve() if args.target_dir else work / "target"
        archive = work / "base.tar"
        subprocess.run(["git", "-C", ROOT, "archive", "-o", archive, base], check=True)
        subprocess.run(["tar", "-xf", archive, "-C", work], check=True)
        archive.unlink()
        original = {path: (work / path).read_bytes() for path in PRODUCTION}
        example = (EVIDENCE / "sampling.rs").read_text()
        (work / EXAMPLE).parent.mkdir(exist_ok=True)
        shutil.copyfile(EVIDENCE / "allocation_probe.rs", work / CRATE / "src/tensorci2/allocation_probe.rs")

        def build(name, patch, probe=False):
            for path, data in original.items():
                (work / path).write_bytes(data)
            if patch:
                subprocess.run(["git", "apply", "-"], input=patch, cwd=work, check=True)
            if probe:
                source = work / SOURCE
                source.write_text(instrument(source.read_text()))
            (work / EXAMPLE).write_text(
                example.replace(
                    "    let mut out = ",
                    "    tensor4all_tensorci::tensorci2::allocation_probe::report();\n    let mut out = ",
                ) if probe else example
            )
            with (results / f"{name}.build.log").open("w") as log:
                subprocess.run(
                    ["cargo", "build", "--locked", "--release", "--jobs", "2", "--target-dir", target,
                     "-p", "tensor4all-tensorci", "--example", "issue1460_sampling"],
                    cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True,
                )
            executable = work / name
            shutil.copy2(target / "release/examples/issue1460_sampling", executable)
            return executable

        variants = [("base", b""), ("A", reuse), ("AB", changed)]
        binaries = {name: build(name, patch) for name, patch in variants}
        for size, strategy in CELLS:
            traces = []
            for name, binary in binaries.items():
                trace = subprocess.check_output([binary, str(size), strategy, "trace"])
                (results / f"{size}-{strategy}-{name}.trace").write_bytes(trace)
                traces.append(trace)
            if not traces[0] == traces[1] == traces[2]:
                raise ValueError(f"trace mismatch for {size} {strategy}")
            print(f"{size} {strategy}: {traces[0].splitlines()[0].decode()}", flush=True)

        for offset, (size, strategy) in enumerate(CELLS):
            order = list(binaries)
            order = order[offset % len(order):] + order[:offset % len(order)]
            for name in order:
                raw = subprocess.check_output([binaries[name], str(size), strategy, "time"])
                (results / f"{size}-{strategy}-{name}.ns").write_bytes(raw)
                times = [int(value) / 1e6 for value in raw.splitlines()]
                print(f"{size:2} {strategy:4} {name:4} med={median(times):.3f} "
                      f"min={min(times):.3f} max={max(times):.3f} ms", flush=True)

        for name, patch in variants:
            binary = build(name + "-alloc", patch, probe=True)
            for size, strategy in CELLS:
                run = subprocess.run([binary, str(size), strategy, "trace"], capture_output=True, check=True)
                if run.stdout != (results / f"{size}-{strategy}-{name}.trace").read_bytes():
                    raise ValueError(f"allocation probe changed the {size} {strategy} {name} trace")
                (results / f"{size}-{strategy}-{name}.alloc").write_bytes(run.stderr)
                print(f"{size} {strategy} {name}\n{run.stderr.decode()}", end="", flush=True)

        if args.release_tests:
            # Restore production A+B before testing. The probe is deliberately
            # excluded from the test build.
            for path, data in original.items():
                (work / path).write_bytes(data)
            subprocess.run(["git", "apply", "-"], input=changed, cwd=work, check=True)
            for path in ("tests/main.rs", "tests/scalar_sampling.rs", "src/tensorci2/tests/mod.rs"):
                shutil.copyfile(ROOT / CRATE / path, work / CRATE / path)
            for test in ("scalar_sampling", "batch"):
                with (results / f"test-release-{test}.log").open("w") as log:
                    subprocess.run(
                        ["cargo", "test", "--locked", "--release", "--jobs", "2", "--target-dir", target,
                         "-p", "tensor4all-tensorci", test],
                        cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True,
                    )


if __name__ == "__main__":
    main()
