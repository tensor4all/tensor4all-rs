#!/usr/bin/env python3
"""Add observational serial scopes to a clean, detached diagnostic checkout."""
import argparse
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    root = args.checkout.resolve()
    branch = subprocess.run(["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"], capture_output=True)
    if branch.returncode == 0:
        raise SystemExit("Use a detached diagnostic checkout")
    clean = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if clean:
        raise SystemExit("Use a clean diagnostic checkout")
    relative = Path("crates/tensor4all-partitionedtreetn/src/reconstruction/schedule.rs")
    path = root / relative
    text = path.read_text()
    text = replace(text, "mod tolerance;", "mod tolerance;\nmod profiling1497;")
    text = replace(text,
        "for (region, items) in std::mem::take(&mut level) {",
        "for (ordinal, (region, items)) in std::mem::take(&mut level).into_iter().enumerate() {\n                let _region = profiling1497::Region::new(level_index, ordinal);")
    start = text.index("fn restrict_and_merge<V>(")
    end = text.index("/// One kept output region", start)
    body = text[start:end]
    body = replace(body, "    incurred: &mut f64,\n", "    incurred: &mut f64,\n    label: &str,\n")
    body = replace(body, "    let mut children:", "    let mut profile = profiling1497::Probe::new(label, items.len(), counters.additions);\n    let mut children:")
    body = replace(body, "        let restricted =", "        let start = profile.timer();\n        let before = counters.additions;\n        let restricted =")
    body = replace(body,
        "        match children.entry(ScheduleGeometry::input_parent(input_key)) {",
        "        let parent = ScheduleGeometry::input_parent(input_key);\n        match children.entry(parent.clone()) {")
    body = replace(body, "    }\n    let projections", "        profile.record(parent, start, counters.additions - before);\n    }\n    let projections")
    body = replace(body, "    Ok(children)", "    profile.finish(counters.additions);\n    Ok(children)")
    text = text[:start] + body + text[end:]
    text = replace(text,
        "                                &mut child_incurred,\n",
        "                                &mut child_incurred,\n                                &format!(\"axis-{axis}\"),\n")
    text = replace(text,
        "                            &mut incurred,\n",
        "                            &mut incurred,\n                            \"joint\",\n")
    text = replace(text,
        "restrict_and_merge(context, items, &projector, counters, &mut incurred)?;",
        "restrict_and_merge(context, items, &projector, counters, &mut incurred, \"own\")?;")
    start = text.index("fn merge_own_region<V>(")
    end = text.index("/// Combine two sibling", start)
    body = text[start:end]
    body = replace(body, "    let projector =", "    let mut candidate = profiling1497::Candidate::new(\"own\");\n    let projector =")
    body = replace(body, "    Ok((merged, incurred))", "    candidate.complete();\n    Ok((merged, incurred))")
    text = text[:start] + body + text[end:]
    text = replace(text, "                    for &axis in &axes {", "                    for &axis in &axes {\n                        let mut candidate = profiling1497::Candidate::new(format!(\"axis-{axis}\"));")
    text = replace(text, "                    }\n                    if passing.is_empty()", "                        candidate.complete();\n                    }\n                    if passing.is_empty()")
    text = replace(text, "                    probes.clear();", "                    let mut candidate = profiling1497::Candidate::new(\"joint\");\n                    probes.clear();")
    text = replace(text, "                    children\n                };", "                    candidate.complete();\n                    children\n                };")
    tolerance = root / relative.parent / "schedule/tolerance.rs"
    other = tolerance.read_text()
    other = replace(other,
        "restrict_and_merge(&child_context, items, &projector, counters, &mut incurred)?;",
        "restrict_and_merge(&child_context, items, &projector, counters, &mut incurred, &format!(\"{kind:?}\"))?;")
    start = other.index("    fn candidate(")
    end = other.index("    /// Resource limits", start)
    body = other[start:end]
    body = replace(body, "        let context = self.context;", "        let mut candidate = profiling1497::Candidate::new(format!(\"{kind:?}\"));\n        let context = self.context;")
    body = replace(body, "        Ok(Candidate {", "        candidate.complete();\n        Ok(Candidate {")
    other = other[:start] + body + other[end:]
    other = replace(other,
        "    for ((region, items), shape) in level.into_iter().zip(shapes) {",
        "    for (ordinal, ((region, items), shape)) in level.into_iter().zip(shapes).enumerate() {\n        let _region = profiling1497::Region::new(level_index, ordinal);")
    patch = HERE.parent / "1497-baseline/lazy-reader.patch"
    subprocess.run(["git", "-C", str(root), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(root), "apply", str(patch)], check=True)
    path.write_text(text)
    tolerance.write_text(other)
    benchmark = root / "benchmarks/rust/benchmark_m6a_fourier.rs"
    summary = benchmark.read_text()
    summary = replace(summary,
        '                "work_items_per_level": result.report().work_items_per_level,',
        '                "work_items_per_level": result.report().work_items_per_level,\n'
        '                "additions": result.report().additions,\n'
        '                "projections": result.report().projections,\n'
        '                "compression_attempts": result.report().compression_attempts,\n'
        '                "compressions": result.report().compressions,\n'
        '                "pruned_regions": result.report().pruned_regions,')
    benchmark.write_text(summary)
    shutil.copy2(HERE / "profiler.rs", root / relative.parent / "schedule/profiling1497.rs")
    print("Added diagnostic scopes without reordering numerical calls")


if __name__ == "__main__":
    main()
