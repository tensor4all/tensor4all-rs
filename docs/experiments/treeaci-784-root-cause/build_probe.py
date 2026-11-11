#!/usr/bin/env python3
"""Build the private #784 nesting probe and restore production source bytes."""
from pathlib import Path
import subprocess, json, hashlib, shutil, argparse
parser = argparse.ArgumentParser(description='Build a private nesting diagnostic; restore production sources after compilation.')
parser.add_argument('--clean', action='store_true', help='Build the same workload without diagnostic instrumentation.')
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = here.parents[2]
out = root / 'target/issue784/root-cause'
out.mkdir(parents=True, exist_ok=True)
base=here.parent/'treeaci-784-nesting'
(out/'audit.rs').write_text((base/'audit.rs').read_text()+'\n'+(here/'extra_audit.rs').read_text())
probe=(base/'probe.rs').read_text()
probe=probe.replace('    println!(\n', '    let importance = if std::env::var("T4A_ROOT_IMPORTANCE").as_deref()==Ok("on") {\n        let grid=Grid {r,temperature};let samples=draw(&grid,1,5000,seed^0x5eed);\n        let approx=evaluator.evaluate_batched_typed::<Complex64>(ColMajorArrayRef::new(&samples.points,&[3*r,samples.coords.len()])?,EvaluationHint::default())?;\n        let exact:Vec<_>=samples.coords.iter().map(|&[x,y,n]|grid.g0(x,y,n)).collect();\n        let (rel,se)=relative_l2(&samples,&approx,&exact);\n        Some(json!({"rel_l2":rel,"rel_l2_se":se,"samples":samples.coords.len(),"seed":seed^0x5eed,"norm":samples.norm}))\n    } else {None};\n'+ '    println!(\n',1).replace('"r":r,"topology":mode','"importance":importance,"r":r,"topology":mode',1)
(out/'probe.rs').write_text(probe+'\n'+(here/'importance.rs').read_text())
paths = ['crates/tensor4all-treeaci/src/lib.rs', 'crates/tensor4all-treeaci/src/state.rs', 'crates/tensor4all-treeaci/src/transaction.rs', 'crates/tensor4all-treeaci/src/schedule.rs', 'crates/tensor4all-treeaci/tests/support/g0.rs']
expected = {'crates/tensor4all-treeaci/src/lib.rs': '849e5038577f59caacd32b29374f64652da8767b1c128aa57afb652bce848fcd', 'crates/tensor4all-treeaci/src/state.rs': '1e4feaeb132ac78fe6bdf23b7f2073def4013174df68d9ceec29ef52d5c1699f', 'crates/tensor4all-treeaci/src/transaction.rs': '319719a736682632964550b5de1ec53df1464aac1412365da6bdeadbe1f4ffe9', 'crates/tensor4all-treeaci/src/schedule.rs': 'a8f3b752e6e2b522c79f8705038cca1a03ad3a27bec9f04f8c0361e83a7341ad', 'crates/tensor4all-treeaci/tests/support/g0.rs': '51d5c74729a9cc90797416d458f5ef7ac2f0ce107557691dc1c7f0391bd7a500'}
paths.append('crates/tensor4all-core/src/matrixlu.rs')
expected[paths[-1]]='54082ef06ecc79e0692c3ade6c5aa13bcbf6368095019a23039e196071f7904b'
original = {p: (root / p).read_text() for p in paths}
modified = {}
patches = {}
for (p, s) in original.items():
    assert hashlib.sha256(s.encode()).hexdigest() == expected[p], f'Source changed since audit: {p}'
wrapper = root / 'crates/tensor4all-treeaci/examples/nesting_probe.rs'
assert not wrapper.exists(), f'Temporary example already exists: {wrapper}'
try:
    p = paths[0]
    s = original[p] + f"\n#[path = {json.dumps(str(out / 'audit.rs'))}]\nmod nesting_probe;\n"
    modified[p] = s
    p = paths[1]
    s = original[p].replace('        Ok(Self {\n            problem,', '        let state = Self {\n            problem,', 1).replace('            generation,\n        })', '            generation,\n        };\n        crate::nesting_probe::record(&state, "initialize", None, None, None)?;\n        Ok(state)', 1)
    modified[p] = s
    p = paths[2]
    s = original[p].replace('    let hints = local_pivot_hints(state, forward, options.max_working_bytes)?;', '    crate::nesting_probe::record(state, "before-update", Some(forward), None, None)?;\n    let hints = if std::env::var("T4A_NESTING_HINTS").as_deref()==Ok("off") {Vec::new()} else {local_pivot_hints(state, forward, options.max_working_bytes)?};', 1).replace('    commit_edge_proposal(state, forward, proposal, exec)', '    let report = commit_edge_proposal(state, forward, proposal, exec)?;\n    crate::nesting_probe::record(state, "after-update", Some(forward), None, None)?;\n    Ok(report)', 1)
    modified[p] = s
    p = paths[3]
    s = original[p].replace('    for pass in 0..options.max_sweeps {', '    for pass in 0..options.max_sweeps {\n        crate::nesting_probe::set_pass(pass+1);', 1).replace('        max_ranks.push(report.max_rank);', '        crate::nesting_probe::record(state, "after-pass", None, Some(stable_rank_passes), None)?;\n        max_ranks.push(report.max_rank);', 1).replace('        global_pivots.push(found);', '        crate::nesting_probe::record(state, "after-guard", None, Some(stable_rank_passes), Some(found))?;\n        global_pivots.push(found);', 1)
    modified[p] = s
    p = paths[4]
    s = original[p].replace('-2.0 * PI * temperature * (1usize << (r - 1)) as f64\n                                    + PI * temperature', '(2.0 * PI * temperature) * (0.5 - (1usize << (r - 1)) as f64)')
    assert s != original[p]
    modified[p] = s
    p=paths[-1]
    s=original[p]
    start=s.index('fn factorize_with_hints<');end=s.index('/// Pivot magnitudes',start)
    block=s[start:end]
    block=block.replace('    while n_pivot < max_bond_dim {','    let mut root_full=Vec::new();\n    let mut root_hint_steps=Vec::new();\n    while n_pivot < max_bond_dim {',1)
    block=block.replace('        error = pivot_abs;','        root_full.push(pivot_abs);\n        error = pivot_abs;',1)
    block=block.replace('                    pivot_row = i;','                    root_hint_steps.push(k);\n                    pivot_row = i;',1)
    block=block.replace('    Ok(RrLUFactorization {','    eprintln!("ROOT_LU {{\\\"rows\\\":{},\\\"cols\\\":{},\\\"rank\\\":{},\\\"hints\\\":{},\\\"hint_steps\\\":{:?},\\\"full_residuals\\\":{:?}}}",nr,nc,n_pivot,hints.len(),root_hint_steps,root_full);\n    Ok(RrLUFactorization {',1)
    modified[p]=s[:start]+block+s[end:]
    p=paths[2]
    modified[p]=modified[p].replace('    let proposal = materialize_and_factor_edge(','    crate::nesting_probe::edge_before(state,forward,hints.len())?;\n    let proposal = materialize_and_factor_edge(',1)
    p=paths[3]
    modified[p]=modified[p].replace('        crate::nesting_probe::record(state, "after-guard"','        crate::nesting_probe::repair_guard_candidates(state)?;\n        crate::nesting_probe::record(state, "after-guard"',1)
    if args.clean:
        modified = {paths[4]: modified[paths[4]]}
    for (p, s) in modified.items():
        assert s != original[p], p
        (root / p).write_text(s)
        patches[p] = {'original_sha256': hashlib.sha256(original[p].encode()).hexdigest(), 'instrumented_source': s}
    s = (out / 'probe.rs').read_text()
    wrapper = root / 'crates/tensor4all-treeaci/examples/nesting_probe.rs'
    assert not wrapper.exists()
    wrapper.write_text(s)
    (out / ('clean_probe_example.rs' if args.clean else 'probe_example.rs')).write_text(s)
    (out / ('clean_sources.json' if args.clean else 'instrumented_sources.json')).write_text(json.dumps(patches, indent=2) + '\n')
    subprocess.run(['cargo', 'build', '--locked', '--release', '-p', 'tensor4all-treeaci', '--example', 'nesting_probe'], cwd=root, check=True)
    name = 'root-clean-control' if args.clean else 'root-probe'
    shutil.copy2(root / 'target/release/examples/nesting_probe', out / name)
    (out / (name + '.build.json')).write_text(json.dumps({'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), 'sha256': hashlib.sha256((out / name).read_bytes()).hexdigest(), 'instrumented': not args.clean}, indent=2) + '\n')
finally:
    for (p, s) in original.items():
        (root / p).write_text(s)
    wrapper = root / 'crates/tensor4all-treeaci/examples/nesting_probe.rs'
    if wrapper.exists():
        wrapper.unlink()
print('Original tracked sources restored byte-for-byte.')
