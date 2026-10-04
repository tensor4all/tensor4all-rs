#!/usr/bin/env python3
"""Build, run and independently validate the fixed native product experiment."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
CASES = [('chain', 16, 2), ('chain', 32, 2), ('chain', 32, 4),
         ('chain', 64, 4), ('binary', 15, 2), ('star', 64, 1)]
SEEDS = [1, 2, 3]
BLOCKS = 3
TARGET = 1e-8


def source_hash():
    """Hash contents, including untracked source; exclude generated output."""
    files = {ROOT / 'Cargo.toml', ROOT / 'Cargo.lock', ROOT / 'rust-toolchain.toml'}
    for directory in ['crates', '.cargo', 'benchmarks/tree-rsi']:
        files.update(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and p.suffix in {'.rs', '.toml', '.py'}
                     and not {'target', 'results', 'scratch', '__pycache__'} & set(p.relative_to(ROOT).parts))
    digest = hashlib.sha256()
    for path in sorted(files):
        if path.exists():
            name = str(path.relative_to(ROOT)).encode()
            data = path.read_bytes()
            digest.update(len(name).to_bytes(8, 'little') + name)
            digest.update(len(data).to_bytes(8, 'little') + data)
    return digest.hexdigest()


def validate(records):
    """Fail closed on missing, duplicate, nonfinite or inaccurate attempts."""
    if not records or records[0].get('type') != 'protocol':
        raise ValueError('missing protocol')
    protocol = records[0]
    required = {'cases': [list(c) for c in CASES], 'seeds': SEEDS,
                'blocks': BLOCKS, 'samples': 2048, 'target': TARGET, 'sample_seed': 20260929}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError('protocol differs from fixed experiment')
    caps = {f'{t}-n{n}-chi{chi}': chi * chi for t, n, chi in CASES}
    names = list(caps)
    expected = {(name, phase, block, seed, alg)
                for name in names for phase in ['accuracy', 'timing']
                for block in range(1 if phase == 'accuracy' else BLOCKS)
                for seed in SEEDS for alg in ['rsi', 'treeaci']}
    seen = {}
    for row in records[1:]:
        key = tuple(row.get(k) for k in ['case', 'phase', 'block', 'seed', 'algorithm'])
        if row.get('type') != 'observation' or key not in expected or key in seen:
            raise ValueError(f'failed, unexpected or duplicate attempt: {key}')
        for field in ['error', 'seconds']:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid {field}: {key}')
        if row.get('cap') != caps[row['case']]:
            raise ValueError(f'incorrect rank budget: {key}')
        diagnostics = row.get('diagnostics', {})
        if not isinstance(diagnostics, dict):
            raise ValueError(f'invalid diagnostics: {key}')
        ranks = diagnostics.get('edge_ranks')
        topology, nodes, _ = next(c for c in CASES if f'{c[0]}-n{c[1]}-chi{c[2]}' == row['case'])
        expected_edges = {(i - 1 if topology == 'chain' else (i - 1) // 2 if topology == 'binary' else 0, i)
                          for i in range(1, nodes)}
        observed_edges = set()
        if not isinstance(ranks, list) or len(ranks) != nodes - 1:
            raise ValueError(f'missing actual edge ranks: {key}')
        for edge in ranks:
            if not isinstance(edge, list) or len(edge) != 3 or any(type(v) is not int for v in edge):
                raise ValueError(f'invalid edge rank: {key}')
            a, b, rank = edge
            observed_edges.add((min(a, b), max(a, b)))
            if not 1 <= rank <= row['cap']:
                raise ValueError(f'actual rank exceeds budget: {key}')
        if observed_edges != expected_edges:
            raise ValueError(f'edge-rank topology mismatch: {key}')
        if row['algorithm'] == 'treeaci':
            if diagnostics.get('termination') not in {'Converged', 'RankLimited', 'MaxSweeps'}:
                raise ValueError(f'missing termination: {key}')
            history = [diagnostics.get(k) for k in ['sweep_ranks', 'local_errors', 'global_pivots']]
            if any(not isinstance(h, list) or not h for h in history) or len({len(h) for h in history}) != 1:
                raise ValueError(f'incomplete sweep history: {key}')
            if (any(type(v) is not int or not 1 <= v <= row['cap'] for v in history[0])
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in history[1])
                    or any(type(v) is not int or v < 0 for v in history[2])):
                raise ValueError(f'invalid sweep history: {key}')
        else:
            pivots = diagnostics.get('local_pivots')
            if (type(diagnostics.get('sketch_dim')) is not int
                    or diagnostics['sketch_dim'] != (row['cap'] + 1) // 2 + 5
                    or not isinstance(pivots, list) or len(pivots) != nodes - 1
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in pivots)):
                raise ValueError(f'invalid RSI sketch diagnostics: {key}')
        if row['error'] > TARGET or row['seconds'] <= 0:
            raise ValueError(f'fixed-budget accuracy target or timer check not met: {key}')
        seen[key] = row
    if set(seen) != expected:
        raise ValueError(f'missing {len(expected - set(seen))} attempts')
    return [(name, statistics.median(seen[name, 'timing', b, s, 'rsi']['seconds']
                                   for b in range(BLOCKS) for s in SEEDS),
             statistics.median(seen[name, 'timing', b, s, 'treeaci']['seconds']
                               for b in range(BLOCKS) for s in SEEDS),
             statistics.median(seen[name, 'timing', b, s, 'treeaci']['seconds'] /
                               seen[name, 'timing', b, s, 'rsi']['seconds']
                               for b in range(BLOCKS) for s in SEEDS)) for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int, help='one CPU from the current Linux affinity set')
    parser.add_argument('--output', type=Path, help='new JSONL path (default: ignored target/tree-rsi)')
    args = parser.parse_args()
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--no-deps', '--format-version', '1'], cwd=ROOT))
    target = Path(metadata['target_directory'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = (args.output or target / 'tree-rsi' / f'{stamp}.jsonl').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('output already exists')
    source = source_hash()
    subprocess.run(['cargo', 'build', '--release', '-p', 'tree-rsi-benchmark'], cwd=ROOT, check=True)
    if source_hash() != source:
        raise ValueError('source changed while building; rerun')
    binary = target / 'release/tree-rsi-benchmark'
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS']:
        env[key] = '1'
    env.update(RSI_SOURCE_SHA256=source, RSI_BINARY_SHA256=binary_hash)
    available = os.sched_getaffinity(0)
    cpu = args.cpu if args.cpu is not None else min(available)
    if cpu not in available:
        raise ValueError('CPU outside current affinity')
    host = {'source_sha256': source, 'binary_sha256': binary_hash, 'cpu': cpu,
            'uname': list(os.uname()), 'load_average': list(os.getloadavg()),
            'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
            'thread_environment': {k: env[k] for k in env if k.endswith('NUM_THREADS')},
            'build_environment': {k: env[k] for k in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_TARGET', 'CARGO_PROFILE_RELEASE_LTO', 'CARGO_PROFILE_RELEASE_OPT_LEVEL'] if k in env},
            'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    with output.with_suffix('.host.json').open('x') as handle:
        json.dump(host, handle, indent=2)
    os.sched_setaffinity(0, {cpu})
    result = subprocess.run([str(binary), str(output)], cwd=ROOT, env=env)
    print(f'All attempts: {output}', flush=True)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    summary = validate(records)
    if result.returncode:
        raise ValueError(f'executable exited {result.returncode}')
    if records[0]['source_sha256'] != source or records[0]['binary_sha256'] != binary_hash:
        raise ValueError('provenance mismatch')
    print('Sampled accuracy passed; this is not a global error bound or GW acceptance.')
    print('case | RSI median s | TreeACI median s | median paired TreeACI/RSI')
    for name, rsi, aci, ratio in summary:
        print(f'{name} | {rsi:.6g} | {aci:.6g} | {ratio:.3f}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f'Experiment rejected: {error}', file=sys.stderr)
        sys.exit(1)
