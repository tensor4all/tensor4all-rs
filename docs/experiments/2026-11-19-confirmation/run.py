#!/usr/bin/env python3
"""Run the predeclared four-arm comparison using already built binaries.

Usage: taskset -c 0-7 python3 run.py OUTPUT CANDIDATE MAIN HISTORICAL
Each checkout must contain a release example in target/1372-reference/.
The output directory must not exist. Every attempt is retained.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]


def key(row):
    return row['case'], row['tolerance'], row['seed']


def deterministic(row):
    return {k: v for k, v in row.items() if k not in ('seconds', 'confirm')}


def main():
    output, candidate, current, historical = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=False)
    arms = {'historical': (historical, 'off'), 'main': (current, 'off'),
            'off': (candidate, 'off'), 'on': (candidate, 'on')}
    manifest = {'protocol': str(Path(__file__).with_name('README.md')),
                'observer_affinity': sorted(os.sched_getaffinity(0)),
                'uname': command(['uname', '-a']), 'lscpu': command(['lscpu']),
                'ai_run_sha256': sha('/root/.local/bin/ai-run'),
                'thread_environment': {k: v for k, v in os.environ.items()
                    if k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                             'RAYON_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}, 'arms': {}}
    if not os.sched_getaffinity(0) <= set(range(8)):
        raise RuntimeError('Observer must be confined to CPUs 0-7')
    (output/'compiler.txt').write_text(command(['ai-run', 'build', '--mem', '1G', '--', 'rustc', '-Vv']))
    for arm, (checkout, setting) in arms.items():
        binary = checkout/'target/1372-reference/release/examples/confirmation_comparison'
        source = checkout/'crates/tensor4all-treetci/examples/confirmation_comparison.rs'
        manifest['arms'][arm] = {
            'checkout': str(checkout), 'head': command(['git', 'rev-parse', 'HEAD'], checkout),
            'tracked_dirty': command(['git', 'diff', 'HEAD', '--name-only'], checkout),
            'untracked': command(['git', 'ls-files', '--others', '--exclude-standard'], checkout),
            'source_sha256': sha(source), 'binary': str(binary), 'binary_sha256': sha(binary),
            'lock_sha256': sha(checkout/'Cargo.lock'), 'confirmation': setting}
        shutil.copyfile(source, output/(arm+'-driver.rs'))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # No source or executable changes are permitted after freezing the manifest.
    def unchanged():
        for arm, (checkout, _) in arms.items():
            pin = manifest['arms'][arm]
            assert command(['git', 'rev-parse', 'HEAD'], checkout) == pin['head']
            assert command(['git', 'diff', 'HEAD', '--name-only'], checkout) == pin['tracked_dirty']
            assert sha(pin['binary']) == pin['binary_sha256']
            assert sha(checkout/'crates/tensor4all-treetci/examples/confirmation_comparison.rs') == pin['source_sha256']
    checked = {}
    for arm, (checkout, setting) in arms.items():
        unchanged()
        with (output/(arm+'-checked.jsonl')).open('w') as out, (output/(arm+'-checked.log')).open('w') as log:
            subprocess.run(['ai-run', 'build', '--mem', '6G', '--jobs', '2', '--',
                manifest['arms'][arm]['binary'], 'all', setting, 'checked'],
                cwd=checkout, stdout=out, stderr=log, check=True)
        entries = rows(output/(arm+'-checked.jsonl'))
        assert len(entries) == 21, (arm, len(entries))
        checked[arm] = {key(r): deterministic(r) for r in entries}
        assert len(checked[arm]) == 21
    assert checked['main'] == checked['off'], 'Default differs from main'
    rounds = [['historical', 'main', 'off', 'on'], ['on', 'off', 'main', 'historical']]
    for number, order in enumerate(rounds, 1):
        for arm in order:
            unchanged()
            checkout, setting = arms[arm]
            stem = output/f'round{number}-{arm}'
            with stem.with_suffix('.jsonl').open('w') as out, stem.with_suffix('.log').open('w') as log:
                subprocess.run(['ai-run', 'measure', '--mem', '6G', '--cpus', '8-15',
                    '--report', str(stem.with_suffix('.guard.json')), '--',
                    manifest['arms'][arm]['binary'], 'comb60', setting, 'timed'],
                    cwd=checkout, stdout=out, stderr=log, check=True)
            report = json.loads(stem.with_suffix('.guard.json').read_text())
            assert report['verdict'] == 'CLEAN', f'INCONCLUSIVE: {stem}'
            entries = rows(stem.with_suffix('.jsonl'))
            assert len(entries) == 3
            for row in entries:
                assert deterministic(row) == checked[arm][key(row)], (arm, key(row))
    unchanged()
    (output/'VALID.json').write_text(json.dumps({'status': 'VALID', 'checked_cells': 84, 'timed_cells': 24})+'\n')


if __name__ == '__main__':
    main()
