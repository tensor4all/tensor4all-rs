#!/usr/bin/env python3
"""Run the pre-registered #1130 paired experiment; never retry selected cells."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import runpy
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).with_name('measurement-protocol.json')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources(head):
    if git('rev-parse', 'HEAD') != head or git('status', '--porcelain'):
        raise RuntimeError('Measured checkout moved or became dirty')


def pin_agents(observer):
    """Confine agent processes and existing children, excluding gated jobs."""
    processes = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            processes[int(entry.name)] = {
                'parent': int(stat[1]), 'start': stat[19],
                'name': (entry / 'comm').read_text().strip(),
                'cgroup': (entry / 'cgroup').read_text(),
            }
        except (OSError, ValueError, IndexError):
            continue
    chosen = {pid for pid, p in processes.items()
              if p['name'].startswith('codex') or p['name'] in ('claude', 'pi')}
    while True:
        expanded = chosen | {pid for pid, p in processes.items() if p['parent'] in chosen}
        if expanded == chosen:
            break
        chosen = expanded
    records = []
    try:
        for pid in sorted(chosen):
            p = processes[pid]
            if 'ai-measure.slice' in p['cgroup'] or 'ai-build.slice' in p['cgroup']:
                continue
            try:
                record = dict(pid=pid, start=p['start'], name=p['name'],
                              affinity=sorted(os.sched_getaffinity(pid)), tasks={})
                records.append(record)
                for task in Path(f'/proc/{pid}/task').iterdir():
                    tid = int(task.name)
                    try:
                        record['tasks'][str(tid)] = sorted(os.sched_getaffinity(tid))
                        os.sched_setaffinity(tid, observer)
                    except ProcessLookupError:
                        pass
            except ProcessLookupError:
                continue
        return records
    except BaseException:
        restore_agents(records)
        raise


def restore_agents(records):
    for record in reversed(records):
        pid = record['pid']
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            if stat[19] != record['start']:
                continue
            for task in Path(f'/proc/{pid}/task').iterdir():
                try:
                    os.sched_setaffinity(int(task.name), record['tasks'].get(task.name, record['affinity']))
                except ProcessLookupError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass


def run_cell(binary, case, policy):
    protocol = json.loads(PROTOCOL.read_text())
    if os.sched_getaffinity(0) != {protocol['measurement_cpu']}:
        raise RuntimeError('Cell did not receive the registered single CPU')
    limit = protocol['address_space_limit_bytes']
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    environment = dict(os.environ, **protocol['thread_environment'], LC_ALL='C')
    os.execve(binary, [binary, case, policy, 'measure'], environment)


def numerical_valid(row, case, policy, protocol):
    if row['case'] != case or row['policy'] != policy:
        return False
    for key in ('scheduler_seconds', 'l2_error', 'max_abs_error', 'error_bound', 'allowance'):
        if not math.isfinite(row[key]) or row[key] < 0:
            return False
    margin = protocol['absolute_roundoff_allowance']
    return (row['scheduler_seconds'] > 0 and row['peak_rss_kib'] > 0
            and row['error_bound'] <= row['allowance']
            and row['l2_error'] <= row['error_bound'] + margin
            and row['max_abs_error'] <= row['error_bound'] + margin
            and row['peak_rss_kib'] * 1024 < protocol['address_space_limit_bytes']
            and (policy != 'tolerance' or row['terms'] == row['regions']))


def summarize(rows, protocol):
    summary = {}
    rng = random.Random(protocol['bootstrap']['seed'])
    for case in protocol['cases']:
        pairs = []
        for pair in range(protocol['pairs']):
            arms = {r['policy']: r['result'] for r in rows if r['case'] == case and r['pair'] == pair}
            pairs.append({'pair': pair + 1,
                          'time_ratio': arms['tolerance']['scheduler_seconds'] / arms['goal8']['scheduler_seconds'],
                          'rss_ratio': arms['tolerance']['peak_rss_kib'] / arms['goal8']['peak_rss_kib']})
        ratios = [p['time_ratio'] for p in pairs]
        bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios)))
                           for _ in range(protocol['bootstrap']['resamples']))
        count = len(bootstrap)
        median = statistics.median(ratios)
        hard = median <= protocol['hard_max_median_paired_ratio'][case]
        rss = case != 'original' or max(p['rss_ratio'] for p in pairs) <= protocol['original_max_paired_rss_ratio']
        summary[case] = {
            'pairs': pairs, 'median_paired_time_ratio': median,
            'bootstrap_95_percentile_interval': [bootstrap[int(.025 * count)], bootstrap[int(.975 * count) - 1]],
            'median_scheduler_seconds': {policy: statistics.median(r['result']['scheduler_seconds'] for r in rows if r['case'] == case and r['policy'] == policy) for policy in ('goal8', 'tolerance')},
            'max_peak_rss_kib': {policy: max(r['result']['peak_rss_kib'] for r in rows if r['case'] == case and r['policy'] == policy) for policy in ('goal8', 'tolerance')},
            'soft_review_triggered': median > protocol['soft_review_ratio'],
            'hard_time_gate_passed': hard, 'rss_gate_passed': rss,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--registration-comment', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise RuntimeError('Write measurements outside the measured checkout')
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    check_sources(args.expected_head)
    gate_config = runpy.run_path(shutil.which('ai-run'))['load_config']()
    if any(gate_config[k] != v for k, v in protocol['noise_thresholds'].items()):
        raise RuntimeError('Host-noise settings differ from the registered protocol')
    observer = set(protocol['observer_cpus'])
    os.sched_setaffinity(0, observer)
    lock = open(git('rev-parse', '--git-path', 'issue1130-acceptance.lock'), 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    affinities = pin_agents(observer)
    write_json(output / 'observer-affinities.json', affinities)
    try:
        environment = dict(os.environ, **protocol['thread_environment'],
                           CARGO_TARGET_DIR=str(ROOT.parent / 'tensor4all-rs/target'))
        environment.pop('RUSTFLAGS', None)
        environment.pop('CARGO_ENCODED_RUSTFLAGS', None)
        build = ['ai-run', 'build', '--mem', '6G', '--jobs', '2', '--', 'bash', '-c',
                 'cargo fmt --all -- --check && cargo build --locked --release -p tensor4all-benchmarks --features fourier --example benchmark_merge_policy_1130 && rustc --version --verbose']
        print('Building the registered release binary through ai-run build.', flush=True)
        with (output / 'build.txt').open('w') as log:
            subprocess.run(build, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        check_sources(args.expected_head)
        source_binary = Path(environment['CARGO_TARGET_DIR']) / 'release/examples/benchmark_merge_policy_1130'
        binary = output / 'benchmark_merge_policy_1130'
        shutil.copyfile(source_binary, binary)
        binary.chmod(0o555)
        binary_hash = digest(binary)
        metadata = {
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'baseline_commit': git('rev-parse', 'HEAD'), 'candidate_commit': git('rev-parse', 'HEAD'),
            'dirty': False, 'registration_comment': args.registration_comment,
            'protocol': protocol, 'protocol_sha256': digest(PROTOCOL),
            'harness_sha256': digest(ROOT / protocol['source_harness']),
            'controller_sha256': digest(Path(__file__)), 'binary_sha256': binary_hash,
            'cargo_lock_sha256': digest(ROOT / 'Cargo.lock'),
            'ai_run_sha256': digest(Path(shutil.which('ai-run'))),
            'ai_run_noise_settings': {k: gate_config[k] for k in protocol['noise_thresholds']},
            'kernel': platform.release(), 'machine': platform.machine(), 'hostname': platform.node(),
            'build_command': build, 'build_rustflags': None,
            'binary_libraries': subprocess.check_output(['ldd', str(binary)], text=True),
            'cpu_topology': subprocess.check_output(['lscpu', '-e=CPU,CORE,SOCKET,ONLINE'], text=True),
            'cpu_provider': 'platform default: faer on Linux',
        }
        write_json(output / 'metadata.json', metadata)
        for attempt in (1, 2):
            folder = output / f'attempt-{attempt}'
            folder.mkdir()
            rows, noisy, invalid = [], False, False
            print(f'Complete paired experiment, attempt {attempt}.', flush=True)
            for case in protocol['cases']:
                for pair in range(protocol['pairs']):
                    order = protocol['first_pair_order'][::1 if pair % 2 == 0 else -1]
                    for policy in order:
                        check_sources(args.expected_head)
                        if digest(binary) != binary_hash:
                            raise RuntimeError('Immutable binary changed')
                        prefix = f'{case}-pair-{pair + 1}-{policy}'
                        gate_path = folder / f'{prefix}.gate.json'
                        command = ['ai-run', 'measure', '--mem', '6G', '--cpus', str(protocol['measurement_cpu']),
                                   '--wait', '600', '--report', str(gate_path), '--label', 'issue-1130-acceptance', '--',
                                   sys.executable, str(Path(__file__).resolve()), '_cell', str(binary), case, policy]
                        run = subprocess.run(command, cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        (folder / f'{prefix}.stdout.txt').write_text(run.stdout)
                        (folder / f'{prefix}.stderr.txt').write_text(run.stderr)
                        gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
                        clean = (gate.get('verdict') == 'CLEAN' and gate.get('cpus') == str(protocol['measurement_cpu'])
                                 and gate.get('pre_guard', {}).get('ok') is True
                                 and gate.get('thresholds') == {k: protocol['noise_thresholds'][k] for k in ('foreign_frac', 'others_busy_cpus', 'mem_stall_frac')})
                        noisy |= not clean
                        try:
                            row = json.loads(run.stdout)
                            correct = run.returncode == 0 and numerical_valid(row, case, policy, protocol)
                        except (ValueError, KeyError, TypeError):
                            row, correct = None, False
                        invalid |= not correct and run.returncode != 75
                        rows.append({'case': case, 'pair': pair, 'policy': policy, 'command': command,
                                     'exit_code': run.returncode, 'gate': gate_path.name, 'clean': clean,
                                     'numerical_valid': correct, 'result': row})
                        write_json(folder / 'samples.json', rows)
                        print(f'  {prefix}: clean={clean}, numerical_valid={correct}', flush=True)
            check_sources(args.expected_head)
            if invalid:
                outcome = {'status': 'INVALID_NUMERICAL', 'attempt': attempt}
            elif noisy:
                outcome = {'status': 'INCONCLUSIVE', 'attempt': attempt}
            else:
                summary = summarize(rows, protocol)
                passed = all(s['hard_time_gate_passed'] and s['rss_gate_passed'] for s in summary.values())
                outcome = {'status': 'PASS' if passed else 'FAIL', 'attempt': attempt, 'summary': summary}
            write_json(folder / 'outcome.json', outcome)
            write_json(output / 'outcome.json', outcome)
            print(f"Attempt {attempt}: {outcome['status']}", flush=True)
            if outcome['status'] != 'INCONCLUSIVE' or attempt == 2:
                break
        return 0 if outcome['status'] == 'PASS' else 1
    finally:
        restore_agents(affinities)
        write_json(output / 'observer-affinities-restored.json', {'restored': True})
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_cell':
        run_cell(*sys.argv[2:])
    else:
        sys.exit(main())
