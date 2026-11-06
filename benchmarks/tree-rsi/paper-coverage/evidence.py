"""Source-bound worker receipts and exact experiment schedules (no timing code)."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_settings(cwd):
    # Record only build controls, never arbitrary environment variables/secrets.
    return dict(rustc=subprocess.check_output(['rustc', '-vV'], cwd=cwd, text=True),
                cargo=subprocess.check_output(['cargo', '-V'], cwd=cwd, text=True),
                environment={k: os.environ[k] for k in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS',
                    'CARGO_BUILD_TARGET', 'CARGO_PROFILE_RELEASE_LTO', 'CARGO_PROFILE_RELEASE_CODEGEN_UNITS',
                    'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'RUSTC', 'RUSTC_WRAPPER',
                    'RUSTC_WORKSPACE_WRAPPER', 'RUSTUP_TOOLCHAIN') if k in os.environ})


def source_inputs(directory, source, cwd, algorithm, worker_source=None):
    args = ['cargo', 'metadata', '--offline', '--format-version=1', '--manifest-path', str(directory/'Cargo.toml')]
    if algorithm:
        args += ['--features', algorithm]
    metadata = json.loads(subprocess.check_output(args, cwd=cwd, text=True))
    files = {}
    def add(path):
        path = path.resolve()
        files[str(path)] = sha(path)
    add(Path(__file__))
    add(Path(__file__).with_name('build.py'))
    for package in metadata['packages']:
        if package['source'] is not None:
            continue  # Registry/git revisions and checksums are captured in Cargo.lock.
        folder = Path(package['manifest_path']).parent
        add(folder/'Cargo.toml')
        for parent in folder.parents:
            if (parent/'Cargo.toml').is_file():
                add(parent/'Cargo.toml')
        # Package inputs include build scripts, native sources and included data.
        # Pristine git-archive baselines have no .git; enumerate their package files.
        for path in folder.rglob('*'):
            relative = path.relative_to(folder)
            if any(p in ('target', '.git', '__pycache__') for p in relative.parts):
                continue
            if path.is_file() and path.name not in ('worker', 'build.log', 'build-receipt.json', 'Cargo.lock'):
                add(path)
    if worker_source is not None:
        add(Path(worker_source))
    for path in (directory/'Cargo.lock', source/'Cargo.toml', source/'Cargo.lock'):
        add(path)
    # Cargo searches configuration from cwd to /, then CARGO_HOME.
    for folder in [Path(cwd).resolve(), *Path(cwd).resolve().parents,
                   Path(os.environ.get('CARGO_HOME', str(Path.home()/'.cargo')))]:
        for name in ('.cargo/config', '.cargo/config.toml') if folder != Path(os.environ.get('CARGO_HOME', str(Path.home()/'.cargo'))) else ('config', 'config.toml'):
            path = folder/name
            if path.is_file():
                add(path)
    return dict(files=files, settings=build_settings(cwd), algorithm=algorithm, profile='release',
                directory=str(directory.resolve()), source=str(source.resolve()),
                worker_source=str(Path(worker_source).resolve()) if worker_source else None)


def archive_worker(directory, out, inputs):
    receipt = dict(schema=1, inputs=inputs, source_sha256=identity(inputs), worker_sha256=sha(directory/'worker'))
    build_id = identity(receipt)
    archive = out/'builds'/build_id
    if not archive.exists():
        archive.mkdir(parents=True)
        shutil.copy2(directory/'worker', archive/'worker')
        snapshot = archive/'sources'; snapshot.mkdir()
        for original, digest in inputs['files'].items():
            path = Path(original)
            if sha(path) != digest:
                raise ValueError('source changed while archiving worker')
            destination = snapshot/identity(original)
            shutil.copy2(path, destination)
        (archive/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    else:
        verify_archived_worker(archive, build_id)
    (directory/'build-receipt.json').write_text(json.dumps(dict(build_id=build_id))+'\n')
    return archive


def verify_archived_worker(archive, build_id):
    receipt = json.loads((archive/'receipt.json').read_text())
    if receipt.get('schema') != 1 or identity(receipt) != build_id:
        raise ValueError('invalid worker build receipt')
    if identity(receipt['inputs']) != receipt['source_sha256']:
        raise ValueError('worker source identity mismatch')
    if sha(archive/'worker') != receipt['worker_sha256']:
        raise ValueError('worker hash mismatch')
    for original, digest in receipt['inputs']['files'].items():
        if sha(archive/'sources'/identity(original)) != digest:
            raise ValueError('worker build source snapshot mismatch')
    return receipt


def checked_workers(out, source, cwd, algorithms):
    workers = {}; builds = {}; receipts = {}
    for algorithm in algorithms:
        directory = out/algorithm
        pointer = directory/'build-receipt.json'
        if not pointer.is_file():
            raise ValueError(f'{algorithm}: worker has no build receipt; run build.py')
        build_id = json.loads(pointer.read_text())['build_id']
        archive = out/'builds'/build_id
        receipt = verify_archived_worker(archive, build_id)
        root = source[algorithm]
        if source_inputs(directory, root, cwd, 'treeaci' if algorithm.startswith('treeaci') else algorithm,receipt['inputs'].get('worker_source')) != receipt['inputs']:
            raise ValueError(f'{algorithm}: stale worker sources or build settings; run build.py')
        workers[algorithm] = archive/'worker'; builds[algorithm] = build_id; receipts[algorithm] = receipt
    return workers, dict(worker_builds=builds,
        worker_hashes={a:r['worker_sha256'] for a,r in receipts.items()},
        candidate_source_sha256=receipts['rsi']['source_sha256'] if 'rsi' in receipts else None)


def workers_unchanged(out, builds, cwd):
    for build_id in builds.values():
        receipt = verify_archived_worker(out/'builds'/build_id, build_id)
        inputs=receipt['inputs']
        if source_inputs(Path(inputs['directory']),Path(inputs['source']),cwd,inputs['algorithm'],inputs.get('worker_source')) != inputs:
            return False
    return True


def snapshot_harness(folder, source):
    snapshot = folder/'source-snapshot'; snapshot.mkdir()
    for path in Path(source).iterdir():
        if path.is_file():
            shutil.copy2(path, snapshot/path.name)
    return {p.name:sha(p) for p in snapshot.iterdir()}


def harness_unchanged(source, manifest):
    return {p.name:sha(p) for p in Path(source).iterdir() if p.is_file()} == manifest


def observation_key(row):
    if any(not isinstance(row[k],int) or isinstance(row[k],bool) for k in ('seed','block')):
        raise ValueError('observation has an invalid schedule identity')
    if any(not isinstance(row[k],str) or not row[k] for k in ('algorithm','phase')):
        raise ValueError('observation has an invalid schedule identity')
    return (identity(row['case']), row['algorithm'], row['seed'], row['phase'], row['block'])


def expected_schedule(protocol):
    cases = protocol.get('cases'); seeds = protocol.get('seeds'); algorithms = protocol.get('algorithms')
    if not all(isinstance(values, list) and values for values in (cases, seeds, algorithms)):
        raise ValueError('protocol must contain non-empty cases, seeds and algorithms')
    blocks = protocol.get('blocks', 1); phase = protocol.get('phase')
    if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
        raise ValueError('protocol has an invalid block count')
    if not isinstance(phase, str) or not phase:
        raise ValueError('protocol has no phase')
    if any(not isinstance(c, dict) or not isinstance(c.get('fixture'), str) or not c['fixture']
           or not isinstance(c.get('cap'), int) or isinstance(c['cap'], bool) or c['cap'] < 1
           or not isinstance(c.get('options'), dict) for c in cases):
        raise ValueError('protocol has an invalid case')
    if any(not isinstance(s, int) or isinstance(s, bool) for s in seeds) or any(not isinstance(a, str) or not a for a in algorithms):
        raise ValueError('protocol has invalid seeds or algorithms')
    schedule = {(identity(c), a, s, phase, b) for c,a,s,b in itertools.product(cases, algorithms, seeds, range(blocks))}
    if len(schedule) != len(cases)*len(seeds)*len(algorithms)*blocks:
        raise ValueError('protocol has duplicate schedule dimensions')
    return schedule


def verify_schedule(protocol, completion, rows):
    expected = expected_schedule(protocol)
    if completion.get('expected') != len(expected):
        raise ValueError('completion count differs from protocol')
    if completion.get('observations') != len(rows) or len(rows) != len(expected):
        raise ValueError('incomplete or extra observations')
    if completion.get('candidate_source_unchanged') is not True:
        raise ValueError('candidate source changed or completion evidence is missing')
    try:
        keys = [observation_key(r) for r in rows]
        tags = [r['tag'] for r in rows]
    except (KeyError, TypeError) as error:
        raise ValueError('observation has no schedule identity') from error
    if len(set(keys)) != len(expected) or set(keys) != expected:
        raise ValueError('observation schedule differs from protocol')
    if any(not isinstance(t, str) or not t or Path(t).name != t or t in ('.', '..') for t in tags) or len(set(tags)) != len(tags):
        raise ValueError('invalid or duplicate observation tags')
    statuses = {'completed', 'missing_fixture', 'timeout', 'algorithm_error', 'validation_error'}
    if any(r.get('status') not in statuses for r in rows):
        raise ValueError('invalid observation status')
