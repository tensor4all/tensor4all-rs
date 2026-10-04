import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evidence import archive_worker, checked_workers, identity, sha, source_inputs, workers_unchanged
from summarize import records
from verify_artifacts import expected_observation_count, verify_run_evidence, verify_trace_parity


class ArtifactEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root/'run'; self.folder.mkdir()
        self.out = self.root/'workers'
        builds = {}; hashes = {}; self.receipts = {}
        for algorithm in ('treeaci', 'rsi'):
            directory = self.out/algorithm; directory.mkdir(parents=True)
            (directory/'worker').write_bytes(f'{algorithm}-worker'.encode())
            (directory/'main.rs').write_text('fn main() {}')
            inputs = dict(files={str(directory/'main.rs'):sha(directory/'main.rs')},
                          settings={}, algorithm=algorithm, profile='release', directory=str(directory), source=str(self.root))
            archive = archive_worker(directory, self.out, inputs)
            builds[algorithm] = archive.name; hashes[algorithm] = sha(archive/'worker')
            self.receipts[algorithm] = json.loads((archive/'receipt.json').read_text())
        self.protocol = dict(cases=[dict(fixture='a', cap=2, options={}),
                                   dict(fixture='b', cap=3, options={'local_tolerance':1e-14})],
            seeds=[1, 2], algorithms=['treeaci', 'rsi'], blocks=3, phase='repeat',
            worker_builds=builds, worker_hashes=hashes,
            candidate_source_sha256=self.receipts['rsi']['source_sha256'])
        snapshot = self.folder/'source-snapshot'; snapshot.mkdir()
        (snapshot/'worker.rs').write_text('source')
        self.protocol['source_files'] = {'worker.rs':sha(snapshot/'worker.rs')}
        self.expected = expected_observation_count(self.protocol)
        self.completion = dict(expected=self.expected, observations=self.expected, candidate_source_unchanged=True)
        self.rows = [dict(case=c, algorithm=a, seed=s, phase='repeat', block=b,
                          tag=f'observation-{i}', status='algorithm_error')
            for i,(c,a,s,b) in enumerate(itertools.product(self.protocol['cases'],
                self.protocol['algorithms'], self.protocol['seeds'], range(3)))]

    def verify(self):
        verify_run_evidence(self.folder, self.protocol, self.completion, self.rows, self.out)

    def test_complete_evidence_and_cartesian_schedule_are_accepted(self):
        self.assertEqual(self.expected, 24)
        self.verify()

    def test_changed_or_missing_completion_evidence_is_rejected(self):
        for value in (False, None):
            self.completion['candidate_source_unchanged'] = value
            with self.assertRaisesRegex(ValueError, 'candidate source changed'):
                self.verify()

    def test_incomplete_observation_schedule_is_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, 'incomplete or extra observations'):
            self.verify()

    def test_count_correct_wrong_schedule_is_rejected(self):
        for field,value in [('case', dict(fixture='unexpected', cap=2, options={})),
                            ('algorithm','unexpected'), ('seed',999), ('phase','wrong'), ('block',99)]:
            rows = copy.deepcopy(self.rows); rows[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'schedule differs'):
                verify_run_evidence(self.folder, self.protocol, self.completion, rows, self.out)
        rows = copy.deepcopy(self.rows); rows[0]['case']['options']={'local_tolerance':0.0}
        with self.assertRaisesRegex(ValueError, 'schedule differs'):
            verify_run_evidence(self.folder, self.protocol, self.completion, rows, self.out)

    def test_boolean_and_float_schedule_coordinates_are_rejected(self):
        for field,value in [('seed',True),('seed',1.0),('block',False),('block',0.0)]:
            rows=copy.deepcopy(self.rows);rows[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaisesRegex(ValueError,'invalid schedule identity'):
                verify_run_evidence(self.folder,self.protocol,self.completion,rows,self.out)

    def test_duplicate_keys_with_unique_tags_are_rejected(self):
        self.rows[0] = dict(self.rows[1], tag='different-tag')
        with self.assertRaisesRegex(ValueError, 'schedule differs'):
            self.verify()

    def test_missing_source_manifest_is_rejected(self):
        for value in (None, {}):
            self.protocol['source_files'] = value
            with self.assertRaisesRegex(ValueError, 'source-file manifest'):
                self.verify()

    def test_modified_harness_snapshot_is_rejected(self):
        (self.folder/'source-snapshot'/'worker.rs').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source snapshot mismatch'):
            self.verify()

    def test_replaced_archived_binary_is_rejected(self):
        (self.out/'builds'/self.protocol['worker_builds']['rsi']/'worker').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'worker hash mismatch'):
            self.verify()

    def test_modified_archived_library_source_is_rejected(self):
        receipt = self.receipts['rsi']; original = next(iter(receipt['inputs']['files']))
        (self.out/'builds'/self.protocol['worker_builds']['rsi']/'sources'/identity(original)).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'build source snapshot mismatch'):
            self.verify()

    def test_candidate_identity_must_match_build_receipt(self):
        self.protocol['candidate_source_sha256']='new-source-without-rebuild'
        with self.assertRaisesRegex(ValueError, 'candidate source differs'):
            self.verify()

    def test_no_receipt_cannot_schedule_measurements(self):
        (self.out/'rsi'/'build-receipt.json').unlink()
        with self.assertRaisesRegex(ValueError, 'no build receipt'):
            checked_workers(self.out, {'rsi':self.root}, self.root, ['rsi'])

    def test_stale_worker_source_or_build_settings_cannot_schedule_measurements(self):
        inputs=self.receipts['rsi']['inputs']
        with patch('evidence.source_inputs', return_value=inputs):
            workers,evidence=checked_workers(self.out, {'rsi':self.root}, self.root, ['rsi'])
            self.assertEqual(workers['rsi'], self.out/'builds'/self.protocol['worker_builds']['rsi']/'worker')
            self.assertEqual(evidence['candidate_source_sha256'],self.protocol['candidate_source_sha256'])
        for key,value in [('files',{'new-source':'changed'}),('settings',{'rustflags':'changed'})]:
            changed=dict(inputs,**{key:value})
            with patch('evidence.source_inputs', return_value=changed), self.assertRaisesRegex(ValueError, 'stale worker'):
                checked_workers(self.out, {'rsi':self.root}, self.root, ['rsi'])
            with patch('evidence.source_inputs', return_value=changed):
                self.assertFalse(workers_unchanged(self.out, {'rsi':self.protocol['worker_builds']['rsi']}, self.root))

    def test_protocol_requires_explicit_unique_dimensions(self):
        for field,value in [('cases', []),('seeds',[1,1]),('algorithms',None),('blocks',True),('phase',None)]:
            protocol=dict(self.protocol,**{field:value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                expected_observation_count(protocol)

    def test_summarizer_rejects_count_correct_wrong_schedule(self):
        self.rows[0]['seed']=999
        for name,data in [('protocol.json',self.protocol),('completion.json',self.completion)]:
            (self.folder/name).write_text(json.dumps(data))
        (self.folder/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        with self.assertRaisesRegex(ValueError, 'schedule differs'):
            records(self.folder)

    def test_trace_parity_is_checked_against_actual_baseline(self):
        baseline=self.rows[0]; replay=dict(baseline, phase='trace', baseline_directory='run',
            baseline_tag=baseline['tag'], matches_untouched_main=True)
        with patch('verify_artifacts.records',return_value=[baseline]):
            verify_trace_parity([replay],self.root)
            replay['status']='completed'
            with self.assertRaisesRegex(ValueError,'parity failure'):
                verify_trace_parity([replay],self.root)


class BuildSourceClosureTests(unittest.TestCase):
    def test_metadata_closure_records_dependency_source_lock_config_and_new_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); worker=root/'rsi'; worker.mkdir(); dep=root/'dep'; dep.mkdir()
            (root/'Cargo.toml').write_text('[workspace]\nmembers=["dep"]\nresolver="2"\n')
            (root/'Cargo.lock').write_text('# source lock\n')
            (dep/'Cargo.toml').write_text('[package]\nname="receipt-dep"\nversion="0.1.0"\nedition="2024"\n')
            (dep/'src').mkdir(); (dep/'src/lib.rs').write_text('pub fn value()->i32{1}')
            (worker/'Cargo.toml').write_text('[workspace]\n[package]\nname="receipt-worker"\nversion="0.1.0"\nedition="2024"\n[features]\nrsi=[]\n[dependencies]\nreceipt-dep={path="../dep"}\n')
            (worker/'src').mkdir(); (worker/'src/main.rs').write_text('fn main(){assert_eq!(receipt_dep::value(),1);}')
            original=root/'original-worker.rs'; original.write_text('fn main() {}')
            before=source_inputs(worker,root,root,'rsi',original)
            self.assertIn(str(dep/'src/lib.rs'),before['files'])
            self.assertIn(str(root/'Cargo.toml'),before['files'])
            self.assertIn(str(worker/'Cargo.lock'),before['files'])
            self.assertIn('rustc',before['settings'])
            self.assertIn(str(original),before['files'])
            original.write_text('fn main() { println!("changed"); }')
            self.assertNotEqual(source_inputs(worker,root,root,'rsi',original),before)
            (dep/'src/included.txt').write_text('new input')
            self.assertNotEqual(source_inputs(worker,root,root,'rsi'),before)
            (root/'.cargo').mkdir();(root/'.cargo/config.toml').write_text('[build]\nrustflags=["--cfg=receipt_test"]\n')
            self.assertIn(str(root/'.cargo/config.toml'),source_inputs(worker,root,root,'rsi')['files'])


if __name__=='__main__':
    unittest.main()
