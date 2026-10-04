"""Synthetic records test the validator, not RSI accuracy."""
import copy
import unittest
import subprocess
import run


def records():
    result = [dict(type='protocol', cases=[list(c) for c in run.CASES], seeds=run.SEEDS,
                   blocks=run.BLOCKS, samples=2048, target=run.TARGET, sample_seed=20260929)]
    for topology, n, chi in run.CASES:
        for phase in ['accuracy', 'timing']:
            for block in range(1 if phase == 'accuracy' else run.BLOCKS):
                for seed in run.SEEDS:
                    for algorithm in ['rsi', 'treeaci']:
                        diagnostics = dict(edge_ranks=[
                            [i - 1 if topology == 'chain' else (i - 1) // 2 if topology == 'binary' else 0, i, 1]
                            for i in range(1, n)])
                        if algorithm == 'treeaci':
                            diagnostics.update(termination='Converged', sweep_ranks=[1, 1],
                                               local_errors=[0.0, 0.0], global_pivots=[0, 0])
                        else:
                            diagnostics.update(sketch_dim=(chi * chi + 1) // 2 + 5,
                                               local_pivots=[0.0] * (n - 1))
                        result.append(dict(type='observation', case=f'{topology}-n{n}-chi{chi}',
                                           phase=phase, block=block, seed=seed, algorithm=algorithm,
                                           seconds=1.0, error=0.0, cap=chi * chi, diagnostics=diagnostics))
    return result


class Validation(unittest.TestCase):
    def test_rsi_has_no_aci_dependency(self):
        graph = subprocess.check_output(
            ["cargo", "tree", "--locked", "-p", "tensor4all-treersi", "--edges", "normal,build,dev"],
            cwd=run.ROOT, text=True)
        self.assertNotIn("tensor4all-treeaci", graph)
        self.assertNotIn("tensor4all-aci ", graph)

    def test_complete(self):
        self.assertEqual(len(run.validate(records())), len(run.CASES))

    def test_rejects_false_success(self):
        for field, value in [('error', 1.0), ('error', float('nan')), ('error', None),
                             ('error', True), ('seconds', 0.0), ('type', 'failure'), ('cap', 999)]:
            data = records()
            data[-1].update({field: value, 'matched': True})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                run.validate(data)

    def test_rejects_missing_diagnostics_or_hidden_larger_ranks(self):
        for bad in [None, {}, {'edge_ranks': [[0, 1, 999]]}]:
            data = records()
            data[-1]['diagnostics'] = bad
            with self.assertRaises(ValueError):
                run.validate(data)
        data = records()
        data[-1]['diagnostics']['edge_ranks'][0][2] = data[-1]['cap'] + 1
        with self.assertRaises(ValueError):
            run.validate(data)

    def test_rejects_invalid_algorithm_diagnostics(self):
        for algorithm, field, value in [
                ('rsi', 'sketch_dim', 999), ('rsi', 'local_pivots', []),
                ('rsi', 'local_pivots', [float('nan')] * 15),
                ('treeaci', 'termination', 'Unknown'),
                ('treeaci', 'local_errors', [float('nan'), 0.0]),
                ('treeaci', 'sweep_ranks', [999, 999]),
                ('treeaci', 'global_pivots', [False, 0])]:
            data = records()
            row = next(r for r in data[1:] if r['algorithm'] == algorithm)
            row['diagnostics'][field] = value
            with self.subTest(algorithm=algorithm, field=field), self.assertRaises(ValueError):
                run.validate(data)

    def test_rank_limited_does_not_by_itself_mean_bad_accuracy(self):
        data = records()
        for row in data[1:]:
            if row['algorithm'] == 'treeaci':
                row['diagnostics']['termination'] = 'RankLimited'
        self.assertEqual(len(run.validate(data)), len(run.CASES))

    def test_missing_duplicate_and_changed_protocol(self):
        original = records()
        changed = copy.deepcopy(original)
        changed[0]['target'] = 1.0
        for data in [original[:-1], original + [original[-1]], changed, []]:
            with self.assertRaises(ValueError):
                run.validate(data)


if __name__ == '__main__':
    unittest.main()
