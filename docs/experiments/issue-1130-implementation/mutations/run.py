from pathlib import Path
import json, subprocess, time
root = Path(__file__).resolve().parents[4]
schedule = root / 'crates/tensor4all-partitionedtreetn/src/reconstruction/schedule.rs'
tolerance = schedule.parent / 'schedule/tolerance.rs'
snapshots = {p: p.read_text() for p in (schedule, tolerance)}
out = root / 'docs/experiments/issue-1130-implementation/mutations'
unit = ['cargo', 'test', '-p', 'tensor4all-partitionedtreetn', '--lib']
public = ['cargo', 'test', '-p', 'tensor4all-quanticstransform', '--features', 'partitioned', '--test', 'qft_reconstruction']
mutations = [
    ('forecast-collisions', tolerance, 'active = true;', 'active = false;', [
        (unit + ['forecast_truncates_depth_and_skips_inactive_levels'], ['forecast_truncates_depth_and_skips_inactive_levels']),
        (public + ['tolerance_merges_report_inactive_levels_early_stops_and_zero_budget'], ['tolerance_merges_report_inactive_levels_early_stops_and_zero_budget'])]),
    ('joint-tie-order', tolerance, 'Self::Joint => (0, 0),', 'Self::Joint => (3, 0),', [
        (unit + ['candidate_order_chooses_cheaper_singles_and_deterministic_ties'], ['candidate_order_chooses_cheaper_singles_and_deterministic_ties']),
        (unit + ['serial_joint_probes_keep_the_chosen_partition_and_its_ledger'], ['serial_joint_probes_keep_the_chosen_partition_and_its_ledger'])]),
    ('child-cap-off-by-one', schedule, 'if next.len() >= max_children {', 'if next.len() > max_children {', [
        (unit + ['child_planning_bounds_allocation_and_prunes_before_expansion'], ['child_planning_bounds_allocation_and_prunes_before_expansion'])]),
    ('ignore-term-cap', tolerance, 'let terms_left = context.options.max_terms.saturating_sub(used_terms);', 'let terms_left = context.options.max_work_items.saturating_sub(used_terms);', [
        (unit + ['candidate_limits_cover_work_terms_and_empty_shapes'], ['candidate_limits_cover_work_terms_and_empty_shapes'])]),
    ('maximum-bond-only', schedule, 'logical_parameter_count(&candidate)? < logical_parameter_count(sum)?', 'candidate.max_bond_dim() < sum.max_bond_dim()', [
        (unit + ['tolerance_compression_reduces_other_bonds_when_the_maximum_stays_fixed'], ['tolerance_compression_reduces_other_bonds_when_the_maximum_stays_fixed'])]),
    ('skip-tolerance-compression', schedule, 'compress_item(&sum, context.center, None, context.share, counters)?;', '(sum.clone(), 0.0);', [
        (public + ['tolerance_merges_compress_without_a_rank_goal'], ['tolerance_merges_compress_without_a_rank_goal']),
        (public + ['tolerance_merges_bound_cancellation_and_charge_only_kept_candidates'], ['tolerance_merges_bound_cancellation_and_charge_only_kept_candidates']),
        (public + ['tolerance_merges_handle_selected_multiaxis_sparse_spectator_cells'], ['tolerance_merges_handle_selected_multiaxis_sparse_spectator_cells'])]),
    ('omit-application-ledger', schedule, 'application_error_bound: application_error,', 'application_error_bound: 0.0,', [
        (public + ['tolerance_merges_charge_application_before_the_level_budget'], ['tolerance_merges_charge_application_before_the_level_budget'])]),
]
results = []
try:
    for name, path, old, new, checks in mutations:
        for p, content in snapshots.items():
            p.write_text(content)
        assert snapshots[path].count(old) == 1, (name, snapshots[path].count(old))
        path.write_text(snapshots[path].replace(old, new))
        for i, (command, expected) in enumerate(checks):
            log = out / f'{name}-{i + 1}.txt'
            print(f'MUTATION {name}: {expected[0]}', flush=True)
            run = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write_text(run.stdout.replace(str(root), '<worktree>').replace(str(root.parent / 'tensor4all-rs/target'), '<cargo-target>'))
            killed = run.returncode == 101 and all(any(test in line and ' ... FAILED' in line for line in run.stdout.splitlines()) for test in expected) and 'could not compile' not in run.stdout
            results.append(dict(mutation=name, source=str(path.relative_to(root)), before=old, after=new, tests=expected, command=command, exit_code=run.returncode, killed=killed, log=str(log.relative_to(root))))
            (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(f'  killed={killed}, exit={run.returncode}', flush=True)
            if not killed:
                print(run.stdout[-6000:], flush=True)
                raise SystemExit(f'Mutation not killed: {name}')
finally:
    for p, content in snapshots.items():
        p.write_text(content)
print(f'All {len(results)} test/mutation checks killed their mutation; sources restored.', flush=True)
