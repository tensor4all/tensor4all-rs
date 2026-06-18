# tensor4all-aci

## src/batch.rs

### `pub fn new(values: & 'a [T], n_inputs: usize, n_points: usize) -> Result < Self >` (impl ElementwiseBatch < 'a , T >)

Creates a borrowed column-major batch view. `values` must contain exactly `n_inputs * n_points` entries in column-major order. Both `n_inputs` and `n_points` must be nonzero.

### `pub fn n_inputs(&self) -> usize` (impl ElementwiseBatch < 'a , T >)

Returns the number of operator inputs per interpolation point.

### `pub fn n_points(&self) -> usize` (impl ElementwiseBatch < 'a , T >)

Returns the number of interpolation points in the batch.

### `pub fn get(&self, input: usize, point: usize) -> Result < T >` (impl ElementwiseBatch < 'a , T >)

Returns one value using column-major indexing. The returned value is `values[input + n_inputs * point]`, so `input` varies fastest in the flat buffer.

### `pub fn as_col_major_slice(&self) -> & 'a [T]` (impl ElementwiseBatch < 'a , T >)

Returns the borrowed flat slice in column-major input/point layout.

## src/elementwise.rs

### `pub fn elementwise_batched(op: F, inputs: & [TensorTrain < T >], options: & AciOptions < T >) -> Result < AciResult < T > >`

Runs batched elementwise ACI over tensor-train inputs. This function approximates the pointwise application of `op` to `inputs`. The callback receives batches in column-major input/point layout through

### ` fn elementwise_batched_one_site(op: F, inputs: & [TensorTrain < T >]) -> Result < AciResult < T > >`

### `pub fn elementwise(op: F, inputs: & [TensorTrain < T >], options: & AciOptions < T >) -> Result < AciResult < T > >`

Runs scalar elementwise ACI over tensor-train inputs. This convenience wrapper evaluates `op` once per interpolation point. The callback receives one value from each input tensor train in input order and

### `pub(crate) fn ranks_are_stable(ranks: & [usize], min_iters: usize) -> bool`

### `pub(crate) fn convergence_criterion_like_julia(iteration: usize, ranks: & [usize], errors: & [f64], min_iters: usize, tolerance: f64) -> bool`

### `pub(crate) fn error_metric(max_pivot_error: f64, max_sampled_scale: f64, scale_tolerance: bool) -> f64`

### `pub(crate) fn max_error_metric(pivot_errors: & [f64], pivot_scales: & [f64], scale_tolerance: bool) -> f64`

## src/local.rs

### ` fn local_setup_batching_enabled() -> bool`

### ` fn local_setup_batching_enabled() -> bool`

### ` fn local_materialize_batching_enabled() -> bool`

### ` fn local_materialize_batching_enabled() -> bool`

### `pub(crate) fn value(&self, row: usize, col: usize) -> Result < T >` (impl LocalInputFactors < T >)

### ` fn materialize_values(&self) -> Result < Vec < T > >` (impl LocalInputFactors < T >)

### ` fn left_offset(&self, left_pivot: usize, site_left: usize, middle: usize) -> usize` (impl LocalInputFactors < T >)

### ` fn right_offset(&self, middle: usize, site_right: usize, right_pivot: usize) -> usize` (impl LocalInputFactors < T >)

### `pub(crate) fn new(problem: & 'a ElementwiseProblem < T >, bond: usize, operator: & 'a F) -> Result < Self >` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn new_with_setup_timing(problem: & 'a ElementwiseProblem < T >, bond: usize, operator: & 'a F, timing: & mut LocalInputSetupTiming) -> Result < Self >` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn nrows(&self) -> usize` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn ncols(&self) -> usize` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn fill_local_block(&self, rows: & [usize], cols: & [usize], out: & mut [T]) -> Result < () >` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn materialize_local_matrix(&self) -> Result < Matrix < T > >` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn materialize_input_values(&self) -> Result < Vec < T > >` (impl LocalBlockEvaluator < 'a , T >)

### ` fn shared_middle_dim(&self) -> Option < usize >` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn apply_operator_to_input_values(&self, input_values: & [T]) -> Result < Matrix < T > >` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn fill_local_block_or_zero(&self, rows: & [usize], cols: & [usize], out: & mut [T])` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn take_error(&self) -> Option < AciError >` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn clear_cache(&self)` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn max_output_abs(&self) -> f64` (impl LocalBlockEvaluator < 'a , T >)

### ` fn record_error(&self, err: AciError)` (impl LocalBlockEvaluator < 'a , T >)

### ` fn entry_key(&self, row: usize, col: usize) -> Result < usize >` (impl LocalBlockEvaluator < 'a , T >)

### ` fn record_output_scale(&self, values: & [T])` (impl LocalBlockEvaluator < 'a , T >)

### `pub(crate) fn local_input_factors_for_problem(problem: & ElementwiseProblem < T >, bond: usize) -> Result < Vec < LocalInputFactors < T > > >`

### `pub(crate) fn local_input_factors_for_problem_with_timing(problem: & ElementwiseProblem < T >, bond: usize, timing: & mut LocalInputSetupTiming) -> Result < Vec < LocalInputFactors < T > > >`

### ` fn local_input_factor_dims_for_problem(problem: & ElementwiseProblem < T >, bond: usize) -> Result < Vec < LocalInputFactorDims > >`

### ` fn local_input_factor_dims(problem: & ElementwiseProblem < T >, input: usize, bond: usize) -> Result < LocalInputFactorDims >`

### ` fn build_local_input_factors_from_dims_with_timing(problem: & ElementwiseProblem < T >, bond: usize, dims: & [LocalInputFactorDims], timing: & mut LocalInputSetupTiming) -> Result < Vec < LocalInputFactors < T > > >`

### ` fn build_local_input_factors_from_dims(problem: & ElementwiseProblem < T >, bond: usize, dims: & [LocalInputFactorDims]) -> Result < Vec < LocalInputFactors < T > > >`

### ` fn assemble_local_input_factors(dims: & [LocalInputFactorDims], left_values: Vec < Vec < T > >, right_values: Vec < Vec < T > >) -> Result < Vec < LocalInputFactors < T > > >`

### ` fn local_input_factors_from_values(dims: LocalInputFactorDims, left_values: Vec < T >, right_values: Vec < T >) -> Result < LocalInputFactors < T > >`

### ` fn shared_input_factor_dims(dims: & [LocalInputFactorDims]) -> Option < LocalInputFactorDims >`

### ` fn build_left_factors(problem: & ElementwiseProblem < T >, bond: usize, dims: & [LocalInputFactorDims]) -> Result < Vec < Vec < T > > >`

### ` fn build_right_factors(problem: & ElementwiseProblem < T >, bond: usize, dims: & [LocalInputFactorDims]) -> Result < Vec < Vec < T > > >`

### ` fn build_left_factor(problem: & ElementwiseProblem < T >, input: usize, bond: usize) -> Result < Vec < T > >`

### ` fn build_right_factor(problem: & ElementwiseProblem < T >, input: usize, bond: usize) -> Result < Vec < T > >`

### ` fn validate_local_shapes(problem: & ElementwiseProblem < T >, bond: usize) -> Result < (usize , usize) >`

### ` fn local_left_frame(problem: & ElementwiseProblem < T >, input: usize, bond: usize) -> Result < & tensor4all_tensorbackend :: Matrix < T > >`

### ` fn local_right_frame(problem: & ElementwiseProblem < T >, input: usize, bond: usize) -> Result < & tensor4all_tensorbackend :: Matrix < T > >`

### ` fn local_factor_error(context: & str, err: impl std :: fmt :: Display) -> AciError`

### ` fn validate_indices(kind: & 'static str, indices: & [usize], len: usize) -> Result < () >`

### ` fn checked_local_mul(lhs: usize, rhs: usize, description: & str) -> Result < usize >`

## src/options.rs

### ` fn default() -> Self` (impl AciOptions < T >)

## src/random_tt.rs

### `pub(crate) fn initial_guess(inputs: & [TensorTrain < T >], options: & AciOptions < T >) -> Result < TensorTrain < T > >`

### ` fn validate_existing_initial_guess(guess: & TensorTrain < T >, site_dims: & [usize], max_bond_dim: usize) -> Result < () >`

### ` fn initial_guess_core_dims(site_dims: & [usize], link_dims: & [usize]) -> Vec < (usize , usize , usize) >`

### ` fn default_link_dims(inputs: & [TensorTrain < T >], site_dims: & [usize], max_bond_dim: usize) -> Result < Vec < usize > >`

### ` fn random_core(left_dim: usize, site_dim: usize, right_dim: usize, rng: & mut ChaCha8Rng) -> Result < Tensor3 < T > >`

### `pub(crate) fn initial_guess_core_entry_count(left_dim: usize, site_dim: usize, right_dim: usize) -> Result < usize >`

### `pub(crate) fn initial_guess_total_entry_count(core_dims: & [(usize , usize , usize)]) -> Result < usize >`

### `pub(crate) fn initial_guess_existing_entry_count(guess: & TensorTrain < T >) -> Result < usize >`

### ` fn checked_add(lhs: usize, rhs: usize, description: & str) -> Result < usize >`

### ` fn checked_mul(lhs: usize, rhs: usize, description: & str) -> Result < usize >`

## src/scalar.rs

### `pub(super) fn sample_standard_normal(rng: & mut rand_chacha :: ChaCha8Rng) -> Self` (trait Sealed)

### ` fn sample_standard_normal(rng: & mut rand_chacha :: ChaCha8Rng) -> Self` (impl f64)

### ` fn sample_standard_normal(rng: & mut rand_chacha :: ChaCha8Rng) -> Self` (impl Complex64)

### `pub(crate) fn sample_standard_normal(rng: & mut rand_chacha :: ChaCha8Rng) -> T`

## src/state.rs

### ` fn frame_batching_enabled() -> bool`

### ` fn frame_batching_enabled() -> bool`

### ` fn from_core(core: & Tensor3 < T >) -> Self` (impl InputCoreMatrices < T >)

### `pub(crate) fn new(inputs: Vec < TensorTrain < T > >, options: AciOptions < T >) -> Result < Self >` (impl ElementwiseProblem < T >)

### `pub(crate) fn len(&self) -> usize` (impl ElementwiseProblem < T >)

### `pub(crate) fn n_inputs(&self) -> usize` (impl ElementwiseProblem < T >)

### `pub(crate) fn input_core_left_matrix(&self, input: usize, site: usize) -> & Matrix < T >` (impl ElementwiseProblem < T >)

### `pub(crate) fn input_core_right_matrix(&self, input: usize, site: usize) -> & Matrix < T >` (impl ElementwiseProblem < T >)

### `pub(crate) fn left_frame_shape(&self, input: usize, site: usize) -> Option < (usize , usize) >` (impl ElementwiseProblem < T >)

### `pub(crate) fn right_frame_shape(&self, input: usize, site: usize) -> Option < (usize , usize) >` (impl ElementwiseProblem < T >)

### `pub(crate) fn left_frame_value(&self, input: usize, site: usize, row: usize, col: usize) -> Option < T >` (impl ElementwiseProblem < T >)

### `pub(crate) fn right_frame_value(&self, input: usize, site: usize, row: usize, col: usize) -> Option < T >` (impl ElementwiseProblem < T >)

### `pub(crate) fn local_input_shape(&self, input: usize, bond: usize) -> Result < (usize , usize) >` (impl ElementwiseProblem < T >)

### `pub(crate) fn local_input_value(&self, input: usize, bond: usize, row: usize, col: usize) -> Result < T >` (impl ElementwiseProblem < T >)

### `pub(crate) fn update_left_frame(&mut self, input: usize, site: usize, row_indices: & [usize]) -> Result < () >` (impl ElementwiseProblem < T >)

### `pub(crate) fn update_right_frame(&mut self, input: usize, site: usize, col_indices: & [usize]) -> Result < () >` (impl ElementwiseProblem < T >)

### `pub(crate) fn update_left_frames(&mut self, site: usize, row_indices: & [usize]) -> Result < () >` (impl ElementwiseProblem < T >)

### `pub(crate) fn update_right_frames(&mut self, site: usize, col_indices: & [usize]) -> Result < () >` (impl ElementwiseProblem < T >)

### ` fn batched_left_frame_updates(&self, site: usize, row_indices: & [usize]) -> Result < Option < Vec < Matrix < T > > > >` (impl ElementwiseProblem < T >)

### ` fn batched_right_frame_updates(&self, site: usize, col_indices: & [usize]) -> Result < Option < Vec < Matrix < T > > > >` (impl ElementwiseProblem < T >)

### `pub(crate) fn local_update(&mut self, bond: usize, left_orthogonal: bool, options: & AciOptions < T >, op: & mut F) -> Result < () >` (impl ElementwiseProblem < T >)

### ` fn initialize_right_frames(&mut self) -> Result < () >` (impl ElementwiseProblem < T >)

### ` fn local_input_context(&self, input: usize, bond: usize) -> Result < LocalInputContext < '_ , T > >` (impl ElementwiseProblem < T >)

### ` fn unit_frame() -> Matrix < T >`

### ` fn frame_value(frames: & [Vec < Option < Matrix < T > > >], input: usize, site: usize, row: usize, col: usize) -> Option < T >`

### ` fn validate_input_site(input: usize, site: usize, n_inputs: usize, n_sites: usize) -> Result < () >`

### ` fn validate_selection(kind: & 'static str, indices: & [usize], len: usize) -> Result < () >`

### ` fn checked_frame_mul(lhs: usize, rhs: usize, description: & str) -> Result < usize >`

### ` fn missing_frame(kind: & 'static str, input: usize, site: usize) -> AciError`

### ` fn right_matrix_julia_order(core: & Tensor3 < T >) -> Matrix < T >`

### ` fn left_matrix_julia_order(core: & Tensor3 < T >) -> Matrix < T >`

### `pub(crate) fn matrix_to_tensor3(matrix: & Matrix < T >, left_dim: usize, site_dim: usize, right_dim: usize) -> Result < Tensor3 < T > >`

### `pub(crate) fn matrix_into_tensor3(matrix: Matrix < T >, left_dim: usize, site_dim: usize, right_dim: usize) -> Result < Tensor3 < T > >`

### `pub(crate) fn right_factor_to_tensor3(matrix: & Matrix < T >, left_dim: usize, site_dim: usize, right_dim: usize) -> Result < Tensor3 < T > >`

### `pub(crate) fn right_factor_into_tensor3(matrix: Matrix < T >, left_dim: usize, site_dim: usize, right_dim: usize) -> Result < Tensor3 < T > >`

### ` fn matmul_checked_owned(left: Matrix < T >, right: Matrix < T >, site: usize) -> Result < Matrix < T > >`

### ` fn frame_matmul_checked(left: & Matrix < T >, right: & Matrix < T >, direction: & 'static str, input: usize, site: usize) -> Result < Matrix < T > >`

### ` fn frame_matmul_error(direction: & 'static str, err: impl std :: fmt :: Display) -> AciError`

## src/tests.rs

### ` fn tensor_train_with_link_dims(site_dims: & [usize], link_dims: & [usize]) -> TensorTrain < f64 >`

### ` fn local_test_problem() -> ElementwiseProblem < f64 >`

### ` fn explicit_local_value(problem: & ElementwiseProblem < f64 >, input: usize, bond: usize, row: usize, col: usize) -> f64`

### ` fn multiply_batch(batch: ElementwiseBatch < '_ , f64 >, output: & mut [f64]) -> crate :: Result < () >`

### ` fn zero_batch(batch: ElementwiseBatch < '_ , f64 >, output: & mut [f64]) -> crate :: Result < () >`

### ` fn assert_solution_is_zero_on_binary_three_site_grid(problem: & ElementwiseProblem < f64 >)`

### ` fn elementwise_multiplies_constant_tensor_trains()`

### ` fn scalar_and_batched_paths_match()`

### ` fn elementwise_batched_propagates_operator_error()`

### ` fn elementwise_single_site_scalar_evaluates_operator()`

### ` fn elementwise_batched_single_site_uses_column_major_batch_layout()`

### ` fn elementwise_batched_single_site_propagates_operator_error()`

### ` fn relative_error_metric_normalizes_by_sampled_scale()`

### ` fn relative_error_metric_pairs_each_bond_with_its_scale()`

### ` fn rank_stability_matches_julia_min_iter_window()`

### ` fn convergence_criterion_matches_julia_algorithm()`

### ` fn default_options_are_conservative()`

### ` fn elementwise_batch_uses_column_major_input_point_layout()`

### ` fn elementwise_batch_rejects_bad_length()`

### ` fn elementwise_batch_rejects_zero_inputs()`

### ` fn elementwise_batch_rejects_zero_points()`

### ` fn elementwise_batch_rejects_shape_overflow()`

### ` fn elementwise_batch_rejects_out_of_range_input_index()`

### ` fn elementwise_batch_rejects_out_of_range_point_index()`

### ` fn elementwise_problem_initializes_boundary_frames()`

### ` fn elementwise_problem_initializes_all_rank_one_right_frames()`

### ` fn elementwise_problem_handles_one_site_input()`

### ` fn elementwise_problem_preserves_initial_guess_values()`

### ` fn elementwise_problem_updates_left_frame_for_selected_rows()`

### ` fn elementwise_problem_updates_left_frame_values()`

### ` fn elementwise_problem_updates_all_left_frames_for_selected_rows()`

### ` fn elementwise_problem_updates_right_frame_for_selected_columns()`

### ` fn elementwise_problem_updates_right_frame_values()`

### ` fn elementwise_problem_rejects_invalid_frame_selection_indices()`

### ` fn elementwise_problem_one_bond_local_update_matches_dense_product_left_orthogonal()`

### ` fn elementwise_problem_one_bond_local_update_matches_dense_product_right_orthogonal()`

### ` fn elementwise_problem_one_bond_local_update_returns_operator_error_side_channel()`

### ` fn elementwise_problem_one_bond_zero_update_keeps_left_frames_nonzero_dimensional()`

### ` fn elementwise_problem_one_bond_zero_update_keeps_right_frames_nonzero_dimensional()`

### ` fn local_input_value_matches_explicit_two_site_contraction()`

### ` fn local_input_factors_match_explicit_two_site_contraction_for_all_inputs()`

### ` fn local_input_factors_report_bounds_errors_and_record_timing()`

### ` fn tensor3_reshape_helpers_accept_and_reject_shapes()`

### ` fn local_block_evaluator_uses_matrix_luci_point_order_and_batch_layout()`

### ` fn local_block_evaluator_materializes_full_matrix_in_column_major_order()`

### ` fn local_block_evaluator_serves_duplicate_entries_from_cache()`

### ` fn local_block_evaluator_cache_can_be_cleared()`

### ` fn local_block_evaluators_with_different_operators_have_separate_caches()`

### ` fn local_block_evaluator_or_zero_records_operator_error_once()`

### ` fn local_block_evaluator_or_zero_records_local_error_once()`

### ` fn local_input_value_rejects_out_of_range_indices()`

### ` fn local_input_value_rejects_missing_frames()`

### ` fn validate_inputs_rejects_empty_inputs()`

### ` fn validate_inputs_rejects_zero_site_tensor_train()`

### ` fn validate_inputs_rejects_zero_physical_dim_in_first_input()`

### ` fn validate_inputs_rejects_zero_physical_dim_in_later_input()`

### ` fn validate_inputs_rejects_zero_internal_bond_dim_in_first_input()`

### ` fn validate_inputs_rejects_zero_internal_bond_dim_in_later_input()`

### ` fn validate_inputs_rejects_length_mismatch()`

### ` fn validate_inputs_rejects_site_dim_mismatch()`

### ` fn validate_options_rejects_zero_max_iters()`

### ` fn validate_options_rejects_zero_max_bond_dim()`

### ` fn validate_options_rejects_min_iters_above_max_iters()`

### ` fn validate_options_rejects_zero_min_iters()`

### ` fn validate_options_rejects_negative_tolerance()`

### ` fn validate_options_rejects_nan_tolerance()`

### ` fn validate_options_rejects_infinite_tolerance()`

### ` fn default_initial_guess_matches_input_site_dims()`

### ` fn initial_guess_link_dims_are_empty_for_one_site_input()`

### ` fn initial_guess_link_dims_are_limited_by_max_bond_dim()`

### ` fn initial_guess_link_dims_are_limited_by_physical_left_right_products()`

### ` fn initial_guess_link_dims_are_limited_by_minimum_input_link_dim()`

### ` fn initial_guess_is_deterministic_for_same_seed()`

### ` fn initial_guess_accepts_compatible_explicit_guess()`

### ` fn initial_guess_zero_initializes_nonzero_dimensional_right_frames()`

### ` fn initial_guess_rejects_incompatible_explicit_guess_site_dims()`

### ` fn explicit_initial_guess_rejects_rank_above_max_bond_dim()`

### ` fn explicit_initial_guess_rejects_zero_bond_dimension()`

### ` fn complex_initial_guess_is_deterministic()`

### ` fn initial_guess_existing_entry_count_matches_explicit_guess_cores()`

### ` fn initial_guess_rejects_huge_non_overflowing_core_size()`

### ` fn initial_guess_rejects_oversized_total_entries()`

### ` fn total(self) -> Duration` (impl LocalStepTiming)

### ` fn setup_other(self) -> Duration` (impl LocalStepTiming)

### ` fn step_timing_link_dims(n_sites: usize, local_dim: usize, chi: usize) -> Vec < usize >`

### ` fn step_timing_core_value(input_index: usize, site: usize, physical: usize, left: usize, right: usize, left_dim: usize, right_dim: usize) -> f64`

### ` fn step_timing_deterministic_tt(input_index: usize, n_sites: usize, chi: usize) -> TensorTrain < f64 >`

### ` fn step_timing_inputs(n_sites: usize, chi: usize) -> Vec < TensorTrain < f64 > >`

### ` fn step_timing_multiply_batch(batch: ElementwiseBatch < '_ , f64 >, output: & mut [f64]) -> crate :: Result < () >`

### ` fn timed_local_update(problem: & mut ElementwiseProblem < f64 >, bond: usize, left_orthogonal: bool, options: & AciOptions < f64 >, op: & mut F, timing: & mut LocalStepTiming) -> crate :: Result < () >`

### ` fn timed_aci_run(n_sites: usize, chi: usize, min_iters: usize, fixed_sweeps: Option < usize >) -> LocalStepTiming`

### ` fn duration_ms(duration: Duration) -> f64`

### ` fn median_ms(values: Vec < f64 >) -> f64`

### ` fn local_update_step_timing()`

## src/validation.rs

### `pub(crate) fn validate_options(options: & crate :: AciOptions < T >) -> Result < () >`

### `pub(crate) fn validate_inputs(inputs: & [TensorTrain < T >]) -> Result < Vec < usize > >`

### ` fn validate_positive_site_dims(site_dims: & [usize]) -> Result < () >`

### ` fn validate_positive_core_dims(input: & TensorTrain < T >, input_index: usize) -> Result < () >`
