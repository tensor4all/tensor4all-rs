"""CFFI definitions for tensor4all C API."""

import cffi

ffi = cffi.FFI()

# C API declarations
ffi.cdef("""
    // Status codes
    typedef int StatusCode;

    // Opaque types
    typedef struct { void* _private; } t4a_index;
    typedef struct { void* _private; } t4a_tensor;

    // Storage kind enum
    typedef enum {
        DenseF64 = 0,
        DenseC64 = 1,
        DiagF64 = 2,
        DiagC64 = 3,
    } t4a_storage_kind;

    // ========================================================================
    // Index functions
    // ========================================================================

    // Lifecycle
    t4a_index* t4a_index_new(size_t dim);
    t4a_index* t4a_index_new_with_tags(size_t dim, const char* tags_csv);
    t4a_index* t4a_index_new_with_id(size_t dim, uint64_t id, const char* tags_csv);
    void t4a_index_release(t4a_index* ptr);
    t4a_index* t4a_index_clone(const t4a_index* ptr);
    int t4a_index_is_assigned(const t4a_index* ptr);

    // Accessors
    StatusCode t4a_index_dim(const t4a_index* ptr, size_t* out_dim);
    StatusCode t4a_index_id(const t4a_index* ptr, uint64_t* out_id);
    StatusCode t4a_index_get_tags(const t4a_index* ptr, uint8_t* buf, size_t buf_len, size_t* out_len);
    int t4a_index_has_tag(const t4a_index* ptr, const char* tag);

    // Modifiers
    StatusCode t4a_index_add_tag(t4a_index* ptr, const char* tag);
    StatusCode t4a_index_set_tags_csv(t4a_index* ptr, const char* tags_csv);

    // ========================================================================
    // Tensor functions
    // ========================================================================

    // Lifecycle
    void t4a_tensor_release(t4a_tensor* ptr);
    t4a_tensor* t4a_tensor_clone(const t4a_tensor* ptr);
    int t4a_tensor_is_assigned(const t4a_tensor* ptr);

    // Constructors
    t4a_tensor* t4a_tensor_new_dense_f64(
        size_t rank,
        const t4a_index** index_ptrs,
        const size_t* dims,
        const double* data,
        size_t data_len
    );
    t4a_tensor* t4a_tensor_new_dense_c64(
        size_t rank,
        const t4a_index** index_ptrs,
        const size_t* dims,
        const double* data_re,
        const double* data_im,
        size_t data_len
    );

    // Accessors
    StatusCode t4a_tensor_get_rank(const t4a_tensor* ptr, size_t* out_rank);
    StatusCode t4a_tensor_get_dims(const t4a_tensor* ptr, size_t* out_dims, size_t buf_len);
    StatusCode t4a_tensor_get_indices(const t4a_tensor* ptr, t4a_index** out_indices, size_t buf_len);
    StatusCode t4a_tensor_get_storage_kind(const t4a_tensor* ptr, t4a_storage_kind* out_kind);
    StatusCode t4a_tensor_get_data_f64(const t4a_tensor* ptr, double* buf, size_t buf_len, size_t* out_len);
    StatusCode t4a_tensor_get_data_c64(const t4a_tensor* ptr, double* buf_re, double* buf_im, size_t buf_len, size_t* out_len);

    // ========================================================================
    // TensorTrain F64 functions
    // ========================================================================

    // Opaque type
    typedef struct { void* _private; } t4a_tt_f64;
    typedef struct { void* _private; } t4a_tt_c64;

    // Lifecycle - F64
    t4a_tt_f64* t4a_tt_f64_new_zeros(const size_t* site_dims, size_t num_sites);
    t4a_tt_f64* t4a_tt_f64_new_constant(const size_t* site_dims, size_t num_sites, double value);
    void t4a_tt_f64_release(t4a_tt_f64* ptr);
    t4a_tt_f64* t4a_tt_f64_clone(const t4a_tt_f64* ptr);

    // Properties - F64
    StatusCode t4a_tt_f64_len(const t4a_tt_f64* ptr, size_t* out_len);
    StatusCode t4a_tt_f64_site_dims(const t4a_tt_f64* ptr, size_t* out_dims, size_t buf_len);
    StatusCode t4a_tt_f64_link_dims(const t4a_tt_f64* ptr, size_t* out_dims, size_t buf_len);
    StatusCode t4a_tt_f64_rank(const t4a_tt_f64* ptr, size_t* out_rank);

    // Evaluation - F64
    StatusCode t4a_tt_f64_evaluate(const t4a_tt_f64* ptr, const size_t* indices, size_t num_indices, double* out_value);
    StatusCode t4a_tt_f64_sum(const t4a_tt_f64* ptr, double* out_sum);
    StatusCode t4a_tt_f64_norm(const t4a_tt_f64* ptr, double* out_norm);
    StatusCode t4a_tt_f64_log_norm(const t4a_tt_f64* ptr, double* out_log_norm);

    // Scaling - F64
    StatusCode t4a_tt_f64_scale_inplace(t4a_tt_f64* ptr, double factor);
    t4a_tt_f64* t4a_tt_f64_scaled(const t4a_tt_f64* ptr, double factor);

    // Full tensor - F64
    StatusCode t4a_tt_f64_fulltensor(const t4a_tt_f64* ptr, double* out_data, size_t buf_len, size_t* out_len);

    // Arithmetic - F64
    t4a_tt_f64* t4a_tt_f64_add(const t4a_tt_f64* a, const t4a_tt_f64* b);
    t4a_tt_f64* t4a_tt_f64_sub(const t4a_tt_f64* a, const t4a_tt_f64* b);
    t4a_tt_f64* t4a_tt_f64_negate(const t4a_tt_f64* ptr);
    t4a_tt_f64* t4a_tt_f64_reverse(const t4a_tt_f64* ptr);
    t4a_tt_f64* t4a_tt_f64_hadamard(const t4a_tt_f64* a, const t4a_tt_f64* b);
    t4a_tt_f64* t4a_tt_f64_hadamard_zipup(const t4a_tt_f64* a, const t4a_tt_f64* b, double tolerance, size_t max_bond_dim);
    StatusCode t4a_tt_f64_dot(const t4a_tt_f64* a, const t4a_tt_f64* b, double* out_dot);

    // Compression - F64
    StatusCode t4a_tt_f64_compress(t4a_tt_f64* ptr, double tolerance, size_t max_bond_dim);
    t4a_tt_f64* t4a_tt_f64_compressed(const t4a_tt_f64* ptr, double tolerance, size_t max_bond_dim);

    // ========================================================================
    // TensorTrain C64 functions
    // ========================================================================

    // Lifecycle - C64
    t4a_tt_c64* t4a_tt_c64_new_zeros(const size_t* site_dims, size_t num_sites);
    t4a_tt_c64* t4a_tt_c64_new_constant(const size_t* site_dims, size_t num_sites, double re, double im);
    void t4a_tt_c64_release(t4a_tt_c64* ptr);
    t4a_tt_c64* t4a_tt_c64_clone(const t4a_tt_c64* ptr);

    // Properties - C64
    StatusCode t4a_tt_c64_len(const t4a_tt_c64* ptr, size_t* out_len);
    StatusCode t4a_tt_c64_site_dims(const t4a_tt_c64* ptr, size_t* out_dims, size_t buf_len);
    StatusCode t4a_tt_c64_link_dims(const t4a_tt_c64* ptr, size_t* out_dims, size_t buf_len);
    StatusCode t4a_tt_c64_rank(const t4a_tt_c64* ptr, size_t* out_rank);

    // Evaluation - C64
    StatusCode t4a_tt_c64_evaluate(const t4a_tt_c64* ptr, const size_t* indices, size_t num_indices, double* out_re, double* out_im);
    StatusCode t4a_tt_c64_sum(const t4a_tt_c64* ptr, double* out_re, double* out_im);
    StatusCode t4a_tt_c64_norm(const t4a_tt_c64* ptr, double* out_norm);
    StatusCode t4a_tt_c64_log_norm(const t4a_tt_c64* ptr, double* out_log_norm);

    // Scaling - C64
    StatusCode t4a_tt_c64_scale_inplace(t4a_tt_c64* ptr, double factor_re, double factor_im);
    t4a_tt_c64* t4a_tt_c64_scaled(const t4a_tt_c64* ptr, double factor_re, double factor_im);

    // Full tensor - C64
    StatusCode t4a_tt_c64_fulltensor(const t4a_tt_c64* ptr, double* out_re, double* out_im, size_t buf_len, size_t* out_len);

    // Arithmetic - C64
    t4a_tt_c64* t4a_tt_c64_add(const t4a_tt_c64* a, const t4a_tt_c64* b);
    t4a_tt_c64* t4a_tt_c64_sub(const t4a_tt_c64* a, const t4a_tt_c64* b);
    t4a_tt_c64* t4a_tt_c64_negate(const t4a_tt_c64* ptr);
    t4a_tt_c64* t4a_tt_c64_reverse(const t4a_tt_c64* ptr);
    t4a_tt_c64* t4a_tt_c64_hadamard(const t4a_tt_c64* a, const t4a_tt_c64* b);
    t4a_tt_c64* t4a_tt_c64_hadamard_zipup(const t4a_tt_c64* a, const t4a_tt_c64* b, double tolerance, size_t max_bond_dim);
    StatusCode t4a_tt_c64_dot(const t4a_tt_c64* a, const t4a_tt_c64* b, double* out_re, double* out_im);

    // Compression - C64
    StatusCode t4a_tt_c64_compress(t4a_tt_c64* ptr, double tolerance, size_t max_bond_dim);
    t4a_tt_c64* t4a_tt_c64_compressed(const t4a_tt_c64* ptr, double tolerance, size_t max_bond_dim);

    // ========================================================================
    // Algorithm functions (tensor4all-core-common)
    // ========================================================================

    // Default tolerance
    double t4a_get_default_svd_rtol(void);

    // Algorithm enum types
    typedef enum {
        T4A_FACTORIZE_SVD = 0,
        T4A_FACTORIZE_LU = 1,
        T4A_FACTORIZE_CI = 2,
    } t4a_factorize_algorithm;

    typedef enum {
        T4A_CONTRACTION_NAIVE = 0,
        T4A_CONTRACTION_ZIPUP = 1,
        T4A_CONTRACTION_FIT = 2,
    } t4a_contraction_algorithm;

    typedef enum {
        T4A_COMPRESSION_SVD = 0,
        T4A_COMPRESSION_LU = 1,
        T4A_COMPRESSION_CI = 2,
        T4A_COMPRESSION_VARIATIONAL = 3,
    } t4a_compression_algorithm;

    // Algorithm name functions
    const char* t4a_factorize_algorithm_name(t4a_factorize_algorithm alg);
    const char* t4a_contraction_algorithm_name(t4a_contraction_algorithm alg);
    const char* t4a_compression_algorithm_name(t4a_compression_algorithm alg);

    // Algorithm from name functions
    StatusCode t4a_factorize_algorithm_from_name(const char* name, t4a_factorize_algorithm* out_alg);
    StatusCode t4a_contraction_algorithm_from_name(const char* name, t4a_contraction_algorithm* out_alg);
    StatusCode t4a_compression_algorithm_from_name(const char* name, t4a_compression_algorithm* out_alg);

    // Algorithm from i32 functions
    StatusCode t4a_factorize_algorithm_from_i32(int32_t value, t4a_factorize_algorithm* out_alg);
    StatusCode t4a_contraction_algorithm_from_i32(int32_t value, t4a_contraction_algorithm* out_alg);
    StatusCode t4a_compression_algorithm_from_i32(int32_t value, t4a_compression_algorithm* out_alg);

    // ========================================================================
    // MPO F64 functions
    // ========================================================================

    // Opaque type
    typedef struct { void* _private; } t4a_mpo_f64;
    typedef struct { void* _private; } t4a_mpo_c64;

    // Lifecycle - F64
    t4a_mpo_f64* t4a_mpo_f64_new_zeros(const size_t* dims1, const size_t* dims2, size_t num_sites);
    t4a_mpo_f64* t4a_mpo_f64_new_constant(const size_t* dims1, const size_t* dims2, size_t num_sites, double value);
    t4a_mpo_f64* t4a_mpo_f64_new_identity(const size_t* site_dims, size_t num_sites);
    void t4a_mpo_f64_release(t4a_mpo_f64* ptr);
    t4a_mpo_f64* t4a_mpo_f64_clone(const t4a_mpo_f64* ptr);
    int t4a_mpo_f64_is_assigned(const t4a_mpo_f64* ptr);

    // Properties - F64
    StatusCode t4a_mpo_f64_len(const t4a_mpo_f64* ptr, size_t* out_len);
    StatusCode t4a_mpo_f64_site_dims(const t4a_mpo_f64* ptr, size_t* out_dims1, size_t* out_dims2, size_t buf_len);
    StatusCode t4a_mpo_f64_link_dims(const t4a_mpo_f64* ptr, size_t* out_dims, size_t buf_len);
    StatusCode t4a_mpo_f64_rank(const t4a_mpo_f64* ptr, size_t* out_rank);

    // Evaluation - F64
    StatusCode t4a_mpo_f64_evaluate(const t4a_mpo_f64* ptr, const size_t* indices, size_t num_indices, double* out_value);
    StatusCode t4a_mpo_f64_sum(const t4a_mpo_f64* ptr, double* out_sum);

    // Scaling - F64
    StatusCode t4a_mpo_f64_scale_inplace(t4a_mpo_f64* ptr, double factor);
    t4a_mpo_f64* t4a_mpo_f64_scaled(const t4a_mpo_f64* ptr, double factor);

    // Contraction - F64
    t4a_mpo_f64* t4a_mpo_f64_contract_naive(const t4a_mpo_f64* a, const t4a_mpo_f64* b);
    t4a_mpo_f64* t4a_mpo_f64_contract_zipup(const t4a_mpo_f64* a, const t4a_mpo_f64* b, double tolerance, size_t max_bond_dim);
    t4a_mpo_f64* t4a_mpo_f64_contract_fit(const t4a_mpo_f64* a, const t4a_mpo_f64* b, double tolerance, size_t max_bond_dim, size_t max_sweeps);
    t4a_mpo_f64* t4a_mpo_f64_contract(const t4a_mpo_f64* a, const t4a_mpo_f64* b, t4a_contraction_algorithm algorithm, double tolerance, size_t max_bond_dim);

    // ========================================================================
    // MPO C64 functions
    // ========================================================================

    // Lifecycle - C64
    t4a_mpo_c64* t4a_mpo_c64_new_zeros(const size_t* dims1, const size_t* dims2, size_t num_sites);
    t4a_mpo_c64* t4a_mpo_c64_new_constant(const size_t* dims1, const size_t* dims2, size_t num_sites, double value_re, double value_im);
    t4a_mpo_c64* t4a_mpo_c64_new_identity(const size_t* site_dims, size_t num_sites);
    void t4a_mpo_c64_release(t4a_mpo_c64* ptr);
    t4a_mpo_c64* t4a_mpo_c64_clone(const t4a_mpo_c64* ptr);
    int t4a_mpo_c64_is_assigned(const t4a_mpo_c64* ptr);

    // Properties - C64
    StatusCode t4a_mpo_c64_len(const t4a_mpo_c64* ptr, size_t* out_len);

    // Evaluation - C64
    StatusCode t4a_mpo_c64_evaluate(const t4a_mpo_c64* ptr, const size_t* indices, size_t num_indices, double* out_re, double* out_im);
    StatusCode t4a_mpo_c64_sum(const t4a_mpo_c64* ptr, double* out_re, double* out_im);

    // Contraction - C64
    t4a_mpo_c64* t4a_mpo_c64_contract_naive(const t4a_mpo_c64* a, const t4a_mpo_c64* b);
    t4a_mpo_c64* t4a_mpo_c64_contract_zipup(const t4a_mpo_c64* a, const t4a_mpo_c64* b, double tolerance, size_t max_bond_dim);
    t4a_mpo_c64* t4a_mpo_c64_contract_fit(const t4a_mpo_c64* a, const t4a_mpo_c64* b, double tolerance, size_t max_bond_dim, size_t max_sweeps);
    t4a_mpo_c64* t4a_mpo_c64_contract(const t4a_mpo_c64* a, const t4a_mpo_c64* b, t4a_contraction_algorithm algorithm, double tolerance, size_t max_bond_dim);

    // ========================================================================
    // SimpleTT F64 functions (tensor4all-simplett)
    // ========================================================================

    // Opaque type
    typedef struct { void* _private; } t4a_simplett_f64;

    // Lifecycle
    void t4a_simplett_f64_release(t4a_simplett_f64* ptr);
    t4a_simplett_f64* t4a_simplett_f64_clone(const t4a_simplett_f64* ptr);

    // Constructors
    t4a_simplett_f64* t4a_simplett_f64_constant(const size_t* site_dims, size_t n_sites, double value);
    t4a_simplett_f64* t4a_simplett_f64_zeros(const size_t* site_dims, size_t n_sites);

    // Accessors
    StatusCode t4a_simplett_f64_len(const t4a_simplett_f64* ptr, size_t* out_len);
    StatusCode t4a_simplett_f64_site_dims(const t4a_simplett_f64* ptr, size_t* out_dims, size_t buf_len);
    StatusCode t4a_simplett_f64_link_dims(const t4a_simplett_f64* ptr, size_t* out_dims, size_t buf_len);
    StatusCode t4a_simplett_f64_rank(const t4a_simplett_f64* ptr, size_t* out_rank);
    StatusCode t4a_simplett_f64_evaluate(const t4a_simplett_f64* ptr, const size_t* indices, size_t n_indices, double* out_value);
    StatusCode t4a_simplett_f64_sum(const t4a_simplett_f64* ptr, double* out_value);
    StatusCode t4a_simplett_f64_norm(const t4a_simplett_f64* ptr, double* out_value);
    StatusCode t4a_simplett_f64_site_tensor(const t4a_simplett_f64* ptr, size_t site,
                                             double* out_data, size_t buf_len,
                                             size_t* out_left_dim, size_t* out_site_dim, size_t* out_right_dim);

    // ========================================================================
    // TensorCI2 F64 functions (tensor4all-tensorci)
    // ========================================================================

    // Opaque type
    typedef struct { void* _private; } t4a_tci2_f64;

    // Callback type for evaluation function
    typedef int (*t4a_eval_callback)(const int64_t* indices, size_t n_indices, double* result, void* user_data);

    // Lifecycle
    void t4a_tci2_f64_release(t4a_tci2_f64* ptr);
    t4a_tci2_f64* t4a_tci2_f64_new(const size_t* local_dims, size_t n_sites);

    // Accessors
    StatusCode t4a_tci2_f64_len(const t4a_tci2_f64* ptr, size_t* out_len);
    StatusCode t4a_tci2_f64_rank(const t4a_tci2_f64* ptr, size_t* out_rank);
    StatusCode t4a_tci2_f64_link_dims(const t4a_tci2_f64* ptr, size_t* out_dims, size_t buf_len);
    StatusCode t4a_tci2_f64_max_sample_value(const t4a_tci2_f64* ptr, double* out_value);
    StatusCode t4a_tci2_f64_max_bond_error(const t4a_tci2_f64* ptr, double* out_value);

    // Pivot operations
    StatusCode t4a_tci2_f64_add_global_pivots(t4a_tci2_f64* ptr, const size_t* pivots, size_t n_pivots, size_t n_sites);

    // Conversion
    t4a_simplett_f64* t4a_tci2_f64_to_tensor_train(const t4a_tci2_f64* ptr);

    // High-level crossinterpolate2
    StatusCode t4a_crossinterpolate2_f64(
        const size_t* local_dims,
        size_t n_sites,
        const size_t* initial_pivots,
        size_t n_initial_pivots,
        t4a_eval_callback eval_fn,
        void* user_data,
        double tolerance,
        size_t max_bonddim,
        size_t max_iter,
        t4a_tci2_f64** out_tci,
        double* out_final_error
    );
""")
