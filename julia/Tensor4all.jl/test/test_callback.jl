# Test for C callback function pointer with trampoline pattern
#
# This tests passing Julia closures to Rust via C function pointers using
# the pointer_from_objref pattern.

using Test

const CALLBACK_LIB = joinpath(@__DIR__, "..", "..", "..", "target", "release", "libtensor4all_callback_test.dylib")

# Check if the library exists
if !isfile(CALLBACK_LIB)
    error("Callback test library not found at: $CALLBACK_LIB\n" *
          "Run: cargo build -p tensor4all-callback-test --release")
end

# ============================================================================
# Trampoline implementation
# ============================================================================

"""
    _trampoline(indices_ptr, n_indices, result_ptr, user_data) -> Cint

Top-level trampoline function that can be converted to a C function pointer.
Dispatches to the actual Julia function stored in user_data.
"""
function _trampoline(indices_ptr::Ptr{Int64}, n_indices::Csize_t,
                     result_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})::Cint
    try
        # Recover the Ref{Any} from the pointer
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]

        # Read indices from pointer
        indices = unsafe_wrap(Array, indices_ptr, Int(n_indices))

        # Call the user function
        val = Float64(f(indices...))

        # Store result
        unsafe_store!(result_ptr, val)
        return Cint(0)
    catch e
        @error "Error in trampoline" exception=(e, catch_backtrace())
        return Cint(-1)
    end
end

# Create the C function pointer (once, at module load time)
const TRAMPOLINE_PTR = @cfunction(
    _trampoline,
    Cint,
    (Ptr{Int64}, Csize_t, Ptr{Float64}, Ptr{Cvoid})
)

"""
    with_callback(f::Function, rust_call::Function)

Execute `rust_call` with the Julia function `f` wrapped for C interop.
The `rust_call` receives `(trampoline_ptr, user_data_ptr)`.
"""
function with_callback(f::Function, rust_call::Function)
    # Store function in a Ref to get a stable pointer
    f_ref = Ref{Any}(f)

    # GC.@preserve ensures f_ref is not garbage collected during the ccall
    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        return rust_call(TRAMPOLINE_PTR, user_data)
    end
end

# ============================================================================
# Helper function for calling Rust with callback
# ============================================================================

"""
    call_simple(f::Function, result::Ref{Float64}) -> Cint

Call the simple test function with a Julia callback.
"""
function call_simple(f::Function, result::Ref{Float64})
    f_ref = Ref{Any}(f)
    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        return ccall(
            (:t4a_callback_test_simple, CALLBACK_LIB),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
            TRAMPOLINE_PTR, user_data, result
        )
    end
end

"""
    call_multiple(f::Function, n_calls::Integer, result::Ref{Float64}) -> Cint

Call the multiple test function with a Julia callback.
"""
function call_multiple(f::Function, n_calls::Integer, result::Ref{Float64})
    f_ref = Ref{Any}(f)
    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        return ccall(
            (:t4a_callback_test_multiple, CALLBACK_LIB),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{Float64}),
            TRAMPOLINE_PTR, user_data, Csize_t(n_calls), result
        )
    end
end

"""
    call_with_indices(f::Function, indices::Vector{Int64}, result::Ref{Float64}) -> Cint

Call the custom indices test function with a Julia callback.
"""
function call_with_indices(f::Function, indices::Vector{Int64}, result::Ref{Float64})
    f_ref = Ref{Any}(f)
    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        return ccall(
            (:t4a_callback_test_with_indices, CALLBACK_LIB),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Float64}),
            TRAMPOLINE_PTR, user_data, indices, Csize_t(length(indices)), result
        )
    end
end

# ============================================================================
# Tests
# ============================================================================

@testset "Callback Tests" begin

    @testset "Simple callback - sum indices" begin
        # Function that sums its arguments: f(1, 2, 3) = 6
        sum_fn = (args...) -> sum(args)

        result = Ref{Float64}(0.0)
        status = call_simple(sum_fn, result)

        @test status == 0
        @test result[] == 6.0  # 1 + 2 + 3
    end

    @testset "Callback with captured state" begin
        # Closure that captures a multiplier
        multiplier = 10.0
        scaled_sum = (args...) -> sum(args) * multiplier

        result = Ref{Float64}(0.0)
        status = call_simple(scaled_sum, result)

        @test status == 0
        @test result[] == 60.0  # (1 + 2 + 3) * 10
    end

    @testset "Multiple callback invocations" begin
        # Count how many times the function is called
        call_count = Ref(0)

        counting_fn = (args...) -> begin
            call_count[] += 1
            return sum(args)
        end

        result = Ref{Float64}(0.0)
        n_calls = 5

        status = call_multiple(counting_fn, n_calls, result)

        @test status == 0
        @test call_count[] == n_calls
        # Sum: (0+1+2) + (1+2+3) + (2+3+4) + (3+4+5) + (4+5+6) = 3+6+9+12+15 = 45
        @test result[] == 45.0
    end

    @testset "Callback with custom indices" begin
        product_fn = (args...) -> prod(args)

        indices = Int64[2, 3, 4]
        result = Ref{Float64}(0.0)

        status = call_with_indices(product_fn, indices, result)

        @test status == 0
        @test result[] == 24.0  # 2 * 3 * 4
    end

    @testset "Complex closure with mutable state" begin
        # Accumulate all indices seen
        all_indices = Vector{Int64}[]

        recording_fn = (args...) -> begin
            push!(all_indices, collect(args))
            return Float64(length(all_indices))
        end

        result = Ref{Float64}(0.0)
        n_calls = 3

        status = call_multiple(recording_fn, n_calls, result)

        @test status == 0
        @test length(all_indices) == n_calls
        @test all_indices[1] == [0, 1, 2]
        @test all_indices[2] == [1, 2, 3]
        @test all_indices[3] == [2, 3, 4]
    end

end

println("All callback tests passed!")
