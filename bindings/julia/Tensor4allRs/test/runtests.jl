using Test
using Tensor4allRs

# Initialize library - adjust path as needed
# For development, build with: cargo build --release -p tensor4all-capi
const LIBPATH = if Sys.isapple()
    joinpath(@__DIR__, "..", "..", "..", "..", "target", "release", "libtensor4all_capi.dylib")
elseif Sys.islinux()
    joinpath(@__DIR__, "..", "..", "..", "..", "target", "release", "libtensor4all_capi.so")
elseif Sys.iswindows()
    joinpath(@__DIR__, "..", "..", "..", "..", "target", "release", "tensor4all_capi.dll")
else
    error("Unsupported platform")
end

@testset "Tensor4allRs" begin
    # Check if library exists
    if !isfile(LIBPATH)
        @warn "Library not found at $LIBPATH. Skipping tests. Build with: cargo build --release -p tensor4all-capi"
        return
    end

    Tensor4allRs.init_library(LIBPATH)

    @testset "TensorTrainF64" begin
        @testset "zeros" begin
            tt = zeros_tt(Float64, [2, 3, 2])
            @test length(tt) == 3
            @test site_dims(tt) == [2, 3, 2]
            @test rank(tt) == 1
        end

        @testset "constant" begin
            tt = constant_tt(Float64, [2, 2], 5.0)
            @test length(tt) == 2
            @test site_dims(tt) == [2, 2]

            # Sum should be 5.0 * 2 * 2 = 20.0
            @test isapprox(sum_tt(tt), 20.0; atol=1e-10)

            # Evaluate at [1, 2] (1-based)
            @test isapprox(evaluate(tt, [1, 2]), 5.0; atol=1e-10)
        end

        @testset "scale" begin
            tt = constant_tt(Float64, [2, 2], 1.0)

            # Immutable scale
            tt2 = scaled(tt, 3.0)
            @test isapprox(sum_tt(tt2), 12.0; atol=1e-10)
            @test isapprox(sum_tt(tt), 4.0; atol=1e-10)  # Original unchanged

            # In-place scale
            scale!(tt, 2.0)
            @test isapprox(sum_tt(tt), 8.0; atol=1e-10)
        end

        @testset "norm" begin
            tt = constant_tt(Float64, [2, 3], 2.0)

            norm_val = norm_tt(tt)
            # norm = sqrt(sum of squares) = sqrt(6 * 4) = sqrt(24)
            @test isapprox(norm_val, sqrt(24.0); atol=1e-10)

            log_norm_val = log_norm_tt(tt)
            @test isapprox(log_norm_val, log(norm_val); atol=1e-10)
        end

        @testset "fulltensor" begin
            tt = constant_tt(Float64, [2, 3], 5.0)
            arr = fulltensor(tt)

            @test size(arr) == (2, 3)
            @test all(isapprox.(arr, 5.0; atol=1e-10))
        end

        @testset "copy" begin
            tt = constant_tt(Float64, [2, 2], 3.0)
            tt2 = copy(tt)

            @test isapprox(sum_tt(tt2), sum_tt(tt); atol=1e-10)

            # Modify original, copy should be unchanged
            scale!(tt, 2.0)
            @test isapprox(sum_tt(tt), 24.0; atol=1e-10)
            @test isapprox(sum_tt(tt2), 12.0; atol=1e-10)
        end
    end

    @testset "TensorTrainC64" begin
        @testset "zeros" begin
            tt = zeros_tt(ComplexF64, [2, 3, 2])
            @test length(tt) == 3
            @test site_dims(tt) == [2, 3, 2]
        end

        @testset "constant" begin
            tt = constant_tt(ComplexF64, [2, 2], 3.0 + 4.0im)
            @test length(tt) == 2

            val = evaluate(tt, [1, 2])
            @test isapprox(real(val), 3.0; atol=1e-10)
            @test isapprox(imag(val), 4.0; atol=1e-10)
        end

        @testset "scale" begin
            tt = constant_tt(ComplexF64, [2, 2], 1.0 + 0.0im)

            # Scale by complex number
            tt2 = scaled(tt, 0.0 + 1.0im)
            s = sum_tt(tt2)
            @test isapprox(real(s), 0.0; atol=1e-10)
            @test isapprox(imag(s), 4.0; atol=1e-10)
        end

        @testset "fulltensor" begin
            tt = constant_tt(ComplexF64, [2, 3], 1.0 + 2.0im)
            arr = fulltensor(tt)

            @test size(arr) == (2, 3)
            @test all(x -> isapprox(x, 1.0 + 2.0im; atol=1e-10), arr)
        end
    end
end
