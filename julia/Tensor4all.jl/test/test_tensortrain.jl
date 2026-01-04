import Tensor4all.TensorTrain as TT

@testset "TensorTrain" begin
    @testset "zeros (f64)" begin
        tt = TT.zeros(Float64, [2, 3, 2])
        @test length(tt) == 3
        @test TT.site_dims(tt) == [2, 3, 2]
        @test sum(tt) ≈ 0.0
    end

    @testset "constant (f64)" begin
        tt = TT.constant(Float64, [2, 3, 2], 1.0)
        @test length(tt) == 3
        @test TT.site_dims(tt) == [2, 3, 2]
        @test sum(tt) ≈ 2 * 3 * 2  # All elements are 1.0
        @test TT.rank(tt) == 1  # Constant has bond dimension 1
    end

    @testset "zeros (c64)" begin
        tt = TT.zeros(ComplexF64, [2, 3, 2])
        @test length(tt) == 3
        @test TT.site_dims(tt) == [2, 3, 2]
        @test sum(tt) ≈ 0.0 + 0.0im
    end

    @testset "constant (c64)" begin
        tt = TT.constant(ComplexF64, [2, 3, 2], 1.0 + 2.0im)
        @test length(tt) == 3
        @test TT.site_dims(tt) == [2, 3, 2]
        @test sum(tt) ≈ (1.0 + 2.0im) * 12
    end

    @testset "evaluate (f64)" begin
        tt = TT.constant(Float64, [2, 3, 2], 5.0)
        # All elements should be 5.0
        @test TT.evaluate(tt, [1, 1, 1]) ≈ 5.0
        @test TT.evaluate(tt, [2, 3, 2]) ≈ 5.0
        @test TT.evaluate(tt, [1, 2, 1]) ≈ 5.0
    end

    @testset "evaluate (c64)" begin
        val = 3.0 + 4.0im
        tt = TT.constant(ComplexF64, [2, 2], val)
        @test TT.evaluate(tt, [1, 1]) ≈ val
        @test TT.evaluate(tt, [2, 2]) ≈ val
    end

    @testset "norm" begin
        tt = TT.constant(Float64, [2, 2, 2], 1.0)
        # ||tt||_F = sqrt(sum of |elements|^2) = sqrt(8 * 1^2) = sqrt(8)
        @test TT.norm(tt) ≈ sqrt(8.0)
    end

    @testset "log_norm" begin
        tt = TT.constant(Float64, [2, 2, 2], 1.0)
        @test TT.log_norm(tt) ≈ log(sqrt(8.0))
    end

    @testset "copy" begin
        tt = TT.constant(Float64, [2, 3], 3.0)
        tt2 = copy(tt)
        @test TT.site_dims(tt2) == [2, 3]
        @test sum(tt2) ≈ sum(tt)
    end

    @testset "scale!" begin
        tt = TT.constant(Float64, [2, 2], 1.0)
        TT.scale!(tt, 2.0)
        @test sum(tt) ≈ 8.0  # 4 elements * 2.0
    end

    @testset "scaled" begin
        tt = TT.constant(Float64, [2, 2], 1.0)
        tt2 = TT.scaled(tt, 3.0)
        @test sum(tt) ≈ 4.0   # Original unchanged
        @test sum(tt2) ≈ 12.0 # Scaled copy
    end

    @testset "fulltensor" begin
        tt = TT.constant(Float64, [2, 3], 2.0)
        arr = TT.fulltensor(tt)
        @test size(arr) == (2, 3)
        @test all(arr .≈ 2.0)
    end

    @testset "fulltensor (c64)" begin
        val = 1.0 + 1.0im
        tt = TT.constant(ComplexF64, [2, 2], val)
        arr = TT.fulltensor(tt)
        @test size(arr) == (2, 2)
        @test all(arr .≈ val)
    end

    @testset "add" begin
        tt1 = TT.constant(Float64, [2, 2], 1.0)
        tt2 = TT.constant(Float64, [2, 2], 2.0)
        tt3 = TT.add(tt1, tt2)
        @test sum(tt3) ≈ 12.0  # 4 * (1 + 2)
    end

    @testset "operator +" begin
        tt1 = TT.constant(Float64, [2, 2], 1.0)
        tt2 = TT.constant(Float64, [2, 2], 2.0)
        tt3 = tt1 + tt2
        @test sum(tt3) ≈ 12.0
    end

    @testset "sub" begin
        tt1 = TT.constant(Float64, [2, 2], 5.0)
        tt2 = TT.constant(Float64, [2, 2], 2.0)
        tt3 = TT.sub(tt1, tt2)
        @test sum(tt3) ≈ 12.0  # 4 * (5 - 2)
    end

    @testset "operator -" begin
        tt1 = TT.constant(Float64, [2, 2], 5.0)
        tt2 = TT.constant(Float64, [2, 2], 2.0)
        tt3 = tt1 - tt2
        @test sum(tt3) ≈ 12.0
    end

    @testset "negate" begin
        tt = TT.constant(Float64, [2, 2], 3.0)
        neg_tt = TT.negate(tt)
        @test sum(neg_tt) ≈ -12.0
    end

    @testset "operator unary -" begin
        tt = TT.constant(Float64, [2, 2], 3.0)
        neg_tt = -tt
        @test sum(neg_tt) ≈ -12.0
    end

    @testset "hadamard" begin
        tt1 = TT.constant(Float64, [2, 2], 2.0)
        tt2 = TT.constant(Float64, [2, 2], 3.0)
        tt3 = TT.hadamard(tt1, tt2)
        @test sum(tt3) ≈ 24.0  # 4 * (2 * 3)
    end

    @testset "operator *" begin
        tt1 = TT.constant(Float64, [2, 2], 2.0)
        tt2 = TT.constant(Float64, [2, 2], 3.0)
        tt3 = tt1 * tt2
        @test sum(tt3) ≈ 24.0
    end

    @testset "scalar multiplication" begin
        tt = TT.constant(Float64, [2, 2], 1.0)
        tt2 = tt * 5.0
        tt3 = 5.0 * tt
        @test sum(tt2) ≈ 20.0
        @test sum(tt3) ≈ 20.0
    end

    @testset "dot" begin
        tt1 = TT.constant(Float64, [2, 2], 1.0)
        tt2 = TT.constant(Float64, [2, 2], 2.0)
        # dot(tt1, tt2) = sum(tt1 .* tt2) = 4 * (1 * 2) = 8
        @test TT.dot(tt1, tt2) ≈ 8.0
    end

    @testset "compress!" begin
        # Addition increases bond dimension
        tt1 = TT.constant(Float64, [2, 2], 1.0)
        tt2 = TT.constant(Float64, [2, 2], 2.0)
        tt3 = tt1 + tt2
        @test TT.rank(tt3) == 2  # Bond dim increases after addition

        TT.compress!(tt3; tolerance=1e-12)
        @test TT.rank(tt3) == 1  # Compressed back to 1
        @test sum(tt3) ≈ 12.0   # Values preserved
    end

    @testset "compressed" begin
        tt1 = TT.constant(Float64, [2, 2], 1.0)
        tt2 = TT.constant(Float64, [2, 2], 2.0)
        tt3 = tt1 + tt2

        tt4 = TT.compressed(tt3; tolerance=1e-12)
        @test TT.rank(tt4) == 1
        @test TT.rank(tt3) == 2  # Original unchanged
    end

    @testset "reverse" begin
        tt = TT.constant(Float64, [2, 3, 4], 1.0)
        tt_rev = reverse(tt)
        @test TT.site_dims(tt_rev) == [4, 3, 2]
        @test sum(tt_rev) ≈ sum(tt)
    end

    @testset "complex arithmetic" begin
        tt1 = TT.constant(ComplexF64, [2, 2], 1.0 + 1.0im)
        tt2 = TT.constant(ComplexF64, [2, 2], 2.0 + 0.0im)
        tt3 = tt1 + tt2
        @test sum(tt3) ≈ 4 * (3.0 + 1.0im)
    end

    @testset "complex dot" begin
        tt1 = TT.constant(ComplexF64, [2, 2], 1.0 + 1.0im)
        tt2 = TT.constant(ComplexF64, [2, 2], 1.0 - 1.0im)
        # dot computes sum(conj(tt1) .* tt2) or sum(tt1 .* tt2) depending on implementation
        # For element-wise: (1+i)(1-i) = 1 - i^2 = 2, so 4 * 2 = 8
        d = TT.dot(tt1, tt2)
        @test real(d) ≈ 8.0 atol = 1e-10
    end
end
