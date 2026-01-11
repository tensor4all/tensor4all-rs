using Test
using Tensor4all
using Tensor4all.SimpleTT

# Import functions from SimpleTT module
import Tensor4all.SimpleTT: site_dims, link_dims, rank, evaluate, site_tensor

@testset "SimpleTT" begin
    @testset "constant tensor train" begin
        # Create a constant tensor train
        tt = SimpleTensorTrain([2, 3, 4], 1.5)

        @test length(tt) == 3
        @test site_dims(tt) == [2, 3, 4]
        @test rank(tt) == 1  # Constant has rank 1

        # Sum should be value * product of dimensions
        expected_sum = 1.5 * 2 * 3 * 4
        @test sum(tt) ≈ expected_sum
    end

    @testset "zeros tensor train" begin
        tt = zeros(SimpleTensorTrain, [2, 3])

        @test length(tt) == 2
        @test site_dims(tt) == [2, 3]
        @test sum(tt) == 0.0
    end

    @testset "evaluate" begin
        tt = SimpleTensorTrain([2, 3, 4], 2.0)

        # All elements should be 2.0
        @test evaluate(tt, [0, 0, 0]) ≈ 2.0
        @test evaluate(tt, [1, 2, 3]) ≈ 2.0

        # Test callable interface
        @test tt([0, 1, 2]) ≈ 2.0
        @test tt(0, 1, 2) ≈ 2.0
    end

    @testset "copy" begin
        tt1 = SimpleTensorTrain([2, 3], 3.0)
        tt2 = copy(tt1)

        @test length(tt2) == length(tt1)
        @test site_dims(tt2) == site_dims(tt1)
        @test sum(tt2) ≈ sum(tt1)
    end

    @testset "link_dims" begin
        tt = SimpleTensorTrain([2, 3, 4], 1.0)

        ldims = link_dims(tt)
        @test length(ldims) == 2  # n_sites - 1

        # For rank-1 constant, link dims should all be 1
        @test all(d -> d == 1, ldims)
    end

    @testset "site_tensor" begin
        tt = SimpleTensorTrain([2, 3], 1.0)

        # Get site tensor at site 0
        t0 = site_tensor(tt, 0)
        @test size(t0, 1) == 1  # left dim
        @test size(t0, 2) == 2  # site dim
        @test size(t0, 3) == 1  # right dim

        # Get site tensor at site 1
        t1 = site_tensor(tt, 1)
        @test size(t1, 1) == 1  # left dim
        @test size(t1, 2) == 3  # site dim
        @test size(t1, 3) == 1  # right dim
    end

    @testset "show" begin
        tt = SimpleTensorTrain([2, 3, 4], 1.0)

        # Test that show doesn't error
        io = IOBuffer()
        show(io, tt)
        s = String(take!(io))
        @test occursin("SimpleTensorTrain", s)
        @test occursin("3", s)  # sites

        show(io, MIME"text/plain"(), tt)
        s = String(take!(io))
        @test occursin("Sites:", s)
    end
end
