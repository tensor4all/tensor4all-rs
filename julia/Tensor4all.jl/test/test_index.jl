@testset "Index" begin
    @testset "creation" begin
        # Basic creation
        i = T4AIndex(5)
        @test Tensor4all.dim(i) == 5
        @test Tensor4all.tags(i) == ""
        @test Tensor4all.id(i) != 0

        # With tags
        j = T4AIndex(3; tags="Site,n=1")
        @test Tensor4all.dim(j) == 3
        @test Tensor4all.hastag(j, "Site")
        @test Tensor4all.hastag(j, "n=1")
        @test !Tensor4all.hastag(j, "Missing")

        # Tags string contains both
        t = Tensor4all.tags(j)
        @test occursin("Site", t)
        @test occursin("n=1", t)
    end

    @testset "custom ID" begin
        id_val = UInt64(0x12345678_9ABCDEF0)
        i = T4AIndex(4, id_val; tags="Custom")
        @test Tensor4all.dim(i) == 4
        @test Tensor4all.id(i) == id_val
        @test Tensor4all.hastag(i, "Custom")
    end

    @testset "copy" begin
        i = T4AIndex(5; tags="Original")
        j = copy(i)

        @test Tensor4all.dim(i) == Tensor4all.dim(j)
        @test Tensor4all.id(i) == Tensor4all.id(j)
        @test Tensor4all.tags(i) == Tensor4all.tags(j)
        @test i == j  # Equal by ID
    end

    @testset "equality and hashing" begin
        i = T4AIndex(5)
        j = copy(i)
        k = T4AIndex(5)  # Different ID

        @test i == j
        @test i != k
        @test hash(i) == hash(j)
        @test hash(i) != hash(k)  # Very likely different
    end

    @testset "display" begin
        i = T4AIndex(3; tags="Site")
        s = sprint(show, i)
        @test occursin("dim=3", s)
        @test occursin("Site", s)
    end

    @testset "error handling" begin
        @test_throws ArgumentError T4AIndex(0)
        @test_throws ArgumentError T4AIndex(-1)
    end
end
