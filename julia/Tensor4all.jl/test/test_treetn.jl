using Test
using Tensor4all: Index as T4AIndex, Tensor as T4ATensor
using Tensor4all: dim, rank
using Tensor4all.TreeTN: MPS, MPO, TreeTensorNetwork
using Tensor4all.TreeTN: nv, ne, linkdims, maxbonddim, linkind, linkinds, linkdim
using Tensor4all.TreeTN: canonical_form, Unitary, LU, CI
using Tensor4all.TreeTN: orthogonalize!, truncate!, inner
using Tensor4all.TreeTN: contract, to_dense, truncate
using Tensor4all.TreeTN: random_mps, random_tt
using Tensor4all.TreeTN: findsite, findsites, siteinds, siteind
using LinearAlgebra

@testset "TreeTN" begin
    @testset "MPS construction" begin
        # Create indices
        s0 = T4AIndex(2; tags="Site,n=0")
        l01 = T4AIndex(3; tags="Link,l=0")
        s1 = T4AIndex(2; tags="Site,n=1")

        # Create tensors with shared link index
        data0 = rand(2, 3)
        data1 = rand(3, 2)
        t0 = T4ATensor([s0, l01], data0)
        t1 = T4ATensor([l01, s1], data1)

        # Create MPS
        mps = MPS([t0, t1])
        @test length(mps) == 2
        @test nv(mps) == 2
        @test ne(mps) == 1
    end

    @testset "MPS accessors" begin
        # Create a simple 3-site MPS
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)
        l12 = T4AIndex(3)
        s2 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 3))
        t1 = T4ATensor([l01, s1, l12], rand(3, 2, 3))
        t2 = T4ATensor([l12, s2], rand(3, 2))

        mps = MPS([t0, t1, t2])

        @test length(mps) == 3
        @test linkdims(mps) == [3, 3]
        @test maxbonddim(mps) == 3

        # Test tensor access (1-indexed)
        tensor1 = mps[1]
        @test rank(tensor1) == 2

        tensor2 = mps[2]
        @test rank(tensor2) == 3

        # Test iteration (collect replaces old tensors() function)
        ts = collect(mps)
        @test length(ts) == 3

        # Test link indices
        link1 = linkind(mps, 1)
        @test link1 !== nothing
        @test dim(link1) == 3

        links = linkinds(mps)
        @test length(links) == 2

        # Test linkdim
        @test linkdim(mps, 1) == 3
        @test linkdim(mps, 2) == 3
    end

    @testset "MPS orthogonalize" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(4)
        s1 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 4))
        t1 = T4ATensor([l01, s1], rand(4, 2))

        mps = MPS([t0, t1])

        # Initially no canonical form
        @test canonical_form(mps) === nothing

        # Orthogonalize to site 1
        orthogonalize!(mps, 1)
        @test canonical_form(mps) == Unitary

        # Orthogonalize to site 2
        orthogonalize!(mps, 2)
        @test canonical_form(mps) == Unitary
    end

    @testset "MPS canonical forms" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(4)
        s1 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 4))
        t1 = T4ATensor([l01, s1], rand(4, 2))

        # Test Unitary (QR) form
        mps1 = MPS([t0, t1])
        orthogonalize!(mps1, 1; form=Unitary)
        @test canonical_form(mps1) == Unitary

        # Test LU form
        mps2 = MPS([t0, t1])
        orthogonalize!(mps2, 1; form=LU)
        @test canonical_form(mps2) == LU

        # Test CI form
        mps3 = MPS([t0, t1])
        orthogonalize!(mps3, 1; form=CI)
        @test canonical_form(mps3) == CI
    end

    @testset "MPS norm" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)

        # Create random tensors
        data0 = rand(2, 3)
        data1 = rand(3, 2)
        t0 = T4ATensor([s0, l01], data0)
        t1 = T4ATensor([l01, s1], data1)

        mps = MPS([t0, t1])

        # Norm should be positive
        n = norm(mps)
        @test n > 0

        # Norm should be consistent with inner product
        inner_val = inner(mps, mps)
        @test isapprox(real(inner_val), n^2; rtol=1e-10)
        @test isapprox(imag(inner_val), 0.0; atol=1e-10)
    end

    @testset "MPS truncation" begin
        s0 = T4AIndex(4)
        l01 = T4AIndex(8)  # Large bond dimension
        s1 = T4AIndex(4)
        l12 = T4AIndex(8)
        s2 = T4AIndex(4)

        t0 = T4ATensor([s0, l01], rand(4, 8))
        t1 = T4ATensor([l01, s1, l12], rand(8, 4, 8))
        t2 = T4ATensor([l12, s2], rand(8, 4))

        mps = MPS([t0, t1, t2])
        @test maxbonddim(mps) == 8

        # Truncate to max dimension 4
        truncate!(mps; maxdim=4)
        @test maxbonddim(mps) <= 4
    end

    @testset "MPS copy" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 3))
        t1 = T4ATensor([l01, s1], rand(3, 2))

        mps = MPS([t0, t1])
        mps_copy = copy(mps)

        # Copy should have same properties
        @test length(mps_copy) == length(mps)
        @test linkdims(mps_copy) == linkdims(mps)

        # Modifying copy shouldn't affect original
        orthogonalize!(mps_copy, 1)
        @test canonical_form(mps_copy) == Unitary
        @test canonical_form(mps) === nothing
    end

    @testset "MPS display" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 3))
        t1 = T4ATensor([l01, s1], rand(3, 2))

        mps = MPS([t0, t1])

        # Test show method doesn't error
        io = IOBuffer()
        show(io, mps)
        str = String(take!(io))
        @test occursin("TreeTensorNetwork", str)
        @test occursin("nv=2", str)
    end

    @testset "MPS contract" begin
        # Create two MPS representing MPO-like objects
        # Shared site indices (to be contracted)
        s0 = T4AIndex(2)
        s1 = T4AIndex(2)
        s2 = T4AIndex(2)

        # External indices for MPS_A
        a0 = T4AIndex(2)
        a1 = T4AIndex(2)
        a2 = T4AIndex(2)

        # External indices for MPS_B
        b0 = T4AIndex(2)
        b1 = T4AIndex(2)
        b2 = T4AIndex(2)

        # First MPS (MPO-like)
        l01_a = T4AIndex(3)
        l12_a = T4AIndex(3)
        t0_a = T4ATensor([a0, s0, l01_a], rand(2, 2, 3))
        t1_a = T4ATensor([l01_a, a1, s1, l12_a], rand(3, 2, 2, 3))
        t2_a = T4ATensor([l12_a, a2, s2], rand(3, 2, 2))
        mps_a = MPS([t0_a, t1_a, t2_a])

        # Second MPS (MPO-like)
        l01_b = T4AIndex(3)
        l12_b = T4AIndex(3)
        t0_b = T4ATensor([b0, s0, l01_b], rand(2, 2, 3))
        t1_b = T4ATensor([l01_b, b1, s1, l12_b], rand(3, 2, 2, 3))
        t2_b = T4ATensor([l12_b, b2, s2], rand(3, 2, 2))
        mps_b = MPS([t0_b, t1_b, t2_b])

        # Test zipup contract (default)
        result = contract(mps_a, mps_b)
        @test length(result) == 3
        @test maxbonddim(result) > 0

        # Test contract with explicit :zipup symbol
        result_zipup = contract(mps_a, mps_b; method=:zipup)
        @test length(result_zipup) == 3

        # Test contract with max dimension
        result_truncated = contract(mps_a, mps_b; maxdim=4)
        @test length(result_truncated) == 3
        truncate!(result_truncated; maxdim=4)
        @test maxbonddim(result_truncated) <= 4

        # Test contract with :fit symbol
        result_fit = contract(mps_a, mps_b; method=:fit)
        @test length(result_fit) == 3

        # Test contract with string methods
        result_fit_str = contract(mps_a, mps_b; method="fit")
        @test length(result_fit_str) == 3

        result_zipup_str = contract(mps_a, mps_b; method="zipup")
        @test length(result_zipup_str) == 3

        result_naive = contract(mps_a, mps_b; method=:naive)
        @test length(result_naive) == 3

        result_naive_str = contract(mps_a, mps_b; method="naive")
        @test length(result_naive_str) == 3
    end

    @testset "MPS immutable truncate" begin
        s0 = T4AIndex(4)
        l01 = T4AIndex(8)
        s1 = T4AIndex(4)
        l12 = T4AIndex(8)
        s2 = T4AIndex(4)

        t0 = T4ATensor([s0, l01], rand(4, 8))
        t1 = T4ATensor([l01, s1, l12], rand(8, 4, 8))
        t2 = T4ATensor([l12, s2], rand(8, 4))

        mps = MPS([t0, t1, t2])
        @test maxbonddim(mps) == 8

        # Immutable truncate should return new MPS
        mps_truncated = truncate(mps; maxdim=4)
        @test maxbonddim(mps_truncated) <= 4
        # Original should be unchanged
        @test maxbonddim(mps) == 8
    end

    @testset "MPS deepcopy" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 3))
        t1 = T4ATensor([l01, s1], rand(3, 2))

        mps = MPS([t0, t1])
        mps_deep = deepcopy(mps)

        # Deepcopy should have same properties
        @test length(mps_deep) == length(mps)
        @test linkdims(mps_deep) == linkdims(mps)

        # Modifying deepcopy shouldn't affect original
        orthogonalize!(mps_deep, 1)
        @test canonical_form(mps_deep) == Unitary
        @test canonical_form(mps) === nothing
    end

    @testset "MPS iterator" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)
        l12 = T4AIndex(4)
        s2 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 3))
        t1 = T4ATensor([l01, s1, l12], rand(3, 2, 4))
        t2 = T4ATensor([l12, s2], rand(4, 2))

        mps = MPS([t0, t1, t2])

        # Test iteration
        collected = collect(mps)
        @test length(collected) == 3
        @test all(t -> t isa T4ATensor, collected)

        # Test for loop
        count = 0
        for t in mps
            count += 1
            @test t isa T4ATensor
        end
        @test count == 3

        # Test indexing helpers
        @test firstindex(mps) == 1
        @test lastindex(mps) == 3
        @test eachindex(mps) == 1:3
        @test eltype(typeof(mps)) == T4ATensor
    end

    @testset "MPS from Arrays" begin
        # Create site indices
        s1 = T4AIndex(2)
        s2 = T4AIndex(3)
        s3 = T4AIndex(2)

        # Create arrays with proper bond dimensions
        arr1 = rand(2, 4)      # site_dim=2, right_bond=4
        arr2 = rand(4, 3, 5)   # left_bond=4, site_dim=3, right_bond=5
        arr3 = rand(5, 2)      # left_bond=5, site_dim=2

        # Create MPS from arrays
        mps = MPS([arr1, arr2, arr3], [s1, s2, s3])

        @test length(mps) == 3
        @test linkdims(mps) == [4, 5]

        # Test with single site
        s_single = T4AIndex(4)
        arr_single = rand(4)
        mps_single = MPS([arr_single], [s_single])
        @test length(mps_single) == 1
    end

    @testset "random_mps" begin
        # Create site indices
        sites = [T4AIndex(2) for _ in 1:5]

        # Test with uniform bond dimension
        mps = random_mps(sites; linkdims=4)
        @test length(mps) == 5
        @test all(==(4), linkdims(mps))
        @test maxbonddim(mps) == 4

        # Test with variable bond dimensions
        mps2 = random_mps(sites; linkdims=[2, 3, 4, 5])
        @test linkdims(mps2) == [2, 3, 4, 5]

        # Test with complex type
        mps_complex = random_mps(ComplexF64, sites; linkdims=3)
        @test length(mps_complex) == 5

        # Test single site
        mps_single = random_mps([T4AIndex(4)])
        @test length(mps_single) == 1

        # Verify tensor structure
        mps3 = random_mps([T4AIndex(2), T4AIndex(3), T4AIndex(4)]; linkdims=5)
        @test length(mps3) == 3
        # First tensor should have rank 2 (site + right link)
        @test rank(mps3[1]) == 2
        # Middle tensor should have rank 3 (left link + site + right link)
        @test rank(mps3[2]) == 3
        # Last tensor should have rank 2 (left link + site)
        @test rank(mps3[3]) == 2

        # Test random_tt alias
        mps_alias = random_tt(sites; linkdims=4)
        @test length(mps_alias) == 5
    end

    @testset "MPS setindex!" begin
        using Tensor4all: indices

        # Create a random MPS
        sites = [T4AIndex(2) for _ in 1:3]
        mps = random_mps(sites; linkdims=4)

        # Orthogonalize first
        orthogonalize!(mps, 2)
        @test canonical_form(mps) == Unitary

        # Get the middle tensor and modify it
        old_tensor = mps[2]
        old_inds = indices(old_tensor)

        # Create a new tensor with the same indices but different data
        new_data = rand(dims(old_tensor)...)
        new_tensor = T4ATensor(old_inds, new_data)

        # Set the tensor
        mps[2] = new_tensor

        # The tensor should be updated
        updated_tensor = mps[2]
        @test Tensor4all.data(updated_tensor) ≈ new_data
    end

    @testset "MPS site search" begin
        using Tensor4all: hasind, hasinds

        # Create site indices with tags for identification
        s1 = T4AIndex(2; tags="Site,n=1")
        s2 = T4AIndex(3; tags="Site,n=2")
        s3 = T4AIndex(4; tags="Site,n=3")
        sites = [s1, s2, s3]

        mps = random_mps(sites; linkdims=5)

        # Test findsite - should find the site containing the given index
        @test findsite(mps, s1) == 1
        @test findsite(mps, s2) == 2
        @test findsite(mps, s3) == 3

        # Test findsite with non-existent index
        other_idx = T4AIndex(10)
        @test findsite(mps, other_idx) === nothing

        # Test findsites - should return all sites with common indices
        @test findsites(mps, s1) == [1]
        @test findsites(mps, s2) == [2]
        @test findsites(mps, [s1, s3]) == [1, 3]

        # Test findsites with link index (should find both adjacent sites)
        link1 = linkind(mps, 1)
        @test link1 !== nothing
        found = findsites(mps, link1)
        @test length(found) == 2
        @test 1 in found && 2 in found

        # Test siteinds - should return site indices only (not link indices)
        si1 = siteinds(mps, 1)
        @test length(si1) == 1
        @test Tensor4all.id(si1[1]) == Tensor4all.id(s1)

        si2 = siteinds(mps, 2)
        @test length(si2) == 1
        @test Tensor4all.id(si2[1]) == Tensor4all.id(s2)

        si3 = siteinds(mps, 3)
        @test length(si3) == 1
        @test Tensor4all.id(si3[1]) == Tensor4all.id(s3)

        # Test siteind - convenience for single site index
        @test siteind(mps, 1) !== nothing
        @test Tensor4all.id(siteind(mps, 1)) == Tensor4all.id(s1)

        # Test hasind predicate
        t1 = mps[1]
        @test hasind(s1)(t1)
        @test !hasind(s2)(t1)
        @test !hasind(s3)(t1)

        # Test hasinds predicate
        @test hasinds([s1])(t1)
        @test hasinds(s1)(t1)

        # Test findfirst with predicate on MPS
        @test findfirst(hasind(s2), mps) == 2
        @test findfirst(hasind(s3), mps) == 3
    end

    @testset "TreeTN basic smoke test" begin
        # Create a simple MPS and test core operations
        sites = [T4AIndex(2) for _ in 1:4]
        mps = random_mps(sites; linkdims=3)

        # Test basic properties
        @test nv(mps) == 4
        @test ne(mps) == 3
        @test maxbonddim(mps) == 3
        @test linkdims(mps) == [3, 3, 3]
        @test linkdim(mps, 1) == 3

        # Test orthogonalize
        orthogonalize!(mps, 2)
        @test canonical_form(mps) == Unitary

        # Test inner product
        ip = inner(mps, mps)
        n = norm(mps)
        @test isapprox(real(ip), n^2; rtol=1e-10)

        # Test truncate
        truncate!(mps; maxdim=2)
        @test maxbonddim(mps) <= 2

        # Test to_dense
        dense = to_dense(mps)
        @test rank(dense) == 4
    end
end
