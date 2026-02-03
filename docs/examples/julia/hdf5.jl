# Julia documentation examples: HDF5 I/O (ITensors.jl compatible)
#
# Run with:
#   julia --project=julia/Tensor4all.jl docs/examples/julia/hdf5.jl

using Random
using Tensor4all
using Tensor4all.TreeTN

# ANCHOR: save_load_tensor
mktempdir() do dir
    path = joinpath(dir, "tensor.h5")
    i = Index(2; tags="Site,n=1")
    j = Index(3; tags="Link,l=1")
    t = Tensor([i, j], ones(2, 3))

    save_itensor(path, "my_tensor", t)
    t2 = load_itensor(path, "my_tensor")

    @assert dims(t2) == (2, 3)
    @assert tags(indices(t2)[1]) == "Site,n=1"
end
# ANCHOR_END: save_load_tensor

# ANCHOR: save_load_mps
mktempdir() do dir
    path = joinpath(dir, "mps.h5")
    Random.seed!(1)
    sites = [Index(2; tags="Site,n=$n") for n in 1:3]
    mps = random_mps(sites; linkdims=2)

    save_mps(path, "psi", mps)
    mps2 = load_mps(path, "psi")

    @assert length(mps2) == length(mps)
end
# ANCHOR_END: save_load_mps

