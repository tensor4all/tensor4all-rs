# Build script for tensor4all-capi Rust library
#
# This script builds the Rust library and copies it to the deps directory.
# Run via `Pkg.build("Tensor4all")` or `julia deps/build.jl`.
#
# Rust source resolution (in priority order):
#   1. TENSOR4ALL_RS_PATH environment variable
#   2. Sibling directory ../tensor4all-rs/ (relative to package root)
#   3. Clone from GitHub

using Libdl
using RustToolChain: cargo

# Configuration
const TENSOR4ALL_RS_VERSION = "v0.1.0"  # Tag or commit hash
const TENSOR4ALL_RS_REPO = "https://github.com/tensor4all/tensor4all-rs.git"

# Paths
const DEPS_DIR = @__DIR__
const PACKAGE_DIR = dirname(DEPS_DIR)
const SIBLING_RUST_DIR = joinpath(dirname(PACKAGE_DIR), "tensor4all-rs")

# Output library name (platform-specific)
const LIB_NAME = "libtensor4all_capi." * Libdl.dlext

"""
    find_rust_source() -> Union{String, Nothing}

Find the tensor4all-rs source directory.
Returns the path if found locally, or `nothing` if GitHub clone is needed.
"""
function find_rust_source()
    # Priority 1: Environment variable
    env_path = get(ENV, "TENSOR4ALL_RS_PATH", nothing)
    if env_path !== nothing && isdir(env_path)
        cargo_toml = joinpath(env_path, "Cargo.toml")
        if isfile(cargo_toml)
            return env_path
        else
            @warn "TENSOR4ALL_RS_PATH is set but Cargo.toml not found" path=env_path
        end
    end

    # Priority 2: Sibling directory
    if isdir(SIBLING_RUST_DIR)
        cargo_toml = joinpath(SIBLING_RUST_DIR, "Cargo.toml")
        if isfile(cargo_toml)
            return SIBLING_RUST_DIR
        end
    end

    # No local source found
    return nothing
end

"""
    clone_from_github(dest::String)

Clone tensor4all-rs from GitHub to the specified directory.
"""
function clone_from_github(dest::String)
    println("Cloning tensor4all-rs from GitHub (version: $TENSOR4ALL_RS_VERSION)...")
    run(`git clone --depth 1 --branch $TENSOR4ALL_RS_VERSION $TENSOR4ALL_RS_REPO $dest`)
end

"""
    build_library(rust_dir::String)

Build tensor4all-capi in the specified Rust workspace directory.
"""
function build_library(rust_dir::String)
    println("Building tensor4all-capi...")
    println("  Rust source: $rust_dir")
    println("  Using cargo from RustToolChain.jl")

    cd(rust_dir) do
        run(`$(cargo()) build -p tensor4all-capi --release`)
    end

    # Find the built library
    src_lib = joinpath(rust_dir, "target", "release", LIB_NAME)
    if !isfile(src_lib)
        error("Built library not found at: $src_lib")
    end

    return src_lib
end

"""
    install_library(src_lib::String)

Copy the built library to the deps directory.
"""
function install_library(src_lib::String)
    dst_lib = joinpath(DEPS_DIR, LIB_NAME)
    println("Installing library...")
    println("  $src_lib -> $dst_lib")
    cp(src_lib, dst_lib; force=true)
    println("Build complete!")
    println("Library installed to: $dst_lib")
end

function main()
    rust_dir = find_rust_source()
    cleanup = false

    if rust_dir === nothing
        # Clone from GitHub
        rust_dir = mktempdir()
        clone_from_github(rust_dir)
        cleanup = true
    else
        println("Using local Rust source: $rust_dir")
    end

    try
        src_lib = build_library(rust_dir)
        install_library(src_lib)
    finally
        if cleanup
            println("Cleaning up temporary directory...")
            rm(rust_dir; recursive=true, force=true)
        end
    end
end

# Run build
main()
