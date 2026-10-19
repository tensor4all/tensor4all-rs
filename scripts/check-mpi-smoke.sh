#!/usr/bin/env bash
# Runs the MPI smoke test of the distributed adaptive patch driver.
#
# The test initializes MPI itself, so it is one `[[test]]` binary and must run
# under a launcher. It uses hataori's runtime backend (`rsmpi-rt`), whose
# `mpi` crate loads the MPI wrapper library from `MPI_RT_LIB`.
#
# Usage:
#   MPI_RT_LIB=/path/to/libmpiwrapper.so scripts/check-mpi-smoke.sh [worlds]
#
# `worlds` is a space-separated list of world sizes; the default is "1 2".
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mpi_rt_lib="${MPI_RT_LIB:-}"
[[ $mpi_rt_lib = /* && -f $mpi_rt_lib ]] || {
    printf 'check-mpi-smoke: set MPI_RT_LIB to an absolute MPI wrapper library path\n' >&2
    exit 2
}

worlds=${1:-1 2}
launcher=${MPIEXEC:-mpiexec}
launcher_flags=()
if "$launcher" --version 2>&1 | grep -Eq 'Open MPI|OpenRTE'; then
    launcher_flags+=(--oversubscribe)
    if ((EUID == 0)); then
        launcher_flags+=(--allow-run-as-root)
    fi
fi

binary=$(cargo test --locked -p tensor4all-partitionedtreetn \
    --features hataori-mpi-rsmpi-rt --test adaptive_mpi --no-run \
    --message-format=json 2>/dev/null |
    grep -o '"executable":"[^"]*"' | tail -n 1 | cut -d'"' -f4)
[[ -x $binary ]] || {
    printf 'check-mpi-smoke: could not locate the adaptive_mpi test binary\n' >&2
    exit 1
}

for count in $worlds; do
    printf 'check-mpi-smoke: world size %s\n' "$count"
    env MPI_RT_LIB="$mpi_rt_lib" setsid timeout --signal=TERM --kill-after=5s 300s \
        "$launcher" "${launcher_flags[@]}" --bind-to none -n "$count" \
        "$binary" --nocapture --ignored
done

printf 'check-mpi-smoke: distributed adaptive patching passed\n'
